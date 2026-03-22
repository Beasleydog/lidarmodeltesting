from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train import (
    DEFAULT_LIDAR_MAX_RANGE_CM,
    MODEL_SENSOR_YAW_PITCH_DEG,
    configure_runtime_for_device,
    log,
    log_plain,
    optimizer_step_for_device,
    resolve_data_loader_workers,
    save_checkpoint_for_device,
    select_runtime_device,
    set_log_file,
)


@dataclass(frozen=True)
class RealSequence:
    name: str
    pose_xyz_cm: np.ndarray
    heading_pitch_roll_deg: np.ndarray
    lidar_cm: np.ndarray
    obstacle_labels: np.ndarray
    sensor_names: tuple[str, ...]


@dataclass(frozen=True)
class DatasetMeta:
    train_files: list[str]
    val_files: list[str]
    train_windows: int
    val_windows: int
    num_sensors: int
    history_steps: int
    current_plus_history: int
    pos_weight: float
    sample_positive_fraction: float
    beam_positive_fraction: float


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _parse_float(value: object, default: float, field_name: str, file_name: str, row_idx: int) -> float:
    if value is None:
        return float(default)
    text = str(value).strip()
    if not text:
        return float(default)
    lowered = text.lower()
    if lowered in {"nan", "none", "null", "na", "n/a"}:
        return float(default)
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse float in {file_name} row {row_idx} field {field_name}: {text!r}"
        ) from exc
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _sensor_columns(fieldnames: list[str]) -> list[str]:
    cols = [name for name in fieldnames if name.startswith("lidar_") and name.endswith("_cm")]
    if not cols:
        raise ValueError("No lidar *_cm columns found in raw cleanlog CSV")
    return cols


def _label_columns(fieldnames: list[str], sensor_columns: list[str]) -> list[str]:
    label_cols = []
    for sensor_col in sensor_columns:
        label_col = f"{sensor_col}_is_obstacle"
        if label_col not in fieldnames:
            raise ValueError(f"Missing label column {label_col!r}")
        label_cols.append(label_col)
    return label_cols


def _read_cleanlog_pair(raw_path: Path, label_path: Path) -> RealSequence:
    with raw_path.open("r", encoding="utf-8", newline="") as raw_fh:
        raw_reader = csv.DictReader(raw_fh)
        raw_fieldnames = list(raw_reader.fieldnames or [])
        sensor_columns = _sensor_columns(raw_fieldnames)
        raw_rows = list(raw_reader)

    with label_path.open("r", encoding="utf-8", newline="") as label_fh:
        label_reader = csv.DictReader(label_fh)
        label_fieldnames = list(label_reader.fieldnames or [])
        _label_columns(label_fieldnames, sensor_columns)
        label_rows = list(label_reader)

    if len(raw_rows) != len(label_rows):
        raise ValueError(
            f"Row count mismatch for {raw_path.name}: raw={len(raw_rows)} label={len(label_rows)}"
        )
    if len(raw_rows) == 0:
        raise ValueError(f"{raw_path.name} contains zero rows")

    num_steps = len(raw_rows)
    num_sensors = len(sensor_columns)
    pose_xyz = np.zeros((num_steps, 3), dtype=np.float32)
    angles = np.zeros((num_steps, 3), dtype=np.float32)
    lidar_cm = np.zeros((num_steps, num_sensors), dtype=np.float32)
    labels = np.zeros((num_steps, num_sensors), dtype=np.float32)

    for idx, (raw_row, label_row) in enumerate(zip(raw_rows, label_rows, strict=True)):
        for key in ("iso_time_utc", "step_idx"):
            raw_value = raw_row.get(key, "")
            label_value = label_row.get(key, "")
            if raw_value != label_value:
                raise ValueError(
                    f"Alignment mismatch for {raw_path.name} row {idx}: "
                    f"{key} raw={raw_value!r} label={label_value!r}"
                )

        pose_xyz[idx, 0] = _parse_float(raw_row.get("rover_pos_x"), 0.0, "rover_pos_x", raw_path.name, idx)
        pose_xyz[idx, 1] = _parse_float(raw_row.get("rover_pos_y"), 0.0, "rover_pos_y", raw_path.name, idx)
        pose_xyz[idx, 2] = _parse_float(raw_row.get("rover_pos_z"), 0.0, "rover_pos_z", raw_path.name, idx)
        angles[idx, 0] = _parse_float(raw_row.get("heading"), 0.0, "heading", raw_path.name, idx)
        angles[idx, 1] = _parse_float(raw_row.get("pitch"), 0.0, "pitch", raw_path.name, idx)
        angles[idx, 2] = _parse_float(raw_row.get("roll"), 0.0, "roll", raw_path.name, idx)

        for sensor_idx, sensor_col in enumerate(sensor_columns):
            lidar_cm[idx, sensor_idx] = _parse_float(raw_row.get(sensor_col), -1.0, sensor_col, raw_path.name, idx)
            label_value = _parse_float(
                label_row.get(f"{sensor_col}_is_obstacle"),
                0.0,
                f"{sensor_col}_is_obstacle",
                label_path.name,
                idx,
            )
            labels[idx, sensor_idx] = 1.0 if label_value > 0.5 else 0.0

    return RealSequence(
        name=raw_path.name,
        pose_xyz_cm=pose_xyz,
        heading_pitch_roll_deg=angles,
        lidar_cm=lidar_cm,
        obstacle_labels=labels,
        sensor_names=tuple(sensor_columns),
    )


def load_real_sequences(cleanlog_dir: Path) -> list[RealSequence]:
    label_dir = cleanlog_dir / "labeled_obstacles_liveexport"
    if not cleanlog_dir.exists():
        raise FileNotFoundError(f"Missing cleanlog directory: {cleanlog_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    raw_files = sorted(path for path in cleanlog_dir.glob("*.csv") if (label_dir / path.name).exists())
    if not raw_files:
        raise FileNotFoundError(f"No paired raw/label CSVs found in {cleanlog_dir}")

    sequences = [_read_cleanlog_pair(raw_path, label_dir / raw_path.name) for raw_path in raw_files]
    sensor_names = sequences[0].sensor_names
    for seq in sequences[1:]:
        if seq.sensor_names != sensor_names:
            raise ValueError(f"Sensor schema mismatch in {seq.name}")
    return sequences


def split_sequences(
    sequences: list[RealSequence],
    val_fraction: float,
    seed: int,
) -> tuple[list[RealSequence], list[RealSequence]]:
    if len(sequences) < 2:
        raise ValueError("Need at least two paired cleanlog files to create a validation split")
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(sequences))
    rng.shuffle(order)

    val_count = int(round(len(sequences) * float(val_fraction)))
    val_count = max(1, min(len(sequences) - 1, val_count))
    val_ids = set(int(i) for i in order[:val_count])
    train = [seq for idx, seq in enumerate(sequences) if idx not in val_ids]
    val = [seq for idx, seq in enumerate(sequences) if idx in val_ids]
    return train, val


class RealLidarSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[RealSequence],
        history_steps: int,
        max_range_cm: float,
        augment: bool = False,
        xy_offset_max_cm: float = 10000.0,
        z_offset_max_cm: float = 10.0,
        absolute_xy_scale_cm: float = 20000.0,
        absolute_z_scale_cm: float = 5000.0,
        relative_xy_scale_cm: float = 5000.0,
        relative_z_scale_cm: float = 500.0,
    ):
        self.sequences = list(sequences)
        self.history_steps = int(history_steps)
        self.window_size = self.history_steps + 1
        self.max_range_cm = float(max(max_range_cm, 1.0))
        self.augment = bool(augment)
        self.xy_offset_max_cm = float(max(xy_offset_max_cm, 0.0))
        self.z_offset_max_cm = float(max(z_offset_max_cm, 0.0))
        self.absolute_xy_scale_cm = float(max(absolute_xy_scale_cm, 1.0))
        self.absolute_z_scale_cm = float(max(absolute_z_scale_cm, 1.0))
        self.relative_xy_scale_cm = float(max(relative_xy_scale_cm, 1.0))
        self.relative_z_scale_cm = float(max(relative_z_scale_cm, 1.0))
        self.rng = np.random.default_rng()

        self.index: list[tuple[int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            if seq.lidar_cm.shape[0] < self.window_size:
                continue
            for end_idx in range(self.history_steps, seq.lidar_cm.shape[0]):
                self.index.append((seq_idx, end_idx))

        if not self.index:
            raise ValueError(
                f"No valid windows: history_steps={self.history_steps} exceeds all cleanlog sequence lengths"
            )

        self.num_sensors = int(self.sequences[0].lidar_cm.shape[1])
        sensor_yaw = np.deg2rad(MODEL_SENSOR_YAW_PITCH_DEG[: self.num_sensors, 0].astype(np.float32))
        sensor_pitch = np.deg2rad(MODEL_SENSOR_YAW_PITCH_DEG[: self.num_sensors, 1].astype(np.float32))
        sensor_idx = np.linspace(0.0, 1.0, self.num_sensors, dtype=np.float32)
        self.sensor_features = {
            "yaw_sin": np.sin(sensor_yaw).astype(np.float32),
            "yaw_cos": np.cos(sensor_yaw).astype(np.float32),
            "pitch_sin": np.sin(sensor_pitch).astype(np.float32),
            "pitch_cos": np.cos(sensor_pitch).astype(np.float32),
            "index": sensor_idx.astype(np.float32),
        }
        self.current_has_obstacle = np.asarray(
            [self.sequences[seq_idx].obstacle_labels[end_idx].max() > 0.5 for seq_idx, end_idx in self.index],
            dtype=np.bool_,
        )

    def __len__(self) -> int:
        return len(self.index)

    @property
    def input_channels(self) -> int:
        return 20

    def _sample_translation_offsets(self) -> np.ndarray:
        if not self.augment:
            return np.zeros((3,), dtype=np.float32)
        return np.asarray(
            [
                self.rng.uniform(-self.xy_offset_max_cm, self.xy_offset_max_cm),
                self.rng.uniform(-self.xy_offset_max_cm, self.xy_offset_max_cm),
                self.rng.uniform(-self.z_offset_max_cm, self.z_offset_max_cm),
            ],
            dtype=np.float32,
        )

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_idx, end_idx = self.index[item]
        seq = self.sequences[seq_idx]
        start_idx = end_idx - self.history_steps

        pose_xyz = seq.pose_xyz_cm[start_idx : end_idx + 1].copy()
        pose_xyz += self._sample_translation_offsets().reshape(1, 3)
        rel_xyz = pose_xyz - pose_xyz[-1:].copy()
        angles_deg = seq.heading_pitch_roll_deg[start_idx : end_idx + 1]
        lidar = seq.lidar_cm[start_idx : end_idx + 1]
        target = seq.obstacle_labels[end_idx]

        hit = (lidar >= 0.0).astype(np.float32)
        clipped = np.where(hit > 0.0, np.clip(lidar, 0.0, self.max_range_cm), self.max_range_cm).astype(np.float32)
        range_norm = (clipped / self.max_range_cm).astype(np.float32)

        headings = np.deg2rad(angles_deg[:, 0].astype(np.float32))
        pitches = np.deg2rad(angles_deg[:, 1].astype(np.float32))
        rolls = np.deg2rad(angles_deg[:, 2].astype(np.float32))
        time_age = np.linspace(-1.0, 0.0, self.window_size, dtype=np.float32)

        abs_x = np.broadcast_to((pose_xyz[:, 0] / self.absolute_xy_scale_cm)[:, None], (self.window_size, self.num_sensors))
        abs_y = np.broadcast_to((pose_xyz[:, 1] / self.absolute_xy_scale_cm)[:, None], (self.window_size, self.num_sensors))
        abs_z = np.broadcast_to((pose_xyz[:, 2] / self.absolute_z_scale_cm)[:, None], (self.window_size, self.num_sensors))
        rel_x = np.broadcast_to((rel_xyz[:, 0] / self.relative_xy_scale_cm)[:, None], (self.window_size, self.num_sensors))
        rel_y = np.broadcast_to((rel_xyz[:, 1] / self.relative_xy_scale_cm)[:, None], (self.window_size, self.num_sensors))
        rel_z = np.broadcast_to((rel_xyz[:, 2] / self.relative_z_scale_cm)[:, None], (self.window_size, self.num_sensors))
        heading_sin = np.broadcast_to(np.sin(headings)[:, None], (self.window_size, self.num_sensors))
        heading_cos = np.broadcast_to(np.cos(headings)[:, None], (self.window_size, self.num_sensors))
        pitch_sin = np.broadcast_to(np.sin(pitches)[:, None], (self.window_size, self.num_sensors))
        pitch_cos = np.broadcast_to(np.cos(pitches)[:, None], (self.window_size, self.num_sensors))
        roll_sin = np.broadcast_to(np.sin(rolls)[:, None], (self.window_size, self.num_sensors))
        roll_cos = np.broadcast_to(np.cos(rolls)[:, None], (self.window_size, self.num_sensors))
        sensor_yaw_sin = np.broadcast_to(self.sensor_features["yaw_sin"][None, :], (self.window_size, self.num_sensors))
        sensor_yaw_cos = np.broadcast_to(self.sensor_features["yaw_cos"][None, :], (self.window_size, self.num_sensors))
        sensor_pitch_sin = np.broadcast_to(self.sensor_features["pitch_sin"][None, :], (self.window_size, self.num_sensors))
        sensor_pitch_cos = np.broadcast_to(self.sensor_features["pitch_cos"][None, :], (self.window_size, self.num_sensors))
        sensor_idx = np.broadcast_to(self.sensor_features["index"][None, :], (self.window_size, self.num_sensors))
        time_map = np.broadcast_to(time_age[:, None], (self.window_size, self.num_sensors))

        features = np.stack(
            [
                range_norm,
                hit,
                abs_x,
                abs_y,
                abs_z,
                rel_x,
                rel_y,
                rel_z,
                heading_sin,
                heading_cos,
                pitch_sin,
                pitch_cos,
                roll_sin,
                roll_cos,
                sensor_yaw_sin,
                sensor_yaw_cos,
                sensor_pitch_sin,
                sensor_pitch_cos,
                sensor_idx,
                time_map,
            ],
            axis=0,
        ).astype(np.float32)

        return torch.from_numpy(features), torch.from_numpy(target.astype(np.float32))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch >= 8 and out_ch % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallTemporalUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, dropout: float = 0.10):
        super().__init__()
        self.model_type = "real_cleanlog_temporal_unet"
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.drop = nn.Dropout2d(dropout)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def _upsample_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(self.drop(e1)))
        e3 = self.enc3(self.pool(self.drop(e2)))
        b = self.bottleneck(self.pool(self.drop(e3)))

        d3 = self._upsample_to(b, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self._upsample_to(d3, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._upsample_to(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.head(d1).squeeze(1)
        return logits[:, -1, :]


def build_sampler(dataset: RealLidarSequenceDataset) -> WeightedRandomSampler | None:
    sample_has_obstacle = dataset.current_has_obstacle.astype(np.float32)
    pos = float(sample_has_obstacle.sum())
    neg = float(len(sample_has_obstacle) - pos)
    if pos <= 0.0 or neg <= 0.0:
        return None
    pos_weight = neg / pos
    sample_weights = np.where(sample_has_obstacle > 0.5, pos_weight, 1.0).astype(np.float32)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_pos_weight(sequences: list[RealSequence], max_weight: float) -> tuple[float, float, float]:
    labels = np.concatenate([seq.obstacle_labels.reshape(-1) for seq in sequences], axis=0).astype(np.float64)
    positive = float(np.sum(labels > 0.5))
    negative = float(np.sum(labels <= 0.5))
    raw_pos_weight = negative / max(positive, 1.0)
    pos_weight = np.sqrt(raw_pos_weight)
    pos_weight = float(np.clip(pos_weight, 1.0, float(max_weight)))

    sample_positive_flags = np.concatenate(
        [np.any(seq.obstacle_labels > 0.5, axis=1).astype(np.float64) for seq in sequences],
        axis=0,
    )
    sample_positive_fraction = float(sample_positive_flags.mean()) if sample_positive_flags.size else 0.0
    beam_positive_fraction = float(np.mean(labels > 0.5)) if labels.size else 0.0
    return pos_weight, sample_positive_fraction, beam_positive_fraction


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    device_kind: str,
    pos_weight: float,
    decision_threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    pos_weight_t = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    threshold = float(decision_threshold)
    logits_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight_t)

            total_loss += float(loss.item()) * float(y.shape[0])
            total_samples += int(y.shape[0])
            logits_chunks.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
            target_chunks.append(y.detach().cpu().numpy().astype(np.float32, copy=False))

            if device_kind == "xla":
                import torch_xla.core.xla_model as xm

                xm.mark_step()

    logits_np = np.concatenate(logits_chunks, axis=0)
    targets_np = np.concatenate(target_chunks, axis=0)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    preds_np = probs_np >= threshold
    target_bool = targets_np > 0.5

    tp = int(np.sum(preds_np & target_bool))
    tn = int(np.sum((~preds_np) & (~target_bool)))
    fp = int(np.sum(preds_np & (~target_bool)))
    fn = int(np.sum((~preds_np) & target_bool))
    total_beams = tp + tn + fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(total_beams, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    false_positive_rate = fp / max(fp + tn, 1)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _compute_binary_metrics_from_probs(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    preds = probs >= float(threshold)
    target_bool = targets > 0.5
    tp = int(np.sum(preds & target_bool))
    tn = int(np.sum((~preds) & (~target_bool)))
    fp = int(np.sum(preds & (~target_bool)))
    fn = int(np.sum((~preds) & target_bool))
    total = tp + tn + fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(total, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    false_positive_rate = fp / max(fp + tn, 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def sweep_validation_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    device_kind: str,
    thresholds: list[float],
    min_recall: float,
) -> dict[str, float | list[list[int]]]:
    model.eval()
    logits_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            logits_chunks.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
            target_chunks.append(y.detach().cpu().numpy().astype(np.float32, copy=False))

            if device_kind == "xla":
                import torch_xla.core.xla_model as xm

                xm.mark_step()

    logits_np = np.concatenate(logits_chunks, axis=0)
    targets_np = np.concatenate(target_chunks, axis=0)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))

    best_metrics: dict[str, float | list[list[int]]] | None = None
    best_fallback: dict[str, float | list[list[int]]] | None = None
    for threshold in thresholds:
        metrics = _compute_binary_metrics_from_probs(probs_np, targets_np, threshold)
        if best_fallback is None:
            best_fallback = metrics
        else:
            if float(metrics["f1"]) > float(best_fallback["f1"]) + 1e-12:
                best_fallback = metrics
            elif (
                abs(float(metrics["f1"]) - float(best_fallback["f1"])) <= 1e-12
                and float(metrics["precision"]) > float(best_fallback["precision"])
            ):
                best_fallback = metrics

        if float(metrics["recall"]) + 1e-12 < float(min_recall):
            continue
        if best_metrics is None:
            best_metrics = metrics
            continue
        if float(metrics["precision"]) > float(best_metrics["precision"]) + 1e-12:
            best_metrics = metrics
            continue
        if abs(float(metrics["precision"]) - float(best_metrics["precision"])) <= 1e-12:
            if float(metrics["f1"]) > float(best_metrics["f1"]) + 1e-12:
                best_metrics = metrics
                continue
            if (
                abs(float(metrics["f1"]) - float(best_metrics["f1"])) <= 1e-12
                and float(metrics["specificity"]) > float(best_metrics["specificity"])
            ):
                best_metrics = metrics

    if best_metrics is None:
        best_metrics = best_fallback
    if best_metrics is None:
        raise RuntimeError("Threshold sweep produced no metrics")
    return best_metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    device_kind: str,
    epoch_idx: int,
    total_epochs: int,
    pos_weight: float,
    grad_clip_norm: float,
    use_amp: bool,
    log_every_batches: int,
) -> dict[str, float]:
    model.train(True)
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_beams = 0
    tp = 0
    fp = 0
    fn = 0
    batch_total = len(loader)
    epoch_t0 = perf_counter()
    pos_weight_t = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and device_kind == "cuda"))

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=device_kind == "cuda")
        y = y.to(device, non_blocking=device_kind == "cuda")

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if device_kind == "cuda" and use_amp
            else nullcontext()
        )
        with amp_context:
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight_t)

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer_step_for_device(optimizer, device_kind)

        preds = (logits > 0.0).float()
        total_loss += float(loss.item()) * float(y.shape[0])
        total_samples += int(y.shape[0])
        total_correct += int((preds == y).sum().item())
        total_beams += int(y.numel())
        tp += int(((preds == 1.0) & (y == 1.0)).sum().item())
        fp += int(((preds == 1.0) & (y == 0.0)).sum().item())
        fn += int(((preds == 0.0) & (y == 1.0)).sum().item())

        if log_every_batches > 0 and (batch_idx % log_every_batches == 0 or batch_idx == batch_total):
            running_loss = total_loss / max(total_samples, 1)
            running_acc = total_correct / max(total_beams, 1)
            running_precision = tp / max(tp + fp, 1)
            running_recall = tp / max(tp + fn, 1)
            log(
                f"train epoch {epoch_idx:03d}/{total_epochs:03d} "
                f"batch {batch_idx:04d}/{batch_total:04d} "
                f"loss={running_loss:.5f} acc={running_acc:.4f} "
                f"prec={running_precision:.4f} recall={running_recall:.4f}"
            )

    elapsed_s = perf_counter() - epoch_t0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    specificity = (max(total_beams - tp - fp - fn, 0)) / max(max(total_beams - tp - fn, 0), 1)
    false_positive_rate = fp / max(fp + max(total_beams - tp - fp - fn, 0), 1)
    tn = max(total_beams - tp - fp - fn, 0)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_beams, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "time_s": elapsed_s,
    }


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DatasetMeta, tuple[str, ...], int]:
    sequences = load_real_sequences(args.cleanlog_dir)
    train_sequences, val_sequences = split_sequences(sequences, val_fraction=args.val_fraction, seed=args.seed)
    pos_weight, sample_positive_fraction, beam_positive_fraction = compute_pos_weight(
        train_sequences,
        max_weight=args.pos_weight_cap,
    )

    train_ds = RealLidarSequenceDataset(
        train_sequences,
        history_steps=args.history_steps,
        max_range_cm=args.max_range_cm,
        augment=True,
        xy_offset_max_cm=args.xy_offset_max_cm,
        z_offset_max_cm=args.z_offset_max_cm,
        absolute_xy_scale_cm=args.absolute_xy_scale_cm,
        absolute_z_scale_cm=args.absolute_z_scale_cm,
        relative_xy_scale_cm=args.relative_xy_scale_cm,
        relative_z_scale_cm=args.relative_z_scale_cm,
    )
    val_ds = RealLidarSequenceDataset(
        val_sequences,
        history_steps=args.history_steps,
        max_range_cm=args.max_range_cm,
        augment=False,
        xy_offset_max_cm=0.0,
        z_offset_max_cm=0.0,
        absolute_xy_scale_cm=args.absolute_xy_scale_cm,
        absolute_z_scale_cm=args.absolute_z_scale_cm,
        relative_xy_scale_cm=args.relative_xy_scale_cm,
        relative_z_scale_cm=args.relative_z_scale_cm,
    )
    workers = resolve_data_loader_workers(args.workers)
    pin_memory = str(args.device).lower() in {"auto", "cuda"}
    sampler = build_sampler(train_ds) if args.balance_positive_windows else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=bool(workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=bool(workers > 0),
    )
    meta = DatasetMeta(
        train_files=[seq.name for seq in train_sequences],
        val_files=[seq.name for seq in val_sequences],
        train_windows=len(train_ds),
        val_windows=len(val_ds),
        num_sensors=train_ds.num_sensors,
        history_steps=args.history_steps,
        current_plus_history=args.history_steps + 1,
        pos_weight=pos_weight,
        sample_positive_fraction=sample_positive_fraction,
        beam_positive_fraction=beam_positive_fraction,
    )
    return train_loader, val_loader, meta, train_sequences[0].sensor_names, train_ds.input_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a real cleanlog beam classifier on the last N lidar frames plus the current frame."
    )
    parser.add_argument("--cleanlog-dir", type=Path, default=Path("cleanlog"))
    parser.add_argument("--output", type=Path, default=Path("runs/realdatabeam_unet.pt"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--history-steps", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-range-cm", type=float, default=DEFAULT_LIDAR_MAX_RANGE_CM)
    parser.add_argument("--xy-offset-max-cm", type=float, default=10000.0)
    parser.add_argument("--z-offset-max-cm", type=float, default=10.0)
    parser.add_argument("--absolute-xy-scale-cm", type=float, default=20000.0)
    parser.add_argument("--absolute-z-scale-cm", type=float, default=5000.0)
    parser.add_argument("--relative-xy-scale-cm", type=float, default=5000.0)
    parser.add_argument("--relative-z-scale-cm", type=float, default=500.0)
    parser.add_argument("--pos-weight-cap", type=float, default=25.0)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--eval-threshold", type=float, default=0.7)
    parser.add_argument("--threshold-sweep-start", type=float, default=0.60)
    parser.add_argument("--threshold-sweep-end", type=float, default=0.90)
    parser.add_argument("--threshold-sweep-step", type=float, default=0.02)
    parser.add_argument("--threshold-min-recall", type=float, default=0.50)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--balance-positive-windows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log.txt")
    metrics_path = args.output.with_suffix(".metrics.csv")
    split_path = args.output.with_suffix(".split.json")
    val_report_path = args.output.with_suffix(".val_report.json")

    with log_path.open("w", encoding="utf-8") as log_fh:
        set_log_file(log_fh)
        try:
            device, device_kind = select_runtime_device(args.device)
            configure_runtime_for_device(device_kind)
            log("Starting real cleanlog temporal U-Net trainer")
            log(f"Device selected: {device_kind} -> {device}")

            train_loader, val_loader, meta, sensor_names, input_channels = build_loaders(args)
            log(
                "Dataset split summary: "
                f"train_files={len(meta.train_files)} val_files={len(meta.val_files)} "
                f"train_windows={meta.train_windows} val_windows={meta.val_windows}"
            )
            log(
                "Beam stats: "
                f"num_sensors={meta.num_sensors} history_steps={meta.history_steps} "
                f"window_size={meta.current_plus_history} pos_weight={meta.pos_weight:.3f}"
            )
            log(
                "Positive fractions from train split: "
                f"window_has_obstacle={meta.sample_positive_fraction:.4f} "
                f"beam_is_obstacle={meta.beam_positive_fraction:.4f}"
            )
            log(
                "Augmentation: "
                f"xy_offset=[{-args.xy_offset_max_cm:.1f},{args.xy_offset_max_cm:.1f}]cm "
                f"z_offset=[{-args.z_offset_max_cm:.1f},{args.z_offset_max_cm:.1f}]cm"
            )
            threshold_values = np.arange(
                float(args.threshold_sweep_start),
                float(args.threshold_sweep_end) + (0.5 * float(args.threshold_sweep_step)),
                float(args.threshold_sweep_step),
                dtype=np.float32,
            )
            threshold_values = np.clip(threshold_values, 0.0, 1.0)
            threshold_values = sorted(set(float(round(v, 6)) for v in threshold_values.tolist()))
            if float(args.eval_threshold) not in threshold_values:
                threshold_values.append(float(args.eval_threshold))
                threshold_values = sorted(set(threshold_values))
            log(
                f"Validation threshold sweep: {threshold_values[0]:.2f}..{threshold_values[-1]:.2f} "
                f"step~{args.threshold_sweep_step:.2f} default_eval={args.eval_threshold:.2f} "
                f"min_recall={args.threshold_min_recall:.2f}"
            )

            with split_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "train_files": meta.train_files,
                        "val_files": meta.val_files,
                        "train_windows": meta.train_windows,
                        "val_windows": meta.val_windows,
                        "num_sensors": meta.num_sensors,
                        "sensor_names": list(sensor_names),
                        "history_steps": meta.history_steps,
                        "window_size": meta.current_plus_history,
                        "beam_positive_fraction": meta.beam_positive_fraction,
                        "sample_positive_fraction": meta.sample_positive_fraction,
                        "pos_weight": meta.pos_weight,
                        "args": _jsonable_args(args),
                    },
                    fh,
                    indent=2,
                )
            log(f"Wrote split manifest: {split_path}")

            model = SmallTemporalUNet(
                in_channels=input_channels,
                base_channels=args.base_channels,
                dropout=args.dropout,
            ).to(device)
            param_count = sum(p.numel() for p in model.parameters())
            log(
                f"Model summary: type={model.model_type} base_channels={args.base_channels} "
                f"dropout={args.dropout:.2f} params={param_count}"
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.plateau_factor,
                patience=args.plateau_patience,
                min_lr=args.min_lr,
            )

            best = {"val_loss": float("inf"), "val_f1": -1.0, "val_precision": -1.0, "epoch": -1, "metrics": None}
            epochs_without_improve = 0
            with metrics_path.open("w", encoding="utf-8", newline="") as metrics_fh:
                writer = csv.writer(metrics_fh)
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "train_acc",
                        "train_precision",
                        "train_recall",
                        "train_f1",
                        "train_specificity",
                        "train_fpr",
                        "train_tn",
                        "train_fp",
                        "train_fn",
                        "train_tp",
                        "val_loss",
                        "val_acc",
                        "val_precision",
                        "val_recall",
                        "val_f1",
                        "val_specificity",
                        "val_fpr",
                        "val_threshold",
                        "val_tn",
                        "val_fp",
                        "val_fn",
                        "val_tp",
                        "best_val_precision",
                        "best_val_f1",
                        "lr",
                    ]
                )

                for epoch in range(1, args.epochs + 1):
                    train_metrics = train_epoch(
                        model=model,
                        loader=train_loader,
                        optimizer=optimizer,
                        device=device,
                        device_kind=device_kind,
                        epoch_idx=epoch,
                        total_epochs=args.epochs,
                        pos_weight=meta.pos_weight,
                        grad_clip_norm=args.grad_clip_norm,
                        use_amp=bool(args.amp),
                        log_every_batches=args.log_every_batches,
                    )
                    val_metrics = evaluate(
                        model=model,
                        loader=val_loader,
                        device=device,
                        device_kind=device_kind,
                        pos_weight=meta.pos_weight,
                        decision_threshold=float(args.eval_threshold),
                    )
                    sweep_metrics = sweep_validation_thresholds(
                        model=model,
                        loader=val_loader,
                        device=device,
                        device_kind=device_kind,
                        thresholds=threshold_values,
                        min_recall=float(args.threshold_min_recall),
                    )
                    val_metrics.update(
                        {
                            "accuracy": sweep_metrics["accuracy"],
                            "precision": sweep_metrics["precision"],
                            "recall": sweep_metrics["recall"],
                            "f1": sweep_metrics["f1"],
                            "threshold": sweep_metrics["threshold"],
                            "tn": sweep_metrics["tn"],
                            "fp": sweep_metrics["fp"],
                            "fn": sweep_metrics["fn"],
                            "tp": sweep_metrics["tp"],
                            "confusion_matrix": sweep_metrics["confusion_matrix"],
                        }
                    )
                    scheduler.step(float(val_metrics["loss"]))

                    current_lr = float(optimizer.param_groups[0]["lr"])
                    writer.writerow(
                        [
                            epoch,
                            train_metrics["loss"],
                            train_metrics["accuracy"],
                            train_metrics["precision"],
                            train_metrics["recall"],
                            train_metrics["f1"],
                            train_metrics["specificity"],
                            train_metrics["false_positive_rate"],
                            train_metrics["tn"],
                            train_metrics["fp"],
                            train_metrics["fn"],
                            train_metrics["tp"],
                            val_metrics["loss"],
                            val_metrics["accuracy"],
                            val_metrics["precision"],
                            val_metrics["recall"],
                            val_metrics["f1"],
                            val_metrics["specificity"],
                            val_metrics["false_positive_rate"],
                            val_metrics["threshold"],
                            val_metrics["tn"],
                            val_metrics["fp"],
                            val_metrics["fn"],
                            val_metrics["tp"],
                            max(float(best["val_precision"]), float(val_metrics["precision"])),
                            max(float(best["val_f1"]), float(val_metrics["f1"])),
                            current_lr,
                        ]
                    )
                    metrics_fh.flush()

                    log(
                        f"epoch {epoch:03d}/{args.epochs:03d} "
                        f"train_loss={train_metrics['loss']:.5f} train_acc={train_metrics['accuracy']:.4f} "
                        f"train_f1={train_metrics['f1']:.4f} train_spec={train_metrics['specificity']:.4f} "
                        f"train_cm=[[{int(train_metrics['tn'])},{int(train_metrics['fp'])}],"
                        f"[{int(train_metrics['fn'])},{int(train_metrics['tp'])}]] "
                        f"val_loss={float(val_metrics['loss']):.5f} val_acc={float(val_metrics['accuracy']):.4f} "
                        f"val_prec={float(val_metrics['precision']):.4f} val_recall={float(val_metrics['recall']):.4f} "
                        f"val_f1={float(val_metrics['f1']):.4f} val_spec={float(val_metrics['specificity']):.4f} "
                        f"val_fpr={float(val_metrics['false_positive_rate']):.4f} "
                        f"thr={float(val_metrics['threshold']):.2f} "
                        f"val_cm=[[{int(val_metrics['tn'])},{int(val_metrics['fp'])}],"
                        f"[{int(val_metrics['fn'])},{int(val_metrics['tp'])}]] "
                        f"lr={current_lr:.6f}"
                    )

                    improved = False
                    if float(val_metrics["precision"]) > (float(best["val_precision"]) + float(args.early_stop_min_delta)):
                        improved = True
                    elif abs(float(val_metrics["precision"]) - float(best["val_precision"])) <= float(
                        args.early_stop_min_delta
                    ) and float(val_metrics["f1"]) > (float(best["val_f1"]) + float(args.early_stop_min_delta)):
                        improved = True
                    if improved:
                        best = {
                            "val_loss": float(val_metrics["loss"]),
                            "val_f1": float(val_metrics["f1"]),
                            "val_precision": float(val_metrics["precision"]),
                            "epoch": epoch,
                            "metrics": val_metrics,
                        }
                        epochs_without_improve = 0
                        checkpoint = {
                            "model_state_dict": model.state_dict(),
                            "model_config": {
                                "model_type": model.model_type,
                                "in_channels": input_channels,
                                "base_channels": args.base_channels,
                                "dropout": args.dropout,
                                "num_sensors": meta.num_sensors,
                                "history_steps": meta.history_steps,
                                "max_range_cm": args.max_range_cm,
                            },
                            "meta": {
                                "train_files": meta.train_files,
                                "val_files": meta.val_files,
                                "sensor_names": list(sensor_names),
                                "beam_positive_fraction": meta.beam_positive_fraction,
                                "sample_positive_fraction": meta.sample_positive_fraction,
                                "pos_weight": meta.pos_weight,
                                "best_val_loss": best["val_loss"],
                                "best_val_precision": best["val_precision"],
                                "best_val_f1": best["val_f1"],
                                "best_threshold": float(val_metrics["threshold"]),
                                "best_epoch": best["epoch"],
                            },
                        }
                        save_checkpoint_for_device(checkpoint, args.output, device_kind)
                        with val_report_path.open("w", encoding="utf-8") as fh:
                            json.dump(
                                {
                                    "best_epoch": best["epoch"],
                                    "best_val_loss": best["val_loss"],
                                    "best_val_precision": best["val_precision"],
                                    "best_val_f1": best["val_f1"],
                                    "best_threshold": float(val_metrics["threshold"]),
                                    "metrics": best["metrics"],
                                },
                                fh,
                                indent=2,
                            )
                        log(
                            f"New best validation precision at epoch {epoch:03d}: "
                            f"{best['val_precision']:.6f} with F1={best['val_f1']:.6f} "
                            f"at threshold {float(val_metrics['threshold']):.2f}. "
                            f"Saved checkpoint -> {args.output}"
                        )
                    else:
                        epochs_without_improve += 1
                        log(
                            f"no val improvement for {epochs_without_improve} epoch(s) "
                            f"(patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta:.6f})"
                        )
                        if epochs_without_improve >= int(args.early_stop_patience):
                            log(
                                f"Early stopping at epoch {epoch:03d}: "
                                f"best_val_precision={best['val_precision']:.6f} "
                                f"best_val_f1={best['val_f1']:.6f} from epoch {best['epoch']:03d}"
                            )
                            break

            log_plain(f"saved checkpoint: {args.output}")
            log_plain(f"saved split manifest: {split_path}")
            log_plain(f"saved metrics: {metrics_path}")
            log_plain(f"saved val report: {val_report_path}")
            log_plain(f"saved log: {log_path}")
            if best["metrics"] is not None:
                best_metrics = best["metrics"]
                log_plain("FINAL_SUMMARY_BEGIN")
                log_plain(
                    "config "
                    f"history_steps={args.history_steps} batch_size={args.batch_size} lr={args.lr} "
                    f"dropout={args.dropout} weight_decay={args.weight_decay} "
                    f"pos_weight={meta.pos_weight:.4f} min_recall={args.threshold_min_recall:.2f} "
                    f"threshold_sweep={args.threshold_sweep_start:.2f}:{args.threshold_sweep_step:.2f}:{args.threshold_sweep_end:.2f}"
                )
                log_plain(
                    "data "
                    f"train_files={len(meta.train_files)} val_files={len(meta.val_files)} "
                    f"train_windows={meta.train_windows} val_windows={meta.val_windows} "
                    f"beam_positive_fraction={meta.beam_positive_fraction:.4f} "
                    f"window_positive_fraction={meta.sample_positive_fraction:.4f}"
                )
                log_plain(
                    "best "
                    f"epoch={best['epoch']} val_loss={best['val_loss']:.6f} "
                    f"val_precision={best['val_precision']:.6f} val_f1={best['val_f1']:.6f} "
                    f"threshold={float(best_metrics['threshold']):.2f} "
                    f"precision={float(best_metrics['precision']):.4f} "
                    f"recall={float(best_metrics['recall']):.4f} "
                    f"specificity={float(best_metrics['specificity']):.4f} "
                    f"fpr={float(best_metrics['false_positive_rate']):.4f}"
                )
                log_plain(
                    "best_confusion "
                    f"tn={int(best_metrics['tn'])} fp={int(best_metrics['fp'])} "
                    f"fn={int(best_metrics['fn'])} tp={int(best_metrics['tp'])}"
                )
                log_plain(
                    "artifacts "
                    f"checkpoint={args.output} split={split_path} metrics={metrics_path} val_report={val_report_path} log={log_path}"
                )
                log_plain("FINAL_SUMMARY_END")
        finally:
            set_log_file(None)


if __name__ == "__main__":
    main()
