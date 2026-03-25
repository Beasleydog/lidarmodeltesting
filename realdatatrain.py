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


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_type: str
    hidden_dim: int
    dropout: float
    lr: float
    weight_decay: float
    pos_weight_scale: float
    threshold_min_recall: float
    threshold_sweep_start: float
    threshold_sweep_end: float
    threshold_sweep_step: float
    loss_type: str = "bce"
    focal_gamma: float = 0.0
    label_smoothing: float = 0.0
    aux_hit_loss_weight: float = 0.0
    temporal_layers: int = 1
    transformer_layers: int = 2
    attention_heads: int = 4


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


class TemporalBeamTransformerClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_sensors: int,
        hidden_dim: int = 48,
        dropout: float = 0.15,
        temporal_layers: int = 1,
        transformer_layers: int = 2,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.model_type = "real_cleanlog_temporal_beam_transformer"
        self.in_channels = int(in_channels)
        self.num_sensors = int(num_sensors)
        self.hidden_dim = int(hidden_dim)
        self.temporal_layers = int(max(temporal_layers, 1))
        self.transformer_layers = int(max(transformer_layers, 1))
        self.attention_heads = int(max(attention_heads, 1))
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")

        sensor_geom = MODEL_SENSOR_YAW_PITCH_DEG[: self.num_sensors].astype(np.float32)
        sensor_geom_rad = np.deg2rad(sensor_geom)
        sensor_geom_feats = np.concatenate(
            [
                np.sin(sensor_geom_rad[:, :1]),
                np.cos(sensor_geom_rad[:, :1]),
                np.sin(sensor_geom_rad[:, 1:2]),
                np.cos(sensor_geom_rad[:, 1:2]),
            ],
            axis=1,
        ).astype(np.float32)
        self.register_buffer("sensor_geom_feats", torch.from_numpy(sensor_geom_feats), persistent=False)
        self.register_buffer("sensor_ids", torch.arange(self.num_sensors, dtype=torch.long), persistent=False)

        temporal_hidden = max(self.hidden_dim // 2, 16)
        self.temporal_input = nn.Sequential(
            nn.Linear(self.in_channels, temporal_hidden),
            nn.LayerNorm(temporal_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_gru = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            num_layers=self.temporal_layers,
            batch_first=True,
            dropout=dropout if self.temporal_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.current_mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.sensor_embedding = nn.Embedding(self.num_sensors, self.hidden_dim)
        self.sensor_geom_mlp = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.attention_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.beam_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
        self.beam_fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps, sensor_count = x.shape
        if sensor_count != self.num_sensors:
            raise ValueError(f"Expected {self.num_sensors} sensors, got {sensor_count}")

        beam_sequences = x.permute(0, 3, 2, 1).reshape(batch_size * sensor_count, time_steps, channels)
        temporal_inputs = self.temporal_input(beam_sequences)
        temporal_outputs, _ = self.temporal_gru(temporal_inputs)
        temporal_tokens = temporal_outputs[:, -1, :].reshape(batch_size, sensor_count, self.hidden_dim)

        current_features = x[:, :, -1, :].permute(0, 2, 1).reshape(batch_size * sensor_count, channels)
        current_tokens = self.current_mlp(current_features).reshape(batch_size, sensor_count, self.hidden_dim)

        beam_ids = self.sensor_ids.to(device=x.device)
        beam_embed = self.sensor_embedding(beam_ids).unsqueeze(0).expand(batch_size, -1, -1)
        geom_embed = self.sensor_geom_mlp(self.sensor_geom_feats.to(device=x.device)).unsqueeze(0).expand(batch_size, -1, -1)

        tokens = temporal_tokens + current_tokens + beam_embed + geom_embed
        encoded = self.beam_encoder(tokens)
        global_context = encoded.mean(dim=1, keepdim=True).expand(-1, sensor_count, -1)
        fused = self.beam_fuse(torch.cat([encoded, global_context], dim=-1))
        logits = self.head(fused).squeeze(-1)
        return logits


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
    def __init__(self, in_channels: int, num_sensors: int, base_channels: int = 32, dropout: float = 0.10):
        super().__init__()
        del num_sensors
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
        return self.head(d1).squeeze(1)[:, -1, :]


class TemporalBeamGRUClassifier(nn.Module):
    def __init__(self, in_channels: int, num_sensors: int, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.model_type = "real_cleanlog_temporal_beam_gru"
        self.in_channels = int(in_channels)
        self.num_sensors = int(num_sensors)
        self.hidden_dim = int(hidden_dim)
        temporal_hidden = max(self.hidden_dim // 2, 16)

        sensor_geom = MODEL_SENSOR_YAW_PITCH_DEG[: self.num_sensors].astype(np.float32)
        sensor_geom_rad = np.deg2rad(sensor_geom)
        sensor_geom_feats = np.concatenate(
            [
                np.sin(sensor_geom_rad[:, :1]),
                np.cos(sensor_geom_rad[:, :1]),
                np.sin(sensor_geom_rad[:, 1:2]),
                np.cos(sensor_geom_rad[:, 1:2]),
            ],
            axis=1,
        ).astype(np.float32)
        self.register_buffer("sensor_geom_feats", torch.from_numpy(sensor_geom_feats), persistent=False)
        self.register_buffer("sensor_ids", torch.arange(self.num_sensors, dtype=torch.long), persistent=False)

        self.temporal_input = nn.Sequential(
            nn.Linear(self.in_channels, temporal_hidden),
            nn.LayerNorm(temporal_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_gru = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.current_mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sensor_embedding = nn.Embedding(self.num_sensors, self.hidden_dim)
        self.sensor_geom_mlp = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps, sensor_count = x.shape
        beam_sequences = x.permute(0, 3, 2, 1).reshape(batch_size * sensor_count, time_steps, channels)
        temporal_inputs = self.temporal_input(beam_sequences)
        temporal_outputs, _ = self.temporal_gru(temporal_inputs)
        temporal_tokens = temporal_outputs[:, -1, :].reshape(batch_size, sensor_count, self.hidden_dim)

        current_features = x[:, :, -1, :].permute(0, 2, 1).reshape(batch_size * sensor_count, channels)
        current_tokens = self.current_mlp(current_features).reshape(batch_size, sensor_count, self.hidden_dim)
        beam_embed = self.sensor_embedding(self.sensor_ids.to(device=x.device)).unsqueeze(0).expand(batch_size, -1, -1)
        geom_embed = self.sensor_geom_mlp(self.sensor_geom_feats.to(device=x.device)).unsqueeze(0).expand(batch_size, -1, -1)

        fused = torch.cat([temporal_tokens + beam_embed + geom_embed, current_tokens], dim=-1)
        return self.head(fused).squeeze(-1)


class TemporalConvTransformerClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_sensors: int,
        hidden_dim: int = 96,
        dropout: float = 0.15,
        transformer_layers: int = 3,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.model_type = "real_cleanlog_temporal_conv_transformer"
        self.in_channels = int(in_channels)
        self.num_sensors = int(num_sensors)
        self.hidden_dim = int(hidden_dim)
        self.transformer_layers = int(max(transformer_layers, 1))
        self.attention_heads = int(max(attention_heads, 1))
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")

        sensor_geom = MODEL_SENSOR_YAW_PITCH_DEG[: self.num_sensors].astype(np.float32)
        sensor_geom_rad = np.deg2rad(sensor_geom)
        sensor_geom_feats = np.concatenate(
            [
                np.sin(sensor_geom_rad[:, :1]),
                np.cos(sensor_geom_rad[:, :1]),
                np.sin(sensor_geom_rad[:, 1:2]),
                np.cos(sensor_geom_rad[:, 1:2]),
            ],
            axis=1,
        ).astype(np.float32)
        self.register_buffer("sensor_geom_feats", torch.from_numpy(sensor_geom_feats), persistent=False)
        self.register_buffer("sensor_ids", torch.arange(self.num_sensors, dtype=torch.long), persistent=False)

        self.temporal_in = nn.Sequential(
            nn.Conv1d(self.in_channels, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if self.hidden_dim % 8 == 0 else 1, self.hidden_dim),
            nn.GELU(),
        )
        self.temporal_block1 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if self.hidden_dim % 8 == 0 else 1, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_block2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8 if self.hidden_dim % 8 == 0 else 1, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.current_mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sensor_embedding = nn.Embedding(self.num_sensors, self.hidden_dim)
        self.sensor_geom_mlp = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.attention_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.beam_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps, sensor_count = x.shape
        beam_sequences = x.permute(0, 3, 1, 2).reshape(batch_size * sensor_count, channels, time_steps)
        temporal = self.temporal_in(beam_sequences)
        temporal = temporal + self.temporal_block1(temporal)
        temporal = temporal + self.temporal_block2(temporal)
        temporal_tokens = temporal[:, :, -1].reshape(batch_size, sensor_count, self.hidden_dim)

        current_features = x[:, :, -1, :].permute(0, 2, 1).reshape(batch_size * sensor_count, channels)
        current_tokens = self.current_mlp(current_features).reshape(batch_size, sensor_count, self.hidden_dim)
        beam_embed = self.sensor_embedding(self.sensor_ids.to(device=x.device)).unsqueeze(0).expand(batch_size, -1, -1)
        geom_embed = self.sensor_geom_mlp(self.sensor_geom_feats.to(device=x.device)).unsqueeze(0).expand(batch_size, -1, -1)

        encoded = self.beam_encoder(temporal_tokens + current_tokens + beam_embed + geom_embed)
        global_context = encoded.mean(dim=1, keepdim=True).expand(-1, sensor_count, -1)
        return self.head(torch.cat([encoded, global_context], dim=-1)).squeeze(-1)


def _compute_local_beam_geometry(
    x: torch.Tensor,
    max_range_cm: float,
    relative_xy_scale_cm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    range_cm = torch.clamp(x[:, 0], 0.0, 1.0) * float(max_range_cm)
    hit = x[:, 1] > 0.5
    rel_x_global = x[:, 5] * float(relative_xy_scale_cm)
    rel_y_global = x[:, 6] * float(relative_xy_scale_cm)
    heading = torch.atan2(x[:, 8], x[:, 9])
    sensor_yaw = torch.atan2(x[:, 14], x[:, 15])
    beam_yaw_global = heading + sensor_yaw

    endpoint_x_global_rel = rel_x_global + range_cm * torch.cos(beam_yaw_global)
    endpoint_y_global_rel = rel_y_global + range_cm * torch.sin(beam_yaw_global)

    current_heading = heading[:, -1:, :1]
    cos_h = torch.cos(current_heading)
    sin_h = torch.sin(current_heading)
    origin_x = cos_h * rel_x_global + sin_h * rel_y_global
    origin_y = -sin_h * rel_x_global + cos_h * rel_y_global
    endpoint_x = cos_h * endpoint_x_global_rel + sin_h * endpoint_y_global_rel
    endpoint_y = -sin_h * endpoint_x_global_rel + cos_h * endpoint_y_global_rel
    dir_x = endpoint_x - origin_x
    dir_y = endpoint_y - origin_y
    norm = torch.clamp(torch.sqrt(dir_x.square() + dir_y.square()), min=1.0)
    dir_x = dir_x / norm
    dir_y = dir_y / norm
    time_age = 0.5 * (x[:, 19] + 1.0)
    return origin_x, origin_y, endpoint_x, endpoint_y, dir_x, dir_y, hit, time_age


def _rasterize_bev_evidence(
    origin_x: torch.Tensor,
    origin_y: torch.Tensor,
    endpoint_x: torch.Tensor,
    endpoint_y: torch.Tensor,
    hit: torch.Tensor,
    time_age: torch.Tensor,
    grid_size: int,
    grid_extent_cm: float,
    ray_samples: int = 12,
) -> torch.Tensor:
    batch_size, time_steps, sensor_count = endpoint_x.shape
    grid_size_i = int(grid_size)
    extent = float(grid_extent_cm)

    batch_idx = torch.arange(batch_size, device=endpoint_x.device).view(batch_size, 1, 1).expand_as(endpoint_x)

    def scatter_channel(mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        flat = torch.zeros(batch_size * grid_size_i * grid_size_i, device=endpoint_x.device, dtype=torch.float32)
        if torch.any(mask):
            flat.scatter_add_(0, lin[mask], weight[mask].to(dtype=torch.float32))
        return flat.view(batch_size, 1, grid_size_i, grid_size_i)

    def to_lin(x_cm: torch.Tensor, y_cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm_x = torch.clamp(x_cm / extent, -1.0, 1.0)
        norm_y = torch.clamp(-y_cm / extent, -1.0, 1.0)
        ix = torch.round((norm_x + 1.0) * 0.5 * float(grid_size_i - 1)).long()
        iy = torch.round((norm_y + 1.0) * 0.5 * float(grid_size_i - 1)).long()
        valid = (norm_x >= -1.0) & (norm_x <= 1.0) & (norm_y >= -1.0) & (norm_y <= 1.0)
        lin = batch_idx * (grid_size_i * grid_size_i) + iy * grid_size_i + ix
        return lin, valid

    recency = torch.clamp(time_age, 0.05, 1.0).to(dtype=torch.float32)
    endpoint_lin, valid = to_lin(endpoint_x, endpoint_y)
    lin = endpoint_lin
    current_mask = torch.zeros_like(valid)
    current_mask[:, -1, :] = True

    occupied = scatter_channel(valid & hit, recency)
    current_occupied = scatter_channel(valid & current_mask & hit, torch.ones_like(recency))
    recent_occupied = scatter_channel(valid & hit, recency.square())
    nohit_endpoints = scatter_channel(valid & (~hit), 0.5 * recency)

    fractions = torch.linspace(0.1, 0.95, steps=max(int(ray_samples), 2), device=endpoint_x.device, dtype=torch.float32)
    free = torch.zeros(batch_size, 1, grid_size_i, grid_size_i, device=endpoint_x.device, dtype=torch.float32)
    current_free = torch.zeros_like(free)
    recent_free = torch.zeros_like(free)
    known = occupied + nohit_endpoints
    for frac in fractions:
        sample_x = origin_x + frac * (endpoint_x - origin_x)
        sample_y = origin_y + frac * (endpoint_y - origin_y)
        lin, ray_valid = to_lin(sample_x, sample_y)
        free_mask = ray_valid
        weight = recency * (1.1 - 0.5 * frac)
        if torch.any(free_mask):
            flat = torch.zeros(batch_size * grid_size_i * grid_size_i, device=endpoint_x.device, dtype=torch.float32)
            flat.scatter_add_(0, lin[free_mask], weight[free_mask])
            free = free + flat.view(batch_size, 1, grid_size_i, grid_size_i)

            flat_current = torch.zeros_like(flat)
            current_free_mask = free_mask & current_mask
            if torch.any(current_free_mask):
                flat_current.scatter_add_(0, lin[current_free_mask], torch.ones_like(weight[current_free_mask]))
            current_free = current_free + flat_current.view(batch_size, 1, grid_size_i, grid_size_i)

            flat_recent = torch.zeros_like(flat)
            flat_recent.scatter_add_(0, lin[free_mask], (weight[free_mask] * weight[free_mask]))
            recent_free = recent_free + flat_recent.view(batch_size, 1, grid_size_i, grid_size_i)
    known = known + free
    return torch.cat(
        [
            occupied,
            free,
            current_occupied,
            current_free,
            recent_occupied,
            recent_free,
            nohit_endpoints,
            known,
        ],
        dim=1,
    )


def _sample_bev_features(
    bev_features: torch.Tensor,
    sample_x_cm: torch.Tensor,
    sample_y_cm: torch.Tensor,
    grid_extent_cm: float,
) -> torch.Tensor:
    extent = float(grid_extent_cm)
    norm_x = torch.clamp(sample_x_cm / extent, -1.0, 1.0)
    norm_y = torch.clamp(-sample_y_cm / extent, -1.0, 1.0)
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(bev_features, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(-1).permute(0, 2, 1)


def _sample_bev_ray_profile(
    bev_features: torch.Tensor,
    dir_x: torch.Tensor,
    dir_y: torch.Tensor,
    max_range_cm: float,
    grid_extent_cm: float,
    num_samples: int = 10,
) -> torch.Tensor:
    fractions = torch.linspace(
        0.1,
        1.0,
        steps=max(int(num_samples), 2),
        device=bev_features.device,
        dtype=torch.float32,
    ).view(1, 1, -1, 1)
    sample_x = dir_x.unsqueeze(-1) * float(max_range_cm) * fractions.squeeze(-1)
    sample_y = dir_y.unsqueeze(-1) * float(max_range_cm) * fractions.squeeze(-1)
    extent = float(grid_extent_cm)
    norm_x = torch.clamp(sample_x / extent, -1.0, 1.0)
    norm_y = torch.clamp(-sample_y / extent, -1.0, 1.0)
    grid = torch.stack([norm_x, norm_y], dim=-1)
    sampled = F.grid_sample(bev_features, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.permute(0, 2, 3, 1)


def _flatten_time_sensor_rays(values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    batch_size, time_steps, sensor_count = values.shape
    flat = values.reshape(batch_size, time_steps * sensor_count)
    return flat, (batch_size, time_steps, sensor_count)


class TemporalBEVUNetClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_sensors: int,
        hidden_dim: int = 96,
        dropout: float = 0.15,
        max_range_cm: float = DEFAULT_LIDAR_MAX_RANGE_CM,
        relative_xy_scale_cm: float = 5000.0,
        grid_size: int = 64,
    ):
        super().__init__()
        self.model_type = "real_cleanlog_temporal_bev_unet"
        self.in_channels = int(in_channels)
        self.num_sensors = int(num_sensors)
        self.hidden_dim = int(hidden_dim)
        self.max_range_cm = float(max_range_cm)
        self.relative_xy_scale_cm = float(relative_xy_scale_cm)
        self.grid_size = int(grid_size)
        self.grid_extent_cm = float(max(self.max_range_cm * 0.9, 6000.0))

        bev_channels = max(self.hidden_dim // 2, 32)
        self.bev_in = ConvBlock(8, bev_channels)
        self.bev_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bev_mid = ConvBlock(bev_channels, bev_channels * 2)
        self.bev_bottleneck = ConvBlock(bev_channels * 2, bev_channels * 2)
        self.bev_out = ConvBlock(bev_channels * 4, bev_channels)
        self.drop = nn.Dropout2d(dropout)

        self.current_mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sensor_embedding = nn.Embedding(self.num_sensors, self.hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.bev_proj = nn.Sequential(
            nn.Linear(bev_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, sensor_count = x.shape
        if sensor_count != self.num_sensors:
            raise ValueError(f"Expected {self.num_sensors} sensors, got {sensor_count}")

        origin_x, origin_y, endpoint_x, endpoint_y, dir_x, dir_y, hit, time_age = _compute_local_beam_geometry(
            x,
            max_range_cm=self.max_range_cm,
            relative_xy_scale_cm=self.relative_xy_scale_cm,
        )
        bev = _rasterize_bev_evidence(
            origin_x,
            origin_y,
            endpoint_x,
            endpoint_y,
            hit,
            time_age,
            self.grid_size,
            self.grid_extent_cm,
        )
        e1 = self.bev_in(bev)
        e2 = self.bev_mid(self.bev_pool(self.drop(e1)))
        b = self.bev_bottleneck(self.bev_pool(self.drop(e2)))
        up = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        up = torch.cat([up, e2], dim=1)
        bev_features = self.bev_out(F.interpolate(up, size=e1.shape[-2:], mode="bilinear", align_corners=False))

        current_endpoint = self.bev_proj(_sample_bev_features(bev_features, endpoint_x[:, -1, :], endpoint_y[:, -1, :], self.grid_extent_cm))
        ray_profile = _sample_bev_ray_profile(
            bev_features,
            dir_x[:, -1, :],
            dir_y[:, -1, :],
            max_range_cm=self.max_range_cm,
            grid_extent_cm=self.grid_extent_cm,
        )
        ray_mean = self.bev_proj(ray_profile.mean(dim=2))
        ray_max = self.bev_proj(ray_profile.max(dim=2).values)
        global_bev = self.bev_proj(bev_features.mean(dim=(-2, -1))).unsqueeze(1).expand(-1, sensor_count, -1)

        current_features = x[:, :, -1, :].permute(0, 2, 1)
        current_tokens = self.current_mlp(current_features)
        beam_ids = torch.arange(self.num_sensors, device=x.device)
        beam_embed = self.sensor_embedding(beam_ids).unsqueeze(0).expand(batch_size, -1, -1)
        fused = torch.cat([current_endpoint + ray_mean, ray_max, current_tokens + beam_embed, global_bev], dim=-1)
        return self.head(fused).squeeze(-1)


class TemporalBEVFusionClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_sensors: int,
        hidden_dim: int = 128,
        dropout: float = 0.18,
        transformer_layers: int = 3,
        attention_heads: int = 4,
        max_range_cm: float = DEFAULT_LIDAR_MAX_RANGE_CM,
        relative_xy_scale_cm: float = 5000.0,
        grid_size: int = 64,
    ):
        super().__init__()
        self.model_type = "real_cleanlog_temporal_bev_fusion"
        self.in_channels = int(in_channels)
        self.num_sensors = int(num_sensors)
        self.hidden_dim = int(hidden_dim)
        self.transformer_layers = int(max(transformer_layers, 1))
        self.attention_heads = int(max(attention_heads, 1))
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.max_range_cm = float(max_range_cm)
        self.relative_xy_scale_cm = float(relative_xy_scale_cm)
        self.grid_size = int(grid_size)
        self.grid_extent_cm = float(max(self.max_range_cm * 0.9, 6000.0))

        bev_channels = max(self.hidden_dim // 2, 48)
        self.bev_stem = nn.Sequential(
            nn.Conv2d(8, bev_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if bev_channels % 8 == 0 else 1, bev_channels),
            nn.GELU(),
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if bev_channels % 8 == 0 else 1, bev_channels),
            nn.GELU(),
        )
        self.bev_block = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8 if bev_channels % 8 == 0 else 1, bev_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if bev_channels % 8 == 0 else 1, bev_channels),
            nn.GELU(),
        )
        self.bev_proj = nn.Sequential(
            nn.Linear(bev_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.current_mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sensor_embedding = nn.Embedding(self.num_sensors, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.attention_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.beam_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.aux_hit_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, _, _, sensor_count = x.shape
        if sensor_count != self.num_sensors:
            raise ValueError(f"Expected {self.num_sensors} sensors, got {sensor_count}")

        origin_x, origin_y, endpoint_x, endpoint_y, dir_x, dir_y, hit, time_age = _compute_local_beam_geometry(
            x,
            max_range_cm=self.max_range_cm,
            relative_xy_scale_cm=self.relative_xy_scale_cm,
        )
        bev = _rasterize_bev_evidence(
            origin_x,
            origin_y,
            endpoint_x,
            endpoint_y,
            hit,
            time_age,
            self.grid_size,
            self.grid_extent_cm,
        )
        bev_features = self.bev_stem(bev)
        bev_features = bev_features + self.bev_block(bev_features)

        endpoint_tokens = self.bev_proj(
            _sample_bev_features(bev_features, endpoint_x[:, -1, :], endpoint_y[:, -1, :], self.grid_extent_cm)
        )
        ray_profile = _sample_bev_ray_profile(
            bev_features,
            dir_x[:, -1, :],
            dir_y[:, -1, :],
            max_range_cm=self.max_range_cm,
            grid_extent_cm=self.grid_extent_cm,
        )
        ray_mean = self.bev_proj(ray_profile.mean(dim=2))
        ray_max = self.bev_proj(ray_profile.max(dim=2).values)
        global_tokens = self.bev_proj(bev_features.mean(dim=(-2, -1))).unsqueeze(1).expand(-1, sensor_count, -1)
        current_tokens = self.current_mlp(x[:, :, -1, :].permute(0, 2, 1))
        beam_ids = torch.arange(self.num_sensors, device=x.device)
        beam_embed = self.sensor_embedding(beam_ids).unsqueeze(0).expand(batch_size, -1, -1)
        tokens = endpoint_tokens + ray_mean + current_tokens + beam_embed
        encoded = self.beam_encoder(tokens)
        logits = self.head(torch.cat([encoded, ray_max, global_tokens], dim=-1)).squeeze(-1)

        flat_endpoint_x, aux_shape = _flatten_time_sensor_rays(endpoint_x)
        flat_endpoint_y, _ = _flatten_time_sensor_rays(endpoint_y)
        flat_dir_x, _ = _flatten_time_sensor_rays(dir_x)
        flat_dir_y, _ = _flatten_time_sensor_rays(dir_y)
        aux_endpoint = self.bev_proj(_sample_bev_features(bev_features, flat_endpoint_x, flat_endpoint_y, self.grid_extent_cm))
        aux_profile = _sample_bev_ray_profile(
            bev_features,
            flat_dir_x,
            flat_dir_y,
            max_range_cm=self.max_range_cm,
            grid_extent_cm=self.grid_extent_cm,
        )
        aux_ray_mean = self.bev_proj(aux_profile.mean(dim=2))
        aux_logits = self.aux_hit_head(torch.cat([aux_endpoint, aux_ray_mean], dim=-1)).squeeze(-1)
        aux_logits = aux_logits.reshape(aux_shape)
        return {"logits": logits, "aux_hit_logits": aux_logits}


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


def compute_experiment_loss(
    outputs: torch.Tensor | dict[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight_t: torch.Tensor,
    exp: ExperimentConfig,
    training: bool,
) -> torch.Tensor:
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    smooth = float(exp.label_smoothing) if training else 0.0
    if smooth > 0.0:
        clamped = min(max(smooth, 0.0), 0.49)
        loss_targets = targets * (1.0 - clamped) + (1.0 - targets) * clamped
    else:
        loss_targets = targets

    per_entry = F.binary_cross_entropy_with_logits(
        logits,
        loss_targets,
        pos_weight=pos_weight_t,
        reduction="none",
    )
    if exp.loss_type != "focal":
        main_loss = per_entry.mean()
    else:
        gamma = max(float(exp.focal_gamma), 0.0)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = torch.pow(torch.clamp(1.0 - pt, min=1e-6), gamma)
        main_loss = (per_entry * focal_weight).mean()

    aux_weight = float(exp.aux_hit_loss_weight)
    if aux_weight <= 0.0 or not isinstance(outputs, dict) or "aux_hit_logits" not in outputs:
        return main_loss

    aux_targets = inputs[:, 1]
    aux_logits = outputs["aux_hit_logits"]
    aux_loss = F.binary_cross_entropy_with_logits(aux_logits, aux_targets, reduction="mean")
    return main_loss + aux_weight * aux_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    device_kind: str,
    pos_weight: float,
    exp: ExperimentConfig,
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
            outputs = model(x)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = compute_experiment_loss(outputs, x, y, pos_weight_t, exp, training=False)

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
            outputs = model(x)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
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
    exp: ExperimentConfig,
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
            outputs = model(x)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = compute_experiment_loss(outputs, x, y, pos_weight_t, exp, training=True)

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


def build_full_experiment_suite() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="bev_fusion_occ_balanced",
            model_type="bev_fusion",
            hidden_dim=160,
            dropout=0.18,
            lr=6.5e-4,
            weight_decay=2.5e-4,
            pos_weight_scale=0.95,
            threshold_min_recall=0.55,
            threshold_sweep_start=0.80,
            threshold_sweep_end=0.98,
            threshold_sweep_step=0.01,
            transformer_layers=4,
            attention_heads=8,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_balanced_aux025",
            model_type="bev_fusion",
            hidden_dim=160,
            dropout=0.18,
            lr=6.5e-4,
            weight_decay=2.5e-4,
            pos_weight_scale=0.95,
            threshold_min_recall=0.55,
            threshold_sweep_start=0.80,
            threshold_sweep_end=0.98,
            threshold_sweep_step=0.01,
            transformer_layers=4,
            attention_heads=8,
            aux_hit_loss_weight=0.25,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_balanced_aux050",
            model_type="bev_fusion",
            hidden_dim=160,
            dropout=0.18,
            lr=6.5e-4,
            weight_decay=2.5e-4,
            pos_weight_scale=0.95,
            threshold_min_recall=0.55,
            threshold_sweep_start=0.80,
            threshold_sweep_end=0.98,
            threshold_sweep_step=0.01,
            transformer_layers=4,
            attention_heads=8,
            aux_hit_loss_weight=0.50,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_recall",
            model_type="bev_fusion",
            hidden_dim=160,
            dropout=0.18,
            lr=6e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.10,
            threshold_min_recall=0.65,
            threshold_sweep_start=0.65,
            threshold_sweep_end=0.92,
            threshold_sweep_step=0.02,
            transformer_layers=4,
            attention_heads=8,
            label_smoothing=0.02,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_recall_aux025",
            model_type="bev_fusion",
            hidden_dim=160,
            dropout=0.18,
            lr=6e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.10,
            threshold_min_recall=0.65,
            threshold_sweep_start=0.65,
            threshold_sweep_end=0.92,
            threshold_sweep_step=0.02,
            transformer_layers=4,
            attention_heads=8,
            label_smoothing=0.02,
            aux_hit_loss_weight=0.25,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_recall_aux050",
            model_type="bev_fusion",
            hidden_dim=160,
            dropout=0.18,
            lr=6e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.10,
            threshold_min_recall=0.65,
            threshold_sweep_start=0.65,
            threshold_sweep_end=0.92,
            threshold_sweep_step=0.02,
            transformer_layers=4,
            attention_heads=8,
            label_smoothing=0.02,
            aux_hit_loss_weight=0.50,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_recall_harder",
            model_type="bev_fusion",
            hidden_dim=160,
            dropout=0.18,
            lr=5.5e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.20,
            threshold_min_recall=0.70,
            threshold_sweep_start=0.55,
            threshold_sweep_end=0.88,
            threshold_sweep_step=0.02,
            transformer_layers=4,
            attention_heads=8,
            label_smoothing=0.02,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_deep",
            model_type="bev_fusion",
            hidden_dim=192,
            dropout=0.20,
            lr=5.5e-4,
            weight_decay=3e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.60,
            threshold_sweep_start=0.72,
            threshold_sweep_end=0.96,
            threshold_sweep_step=0.02,
            transformer_layers=5,
            attention_heads=8,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_deep_aux025",
            model_type="bev_fusion",
            hidden_dim=192,
            dropout=0.20,
            lr=5.5e-4,
            weight_decay=3e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.60,
            threshold_sweep_start=0.72,
            threshold_sweep_end=0.96,
            threshold_sweep_step=0.02,
            transformer_layers=5,
            attention_heads=8,
            aux_hit_loss_weight=0.25,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_deep_aux050",
            model_type="bev_fusion",
            hidden_dim=192,
            dropout=0.20,
            lr=5.5e-4,
            weight_decay=3e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.60,
            threshold_sweep_start=0.72,
            threshold_sweep_end=0.96,
            threshold_sweep_step=0.02,
            transformer_layers=5,
            attention_heads=8,
            aux_hit_loss_weight=0.50,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_deep_recall",
            model_type="bev_fusion",
            hidden_dim=192,
            dropout=0.20,
            lr=5.0e-4,
            weight_decay=2.5e-4,
            pos_weight_scale=1.15,
            threshold_min_recall=0.68,
            threshold_sweep_start=0.58,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.02,
            transformer_layers=5,
            attention_heads=8,
            label_smoothing=0.02,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_focal_light",
            model_type="bev_fusion",
            hidden_dim=160,
            dropout=0.18,
            lr=6.5e-4,
            weight_decay=2.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.60,
            threshold_sweep_start=0.72,
            threshold_sweep_end=0.96,
            threshold_sweep_step=0.02,
            transformer_layers=4,
            attention_heads=8,
            loss_type="focal",
            focal_gamma=1.0,
        ),
        ExperimentConfig(
            name="bev_fusion_occ_smooth_heavy",
            model_type="bev_fusion",
            hidden_dim=192,
            dropout=0.20,
            lr=5.5e-4,
            weight_decay=2.5e-4,
            pos_weight_scale=1.05,
            threshold_min_recall=0.65,
            threshold_sweep_start=0.62,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.02,
            transformer_layers=5,
            attention_heads=8,
            label_smoothing=0.04,
        ),
    ]


def build_quick_experiment_suite() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="quick_beam_gru_baseline",
            model_type="beam_gru",
            hidden_dim=96,
            dropout=0.12,
            lr=9e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.55,
            threshold_sweep_start=0.60,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.05,
        ),
        ExperimentConfig(
            name="quick_beam_transformer_small",
            model_type="beam_transformer",
            hidden_dim=96,
            dropout=0.12,
            lr=8e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.55,
            threshold_sweep_start=0.60,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.05,
            temporal_layers=1,
            transformer_layers=2,
            attention_heads=4,
        ),
        ExperimentConfig(
            name="quick_conv_transformer_small",
            model_type="conv_transformer",
            hidden_dim=96,
            dropout=0.12,
            lr=8e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.55,
            threshold_sweep_start=0.60,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.05,
            transformer_layers=2,
            attention_heads=4,
        ),
        ExperimentConfig(
            name="quick_bev_unet_small",
            model_type="bev_unet",
            hidden_dim=96,
            dropout=0.12,
            lr=8e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.55,
            threshold_sweep_start=0.60,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.05,
        ),
        ExperimentConfig(
            name="quick_bev_fusion_small",
            model_type="bev_fusion",
            hidden_dim=96,
            dropout=0.12,
            lr=8e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.55,
            threshold_sweep_start=0.60,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.05,
            transformer_layers=2,
            attention_heads=4,
        ),
        ExperimentConfig(
            name="quick_bev_fusion_recall",
            model_type="bev_fusion",
            hidden_dim=96,
            dropout=0.14,
            lr=7e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.15,
            threshold_min_recall=0.65,
            threshold_sweep_start=0.50,
            threshold_sweep_end=0.85,
            threshold_sweep_step=0.05,
            transformer_layers=2,
            attention_heads=4,
            label_smoothing=0.02,
        ),
        ExperimentConfig(
            name="quick_bev_fusion_aux",
            model_type="bev_fusion",
            hidden_dim=96,
            dropout=0.12,
            lr=8e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.55,
            threshold_sweep_start=0.60,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.05,
            transformer_layers=2,
            attention_heads=4,
            aux_hit_loss_weight=0.25,
        ),
        ExperimentConfig(
            name="quick_bev_fusion_focal",
            model_type="bev_fusion",
            hidden_dim=96,
            dropout=0.12,
            lr=8e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.60,
            threshold_sweep_start=0.55,
            threshold_sweep_end=0.85,
            threshold_sweep_step=0.05,
            loss_type="focal",
            focal_gamma=1.5,
            transformer_layers=2,
            attention_heads=4,
        ),
    ]


def build_focused_experiment_suite() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="focused_beam_transformer_h128",
            model_type="beam_transformer",
            hidden_dim=128,
            dropout=0.12,
            lr=7e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.58,
            threshold_sweep_start=0.70,
            threshold_sweep_end=0.88,
            threshold_sweep_step=0.03,
            temporal_layers=1,
            transformer_layers=3,
            attention_heads=4,
        ),
        ExperimentConfig(
            name="focused_conv_transformer_h128",
            model_type="conv_transformer",
            hidden_dim=128,
            dropout=0.12,
            lr=7e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.60,
            threshold_sweep_start=0.76,
            threshold_sweep_end=0.94,
            threshold_sweep_step=0.03,
            transformer_layers=3,
            attention_heads=4,
        ),
        ExperimentConfig(
            name="focused_bev_fusion_small_h96",
            model_type="bev_fusion",
            hidden_dim=96,
            dropout=0.12,
            lr=8e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.58,
            threshold_sweep_start=0.72,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.03,
            transformer_layers=2,
            attention_heads=4,
        ),
        ExperimentConfig(
            name="focused_bev_fusion_recall_h96",
            model_type="bev_fusion",
            hidden_dim=96,
            dropout=0.14,
            lr=7e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.15,
            threshold_min_recall=0.64,
            threshold_sweep_start=0.62,
            threshold_sweep_end=0.82,
            threshold_sweep_step=0.03,
            transformer_layers=2,
            attention_heads=4,
            label_smoothing=0.02,
        ),
        ExperimentConfig(
            name="focused_bev_fusion_small_h128",
            model_type="bev_fusion",
            hidden_dim=128,
            dropout=0.12,
            lr=7e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.58,
            threshold_sweep_start=0.72,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.03,
            transformer_layers=3,
            attention_heads=4,
        ),
        ExperimentConfig(
            name="focused_bev_fusion_recall_h128",
            model_type="bev_fusion",
            hidden_dim=128,
            dropout=0.14,
            lr=6.5e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.12,
            threshold_min_recall=0.64,
            threshold_sweep_start=0.60,
            threshold_sweep_end=0.80,
            threshold_sweep_step=0.03,
            transformer_layers=3,
            attention_heads=4,
            label_smoothing=0.02,
        ),
        ExperimentConfig(
            name="focused_bev_fusion_recall_aux_h128",
            model_type="bev_fusion",
            hidden_dim=128,
            dropout=0.14,
            lr=6.5e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.12,
            threshold_min_recall=0.64,
            threshold_sweep_start=0.60,
            threshold_sweep_end=0.80,
            threshold_sweep_step=0.03,
            transformer_layers=3,
            attention_heads=4,
            label_smoothing=0.02,
            aux_hit_loss_weight=0.15,
        ),
        ExperimentConfig(
            name="focused_bev_fusion_precision_h128",
            model_type="bev_fusion",
            hidden_dim=128,
            dropout=0.10,
            lr=6.5e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.05,
            threshold_min_recall=0.58,
            threshold_sweep_start=0.74,
            threshold_sweep_end=0.92,
            threshold_sweep_step=0.03,
            transformer_layers=3,
            attention_heads=4,
            label_smoothing=0.01,
        ),
        ExperimentConfig(
            name="focused_bev_fusion_focal_h128",
            model_type="bev_fusion",
            hidden_dim=128,
            dropout=0.12,
            lr=6.5e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.05,
            threshold_min_recall=0.62,
            threshold_sweep_start=0.64,
            threshold_sweep_end=0.84,
            threshold_sweep_step=0.03,
            loss_type="focal",
            focal_gamma=1.25,
            transformer_layers=3,
            attention_heads=4,
        ),
    ]


def build_refine_experiment_suite() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="refine_bev_fusion_precision_h96",
            model_type="bev_fusion",
            hidden_dim=96,
            dropout=0.12,
            lr=7.5e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.00,
            threshold_min_recall=0.58,
            threshold_sweep_start=0.76,
            threshold_sweep_end=0.90,
            threshold_sweep_step=0.02,
            transformer_layers=2,
            attention_heads=4,
        ),
        ExperimentConfig(
            name="refine_bev_fusion_recall_h96",
            model_type="bev_fusion",
            hidden_dim=96,
            dropout=0.14,
            lr=6.8e-4,
            weight_decay=1.5e-4,
            pos_weight_scale=1.15,
            threshold_min_recall=0.64,
            threshold_sweep_start=0.64,
            threshold_sweep_end=0.78,
            threshold_sweep_step=0.02,
            transformer_layers=2,
            attention_heads=4,
            label_smoothing=0.02,
        ),
        ExperimentConfig(
            name="refine_bev_fusion_precision_h128",
            model_type="bev_fusion",
            hidden_dim=128,
            dropout=0.10,
            lr=6.2e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.04,
            threshold_min_recall=0.58,
            threshold_sweep_start=0.80,
            threshold_sweep_end=0.92,
            threshold_sweep_step=0.02,
            transformer_layers=3,
            attention_heads=4,
            label_smoothing=0.01,
        ),
        ExperimentConfig(
            name="refine_bev_fusion_recall_aux_h128",
            model_type="bev_fusion",
            hidden_dim=128,
            dropout=0.14,
            lr=6.2e-4,
            weight_decay=2e-4,
            pos_weight_scale=1.10,
            threshold_min_recall=0.64,
            threshold_sweep_start=0.66,
            threshold_sweep_end=0.80,
            threshold_sweep_step=0.02,
            transformer_layers=3,
            attention_heads=4,
            label_smoothing=0.02,
            aux_hit_loss_weight=0.10,
        ),
    ]


def build_experiment_suite(profile: str) -> list[ExperimentConfig]:
    normalized = str(profile).strip().lower()
    if normalized == "quick":
        return build_quick_experiment_suite()
    if normalized == "focused":
        return build_focused_experiment_suite()
    if normalized == "refine":
        return build_refine_experiment_suite()
    if normalized == "full":
        return build_full_experiment_suite()
    raise ValueError(f"Unknown suite profile {profile!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a real cleanlog beam classifier on the last N lidar frames plus the current frame."
    )
    parser.add_argument("--cleanlog-dir", type=Path, default=Path("cleanlog"))
    parser.add_argument("--output", type=Path, default=Path("runs/realdatabeam_transformer.pt"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--history-steps", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
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
    parser.add_argument("--suite-profile", choices=("quick", "focused", "refine", "full"), default="full")
    parser.add_argument("--suite-epochs", type=int, default=20)
    parser.add_argument("--suite-early-stop-patience", type=int, default=7)
    return parser.parse_args()


def _build_model_for_experiment(
    exp: ExperimentConfig,
    args: argparse.Namespace,
    input_channels: int,
    num_sensors: int,
) -> nn.Module:
    if exp.model_type == "conv_unet":
        return SmallTemporalUNet(
            in_channels=input_channels,
            num_sensors=num_sensors,
            base_channels=exp.hidden_dim,
            dropout=exp.dropout,
        )
    if exp.model_type == "beam_gru":
        return TemporalBeamGRUClassifier(
            in_channels=input_channels,
            num_sensors=num_sensors,
            hidden_dim=exp.hidden_dim,
            dropout=exp.dropout,
        )
    if exp.model_type == "beam_transformer":
        return TemporalBeamTransformerClassifier(
            in_channels=input_channels,
            num_sensors=num_sensors,
            hidden_dim=exp.hidden_dim,
            dropout=exp.dropout,
            temporal_layers=exp.temporal_layers,
            transformer_layers=exp.transformer_layers,
            attention_heads=exp.attention_heads,
        )
    if exp.model_type == "conv_transformer":
        return TemporalConvTransformerClassifier(
            in_channels=input_channels,
            num_sensors=num_sensors,
            hidden_dim=exp.hidden_dim,
            dropout=exp.dropout,
            transformer_layers=exp.transformer_layers,
            attention_heads=exp.attention_heads,
        )
    if exp.model_type == "bev_unet":
        return TemporalBEVUNetClassifier(
            in_channels=input_channels,
            num_sensors=num_sensors,
            hidden_dim=exp.hidden_dim,
            dropout=exp.dropout,
            max_range_cm=args.max_range_cm,
            relative_xy_scale_cm=args.relative_xy_scale_cm,
        )
    if exp.model_type == "bev_fusion":
        return TemporalBEVFusionClassifier(
            in_channels=input_channels,
            num_sensors=num_sensors,
            hidden_dim=exp.hidden_dim,
            dropout=exp.dropout,
            transformer_layers=exp.transformer_layers,
            attention_heads=exp.attention_heads,
            max_range_cm=args.max_range_cm,
            relative_xy_scale_cm=args.relative_xy_scale_cm,
        )
    raise ValueError(f"Unknown model_type {exp.model_type!r}")


def _build_threshold_values(start: float, end: float, step: float) -> list[float]:
    values = np.arange(float(start), float(end) + (0.5 * float(step)), float(step), dtype=np.float32)
    values = np.clip(values, 0.0, 1.0)
    return sorted(set(float(round(v, 6)) for v in values.tolist()))


def run_experiment(
    exp: ExperimentConfig,
    args: argparse.Namespace,
    device: torch.device,
    device_kind: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    meta: DatasetMeta,
    sensor_names: tuple[str, ...],
    input_channels: int,
) -> dict[str, object]:
    output_base = args.output.parent / f"{args.output.stem}.{exp.name}"
    ckpt_path = output_base.with_suffix(".pt")
    val_report_path = output_base.with_suffix(".val_report.json")
    metrics_path = output_base.with_suffix(".metrics.csv")

    model = _build_model_for_experiment(
        exp,
        args=args,
        input_channels=input_channels,
        num_sensors=meta.num_sensors,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        min_lr=args.min_lr,
    )
    threshold_values = _build_threshold_values(
        exp.threshold_sweep_start,
        exp.threshold_sweep_end,
        exp.threshold_sweep_step,
    )
    scaled_pos_weight = max(1.0, float(meta.pos_weight) * float(exp.pos_weight_scale))

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
                "val_loss",
                "val_acc",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_specificity",
                "val_fpr",
                "val_threshold",
            ]
        )

        for epoch in range(1, args.suite_epochs + 1):
            train_metrics = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                exp=exp,
                device=device,
                device_kind=device_kind,
                epoch_idx=epoch,
                total_epochs=args.suite_epochs,
                pos_weight=scaled_pos_weight,
                grad_clip_norm=args.grad_clip_norm,
                use_amp=bool(args.amp),
                log_every_batches=0,
            )
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                device_kind=device_kind,
                pos_weight=scaled_pos_weight,
                exp=exp,
                decision_threshold=float(exp.threshold_sweep_start),
            )
            sweep_metrics = sweep_validation_thresholds(
                model=model,
                loader=val_loader,
                device=device,
                device_kind=device_kind,
                thresholds=threshold_values,
                min_recall=float(exp.threshold_min_recall),
            )
            val_metrics.update(
                {
                    "accuracy": sweep_metrics["accuracy"],
                    "precision": sweep_metrics["precision"],
                    "recall": sweep_metrics["recall"],
                    "f1": sweep_metrics["f1"],
                    "specificity": sweep_metrics["specificity"],
                    "false_positive_rate": sweep_metrics["false_positive_rate"],
                    "threshold": sweep_metrics["threshold"],
                    "tn": sweep_metrics["tn"],
                    "fp": sweep_metrics["fp"],
                    "fn": sweep_metrics["fn"],
                    "tp": sweep_metrics["tp"],
                    "confusion_matrix": sweep_metrics["confusion_matrix"],
                }
            )
            scheduler.step(float(val_metrics["loss"]))

            writer.writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    train_metrics["accuracy"],
                    train_metrics["precision"],
                    train_metrics["recall"],
                    train_metrics["f1"],
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    val_metrics["precision"],
                    val_metrics["recall"],
                    val_metrics["f1"],
                    val_metrics["specificity"],
                    val_metrics["false_positive_rate"],
                    val_metrics["threshold"],
                ]
            )
            metrics_fh.flush()

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
                    "metrics": dict(val_metrics),
                }
                epochs_without_improve = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "model_type": model.model_type,
                        "hidden_dim": exp.hidden_dim,
                        "dropout": exp.dropout,
                        "loss_type": exp.loss_type,
                        "focal_gamma": exp.focal_gamma,
                        "label_smoothing": exp.label_smoothing,
                        "num_sensors": meta.num_sensors,
                        "history_steps": meta.history_steps,
                    },
                    "meta": {
                        "experiment": exp.name,
                        "sensor_names": list(sensor_names),
                        "best_val_precision": best["val_precision"],
                        "best_val_f1": best["val_f1"],
                        "best_threshold": float(val_metrics["threshold"]),
                        "best_epoch": best["epoch"],
                    },
                }
                save_checkpoint_for_device(checkpoint, ckpt_path, device_kind)
                with val_report_path.open("w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "experiment": exp.name,
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
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= int(args.suite_early_stop_patience):
                    break

    if best["metrics"] is None:
        raise RuntimeError(f"Experiment {exp.name} produced no best metrics")

    return {
        "name": exp.name,
        "model_type": exp.model_type,
        "params": param_count,
        "hidden_dim": exp.hidden_dim,
        "dropout": exp.dropout,
        "lr": exp.lr,
        "weight_decay": exp.weight_decay,
        "pos_weight_scale": exp.pos_weight_scale,
        "loss_type": exp.loss_type,
        "focal_gamma": exp.focal_gamma,
        "label_smoothing": exp.label_smoothing,
        "aux_hit_loss_weight": exp.aux_hit_loss_weight,
        "threshold_min_recall": exp.threshold_min_recall,
        "threshold_range": [exp.threshold_sweep_start, exp.threshold_sweep_end, exp.threshold_sweep_step],
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "best_val_precision": float(best["val_precision"]),
        "best_val_f1": float(best["val_f1"]),
        "best_metrics": best["metrics"],
        "checkpoint": str(ckpt_path),
        "metrics_csv": str(metrics_path),
        "val_report": str(val_report_path),
    }


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
            log("Starting real cleanlog temporal beam-transformer trainer")
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
            log("Running experiment suite with compact output; per-batch logs are suppressed.")
            experiments = build_experiment_suite(args.suite_profile)
            log(
                "Suite config: "
                f"profile={args.suite_profile} experiments={len(experiments)} suite_epochs={args.suite_epochs} "
                f"early_stop_patience={args.suite_early_stop_patience}"
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
            suite_results: list[dict[str, object]] = []
            failed_results: list[dict[str, object]] = []
            for exp in experiments:
                log(
                    f"Experiment start: {exp.name} model={exp.model_type} hidden={exp.hidden_dim} "
                    f"dropout={exp.dropout:.2f} lr={exp.lr:.5f} pos_weight_scale={exp.pos_weight_scale:.2f} "
                    f"loss={exp.loss_type} gamma={exp.focal_gamma:.2f} smooth={exp.label_smoothing:.3f} "
                    f"aux_hit={exp.aux_hit_loss_weight:.2f} "
                    f"recall_floor={exp.threshold_min_recall:.2f}"
                )
                try:
                    result = run_experiment(
                        exp=exp,
                        args=args,
                        device=device,
                        device_kind=device_kind,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        meta=meta,
                        sensor_names=sensor_names,
                        input_channels=input_channels,
                    )
                except Exception as exc:
                    error_result = {
                        "name": exp.name,
                        "model_type": exp.model_type,
                        "hidden_dim": exp.hidden_dim,
                        "dropout": exp.dropout,
                        "lr": exp.lr,
                        "weight_decay": exp.weight_decay,
                        "pos_weight_scale": exp.pos_weight_scale,
                        "loss_type": exp.loss_type,
                        "focal_gamma": exp.focal_gamma,
                        "label_smoothing": exp.label_smoothing,
                        "threshold_min_recall": exp.threshold_min_recall,
                        "threshold_range": [
                            exp.threshold_sweep_start,
                            exp.threshold_sweep_end,
                            exp.threshold_sweep_step,
                        ],
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failed_results.append(error_result)
                    log(
                        f"Experiment failed: {exp.name} "
                        f"type={type(exc).__name__} message={str(exc)}"
                    )
                    continue

                result["status"] = "ok"
                suite_results.append(result)
                best_metrics = result["best_metrics"]
                log(
                    f"Experiment done: {result['name']} "
                    f"precision={float(result['best_val_precision']):.4f} "
                    f"f1={float(result['best_val_f1']):.4f} "
                    f"recall={float(best_metrics['recall']):.4f} "
                    f"fpr={float(best_metrics['false_positive_rate']):.4f} "
                    f"threshold={float(best_metrics['threshold']):.2f}"
                )

            suite_json_path = args.output.parent / f"{args.output.stem}.suite_results.json"
            with suite_json_path.open("w", encoding="utf-8") as fh:
                json.dump({"successful": suite_results, "failed": failed_results}, fh, indent=2)

            if not suite_results:
                log_plain(f"saved split manifest: {split_path}")
                log_plain(f"saved suite results: {suite_json_path}")
                log_plain(f"saved log: {log_path}")
                log_plain("FINAL_SUITE_SUMMARY_BEGIN")
                log_plain(
                    "suite "
                    f"experiments={len(experiments)} completed=0 failed={len(failed_results)} "
                    f"suite_epochs={args.suite_epochs} early_stop_patience={args.suite_early_stop_patience}"
                )
                log_plain(
                    "data "
                    f"train_files={len(meta.train_files)} val_files={len(meta.val_files)} "
                    f"train_windows={meta.train_windows} val_windows={meta.val_windows} "
                    f"beam_positive_fraction={meta.beam_positive_fraction:.4f} "
                    f"window_positive_fraction={meta.sample_positive_fraction:.4f}"
                )
                for failed in failed_results:
                    log_plain(
                        f"failed name={failed['name']} model={failed['model_type']} "
                        f"error_type={failed['error_type']} error={failed['error']}"
                    )
                log_plain(f"artifacts suite={suite_json_path} split={split_path} log={log_path}")
                log_plain("FINAL_SUITE_SUMMARY_END")
                return

            sorted_by_precision = sorted(
                suite_results,
                key=lambda r: (
                    float(r["best_val_precision"]),
                    float(r["best_val_f1"]),
                    -float(r["best_metrics"]["false_positive_rate"]),
                ),
                reverse=True,
            )
            sorted_by_f1 = sorted(
                suite_results,
                key=lambda r: (
                    float(r["best_val_f1"]),
                    float(r["best_val_precision"]),
                ),
                reverse=True,
            )
            best_precision = sorted_by_precision[0]
            best_f1 = sorted_by_f1[0]
            log_plain(f"saved split manifest: {split_path}")
            log_plain(f"saved suite results: {suite_json_path}")
            log_plain(f"saved log: {log_path}")
            log_plain("FINAL_SUITE_SUMMARY_BEGIN")
            log_plain(
                "suite "
                f"experiments={len(experiments)} completed={len(suite_results)} failed={len(failed_results)} "
                f"suite_epochs={args.suite_epochs} early_stop_patience={args.suite_early_stop_patience}"
            )
            log_plain(
                "data "
                f"train_files={len(meta.train_files)} val_files={len(meta.val_files)} "
                f"train_windows={meta.train_windows} val_windows={meta.val_windows} "
                f"beam_positive_fraction={meta.beam_positive_fraction:.4f} "
                f"window_positive_fraction={meta.sample_positive_fraction:.4f}"
            )
            log_plain(
                "best_precision "
                f"name={best_precision['name']} model={best_precision['model_type']} params={best_precision['params']} "
                f"precision={float(best_precision['best_val_precision']):.4f} "
                f"f1={float(best_precision['best_val_f1']):.4f} "
                f"recall={float(best_precision['best_metrics']['recall']):.4f} "
                f"fpr={float(best_precision['best_metrics']['false_positive_rate']):.4f} "
                f"threshold={float(best_precision['best_metrics']['threshold']):.2f} "
                f"loss={best_precision['loss_type']} gamma={best_precision['focal_gamma']:.2f} "
                f"smooth={best_precision['label_smoothing']:.3f} "
                f"aux_hit={best_precision['aux_hit_loss_weight']:.2f}"
            )
            log_plain(
                "best_f1 "
                f"name={best_f1['name']} model={best_f1['model_type']} params={best_f1['params']} "
                f"precision={float(best_f1['best_val_precision']):.4f} "
                f"f1={float(best_f1['best_val_f1']):.4f} "
                f"recall={float(best_f1['best_metrics']['recall']):.4f} "
                f"fpr={float(best_f1['best_metrics']['false_positive_rate']):.4f} "
                f"threshold={float(best_f1['best_metrics']['threshold']):.2f} "
                f"loss={best_f1['loss_type']} gamma={best_f1['focal_gamma']:.2f} "
                f"smooth={best_f1['label_smoothing']:.3f} "
                f"aux_hit={best_f1['aux_hit_loss_weight']:.2f}"
            )
            for rank, result in enumerate(sorted_by_precision, start=1):
                metrics = result["best_metrics"]
                log_plain(
                    f"rank {rank:02d} "
                    f"name={result['name']} model={result['model_type']} params={result['params']} "
                    f"epoch={result['best_epoch']} prec={float(result['best_val_precision']):.4f} "
                    f"f1={float(result['best_val_f1']):.4f} "
                    f"recall={float(metrics['recall']):.4f} "
                    f"spec={float(metrics['specificity']):.4f} "
                    f"fpr={float(metrics['false_positive_rate']):.4f} "
                    f"thr={float(metrics['threshold']):.2f} "
                    f"cfg=modeltype:{result['model_type']},hidden:{result['hidden_dim']},dropout:{result['dropout']},"
                    f"lr:{result['lr']},wd:{result['weight_decay']},pws:{result['pos_weight_scale']},"
                    f"loss:{result['loss_type']},gamma:{result['focal_gamma']},smooth:{result['label_smoothing']},"
                    f"aux_hit:{result['aux_hit_loss_weight']},"
                    f"recall_floor:{result['threshold_min_recall']}"
                )
            for failed in failed_results:
                log_plain(
                    f"failed name={failed['name']} model={failed['model_type']} "
                    f"error_type={failed['error_type']} error={failed['error']}"
                )
            log_plain(f"artifacts suite={suite_json_path} split={split_path} log={log_path}")
            log_plain("FINAL_SUITE_SUMMARY_END")
        finally:
            set_log_file(None)


if __name__ == "__main__":
    main()
