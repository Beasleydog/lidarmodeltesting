from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable, TextIO

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass(frozen=True)
class WorldData:
    pose: np.ndarray
    lidar_cm: np.ndarray
    lidar_class: np.ndarray
    teleport_flag: np.ndarray


@dataclass(frozen=True)
class NormStats:
    pose_mean: np.ndarray
    pose_std: np.ndarray
    dist_mean: float
    dist_std: float


_LOG_FILE: TextIO | None = None


def set_log_file(log_file: TextIO | None) -> None:
    global _LOG_FILE
    _LOG_FILE = log_file


def _emit_log_line(line: str) -> None:
    print(line, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    _emit_log_line(f"[{stamp}] {message}")


def log_plain(message: str) -> None:
    _emit_log_line(message)


def select_runtime_device(preferred: str | None = None) -> tuple[torch.device, str]:
    choice = "auto" if preferred is None else str(preferred).strip().lower()
    valid = {"auto", "cpu", "cuda", "mps", "xla"}
    if choice not in valid:
        raise ValueError(f"Unsupported device selection '{preferred}'. Choose from {sorted(valid)}")

    if choice in {"auto", "xla"}:
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device(), "xla"
        except Exception:
            if choice == "xla":
                raise

    if choice in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if choice in {"auto", "mps"} and mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"

    if choice == "auto" or choice == "cpu":
        return torch.device("cpu"), "cpu"

    raise RuntimeError(f"Requested device '{choice}' is not available in this environment")


def optimizer_step_for_device(optimizer: torch.optim.Optimizer, device_kind: str) -> None:
    if device_kind == "xla":
        import torch_xla.core.xla_model as xm

        xm.optimizer_step(optimizer, barrier=False)
        xm.mark_step()
        return
    optimizer.step()


def save_checkpoint_for_device(checkpoint: dict, path: Path, device_kind: str) -> None:
    if device_kind == "xla":
        import torch_xla.core.xla_model as xm

        xm.save(checkpoint, path)
        return
    torch.save(checkpoint, path)


def load_checkpoint_for_device(path: str | Path, device: torch.device, device_kind: str) -> dict:
    map_location: str | torch.device = "cpu" if device_kind == "xla" else device
    return torch.load(path, map_location=map_location)


def _split_single_world_for_validation(world: WorldData, val_fraction: float) -> tuple[WorldData, WorldData]:
    t_steps = world.pose.shape[0]
    if t_steps < 2:
        # Not enough timeline for a clean temporal split; fallback to shared world.
        return world, world
    split_idx = int(round(t_steps * (1.0 - val_fraction)))
    split_idx = max(1, min(t_steps - 1, split_idx))
    train_world = WorldData(
        pose=world.pose[:split_idx].copy(),
        lidar_cm=world.lidar_cm[:split_idx].copy(),
        lidar_class=world.lidar_class[:split_idx].copy(),
        teleport_flag=world.teleport_flag[:split_idx].copy(),
    )
    val_world = WorldData(
        pose=world.pose[split_idx:].copy(),
        lidar_cm=world.lidar_cm[split_idx:].copy(),
        lidar_class=world.lidar_class[split_idx:].copy(),
        teleport_flag=world.teleport_flag[split_idx:].copy(),
    )
    return train_world, val_world


def _sensor_keys(fieldnames: list[str], prefix: str) -> list[str]:
    keys = [k for k in fieldnames if k.startswith(prefix)]
    keys.sort(key=lambda k: int(k.rsplit("_", 1)[1]))
    return keys


def load_world_file(path: Path) -> WorldData:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        cm_keys = _sensor_keys(reader.fieldnames, "lidar_cm_")
        class_keys = _sensor_keys(reader.fieldnames, "lidar_class_")
        if len(cm_keys) == 0 or len(cm_keys) != len(class_keys):
            raise ValueError(f"{path} has invalid lidar columns")

        pose_rows: list[list[float]] = []
        cm_rows: list[list[float]] = []
        cls_rows: list[list[int]] = []
        teleport_rows: list[int] = []
        has_teleport_col = "teleport_flag" in reader.fieldnames
        for row in reader:
            pose_rows.append(
                [
                    float(row["x_cm"]),
                    float(row["y_cm"]),
                    float(row["z_cm"]),
                    float(row["yaw_deg"]),
                ]
            )
            cm_rows.append([float(row[k]) for k in cm_keys])
            cls_rows.append([int(row[k]) for k in class_keys])
            teleport_rows.append(int(float(row["teleport_flag"])) if has_teleport_col else 0)

    if len(pose_rows) < 1:
        raise ValueError(f"{path} needs at least 1 timestep for current-step classification")
    return WorldData(
        pose=np.asarray(pose_rows, dtype=np.float32),
        lidar_cm=np.asarray(cm_rows, dtype=np.float32),
        lidar_class=np.asarray(cls_rows, dtype=np.int64),
        teleport_flag=np.asarray(teleport_rows, dtype=np.int64),
    )


def compute_norm_stats(worlds: Iterable[WorldData]) -> NormStats:
    pose_all = np.concatenate([w.pose for w in worlds], axis=0)
    pose_mean = pose_all.mean(axis=0)
    pose_std = pose_all.std(axis=0) + 1e-6

    dists = np.concatenate([w.lidar_cm for w in worlds], axis=0)
    valid = dists >= 0.0
    if np.any(valid):
        dist_mean = float(dists[valid].mean())
        dist_std = float(dists[valid].std() + 1e-6)
    else:
        dist_mean = 0.0
        dist_std = 1.0
    return NormStats(
        pose_mean=pose_mean.astype(np.float32),
        pose_std=pose_std.astype(np.float32),
        dist_mean=dist_mean,
        dist_std=dist_std,
    )


def world_to_features(world: WorldData, stats: NormStats) -> tuple[np.ndarray, np.ndarray]:
    pose = (world.pose - stats.pose_mean[None, :]) / stats.pose_std[None, :]

    hit = (world.lidar_cm >= 0.0).astype(np.float32)
    dist = np.where(hit > 0.0, world.lidar_cm, 0.0)
    dist = (dist - stats.dist_mean) / stats.dist_std
    dist = np.where(hit > 0.0, dist, 0.0)

    features = np.concatenate([pose, dist.astype(np.float32), hit], axis=1)
    targets = world.lidar_class.astype(np.int64)
    return features, targets


def featurize_timestep(pose_xyzyaw: np.ndarray, lidar_cm: np.ndarray, stats: NormStats) -> np.ndarray:
    pose = (pose_xyzyaw.astype(np.float32) - stats.pose_mean) / stats.pose_std
    hit = (lidar_cm >= 0.0).astype(np.float32)
    dist = np.where(hit > 0.0, lidar_cm.astype(np.float32), 0.0)
    dist = (dist - stats.dist_mean) / stats.dist_std
    dist = np.where(hit > 0.0, dist, 0.0)
    return np.concatenate([pose, dist, hit], axis=0).astype(np.float32)


class SequencePieceDataset(Dataset):
    def __init__(
        self,
        features_by_world: list[np.ndarray],
        targets_by_world: list[np.ndarray],
        teleport_flags_by_world: list[np.ndarray],
        max_history: int,
        histories_per_target: int = 3,
        exclude_after_teleport_steps: int = 1,
        seed: int = 0,
    ):
        self.features_by_world = features_by_world
        self.targets_by_world = targets_by_world
        self.histories_per_target = int(max(histories_per_target, 1))
        self.exclude_after_teleport_steps = int(max(exclude_after_teleport_steps, 0))
        self.index: list[tuple[int, int, int]] = []
        self.sample_has_obstacle_target: list[bool] = []
        rng = np.random.default_rng(seed)

        for wi, feat in enumerate(features_by_world):
            t_steps = feat.shape[0]
            tele_flags = teleport_flags_by_world[wi]
            invalid_targets = np.zeros((t_steps,), dtype=bool)
            tele_indices = np.flatnonzero(tele_flags > 0)
            for idx in tele_indices:
                start_excl = int(idx)
                end_excl = min(t_steps - 1, int(idx + self.exclude_after_teleport_steps))
                invalid_targets[start_excl : end_excl + 1] = True
            for end in range(t_steps):
                if invalid_targets[end]:
                    continue
                # Sequence is [start:end+1), predict class at end.
                history_max = end + 1 if max_history <= 0 else min(end + 1, max_history)
                history_lengths = self._sample_history_lengths(history_max, self.histories_per_target, rng)
                for hist_len in history_lengths:
                    start = end + 1 - hist_len
                    self.index.append((wi, start, end))
                    self.sample_has_obstacle_target.append(bool(np.any(self.targets_by_world[wi][end] == 1)))

    @staticmethod
    def _sample_history_lengths(history_max: int, k: int, rng: np.random.Generator) -> list[int]:
        if history_max <= 0:
            return []
        if history_max <= k:
            return list(range(1, history_max + 1))
        if k <= 1:
            return [history_max]

        # Always include shortest and longest contexts.
        lengths = [1, history_max]
        remaining = k - len(lengths)
        if remaining <= 0:
            return sorted(set(lengths))

        candidates = np.arange(2, history_max, dtype=np.int32)
        if candidates.size > 0:
            sampled = rng.choice(candidates, size=min(remaining, candidates.size), replace=False)
            lengths.extend(int(v) for v in sampled.tolist())
        return sorted(set(lengths))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wi, start, end = self.index[idx]
        x = self.features_by_world[wi][start : end + 1]
        y = self.targets_by_world[wi][end]
        return torch.from_numpy(x), torch.from_numpy(y)

    def obstacle_target_mask(self) -> np.ndarray:
        return np.asarray(self.sample_has_obstacle_target, dtype=bool)


def collate_padded(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([x.shape[0] for x, _ in batch], dtype=torch.long)
    feat_dim = batch[0][0].shape[1]
    max_len = int(lengths.max().item())
    x_pad = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    y = torch.stack([yb for _, yb in batch], dim=0).long()

    for i, (xb, _) in enumerate(batch):
        x_pad[i, : xb.shape[0], :] = xb
    return x_pad, lengths, y


def _default_region_sensor_groups(num_sensors: int) -> dict[str, list[int]]:
    if int(num_sensors) == 17:
        return {
            "front": [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16],
            "left": [7],
            "right": [8],
            "rear": [9, 10, 11, 12],
        }
    chunks = np.array_split(np.arange(int(num_sensors), dtype=np.int64), 4)
    return {
        "front": [int(v) for v in chunks[0].tolist()],
        "left": [int(v) for v in chunks[1].tolist()],
        "right": [int(v) for v in chunks[2].tolist()],
        "rear": [int(v) for v in chunks[3].tolist()],
    }


def _normalize_region_sensor_groups(
    region_sensor_groups: dict[str, list[int]] | None,
    num_sensors: int,
) -> dict[str, list[int]]:
    base = _default_region_sensor_groups(num_sensors) if region_sensor_groups is None else region_sensor_groups
    ordered_names = ("front", "left", "right", "rear")
    normalized: dict[str, list[int]] = {}
    used: set[int] = set()
    for name in ordered_names:
        raw = base.get(name, [])
        ids: list[int] = []
        for idx in raw:
            idx_i = int(idx)
            if 0 <= idx_i < int(num_sensors) and idx_i not in used:
                ids.append(idx_i)
                used.add(idx_i)
        if ids:
            normalized[name] = ids

    remaining = [i for i in range(int(num_sensors)) if i not in used]
    if remaining:
        normalized.setdefault("front", []).extend(remaining)
    if not normalized:
        normalized["front"] = list(range(int(num_sensors)))
    return normalized


def _default_region_hidden_dims(
    region_sensor_groups: dict[str, list[int]],
    hidden_dim: int,
) -> dict[str, int]:
    active_items = [(name, ids) for name, ids in region_sensor_groups.items() if ids]
    min_hidden = 8
    if not active_items:
        return {"front": max(min_hidden, int(hidden_dim))}

    target_budget = max(min_hidden * len(active_items), int(round(max(int(hidden_dim), 1) * 1.0)))
    weights = np.sqrt(np.asarray([len(ids) for _, ids in active_items], dtype=np.float32))
    weights /= max(float(weights.sum()), 1e-8)

    region_hidden_dims: dict[str, int] = {}
    for (name, _), weight in zip(active_items, weights):
        region_hidden_dims[name] = max(min_hidden, int(round(target_budget * float(weight))))
    return region_hidden_dims


class GRULidarClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_sensors: int,
        num_classes: int,
        dropout: float,
        region_sensor_groups: dict[str, list[int]] | None = None,
        region_hidden_dims: dict[str, int] | None = None,
        fusion_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.model_type = "regional_gru"
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_sensors = int(num_sensors)
        self.num_classes = int(num_classes)

        if self.input_dim != 4 + 2 * self.num_sensors:
            raise ValueError(
                f"regional GRU expects input_dim=4+2*num_sensors; got input_dim={self.input_dim} "
                f"for num_sensors={self.num_sensors}"
            )

        self.region_sensor_groups = _normalize_region_sensor_groups(region_sensor_groups, self.num_sensors)
        if region_hidden_dims is None:
            region_hidden_dims = _default_region_hidden_dims(self.region_sensor_groups, self.hidden_dim)
        self.region_hidden_dims = {
            name: max(4, int(region_hidden_dims.get(name, 8)))
            for name in self.region_sensor_groups.keys()
        }
        self.fusion_hidden_dim = (
            max(48, int(round(self.hidden_dim * 0.75)))
            if fusion_hidden_dim is None
            else max(8, int(fusion_hidden_dim))
        )
        self.region_names = list(self.region_sensor_groups.keys())

        self.region_grus = nn.ModuleDict()
        total_region_dim = 0
        for name in self.region_names:
            sensor_ids = self.region_sensor_groups[name]
            region_input_dim = 4 + 2 * len(sensor_ids)
            region_hidden = self.region_hidden_dims[name]
            self.region_grus[name] = nn.GRU(
                input_size=region_input_dim,
                hidden_size=region_hidden,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=dropout if self.num_layers > 1 else 0.0,
            )
            total_region_dim += region_hidden

        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion = nn.Linear(total_region_dim, self.fusion_hidden_dim)
        self.head = nn.Linear(self.fusion_hidden_dim, self.num_sensors * self.num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pose = x[:, :, :4]
        dist = x[:, :, 4 : 4 + self.num_sensors]
        hit = x[:, :, 4 + self.num_sensors : 4 + 2 * self.num_sensors]

        region_states: list[torch.Tensor] = []
        for name in self.region_names:
            sensor_ids = self.region_sensor_groups[name]
            x_region = torch.cat([pose, dist[:, :, sensor_ids], hit[:, :, sensor_ids]], dim=-1)
            packed = pack_padded_sequence(x_region, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h_n = self.region_grus[name](packed)
            region_states.append(h_n[-1])

        fused = torch.cat(region_states, dim=-1)
        fused = self.fusion_dropout(fused)
        fused = F.gelu(self.fusion(fused))
        logits = self.head(self.fusion_dropout(fused))
        return logits.view(-1, self.num_sensors, self.num_classes)
class GRULidarInferencer:
    def __init__(
        self,
        model: GRULidarClassifier,
        stats: NormStats,
        device: torch.device,
        num_sensors: int,
        input_dim: int,
        max_history: int = 32,
    ):
        self.model = model.eval()
        self.stats = stats
        self.device = device
        self.num_sensors = int(num_sensors)
        self.input_dim = int(input_dim)
        self.max_history = int(max_history)

    def featurize_timestep(self, pose_xyzyaw: np.ndarray, lidar_cm: np.ndarray) -> np.ndarray:
        return featurize_timestep(pose_xyzyaw, lidar_cm, self.stats)

    def predict_current_from_feature_history(self, feature_history: np.ndarray) -> np.ndarray:
        if feature_history.ndim != 2:
            raise ValueError("feature_history must be shape [T, F]")
        if feature_history.shape[1] != self.input_dim:
            raise ValueError(f"feature_history feature dim {feature_history.shape[1]} != expected {self.input_dim}")
        if feature_history.shape[0] < 1:
            raise ValueError("feature_history must have at least 1 timestep")

        if self.max_history > 0 and feature_history.shape[0] > self.max_history:
            feature_history = feature_history[-self.max_history :]

        x = torch.from_numpy(feature_history.astype(np.float32)).unsqueeze(0).to(self.device)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x, lengths)
            pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)
        if pred.shape[0] != self.num_sensors:
            raise RuntimeError(f"predicted {pred.shape[0]} sensors, expected {self.num_sensors}")
        return pred

    def predict_current_from_history(self, pose_history: np.ndarray, lidar_cm_history: np.ndarray) -> np.ndarray:
        if pose_history.ndim != 2 or pose_history.shape[1] != 4:
            raise ValueError("pose_history must be shape [T,4] for x,y,z,yaw")
        if lidar_cm_history.ndim != 2 or lidar_cm_history.shape[1] != self.num_sensors:
            raise ValueError(f"lidar_cm_history must be shape [T,{self.num_sensors}]")
        if pose_history.shape[0] != lidar_cm_history.shape[0]:
            raise ValueError("pose_history and lidar_cm_history must have matching T")
        if pose_history.shape[0] < 1:
            raise ValueError("history must have at least 1 timestep")

        feats = np.stack(
            [self.featurize_timestep(pose_history[i], lidar_cm_history[i]) for i in range(pose_history.shape[0])],
            axis=0,
        )
        return self.predict_current_from_feature_history(feats)

    def predict_next_from_feature_history(self, feature_history: np.ndarray) -> np.ndarray:
        # Backward-compatible alias retained for callers that still use the old name.
        return self.predict_current_from_feature_history(feature_history)

    def predict_next_from_history(self, pose_history: np.ndarray, lidar_cm_history: np.ndarray) -> np.ndarray:
        # Backward-compatible alias retained for callers that still use the old name.
        return self.predict_current_from_history(pose_history, lidar_cm_history)


def load_gru_lidar_inferencer(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    max_history: int | None = None,
) -> GRULidarInferencer:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if device is None:
        device_t, device_kind = select_runtime_device("auto")
    elif isinstance(device, str) and device.strip().lower() in {"auto", "cpu", "cuda", "mps", "xla"}:
        device_t, device_kind = select_runtime_device(device)
    else:
        device_t = torch.device(device)
        device_kind = str(device_t.type)

    ckpt = load_checkpoint_for_device(checkpoint_path, device_t, device_kind)
    cfg = ckpt["model_config"]
    model = GRULidarClassifier(
        input_dim=int(cfg["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        num_sensors=int(cfg["num_sensors"]),
        num_classes=int(cfg["num_classes"]),
        dropout=float(cfg["dropout"]),
        region_sensor_groups=cfg.get("region_sensor_groups"),
        region_hidden_dims=cfg.get("region_hidden_dims"),
        fusion_hidden_dim=cfg.get("fusion_hidden_dim"),
    ).to(device_t)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm = ckpt["norm"]
    stats = NormStats(
        pose_mean=np.asarray(norm["pose_mean"], dtype=np.float32),
        pose_std=np.asarray(norm["pose_std"], dtype=np.float32),
        dist_mean=float(norm["dist_mean"]),
        dist_std=float(norm["dist_std"]),
    )
    if max_history is None:
        max_history = int(ckpt.get("meta", {}).get("max_history", 32))
    return GRULidarInferencer(
        model=model,
        stats=stats,
        device=device_t,
        num_sensors=int(cfg["num_sensors"]),
        input_dim=int(cfg["input_dim"]),
        max_history=int(max_history),
    )


def focal_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    # Standard multi-class focal loss applied per lidar ray.
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    focal = (1.0 - pt).clamp(min=0.0).pow(float(max(gamma, 0.0)))

    nll = -log_pt
    if class_weights is not None:
        sample_w = class_weights.to(logits.device).gather(0, targets.reshape(-1)).reshape_as(targets)
        nll = nll * sample_w
    return (focal * nll).mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    device_kind: str,
    num_classes: int,
    epoch_idx: int,
    total_epochs: int,
    phase_name: str,
    log_every_batches: int,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 0.0,
    focal_gamma: float = 2.0,
) -> tuple[float, float]:
    def fmt_acc(correct: int, total: int) -> str:
        if total <= 0:
            return "n/a"
        return f"{(correct / total):.3f}"

    is_train = optimizer is not None
    model.train(is_train)
    use_focal = float(focal_gamma) > 0.0
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(max(label_smoothing, 0.0)))

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    seen_sequences = 0
    class_totals = np.zeros((num_classes,), dtype=np.int64)
    class_correct = np.zeros((num_classes,), dtype=np.int64)

    batch_count = len(loader)
    phase_t0 = perf_counter()
    for batch_idx, (x, lengths, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        logits = model(x, lengths)
        logits_flat = logits.reshape(-1, num_classes)
        y_flat = y.reshape(-1)
        if use_focal:
            loss = focal_cross_entropy_loss(
                logits_flat,
                y_flat,
                gamma=focal_gamma,
                class_weights=class_weights,
            )
        else:
            loss = criterion(logits_flat, y_flat)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer_step_for_device(optimizer, device_kind)

        preds = torch.argmax(logits, dim=-1)
        total_correct += int((preds == y).sum().item())
        total_count += int(y.numel())
        batch_sequences = int(y.shape[0])
        total_loss += float(loss.item()) * float(batch_sequences)
        seen_sequences += batch_sequences

        y_flat = y.reshape(-1).detach().cpu().numpy()
        p_flat = preds.reshape(-1).detach().cpu().numpy()
        for cls_id in range(num_classes):
            cls_mask = y_flat == cls_id
            cls_count = int(np.sum(cls_mask))
            if cls_count > 0:
                class_totals[cls_id] += cls_count
                class_correct[cls_id] += int(np.sum(p_flat[cls_mask] == cls_id))

        if log_every_batches > 0 and (batch_idx % log_every_batches == 0 or batch_idx == batch_count):
            running_loss = total_loss / max(seen_sequences, 1)
            running_acc = total_correct / max(total_count, 1)
            obstacle_acc = fmt_acc(int(class_correct[1]), int(class_totals[1]))
            ground_acc = fmt_acc(int(class_correct[0]), int(class_totals[0]))
            nothing_acc = fmt_acc(int(class_correct[2]), int(class_totals[2]))
            log(
                f"{phase_name} epoch {epoch_idx:03d}/{total_epochs:03d} "
                f"batch {batch_idx:04d}/{batch_count:04d} "
                f"loss={running_loss:.5f} acc={running_acc:.4f} "
                f"class_acc(o/g/n)={obstacle_acc}/{ground_acc}/{nothing_acc}"
            )

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = total_correct / max(total_count, 1)
    phase_s = perf_counter() - phase_t0
    log(
        f"{phase_name} epoch {epoch_idx:03d}/{total_epochs:03d} complete "
        f"loss={avg_loss:.5f} acc={acc:.4f} time={phase_s:.2f}s"
    )
    return avg_loss, acc


def evaluate_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    device_kind: str,
    num_classes: int,
) -> dict:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=-1)

            y_flat = y.reshape(-1).cpu().numpy()
            p_flat = preds.reshape(-1).cpu().numpy()
            for yt, yp in zip(y_flat, p_flat):
                confusion[int(yt), int(yp)] += 1
            if device_kind == "xla":
                import torch_xla.core.xla_model as xm

                xm.mark_step()

    per_class_recall = []
    for c in range(num_classes):
        denom = int(confusion[c, :].sum())
        per_class_recall.append(float(confusion[c, c]) / max(denom, 1))

    return {
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": per_class_recall,
    }


def compute_class_weights(class_counts: np.ndarray) -> np.ndarray:
    counts = np.clip(class_counts.astype(np.float64), 1.0, None)
    # Tempered class balancing:
    # Use inverse-sqrt frequency (less aggressive than inverse frequency) and
    # cap the maximum weight so rare classes do not dominate loss.
    weights = 1.0 / np.sqrt(counts)
    weights /= np.mean(weights)
    max_weight = 1.8
    weights = np.clip(weights, 1e-8, max_weight)
    weights /= np.mean(weights)
    return weights.astype(np.float32)


def compute_obstacle_oversample_weights(
    dataset: SequencePieceDataset,
    target_fraction: float,
) -> tuple[np.ndarray, dict]:
    total = len(dataset)
    if total <= 0:
        return np.zeros((0,), dtype=np.float64), {
            "enabled": False,
            "num_samples": 0,
            "obstacle_samples": 0,
            "non_obstacle_samples": 0,
            "natural_obstacle_fraction": 0.0,
            "sampled_obstacle_fraction": 0.0,
        }

    mask = dataset.obstacle_target_mask()
    obstacle_count = int(mask.sum())
    non_obstacle_count = int(total - obstacle_count)
    natural_fraction = float(obstacle_count) / float(total)

    if obstacle_count == 0 or non_obstacle_count == 0 or target_fraction <= 0.0:
        return np.ones((total,), dtype=np.float64), {
            "enabled": False,
            "num_samples": total,
            "obstacle_samples": obstacle_count,
            "non_obstacle_samples": non_obstacle_count,
            "natural_obstacle_fraction": natural_fraction,
            "sampled_obstacle_fraction": natural_fraction,
        }

    target_fraction = float(np.clip(target_fraction, 1e-3, 1.0 - 1e-3))
    obstacle_weight = target_fraction / float(obstacle_count)
    non_obstacle_weight = (1.0 - target_fraction) / float(non_obstacle_count)
    weights = np.where(mask, obstacle_weight, non_obstacle_weight).astype(np.float64)
    weights /= max(weights.mean(), 1e-12)
    return weights, {
        "enabled": True,
        "num_samples": total,
        "obstacle_samples": obstacle_count,
        "non_obstacle_samples": non_obstacle_count,
        "natural_obstacle_fraction": natural_fraction,
        "sampled_obstacle_fraction": target_fraction,
    }


def build_loaders(
    data_dir: Path,
    val_fraction: float,
    batch_size: int,
    max_history: int,
    histories_per_target: int,
    exclude_after_teleport_steps: int,
    obstacle_oversample_target_frac: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict]:
    log(f"Scanning data directory: {data_dir}")
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No data files found in {data_dir}")
    log(f"Found {len(files)} data files")

    rng = np.random.default_rng(seed)
    order = np.arange(len(files))
    rng.shuffle(order)
    files = [files[i] for i in order]
    log("Shuffled file order for train/val split")

    if len(files) == 1:
        log("Single data file detected, applying temporal split for validation")
        try:
            single_world = load_world_file(files[0])
        except ValueError as exc:
            raise SystemExit(f"No valid data files in {data_dir}: {exc}") from exc
        train_world, val_world = _split_single_world_for_validation(single_world, val_fraction)
        train_worlds = [train_world]
        val_worlds = [val_world]
        train_files = [Path(f"{files[0].name}::train_split")]
        val_files = [Path(f"{files[0].name}::val_split")]
    else:
        val_count = max(1, int(round(len(files) * val_fraction)))
        val_files = files[:val_count]
        train_files = files[val_count:]
        if not train_files:
            train_files, val_files = val_files, train_files
        log(f"Split worlds -> train={len(train_files)} val={len(val_files)}")
        train_worlds = []
        loaded_train_files: list[Path] = []
        for i, p in enumerate(train_files, start=1):
            log(f"Loading train world {i}/{len(train_files)}: {p.name}")
            try:
                train_worlds.append(load_world_file(p))
                loaded_train_files.append(p)
            except ValueError as exc:
                log(f"Skipping invalid train world {p.name}: {exc}")
        train_files = loaded_train_files
        val_worlds = []
        loaded_val_files: list[Path] = []
        for i, p in enumerate(val_files, start=1):
            log(f"Loading val world {i}/{len(val_files)}: {p.name}")
            try:
                val_worlds.append(load_world_file(p))
                loaded_val_files.append(p)
            except ValueError as exc:
                log(f"Skipping invalid val world {p.name}: {exc}")
        val_files = loaded_val_files
        if not train_worlds:
            raise SystemExit(
                "No valid training worlds were found. "
                "Remove malformed files or regenerate data."
            )
        if not val_worlds:
            log("No valid validation worlds found; using first training world for validation.")
            val_worlds = [train_worlds[0]]
            val_files = [Path(f"{train_files[0].name}::val_fallback")]
    log("Computing normalization stats from training worlds")
    stats = compute_norm_stats(train_worlds)

    log("Converting worlds to model features")
    train_feats, train_targets = zip(*(world_to_features(w, stats) for w in train_worlds))
    val_feats, val_targets = zip(*(world_to_features(w, stats) for w in val_worlds))
    train_teleports = [w.teleport_flag for w in train_worlds]
    val_teleports = [w.teleport_flag for w in val_worlds]

    log(
        "Constructing sequence-piece datasets "
        f"(sampled histories_per_target={histories_per_target})"
    )
    train_ds = SequencePieceDataset(
        list(train_feats),
        list(train_targets),
        train_teleports,
        max_history=max_history,
        histories_per_target=histories_per_target,
        exclude_after_teleport_steps=exclude_after_teleport_steps,
        seed=seed,
    )
    val_ds = SequencePieceDataset(
        list(val_feats),
        list(val_targets),
        val_teleports,
        max_history=max_history,
        histories_per_target=histories_per_target,
        exclude_after_teleport_steps=exclude_after_teleport_steps,
        seed=seed + 1,
    )
    log(f"Dataset samples -> train={len(train_ds)} val={len(val_ds)}")

    train_sample_weights, oversample_meta = compute_obstacle_oversample_weights(
        train_ds,
        obstacle_oversample_target_frac,
    )
    train_sampler = None
    if oversample_meta["enabled"]:
        train_sampler = WeightedRandomSampler(
            torch.from_numpy(train_sample_weights),
            num_samples=len(train_ds),
            replacement=True,
        )
        log(
            "Using obstacle-target oversampling "
            f"(natural_frac={oversample_meta['natural_obstacle_fraction']:.3f} "
            f"sampled_frac={oversample_meta['sampled_obstacle_fraction']:.3f} "
            f"obstacle_samples={oversample_meta['obstacle_samples']} "
            f"non_obstacle_samples={oversample_meta['non_obstacle_samples']})"
        )
    else:
        log(
            "Obstacle-target oversampling disabled "
            f"(obstacle_samples={oversample_meta['obstacle_samples']} "
            f"non_obstacle_samples={oversample_meta['non_obstacle_samples']})"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_padded,
    )
    log(f"DataLoaders ready -> batch_size={batch_size}")
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_padded,
    )

    meta = {
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
        "num_train_worlds": len(train_worlds),
        "num_val_worlds": len(val_worlds),
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "num_sensors": int(train_worlds[0].lidar_cm.shape[1]),
        "input_dim": int(train_feats[0].shape[1]),
        "num_classes": 3,
        "histories_per_target": int(histories_per_target),
        "exclude_after_teleport_steps": int(exclude_after_teleport_steps),
        "sequence_sampling": "sampled_per_target",
        "obstacle_oversample_target_frac": float(obstacle_oversample_target_frac),
        "train_obstacle_target_samples": int(oversample_meta["obstacle_samples"]),
        "train_non_obstacle_target_samples": int(oversample_meta["non_obstacle_samples"]),
        "train_obstacle_target_fraction": float(oversample_meta["natural_obstacle_fraction"]),
        "train_sampled_obstacle_target_fraction": float(oversample_meta["sampled_obstacle_fraction"]),
        "train_teleport_rows": int(np.sum([int(np.sum(w > 0)) for w in train_teleports])),
        "val_teleport_rows": int(np.sum([int(np.sum(w > 0)) for w in val_teleports])),
        "train_class_counts": np.bincount(
            np.concatenate([t.reshape(-1) for t in train_targets]),
            minlength=3,
        ).astype(np.int64).tolist(),
        "val_class_counts": np.bincount(
            np.concatenate([t.reshape(-1) for t in val_targets]),
            minlength=3,
        ).astype(np.int64).tolist(),
        "norm": {
            "pose_mean": stats.pose_mean.tolist(),
            "pose_std": stats.pose_std.tolist(),
            "dist_mean": stats.dist_mean,
            "dist_std": stats.dist_std,
        },
    }
    return train_loader, val_loader, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU for current-step lidar class prediction.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/gru_lidar_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-history", type=int, default=32)
    parser.add_argument("--histories-per-target", type=int, default=3)
    parser.add_argument("--exclude-after-teleport-steps", type=int, default=1)
    parser.add_argument("--obstacle-oversample-target-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-epochs", type=int, default=4)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log.txt")
    log_fh = log_path.open("w", encoding="utf-8")
    set_log_file(log_fh)

    log("Starting GRU lidar training script")
    log("Parsed args: " + json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    log(f"Writing logs to: {log_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    log(f"Seeds set -> numpy={args.seed} torch={args.seed}")

    train_loader, val_loader, meta = build_loaders(
        data_dir=args.data_dir,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        max_history=args.max_history,
        histories_per_target=args.histories_per_target,
        exclude_after_teleport_steps=args.exclude_after_teleport_steps,
        obstacle_oversample_target_frac=args.obstacle_oversample_target_frac,
        seed=args.seed,
    )
    if len(train_loader.dataset) == 0:
        raise SystemExit("No training samples were constructed.")
    log(
        "Loader summary: "
        f"train_worlds={meta['num_train_worlds']} val_worlds={meta['num_val_worlds']} "
        f"train_samples={meta['num_train_samples']} val_samples={meta['num_val_samples']}"
    )
    log(
        "Teleport rows: "
        f"train={meta['train_teleport_rows']} val={meta['val_teleport_rows']} "
        f"(exclude_after_teleport_steps={meta['exclude_after_teleport_steps']})"
    )
    log(
        "Obstacle-target sample mix: "
        f"natural={meta['train_obstacle_target_fraction']:.3f} "
        f"sampled={meta['train_sampled_obstacle_target_fraction']:.3f} "
        f"counts={meta['train_obstacle_target_samples']}/{meta['train_non_obstacle_target_samples']}"
    )
    log(
        "Class counts (ground, obstacle, none): "
        f"train={meta['train_class_counts']} val={meta['val_class_counts']}"
    )

    device, device_kind = select_runtime_device(args.device)
    log(f"Using device: {device} ({device_kind})")
    model = GRULidarClassifier(
        input_dim=meta["input_dim"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_sensors=meta["num_sensors"],
        num_classes=meta["num_classes"],
        dropout=args.dropout,
    ).to(device)
    model_config = {
        "model_type": "regional_gru",
        "input_dim": meta["input_dim"],
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_sensors": meta["num_sensors"],
        "num_classes": meta["num_classes"],
        "dropout": args.dropout,
        "region_sensor_groups": model.region_sensor_groups,
        "region_hidden_dims": model.region_hidden_dims,
        "fusion_hidden_dim": int(model.fusion_hidden_dim),
    }
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Model initialized: type={model_config['model_type']} params={param_count}")
    log(
        "Regional layout: "
        f"groups={model_config['region_sensor_groups']} "
        f"region_hidden={model_config['region_hidden_dims']} "
        f"fusion_hidden={model_config['fusion_hidden_dim']}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log(f"Optimizer initialized: AdamW lr={args.lr} weight_decay={args.weight_decay}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        min_lr=args.min_lr,
    )
    log(
        "Scheduler initialized: ReduceLROnPlateau "
        f"factor={args.plateau_factor} patience={args.plateau_patience} min_lr={args.min_lr}"
    )
    class_weights_np = compute_class_weights(np.asarray(meta["train_class_counts"], dtype=np.float32))
    class_weights_t = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    log(f"Using tempered sqrt-inverse class weights (capped) : {class_weights_np.tolist()}")
    if args.focal_gamma > 0.0:
        log(f"Using focal loss with gamma={args.focal_gamma}")
    else:
        log(f"Using cross-entropy loss with label_smoothing={args.label_smoothing}")
    best = {"val_loss": float("inf"), "epoch": -1}
    history: list[dict] = []
    metrics_csv_path = args.output.with_suffix(".metrics.csv")
    split_manifest_path = args.output.with_suffix(".split.json")
    log(f"Output directory ready: {args.output.parent}")
    with split_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "train_files": meta["train_files"],
                "val_files": meta["val_files"],
                "num_train_worlds": meta["num_train_worlds"],
                "num_val_worlds": meta["num_val_worlds"],
                "num_train_samples": meta["num_train_samples"],
                "num_val_samples": meta["num_val_samples"],
                "max_history": args.max_history,
                "histories_per_target": args.histories_per_target,
                "exclude_after_teleport_steps": args.exclude_after_teleport_steps,
                "obstacle_oversample_target_frac": args.obstacle_oversample_target_frac,
                "sequence_sampling": meta["sequence_sampling"],
                "train_teleport_rows": meta["train_teleport_rows"],
                "val_teleport_rows": meta["val_teleport_rows"],
                "train_obstacle_target_samples": meta["train_obstacle_target_samples"],
                "train_non_obstacle_target_samples": meta["train_non_obstacle_target_samples"],
                "train_obstacle_target_fraction": meta["train_obstacle_target_fraction"],
                "train_sampled_obstacle_target_fraction": meta["train_sampled_obstacle_target_fraction"],
                "train_class_counts": meta["train_class_counts"],
                "val_class_counts": meta["val_class_counts"],
            },
            fh,
            indent=2,
        )
    log(f"Wrote split manifest: {split_manifest_path}")
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_loss"])
    log(f"Initialized metrics CSV: {metrics_csv_path}")

    epochs_without_improve = 0
    for epoch in range(1, args.epochs + 1):
        log(f"Epoch {epoch:03d}/{args.epochs:03d} started")
        t0 = perf_counter()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            device_kind,
            meta["num_classes"],
            epoch_idx=epoch,
            total_epochs=args.epochs,
            phase_name="train",
            log_every_batches=args.log_every_batches,
            class_weights=class_weights_t,
            label_smoothing=args.label_smoothing,
            grad_clip_norm=args.grad_clip_norm,
            focal_gamma=args.focal_gamma,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            None,
            device,
            device_kind,
            meta["num_classes"],
            epoch_idx=epoch,
            total_epochs=args.epochs,
            phase_name="val",
            log_every_batches=args.log_every_batches,
            class_weights=class_weights_t,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
        )
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        log(f"Scheduler step complete -> lr={current_lr:.8f}")
        epoch_s = perf_counter() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_seconds": epoch_s,
            }
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch}
            epochs_without_improve = 0
            log(
                f"New best validation loss at epoch {epoch:03d}: "
                f"{best['val_loss']:.6f}. Saving checkpoint -> {args.output}"
            )
            save_checkpoint_for_device(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "norm": meta["norm"],
                    "history": history,
                    "meta": {
                        "train_files": meta["train_files"],
                        "val_files": meta["val_files"],
                        "max_history": args.max_history,
                        "histories_per_target": args.histories_per_target,
                        "exclude_after_teleport_steps": args.exclude_after_teleport_steps,
                        "obstacle_oversample_target_frac": args.obstacle_oversample_target_frac,
                        "sequence_sampling": meta["sequence_sampling"],
                        "class_weighting": "sqrt_inverse_capped",
                        "class_weights": class_weights_np.tolist(),
                        "loss_type": "focal" if args.focal_gamma > 0.0 else "cross_entropy",
                        "prediction_target": "current_step_lidar_class",
                        "focal_gamma": args.focal_gamma,
                        "label_smoothing": args.label_smoothing,
                        "weight_decay": args.weight_decay,
                        "grad_clip_norm": args.grad_clip_norm,
                        "train_class_counts": meta["train_class_counts"],
                        "val_class_counts": meta["val_class_counts"],
                        "best_epoch": best["epoch"],
                        "best_val_loss": best["val_loss"],
                    },
                },
                args.output,
                device_kind,
            )
        else:
            epochs_without_improve += 1
        with metrics_csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, best["val_loss"]])
        log_plain(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.5f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.4f} "
            f"best_val={best['val_loss']:.5f}@{best['epoch']:03d} "
            f"lr={current_lr:.7f} "
            f"time={epoch_s:.2f}s"
        )
        log(f"Epoch {epoch:03d}/{args.epochs:03d} finished")
        if epoch >= args.early_stop_min_epochs and epochs_without_improve >= args.early_stop_patience:
            log(
                f"Early stopping triggered after epoch {epoch:03d}; "
                f"no val improvement for {epochs_without_improve} epochs."
            )
            break

    history_path = args.output.with_suffix(".history.json")
    log("Writing final history JSON")
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    if args.output.exists():
        log("Loading best checkpoint for final validation report")
        best_ckpt = load_checkpoint_for_device(args.output, device, device_kind)
        model.load_state_dict(best_ckpt["model_state_dict"])
    log("Running detailed validation evaluation")
    final_report = evaluate_detailed(model, val_loader, device, device_kind, meta["num_classes"])
    final_report.update(
        {
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "num_train_samples": meta["num_train_samples"],
            "num_val_samples": meta["num_val_samples"],
        }
    )
    val_report_path = args.output.with_suffix(".val_report.json")
    with val_report_path.open("w", encoding="utf-8") as fh:
        json.dump(final_report, fh, indent=2)
    log("Training run complete")
    log_plain(f"saved model: {args.output}")
    log_plain(f"saved history: {history_path}")
    log_plain(f"saved metrics csv: {metrics_csv_path}")
    log_plain(f"saved split manifest: {split_manifest_path}")
    log_plain(f"saved val report: {val_report_path}")
    log_plain(f"saved log: {log_path}")
    log_fh.close()
    set_log_file(None)


if __name__ == "__main__":
    main()
