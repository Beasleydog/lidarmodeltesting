from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train import (
    DEFAULT_LIDAR_MAX_RANGE_CM,
    EGO_MAP_FEATURE_MODE,
    MODEL_SENSOR_COORDS_CM,
    MODEL_SENSOR_DIRS_LOCAL,
    build_loaders,
    compute_class_weights,
    configure_runtime_for_device,
    load_checkpoint_for_device,
    log,
    log_plain,
    resolve_data_loader_workers,
    save_checkpoint_for_device,
    select_runtime_device,
    set_log_file,
)


class PoseAlignedBeamTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_sensors: int,
        num_classes: int,
        hidden_dim: int = 96,
        dropout: float = 0.10,
        max_range_cm: float = DEFAULT_LIDAR_MAX_RANGE_CM,
        transformer_layers: int = 4,
        attention_heads: int = 4,
        ff_mult: int = 4,
        decoder_hidden_dim: int = 96,
    ):
        super().__init__()
        self.model_type = "pose_aligned_beam_transformer"
        self.input_dim = int(input_dim)
        self.num_sensors = int(num_sensors)
        self.num_classes = int(num_classes)
        self.pose_dim = int(self.input_dim - 2 * self.num_sensors)
        if self.pose_dim != 12:
            raise ValueError(
                f"pose_aligned_beam_transformer expects input_dim=12+2*num_sensors; got input_dim={self.input_dim}"
            )

        self.hidden_dim = int(hidden_dim)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.max_range_cm = float(max(max_range_cm, 1.0))
        self.transformer_layers = int(max(transformer_layers, 1))
        self.attention_heads = int(max(attention_heads, 1))
        self.ff_mult = int(max(ff_mult, 2))
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("--hidden-dim must be divisible by --attention-heads")

        self.register_buffer(
            "sensor_coords_local",
            torch.tensor(MODEL_SENSOR_COORDS_CM[: self.num_sensors], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sensor_dirs_local",
            torch.tensor(MODEL_SENSOR_DIRS_LOCAL[: self.num_sensors], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sensor_ids",
            torch.arange(self.num_sensors, dtype=torch.long),
            persistent=False,
        )

        self.sensor_embedding = nn.Embedding(self.num_sensors, self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.token_mlp = nn.Sequential(
            nn.Linear(12, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(11, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.attention_heads,
            dim_feedforward=self.hidden_dim * self.ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
        self.query_to_history = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_dim, self.num_classes),
        )

    def _gather_last_valid(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(values.shape[0], device=values.device)
        last_idx = lengths.to(device=values.device).clamp_min(1) - 1
        return values[batch_idx, last_idx]

    def _parse_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose = x[:, :, : self.pose_dim]
        dist = x[:, :, self.pose_dim : self.pose_dim + self.num_sensors]
        hit = x[:, :, self.pose_dim + self.num_sensors : self.pose_dim + 2 * self.num_sensors]
        pose_origin = pose[:, :, :3]
        pose_basis = pose[:, :, 3:].reshape(x.shape[0], x.shape[1], 3, 3)
        return pose_origin, pose_basis, dist, hit

    def _build_history_tokens(
        self,
        pose_origin: torch.Tensor,
        pose_basis: torch.Tensor,
        dist: torch.Tensor,
        hit: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, sensor_count = dist.shape
        time_mask = (
            torch.arange(time_steps, device=dist.device).unsqueeze(0) < lengths.to(device=dist.device).unsqueeze(1)
        )

        current_dist = self._gather_last_valid(dist, lengths).clamp(min=0.0, max=self.max_range_cm)
        current_hit = self._gather_last_valid(hit, lengths)
        sensor_coords = self.sensor_coords_local.view(1, 1, sensor_count, 3).to(device=dist.device, dtype=dist.dtype)
        sensor_dirs = self.sensor_dirs_local.view(1, 1, sensor_count, 3).to(device=dist.device, dtype=dist.dtype)
        pose_basis_exp = pose_basis.unsqueeze(2)
        origin_world = pose_origin.unsqueeze(2) + torch.matmul(pose_basis_exp, sensor_coords.unsqueeze(-1)).squeeze(-1)
        dir_world = torch.matmul(pose_basis_exp, sensor_dirs.unsqueeze(-1)).squeeze(-1)

        ranges = dist.clamp(min=0.0, max=self.max_range_cm)
        endpoint_world = origin_world + dir_world * ranges.unsqueeze(-1)

        time_idx = torch.arange(time_steps, device=dist.device).unsqueeze(0)
        time_offset = (lengths.to(device=dist.device).unsqueeze(1) - 1 - time_idx).clamp_min(0).to(dtype=dist.dtype)
        age = time_offset / torch.maximum(
            (lengths.to(device=dist.device).unsqueeze(1) - 1).to(dtype=dist.dtype),
            torch.ones((batch_size, 1), device=dist.device, dtype=dist.dtype),
        )
        age = age.unsqueeze(-1).unsqueeze(2).expand(-1, -1, sensor_count, -1)

        token_features = torch.cat(
            [
                origin_world / self.max_range_cm,
                dir_world,
                hit.unsqueeze(-1),
                (ranges / self.max_range_cm).unsqueeze(-1),
                endpoint_world / self.max_range_cm,
                age,
            ],
            dim=-1,
        )
        token_features = token_features.reshape(batch_size, time_steps * sensor_count, -1)
        token_age = age.reshape(batch_size, time_steps * sensor_count, 1)
        token_sensor_ids = self.sensor_ids.view(1, 1, sensor_count).expand(batch_size, time_steps, -1).reshape(batch_size, -1)
        token_mask = (~time_mask).unsqueeze(-1).expand(-1, -1, sensor_count).reshape(batch_size, -1)
        return token_features, token_age, token_sensor_ids, token_mask, current_dist, current_hit

    def _encode_history(
        self,
        token_features: torch.Tensor,
        token_age: torch.Tensor,
        token_sensor_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_embed = self.token_mlp(token_features)
        token_embed = token_embed + self.sensor_embedding(token_sensor_ids) + self.time_mlp(token_age)
        return self.history_encoder(token_embed, src_key_padding_mask=token_mask)

    def _build_queries(
        self,
        current_dist: torch.Tensor,
        current_hit: torch.Tensor,
        current_pose_origin: torch.Tensor,
        current_pose_basis: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = current_dist.shape[0]
        sensor_coords_local = self.sensor_coords_local.to(device=current_dist.device, dtype=current_dist.dtype).unsqueeze(0)
        sensor_dirs_local = self.sensor_dirs_local.to(device=current_dist.device, dtype=current_dist.dtype).unsqueeze(0)
        sensor_coords = current_pose_origin.unsqueeze(1) + torch.matmul(
            current_pose_basis.unsqueeze(1), sensor_coords_local.unsqueeze(-1)
        ).squeeze(-1)
        sensor_dirs = torch.matmul(current_pose_basis.unsqueeze(1), sensor_dirs_local.unsqueeze(-1)).squeeze(-1)
        endpoint_local = sensor_coords + sensor_dirs * current_dist.unsqueeze(-1)
        query_features = torch.cat(
            [
                sensor_coords / self.max_range_cm,
                sensor_dirs,
                current_hit.unsqueeze(-1),
                (current_dist / self.max_range_cm).unsqueeze(-1),
                endpoint_local / self.max_range_cm,
            ],
            dim=-1,
        )
        sensor_embed = self.sensor_embedding(self.sensor_ids.to(device=current_dist.device)).unsqueeze(0)
        return self.query_mlp(query_features) + sensor_embed.expand(batch_size, -1, -1)

    def _decode_queries(
        self,
        encoded_history: torch.Tensor,
        token_mask: torch.Tensor,
        current_dist: torch.Tensor,
        current_hit: torch.Tensor,
        current_pose_origin: torch.Tensor,
        current_pose_basis: torch.Tensor,
    ) -> torch.Tensor:
        queries = self._build_queries(current_dist, current_hit, current_pose_origin, current_pose_basis)
        attended, _ = self.query_to_history(
            queries,
            encoded_history,
            encoded_history,
            key_padding_mask=token_mask,
            need_weights=False,
        )
        history_sum = encoded_history.masked_fill(token_mask.unsqueeze(-1), 0.0).sum(dim=1)
        history_count = (~token_mask).sum(dim=1, keepdim=True).clamp_min(1).to(dtype=encoded_history.dtype)
        history_global = (history_sum / history_count).unsqueeze(1).expand(-1, self.num_sensors, -1)
        fused = self.fuse(torch.cat([queries, attended, history_global], dim=-1))
        return self.decoder(fused)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pose_origin, pose_basis, dist, hit = self._parse_inputs(x)
        current_pose_origin = self._gather_last_valid(pose_origin, lengths)
        current_pose_basis = self._gather_last_valid(pose_basis, lengths)
        token_features, token_age, token_sensor_ids, token_mask, current_dist, current_hit = self._build_history_tokens(
            pose_origin, pose_basis, dist, hit, lengths
        )
        encoded = self._encode_history(token_features, token_age, token_sensor_ids, token_mask)
        return self._decode_queries(
            encoded,
            token_mask,
            current_dist,
            current_hit,
            current_pose_origin,
            current_pose_basis,
        )


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    device_kind: str,
    epoch_idx: int,
    total_epochs: int,
    phase_name: str,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 0.0,
    use_amp: bool = False,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    seen_sequences = 0
    phase_t0 = perf_counter()
    non_blocking = device_kind == "cuda"
    amp_enabled = bool(use_amp and device_kind == "cuda")

    class_weights_t = None
    if class_weights is not None:
        class_weights_t = class_weights.to(device)

    for x, lengths, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        lengths = lengths.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if amp_enabled
            else nullcontext()
        )
        grad_context = nullcontext() if is_train else torch.inference_mode()
        with grad_context:
            with amp_context:
                logits = model(x, lengths)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    weight=class_weights_t,
                    label_smoothing=float(max(label_smoothing, 0.0)),
                )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        total_correct += int((preds == y).sum().item())
        total_count += int(y.numel())
        batch_sequences = int(y.shape[0])
        total_loss += float(loss.item()) * float(batch_sequences)
        seen_sequences += batch_sequences

    avg_loss = total_loss / max(seen_sequences, 1)
    acc = total_correct / max(total_count, 1)
    phase_s = perf_counter() - phase_t0
    log(
        f"{phase_name} epoch {epoch_idx:03d}/{total_epochs:03d} complete "
        f"loss={avg_loss:.5f} acc={acc:.4f} time={phase_s:.2f}s"
    )
    return avg_loss, acc


def evaluate_detailed(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    confusion = np.zeros((3, 3), dtype=np.int64)
    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=-1)
            y_flat = y.reshape(-1).cpu().numpy()
            p_flat = preds.reshape(-1).cpu().numpy()
            for yt, yp in zip(y_flat, p_flat):
                confusion[int(yt), int(yp)] += 1

    per_class_recall = []
    for c in range(3):
        denom = int(confusion[c, :].sum())
        per_class_recall.append(float(confusion[c, c]) / max(denom, 1))
    overall_accuracy = float(np.trace(confusion)) / max(int(confusion.sum()), 1)
    return {
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": per_class_recall,
        "overall_accuracy": overall_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pose-aligned transformer lidar classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/pose_aligned_beam_transformer.pt"))
    parser.add_argument("--eval-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--decoder-hidden-dim", type=int, default=96)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--amp", type=str, choices=("auto", "on", "off"), default="off")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-history", type=int, default=48)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--histories-per-target", type=int, default=2)
    parser.add_argument("--exclude-after-teleport-steps", type=int, default=1)
    parser.add_argument("--history-step-min", type=int, default=8)
    parser.add_argument("--history-step-max", type=int, default=24)
    parser.add_argument("--train-lidar-offset-max-cm", type=float, default=200.0)
    parser.add_argument("--train-diversity-max-per-signature", type=int, default=2)
    parser.add_argument("--train-diversity-ref-samples", type=int, default=50000)
    parser.add_argument("--train-diversity-quant-scale", type=float, default=0.20)
    parser.add_argument("--train-diversity-min-step", type=float, default=0.05)
    parser.add_argument("--train-world-scale-x-min", type=float, default=0.90)
    parser.add_argument("--train-world-scale-x-max", type=float, default=1.10)
    parser.add_argument("--train-world-scale-y-min", type=float, default=0.90)
    parser.add_argument("--train-world-scale-y-max", type=float, default=1.10)
    parser.add_argument("--train-world-scale-z-min", type=float, default=0.90)
    parser.add_argument("--train-world-scale-z-max", type=float, default=1.10)
    parser.add_argument("--no-hit-range-cm", type=float, default=DEFAULT_LIDAR_MAX_RANGE_CM)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    args = parser.parse_args()

    report_base = args.eval_checkpoint if args.eval_checkpoint is not None else args.output
    report_base.parent.mkdir(parents=True, exist_ok=True)
    log_path = (
        report_base.with_suffix(".eval.log.txt")
        if args.eval_checkpoint is not None
        else args.output.with_suffix(".log.txt")
    )
    log_fh = log_path.open("w", encoding="utf-8")
    set_log_file(log_fh)

    log("Starting pose-aligned beam transformer training script")
    log("Parsed args: " + json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device, device_kind = select_runtime_device(args.device)
    configure_runtime_for_device(device_kind)
    loader_num_workers = resolve_data_loader_workers(args.num_workers)
    loader_pin_memory = bool(device_kind == "cuda" and not args.disable_pin_memory)
    amp_enabled = bool(device_kind == "cuda" and args.amp != "off")

    train_loader, val_loader, meta = build_loaders(
        data_dir=args.data_dir,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=loader_num_workers,
        pin_memory=loader_pin_memory,
        prefetch_factor=args.prefetch_factor,
        max_history=args.max_history,
        min_history=args.min_history,
        histories_per_target=args.histories_per_target,
        exclude_after_teleport_steps=args.exclude_after_teleport_steps,
        history_step_min=args.history_step_min,
        history_step_max=args.history_step_max,
        obstacle_oversample_target_frac=0.0,
        train_lidar_offset_max_cm=args.train_lidar_offset_max_cm,
        train_diversity_max_per_signature=args.train_diversity_max_per_signature,
        train_diversity_ref_samples=args.train_diversity_ref_samples,
        train_diversity_quant_scale=args.train_diversity_quant_scale,
        train_diversity_min_step=args.train_diversity_min_step,
        train_world_scale_x_min=args.train_world_scale_x_min,
        train_world_scale_x_max=args.train_world_scale_x_max,
        train_world_scale_y_min=args.train_world_scale_y_min,
        train_world_scale_y_max=args.train_world_scale_y_max,
        train_world_scale_z_min=args.train_world_scale_z_min,
        train_world_scale_z_max=args.train_world_scale_z_max,
        feature_mode=EGO_MAP_FEATURE_MODE,
        no_hit_range_cm=float(args.no_hit_range_cm),
        seed=args.seed,
    )
    if len(train_loader.dataset) == 0:
        raise SystemExit("Training split produced zero samples; adjust history sampling or dataset filters.")
    log(
        "Dataset split summary: "
        f"train_worlds={meta['num_train_worlds']} val_worlds={meta['num_val_worlds']} "
        f"train_samples={meta['num_train_samples']} val_samples={meta['num_val_samples']}"
    )
    log(
        "Teleport exclusion summary: "
        f"train={meta['train_teleport_rows']} val={meta['val_teleport_rows']} "
        f"(exclude_after_teleport_steps={meta['exclude_after_teleport_steps']})"
    )
    log(
        "Sequence sampling summary: "
        f"length={meta['min_history']}..{('unbounded' if int(meta['max_history']) <= 0 else meta['max_history'])} "
        f"sampled={meta['histories_per_target']} "
        f"step_range={meta['history_step_min']}..{meta['history_step_max']}"
    )
    log(
        "Train lidar offset augmentation: "
        f"max_cm={meta['train_lidar_offset_max_cm']:.1f} (uniform in [0,max] per sampled history)"
    )
    if bool(meta.get("train_diversity_enabled", False)):
        log(
            "Train diversity filter: "
            f"before={meta['train_diversity_before']} after={meta['train_diversity_after']} "
            f"kept_frac={meta['train_diversity_kept_fraction']:.3f} "
            f"unique={meta['train_diversity_unique_signatures']} "
            f"max_per_signature={meta['train_diversity_max_per_signature']}"
        )
    else:
        log("Train diversity filter: disabled")
    if bool(meta.get("train_world_scale_enabled", False)):
        log(
            "Train world scaling: "
            f"x={meta['train_world_scale_x_min']:.3f}..{meta['train_world_scale_x_max']:.3f} "
            f"y={meta['train_world_scale_y_min']:.3f}..{meta['train_world_scale_y_max']:.3f} "
            f"z={meta['train_world_scale_z_min']:.3f}..{meta['train_world_scale_z_max']:.3f}"
        )
    else:
        log("Train world scaling: disabled")
    log(
        "Obstacle target balance: "
        f"natural={meta['train_obstacle_target_fraction']:.3f} "
        f"sampled={meta['train_sampled_obstacle_target_fraction']:.3f} "
        f"counts={meta['train_obstacle_target_samples']}/{meta['train_non_obstacle_target_samples']}"
    )
    log(
        "Per-class counts: "
        f"train={meta['train_class_counts']} val={meta['val_class_counts']}"
    )

    model_config = {
        "model_type": "pose_aligned_beam_transformer",
        "input_dim": int(meta["input_dim"]),
        "num_sensors": int(meta["num_sensors"]),
        "num_classes": int(meta["num_classes"]),
        "hidden_dim": int(args.hidden_dim),
        "decoder_hidden_dim": int(args.decoder_hidden_dim),
        "transformer_layers": int(args.transformer_layers),
        "attention_heads": int(args.attention_heads),
        "ff_mult": int(args.ff_mult),
        "dropout": float(args.dropout),
        "no_hit_range_cm": float(meta["no_hit_range_cm"]),
        "feature_mode": meta["feature_mode"],
    }

    if args.eval_checkpoint is not None:
        ckpt = load_checkpoint_for_device(args.eval_checkpoint, device, device_kind)
        cfg = dict(ckpt["model_config"])
        if str(cfg.get("model_type", "")) not in {"", "pose_aligned_beam_transformer"}:
            raise SystemExit(
                f"{args.eval_checkpoint} has model_type={cfg.get('model_type')!r}; "
                "this script now only evaluates pose_aligned_beam_transformer checkpoints."
            )
        log(f"Eval-only mode: loading checkpoint {args.eval_checkpoint}")
        model = PoseAlignedBeamTransformerClassifier(
            input_dim=int(cfg["input_dim"]),
            num_sensors=int(cfg["num_sensors"]),
            num_classes=int(cfg["num_classes"]),
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
            max_range_cm=float(cfg["no_hit_range_cm"]),
            transformer_layers=int(cfg.get("transformer_layers", 4)),
            attention_heads=int(cfg.get("attention_heads", 4)),
            ff_mult=int(cfg.get("ff_mult", 4)),
            decoder_hidden_dim=int(cfg["decoder_hidden_dim"]),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        param_count = sum(p.numel() for p in model.parameters())
        log(f"Model initialized: type={model.model_type} params={param_count}")
        log(
            "Feature pipeline: "
            f"mode={meta['feature_mode']} "
            f"no_hit_range_cm={meta['no_hit_range_cm']:.1f}"
        )
        log(
            "Pose-aligned beam transformer layout: "
            f"hidden={cfg['hidden_dim']} "
            f"layers={cfg.get('transformer_layers', 4)} "
            f"heads={cfg.get('attention_heads', 4)} "
            f"ff_mult={cfg.get('ff_mult', 4)} "
            f"decoder_hidden={cfg['decoder_hidden_dim']}"
        )
        log("Running eval-only detailed validation evaluation")
        final_report = evaluate_detailed(model, val_loader, device)
        final_report["source_checkpoint"] = str(args.eval_checkpoint)
        final_report["num_train_samples"] = meta["num_train_samples"]
        final_report["num_val_samples"] = meta["num_val_samples"]
        val_report_path = args.eval_checkpoint.with_suffix(".eval_val_report.json")
        with val_report_path.open("w", encoding="utf-8") as fh:
            json.dump(final_report, fh, indent=2)
        log_plain(f"saved val report: {val_report_path}")
        log_plain(f"saved log: {log_path}")
        log_fh.close()
        set_log_file(None)
        return

    model = PoseAlignedBeamTransformerClassifier(
        input_dim=int(meta["input_dim"]),
        num_sensors=int(meta["num_sensors"]),
        num_classes=int(meta["num_classes"]),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        max_range_cm=float(meta["no_hit_range_cm"]),
        transformer_layers=int(args.transformer_layers),
        attention_heads=int(args.attention_heads),
        ff_mult=int(args.ff_mult),
        decoder_hidden_dim=int(args.decoder_hidden_dim),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    log(f"Model initialized: type={model.model_type} params={param_count}")
    log(
        "Feature pipeline: "
        f"mode={meta['feature_mode']} "
        f"no_hit_range_cm={meta['no_hit_range_cm']:.1f}"
    )
    log(
        "Pose-aligned beam transformer layout: "
        f"hidden={model_config['hidden_dim']} "
        f"layers={model_config['transformer_layers']} "
        f"heads={model_config['attention_heads']} "
        f"ff_mult={model_config['ff_mult']} "
        f"decoder_hidden={model_config['decoder_hidden_dim']}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log(f"Optimizer initialized: AdamW lr={args.lr} weight_decay={args.weight_decay}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )
    log("Scheduler initialized: ReduceLROnPlateau factor=0.5 patience=2 min_lr=1e-05")
    class_weights = torch.tensor(
        compute_class_weights(np.asarray(meta["train_class_counts"], dtype=np.float32)),
        dtype=torch.float32,
    )

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
                "min_history": meta["min_history"],
                "max_history": meta["max_history"],
                "histories_per_target": args.histories_per_target,
                "exclude_after_teleport_steps": args.exclude_after_teleport_steps,
                "history_step_min": args.history_step_min,
                "history_step_max": args.history_step_max,
                "train_lidar_offset_max_cm": float(meta["train_lidar_offset_max_cm"]),
                "train_diversity_enabled": bool(meta.get("train_diversity_enabled", False)),
                "train_diversity_before": int(meta.get("train_diversity_before", meta["num_train_samples"])),
                "train_diversity_after": int(meta.get("train_diversity_after", meta["num_train_samples"])),
                "train_diversity_kept_fraction": float(meta.get("train_diversity_kept_fraction", 1.0)),
                "train_diversity_unique_signatures": int(meta.get("train_diversity_unique_signatures", 0)),
                "train_diversity_max_per_signature": int(meta.get("train_diversity_max_per_signature", 0)),
                "train_diversity_ref_samples": int(meta.get("train_diversity_ref_samples", 0)),
                "train_diversity_quant_scale": float(meta.get("train_diversity_quant_scale", 0.0)),
                "train_diversity_min_step": float(meta.get("train_diversity_min_step", 0.0)),
                "train_world_scale_enabled": bool(meta.get("train_world_scale_enabled", False)),
                "train_world_scale_x_min": float(meta.get("train_world_scale_x_min", 1.0)),
                "train_world_scale_x_max": float(meta.get("train_world_scale_x_max", 1.0)),
                "train_world_scale_y_min": float(meta.get("train_world_scale_y_min", 1.0)),
                "train_world_scale_y_max": float(meta.get("train_world_scale_y_max", 1.0)),
                "train_world_scale_z_min": float(meta.get("train_world_scale_z_min", 1.0)),
                "train_world_scale_z_max": float(meta.get("train_world_scale_z_max", 1.0)),
                "feature_mode": meta["feature_mode"],
                "no_hit_range_cm": float(meta["no_hit_range_cm"]),
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

    for epoch in range(1, args.epochs + 1):
        log(f"Epoch {epoch:03d}/{args.epochs:03d} started")
        t0 = perf_counter()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            device_kind,
            epoch,
            args.epochs,
            "train",
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            grad_clip_norm=args.grad_clip_norm,
            use_amp=amp_enabled,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            None,
            device,
            device_kind,
            epoch,
            args.epochs,
            "val",
            class_weights=class_weights,
            label_smoothing=0.0,
            grad_clip_norm=0.0,
            use_amp=amp_enabled,
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
                "lr": current_lr,
                "epoch_seconds": epoch_s,
            }
        )

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch}
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
                        "min_history": meta["min_history"],
                        "max_history": meta["max_history"],
                        "histories_per_target": args.histories_per_target,
                        "exclude_after_teleport_steps": args.exclude_after_teleport_steps,
                        "history_step_min": args.history_step_min,
                        "history_step_max": args.history_step_max,
                        "train_lidar_offset_max_cm": float(meta["train_lidar_offset_max_cm"]),
                        "train_diversity_enabled": bool(meta.get("train_diversity_enabled", False)),
                        "train_diversity_before": int(meta.get("train_diversity_before", meta["num_train_samples"])),
                        "train_diversity_after": int(meta.get("train_diversity_after", meta["num_train_samples"])),
                        "train_diversity_kept_fraction": float(meta.get("train_diversity_kept_fraction", 1.0)),
                        "train_diversity_unique_signatures": int(meta.get("train_diversity_unique_signatures", 0)),
                        "train_diversity_max_per_signature": int(meta.get("train_diversity_max_per_signature", 0)),
                        "train_diversity_ref_samples": int(meta.get("train_diversity_ref_samples", 0)),
                        "train_diversity_quant_scale": float(meta.get("train_diversity_quant_scale", 0.0)),
                        "train_diversity_min_step": float(meta.get("train_diversity_min_step", 0.0)),
                        "train_world_scale_enabled": bool(meta.get("train_world_scale_enabled", False)),
                        "train_world_scale_x_min": float(meta.get("train_world_scale_x_min", 1.0)),
                        "train_world_scale_x_max": float(meta.get("train_world_scale_x_max", 1.0)),
                        "train_world_scale_y_min": float(meta.get("train_world_scale_y_min", 1.0)),
                        "train_world_scale_y_max": float(meta.get("train_world_scale_y_max", 1.0)),
                        "train_world_scale_z_min": float(meta.get("train_world_scale_z_min", 1.0)),
                        "train_world_scale_z_max": float(meta.get("train_world_scale_z_max", 1.0)),
                        "feature_mode": meta["feature_mode"],
                        "no_hit_range_cm": float(meta["no_hit_range_cm"]),
                        "sequence_sampling": meta["sequence_sampling"],
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
        with metrics_csv_path.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow([epoch, train_loss, train_acc, val_loss, val_acc, best["val_loss"]])

        log_plain(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.5f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.4f} "
            f"best_val={best['val_loss']:.5f}@{best['epoch']:03d} "
            f"lr={current_lr:.7f} "
            f"time={epoch_s:.2f}s"
        )
        log(f"Epoch {epoch:03d}/{args.epochs:03d} finished")

    log("Writing final history JSON")
    history_path = args.output.with_suffix(".history.json")
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    if args.output.exists():
        log("Loading best checkpoint for final validation report")
        ckpt = load_checkpoint_for_device(args.output, device, device_kind)
        model.load_state_dict(ckpt["model_state_dict"])

    log("Running detailed validation evaluation")
    final_report = evaluate_detailed(model, val_loader, device)
    final_report["best_epoch"] = best["epoch"]
    final_report["best_val_loss"] = best["val_loss"]
    final_report["num_train_samples"] = meta["num_train_samples"]
    final_report["num_val_samples"] = meta["num_val_samples"]
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
