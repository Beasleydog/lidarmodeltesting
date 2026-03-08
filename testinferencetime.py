from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from train import DEFAULT_LIDAR_MAX_RANGE_CM, load_gru_lidar_inferencer


def _flat_basis_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.asarray(
        [
            c,
            -s,
            0.0,
            s,
            c,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )


def _random_pose_values(rng: np.random.Generator, pose_dim: int) -> np.ndarray:
    x = float(rng.uniform(-1500.0, 1500.0))
    y = float(rng.uniform(-1500.0, 1500.0))
    z = float(rng.uniform(-120.0, 120.0))
    if pose_dim == 12:
        yaw_deg = float(rng.uniform(-180.0, 180.0))
        return np.concatenate(
            [np.asarray([x, y, z], dtype=np.float32), _flat_basis_from_yaw_deg(yaw_deg)],
            axis=0,
        ).astype(np.float32, copy=False)
    if pose_dim == 4:
        yaw_deg = float(rng.uniform(-180.0, 180.0))
        return np.asarray([x, y, z, yaw_deg], dtype=np.float32)
    if pose_dim == 3:
        return np.asarray([x, y, z], dtype=np.float32)
    # Fallback: if a checkpoint has an unexpected pose dim, still produce valid shape.
    return rng.normal(0.0, 1.0, size=(pose_dim,)).astype(np.float32)


def _random_lidar_cm(
    rng: np.random.Generator,
    num_sensors: int,
    max_range_cm: float,
) -> np.ndarray:
    hit_mask = rng.random(num_sensors) < 0.75
    lidar = np.full((num_sensors,), -1.0, dtype=np.float32)
    hit_count = int(np.count_nonzero(hit_mask))
    if hit_count > 0:
        lidar[hit_mask] = rng.uniform(20.0, float(max_range_cm), size=(hit_count,)).astype(np.float32)
    return lidar


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _predict_current(
    inferencer,
    feature_history: np.ndarray,
    obstacle_logit_bias: float,
) -> np.ndarray:
    if inferencer.binary_obstacle_only:
        return inferencer.predict_current_obstacle_mask_from_feature_history_with_bias(
            feature_history,
            obstacle_logit_bias=float(obstacle_logit_bias),
        )
    return inferencer.predict_current_from_feature_history(feature_history)


def _run_one_step(
    inferencer,
    rng: np.random.Generator,
    history: list[np.ndarray],
    obstacle_logit_bias: float,
    max_range_cm: float,
) -> tuple[float, np.ndarray]:
    pose_values = _random_pose_values(rng, int(inferencer.pose_dim))
    lidar_cm = _random_lidar_cm(rng, int(inferencer.num_sensors), max_range_cm=float(max_range_cm))
    feature_t = inferencer.featurize_timestep(pose_values, lidar_cm)
    feature_history = np.asarray([*history, feature_t], dtype=np.float32)

    _sync_if_cuda(inferencer.device)
    t0 = perf_counter()
    pred = _predict_current(inferencer, feature_history, obstacle_logit_bias=float(obstacle_logit_bias))
    _sync_if_cuda(inferencer.device)
    dt_s = perf_counter() - t0

    history.append(feature_t)
    if inferencer.max_history > 0 and len(history) > inferencer.max_history:
        del history[:-inferencer.max_history]
    return dt_s, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure random single-step and looped inference latency.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/gru_lidar_classifier.pt"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-history", type=int, default=64)
    parser.add_argument("--max-range-cm", type=float, default=DEFAULT_LIDAR_MAX_RANGE_CM)
    parser.add_argument("--obstacle-logit-bias", type=float, default=0.0)
    args = parser.parse_args()

    inferencer = load_gru_lidar_inferencer(
        args.checkpoint,
        device=args.device,
        max_history=int(args.max_history),
    )
    rng = np.random.default_rng(int(args.seed))

    print(
        "Loaded inferencer: "
        f"checkpoint={args.checkpoint} "
        f"device={inferencer.device} "
        f"sensors={inferencer.num_sensors} "
        f"pose_dim={inferencer.pose_dim} "
        f"binary_obstacle_only={inferencer.binary_obstacle_only}"
    )

    # Single random step from empty history.
    single_history: list[np.ndarray] = []
    single_dt_s, single_pred = _run_one_step(
        inferencer,
        rng,
        single_history,
        obstacle_logit_bias=float(args.obstacle_logit_bias),
        max_range_cm=float(args.max_range_cm),
    )
    single_obstacles = int(np.count_nonzero(single_pred))
    print(
        "Single step: "
        f"{single_dt_s * 1000.0:.3f} ms "
        f"(predicted_obstacle_rays={single_obstacles}/{inferencer.num_sensors})"
    )

    # Timed loop after warmup to capture steadier latency.
    loop_history: list[np.ndarray] = []
    for _ in range(int(max(args.warmup, 0))):
        _run_one_step(
            inferencer,
            rng,
            loop_history,
            obstacle_logit_bias=float(args.obstacle_logit_bias),
            max_range_cm=float(args.max_range_cm),
        )

    latencies_ms: list[float] = []
    for _ in range(int(max(args.loops, 1))):
        dt_s, _ = _run_one_step(
            inferencer,
            rng,
            loop_history,
            obstacle_logit_bias=float(args.obstacle_logit_bias),
            max_range_cm=float(args.max_range_cm),
        )
        latencies_ms.append(dt_s * 1000.0)

    total_ms = float(np.sum(latencies_ms))
    avg_ms = float(np.mean(latencies_ms))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    p95_ms = float(np.percentile(np.asarray(latencies_ms, dtype=np.float64), 95.0))
    print(
        f"Loop {int(max(args.loops, 1))} steps (warmup={int(max(args.warmup, 0))}): "
        f"total={total_ms:.3f} ms "
        f"avg={avg_ms:.3f} ms "
        f"min={min_ms:.3f} ms "
        f"max={max_ms:.3f} ms "
        f"p95={p95_ms:.3f} ms"
    )


if __name__ == "__main__":
    main()
