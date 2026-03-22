from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _quantile_str(values: np.ndarray, q: float) -> str:
    if values.size == 0:
        return "nan"
    return f"{float(np.quantile(values, q)):.1f}"


def _median_str(values: np.ndarray) -> str:
    if values.size == 0:
        return "nan"
    return f"{float(np.median(values)):.1f}"


def _pct(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return 100.0 * float(numer) / float(denom)


def analyze_training(train_dir: Path) -> dict:
    files = sorted(train_dir.glob("world_*.txt"))
    per_sensor_total = defaultdict(int)
    per_sensor_valid = defaultdict(int)
    per_sensor_zero = defaultdict(int)
    per_sensor_values: dict[int, list[float]] = defaultdict(list)
    per_step_valid_counts: list[int] = []
    per_step_short_counts: list[int] = []
    per_step_far_counts: list[int] = []
    motion_dxy: list[float] = []
    motion_dyaw: list[float] = []

    for path in files:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            prev_pose: tuple[float, float, float] | None = None
            for row in reader:
                vals: list[float] = []
                sensor_idx = 0
                while f"lidar_cm_{sensor_idx}" in row:
                    value = float(row[f"lidar_cm_{sensor_idx}"])
                    vals.append(value)
                    per_sensor_total[sensor_idx] += 1
                    if value == 0.0:
                        per_sensor_zero[sensor_idx] += 1
                    if value >= 0.0:
                        per_sensor_valid[sensor_idx] += 1
                        per_sensor_values[sensor_idx].append(value)
                    sensor_idx += 1

                arr = np.asarray(vals, dtype=np.float64)
                per_step_valid_counts.append(int(np.sum(arr > 0.0)))
                per_step_short_counts.append(int(np.sum((arr > 0.0) & (arr < 150.0))))
                per_step_far_counts.append(int(np.sum(arr >= 900.0)))

                x = float(row["x_cm"])
                y = float(row["y_cm"])
                yaw = float(row["yaw_deg"])
                if prev_pose is not None:
                    motion_dxy.append(math.hypot(x - prev_pose[0], y - prev_pose[1]))
                    dyaw = ((yaw - prev_pose[2] + 180.0) % 360.0) - 180.0
                    motion_dyaw.append(abs(dyaw))
                prev_pose = (x, y, yaw)

    return {
        "files": files,
        "per_sensor_total": per_sensor_total,
        "per_sensor_valid": per_sensor_valid,
        "per_sensor_zero": per_sensor_zero,
        "per_sensor_values": per_sensor_values,
        "per_step_valid_counts": np.asarray(per_step_valid_counts, dtype=np.float64),
        "per_step_short_counts": np.asarray(per_step_short_counts, dtype=np.float64),
        "per_step_far_counts": np.asarray(per_step_far_counts, dtype=np.float64),
        "motion_dxy": np.asarray(motion_dxy, dtype=np.float64),
        "motion_dyaw": np.asarray(motion_dyaw, dtype=np.float64),
    }


def analyze_real(realdata_dir: Path) -> dict:
    dirs = sorted(realdata_dir.glob("smartdrive_debug_*"))
    per_sensor_total = defaultdict(int)
    per_sensor_enabled = defaultdict(int)
    per_sensor_raw_nonneg = defaultdict(int)
    per_sensor_raw_positive = defaultdict(int)
    per_sensor_raw_zero = defaultdict(int)
    per_sensor_raw_neg1 = defaultdict(int)
    per_sensor_raw_values: dict[int, list[float]] = defaultdict(list)
    per_sensor_positive = defaultdict(int)
    per_sensor_zero = defaultdict(int)
    per_sensor_neg1 = defaultdict(int)
    per_sensor_values: dict[int, list[float]] = defaultdict(list)
    per_sensor_reason_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_step_enabled_counts: list[int] = []
    per_step_positive_counts: list[int] = []
    per_step_nonpositive_counts: list[int] = []
    per_step_short_counts: list[int] = []
    per_step_far_counts: list[int] = []
    motion_dxy: list[float] = []
    motion_dyaw: list[float] = []

    for run_dir in dirs:
        path = run_dir / "lidar_points.txt"
        if not path.exists():
            continue

        step_ranges: dict[int, dict[int, float]] = {}
        step_enabled: dict[int, dict[int, int]] = {}
        step_pose: dict[int, tuple[float, float, float]] = {}
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                step_idx = int(row["step_idx"])
                sensor_idx = int(row["sensor_idx"])
                enabled = int(row["sensor_enabled"])
                raw_cm = float(row["raw_cm"]) if row["raw_cm"].lower() != "nan" else float("nan")
                reason = row.get("usable_reason", "")

                per_sensor_total[sensor_idx] += 1
                per_sensor_enabled[sensor_idx] += enabled
                per_sensor_reason_counts[sensor_idx][reason] += 1
                if raw_cm >= 0.0:
                    per_sensor_raw_nonneg[sensor_idx] += 1
                if raw_cm > 0.0:
                    per_sensor_raw_positive[sensor_idx] += 1
                    per_sensor_raw_values[sensor_idx].append(raw_cm)
                if raw_cm == 0.0:
                    per_sensor_raw_zero[sensor_idx] += 1
                if raw_cm == -1.0:
                    per_sensor_raw_neg1[sensor_idx] += 1

                if enabled:
                    if raw_cm == 0.0:
                        per_sensor_zero[sensor_idx] += 1
                    if raw_cm == -1.0:
                        per_sensor_neg1[sensor_idx] += 1
                    if raw_cm > 0.0:
                        per_sensor_positive[sensor_idx] += 1
                        per_sensor_values[sensor_idx].append(raw_cm)

                step_ranges.setdefault(step_idx, {})[sensor_idx] = raw_cm
                step_enabled.setdefault(step_idx, {})[sensor_idx] = enabled
                if step_idx not in step_pose:
                    step_pose[step_idx] = (
                        float(row["rover_x_cm"]),
                        float(row["rover_y_cm"]),
                        float(row["rover_heading_deg"]),
                    )

        prev_pose: tuple[float, float, float] | None = None
        for step_idx in sorted(step_ranges):
            enabled = np.asarray([step_enabled[step_idx].get(i, 0) for i in range(17)], dtype=np.int64)
            ranges = np.asarray([step_ranges[step_idx].get(i, float("nan")) for i in range(17)], dtype=np.float64)
            valid_positive = (enabled == 1) & (ranges > 0.0)
            per_step_enabled_counts.append(int(np.sum(enabled)))
            per_step_positive_counts.append(int(np.sum(valid_positive)))
            per_step_nonpositive_counts.append(int(np.sum((enabled == 1) & (ranges <= 0.0))))
            per_step_short_counts.append(int(np.sum(valid_positive & (ranges < 150.0))))
            per_step_far_counts.append(int(np.sum((enabled == 1) & (ranges >= 900.0))))

            pose = step_pose[step_idx]
            if prev_pose is not None:
                motion_dxy.append(math.hypot(pose[0] - prev_pose[0], pose[1] - prev_pose[1]))
                dyaw = ((pose[2] - prev_pose[2] + 180.0) % 360.0) - 180.0
                motion_dyaw.append(abs(dyaw))
            prev_pose = pose

    return {
        "dirs": dirs,
        "per_sensor_total": per_sensor_total,
        "per_sensor_enabled": per_sensor_enabled,
        "per_sensor_raw_nonneg": per_sensor_raw_nonneg,
        "per_sensor_raw_positive": per_sensor_raw_positive,
        "per_sensor_raw_zero": per_sensor_raw_zero,
        "per_sensor_raw_neg1": per_sensor_raw_neg1,
        "per_sensor_raw_values": per_sensor_raw_values,
        "per_sensor_positive": per_sensor_positive,
        "per_sensor_zero": per_sensor_zero,
        "per_sensor_neg1": per_sensor_neg1,
        "per_sensor_values": per_sensor_values,
        "per_sensor_reason_counts": per_sensor_reason_counts,
        "per_step_enabled_counts": np.asarray(per_step_enabled_counts, dtype=np.float64),
        "per_step_positive_counts": np.asarray(per_step_positive_counts, dtype=np.float64),
        "per_step_nonpositive_counts": np.asarray(per_step_nonpositive_counts, dtype=np.float64),
        "per_step_short_counts": np.asarray(per_step_short_counts, dtype=np.float64),
        "per_step_far_counts": np.asarray(per_step_far_counts, dtype=np.float64),
        "motion_dxy": np.asarray(motion_dxy, dtype=np.float64),
        "motion_dyaw": np.asarray(motion_dyaw, dtype=np.float64),
    }


def _array_summary(name: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"{name}: no data"
    return (
        f"{name}: mean={float(values.mean()):.2f} "
        f"p50={_median_str(values)} p95={_quantile_str(values, 0.95)} "
        f"p99={_quantile_str(values, 0.99)}"
    )


def build_report(train_stats: dict, real_stats: dict) -> str:
    lines: list[str] = []
    train_files = train_stats["files"]
    real_dirs = real_stats["dirs"]
    lines.append(f"Training files: {len(train_files)}")
    lines.append(f"Real smartdrive runs: {len(real_dirs)}")
    lines.append("")
    lines.append("Per-sensor range comparison")
    lines.append(
        "sensor train_valid% train_q50 train_q95 real_pipeline_enabled% "
        "real_raw_pos% real_raw_q50 real_raw_q95 real_raw_zero% real_raw_neg1%"
    )
    for sensor_idx in sorted(train_stats["per_sensor_total"]):
        train_values = np.asarray(train_stats["per_sensor_values"][sensor_idx], dtype=np.float64)
        real_values = np.asarray(real_stats["per_sensor_raw_values"][sensor_idx], dtype=np.float64)
        lines.append(
            f"{sensor_idx:>2} "
            f"{_pct(train_stats['per_sensor_valid'][sensor_idx], train_stats['per_sensor_total'][sensor_idx]):>11.1f} "
            f"{_median_str(train_values):>9} "
            f"{_quantile_str(train_values, 0.95):>9} "
            f"{_pct(real_stats['per_sensor_enabled'][sensor_idx], real_stats['per_sensor_total'][sensor_idx]):>21.1f} "
            f"{_pct(real_stats['per_sensor_raw_positive'][sensor_idx], real_stats['per_sensor_total'][sensor_idx]):>13.1f} "
            f"{_median_str(real_values):>12} "
            f"{_quantile_str(real_values, 0.95):>12} "
            f"{_pct(real_stats['per_sensor_raw_zero'][sensor_idx], real_stats['per_sensor_total'][sensor_idx]):>14.1f} "
            f"{_pct(real_stats['per_sensor_raw_neg1'][sensor_idx], real_stats['per_sensor_total'][sensor_idx]):>14.1f}"
        )

    lines.append("")
    lines.append("Top mean distance shifts")
    mean_shifts: list[tuple[float, int, float, float]] = []
    for sensor_idx in sorted(train_stats["per_sensor_total"]):
        train_values = np.asarray(train_stats["per_sensor_values"][sensor_idx], dtype=np.float64)
        real_values = np.asarray(real_stats["per_sensor_values"][sensor_idx], dtype=np.float64)
        if train_values.size == 0 or real_values.size == 0:
            continue
        train_mean = float(train_values.mean())
        real_mean = float(real_values.mean())
        mean_shifts.append((abs(real_mean - train_mean), sensor_idx, train_mean, real_mean))
    for _, sensor_idx, train_mean, real_mean in sorted(mean_shifts, reverse=True)[:10]:
        lines.append(
            f"sensor {sensor_idx}: train_mean={train_mean:.1f} "
            f"real_mean={real_mean:.1f} delta={real_mean - train_mean:.1f}"
        )

    lines.append("")
    lines.append(_array_summary("Train valid sensors per step", train_stats["per_step_valid_counts"]))
    lines.append(_array_summary("Real enabled sensors per step", real_stats["per_step_enabled_counts"]))
    lines.append(_array_summary("Real positive-range sensors per step", real_stats["per_step_positive_counts"]))
    lines.append(_array_summary("Real nonpositive enabled sensors per step", real_stats["per_step_nonpositive_counts"]))
    lines.append(_array_summary("Train short-range hits <150 cm per step", train_stats["per_step_short_counts"]))
    lines.append(_array_summary("Real short-range hits <150 cm per step", real_stats["per_step_short_counts"]))
    lines.append(_array_summary("Train far hits >=900 cm per step", train_stats["per_step_far_counts"]))
    lines.append(_array_summary("Real far hits >=900 cm per step", real_stats["per_step_far_counts"]))

    lines.append("")
    lines.append(_array_summary("Train step delta XY cm", train_stats["motion_dxy"]))
    lines.append(_array_summary("Real step delta XY cm", real_stats["motion_dxy"]))
    lines.append(_array_summary("Train step delta yaw deg", train_stats["motion_dyaw"]))
    lines.append(_array_summary("Real step delta yaw deg", real_stats["motion_dyaw"]))

    if train_stats["motion_dxy"].size > 0:
        lines.append(
            f"Train stationary delta_xy==0 fraction: {100.0 * float(np.mean(train_stats['motion_dxy'] == 0.0)):.2f}%"
        )
    if real_stats["motion_dxy"].size > 0:
        lines.append(
            f"Real stationary delta_xy==0 fraction: {100.0 * float(np.mean(real_stats['motion_dxy'] == 0.0)):.2f}%"
        )
    if train_stats["motion_dyaw"].size > 0:
        lines.append(
            f"Train stationary delta_yaw==0 fraction: {100.0 * float(np.mean(train_stats['motion_dyaw'] == 0.0)):.2f}%"
        )
    if real_stats["motion_dyaw"].size > 0:
        lines.append(
            f"Real stationary delta_yaw==0 fraction: {100.0 * float(np.mean(real_stats['motion_dyaw'] == 0.0)):.2f}%"
        )

    lines.append("")
    real_total = int(sum(real_stats["per_sensor_total"].values()))
    real_total_enabled = int(sum(real_stats["per_sensor_enabled"].values()))
    real_total_positive = int(sum(real_stats["per_sensor_raw_positive"].values()))
    real_total_zero = int(sum(real_stats["per_sensor_raw_zero"].values()))
    real_total_neg1 = int(sum(real_stats["per_sensor_raw_neg1"].values()))
    train_total = int(sum(train_stats["per_sensor_total"].values()))
    train_total_zero = int(sum(train_stats["per_sensor_zero"].values()))
    lines.append(f"Training exact-zero lidar readings: {_pct(train_total_zero, train_total):.2f}%")
    lines.append(f"Real raw exact-zero lidar readings: {_pct(real_total_zero, real_total):.2f}%")
    lines.append(f"Real raw -1 lidar readings: {_pct(real_total_neg1, real_total):.2f}%")
    lines.append(f"Real raw positive lidar readings: {_pct(real_total_positive, real_total):.2f}%")
    lines.append(f"Real pipeline-enabled fraction: {_pct(real_total_enabled, real_total):.2f}%")

    lines.append("")
    lines.append("Most common unusable reasons by sensor")
    for sensor_idx in sorted(real_stats["per_sensor_reason_counts"]):
        top = sorted(
            real_stats["per_sensor_reason_counts"][sensor_idx].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        top_fmt = ", ".join(f"{name}={count}" for name, count in top)
        lines.append(f"sensor {sensor_idx}: {top_fmt}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare real rover lidar logs against synthetic training data.")
    parser.add_argument("--train-dir", type=Path, default=Path("data"))
    parser.add_argument("--realdata-dir", type=Path, default=Path("realdata"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    train_stats = analyze_training(args.train_dir)
    real_stats = analyze_real(args.realdata_dir)
    report = build_report(train_stats, real_stats)
    print(report)
    if args.output is not None:
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
