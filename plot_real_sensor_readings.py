from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def collect_real_sensor_data(realdata_dir: Path) -> tuple[dict[int, list[float]], list[Path]]:
    per_sensor: dict[int, list[float]] = defaultdict(list)
    runs = sorted(realdata_dir.glob("smartdrive_debug_*/lidar_points.txt"))
    for path in runs:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                raw = row["raw_cm"].strip().lower()
                if raw == "nan":
                    continue
                per_sensor[int(row["sensor_idx"])].append(float(raw))
    return per_sensor, runs


def _sensor_label_lookup(realdata_dir: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    runs = sorted(realdata_dir.glob("smartdrive_debug_*/lidar_points.txt"))
    for path in runs:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                idx = int(row["sensor_idx"])
                labels.setdefault(idx, row["sensor_label"])
    return labels


def save_zero_summary(
    per_sensor: dict[int, list[float]],
    labels: dict[int, str],
    output_csv: Path,
) -> None:
    rows: list[tuple[int, str, int, int, int, int, float, float]] = []
    for sensor_idx in sorted(per_sensor):
        values = np.asarray(per_sensor[sensor_idx], dtype=np.float64)
        total = int(values.size)
        zero_count = int(np.sum(values == 0.0))
        neg1_count = int(np.sum(values == -1.0))
        positive_count = int(np.sum(values > 0.0))
        zero_pct = (100.0 * zero_count / total) if total else 0.0
        neg1_pct = (100.0 * neg1_count / total) if total else 0.0
        rows.append(
            (
                sensor_idx,
                labels.get(sensor_idx, f"sensor_{sensor_idx}"),
                total,
                zero_count,
                neg1_count,
                positive_count,
                zero_pct,
                neg1_pct,
            )
        )

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "sensor_idx",
                "sensor_label",
                "total_readings",
                "zero_count",
                "neg1_count",
                "positive_count",
                "zero_pct",
                "neg1_pct",
            ]
        )
        writer.writerows(rows)


def plot_sensor_distribution(
    sensor_idx: int,
    sensor_label: str,
    values: np.ndarray,
    out_path_with_neg1: Path,
    out_path_without_neg1: Path,
) -> None:
    zero_count = int(np.sum(values == 0.0))
    neg1_count = int(np.sum(values == -1.0))
    positive = values[values > 0.0]

    def _draw_common_density(ax) -> None:
        if positive.size > 1:
            density_bins = np.linspace(0.0, max(1000.0, float(np.quantile(positive, 0.995))), 120)
            hist, edges = np.histogram(positive, bins=density_bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, hist, color="#2ca02c", linewidth=2.0)
            ax.fill_between(centers, hist, color="#2ca02c", alpha=0.2)
        ax.set_title(f"Sensor {sensor_idx} positive-reading density")
        ax.set_xlabel("raw_cm")
        ax.set_ylabel("density")

    # Version 1: explicit separate -1 and 0 bars.
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 4.8), dpi=150)
    pos_bins = np.arange(0.0, 1000.0 + 25.0, 25.0)
    pos_hist, pos_edges = np.histogram(positive, bins=pos_bins)
    pos_centers = 0.5 * (pos_edges[:-1] + pos_edges[1:])
    axes1[0].bar(pos_centers, pos_hist, width=22.0, color="#1f77b4", alpha=0.85, edgecolor="white")
    axes1[0].bar([-1.0], [neg1_count], width=0.8, color="#9467bd", alpha=0.95, label="-1")
    axes1[0].bar([0.0], [zero_count], width=0.8, color="#d62728", alpha=0.95, label="0")
    axes1[0].set_title(f"Sensor {sensor_idx} histogram with explicit -1/0 bars")
    axes1[0].set_xlabel("raw_cm")
    axes1[0].set_ylabel("count")
    axes1[0].legend()
    _draw_common_density(axes1[1])
    fig1.suptitle(
        f"{sensor_label} | total={values.size} zero={zero_count} ({(100.0 * zero_count / values.size):.1f}%) "
        f"-1={neg1_count} ({(100.0 * neg1_count / values.size):.1f}%)",
        fontsize=11,
    )
    fig1.tight_layout()
    fig1.savefig(out_path_with_neg1, bbox_inches="tight")
    plt.close(fig1)

    # Version 2: only nonnegative values, so the positive-range shape is easy to read.
    nonnegative = values[values >= 0.0]
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4.8), dpi=150)
    axes2[0].hist(nonnegative, bins=pos_bins, color="#1f77b4", alpha=0.85, edgecolor="white")
    axes2[0].axvline(0.0, color="#d62728", linestyle="--", linewidth=1.5, label="zero")
    axes2[0].set_title(f"Sensor {sensor_idx} histogram without -1 values")
    axes2[0].set_xlabel("raw_cm")
    axes2[0].set_ylabel("count")
    axes2[0].legend()
    _draw_common_density(axes2[1])
    fig2.suptitle(
        f"{sensor_label} | nonnegative-only view | total={values.size} zero={zero_count} "
        f"-1 hidden={neg1_count}",
        fontsize=11,
    )
    fig2.tight_layout()
    fig2.savefig(out_path_without_neg1, bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-sensor real rover lidar reading distributions.")
    parser.add_argument("--realdata-dir", type=Path, default=Path("realdata"))
    parser.add_argument("--output-dir", type=Path, default=Path("real_sensor_plots"))
    args = parser.parse_args()

    per_sensor, runs = collect_real_sensor_data(args.realdata_dir)
    labels = _sensor_label_lookup(args.realdata_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_zero_summary(per_sensor, labels, args.output_dir / "sensor_zero_summary.csv")

    for sensor_idx in sorted(per_sensor):
        sensor_label = labels.get(sensor_idx, f"sensor_{sensor_idx}")
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sensor_label)
        out_path_with_neg1 = args.output_dir / f"sensor_{sensor_idx:02d}_{safe_label}_with_neg1.png"
        out_path_without_neg1 = args.output_dir / f"sensor_{sensor_idx:02d}_{safe_label}_without_neg1.png"
        plot_sensor_distribution(
            sensor_idx,
            sensor_label,
            np.asarray(per_sensor[sensor_idx], dtype=np.float64),
            out_path_with_neg1,
            out_path_without_neg1,
        )

    print(f"Processed {len(runs)} runs")
    print(f"Wrote plots to {args.output_dir}")
    print(f"Wrote summary to {args.output_dir / 'sensor_zero_summary.csv'}")


if __name__ == "__main__":
    main()
