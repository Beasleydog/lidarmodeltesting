from __future__ import annotations

import argparse
import csv
from pathlib import Path


CLASS_LABELS = {
    0: "ground",
    1: "obstacle",
    2: "none",
}


def _sensor_keys(fieldnames: list[str], prefix: str) -> list[str]:
    keys = [key for key in fieldnames if key.startswith(prefix)]
    keys.sort(key=lambda key: int(key.rsplit("_", 1)[1]))
    return keys


def _fmt_pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * float(count) / float(total)):.2f}%"


def _class_name(class_id: int) -> str:
    return CLASS_LABELS.get(class_id, f"class_{class_id}")


def analyze_dataset(data_dir: Path, pattern: str) -> int:
    files = sorted(data_dir.glob(pattern))
    if not files:
        print(f"No files matched {pattern!r} in {data_dir}")
        return 1

    total_rows = 0
    total_teleports = 0
    total_lidar_samples = 0
    total_valid_lidar = 0
    distance_sum_cm = 0.0
    abs_move_sum_cm = 0.0
    abs_turn_sum_deg = 0.0
    min_rows_per_file: int | None = None
    max_rows_per_file = 0
    sensor_count: int | None = None
    skipped_files = 0

    overall_class_counts: dict[int, int] = {}
    per_sensor_class_counts: list[dict[int, int]] = []

    for path in files:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                print(f"Skipping {path.name}: missing header")
                skipped_files += 1
                continue

            cm_keys = _sensor_keys(reader.fieldnames, "lidar_cm_")
            class_keys = _sensor_keys(reader.fieldnames, "lidar_class_")
            if len(cm_keys) == 0 or len(cm_keys) != len(class_keys):
                print(f"Skipping {path.name}: invalid lidar columns")
                skipped_files += 1
                continue

            if sensor_count is None:
                sensor_count = len(cm_keys)
                per_sensor_class_counts = [{} for _ in range(sensor_count)]
            elif sensor_count != len(cm_keys):
                print(
                    f"Skipping {path.name}: sensor count changed "
                    f"({len(cm_keys)} != {sensor_count})"
                )
                skipped_files += 1
                continue

            rows_in_file = 0
            for row in reader:
                rows_in_file += 1
                total_rows += 1
                total_teleports += int(float(row.get("teleport_flag", "0")))
                abs_move_sum_cm += abs(float(row.get("cmd_move_cm", "0")))
                abs_turn_sum_deg += abs(float(row.get("cmd_turn_deg", "0")))

                for sensor_idx, class_key in enumerate(class_keys):
                    class_id = int(row[class_key])
                    overall_class_counts[class_id] = overall_class_counts.get(class_id, 0) + 1
                    sensor_counts = per_sensor_class_counts[sensor_idx]
                    sensor_counts[class_id] = sensor_counts.get(class_id, 0) + 1
                    total_lidar_samples += 1

                for cm_key in cm_keys:
                    distance_cm = float(row[cm_key])
                    if distance_cm >= 0.0:
                        total_valid_lidar += 1
                        distance_sum_cm += distance_cm

            min_rows_per_file = rows_in_file if min_rows_per_file is None else min(min_rows_per_file, rows_in_file)
            max_rows_per_file = max(max_rows_per_file, rows_in_file)

    processed_files = len(files) - skipped_files
    if processed_files <= 0 or sensor_count is None:
        print("No valid data files were processed.")
        return 1

    avg_rows_per_file = float(total_rows) / float(processed_files)
    avg_valid_distance_cm = distance_sum_cm / float(total_valid_lidar) if total_valid_lidar > 0 else 0.0
    avg_abs_move_cm = abs_move_sum_cm / float(total_rows) if total_rows > 0 else 0.0
    avg_abs_turn_deg = abs_turn_sum_deg / float(total_rows) if total_rows > 0 else 0.0

    print(f"Dataset: {data_dir.resolve()}")
    print(f"Files processed: {processed_files} (skipped {skipped_files})")
    print(
        "Timesteps: "
        f"{total_rows} total | avg/file {avg_rows_per_file:.1f} | "
        f"min/file {min_rows_per_file} | max/file {max_rows_per_file}"
    )
    print(
        "Motion: "
        f"avg |cmd_move_cm| {avg_abs_move_cm:.2f} | "
        f"avg |cmd_turn_deg| {avg_abs_turn_deg:.2f}"
    )
    print(f"Teleports: {total_teleports} ({_fmt_pct(total_teleports, total_rows)} of timesteps)")
    print(
        "Lidar returns: "
        f"{total_valid_lidar}/{total_lidar_samples} valid "
        f"({_fmt_pct(total_valid_lidar, total_lidar_samples)}) | "
        f"mean valid distance {avg_valid_distance_cm:.2f} cm"
    )

    print("\nOverall lidar class breakdown:")
    total_classified = sum(overall_class_counts.values())
    for class_id in sorted(overall_class_counts):
        count = overall_class_counts[class_id]
        print(f"  {_class_name(class_id):>8}: {count:>10} ({_fmt_pct(count, total_classified)})")

    print("\nPer-sensor lidar class breakdown:")
    for sensor_idx, sensor_counts in enumerate(per_sensor_class_counts):
        sensor_total = sum(sensor_counts.values())
        parts = []
        for class_id in sorted(CLASS_LABELS):
            count = sensor_counts.get(class_id, 0)
            parts.append(f"{_class_name(class_id)}={_fmt_pct(count, sensor_total)}")
        extra_class_ids = sorted(class_id for class_id in sensor_counts if class_id not in CLASS_LABELS)
        for class_id in extra_class_ids:
            count = sensor_counts[class_id]
            parts.append(f"{_class_name(class_id)}={_fmt_pct(count, sensor_total)}")
        print(f"  sensor {sensor_idx:02d}: " + " | ".join(parts))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize lidar dataset files in the data directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pattern", default="world_*.txt")
    args = parser.parse_args()
    return analyze_dataset(args.data_dir, args.pattern)


if __name__ == "__main__":
    raise SystemExit(main())
