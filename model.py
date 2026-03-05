from __future__ import annotations

from pathlib import Path

import numpy as np

from train import load_gru_lidar_inferencer


RUNS_DIR = Path(__file__).resolve().parent / "runs"
CHECKPOINT_PATH = RUNS_DIR / "gru_lidar_classifier.pt"
MAX_HISTORY = 64
DEFAULT_OBSTACLE_LOGIT_BIAS = 0.0

_INFERENCER = load_gru_lidar_inferencer(CHECKPOINT_PATH, max_history=MAX_HISTORY)
_FEATURE_HISTORY: list[np.ndarray] = []


def reset_history() -> None:
    _FEATURE_HISTORY.clear()


def ingest_lidar(
    lidar_cm: np.ndarray,
    pose_xyz_cm: np.ndarray,
    basis: np.ndarray | None = None,
    obstacle_logit_bias: float = DEFAULT_OBSTACLE_LOGIT_BIAS,
) -> dict[str, np.ndarray]:
    lidar_arr = np.asarray(lidar_cm, dtype=np.float32).reshape(-1)
    pose_arr = np.asarray(pose_xyz_cm, dtype=np.float32).reshape(-1)
    basis_arr = None if basis is None else np.asarray(basis, dtype=np.float32).reshape(3, 3)

    feature_t = _INFERENCER.featurize_timestep(
        pose_arr,
        lidar_arr,
        basis=basis_arr,
    )
    history = np.asarray([*_FEATURE_HISTORY, feature_t], dtype=np.float32)

    if _INFERENCER.binary_obstacle_only:
        obstacle_logits = _INFERENCER.predict_current_obstacle_logits_from_feature_history(history)
        obstacle_mask = _INFERENCER.predict_current_obstacle_mask_from_feature_history_with_bias(
            history,
            obstacle_logit_bias=float(obstacle_logit_bias),
        )
        class_ids = np.where(obstacle_mask, 1, 0).astype(np.int64)
        output = {
            "class_ids": class_ids,
            "obstacle_mask": obstacle_mask,
            "obstacle_logits": obstacle_logits,
        }
    else:
        class_ids = _INFERENCER.predict_current_from_feature_history(history).astype(np.int64)
        output = {
            "class_ids": class_ids,
            "obstacle_mask": (class_ids == 1),
        }

    _FEATURE_HISTORY.append(feature_t)
    if _INFERENCER.max_history > 0 and len(_FEATURE_HISTORY) > _INFERENCER.max_history:
        del _FEATURE_HISTORY[:-_INFERENCER.max_history]

    return output
