from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from environment_demo import (
    DEFORMATION_EPS_CM,
    GRID_SIZE,
    ROVER_MOVE_STEP_CM,
    ROVER_TURN_STEP_DEG,
    WORLD_SIZE_CM,
    WORLDGEN_NOISE_BASE_CELLS_RANGE,
    WORLDGEN_NOISE_OCTAVES_RANGE,
    WORLDGEN_NOISE_PERSISTENCE_RANGE,
    WORLDGEN_OBSTACLE_COUNT_RANGE,
    WORLDGEN_OBSTACLE_EDGE_START_FRAC_RANGE,
    WORLDGEN_OBSTACLE_HEIGHT_RANGE_CM,
    WORLDGEN_OBSTACLE_MAX_OVERLAP_RATIO_RANGE,
    WORLDGEN_OBSTACLE_RADIUS_MAX_FRAC_RANGE,
    WORLDGEN_OBSTACLE_RADIUS_MIN_FRAC_RANGE,
    WORLDGEN_OBSTACLE_SIDE_EXPONENT_RANGE,
    WORLDGEN_OBSTACLE_TALL_HEIGHT_RANGE_CM,
    WORLDGEN_OBSTACLE_TALL_PROB_RANGE,
    WORLDGEN_TERRAIN_HEIGHT_SCALE_RANGE_CM,
    RoverState,
    build_height_sampler,
    clamp_rover_state_xy,
    compute_rover_pose,
    generate_environment,
    get_rover_edge_local_xy,
    get_rover_footprint_local,
    make_lidar_distance_samples,
    move_rover_forward,
    run_lidar_scan,
    turn_rover,
    view_environment,
)

# ===== Data Config =====
SHOW_VIEWER = False
DATA_DIR = Path("data")
WORLDS_TO_GENERATE = 3000
TIMESTEPS_PER_WORLD = 300
LOG_EVERY_N_STEPS = 1
BASE_RANDOM_SEED: int | None = None
VIEWER_PLAYBACK_SLEEP_S = 0.04
HOLD_VIEWER_AFTER_PLAYBACK = False
NUM_WORKERS = 0  # Use >1 to generate worlds in parallel. If <=0, auto-selects CPU-based worker count.

# Motion model control
MAX_MOVE_CM_PER_STEP = float(ROVER_MOVE_STEP_CM)
MAX_TURN_DEG_PER_STEP = float(ROVER_TURN_STEP_DEG)
CONTROL_SEGMENT_STEPS_MIN = 8
CONTROL_SEGMENT_STEPS_MAX = 32
THROTTLE_SLEW_PER_STEP = 0.08
STEERING_SLEW_PER_STEP = 0.12

# Throttle target sampling
THROTTLE_FORWARD_PROB = 0.88
THROTTLE_FORWARD_MIN = 0.35
THROTTLE_FORWARD_BETA_A = 5.0
THROTTLE_FORWARD_BETA_B = 2.0
THROTTLE_REVERSE_MIN = 0.08
THROTTLE_REVERSE_MAX = 0.55

# Steering target sampling (random base target, then obstacle attractor is applied per step)
STEERING_BASE_STD = 0.70

# Per-step obstacle attractors (nearest obstacle)
STEERING_ATTRACTION_YAW_SCALE_DEG = 70.0
STEERING_ATTRACTOR_STRENGTH = 0.32
THROTTLE_ATTRACTOR_STRENGTH = 0.26
ATTRACTOR_EFFECT_MIN_DIST_CM = 220.0
ATTRACTOR_EFFECT_MAX_DIST_CM = 2200.0

# Main navigation target behavior: when reached, jump to farthest obstacle.
NAV_TARGET_REACHED_DIST_CM = 420.0
NAV_TARGET_STEER_STRENGTH = 0.52
NAV_TARGET_THROTTLE_STRENGTH = 0.34
NAV_TARGET_MAX_STEPS = 96

# Per-world navigation policy mix.
NAV_MODE_NEAREST_THEN_FARTHEST = "nearest_then_farthest"
NAV_MODE_FARTHEST_THEN_NEAREST = "farthest_then_nearest"
NAV_MODE_RANDOM_WAYPOINT = "random_waypoint"
NAV_MODE_CHOICES = (
    NAV_MODE_NEAREST_THEN_FARTHEST,
    NAV_MODE_FARTHEST_THEN_NEAREST,
    NAV_MODE_RANDOM_WAYPOINT,
)
NAV_MODE_PROBS = (0.45, 0.35, 0.20)
RANDOM_MODE_OBSTACLE_TARGET_PROB = 0.70
RANDOM_MODE_NEAREST_ATTRACTOR_SCALE = 0.55

# Spawn behavior
SPAWN_NEAR_OBSTACLE_MIN_CM = 140.0
SPAWN_NEAR_OBSTACLE_MAX_CM = 900.0
SPAWN_NEAR_OBSTACLE_DIST_BIAS = 0.72
SPAWN_NEAR_OBSTACLE_RETRIES = 24
SPAWN_YAW_NOISE_STD_DEG = 30.0

# Safety behavior: if rover center ends up on obstacle-deformed terrain, relocate to free ground.
TELEPORT_ON_OBSTACLE_CONTACT = True
TELEPORT_SAFE_CLEARANCE_CM = 260.0
TELEPORT_PREFERRED_CLEARANCE_CM = 420.0
TELEPORT_INNER_WORLD_FRACTION = 0.50
TELEPORT_MAX_ATTEMPTS = 90
# Buffer so teleport triggers only after meaningful penetration into obstacle deformation.
TELEPORT_MIN_DEFORMATION_CM = 24.0
TELEPORT_MIN_DEFORMATION_WORLD_P90_RATIO = 0.08
TELEPORT_MIN_PROBE_HITS = 2

# Per-world world-style mixture. This biases random worldgen ranges.
WORLD_STYLE_OPEN = "open"
WORLD_STYLE_BALANCED = "balanced"
WORLD_STYLE_DENSE = "dense"
WORLD_STYLE_CHOICES = (WORLD_STYLE_OPEN, WORLD_STYLE_BALANCED, WORLD_STYLE_DENSE)
WORLD_STYLE_PROBS = (0.28, 0.48, 0.24)


def _new_world_filename(world_idx: int, world_seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return DATA_DIR / f"world_{world_idx:03d}_{stamp}_seed_{world_seed}.txt"


def _world_rng_seed(world_idx: int, world_seed: int) -> int:
    # Stable per-world RNG seed so parallel and serial generation produce consistent behavior.
    mix = (int(world_seed) ^ (int(world_idx) * 2654435761)) & 0xFFFFFFFF
    return int(mix)


def _wrap_angle_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _move_toward(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target


def _sample_uniform_biased(rng: np.random.Generator, low: float, high: float, bias01: float) -> float:
    if high <= low:
        return float(low)
    mix = float(np.clip(0.65 * bias01 + 0.35 * rng.random(), 0.0, 1.0))
    return float(low + (high - low) * mix)


def _sample_int_biased(rng: np.random.Generator, low: int, high: int, bias01: float) -> int:
    if high <= low:
        return int(low)
    value = _sample_uniform_biased(rng, float(low), float(high), bias01)
    return int(np.clip(int(round(value)), low, high))


def _sample_world_generation_kwargs(rng: np.random.Generator) -> tuple[str, dict[str, float | int]]:
    style = str(rng.choice(np.asarray(WORLD_STYLE_CHOICES), p=np.asarray(WORLD_STYLE_PROBS, dtype=float)))
    if style == WORLD_STYLE_OPEN:
        bias = 0.23
    elif style == WORLD_STYLE_DENSE:
        bias = 0.82
    else:
        bias = 0.52

    obstacle_count = _sample_int_biased(
        rng,
        int(WORLDGEN_OBSTACLE_COUNT_RANGE[0]),
        int(WORLDGEN_OBSTACLE_COUNT_RANGE[1]),
        bias,
    )
    terrain_height_scale_cm = _sample_uniform_biased(
        rng,
        float(WORLDGEN_TERRAIN_HEIGHT_SCALE_RANGE_CM[0]),
        float(WORLDGEN_TERRAIN_HEIGHT_SCALE_RANGE_CM[1]),
        bias,
    )
    noise_octaves = _sample_int_biased(
        rng,
        int(WORLDGEN_NOISE_OCTAVES_RANGE[0]),
        int(WORLDGEN_NOISE_OCTAVES_RANGE[1]),
        0.42 + 0.25 * bias,
    )
    noise_base_cells = _sample_int_biased(
        rng,
        int(WORLDGEN_NOISE_BASE_CELLS_RANGE[0]),
        int(WORLDGEN_NOISE_BASE_CELLS_RANGE[1]),
        0.50 - 0.25 * bias,
    )
    noise_persistence = _sample_uniform_biased(
        rng,
        float(WORLDGEN_NOISE_PERSISTENCE_RANGE[0]),
        float(WORLDGEN_NOISE_PERSISTENCE_RANGE[1]),
        0.55 + 0.30 * bias,
    )

    radius_min_frac = _sample_uniform_biased(
        rng,
        float(WORLDGEN_OBSTACLE_RADIUS_MIN_FRAC_RANGE[0]),
        float(WORLDGEN_OBSTACLE_RADIUS_MIN_FRAC_RANGE[1]),
        0.32 + 0.45 * bias,
    )
    radius_max_frac = _sample_uniform_biased(
        rng,
        float(max(radius_min_frac + 0.001, WORLDGEN_OBSTACLE_RADIUS_MAX_FRAC_RANGE[0])),
        float(WORLDGEN_OBSTACLE_RADIUS_MAX_FRAC_RANGE[1]),
        0.38 + 0.48 * bias,
    )

    obs_h_low = float(WORLDGEN_OBSTACLE_HEIGHT_RANGE_CM[0])
    obs_h_high = float(WORLDGEN_OBSTACLE_HEIGHT_RANGE_CM[1])
    obstacle_height_min_cm = _sample_uniform_biased(rng, obs_h_low, obs_h_low + 0.50 * (obs_h_high - obs_h_low), bias)
    obstacle_height_max_cm = _sample_uniform_biased(
        rng,
        obstacle_height_min_cm + 70.0,
        obs_h_high,
        0.45 + 0.50 * bias,
    )

    tall_h_low = float(WORLDGEN_OBSTACLE_TALL_HEIGHT_RANGE_CM[0])
    tall_h_high = float(WORLDGEN_OBSTACLE_TALL_HEIGHT_RANGE_CM[1])
    obstacle_tall_min_cm = _sample_uniform_biased(rng, tall_h_low, tall_h_high - 120.0, 0.40 + 0.50 * bias)
    obstacle_tall_min_cm = max(obstacle_tall_min_cm, obstacle_height_max_cm * 0.85)
    obstacle_tall_max_cm = _sample_uniform_biased(
        rng,
        obstacle_tall_min_cm + 90.0,
        tall_h_high,
        0.52 + 0.38 * bias,
    )

    obstacle_tall_prob = _sample_uniform_biased(
        rng,
        float(WORLDGEN_OBSTACLE_TALL_PROB_RANGE[0]),
        float(WORLDGEN_OBSTACLE_TALL_PROB_RANGE[1]),
        0.36 + 0.48 * bias,
    )
    obstacle_max_overlap_ratio = _sample_uniform_biased(
        rng,
        float(WORLDGEN_OBSTACLE_MAX_OVERLAP_RATIO_RANGE[0]),
        float(WORLDGEN_OBSTACLE_MAX_OVERLAP_RATIO_RANGE[1]),
        0.40 + 0.45 * bias,
    )
    obstacle_edge_start_frac = _sample_uniform_biased(
        rng,
        float(WORLDGEN_OBSTACLE_EDGE_START_FRAC_RANGE[0]),
        float(WORLDGEN_OBSTACLE_EDGE_START_FRAC_RANGE[1]),
        0.55 - 0.28 * bias,
    )
    obstacle_side_exponent = _sample_uniform_biased(
        rng,
        float(WORLDGEN_OBSTACLE_SIDE_EXPONENT_RANGE[0]),
        float(WORLDGEN_OBSTACLE_SIDE_EXPONENT_RANGE[1]),
        0.30 + 0.38 * bias,
    )
    return style, {
        "obstacle_count": obstacle_count,
        "noise_octaves": noise_octaves,
        "noise_base_cells": noise_base_cells,
        "noise_persistence": noise_persistence,
        "terrain_height_scale_cm": terrain_height_scale_cm,
        "obstacle_radius_min_frac": radius_min_frac,
        "obstacle_radius_max_frac": radius_max_frac,
        "obstacle_height_min_cm": obstacle_height_min_cm,
        "obstacle_height_max_cm": obstacle_height_max_cm,
        "obstacle_tall_min_cm": obstacle_tall_min_cm,
        "obstacle_tall_max_cm": obstacle_tall_max_cm,
        "obstacle_tall_prob": obstacle_tall_prob,
        "obstacle_max_overlap_ratio": obstacle_max_overlap_ratio,
        "obstacle_edge_start_frac": obstacle_edge_start_frac,
        "obstacle_side_exponent": obstacle_side_exponent,
    }


def _pick_random_obstacle_target(
    env,
    occupied_indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if occupied_indices.size == 0:
        x = float(rng.uniform(float(env.x.min()), float(env.x.max())))
        y = float(rng.uniform(float(env.y.min()), float(env.y.max())))
        return x, y
    idx = int(rng.integers(0, occupied_indices.shape[0]))
    iy, ix = occupied_indices[idx]
    return float(env.x[iy, ix]), float(env.y[iy, ix])


def _pick_random_world_target(env, rng: np.random.Generator) -> tuple[float, float]:
    return (
        float(rng.uniform(float(env.x.min()), float(env.x.max()))),
        float(rng.uniform(float(env.y.min()), float(env.y.max()))),
    )


def _xy_to_grid_index(env, x: float, y: float) -> tuple[int, int]:
    x_axis = env.x[0]
    y_axis = env.y[:, 0]
    ix = int(np.argmin(np.abs(x_axis - x)))
    iy = int(np.argmin(np.abs(y_axis - y)))
    return iy, ix


def _sample_grid_bilinear(grid: np.ndarray, sampler, x: float, y: float) -> float:
    if x < sampler.x0 or x > sampler.x1 or y < sampler.y0 or y > sampler.y1:
        return 0.0
    fx = (x - sampler.x0) / sampler.dx
    fy = (y - sampler.y0) / sampler.dy
    ix = int(np.clip(np.floor(fx), 0, sampler.max_ix))
    iy = int(np.clip(np.floor(fy), 0, sampler.max_iy))
    tx = float(fx - ix)
    ty = float(fy - iy)
    z00 = float(grid[iy, ix])
    z10 = float(grid[iy, ix + 1])
    z01 = float(grid[iy + 1, ix])
    z11 = float(grid[iy + 1, ix + 1])
    return (1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11


def _deformation_at_xy_cm(deformation_grid: np.ndarray, sampler, x: float, y: float) -> float:
    return float(_sample_grid_bilinear(deformation_grid, sampler, x, y))


def _is_state_on_obstacle(
    state: RoverState,
    deformation_grid: np.ndarray,
    sampler,
    footprint_local_xy: np.ndarray,
    deformation_threshold_cm: float,
) -> bool:
    yaw = np.deg2rad(state.yaw_deg)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    footprint_world = (rot @ footprint_local_xy.T).T + np.array([state.x, state.y], dtype=float)
    center = footprint_world.mean(axis=0, keepdims=True)
    mids = np.vstack(
        (
            0.5 * (footprint_world[0] + footprint_world[1]),
            0.5 * (footprint_world[1] + footprint_world[2]),
            0.5 * (footprint_world[2] + footprint_world[3]),
            0.5 * (footprint_world[3] + footprint_world[0]),
        )
    )
    probes = np.vstack((center, footprint_world, mids))
    hit_count = 0
    for px, py in probes:
        if _deformation_at_xy_cm(deformation_grid, sampler, float(px), float(py)) >= deformation_threshold_cm:
            hit_count += 1
            if hit_count >= TELEPORT_MIN_PROBE_HITS:
                return True
    return False


def _nearest_obstacle_distance_cm(x: float, y: float, obstacle_points_xy: np.ndarray) -> float:
    if obstacle_points_xy.size == 0:
        return float("inf")
    dx = obstacle_points_xy[:, 0] - x
    dy = obstacle_points_xy[:, 1] - y
    return float(np.sqrt(np.min(dx * dx + dy * dy)))


def _pick_nearest_obstacle_to_xy(x: float, y: float, obstacle_points_xy: np.ndarray) -> tuple[float, float]:
    if obstacle_points_xy.size == 0:
        return x, y
    dx = obstacle_points_xy[:, 0] - x
    dy = obstacle_points_xy[:, 1] - y
    idx = int(np.argmin(dx * dx + dy * dy))
    return float(obstacle_points_xy[idx, 0]), float(obstacle_points_xy[idx, 1])


def _is_within_inner_teleport_region(env, x: float, y: float) -> bool:
    x_min = float(env.x.min())
    x_max = float(env.x.max())
    y_min = float(env.y.min())
    y_max = float(env.y.max())
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_half = 0.5 * (x_max - x_min) * TELEPORT_INNER_WORLD_FRACTION
    y_half = 0.5 * (y_max - y_min) * TELEPORT_INNER_WORLD_FRACTION
    return (
        (x >= (x_center - x_half))
        and (x <= (x_center + x_half))
        and (y >= (y_center - y_half))
        and (y <= (y_center + y_half))
    )


def _yaw_toward_target_with_noise(
    x: float,
    y: float,
    target_x: float,
    target_y: float,
    rng: np.random.Generator,
    noise_std_deg: float = SPAWN_YAW_NOISE_STD_DEG,
) -> float:
    dx = float(target_x - x)
    dy = float(target_y - y)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return float(rng.uniform(-180.0, 180.0))
    yaw_deg = float(np.rad2deg(np.arctan2(dy, dx)))
    return yaw_deg + float(rng.normal(0.0, noise_std_deg))


def _teleport_state_to_safe_ground(
    state: RoverState,
    env,
    deformation_grid: np.ndarray,
    sampler,
    footprint_local_xy: np.ndarray,
    deformation_threshold_cm: float,
    obstacle_points_xy: np.ndarray,
    rng: np.random.Generator,
) -> None:
    free_indices = np.argwhere(~env.occupancy)
    if free_indices.size == 0:
        state.x, state.y = _pick_random_world_target(env, rng)
        state.yaw_deg = float(rng.uniform(-180.0, 180.0))
        clamp_rover_state_xy(state, env)
        return
    inner_free_indices = np.asarray(
        [
            (iy, ix)
            for iy, ix in free_indices
            if _is_within_inner_teleport_region(env, float(env.x[iy, ix]), float(env.y[iy, ix]))
        ],
        dtype=np.int64,
    )

    x_axis = env.x[0]
    y_axis = env.y[:, 0]
    dx_cell = float(abs(x_axis[1] - x_axis[0])) if x_axis.shape[0] > 1 else 0.0
    dy_cell = float(abs(y_axis[1] - y_axis[0])) if y_axis.shape[0] > 1 else 0.0
    jitter_x = 0.35 * dx_cell
    jitter_y = 0.35 * dy_cell

    best_safe_candidate: tuple[float, float, float] | None = None
    best_safe_clearance = float("inf")
    fallback_candidate: tuple[float, float, float] | None = None
    fallback_clearance = -1.0
    for _ in range(TELEPORT_MAX_ATTEMPTS):
        if obstacle_points_xy.size > 0:
            pick = int(rng.integers(0, obstacle_points_xy.shape[0]))
            ox = float(obstacle_points_xy[pick, 0])
            oy = float(obstacle_points_xy[pick, 1])
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            dist = _sample_uniform_biased(
                rng,
                TELEPORT_SAFE_CLEARANCE_CM,
                TELEPORT_PREFERRED_CLEARANCE_CM,
                0.75,
            )
            cx = ox + dist * float(np.cos(angle)) + float(rng.uniform(-jitter_x, jitter_x))
            cy = oy + dist * float(np.sin(angle)) + float(rng.uniform(-jitter_y, jitter_y))
        else:
            source_indices = inner_free_indices if inner_free_indices.size > 0 else free_indices
            pick = int(rng.integers(0, source_indices.shape[0]))
            iy, ix = source_indices[pick]
            cx = float(env.x[iy, ix] + rng.uniform(-jitter_x, jitter_x))
            cy = float(env.y[iy, ix] + rng.uniform(-jitter_y, jitter_y))
        probe = RoverState(x=cx, y=cy, yaw_deg=float(rng.uniform(-180.0, 180.0)))
        clamp_rover_state_xy(probe, env)
        if not _is_within_inner_teleport_region(env, probe.x, probe.y):
            continue
        if _is_state_on_obstacle(
            probe,
            deformation_grid,
            sampler,
            footprint_local_xy,
            deformation_threshold_cm=deformation_threshold_cm,
        ):
            continue
        clearance = _nearest_obstacle_distance_cm(probe.x, probe.y, obstacle_points_xy)
        if clearance > fallback_clearance:
            fallback_clearance = clearance
            fallback_candidate = (probe.x, probe.y, probe.yaw_deg)
        if clearance < TELEPORT_SAFE_CLEARANCE_CM:
            continue
        near_ox, near_oy = _pick_nearest_obstacle_to_xy(probe.x, probe.y, obstacle_points_xy)
        probe_yaw_deg = _yaw_toward_target_with_noise(probe.x, probe.y, near_ox, near_oy, rng)
        if clearance < best_safe_clearance:
            best_safe_clearance = clearance
            best_safe_candidate = (probe.x, probe.y, probe_yaw_deg)
        if clearance <= TELEPORT_PREFERRED_CLEARANCE_CM:
            state.x, state.y, state.yaw_deg = probe.x, probe.y, probe_yaw_deg
            return

    if best_safe_candidate is not None:
        state.x, state.y, state.yaw_deg = best_safe_candidate
    elif fallback_candidate is not None:
        state.x, state.y, state.yaw_deg = fallback_candidate
    else:
        source_indices = inner_free_indices if inner_free_indices.size > 0 else free_indices
        pick = int(rng.integers(0, source_indices.shape[0]))
        iy, ix = source_indices[pick]
        state.x = float(env.x[iy, ix])
        state.y = float(env.y[iy, ix])
        state.yaw_deg = float(rng.uniform(-180.0, 180.0))
        clamp_rover_state_xy(state, env)


def _nearest_obstacle_metrics(
    state: RoverState,
    obstacle_points_xy: np.ndarray,
) -> tuple[float, float, float]:
    if obstacle_points_xy.size == 0:
        return float("inf"), 0.0, 0.0
    dx = obstacle_points_xy[:, 0] - state.x
    dy = obstacle_points_xy[:, 1] - state.y
    d2 = dx * dx + dy * dy
    idx = int(np.argmin(d2))
    nearest_dx = float(dx[idx])
    nearest_dy = float(dy[idx])
    dist_cm = float(np.sqrt(max(float(d2[idx]), 1e-8)))
    desired_yaw_deg = float(np.rad2deg(np.arctan2(nearest_dy, nearest_dx)))
    yaw_err_deg = _wrap_angle_deg(desired_yaw_deg - state.yaw_deg)
    ahead_score = float(np.cos(np.deg2rad(yaw_err_deg)))
    return dist_cm, yaw_err_deg, ahead_score


def _target_metrics(
    state: RoverState,
    target_x: float,
    target_y: float,
) -> tuple[float, float, float]:
    dx = float(target_x - state.x)
    dy = float(target_y - state.y)
    dist_cm = float(np.sqrt(max(dx * dx + dy * dy, 1e-8)))
    desired_yaw_deg = float(np.rad2deg(np.arctan2(dy, dx)))
    yaw_err_deg = _wrap_angle_deg(desired_yaw_deg - state.yaw_deg)
    ahead_score = float(np.cos(np.deg2rad(yaw_err_deg)))
    return dist_cm, yaw_err_deg, ahead_score


def _distance_weight(dist_cm: float) -> float:
    if not np.isfinite(dist_cm):
        return 0.0
    if dist_cm <= ATTRACTOR_EFFECT_MIN_DIST_CM:
        return 1.0
    if dist_cm >= ATTRACTOR_EFFECT_MAX_DIST_CM:
        return 0.0
    span = ATTRACTOR_EFFECT_MAX_DIST_CM - ATTRACTOR_EFFECT_MIN_DIST_CM
    return float((ATTRACTOR_EFFECT_MAX_DIST_CM - dist_cm) / max(span, 1e-6))


def _sample_throttle_target(rng: np.random.Generator) -> float:
    if rng.random() < THROTTLE_FORWARD_PROB:
        mag = THROTTLE_FORWARD_MIN + (1.0 - THROTTLE_FORWARD_MIN) * float(
            rng.beta(THROTTLE_FORWARD_BETA_A, THROTTLE_FORWARD_BETA_B)
        )
        return float(np.clip(mag, 0.0, 1.0))
    return -float(rng.uniform(THROTTLE_REVERSE_MIN, THROTTLE_REVERSE_MAX))


def _sample_steering_base_target(rng: np.random.Generator) -> float:
    return float(np.clip(rng.normal(0.0, STEERING_BASE_STD), -1.0, 1.0))


def _choose_control_target_segment(rng: np.random.Generator) -> tuple[int, float, float]:
    segment_steps = int(rng.integers(CONTROL_SEGMENT_STEPS_MIN, CONTROL_SEGMENT_STEPS_MAX + 1))
    throttle_base_target = _sample_throttle_target(rng)
    steering_base_target = _sample_steering_base_target(rng)
    return segment_steps, throttle_base_target, steering_base_target


def _spawn_state_near_obstacle(env, occupied_indices: np.ndarray, rng: np.random.Generator) -> RoverState:
    if occupied_indices.size == 0:
        state = RoverState(
            x=float(rng.uniform(float(env.x.min()), float(env.x.max()))),
            y=float(rng.uniform(float(env.y.min()), float(env.y.max()))),
            yaw_deg=float(rng.uniform(-180.0, 180.0)),
        )
        clamp_rover_state_xy(state, env)
        return state

    for _ in range(SPAWN_NEAR_OBSTACLE_RETRIES):
        ox, oy = _pick_random_obstacle_target(env, occupied_indices, rng)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        dist = _sample_uniform_biased(
            rng,
            SPAWN_NEAR_OBSTACLE_MIN_CM,
            SPAWN_NEAR_OBSTACLE_MAX_CM,
            SPAWN_NEAR_OBSTACLE_DIST_BIAS,
        )
        x = ox + dist * float(np.cos(angle))
        y = oy + dist * float(np.sin(angle))
        yaw_deg = _yaw_toward_target_with_noise(x, y, ox, oy, rng)
        state = RoverState(x=x, y=y, yaw_deg=yaw_deg)
        clamp_rover_state_xy(state, env)
        return state

    state = RoverState(yaw_deg=float(rng.uniform(-180.0, 180.0)))
    clamp_rover_state_xy(state, env)
    return state


def _pick_nearest_obstacle_target(state: RoverState, obstacle_points_xy: np.ndarray) -> tuple[float, float]:
    if obstacle_points_xy.size == 0:
        return state.x, state.y
    dx = obstacle_points_xy[:, 0] - state.x
    dy = obstacle_points_xy[:, 1] - state.y
    idx = int(np.argmin(dx * dx + dy * dy))
    return float(obstacle_points_xy[idx, 0]), float(obstacle_points_xy[idx, 1])


def _pick_farthest_obstacle_target(state: RoverState, obstacle_points_xy: np.ndarray) -> tuple[float, float]:
    if obstacle_points_xy.size == 0:
        return state.x, state.y
    dx = obstacle_points_xy[:, 0] - state.x
    dy = obstacle_points_xy[:, 1] - state.y
    idx = int(np.argmax(dx * dx + dy * dy))
    return float(obstacle_points_xy[idx, 0]), float(obstacle_points_xy[idx, 1])


def _sample_nav_mode(rng: np.random.Generator) -> str:
    return str(rng.choice(np.asarray(NAV_MODE_CHOICES), p=np.asarray(NAV_MODE_PROBS, dtype=float)))


def _pick_nav_target_for_mode(
    nav_mode: str,
    cycle_idx: int,
    state: RoverState,
    obstacle_points_xy: np.ndarray,
    env,
    occupied_indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if nav_mode == NAV_MODE_NEAREST_THEN_FARTHEST:
        if (cycle_idx % 2) == 0:
            return _pick_nearest_obstacle_target(state, obstacle_points_xy)
        return _pick_farthest_obstacle_target(state, obstacle_points_xy)
    if nav_mode == NAV_MODE_FARTHEST_THEN_NEAREST:
        if (cycle_idx % 2) == 0:
            return _pick_farthest_obstacle_target(state, obstacle_points_xy)
        return _pick_nearest_obstacle_target(state, obstacle_points_xy)
    if rng.random() < RANDOM_MODE_OBSTACLE_TARGET_PROB:
        return _pick_random_obstacle_target(env, occupied_indices, rng)
    return _pick_random_world_target(env, rng)


def _write_header(fh, sensor_count: int) -> None:
    header = [
        "timestep",
        "x_cm",
        "y_cm",
        "z_cm",
        "yaw_deg",
        "teleport_flag",
        "throttle_cmd",
        "steering_cmd",
        "cmd_move_cm",
        "cmd_turn_deg",
    ]
    header.extend([f"lidar_cm_{i}" for i in range(sensor_count)])
    header.extend([f"lidar_class_{i}" for i in range(sensor_count)])
    fh.write(",".join(header) + "\n")


def _write_timestep_row(
    fh,
    timestep: int,
    state: RoverState,
    z_cm: float,
    teleport_flag: int,
    throttle_cmd: float,
    steering_cmd: float,
    cmd_move_cm: float,
    cmd_turn_deg: float,
    scan,
) -> None:
    row = [
        str(timestep),
        f"{state.x:.3f}",
        f"{state.y:.3f}",
        f"{z_cm:.3f}",
        f"{state.yaw_deg:.3f}",
        str(int(teleport_flag)),
        f"{throttle_cmd:.5f}",
        f"{steering_cmd:.5f}",
        f"{cmd_move_cm:.3f}",
        f"{cmd_turn_deg:.3f}",
    ]
    row.extend(f"{v:.3f}" if v >= 0.0 else "-1" for v in scan.distances_cm)
    row.extend(str(int(v)) for v in scan.class_ids)
    fh.write(",".join(row) + "\n")


def generate_world_dataset(
    world_idx: int,
    world_seed: int,
    rng: np.random.Generator | None = None,
    show_viewer: bool | None = None,
) -> Path:
    if rng is None:
        rng = np.random.default_rng(_world_rng_seed(world_idx, world_seed))
    if show_viewer is None:
        show_viewer = SHOW_VIEWER

    world_style, world_gen_kwargs = _sample_world_generation_kwargs(rng)
    env = generate_environment(
        size=GRID_SIZE,
        world_size=WORLD_SIZE_CM,
        seed=world_seed,
        **world_gen_kwargs,
    )

    sampler = build_height_sampler(env)
    deformation_grid = env.z - env.z_base
    positive_deformation = deformation_grid[deformation_grid > DEFORMATION_EPS_CM]
    world_obstacle_p90_cm = float(np.quantile(positive_deformation, 0.90)) if positive_deformation.size > 0 else 0.0
    teleport_deformation_threshold_cm = max(
        TELEPORT_MIN_DEFORMATION_CM,
        TELEPORT_MIN_DEFORMATION_WORLD_P90_RATIO * world_obstacle_p90_cm,
    )
    lidar_distance_samples = make_lidar_distance_samples()
    rover_edge_local_xy = get_rover_edge_local_xy()
    rover_footprint_local_xy = get_rover_footprint_local()
    occupied_indices = np.argwhere(env.occupancy)
    obstacle_points_xy = np.column_stack((env.x[env.occupancy], env.y[env.occupancy]))

    state = _spawn_state_near_obstacle(env, occupied_indices, rng)
    teleported_since_last_log = False
    if TELEPORT_ON_OBSTACLE_CONTACT and _is_state_on_obstacle(
        state,
        deformation_grid,
        sampler,
        rover_footprint_local_xy,
        deformation_threshold_cm=teleport_deformation_threshold_cm,
    ):
        _teleport_state_to_safe_ground(
            state,
            env,
            deformation_grid,
            sampler,
            rover_footprint_local_xy,
            teleport_deformation_threshold_cm,
            obstacle_points_xy,
            rng,
        )
        teleported_since_last_log = True

    current_throttle = 0.0
    current_steering = 0.0
    segment_steps_left = 0
    throttle_base_target = 0.0
    steering_base_target = 0.0
    nav_mode = _sample_nav_mode(rng)
    nav_cycle_idx = 0
    nav_steps_on_target = 0
    nav_target_x, nav_target_y = _pick_nav_target_for_mode(
        nav_mode,
        nav_cycle_idx,
        state,
        obstacle_points_xy,
        env,
        occupied_indices,
        rng,
    )
    playback_controls: list[tuple[str, float | tuple[float, float, float]]] = [
        ("teleport", (float(state.x), float(state.y), float(state.yaw_deg)))
    ]

    out_path = _new_world_filename(world_idx, world_seed)
    with out_path.open("w", encoding="utf-8") as fh:
        _write_header(fh, sensor_count=len(rover_edge_local_xy))

        for t in range(TIMESTEPS_PER_WORLD):
            if TELEPORT_ON_OBSTACLE_CONTACT and _is_state_on_obstacle(
                state,
                deformation_grid,
                sampler,
                rover_footprint_local_xy,
                deformation_threshold_cm=teleport_deformation_threshold_cm,
            ):
                _teleport_state_to_safe_ground(
                    state,
                    env,
                    deformation_grid,
                    sampler,
                    rover_footprint_local_xy,
                    teleport_deformation_threshold_cm,
                    obstacle_points_xy,
                    rng,
                )
                playback_controls.append(("teleport", (float(state.x), float(state.y), float(state.yaw_deg))))
                teleported_since_last_log = True
                current_throttle = 0.0
                current_steering = 0.0
                segment_steps_left = 0
                nav_steps_on_target = NAV_TARGET_MAX_STEPS

            nav_dist_cm, nav_yaw_err_deg, nav_ahead_score = _target_metrics(state, nav_target_x, nav_target_y)
            should_retarget = (nav_dist_cm <= NAV_TARGET_REACHED_DIST_CM) or (
                nav_steps_on_target >= NAV_TARGET_MAX_STEPS
            )
            if should_retarget:
                nav_cycle_idx += 1
                nav_target_x, nav_target_y = _pick_nav_target_for_mode(
                    nav_mode,
                    nav_cycle_idx,
                    state,
                    obstacle_points_xy,
                    env,
                    occupied_indices,
                    rng,
                )
                nav_steps_on_target = 0
                nav_dist_cm, nav_yaw_err_deg, nav_ahead_score = _target_metrics(state, nav_target_x, nav_target_y)
            nav_steps_on_target += 1

            if segment_steps_left <= 0:
                segment_steps_left, throttle_base_target, steering_base_target = _choose_control_target_segment(rng)

            nearest_dist_cm, nearest_yaw_err_deg, nearest_ahead_score = _nearest_obstacle_metrics(
                state, obstacle_points_xy
            )
            w_dist = _distance_weight(nearest_dist_cm)
            w_nav = _distance_weight(nav_dist_cm)
            nearest_attractor_scale = (
                RANDOM_MODE_NEAREST_ATTRACTOR_SCALE if nav_mode == NAV_MODE_RANDOM_WAYPOINT else 1.0
            )
            steer_attractor = float(np.clip(nearest_yaw_err_deg / STEERING_ATTRACTION_YAW_SCALE_DEG, -1.0, 1.0))
            nav_steer_attractor = float(np.clip(nav_yaw_err_deg / STEERING_ATTRACTION_YAW_SCALE_DEG, -1.0, 1.0))
            steering_target = float(
                np.clip(
                    steering_base_target
                    + NAV_TARGET_STEER_STRENGTH * w_nav * nav_steer_attractor
                    + nearest_attractor_scale * STEERING_ATTRACTOR_STRENGTH * w_dist * steer_attractor,
                    -1.0,
                    1.0,
                )
            )
            throttle_target = float(
                np.clip(
                    throttle_base_target
                    + NAV_TARGET_THROTTLE_STRENGTH * w_nav * nav_ahead_score
                    + nearest_attractor_scale * THROTTLE_ATTRACTOR_STRENGTH * w_dist * nearest_ahead_score,
                    -1.0,
                    1.0,
                )
            )

            current_throttle = _move_toward(current_throttle, throttle_target, THROTTLE_SLEW_PER_STEP)
            current_steering = _move_toward(current_steering, steering_target, STEERING_SLEW_PER_STEP)
            current_throttle = float(np.clip(current_throttle, -1.0, 1.0))
            current_steering = float(np.clip(current_steering, -1.0, 1.0))

            cmd_turn_deg = current_steering * MAX_TURN_DEG_PER_STEP
            cmd_move_cm = current_throttle * MAX_MOVE_CM_PER_STEP

            if abs(cmd_turn_deg) > 1e-6:
                playback_controls.append(("turn", cmd_turn_deg))
                turn_rover(state, cmd_turn_deg)
            if abs(cmd_move_cm) > 1e-6:
                playback_controls.append(("move", cmd_move_cm))
                move_rover_forward(state, cmd_move_cm)

            pre_clamp_x, pre_clamp_y = state.x, state.y
            clamp_rover_state_xy(state, env)
            segment_steps_left -= 1

            # Boundary recovery: if clamp changed pose, retarget steering soon.
            if abs(state.x - pre_clamp_x) > 1e-6 or abs(state.y - pre_clamp_y) > 1e-6:
                segment_steps_left = min(segment_steps_left, 3)
                steering_base_target = float(np.clip(steering_base_target + rng.uniform(-0.8, 0.8), -1.0, 1.0))
                throttle_base_target = max(0.25, throttle_base_target)
                nav_steps_on_target = NAV_TARGET_MAX_STEPS

            if TELEPORT_ON_OBSTACLE_CONTACT and _is_state_on_obstacle(
                state,
                deformation_grid,
                sampler,
                rover_footprint_local_xy,
                deformation_threshold_cm=teleport_deformation_threshold_cm,
            ):
                _teleport_state_to_safe_ground(
                    state,
                    env,
                    deformation_grid,
                    sampler,
                    rover_footprint_local_xy,
                    teleport_deformation_threshold_cm,
                    obstacle_points_xy,
                    rng,
                )
                playback_controls.append(("teleport", (float(state.x), float(state.y), float(state.yaw_deg))))
                teleported_since_last_log = True
                current_throttle = 0.0
                current_steering = 0.0
                segment_steps_left = 0
                nav_steps_on_target = NAV_TARGET_MAX_STEPS

            should_log = ((t % LOG_EVERY_N_STEPS) == 0) or (t == TIMESTEPS_PER_WORLD - 1)
            if not should_log:
                continue

            pose = compute_rover_pose(env, state, sampler)
            if pose is None:
                continue
            scan = run_lidar_scan(
                env,
                pose,
                sampler,
                lidar_distance_samples,
                rover_edge_local_xy=rover_edge_local_xy,
            )
            _write_timestep_row(
                fh,
                t,
                state,
                float(pose.origin[2]),
                int(teleported_since_last_log),
                current_throttle,
                current_steering,
                cmd_move_cm,
                cmd_turn_deg,
                scan,
            )
            teleported_since_last_log = False

    if show_viewer:
        view_environment(
            env,
            playback_controls=playback_controls,
            playback_sleep_s=VIEWER_PLAYBACK_SLEEP_S,
            hold_after_playback=HOLD_VIEWER_AFTER_PLAYBACK,
            window_name=(
                f"Rover Terrain Viewer | world={world_idx} seed={world_seed} "
                f"style={world_style} nav={nav_mode}"
            ),
        )
    return out_path


def _generate_world_dataset_worker(world_idx: int, world_seed: int) -> str:
    # Worker entrypoint for multiprocessing; viewer is disabled for parallel runs.
    rng = np.random.default_rng(_world_rng_seed(world_idx, world_seed))
    out_path = generate_world_dataset(world_idx, world_seed, rng, show_viewer=False)
    return str(out_path)


def _stop_process_pool(executor: ProcessPoolExecutor) -> None:
    # Best-effort hard stop for Ctrl+C on Windows: cancel futures and terminate workers.
    executor.shutdown(wait=False, cancel_futures=True)
    proc_map = getattr(executor, "_processes", None)
    if not proc_map:
        return
    for proc in list(proc_map.values()):
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            continue
    for proc in list(proc_map.values()):
        try:
            proc.join(timeout=0.5)
        except Exception:
            continue
    for proc in list(proc_map.values()):
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:
            continue
    for proc in list(proc_map.values()):
        try:
            proc.join(timeout=0.25)
        except Exception:
            continue


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed_rng = np.random.default_rng(BASE_RANDOM_SEED if BASE_RANDOM_SEED is not None else None)
    world_seeds = [int(seed_rng.integers(0, 2_147_483_647)) for _ in range(WORLDS_TO_GENERATE)]

    if NUM_WORKERS <= 0:
        num_workers = max(1, (os.cpu_count() or 2) - 1)
    else:
        num_workers = int(NUM_WORKERS)

    if SHOW_VIEWER and num_workers > 1:
        print("SHOW_VIEWER=True -> forcing NUM_WORKERS=1 (viewer is single-process only).")
        num_workers = 1

    if num_workers <= 1:
        try:
            for i, world_seed in enumerate(world_seeds):
                world_rng = np.random.default_rng(_world_rng_seed(i, world_seed))
                out_path = generate_world_dataset(i, world_seed, world_rng, show_viewer=SHOW_VIEWER)
                print(f"Wrote {out_path}")
        except KeyboardInterrupt:
            print("\nCtrl+C received. Stopping generation.")
            raise SystemExit(130)
        return

    print(f"Generating {WORLDS_TO_GENERATE} worlds with {num_workers} workers (headless).")
    executor = ProcessPoolExecutor(max_workers=num_workers)
    future_by_world: dict = {}
    try:
        future_by_world = {
            executor.submit(_generate_world_dataset_worker, i, world_seed): i
            for i, world_seed in enumerate(world_seeds)
        }
        for future in as_completed(future_by_world):
            world_idx = future_by_world[future]
            out_path = future.result()
            print(f"Wrote world {world_idx:03d}: {out_path}")
    except KeyboardInterrupt:
        print("\nCtrl+C received. Cancelling pending worlds and terminating workers...")
        for future in future_by_world:
            future.cancel()
        _stop_process_pool(executor)
        raise SystemExit(130)
    except Exception:
        _stop_process_pool(executor)
        raise
    executor.shutdown(wait=True, cancel_futures=False)


if __name__ == "__main__":
    main()
