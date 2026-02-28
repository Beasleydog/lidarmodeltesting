from dataclasses import dataclass
import time

import numpy as np

# ===== Config =====
GRID_SIZE = 60
OBSTACLE_COUNT = 220
SEED = 7

# Rover LIDAR sensor coordinates from documentation (assumed centimeters).
# Format: (x, y, z) relative to rover origin.
LIDAR_SENSOR_COORDS_CM = np.array(
    [
        (250.0, 245.0, 50.0),
        (325.0, 75.0, 130.0),
        (325.0, 0.0, 130.0),
        (325.0, -75.0, 130.0),
        (250.0, -245.0, 50.0),
        (325.0, 75.0, 130.0),
        (325.0, -75.0, 130.0),
        (40.0, 235.0, 100.0),
        (40.0, -235.0, 100.0),
        (-215.0, 270.0, 70.0),
        (-320.0, 80.0, 10.0),
        (-320.0, -50.0, 10.0),
        (-215.0, -215.0, 70.0),
        (325.0, 75.0, 130.0),
        (325.0, -75.0, 130.0),
        (250.0, 245.0, 50.0),
        (250.0, -245.0, 50.0),
    ],
    dtype=float,
)
ROVER_X_MIN_CM = float(LIDAR_SENSOR_COORDS_CM[:, 0].min())
ROVER_X_MAX_CM = float(LIDAR_SENSOR_COORDS_CM[:, 0].max())
ROVER_Y_MIN_CM = float(LIDAR_SENSOR_COORDS_CM[:, 1].min())
ROVER_Y_MAX_CM = float(LIDAR_SENSOR_COORDS_CM[:, 1].max())
ROVER_LENGTH_CM = ROVER_X_MAX_CM - ROVER_X_MIN_CM
ROVER_WIDTH_CM = ROVER_Y_MAX_CM - ROVER_Y_MIN_CM
ROVER_SPAN_CM = max(ROVER_LENGTH_CM, ROVER_WIDTH_CM)
ROVER_FOOTPRINT_LOCAL = np.array(
    [
        [ROVER_X_MIN_CM, ROVER_Y_MIN_CM],
        [ROVER_X_MAX_CM, ROVER_Y_MIN_CM],
        [ROVER_X_MAX_CM, ROVER_Y_MAX_CM],
        [ROVER_X_MIN_CM, ROVER_Y_MAX_CM],
    ],
    dtype=float,
)
ROVER_FOOTPRINT_CENTER_LOCAL = ROVER_FOOTPRINT_LOCAL.mean(axis=0)
WORLD_SIZE_MULTIPLIER = 60.0
WORLD_SIZE_CM = ROVER_SPAN_CM * WORLD_SIZE_MULTIPLIER

# Terrain smoothness controls
NOISE_OCTAVES = 5
NOISE_BASE_CELLS = 2
NOISE_PERSISTENCE = 0.62
TERRAIN_HEIGHT_SCALE_CM = 520.0
TERRAIN_VERTICAL_EXAGGERATION = 3.4
LIDAR_RANGE_CM = 1000.0
OBSTACLE_RADIUS_MIN_FRAC = 0.015
OBSTACLE_RADIUS_MAX_FRAC = 0.022
OBSTACLE_HEIGHT_MIN_CM = 110.0
OBSTACLE_HEIGHT_MAX_CM = 320.0
OBSTACLE_TALL_MIN_CM = 300.0
OBSTACLE_TALL_MAX_CM = 520.0
OBSTACLE_TALL_PROB = 0.30
OBSTACLE_MAX_OVERLAP_RATIO = 0.32
OBSTACLE_MAX_ATTEMPTS_FACTOR = 35
OBSTACLE_EDGE_START_FRAC = 0.78
OBSTACLE_SIDE_EXPONENT = 0.22
# Ranges for dataset world randomization (used by data.py).
WORLDGEN_OBSTACLE_COUNT_RANGE = (70, 140)
WORLDGEN_TERRAIN_HEIGHT_SCALE_RANGE_CM = (300.0, 900.0)
WORLDGEN_NOISE_OCTAVES_RANGE = (3, 6)
WORLDGEN_NOISE_BASE_CELLS_RANGE = (2, 6)
WORLDGEN_NOISE_PERSISTENCE_RANGE = (0.35, 0.72)
WORLDGEN_OBSTACLE_RADIUS_MIN_FRAC_RANGE = (0.009, 0.020)
WORLDGEN_OBSTACLE_RADIUS_MAX_FRAC_RANGE = (0.018, 0.040)
WORLDGEN_OBSTACLE_HEIGHT_RANGE_CM = (70.0, 420.0)
WORLDGEN_OBSTACLE_TALL_HEIGHT_RANGE_CM = (220.0, 700.0)
WORLDGEN_OBSTACLE_TALL_PROB_RANGE = (0.12, 0.48)
WORLDGEN_OBSTACLE_MAX_OVERLAP_RATIO_RANGE = (0.12, 0.48)
WORLDGEN_OBSTACLE_EDGE_START_FRAC_RANGE = (0.58, 0.88)
WORLDGEN_OBSTACLE_SIDE_EXPONENT_RANGE = (0.14, 0.52)
OCCUPANCY_OVERLAY_OFFSET_CM = 10.0
OCCUPANCY_FREE_COLOR = "#2f8f5b"
OCCUPANCY_OCCUPIED_COLOR = "#ff2b2b"
OCCUPANCY_FREE_OPACITY = 0.08
OCCUPANCY_OCCUPIED_OPACITY = 0.45
OCCUPANCY_MAP_CLEARANCE_CM = 900.0
DEFORMATION_EPS_CM = 1e-6
OCCUPIED_MARKER_SIZE = 5.0
ROVER_MOVE_STEP_CM = 140.0
ROVER_TURN_STEP_DEG = 9.0
LIDAR_SAMPLE_STEP_CM = 25.0
LIDAR_OBSTACLE_COLOR = "#ff2b2b"
LIDAR_GROUND_COLOR = "#2b7bff"
LIDAR_CLEAR_COLOR = "#32d14f"
LIDAR_CLASS_GROUND = 0
LIDAR_CLASS_OBSTACLE = 1
LIDAR_CLASS_NONE = 2
TERRAIN_CMAP = ["#ffffff", "#f3f3f3", "#d8d8d8", "#b88e62", "#915a2b", "#5a2a10"]
USE_MODEL_LIDAR_CLASSIFIER = False
MODEL_LIDAR_CHECKPOINT_PATH = "runs/gru_lidar_classifier.pt"
MODEL_LIDAR_MAX_HISTORY = 64

# (yaw_deg, pitch_deg) in rover frame where +X is forward, +Y is left, +Z is up.
# Positive yaw rotates CCW from +X toward +Y. Negative pitch points downward.
LIDAR_YAW_PITCH_DEG = np.array(
    [
        (30.0, 0.0),     # 0
        (20.0, -20.0),   # 1
        (0.0, 0.0),      # 2
        (-20.0, -20.0),  # 3
        (-30.0, 0.0),    # 4
        (0.0, -25.0),    # 5
        (0.0, -25.0),    # 6
        (90.0, -20.0),   # 7
        (-90.0, -20.0),  # 8
        (140.0, 0.0),    # 9
        (180.0, 0.0),    # 10
        (180.0, 0.0),    # 11
        (-140.0, 0.0),   # 12
        (20.0, -10.0),   # 13
        (-20.0, -10.0),  # 14
        (15.0, 0.0),     # 15
        (-15.0, 0.0),    # 16
    ],
    dtype=float,
)

@dataclass
class Environment:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    z_base: np.ndarray
    occupancy: np.ndarray


@dataclass
class RoverState:
    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class RoverPose:
    origin: np.ndarray
    basis: np.ndarray
    center: np.ndarray


@dataclass(frozen=True)
class LidarScan:
    distances_cm: np.ndarray
    class_ids: np.ndarray
    hit_types: np.ndarray
    start_points: np.ndarray
    end_points: np.ndarray


@dataclass(frozen=True)
class HeightSampler:
    x0: float
    x1: float
    y0: float
    y1: float
    dx: float
    dy: float
    max_ix: int
    max_iy: int


def resize_bilinear(grid: np.ndarray, out_size: int) -> np.ndarray:
    h, w = grid.shape
    y = np.linspace(0, h - 1, out_size)
    x = np.linspace(0, w - 1, out_size)

    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = x - x0
    wy = y - y0

    top = (1 - wx)[None, :] * grid[y0[:, None], x0[None, :]] + wx[None, :] * grid[
        y0[:, None], x1[None, :]
    ]
    bottom = (1 - wx)[None, :] * grid[y1[:, None], x0[None, :]] + wx[None, :] * grid[
        y1[:, None], x1[None, :]
    ]
    return (1 - wy)[:, None] * top + wy[:, None] * bottom


def generate_smooth_heightmap(
    size: int,
    rng: np.random.Generator,
    octaves: int = NOISE_OCTAVES,
    base_cells: int = NOISE_BASE_CELLS,
    persistence: float = NOISE_PERSISTENCE,
    height_scale: float = TERRAIN_HEIGHT_SCALE_CM,
) -> np.ndarray:
    z = np.zeros((size, size), dtype=float)
    amplitude = 1.0
    amplitude_sum = 0.0
    cells = base_cells

    for _ in range(octaves):
        coarse = rng.random((cells + 1, cells + 1))
        z += amplitude * resize_bilinear(coarse, size)
        amplitude_sum += amplitude
        amplitude *= persistence
        cells *= 2

    z /= max(amplitude_sum, 1e-8)
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    return (z - 0.5) * 2.0 * height_scale


def random_polygon(
    rng: np.random.Generator,
    center_x: float,
    center_y: float,
    radius: float,
    vertices: int,
) -> np.ndarray:
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=vertices))
    radii = radius * rng.uniform(0.6, 1.0, size=vertices)
    px = center_x + radii * np.cos(angles)
    py = center_y + radii * np.sin(angles)
    return np.column_stack((px, py))


def rasterize_polygon(poly: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    inside = np.zeros_like(x_grid, dtype=bool)
    x0, y0 = poly[-1]

    for x1, y1 in poly:
        crosses = (y1 > y_grid) != (y0 > y_grid)
        x_cross = (x0 - x1) * (y_grid - y1) / (y0 - y1 + 1e-12) + x1
        inside ^= crosses & (x_grid < x_cross)
        x0, y0 = x1, y1

    return inside


def generate_environment(
    size: int = GRID_SIZE,
    world_size: float = WORLD_SIZE_CM,
    obstacle_count: int = OBSTACLE_COUNT,
    seed: int = SEED,
    noise_octaves: int = NOISE_OCTAVES,
    noise_base_cells: int = NOISE_BASE_CELLS,
    noise_persistence: float = NOISE_PERSISTENCE,
    terrain_height_scale_cm: float = TERRAIN_HEIGHT_SCALE_CM,
    obstacle_radius_min_frac: float = OBSTACLE_RADIUS_MIN_FRAC,
    obstacle_radius_max_frac: float = OBSTACLE_RADIUS_MAX_FRAC,
    obstacle_height_min_cm: float = OBSTACLE_HEIGHT_MIN_CM,
    obstacle_height_max_cm: float = OBSTACLE_HEIGHT_MAX_CM,
    obstacle_tall_min_cm: float = OBSTACLE_TALL_MIN_CM,
    obstacle_tall_max_cm: float = OBSTACLE_TALL_MAX_CM,
    obstacle_tall_prob: float = OBSTACLE_TALL_PROB,
    obstacle_max_overlap_ratio: float = OBSTACLE_MAX_OVERLAP_RATIO,
    obstacle_max_attempts_factor: int = OBSTACLE_MAX_ATTEMPTS_FACTOR,
    obstacle_edge_start_frac: float = OBSTACLE_EDGE_START_FRAC,
    obstacle_side_exponent: float = OBSTACLE_SIDE_EXPONENT,
) -> Environment:
    rng = np.random.default_rng(seed)
    obstacle_count = int(max(obstacle_count, 0))
    noise_octaves = int(max(noise_octaves, 1))
    noise_base_cells = int(max(noise_base_cells, 2))
    noise_persistence = float(np.clip(noise_persistence, 0.05, 0.95))
    terrain_height_scale_cm = float(max(terrain_height_scale_cm, 1.0))
    radius_min_frac = float(min(obstacle_radius_min_frac, obstacle_radius_max_frac))
    radius_max_frac = float(max(obstacle_radius_min_frac, obstacle_radius_max_frac))
    obstacle_height_min_cm = float(min(obstacle_height_min_cm, obstacle_height_max_cm))
    obstacle_height_max_cm = float(max(obstacle_height_min_cm, obstacle_height_max_cm))
    obstacle_tall_min_cm = float(min(obstacle_tall_min_cm, obstacle_tall_max_cm))
    obstacle_tall_max_cm = float(max(obstacle_tall_min_cm, obstacle_tall_max_cm))
    obstacle_tall_prob = float(np.clip(obstacle_tall_prob, 0.0, 1.0))
    obstacle_max_overlap_ratio = float(np.clip(obstacle_max_overlap_ratio, 0.0, 1.0))
    obstacle_max_attempts_factor = int(max(obstacle_max_attempts_factor, 1))
    obstacle_edge_start_frac = float(np.clip(obstacle_edge_start_frac, 0.0, 0.98))
    obstacle_side_exponent = float(max(obstacle_side_exponent, 0.05))

    axis = np.linspace(-world_size / 2.0, world_size / 2.0, size)
    x, y = np.meshgrid(axis, axis)

    z_base = generate_smooth_heightmap(
        size=size,
        rng=rng,
        octaves=noise_octaves,
        base_cells=noise_base_cells,
        persistence=noise_persistence,
        height_scale=terrain_height_scale_cm,
    )
    z = z_base.copy()
    occupancy_core = np.zeros((size, size), dtype=bool)

    placed = 0
    attempts = 0
    max_attempts = obstacle_count * obstacle_max_attempts_factor

    while placed < obstacle_count and attempts < max_attempts:
        attempts += 1
        center_x = rng.uniform(-0.45 * world_size, 0.45 * world_size)
        center_y = rng.uniform(-0.45 * world_size, 0.45 * world_size)
        radius = rng.uniform(radius_min_frac * world_size, radius_max_frac * world_size)
        vertices = int(rng.integers(5, 9))
        if rng.random() < obstacle_tall_prob:
            obstacle_height = rng.uniform(obstacle_tall_min_cm, obstacle_tall_max_cm)
        else:
            obstacle_height = rng.uniform(obstacle_height_min_cm, obstacle_height_max_cm)

        poly = random_polygon(rng, center_x, center_y, radius, vertices)
        mask = rasterize_polygon(poly, x, y)
        mask_area = int(mask.sum())
        if mask_area < 4:
            continue

        overlap_ratio = float((mask & occupancy_core).sum()) / float(mask_area)
        if overlap_ratio > obstacle_max_overlap_ratio:
            continue

        # Steeper edge-band profile so rocks rise abruptly near the perimeter.
        dx = (x[mask] - center_x) / (radius + 1e-8)
        dy = (y[mask] - center_y) / (radius + 1e-8)
        d = np.sqrt(dx * dx + dy * dy)
        edge_band = np.clip(
            (1.0 - d) / (1.0 - obstacle_edge_start_frac + 1e-8),
            0.0,
            1.0,
        )
        profile = edge_band ** obstacle_side_exponent
        roughness = rng.uniform(0.80, 1.22, size=profile.shape)
        z[mask] += obstacle_height * profile * roughness
        occupancy_core |= mask
        placed += 1

    deformation = z - z_base
    occupancy = deformation > DEFORMATION_EPS_CM
    return Environment(x=x, y=y, z=z, z_base=z_base, occupancy=occupancy)


def yaw_pitch_to_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cp = np.cos(pitch)
    return np.array(
        [
            cp * np.cos(yaw),
            cp * np.sin(yaw),
            np.sin(pitch),
        ],
        dtype=float,
    )


def snap_point_to_rover_edge(x: float, y: float) -> tuple[float, float]:
    d_left = abs(x - ROVER_X_MIN_CM)
    d_right = abs(x - ROVER_X_MAX_CM)
    d_bottom = abs(y - ROVER_Y_MIN_CM)
    d_top = abs(y - ROVER_Y_MAX_CM)
    edge = int(np.argmin([d_left, d_right, d_bottom, d_top]))

    if edge == 0:
        return ROVER_X_MIN_CM, y
    if edge == 1:
        return ROVER_X_MAX_CM, y
    if edge == 2:
        return x, ROVER_Y_MIN_CM
    return x, ROVER_Y_MAX_CM


ROVER_EDGE_LOCAL_XY = np.array(
    [snap_point_to_rover_edge(float(sx), float(sy)) for sx, sy, _ in LIDAR_SENSOR_COORDS_CM],
    dtype=float,
)


def get_rover_footprint_local() -> np.ndarray:
    return ROVER_FOOTPRINT_LOCAL.copy()


def get_rover_footprint_center_local() -> np.ndarray:
    return ROVER_FOOTPRINT_CENTER_LOCAL.copy()


def get_rover_edge_local_xy() -> np.ndarray:
    return ROVER_EDGE_LOCAL_XY.copy()


def rotate_xy(x: float, y: float, yaw_deg: float) -> tuple[float, float]:
    yaw = np.deg2rad(yaw_deg)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return c * x - s * y, s * x + c * y


def make_lidar_distance_samples(
    max_range_cm: float = LIDAR_RANGE_CM,
    step_cm: float = LIDAR_SAMPLE_STEP_CM,
) -> np.ndarray:
    return np.arange(
        0.0,
        max_range_cm + step_cm,
        step_cm,
        dtype=float,
    )


def lidar_hit_type_to_class_id(hit_type: str) -> int:
    if hit_type == "ground":
        return LIDAR_CLASS_GROUND
    if hit_type == "obstacle":
        return LIDAR_CLASS_OBSTACLE
    if hit_type == "none":
        return LIDAR_CLASS_NONE
    raise ValueError(f"Unknown lidar hit type: {hit_type}")


def lidar_class_id_to_color_hex(class_id: int) -> str:
    if class_id == LIDAR_CLASS_OBSTACLE:
        return LIDAR_OBSTACLE_COLOR
    if class_id == LIDAR_CLASS_GROUND:
        return LIDAR_GROUND_COLOR
    return LIDAR_CLEAR_COLOR


def build_height_sampler(env: Environment) -> HeightSampler:
    x_axis = env.x[0]
    y_axis = env.y[:, 0]
    return HeightSampler(
        x0=float(x_axis[0]),
        x1=float(x_axis[-1]),
        y0=float(y_axis[0]),
        y1=float(y_axis[-1]),
        dx=float(x_axis[1] - x_axis[0]),
        dy=float(y_axis[1] - y_axis[0]),
        max_ix=env.z.shape[1] - 2,
        max_iy=env.z.shape[0] - 2,
    )


def sample_height_bilinear_many(
    env: Environment,
    xs: np.ndarray,
    ys: np.ndarray,
    sampler: HeightSampler,
) -> np.ndarray:
    heights = np.full(xs.shape, np.nan, dtype=float)
    inside = (
        (xs >= sampler.x0)
        & (xs <= sampler.x1)
        & (ys >= sampler.y0)
        & (ys <= sampler.y1)
    )
    if not np.any(inside):
        return heights

    fx = (xs[inside] - sampler.x0) / sampler.dx
    fy = (ys[inside] - sampler.y0) / sampler.dy
    ix = np.clip(np.floor(fx).astype(np.int32), 0, sampler.max_ix)
    iy = np.clip(np.floor(fy).astype(np.int32), 0, sampler.max_iy)
    tx = fx - ix
    ty = fy - iy

    z00 = env.z[iy, ix]
    z10 = env.z[iy, ix + 1]
    z01 = env.z[iy + 1, ix]
    z11 = env.z[iy + 1, ix + 1]
    z0v = z00 * (1.0 - tx) + z10 * tx
    z1v = z01 * (1.0 - tx) + z11 * tx
    heights[inside] = z0v * (1.0 - ty) + z1v * ty
    return heights


def sample_height_bilinear(
    env: Environment,
    x: float,
    y: float,
    sampler: HeightSampler,
) -> float | None:
    if x < sampler.x0 or x > sampler.x1 or y < sampler.y0 or y > sampler.y1:
        return None

    fx = (x - sampler.x0) / sampler.dx
    fy = (y - sampler.y0) / sampler.dy
    ix = int(np.clip(np.floor(fx), 0, sampler.max_ix))
    iy = int(np.clip(np.floor(fy), 0, sampler.max_iy))
    tx = fx - ix
    ty = fy - iy

    z00 = env.z[iy, ix]
    z10 = env.z[iy, ix + 1]
    z01 = env.z[iy + 1, ix]
    z11 = env.z[iy + 1, ix + 1]
    z0v = z00 * (1.0 - tx) + z10 * tx
    z1v = z01 * (1.0 - tx) + z11 * tx
    return float(z0v * (1.0 - ty) + z1v * ty)


def estimate_terrain_normal(
    env: Environment,
    x: float,
    y: float,
    sampler: HeightSampler,
) -> np.ndarray:
    if x < sampler.x0 or x > sampler.x1 or y < sampler.y0 or y > sampler.y1:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    fx = (x - sampler.x0) / sampler.dx
    fy = (y - sampler.y0) / sampler.dy
    ix = int(np.clip(np.floor(fx), 0, sampler.max_ix))
    iy = int(np.clip(np.floor(fy), 0, sampler.max_iy))
    tx = fx - ix
    ty = fy - iy

    z00 = env.z[iy, ix]
    z10 = env.z[iy, ix + 1]
    z01 = env.z[iy + 1, ix]
    z11 = env.z[iy + 1, ix + 1]
    dz_dtx = (z10 - z00) * (1.0 - ty) + (z11 - z01) * ty
    dz_dty = (z01 - z00) * (1.0 - tx) + (z11 - z10) * tx
    dzdx = dz_dtx / max(sampler.dx, 1e-6)
    dzdy = dz_dty / max(sampler.dy, 1e-6)
    normal = np.array([-dzdx, -dzdy, 1.0], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / norm


def build_rover_basis(yaw_deg: float, terrain_up: np.ndarray) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    forward_hint = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
    forward = forward_hint - float(np.dot(forward_hint, terrain_up)) * terrain_up
    f_norm = float(np.linalg.norm(forward))
    if f_norm < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        forward = fallback - float(np.dot(fallback, terrain_up)) * terrain_up
        f_norm = float(np.linalg.norm(forward))
        if f_norm < 1e-8:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
            forward = fallback - float(np.dot(fallback, terrain_up)) * terrain_up
            f_norm = float(np.linalg.norm(forward))
    forward /= max(f_norm, 1e-8)
    left = np.cross(terrain_up, forward)
    left_norm = float(np.linalg.norm(left))
    if left_norm < 1e-8:
        left = np.array([0.0, 1.0, 0.0], dtype=float)
        left_norm = float(np.linalg.norm(left))
    left /= max(left_norm, 1e-8)
    return np.column_stack((forward, left, terrain_up))


def clamp_rover_state_xy(state: RoverState, env: Environment, margin_scale: float = 0.7) -> None:
    margin = ROVER_SPAN_CM * margin_scale
    state.x = float(np.clip(state.x, float(env.x.min()) + margin, float(env.x.max()) - margin))
    state.y = float(np.clip(state.y, float(env.y.min()) + margin, float(env.y.max()) - margin))


def move_rover_forward(state: RoverState, distance_cm: float) -> None:
    dx, dy = rotate_xy(distance_cm, 0.0, state.yaw_deg)
    state.x += dx
    state.y += dy


def turn_rover(state: RoverState, delta_yaw_deg: float) -> None:
    state.yaw_deg += delta_yaw_deg


def compute_rover_pose(
    env: Environment,
    state: RoverState,
    sampler: HeightSampler,
    footprint_center_local: np.ndarray | None = None,
) -> RoverPose | None:
    if footprint_center_local is None:
        footprint_center_local = ROVER_FOOTPRINT_CENTER_LOCAL
    center_dx, center_dy = rotate_xy(
        float(footprint_center_local[0]),
        float(footprint_center_local[1]),
        state.yaw_deg,
    )
    center_x = state.x + center_dx
    center_y = state.y + center_dy
    center_z = sample_height_bilinear(env, center_x, center_y, sampler)
    if center_z is None:
        return None
    terrain_up = estimate_terrain_normal(env, center_x, center_y, sampler)
    rover_basis = build_rover_basis(state.yaw_deg, terrain_up)
    center_world = np.array([center_x, center_y, center_z], dtype=float)
    rover_origin = center_world - rover_basis @ np.array(
        [float(footprint_center_local[0]), float(footprint_center_local[1]), 0.0],
        dtype=float,
    )
    return RoverPose(origin=rover_origin, basis=rover_basis, center=center_world)


def cast_lidar_ray(
    env: Environment,
    start: np.ndarray,
    direction: np.ndarray,
    distance_samples_cm: np.ndarray,
    sampler: HeightSampler,
) -> tuple[str, np.ndarray]:
    def is_deformed_at(x: float, y: float) -> bool:
        if x < sampler.x0 or x > sampler.x1 or y < sampler.y0 or y > sampler.y1:
            return False
        fx = (x - sampler.x0) / sampler.dx
        fy = (y - sampler.y0) / sampler.dy
        ix = int(np.clip(np.floor(fx), 0, sampler.max_ix))
        iy = int(np.clip(np.floor(fy), 0, sampler.max_iy))
        tx = fx - ix
        ty = fy - iy

        d00 = env.z[iy, ix] - env.z_base[iy, ix]
        d10 = env.z[iy, ix + 1] - env.z_base[iy, ix + 1]
        d01 = env.z[iy + 1, ix] - env.z_base[iy + 1, ix]
        d11 = env.z[iy + 1, ix + 1] - env.z_base[iy + 1, ix + 1]
        d0v = d00 * (1.0 - tx) + d10 * tx
        d1v = d01 * (1.0 - tx) + d11 * tx
        deformation_cm = d0v * (1.0 - ty) + d1v * ty
        return bool(deformation_cm > DEFORMATION_EPS_CM)

    ray_points = start[None, :] + distance_samples_cm[:, None] * direction[None, :]
    heights = sample_height_bilinear_many(env, ray_points[:, 0], ray_points[:, 1], sampler)
    in_bounds = ~np.isnan(heights)

    first_oob_idx = int(np.argmax(~in_bounds)) if np.any(~in_bounds) else -1
    hit_mask = in_bounds & (heights >= ray_points[:, 2])
    if np.any(hit_mask):
        hit_idx = int(np.argmax(hit_mask))
        if first_oob_idx == -1 or hit_idx < first_oob_idx:
            hit_point = ray_points[hit_idx]
            hit_type = "obstacle" if is_deformed_at(float(hit_point[0]), float(hit_point[1])) else "ground"
            return hit_type, hit_point

    if first_oob_idx != -1:
        return "none", ray_points[first_oob_idx]
    return "none", ray_points[-1]


def run_lidar_scan(
    env: Environment,
    pose: RoverPose,
    sampler: HeightSampler,
    lidar_distance_samples: np.ndarray,
    rover_edge_local_xy: np.ndarray | None = None,
) -> LidarScan:
    if rover_edge_local_xy is None:
        rover_edge_local_xy = ROVER_EDGE_LOCAL_XY

    sensor_count = len(LIDAR_SENSOR_COORDS_CM)
    distances_cm = np.full((sensor_count,), -1.0, dtype=float)
    class_ids = np.full((sensor_count,), LIDAR_CLASS_NONE, dtype=np.int32)
    hit_types = np.full((sensor_count,), "none", dtype="<U8")
    start_points = np.zeros((sensor_count, 3), dtype=float)
    end_points = np.zeros((sensor_count, 3), dtype=float)

    for i, ((_, _, sz), (sensor_yaw, sensor_pitch)) in enumerate(
        zip(LIDAR_SENSOR_COORDS_CM, LIDAR_YAW_PITCH_DEG)
    ):
        lx, ly = rover_edge_local_xy[i]
        local_start = np.array([float(lx), float(ly), float(sz)], dtype=float)
        start = pose.origin + pose.basis @ local_start
        local_direction = yaw_pitch_to_direction(float(sensor_yaw), float(sensor_pitch))
        direction = pose.basis @ local_direction
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        hit_type, end = cast_lidar_ray(
            env,
            start,
            direction,
            lidar_distance_samples,
            sampler,
        )
        hit_types[i] = hit_type
        class_ids[i] = lidar_hit_type_to_class_id(hit_type)
        if hit_type != "none":
            distances_cm[i] = float(np.linalg.norm(end - start))
        start_points[i] = start
        end_points[i] = end

    return LidarScan(
        distances_cm=distances_cm,
        class_ids=class_ids,
        hit_types=hit_types,
        start_points=start_points,
        end_points=end_points,
    )


def view_environment(
    env: Environment,
    playback_controls: list[tuple[str, float | tuple[float, float, float]]] | None = None,
    playback_sleep_s: float = 0.04,
    hold_after_playback: bool = True,
    window_name: str = "Rover Terrain Viewer",
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("Open3D is required. Install with: pip install open3d") from exc

    def hex_to_rgb01(hex_color: str) -> np.ndarray:
        value = hex_color.lstrip("#")
        return np.array(
            [int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)],
            dtype=float,
        ) / 255.0

    def interpolate_cmap(values: np.ndarray, vmin: float, vmax: float, colors: np.ndarray) -> np.ndarray:
        t = np.clip((values - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
        seg = t * (len(colors) - 1)
        i0 = np.floor(seg).astype(np.int32)
        i1 = np.clip(i0 + 1, 0, len(colors) - 1)
        w = seg - i0
        return colors[i0] * (1.0 - w)[:, None] + colors[i1] * w[:, None]

    terrain_samples = env.z[~env.occupancy]
    if terrain_samples.size < 16:
        terrain_samples = env.z.ravel()
    z_lo = float(np.percentile(terrain_samples, 2.0))
    z_hi = float(np.percentile(terrain_samples, 98.0))

    rows, cols = env.z.shape
    z_vis = env.z * TERRAIN_VERTICAL_EXAGGERATION
    terrain_vertices = np.column_stack((env.x.ravel(), env.y.ravel(), z_vis.ravel()))
    cell_ids = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    v00 = cell_ids
    v10 = cell_ids + 1
    v01 = cell_ids + cols
    v11 = v01 + 1
    terrain_triangles = np.vstack(
        (
            np.column_stack((v00, v10, v11)),
            np.column_stack((v00, v11, v01)),
        )
    ).astype(np.int32)

    terrain_cmap = np.array([hex_to_rgb01(c) for c in TERRAIN_CMAP], dtype=float)
    terrain_colors = interpolate_cmap(env.z.ravel(), z_lo, z_hi, terrain_cmap)
    occ_flat = env.occupancy.ravel()
    occ_color = hex_to_rgb01(OCCUPANCY_OCCUPIED_COLOR)
    if np.any(occ_flat):
        terrain_colors[occ_flat] = 0.45 * terrain_colors[occ_flat] + 0.55 * occ_color[None, :]

    terrain_mesh = o3d.geometry.TriangleMesh()
    terrain_mesh.vertices = o3d.utility.Vector3dVector(terrain_vertices)
    terrain_mesh.triangles = o3d.utility.Vector3iVector(terrain_triangles)
    terrain_mesh.vertex_colors = o3d.utility.Vector3dVector(terrain_colors)
    terrain_mesh.compute_vertex_normals()

    occupied_points = np.column_stack(
        (
            env.x[env.occupancy],
            env.y[env.occupancy],
            (env.z[env.occupancy] + OCCUPANCY_OVERLAY_OFFSET_CM) * TERRAIN_VERTICAL_EXAGGERATION,
        )
    )
    occupied_cloud = o3d.geometry.PointCloud()
    if occupied_points.size > 0:
        occupied_cloud.points = o3d.utility.Vector3dVector(occupied_points)
        occupied_cloud.colors = o3d.utility.Vector3dVector(
            np.tile(occ_color[None, :], (occupied_points.shape[0], 1))
        )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name=window_name, width=1400, height=900):
        raise SystemExit("Failed to create Open3D window.")

    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.06, 0.08, 0.10], dtype=float)
    render_opt.mesh_show_back_face = True
    render_opt.point_size = float(OCCUPIED_MARKER_SIZE)

    vis.add_geometry(terrain_mesh, reset_bounding_box=True)
    if occupied_points.size > 0:
        vis.add_geometry(occupied_cloud, reset_bounding_box=False)

    rover_edge_local_xy = get_rover_edge_local_xy()
    footprint_local = get_rover_footprint_local()
    footprint_center_local = get_rover_footprint_center_local()
    rover_state = RoverState()
    height_sampler = build_height_sampler(env)
    lidar_distance_samples = make_lidar_distance_samples()
    sensor_count = len(LIDAR_SENSOR_COORDS_CM)
    model_inferencer = None
    model_inferencer_failed = False
    model_feature_history: list[np.ndarray] = []
    if USE_MODEL_LIDAR_CLASSIFIER:
        try:
            from train import load_gru_lidar_inferencer

            model_inferencer = load_gru_lidar_inferencer(
                MODEL_LIDAR_CHECKPOINT_PATH,
                max_history=MODEL_LIDAR_MAX_HISTORY,
            )
            print(f"Model lidar classifier enabled: {MODEL_LIDAR_CHECKPOINT_PATH}")
        except Exception as exc:
            model_inferencer_failed = True
            print(f"Model lidar classifier disabled (load failed): {exc}")

    rover_poly_points = np.zeros((4, 3), dtype=float)
    rover_outline_dyn = o3d.geometry.LineSet()
    rover_outline_dyn.points = o3d.utility.Vector3dVector(rover_poly_points)
    rover_outline_dyn.lines = o3d.utility.Vector2iVector(
        np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    )
    rover_outline_dyn.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.91, 0.94, 1.00]], dtype=float), (4, 1))
    )
    vis.add_geometry(rover_outline_dyn, reset_bounding_box=False)

    ray_points = np.zeros((sensor_count * 2, 3), dtype=float)
    ray_lines = np.column_stack(
        (np.arange(sensor_count, dtype=np.int32) * 2, np.arange(sensor_count, dtype=np.int32) * 2 + 1)
    )
    ray_colors = np.tile(hex_to_rgb01(LIDAR_CLEAR_COLOR)[None, :], (sensor_count, 1))
    ray_mesh = o3d.geometry.LineSet()
    ray_mesh.points = o3d.utility.Vector3dVector(ray_points)
    ray_mesh.lines = o3d.utility.Vector2iVector(ray_lines)
    ray_mesh.colors = o3d.utility.Vector3dVector(ray_colors)
    vis.add_geometry(ray_mesh, reset_bounding_box=False)

    def draw_dynamic() -> None:
        nonlocal model_inferencer_failed
        pose = compute_rover_pose(env, rover_state, height_sampler, footprint_center_local=footprint_center_local)
        if pose is None:
            return

        for i, (lx, ly) in enumerate(footprint_local):
            local_pt = np.array([float(lx), float(ly), 2.0], dtype=float)
            rover_poly_points[i] = pose.origin + pose.basis @ local_pt
        rover_poly_points[:, 2] *= TERRAIN_VERTICAL_EXAGGERATION
        rover_outline_dyn.points = o3d.utility.Vector3dVector(rover_poly_points)

        scan = run_lidar_scan(
            env,
            pose,
            height_sampler,
            lidar_distance_samples,
            rover_edge_local_xy=rover_edge_local_xy,
        )
        ray_points[0::2] = scan.start_points
        ray_points[1::2] = scan.end_points

        class_ids_for_color = scan.class_ids
        if model_inferencer is not None and not model_inferencer_failed and len(model_feature_history) > 0:
            try:
                model_pred = model_inferencer.predict_next_from_feature_history(
                    np.asarray(model_feature_history, dtype=np.float32)
                )
                if model_pred.shape[0] == sensor_count:
                    class_ids_for_color = model_pred.astype(np.int32)
            except Exception as exc:
                model_inferencer_failed = True
                print(f"Model lidar classifier disabled (runtime error): {exc}")

        for i, class_id in enumerate(class_ids_for_color):
            ray_colors[i] = hex_to_rgb01(lidar_class_id_to_color_hex(int(class_id)))

        if model_inferencer is not None and not model_inferencer_failed:
            pose_xyzyaw = np.array(
                [rover_state.x, rover_state.y, float(pose.origin[2]), rover_state.yaw_deg],
                dtype=np.float32,
            )
            feature_t = model_inferencer.featurize_timestep(pose_xyzyaw, scan.distances_cm.astype(np.float32))
            model_feature_history.append(feature_t)
            if MODEL_LIDAR_MAX_HISTORY > 0 and len(model_feature_history) > MODEL_LIDAR_MAX_HISTORY:
                del model_feature_history[:-MODEL_LIDAR_MAX_HISTORY]

        ray_points[:, 2] *= TERRAIN_VERTICAL_EXAGGERATION
        ray_mesh.points = o3d.utility.Vector3dVector(ray_points)
        ray_mesh.colors = o3d.utility.Vector3dVector(ray_colors)

        vis.update_geometry(rover_outline_dyn)
        vis.update_geometry(ray_mesh)
        vis.update_renderer()
        print(
            f"Pose x={rover_state.x:.0f} y={rover_state.y:.0f} yaw={rover_state.yaw_deg:.1f}",
            end="\r",
        )

    def move_forward(distance_cm: float) -> None:
        move_rover_forward(rover_state, distance_cm)
        clamp_rover_state_xy(rover_state, env)
        draw_dynamic()

    def turn(delta_deg: float) -> None:
        turn_rover(rover_state, delta_deg)
        draw_dynamic()

    def teleport_pose(x: float, y: float, yaw_deg: float) -> None:
        rover_state.x = float(x)
        rover_state.y = float(y)
        rover_state.yaw_deg = float(yaw_deg)
        clamp_rover_state_xy(rover_state, env)
        draw_dynamic()

    draw_dynamic()
    if playback_controls is None:
        vis.register_key_callback(ord("W"), lambda _: (move_forward(ROVER_MOVE_STEP_CM), False)[1])
        vis.register_key_callback(ord("S"), lambda _: (move_forward(-ROVER_MOVE_STEP_CM), False)[1])
        vis.register_key_callback(ord("A"), lambda _: (turn(ROVER_TURN_STEP_DEG), False)[1])
        vis.register_key_callback(ord("D"), lambda _: (turn(-ROVER_TURN_STEP_DEG), False)[1])

        # Fallback keys in case an OS or window manager consumes WASD.
        vis.register_key_callback(ord("I"), lambda _: (move_forward(ROVER_MOVE_STEP_CM), False)[1])
        vis.register_key_callback(ord("K"), lambda _: (move_forward(-ROVER_MOVE_STEP_CM), False)[1])
        vis.register_key_callback(ord("U"), lambda _: (turn(ROVER_TURN_STEP_DEG), False)[1])
        vis.register_key_callback(ord("O"), lambda _: (turn(-ROVER_TURN_STEP_DEG), False)[1])

        print("Open3D controls: WASD move/turn (fallback I/K/U/O), mouse to orbit/pan/zoom")
        vis.run()
    else:
        print("Open3D playback mode: running autonomous trajectory")
        for cmd, value in playback_controls:
            if cmd == "move":
                move_forward(float(value))
            elif cmd == "turn":
                turn(float(value))
            elif cmd == "teleport":
                try:
                    px, py, pyaw = value  # type: ignore[misc]
                    teleport_pose(float(px), float(py), float(pyaw))
                except Exception:
                    continue
            if not vis.poll_events():
                break
            if playback_sleep_s > 0.0:
                time.sleep(playback_sleep_s)
        if hold_after_playback:
            print("Playback complete. Close window to continue.")
            vis.run()
        else:
            vis.poll_events()
    vis.destroy_window()


def main() -> None:
    env = generate_environment(
        size=GRID_SIZE,
        world_size=WORLD_SIZE_CM,
        obstacle_count=OBSTACLE_COUNT,
        seed=SEED,
    )
    view_environment(env)


if __name__ == "__main__":
    main()
