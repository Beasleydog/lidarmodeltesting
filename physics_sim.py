from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

LIDAR_SENSOR_COORDS_CM = np.array([
    (250.0, 245.0, 50.0), (325.0, 75.0, 130.0), (325.0, 0.0, 130.0), (325.0, -75.0, 130.0),
    (250.0, -245.0, 50.0), (325.0, 75.0, 130.0), (325.0, -75.0, 130.0), (40.0, 235.0, 100.0),
    (40.0, -235.0, 100.0), (-215.0, 270.0, 70.0), (-320.0, 80.0, 10.0), (-320.0, -50.0, 10.0),
    (-215.0, -215.0, 70.0), (325.0, 75.0, 130.0), (325.0, -75.0, 130.0), (250.0, 245.0, 50.0),
    (250.0, -245.0, 50.0),
], dtype=np.float32)
LIDAR_YAW_PITCH_DEG = np.array([
    (30.0, 0.0), (20.0, -20.0), (0.0, 0.0), (-20.0, -20.0), (-30.0, 0.0), (0.0, -25.0),
    (0.0, -25.0), (90.0, -20.0), (-90.0, -20.0), (140.0, 0.0), (180.0, 0.0), (180.0, 0.0),
    (-140.0, 0.0), (20.0, -10.0), (-20.0, -10.0), (15.0, 0.0), (-15.0, 0.0),
], dtype=np.float32)
LIDAR_CLASS_GROUND = 0
LIDAR_CLASS_OBSTACLE = 1
LIDAR_CLASS_NONE = 2
DEFAULT_LIDAR_RANGE_CM = 1000.0
WORLD_SIZE_CM = 22000.0

SENSOR_POS_LOCAL_M = LIDAR_SENSOR_COORDS_CM / 100.0


def _sensor_dirs() -> np.ndarray:
    yaw = np.deg2rad(LIDAR_YAW_PITCH_DEG[:, 0])
    pitch = np.deg2rad(LIDAR_YAW_PITCH_DEG[:, 1])
    cp = np.cos(pitch)
    return np.stack([cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)], axis=1).astype(np.float32)


SENSOR_DIRS_LOCAL = _sensor_dirs()


@dataclass(frozen=True)
class WorldConfig:
    world_size_cm: float = WORLD_SIZE_CM
    terrain_size: int = 192
    terrain_height_scale_cm: float = 450.0
    streamed_hazards_enabled: bool = True
    stream_radius_cm: float = 3600.0
    rock_slot_count: int = 56
    bump_slot_count: int = 18
    crater_max_count: int = 12
    target_local_rocks: int = 8
    target_local_bumps: int = 0
    target_local_craters: int = 2
    hazard_keepout_radius_cm: float = 1000.0
    hazard_overlap_margin_cm: float = 90.0
    hazard_spawn_attempts_per_tick: int = 28
    rock_radius_min_cm: float = 55.0
    rock_radius_max_cm: float = 720.0
    rock_height_min_cm: float = 30.0
    rock_height_max_cm: float = 420.0
    bump_radius_min_cm: float = 14.0
    bump_radius_max_cm: float = 42.0
    bump_height_min_cm: float = 6.0
    bump_height_max_cm: float = 20.0
    force_front_wheel_bumps: bool = True
    front_wheel_bump_forward_min_cm: float = 120.0
    front_wheel_bump_forward_max_cm: float = 240.0
    front_wheel_bump_lateral_max_cm: float = 12.0
    front_wheel_bump_probe_forward_cm: float = 320.0
    crater_radius_min_cm: float = 260.0
    crater_radius_max_cm: float = 900.0
    crater_depth_ratio_min: float = 0.20
    crater_depth_ratio_max: float = 0.34
    crater_lip_height_ratio_min: float = 0.07
    crater_lip_height_ratio_max: float = 0.15
    crater_lip_inner_ratio: float = 0.90
    crater_lip_outer_ratio: float = 1.20
    hazard_epsilon_cm: float = 5.0


@dataclass(frozen=True)
class RoverConfig:
    chassis_mass_kg: float = 130.0
    chassis_size_m: tuple[float, float, float] = (3.2, 3.0, 0.34)
    visual_frame_size_m: tuple[float, float, float] = (6.5, 5.15, 1.15)
    visual_frame_offset_m: tuple[float, float, float] = (0.025, 0.125, 0.625)
    wheel_radius_m: float = 0.68
    wheel_width_m: float = 0.34
    wheel_mass_kg: float = 32.0
    wheelbase_m: float = 5.3
    track_m: float = 4.7
    suspension_rest_m: float = 0.5
    suspension_travel_up_m: float = 0.42
    suspension_travel_down_m: float = 0.36
    suspension_stiffness: float = 2200.0
    suspension_damping: float = 70.0
    steering_max_deg: float = 32.0
    steering_joint_damping: float = 28.0
    steering_joint_frictionloss: float = 1.6
    steering_servo_kp: float = 18000.0
    steering_servo_kd: float = 2400.0
    steering_force_max: float = 36000.0
    wheel_speed_max_rad_s: float = 22.0
    wheel_velocity_kv: float = 2400.0
    wheel_force_max: float = 80000.0
    brake_force_max: float = 180.0
    drive_assist_gain: float = 0.0
    drive_assist_force_max: float = 0.0
    steer_assist_torque_max: float = 0.0
    wheel_pair_friction: tuple[float, float, float, float, float] = (28.0, 14.0, 0.014, 0.001, 0.001)


@dataclass(frozen=True)
class SimConfig:
    time_step: float = 1.0 / 120.0
    action_repeat: int = 5
    gravity_m_s2: float = -7.5
    lidar_range_cm: float = DEFAULT_LIDAR_RANGE_CM
    settle_steps: int = 80
    ray_start_offset_cm: float = 8.0


@dataclass(frozen=True)
class RoverPose:
    origin: np.ndarray
    basis: np.ndarray
    yaw_deg: float


@dataclass(frozen=True)
class LidarScan:
    distances_cm: np.ndarray
    class_ids: np.ndarray
    hit_types: np.ndarray
    start_points: np.ndarray
    end_points: np.ndarray


@dataclass(frozen=True)
class RoverDebugState:
    sim_time_s: float
    pose_origin_cm: np.ndarray
    pose_basis: np.ndarray
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    steering_angle_deg: np.ndarray
    steering_rate_rad_s: np.ndarray
    wheel_rate_rad_s: np.ndarray
    suspension_pos_m: np.ndarray
    suspension_rate_m_s: np.ndarray
    ctrl_values: np.ndarray
    wheel_ground_contacts: np.ndarray
    wheel_rock_contacts: np.ndarray
    chassis_contacts: int
    contact_count: int
    ground_height_cm: float
    hazard_height_cm: float


@dataclass(frozen=True)
class HazardRefreshDebug:
    refresh_count: int
    skipped_distance_gate: bool
    force: bool
    center_x_cm: float
    center_y_cm: float
    yaw_deg: float
    rocks_near_before: int
    bumps_near_before: int
    craters_near_before: int
    rocks_added: int
    bumps_added: int
    craters_added: int
    placed_rocks: int
    placed_bumps: int
    placed_craters: int


@dataclass(frozen=True)
class RockSpec:
    center_cm: tuple[float, float]
    base_z_cm: float
    yaw_deg: float
    size_m: tuple[float, float, float]
    vertices_m: list[list[float]]
    faces: list[int]
    outline_cm: np.ndarray


@dataclass(frozen=True)
class BumpSpec:
    center_cm: tuple[float, float]
    base_z_cm: float
    yaw_deg: float
    size_m: tuple[float, float, float]


@dataclass(frozen=True)
class CraterSpec:
    center_cm: tuple[float, float]
    radius_cm: float
    depth_cm: float
    lip_height_cm: float


def _resize_bilinear(grid: np.ndarray, out_size: int) -> np.ndarray:
    h, w = grid.shape
    y = np.linspace(0, h - 1, out_size)
    x = np.linspace(0, w - 1, out_size)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = x - x0
    wy = y - y0
    top = (1.0 - wx)[None, :] * grid[y0[:, None], x0[None, :]] + wx[None, :] * grid[y0[:, None], x1[None, :]]
    bottom = (1.0 - wx)[None, :] * grid[y1[:, None], x0[None, :]] + wx[None, :] * grid[y1[:, None], x1[None, :]]
    return (1.0 - wy)[:, None] * top + wy[:, None] * bottom


def _blur(grid: np.ndarray) -> np.ndarray:
    padded = np.pad(grid, 1, mode='edge')
    return (
        padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:] + 2.0 * padded[1:-1, :-2]
        + 4.0 * padded[1:-1, 1:-1] + 2.0 * padded[1:-1, 2:] + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
    ) / 16.0


def generate_heightmap(cfg: WorldConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = int(cfg.terrain_size)
    z = np.zeros((size, size), dtype=np.float32)
    amp = 1.0
    cells = 2
    total = 0.0
    for _ in range(5):
        coarse = rng.random((cells + 1, cells + 1), dtype=np.float32)
        z += amp * _resize_bilinear(coarse, size).astype(np.float32)
        total += amp
        amp *= 0.58
        cells *= 2
    z /= max(total, 1e-6)
    z = (z - z.min()) / max(float(z.max() - z.min()), 1e-6)
    for _ in range(2):
        z = (0.35 * z + 0.65 * _blur(z)).astype(np.float32)
    z = ((z - 0.5) * 2.0 * float(cfg.terrain_height_scale_cm)).astype(np.float32)

    hazard = np.zeros_like(z)
    axis = np.linspace(-0.5 * cfg.world_size_cm, 0.5 * cfg.world_size_cm, size, dtype=np.float32)
    return axis, z, hazard


def sample_grid_bilinear(grid: np.ndarray, axis_cm: np.ndarray, x_cm: float, y_cm: float) -> float:
    if x_cm < float(axis_cm[0]) or x_cm > float(axis_cm[-1]) or y_cm < float(axis_cm[0]) or y_cm > float(axis_cm[-1]):
        return float('nan')
    step = float(axis_cm[1] - axis_cm[0])
    fx = (x_cm - float(axis_cm[0])) / step
    fy = (y_cm - float(axis_cm[0])) / step
    max_i = grid.shape[0] - 2
    ix = int(np.clip(np.floor(fx), 0, max_i))
    iy = int(np.clip(np.floor(fy), 0, max_i))
    tx = fx - ix
    ty = fy - iy
    z00 = float(grid[iy, ix])
    z10 = float(grid[iy, ix + 1])
    z01 = float(grid[iy + 1, ix])
    z11 = float(grid[iy + 1, ix + 1])
    z0 = z00 * (1.0 - tx) + z10 * tx
    z1 = z01 * (1.0 - tx) + z11 * tx
    return z0 * (1.0 - ty) + z1 * ty


def _random_rock_mesh(rng: np.random.Generator, radius_cm: float, height_cm: float) -> tuple[list[list[float]], list[int], np.ndarray]:
    verts_n = int(rng.integers(5, 14))
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=verts_n))
    anisotropy = float(rng.uniform(0.45, 1.0))
    x_scale = float(rng.uniform(0.65, 1.35))
    y_scale = x_scale * anisotropy
    radii = radius_cm * rng.uniform(0.35, 1.15, size=verts_n)
    jitter = rng.normal(0.0, 0.11 * radius_cm, size=(verts_n, 2)).astype(np.float32)
    bottom = np.column_stack((
        x_scale * radii * np.cos(angles),
        y_scale * radii * np.sin(angles),
    )).astype(np.float32)
    bottom = (bottom + jitter).astype(np.float32)
    squash = float(rng.uniform(0.18, 0.92))
    lean = rng.normal(0.0, 0.16 * radius_cm, size=bottom.shape).astype(np.float32)
    top = (bottom * squash + lean).astype(np.float32)
    top_z = (height_cm * rng.uniform(0.45, 1.15, size=verts_n)).astype(np.float32)
    vertices: list[list[float]] = []
    indices: list[int] = []
    for x, y in bottom:
        vertices.append([float(x / 100.0), float(y / 100.0), 0.0])
    for i, (x, y) in enumerate(top):
        vertices.append([float(x / 100.0), float(y / 100.0), float(top_z[i] / 100.0)])
    base_center = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    top_center = len(vertices)
    vertices.append([0.0, 0.0, float(height_cm / 100.0)])
    for i in range(verts_n):
        j = (i + 1) % verts_n
        indices.extend([base_center, j, i])
        indices.extend([top_center, verts_n + i, verts_n + j])
        indices.extend([i, j, verts_n + j])
        indices.extend([i, verts_n + j, verts_n + i])
    return vertices, indices, bottom


def _floats_to_str(values: np.ndarray | list[float]) -> str:
    return ' '.join(f'{float(v):.8g}' for v in values)


def _mesh_vertices_to_str(vertices_m: list[list[float]]) -> str:
    flat: list[float] = []
    for vertex in vertices_m:
        flat.extend(vertex)
    return _floats_to_str(flat)


def _build_rock_slot_specs(cfg: WorldConfig, rng: np.random.Generator) -> list[RockSpec]:
    rock_specs: list[RockSpec] = []
    for _ in range(int(cfg.rock_slot_count)):
        radius_x_cm = float(rng.uniform(cfg.rock_radius_min_cm, cfg.rock_radius_max_cm))
        radius_y_cm = float(rng.uniform(cfg.rock_radius_min_cm, cfg.rock_radius_max_cm))
        height_top_cm = float(rng.uniform(cfg.rock_height_min_cm, cfg.rock_height_max_cm))
        mesh_radius_cm = max(radius_x_cm, radius_y_cm)
        vertices_m, faces, outline_cm = _random_rock_mesh(rng, mesh_radius_cm, height_top_cm)
        rock_specs.append(RockSpec(
            (0.0, 0.0),
            0.0,
            0.0,
            (radius_x_cm / 100.0, radius_y_cm / 100.0, height_top_cm / 100.0),
            vertices_m,
            faces,
            outline_cm,
        ))
    return rock_specs


def _build_bump_slot_specs(cfg: WorldConfig, rng: np.random.Generator) -> list[BumpSpec]:
    bump_specs: list[BumpSpec] = []
    for _ in range(int(cfg.bump_slot_count)):
        radius_x_cm = float(rng.uniform(cfg.bump_radius_min_cm, cfg.bump_radius_max_cm))
        radius_y_cm = float(rng.uniform(cfg.bump_radius_min_cm, cfg.bump_radius_max_cm))
        height_cm_bump = float(rng.uniform(cfg.bump_height_min_cm, cfg.bump_height_max_cm))
        bump_specs.append(BumpSpec(
            center_cm=(0.0, 0.0),
            base_z_cm=0.0,
            yaw_deg=0.0,
            size_m=(radius_x_cm / 100.0, radius_y_cm / 100.0, height_cm_bump / 100.0),
        ))
    return bump_specs


def _build_model_xml(
    world_cfg: WorldConfig,
    rover_cfg: RoverConfig,
    sim_cfg: SimConfig,
    terrain_height_cm: np.ndarray,
    terrain_min_cm: float,
    terrain_span_cm: float,
    rock_specs: list[RockSpec],
    bump_specs: list[BumpSpec],
) -> str:
    half_extent_m = world_cfg.world_size_cm / 200.0
    max_height_m = max(terrain_span_cm / 100.0, 0.1)
    base_depth_m = 6.0
    chassis_half = [0.5 * value for value in rover_cfg.chassis_size_m]
    frame_half = [0.5 * value for value in rover_cfg.visual_frame_size_m]
    wheel_range = f'-{rover_cfg.wheel_force_max:.3f} {rover_cfg.wheel_force_max:.3f}'
    brake_range = f'-{rover_cfg.brake_force_max:.3f} {rover_cfg.brake_force_max:.3f}'
    steer_force_range = f'-{rover_cfg.steering_force_max:.3f} {rover_cfg.steering_force_max:.3f}'
    slide_range = f'-{rover_cfg.suspension_travel_down_m:.3f} {rover_cfg.suspension_travel_up_m:.3f}'
    wheel_ctrl_range = f'-{rover_cfg.wheel_speed_max_rad_s:.3f} {rover_cfg.wheel_speed_max_rad_s:.3f}'
    wheel_half_length = 0.5 * rover_cfg.wheel_width_m
    half_wheelbase = 0.5 * rover_cfg.wheelbase_m
    half_track = 0.5 * rover_cfg.track_m
    frame_pos = rover_cfg.visual_frame_offset_m
    pair_friction = ' '.join(f'{value:.6f}' for value in rover_cfg.wheel_pair_friction)

    xml_lines = [
        '<mujoco model="lidar_rover">',
        '  <compiler angle="degree" autolimits="true" eulerseq="xyz"/>',
        f'  <option timestep="{sim_cfg.time_step:.8f}" gravity="0 0 {sim_cfg.gravity_m_s2:.8f}" integrator="implicitfast" cone="elliptic" iterations="100" impratio="4.0">',
        '    <flag nativeccd="enable" multiccd="enable"/>',
        '  </option>',
        '  <visual>',
        '    <global offwidth="1600" offheight="900"/>',
        '    <quality shadowsize="2048" offsamples="4"/>',
        '    <map zfar="800"/>' ,
        '  </visual>',
        '  <asset>',
        f'    <hfield name="terrain_hf" nrow="{terrain_height_cm.shape[0]}" ncol="{terrain_height_cm.shape[1]}" size="{half_extent_m:.6f} {half_extent_m:.6f} {max_height_m:.6f} {base_depth_m:.6f}"/>',
    ]
    for idx, rock in enumerate(rock_specs):
        if rock.vertices_m and rock.faces:
            xml_lines.append(
                '    <mesh name="rock_mesh_{idx}" vertex="{vertex}" face="{face}"/>'.format(
                    idx=idx,
                    vertex=_mesh_vertices_to_str(rock.vertices_m),
                    face=' '.join(str(int(v)) for v in rock.faces),
                )
            )
    xml_lines.extend([
        '  </asset>',
        '  <default>',
        '    <geom friction="1.4 0.02 0.01" solref="0.018 1" solimp="0.97 0.995 0.001" condim="6"/>',
        '    <joint damping="2.0" armature="0.02"/>',
        f'    <default class="suspension"><joint type="slide" axis="0 0 1" range="{slide_range}" stiffness="{rover_cfg.suspension_stiffness:.4f}" damping="{rover_cfg.suspension_damping:.4f}"/></default>',
        '    <default class="steer"><joint type="hinge" axis="0 0 1" damping="{:.6f}" frictionloss="{:.6f}" armature="0.60" solreflimit="0.003 1" solimplimit="0.995 0.9995 0.0001"/></default>'.format(
            rover_cfg.steering_joint_damping, rover_cfg.steering_joint_frictionloss
        ),
        '    <default class="wheel"><joint type="hinge" axis="0 1 0" damping="0.2" frictionloss="0.05" armature="0.04"/><geom type="cylinder" size="{:.6f} {:.6f}" euler="90 0 0" friction="1.1 0.01 0.002" rgba="0.16 0.16 0.18 1" condim="6"/></default>'.format(rover_cfg.wheel_radius_m, wheel_half_length),
        '  </default>',
        '  <worldbody>',
        '    <light pos="0 0 160" dir="0 0 -1" diffuse="0.95 0.95 0.95" specular="0.18 0.18 0.18" directional="true" castshadow="true"/>',
        '    <geom name="terrain" type="hfield" hfield="terrain_hf" pos="0 0 {:.6f}" rgba="0.78 0.73 0.66 1" priority="1"/>'.format(terrain_min_cm / 100.0),
    ])
    for idx, rock in enumerate(rock_specs):
        xml_lines.append(
            '    <geom name="rock_{idx}" type="mesh" mesh="rock_mesh_{idx}" pos="{x:.6f} {y:.6f} {z:.6f}" euler="0 0 {yaw:.6f}" rgba="0.42 0.31 0.19 1" friction="1.2 0.01 0.002" condim="6" priority="1"/>'.format(
                idx=idx,
                x=rock.center_cm[0] / 100.0,
                y=rock.center_cm[1] / 100.0,
                z=rock.base_z_cm / 100.0,
                yaw=rock.yaw_deg,
            )
        )
    for idx, bump in enumerate(bump_specs):
        xml_lines.append(
            '    <geom name="bump_{idx}" type="ellipsoid" size="{sx:.6f} {sy:.6f} {sz:.6f}" pos="{x:.6f} {y:.6f} {z:.6f}" euler="0 0 {yaw:.6f}" rgba="0.56 0.44 0.31 1" friction="1.15 0.01 0.002" condim="6" priority="1"/>'.format(
                idx=idx,
                sx=bump.size_m[0],
                sy=bump.size_m[1],
                sz=bump.size_m[2],
                x=bump.center_cm[0] / 100.0,
                y=bump.center_cm[1] / 100.0,
                z=(bump.base_z_cm / 100.0) + bump.size_m[2],
                yaw=bump.yaw_deg,
            )
        )
    xml_lines.extend([
        '    <body name="rover" pos="0 0 0">',
        '      <freejoint/>',
        '      <geom name="chassis" type="box" size="{:.6f} {:.6f} {:.6f}" rgba="0.83 0.84 0.86 1" mass="{:.6f}" friction="1.1 0.02 0.01"/>'.format(
            chassis_half[0], chassis_half[1], chassis_half[2], rover_cfg.chassis_mass_kg
        ),
        '      <geom name="frame_visual" type="box" pos="{:.6f} {:.6f} {:.6f}" size="{:.6f} {:.6f} {:.6f}" rgba="0.72 0.76 0.80 0.55" contype="0" conaffinity="0"/>'.format(
            frame_pos[0], frame_pos[1], frame_pos[2], frame_half[0], frame_half[1], frame_half[2]
        ),
    ])
    wheel_specs = [
        ('fl', half_wheelbase, half_track, True),
        ('fr', half_wheelbase, -half_track, True),
        ('rl', -half_wheelbase, half_track, False),
        ('rr', -half_wheelbase, -half_track, False),
    ]
    for name, x_pos, y_pos, steerable in wheel_specs:
        xml_lines.append(f'      <body name="susp_{name}" pos="{x_pos:.6f} {y_pos:.6f} {-rover_cfg.suspension_rest_m:.6f}">')
        xml_lines.append(f'        <joint name="susp_{name}" class="suspension"/>')
        xml_lines.append('        <inertial pos="0 0 0" mass="0.4" diaginertia="0.002 0.002 0.002"/>')
        if steerable:
            xml_lines.append(f'        <body name="steer_{name}">')
            xml_lines.append(f'          <joint name="steer_{name}" class="steer" range="{-rover_cfg.steering_max_deg:.6f} {rover_cfg.steering_max_deg:.6f}"/>')
            xml_lines.append('          <inertial pos="0 0 0" mass="0.3" diaginertia="0.0015 0.0015 0.0015"/>')
            xml_lines.append(f'          <body name="wheel_{name}">')
            xml_lines.append(f'            <joint name="wheel_{name}" class="wheel"/>')
            xml_lines.append(f'            <geom name="wheel_geom_{name}" class="wheel" mass="{rover_cfg.wheel_mass_kg:.6f}"/>')
            xml_lines.append('          </body>')
            xml_lines.append('        </body>')
        else:
            xml_lines.append(f'        <body name="wheel_{name}">')
            xml_lines.append(f'          <joint name="wheel_{name}" class="wheel"/>')
            xml_lines.append(f'          <geom name="wheel_geom_{name}" class="wheel" mass="{rover_cfg.wheel_mass_kg:.6f}"/>')
            xml_lines.append('        </body>')
        xml_lines.append('      </body>')
    xml_lines.extend([
        '    </body>',
        '  </worldbody>',
        '  <contact>',
        '    <exclude body1="rover" body2="wheel_fl"/>',
        '    <exclude body1="rover" body2="wheel_fr"/>',
        '    <exclude body1="rover" body2="wheel_rl"/>',
        '    <exclude body1="rover" body2="wheel_rr"/>',
        '    <exclude body1="rover" body2="steer_fl"/>',
        '    <exclude body1="rover" body2="steer_fr"/>',
    ])
    for wheel_name in ('fl', 'fr', 'rl', 'rr'):
        xml_lines.append(
            f'    <pair geom1="wheel_geom_{wheel_name}" geom2="terrain" condim="6" friction="{pair_friction}"/>'
        )
        for idx in range(len(rock_specs)):
            xml_lines.append(
                f'    <pair geom1="wheel_geom_{wheel_name}" geom2="rock_{idx}" condim="6" friction="{pair_friction}"/>'
            )
        for idx in range(len(bump_specs)):
            xml_lines.append(
                f'    <pair geom1="wheel_geom_{wheel_name}" geom2="bump_{idx}" condim="6" friction="{pair_friction}"/>'
            )
    xml_lines.extend([
        '  </contact>',
        '  <equality>',
        '    <joint joint1="steer_fr" joint2="steer_fl" polycoef="0 1 0 0 0" solref="0.002 1" solimp="0.995 0.9997 0.00005"/>',
        '  </equality>',
        '  <actuator>',
        f'    <motor name="steer_act_fl" joint="steer_fl" gear="1" forcerange="{steer_force_range}"/>',
        f'    <motor name="steer_act_fr" joint="steer_fr" gear="1" forcerange="{steer_force_range}"/>',
        f'    <velocity name="drive_fl" joint="wheel_fl" kv="{rover_cfg.wheel_velocity_kv:.6f}" ctrlrange="{wheel_ctrl_range}" ctrllimited="true" forcerange="{wheel_range}"/>',
        f'    <velocity name="drive_fr" joint="wheel_fr" kv="{rover_cfg.wheel_velocity_kv:.6f}" ctrlrange="{wheel_ctrl_range}" ctrllimited="true" forcerange="{wheel_range}"/>',
        f'    <velocity name="drive_rl" joint="wheel_rl" kv="{rover_cfg.wheel_velocity_kv:.6f}" ctrlrange="{wheel_ctrl_range}" ctrllimited="true" forcerange="{wheel_range}"/>',
        f'    <velocity name="drive_rr" joint="wheel_rr" kv="{rover_cfg.wheel_velocity_kv:.6f}" ctrlrange="{wheel_ctrl_range}" ctrllimited="true" forcerange="{wheel_range}"/>',
        f'    <motor name="brake_fl" joint="wheel_fl" gear="1" forcerange="{brake_range}"/>',
        f'    <motor name="brake_fr" joint="wheel_fr" gear="1" forcerange="{brake_range}"/>',
        f'    <motor name="brake_rl" joint="wheel_rl" gear="1" forcerange="{brake_range}"/>',
        f'    <motor name="brake_rr" joint="wheel_rr" gear="1" forcerange="{brake_range}"/>',
        '  </actuator>',
        '</mujoco>',
    ])
    return '\n'.join(xml_lines)


class MujocoRoverWorld:
    def __init__(
        self,
        gui: bool = False,
        seed: int = 0,
        world_cfg: WorldConfig | None = None,
        rover_cfg: RoverConfig | None = None,
        sim_cfg: SimConfig | None = None,
        axis_cm: np.ndarray | None = None,
        height_cm: np.ndarray | None = None,
        hazard_cm: np.ndarray | None = None,
        rock_specs: list[RockSpec] | None = None,
        bump_specs: list[BumpSpec] | None = None,
        obstacle_points_xy_cm: np.ndarray | None = None,
    ):
        self.gui = bool(gui)
        self.seed = int(seed)
        self.world_cfg = world_cfg or WorldConfig()
        self.rover_cfg = rover_cfg or RoverConfig()
        self.sim_cfg = sim_cfg or SimConfig()
        self.rng = np.random.default_rng(self.seed)

        if axis_cm is None or height_cm is None or hazard_cm is None:
            self.axis_cm, base_height_cm, base_hazard_cm = generate_heightmap(self.world_cfg, self.rng)
        else:
            self.axis_cm = np.asarray(axis_cm, dtype=np.float32).copy()
            base_height_cm = np.asarray(height_cm, dtype=np.float32).copy()
            base_hazard_cm = np.asarray(hazard_cm, dtype=np.float32).copy()
        self.base_height_cm = base_height_cm.astype(np.float32)
        self.height_cm = self.base_height_cm.copy()
        self.hazard_cm = base_hazard_cm.astype(np.float32).copy()
        self._stream_center_cm = np.zeros(2, dtype=np.float32)
        self._placed_rocks: list[RockSpec] = []
        self._placed_bumps: list[BumpSpec] = []
        self._placed_craters: list[CraterSpec] = []
        self._hazard_spawn_attempts = {'rock': 0, 'bump': 0, 'crater': 0}
        self._hazard_spawn_success = {'rock': 0, 'bump': 0, 'crater': 0}
        self._hazard_refresh_count = 0
        self._last_hazard_refresh = HazardRefreshDebug(
            refresh_count=0,
            skipped_distance_gate=False,
            force=False,
            center_x_cm=0.0,
            center_y_cm=0.0,
            yaw_deg=0.0,
            rocks_near_before=0,
            bumps_near_before=0,
            craters_near_before=0,
            rocks_added=0,
            bumps_added=0,
            craters_added=0,
            placed_rocks=0,
            placed_bumps=0,
            placed_craters=0,
        )
        self._hfield_render_dirty = True
        if rock_specs is None:
            self.rock_specs = _build_rock_slot_specs(self.world_cfg, np.random.default_rng(self.seed ^ 0xA5A5A5A5))
            rock_nav = np.zeros((0, 2), dtype=np.float32)
        else:
            self.rock_specs = list(rock_specs)
            if obstacle_points_xy_cm is None:
                rock_nav = np.zeros((0, 2), dtype=np.float32)
            else:
                rock_nav = np.asarray(obstacle_points_xy_cm, dtype=np.float32).copy()
        if bump_specs is None:
            self.bump_specs = _build_bump_slot_specs(self.world_cfg, np.random.default_rng(self.seed ^ 0x5A5A5A5A))
        else:
            self.bump_specs = list(bump_specs)
        if obstacle_points_xy_cm is None:
            self._obstacle_points_xy_cm = rock_nav.copy()
        else:
            self._obstacle_points_xy_cm = np.asarray(obstacle_points_xy_cm, dtype=np.float32).copy()

        max_crater_depth = self.world_cfg.crater_radius_max_cm * self.world_cfg.crater_depth_ratio_max
        self.terrain_min_cm = float(np.min(self.base_height_cm) - max_crater_depth - 20.0)
        self.terrain_max_cm = float(np.max(self.base_height_cm) + 20.0)
        self.terrain_span_cm = max(self.terrain_max_cm - self.terrain_min_cm, 1.0)
        terrain_norm = ((self.height_cm - self.terrain_min_cm) / self.terrain_span_cm).astype(np.float32)
        xml = _build_model_xml(
            self.world_cfg,
            self.rover_cfg,
            self.sim_cfg,
            self.height_cm,
            self.terrain_min_cm,
            self.terrain_span_cm,
            self.rock_specs,
            self.bump_specs,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.hfield_data[:] = terrain_norm.reshape(-1)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'rover')
        self.chassis_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'chassis')
        self.terrain_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'terrain')
        self.rock_geom_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'rock_{idx}') for idx in range(len(self.rock_specs))
        ], dtype=np.int32)
        self.bump_geom_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'bump_{idx}') for idx in range(len(self.bump_specs))
        ], dtype=np.int32)
        self.rock_geom_id_set = set(int(v) for v in self.rock_geom_ids.tolist())
        self.bump_geom_id_set = set(int(v) for v in self.bump_geom_ids.tolist())
        self.wheel_geom_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'wheel_geom_fl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'wheel_geom_fr'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'wheel_geom_rl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'wheel_geom_rr'),
        ], dtype=np.int32)
        self.rover_geom_id_set = self._collect_body_subtree_geom_ids(self.rover_body_id)
        self.steering_actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'steer_act_fl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'steer_act_fr'),
        ], dtype=np.int32)
        self.drive_actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_fl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_fr'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_rl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_rr'),
        ], dtype=np.int32)
        self.brake_actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'brake_fl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'brake_fr'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'brake_rl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'brake_rr'),
        ], dtype=np.int32)
        self.drive_joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'wheel_fl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'wheel_fr'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'wheel_rl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'wheel_rr'),
        ], dtype=np.int32)
        self.steering_joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'steer_fl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'steer_fr'),
        ], dtype=np.int32)
        self.suspension_joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'susp_fl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'susp_fr'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'susp_rl'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'susp_rr'),
        ], dtype=np.int32)

        self._geomgroup_all = np.ones(6, dtype=np.uint8)
        self._deactivate_all_streamed_geoms()
        self.respawn_random()

    def __enter__(self) -> MujocoRoverWorld:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        return None

    def obstacle_points_xy_cm(self) -> np.ndarray:
        return self._obstacle_points_xy_cm.copy()

    def _collect_body_subtree_geom_ids(self, root_body_id: int) -> set[int]:
        root_id = int(root_body_id)
        parent_ids = np.asarray(self.model.body_parentid, dtype=np.int32)
        geom_body_ids = np.asarray(self.model.geom_bodyid, dtype=np.int32)
        body_ids: set[int] = set()
        for body_id in range(int(self.model.nbody)):
            cursor = int(body_id)
            while True:
                if cursor == root_id:
                    body_ids.add(int(body_id))
                    break
                if cursor <= 0 or cursor == int(parent_ids[cursor]):
                    break
                cursor = int(parent_ids[cursor])
        return {int(geom_id) for geom_id, body_id in enumerate(geom_body_ids.tolist()) if int(body_id) in body_ids}

    def consume_hfield_render_dirty(self) -> bool:
        dirty = bool(self._hfield_render_dirty)
        self._hfield_render_dirty = False
        return dirty

    def hazard_generation_stats(self) -> dict[str, int]:
        return {
            'rock_attempts': int(self._hazard_spawn_attempts['rock']),
            'rock_success': int(self._hazard_spawn_success['rock']),
            'bump_attempts': int(self._hazard_spawn_attempts['bump']),
            'bump_success': int(self._hazard_spawn_success['bump']),
            'crater_attempts': int(self._hazard_spawn_attempts['crater']),
            'crater_success': int(self._hazard_spawn_success['crater']),
            'placed_rocks': len(self._placed_rocks),
            'placed_bumps': len(self._placed_bumps),
            'placed_craters': len(self._placed_craters),
        }

    def hazard_counts_near_pose(self) -> dict[str, int]:
        pose = self.get_pose()
        rocks, bumps, craters = self._hazard_counts_near(float(pose.origin[0]), float(pose.origin[1]))
        return {'rocks': int(rocks), 'bumps': int(bumps), 'craters': int(craters)}

    def hazard_refresh_debug(self) -> HazardRefreshDebug:
        return self._last_hazard_refresh

    def _set_geom_yaw_deg(self, geom_id: int, yaw_deg: float) -> None:
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_euler2Quat(quat, np.array([0.0, 0.0, math.radians(yaw_deg)], dtype=np.float64), 'xyz')
        self.model.geom_quat[int(geom_id)] = quat

    def _deactivate_all_streamed_geoms(self) -> None:
        hidden_z = (self.terrain_min_cm / 100.0) - 120.0
        for geom_id in self.rock_geom_ids:
            self.model.geom_pos[int(geom_id)] = np.array([0.0, 0.0, hidden_z], dtype=np.float64)
            self._set_geom_yaw_deg(int(geom_id), 0.0)
        for geom_id in self.bump_geom_ids:
            self.model.geom_pos[int(geom_id)] = np.array([0.0, 0.0, hidden_z], dtype=np.float64)
            self.model.geom_size[int(geom_id)] = np.array([0.01, 0.01, 0.01], dtype=np.float64)
            self._set_geom_yaw_deg(int(geom_id), 0.0)
        mujoco.mj_forward(self.model, self.data)

    def _apply_streamed_craters(self, craters: list[CraterSpec]) -> None:
        hazard = np.zeros_like(self.base_height_cm, dtype=np.float32)
        uplift = np.zeros_like(self.base_height_cm, dtype=np.float32)
        if craters:
            xg, yg = np.meshgrid(self.axis_cm, self.axis_cm)
            for crater in craters:
                dist = np.sqrt(((xg - crater.center_cm[0]) / crater.radius_cm) ** 2 + ((yg - crater.center_cm[1]) / crater.radius_cm) ** 2)
                mask = dist <= 1.0
                bowl = np.zeros_like(hazard)
                inner_profile = np.clip(1.0 - dist[mask], 0.0, 1.0)
                bowl[mask] = crater.depth_cm * np.power(inner_profile, 0.28).astype(np.float32)
                hazard = np.maximum(hazard, bowl)
                lip = np.zeros_like(hazard)
                lip_inner = float(self.world_cfg.crater_lip_inner_ratio)
                lip_outer = float(self.world_cfg.crater_lip_outer_ratio)
                lip_mask = (dist >= lip_inner) & (dist <= lip_outer)
                if np.any(lip_mask):
                    lip_mid = 0.5 * (lip_inner + lip_outer)
                    lip_half = max(0.5 * (lip_outer - lip_inner), 1e-3)
                    lip_profile = 1.0 - np.abs((dist[lip_mask] - lip_mid) / lip_half)
                    lip[lip_mask] = crater.lip_height_cm * np.power(np.clip(lip_profile, 0.0, 1.0), 1.15).astype(np.float32)
                    uplift = np.maximum(uplift, lip)
        self.hazard_cm = hazard
        self.height_cm = (self.base_height_cm + uplift - self.hazard_cm).astype(np.float32)
        terrain_norm = np.clip((self.height_cm - self.terrain_min_cm) / self.terrain_span_cm, 0.0, 1.0).astype(np.float32)
        self.model.hfield_data[:] = terrain_norm.reshape(-1)
        self._hfield_render_dirty = True

    def _rock_radius_cm(self, rock: RockSpec) -> float:
        return 100.0 * max(float(rock.size_m[0]), float(rock.size_m[1]))

    def _bump_radius_cm(self, bump: BumpSpec) -> float:
        return 100.0 * max(float(bump.size_m[0]), float(bump.size_m[1]))

    def _sync_persistent_hazards_to_model(self) -> None:
        self._apply_streamed_craters(self._placed_craters)
        hidden_z = (self.terrain_min_cm / 100.0) - 120.0
        for slot_idx, geom_id in enumerate(self.rock_geom_ids):
            if slot_idx < len(self._placed_rocks):
                rock = self._placed_rocks[slot_idx]
                surface_z_cm = sample_grid_bilinear(self.height_cm, self.axis_cm, rock.center_cm[0], rock.center_cm[1])
                self.model.geom_size[int(geom_id)] = np.asarray(rock.size_m, dtype=np.float64)
                self.model.geom_pos[int(geom_id)] = np.array([rock.center_cm[0] / 100.0, rock.center_cm[1] / 100.0, surface_z_cm / 100.0], dtype=np.float64)
                self._set_geom_yaw_deg(int(geom_id), rock.yaw_deg)
            else:
                self.model.geom_size[int(geom_id)] = np.array([0.01, 0.01, 0.01], dtype=np.float64)
                self.model.geom_pos[int(geom_id)] = np.array([0.0, 0.0, hidden_z], dtype=np.float64)
                self._set_geom_yaw_deg(int(geom_id), 0.0)
        for slot_idx, geom_id in enumerate(self.bump_geom_ids):
            if slot_idx < len(self._placed_bumps):
                bump = self._placed_bumps[slot_idx]
                surface_z_cm = sample_grid_bilinear(self.height_cm, self.axis_cm, bump.center_cm[0], bump.center_cm[1])
                self.model.geom_size[int(geom_id)] = np.asarray(bump.size_m, dtype=np.float64)
                self.model.geom_pos[int(geom_id)] = np.array([
                    bump.center_cm[0] / 100.0,
                    bump.center_cm[1] / 100.0,
                    (surface_z_cm / 100.0) + bump.size_m[2],
                ], dtype=np.float64)
                self._set_geom_yaw_deg(int(geom_id), bump.yaw_deg)
            else:
                self.model.geom_size[int(geom_id)] = np.array([0.01, 0.01, 0.01], dtype=np.float64)
                self.model.geom_pos[int(geom_id)] = np.array([0.0, 0.0, hidden_z], dtype=np.float64)
                self._set_geom_yaw_deg(int(geom_id), 0.0)
        rock_points = np.asarray([[item.center_cm[0], item.center_cm[1]] for item in self._placed_rocks], dtype=np.float32) if self._placed_rocks else np.zeros((0, 2), dtype=np.float32)
        crater_points = np.asarray([[item.center_cm[0], item.center_cm[1]] for item in self._placed_craters], dtype=np.float32) if self._placed_craters else np.zeros((0, 2), dtype=np.float32)
        self._obstacle_points_xy_cm = crater_points if rock_points.size == 0 else np.vstack([crater_points, rock_points]).astype(np.float32)
        mujoco.mj_forward(self.model, self.data)

    def _hazard_counts_near(self, center_x_cm: float, center_y_cm: float) -> tuple[int, int, int]:
        radius2 = float(self.world_cfg.stream_radius_cm * self.world_cfg.stream_radius_cm)
        def _count(specs, get_center):
            count = 0
            for spec in specs:
                dx = get_center(spec)[0] - center_x_cm
                dy = get_center(spec)[1] - center_y_cm
                if dx * dx + dy * dy <= radius2:
                    count += 1
            return count
        return (
            _count(self._placed_rocks, lambda item: item.center_cm),
            _count(self._placed_bumps, lambda item: item.center_cm),
            _count(self._placed_craters, lambda item: item.center_cm),
        )

    def _hazard_overlaps(self, center_x_cm: float, center_y_cm: float, radius_cm: float) -> bool:
        margin = float(self.world_cfg.hazard_overlap_margin_cm)
        for rock in self._placed_rocks:
            dx = rock.center_cm[0] - center_x_cm
            dy = rock.center_cm[1] - center_y_cm
            if dx * dx + dy * dy < (radius_cm + self._rock_radius_cm(rock) + margin) ** 2:
                return True
        for bump in self._placed_bumps:
            dx = bump.center_cm[0] - center_x_cm
            dy = bump.center_cm[1] - center_y_cm
            if dx * dx + dy * dy < (radius_cm + self._bump_radius_cm(bump) + margin) ** 2:
                return True
        for crater in self._placed_craters:
            dx = crater.center_cm[0] - center_x_cm
            dy = crater.center_cm[1] - center_y_cm
            if dx * dx + dy * dy < (radius_cm + crater.radius_cm + margin) ** 2:
                return True
        return False

    def _sample_hazard_point_near(self, kind: str, center_x_cm: float, center_y_cm: float, hazard_radius_cm: float) -> tuple[float, float] | None:
        inner = max(self.world_cfg.hazard_keepout_radius_cm + hazard_radius_cm, 0.35 * self.world_cfg.stream_radius_cm)
        outer = self.world_cfg.stream_radius_cm
        bounds = 0.47 * self.world_cfg.world_size_cm
        for _ in range(int(self.world_cfg.hazard_spawn_attempts_per_tick)):
            self._hazard_spawn_attempts[kind] += 1
            ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
            dist = float(np.sqrt(self.rng.uniform(inner * inner, outer * outer)))
            x_cm = float(np.clip(center_x_cm + dist * math.cos(ang), -bounds, bounds))
            y_cm = float(np.clip(center_y_cm + dist * math.sin(ang), -bounds, bounds))
            if not np.isfinite(sample_grid_bilinear(self.base_height_cm, self.axis_cm, x_cm, y_cm)):
                continue
            if self._hazard_overlaps(x_cm, y_cm, hazard_radius_cm):
                continue
            return x_cm, y_cm
        return None

    def _sample_front_wheel_bump_point(self, center_x_cm: float, center_y_cm: float, yaw_deg: float, side_index: int, hazard_radius_cm: float) -> tuple[float, float] | None:
        anchors = self._wheel_anchor_xy_cm(center_x_cm, center_y_cm, yaw_deg)
        front_anchor = anchors[side_index]
        yaw = math.radians(yaw_deg)
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
        for _ in range(6):
            forward_offset = (
                0.55 * self.rover_cfg.wheel_radius_m * 100.0
                + hazard_radius_cm
                + float(self.rng.uniform(
                    self.world_cfg.front_wheel_bump_forward_min_cm,
                    self.world_cfg.front_wheel_bump_forward_max_cm,
                ))
            )
            lateral_offset = float(self.rng.uniform(
                -self.world_cfg.front_wheel_bump_lateral_max_cm,
                self.world_cfg.front_wheel_bump_lateral_max_cm,
            ))
            point = front_anchor.astype(np.float64) + forward * forward_offset + lateral * lateral_offset
            x_cm = float(np.clip(point[0], -0.47 * self.world_cfg.world_size_cm, 0.47 * self.world_cfg.world_size_cm))
            y_cm = float(np.clip(point[1], -0.47 * self.world_cfg.world_size_cm, 0.47 * self.world_cfg.world_size_cm))
            if not np.isfinite(sample_grid_bilinear(self.base_height_cm, self.axis_cm, x_cm, y_cm)):
                continue
            if self._hazard_overlaps(x_cm, y_cm, hazard_radius_cm):
                continue
            return x_cm, y_cm
        return None

    def _make_bump_spec(self, x_cm: float, y_cm: float, radius_x_cm: float, radius_y_cm: float, height_cm_bump: float) -> BumpSpec:
        base_z = sample_grid_bilinear(self.height_cm, self.axis_cm, x_cm, y_cm)
        return BumpSpec(
            center_cm=(x_cm, y_cm),
            base_z_cm=float(base_z),
            yaw_deg=float(self.rng.uniform(-180.0, 180.0)),
            size_m=(radius_x_cm / 100.0, radius_y_cm / 100.0, height_cm_bump / 100.0),
        )

    def _has_bump_ahead_of_front_wheel(self, center_x_cm: float, center_y_cm: float, yaw_deg: float, side_index: int) -> bool:
        anchors = self._wheel_anchor_xy_cm(center_x_cm, center_y_cm, yaw_deg)
        front_anchor = anchors[side_index].astype(np.float64)
        yaw = math.radians(yaw_deg)
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
        wheel_front_cm = 0.55 * self.rover_cfg.wheel_radius_m * 100.0
        forward_min = wheel_front_cm
        forward_max = wheel_front_cm + self.world_cfg.front_wheel_bump_probe_forward_cm
        lateral_max = self.world_cfg.front_wheel_bump_lateral_max_cm + self.world_cfg.bump_radius_max_cm
        for bump in self._placed_bumps:
            rel = np.array(bump.center_cm, dtype=np.float64) - front_anchor
            forward_cm = float(rel @ forward)
            if forward_cm < forward_min or forward_cm > forward_max:
                continue
            lateral_cm = abs(float(rel @ lateral))
            if lateral_cm <= lateral_max:
                return True
        return False

    def _refresh_streamed_hazards(self, center_x_cm: float, center_y_cm: float, yaw_deg: float | None = None, force: bool = False) -> None:
        if not self.world_cfg.streamed_hazards_enabled:
            return
        if yaw_deg is None:
            yaw_deg = self.get_pose().yaw_deg
        if not force and np.sum((self._stream_center_cm - np.array([center_x_cm, center_y_cm], dtype=np.float32)) ** 2) < 120.0 * 120.0:
            self._last_hazard_refresh = HazardRefreshDebug(
                refresh_count=self._hazard_refresh_count,
                skipped_distance_gate=True,
                force=bool(force),
                center_x_cm=float(center_x_cm),
                center_y_cm=float(center_y_cm),
                yaw_deg=float(yaw_deg),
                rocks_near_before=0,
                bumps_near_before=0,
                craters_near_before=0,
                rocks_added=0,
                bumps_added=0,
                craters_added=0,
                placed_rocks=len(self._placed_rocks),
                placed_bumps=len(self._placed_bumps),
                placed_craters=len(self._placed_craters),
            )
            return
        self._hazard_refresh_count += 1
        rocks_near, bumps_near, craters_near = self._hazard_counts_near(center_x_cm, center_y_cm)
        rocks_near_before = int(rocks_near)
        bumps_near_before = int(bumps_near)
        craters_near_before = int(craters_near)
        rocks_added = 0
        bumps_added = 0
        craters_added = 0
        while rocks_near < int(self.world_cfg.target_local_rocks) and len(self._placed_rocks) < len(self.rock_geom_ids):
            slot_idx = len(self._placed_rocks)
            rock_template = self.rock_specs[slot_idx]
            footprint_radius_cm = self._rock_radius_cm(rock_template)
            pos = self._sample_hazard_point_near('rock', center_x_cm, center_y_cm, footprint_radius_cm)
            if pos is None:
                break
            x_cm, y_cm = pos
            base_z = sample_grid_bilinear(self.height_cm, self.axis_cm, x_cm, y_cm)
            self._placed_rocks.append(RockSpec(
                (x_cm, y_cm),
                float(base_z),
                float(self.rng.uniform(-180.0, 180.0)),
                rock_template.size_m,
                rock_template.vertices_m,
                rock_template.faces,
                rock_template.outline_cm,
            ))
            self._hazard_spawn_success['rock'] += 1
            rocks_near += 1
            rocks_added += 1
        if self.world_cfg.force_front_wheel_bumps:
            for side_index in (0, 1):
                if self._has_bump_ahead_of_front_wheel(center_x_cm, center_y_cm, yaw_deg, side_index):
                    continue
                radius_x_cm = float(self.rng.uniform(self.world_cfg.bump_radius_min_cm, self.world_cfg.bump_radius_max_cm))
                radius_y_cm = float(self.rng.uniform(self.world_cfg.bump_radius_min_cm, self.world_cfg.bump_radius_max_cm))
                height_cm_bump = float(self.rng.uniform(self.world_cfg.bump_height_min_cm, self.world_cfg.bump_height_max_cm))
                footprint_radius_cm = max(radius_x_cm, radius_y_cm)
                replaced_bump = None
                if side_index < len(self._placed_bumps):
                    replaced_bump = self._placed_bumps.pop(side_index)
                pos = self._sample_front_wheel_bump_point(center_x_cm, center_y_cm, yaw_deg, side_index, footprint_radius_cm)
                self._hazard_spawn_attempts['bump'] += 1
                if pos is None:
                    if replaced_bump is not None:
                        self._placed_bumps.insert(side_index, replaced_bump)
                    continue
                x_cm, y_cm = pos
                bump = self._make_bump_spec(x_cm, y_cm, radius_x_cm, radius_y_cm, height_cm_bump)
                self._placed_bumps.insert(side_index, bump)
                self._hazard_spawn_success['bump'] += 1
                if replaced_bump is None:
                    bumps_near += 1
                bumps_added += 1
        while bumps_near < int(self.world_cfg.target_local_bumps) and len(self._placed_bumps) < len(self.bump_geom_ids):
            radius_x_cm = float(self.rng.uniform(self.world_cfg.bump_radius_min_cm, self.world_cfg.bump_radius_max_cm))
            radius_y_cm = float(self.rng.uniform(self.world_cfg.bump_radius_min_cm, self.world_cfg.bump_radius_max_cm))
            height_cm_bump = float(self.rng.uniform(self.world_cfg.bump_height_min_cm, self.world_cfg.bump_height_max_cm))
            footprint_radius_cm = max(radius_x_cm, radius_y_cm)
            pos = self._sample_hazard_point_near('bump', center_x_cm, center_y_cm, footprint_radius_cm)
            if pos is None:
                break
            x_cm, y_cm = pos
            self._placed_bumps.append(self._make_bump_spec(x_cm, y_cm, radius_x_cm, radius_y_cm, height_cm_bump))
            self._hazard_spawn_success['bump'] += 1
            bumps_near += 1
            bumps_added += 1
        while craters_near < int(self.world_cfg.target_local_craters) and len(self._placed_craters) < int(self.world_cfg.crater_max_count):
            radius_cm = float(self.rng.uniform(self.world_cfg.crater_radius_min_cm, self.world_cfg.crater_radius_max_cm))
            pos = self._sample_hazard_point_near('crater', center_x_cm, center_y_cm, radius_cm)
            if pos is None:
                break
            x_cm, y_cm = pos
            depth_cm = radius_cm * float(self.rng.uniform(self.world_cfg.crater_depth_ratio_min, self.world_cfg.crater_depth_ratio_max))
            lip_height_cm = depth_cm * float(self.rng.uniform(
                self.world_cfg.crater_lip_height_ratio_min,
                self.world_cfg.crater_lip_height_ratio_max,
            ))
            self._placed_craters.append(CraterSpec((x_cm, y_cm), radius_cm, depth_cm, lip_height_cm))
            self._hazard_spawn_success['crater'] += 1
            craters_near += 1
            craters_added += 1
            self._apply_streamed_craters(self._placed_craters)
        self._sync_persistent_hazards_to_model()
        self._stream_center_cm[:] = [center_x_cm, center_y_cm]
        self._last_hazard_refresh = HazardRefreshDebug(
            refresh_count=self._hazard_refresh_count,
            skipped_distance_gate=False,
            force=bool(force),
            center_x_cm=float(center_x_cm),
            center_y_cm=float(center_y_cm),
            yaw_deg=float(yaw_deg),
            rocks_near_before=rocks_near_before,
            bumps_near_before=bumps_near_before,
            craters_near_before=craters_near_before,
            rocks_added=int(rocks_added),
            bumps_added=int(bumps_added),
            craters_added=int(craters_added),
            placed_rocks=len(self._placed_rocks),
            placed_bumps=len(self._placed_bumps),
            placed_craters=len(self._placed_craters),
        )

    def _spawn_quality(self, x_cm: float, y_cm: float) -> float:
        h0 = self.sample_height_cm(x_cm, y_cm)
        if not np.isfinite(h0):
            return float('inf')
        step = 120.0
        hx1 = self.sample_height_cm(x_cm + step, y_cm)
        hx0 = self.sample_height_cm(x_cm - step, y_cm)
        hy1 = self.sample_height_cm(x_cm, y_cm + step)
        hy0 = self.sample_height_cm(x_cm, y_cm - step)
        if not all(np.isfinite(v) for v in (hx1, hx0, hy1, hy0)):
            return float('inf')
        slope = abs(hx1 - hx0) + abs(hy1 - hy0)
        hazard = max(self.sample_hazard_cm(x_cm, y_cm), 0.0)
        return float(slope + 2.5 * hazard)

    def _nearest_active_obstacle_distance_cm(self, x_cm: float, y_cm: float) -> float:
        if self._obstacle_points_xy_cm.size == 0:
            return float('inf')
        delta = self._obstacle_points_xy_cm.astype(np.float64) - np.array([x_cm, y_cm], dtype=np.float64)
        return float(np.sqrt(np.min(np.sum(delta * delta, axis=1))))

    def _wheel_anchor_xy_cm(self, x_cm: float, y_cm: float, yaw_deg: float) -> np.ndarray:
        half_wheelbase = 50.0 * self.rover_cfg.wheelbase_m
        half_track = 50.0 * self.rover_cfg.track_m
        local = np.array([
            [half_wheelbase, half_track],
            [half_wheelbase, -half_track],
            [-half_wheelbase, half_track],
            [-half_wheelbase, -half_track],
        ], dtype=np.float64)
        yaw = math.radians(yaw_deg)
        rot = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)],
        ], dtype=np.float64)
        return (local @ rot.T + np.array([x_cm, y_cm], dtype=np.float64)).astype(np.float32)

    def _sample_surface_height_cm(self, x_cm: float, y_cm: float) -> float:
        ray_origin = np.array([x_cm / 100.0, y_cm / 100.0, (self.terrain_max_cm + 4000.0) / 100.0], dtype=np.float64)
        ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        normal = np.zeros(3, dtype=np.float64)
        dist_m = mujoco.mj_ray(
            self.model,
            self.data,
            ray_origin,
            ray_dir,
            self._geomgroup_all,
            1,
            self.rover_body_id,
            geomid,
            normal,
        )
        if dist_m < 0.0:
            return float('nan')
        return float((ray_origin[2] - dist_m) * 100.0)

    def _spawn_support_height_cm(self, x_cm: float, y_cm: float, yaw_deg: float) -> float:
        sample_points = self._wheel_anchor_xy_cm(x_cm, y_cm, yaw_deg)
        center = np.array([[x_cm, y_cm]], dtype=np.float32)
        heights = []
        for px, py in np.vstack([center, sample_points]):
            surface = self._sample_surface_height_cm(float(px), float(py))
            if np.isfinite(surface):
                heights.append(surface)
        if not heights:
            return float('nan')
        return float(max(heights))

    def _settled_spawn_is_valid(self) -> bool:
        pose = self.get_pose()
        support_cm = self._spawn_support_height_cm(float(pose.origin[0]), float(pose.origin[1]), pose.yaw_deg)
        clearance_cm = float(pose.origin[2] - support_cm) if np.isfinite(support_cm) else float('inf')
        lin_vel, ang_vel = self.get_velocity()
        speed_m_s = float(np.linalg.norm(lin_vel))
        ang_speed = float(np.linalg.norm(ang_vel))
        debug = self.get_debug_state()
        wheel_contacts = int(np.sum(debug.wheel_ground_contacts) + np.sum(debug.wheel_rock_contacts))
        if not np.isfinite(support_cm):
            return False
        if clearance_cm > 120.0:
            return False
        if speed_m_s > 1.5 or ang_speed > 1.5:
            return False
        if wheel_contacts <= 0:
            return False
        return True

    def _apply_controls(self, throttle: float, steering: float) -> None:
        self.data.xfrc_applied[self.rover_body_id, :] = 0.0
        steer_input = float(np.clip(steering, -1.0, 1.0))
        desired_angle = steer_input * math.radians(self.rover_cfg.steering_max_deg)
        steer_qpos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.steering_joint_ids], dtype=np.float64)
        steer_qvel = np.array([self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self.steering_joint_ids], dtype=np.float64)
        rack_angle = float(np.mean(steer_qpos))
        rack_rate = float(np.mean(steer_qvel))
        rack_torque = (
            self.rover_cfg.steering_servo_kp * (desired_angle - rack_angle)
            - self.rover_cfg.steering_servo_kd * rack_rate
        )
        rack_torque = float(np.clip(rack_torque, -self.rover_cfg.steering_force_max, self.rover_cfg.steering_force_max))
        self.data.ctrl[self.steering_actuator_ids[0]] = rack_torque
        self.data.ctrl[self.steering_actuator_ids[1]] = 0.0
        desired_wheel_rate = float(np.clip(throttle, -1.0, 1.0)) * self.rover_cfg.wheel_speed_max_rad_s
        wheel_vel = np.array([self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self.drive_joint_ids], dtype=np.float64)
        wheel_targets = np.full((self.drive_actuator_ids.shape[0],), desired_wheel_rate, dtype=np.float64)
        steer_mag = abs(desired_angle)
        if abs(desired_wheel_rate) > 1e-4 and steer_mag > math.radians(1.0):
            turn_radius = max(self.rover_cfg.wheelbase_m / max(abs(math.tan(desired_angle)), 1e-4), 0.55 * self.rover_cfg.track_m)
            inner_scale = max((turn_radius - 0.5 * self.rover_cfg.track_m) / turn_radius, 0.35)
            outer_scale = min((turn_radius + 0.5 * self.rover_cfg.track_m) / turn_radius, 1.85)
            if steer_input > 0.0:
                left_scale, right_scale = inner_scale, outer_scale
            else:
                left_scale, right_scale = outer_scale, inner_scale
            wheel_targets[:] = desired_wheel_rate * np.array([left_scale, right_scale, left_scale, right_scale], dtype=np.float64)
        self.data.ctrl[self.drive_actuator_ids] = wheel_targets
        if abs(throttle) > 1e-4:
            self.data.ctrl[self.brake_actuator_ids] = 0.0
        else:
            self.data.ctrl[self.brake_actuator_ids] = np.clip(-0.08 * wheel_vel * self.rover_cfg.brake_force_max, -self.rover_cfg.brake_force_max, self.rover_cfg.brake_force_max)

    def sample_height_cm(self, x_cm: float, y_cm: float) -> float:
        return sample_grid_bilinear(self.height_cm, self.axis_cm, x_cm, y_cm)

    def sample_hazard_cm(self, x_cm: float, y_cm: float) -> float:
        return sample_grid_bilinear(self.hazard_cm, self.axis_cm, x_cm, y_cm)

    def reset_pose(self, x_cm: float, y_cm: float, yaw_deg: float, settle_steps: int | None = None) -> None:
        self._refresh_streamed_hazards(x_cm, y_cm, yaw_deg=yaw_deg, force=True)
        support_cm = self._spawn_support_height_cm(x_cm, y_cm, yaw_deg)
        if not np.isfinite(support_cm):
            raise ValueError('Requested pose is outside the valid terrain support area')
        spawn_z_m = (support_cm / 100.0) + self.rover_cfg.wheel_radius_m + self.rover_cfg.suspension_rest_m + 0.06
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qpos[0:3] = [x_cm / 100.0, y_cm / 100.0, spawn_z_m]
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_euler2Quat(quat, np.array([0.0, 0.0, math.radians(yaw_deg)], dtype=np.float64), 'xyz')
        self.data.qpos[3:7] = quat
        for joint_group in (self.suspension_joint_ids, self.steering_joint_ids, self.drive_joint_ids):
            for joint_id in joint_group:
                qadr = self.model.jnt_qposadr[joint_id]
                dadr = self.model.jnt_dofadr[joint_id]
                self.data.qpos[qadr] = 0.0
                self.data.qvel[dadr] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._apply_controls(0.0, 0.0)
        for _ in range(int(self.sim_cfg.settle_steps if settle_steps is None else settle_steps)):
            mujoco.mj_step(self.model, self.data)

    def respawn_random(self) -> None:
        bounds = 0.38 * self.world_cfg.world_size_cm
        for _ in range(20):
            best_choice: tuple[float, float, float] | None = None
            best_score = float('inf')
            for _ in range(24):
                if self._obstacle_points_xy_cm.size > 0:
                    target = self._obstacle_points_xy_cm[int(self.rng.integers(0, self._obstacle_points_xy_cm.shape[0]))]
                    ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
                    if self.world_cfg.streamed_hazards_enabled:
                        dist = float(self.rng.uniform(900.0, 2200.0))
                    else:
                        dist = float(self.rng.uniform(260.0, 980.0))
                    cand_x = float(np.clip(target[0] + dist * math.cos(ang), -bounds, bounds))
                    cand_y = float(np.clip(target[1] + dist * math.sin(ang), -bounds, bounds))
                    cand_yaw = float(np.rad2deg(np.arctan2(float(target[1] - cand_y), float(target[0] - cand_x))))
                else:
                    cand_x = float(self.rng.uniform(-bounds, bounds))
                    cand_y = float(self.rng.uniform(-bounds, bounds))
                    cand_yaw = float(self.rng.uniform(-180.0, 180.0))
                support_cm = self._spawn_support_height_cm(cand_x, cand_y, cand_yaw)
                if not np.isfinite(support_cm):
                    continue
                obstacle_clearance_cm = self._nearest_active_obstacle_distance_cm(cand_x, cand_y)
                score = self._spawn_quality(cand_x, cand_y) + 0.35 * abs(support_cm - self.sample_height_cm(cand_x, cand_y))
                if self.world_cfg.streamed_hazards_enabled and obstacle_clearance_cm < 1200.0:
                    score += 25.0 * (1200.0 - obstacle_clearance_cm)
                if score < best_score:
                    best_score = score
                    best_choice = (cand_x, cand_y, cand_yaw)
            if best_choice is None:
                x_cm = float(self.rng.uniform(-bounds, bounds))
                y_cm = float(self.rng.uniform(-bounds, bounds))
                yaw_deg = float(self.rng.uniform(-180.0, 180.0))
            else:
                x_cm, y_cm, yaw_deg = best_choice
            self.reset_pose(x_cm, y_cm, yaw_deg)
            if self._settled_spawn_is_valid():
                return
        raise RuntimeError('Failed to find a stable rover spawn state')

    def step(self, throttle: float, steering: float, repeats: int | None = None) -> None:
        repeats = int(self.sim_cfg.action_repeat if repeats is None else repeats)
        for _ in range(max(repeats, 1)):
            self._apply_controls(throttle, steering)
            mujoco.mj_step(self.model, self.data)
            rover_pos = self.data.xpos[self.rover_body_id]
            pose = self.get_pose()
            self._refresh_streamed_hazards(float(rover_pos[0] * 100.0), float(rover_pos[1] * 100.0), yaw_deg=pose.yaw_deg)

    def get_pose(self) -> RoverPose:
        origin = self.data.xpos[self.rover_body_id].copy() * 100.0
        basis = self.data.xmat[self.rover_body_id].reshape(3, 3).copy().astype(np.float32)
        yaw_deg = float(np.rad2deg(np.arctan2(float(basis[1, 0]), float(basis[0, 0]))))
        return RoverPose(origin=origin.astype(np.float32), basis=basis, yaw_deg=yaw_deg)

    def get_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.cvel[self.rover_body_id, 3:].copy().astype(np.float32), self.data.cvel[self.rover_body_id, :3].copy().astype(np.float32)

    def get_debug_state(self) -> RoverDebugState:
        pose = self.get_pose()
        basis = pose.basis.astype(np.float64)
        pitch_deg = float(np.rad2deg(math.asin(np.clip(-basis[2, 0], -1.0, 1.0))))
        roll_deg = float(np.rad2deg(math.atan2(float(basis[2, 1]), float(basis[2, 2]))))
        linear_velocity = self.data.cvel[self.rover_body_id, 3:].copy().astype(np.float32)
        angular_velocity = self.data.cvel[self.rover_body_id, :3].copy().astype(np.float32)
        steering_qpos = []
        steering_qvel = []
        for joint_id in self.steering_joint_ids:
            steering_qpos.append(math.degrees(float(self.data.qpos[self.model.jnt_qposadr[joint_id]])))
            steering_qvel.append(float(self.data.qvel[self.model.jnt_dofadr[joint_id]]))
        wheel_qvel = []
        for joint_id in self.drive_joint_ids:
            wheel_qvel.append(float(self.data.qvel[self.model.jnt_dofadr[joint_id]]))
        suspension_qpos = []
        suspension_qvel = []
        for joint_id in self.suspension_joint_ids:
            suspension_qpos.append(float(self.data.qpos[self.model.jnt_qposadr[joint_id]]))
            suspension_qvel.append(float(self.data.qvel[self.model.jnt_dofadr[joint_id]]))

        wheel_ground_contacts = np.zeros((len(self.drive_joint_ids),), dtype=np.int32)
        wheel_rock_contacts = np.zeros((len(self.drive_joint_ids),), dtype=np.int32)
        chassis_contacts = 0
        rock_geom_ids = self.rock_geom_id_set
        bump_geom_ids = self.bump_geom_id_set
        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            pair = (int(contact.geom1), int(contact.geom2))
            if self.chassis_geom_id in pair:
                chassis_contacts += 1
            for wheel_idx, geom_id in enumerate(self.wheel_geom_ids):
                if geom_id not in pair:
                    continue
                other = pair[0] if pair[1] == geom_id else pair[1]
                if other == self.terrain_geom_id:
                    wheel_ground_contacts[wheel_idx] += 1
                elif other in bump_geom_ids:
                    wheel_ground_contacts[wheel_idx] += 1
                elif other in rock_geom_ids:
                    wheel_rock_contacts[wheel_idx] += 1

        ground_height = self.sample_height_cm(float(pose.origin[0]), float(pose.origin[1]))
        hazard_height = self.sample_hazard_cm(float(pose.origin[0]), float(pose.origin[1]))
        return RoverDebugState(
            sim_time_s=float(self.data.time),
            pose_origin_cm=pose.origin.copy(),
            pose_basis=pose.basis.copy(),
            yaw_deg=pose.yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            linear_velocity_m_s=linear_velocity,
            angular_velocity_rad_s=angular_velocity,
            steering_angle_deg=np.asarray(steering_qpos, dtype=np.float32),
            steering_rate_rad_s=np.asarray(steering_qvel, dtype=np.float32),
            wheel_rate_rad_s=np.asarray(wheel_qvel, dtype=np.float32),
            suspension_pos_m=np.asarray(suspension_qpos, dtype=np.float32),
            suspension_rate_m_s=np.asarray(suspension_qvel, dtype=np.float32),
            ctrl_values=self.data.ctrl.copy().astype(np.float32),
            wheel_ground_contacts=wheel_ground_contacts,
            wheel_rock_contacts=wheel_rock_contacts,
            chassis_contacts=chassis_contacts,
            contact_count=int(self.data.ncon),
            ground_height_cm=float(ground_height),
            hazard_height_cm=float(hazard_height),
        )

    def is_invalid(self) -> bool:
        pose = self.get_pose()
        half = 0.5 * self.world_cfg.world_size_cm
        if abs(float(pose.origin[0])) > half or abs(float(pose.origin[1])) > half:
            return True
        if float(pose.basis[2, 2]) < 0.15:
            return True
        ground_cm = self.sample_height_cm(float(pose.origin[0]), float(pose.origin[1]))
        if np.isfinite(ground_cm) and float(pose.origin[2]) < ground_cm - 30.0:
            return True
        return False

    def run_lidar_scan(self) -> LidarScan:
        pose = self.get_pose()
        starts = pose.origin[None, :] / 100.0 + SENSOR_POS_LOCAL_M @ pose.basis.T
        dirs = SENSOR_DIRS_LOCAL @ pose.basis.T
        ray_starts = starts + dirs * (self.sim_cfg.ray_start_offset_cm / 100.0)
        distances_cm = np.full((starts.shape[0],), -1.0, dtype=np.float32)
        class_ids = np.full((starts.shape[0],), LIDAR_CLASS_NONE, dtype=np.int32)
        hit_types = np.full((starts.shape[0],), 'none', dtype='<U8')
        end_points = (starts + dirs * (self.sim_cfg.lidar_range_cm / 100.0)).copy()
        geomid = np.array([-1], dtype=np.int32)
        normal = np.zeros(3, dtype=np.float64)
        max_range_m = self.sim_cfg.lidar_range_cm / 100.0
        for idx in range(starts.shape[0]):
            remaining_m = float(max_range_m)
            march_start = ray_starts[idx].astype(np.float64).copy()
            hit_point = None
            hit_geom = -1
            while remaining_m > 0.0:
                geomid[0] = -1
                dist_m = mujoco.mj_ray(
                    self.model,
                    self.data,
                    march_start,
                    dirs[idx].astype(np.float64),
                    self._geomgroup_all,
                    1,
                    self.rover_body_id,
                    geomid,
                    normal,
                )
                if dist_m < 0.0 or dist_m > remaining_m:
                    hit_geom = -1
                    break
                candidate_hit_point = march_start + dirs[idx] * dist_m
                candidate_geom = int(geomid[0])
                if candidate_geom in self.rover_geom_id_set:
                    advance_m = max(float(dist_m) + 0.02, 0.02)
                    march_start = march_start + dirs[idx].astype(np.float64) * advance_m
                    remaining_m -= advance_m
                    continue
                hit_point = candidate_hit_point
                hit_geom = candidate_geom
                break
            if hit_point is None:
                continue
            dist_cm = float(np.linalg.norm(hit_point - starts[idx]) * 100.0)
            if dist_cm < 5.0:
                continue
            distances_cm[idx] = dist_cm
            end_points[idx] = hit_point
            if hit_geom in self.rock_geom_id_set:
                class_ids[idx] = LIDAR_CLASS_OBSTACLE
                hit_types[idx] = 'obstacle'
                continue
            if hit_geom in self.bump_geom_id_set:
                class_ids[idx] = LIDAR_CLASS_GROUND
                hit_types[idx] = 'ground'
                continue
            if hit_geom == self.terrain_geom_id:
                class_ids[idx] = LIDAR_CLASS_GROUND
                hit_types[idx] = 'ground'
                continue
            class_ids[idx] = LIDAR_CLASS_OBSTACLE
            hit_types[idx] = 'obstacle'
        return LidarScan(
            distances_cm=distances_cm,
            class_ids=class_ids,
            hit_types=hit_types,
            start_points=(starts * 100.0).astype(np.float32),
            end_points=(end_points * 100.0).astype(np.float32),
        )


BulletRoverWorld = MujocoRoverWorld
