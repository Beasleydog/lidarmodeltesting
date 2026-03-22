from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import glfw
import mujoco
import numpy as np

from physics_sim import LIDAR_CLASS_NONE, MujocoRoverWorld, SimConfig, WorldConfig

DATA_DIR = Path('data')
WORLDS_TO_GENERATE = 1000
TIMESTEPS_PER_WORLD = 500
BASE_RANDOM_SEED: int | None = None
SHOW_VIEWER = True
NUM_WORKERS = 1

THROTTLE_FORWARD_PROB = 0.9
THROTTLE_FORWARD_MIN = 0.35
THROTTLE_REVERSE_MAX = 0.35
STEERING_BASE_STD = 0.55
CONTROL_SEGMENT_STEPS_MIN = 10
CONTROL_SEGMENT_STEPS_MAX = 28
TARGET_REACHED_CM = 480.0
STUCK_POS_XY_EPS_CM = 2.0
STUCK_POS_Z_EPS_CM = 8.0
STUCK_NEAR_OBSTACLE_CM = 600.0
WORLD_CFG_HEIGHT_RANGE = (700.0, 1400.0)
WORLD_CFG_ROCK_SLOT_RANGE = (48, 64)
WORLD_CFG_BUMP_SLOT_RANGE = (14, 24)
WORLD_CFG_CRATER_MAX_RANGE = (10, 14)
WORLD_CFG_TARGET_LOCAL_ROCK_RANGE = (7, 10)
WORLD_CFG_TARGET_LOCAL_BUMP_RANGE = (0, 1)
WORLD_CFG_TARGET_LOCAL_CRATER_RANGE = (1, 3)
VIEWER_WIDTH = 1440
VIEWER_HEIGHT = 900
VIEWER_CAMERA_DISTANCE = 9.0
VIEWER_CAMERA_AZIMUTH_DEG = 135.0
VIEWER_CAMERA_ELEVATION_DEG = -24.0
VIEWER_CAMERA_LOOKAT_Z_OFFSET_M = 1.4
VIEWER_SCENE_MAX_GEOMS = 20000


def _new_world_filename(world_idx: int, world_seed: int) -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return DATA_DIR / f'world_{world_idx:03d}_{stamp}_seed_{world_seed}.txt'


def _world_rng_seed(world_idx: int, world_seed: int) -> int:
    return int((int(world_seed) ^ (int(world_idx) * 2654435761)) & 0xFFFFFFFF)


def _wrap_angle_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _sample_world_config(rng: np.random.Generator) -> WorldConfig:
    return WorldConfig(
        terrain_height_scale_cm=float(rng.uniform(*WORLD_CFG_HEIGHT_RANGE)),
        rock_slot_count=int(rng.integers(WORLD_CFG_ROCK_SLOT_RANGE[0], WORLD_CFG_ROCK_SLOT_RANGE[1] + 1)),
        bump_slot_count=int(rng.integers(WORLD_CFG_BUMP_SLOT_RANGE[0], WORLD_CFG_BUMP_SLOT_RANGE[1] + 1)),
        crater_max_count=int(rng.integers(WORLD_CFG_CRATER_MAX_RANGE[0], WORLD_CFG_CRATER_MAX_RANGE[1] + 1)),
        target_local_rocks=int(rng.integers(WORLD_CFG_TARGET_LOCAL_ROCK_RANGE[0], WORLD_CFG_TARGET_LOCAL_ROCK_RANGE[1] + 1)),
        target_local_bumps=int(rng.integers(WORLD_CFG_TARGET_LOCAL_BUMP_RANGE[0], WORLD_CFG_TARGET_LOCAL_BUMP_RANGE[1] + 1)),
        target_local_craters=int(rng.integers(WORLD_CFG_TARGET_LOCAL_CRATER_RANGE[0], WORLD_CFG_TARGET_LOCAL_CRATER_RANGE[1] + 1)),
    )


def _sample_control_segment(rng: np.random.Generator) -> tuple[int, float, float]:
    steps = int(rng.integers(CONTROL_SEGMENT_STEPS_MIN, CONTROL_SEGMENT_STEPS_MAX + 1))
    if rng.random() < THROTTLE_FORWARD_PROB:
        throttle = float(rng.uniform(THROTTLE_FORWARD_MIN, 1.0))
    else:
        throttle = -float(rng.uniform(0.0, THROTTLE_REVERSE_MAX))
    steering = float(np.clip(rng.normal(0.0, STEERING_BASE_STD), -1.0, 1.0))
    return steps, throttle, steering


def _append_lidar_overlay(scene: mujoco.MjvScene, scan) -> None:
    palette = {
        0: np.array([0.16, 0.47, 1.0, 1.0], dtype=np.float32),
        1: np.array([1.0, 0.18, 0.18, 1.0], dtype=np.float32),
        LIDAR_CLASS_NONE: np.array([0.28, 0.84, 0.34, 0.35], dtype=np.float32),
    }
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    zero = np.zeros(3, dtype=np.float64)
    for idx in range(scan.start_points.shape[0]):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.ones(3, dtype=np.float64), zero, identity, palette[int(scan.class_ids[idx])])
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            2.0,
            (scan.start_points[idx] / 100.0).astype(np.float64),
            (scan.end_points[idx] / 100.0).astype(np.float64),
        )
        scene.ngeom += 1


class DatasetViewer:
    def __init__(self, world: MujocoRoverWorld):
        self.world = world
        if not glfw.init():
            raise RuntimeError('Failed to initialize GLFW')
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(VIEWER_WIDTH, VIEWER_HEIGHT, 'Dataset Generation Viewer', None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError('Failed to create GLFW window')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(world.model, maxgeom=VIEWER_SCENE_MAX_GEOMS)
        self.context = mujoco.MjrContext(world.model, int(mujoco.mjtFontScale.mjFONTSCALE_150))
        if world.consume_hfield_render_dirty():
            mujoco.mjr_uploadHField(world.model, self.context, 0)

    def should_close(self) -> bool:
        return glfw.window_should_close(self.window)

    def render(self, scan, status_left: str, status_right: str) -> None:
        glfw.poll_events()
        pose = self.world.get_pose()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.trackbodyid = -1
        self.cam.lookat[:] = [
            float(pose.origin[0] / 100.0),
            float(pose.origin[1] / 100.0),
            float(pose.origin[2] / 100.0) + VIEWER_CAMERA_LOOKAT_Z_OFFSET_M,
        ]
        self.cam.distance = VIEWER_CAMERA_DISTANCE
        self.cam.azimuth = VIEWER_CAMERA_AZIMUTH_DEG
        self.cam.elevation = VIEWER_CAMERA_ELEVATION_DEG
        if self.world.consume_hfield_render_dirty():
            mujoco.mjr_uploadHField(self.world.model, self.context, 0)
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.world.model, self.world.data, self.opt, self.pert, self.cam, int(mujoco.mjtCatBit.mjCAT_ALL), self.scene)
        _append_lidar_overlay(self.scene, scan)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(
            int(mujoco.mjtFontScale.mjFONTSCALE_150),
            int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
            viewport,
            status_left,
            status_right,
            self.context,
        )
        glfw.swap_buffers(self.window)

    def close(self) -> None:
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()


def _pick_target(world: MujocoRoverWorld, rng: np.random.Generator) -> tuple[float, float]:
    pts = world.obstacle_points_xy_cm()
    if pts.size > 0 and rng.random() < 0.7:
        pick = int(rng.integers(0, pts.shape[0]))
        return float(pts[pick, 0]), float(pts[pick, 1])
    half = 0.4 * world.world_cfg.world_size_cm
    return float(rng.uniform(-half, half)), float(rng.uniform(-half, half))


def _nearest_obstacle_distance_cm(world: MujocoRoverWorld, pose) -> float:
    pts = world.obstacle_points_xy_cm()
    if pts.size == 0:
        return float('inf')
    delta = pts.astype(np.float64) - pose.origin[:2].astype(np.float64)
    return float(np.sqrt(np.min(np.sum(delta * delta, axis=1))))


def _write_header(fh, sensor_count: int) -> None:
    header = [
        'timestep', 'x_cm', 'y_cm', 'z_cm', 'yaw_deg',
        'basis_xx', 'basis_xy', 'basis_xz', 'basis_yx', 'basis_yy', 'basis_yz', 'basis_zx', 'basis_zy', 'basis_zz',
        'teleport_flag', 'throttle_cmd', 'steering_cmd', 'cmd_move_cm', 'cmd_turn_deg',
    ]
    header.extend([f'lidar_cm_{i}' for i in range(sensor_count)])
    header.extend([f'lidar_class_{i}' for i in range(sensor_count)])
    fh.write(','.join(header) + '\n')


def _write_row(fh, timestep: int, pose, teleport_flag: int, throttle_cmd: float, steering_cmd: float, cmd_move_cm: float, cmd_turn_deg: float, scan) -> None:
    row = [
        str(timestep),
        f'{float(pose.origin[0]):.3f}', f'{float(pose.origin[1]):.3f}', f'{float(pose.origin[2]):.3f}', f'{float(pose.yaw_deg):.3f}',
        f'{float(pose.basis[0,0]):.6f}', f'{float(pose.basis[0,1]):.6f}', f'{float(pose.basis[0,2]):.6f}',
        f'{float(pose.basis[1,0]):.6f}', f'{float(pose.basis[1,1]):.6f}', f'{float(pose.basis[1,2]):.6f}',
        f'{float(pose.basis[2,0]):.6f}', f'{float(pose.basis[2,1]):.6f}', f'{float(pose.basis[2,2]):.6f}',
        str(int(teleport_flag)), f'{throttle_cmd:.5f}', f'{steering_cmd:.5f}', f'{cmd_move_cm:.3f}', f'{cmd_turn_deg:.3f}',
    ]
    row.extend(f'{v:.3f}' if v >= 0.0 else '-1' for v in scan.distances_cm)
    row.extend(str(int(v)) for v in scan.class_ids)
    fh.write(','.join(row) + '\n')


def generate_world_dataset(world_idx: int, world_seed: int, rng: np.random.Generator | None = None, show_viewer: bool | None = None) -> Path:
    rng = rng or np.random.default_rng(_world_rng_seed(world_idx, world_seed))
    show_viewer = SHOW_VIEWER if show_viewer is None else show_viewer
    world_cfg = _sample_world_config(rng)
    sim_cfg = SimConfig()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _new_world_filename(world_idx, world_seed)
    with MujocoRoverWorld(gui=bool(show_viewer), seed=world_seed, world_cfg=world_cfg, sim_cfg=sim_cfg) as world:
        viewer = DatasetViewer(world) if show_viewer else None
        target_x, target_y = _pick_target(world, rng)
        segment_left, throttle_base, steering_base = 0, 0.0, 0.0
        prev_pose = world.get_pose()
        sensor_count = len(world.run_lidar_scan().distances_cm)
        try:
            with out_path.open('w', encoding='utf-8') as fh:
                _write_header(fh, sensor_count)
                for t in range(TIMESTEPS_PER_WORLD):
                    if viewer is not None and viewer.should_close():
                        break
                    teleported = 0
                    if world.is_invalid():
                        world.respawn_random()
                        prev_pose = world.get_pose()
                        target_x, target_y = _pick_target(world, rng)
                        segment_left = 0
                        teleported = 1
                    pose_before = world.get_pose()
                    dx = target_x - float(pose_before.origin[0])
                    dy = target_y - float(pose_before.origin[1])
                    if (dx * dx + dy * dy) ** 0.5 <= TARGET_REACHED_CM:
                        target_x, target_y = _pick_target(world, rng)
                    if segment_left <= 0:
                        segment_left, throttle_base, steering_base = _sample_control_segment(rng)
                    desired_yaw = float(np.rad2deg(np.arctan2(target_y - float(pose_before.origin[1]), target_x - float(pose_before.origin[0]))))
                    yaw_err = _wrap_angle_deg(desired_yaw - float(pose_before.yaw_deg))
                    steering_cmd = float(np.clip(0.55 * steering_base + 0.75 * np.clip(yaw_err / 75.0, -1.0, 1.0), -1.0, 1.0))
                    throttle_cmd = float(np.clip(throttle_base * (1.0 - 0.45 * min(abs(yaw_err) / 90.0, 1.0)), -1.0, 1.0))
                    step_start = time.perf_counter()
                    world.step(throttle_cmd, steering_cmd)
                    segment_left -= 1
                    if world.is_invalid():
                        world.respawn_random()
                        prev_pose = world.get_pose()
                        target_x, target_y = _pick_target(world, rng)
                        segment_left = 0
                        teleported = 1
                    pose = world.get_pose()
                    delta_xy_cm = float(np.linalg.norm(pose.origin[:2] - prev_pose.origin[:2]))
                    delta_z_cm = abs(float(pose.origin[2]) - float(prev_pose.origin[2]))
                    obstacle_dist_cm = _nearest_obstacle_distance_cm(world, pose)
                    if obstacle_dist_cm <= STUCK_NEAR_OBSTACLE_CM and delta_xy_cm <= STUCK_POS_XY_EPS_CM and delta_z_cm <= STUCK_POS_Z_EPS_CM:
                        world.respawn_random()
                        pose = world.get_pose()
                        prev_pose = pose
                        target_x, target_y = _pick_target(world, rng)
                        segment_left = 0
                        teleported = 1
                    scan = world.run_lidar_scan()
                    cmd_move_cm = float(np.linalg.norm(pose.origin[:2] - prev_pose.origin[:2]))
                    cmd_turn_deg = _wrap_angle_deg(float(pose.yaw_deg) - float(prev_pose.yaw_deg))
                    _write_row(fh, t, pose, teleported, throttle_cmd, steering_cmd, cmd_move_cm, cmd_turn_deg, scan)
                    if viewer is not None:
                        speed_cm_s = float(np.linalg.norm(world.get_velocity()[0][:2]) * 100.0)
                        viewer.render(
                            scan,
                            (
                                f'World {world_idx:03d} step {t:03d}\n'
                                f'delta_xy={delta_xy_cm:.1f}cm delta_z={delta_z_cm:.1f}cm'
                            ),
                            (
                                f'throttle={throttle_cmd:+.2f} steer={steering_cmd:+.2f} speed={speed_cm_s:.1f} cm/s\n'
                                f'reset if delta_xy<={STUCK_POS_XY_EPS_CM:.1f}cm '
                                f'and delta_z<={STUCK_POS_Z_EPS_CM:.1f}cm, '
                                f'obs_dist={obstacle_dist_cm:.1f}cm '
                                f'(threshold {STUCK_NEAR_OBSTACLE_CM:.1f}cm)'
                            ),
                        )
                        remaining = world.sim_cfg.time_step - (time.perf_counter() - step_start)
                        if remaining > 0.0:
                            time.sleep(remaining)
                    prev_pose = pose
        finally:
            if viewer is not None:
                viewer.close()
    return out_path


def _generate_worker(world_idx: int, world_seed: int) -> str:
    rng = np.random.default_rng(_world_rng_seed(world_idx, world_seed))
    return str(generate_world_dataset(world_idx, world_seed, rng, show_viewer=False))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed_rng = np.random.default_rng(BASE_RANDOM_SEED if BASE_RANDOM_SEED is not None else None)
    world_seeds = [int(seed_rng.integers(0, 2_147_483_647)) for _ in range(WORLDS_TO_GENERATE)]
    if SHOW_VIEWER:
        for i, seed in enumerate(world_seeds):
            print(f'Wrote {generate_world_dataset(i, seed, np.random.default_rng(_world_rng_seed(i, seed)), show_viewer=True)}')
        return
    num_workers = max(1, (os.cpu_count() or 2) - 1) if NUM_WORKERS <= 0 else int(NUM_WORKERS)
    if num_workers <= 1:
        for i, seed in enumerate(world_seeds):
            print(f'Wrote {generate_world_dataset(i, seed, np.random.default_rng(_world_rng_seed(i, seed)), show_viewer=False)}')
        return
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_map = {executor.submit(_generate_worker, i, seed): i for i, seed in enumerate(world_seeds)}
        for future in as_completed(future_map):
            print(f'Wrote world {future_map[future]:03d}: {future.result()}')


if __name__ == '__main__':
    main()
