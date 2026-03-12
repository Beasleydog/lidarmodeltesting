from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import glfw
import mujoco
import numpy as np

from physics_sim import LIDAR_CLASS_NONE, MujocoRoverWorld, SimConfig, WorldConfig

SHOW_LIDAR = True
ENABLE_MODEL_INFERENCE = True
MODEL_INFERENCE_HISTORY_STEP_GAP = 20
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
CAMERA_DISTANCE = 4.8
CAMERA_AZIMUTH_DEG = 135.0
CAMERA_ELEVATION_DEG = -24.0
CAMERA_LOOKAT_Z_OFFSET_M = 1.4
CAMERA_AUTO_VISIBILITY_FRAMES = 24
DEMO_TERRAIN_HEIGHT_SCALE_CM = 1000.0
STATUS_EVERY_S = 0.2
SCENE_MAX_GEOMS = 20000
LOG_DIR = Path('logs')


@dataclass
class DriveInput:
    show_lidar: bool = SHOW_LIDAR
    respawn_requested: bool = False
    camera_distance: float = CAMERA_DISTANCE
    auto_camera_frames_left: int = CAMERA_AUTO_VISIBILITY_FRAMES

    def on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        del scancode, mods
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            self.respawn_requested = True
            self.auto_camera_frames_left = CAMERA_AUTO_VISIBILITY_FRAMES
        elif key == glfw.KEY_L:
            self.show_lidar = not self.show_lidar
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD, glfw.KEY_PAGE_UP):
            self.camera_distance = max(1.0, self.camera_distance - 0.5)
            self.auto_camera_frames_left = 0
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT, glfw.KEY_PAGE_DOWN):
            self.camera_distance = min(20.0, self.camera_distance + 0.5)
            self.auto_camera_frames_left = 0

    def on_scroll(self, window, xoffset: float, yoffset: float) -> None:
        del window, xoffset
        self.camera_distance = float(np.clip(self.camera_distance - 0.5 * yoffset, 1.0, 20.0))
        self.auto_camera_frames_left = 0

    def sample_drive(self, window) -> tuple[float, float]:
        forward = glfw.get_key(window, glfw.KEY_W) == glfw.PRESS or glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS
        backward = glfw.get_key(window, glfw.KEY_S) == glfw.PRESS or glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS
        left = glfw.get_key(window, glfw.KEY_A) == glfw.PRESS or glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
        right = glfw.get_key(window, glfw.KEY_D) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
        throttle = 1.0 if forward and not backward else -1.0 if backward and not forward else 0.0
        steering = 1.0 if left and not right else -1.0 if right and not left else 0.0
        return throttle, steering


def _configure_camera(cam: mujoco.MjvCamera, world: MujocoRoverWorld, controls: DriveInput) -> None:
    pose = world.get_pose()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.trackbodyid = -1
    cam.lookat[:] = [
        float(pose.origin[0] / 100.0),
        float(pose.origin[1] / 100.0),
        float(pose.origin[2] / 100.0) + CAMERA_LOOKAT_Z_OFFSET_M,
    ]
    cam.distance = controls.camera_distance
    cam.azimuth = CAMERA_AZIMUTH_DEG
    cam.elevation = CAMERA_ELEVATION_DEG


def _camera_visibility_score(context: mujoco.MjrContext, viewport: mujoco.MjrRect) -> float:
    rgb = np.zeros((max(viewport.height, 1), max(viewport.width, 1), 3), dtype=np.uint8)
    depth = np.zeros((max(viewport.height, 1), max(viewport.width, 1)), dtype=np.float32)
    mujoco.mjr_readPixels(rgb, depth, viewport, context)
    return float(rgb.mean())


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


def _append_persistent_hit_overlay(scene: mujoco.MjvScene, hit_points_cm: list[np.ndarray]) -> None:
    if not hit_points_cm:
        return
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    zero = np.zeros(3, dtype=np.float64)
    color = np.array([1.0, 0.08, 0.08, 1.0], dtype=np.float64)
    half_height_cm = 45.0
    for hit_point_cm in hit_points_cm:
        if scene.ngeom >= scene.maxgeom:
            break
        start_pt_cm = np.asarray(hit_point_cm, dtype=np.float64).copy()
        end_pt_cm = np.asarray(hit_point_cm, dtype=np.float64).copy()
        start_pt_cm[2] -= half_height_cm
        end_pt_cm[2] += half_height_cm
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.ones(3, dtype=np.float64),
            zero,
            identity,
            color,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            3.0,
            (start_pt_cm / 100.0).astype(np.float64),
            (end_pt_cm / 100.0).astype(np.float64),
        )
        scene.ngeom += 1


class DemoInferencer:
    def __init__(self, history_step_gap: int) -> None:
        import model as lidar_model

        self._model = lidar_model
        self.history_step_gap = int(max(history_step_gap, 1))
        self._sample_counter = 0
        self._last_obstacle_mask: np.ndarray | None = None

    def reset_history(self) -> None:
        self._model.reset_history()
        self._sample_counter = 0
        self._last_obstacle_mask = None

    def predict_obstacle_mask(
        self,
        lidar_cm: np.ndarray,
        pose_origin_cm: np.ndarray,
        basis: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        should_sample = (self._sample_counter % self.history_step_gap) == 0 or self._last_obstacle_mask is None
        self._sample_counter += 1
        if should_sample:
            result = self._model.ingest_lidar(
                lidar_cm=lidar_cm,
                pose_xyz_cm=pose_origin_cm,
                basis=basis,
            )
            self._last_obstacle_mask = np.asarray(result["obstacle_mask"], dtype=bool)
        assert self._last_obstacle_mask is not None
        return self._last_obstacle_mask.copy(), bool(should_sample)


def _quantized_hit_key(point_cm: np.ndarray) -> tuple[int, int, int]:
    scale = 20.0
    return tuple(int(np.rint(float(v) / scale)) for v in point_cm.tolist())


def _capture_inferred_hit_beams(
    hit_points_cm: list[np.ndarray],
    seen_keys: set[tuple[int, int, int]],
    scan,
    obstacle_mask: np.ndarray,
) -> int:
    added = 0
    valid_hits = np.asarray(scan.distances_cm, dtype=np.float32) >= 0.0
    for idx in np.flatnonzero(np.asarray(obstacle_mask, dtype=bool) & valid_hits):
        end_pt_cm = np.asarray(scan.end_points[int(idx)], dtype=np.float32)
        key = _quantized_hit_key(end_pt_cm)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        hit_points_cm.append(end_pt_cm.copy())
        added += 1
        if len(hit_points_cm) > 12000:
            old_end = hit_points_cm.pop(0)
            seen_keys.discard(_quantized_hit_key(old_end))
    return added


def _create_window() -> glfw._GLFWwindow:
    if not glfw.init():
        raise RuntimeError('Failed to initialize GLFW')
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, 'Lidar Rover Demo', None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError('Failed to create GLFW window')
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    return window


def _open_demo_log() -> tuple[Path, object, csv.DictWriter]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f'environment_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    handle = log_path.open('w', newline='', encoding='utf-8')
    writer = csv.DictWriter(handle, fieldnames=[
        'sim_time_s',
        'pose_x_cm',
        'pose_y_cm',
        'pose_z_cm',
        'yaw_deg',
        'throttle_cmd',
        'steer_cmd',
        'speed_cm_s',
        'near_rocks',
        'near_bumps',
        'near_craters',
        'rock_attempts',
        'rock_success',
        'bump_attempts',
        'bump_success',
        'crater_attempts',
        'crater_success',
        'placed_rocks',
        'placed_bumps',
        'placed_craters',
        'refresh_count',
        'refresh_skipped_distance_gate',
        'refresh_force',
        'refresh_center_x_cm',
        'refresh_center_y_cm',
        'refresh_yaw_deg',
        'refresh_rocks_near_before',
        'refresh_bumps_near_before',
        'refresh_craters_near_before',
        'refresh_rocks_added',
        'refresh_bumps_added',
        'refresh_craters_added',
    ])
    writer.writeheader()
    handle.flush()
    return log_path, handle, writer


def main() -> None:
    controls = DriveInput()
    window = _create_window()
    glfw.set_key_callback(window, controls.on_key)
    glfw.set_scroll_callback(window, controls.on_scroll)
    print('Controls: hold W/S for full throttle, hold A/D for full steering, mouse wheel or PageUp/PageDown to zoom, R respawn, L toggle lidar, Esc quit')
    inferencer = DemoInferencer(MODEL_INFERENCE_HISTORY_STEP_GAP) if ENABLE_MODEL_INFERENCE else None
    if inferencer is not None:
        print(
            'Model inference overlay enabled: feeding live pose + lidar into runs/gru_lidar_classifier.pt '
            f'(history step gap={MODEL_INFERENCE_HISTORY_STEP_GAP})'
        )
    log_path, log_handle, log_writer = _open_demo_log()
    print(f'Logging demo telemetry to {log_path}')
    try:
        with MujocoRoverWorld(
            gui=True,
            world_cfg=WorldConfig(terrain_height_scale_cm=DEMO_TERRAIN_HEIGHT_SCALE_CM),
            sim_cfg=SimConfig(),
        ) as world:
            cam = mujoco.MjvCamera()
            opt = mujoco.MjvOption()
            pert = mujoco.MjvPerturb()
            scene = mujoco.MjvScene(world.model, maxgeom=SCENE_MAX_GEOMS)
            context = mujoco.MjrContext(world.model, int(mujoco.mjtFontScale.mjFONTSCALE_150))
            if world.consume_hfield_render_dirty():
                mujoco.mjr_uploadHField(world.model, context, 0)
            _configure_camera(cam, world, controls)
            last_status_t = 0.0
            persistent_hit_points_cm: list[np.ndarray] = []
            persistent_hit_keys: set[tuple[int, int, int]] = set()
            while not glfw.window_should_close(window):
                step_start = time.perf_counter()
                glfw.poll_events()
                throttle, steering = controls.sample_drive(window)
                if controls.respawn_requested:
                    world.respawn_random()
                    controls.respawn_requested = False
                    if inferencer is not None:
                        inferencer.reset_history()
                world.step(throttle, steering)
                if world.is_invalid():
                    world.respawn_random()
                    controls.auto_camera_frames_left = CAMERA_AUTO_VISIBILITY_FRAMES
                    if inferencer is not None:
                        inferencer.reset_history()
                if world.consume_hfield_render_dirty():
                    mujoco.mjr_uploadHField(world.model, context, 0)
                pose = world.get_pose()
                cam.lookat[:] = [
                    float(pose.origin[0] / 100.0),
                    float(pose.origin[1] / 100.0),
                    float(pose.origin[2] / 100.0) + CAMERA_LOOKAT_Z_OFFSET_M,
                ]
                cam.distance = controls.camera_distance
                scan = world.run_lidar_scan()
                inferred_obstacle_mask: np.ndarray | None = None
                added_hit_beams = 0
                sampled_model_history = False
                if inferencer is not None:
                    inferred_obstacle_mask, sampled_model_history = inferencer.predict_obstacle_mask(
                        lidar_cm=np.asarray(scan.distances_cm, dtype=np.float32),
                        pose_origin_cm=np.asarray(pose.origin, dtype=np.float32),
                        basis=np.asarray(pose.basis, dtype=np.float32),
                    )
                    if sampled_model_history:
                        added_hit_beams = _capture_inferred_hit_beams(
                            persistent_hit_points_cm,
                            persistent_hit_keys,
                            scan,
                            inferred_obstacle_mask,
                        )
                width, height = glfw.get_framebuffer_size(window)
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjv_updateScene(world.model, world.data, opt, pert, cam, int(mujoco.mjtCatBit.mjCAT_ALL), scene)
                if controls.show_lidar:
                    _append_lidar_overlay(scene, scan)
                if inferencer is not None:
                    _append_persistent_hit_overlay(scene, persistent_hit_points_cm)
                mujoco.mjr_render(viewport, scene, context)
                if controls.auto_camera_frames_left > 0:
                    visibility = _camera_visibility_score(context, viewport)
                    if visibility < 4.0:
                        controls.camera_distance = min(20.0, controls.camera_distance + 0.5)
                    controls.auto_camera_frames_left -= 1
                lin_vel, _ = world.get_velocity()
                speed_cm_s = float(np.linalg.norm(lin_vel[:2]) * 100.0)
                status_left = 'Drive'
                status_right = f'throttle={throttle:+.2f} steer={steering:+.2f} speed={speed_cm_s:.1f} cm/s lidar={"on" if controls.show_lidar else "off"}'
                if inferred_obstacle_mask is not None:
                    predicted_count = int(np.count_nonzero(inferred_obstacle_mask))
                    status_left = 'Drive + Model'
                    status_right += (
                        f' infer_obs={predicted_count}/{len(inferred_obstacle_mask)} '
                        f'persist={len(persistent_hit_points_cm)} +{added_hit_beams} '
                        f'sampled={"yes" if sampled_model_history else "no"}'
                    )
                mujoco.mjr_overlay(
                    int(mujoco.mjtFontScale.mjFONTSCALE_150),
                    int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                    viewport,
                    status_left,
                    status_right,
                    context,
                )
                glfw.swap_buffers(window)
                now = time.perf_counter()
                if (now - last_status_t) >= STATUS_EVERY_S:
                    near_counts = world.hazard_counts_near_pose()
                    hazard_stats = world.hazard_generation_stats()
                    refresh = world.hazard_refresh_debug()
                    log_writer.writerow({
                        'sim_time_s': float(world.data.time),
                        'pose_x_cm': float(pose.origin[0]),
                        'pose_y_cm': float(pose.origin[1]),
                        'pose_z_cm': float(pose.origin[2]),
                        'yaw_deg': float(pose.yaw_deg),
                        'throttle_cmd': float(throttle),
                        'steer_cmd': float(steering),
                        'speed_cm_s': speed_cm_s,
                        'near_rocks': int(near_counts['rocks']),
                        'near_bumps': int(near_counts['bumps']),
                        'near_craters': int(near_counts['craters']),
                        'rock_attempts': int(hazard_stats['rock_attempts']),
                        'rock_success': int(hazard_stats['rock_success']),
                        'bump_attempts': int(hazard_stats['bump_attempts']),
                        'bump_success': int(hazard_stats['bump_success']),
                        'crater_attempts': int(hazard_stats['crater_attempts']),
                        'crater_success': int(hazard_stats['crater_success']),
                        'placed_rocks': int(hazard_stats['placed_rocks']),
                        'placed_bumps': int(hazard_stats['placed_bumps']),
                        'placed_craters': int(hazard_stats['placed_craters']),
                        'refresh_count': int(refresh.refresh_count),
                        'refresh_skipped_distance_gate': int(refresh.skipped_distance_gate),
                        'refresh_force': int(refresh.force),
                        'refresh_center_x_cm': float(refresh.center_x_cm),
                        'refresh_center_y_cm': float(refresh.center_y_cm),
                        'refresh_yaw_deg': float(refresh.yaw_deg),
                        'refresh_rocks_near_before': int(refresh.rocks_near_before),
                        'refresh_bumps_near_before': int(refresh.bumps_near_before),
                        'refresh_craters_near_before': int(refresh.craters_near_before),
                        'refresh_rocks_added': int(refresh.rocks_added),
                        'refresh_bumps_added': int(refresh.bumps_added),
                        'refresh_craters_added': int(refresh.craters_added),
                    })
                    log_handle.flush()
                    print(
                        f'Pose x={pose.origin[0]:.0f} y={pose.origin[1]:.0f} z={pose.origin[2]:.0f} yaw={pose.yaw_deg:.1f} '
                        f'speed_cm_s={speed_cm_s:.1f} throttle={throttle:+.2f} steer={steering:+.2f}',
                        end='\r',
                        flush=True,
                    )
                    last_status_t = now
                remaining = world.sim_cfg.time_step - (time.perf_counter() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)
            context.free()
    finally:
        log_handle.close()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == '__main__':
    main()
