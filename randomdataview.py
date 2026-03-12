from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import glfw
import mujoco
import numpy as np

from physics_sim import MujocoRoverWorld, SimConfig, WorldConfig


DATA_DIR = Path("data")
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900
SCENE_MAX_GEOMS = 20000
CAMERA_DISTANCE_M = 18.0
CAMERA_AZIMUTH_DEG = 135.0
CAMERA_ELEVATION_DEG = -28.0
CAMERA_LOOKAT_Z_OFFSET_M = 1.4
PLAYBACK_STEP_SECONDS = 0.06
PATH_COLOR = np.array([0.15, 0.88, 0.55, 1.0], dtype=np.float32)
TELEPORT_COLOR = np.array([1.0, 0.42, 0.16, 1.0], dtype=np.float32)
PREDICTED_PATH_COLOR = np.array([0.20, 0.55, 1.0, 1.0], dtype=np.float32)
DEFAULT_PREDICTED_PATH = Path("world_smartdrive_20260309_183830.txt")
# DEFAULT_PREDICTED_PATH = "off"
DISABLED_PATH_VALUES = {"", "none", "off", "false", "0"}


@dataclass(frozen=True)
class ReplayFrame:
    timestep: int
    origin_cm: np.ndarray
    basis: np.ndarray
    yaw_deg: float
    teleport_flag: int


@dataclass(frozen=True)
class ReplayData:
    path: Path
    frames: list[ReplayFrame]


def _basis_fieldnames() -> list[str]:
    return [
        "basis_xx",
        "basis_xy",
        "basis_xz",
        "basis_yx",
        "basis_yy",
        "basis_yz",
        "basis_zx",
        "basis_zy",
        "basis_zz",
    ]


def _yaw_basis(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_replay_file(path: Path) -> ReplayData:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row")
        basis_keys = [name for name in _basis_fieldnames() if name in reader.fieldnames]
        use_basis = len(basis_keys) == 9
        frames: list[ReplayFrame] = []
        for row in reader:
            yaw_deg = float(row["yaw_deg"])
            basis = (
                np.asarray([float(row[key]) for key in basis_keys], dtype=np.float32).reshape(3, 3)
                if use_basis
                else _yaw_basis(yaw_deg)
            )
            frames.append(
                ReplayFrame(
                    timestep=int(float(row.get("timestep", len(frames)))),
                    origin_cm=np.asarray(
                        [
                            float(row["x_cm"]),
                            float(row["y_cm"]),
                            float(row["z_cm"]),
                        ],
                        dtype=np.float32,
                    ),
                    basis=basis,
                    yaw_deg=yaw_deg,
                    teleport_flag=int(float(row.get("teleport_flag", "0"))),
                )
            )
    if not frames:
        raise ValueError(f"{path} has no timestep rows")
    return ReplayData(path=path, frames=frames)


def pick_random_file(data_dir: Path, seed: int | None = None) -> Path:
    files = [p for p in sorted(data_dir.glob("world_*.txt")) if p.stat().st_size > 0]
    if not files:
        raise SystemExit(f"No non-empty data files found in {data_dir}")
    rng = random.Random(seed)
    return rng.choice(files)


def _create_window() -> glfw._GLFWwindow:
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Random Data Replay", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    return window


def _flat_replay_world_config() -> WorldConfig:
    return WorldConfig(
        terrain_height_scale_cm=25.0,
        streamed_hazards_enabled=False,
        rock_slot_count=0,
        bump_slot_count=0,
        crater_max_count=0,
        target_local_rocks=0,
        target_local_bumps=0,
        target_local_craters=0,
        force_front_wheel_bumps=False,
    )


def _set_world_pose(world: MujocoRoverWorld, frame: ReplayFrame) -> None:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, frame.basis.astype(np.float64).reshape(-1))
    world.data.qpos[:] = 0.0
    world.data.qvel[:] = 0.0
    world.data.ctrl[:] = 0.0
    world.data.qpos[0:3] = frame.origin_cm.astype(np.float64) / 100.0
    world.data.qpos[3:7] = quat
    for joint_group in (world.suspension_joint_ids, world.steering_joint_ids, world.drive_joint_ids):
        for joint_id in joint_group:
            qadr = world.model.jnt_qposadr[joint_id]
            dadr = world.model.jnt_dofadr[joint_id]
            world.data.qpos[qadr] = 0.0
            world.data.qvel[dadr] = 0.0
    mujoco.mj_forward(world.model, world.data)


def _append_path_overlay(
    scene: mujoco.MjvScene,
    data: ReplayData,
    current_idx: int,
    *,
    path_color: np.ndarray = PATH_COLOR,
    teleport_color: np.ndarray = TELEPORT_COLOR,
) -> None:
    if current_idx <= 0:
        return
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    zero = np.zeros(3, dtype=np.float64)
    max_idx = min(int(current_idx), len(data.frames) - 1)
    for idx in range(1, max_idx + 1):
        if scene.ngeom >= scene.maxgeom:
            break
        prev_pos = data.frames[idx - 1].origin_cm.astype(np.float64) / 100.0
        curr_pos = data.frames[idx].origin_cm.astype(np.float64) / 100.0
        color = teleport_color if int(data.frames[idx].teleport_flag) > 0 else path_color
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.ones(3, dtype=np.float64),
            zero,
            identity,
            color.astype(np.float64),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            3.0 if int(data.frames[idx].teleport_flag) > 0 else 2.0,
            prev_pos,
            curr_pos,
        )
        scene.ngeom += 1


class ReplayControls:
    def __init__(self) -> None:
        self.camera_distance = CAMERA_DISTANCE_M
        self.playing = True
        self.step_once = 0
        self.jump_to_start = False
        self.jump_to_end = False

    def on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        del scancode, mods
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.playing = not self.playing
        elif key == glfw.KEY_RIGHT:
            self.step_once += 1
            self.playing = False
        elif key == glfw.KEY_LEFT:
            self.step_once -= 1
            self.playing = False
        elif key == glfw.KEY_HOME:
            self.jump_to_start = True
            self.playing = False
        elif key == glfw.KEY_END:
            self.jump_to_end = True
            self.playing = False
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD, glfw.KEY_PAGE_UP):
            self.camera_distance = max(4.0, self.camera_distance - 0.8)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT, glfw.KEY_PAGE_DOWN):
            self.camera_distance = min(60.0, self.camera_distance + 0.8)

    def on_scroll(self, window, xoffset: float, yoffset: float) -> None:
        del window, xoffset
        self.camera_distance = float(np.clip(self.camera_distance - 0.6 * yoffset, 4.0, 60.0))


def _status_left(data: ReplayData, idx: int, predicted_path: ReplayData | None) -> str:
    frame = data.frames[idx]
    predicted_line = f"\npredicted={predicted_path.path.name}" if predicted_path is not None else "\npredicted=off"
    return (
        f"{data.path.name}\n"
        f"frame {idx + 1:03d}/{len(data.frames):03d} timestep={frame.timestep:03d}\n"
        f"x={frame.origin_cm[0]:.0f} y={frame.origin_cm[1]:.0f} z={frame.origin_cm[2]:.0f} yaw={frame.yaw_deg:.1f}"
        f"{predicted_line}"
    )


def _status_right(data: ReplayData, idx: int, controls: ReplayControls, step_seconds: float) -> str:
    frame = data.frames[idx]
    teleports_seen = int(sum(int(f.teleport_flag > 0) for f in data.frames[: idx + 1]))
    return (
        f"teleport={'yes' if frame.teleport_flag > 0 else 'no'} total_seen={teleports_seen}\n"
        f"play={'on' if controls.playing else 'off'} dt={step_seconds:.2f}s zoom={controls.camera_distance:.1f}m\n"
        "Space play/pause  Left/Right step  Home/End jump  PgUp/PgDn zoom"
    )


def replay(data: ReplayData, step_seconds: float, predicted_path: ReplayData | None = None) -> None:
    controls = ReplayControls()
    window = _create_window()
    glfw.set_key_callback(window, controls.on_key)
    glfw.set_scroll_callback(window, controls.on_scroll)

    world_cfg = _flat_replay_world_config()
    sim_cfg = SimConfig()
    try:
        with MujocoRoverWorld(gui=True, world_cfg=world_cfg, sim_cfg=sim_cfg) as world:
            cam = mujoco.MjvCamera()
            opt = mujoco.MjvOption()
            pert = mujoco.MjvPerturb()
            scene = mujoco.MjvScene(world.model, maxgeom=SCENE_MAX_GEOMS)
            context = mujoco.MjrContext(world.model, int(mujoco.mjtFontScale.mjFONTSCALE_150))
            if world.consume_hfield_render_dirty():
                mujoco.mjr_uploadHField(world.model, context, 0)

            idx = 0
            last_advance = time.perf_counter()
            while not glfw.window_should_close(window):
                glfw.poll_events()
                if controls.jump_to_start:
                    idx = 0
                    controls.jump_to_start = False
                if controls.jump_to_end:
                    idx = len(data.frames) - 1
                    controls.jump_to_end = False
                if controls.step_once != 0:
                    idx = max(0, min(len(data.frames) - 1, idx + int(np.sign(controls.step_once))))
                    controls.step_once -= int(np.sign(controls.step_once))

                now = time.perf_counter()
                if controls.playing and (now - last_advance) >= step_seconds:
                    if idx < len(data.frames) - 1:
                        idx += 1
                    last_advance = now

                _set_world_pose(world, data.frames[idx])
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

                width, height = glfw.get_framebuffer_size(window)
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjv_updateScene(
                    world.model,
                    world.data,
                    opt,
                    pert,
                    cam,
                    int(mujoco.mjtCatBit.mjCAT_ALL),
                    scene,
                )
                _append_path_overlay(scene, data, idx)
                if predicted_path is not None:
                    _append_path_overlay(
                        scene,
                        predicted_path,
                        len(predicted_path.frames) - 1,
                        path_color=PREDICTED_PATH_COLOR,
                        teleport_color=PREDICTED_PATH_COLOR,
                    )
                mujoco.mjr_render(viewport, scene, context)
                mujoco.mjr_overlay(
                    int(mujoco.mjtFontScale.mjFONTSCALE_150),
                    int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                    viewport,
                    _status_left(data, idx, predicted_path),
                    _status_right(data, idx, controls, step_seconds),
                    context,
                )
                glfw.swap_buffers(window)

            context.free()
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a random rover data file in the MuJoCo viewer.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--file", type=Path, default=None, help="Optional specific data file to replay.")
    parser.add_argument(
        "--predicted-path",
        type=Path,
        default=DEFAULT_PREDICTED_PATH,
        help="Optional predicted path file to overlay. Use 'none' or 'off' to disable it.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for random file selection.")
    parser.add_argument("--step-seconds", type=float, default=PLAYBACK_STEP_SECONDS)
    args = parser.parse_args()

    target = args.file if args.file is not None else pick_random_file(args.data_dir, seed=args.seed)
    if not target.is_absolute():
        target = target if target.exists() else args.data_dir / target
    data = load_replay_file(target.resolve())
    predicted_path: ReplayData | None = None
    if args.predicted_path is not None:
        predicted_target = args.predicted_path
        if str(predicted_target).strip().lower() not in DISABLED_PATH_VALUES:
            if not predicted_target.is_absolute():
                predicted_target = predicted_target if predicted_target.exists() else args.data_dir / predicted_target
            if predicted_target.exists():
                predicted_path = load_replay_file(predicted_target.resolve())
    replay(data, step_seconds=float(max(args.step_seconds, 0.01)), predicted_path=predicted_path)


if __name__ == "__main__":
    main()
