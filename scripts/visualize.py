#!/usr/bin/env python3
"""Visualize retargeted .pkl motions alongside their robot.xml in MuJoCo.

Usage
-----
# Single robot folder:
python scripts/visualize.py outputs/retargeting/robot_000

# Parent folder (plays all robot_XXX sub-folders sequentially):
python scripts/visualize.py outputs/retargeting

# Extra flags:
python scripts/visualize.py outputs/retargeting --loop --record-video
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import List

import mujoco
import mujoco.viewer as mjv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Body-name candidates used to pick the camera look-at target.
# ──────────────────────────────────────────────────────────────────────────────
_BASE_BODY_CANDIDATES = [
    "pelvis",
    "base_link",
    "base",
    "Waist",
    "waist_link",
    "trunk",
    "Trunk",
    "torso",
]


class RobotMotionViewer:
    """MuJoCo passive viewer for retargeted motion playback."""

    def __init__(
        self,
        xml_path: Path,
        motion_fps: float = 30.0,
        camera_follow: bool = True,
        cam_distance: float = 2.5,
        cam_elevation: float = -10.0,
        record_video: bool = False,
        video_path: str | None = None,
        video_width: int = 1280,
        video_height: int = 720,
    ) -> None:
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        self.motion_fps = motion_fps
        self.camera_follow = camera_follow
        self.cam_distance = cam_distance
        self.cam_elevation = cam_elevation
        self.record_video = record_video

        self._base_body_id = self._find_base_body()

        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )

        if self.record_video:
            import imageio  # type: ignore[import]
            assert video_path is not None, "Provide video_path when record_video=True."
            self.video_path = video_path
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            self.mp4_writer = imageio.get_writer(video_path, fps=self.motion_fps)
            self.renderer = mujoco.Renderer(self.model, height=video_height, width=video_width)
            print(f"[Viewer] Recording to {video_path}")

    # ------------------------------------------------------------------
    def _find_base_body(self) -> int:
        for name in _BASE_BODY_CANDIDATES:
            try:
                return self.model.body(name).id
            except (KeyError, mujoco.FatalError):
                continue
        # Fallback: first non-world body.
        return 1 if self.model.nbody > 1 else 0

    # ------------------------------------------------------------------
    def step(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof_pos: np.ndarray,
    ) -> None:
        """Advance the viewer by one frame.

        Parameters
        ----------
        root_pos:       (3,) root translation in world frame.
        root_rot_wxyz:  (4,) root quaternion in MuJoCo wxyz convention.
        dof_pos:        (njoints,) joint positions.
        """
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot_wxyz
        n_dof = min(dof_pos.shape[0], self.data.qpos.shape[0] - 7)
        self.data.qpos[7 : 7 + n_dof] = dof_pos[:n_dof]

        mujoco.mj_forward(self.model, self.data)

        if self.camera_follow:
            self.viewer.cam.lookat[:] = self.data.xpos[self._base_body_id]
            self.viewer.cam.distance = self.cam_distance
            self.viewer.cam.elevation = self.cam_elevation

        self.viewer.sync()

        if self.record_video:
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)

    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self) -> None:
        self.viewer.close()
        time.sleep(0.3)
        if self.record_video:
            self.mp4_writer.close()
            print(f"[Viewer] Video saved to {self.video_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_motion(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _xyzw_to_wxyz(rot_xyzw: np.ndarray) -> np.ndarray:
    """Convert (T, 4) xyzw quaternion array to (T, 4) wxyz."""
    return np.concatenate([rot_xyzw[:, 3:4], rot_xyzw[:, :3]], axis=-1)


def _discover_robot_folders(base: Path) -> List[Path]:
    """Return a sorted list of folders that contain robot.xml + at least one .pkl."""
    if (base / "robot.xml").exists() and list(base.glob("*.pkl")):
        return [base]
    candidates = sorted(p.parent for p in base.glob("*/robot.xml"))
    valid = [p for p in candidates if list(p.glob("*.pkl"))]
    if not valid:
        raise FileNotFoundError(
            f"No folder with both robot.xml and a .pkl found under: {base}"
        )
    return valid


# ──────────────────────────────────────────────────────────────────────────────
# Playback
# ──────────────────────────────────────────────────────────────────────────────

def play_folder(robot_folder: Path, args: argparse.Namespace) -> None:
    xml_path = robot_folder / "robot.xml"
    pkl_files = sorted(robot_folder.glob("*.pkl"))
    pkl_path = pkl_files[0]

    print(f"\n[Viewer] ── {robot_folder.name} ──")
    print(f"         XML : {xml_path}")
    print(f"         PKL : {pkl_path}")

    motion = _load_motion(pkl_path)
    fps: float = float(motion["fps"])
    root_pos: np.ndarray = np.asarray(motion["root_pos"])        # (T, 3)
    root_rot_xyzw: np.ndarray = np.asarray(motion["root_rot"])   # (T, 4) xyzw
    dof_pos: np.ndarray = np.asarray(motion["dof_pos"])          # (T, njoints)

    # retargeting.py saves root_rot as xyzw; MuJoCo qpos expects wxyz.
    root_rot_wxyz = _xyzw_to_wxyz(root_rot_xyzw)

    T = root_pos.shape[0]
    dt = 1.0 / fps

    video_path: str | None = None
    if args.record_video:
        video_path = str(robot_folder / f"{pkl_path.stem}.mp4")

    viewer = RobotMotionViewer(
        xml_path=xml_path,
        motion_fps=fps,
        camera_follow=args.camera_follow,
        cam_distance=args.cam_distance,
        cam_elevation=args.cam_elevation,
        record_video=args.record_video,
        video_path=video_path,
        video_width=args.video_width,
        video_height=args.video_height,
    )

    print(f"[Viewer] Playing {T} frames @ {fps:.2f} fps  (press Esc / close window to stop)")

    try:
        for i in range(T):
            if not viewer.is_running():
                break
            t0 = time.perf_counter()
            viewer.step(
                root_pos=root_pos[i],
                root_rot_wxyz=root_rot_wxyz[i],
                dof_pos=dof_pos[i],
            )
            elapsed = time.perf_counter() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)
    finally:
        viewer.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize retargeted .pkl motions in MuJoCo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help=(
            "Robot folder containing robot.xml + .pkl, "
            "or a parent directory with robot_XXX/ sub-folders."
        ),
    )
    parser.add_argument(
        "--no-camera-follow",
        dest="camera_follow",
        action="store_false",
        default=True,
        help="Disable camera tracking of the robot base.",
    )
    parser.add_argument(
        "--cam-distance",
        type=float,
        default=2.5,
        help="Camera distance from the tracked body.",
    )
    parser.add_argument(
        "--cam-elevation",
        type=float,
        default=-10.0,
        help="Camera elevation angle (degrees).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Loop each motion clip indefinitely.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=False,
        help="Save an .mp4 alongside the .pkl for each robot.",
    )
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.folder).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Path not found: {base_path}")

    folders = _discover_robot_folders(base_path)
    print(f"[Viewer] Found {len(folders)} robot folder(s) to visualize.")

    for robot_folder in folders:
        if args.loop:
            while True:
                play_folder(robot_folder, args)
        else:
            play_folder(robot_folder, args)

    print("[Viewer] Done.")


if __name__ == "__main__":
    main()
