"""Full evaluation pipeline: human motion → random robots → retargeting → scoring → grid visualization.

Usage examples (run from project root):

  # Minimal run: 4 robots, default assets
  python scripts/evaluation.py --motion-npz data/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz

  # Custom number of robots with visualization
  python scripts/evaluation.py \
      --motion-npz data/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz \
      --num-robots 6 --visualize-grid

  # Grid view with wider spacing and phase-stagger so every robot shows a different pose
  python scripts/evaluation.py \
      --motion-npz data/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz \
      --num-robots 9 --visualize-grid --grid-spacing 3.0 --grid-phase-offset 0.5 --grid-loop

  # Re-use already generated robots (skip retargeting) and jump straight to the grid
  python scripts/evaluation.py \
      --motion-npz data/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz \
      --num-robots 4 --keep-existing --visualize-grid

  # Only show ranked single-robot visualizations (legacy mode)
  python scripts/evaluation.py \
      --motion-npz data/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz \
      --visualize-ranked --visualize-top-k 3 --visualize-worst-k 1

"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import mujoco.viewer as mjv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import retargeting as rt  # noqa: E402

from optimal_embodiment.eval import robot2human as r2h  # noqa: E402
from optimal_embodiment.smpl import  load_human_motion_frames
from optimal_embodiment.robot.utils import build_random_robot_xml, generate_ik_config


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_seed_list(seed_list: Optional[str]) -> Optional[List[int]]:
    if seed_list is None:
        return None
    raw = [x.strip() for x in seed_list.split(",")]
    seeds = [int(x) for x in raw if x]
    if not seeds:
        raise ValueError("`--robot-seeds` was provided but ended up empty.")
    return seeds


def _build_seed_plan(seed: int, num_robots: int, seed_list: Optional[str]) -> List[int]:
    parsed = _parse_seed_list(seed_list)
    if parsed is not None:
        return parsed
    if num_robots <= 0:
        raise ValueError("`--num-robots` must be > 0.")
    return [int(seed) + i for i in range(int(num_robots))]


def _fmt(x: float, d: int = 3) -> str:
    try:
        if x != x:
            return "nan"
    except Exception:
        return "nan"
    return f"{x:.{d}f}"


def _robot_dirs_with_scores(
    results: Sequence[Tuple[int, r2h.RobotEval]],
    top_k: int,
    worst_k: int,
) -> List[Tuple[int, r2h.RobotEval]]:
    if not results:
        return []
    ordered = sorted(results, key=lambda x: x[1].overall_score, reverse=True)
    selected: List[Tuple[int, r2h.RobotEval]] = []
    seen: set[str] = set()

    for pair in ordered[: max(0, top_k)]:
        if pair[1].robot_dir not in seen:
            selected.append(pair)
            seen.add(pair[1].robot_dir)
    if worst_k > 0:
        for pair in reversed(ordered[-worst_k:]):
            if pair[1].robot_dir not in seen:
                selected.append(pair)
                seen.add(pair[1].robot_dir)
    return selected


def _save_per_robot_score(robot_dir: Path, seed: int, result: r2h.RobotEval) -> None:
    payload = {
        "seed": seed,
        "result": asdict(result),
    }
    out = robot_dir / "score.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Grid visualization
# ──────────────────────────────────────────────────────────────────────────────

def _grid_positions(n: int, spacing: float) -> List[Tuple[float, float]]:
    """Return centered XY positions for n robots arranged in the tightest square grid."""
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    positions: List[Tuple[float, float]] = []
    x_offset = (cols - 1) * spacing / 2.0
    y_offset = (rows - 1) * spacing / 2.0
    for i in range(n):
        row = i // cols
        col = i % cols
        positions.append((col * spacing - x_offset, row * spacing - y_offset))
    return positions


def _load_motion_pkl(pkl_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load a retargeted motion PKL and return (root_pos, root_rot_wxyz, dof_pos, fps)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    fps: float = float(data["fps"])
    root_pos: np.ndarray = np.asarray(data["root_pos"], dtype=np.float64)     # (T, 3)
    root_rot_xyzw: np.ndarray = np.asarray(data["root_rot"], dtype=np.float64) # (T, 4) xyzw
    dof_pos: np.ndarray = np.asarray(data["dof_pos"], dtype=np.float64)         # (T, N)
    # MuJoCo qpos stores quaternion as wxyz; PKL stores xyzw
    root_rot_wxyz = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1)
    return root_pos, root_rot_wxyz, dof_pos, fps


def _build_combined_grid_model(
    robot_xml_paths: List[Path],
    positions: List[Tuple[float, float]],
) -> Tuple[mujoco.MjModel, List[int]]:
    """Merge all robots into one MuJoCo scene arranged at the given XY positions.

    Returns (combined_model, free_qpos_adrs) where free_qpos_adrs[i] is the
    qpos address of robot i's free joint in the merged model.
    """
    spec = mujoco.MjSpec()

    # Ground plane (scene floor) with checkerboard texture
    tex = spec.add_texture()
    tex.name = "texplane"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1 = [0.2, 0.3, 0.4]
    tex.rgb2 = [0.1, 0.2, 0.3]
    tex.width = 512
    tex.height = 512

    mat = spec.add_material()
    mat.name = "matplane"
    mat.textures = ["texplane"] + [""] * 9  # exactly 10 slots; empty for unused
    mat.texrepeat = [4.0, 4.0]
    mat.texuniform = True

    floor = spec.worldbody.add_geom()
    floor.name = "grid_floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.material = "matplane"

    # Attach each robot with a unique prefix and a frame at TQits grid position
    for i, (xml_path, (gx, gy)) in enumerate(zip(robot_xml_paths, positions)):
        child = mujoco.MjSpec.from_file(str(xml_path))
        frame = spec.worldbody.add_frame()
        frame.pos = [gx, gy, 0.0]
        spec.attach(child, prefix=f"r{i}_", suffix="", frame=frame)

    model = spec.compile()

    # Collect the qposadr for each robot's free joint (type 0 = mjJNT_FREE)
    free_qpos_adrs: List[int] = []
    for i in range(len(robot_xml_paths)):
        prefix = f"r{i}_"
        adr = -1
        for j in range(model.njnt):
            if model.jnt_type[j] != 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.startswith(prefix):
                adr = int(model.jnt_qposadr[j])
                break
        if adr == -1:
            raise RuntimeError(
                f"Could not find free joint for robot {i} (prefix={prefix!r}). "
                "Make sure the robot XML has a floating-base free joint."
            )
        free_qpos_adrs.append(adr)

    return model, free_qpos_adrs


def visualize_robots_grid(
    robot_dirs: List[Path],
    motion_stem: str,
    spacing: float = 2.5,
    phase_offset: float = 0.5,
    loop: bool = False,
    cam_distance: float = 8.0,
    cam_elevation: float = -15.0,
) -> None:
    """Display all robots simultaneously in a single MuJoCo environment.

    Robots are arranged in a square grid, each playing the retargeted motion
    at a different phase so they appear to perform diverse movements at any
    given instant.

    Parameters
    ----------
    robot_dirs:    Directories that each contain robot.xml and <motion_stem>.pkl.
    motion_stem:   Stem of the PKL file (e.g. 'Extended_1_stageii').
    spacing:       Distance in metres between adjacent robots in the grid.
    phase_offset:  Fraction of total frames to stagger consecutive robots
                   (0.0 = all in sync, 0.5 = half-cycle apart).
    loop:          Whether to loop the animation indefinitely.
    cam_distance:  Passive viewer camera distance.
    cam_elevation: Passive viewer camera elevation in degrees.
    """
    n = len(robot_dirs)
    if n == 0:
        print("      [grid] No robot directories provided — skipping.")
        return

    print(f"\n[Grid] Building combined scene for {n} robot(s) | spacing={spacing}m")

    xml_paths = [d / "robot.xml" for d in robot_dirs]
    pkl_paths = [d / f"{motion_stem}.pkl" for d in robot_dirs]

    for p in xml_paths + pkl_paths:
        if not p.is_file():
            raise FileNotFoundError(f"[Grid] Required file not found: {p}")

    positions = _grid_positions(n, spacing)
    model, free_qpos_adrs = _build_combined_grid_model(xml_paths, positions)
    data = mujoco.MjData(model)

    # Load all motions
    motions: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for pkl in pkl_paths:
        motions.append(_load_motion_pkl(pkl))

    fps = motions[0][3]
    T = motions[0][0].shape[0]

    # Phase offsets: robot i starts at frame (i * phase_frames) % T
    phase_frames = [round(i * phase_offset * T) % T for i in range(n)]
    print(f"[Grid] T={T} frames | fps={fps:.1f} | phase offsets={phase_frames}")
    print(f"[Grid] Grid layout: {[f'({p[0]:.1f},{p[1]:.1f})' for p in positions]}")
    print("[Grid] Press Esc or close the window to stop.")

    viewer = mjv.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False)

    viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
    viewer.cam.distance = cam_distance
    viewer.cam.elevation = cam_elevation
    viewer.cam.azimuth = 45.0

    # Pre-compute the full qpos buffer for every frame so the render loop only
    # does a single numpy copy + mj_kinematics per frame.
    print("[Grid] Pre-computing qpos buffer…", end=" ", flush=True)
    nq = model.nq
    all_frames = np.arange(T)
    qpos_buf = np.empty((T, nq), dtype=np.float64)
    qpos_buf[:] = data.qpos[np.newaxis, :]

    for i, (root_pos, root_rot_wxyz, dof_pos, _) in enumerate(motions):
        adr = free_qpos_adrs[i]
        gx, gy = positions[i]
        pos0 = root_pos[0]
        frame_indices = (all_frames + phase_frames[i]) % T

        rp = root_pos[frame_indices]
        qpos_buf[:, adr]     = gx + (rp[:, 0] - pos0[0])
        qpos_buf[:, adr + 1] = gy + (rp[:, 1] - pos0[1])
        qpos_buf[:, adr + 2] = rp[:, 2]
        qpos_buf[:, adr + 3:adr + 7] = root_rot_wxyz[frame_indices]

        n_dof = min(dof_pos.shape[1], nq - adr - 7)
        qpos_buf[:, adr + 7:adr + 7 + n_dof] = dof_pos[frame_indices, :n_dof]

    print("done.")

    mujoco.mj_kinematics(model, data)

    # Drive playback from wall-clock time so the animation always stays in sync
    # with real time. If a frame renders slowly we skip ahead to the correct
    # position instead of falling behind, which is what caused the lag.
    duration = T / fps
    start_wall = time.perf_counter()

    try:
        while viewer.is_running():
            elapsed = time.perf_counter() - start_wall

            if not loop and elapsed >= duration:
                break

            t = int((elapsed % duration) * fps)
            t = min(t, T - 1)

            data.qpos[:] = qpos_buf[t]
            mujoco.mj_kinematics(model, data)
            viewer.sync()

    finally:
        viewer.close()
        time.sleep(0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full pipeline: human npz -> random robots -> retargeting -> scoring -> "
            "grid visualization in MuJoCo."
        )
    )
    parser.add_argument("--motion-npz", type=str, required=True, help="Human motion .npz (AMASS/SMPLH).")
    parser.add_argument(
        "--body-models-dir",
        type=str,
        default=str(ROOT / "data"),
        help="Directory with body_models/ (SMPLH + DMPLs).",
    )
    parser.add_argument(
        "--template-xml",
        type=str,
        default=str(ROOT / "assets" / "g1_29dof" / "g1_29dof.xml"),
        help="Base XML for randomization.",
    )
    parser.add_argument(
        "--template-ik",
        type=str,
        default=str(
            ROOT
            / "third_party"
            / "gmr"
            / "general_motion_retargeting"
            / "ik_configs"
            / "smplx_to_g1.json"
        ),
        help="Reference IK to generate dynamic IK per robot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "full_pipeline_eval"),
        help="Output folder.",
    )
    parser.add_argument(
        "--motion-stem",
        type=str,
        default=None,
        help="Name of pkl without extension (default: npz stem).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed (if not using --robot-seeds).")
    parser.add_argument("--num-robots", type=int, default=4, help="Number of robots to generate and evaluate.")
    parser.add_argument(
        "--robot-seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds instead of --num-robots. Example: 1,7,42,105",
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--disable-head-task", action="store_true", default=False)
    parser.add_argument("--disable-head-joints", action="store_true", default=False)
    parser.add_argument("--enable-dtw", action="store_true", default=False, help="Include DTW in score (slower).")
    parser.add_argument("--max-dtw-frames", type=int, default=600)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        default=False,
        help="If robot.xml/IK/PKL already exists for a robot, do not regenerate.",
    )

    # ── legacy per-robot visualization ──────────────────────────────────────
    parser.add_argument(
        "--visualize-ranked",
        action="store_true",
        default=False,
        help="Play back ranked robots one by one (original human + robot, sequential windows).",
    )
    parser.add_argument("--visualize-top-k", type=int, default=3, help="Top-K to visualize (--visualize-ranked).")
    parser.add_argument("--visualize-worst-k", type=int, default=1, help="Worst-K to visualize (--visualize-ranked).")
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        default=False,
        help="Skip real-time FPS throttle in legacy per-robot visualization.",
    )
    parser.add_argument("--summary-json", type=str, default=None, help="Path of final JSON summary.")

    # ── grid visualization ───────────────────────────────────────────────────
    parser.add_argument(
        "--visualize-grid",
        action="store_true",
        default=False,
        help="After the pipeline, show ALL robots together in one MuJoCo scene arranged in a grid.",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=2.5,
        help="Distance in metres between adjacent robots in the grid (default: 2.5).",
    )
    parser.add_argument(
        "--grid-phase-offset",
        type=float,
        default=0.5,
        help=(
            "Fraction of the total motion to stagger each consecutive robot's playback "
            "(0 = all in sync, 0.5 = half-cycle apart). Default: 0.5."
        ),
    )
    parser.add_argument(
        "--grid-loop",
        action="store_true",
        default=False,
        help="Loop the grid animation indefinitely (default: play once and exit).",
    )
    parser.add_argument(
        "--grid-cam-distance",
        type=float,
        default=8.0,
        help="Camera distance for the grid viewer (default: 8.0).",
    )
    parser.add_argument(
        "--grid-cam-elevation",
        type=float,
        default=-15.0,
        help="Camera elevation for the grid viewer in degrees (default: -15).",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    motion_npz = Path(args.motion_npz).resolve()
    body_models_dir = Path(args.body_models_dir).resolve()
    template_xml = Path(args.template_xml).resolve()
    template_ik = Path(args.template_ik).resolve()
    output_dir = Path(args.output_dir).resolve()
    motion_stem = args.motion_stem if args.motion_stem else motion_npz.stem
    summary_json = Path(args.summary_json).resolve() if args.summary_json else output_dir / "summary.json"

    if not motion_npz.is_file():
        raise FileNotFoundError(f"Motion npz does not exist: {motion_npz}")
    if not template_xml.is_file():
        raise FileNotFoundError(f"Template xml does not exist: {template_xml}")
    if not template_ik.is_file():
        raise FileNotFoundError(f"Template ik does not exist: {template_ik}")

    seeds = _build_seed_plan(seed=int(args.seed), num_robots=int(args.num_robots), seed_list=args.robot_seeds)
    print(f"Selected seeds ({len(seeds)}): {seeds}")

    print("[1/5] Loading human motion...")
    human_frames, aligned_fps, human_height = load_human_motion_frames(
        npz_path=motion_npz,
        body_models_dir=body_models_dir,
        target_fps=float(args.target_fps),
    )
    print(
        f"      frames={len(human_frames)} | aligned_fps={_fmt(aligned_fps, 4)} | "
        f"height_est={_fmt(human_height, 3)} m"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    human_frames_cache: Dict[float, List[Dict[str, Tuple[object, object]]]] = {float(aligned_fps): human_frames}  # type: ignore[assignment]
    eval_results: List[Tuple[int, r2h.RobotEval]] = []
    failures: List[Dict[str, str]] = []

    print("[2/5] Generating robots + dynamic IK + retargeting + scoring...")
    for idx, seed in enumerate(seeds):
        robot_dir = output_dir / f"robot_{idx:03d}"
        xml_path = robot_dir / "robot.xml"
        ik_path = robot_dir / "smplx_to_robot.json"
        pkl_path = robot_dir / f"{motion_stem}.pkl"
        metadata_path = robot_dir / "metadata.json"
        robot_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx + 1}/{len(seeds)}] {robot_dir.name} | seed={seed}")
        try:
            generate_needed = not bool(args.keep_existing and xml_path.is_file() and ik_path.is_file())
            if generate_needed:
                model = build_random_robot_xml(
                    output_xml_path=xml_path,
                    template_xml=template_xml,
                    seed=int(seed),
                    add_head_joints=not bool(args.disable_head_joints),
                )
                ik_cfg = generate_ik_config(
                    model=model,
                    output_path=ik_path,
                    template_ik_path=template_ik,
                    include_head_task=not bool(args.disable_head_task),
                )
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "seed": int(seed),
                            "robot_dir": str(robot_dir),
                            "nbody": int(model.nbody),
                            "ik1_size": int(len(ik_cfg["ik_match_table1"])),
                            "ik2_size": int(len(ik_cfg["ik_match_table2"])),
                        },
                        f,
                        indent=2,
                    )
                print(
                    f"      generated: nbody={model.nbody}, "
                    f"ik1={len(ik_cfg['ik_match_table1'])}, ik2={len(ik_cfg['ik_match_table2'])}"
                )
            else:
                print("      using existing robot.xml + IK")

            retarget_needed = not bool(args.keep_existing and pkl_path.is_file())
            if retarget_needed:
                robot_key = f"random_humanoid_eval_{int(time.time())}_{idx}"
                rt.retarget_motion(
                    robot_key=robot_key,
                    robot_xml_path=xml_path,
                    ik_config_path=ik_path,
                    human_frames=human_frames,
                    human_height=float(human_height),
                    aligned_fps=float(aligned_fps),
                    save_path=pkl_path,
                    visualize=False,
                    rate_limit=False,
                    record_video=False,
                    video_path=None,
                )
                print(f"      retarget OK -> {pkl_path.name}")
            else:
                print("      using existing pkl")

            result = r2h.evaluate_robot(
                motion_npz=motion_npz,
                body_models_dir=body_models_dir,
                robot_dir=robot_dir,
                motion_stem=motion_stem,
                human_frames_cache=human_frames_cache,  # type: ignore[arg-type]
                enable_dtw=bool(args.enable_dtw),
                max_dtw_frames=int(args.max_dtw_frames),
            )
            eval_results.append((seed, result))
            _save_per_robot_score(robot_dir=robot_dir, seed=seed, result=result)
            print(
                f"      score: overall={_fmt(result.overall_score, 2)} | "
                f"imit={_fmt(result.imitation_score, 2)} | quality={_fmt(result.quality_score, 2)} | "
                f"mpjpe={_fmt(result.mpjpe_cm, 2)}cm"
            )
        except Exception as exc:
            failures.append({"robot_dir": str(robot_dir), "seed": str(seed), "error": str(exc)})
            print(f"      [FAIL] {exc}")

    if not eval_results:
        raise RuntimeError("No valid score was obtained.")

    print("\n[3/5] Final ranking")
    ranked = sorted(eval_results, key=lambda x: x[1].overall_score, reverse=True)
    for rank_idx, (seed, res) in enumerate(ranked, start=1):
        print(
            f"{rank_idx:02d}. {Path(res.robot_dir).name} (seed={seed}) | "
            f"overall={_fmt(res.overall_score, 2)} | "
            f"mpjpe={_fmt(res.mpjpe_cm, 2)}cm | pck10={_fmt(res.pck_10cm, 1)}%"
        )

    print("[4/5] Saving summary...")
    payload = {
        "motion_npz": str(motion_npz),
        "body_models_dir": str(body_models_dir),
        "template_xml": str(template_xml),
        "template_ik": str(template_ik),
        "output_dir": str(output_dir),
        "motion_stem": motion_stem,
        "target_fps_requested": float(args.target_fps),
        "aligned_fps": float(aligned_fps),
        "human_height_est_m": float(human_height),
        "seeds_requested": seeds,
        "num_success": len(ranked),
        "num_failures": len(failures),
        "results_ranked": [
            {
                "seed": int(seed),
                **asdict(res),
            }
            for seed, res in ranked
        ],
        "failures": failures,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"      -> {summary_json}")

    print("[5/5] Visualization")

    # ── Grid: all robots in one shared MuJoCo environment ───────────────────
    if args.visualize_grid:
        successful_dirs = [Path(res.robot_dir) for _, res in ranked]
        visualize_robots_grid(
            robot_dirs=successful_dirs,
            motion_stem=motion_stem,
            spacing=float(args.grid_spacing),
            phase_offset=float(args.grid_phase_offset),
            loop=bool(args.grid_loop),
            cam_distance=float(args.grid_cam_distance),
            cam_elevation=float(args.grid_cam_elevation),
        )

    # ── Legacy: ranked single-robot playback ────────────────────────────────
    if args.visualize_ranked:
        to_visualize = _robot_dirs_with_scores(
            results=ranked,
            top_k=max(0, int(args.visualize_top_k)),
            worst_k=max(0, int(args.visualize_worst_k)),
        )
        if not to_visualize:
            print("      no robots selected for visualization")
        else:
            print(
                "      Original human + robot will be shown (sequential). "
                "Close the window to go to the next."
            )
            for vis_idx, (seed, res) in enumerate(to_visualize, start=1):
                robot_dir = Path(res.robot_dir)
                xml_path = Path(res.xml_path)
                ik_path = robot_dir / "smplx_to_robot.json"
                pkl_path = Path(res.motion_pkl)
                video_path = robot_dir / f"{motion_stem}.scored.mp4"

                print(
                    f"\n      [{vis_idx}/{len(to_visualize)}] {robot_dir.name} (seed={seed}) | "
                    f"overall={_fmt(res.overall_score, 2)} | mpjpe={_fmt(res.mpjpe_cm, 2)}cm"
                )
                robot_key = f"viz_robot_{vis_idx}_{int(time.time())}"
                rt.retarget_motion(
                    robot_key=robot_key,
                    robot_xml_path=xml_path,
                    ik_config_path=ik_path,
                    human_frames=human_frames,
                    human_height=float(human_height),
                    aligned_fps=float(aligned_fps),
                    save_path=pkl_path,
                    visualize=True,
                    rate_limit=not bool(args.no_rate_limit),
                    record_video=False,
                    video_path=None,
                )

    if not args.visualize_grid and not args.visualize_ranked:
        print("      skipped (use --visualize-grid or --visualize-ranked to enable)")

    print("\nDone.")


if __name__ == "__main__":
    main()
