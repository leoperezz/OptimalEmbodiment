from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import retargeting as rt  # noqa: E402


JOINT_ORDER: List[str] = [
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "spine3",
    "left_foot",
    "right_foot",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


JOINT_TO_TEMPLATE_FRAME: Dict[str, str] = {
    "pelvis": "pelvis",
    "left_hip": "left_hip_roll_link",
    "right_hip": "right_hip_roll_link",
    "left_knee": "left_knee_link",
    "right_knee": "right_knee_link",
    "spine3": "torso_link",
    "left_foot": "left_toe_link",
    "right_foot": "right_toe_link",
    "head": "head",
    "left_shoulder": "left_shoulder_yaw_link",
    "right_shoulder": "right_shoulder_yaw_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
}


@dataclass
class RobotEval:
    robot_dir: str
    motion_pkl: str
    xml_path: str
    fps: float
    frames_compared: int
    valid_joints: int
    valid_joint_names: List[str]
    unresolved_joint_names: List[str]
    mpjpe_cm: float
    pck_10cm: float
    vel_error_cm_s: float
    root_traj_rmse_cm: float
    dtw_pose_cm: float
    joint_limit_violation_rate: float
    dof_acc_mean: float
    foot_slip_cm_s: float
    imitation_score: float
    quality_score: float
    overall_score: float


def _load_robot_motion(pkl_path: Path) -> Dict[str, np.ndarray | float]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Formato inesperado en {pkl_path}: se esperaba dict.")

    required = ["root_pos", "root_rot", "dof_pos"]
    for key in required:
        if key not in data:
            raise KeyError(f"Falta '{key}' en {pkl_path}.")

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    fps = float(data.get("fps", 30.0))

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos inválido en {pkl_path}: {root_pos.shape}")
    if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
        raise ValueError(f"root_rot inválido en {pkl_path}: {root_rot_xyzw.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos inválido en {pkl_path}: {dof_pos.shape}")
    if not (root_pos.shape[0] == root_rot_xyzw.shape[0] == dof_pos.shape[0]):
        raise ValueError(
            f"Cantidad de frames inconsistente en {pkl_path}: "
            f"root_pos={root_pos.shape[0]}, root_rot={root_rot_xyzw.shape[0]}, dof_pos={dof_pos.shape[0]}"
        )

    return {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot_xyzw": root_rot_xyzw,
        "dof_pos": dof_pos,
    }


def _find_motion_pkl(robot_dir: Path, motion_stem: Optional[str]) -> Path:
    if motion_stem:
        p = robot_dir / f"{motion_stem}.pkl"
        if not p.is_file():
            raise FileNotFoundError(f"No existe motion pkl esperado: {p}")
        return p

    pkls = sorted(robot_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No se encontró ningún .pkl en {robot_dir}")
    if len(pkls) > 1:
        raise RuntimeError(
            f"Hay múltiples .pkl en {robot_dir}. Usa --motion-stem para indicar uno específico."
        )
    return pkls[0]


def _collect_robot_dirs(robot_dir: Optional[Path], retargeting_dir: Optional[Path]) -> List[Path]:
    if robot_dir is not None:
        if not robot_dir.is_dir():
            raise FileNotFoundError(f"No existe robot_dir: {robot_dir}")
        return [robot_dir]

    if retargeting_dir is None or not retargeting_dir.is_dir():
        raise FileNotFoundError(f"No existe retargeting_dir: {retargeting_dir}")

    dirs = sorted([p for p in retargeting_dir.iterdir() if p.is_dir() and p.name.startswith("robot_")])
    if not dirs:
        if (retargeting_dir / "robot.xml").is_file():
            return [retargeting_dir]
        raise RuntimeError(f"No se encontraron carpetas robot_* en {retargeting_dir}")
    return dirs


def _resolve_body_name(model: mujoco.MjModel, template_frame: str) -> Optional[str]:
    body_names = rt._body_names_from_model(model)  # type: ignore[attr-defined]
    if template_frame == "head":
        return rt._resolve_head_body(body_names)  # type: ignore[attr-defined]
    return rt._resolve_frame_name(template_frame, body_names)  # type: ignore[attr-defined]


def _extract_human_positions(
    human_frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    num_frames: int,
) -> np.ndarray:
    arr = np.full((num_frames, len(JOINT_ORDER), 3), np.nan, dtype=np.float64)
    for t in range(num_frames):
        frame = human_frames[t]
        for j, joint_name in enumerate(JOINT_ORDER):
            if joint_name in frame:
                arr[t, j, :] = np.asarray(frame[joint_name][0], dtype=np.float64)
    return arr


def _reconstruct_robot_tracks(
    model: mujoco.MjModel,
    motion: Dict[str, np.ndarray | float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Optional[str]], np.ndarray]:
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(motion["root_rot_xyzw"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)

    T = int(root_pos.shape[0])
    data = mujoco.MjData(model)

    qpos_full = np.zeros((T, model.nq), dtype=np.float64)
    joint_tracks = np.full((T, len(JOINT_ORDER), 3), np.nan, dtype=np.float64)

    resolved_names: Dict[str, Optional[str]] = {}
    body_ids = np.full((len(JOINT_ORDER),), -1, dtype=np.int32)
    for j, joint_name in enumerate(JOINT_ORDER):
        template_frame = JOINT_TO_TEMPLATE_FRAME[joint_name]
        body_name = _resolve_body_name(model, template_frame)
        resolved_names[joint_name] = body_name
        if body_name is not None:
            body_ids[j] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    dof_expected = max(0, model.nq - 7)
    dof_used = min(dof_expected, dof_pos.shape[1])

    for t in range(T):
        q = np.zeros((model.nq,), dtype=np.float64)
        q[:3] = root_pos[t]
        quat_xyzw = root_rot_xyzw[t]
        q[3:7] = quat_xyzw[[3, 0, 1, 2]]
        if dof_used > 0:
            q[7 : 7 + dof_used] = dof_pos[t, :dof_used]
        qpos_full[t] = q

        data.qpos[:] = q
        mujoco.mj_forward(model, data)

        for j, bid in enumerate(body_ids):
            if bid >= 0:
                joint_tracks[t, j, :] = data.xpos[bid]

    return joint_tracks, qpos_full, resolved_names, body_ids


def _scalar_procrustes_fit(src: np.ndarray, tgt: np.ndarray) -> Tuple[float, np.ndarray]:
    src_flat = src.reshape(-1, 3)
    tgt_flat = tgt.reshape(-1, 3)

    denom = float(np.sum(src_flat * src_flat))
    if denom < 1e-12:
        scale = 1.0
    else:
        scale = float(np.sum(src_flat * tgt_flat) / denom)
        scale = max(scale, 1e-6)

    scaled = src_flat * scale
    h = scaled.T @ tgt_flat
    u, _, vt = np.linalg.svd(h)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return scale, rot


def _dtw_pose_error_m(human_pose: np.ndarray, robot_pose: np.ndarray, max_frames: int) -> float:
    if human_pose.shape[0] == 0 or robot_pose.shape[0] == 0:
        return float("nan")

    step_h = max(1, math.ceil(human_pose.shape[0] / max_frames))
    step_r = max(1, math.ceil(robot_pose.shape[0] / max_frames))

    h = human_pose[::step_h].reshape(-1, human_pose.shape[1] * 3)
    r = robot_pose[::step_r].reshape(-1, robot_pose.shape[1] * 3)
    n = h.shape[0]
    m = r.shape[0]
    if n == 0 or m == 0:
        return float("nan")

    num_joints = max(1, human_pose.shape[1])
    norm_scale = math.sqrt(float(num_joints))

    prev = np.full((m + 1,), np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = np.full((m + 1,), np.inf, dtype=np.float64)
        hi = h[i - 1]
        for j in range(1, m + 1):
            cost = np.linalg.norm(hi - r[j - 1]) / norm_scale
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return float(prev[m] / (n + m))


def _joint_limit_violation_rate(model: mujoco.MjModel, qpos_full: np.ndarray) -> float:
    violations = 0
    total = 0
    for jid in range(model.njnt):
        joint_type = int(model.jnt_type[jid])
        if joint_type in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
            continue
        if int(model.jnt_limited[jid]) == 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        if qadr < 0 or qadr >= qpos_full.shape[1]:
            continue
        lo, hi = model.jnt_range[jid]
        vals = qpos_full[:, qadr]
        total += vals.size
        violations += int(np.count_nonzero((vals < lo) | (vals > hi)))
    if total == 0:
        return float("nan")
    return float(violations / total)


def _dof_acc_mean(dof_pos: np.ndarray, fps: float) -> float:
    if dof_pos.shape[0] < 3:
        return float("nan")
    dt = 1.0 / max(fps, 1e-8)
    acc = np.diff(dof_pos, n=2, axis=0) / (dt * dt)
    if acc.size == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(acc, axis=1)))


def _foot_slip_cm_s(robot_tracks: np.ndarray, fps: float) -> float:
    if robot_tracks.shape[0] < 2:
        return float("nan")
    dt = 1.0 / max(fps, 1e-8)
    foot_indices = [JOINT_ORDER.index("left_foot"), JOINT_ORDER.index("right_foot")]

    slips: List[float] = []
    for idx in foot_indices:
        foot = robot_tracks[:, idx, :]
        if not np.isfinite(foot).all():
            continue
        speed_xy = np.linalg.norm(np.diff(foot[:, :2], axis=0) / dt, axis=1)
        ground = float(np.percentile(foot[:, 2], 5.0))
        stance = foot[:-1, 2] <= (ground + 0.03)
        if np.any(stance):
            slips.append(float(np.mean(speed_xy[stance]) * 100.0))
    if not slips:
        return float("nan")
    return float(np.mean(slips))


def _safe_exp_score(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.exp(-value / max(scale, 1e-8)))


def _compute_scores(
    mpjpe_cm: float,
    pck_10cm: float,
    vel_error_cm_s: float,
    root_rmse_cm: float,
    dtw_pose_cm: float,
    joint_limit_violation_rate: float,
    dof_acc_mean: float,
    foot_slip_cm_s: float,
) -> Tuple[float, float, float]:
    imitation_score = (
        100.0
        * (
            0.45 * _safe_exp_score(mpjpe_cm, 20.0)
            + 0.20 * (pck_10cm / 100.0 if np.isfinite(pck_10cm) else 0.0)
            + 0.20 * _safe_exp_score(vel_error_cm_s, 80.0)
            + 0.15 * _safe_exp_score(root_rmse_cm, 30.0)
        )
    )
    if np.isfinite(dtw_pose_cm):
        imitation_score = 0.85 * imitation_score + 15.0 * _safe_exp_score(dtw_pose_cm, 25.0)

    quality_score = 100.0
    if np.isfinite(joint_limit_violation_rate):
        quality_score *= float(np.exp(-80.0 * joint_limit_violation_rate))
    if np.isfinite(foot_slip_cm_s):
        quality_score *= _safe_exp_score(foot_slip_cm_s, 30.0)
    if np.isfinite(dof_acc_mean):
        quality_score *= _safe_exp_score(dof_acc_mean, 300.0)

    overall_score = 0.80 * imitation_score + 0.20 * quality_score
    imitation_score = float(np.clip(imitation_score, 0.0, 100.0))
    quality_score = float(np.clip(quality_score, 0.0, 100.0))
    overall_score = float(np.clip(overall_score, 0.0, 100.0))
    return imitation_score, quality_score, overall_score


def evaluate_robot(
    motion_npz: Path,
    body_models_dir: Path,
    robot_dir: Path,
    motion_stem: Optional[str],
    human_frames_cache: Dict[float, List[Dict[str, Tuple[np.ndarray, np.ndarray]]]],
    enable_dtw: bool,
    max_dtw_frames: int,
) -> RobotEval:
    xml_path = robot_dir / "robot.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"No existe robot.xml en {robot_dir}")

    motion_pkl = _find_motion_pkl(robot_dir, motion_stem)
    motion = _load_robot_motion(motion_pkl)
    fps = float(motion["fps"])

    if fps not in human_frames_cache:
        human_frames, _, _ = rt.load_human_motion_frames(
            npz_path=motion_npz,
            body_models_dir=body_models_dir,
            target_fps=fps,
        )
        human_frames_cache[fps] = human_frames
    human_frames = human_frames_cache[fps]

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    robot_tracks, qpos_full, resolved_names, _ = _reconstruct_robot_tracks(model=model, motion=motion)

    T = min(len(human_frames), robot_tracks.shape[0])
    if T <= 0:
        raise RuntimeError(f"Sin frames comparables en {robot_dir}")

    human_tracks = _extract_human_positions(human_frames=human_frames, num_frames=T)
    robot_tracks = robot_tracks[:T]
    qpos_full = qpos_full[:T]
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)[:T]

    valid_joint_mask = np.isfinite(human_tracks).all(axis=(0, 2)) & np.isfinite(robot_tracks).all(axis=(0, 2))
    valid_joint_ids = np.where(valid_joint_mask)[0]
    if valid_joint_ids.size < 5:
        raise RuntimeError(
            f"Solo {valid_joint_ids.size} joints válidos en {robot_dir}. "
            f"Revisa mapeo IK/body names de ese robot."
        )

    human_valid = human_tracks[:, valid_joint_ids, :]
    robot_valid = robot_tracks[:, valid_joint_ids, :]

    human_center = human_valid - human_valid[:, :1, :]
    robot_center = robot_valid - robot_valid[:, :1, :]

    scale, rot = _scalar_procrustes_fit(src=robot_center, tgt=human_center)
    robot_aligned = np.einsum("tjd,dk->tjk", robot_center * scale, rot)

    per_joint_err_m = np.linalg.norm(human_center - robot_aligned, axis=-1)
    mpjpe_cm = float(np.mean(per_joint_err_m) * 100.0)
    pck_10cm = float(np.mean(per_joint_err_m < 0.10) * 100.0)

    if T > 1:
        dt = 1.0 / max(fps, 1e-8)
        v_h = np.diff(human_center, axis=0) / dt
        v_r = np.diff(robot_aligned, axis=0) / dt
        vel_error_cm_s = float(np.mean(np.linalg.norm(v_h - v_r, axis=-1)) * 100.0)
    else:
        vel_error_cm_s = float("nan")

    h_root = human_tracks[:, 0, :] - human_tracks[0:1, 0, :]
    r_root = robot_tracks[:, 0, :] - robot_tracks[0:1, 0, :]
    r_root_aligned = np.einsum("td,dk->tk", r_root * scale, rot)
    root_traj_rmse_cm = float(np.sqrt(np.mean(np.sum((h_root - r_root_aligned) ** 2, axis=-1))) * 100.0)

    if enable_dtw:
        dtw_pose_cm = float(_dtw_pose_error_m(human_center, robot_aligned, max_frames=max_dtw_frames) * 100.0)
    else:
        dtw_pose_cm = float("nan")

    joint_limit_violation_rate = _joint_limit_violation_rate(model=model, qpos_full=qpos_full)
    dof_acc_mean = _dof_acc_mean(dof_pos=dof_pos, fps=fps)
    foot_slip_cm_s = _foot_slip_cm_s(robot_tracks=robot_tracks, fps=fps)

    imitation_score, quality_score, overall_score = _compute_scores(
        mpjpe_cm=mpjpe_cm,
        pck_10cm=pck_10cm,
        vel_error_cm_s=vel_error_cm_s,
        root_rmse_cm=root_traj_rmse_cm,
        dtw_pose_cm=dtw_pose_cm,
        joint_limit_violation_rate=joint_limit_violation_rate,
        dof_acc_mean=dof_acc_mean,
        foot_slip_cm_s=foot_slip_cm_s,
    )

    valid_joint_names = [JOINT_ORDER[i] for i in valid_joint_ids]
    unresolved_joint_names = [k for k, v in resolved_names.items() if v is None]

    return RobotEval(
        robot_dir=str(robot_dir),
        motion_pkl=str(motion_pkl),
        xml_path=str(xml_path),
        fps=fps,
        frames_compared=T,
        valid_joints=int(valid_joint_ids.size),
        valid_joint_names=valid_joint_names,
        unresolved_joint_names=unresolved_joint_names,
        mpjpe_cm=mpjpe_cm,
        pck_10cm=pck_10cm,
        vel_error_cm_s=vel_error_cm_s,
        root_traj_rmse_cm=root_traj_rmse_cm,
        dtw_pose_cm=dtw_pose_cm,
        joint_limit_violation_rate=joint_limit_violation_rate,
        dof_acc_mean=dof_acc_mean,
        foot_slip_cm_s=foot_slip_cm_s,
        imitation_score=imitation_score,
        quality_score=quality_score,
        overall_score=overall_score,
    )


def _fmt_float(x: float, dec: int = 3) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{dec}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalua fidelidad de imitación entre motion humano (.npz) y motion retargeteado de robots (.pkl). "
            "Calcula métricas tipo humano->robot y métricas de calidad en robot."
        )
    )
    parser.add_argument("--motion-npz", type=str, required=True, help="Path al motion humano original .npz.")
    parser.add_argument(
        "--body-models-dir",
        type=str,
        default=str(ROOT / "data"),
        help="Directorio con body_models/ (SMPLH + DMPLs).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--robot-dir", type=str, default=None, help="Carpeta de un robot (contiene robot.xml y .pkl).")
    group.add_argument(
        "--retargeting-dir",
        type=str,
        default=None,
        help="Carpeta con múltiples robot_XXX para evaluar y rankear.",
    )
    parser.add_argument(
        "--motion-stem",
        type=str,
        default=None,
        help="Nombre base del pkl sin extension (si hay más de un .pkl por robot).",
    )
    parser.add_argument("--enable-dtw", action="store_true", default=False, help="Activa métrica DTW (más lenta).")
    parser.add_argument("--max-dtw-frames", type=int, default=600, help="Máximo frames por secuencia para DTW.")
    parser.add_argument("--output-json", type=str, default=None, help="Guarda resultados en JSON.")
    parser.add_argument("--top-k", type=int, default=10, help="Cuántos robots mostrar en ranking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    motion_npz = Path(args.motion_npz).resolve()
    body_models_dir = Path(args.body_models_dir).resolve()
    robot_dir = Path(args.robot_dir).resolve() if args.robot_dir else None
    retargeting_dir = Path(args.retargeting_dir).resolve() if args.retargeting_dir else None

    if not motion_npz.is_file():
        raise FileNotFoundError(f"No existe motion npz: {motion_npz}")

    robot_dirs = _collect_robot_dirs(robot_dir=robot_dir, retargeting_dir=retargeting_dir)
    print(f"Evaluando {len(robot_dirs)} robot(s)...")

    human_frames_cache: Dict[float, List[Dict[str, Tuple[np.ndarray, np.ndarray]]]] = {}
    results: List[RobotEval] = []
    failures: List[Tuple[str, str]] = []

    for rdir in robot_dirs:
        try:
            eval_result = evaluate_robot(
                motion_npz=motion_npz,
                body_models_dir=body_models_dir,
                robot_dir=rdir,
                motion_stem=args.motion_stem,
                human_frames_cache=human_frames_cache,
                enable_dtw=bool(args.enable_dtw),
                max_dtw_frames=int(args.max_dtw_frames),
            )
            results.append(eval_result)
            print(
                f"[OK] {rdir.name}: overall={_fmt_float(eval_result.overall_score, 2)} | "
                f"mpjpe={_fmt_float(eval_result.mpjpe_cm, 2)} cm | pck10={_fmt_float(eval_result.pck_10cm, 1)}%"
            )
        except Exception as exc:
            failures.append((str(rdir), str(exc)))
            print(f"[FAIL] {rdir.name}: {exc}")

    if not results:
        raise RuntimeError("No se pudo evaluar ningún robot.")

    results_sorted = sorted(results, key=lambda x: x.overall_score, reverse=True)
    top_k = max(1, min(int(args.top_k), len(results_sorted)))

    print("\nRanking (top):")
    for i, res in enumerate(results_sorted[:top_k], start=1):
        print(
            f"{i:02d}. {Path(res.robot_dir).name} | overall={_fmt_float(res.overall_score, 2)} | "
            f"imit={_fmt_float(res.imitation_score, 2)} | quality={_fmt_float(res.quality_score, 2)} | "
            f"mpjpe={_fmt_float(res.mpjpe_cm, 2)} cm | slip={_fmt_float(res.foot_slip_cm_s, 2)} cm/s"
        )

    payload = {
        "motion_npz": str(motion_npz),
        "body_models_dir": str(body_models_dir),
        "num_robots_requested": len(robot_dirs),
        "num_robots_evaluated": len(results_sorted),
        "num_failures": len(failures),
        "results": [asdict(r) for r in results_sorted],
        "failures": [{"robot_dir": r, "error": e} for r, e in failures],
    }

    if args.output_json:
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON guardado en: {out}")

    if failures:
        print("\nRobots con fallo:")
        for robot_path, err in failures:
            print(f"- {robot_path}: {err}")


if __name__ == "__main__":
    main()
