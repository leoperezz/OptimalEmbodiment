from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = ROOT / "third_party" / "gmr"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))


SMPLH_PRIMARY_JOINTS: Dict[str, int] = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}


FRAME_FALLBACKS: Dict[str, List[str]] = {
    "pelvis": ["pelvis", "base_link", "base", "Waist", "waist_link", "trunk", "Trunk"],
    "torso_link": ["torso_link", "torso", "spine", "waist_pitch_link", "waist_roll_link"],
    "left_hip_roll_link": ["left_hip_roll_link", "left_hip_yaw_link", "left_hip_pitch_link", "left_hip_link"],
    "left_knee_link": ["left_knee_link"],
    "left_toe_link": ["left_toe_link", "left_foot_link", "left_ankle_roll_link", "left_ankle_pitch_link"],
    "right_hip_roll_link": ["right_hip_roll_link", "right_hip_yaw_link", "right_hip_pitch_link", "right_hip_link"],
    "right_knee_link": ["right_knee_link"],
    "right_toe_link": ["right_toe_link", "right_foot_link", "right_ankle_roll_link", "right_ankle_pitch_link"],
    "left_shoulder_yaw_link": ["left_shoulder_yaw_link", "left_shoulder_roll_link", "left_shoulder_pitch_link"],
    "left_elbow_link": ["left_elbow_link"],
    "left_wrist_yaw_link": ["left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link"],
    "right_shoulder_yaw_link": ["right_shoulder_yaw_link", "right_shoulder_roll_link", "right_shoulder_pitch_link"],
    "right_elbow_link": ["right_elbow_link"],
    "right_wrist_yaw_link": ["right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"],
}


@dataclass
class AmassMotion:
    gender: str
    fps: float
    root_orient: np.ndarray  # [T, 3]
    pose_body: np.ndarray  # [T, 63]
    pose_hand: np.ndarray  # [T, 90]
    trans: np.ndarray  # [T, 3]
    betas: np.ndarray  # [B]
    poses: np.ndarray  # [T, 3*J]
    dmpls: Optional[np.ndarray]


def _resolve_model_paths(body_models_dir: Path, gender: str) -> Tuple[Path, Optional[Path]]:
    g = gender.lower()
    if g not in {"male", "female", "neutral"}:
        g = "neutral"

    candidate_roots = [
        body_models_dir / "body_models",
        body_models_dir,
    ]

    for root in candidate_roots:
        smplh_path = root / "smplh" / g / "model.npz"
        dmpl_path = root / "dmpls" / g / "model.npz"
        if smplh_path.is_file():
            return smplh_path, dmpl_path if dmpl_path.is_file() else None

    tried = ", ".join(str(x) for x in candidate_roots)
    raise FileNotFoundError(
        f"No se encontró SMPLH model.npz para gender={g}. Paths base probados: {tried}"
    )


def _load_amass_npz(npz_path: Path) -> AmassMotion:
    with np.load(npz_path, allow_pickle=True) as d:
        keys = set(d.files)

        def req(name: str) -> np.ndarray:
            if name not in d:
                raise KeyError(f"Falta '{name}' en {npz_path}. Keys: {sorted(keys)}")
            return np.asarray(d[name])

        gender = str(d["gender"]) if "gender" in d else "neutral"
        if "mocap_frame_rate" in d:
            fps = float(d["mocap_frame_rate"])
        elif "mocap_framerate" in d:
            fps = float(d["mocap_framerate"])
        else:
            fps = 120.0

        root_orient = req("root_orient")
        pose_body = req("pose_body")
        trans = req("trans")
        betas = req("betas")
        dmpls = np.asarray(d["dmpls"]) if "dmpls" in d else None

        if "pose_hand" in d:
            pose_hand = np.asarray(d["pose_hand"])
        elif "poses" in d and np.asarray(d["poses"]).shape[1] >= 156:
            pose_hand = np.asarray(d["poses"])[:, 66:156]
        else:
            pose_hand = np.zeros((root_orient.shape[0], 90), dtype=np.float32)

        if "poses" in d and np.asarray(d["poses"]).shape[1] >= 156:
            poses = np.asarray(d["poses"])[:, :156]
        else:
            poses = np.concatenate([root_orient, pose_body, pose_hand], axis=1)

    return AmassMotion(
        gender=gender,
        fps=fps,
        root_orient=root_orient,
        pose_body=pose_body,
        pose_hand=pose_hand,
        trans=trans,
        betas=betas,
        poses=poses,
        dmpls=dmpls,
    )


def _estimate_human_height(betas: np.ndarray) -> float:
    b = np.asarray(betas).reshape(-1)
    beta0 = float(b[0]) if b.size > 0 else 0.0
    return 1.66 + 0.1 * beta0


def _build_body_model_joints(motion: AmassMotion, smplh_path: Path, dmpl_path: Optional[Path]) -> np.ndarray:
    try:
        import smplx.lbs as _smplx_lbs
    except Exception as exc:
        raise RuntimeError("Se requiere `smplx` para usar human_body_prior en retargeting.") from exc

    original_lbs = _smplx_lbs.lbs

    def _lbs_compat(*args: object, dtype: object = None, **kwargs: object) -> object:
        return original_lbs(*args, **kwargs)

    _smplx_lbs.lbs = _lbs_compat  # type: ignore[misc]

    try:
        import torch
        from human_body_prior.body_model.body_model import BodyModel
    except Exception as exc:
        raise RuntimeError(
            "Se requiere `human_body_prior` y `torch` para reconstruir joints SMPLH."
        ) from exc

    T = int(motion.trans.shape[0])
    num_betas = int(min(16, np.asarray(motion.betas).reshape(-1).shape[0]))

    device = torch.device("cpu")

    body_model: BodyModel
    dmpl_enabled = False
    requested_num_dmpls = 0
    if motion.dmpls is not None and dmpl_path is not None and np.asarray(motion.dmpls).ndim == 2:
        requested_num_dmpls = int(np.asarray(motion.dmpls).shape[1])

    try:
        body_model = BodyModel(
            bm_path=str(smplh_path),
            model_type="smplh",
            num_betas=num_betas,
            batch_size=T,
            num_dmpls=requested_num_dmpls if requested_num_dmpls > 0 else None,
            path_dmpl=str(dmpl_path) if requested_num_dmpls > 0 and dmpl_path is not None else None,
        ).to(device)
        dmpl_enabled = requested_num_dmpls > 0
    except TypeError:
        body_model = BodyModel(
            bm_path=str(smplh_path),
            num_betas=num_betas,
            batch_size=T,
        ).to(device)
    except Exception:
        body_model = BodyModel(
            bm_path=str(smplh_path),
            model_type="smplh",
            num_betas=num_betas,
            batch_size=T,
            num_dmpls=None,
            path_dmpl=None,
        ).to(device)

    root_orient = torch.as_tensor(motion.root_orient, dtype=torch.float32, device=device)
    pose_body = torch.as_tensor(motion.pose_body, dtype=torch.float32, device=device)
    pose_hand = torch.as_tensor(motion.pose_hand, dtype=torch.float32, device=device)
    trans = torch.as_tensor(motion.trans, dtype=torch.float32, device=device)

    betas = torch.as_tensor(np.asarray(motion.betas).reshape(-1)[:num_betas], dtype=torch.float32, device=device)
    betas_time = betas.unsqueeze(0).expand(T, -1)

    params = {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "pose_hand": pose_hand,
        "trans": trans,
        "betas": betas_time,
    }

    if dmpl_enabled and motion.dmpls is not None:
        dmpls = np.asarray(motion.dmpls)
        if dmpls.ndim == 2 and dmpls.shape[0] == T:
            params["dmpls"] = torch.as_tensor(dmpls, dtype=torch.float32, device=device)

    with torch.no_grad():
        body = body_model(**params)

    if not hasattr(body, "Jtr"):
        raise RuntimeError("BodyModel no devolvió `Jtr`; no se pueden extraer joints.")

    joints = body.Jtr.detach().cpu().numpy()
    if joints.ndim != 3:
        raise RuntimeError(f"Forma inesperada para joints: {joints.shape}")
    return joints


def _load_parents_from_smplh_model(smplh_path: Path) -> np.ndarray:
    with np.load(smplh_path, allow_pickle=True) as m:
        if "kintree_table" not in m:
            raise KeyError(f"El modelo {smplh_path} no contiene kintree_table.")
        parents = np.asarray(m["kintree_table"])[0].astype(np.int64)
    parents[(parents < 0) | (parents > 10_000)] = -1
    if parents.shape[0] > 0:
        parents[0] = -1
    return parents


def _compute_global_quats_wxyz(local_rotvecs: np.ndarray, parents: np.ndarray) -> np.ndarray:
    T, J, _ = local_rotvecs.shape
    local_xyzw = R.from_rotvec(local_rotvecs.reshape(-1, 3)).as_quat().reshape(T, J, 4)
    global_xyzw = np.zeros_like(local_xyzw)

    for j in range(J):
        p = int(parents[j]) if j < len(parents) else -1
        if p < 0:
            global_xyzw[:, j, :] = local_xyzw[:, j, :]
        else:
            parent_rot = R.from_quat(global_xyzw[:, p, :])
            child_local = R.from_quat(local_xyzw[:, j, :])
            global_xyzw[:, j, :] = (parent_rot * child_local).as_quat()

    global_wxyz = np.concatenate([global_xyzw[..., 3:4], global_xyzw[..., :3]], axis=-1)
    return global_wxyz


def _resample_positions(pos: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    out = np.empty((len(t_tgt), 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.interp(t_tgt, t_src, pos[:, k])
    return out


def _resample_quats_wxyz(quat_wxyz: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    if len(t_src) == 1:
        return np.repeat(quat_wxyz[:1], repeats=len(t_tgt), axis=0)

    quat_xyzw = np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=-1)
    rots = R.from_quat(quat_xyzw)
    slerp = Slerp(t_src, rots)
    tgt_xyzw = slerp(t_tgt).as_quat()
    return np.concatenate([tgt_xyzw[:, 3:4], tgt_xyzw[:, :3]], axis=-1)


def load_human_motion_frames(
    npz_path: Path,
    body_models_dir: Path,
    target_fps: float,
) -> Tuple[List[Dict[str, Tuple[np.ndarray, np.ndarray]]], float, float]:
    motion = _load_amass_npz(npz_path)
    smplh_path, dmpl_path = _resolve_model_paths(body_models_dir, motion.gender)

    joints = _build_body_model_joints(motion, smplh_path=smplh_path, dmpl_path=dmpl_path)
    parents = _load_parents_from_smplh_model(smplh_path)

    poses = np.asarray(motion.poses, dtype=np.float64)
    T = poses.shape[0]
    J = poses.shape[1] // 3
    local_rotvecs = poses.reshape(T, J, 3)
    global_quat_wxyz = _compute_global_quats_wxyz(local_rotvecs, parents[:J])

    src_fps = float(motion.fps)
    if target_fps <= 0:
        target_fps = src_fps

    if T <= 1:
        t_src = np.array([0.0], dtype=np.float64)
        t_tgt = t_src
        aligned_fps = src_fps
    else:
        duration = (T - 1) / src_fps
        num_tgt = max(1, int(round(duration * target_fps)) + 1)
        t_src = np.linspace(0.0, duration, T)
        t_tgt = np.linspace(0.0, duration, num_tgt)
        aligned_fps = (num_tgt - 1) / duration if duration > 0 else src_fps

    joint_tracks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for joint_name, idx in SMPLH_PRIMARY_JOINTS.items():
        if idx >= joints.shape[1] or idx >= global_quat_wxyz.shape[1]:
            continue
        pos = joints[:, idx, :]
        quat = global_quat_wxyz[:, idx, :]
        if len(t_tgt) != len(t_src):
            pos = _resample_positions(pos, t_src=t_src, t_tgt=t_tgt)
            quat = _resample_quats_wxyz(quat, t_src=t_src, t_tgt=t_tgt)
        joint_tracks[joint_name] = (pos, quat)

    if not joint_tracks:
        raise RuntimeError("No se pudieron extraer joints SMPLH requeridos para retargeting.")

    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
    num_frames = next(iter(joint_tracks.values()))[0].shape[0]
    for i in range(num_frames):
        frame: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for joint_name, (pos_track, quat_track) in joint_tracks.items():
            frame[joint_name] = (pos_track[i], quat_track[i])
        frames.append(frame)

    human_height = _estimate_human_height(motion.betas)
    return frames, float(aligned_fps), float(human_height)


def _body_names_from_model(model: mujoco.MjModel) -> List[str]:
    names: List[str] = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name is not None:
            names.append(name)
    return names


def _resolve_frame_name(template_frame: str, body_names: Iterable[str]) -> Optional[str]:
    body_set = set(body_names)
    if template_frame in body_set:
        return template_frame
    for candidate in FRAME_FALLBACKS.get(template_frame, []):
        if candidate in body_set:
            return candidate
    return None


def _resolve_head_body(body_names: Iterable[str]) -> Optional[str]:
    body_set = set(body_names)
    for candidate in ["head_yaw", "head_pitch", "head_roll", "head_main", "head_link", "head"]:
        if candidate in body_set:
            return candidate
    return None


def generate_ik_config(
    model: mujoco.MjModel,
    output_path: Path,
    template_ik_path: Path,
    include_head_task: bool = True,
) -> Dict[str, object]:
    with open(template_ik_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    body_names = _body_names_from_model(model)

    root_name = _resolve_frame_name("pelvis", body_names)
    if root_name is None:
        raise ValueError("No se pudo resolver `robot_root_name` para el robot randomizado.")

    out = {
        "robot_root_name": root_name,
        "human_root_name": template["human_root_name"],
        "ground_height": template["ground_height"],
        "human_height_assumption": template["human_height_assumption"],
        "use_ik_match_table1": bool(template.get("use_ik_match_table1", True)),
        "use_ik_match_table2": bool(template.get("use_ik_match_table2", True)),
        "human_scale_table": dict(template["human_scale_table"]),
        "ik_match_table1": {},
        "ik_match_table2": {},
    }

    for table_name in ["ik_match_table1", "ik_match_table2"]:
        src_table = template.get(table_name, {})
        used: set[str] = set()
        dst: Dict[str, List[object]] = {}
        for template_frame, entry in src_table.items():
            resolved = _resolve_frame_name(template_frame, body_names)
            if resolved is None:
                continue
            if resolved in used:
                continue
            used.add(resolved)
            dst[resolved] = list(entry)
        out[table_name] = dst

    if include_head_task:
        head_body = _resolve_head_body(body_names)
        if head_body is not None:
            if "head" not in out["human_scale_table"]:
                out["human_scale_table"]["head"] = 0.8
            if head_body not in out["ik_match_table1"]:
                out["ik_match_table1"][head_body] = [
                    "head",
                    0,
                    10,
                    [0.0, 0.0, 0.0],
                    [-0.5, 0.5, 0.5, 0.5],
                ]
            if head_body not in out["ik_match_table2"]:
                out["ik_match_table2"][head_body] = [
                    "head",
                    10,
                    5,
                    [0.0, 0.0, 0.0],
                    [-0.5, 0.5, 0.5, 0.5],
                ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return out


def build_random_robot_xml(
    output_xml_path: Path,
    template_xml: Path,
    seed: int,
    add_head_joints: bool = True,
) -> mujoco.MjModel:
    from optimal_embodiment.randomization.build import HumanoidBuilder, MuJoCoCompiler

    np.random.seed(seed)
    builder = HumanoidBuilder(template_xml=template_xml, ref_mass=25.0, add_head_joints=add_head_joints)
    tree = builder.build()
    semantic_meta = builder.last_semantic_description
    tree.joint_params["type"] = "free"

    xml_str = MuJoCoCompiler().compile(tree)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    output_xml_path.write_text(xml_str, encoding="utf-8")

    if semantic_meta is not None:
        semantic_path = output_xml_path.with_name("semantic_joint_space.json")
        payload = {
            "seed": int(seed),
            "template_xml": str(template_xml),
            "add_head_joints": bool(add_head_joints),
            "semantic": semantic_meta,
        }
        with open(semantic_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return mujoco.MjModel.from_xml_path(str(output_xml_path))


def _register_robot_in_gmr_dicts(robot_key: str, xml_path: Path, ik_config_path: Path, robot_root: str) -> None:
    import general_motion_retargeting.params as gmr_params

    gmr_params.ROBOT_XML_DICT[robot_key] = xml_path
    gmr_params.IK_CONFIG_DICT.setdefault("smplx", {})[robot_key] = ik_config_path
    gmr_params.ROBOT_BASE_DICT[robot_key] = robot_root
    gmr_params.VIEWER_CAM_DISTANCE_DICT[robot_key] = 2.0


def retarget_motion(
    robot_key: str,
    robot_xml_path: Path,
    ik_config_path: Path,
    human_frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    human_height: float,
    aligned_fps: float,
    save_path: Path,
    visualize: bool = False,
    rate_limit: bool = False,
    record_video: bool = False,
    video_path: Optional[Path] = None,
) -> None:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer

    with open(ik_config_path, "r", encoding="utf-8") as f:
        ik_cfg = json.load(f)

    _register_robot_in_gmr_dicts(
        robot_key=robot_key,
        xml_path=robot_xml_path,
        ik_config_path=ik_config_path,
        robot_root=str(ik_cfg["robot_root_name"]),
    )

    retargeter = GMR(
        src_human="smplx",
        tgt_robot=robot_key,
        actual_human_height=human_height,
        verbose=False,
    )

    viewer = None
    if visualize:
        viewer = RobotMotionViewer(
            robot_type=robot_key,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=record_video,
            video_path=str(video_path) if video_path is not None else None,
        )

    qpos_list: List[np.ndarray] = []
    for frame in human_frames:
        qpos = retargeter.retarget(frame)
        qpos_list.append(qpos.copy())

        if viewer is not None:
            if qpos.shape[0] < 7:
                raise RuntimeError("El robot no tiene floating base (qpos<7).")
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=rate_limit,
                follow_camera=False,
            )

    if viewer is not None:
        viewer.close()

    qpos_arr = np.asarray(qpos_list)
    if qpos_arr.shape[1] < 7:
        raise RuntimeError("qpos inválido: se esperaba base libre con al menos 7 estados.")

    root_pos = qpos_arr[:, :3]
    root_rot = qpos_arr[:, 3:7][:, [1, 2, 3, 0]]  # wxyz -> xyzw
    dof_pos = qpos_arr[:, 7:]

    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget AMASS/SMPLH .npz motion to randomized humanoids using dynamic IK config."
    )
    parser.add_argument("--motion-npz", type=str, required=True, help="Path al motion .npz (AMASS/ACCAD stageii).")
    parser.add_argument(
        "--body-models-dir",
        type=str,
        default=str(ROOT / "data"),
        help="Directorio que contiene body_models/ (o directamente smplh/, dmpls/).",
    )
    parser.add_argument(
        "--template-xml",
        type=str,
        default=str(ROOT / "assets" / "g1_29dof" / "g1_29dof.xml"),
        help="XML base para randomización de morfología.",
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
        help="IK config de referencia (G1) para clonar y adaptar dinámicamente.",
    )
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "retargeting"))
    parser.add_argument("--num-robots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--disable-head-task", action="store_true", default=False)
    parser.add_argument("--disable-head-joints", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--rate-limit", action="store_true", default=False)
    parser.add_argument("--record-video", action="store_true", default=False)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        default=False,
        help="Solo genera XML+IK de robots randomizados, sin ejecutar retargeting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    motion_npz = Path(args.motion_npz).resolve()
    body_models_dir = Path(args.body_models_dir).resolve()
    template_xml = Path(args.template_xml).resolve()
    template_ik = Path(args.template_ik).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not motion_npz.is_file():
        raise FileNotFoundError(f"No existe motion file: {motion_npz}")
    if not template_xml.is_file():
        raise FileNotFoundError(f"No existe template xml: {template_xml}")
    if not template_ik.is_file():
        raise FileNotFoundError(f"No existe template ik json: {template_ik}")

    human_frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
    aligned_fps = float(args.target_fps)
    human_height = 1.8

    if not args.generate_only:
        print("[1/3] Loading human motion and reconstructing SMPLH joints/orientations...")
        human_frames, aligned_fps, human_height = load_human_motion_frames(
            npz_path=motion_npz,
            body_models_dir=body_models_dir,
            target_fps=float(args.target_fps),
        )
        print(f"      Frames: {len(human_frames)} | FPS alineado: {aligned_fps:.3f} | altura estimada: {human_height:.3f} m")

    print("[2/3] Generating randomized robots + dynamic IK configs...")
    for idx in range(int(args.num_robots)):
        robot_seed = int(args.seed) + idx
        robot_key = f"random_humanoid_{int(time.time())}_{idx}"
        robot_dir = output_dir / f"robot_{idx:03d}"
        robot_xml_path = robot_dir / "robot.xml"
        ik_config_path = robot_dir / "smplx_to_robot.json"
        retargeted_motion_path = robot_dir / f"{motion_npz.stem}.pkl"
        video_path = robot_dir / f"{motion_npz.stem}.mp4"

        model = build_random_robot_xml(
            output_xml_path=robot_xml_path,
            template_xml=template_xml,
            seed=robot_seed,
            add_head_joints=not bool(args.disable_head_joints),
        )
        generated = generate_ik_config(
            model=model,
            output_path=ik_config_path,
            template_ik_path=template_ik,
            include_head_task=not bool(args.disable_head_task),
        )
        print(
            f"      robot_{idx:03d}: seed={robot_seed}, bodies={model.nbody}, "
            f"ik1={len(generated['ik_match_table1'])}, ik2={len(generated['ik_match_table2'])}"
        )

        if args.generate_only:
            continue

        print(f"[3/3] Retargeting robot_{idx:03d} motion...")
        retarget_motion(
            robot_key=robot_key,
            robot_xml_path=robot_xml_path,
            ik_config_path=ik_config_path,
            human_frames=human_frames,
            human_height=human_height,
            aligned_fps=aligned_fps,
            save_path=retargeted_motion_path,
            visualize=bool(args.visualize),
            rate_limit=bool(args.rate_limit),
            record_video=bool(args.record_video),
            video_path=video_path if args.record_video else None,
        )
        print(f"      Motion saved in: {retargeted_motion_path}")

    print(f"Done. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
