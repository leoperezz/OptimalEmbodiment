"""
Retarget AMASS/SMPLH motion (.npz) to randomized humanoid robots using dynamic IK.

Example usage (run from project root):

  # Retarget a single motion to one randomized robot (defaults)
  python scripts/retargeting.py --motion-npz data/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz

  # Retarget to 3 different randomized robots, 30 fps, with visualization
  python scripts/retargeting.py --motion-npz data/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz \
      --num-robots 3 --target-fps 30 --visualize --rate-limit

  # Only generate randomized robot XMLs and IK configs (no retargeting)
  python scripts/retargeting.py --motion-npz data/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz --generate-only

  # Full example: custom paths, record video, fixed seed
  python scripts/retargeting.py \
      --motion-npz data/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz \
      --body-models-dir data \
      --template-xml assets/g1_29dof/g1_29dof.xml \
      --template-ik third_party/gmr/general_motion_retargeting/ik_configs/smplx_to_g1.json \
      --output-dir outputs/retargeting \
      --num-robots 1 --seed 42 --target-fps 30 \
      --record-video --visualize --rate-limit
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mujoco
import numpy as np

from optimal_embodiment.constants import FRAME_FALLBACKS
from optimal_embodiment.smpl.human import load_human_motion_frames

ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = ROOT / "third_party" / "gmr"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))


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
    from optimal_embodiment.robot.build import HumanoidBuilder, MuJoCoCompiler

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
    parser.add_argument("--target-fps", type=float, default=20.0)
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
