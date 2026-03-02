import json
from pathlib import Path
import numpy as np
from typing import Iterable, Optional, List, Dict
from optimal_embodiment.constants import FRAME_FALLBACKS
import mujoco


def body_names_from_model(model: mujoco.MjModel) -> List[str]:
    names: List[str] = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name is not None:
            names.append(name)
    return names


def resolve_frame_name(template_frame: str, body_names: Iterable[str]) -> Optional[str]:
    body_set = set(body_names)
    if template_frame in body_set:
        return template_frame
    for candidate in FRAME_FALLBACKS.get(template_frame, []):
        if candidate in body_set:
            return candidate
    return None

def resolve_head_body(body_names: Iterable[str]) -> Optional[str]:
    body_set = set(body_names)
    for candidate in ["head_yaw", "head_pitch", "head_roll", "head_main", "head_link", "head"]:
        if candidate in body_set:
            return candidate
    return None


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