import mujoco
from typing import Iterable, Optional, List
from optimal_embodiment.constants import FRAME_FALLBACKS


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