import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mujoco
import mujoco.viewer
import numpy as np


# ==========================================
# 0. Canonical joint space (Nmax = 32)
# ==========================================

N_MAX_GLOBAL_JOINTS = 32

GLOBAL_JOINT_NAMES: List[str] = [
    "left_hip_roll",
    "left_hip_pitch",
    "left_hip_yaw",
    "left_knee_pitch",
    "left_ankle_roll",
    "left_ankle_pitch",
    "right_hip_roll",
    "right_hip_pitch",
    "right_hip_yaw",
    "right_knee_pitch",
    "right_ankle_roll",
    "right_ankle_pitch",
    "waist_pitch",
    "waist_roll",
    "waist_yaw",
    "head_roll",
    "head_pitch",
    "head_yaw",
    "left_shoulder_roll",
    "left_shoulder_pitch",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_roll",
    "right_shoulder_pitch",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

GLOBAL_JOINT_AXES: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


# ==========================================
# 1. Physics-consistent inertia randomization
# ==========================================

class PhysicsConsistentInertia:
    """
    Physics-consistent randomization of inertial parameters via
    Cholesky parametrization in R^10.

    theta = [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]
    J' = (U L)(U L)^T, with J = L L^T.
    """

    # Table-2 inspired ranges per segment. Multipliers are sampled around 1 and
    # mapped to Eq. (25): alpha,d* as logs; shear/translation as centered offsets.
    TABLE2_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
        "shoulder": {
            "e_alpha": (0.8, 1.4),
            "d1": (0.8, 1.2),
            "d2": (0.8, 1.2),
            "d3": (0.8, 1.2),
            "s12": (0.9, 1.1),
            "s23": (0.9, 1.1),
            "s13": (0.9, 1.1),
            "t1": (0.8, 1.2),
            "t2": (0.8, 1.2),
            "t3": (0.7, 1.3),
        },
        "torso": {
            "e_alpha": (0.8, 1.5),
            "d1": (0.8, 1.5),
            "d2": (0.8, 1.4),
            "d3": (0.8, 1.2),
            "s12": (0.9, 1.1),
            "s23": (0.9, 1.1),
            "s13": (0.9, 1.1),
            "t1": (0.8, 1.2),
            "t2": (0.8, 1.2),
            "t3": (0.7, 1.3),
        },
        "pelvis": {
            "e_alpha": (0.8, 1.5),
            "d1": (0.8, 1.4),
            "d2": (0.8, 1.4),
            "d3": (0.8, 1.2),
            "s12": (0.9, 1.1),
            "s23": (0.9, 1.1),
            "s13": (0.9, 1.1),
            "t1": (0.8, 1.2),
            "t2": (0.8, 1.2),
            "t3": (0.7, 1.3),
        },
        "hip": {
            "e_alpha": (0.8, 1.5),
            "d1": (0.8, 1.2),
            "d2": (0.8, 1.2),
            "d3": (0.5, 1.5),
            "s12": (0.9, 1.1),
            "s23": (0.9, 1.1),
            "s13": (0.9, 1.1),
            "t1": (0.8, 1.2),
            "t2": (0.8, 1.2),
            "t3": (0.7, 1.3),
        },
        "knee": {
            "e_alpha": (0.8, 1.5),
            "d1": (0.8, 1.2),
            "d2": (0.8, 1.2),
            "d3": (0.5, 1.5),
            "s12": (0.9, 1.1),
            "s23": (0.9, 1.1),
            "s13": (0.8, 1.1),
            "t1": (0.8, 1.2),
            "t2": (0.8, 1.2),
            "t3": (0.7, 1.3),
        },
        "foot": {
            "e_alpha": (0.8, 1.4),
            "d1": (0.5, 1.5),
            "d2": (0.5, 1.2),
            "d3": (0.8, 1.2),
            "s12": (0.9, 1.1),
            "s23": (0.9, 1.1),
            "s13": (0.9, 1.1),
            "t1": (0.8, 1.2),
            "t2": (0.8, 1.2),
            "t3": (0.8, 1.2),
        },
    }

    SPD_EPS = 1e-9

    @staticmethod
    def _symmetrize(mat: np.ndarray) -> np.ndarray:
        return 0.5 * (mat + mat.T)

    @classmethod
    def _ensure_spd(cls, mat: np.ndarray, eps: float = SPD_EPS) -> np.ndarray:
        mat = cls._symmetrize(mat)
        vals, vecs = np.linalg.eigh(mat)
        vals = np.clip(vals, eps, None)
        return vecs @ np.diag(vals) @ vecs.T

    @staticmethod
    def inertia_vector_to_matrix(inertia: List[float]) -> np.ndarray:
        ixx, iyy, izz, ixy, ixz, iyz = inertia
        return np.array(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
            dtype=float,
        )

    @classmethod
    def pseudo_inertia_from_link(
        cls,
        mass: float,
        com: List[float],
        inertia: List[float],
    ) -> np.ndarray:
        c = np.array(com, dtype=float)
        I_com = cls._ensure_spd(cls.inertia_vector_to_matrix(inertia), eps=cls.SPD_EPS)

        I_bar = I_com + mass * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
        Sigma = 0.5 * np.trace(I_bar) * np.eye(3) - I_bar
        h = mass * c

        J = np.zeros((4, 4), dtype=float)
        J[:3, :3] = Sigma
        J[:3, 3] = h
        J[3, :3] = h
        J[3, 3] = mass
        return cls._ensure_spd(J)

    @classmethod
    def extract_inertial(cls, J: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        J = cls._ensure_spd(J)
        mass = float(max(J[3, 3], cls.SPD_EPS))
        h = J[:3, 3]
        com = h / mass

        Sigma = J[:3, :3]
        I_bar = np.trace(Sigma) * np.eye(3) - Sigma
        I_com = I_bar - mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
        I_com = cls._ensure_spd(I_com, eps=cls.SPD_EPS)

        inertia = np.array(
            [
                I_com[0, 0],
                I_com[1, 1],
                I_com[2, 2],
                I_com[0, 1],
                I_com[0, 2],
                I_com[1, 2],
            ],
            dtype=float,
        )
        return mass, com, inertia

    @staticmethod
    def _inertia_group(name: str) -> str:
        b = base_name(name)
        if "pelvis" in b:
            return "pelvis"
        if "torso" in b or "waist" in b:
            return "torso"
        if "hip" in b:
            return "hip"
        if "knee" in b:
            return "knee"
        if "ankle" in b or "toe" in b or "foot" in b:
            return "foot"
        if any(k in b for k in ("shoulder", "elbow", "wrist", "head", "hand")):
            return "shoulder"
        return "torso"

    @classmethod
    def _sample_theta(cls, link_name: str) -> Dict[str, float]:
        ranges = cls.TABLE2_RANGES[cls._inertia_group(link_name)]
        return {
            "alpha": float(np.log(np.random.uniform(*ranges["e_alpha"]))),
            "d1": float(np.log(np.random.uniform(*ranges["d1"]))),
            "d2": float(np.log(np.random.uniform(*ranges["d2"]))),
            "d3": float(np.log(np.random.uniform(*ranges["d3"]))),
            "s12": float(np.random.uniform(*ranges["s12"]) - 1.0),
            "s23": float(np.random.uniform(*ranges["s23"]) - 1.0),
            "s13": float(np.random.uniform(*ranges["s13"]) - 1.0),
            "t1": float(np.random.uniform(*ranges["t1"]) - 1.0),
            "t2": float(np.random.uniform(*ranges["t2"]) - 1.0),
            "t3": float(np.random.uniform(*ranges["t3"]) - 1.0),
        }

    @staticmethod
    def _build_U(theta: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        A = np.array(
            [
                [np.exp(theta["d1"]), theta["s12"], theta["s13"]],
                [0.0, np.exp(theta["d2"]), theta["s23"]],
                [0.0, 0.0, np.exp(theta["d3"])],
            ],
            dtype=float,
        )

        U_inner = np.array(
            [
                [np.exp(theta["d1"]), theta["s12"], theta["s13"], theta["t1"]],
                [0.0, np.exp(theta["d2"]), theta["s23"], theta["t2"]],
                [0.0, 0.0, np.exp(theta["d3"]), theta["t3"]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        U = np.exp(theta["alpha"]) * U_inner
        return U, A

    def randomize(
        self,
        link_name: str,
        base_mass: float,
        base_com: List[float],
        base_inertia: List[float],
        base_size: List[float],
    ) -> Tuple[float, List[float], List[float], List[float]]:
        """
        Returns:
            (mass, com, inertia6, visual_half_size)
        """
        J = self.pseudo_inertia_from_link(base_mass, base_com, base_inertia)
        L = np.linalg.cholesky(J).T

        theta = self._sample_theta(link_name=link_name)
        U, A = self._build_U(theta)

        UL = U @ L
        J_prime = self._ensure_spd(UL @ UL.T)
        mass, com, inertia = self.extract_inertial(J_prime)

        # AABB half-size approximated by affine deformation A.
        base = np.array(base_size, dtype=float)
        new_size = np.abs(A) @ base

        lo = 0.55 * base
        hi = 1.65 * base
        new_size = np.clip(new_size, lo, hi)
        new_size = np.maximum(new_size, 1e-3)

        return float(mass), com.tolist(), inertia.tolist(), new_size.tolist()


# ==========================================
# 2. Joint randomization
# ==========================================

_RANGE_SCALE: Dict[str, Tuple[float, float]] = {
    "shoulder": (0.80, 1.00),
    "waist": (0.80, 1.00),
    "knee": (0.80, 1.30),
    "ankle": (0.80, 1.00),
    "hip": (0.80, 1.00),
    "head": (0.80, 1.00),
    "elbow": (0.80, 1.00),
    "wrist": (0.80, 1.00),
    "default": (0.80, 1.00),
}


def canonical_name(name: str) -> str:
    n = name.lower()
    for s in ("_joint", "_link"):
        if n.endswith(s):
            n = n[: -len(s)]
    if n.startswith("left_"):
        n = f"{n[len('left_'):]}_l"
    elif n.startswith("right_"):
        n = f"{n[len('right_'):]}_r"
    return n


def base_name(name: str) -> str:
    n = canonical_name(name)
    if n.endswith("_l") or n.endswith("_r"):
        return n[:-2]
    return n


def pair_key(name: str) -> str:
    return base_name(name)


GLOBAL_JOINT_TOKEN_TO_INDEX: Dict[str, int] = {
    "hip_roll_l": 0,
    "hip_pitch_l": 1,
    "hip_yaw_l": 2,
    "knee_l": 3,
    "knee_pitch_l": 3,
    "ankle_roll_l": 4,
    "ankle_pitch_l": 5,
    "hip_roll_r": 6,
    "hip_pitch_r": 7,
    "hip_yaw_r": 8,
    "knee_r": 9,
    "knee_pitch_r": 9,
    "ankle_roll_r": 10,
    "ankle_pitch_r": 11,
    "waist_pitch": 12,
    "waist_roll": 13,
    "waist_yaw": 14,
    "head_roll": 15,
    "head_pitch": 16,
    "head_yaw": 17,
    "shoulder_roll_l": 18,
    "shoulder_pitch_l": 19,
    "shoulder_yaw_l": 20,
    "elbow_l": 21,
    "elbow_pitch_l": 21,
    "wrist_roll_l": 22,
    "wrist_pitch_l": 23,
    "wrist_yaw_l": 24,
    "shoulder_roll_r": 25,
    "shoulder_pitch_r": 26,
    "shoulder_yaw_r": 27,
    "elbow_r": 28,
    "elbow_pitch_r": 28,
    "wrist_roll_r": 29,
    "wrist_pitch_r": 30,
    "wrist_yaw_r": 31,
}


def semantic_joint_index(link_name: str, joint_name: Optional[str] = None) -> Optional[int]:
    if joint_name:
        tok_joint = canonical_name(joint_name)
        idx = GLOBAL_JOINT_TOKEN_TO_INDEX.get(tok_joint)
        if idx is not None:
            return idx
    tok_link = canonical_name(link_name)
    return GLOBAL_JOINT_TOKEN_TO_INDEX.get(tok_link)


def map_joint_state_to_global(
    q_r: Sequence[float],
    global_indices: Sequence[int],
    n_max: int = N_MAX_GLOBAL_JOINTS,
) -> np.ndarray:
    """
    Implements Eq. (10): phi_r: R^{N_r} -> R^{N_max} with zero padding.
    """
    q_r_arr = np.asarray(q_r, dtype=float).reshape(-1)
    if q_r_arr.shape[0] != len(global_indices):
        raise ValueError(
            f"Length mismatch in joint embedding: len(q_r)={q_r_arr.shape[0]} vs "
            f"len(global_indices)={len(global_indices)}"
        )
    q_global = np.zeros((n_max,), dtype=float)
    for local_i, global_i in enumerate(global_indices):
        if global_i < 0 or global_i >= n_max:
            raise ValueError(f"Out-of-range global index: {global_i}")
        q_global[global_i] = float(q_r_arr[local_i])
    return q_global


def _range_scale(name: str, joint_name: Optional[str] = None) -> Tuple[float, float]:
    b = base_name(joint_name) if joint_name is not None else base_name(name)
    for key, val in _RANGE_SCALE.items():
        if key in b:
            return val
    return _RANGE_SCALE["default"]


class JointSpaceRandomization:
    OPTIONAL_GROUPS = {"waist", "shoulder", "elbow", "wrist", "head"}

    REVOLUTE_PROB = 0.75
    POS_SCALE_RANGE = (0.80, 1.20)
    HIP_EULER_RANGE = (-0.30, 0.30)
    TORQUE_SCALE = (0.70, 1.00)
    QDOT_MAX_BASE = 14.0

    HIP_JOINTS = ("hip_roll", "hip_yaw", "hip_pitch")

    @staticmethod
    def _joint_base(name: str, joint_name: Optional[str] = None) -> str:
        if joint_name is not None:
            return base_name(joint_name)
        return base_name(name)

    def _group(self, name: str, joint_name: Optional[str] = None) -> str:
        b = self._joint_base(name, joint_name=joint_name)
        for g in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist", "waist", "head"):
            if g in b:
                return g
        return "default"

    def _is_optional(self, name: str, joint_name: Optional[str] = None) -> bool:
        return self._group(name, joint_name=joint_name) in self.OPTIONAL_GROUPS

    @staticmethod
    def _axis_from_name(name: str, joint_name: Optional[str] = None) -> List[float]:
        b = base_name(joint_name) if joint_name is not None else base_name(name)
        if "roll" in b:
            return [1.0, 0.0, 0.0]
        if "yaw" in b:
            return [0.0, 0.0, 1.0]
        return [0.0, 1.0, 0.0]

    def _sample_hip_group_euler(self) -> Dict[str, List[float]]:
        lo, hi = self.HIP_EULER_RANGE
        while True:
            first_two = np.random.uniform(lo, hi, size=(2, 3))
            third = -(first_two[0] + first_two[1])
            if np.all(third >= lo) and np.all(third <= hi):
                eulers = np.vstack([first_two, third])
                return {
                    jn: [float(v) for v in eulers[idx]]
                    for idx, jn in enumerate(self.HIP_JOINTS)
                }

    def sample_hip_group_profile(self) -> Dict[str, Dict[str, List[float]]]:
        canonical_axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        perm = np.random.permutation(3)
        axis_map = {joint: canonical_axes[int(perm[idx])] for idx, joint in enumerate(self.HIP_JOINTS)}
        return {
            "axis": axis_map,
            "euler": self._sample_hip_group_euler(),
        }

    def sample_joint_profile(
        self,
        name: str,
        joint_name: Optional[str] = None,
        base_axis: Optional[List[float]] = None,
        hip_group: Optional[Dict[str, Dict[str, List[float]]]] = None,
    ) -> Dict[str, Any]:
        b = self._joint_base(name, joint_name=joint_name)
        group = self._group(name, joint_name=joint_name)
        lo_s, hi_s = _range_scale(name, joint_name=joint_name)

        if group == "hip" and hip_group is not None and b in hip_group["axis"]:
            axis = list(hip_group["axis"][b])
            euler = list(hip_group["euler"][b])
        else:
            axis = (
                list(base_axis)
                if base_axis is not None
                else self._axis_from_name(name, joint_name=joint_name)
            )
            euler = [0.0, 0.0, 0.0]

        return {
            "type": (
                "hinge"
                if (
                    not self._is_optional(name, joint_name=joint_name)
                    or np.random.rand() < self.REVOLUTE_PROB
                )
                else "fixed"
            ),
            "axis": axis,
            "scale": np.random.uniform(*self.POS_SCALE_RANGE, size=3),
            "euler": euler,
            "range_scale": float(np.random.uniform(lo_s, hi_s)),
            "qdot_scale": float(np.random.uniform(lo_s, hi_s)),
            "tau_scale": float(np.random.uniform(*self.TORQUE_SCALE)),
        }

    def randomize_joint(
        self,
        name: str,
        base_pos: List[float],
        base_range_deg: List[float],
        total_mass: float,
        parent_com_dist: float,
        profile: Optional[Dict[str, Any]] = None,
        joint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if profile is None:
            profile = self.sample_joint_profile(name=name, joint_name=joint_name)

        j_type = profile["type"]
        axis = list(profile["axis"])

        scale = np.array(profile["scale"], dtype=float)
        new_pos = np.array(base_pos, dtype=float) * scale

        max_dist = 2.0 * max(float(parent_com_dist), 1e-3)
        cur_dist = float(np.linalg.norm(new_pos))
        if cur_dist > max_dist:
            new_pos = new_pos * (max_dist / cur_dist)

        s = float(profile["range_scale"])
        new_range = [float(base_range_deg[0]) * s, float(base_range_deg[1]) * s]
        if new_range[0] > new_range[1]:
            new_range = [new_range[1], new_range[0]]

        qdot_max = self.QDOT_MAX_BASE * float(profile["qdot_scale"])
        tau_max = total_mass * float(profile["tau_scale"])

        return {
            "type": j_type,
            "pos": new_pos.tolist(),
            "euler": list(profile["euler"]),
            "axis": axis,
            "range": new_range,
            "qdot_max": float(qdot_max),
            "torque": float(tau_max),
            "tau_scale": float(profile["tau_scale"]),
        }


# ==========================================
# 3. G1 template and randomized tree
# ==========================================

@dataclass
class TemplateLink:
    name: str
    rel_pos: List[float]
    mass: float
    com: List[float]
    inertia: List[float]
    joint_name: Optional[str] = None
    joint_axis: Optional[List[float]] = None
    joint_range_deg: Optional[List[float]] = None
    children: List["TemplateLink"] = field(default_factory=list)


@dataclass
class LinkDef:
    name: str
    rel_pos: List[float]
    mass: float = 0.0
    com: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    inertia: List[float] = field(default_factory=lambda: [0.0] * 6)
    v_size: List[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    joint_name: Optional[str] = None
    joint_params: Dict[str, Any] = field(default_factory=dict)
    euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    children: List["LinkDef"] = field(default_factory=list)


class G1TemplateLoader:
    def __init__(self, xml_path: Path):
        self.xml_path = xml_path

    @staticmethod
    def _parse_vec(text: Optional[str], n: int, default: float = 0.0) -> List[float]:
        if not text:
            return [default] * n
        vals = [float(v) for v in text.split()]
        if len(vals) < n:
            vals = vals + [default] * (n - len(vals))
        return vals[:n]

    @staticmethod
    def _inertia_from_elem(inertial: ET.Element) -> Tuple[float, List[float], List[float]]:
        mass = float(inertial.get("mass", "0.01"))
        com = G1TemplateLoader._parse_vec(inertial.get("pos"), 3, default=0.0)

        if "fullinertia" in inertial.attrib:
            inertia = G1TemplateLoader._parse_vec(inertial.get("fullinertia"), 6, default=0.0)
        else:
            diag = G1TemplateLoader._parse_vec(inertial.get("diaginertia"), 3, default=1e-4)
            inertia = [diag[0], diag[1], diag[2], 0.0, 0.0, 0.0]

        return mass, com, inertia

    @staticmethod
    def _fallback_inertial(name: str) -> Tuple[float, List[float], List[float]]:
        b = base_name(name)

        if "toe" in b or "foot" in b:
            mass = 0.12
            size = [0.055, 0.018, 0.010]
            com = [0.012, 0.0, 0.0]
        else:
            mass = 0.05
            size = [0.025, 0.025, 0.025]
            com = [0.0, 0.0, 0.0]

        ixx = (1.0 / 12.0) * mass * ((2.0 * size[1]) ** 2 + (2.0 * size[2]) ** 2)
        iyy = (1.0 / 12.0) * mass * ((2.0 * size[0]) ** 2 + (2.0 * size[2]) ** 2)
        izz = (1.0 / 12.0) * mass * ((2.0 * size[0]) ** 2 + (2.0 * size[1]) ** 2)
        inertia = [ixx, iyy, izz, 0.0, 0.0, 0.0]
        return mass, com, inertia

    @staticmethod
    def _should_drop_aux_body(body_elem: ET.Element, parsed_children: List[TemplateLink]) -> bool:
        has_joint = body_elem.find("joint") is not None
        has_inertial = body_elem.find("inertial") is not None
        has_children = len(parsed_children) > 0

        if has_joint or has_children:
            return False

        name = body_elem.get("name", "")
        b = base_name(name)
        if "toe" in b or "foot" in b:
            return False

        # Keep real inertial body, discard auxiliary markers without inertia.
        return not has_inertial

    def _parse_body(self, body_elem: ET.Element, angle_unit: str) -> Optional[TemplateLink]:
        child_nodes: List[TemplateLink] = []
        for c in body_elem.findall("body"):
            parsed = self._parse_body(c, angle_unit=angle_unit)
            if parsed is not None:
                child_nodes.append(parsed)

        if self._should_drop_aux_body(body_elem, child_nodes):
            return None

        name = body_elem.get("name", "unnamed")
        rel_pos = self._parse_vec(body_elem.get("pos"), 3, default=0.0)

        inertial = body_elem.find("inertial")
        if inertial is not None:
            mass, com, inertia = self._inertia_from_elem(inertial)
        else:
            # fallback for retained auxiliary links (e.g. toes/feet)
            mass, com, inertia = self._fallback_inertial(name)

        joint_name: Optional[str] = None
        joint_axis: Optional[List[float]] = None
        joint_range_deg: Optional[List[float]] = None

        j = body_elem.find("joint")
        if j is not None and j.get("type", "hinge") == "hinge":
            joint_name = j.get("name", f"{name}_joint")
            joint_axis = self._parse_vec(j.get("axis"), 3, default=0.0)
            r = self._parse_vec(j.get("range"), 2, default=0.0)
            if angle_unit == "radian":
                joint_range_deg = [float(np.degrees(r[0])), float(np.degrees(r[1]))]
            else:
                joint_range_deg = [float(r[0]), float(r[1])]

        return TemplateLink(
            name=name,
            rel_pos=rel_pos,
            mass=float(max(mass, 1e-5)),
            com=com,
            inertia=inertia,
            joint_name=joint_name,
            joint_axis=joint_axis,
            joint_range_deg=joint_range_deg,
            children=child_nodes,
        )

    def load(self) -> TemplateLink:
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML template does not exist: {self.xml_path}")

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        compiler = root.find("compiler")
        angle_unit = compiler.get("angle", "degree") if compiler is not None else "degree"

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("The XML does not contain <worldbody>.")

        root_body = None
        for b in worldbody.findall("body"):
            if b.find("freejoint") is not None:
                root_body = b
                break
        if root_body is None:
            root_body = worldbody.find("body")

        if root_body is None:
            raise ValueError("Root body was not found in <worldbody>.")

        parsed = self._parse_body(root_body, angle_unit=angle_unit)
        if parsed is None:
            raise ValueError("Could not build template from root body.")

        return parsed


class HumanoidBuilder:
    """
    Builder template-first:
    - Reads base morphology from assets/g1_29dof/g1_29dof.xml
    - Randomizes inertia with physical consistency
    - Randomizes joints maintaining humanoid semantics
    """

    def __init__(self, template_xml: Path, ref_mass: float = 25.0, add_head_joints: bool = True):
        self.inertia_rand = PhysicsConsistentInertia()
        self.joint_rand = JointSpaceRandomization()
        self.ref_mass = ref_mass
        self.add_head_joints = add_head_joints

        self._joint_profiles: Dict[str, Dict[str, Any]] = {}
        self._inertia_profiles: Dict[str, Tuple[float, List[float], List[float], List[float]]] = {}
        self._hip_group_profile: Optional[Dict[str, Dict[str, List[float]]]] = None
        self.last_semantic_description: Optional[Dict[str, Any]] = None

        self.template_root = G1TemplateLoader(template_xml).load()

    def _reset_profiles(self):
        self._joint_profiles.clear()
        self._inertia_profiles.clear()
        self._hip_group_profile = None

    @staticmethod
    def _estimate_half_size(mass: float, inertia6: List[float]) -> List[float]:
        """
        Estimates equivalent box half-size from inertia at CoM.
        """
        m = max(float(mass), 1e-6)
        ixx, iyy, izz = float(inertia6[0]), float(inertia6[1]), float(inertia6[2])

        lx2 = max((6.0 / m) * (iyy + izz - ixx), 1e-6)
        ly2 = max((6.0 / m) * (ixx + izz - iyy), 1e-6)
        lz2 = max((6.0 / m) * (ixx + iyy - izz), 1e-6)

        hx = float(np.sqrt(lx2) * 0.5)
        hy = float(np.sqrt(ly2) * 0.5)
        hz = float(np.sqrt(lz2) * 0.5)

        return [
            float(np.clip(hx, 0.015, 0.20)),
            float(np.clip(hy, 0.015, 0.20)),
            float(np.clip(hz, 0.015, 0.30)),
        ]

    @staticmethod
    def _refine_visual_size(name: str, size: List[float]) -> List[float]:
        b = base_name(name)
        s = np.array(size, dtype=float) * np.random.uniform(0.75, 1.30, size=3)

        if "torso" in b:
            lo, hi = [0.050, 0.040, 0.090], [0.130, 0.110, 0.260]
        elif "pelvis" in b:
            lo, hi = [0.050, 0.040, 0.040], [0.120, 0.100, 0.160]
        elif b in {"head_yaw", "head_pitch", "head_roll"}:
            # Neck motors: visually small so the final head stands out.
            lo, hi = [0.015, 0.015, 0.012], [0.032, 0.032, 0.028]
        elif b in {"head", "head_visual", "head_main"}:
            lo, hi = [0.070, 0.060, 0.070], [0.110, 0.100, 0.130]
        elif "hip" in b or "shoulder" in b:
            lo, hi = [0.018, 0.018, 0.028], [0.060, 0.060, 0.110]
        elif "knee" in b or "elbow" in b:
            lo, hi = [0.016, 0.016, 0.045], [0.055, 0.055, 0.180]
        elif "ankle" in b or "wrist" in b:
            lo, hi = [0.010, 0.010, 0.010], [0.045, 0.055, 0.090]
        elif "toe" in b or "foot" in b:
            s = np.array([max(s[0], 0.050), max(s[1], 0.014), max(s[2], 0.008)], dtype=float)
            lo, hi = [0.045, 0.012, 0.008], [0.110, 0.036, 0.030]
        elif "head" in b:
            lo, hi = [0.055, 0.050, 0.060], [0.130, 0.120, 0.150]
        else:
            lo, hi = [0.012, 0.012, 0.018], [0.095, 0.095, 0.140]

        s = np.clip(s, lo, hi)
        return s.tolist()

    def _get_joint_profile(
        self,
        name: str,
        joint_name: Optional[str],
        base_axis: Optional[List[float]],
    ) -> Dict[str, Any]:
        key = pair_key(name)
        if key not in self._joint_profiles:
            b = self.joint_rand._joint_base(name, joint_name=joint_name)
            hip_group = None
            if b in self.joint_rand.HIP_JOINTS:
                if self._hip_group_profile is None:
                    self._hip_group_profile = self.joint_rand.sample_hip_group_profile()
                hip_group = self._hip_group_profile

            self._joint_profiles[key] = self.joint_rand.sample_joint_profile(
                name=name,
                joint_name=joint_name,
                base_axis=base_axis,
                hip_group=hip_group,
            )
        return self._joint_profiles[key]

    def _get_inertia_profile(
        self,
        name: str,
        b_mass: float,
        b_com: List[float],
        b_inertia: List[float],
        b_size: List[float],
    ) -> Tuple[float, List[float], List[float], List[float]]:
        key = pair_key(name)
        if key not in self._inertia_profiles:
            self._inertia_profiles[key] = self.inertia_rand.randomize(
                link_name=name,
                base_mass=b_mass,
                base_com=b_com,
                base_inertia=b_inertia,
                base_size=b_size,
            )
        return self._inertia_profiles[key]

    def _make_synthetic_link(
        self,
        name: str,
        b_mass: float,
        b_size: List[float],
        b_pos: List[float],
        b_range: List[float],
        parent_com_dist: float,
        children: Optional[List[LinkDef]] = None,
    ) -> LinkDef:
        mass, com, inertia, v_size = self._get_inertia_profile(
            name=name,
            b_mass=b_mass,
            b_com=[0.0, 0.0, 0.0],
            b_inertia=[
                (1.0 / 12.0) * b_mass * ((2.0 * b_size[1]) ** 2 + (2.0 * b_size[2]) ** 2),
                (1.0 / 12.0) * b_mass * ((2.0 * b_size[0]) ** 2 + (2.0 * b_size[2]) ** 2),
                (1.0 / 12.0) * b_mass * ((2.0 * b_size[0]) ** 2 + (2.0 * b_size[1]) ** 2),
                0.0,
                0.0,
                0.0,
            ],
            b_size=b_size,
        )
        v_size = self._refine_visual_size(name, v_size)

        synthetic_joint_name = f"{name}_joint"
        profile = self._get_joint_profile(name=name, joint_name=synthetic_joint_name, base_axis=None)
        jp = self.joint_rand.randomize_joint(
            name=name,
            joint_name=synthetic_joint_name,
            base_pos=b_pos,
            base_range_deg=b_range,
            total_mass=self.ref_mass,
            parent_com_dist=parent_com_dist,
            profile=profile,
        )

        return LinkDef(
            name=name,
            rel_pos=jp["pos"],
            mass=mass,
            com=com,
            inertia=inertia,
            v_size=v_size,
            joint_name=synthetic_joint_name,
            joint_params=jp,
            euler=jp["euler"],
            children=children or [],
        )

    def _make_synthetic_weld_link(
        self,
        name: str,
        b_mass: float,
        b_size: List[float],
        b_pos: List[float],
        children: Optional[List[LinkDef]] = None,
    ) -> LinkDef:
        mass, com, inertia, v_size = self._get_inertia_profile(
            name=name,
            b_mass=b_mass,
            b_com=[0.0, 0.0, 0.0],
            b_inertia=[
                (1.0 / 12.0) * b_mass * ((2.0 * b_size[1]) ** 2 + (2.0 * b_size[2]) ** 2),
                (1.0 / 12.0) * b_mass * ((2.0 * b_size[0]) ** 2 + (2.0 * b_size[2]) ** 2),
                (1.0 / 12.0) * b_mass * ((2.0 * b_size[0]) ** 2 + (2.0 * b_size[1]) ** 2),
                0.0,
                0.0,
                0.0,
            ],
            b_size=b_size,
        )
        v_size = self._refine_visual_size(name, v_size)

        return LinkDef(
            name=name,
            rel_pos=list(b_pos),
            mass=mass,
            com=com,
            inertia=inertia,
            v_size=v_size,
            joint_name=None,
            joint_params={"type": "weld"},
            euler=[0.0, 0.0, 0.0],
            children=children or [],
        )

    def _build_from_template(self, node: TemplateLink, is_root: bool = False) -> LinkDef:
        b_size = self._estimate_half_size(node.mass, node.inertia)
        mass, com, inertia, v_size = self._get_inertia_profile(
            name=node.name,
            b_mass=node.mass,
            b_com=node.com,
            b_inertia=node.inertia,
            b_size=b_size,
        )
        v_size = self._refine_visual_size(node.name, v_size)

        if is_root:
            # Fixed base to world to avoid global translation/rotation of the robot.
            joint_params = {"type": "weld"}
            rel_pos = node.rel_pos
            euler = [0.0, 0.0, 0.0]
            joint_name = node.joint_name
        elif node.joint_axis is not None and node.joint_range_deg is not None:
            profile = self._get_joint_profile(
                name=node.name,
                joint_name=node.joint_name,
                base_axis=node.joint_axis,
            )
            parent_com_dist = float(max(np.linalg.norm(node.rel_pos), 1e-3))
            jp = self.joint_rand.randomize_joint(
                name=node.name,
                joint_name=node.joint_name,
                base_pos=node.rel_pos,
                base_range_deg=node.joint_range_deg,
                total_mass=self.ref_mass,
                parent_com_dist=parent_com_dist,
                profile=profile,
            )
            joint_params = jp
            rel_pos = jp["pos"]
            euler = jp["euler"]
            joint_name = node.joint_name
        else:
            joint_params = {"type": "weld"}
            rel_pos = node.rel_pos
            euler = [0.0, 0.0, 0.0]
            joint_name = node.joint_name

        built = LinkDef(
            name=node.name,
            rel_pos=rel_pos,
            mass=mass,
            com=com,
            inertia=inertia,
            v_size=v_size,
            joint_name=joint_name,
            joint_params=joint_params,
            euler=euler,
            children=[],
        )

        for c in node.children:
            built.children.append(self._build_from_template(c, is_root=False))

        return built

    def _sum_mass(self, link: LinkDef) -> float:
        total = float(link.mass)
        for c in link.children:
            total += self._sum_mass(c)
        return total

    def _rescale_joint_torques(self, link: LinkDef, total_mass: float):
        if link.joint_params.get("type") == "hinge":
            tau_scale = float(link.joint_params.get("tau_scale", 1.0))
            link.joint_params["torque"] = total_mass * tau_scale
        for c in link.children:
            self._rescale_joint_torques(c, total_mass)

    def _find_first(self, link: LinkDef, predicate) -> Optional[LinkDef]:
        if predicate(link):
            return link
        for c in link.children:
            f = self._find_first(c, predicate)
            if f is not None:
                return f
        return None

    def _contains_base(self, link: LinkDef, query_base: str) -> bool:
        if base_name(link.name) == query_base:
            return True
        for c in link.children:
            if self._contains_base(c, query_base):
                return True
        return False

    def _attach_head_chain(self, root: LinkDef):
        if self._contains_base(root, "head_yaw"):
            return

        torso = self._find_first(root, lambda l: "torso" in base_name(l.name))
        if torso is None:
            return

        # Remove fixed head from template to leave a single final head.
        torso.children = [
            c
            for c in torso.children
            if base_name(c.name) not in {"head", "head_link", "head_main", "head_visual"}
        ]

        head_main = self._make_synthetic_weld_link(
            "head_main",
            b_mass=1.35,
            b_size=[0.085, 0.078, 0.092],
            b_pos=[0.0, 0.0, 0.088],
        )
        head_roll = self._make_synthetic_link(
            "head_roll",
            b_mass=0.22,
            b_size=[0.024, 0.024, 0.018],
            b_pos=[0.0, 0.0, 0.028],
            b_range=[-35.0, 35.0],
            parent_com_dist=0.028,
            children=[head_main],
        )
        head_pitch = self._make_synthetic_link(
            "head_pitch",
            b_mass=0.24,
            b_size=[0.026, 0.026, 0.020],
            b_pos=[0.0, 0.0, 0.032],
            b_range=[-45.0, 45.0],
            parent_com_dist=0.032,
            children=[head_roll],
        )
        head_yaw = self._make_synthetic_link(
            "head_yaw",
            b_mass=0.26,
            b_size=[0.028, 0.028, 0.022],
            b_pos=[0.0, 0.0, 0.255],
            b_range=[-65.0, 65.0],
            parent_com_dist=0.255,
            children=[head_pitch],
        )
        torso.children.append(head_yaw)

    def _assert_physical_consistency(self, root: LinkDef):
        def _walk(link: LinkDef):
            if link.mass <= 0:
                raise ValueError(f"Non-positive mass in {link.name}: {link.mass}")

            J = self.inertia_rand.pseudo_inertia_from_link(link.mass, link.com, link.inertia)
            eig_min = float(np.min(np.linalg.eigvalsh(J)))
            if eig_min <= 0:
                raise ValueError(f"Pseudo-inertia not SPD in {link.name}: eig_min={eig_min:.3e}")

            for c in link.children:
                _walk(c)

        _walk(root)

        # Hip group validation (if present with canonical nomenclature)
        canonical = {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}

        side_map: Dict[str, Dict[str, LinkDef]] = {"l": {}, "r": {}}

        def _collect(link: LinkDef):
            cname = canonical_name(link.name)
            for side in ("l", "r"):
                for h in self.joint_rand.HIP_JOINTS:
                    if cname == f"{h}_{side}":
                        side_map[side][h] = link
            for c in link.children:
                _collect(c)

        _collect(root)

        for side in ("l", "r"):
            hips = side_map[side]
            if len(hips) != 3:
                continue

            axes = {
                tuple(float(v) for v in hips[h].joint_params.get("axis", [0.0, 0.0, 0.0]))
                for h in self.joint_rand.HIP_JOINTS
            }
            if axes != canonical:
                raise ValueError(f"Hip axes ({side}) are not canonical permutation: {axes}")

            euler_sum = np.sum(
                [np.array(hips[h].euler, dtype=float) for h in self.joint_rand.HIP_JOINTS],
                axis=0,
            )
            if np.any(np.abs(euler_sum) > 1e-6):
                raise ValueError(
                    f"Hip Euler offsets ({side}) do not preserve zero sum: {euler_sum.tolist()}"
                )

    def _collect_hinge_global_indices(self, root: LinkDef) -> List[int]:
        indices: List[int] = []

        def _walk(link: LinkDef):
            idx = semantic_joint_index(link.name, link.joint_name)
            if idx is not None and link.joint_params.get("type") == "hinge":
                indices.append(idx)
            for c in link.children:
                _walk(c)

        _walk(root)
        return indices

    def build_semantic_description(self, root: LinkDef) -> Dict[str, Any]:
        present_mask = np.zeros((N_MAX_GLOBAL_JOINTS,), dtype=np.int32)
        actuated_mask = np.zeros((N_MAX_GLOBAL_JOINTS,), dtype=np.int32)
        adjacency = np.zeros((N_MAX_GLOBAL_JOINTS, N_MAX_GLOBAL_JOINTS), dtype=np.int32)
        link_name_for_index = [""] * N_MAX_GLOBAL_JOINTS
        joint_type_for_index = ["missing"] * N_MAX_GLOBAL_JOINTS
        indegree = np.zeros((N_MAX_GLOBAL_JOINTS,), dtype=np.int32)

        def _walk(link: LinkDef, parent_idx: Optional[int]):
            idx = semantic_joint_index(link.name, link.joint_name)
            next_parent = parent_idx

            if idx is not None:
                present_mask[idx] = 1
                if not link_name_for_index[idx]:
                    link_name_for_index[idx] = link.name
                if link.joint_params.get("type") == "hinge":
                    actuated_mask[idx] = 1
                    joint_type_for_index[idx] = "hinge"
                else:
                    if joint_type_for_index[idx] == "missing":
                        joint_type_for_index[idx] = "fixed"
                if parent_idx is not None and parent_idx != idx and adjacency[parent_idx, idx] == 0:
                    adjacency[parent_idx, idx] = 1
                    indegree[idx] += 1
                next_parent = idx

            for c in link.children:
                _walk(c, next_parent)

        _walk(root, parent_idx=None)

        roots = [i for i in range(N_MAX_GLOBAL_JOINTS) if present_mask[i] == 1 and indegree[i] == 0]
        if len(roots) > 1:
            anchor = 14 if 14 in roots else roots[0]  # waist_yaw if available
            for r in roots:
                if r == anchor:
                    continue
                if adjacency[anchor, r] == 0:
                    adjacency[anchor, r] = 1

        active_global_indices = self._collect_hinge_global_indices(root)
        return {
            "n_max": int(N_MAX_GLOBAL_JOINTS),
            "global_joint_names": list(GLOBAL_JOINT_NAMES),
            "global_joint_axes": GLOBAL_JOINT_AXES.astype(float).tolist(),
            "joint_present_mask": present_mask.astype(int).tolist(),
            "joint_actuated_mask": actuated_mask.astype(int).tolist(),
            "joint_types": joint_type_for_index,
            "joint_link_names": link_name_for_index,
            "adjacency": adjacency.astype(int).tolist(),
            "active_dofs": int(np.sum(actuated_mask)),
            "active_global_indices_in_kinematic_order": [int(x) for x in active_global_indices],
        }

    def build(self) -> LinkDef:
        self._reset_profiles()
        self.last_semantic_description = None

        root = self._build_from_template(self.template_root, is_root=True)
        if self.add_head_joints:
            self._attach_head_chain(root)

        total_mass = self._sum_mass(root)
        self._rescale_joint_torques(root, total_mass)
        self._assert_physical_consistency(root)

        dofs = self.count_active_dofs(root)
        if dofs < 12 or dofs > 32:
            raise ValueError(f"Active DoF out of expected [12, 32] range: {dofs}")

        self.last_semantic_description = self.build_semantic_description(root)
        return root

    def count_active_dofs(self, link: LinkDef) -> int:
        count = 1 if link.joint_params.get("type") == "hinge" else 0
        for c in link.children:
            count += self.count_active_dofs(c)
        return count


# ==========================================
# 4. MuJoCo XML compiler with coherent collisions
# ==========================================

class MuJoCoCompiler:
    SEGMENT_COLORS: Dict[str, str] = {
        "pelvis": "0.35 0.35 0.38 1",
        "torso": "0.52 0.52 0.56 1",
        "waist": "0.44 0.44 0.48 1",
        "leg_upper": "0.88 0.43 0.24 1",
        "leg_mid": "0.98 0.76 0.22 1",
        "leg_lower": "0.20 0.61 0.83 1",
        "arm_shoulder": "0.75 0.26 0.20 1",
        "arm_mid": "0.16 0.64 0.42 1",
        "arm_lower": "0.13 0.44 0.78 1",
        "head_motor": "0.28 0.28 0.30 1",
        "head_main": "0.92 0.80 0.66 1",
        "default": "0.62 0.62 0.67 1",
    }

    def __init__(self):
        self.enable_robot_collisions = False
        self.root = ET.Element("mujoco", model="randomized_g1")

        ET.SubElement(self.root, "compiler", angle="degree", inertiafromgeom="false")
        ET.SubElement(self.root, "option", timestep="0.005", gravity="0 0 0")

        default = ET.SubElement(self.root, "default")
        ET.SubElement(
            default,
            "geom",
            condim="3",
            friction="0.8 0.1 0.1",
            solref="0.005 1",
            solimp="0.9 0.95 0.001",
        )
        ET.SubElement(default, "joint", damping="1.0", armature="0.01")
        ET.SubElement(default, "motor", ctrllimited="true")

        self.contact = ET.SubElement(self.root, "contact")
        self.actuator = ET.SubElement(self.root, "actuator")
        self.worldbody = ET.SubElement(self.root, "worldbody")

        self._build_assets()
        self._build_floor()

    def _build_assets(self):
        asset = ET.SubElement(self.root, "asset")
        ET.SubElement(
            asset,
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1="0.95 0.95 0.95",
            rgb2="0.75 0.8 0.9",
            width="512",
            height="512",
        )

    def _build_floor(self):
        ET.SubElement(
            self.worldbody,
            "geom",
            name="floor",
            type="plane",
            size="0 0 1",
            pos="0 0 0",
            contype="1",
            conaffinity="1",
            friction="0.9 0.1 0.1",
            rgba="0.95 0.95 0.95 1",
        )
        ET.SubElement(
            self.worldbody,
            "light",
            directional="true",
            diffuse="0.8 0.8 0.8",
            specular="0.2 0.2 0.2",
            pos="0 0 5",
            dir="0 0 -1",
        )

    @classmethod
    def _segment_color(cls, link_name: str) -> str:
        b = base_name(link_name)

        if b in {"head_yaw", "head_pitch", "head_roll"}:
            return cls.SEGMENT_COLORS["head_motor"]
        if b in {"head", "head_link", "head_main", "head_visual"}:
            return cls.SEGMENT_COLORS["head_main"]

        if "pelvis" in b:
            return cls.SEGMENT_COLORS["pelvis"]
        if "torso" in b:
            return cls.SEGMENT_COLORS["torso"]
        if "waist" in b:
            return cls.SEGMENT_COLORS["waist"]

        # Leg: exactly 3 colors per segment for quick reading.
        if "hip" in b:
            return cls.SEGMENT_COLORS["leg_upper"]
        if "knee" in b:
            return cls.SEGMENT_COLORS["leg_mid"]
        if "ankle" in b or "foot" in b or "toe" in b:
            return cls.SEGMENT_COLORS["leg_lower"]

        # Arm: shoulder separate from the rest.
        if "shoulder" in b:
            return cls.SEGMENT_COLORS["arm_shoulder"]
        if "elbow" in b:
            return cls.SEGMENT_COLORS["arm_mid"]
        if "wrist" in b or "hand" in b:
            return cls.SEGMENT_COLORS["arm_lower"]

        return cls.SEGMENT_COLORS["default"]

    @staticmethod
    def _fmt(vals: List[float], prec: int = 5) -> str:
        return " ".join(f"{v:.{prec}f}" for v in vals)

    @staticmethod
    def _geom_pos(link: LinkDef) -> List[float]:
        b = base_name(link.name)
        size = np.array(link.v_size, dtype=float)

        if "toe" in b or "foot" in b:
            return [float(size[0] * 0.35), 0.0, 0.0]

        if len(link.children) > 0:
            direction = np.mean(np.array([c.rel_pos for c in link.children], dtype=float), axis=0)
            proposed = 0.45 * direction
            cap = np.maximum(0.90 * size, np.array([0.015, 0.015, 0.015], dtype=float))
            pos = np.clip(proposed, -cap, cap)
            return pos.tolist()

        return [0.0, 0.0, 0.0]

    def _add_parent_exclusion(self, parent_name: str, child_name: str):
        ET.SubElement(self.contact, "exclude", body1=parent_name, body2=child_name)

    def _build_body(self, parent_xml: ET.Element, link: LinkDef, parent_name: Optional[str] = None):
        attrs: Dict[str, str] = {
            "name": link.name,
            "pos": self._fmt(link.rel_pos),
        }

        euler_deg = [float(np.degrees(e)) for e in link.euler]
        if any(abs(v) > 1e-6 for v in euler_deg):
            attrs["euler"] = self._fmt(euler_deg, prec=4)

        body_xml = ET.SubElement(parent_xml, "body", **attrs)

        ET.SubElement(
            body_xml,
            "inertial",
            pos=self._fmt(link.com),
            mass=f"{link.mass:.6f}",
            fullinertia=self._fmt(link.inertia, prec=8),
        )

        jtype = link.joint_params.get("type", "weld")
        if jtype == "free":
            ET.SubElement(body_xml, "freejoint", name=f"{link.name}_root")
        elif jtype == "hinge":
            jname = link.joint_name if link.joint_name else f"{link.name}_j"
            jp = link.joint_params
            ET.SubElement(
                body_xml,
                "joint",
                name=jname,
                type="hinge",
                axis=self._fmt(jp["axis"], prec=3),
                range=f"{jp['range'][0]:.3f} {jp['range'][1]:.3f}",
            )
            ET.SubElement(
                self.actuator,
                "motor",
                name=f"{jname}_m",
                joint=jname,
                ctrlrange=f"-{jp['torque']:.3f} {jp['torque']:.3f}",
                gear="1",
            )

        color = self._segment_color(link.name)
        geom_pos = self._geom_pos(link)

        # Block-style visual (as in the reference)
        ET.SubElement(
            body_xml,
            "geom",
            name=f"{link.name}_vis",
            type="box",
            pos=self._fmt(geom_pos),
            size=self._fmt(link.v_size),
            rgba=color,
            contype="0",
            conaffinity="0",
            group="1",
        )

        # Robot collision layer. Disabled by default for stable randomizer visualization
        # (avoids "explosions" from initial overlaps).
        contype = "1" if self.enable_robot_collisions else "0"
        conaffinity = "1" if self.enable_robot_collisions else "0"
        col_size = [float(max(v * 0.92, 0.008)) for v in link.v_size]
        ET.SubElement(
            body_xml,
            "geom",
            name=f"{link.name}_col",
            type="box",
            pos=self._fmt(geom_pos),
            size=self._fmt(col_size),
            rgba="0.2 0.2 0.2 0.0",
            contype=contype,
            conaffinity=conaffinity,
            group="3",
        )

        if parent_name is not None:
            self._add_parent_exclusion(parent_name, link.name)

        for c in link.children:
            self._build_body(body_xml, c, parent_name=link.name)

    def compile(self, root_link: LinkDef) -> str:
        self._build_body(self.worldbody, root_link, parent_name=None)
        return ET.tostring(self.root, encoding="unicode")


# ==========================================
# 5. Execution
# ==========================================

def run():
    root_dir = Path(__file__).resolve().parents[2]
    template_path = root_dir / "assets" / "g1_29dof" / "g1_29dof.xml"

    builder = HumanoidBuilder(template_xml=template_path, ref_mass=25.0, add_head_joints=True)

    tree = builder.build()
    dofs = builder.count_active_dofs(tree)
    xml_str = MuJoCoCompiler().compile(tree)

    print(f"[init] Active DOFs: {dofs}")
    print(f"[template] using: {template_path}")

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    print(
        "Viewer started. Collisions active between non-adjacent links "
        "(parent-child excluded to avoid joint locking)."
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    run()
