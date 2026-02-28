import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np


# ==========================================
# 1. Randomizacion fisica-consistente de inercia
# ==========================================

class PhysicsConsistentInertia:
    """
    Randomizacion fisica-consistente de parametros inerciales mediante
    parametrizacion de Cholesky en R^10.

    theta = [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]
    J' = (U L)(U L)^T, con J = L L^T.
    """

    THETA_RANGES = {
        "alpha": (np.log(0.75), np.log(1.35)),
        "d1": (np.log(0.70), np.log(1.35)),
        "d2": (np.log(0.70), np.log(1.35)),
        "d3": (np.log(0.70), np.log(1.35)),
        "s12": (-0.12, 0.12),
        "s23": (-0.12, 0.12),
        "s13": (-0.12, 0.12),
        "t1": (-0.06, 0.06),
        "t2": (-0.06, 0.06),
        "t3": (-0.06, 0.06),
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
    def _sample_theta() -> Dict[str, float]:
        return {
            k: float(np.random.uniform(v[0], v[1]))
            for k, v in PhysicsConsistentInertia.THETA_RANGES.items()
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

        theta = self._sample_theta()
        U, A = self._build_U(theta)

        UL = U @ L
        J_prime = self._ensure_spd(UL @ UL.T)
        mass, com, inertia = self.extract_inertial(J_prime)

        # AABB half-size aproximado por deformacion afín A.
        base = np.array(base_size, dtype=float)
        new_size = np.abs(A) @ base

        lo = 0.55 * base
        hi = 1.65 * base
        new_size = np.clip(new_size, lo, hi)
        new_size = np.maximum(new_size, 1e-3)

        return float(mass), com.tolist(), inertia.tolist(), new_size.tolist()


# ==========================================
# 2. Randomizacion de articulaciones
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


def _range_scale(name: str) -> Tuple[float, float]:
    b = base_name(name)
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

    def _group(self, name: str) -> str:
        b = base_name(name)
        for g in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist", "waist", "head"):
            if g in b:
                return g
        return "default"

    def _is_optional(self, name: str) -> bool:
        return self._group(name) in self.OPTIONAL_GROUPS

    @staticmethod
    def _axis_from_name(name: str) -> List[float]:
        b = base_name(name)
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
        base_axis: Optional[List[float]] = None,
        hip_group: Optional[Dict[str, Dict[str, List[float]]]] = None,
    ) -> Dict[str, Any]:
        b = base_name(name)
        group = self._group(name)
        lo_s, hi_s = _range_scale(name)

        if group == "hip" and hip_group is not None and b in hip_group["axis"]:
            axis = list(hip_group["axis"][b])
            euler = list(hip_group["euler"][b])
        else:
            axis = list(base_axis) if base_axis is not None else self._axis_from_name(name)
            euler = [0.0, 0.0, 0.0]

        return {
            "type": (
                "hinge"
                if (not self._is_optional(name) or np.random.rand() < self.REVOLUTE_PROB)
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
    ) -> Dict[str, Any]:
        if profile is None:
            profile = self.sample_joint_profile(name)

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
# 3. Plantilla G1 y arbol randomizado
# ==========================================

@dataclass
class TemplateLink:
    name: str
    rel_pos: List[float]
    mass: float
    com: List[float]
    inertia: List[float]
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

        # Mantener body inercial real, descartar marcadores auxiliares sin inercia.
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
            # fallback para enlaces auxiliares retenidos (ej: toes/feet)
            mass, com, inertia = self._fallback_inertial(name)

        joint_axis: Optional[List[float]] = None
        joint_range_deg: Optional[List[float]] = None

        j = body_elem.find("joint")
        if j is not None and j.get("type", "hinge") == "hinge":
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
            joint_axis=joint_axis,
            joint_range_deg=joint_range_deg,
            children=child_nodes,
        )

    def load(self) -> TemplateLink:
        if not self.xml_path.exists():
            raise FileNotFoundError(f"No existe plantilla XML: {self.xml_path}")

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        compiler = root.find("compiler")
        angle_unit = compiler.get("angle", "degree") if compiler is not None else "degree"

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("El XML no contiene <worldbody>.")

        root_body = None
        for b in worldbody.findall("body"):
            if b.find("freejoint") is not None:
                root_body = b
                break
        if root_body is None:
            root_body = worldbody.find("body")

        if root_body is None:
            raise ValueError("No se encontro body raiz en <worldbody>.")

        parsed = self._parse_body(root_body, angle_unit=angle_unit)
        if parsed is None:
            raise ValueError("No se pudo construir la plantilla desde el body raiz.")

        return parsed


class HumanoidBuilder:
    """
    Builder template-first:
    - Lee la morfologia base desde assets/g1_29dof/g1_29dof.xml
    - Randomiza inercia con consistencia fisica
    - Randomiza joints manteniendo semantica humanoide
    """

    def __init__(self, template_xml: Path, ref_mass: float = 25.0, add_head_joints: bool = True):
        self.inertia_rand = PhysicsConsistentInertia()
        self.joint_rand = JointSpaceRandomization()
        self.ref_mass = ref_mass
        self.add_head_joints = add_head_joints

        self._joint_profiles: Dict[str, Dict[str, Any]] = {}
        self._inertia_profiles: Dict[str, Tuple[float, List[float], List[float], List[float]]] = {}
        self._hip_group_profile: Optional[Dict[str, Dict[str, List[float]]]] = None

        self.template_root = G1TemplateLoader(template_xml).load()

    def _reset_profiles(self):
        self._joint_profiles.clear()
        self._inertia_profiles.clear()
        self._hip_group_profile = None

    @staticmethod
    def _estimate_half_size(mass: float, inertia6: List[float]) -> List[float]:
        """
        Estima half-size de caja equivalente a partir de inercia en CoM.
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
            # Motores del cuello: visualmente pequenos para que destaque la cabeza final.
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

    def _get_joint_profile(self, name: str, base_axis: Optional[List[float]]) -> Dict[str, Any]:
        key = pair_key(name)
        if key not in self._joint_profiles:
            b = base_name(name)
            hip_group = None
            if b in self.joint_rand.HIP_JOINTS:
                if self._hip_group_profile is None:
                    self._hip_group_profile = self.joint_rand.sample_hip_group_profile()
                hip_group = self._hip_group_profile

            self._joint_profiles[key] = self.joint_rand.sample_joint_profile(
                name=name,
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

        profile = self._get_joint_profile(name, base_axis=None)
        jp = self.joint_rand.randomize_joint(
            name=name,
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
            # Base fija al mundo para evitar traslacion/rotacion global del robot.
            joint_params = {"type": "weld"}
            rel_pos = node.rel_pos
            euler = [0.0, 0.0, 0.0]
        elif node.joint_axis is not None and node.joint_range_deg is not None:
            profile = self._get_joint_profile(node.name, base_axis=node.joint_axis)
            parent_com_dist = float(max(np.linalg.norm(node.rel_pos), 1e-3))
            jp = self.joint_rand.randomize_joint(
                name=node.name,
                base_pos=node.rel_pos,
                base_range_deg=node.joint_range_deg,
                total_mass=self.ref_mass,
                parent_com_dist=parent_com_dist,
                profile=profile,
            )
            joint_params = jp
            rel_pos = jp["pos"]
            euler = jp["euler"]
        else:
            joint_params = {"type": "weld"}
            rel_pos = node.rel_pos
            euler = [0.0, 0.0, 0.0]

        built = LinkDef(
            name=node.name,
            rel_pos=rel_pos,
            mass=mass,
            com=com,
            inertia=inertia,
            v_size=v_size,
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

        # Eliminar cabeza fija del template para dejar una sola cabeza final.
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
                raise ValueError(f"Masa no positiva en {link.name}: {link.mass}")

            J = self.inertia_rand.pseudo_inertia_from_link(link.mass, link.com, link.inertia)
            eig_min = float(np.min(np.linalg.eigvalsh(J)))
            if eig_min <= 0:
                raise ValueError(f"Pseudo-inercia no SPD en {link.name}: eig_min={eig_min:.3e}")

            for c in link.children:
                _walk(c)

        _walk(root)

        # Validacion de grupo de caderas (si estan presentes con nomenclatura canonica)
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
                raise ValueError(f"Ejes de cadera ({side}) no son permutacion canónica: {axes}")

            euler_sum = np.sum(
                [np.array(hips[h].euler, dtype=float) for h in self.joint_rand.HIP_JOINTS],
                axis=0,
            )
            if np.any(np.abs(euler_sum) > 1e-6):
                raise ValueError(
                    f"Offsets Euler de cadera ({side}) no conservan suma cero: {euler_sum.tolist()}"
                )

    def build(self) -> LinkDef:
        self._reset_profiles()

        root = self._build_from_template(self.template_root, is_root=True)
        if self.add_head_joints:
            self._attach_head_chain(root)

        total_mass = self._sum_mass(root)
        self._rescale_joint_torques(root, total_mass)
        self._assert_physical_consistency(root)
        return root

    def count_active_dofs(self, link: LinkDef) -> int:
        count = 1 if link.joint_params.get("type") == "hinge" else 0
        for c in link.children:
            count += self.count_active_dofs(c)
        return count


# ==========================================
# 4. Compilador XML MuJoCo con colisiones coherentes
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

        # Pierna: exactamente 3 colores por segmento para lectura rapida.
        if "hip" in b:
            return cls.SEGMENT_COLORS["leg_upper"]
        if "knee" in b:
            return cls.SEGMENT_COLORS["leg_mid"]
        if "ankle" in b or "foot" in b or "toe" in b:
            return cls.SEGMENT_COLORS["leg_lower"]

        # Brazo: hombro separado del resto.
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
            jname = f"{link.name}_j"
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

        # Visual tipo bloque (como la referencia)
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

        # Capa de colision del robot. Para visualizacion estable del randomizador
        # se deja desactivada por defecto (evita "explosiones" por solapes iniciales).
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
# 5. Ejecucion
# ==========================================

def run():
    root_dir = Path(__file__).resolve().parents[2]
    template_path = root_dir / "assets" / "g1_29dof" / "g1_29dof.xml"

    builder = HumanoidBuilder(template_xml=template_path, ref_mass=25.0, add_head_joints=True)

    tree = builder.build()
    dofs = builder.count_active_dofs(tree)
    xml_str = MuJoCoCompiler().compile(tree)

    print(f"[init] DOFs activos: {dofs}")
    print(f"[template] usando: {template_path}")

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    print(
        "Visualizador iniciado. Colisiones activas entre enlaces no adyacentes "
        "(padre-hijo excluidos para evitar bloqueo en articulaciones)."
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
