from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from optimal_embodiment.constants import SMPLH_PRIMARY_JOINTS

# NumPy 2.0 removed np.infty; restore it for compatibility with pyrender and other deps.
if not hasattr(np, "infty"):
    np.infty = np.inf  # type: ignore[attr-defined]

PathLike = Union[str, Path]


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


@dataclass
class AmassSequence:
    """
    Lightweight wrapper around a single AMASS-style .npz file.

    This assumes a layout compatible with the ACCAD example, i.e. keys like:
    - gender (str)
    - mocap_framerate or mocap_frame_rate (float)
    - root_orient, pose_body, pose_hand, trans, betas, dmpls (arrays)
    """

    path: Path

    gender: str
    frame_rate: float

    root_orient: np.ndarray  # [T, 3]
    pose_body: np.ndarray  # [T, 63]
    pose_hand: np.ndarray  # [T, 90] typically
    trans: np.ndarray  # [T, 3]
    betas: np.ndarray  # [num_betas]
    dmpls: Optional[np.ndarray] = None  # [T, num_dmpls] or None

    # Optional extras (markers, labels, etc.)
    raw: Dict[str, np.ndarray] = None

    @classmethod
    def from_npz(cls, path: PathLike) -> "AmassSequence":
        """
        Load a single motion sequence from an AMASS/ACCAD-style .npz file.

        Parameters
        ----------
        path:
            Path to the .npz file (e.g. something under ``data/ACCAD``).
        """
        p = Path(path)
        with np.load(p, allow_pickle=True) as bdata:
            keys = set(bdata.keys())

            def _get_required(name: str) -> np.ndarray:
                if name not in bdata:
                    raise KeyError(f"Expected key '{name}' in {p}, found {sorted(keys)}")
                return bdata[name]

            def _get_optional(name: str) -> Optional[np.ndarray]:
                return bdata[name] if name in bdata else None

            gender = str(bdata["gender"]) if "gender" in bdata else "neutral"

            if "mocap_framerate" in bdata:
                frame_rate = float(bdata["mocap_framerate"])
            elif "mocap_framerate" in bdata.files:
                frame_rate = float(bdata["mocap_framerate"])
            elif "mocap_frame_rate" in bdata:
                frame_rate = float(bdata["mocap_frame_rate"])
            else:
                # Fall back to a sensible default if metadata is missing.
                frame_rate = 120.0

            # Prefer explicit fields if present; otherwise, split from 'poses'.
            if {"root_orient", "pose_body", "pose_hand"}.issubset(keys):
                root_orient = _get_required("root_orient")
                pose_body = _get_required("pose_body")
                pose_hand = _get_required("pose_hand")
            else:
                poses = _get_required("poses")
                root_orient = poses[:, :3]
                pose_body = poses[:, 3:66]
                pose_hand = poses[:, 66:]

            trans = _get_required("trans")
            betas = _get_required("betas")
            dmpls = _get_optional("dmpls")

            # Keep a shallow copy of all raw arrays for advanced users.
            raw = {k: bdata[k] for k in bdata.files}

        return cls(
            path=p,
            gender=gender,
            frame_rate=frame_rate,
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            trans=trans,
            betas=betas,
            dmpls=dmpls,
            raw=raw,
        )

    @property
    def num_frames(self) -> int:
        return int(self.trans.shape[0])

    @property
    def num_betas(self) -> int:
        return int(self.betas.shape[0])

    @property
    def has_dmpls(self) -> bool:
        return self.dmpls is not None


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


def _body_model_cls_for_motion():
    try:
        import smplx.lbs as _smplx_lbs
    except Exception as exc:
        raise RuntimeError("Se requiere `smplx` para usar human_body_prior en retargeting.") from exc

    original_lbs = _smplx_lbs.lbs

    def _lbs_compat(*args: object, dtype: object = None, **kwargs: object) -> object:
        return original_lbs(*args, **kwargs)

    _smplx_lbs.lbs = _lbs_compat  # type: ignore[misc]

    try:
        from human_body_prior.body_model.body_model import BodyModel
    except Exception as exc:
        raise RuntimeError(
            "Se requiere `human_body_prior` y `torch` para reconstruir joints SMPLH."
        ) from exc
    return BodyModel


def _build_body_model_joints(motion: AmassMotion, smplh_path: Path, dmpl_path: Optional[Path]) -> np.ndarray:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Se requiere `human_body_prior` y `torch` para reconstruir joints SMPLH."
        ) from exc

    BodyModel = _body_model_cls_for_motion()

    t_frames = int(motion.trans.shape[0])
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
            batch_size=t_frames,
            num_dmpls=requested_num_dmpls if requested_num_dmpls > 0 else None,
            path_dmpl=str(dmpl_path) if requested_num_dmpls > 0 and dmpl_path is not None else None,
        ).to(device)
        dmpl_enabled = requested_num_dmpls > 0
    except TypeError:
        body_model = BodyModel(
            bm_path=str(smplh_path),
            num_betas=num_betas,
            batch_size=t_frames,
        ).to(device)
    except Exception:
        body_model = BodyModel(
            bm_path=str(smplh_path),
            model_type="smplh",
            num_betas=num_betas,
            batch_size=t_frames,
            num_dmpls=None,
            path_dmpl=None,
        ).to(device)

    root_orient = torch.as_tensor(motion.root_orient, dtype=torch.float32, device=device)
    pose_body = torch.as_tensor(motion.pose_body, dtype=torch.float32, device=device)
    pose_hand = torch.as_tensor(motion.pose_hand, dtype=torch.float32, device=device)
    trans = torch.as_tensor(motion.trans, dtype=torch.float32, device=device)

    betas = torch.as_tensor(np.asarray(motion.betas).reshape(-1)[:num_betas], dtype=torch.float32, device=device)
    betas_time = betas.unsqueeze(0).expand(t_frames, -1)

    params = {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "pose_hand": pose_hand,
        "trans": trans,
        "betas": betas_time,
    }

    if dmpl_enabled and motion.dmpls is not None:
        dmpls = np.asarray(motion.dmpls)
        if dmpls.ndim == 2 and dmpls.shape[0] == t_frames:
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
    try:
        from scipy.spatial.transform import Rotation as R
    except Exception as exc:
        raise RuntimeError("Se requiere `scipy` para reconstruir orientaciones globales SMPLH.") from exc

    t_frames, num_joints, _ = local_rotvecs.shape
    local_xyzw = R.from_rotvec(local_rotvecs.reshape(-1, 3)).as_quat().reshape(t_frames, num_joints, 4)
    global_xyzw = np.zeros_like(local_xyzw)

    for j in range(num_joints):
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
    try:
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp
    except Exception as exc:
        raise RuntimeError("Se requiere `scipy` para remuestrear quaterniones SMPLH.") from exc

    if len(t_src) == 1:
        return np.repeat(quat_wxyz[:1], repeats=len(t_tgt), axis=0)

    quat_xyzw = np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=-1)
    rots = R.from_quat(quat_xyzw)
    slerp = Slerp(t_src, rots)
    tgt_xyzw = slerp(t_tgt).as_quat()
    return np.concatenate([tgt_xyzw[:, 3:4], tgt_xyzw[:, :3]], axis=-1)


def load_human_motion_frames(
    npz_path: PathLike,
    body_models_dir: PathLike,
    target_fps: float,
) -> Tuple[List[Dict[str, Tuple[np.ndarray, np.ndarray]]], float, float]:
    motion = _load_amass_npz(Path(npz_path))
    smplh_path, dmpl_path = _resolve_model_paths(Path(body_models_dir), motion.gender)

    joints = _build_body_model_joints(motion, smplh_path=smplh_path, dmpl_path=dmpl_path)
    parents = _load_parents_from_smplh_model(smplh_path)

    poses = np.asarray(motion.poses, dtype=np.float64)
    t_frames = poses.shape[0]
    num_joints = poses.shape[1] // 3
    local_rotvecs = poses.reshape(t_frames, num_joints, 3)
    global_quat_wxyz = _compute_global_quats_wxyz(local_rotvecs, parents[:num_joints])

    src_fps = float(motion.fps)
    if target_fps <= 0:
        target_fps = src_fps

    if t_frames <= 1:
        t_src = np.array([0.0], dtype=np.float64)
        t_tgt = t_src
        aligned_fps = src_fps
    else:
        duration = (t_frames - 1) / src_fps
        num_tgt = max(1, int(round(duration * target_fps)) + 1)
        t_src = np.linspace(0.0, duration, t_frames)
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


class SmplHuman:
    """
    High-level helper to work with a SMPL+H body model + AMASS sequence.

    This class is intentionally minimal and keeps all heavy dependencies
    (PyTorch, human_body_prior, body_visualizer, trimesh) optional. The
    corresponding methods will raise a clear error if those packages are not
    installed.
    """

    def __init__(
        self,
        sequence: AmassSequence,
        body_models_dir: PathLike = "data/body_models",
        device: Optional[str] = None,
        num_betas: Optional[int] = None,
        num_dmpls: int = 8,
    ) -> None:
        self.sequence = sequence
        self.body_models_dir = Path(body_models_dir)
        self.num_betas = num_betas or sequence.num_betas
        self.num_dmpls = num_dmpls
        self.device_str = device or ("cuda" if self._torch().cuda.is_available() else "cpu")

        torch = self._torch()
        BodyModel = self._body_model_cls()

        bm_path, _ = self._resolve_body_model_paths(sequence.gender)

        # human_body_prior.BodyModel uses bm_path, model_type; DMPL is optional
        # (many installs raise NotImplementedError for num_dmpls/path_dmpl)
        self._bm = BodyModel(
            bm_path=str(bm_path),
            model_type="smplh",
            num_betas=self.num_betas,
            batch_size=1,
            num_dmpls=None,
            path_dmpl=None,
        ).to(torch.device(self.device_str))

        # Cache faces on CPU for visualization.
        self._faces = self._copy2cpu()(self._bm.f)

    # ---------------------------------------------------------------------
    # Public constructors
    # ---------------------------------------------------------------------
    @classmethod
    def from_npz(
        cls,
        npz_path: PathLike,
        body_models_dir: PathLike = "data/body_models",
        device: Optional[str] = None,
        num_betas: Optional[int] = None,
        num_dmpls: int = 8,
    ) -> "SmplHuman":
        """
        Convenience constructor: build from a single .npz file path.
        """
        seq = AmassSequence.from_npz(npz_path)
        return cls(
            sequence=seq,
            body_models_dir=body_models_dir,
            device=device,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
        )

    # ---------------------------------------------------------------------
    # Low-level: build body parameters and vertices
    # ---------------------------------------------------------------------
    def _body_params_torch(self) -> Dict[str, torch.Tensor]:
        """
        Convert the underlying AMASS sequence into Torch tensors expected by
        human_body_prior.BodyModel.
        """
        torch = self._torch()
        device = torch.device(self.device_str)
        seq = self.sequence

        T = seq.num_frames
        root_orient = torch.as_tensor(seq.root_orient, dtype=torch.float32, device=device)
        pose_body = torch.as_tensor(seq.pose_body, dtype=torch.float32, device=device)
        pose_hand = torch.as_tensor(seq.pose_hand, dtype=torch.float32, device=device)
        trans = torch.as_tensor(seq.trans, dtype=torch.float32, device=device)

        betas = torch.as_tensor(seq.betas[: self.num_betas], dtype=torch.float32, device=device)
        betas_time = betas.unsqueeze(0).expand(T, -1)

        params: Dict[str, "torch.Tensor"] = {
            "root_orient": root_orient,
            "pose_body": pose_body,
            "pose_hand": pose_hand,
            "trans": trans,
            "betas": betas_time,
        }

        if seq.has_dmpls:
            dmpls = torch.as_tensor(seq.dmpls[:, : self.num_dmpls], dtype=torch.float32, device=device)
            params["dmpls"] = dmpls

        return params

    def vertices(self) -> np.ndarray:
        """
        Compute body vertices for all frames as a NumPy array of shape [T, 6890, 3].
        """
        with self._torch().no_grad():
            body = self._bm(**self._body_params_torch())
        return self._copy2cpu()(body.v)

    # ---------------------------------------------------------------------
    # Visualization helpers
    # ---------------------------------------------------------------------
    def show_frame(
        self,
        frame_idx: int = 0,
        with_hands: bool = True,
        with_dmpls: bool = True,
        rotate_front_view: bool = True,
    ) -> None:
        """
        Render a single frame in an off-screen mesh viewer.

        Parameters
        ----------
        frame_idx:
            Which frame index to visualize.
        with_hands:
            Currently unused but kept for API symmetry; the underlying model
            already contains hand articulation as long as pose_hand is present.
        with_dmpls:
            If False, DMPL parameters (soft-tissue dynamics) will be zeroed out.
        rotate_front_view:
            Apply the default AMASS rotation to obtain a more intuitive
            front-ish camera view.
        """
        np_vertices = self._single_frame_vertices(frame_idx, with_dmpls=with_dmpls)

        trimesh, MeshViewer, colors, show_image = self._visualization_backends()

        body_mesh = trimesh.Trimesh(
            vertices=np_vertices,
            faces=self._faces,
            vertex_colors=np.tile(colors["grey"], (np_vertices.shape[0], 1)),
        )

        if rotate_front_view:
            # Match the AMASS notebook: rotate around z, then x.
            angle_z = np.deg2rad(-90.0)
            angle_x = np.deg2rad(30.0)
            body_mesh.apply_transform(trimesh.transformations.rotation_matrix(angle_z, (0, 0, 1)))
            body_mesh.apply_transform(trimesh.transformations.rotation_matrix(angle_x, (1, 0, 0)))

        mv = MeshViewer(width=1600, height=1600, use_offscreen=True)
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)

    def _single_frame_vertices(self, frame_idx: int, with_dmpls: bool) -> np.ndarray:
        """
        Compute vertices for a single frame and return them on CPU.
        """
        torch = self._torch()
        device = torch.device(self.device_str)
        seq = self.sequence

        if frame_idx < 0 or frame_idx >= seq.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {seq.num_frames})")

        root_orient = torch.as_tensor(seq.root_orient[frame_idx : frame_idx + 1], dtype=torch.float32, device=device)
        pose_body = torch.as_tensor(seq.pose_body[frame_idx : frame_idx + 1], dtype=torch.float32, device=device)
        pose_hand = torch.as_tensor(seq.pose_hand[frame_idx : frame_idx + 1], dtype=torch.float32, device=device)
        trans = torch.as_tensor(seq.trans[frame_idx : frame_idx + 1], dtype=torch.float32, device=device)

        betas = torch.as_tensor(seq.betas[: self.num_betas], dtype=torch.float32, device=device)
        betas_time = betas.unsqueeze(0)

        params: Dict[str, "torch.Tensor"] = {
            "root_orient": root_orient,
            "pose_body": pose_body,
            "pose_hand": pose_hand,
            "trans": trans,
            "betas": betas_time,
        }

        if with_dmpls and seq.has_dmpls:
            dmpls = torch.as_tensor(seq.dmpls[frame_idx : frame_idx + 1, : self.num_dmpls], dtype=torch.float32, device=device)
            params["dmpls"] = dmpls

        with torch.no_grad():
            body = self._bm(**params)
        return self._copy2cpu()(body.v[0])

    # ---------------------------------------------------------------------
    # Internal: body model + visualization backends
    # ---------------------------------------------------------------------
    def _resolve_body_model_paths(self, gender: str) -> tuple[Path, Optional[Path]]:
        """
        Resolve SMPL+H and optionally DMPL model paths under ``data/body_models``
        with the layout described in the README. Only SMPL+H is required; DMPL
        is optional (many human_body_prior installs do not support it).
        """
        g = gender.lower()
        if g not in {"male", "female", "neutral"}:
            g = "neutral"

        smplh = self.body_models_dir / "body_models" / "smplh" / g / "model.npz"
        dmpls = self.body_models_dir / "body_models" / "dmpls" / g / "model.npz"

        if not smplh.is_file():
            raise FileNotFoundError(f"Could not find SMPL+H model at {smplh}")
        path_dmpl = dmpls if dmpls.is_file() else None
        return smplh, path_dmpl

    @staticmethod
    def _torch():
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise RuntimeError("PyTorch is required for SMPL visualization. Please install 'torch'.") from exc
        return torch

    @staticmethod
    def _body_model_cls():
        try:
            # human_body_prior calls smplx.lbs(..., dtype=self.dtype), but smplx 0.1.28's
            # lbs() does not accept dtype. Patch lbs to accept and ignore dtype so the call works.
            import smplx.lbs as _smplx_lbs  # type: ignore
            _original_lbs = _smplx_lbs.lbs

            def _lbs_compat(*args: object, dtype: object = None, **kwargs: object) -> object:
                return _original_lbs(*args, **kwargs)

            _smplx_lbs.lbs = _lbs_compat  # type: ignore[misc]

            from human_body_prior.body_model.body_model import BodyModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise RuntimeError(
                "The 'human_body_prior' package is required to construct the body model. "
                "Install it from https://github.com/nghorbani/human_body_prior or via pip."
            ) from exc
        return BodyModel

    @staticmethod
    def _copy2cpu():
        try:
            from human_body_prior.tools.omni_tools import copy2cpu as c2c  # type: ignore
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise RuntimeError(
                "The helper 'copy2cpu' from human_body_prior.tools.omni_tools is required. "
                "Ensure 'human_body_prior' is installed correctly."
            ) from exc
        return c2c

    @staticmethod
    def _visualization_backends():
        try:
            import trimesh  # type: ignore
            from body_visualizer.mesh.mesh_viewer import MeshViewer  # type: ignore
            from body_visualizer.tools.vis_tools import colors, show_image  # type: ignore
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise RuntimeError(
                "Visualization requires 'trimesh' and 'body_visualizer'. "
                "Install them to use 'SmplHuman.show_frame'."
            ) from exc
        return trimesh, MeshViewer, colors, show_image
