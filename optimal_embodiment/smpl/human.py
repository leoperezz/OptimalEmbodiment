from __future__ import annotations
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

# NumPy 2.0 removed np.infty; restore it for compatibility with pyrender and other deps.
if not hasattr(np, "infty"):
    np.infty = np.inf  # type: ignore[attr-defined]

PathLike = Union[str, Path]


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

