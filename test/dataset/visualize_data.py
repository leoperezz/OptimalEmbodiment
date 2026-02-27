"""Render an AMASS sequence as a video (full sequence, person facing camera).

Usage examples (run from project root):

    # Render full sequence to rendered_sequence.mp4
    python test/dataset/visualize_data.py data/ACCAD/MartialArtsWalksTurns_c3d/E13_-_block_right_high_stageii.npz

    # Custom body models directory
    python test/dataset/visualize_data.py data/ACCAD/some_motion.npz --body-models-dir data

    # Save to a specific file
    python test/dataset/visualize_data.py path/to/motion.npz -o my_sequence.mp4
"""
import argparse
import sys
from pathlib import Path

# Add project root so optimal_embodiment is importable when run as script
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

if not hasattr(np, "infty"):
    np.infty = np.inf

from optimal_embodiment.smpl.human import SmplHuman
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors
import imageio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render full AMASS .npz sequence as MP4 video (person facing camera)."
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to the AMASS/ACCAD .npz motion file.",
    )
    parser.add_argument(
        "--body-models-dir",
        type=str,
        default="data",
        help="Directory containing SMPL body model files (default: data).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="rendered_sequence.mp4",
        help="Output video path (default: rendered_sequence.mp4).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    npz_path = args.npz_path
    body_models_dir = args.body_models_dir
    output_path = args.output

    human = SmplHuman.from_npz(npz_path=npz_path, body_models_dir=body_models_dir)

    seq = human.sequence
    num_frames = seq.num_frames
    fps = seq.frame_rate
    print(f"gender={seq.gender}, frames={num_frames}, fps={fps}")

    mv = MeshViewer(width=800, height=800, use_offscreen=True)

    # Render every frame of the sequence
    frames = []
    for fId in range(num_frames):
        np_verts = human._single_frame_vertices(frame_idx=fId, with_dmpls=True)
        body_mesh = trimesh.Trimesh(
            vertices=np_verts,
            faces=human._faces,
            vertex_colors=np.tile(colors["grey"], (np_verts.shape[0], 1)),
        )
        # Center the mesh so the viewer doesn't clip
        body_mesh.vertices -= body_mesh.centroid
        # AMASS root is Z-up; bring to Y-up front view
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(15), (0, 1, 0)))
        # Rotate 180° around Y so the person faces the camera (instead of back)
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (0, 1, 0)))
        mv.set_static_meshes([body_mesh])
        img = mv.render(render_wireframe=False)
        frames.append(img)
        if (fId + 1) % 50 == 0 or fId == 0:
            print(f"Rendered frame {fId + 1}/{num_frames}")

    # Write video (fps from sequence)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved {output_path}  ({num_frames} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
