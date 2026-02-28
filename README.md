## Optimal Embodiment

### Overview

This repository contains the code for the **Optimal Embodiment** project, focused on processing and analyzing human motion based on the AMASS dataset.

### Requirements

- **Python**: recommended >= 3.10  
- **Conda** (Anaconda or Miniconda) for environment management

### Environment setup with Conda

```bash
# Clone the repository (if you have not already)
git clone <URL_OF_THIS_REPOSITORY>
cd OptimalEmbodiment

# Create environment
conda create -n optimal-embodiment python=3.10 -y

# Activate environment
conda activate optimal-embodiment

# Install the package in editable mode (includes the GMR dependency from third_party/gmr)
pip install -e .
```

### Local dependency: GMR (General Motion Retargeting)

The project depends on the **GMR** (_General Motion Retargeting_) package, which is included as a local submodule/code in `third_party/gmr/`. This dependency is declared in `pyproject.toml` as a local path dependency:

```toml
"general_motion_retargeting @ file:./third_party/gmr"
```

When you run `pip install -e .` from the repository root, pip will automatically install the package from `third_party/gmr/` (using its `setup.py`). You do not need to install GMR manually.

If you ever install dependencies only via `pip install -r requirements.txt` (or without using `pyproject.toml`), then you must install GMR explicitly:

```bash
pip install -e ./third_party/gmr
```

Make sure the `third_party/gmr/` directory is available (e.g. by cloning submodules with `git submodule update --init --recursive` if GMR is registered as a submodule).

### Downloading the AMASS dataset

1. Follow the instructions in the **official BlenderProc AMASS guide**:  
   [AMASS Human Poses Dataset – BlenderProc](https://dlr-rm.github.io/BlenderProc/examples/datasets/amass_human_poses/README.html).

2. Once the data is downloaded, create a `data/` directory in the root of the repository (if it does not already exist):

```bash
mkdir -p data
```

3. Copy the downloaded AMASS folders into `data/`, **keeping their original folder names**, for example:

```text
OptimalEmbodiment/
  data/
    AMASS/
      CMU/
      MPI_HDM05/
      ...
```

   The exact structure may vary depending on how you download/extract the dataset, but the idea is to preserve the original AMASS folder names.

### Body models (SMPLH + DMPLS)

For the body models, follow the BlenderProc tutorial **only up to the part where the body model files are mentioned** (for example, the `smplh.tar.xz` and `dmpls.tar.xz` archives) and download those files **manually**.

1. Create the following folder structure inside `data/`:

```text||1
data/
  body_models/
    smplh/
      male/
        model.npz      <-- Extracted from smplh.tar.xz
      female/
        model.npz      <-- Extracted from smplh.tar.xz
      neutral/
        model.npz      <-- Extracted from smplh.tar.xz
    dmpls/
      male/
        model.npz      <-- Extracted from dmpls.tar.xz
      female/
        model.npz      <-- Extracted from dmpls.tar.xz
      neutral/
        model.npz      <-- Extracted from dmpls.tar.xz
```

2. Extract the downloaded archives (`smplh.tar.xz`, `dmpls.tar.xz`, etc.) and copy the corresponding `model.npz` files into the folders indicated above.

### AMASS dataset preprocessing

To obtain the final dataset used by the main algorithm, you should implement a script `scripts/process_amass_data.py` in this repository.

Example usage (once implemented):

```bash
python scripts/process_amass_data.py \
  --input-dir data/AMASS \
  --output-dir data/processed_amass
```

- **`--input-dir`**: path to the root folder containing the original AMASS data (inside `data/`).  
- **`--output-dir`**: path where the processed dataset ready for the algorithm will be stored.

The exact preprocessing steps (normalization, joint selection, output format, feature computation, etc.) should be implemented inside `scripts/process_amass_data.py` according to the needs of your experiments.

### Visualizing AMASS sequences (SMPL+H)

Given the folder structure used in this project:

```text
OptimalEmbodiment/
  data/
    ACCAD/
      ... many *.npz motion files ...
    body_models/
      smplh/{male,female,neutral}/model.npz
      dmpls/{male,female,neutral}/model.npz
```

you can load and visualize an AMASS‑style sequence using the helper classes in `optimal_embodiment.smpl.human` (a light adaptation of the official AMASS visualization notebook):

```python
from optimal_embodiment.smpl.human import AmassSequence, SmplHuman

# Example: pick one ACCAD file
npz_path = "data/ACCAD/MartialArtsWalksTurns_c3d/E13_-_block_right_high_stageii.npz"

# Load the motion and body parameters from the .npz
seq = AmassSequence.from_npz(npz_path)
print(seq.gender, seq.frame_rate, seq.num_frames)

# Build a SMPL+H model using the body models in data/body_models/
human = SmplHuman.from_npz(npz_path, body_models_dir="data")

# Render a single frame (uses PyTorch, human_body_prior, body_visualizer, trimesh)
human.show_frame(frame_idx=0, rotate_front_view=True)
```

Internally this:

- parses AMASS fields such as `root_orient`, `pose_body`, `pose_hand`, `trans`, `betas`, and `dmpls` from the `.npz` file (or splits `poses` if needed),  
- loads the appropriate SMPL+H and DMPL models from `data/body_models/smplh/{gender}/model.npz` and `data/body_models/dmpls/{gender}/model.npz`,  
- calls `human_body_prior.body_model.BodyModel` to obtain vertices, and  
- uses `body_visualizer` + `trimesh` to render a mesh.

To use `SmplHuman.show_frame` you must have the following Python packages installed in your environment in addition to this project’s core dependencies:

- `human_body_prior`  
- `body_visualizer`  
- `trimesh`  
- `matplotlib`

### Citation

If you use this project or the AMASS dataset in scientific work, please cite at least:

**AMASS: Archive of Motion Capture as Surface Shapes**

```bibtex
@inproceedings{mahmood2019amass,
  title     = {AMASS: Archive of Motion Capture as Surface Shapes},
  author    = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F and Pons-Moll, Gerard and Black, Michael J},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
```

Additionally, please cite any other relevant tools or libraries you use (for example, BlenderProc) following the recommendations of their authors.
