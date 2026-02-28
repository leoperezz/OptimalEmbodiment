# Optimal Embodiment

**Finding the best possible embodiment to imitate humans.**

Optimal Embodiment is a research framework for discovering and evaluating robot morphologies that best enable imitation of human motion. Rather than fixing a single robot design, we treat the embodiment as a variable: the goal is to identify—through physics-consistent morphological randomization and motion retargeting—which mechanical and inertial structure best supports robust, generalist humanoid control and human motion tracking.

---

*Human motion retargeted to a randomized humanoid embodiment (Unitree G1 base).*

<video src="images/video_random.mp4" controls width="100%"></video>

---

## Overview

This repository provides:

- **Human motion processing** from the [AMASS](https://amass.is.tue.mpg.de/) dataset (SMPL+H / body models).
- **Motion retargeting** to humanoid robots via [GMR (General Motion Retargeting)](https://github.com/YanjieZe/GMR), integrated as a submodule in `third_party/gmr`.
- **Morphological randomization** for cross-embodiment experimentation, using the **Unitree G1 (29 DOF)** as the base template. Randomization follows physics-consistent reparameterizations (link inertia, joint geometry, limits, actuation) so that policies can be trained or evaluated across a distribution of valid humanoid morphologies.

The approach is inspired by **XHugWBC** ([Xue et al., 2026](https://arxiv.org/abs/2602.05791)), which demonstrates that a single whole-body controller can generalize across many humanoid designs through semantically aligned observation/action spaces and morphological randomization. We build on similar ideas to study *optimal* embodiment for human imitation rather than a fixed robot.

---

## Dependencies

### GMR (General Motion Retargeting)

We use [GMR](https://github.com/YanjieZe/GMR) for retargeting human motion (e.g. from AMASS/SMPL-X) to humanoid robots in real time. **GMR must be available as a Git submodule under `third_party/gmr`.**

Initialize the repository and submodules:

```bash
git clone <URL_OF_THIS_REPOSITORY>
cd OptimalEmbodiment
git submodule update --init --recursive
```

This populates `third_party/gmr` with the [YanjieZe/GMR](https://github.com/YanjieZe/GMR) repository. The project declares GMR as a local path dependency in `pyproject.toml`:

```toml
"general_motion_retargeting @ file:./third_party/gmr"
```

So after cloning and initializing submodules, a single `pip install -e .` installs both this package and GMR from `third_party/gmr`. You do not need to install GMR separately.

If you install dependencies without using `pyproject.toml`, install GMR explicitly:

```bash
pip install -e ./third_party/gmr
```

---

## Environment Setup

- **Python**: ≥ 3.10  
- **Conda** recommended for environment management.

```bash
conda create -n optimal-embodiment python=3.10 -y
conda activate optimal-embodiment
pip install -e .
```

---

## Data Setup

### AMASS dataset

1. Follow the [BlenderProc AMASS guide](https://dlr-rm.github.io/BlenderProc/examples/datasets/amass_human_poses/README.html) to download AMASS.
2. Create a `data/` directory and place the downloaded AMASS folders (e.g. `CMU/`, `MPI_HDM05/`, …) inside it, preserving original folder names:

```text
OptimalEmbodiment/
  data/
    AMASS/
      CMU/
      MPI_HDM05/
      ...
```

### Body models (SMPL+H and DMPLs)

Download the body model archives (`smplh.tar.xz`, `dmpls.tar.xz`) as indicated in the BlenderProc tutorial and extract them under:

```text
data/
  body_models/
    smplh/
      male/    model.npz
      female/  model.npz
      neutral/ model.npz
    dmpls/
      male/    model.npz
      female/  model.npz
      neutral/ model.npz
```

---

## Usage

### Preprocessing AMASS

Run the preprocessing script (implement/adapt as needed for your pipeline):

```bash
python scripts/process_amass_data.py \
  --input-dir data/AMASS \
  --output-dir data/processed_amass
```

### Visualizing AMASS (SMPL+H)

With `data/` and body models in place, you can load and visualize AMASS sequences:

```python
from optimal_embodiment.smpl.human import AmassSequence, SmplHuman

npz_path = "data/ACCAD/MartialArtsWalksTurns_c3d/E13_-_block_right_high_stageii.npz"
seq = AmassSequence.from_npz(npz_path)
human = SmplHuman.from_npz(npz_path, body_models_dir="data")
human.show_frame(frame_idx=0, rotate_front_view=True)
```

Additional Python dependencies for visualization: `human_body_prior`, `body_visualizer`, `trimesh`, `matplotlib`.

---

## Morphological Randomization

Randomization is built on the **Unitree G1 (29 DOF)** as the base morphology. Link and joint parameters (mass, inertia, CoM, joint axes, limits, actuation scaling) are perturbed in a physics-consistent way so that inertial and kinematic validity are preserved. Details are documented in [docs/randomization.md](docs/randomization.md).

---

## Related Work & Citation

- **GMR (General Motion Retargeting):** [YanjieZe/GMR](https://github.com/YanjieZe/GMR) — real-time retargeting of human motion to diverse humanoids; we use it as a submodule in `third_party/gmr`.
- **XHugWBC:** Xue et al., *Scalable and General Whole-Body Control for Cross-Humanoid Locomotion* (2026), [arXiv:2602.05791](https://arxiv.org/abs/2602.05791), [project page](https://xhugwbc.github.io/). Single-policy cross-embodiment control via morphological randomization and aligned spaces; our randomization and cross-embodiment goals are inspired by this line of work.
- **AMASS:** If you use AMASS in your work, please cite the dataset as in the [AMASS project](https://amass.is.tue.mpg.de/).

---

## License

MIT.
