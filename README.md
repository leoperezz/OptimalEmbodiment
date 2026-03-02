# Optimal Embodiment

**Finding the best possible embodiment to imitate humans.**

Research framework for discovering robot morphologies that best enable imitation of human motion. Uses morphological randomization from [XHugWBC](https://arxiv.org/abs/2602.05791) (Unitree G1 base) and motion retargeting via [GMR](https://github.com/YanjieZe/GMR).

![Human motion retargeted to randomized humanoid](images/video_random.gif)

---

## Setup

```bash
git clone <URL>
cd OptimalEmbodiment
git submodule update --init --recursive
conda create -n optimal-embodiment python=3.10 -y
conda activate optimal-embodiment
pip install -e .
```

**Data:** Download [AMASS](https://amass.is.tue.mpg.de/) and body models (SMPL+H, DMPLs) per [BlenderProc guide](https://dlr-rm.github.io/BlenderProc/examples/datasets/amass_human_poses/README.html). Place under `data/AMASS/` and `data/body_models/`.

---

## Scripts

### `scripts/retargeting.py`

Retargets AMASS/SMPLH motion (.npz) to randomized humanoid robots. Generates XML + IK config per robot and saves retargeted motion as .pkl.

```bash
python scripts/retargeting.py --motion-npz data/AMASS/ACCAD/.../motion.npz

# With options:
python scripts/retargeting.py --motion-npz path/to/motion.npz \
  --body-models-dir data \
  --output-dir outputs/retargeting \
  --num-robots 3 \
  --visualize \
  --record-video
```

| Flag | Description |
|------|-------------|
| `--motion-npz` | Path to .npz (AMASS/ACCAD stageii) |
| `--body-models-dir` | Directory with body_models/ (default: data) |
| `--template-xml` | Base XML for randomization (default: assets/g1_29dof/g1_29dof.xml) |
| `--output-dir` | Output directory (default: outputs/retargeting) |
| `--num-robots` | Number of randomized robots |
| `--visualize` | Open MuJoCo viewer during retargeting |
| `--record-video` | Record .mp4 per robot |
| `--generate-only` | Only generate XML+IK, no retargeting |

---

### `scripts/visualize.py`

Visualizes retargeted motions (.pkl) together with robot.xml in MuJoCo.

```bash
# Single robot folder:
python scripts/visualize.py outputs/retargeting/robot_000

# Parent folder (plays all robot_XXX sequentially):
python scripts/visualize.py outputs/retargeting

# With options:
python scripts/visualize.py outputs/retargeting --loop --record-video
```

| Flag | Description |
|------|-------------|
| `--loop` | Loop each clip indefinitely |
| `--record-video` | Save .mp4 next to the .pkl |
| `--no-camera-follow` | Disable camera follow |
| `--cam-distance` | Camera distance (default: 2.5) |
| `--cam-elevation` | Camera elevation angle (default: -10) |

---

## Related Work

- **GMR:** [YanjieZe/GMR](https://github.com/YanjieZe/GMR) — real-time retargeting.
- **XHugWBC:** Xue et al., [arXiv:2602.05791](https://arxiv.org/abs/2602.05791) — cross-embodiment control; we use its morphological randomization.
- **AMASS:** [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/)

---

MIT License.
