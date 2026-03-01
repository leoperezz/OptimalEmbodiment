# Optimal Embodiment

**Finding the best possible embodiment to imitate humans.**

Research framework for discovering robot morphologies that best enable imitation of human motion. Uses physics-consistent morphological randomization (Unitree G1 base) and motion retargeting via [GMR](https://github.com/YanjieZe/GMR).

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

# Con opciones:
python scripts/retargeting.py --motion-npz path/to/motion.npz \
  --body-models-dir data \
  --output-dir outputs/retargeting \
  --num-robots 3 \
  --visualize \
  --record-video
```

| Flag | Descripción |
|------|-------------|
| `--motion-npz` | Ruta al .npz (AMASS/ACCAD stageii) |
| `--body-models-dir` | Directorio con body_models/ (default: data) |
| `--template-xml` | XML base para randomización (default: assets/g1_29dof/g1_29dof.xml) |
| `--output-dir` | Salida (default: outputs/retargeting) |
| `--num-robots` | Número de robots randomizados |
| `--visualize` | Abre viewer MuJoCo durante retargeting |
| `--record-video` | Graba .mp4 por robot |
| `--generate-only` | Solo genera XML+IK, sin retargeting |

---

### `scripts/visualize.py`

Visualiza movimientos retargeteados (.pkl) junto con robot.xml en MuJoCo.

```bash
# Una carpeta de robot:
python scripts/visualize.py outputs/retargeting/robot_000

# Carpeta padre (reproduce todos los robot_XXX secuencialmente):
python scripts/visualize.py outputs/retargeting

# Con opciones:
python scripts/visualize.py outputs/retargeting --loop --record-video
```

| Flag | Descripción |
|------|-------------|
| `--loop` | Repite cada clip indefinidamente |
| `--record-video` | Guarda .mp4 junto al .pkl |
| `--no-camera-follow` | Desactiva seguimiento de cámara |
| `--cam-distance` | Distancia cámara (default: 2.5) |
| `--cam-elevation` | Ángulo elevación cámara (default: -10) |

---

## Related Work

- **GMR:** [YanjieZe/GMR](https://github.com/YanjieZe/GMR) — retargeting en tiempo real.
- **XHugWBC:** Xue et al., [arXiv:2602.05791](https://arxiv.org/abs/2602.05791) — control cross-embodiment.
- **AMASS:** [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/)

---

MIT License.
