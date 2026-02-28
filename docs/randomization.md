# Morphological Randomization in `radomize_env.py`

This document describes what is randomized in
`test/randomizer/radomize_env.py`, how the randomization is physically
constrained, and which ranges are currently used in code.

The implementation follows the same idea as physically consistent
morphological randomization used in cross-humanoid locomotion work:
randomize in a reparameterized space where inertial validity is preserved.

## 1. What is randomized

The randomized morphology has two components:

1. Link parameters (mass, center of mass, inertia, visual size)
2. Joint parameters (joint type, axis, relative position, orientation offset,
   limits, and actuation scaling)

## 2. Link randomization (physically consistent inertia)

For each link, the code builds a pseudo-inertia matrix:

`J = [[Sigma, h], [h^T, m]]`

where:

- `m` is mass
- `h = m * c` is first moment
- `c` is center of mass
- `Sigma = 0.5 * Tr(I_bar) * I - I_bar`

The key physical validity condition is:

`J` must be symmetric positive definite (SPD).

Instead of directly perturbing inertial entries, the code uses a Cholesky-based
reparameterization:

1. Compute `J = L L^T`
2. Sample a 10D parameter vector
3. Build transformation matrix `U`
4. Compute `J' = (U L)(U L)^T`

This guarantees valid inertial parameters after randomization.

### 2.1 Inertial parameter vector and interpretation

The sampled vector is:

`theta = [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]`

Physical interpretation:

- `d1, d2, d3`: axis-wise scaling (stretch/compression)
- `s12, s13, s23`: shear terms
- `t1, t2, t3`: translation-like terms shifting mass distribution
- `alpha`: global density/mass scaling term

### 2.2 Current inertial sampling ranges in code

Uniform sampling is used for all terms:

- `alpha`: `log(0.75)` to `log(1.35)`
- `d1, d2, d3`: `log(0.70)` to `log(1.35)`
- `s12, s13, s23`: `[-0.12, 0.12]`
- `t1, t2, t3`: `[-0.06, 0.06]`

## 3. Joint randomization

Each joint profile includes:

- `type`: `hinge` (revolute) or fixed
- `axis`: rotation axis
- `pos`: relative joint position in parent frame
- `euler`: orientation offsets (used for hip group)
- `range`: angular limits
- `qdot_max`: velocity scaling proxy
- `torque`: actuation scale

### 3.1 Hip group constraints

For `hip_roll`, `hip_yaw`, `hip_pitch`:

- Axes are a random permutation of canonical unit axes
- Euler offsets are sampled with zero-sum constraint over the 3-joint group
  to avoid net orientation bias

### 3.2 Joint position randomization

Joint relative position is multiplied component-wise by a uniform scale in:

- `[0.80, 1.20]`

Then it is clipped to remain within:

- `||pos|| <= 2 * parent_com_dist`

### 3.3 Angular limit scaling by group

Current range-scale multipliers:

- Shoulder: `[0.80, 1.00]`
- Waist: `[0.80, 1.00]`
- Knee: `[0.80, 1.30]`
- Ankle: `[0.80, 1.00]`
- Hip: `[0.80, 1.00]`
- Head: `[0.80, 1.00]`
- Elbow: `[0.80, 1.00]`
- Wrist: `[0.80, 1.00]`
- Default: `[0.80, 1.00]`

### 3.4 Joint orientation and actuation ranges

- Hip Euler offsets: `[-0.30, 0.30]` rad
- Torque scale factor: `[0.70, 1.00]` times total robot mass
- Revolute probability for optional groups: `0.75`

Optional `hinge/fixed` groups in code:

- Waist
- Shoulder
- Elbow
- Wrist
- Head

Leg joints remain non-optional under this grouping logic.

## 4. Visual geometry randomization

Visual box sizes are also randomized (for morphology diversity in rendering):

1. Inertial-derived affine size update from link randomization
2. Extra per-axis visual scale
3. Per-link clamps by anatomical category

Special head setup:

- `head_yaw`, `head_pitch`, `head_roll`: compact motor-like visual blocks
- `head_main`: larger final welded head body for clearer human-like head shape

## 5. Symmetry and consistency checks

- Left/right links share the same sampled profile per canonical base key
  (bilateral consistency).
- Physical consistency check ensures every link pseudo-inertia remains SPD.
- Hip-group checks enforce:
  - axis permutation validity
  - zero-sum Euler offset constraint

## 6. Code references

- Inertial randomization: `PhysicsConsistentInertia`
- Joint randomization: `JointSpaceRandomization`
- Morphology build and checks: `HumanoidBuilder`
- XML generation and rendering colors: `MuJoCoCompiler`
