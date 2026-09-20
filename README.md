<div align="center">

# Bayesian Safety-Constrained A* for Mars Rover Navigation

### AI4MARS · ResNet50-U-Net · Bayesian Safety · A* Planning

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Computer%20Vision-Semantic%20Segmentation-5C3EE8" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Navigation-A*-4B5563" alt="A*">
  <img src="https://img.shields.io/badge/Bayesian-Safety%20Constraint-7C3AED" alt="Bayesian">
</p>

<p>
  A research framework for Mars rover terrain perception,
  probabilistic safety estimation, and safety-constrained navigation.
</p>

</div>

---

## At a Glance

| Component | Role |
|---|---|
| **AI4MARS** | Mars rover imagery and terrain labels |
| **ResNet50-U-Net** | Pixel-wise terrain segmentation |
| **Bayesian Network** | Probabilistic terrain safety estimation |
| **A*** | Safety-constrained path planning |
| **Receding Horizon** | Online multi-image replanning |
| **BNA* Benchmark** | Synthetic evaluation of whole-path safety planning |

> **Core objective:** minimise rover travel distance while satisfying a required path-safety constraint.

---

## Navigation Pipeline

```mermaid
flowchart LR
    A[AI4MARS Rover Image] --> B[ResNet50-U-Net]
    B --> C[Pixel-wise Terrain Probabilities]
    C --> D[16×16 Aggregation]
    D --> E[64×64 Navigation Grid]
    E --> F[Bayesian Safety Model]
    F --> G[P Safe]
    G --> H[Safety-Constrained A*]
    H --> I[Local Safe Path / Waypoint]
    I --> J[Rover State Update]
    J --> K[Next Rover Image]
    K --> B
```

The system is designed as an **online / receding-horizon navigation framework**:
the rover observes its surroundings, builds a local safety map, plans a path,
moves toward a waypoint, receives a new observation, and replans.

---

## Contents

- [Project Overview](#1-project-overview)
- [Research Objective](#2-research-objective)
- [System Architecture](#3-system-architecture)
- [Dataset](#4-dataset)
- [Deep Learning Model](#5-deep-learning-model)
- [Segmentation Output](#6-segmentation-output)
- [Navigation Grid](#7-navigation-grid)
- [Bayesian Network](#8-bayesian-network)
- [A* Path Planning](#9-a-path-planning)
- [Whole-Path Bayesian Safety](#10-whole-path-bayesian-safety)
- [Multi-Image Navigation](#12-multi-image-navigation)
- [Project Structure](#14-project-structure)
- [Installation](#16-installation)
- [Training](#18-training)
- [Running Segmentation Inference](#19-running-segmentation-inference)
- [Navigation Pipeline](#20-navigation-pipeline)
- [Outputs](#21-outputs)
- [Validation](#22-validation)
- [Synthetic BNA* Benchmark](#23-synthetic-bna-benchmark)
- [Important Limitations](#24-important-limitations)
- [Research Integrity](#25-research-integrity)
- [Planned Research Improvements](#26-planned-research-improvements)
- [Intended Research Contribution](#27-intended-research-contribution)
- [Current Status](#28-current-status)
- [Reproducibility](#29-reproducibility)
- [Summary](#30-summary)

---

<details open>
<summary><strong>What this project does</strong></summary>

The project connects **visual terrain understanding** with **probabilistic
safety reasoning** and **path planning**. The segmentation model supplies
terrain probabilities; the Bayesian model converts terrain features into a
safety estimate; A* then uses that estimate during navigation.

</details>

## 1. Project Overview

The project combines:

1. Deep learning to identify terrain types from Mars rover images.
2. Bayesian reasoning to estimate the probability that a terrain cell is safe.
3. A* search to find a path while respecting the safety constraint.
4. Multi-image / receding-horizon navigation, where the rover repeatedly
   observes its surroundings, plans a local path, moves toward a waypoint,
   receives a new image, and replans.

### Terrain Classes

- Soil
- Bedrock
- Sand
- Big rock

The project uses the AI4MARS dataset, primarily MSL rover imagery and
terrain labels.

---

## 2. Research Objective

### Main Objective

Develop an A* navigation algorithm that incorporates Bayesian Network
safety estimates for Mars rover traversal.

### Mathematical Formulation

The intended planning problem is:

$$
\min_P Distance(P)
$$

subject to:

$$
P(Safe(P)\mid X_P) \geq \tau
$$

where:

- $P$ is a candidate rover path.
- $Distance(P)$ is the path length.
- $X_P$ represents terrain observations along the path.
- $P(Safe(P)\mid X_P)$ is the Bayesian safety probability.
- $\tau$ is the required safety threshold.

Safety is therefore treated as a constraint rather than simply being
added as an arbitrary risk penalty.

---

## 3. System Architecture

```text
AI4MARS Rover Image
        |
        v
ResNet50-U-Net
        |
        v
Pixel-wise Terrain Probabilities
        |
        v
16 x 16 Pixel Aggregation
        |
        v
64 x 64 Navigation Grid
        |
        v
Bayesian Network
        |
        v
P(Safe | Terrain)
        |
        v
Safety-Constrained A*
        |
        v
Local Safe Path / Waypoint
        |
        v
Rover State Update
        |
        v
Next Rover Image
        |
        v
Replanning
        |
        v
Final Route
```

The multi-image version is designed as an **online/receding-horizon
navigation system**.

---

## 4. Dataset

Expected AI4MARS MSL structure:

```text
msl/
├── images/
│   ├── edr/
│   ├── mxy/
│   └── rng-30m/
└── labels/
    ├── train/
    └── test/
        ├── masked-gold-min1-100agree/
        ├── masked-gold-min2-100agree/
        └── masked-gold-min3-100agree/
```

The primary image source used by the segmentation pipeline is:

```text
images/edr/
```

### Terrain Classes

| ID | Terrain |
|---:|---|
| 0 | Soil |
| 1 | Bedrock |
| 2 | Sand |
| 3 | Big rock |
| 255 | Ignore / unlabeled |

The value `255` is excluded from training and evaluation.

`Unknown` is treated as a fifth terrain category representing terrain
that cannot be confidently assigned to the four known AI4MARS terrain
classes.

It is conceptually different from `255`:

- `Unknown` is a valid model output.
- `255` remains the ignore/unlabeled value used by the dataset.

---

## 5. Deep Learning Model

<div align="center">

**ResNet50 Encoder → U-Net Decoder → Terrain Probability Map**

</div>

The terrain segmentation stage uses:

**ResNet50 encoder + U-Net decoder**

The encoder uses ImageNet initialization.

### Training Strategy

#### Stage 1 — Frozen Encoder

The ResNet50 encoder is frozen while the U-Net decoder and segmentation
head are trained.

#### Stage 2 — Fine-tuning

Later ResNet50 stages are unfrozen and fine-tuned using a smaller
learning rate.

The model is trained using PyTorch and CUDA.

### Image Resolution

Default image size:

```text
512 x 512
```

Native AI4MARS image representation:

```text
1024 x 1024
```

### Best Checkpoint

```text
checkpoints/best.pt
```

### Training History

```text
outputs/training/
```

---

## 6. Segmentation Output

The navigation system retains the softmax probability distribution for
every pixel:

```text
P(soil)
P(bedrock)
P(sand)
P(big_rock)
```

For example:

```text
soil       = 0.62
bedrock    = 0.08
sand       = 0.25
big_rock   = 0.05
```

This uncertainty is retained instead of immediately reducing the pixel
to only its most likely class.

---

## 7. Navigation Grid

The native image is converted into a lower-resolution navigation grid:

```text
Native image:       1024 x 1024
Cell size:          16 x 16 pixels
Navigation grid:    64 x 64
```

Each navigation cell contains averaged terrain probabilities.

The resulting grid has:

```text
64 x 64 x 4
```

### Additional Features

- Dominant terrain
- Purity
- Entropy
- Confidence

---

## 8. Bayesian Network

> **Safety is treated as a constraint, not merely as an arbitrary penalty added to the path cost.**

The main Bayesian calculation is implemented in:

```text
bayesian_safety.py
```

### Main Class

```python
TerrainBayesianSafety
```

### Main Calculation

```python
score_grid()
```

The model estimates:

$$
P(Safe \mid TerrainFeatures)
$$

The current implementation uses a Naive-Bayes-style structure in which
terrain feature nodes are conditionally independent given the safety
state.

Conceptually:

```text
             Safe
          /   |   |   \
       Soil Bedrock Sand BigRock
             |
           Purity
```

The model uses:

```text
P(feature_level | Safe)
P(feature_level | Unsafe)
P(Safe)
P(Unsafe)
```

and applies Bayes' rule.

### Feature States

```text
0 = Low
1 = Medium
2 = High
```

### Current Features

```text
soil
bedrock
sand
big_rock
purity
```

### Important Limitation

The current Bayesian safety model is an **explicit expert-prior model**.

AI4MARS provides terrain labels but does not directly provide rover
traversability, slip, stall, or terrain-safety ground truth.

Therefore the current Bayesian probabilities should **not** be described
as NASA-calibrated traversability probabilities.

---

## 9. A* Path Planning

<div align="center">

**Shortest feasible route under a safety requirement**

</div>

The project contains:

```text
Dijkstra
A*
Bayesian Safety-Constrained A*
```

Standard A* uses:

$$
f(n)=g(n)+h(n)
$$

The current grid-level Bayesian planner allows cells satisfying:

```text
P(Safe) >= safety_threshold
```

### Default Configuration

```python
DEFAULT_SAFETY_THRESHOLD = 0.80
```

The operational single-grid planner is implemented in:

```text
astar.py
```

A separate research prototype, `bna_star_whole_path.py`, explores the
more rigorous whole-path probability formulation.

---

## 10. Whole-Path Bayesian Safety

For a path containing cells $1,\ldots,k$, a simple
conditional-independence formulation is:

$$
P(S_1,\ldots,S_k\mid X_1,\ldots,X_k)
=
\prod_{i=1}^{k}P(S_i\mid X_i)
$$

A sequential formulation can instead use:

$$
P(S_1\mid X_1)
\prod_{i=2}^{k}P(S_i\mid S_{i-1},X_i)
$$

Because multiplying many probabilities can produce very small values,
the research prototype uses log probabilities:

$$
\log P(path)=\sum_i \log P(S_i\mid\cdots)
$$

This is numerically more stable.

---

## 11. Why Multiple Labels May Be Required in A*

Two partial paths can reach the same node with different distance/safety
trade-offs:

```text
Path A:
distance = 20 m
safety   = 0.02

Path B:
distance = 23 m
safety   = 0.40
```

The longer path cannot automatically be discarded because it may be the
only partial path that can eventually satisfy the whole-path safety
constraint.

A rigorous whole-path implementation can therefore maintain multiple
labels containing:

```text
distance
log probability
battery
node
parent
```

and remove labels using Pareto-dominance rules.

---

## 12. Multi-Image Navigation

The intended final system repeatedly plans from new rover observations:

```text
Image 1
   |
   v
Local safety map
   |
   v
Local candidate paths
   |
   v
Select safe waypoint
   |
   v
Update rover state

Image 2
   |
   v
Local safety map
   |
   v
Local candidate paths
   |
   v
Select next waypoint

...

Final image
   |
   v
Final endpoint
```

Consecutive images do **not** need to contain overlapping visual
landmarks if appropriate navigation state is available.

### Useful State

```text
current rover position
current rover heading
goal bearing
goal position, if available
```

This makes the system a **receding-horizon navigation system**.

If only goal direction is known and there is no global metric pose, the
system can replan toward the goal direction but cannot claim globally
optimal route planning.

---

## 13. Local Candidate Paths

The multi-image system can generate several local candidate paths, for
example:

```text
Forward
Forward-left
Forward-right
```

or angular sectors around the goal bearing.

Candidates are evaluated using:

1. Safety constraint
2. Progress toward the goal
3. Local travel distance
4. Safety as a secondary criterion / tie-breaker

The selected local path ends at a waypoint. After the rover moves toward
that waypoint, the next image is processed and the route is replanned.

---

## 14. Project Structure

The repository is organised around four main layers:

| Layer | Purpose |
|---|---|
| **Perception** | Dataset loading, segmentation, inference and metrics |
| **Safety** | Grid features and Bayesian safety estimation |
| **Planning** | Dijkstra, A*, safety-constrained A* and whole-path research |
| **Evaluation** | Validation, trade-off experiments, checkpoints and outputs |

```text
bayesian A/
│
├── config.py
├── dataset.py
├── model.py
├── train.py
├── metrics.py
├── inference.py
│
├── grid_features.py
├── bayesian_safety.py
├── astar.py
├── navigation_pipeline.py
│
├── rover_state.py
├── candidate_paths.py
├── local_planner.py
├── route_manager.py
├── range_geometry.py
├── navigation_sequence.py
│
├── checkpoints/
│   └── best.pt
│
├── outputs/
│   ├── training/
│   ├── navigation/
│   └── validation/
│
├── datasets/
│   └── ai4mars-dataset-merged-0.1/
│
└── README.md
```

The multi-image modules are part of the planned/ongoing extension and
may not yet exist in the current repository.

---

## 15. Main Files

| File | Purpose |
|---|---|
| `config.py` | Shared configuration and terrain classes |
| `dataset.py` | AI4MARS image/mask loading and preprocessing |
| `model.py` | ResNet50-U-Net architecture |
| `train.py` | Model training and validation |
| `metrics.py` | Segmentation metrics |
| `inference.py` | Model inference |
| `grid_features.py` | Converts pixel probabilities into navigation cells |
| `bayesian_safety.py` | Bayesian terrain-to-safety calculation |
| `astar.py` | Dijkstra, A*, and safety-constrained A* |
| `navigation_pipeline.py` | Single-image end-to-end navigation pipeline |
| `bna_star_whole_path.py` | Whole-path Bayesian A* research prototype |
| `rover_state.py` | Planned rover navigation state |
| `candidate_paths.py` | Planned local candidate path generation |
| `local_planner.py` | Planned local path planning |
| `route_manager.py` | Planned route/waypoint management |
| `navigation_sequence.py` | Planned multi-image navigation runner |
| `range_geometry.py` | Planned image-to-ground/range geometry utilities |

---

## 16. Installation

### Typical Dependencies

```text
Python
PyTorch
TorchVision
segmentation-models-pytorch
NumPy
Pandas
Pillow
OpenCV
Matplotlib
scikit-learn
```

### Check CUDA

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## 17. Dataset Configuration

### Windows PowerShell

```powershell
$env:AI4MARS_MSL_ROOT="C:\path\to\ai4mars-dataset-merged-0.1\msl"
```

The directory should contain:

```text
images\edr
labels\train
```

---

## 18. Training

### Train the Model

```powershell
python train.py --data-root "$env:AI4MARS_MSL_ROOT"
```

### Useful Options

```text
--batch-size
--epochs-frozen
--epochs-finetune
--lr-frozen
--lr-finetune
--val-fraction
--num-workers
--checkpoint-dir
--history-dir
--max-images
--no-amp
```

### Quick Smoke Test

```powershell
python train.py --data-root "$env:AI4MARS_MSL_ROOT" --max-images 100
```

### Best Model

```text
checkpoints/best.pt
```

---

## 19. Running Segmentation Inference

### Single Image

```powershell
python inference.py `
    --checkpoint checkpoints\best.pt `
    --input path\to\image.png
```

### Folder

```powershell
python inference.py `
    --checkpoint checkpoints\best.pt `
    --input path\to\images `
    --out outputs\segmentation_results.csv
```

For navigation, preserve the spatial probability map rather than using
only an image-level summary.

---

## 20. Navigation Pipeline

The single-image pipeline is:

```text
AI4MARS image
      ↓
ResNet50-U-Net
      ↓
Terrain probabilities
      ↓
64 × 64 grid
      ↓
Bayesian safety map
      ↓
Dijkstra / A* / Bayesian Safety A*
```

The pipeline uses:

- Input image
- Trained checkpoint
- Start cell
- Goal cell
- Safety threshold
- Output directory

Use the exact command-line options exposed by the current
`navigation_pipeline.py`.

---

## 21. Outputs

The pipeline can produce:

### Terrain Information

```text
soil_pct
bedrock_pct
sand_pct
big_rock_pct
```

### Grid Statistics

```text
dominant terrain
purity
entropy
confidence
```

### Bayesian Safety

```text
safety_probability
safety_score
```

### Path Metrics

```text
path length
mean safety
minimum safety
unsafe fraction
```

### Visualisations

```text
Original AI4MARS image
Dominant terrain
Bayesian safety map
Bayesian Safety-Constrained A* path
```

---

## 22. Validation

### Segmentation

Evaluate:

- Pixel accuracy
- Per-class IoU
- Mean IoU
- Confusion matrix

### Bayesian Safety

When real traversability ground truth becomes available, evaluate:

- Calibration
- Brier score
- Reliability diagrams
- MAE / RMSE
- Threshold sensitivity

### Path Planning

Compare:

```text
Dijkstra
A*
Bayesian Safety-Constrained A*
```

using:

- Path distance
- Path safety
- Minimum cell safety
- Whole-path safety probability
- Constraint violations
- Expanded nodes
- Planning time
- Feasibility rate

Where possible, include an oracle/reference shortest-safe-path solution.

---

## 23. Synthetic BNA* Benchmark

A synthetic benchmark has been developed to test the path-planning
concept before real traversability ground truth is available.

### Important Files

```text
bna_tradeoff_training.csv
bna_tradeoff_planner_input.csv
bna_tradeoff_test_maps.csv
bna_tradeoff_map_summary.csv
bna_tradeoff_design.json
bna_tradeoff_validation.py
analyze_bna_tradeoff_results.py
bna_star_whole_path.py
```

The benchmark is useful for:

- Debugging
- Safety-constraint testing
- Whole-path probability testing
- Threshold sweeps
- Infeasibility verification
- Distance/safety trade-off experiments

> Synthetic results must not be presented as direct evidence of real Mars rover performance.

---

## 24. Important Limitations

### Terrain Is Not Traversability

A terrain class such as sand or bedrock does not directly establish
whether a rover can safely traverse it.

The current BN is therefore a research prior, not a validated physical
traversability model.

### Battery

AI4MARS does not directly provide battery consumption for every
candidate path.

Battery-aware planning should only be used with a justified energy model
or rover telemetry.

### Metric Distance

A 64×64 grid is not automatically a metric map.

Physical distances require appropriate camera calibration, range/depth,
ground geometry, rover pose, or equivalent metric information.

### Range Data

AI4MARS contains range-related resources, but individual range files
must be inspected before treating them as metric depth.

A `rng-30m` mask should not automatically be interpreted as a full
metric depth map.

### Multi-Image Navigation

The receding-horizon system requires rover navigation state.

Without pose or an appropriate coordinate relationship between
observations, local paths from separate images cannot simply be
concatenated into one metric global path.

---

## 25. Research Integrity

The project separates three types of information.

### Learned from Real AI4MARS Labels

```text
Terrain segmentation
```

### Expert / Model Assumptions

```text
Bayesian terrain-to-safety probabilities
```

### Synthetic Research Data

```text
BNA* benchmark
```

This distinction should be maintained in reports, presentations, and
publications.

---

## 26. Planned Research Improvements

1. Calibrate the Bayesian Network using appropriate traversability data.
2. Use a rigorous whole-path safety constraint.
3. Implement multi-label safety-constrained A*.
4. Implement receding-horizon multi-image navigation.
5. Track rover position, heading, goal direction, and waypoints.
6. Integrate appropriate range/depth and camera geometry.
7. Add oracle, ablation, threshold, and planning-time experiments.
8. Evaluate the complete multi-image route rather than only individual
   images.

---

## 27. Intended Research Contribution

The intended contribution is the integration of:

```text
Mars terrain perception
        +
probabilistic safety estimation
        +
safety-constrained path planning
        +
online multi-image replanning
```

The central idea is to use probabilistic terrain perception to estimate
local safety, then incorporate those estimates into path planning so
that rover navigation considers both distance and uncertainty.

---

## 28. Current Status

| Status | Scope |
|---|---|
| **Implemented** | Core perception, Bayesian safety, A*, visualisation and synthetic BNA* tooling |
| **In progress / planned** | Whole-path constraints, rover state, multi-image navigation, geometry and calibration |

### Implemented

- AI4MARS dataset loading
- Four-class terrain segmentation
- ResNet50-U-Net model
- Two-stage training
- CUDA/AMP support
- Segmentation metrics
- Spatial softmax probability maps
- 16×16 → 64×64 navigation grid
- Bayesian terrain-to-safety model
- Dijkstra
- Standard A*
- Bayesian safety-constrained A*
- Navigation visualisation
- Synthetic whole-path Bayesian A* prototype
- Synthetic BNA* validation tools

### In Progress / Planned

- Rigorous whole-path safety-constrained A*
- Rover state representation
- Local candidate path generation
- Receding-horizon multi-image navigation
- Route management
- Metric range/geometry integration
- Real traversability calibration
- Stronger benchmark and ablation experiments

---

## 29. Reproducibility

```text
Random seed:              42
Image size:               512 x 512
Native image size:        1024 x 1024
Navigation cell size:     16 x 16 pixels
Navigation grid:          64 x 64
Classes:                  5
Terrain classes:          soil, bedrock, sand, big_rock, unknown
Ignore index:             255
Default safety threshold: 0.80
```

---

## 30. Summary

The project develops a probabilistic Mars rover navigation pipeline:

```text
Mars image
   ↓
Terrain segmentation
   ↓
Terrain probability map
   ↓
Navigation grid
   ↓
Bayesian safety probability
   ↓
Safety-constrained A*
   ↓
Safe local path
   ↓
Waypoint
   ↓
New image
   ↓
Replanning
```

The final research direction is an **online Bayesian safety-constrained
A* navigation framework for Mars rover terrain traversal**, with the
objective of finding short routes while maintaining a specified safety
requirement.

---

<div align="center">

### Bayesian Safety-Constrained Mars Rover Navigation

**Terrain perception → Probabilistic safety → Constrained planning → Replanning**

</div>
