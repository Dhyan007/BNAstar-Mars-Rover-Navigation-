# 🚀 Probabilistic Terrain Assessment & Cost-Based Path Selection for a Mars Rover

<p align="center">
  <img src="docs/pipeline.svg" alt="Mars Rover probabilistic navigation pipeline" width="900">
</p>

<p align="center"><b>Semantic Segmentation → Simulated Rover Sensors → Jev Safety & Battery Assessment → Bayesian Fusion → Cost-Based Search</b></p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/AI4Mars-Terrain%20Segmentation-6A5ACD">
<img src="https://img.shields.io/badge/NetworkX-Graph%20Search-2C8EBB?logo=networkx&logoColor=white">
<img src="https://img.shields.io/badge/TypeSafe%20Jev-Probabilistic%20Assessment-111827">
<img src="https://img.shields.io/badge/Status-Research%20Prototype-orange">
</p>

---

## 📌 Overview

This project implements a **probabilistic Mars-rover terrain assessment and cost-based path-selection pipeline**. Instead of making a navigation decision directly from a terrain image, the system propagates uncertainty through several stages:

1. A **ResNet50-FCN semantic segmentation model** evaluates a terrain image.
2. Pixel-wise predictions become image-level probabilities for **soil, bedrock, sand, big rock, and unknown**.
3. Simulated rover sensors generate **slope, roughness, elevation change, distance, speed, and wheel slip**.
4. **Jev / TypeSafe System One** evaluates safety and battery-cost probabilities.
5. A **custom classical Bayesian network** fuses the evidence into a traversal-success probability.
6. Traversal success becomes an additive movement cost:

\[
c(e)=-\log(\max(P_{succ}(e),10^{-6}))
\]

7. At each decision point, two distinct terrain images are sampled and evaluated. The candidate with the lower traversal cost is selected.
8. The process continues through a layered binary-branching search graph until the goal is reached.

> **Important:** The current implementation is a **greedy two-way cost-based search**, not textbook A*. Full A* maintains an open set and evaluates `f(n)=g(n)+h(n)`. This project commits to a candidate after each local comparison and does not backtrack.

---

## 🧠 System Architecture

<p align="center"><img src="docs/pipeline.svg" alt="Detailed system architecture" width="1000"></p>

```text
Terrain Image
     │
     ▼
ResNet50-FCN
     │
     ▼
Terrain Class Probabilities
     │
     ▼
Simulated Rover Sensors
     │
     ▼
Jev Safety + Battery Assessment
     │
     ▼
Custom Bayesian Network
     │
     ▼
Traversal Success Psucc
     │
     ▼
Cost = -log(Psucc)
     │
     ▼
Compare 2 Candidates
     │
     ▼
Next Node
```

---

## 🕸️ Search Space

The navigation graph uses a neural-network-style layered layout:

```text
[1, 2, 4, 8, 16, 8, 4, 2, 1]
```

| Property | Value |
|---|---:|
| Layers | 9 |
| Nodes | 46 |
| Edges | 88 |
| Start node | 0 |
| Goal node | 45 |
| Maximum branching | 2 |
| Moves from start to goal | 8 |
| Candidate images per normal decision | 2 |

<p align="center"><img src="docs/search_space.svg" alt="Neural-network-style Mars rover search graph" width="1000"></p>

**Note:** The neural-network appearance is purely a visualization of the search graph. The graph itself is **not a neural network**.

---

## 🔬 Core Components

### 1. ResNet50-FCN Terrain Segmentation

The perception module uses a ResNet50-backed fully convolutional segmentation model. The five classes are:

| Class | Meaning |
|---|---|
| `soil` | Soil terrain |
| `bedrock` | Bedrock terrain |
| `sand` | Sand terrain |
| `rock` | Big-rock terrain |
| `unknown` | Unlabelled / unknown region |

Image-level class probabilities are computed from the spatial mean of pixel-wise softmax probabilities. This is an **image-level average**, not a guarantee that the entire image belongs to one class.

### 2. Simulated Rover Sensors

Because real rover telemetry is unavailable in this prototype, sensor readings are generated from terrain probabilities with bounded random variation.

| Sensor | Purpose |
|---|---|
| `slope_deg` | Simulated IMU / inclinometer |
| `distance_m` | Rover odometry |
| `roughness` | Simulated LiDAR / depth sensing |
| `elevation_change_m` | Simulated altimeter / DEM |
| `speed_mps` | Simulated wheel odometry |
| `wheel_slip` | Wheel encoder + IMU estimate |

The candidate represents a fixed **10 m terrain segment**. These values are simulation variables and are not calibrated against real rover telemetry.

### 3. Jev Safety & Battery Assessment

Jev receives a structured state containing terrain probabilities, simulated rover sensors, and mission context.

| Output | Description |
|---|---|
| `safety_probability` | Probability that the candidate is safely traversable |
| `battery_cost_probabilities` | Ordinal probability distribution over expected battery use |
| `battery_cost_score` | Derived battery-cost representation |

The battery output is an **ordinal probability distribution**, not a physical energy measurement in watt-hours.

### 4. Custom Bayesian Network

The project uses a classical Bayesian network so that the inference logic and conditional probability tables remain inspectable.

```text
Terrain Risk ─────► Safety ───────┐
                                  │
                                  ▼
                            Traversal Success
                                  ▲
                                  │
Energy Risk ───────► Battery ─────┘
```

The variables are terrain risk `T`, energy risk `E`, safety `S`, battery cost `B`, and traversal outcome `Y`.

The final traversal probability is:

\[
P_{succ}=\sum_s\sum_b P(S=s)P(B=b)P(Y=Success\mid S=s,B=b)
\]

---

## 💰 Cost Function

The planner uses **only the Bayesian-network traversal-success probability** as the movement-cost input:

\[
\boxed{c(e)=-\log(\max(P_{succ}(e),10^{-6}))}
\]

| Traversal success | Approx. cost |
|---:|---:|
| 0.99 | 0.010 |
| 0.95 | 0.051 |
| 0.90 | 0.105 |
| 0.80 | 0.223 |
| 0.50 | 0.693 |
| 0.20 | 1.609 |

**Higher traversal success → lower cost → preferred candidate.**

For independent moves, route cost is additive:

\[
C(\pi)=\sum_k c(e_k)=-\log\left(\prod_k P_{succ}(e_k)\right)
\]

---

## 🔁 Candidate Selection

At every non-terminal node:

```text
Current Node
     │
     ├──── Candidate 1 ────► ResNet50-FCN ─► Sensors ─► Jev ─► BN ─► Cost 1
     │
     └──── Candidate 2 ────► ResNet50-FCN ─► Sensors ─► Jev ─► BN ─► Cost 2
                                      │
                                      ▼
                              Select lower cost
                                      │
                                      ▼
                                  Next Node
```

Each selected edge records the node, image index/name, traversal-success probability, and movement cost.

---

## 📊 Illustrative Results

One complete illustrative run reached the goal using:

```text
0 → 1 → 4 → 10 → 21 → 34 → 40 → 43 → 45
```

| Metric | Result |
|---|---:|
| Start node | 0 |
| Goal node | 45 |
| Moves | 8 |
| Accumulated cost | 1.871 |
| Approx. joint success | 0.154 |
| Mean selected per-move success | 0.795 |

These results are **illustrative rather than a statistical benchmark**.

### Example perception / fusion table

| Image | Soil | Bedrock | Sand | Big Rock | Unknown | Jev Safe | Fused P(Success) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 78 | 0.467 | 0.000 | 0.000 | 0.000 | 0.533 | 0.48 | 0.862 |
| 303 | 0.621 | 0.000 | 0.000 | 0.000 | 0.379 | 0.72 | 0.928 |
| 257 | 0.000 | 0.430 | 0.001 | 0.000 | 0.570 | 0.37 | 0.761 |
| 64 | 0.607 | 0.000 | 0.000 | 0.000 | 0.393 | 0.58 | 0.889 |
| 246 | 0.000 | 0.004 | 0.149 | 0.001 | 0.845 | 0.38 | 0.805 |

---

## 🗂️ Repository Structure

```text
Mars-Rover-Probabilistic-Terrain-Assessment-Repo/
│
├── run_project.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py
│   ├── model.py
│   ├── dataset.py
│   ├── segmentation.py
│   ├── sensors.py
│   ├── jev.py
│   ├── bayesian_network.py
│   ├── graph.py
│   ├── astar.py
│   └── visualization.py
│
└── docs/
    ├── pipeline.svg
    └── search_space.svg
```

> Keep the actual source filenames and configured dataset/model paths unchanged when integrating this README with your existing repository.

---

## ⚙️ Installation

### 1. Clone

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Mars-Rover-Probabilistic-Terrain-Assessment-Repo
```

### 2. Create the environment

```bash
conda create -n mars-rover python=3.11
conda activate mars-rover
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

### 5. Verify the active interpreter

```bash
python -c "import sys; print(sys.executable)"
```

Expected path:

```text
.../anaconda3/envs/mars-rover/bin/python
```

---

## 🔐 Jev / TypeSafe API

Do **not** hard-code API keys in the repository.

On macOS/Linux, for example:

```bash
export TYPESAFE_API_KEY="YOUR_API_KEY"
```

Verify configuration without printing the secret:

```bash
python -c "import os; print('API key configured:', bool(os.getenv('TYPESAFE_API_KEY')))"
```

If the existing project configuration uses a different environment-variable name, keep that name unchanged.

---

## ▶️ Running

From the repository root:

```bash
conda activate mars-rover
python run_project.py
```

Expected high-level flow:

```text
Load configuration
      ↓
Load ResNet50-FCN
      ↓
Build search graph
      ↓
Select two image candidates
      ↓
Extract terrain probabilities
      ↓
Generate sensor probabilities
      ↓
Query Jev
      ↓
Run Bayesian network
      ↓
Calculate traversal cost
      ↓
Choose lower-cost candidate
      ↓
Move to next node
      ↓
Repeat until goal
```

---

## 🎨 Visualization

The search-space visualization intentionally resembles a neural network. Nodes can display the **node ID, image index and cost**, while the selected route is highlighted so the rover's decision sequence is traceable.

This lets a movement be followed from:

**image → segmentation → sensors → Jev → Bayesian network → P(success) → cost → selected edge**.

---

## 🧪 Reproducibility

| Component | Randomness |
|---|---|
| Image candidate selection | Random sampling |
| Sensor generation | NumPy-based simulation |
| Segmentation inference | Determined by model/input |
| Bayesian inference | Deterministic for fixed input |
| Jev | External model/API response |

The reported simulation is illustrative, so an exact rerun can select different images and produce a different route.

---

## ⚠️ Limitations

- **Synthetic sensors:** slope, roughness, elevation change, speed and wheel slip are simulated.
- **Hand-designed CPTs:** Bayesian-network probabilities are engineering assumptions, not learned traversal statistics.
- **Jev is not a physics simulator:** its outputs are probabilistic assessments of the supplied state.
- **Image-level averaging:** spatial averaging can hide small local obstacles.
- **Random edge imagery:** images are sampled for candidate decisions rather than representing a fixed physical map.
- **Greedy search:** the current implementation is not full A* and does not maintain an open list or backtrack.
- **Limited evaluation:** repeated trials, baselines, statistical testing and probability calibration are still required.

---

## 🚧 Future Work

- [ ] Replace synthetic sensors with real rover telemetry
- [ ] Add a physics-based rover simulator
- [ ] Learn Bayesian-network CPTs from traversal data
- [ ] Calibrate traversal-success probabilities
- [ ] Preserve spatial correspondence between images and graph edges
- [ ] Use segmentation masks to identify local obstacles and corridors
- [ ] Add localization uncertainty and dynamic battery state
- [ ] Implement full A* with `f(n)=g(n)+h(n)`
- [ ] Add Dijkstra / A* / greedy baselines
- [ ] Run repeated Monte-Carlo simulations
- [ ] Report confidence intervals and statistical comparisons
- [ ] Evaluate segmentation with IoU / Dice / per-class metrics

---

## 📚 Technologies

| Technology | Role |
|---|---|
| **Python** | Main implementation language |
| **PyTorch** | Deep-learning inference |
| **Torchvision** | ResNet50-FCN segmentation |
| **OpenCV** | Image loading / preprocessing |
| **NumPy** | Numerical operations and sensor simulation |
| **Pandas** | Probability and sensor tables |
| **NetworkX** | Search-graph construction |
| **Matplotlib** | Visualization |
| **TypeSafe / Jev** | External probabilistic assessment |
| **Conda** | Environment management |

---

## 📖 References

- Elian Belot, **ResNet50 Segmentation**, GitHub: https://github.com/ElianBelot/resnet50-segmentation
- C. Genest and J. V. Zidek, “Combining probability distributions: A critique and an annotated bibliography,” *Statistical Science*, vol. 1, no. 1, pp. 114–135, 1986.
- **AI4Mars** terrain-segmentation benchmark and associated Mars rover imagery.
- Project paper: **Probabilistic Terrain Assessment and Cost-Based Path Selection for a Mars Rover: Fusing Semantic Segmentation, Jev Safety Assessment, and a Bayesian Network.**

---

## ⭐ Key Takeaway

The central contribution is the **traceable connection between perception, uncertainty, probabilistic reasoning and navigation**:

\[
\boxed{
\text{Image}
\rightarrow
\text{Terrain Probability}
\rightarrow
\text{Sensors}
\rightarrow
\text{Jev}
\rightarrow
\text{Bayesian Fusion}
\rightarrow
P_{succ}
\rightarrow
\text{Cost}
\rightarrow
\text{Path Decision}
}
\]

> **A terrain image becomes a probabilistic traversal decision.**

The architecture is modular, so future versions can replace simulated sensors, hand-designed probability tables, Jev assessment, or the greedy search rule without redesigning the entire pipeline.

<p align="center"><i>Research prototype for probabilistic terrain-aware Mars rover navigation.</i></p>
