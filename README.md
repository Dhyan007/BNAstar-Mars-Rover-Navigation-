# Probabilistic Terrain Assessment and Cost-Based Path Selection for a Mars Rover

This repository contains the project code extracted from the final project notebook and reorganized into Python modules for GitHub use.

**The original notebook is intentionally not included.** The project logic and the original absolute dataset/model paths have been retained.

## Pipeline

```text
AI4Mars image
     ↓
FCN-ResNet50 semantic segmentation
     ↓
Terrain probabilities
     ↓
Simulated rover IoT sensor values
     ↓
Jev safety probability + battery-cost probabilities
     ↓
Classical Bayesian Network
     ↓
Traversal success probability
     ↓
Cost = -log(P(traversal success))
     ↓
Neural-network-style search graph
     ↓
Mars rover path selection
```

## Search space

The default search-space layers are:

```python
[1, 2, 4, 8, 16, 8, 4, 2, 1]
```

Each node is connected to at most two nodes in the next layer. The visualization therefore resembles a neural network while representing the rover's directed search space.

## Project structure

```text
Mars-Rover-Probabilistic-Terrain-Assessment-Repo/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── run_project.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── model.py
    ├── segmentation.py
    ├── sensors.py
    ├── jev.py
    ├── bayesian_network.py
    ├── search_space.py
    ├── astar.py
    └── visualization.py
```

## Original paths

The paths from the final notebook have intentionally been retained in `src/config.py`:

```text
/Users/dhyan/Desktop/project/BNA*-mars-rover/ai4mars-dataset-merged-0.1/msl/images/edr
/Users/dhyan/Desktop/project/BNA*-mars-rover/ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min1-100agree
/Users/dhyan/Desktop/project/BNA*-mars-rover/ai4mars_fcn_resnet50_state_dict.pth
```

If the repository is used on another machine, these paths will need to exist there or be changed in `src/config.py`.

## API key

The Jev stage uses the TypeSafe SDK. Set the API key as an environment variable rather than committing it:

```bash
export TYPESAFE_API_KEY="your_api_key"
```

A `.env.example` file is included only as a template. **Do not commit your real API key.**

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch installation can be hardware-specific. If the standard `pip install` does not provide the desired CUDA/MPS build, install the appropriate PyTorch build for the target machine first.

## Run

From the repository root:

```bash
python run_project.py
```

The script:

1. Loads the FCN-ResNet50 model.
2. Builds the neural-network-style directed search space.
3. Randomly selects two images for each pair of candidate outgoing paths.
4. Extracts terrain probabilities.
5. Generates the simulated rover sensor values.
6. Queries Jev for safety and battery-cost probabilities.
7. Runs the classical Bayesian network.
8. Converts traversal-success probability into `-log(success)` cost.
9. Selects the lowest-cost candidate and repeats until the destination.
10. Visualizes the search space and the selected rover path.

## Important note about the implementation

The repository preserves the project's current experimental implementation. In particular, the movement cost is derived from the Bayesian-network traversal-success probability:

```python
astar_cost = -np.log(bn_traversal_success)
```

The current simulation selects the lowest movement cost among the two outgoing candidates at each step. The project uses A*-style terminology and a neural-network-shaped heuristic search space, but the current final simulation should be treated as the project's cost-based traversal prototype rather than a conventional global A* implementation with a priority queue/open set.

## GitHub upload

After checking the code:

```bash
git init
git add .
git commit -m "Initial Mars rover probabilistic navigation project"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```
