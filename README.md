# Mars Rover Autonomous Terrain Assessment and Path Planning

A pipeline that combines a terrain segmentation model, simulated rover
sensors, an LLM-based (Jev/TypeSafe) safety assessor, and a classical
Bayesian network to score candidate terrain segments, then greedily
plans a path across a binary-branching search space using an A*-style
edge cost.

## Pipeline

```
AI4Mars image
     |
     v
FCN-ResNet50 segmentation  --------->  per-class terrain probabilities
     |                                  (soil / bedrock / sand / rock / unknown)
     v
Simulated rover sensors  ------------>  slope, roughness, elevation change,
                                         distance, speed, wheel slip
     |
     v
Jev (TypeSafe System One)  ---------->  P(safe traversal), battery-cost distribution
     |
     v
Classical Bayesian network  --------->  P(traversal success)
  (Terrain -> Safety, Sensors -> Battery,
   combined with Jev evidence)
     |
     v
A* edge cost = -log(P(success))
     |
     v
Greedy path planning over a
[1,2,4,8,16,8,4,2,1]-node layered graph
```

At each step of the walk, two candidate images are sampled and pushed
through the full pipeline; the rover moves to whichever candidate has
the lower A* cost.

## Project structure

```
.
├── config.py                  # Paths, device, and search-space constants (edit or use env vars)
├── requirements.txt
├── notebooks/
│   └── final_project.ipynb    # Original notebook, unmodified
├── src/
│   ├── segmentation.py        # ImageSegmentationModel + preprocessing + scoring helpers
│   ├── sensors.py             # Simulated IMU/LiDAR/altimeter/odometry/wheel-slip readings
│   ├── jev_client.py          # Jev (TypeSafe) safety/battery-cost queries
│   ├── bayesian_network.py    # CPTs + inference chain + A* cost
│   ├── path_planning.py       # Search-space graph, visualization, rover simulation loop
│   └── pipeline.py            # Single/pairwise candidate evaluation helpers
├── scripts/
│   └── run_simulation.py      # CLI: run the full simulation, save figures to results/
├── legacy/
│   └── astar_prototype.py     # Earlier, unused A* prototype (see "Cleanup notes" below)
└── results/                   # Output figures land here (gitignored, except .gitkeep)
```

## Setup

```bash
git clone <this-repo-url>
cd mars-rover-terrain-path-planning
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### Data and model checkpoint

The pipeline expects:

1. The [AI4Mars dataset](https://ai4mars.jpl.nasa.gov/) (MSL subset), with
   `images/edr` and `labels/test/masked-gold-min1-100agree` under it.
2. A trained FCN-ResNet50 state dict for `ImageSegmentationModel`.

By default these are expected at:

```
data/ai4mars-dataset-merged-0.1/msl/
models/ai4mars_fcn_resnet50_state_dict.pth
```

Override any of these with environment variables instead of editing
`config.py`:

| Variable | Purpose |
|---|---|
| `MARS_ROVER_DATA_ROOT` | Root of the AI4Mars MSL data |
| `MARS_ROVER_IMAGES_PATH` | Overrides just the images directory |
| `MARS_ROVER_MASK_PATH_TEST` | Overrides just the test-mask directory |
| `MARS_ROVER_MODEL_PATH` | Path to the segmentation model checkpoint |
| `MARS_ROVER_RESULTS_DIR` | Where `scripts/run_simulation.py` saves figures (default `results/`) |
| `TYPESAFE_API_KEY` | Required by `typesafe-sdk` for Jev calls |

### Jev / TypeSafe

`src/jev_client.py` calls TypeSafe's Jev model via `typesafe-sdk`. Set
`TYPESAFE_API_KEY` in your environment before running anything that
touches the Bayesian network stage (it depends on Jev's output).

## Running

```bash
python scripts/run_simulation.py
```

This builds the search-space graph, loads the segmentation model, runs
the rover from the start node to the goal node, and saves
`results/search_space.png` and `results/rover_path.png`.

The original notebook (`notebooks/final_project.ipynb`) is kept
unmodified for reference and can still be run directly if you'd rather
work interactively — the sections after `run_rover_simulation` are the
ones that actually execute the full path-planning walk.

## Cleanup notes

This repo restructures a single working notebook into importable
modules. A few things worth knowing about that restructuring:

- **`add_jev_results` was reconstructed.** The notebook calls a function
  named `add_jev_results(df)` in several places (building the final
  rover simulation), but the cell that defines it isn't present in the
  exported notebook — likely lost before export. `src/jev_client.py`
  reconstructs it to match the manual, per-dataframe pattern used
  elsewhere in the notebook (the cells that build `jev_safety_probability`
  and the five `battery_*_prob` columns on `probability_df`). Worth a
  quick sanity check against your original version if you still have it
  somewhere.
- **`run_mars_rover_astar` was moved to `legacy/`.** This earlier
  prototype of the simulation loop references `select_two_images` and
  `heuristic`, neither of which are defined anywhere in the notebook, so
  it would raise a `NameError` if called. It's never actually invoked —
  the notebook's real end-to-end run uses `run_rover_simulation`
  (now in `src/path_planning.py`), which samples candidates directly and
  uses the A* cost alone with no separate heuristic term. Kept in
  `legacy/` for reference in case you want to finish it.
- **`add_sensor_data` was consolidated.** The notebook defines it three
  times as it evolves; `src/sensors.py` keeps the final, most complete
  version (the one that drops old sensor columns before regenerating
  them).
- **Device selection was unified.** The notebook picks the device
  inconsistently (`cuda`/`cpu` in one cell, `mps`/`cpu` in others).
  `config.py` now has a single `DEVICE` that checks CUDA, then MPS, then
  falls back to CPU.
- **Hardcoded local paths were parameterized.** `IMAGES_PATH`,
  `MASK_PATH_TEST`, and `MODEL_PATH` pointed at a local
  `/Users/dhyan/Desktop/...` directory; they're now in `config.py`,
  overridable via the environment variables listed above.

## License

No license file is included yet — add one (MIT, Apache-2.0, etc.) before
treating this as reusable by others.
