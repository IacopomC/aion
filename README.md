# Aion - Temporal Dynamics for 3D Scene Graphs

[![Static Badge](https://img.shields.io/badge/-arXiv-B31B1B?logo=arxiv)](https://arxiv.org/abs/2512.11903)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://iacopomc.github.io/aion)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository is the official implementation of the paper:

> **Aion: Towards Hierarchical 4D Scene Graphs with Temporal Flow Dynamics**
>
> [Iacopo Catalano](https://scholar.google.com/citations?hl=en&user=VnPwRvkAAAAJ&view_op=list_works&sortby=pubdate), [Eduardo Montijano](https://scholar.google.com/citations?hl=en&user=KD6ysH8AAAAJ&view_op=list_works&sortby=pubdate), [Javier Civera](https://scholar.google.com/citations?hl=en&user=j_sMzokAAAAJ&view_op=list_works&sortby=pubdate), [Julio Placed](https://scholar.google.com/citations?hl=en&user=1ho6W5EAAAAJ&view_op=list_works&sortby=pubdate), and [Jorge Pena-Queralta](https://scholar.google.com/citations?hl=en&user=J1SHJeMAAAAJ&view_op=list_works&sortby=pubdate). <br>
> 
> *arXiv preprint arXiv:2512.11903*, 2025 <br>
> (Accepted for *IEEE International Conference on Robotics and Automation (ICRA)*, 2026.)

**Aion** adds temporal flow modeling to [Hydra](https://github.com/MIT-SPARK/Hydra)'s 3D scene graph. It learns recurring human movement patterns through navigational place nodes using the [FreMEn](https://github.com/gestom/fremen) framework, and provides predictions that can be used for time-aware path planning.

> **Architecture**: Aion runs as a standalone ROS node that subscribes to Hydra's published scene graph — it does not modify Hydra itself.

## Documentation

| Document | Contents |
|----------|----------|
| **[Quick Start](#quick-start)** | Installation and first run |
| **[Configuration Guide](doc/configuration.md)** | All parameters explained |
| **[API Reference](doc/api-reference.md)** | Topics, services, and message types |
| **[Usage Examples](doc/usage-examples.md)** | Service calls, export tools, and A\* planning |
| **[Understanding Aion](doc/understanding-aion.md)** | Concepts, architecture, and theory |
| **[Visualization Guide](doc/temporal_markers_guide.md)** | Marker encoding and RViz setup |
| **[Benchmark suite](https://github.com/IacopomC/aion_benchmark)** | Standalone repo: evaluation scripts and flow-aware routing |

The evaluation and routing scripts live in a separate repository,
[`aion_benchmark`](https://github.com/IacopomC/aion_benchmark) — clone it
alongside this repo to reproduce the paper analysis or run the routing evaluation.

## Installation

### Prerequisites

- **ROS Noetic** on Ubuntu 20.04
- **Hydra** (ROS 1 version) — **must be built first**. Follow the [Hydra installation instructions](https://github.com/MIT-SPARK/Hydra). This provides the required `hydra`, `spark_dsg`, and `hydra_msgs` packages.
- A **people detection system** publishing `geometry_msgs/PoseArray` (positions and facing orientations of detected people)

> **Build order**: Hydra and spark_dsg **must** be built before Aion. Aion's CMakeLists uses `find_package(hydra REQUIRED)` and `find_package(spark_dsg REQUIRED)`, and the C++ code includes Hydra's spatial-search utilities directly. The FremenArray dependency is included as a git submodule inside this repository and gets built alongside Aion automatically.

### Build

```bash
# Create workspace (if not exists)
mkdir -p ~/aion_ws/src && cd ~/aion_ws/src

# Clone repository with submodules
git clone --recursive https://github.com/IacopomC/aion.git

# If you forgot --recursive:
cd aion && git submodule update --init --recursive && cd ..

# Build (Hydra packages must already be in the workspace or installed)
cd ~/aion_ws && catkin build

# Source workspace
source devel/setup.bash
```

> **GTSAM / TBB note (required for loop-closure / LCD-on runs).** The packaged GTSAM is built
> **with TBB** (`GTSAM_USE_TBB`). Its parallel multifrontal elimination races on the dense
> deformation-graph structure produced under many loop closures and crashes nondeterministically
> (`map::at` / segfault inside `OptimizeClique`). Fix: serialize that solve by adding, at the top
> of `kimera_rpgo/src/RobustSolver.cpp::optimize()`,
> `tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 1);` (plus
> `#include <tbb/global_control.h>`). Zero measurable cost (the solve is a few ms). **The Docker
> build applies this automatically** (`docker/Dockerfile` patches the pinned Kimera-RPGO clone);
> for a native build, apply it by hand — Kimera-RPGO is upstream MIT-SPARK, so it's a local patch.
> NB: `TBB_NUM_THREADS=1` does **not** work; only `tbb::global_control` constrains GTSAM's arena.

## Quick Start

```bash
# Terminal 1: Start Hydra
roslaunch hydra_ros your_hydra_launch.launch

# Terminal 2: Start Aion (also launches the FremenArray action server)
roslaunch aion temporal_dynamics.launch

# Terminal 3: Play data
rosbag play --clock your_data.bag
```

The launch file starts both the FremenArray action server and the Aion temporal dynamics node. The system begins learning temporal patterns automatically as soon as it receives scene-graph updates and people detections.

### Key Outputs

| Output | Type | Description |
|--------|------|-------------|
| `/aion/temporal_dynamics` | `MarkerArray` | RViz markers encoding entropy, flow, and activity |
| `/aion/temporal_map` | `AionTemporalMap` | Structured temporal map for programmatic use |
| `/aion/temporal_nodes` | `AionTemporalNode` | Per-node prediction data |
| `/aion/historical_nodes` | `AionHistoricalNode` | Per-node historical data |
| `/aion/get_prediction` | Service | Single-place temporal prediction |
| `/aion/get_all_predictions` | Service | All-places temporal predictions |
| `/aion/export_navigation_data` | Service | Export graph + temporal data to JSON |

**Visualization**: Markers are colored by entropy (blue = predictable, red = chaotic), sized by activity level, and opacity encodes confidence. In `"bins"` mode, all 8 direction bins are shown as individual scaled arrows per place.

## Python (no-ROS) Workflow

In addition to the ROS-based real-time node, Aion ships a **file-driven Python workflow** that runs offline against pre-recorded scenes (works on any `FileDataLoader`-format scene).

The Python workflow has **no ROS dependency** and it embeds the FreMEn solver (`libfremen`) directly in-process, drives Hydra through its own Python
bindings, and persists state to disk via a `--save-flow-state` / `--load-flow-state` flag pair so consecutive sessions accumulate as if the robot had explored continuously.

### Build

```bash
# Build workspace
catkin init
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build
source devel/setup.bash

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip # upgrade for pip 25

# 1. Install spark_dsg from the local copy FIRST
pip install -e src/spark_dsg

# 2. Install hydra
pip install -e src/hydra

# 3. Install aion bindings
pip install -e src/aion/aion/python
```

### Run

```bash
# Train
python -m aion_python.runner \
  --config <dataset> --labelspace ade20k_outdoor \
  --train-scenes /path/to/scene_1 /path/to/scene_1\
  --train-csvs   /path/to/scene_1/tracks.csv /path/to/scene_2/tracks.csv\
  --save-flow-state results/aion_state.bin \
  --output-graph results/dsg.json \
  --assoc-radius 0.7 --grid-size 0.4 --time-window 10.0 --order 1

# Test/Score
python -m aion_python.score \
  --flow-state results/aion_state.bin \
  --test-csvs  /path/to/test/tracks_1.csv /path/to/test/tracks_2.csv \
  --output     results/preds_aion.csv
```

## Quick Configuration

Edit `config/temporal_dynamics_config.yaml`:

```yaml
temporal_dynamics_node:
  # Core
  num_orientation_bins: 8                    # Flow direction bins (typically 8)
  detection_association_distance: 0.7        # Radius to bind hash cells to navigational nodes [m]
  update_interval_seconds: 10.0              # FreMEn model update period [s]
  min_observations: 1                        # Min detections before showing a place

  # ROS Topics
  hydra_dsg_topic: "hydra_ros_node/backend/dsg"
  people_detection_topic: "/people_detections"   # Your people detection topic

  # Layer Selection
  use_navigation_layer: true                 # true = Layer 20 (mesh places), false = Layer 3 (3D places)

  # Delayed Binding (robustness to loop closures)
  enable_delayed_binding: true               # Accumulate in hash cells, then bind to stable nodes
  stability_window_seconds: 5.0             # Wait time before binding [s]

  # Visualization
  marker_scale: 0.3
  use_sphere_markers: false                  # false = arrows (dominant direction), true = spheres
  visualization_mode: "direction"            # "direction", "entropy", or "bins" (all 8 direction bins)
  visualization_data_source: "historical"    # "historical" or "predictions"
```

See the [Configuration Guide](doc/configuration.md) for the full parameter reference.

## Input Data Format

### Hydra Scene Graph

Subscribes to `hydra_msgs/DsgUpdate` and extracts navigational place nodes:
- **Layer 20** (MESH_PLACES, default): 2D mesh-based navigation places
- **Layer 3** (PLACES): 3D GVD-based place nodes

### People Detections

`geometry_msgs/PoseArray` with one pose per detected person:
```yaml
header:
  frame_id: "map"
poses:
  - position: {x: 1.0, y: 2.0, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}  # Movement direction
```

The orientation quaternion encodes the person's movement direction — the yaw component is extracted and discretized into orientation bins.

## Tools

Aion includes utility scripts for data export and temporal-aware path planning:

| Script | Purpose |
|--------|---------|
| `scripts/aion_map_export_service.py` | ROS service node for exporting temporal maps (JSON, numpy, CSV) |
| `scripts/aion_map_exporter.py` | CLI tool for one-shot or periodic auto-saving of map exports |
| `scripts/temporal_astar_comparison.py` | Offline standard A\* vs temporal A\* comparison from exported JSON |
| `scripts/temporal_path_visualizer.py` | RViz visualization of pre-computed A\* paths |
| `temporal_astar_node.py` | Live temporal-aware A\* ROS node (interactive via RViz) |

See [Usage Examples](doc/usage-examples.md) for details on each tool.

## License

MIT License — see [LICENSE](LICENSE) for details.
