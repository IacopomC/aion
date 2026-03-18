# Understanding Aion: Concepts and Theory

This guide explains the concepts, theory, and architecture of Aion for temporal flow prediction in dynamic 3D scene graphs.

## What is Aion?

**Aion** integrates temporal flow modeling into a 3D scene graph system (Hydra's), providing FreMEn functionality for semantic place nodes rather than static grid cells. It learns temporal patterns of human movement through places in the scene graph and predicts future flow patterns.

## Core Concepts

### Scene Graph-Based Temporal Modeling

Unlike grid-based approaches, Aion operates on free space nodes:
- **Dynamic Spatial Structure**: Nodes appear and evolve as the robot explores
- **Semantic Awareness**: Models flow through meaningful locations (rooms, corridors)
- **3D Compatibility**: Handles full 3D environments with Z-layered visualization

### Temporal Pattern Learning

Aion learns cyclical patterns in human movement through places:
- **Short cycles**: Activity peaks and troughs within a day
- **Weekly cycles**: Weekday vs. weekend patterns in office buildings
- **Long-term trends**: Seasonal or structural changes in a space

The system uses the FreMEn framework to model these patterns efficiently, allowing it to:
1. **Estimate current state**: "What should be happening in place X right now?"
2. **Forecast future patterns**: "What will the flow be in place Y tomorrow at 2 PM?"
3. **Adapt continuously**: Patterns evolve as new data arrives

### Prediction vs. Historical Data

Aion tracks two parallel representations per place, controlled by the `visualization_data_source` parameter:

**`"historical"` mode** (default — most common):
- Uses directly accumulated bin counts from observed detections
- No FreMEn model required; available immediately after first detection
- Reflects the actual observation distribution over all time
- Use case: "What have we observed at this place so far?"

**`"predictions"` mode**:
- Uses FreMEn prediction evaluated at the current (or a specified) time
- Requires sufficient history for the FreMEn model to converge
- Captures time-of-day or day-of-week periodicity
- Use case: "What should be happening right now, given learned patterns?"

## Technical Background

### Learning Process

1. **Scene Graph Subscription**: Receive Hydra's real-time DSG updates
2. **Place Extraction**: Extract 2D navigational nodes (Layer 20 MESH_PLACES or Layer 3 PLACES)
3. **Detection Association**: Associate people detections with the nearest spatial hash cell
4. **Delayed Binding**: After a stability window, bind hash cell data to stable DSG nodes
5. **Orientation Binning**: Discretize movement directions into `num_orientation_bins` bins (typically 8)
6. **FreMEn Update**: Periodically send the global state vector to FreMEn
7. **Prediction Generation**: Evaluate learned models at any requested timestamp

### Architecture

```
Hydra DSG + People Detections
       │
       ▼
Spatial Hash Cells ──[stability window]──► DSG Nodes
       │                                        │
       ▼                                        ▼
Orientation Bins (8)               Orientation Bins (8)
       │                                        │
       └──────────────┬─────────────────────────┘
                      ▼
            Global State Vector
          (all places concatenated,
           ordered by spatial hash)
                      │
                      ▼
              FreMEn Global Model
                      │
             ┌────────┴─────────┐
             ▼                  ▼
      Historical Metrics   Predictions
      (entropy, direction,  (time-aware
       flow magnitude)      probabilities)
```

## FreMEn Integration and the Global Model

### Why a Global FreMEn Model?

Aion uses a **single global FreMEn model** that covers all places simultaneously, rather than one independent model per place. The design reasons are:

- **Temporal coherence**: All places are updated at the same time tick, keeping their temporal history aligned.
- **Efficiency**: A single ROS action server handles everything; per-place models would require N concurrent connections and significantly more overhead.

The tradeoff is that a single FreMEn model cannot capture place-specific periodicities independently. In practice, `fremen_model_order: 0` (the default) simply models the mean activity per bin across all time, which is equivalent to historical statistics and is always available even if FreMEn is not connected.

### The Grid vs. Dynamic Nodes Challenge

A static grid would have fixed-size state vectors. The scene graph doesn't:
1. New places appear as the robot explores (new graph nodes)
2. Node IDs are non-sequential (e.g., 42, 173, 891)
3. Places may be re-IDed after loop closures

### Spatial Hashing Solution

Aion solves the ordering problem with **2D spatial hashing**. Each place's (x, y) world position is discretized into a grid cell at resolution `spatial_hash_grid_size`, and the cell coordinates are bit-packed into a `size_t` hash key. The global state vector is built by sorting all temporal entries by their hash key before concatenating bin counts. This gives:

- **Deterministic ordering**: The same spatial positions always produce the same key
- **Robust to ID changes**: Ordering depends on position, not node ID
- **Loop-closure tolerant**: A re-IDed node in the same location gets the same hash key

When a hash cell is **bound** to a DSG node (after the stability window), the original hash key is preserved as `original_hash_cell` so that the FreMEn index slot does not shift.

### Delayed Binding

Detections are first collected in spatial hash cells (keyed by grid position), then migrated to DSG node entries after `stability_window_seconds`. This decouples data collection from graph topology, protecting against:
- Node ID reassignment during loop closures
- Temporary node disappearance during graph updates

```
Detection arrives
      │
      ▼ (findNearestPlace, no binding yet)
Spatial hash cell accumulates counts
      │
      ▼ (after stability_window_seconds)
globalStabilityTimerCallback fires
      │
      ▼ (assignDynamicsToStableNodes)
Data migrates to node-based key (node_id | 0x8000...0)
original_hash_cell preserved for FreMEn index stability
      │
      ▼ (if node disappears: handleNodeRemoval)
Data migrates back to hash cell
```

### FreMEn Update Process

**Step 1**: At each `update_interval_seconds` tick, build the global state vector:
- Sort all temporal entries by spatial hash key
- For each entry, call `getNormalizedCounts()` (scales max bin to 100; returns -1 if no data)
- Concatenate into a flat `vector<int>`

**Step 2**: Send to FreMEn:
```
fremen_->addGlobalState(current_time_ns, global_state_vector)
```

**Step 3**: Move current interval counts into historical totals, reset the interval accumulator.

**Step 4**: Compute historical and/or prediction metrics:
- Historical: Shannon entropy, dominant direction, flow magnitude from accumulated counts
- Predictions: `fremen_->predictGlobalState(time_ns, predictions)` → slice per-place segments

**Step 5**: Publish markers and structured messages.

### Entropy Analysis

**Shannon Entropy** measures flow "chaos":
```
H = -Σ p_i · log₂(p_i)
```
where p_i is the probability of flow in direction bin i.

**Interpretation:**
- **Low entropy (< 1.0)**: Highly directional (e.g., corridor with consistent flow direction)
- **Medium entropy (1.0–2.5)**: Moderately directional (e.g., room with some preferred paths)
- **High entropy (> 2.5)**: Scattered (e.g., open junction with random movement)

### Flow Direction Analysis

- **`num_orientation_bins` bins** (default: 8) cover 360° in equal angular steps (45° each for 8 bins)
- **Dominant direction**: The bin with highest probability, converted to angle = `bin_index × (2π / num_bins)`
- **Multi-modal distributions**: Visible in `"bins"` visualization mode — a corridor intersection shows two or more prominent arrows
- **Flow magnitude**: Sum of all bin counts (proportional to total traffic at the place)
