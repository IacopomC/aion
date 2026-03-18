# Aion Configuration Guide

This guide covers all configuration options, parameter tuning, and visualization setup for Aion.

## Configuration Files

### Main Configuration: `config/temporal_dynamics_config.yaml`

All parameters live in one file under the `temporal_dynamics_node:` namespace (they are loaded with `<rosparam command="load">`). The launch file accepts a `config_file` argument to point to an alternative YAML.

```yaml
temporal_dynamics_node:
  # Spatial Configuration
  num_orientation_bins: 8                    # Flow direction discretization (4, 8, or 16)
  detection_association_distance: 0.7        # Radius to bind hash cells to navigational nodes [m]

  # Temporal Configuration
  update_interval_seconds: 10.0             # FreMEn model update frequency [seconds]
  min_observations: 1                       # Minimum observations before showing a place marker

  # FreMEn Configuration
  fremen_model_order: 0                     # Model complexity (0=basic, higher=seasonal/weekly)
  fremen_operation_timeout: 5.0             # Timeout for FreMEn action calls [seconds]
  fremen_action_server_name: "/fremenarray_aion"  # FremenArray action server name

  # ROS Topics
  hydra_dsg_topic: "hydra_ros_node/backend/dsg"
  people_detection_topic: "/people_detections"
  temporal_places_topic: "/aion/temporal_dynamics"  # MarkerArray visualization topic
  temporal_nodes_topic: "/aion/temporal_nodes"
  historical_nodes_topic: "/aion/historical_nodes"
  temporal_map_topic: "/aion/temporal_map"

  # Layer Selection
  use_navigation_layer: true               # true = Layer 20 (MESH_PLACES), false = Layer 3 (3D places)
  target_layer_id: -1                      # Override layer selection (-1 = use use_navigation_layer)

  # Delayed Binding (robustness to loop closures)
  enable_delayed_binding: true             # Accumulate detections in hash cells, bind to stable nodes
  stability_window_seconds: 5.0           # Time before binding hash cell to a DSG node [s]

  # Visualization
  marker_scale: 0.3                        # Base marker size [m]
  arrow_scale: 1.0                         # Arrow length multiplier (arrow mode only)
  use_sphere_markers: false                # false = arrows (show dominant direction), true = spheres
  visualization_data_source: "historical"  # "historical" or "predictions"
  visualization_mode: "direction"          # "entropy", "direction", or "bins" (see Visualization section)
  frame_id: "map"                          # TF frame for all markers
```

See the full default file at `config/temporal_dynamics_config.yaml` for all parameters.

## Visualization Configuration

### Visualization Mode (`visualization_mode`)

Aion supports three visualization modes, controlled by the `visualization_mode` parameter:

| Mode | Description |
|------|-------------|
| `"direction"` | **Default.** One arrow per place pointing in the dominant flow direction. Arrow length encodes activity. Color encodes entropy. |
| `"entropy"` | One sphere per place. Color encodes entropy (blue=predictable, red=chaotic). Size encodes activity. |
| `"bins"` | All 8 direction bins shown as individual arrows per place, each scaled by its probability. Low-probability bins are hidden. Useful for understanding multi-modal flow distributions. |

Set it in your config:
```yaml
temporal_dynamics_node:
  visualization_mode: "bins"       # Show all 8 direction bins
  use_sphere_markers: false        # Must be false for direction/bins modes
```

### Marker Visual Channels

| Visual Property | Encoding | Interpretation |
|----------------|----------|----------------|
| **Color** | Entropy | Blue = predictable flow, Red = chaotic |
| **Size / Length** | Activity | Bigger = more observations |
| **Transparency** | Confidence | Opaque = many observations, transparent = few |
| **Orientation** | Dominant direction | Points toward main movement direction (direction/bins modes) |

### Frame and Coordinate Settings

```yaml
temporal_dynamics_node:
  frame_id: "map"                  # Coordinate frame for all markers
  marker_z_levels:
    temporal_markers: 5.0          # Z-offset for active temporal markers [m]
    inactive_markers: 5.0          # Z-offset for places with no data yet [m]
```

### Spatial Hashing Configuration

```yaml
temporal_dynamics_node:
  spatial_hash_grid_size: 0.3      # Grid cell size for spatial ordering [m]
  global_model:
    min_places_for_connection: 3   # Minimum places before connecting to FreMEn
```

### Model Parameters

```yaml
temporal_dynamics_node:
  fremen_model_order: 0            # Temporal model complexity
  min_observations: 1              # Minimum observations for prediction
  update_interval_seconds: 10.0    # Model update frequency
```

**Parameter guidelines:**
- **`fremen_model_order`**: 0 = no harmonic components (captures mean activity only). 1+ adds Fourier components for cyclic patterns (daily, weekly, seasonal). More components require more data to converge.
- **`min_observations`**: Set to 1 for debugging/demo, 5-20 for production use.
- **`update_interval_seconds`**: Each tick sends the current bin counts to FreMEn and resets the accumulation window. Shorter = more responsive but more computation.

## Delayed Binding

Detections are accumulated in **spatial hash cells** first, then assigned to DSG place nodes after a stability window. This prevents losing data when Hydra re-IDs nodes during loop closure corrections.

```yaml
temporal_dynamics_node:
  enable_delayed_binding: true
  stability_window_seconds: 5.0    # Seconds before a hash cell is bound to a node
  enable_debug_visualization: false # Enable hash cell and binding debug markers
```

## Performance Parameters

```yaml
temporal_dynamics_node:
  processing_timer_rate: 0.5       # Timer period in SECONDS (0.5 = fires every 0.5s, i.e. 2 Hz)
  detection_buffer_size: 10        # Max buffered PoseArray messages before dropping
  use_efficient_spatial_search: true  # Use KD-tree (O(log n)) for place lookup
  place_sync_interval_seconds: 5.0    # How often to re-read DSG for new places [s]
```

**Note:** `processing_timer_rate` is in **seconds** (period), not Hz. A value of `0.5` means the timer fires twice per second.

## Z-Height Filtering

Useful in multi-floor environments to restrict Aion to a single floor:

```yaml
temporal_dynamics_node:
  filter_by_z_height: false        # Enable height-based filtering
  min_z_height: -1.0               # Minimum Z [m]
  max_z_height: 2.0                # Maximum Z [m]
```

## Service Configuration

Default service names (all configurable):

```yaml
temporal_dynamics_node:
  prediction_service_name: "/aion/get_prediction"
  all_predictions_service_name: "/aion/get_all_predictions"
  reset_service_name: "/aion/reset"
  manual_fremen_service_name: "/aion/update_fremen"
  export_navigation_service_name: "/aion/export_navigation_data"
  export_default_output_directory: "/tmp/aion_navigation_export"
  export_default_filename_prefix: "aion_export"
```

### Example Service Calls

```bash
# Prediction for place ID 42
rosservice call /aion/get_prediction "{place_id: 42, prediction_time: 0}"

# Prediction by position (finds nearest place)
rosservice call /aion/get_prediction "{place_id: 0, x: 1.0, y: 2.0, z: 0.0, prediction_time: 0}"

# All places prediction
rosservice call /aion/get_all_predictions "{prediction_time: 0, order: 0}"

# Manually trigger FreMEn update
rosservice call /aion/update_fremen "{}"

# Reset all temporal data
rosservice call /aion/reset "{}"

# Export data for offline A* analysis
rosservice call /aion/export_navigation_data "{
  output_directory: '/tmp/aion_export',
  include_connectivity: true,
  include_temporal_data: true,
  prediction_time: 0
}"
```

## Saving and Exporting Aion Data

Aion **does not automatically persist data** between runs — all accumulated observations and FreMEn models live in memory. To save the current state:

### Option 1: Export to JSON (recommended)

```bash
# Call the export service while Aion is running
rosservice call /aion/export_navigation_data "{
  output_directory: '/path/to/save',
  include_connectivity: true,
  include_temporal_data: true,
  prediction_time: 0
}"
```

This writes two JSON files into `output_directory`:
- `navigation_layer.json` — place node positions and edges from the Hydra DSG
- `temporal_dynamics.json` — per-place observation counts, probabilities, entropy, and FreMEn predictions

### Option 2: Auto-save with the exporter script

The `scripts/aion_map_exporter.py` script can automatically save exports:

```bash
# One-shot export
rosrun aion aion_map_exporter.py --output /path/to/save

# Periodic auto-save every 60 seconds
rosrun aion aion_map_exporter.py --output /path/to/save --auto_save --save_interval 60

# Override source topic explicitly
rosrun aion aion_map_exporter.py --output /path/to/save --temporal_map_topic /aion/temporal_map
```

The exported JSON files can be loaded by `scripts/temporal_astar_comparison.py` for offline path planning analysis.

> **Note:** There is no built-in offline/playback mode. If you need to replay previously collected data, re-run Aion against a rosbag.

## Profiling / Timing

```yaml
temporal_dynamics_node:
  enable_timing: false             # Write per-iteration timing to CSV
  timing_output_file: ""           # Path for timing CSV (empty = /tmp/aion_timing.csv)
```

When `enable_timing: true`, Aion writes a CSV with columns: `timestamp_ns, callback, elapsed_us, num_places, total_observations, memory_bytes`.

## Troubleshooting

### Spatial Hashing Issues

**No temporal data appearing:**
- Check that `people_detection_topic` matches your detection publisher
- Verify detections are in the `map` frame (same as `frame_id`)
- Lower `min_observations` to 1 for debugging

**Places Not Binding:**
- Increase `detection_association_distance` if hash cells are too far from DSG nodes
- Enable `enable_debug_visualization: true` to see hash cell positions in RViz

**Fremen not updating:**
- Verify the FremenArray action server is running: `rostopic list | grep fremenarray`
- The launch file starts it automatically; if using a custom launch, ensure it is included
- Check `fremen_action_server_name` matches the server's actual name
