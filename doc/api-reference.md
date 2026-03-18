# Aion API Reference

This document provides detailed information about all nodes, topics, services, and message types in the Aion package.

## Nodes

### `temporal_dynamics_node`
Main node that processes Hydra scene graphs and people detections to build temporal flow models for place nodes.

**Subscribed Topics:**
- `/hydra_ros_node/backend/dsg` (hydra_msgs/DsgUpdate) - Hydra scene graph updates
- `/people_detections` (geometry_msgs/PoseArray) - People detection poses

**Published Topics:**
- `/aion/temporal_dynamics` (visualization_msgs/MarkerArray) - Unified temporal visualization (topic name set via `temporal_places_topic` param)
- `/aion/temporal_map` (aion/AionTemporalMap) - Full structured temporal map
- `/aion/temporal_nodes` (aion/AionTemporalNode) - Per-node prediction data stream
- `/aion/historical_nodes` (aion/AionHistoricalNode) - Per-node historical data stream
- `/aion/hash_cells` (visualization_msgs/MarkerArray) - Debug: unbound spatial hash cells (when `enable_debug_visualization: true`)
- `/aion/debug/bindings` (visualization_msgs/MarkerArray) - Debug: binding connections

**Services:**
- `/aion/get_prediction` (aion/GetTemporalPrediction) - Get prediction for single place
- `/aion/get_all_predictions` (aion/GetAllPlacePredictions) - Get predictions for all places
- `/aion/reset` (std_srvs/Trigger) - Reset all temporal data
- `/aion/update_fremen` (std_srvs/Trigger) - Manually trigger a FreMEn model update cycle
- `/aion/export_navigation_data` (aion/ExportNavigationData) - Export navigation graph + temporal data to JSON

## Message Types

### `aion/GetTemporalPrediction`

**Request:**
```
uint64 place_id           # Place ID to query (0 = use position)
float64 x                 # X position for nearest place lookup
float64 y                 # Y position for nearest place lookup  
float64 z                 # Z position for nearest place lookup
uint64 prediction_time    # Time for prediction (nanoseconds, 0 = now)
```

**Response:**
```
bool success                    # Success status
string message                  # Status message
uint64 place_id                 # Actual place ID used
float64[] flow_probabilities    # Per-orientation-bin probabilities [0.0-1.0]
float64 entropy                 # Shannon entropy [0.0-max_entropy]
uint32 num_observations         # Total observations for this place
float64 position_x              # Place position X
float64 position_y              # Place position Y  
float64 position_z              # Place position Z
```

### `aion/GetAllPlacePredictions`

**Request:**
```
uint64 prediction_time    # Time for prediction (nanoseconds, 0 = now)
int32 order              # Fremen model order (0 = default)
```

**Response:**
```
bool success                                    # Success status
string message                                  # Status message
aion/PlacePrediction[] place_predictions  # Predictions for all places
```

### `aion/PlacePrediction`
```
uint64 place_id                 # Place node ID
float64[] flow_probabilities    # Per-orientation-bin probabilities
float64 entropy                 # Predictability measure
uint32 num_observations         # Historical observation count
geometry_msgs/Point position    # Place position in world coordinates
```

### `aion/ExportNavigationData`

**Request:**
```
string output_directory    # Directory to write JSON files
bool include_connectivity  # Include place-to-place edges
bool include_temporal_data # Include FreMEn prediction data
uint64 prediction_time     # Timestamp for predictions (0 = current time)
```

**Response:**
```
bool success               # Success status
string message             # Status message
int32 num_nodes            # Number of exported navigation nodes
int32 num_edges            # Number of exported edges
int32 num_temporal_places  # Number of exported temporal places
```

## Topics

### Subscribed Topics

#### `/hydra_ros_node/backend/dsg` (hydra_msgs/DsgUpdate)
Hydra scene graph updates containing place node information.

**Message Structure:**
- Serialized dynamic scene graph data
- Layer 3 (places) extracted automatically
- Real-time updates as Hydra explores environment

**Processing Details:**
- Place nodes are extracted from the serialized DSG
- Spatial hashing determines consistent ordering for Fremen global model
- New places trigger global model reinitialization
- Place positions are cached for spatial hash calculation

**Spatial Hash Processing:**
```cpp
// Example: How place nodes are processed for spatial consistency
for (const auto& node : places_layer) {
  NodeId place_id = node.id().getKey();
  Eigen::Vector3d position = node.attributes().position;
  
  // Calculate spatial hash for consistent ordering
  size_t spatial_hash = getSpatialHashIndex(place_id, position);
  
  // Store for global state vector construction
  place_spatial_mapping_[spatial_hash] = place_id;
}
```

#### `/people_detections` (geometry_msgs/PoseArray)
People detection poses with position and orientation.

**Expected Format:**
```
header:
  frame_id: "map"          # Should match temporal_dynamics frame_id
poses:
  - position: {x: 1.0, y: 2.0, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}  # Facing direction
```

**Requirements:**
- Orientation must represent person's facing direction
- Position should be in same coordinate frame as Hydra
- Timestamp used for temporal modeling

### Published Topics

#### `/aion/temporal_dynamics` (visualization_msgs/MarkerArray)
Unified visualization markers encoding multiple temporal properties. The topic name is configurable via `temporal_places_topic` parameter.

**Marker Properties:**
- **Type**: SPHERE or ARROW (controlled by `use_sphere_markers`)
- **Color**: Entropy-based (blue=predictable → red=unpredictable)
- **Size**: Activity level (bigger=more observations)
- **Transparency**: Confidence (opaque=confident → transparent=uncertain)
- **Orientation**: Dominant flow direction (arrow mode only)

Places with no temporal data yet are shown as small gray spheres.

**Marker IDs**: Incremental starting from 0, includes a DELETEALL marker to clear stale data each refresh.

#### `/aion/temporal_map` (aion/AionTemporalMap)
Complete snapshot of all nodes with both historical and prediction data. Published on every FreMEn update cycle.

#### `/aion/temporal_nodes` (aion/AionTemporalNode)
Stream of individual node prediction messages, one per place per update cycle.

#### `/aion/historical_nodes` (aion/AionHistoricalNode)
Stream of individual node historical data messages.

## Services

### `get_prediction` (aion/GetTemporalPrediction)
Get temporal flow prediction for a single place.

**Usage Examples:**
```bash
# Query specific place ID
rosservice call /aion/get_prediction "{place_id: 42, prediction_time: 0}"

# Query by position (finds nearest place)
rosservice call /aion/get_prediction "{place_id: 0, x: 1.0, y: 2.0, z: 0.0, prediction_time: 0}"

# Future prediction (Unix timestamp in nanoseconds)
rosservice call /aion/get_prediction "{place_id: 42, prediction_time: 1640995200000000000}"
```

**Returns:**
- Flow probabilities for each orientation bin
- Entropy measure of predictability
- Place metadata (position, observation count)

### `get_all_predictions` (aion/GetAllPlacePredictions)
Get predictions for the entire place layer.

**Usage Examples:**
```bash
# Current predictions for all places
rosservice call /aion/get_all_predictions "{prediction_time: 0, order: 0}"

# Future predictions with higher-order Fremen model
rosservice call /aion/get_all_predictions "{prediction_time: 1640995200000000000, order: 2}"
```

**Returns:**
- Array of predictions for all places with sufficient data
- Global temporal state at requested time

### `reset` (std_srvs/Trigger)
Reset all accumulated temporal data and restart learning.

**Usage:**
```bash
rosservice call /aion/reset "{}"
```

**Effect:**
- Clears all place observation history
- Resets Fremen models
- Reinitializes data structures

### `/aion/update_fremen` (std_srvs/Trigger)
Manually trigger a FreMEn model update outside the normal schedule. Useful for forcing an immediate re-computation of predictions.

**Usage:**
```bash
rosservice call /aion/update_fremen "{}"
```

**Use Cases:**
- Force model update for immediate predictions
- Trigger a FreMEn cycle without waiting for the timer
- Debug Fremen integration

### `/aion/export_navigation_data` (aion/ExportNavigationData)
Export the Hydra navigation layer and temporal dynamics data to JSON files for offline analysis with the A* tools.

**Request fields:**
```
string output_directory         # Directory to write JSON files into
bool include_connectivity       # Include place-to-place edges
bool include_temporal_data      # Include FreMEn prediction data
uint64 prediction_time         # Timestamp for predictions (0 = current time)
```

**Usage:**
```bash
rosservice call /aion/export_navigation_data "{
  output_directory: '/tmp/aion_export',
  include_connectivity: true,
  include_temporal_data: true,
  prediction_time: 0
}"
```

**See also:** `scripts/temporal_astar_comparison.py` for offline A* analysis using the exported files.

## Parameters

### Core Parameters
- `num_orientation_bins` (int, default: 8) - Flow direction discretization
- `detection_association_distance` (double, default: 0.7) - Hash cells-to-node binding radius [m]
- `update_interval_seconds` (double, default: 10.0) - FreMEn update frequency [s]
- `min_observations` (int, default: 1) - Minimum observations for a place to be shown

### Topics and Services
- `hydra_dsg_topic` (string) - Hydra scene graph topic
- `people_detection_topic` (string, default: `/people_detections`) - People detection topic
- `temporal_places_topic` (string, default: `/aion/temporal_dynamics`) - Visualization output topic
- `temporal_nodes_topic` (string, default: `/aion/temporal_nodes`) - Prediction node stream topic
- `historical_nodes_topic` (string, default: `/aion/historical_nodes`) - Historical node stream topic
- `temporal_map_topic` (string, default: `/aion/temporal_map`) - Unified map topic
- `prediction_service_name` (string) - Single prediction service name
- `all_predictions_service_name` (string) - All predictions service name
- `manual_fremen_service_name` (string) - Manual FreMEn update service
- `export_navigation_service_name` (string) - Export service name
- `export_default_output_directory` (string) - Default export directory when request path is empty
- `export_default_filename_prefix` (string) - Default export filename prefix when request prefix is empty

### Visualization
- `marker_scale` (double, default: 0.3) - Base marker size [m]
- `arrow_scale` (double, default: 1.0) - Arrow size multiplier
- `use_sphere_markers` (bool, default: false) - Sphere (`true`) vs arrow (`false`) markers
- `visualization_data_source` (string, default: `"historical"`) - What data to visualize: `"historical"` uses accumulated counts, `"predictions"` uses FreMEn predictions
- `enable_debug_visualization` (bool, default: false) - Publish hash cell and binding debug markers

### Fremen Configuration
- `fremen_model_order` (int, default: 0) - Temporal model complexity
- `fremen_action_server_name` (string, default: `/fremenarray_aion`) - FremenArray server name
- `fremen_operation_timeout` (double, default: 5.0) - Operation timeout [s]

## Dependencies

### ROS Packages
- `hydra_msgs` - Hydra scene graph messages
- `spark_dsg` - Scene graph processing
- `fremenarray` - Temporal modeling framework
- `visualization_msgs` - RViz markers
- `geometry_msgs` - Pose and point messages

### System Libraries
- Eigen3 - Linear algebra
- ROS Noetic - Robot Operating System
- C++17 - Modern C++ features

## Error Codes

### Service Responses
- `success: true` - Operation completed successfully
- `success: false` - Operation failed, check message field

### Common Error Messages
- `"Place not found"` - Invalid place_id or position
- `"Insufficient observations"` - Place has < min_observations
- `"Fremen prediction failed"` - Temporal model unavailable
- `"Invalid prediction time"` - Time parameter out of range
