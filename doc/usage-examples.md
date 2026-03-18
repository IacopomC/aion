# Aion Usage Examples

This document provides practical examples and common usage patterns for Aion. The basic usage is already covered in the main [readme](../README.md).

## Service Usage Examples

### Single Place Prediction

```bash
# Query specific place by ID
rosservice call /aion/get_prediction "{
  place_id: 42,
  prediction_time: 0
}"

# Query nearest place to position
rosservice call /aion/get_prediction "{
  place_id: 0,
  x: 10.0, y: 5.0, z: 0.0,
  prediction_time: 0
}"

# Future prediction (Unix timestamp in nanoseconds)
# Example: January 1, 2024, 2:00 PM = 1704110400000000000
rosservice call /aion/get_prediction "{
  place_id: 42,
  prediction_time: 1704110400000000000
}"
```

**Expected Response:**
```yaml
success: True
message: "Prediction successful"
place_id: 42
flow_probabilities: [0.1, 0.05, 0.3, 0.2, 0.15, 0.1, 0.05, 0.05]  # 8 direction bins
entropy: 2.45
num_observations: 127
position_x: 10.2
position_y: 5.1
position_z: 0.0
```

### All Places Prediction

```bash
# Get current predictions for all places
rosservice call /aion/get_all_predictions "{
  prediction_time: 0,
  order: 0
}"

# Future prediction with higher-order Fremen model
rosservice call /aion/get_all_predictions "{
  prediction_time: 1704110400000000000,
  order: 2
}"
```