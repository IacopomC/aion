# AionTemporalNode.msg - Prediction-based temporal data for a place node
# This message contains temporal flow data computed from Fremen predictions

# Place identification
uint64 place_id                    # Hydra place node ID
geometry_msgs/Point position       # 3D position in map frame

# Temporal flow metrics (from Fremen predictions)
float64 best_angle                 # Dominant flow direction in radians
float64[] probabilities           # Fremen prediction probabilities [num_orientation_bins]
float64 flow_magnitude            # Sum of probabilities 
float64 entropy                   # Shannon entropy from predictions

# Metadata
uint64 prediction_time            # Time of prediction (nanoseconds since epoch)
bool is_bound_to_node             # True if bound to Hydra node, false if in hash space
uint64 spatial_hash               # Hash cell ID (relevant when not bound to node)
