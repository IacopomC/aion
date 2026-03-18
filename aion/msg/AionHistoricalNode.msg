# AionHistoricalNode.msg - Historical temporal data for a place node
# This message contains temporal flow data computed from accumulated historical observations

# Place identification  
uint64 place_id                    # Hydra place node ID
geometry_msgs/Point position       # 3D position in map frame

# Temporal flow metrics (from historical data)
float64 best_angle                 # Dominant flow direction in radians
int32[] raw_counts                # Raw historical counts [num_orientation_bins]
float64[] probabilities           # Normalized probabilities (sum=1.0)
int32[] normalized_counts         # Fremen-style normalized counts (max=100)
float64 flow_magnitude            # Sum of raw counts
float64 entropy                   # Shannon entropy from historical data

# Metadata
uint64 total_observations         # Total number of detections
bool is_bound_to_node             # True if bound to Hydra node, false if in hash space
uint64 spatial_hash               # Hash cell ID (relevant when not bound to node)
