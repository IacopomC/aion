# AionTemporalMap.msg - Collection of all temporal nodes in the map
# This message provides a unified view of all temporal data in the environment

Header header                      # Frame and timestamp information

# Map metadata
float64 update_interval_seconds    # How often this map is updated
uint32 num_orientation_bins        # Number of directional bins used

# Temporal data arrays
AionTemporalNode[] temporal_nodes     # Prediction-based temporal data
AionHistoricalNode[] historical_nodes # Historical temporal data

# Statistics
uint32 total_places                # Total number of tracked places
uint32 bound_places                # Number of places bound to Hydra nodes
uint32 hash_places                 # Number of places in hash space
