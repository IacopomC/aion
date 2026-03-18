# Individual place prediction result
uint64 place_id                        # Place node ID in the scene graph
float64[] flow_probabilities           # Probability distribution over orientation bins
float64 entropy                        # Entropy measure of the temporal model
uint64 num_observations               # Total observations used in the model
geometry_msgs/Point position          # Place position in map frame
