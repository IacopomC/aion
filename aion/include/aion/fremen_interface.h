#pragma once

#include <vector>
#include <map>
#include <memory>
#include <cstdint>
#include <string>

namespace aion {

/**
 * @brief Interface to Fremen temporal modeling framework
 * 
 * This class provides a C++ interface to the Fremen temporal modeling
 * framework for predicting temporal patterns in flow data.
 */
class FremenInterface {
 public:
  struct Config {
    // Fremen model order (0=basic, higher=more complex)
    int model_order = 0;
    
    // Action server name for FremenArray
    std::string action_server_name = "/fremenarray";
    
    // Timeout for Fremen operations (seconds)
    double operation_timeout = 10.0;
  };

  /**
   * @brief Simple constructor with model order only
   * @param model_order Fremen model order (0=basic, higher=more complex)
   */
  FremenInterface(int model_order = 0);
  
  /**
   * @brief Full constructor with config
   * @param config Configuration parameters
   */
  FremenInterface(const Config& config);
  virtual ~FremenInterface();

  /**
   * @brief Add temporal state data to Fremen model
   * @param model_id Unique identifier for this temporal model
   * @param timestamp_ns Timestamp in nanoseconds
   * @param state_value Single state value (probability/count for one bin)
   * @return True if successful
   */
  bool addState(const std::string& model_id,
               uint64_t timestamp_ns,
               double state_value);

  /**
   * @brief Add temporal state data to Fremen model (vector version)
   * @param model_id Unique identifier for this temporal model
   * @param timestamp_ns Timestamp in nanoseconds
   * @param states State vector (flow counts per orientation bin)
   * @return True if successful
   */
  bool addState(const std::string& model_id,
               uint64_t timestamp_ns,
               const std::vector<int>& states);

  /**
   * @brief Get prediction from Fremen model for a single value
   * @param model_id Unique identifier for the temporal model
   * @param prediction_time_ns Time for prediction in nanoseconds
   * @return Predicted value (0.0 if prediction fails)
   */
  double predict(const std::string& model_id,
                 uint64_t prediction_time_ns);

  /**
   * @brief Get prediction from Fremen model
   * @param model_id Unique identifier for the temporal model
   * @param prediction_time_ns Time for prediction in nanoseconds
   * @param predicted_states Output: predicted state vector
   * @return True if successful
   */
  bool predict(const std::string& model_id,
              uint64_t prediction_time_ns,
              std::vector<double>& predicted_states);

  /**
   * @brief Build/update temporal model
   * @param model_id Unique identifier for the temporal model
   * @return True if successful
   */
  bool buildModel(const std::string& model_id);

  /**
   * @brief Add global state vector to single Fremen model
   * @param timestamp_ns Timestamp in nanoseconds
   * @param global_state Complete state vector for all places × orientations
   * @return True if successful
   */
  bool addGlobalState(uint64_t timestamp_ns, const std::vector<int>& global_state);

  /**
   * @brief Get global prediction from single Fremen model
   * @param prediction_time_ns Time for prediction in nanoseconds
   * @param predicted_states Output: predicted global state vector
   * @return True if successful
   */
  bool predictGlobalState(uint64_t prediction_time_ns,
                         std::vector<double>& predicted_states);

  /**
   * @brief Initialize single global Fremen model with empty state
   * @param state_size Total size of global state vector (num_places × num_orientations)
   * @return True if successful
   */
  bool initializeGlobalModel(size_t state_size);

  /**
   * @brief Build/update all temporal models
   * @return True if successful
   */
  bool buildModels();

  /**
   * @brief Check if Fremen action server is available
   * @return True if connected
   */
  bool isConnected() const;

  /**
   * @brief Get model state for serialization
   * @param model_id Unique identifier for the temporal model
   * @param model_data Output: serialized model data
   * @return True if successful
   */
  bool getModelState(const std::string& model_id, 
                    std::vector<uint8_t>& model_data);

  /**
   * @brief Set model state from serialized data
   * @param model_id Unique identifier for the temporal model
   * @param model_data Serialized model data
   * @return True if successful
   */
  bool setModelState(const std::string& model_id,
                    const std::vector<uint8_t>& model_data);

 private:
  Config config_;
  
  // Track which models have been initialized
  std::map<std::string, bool> initialized_models_;
  
  // Internal implementation details
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Utility functions for working with Fremen models
 */
namespace fremen_utils {

// Generate unique model ID for a place node
std::string generateModelId(uint64_t node_id, const std::string& prefix = "place");

// Convert nanoseconds to Fremen time format (seconds since epoch)
double nanosToFremenTime(uint64_t timestamp_ns);

// Convert Fremen time to nanoseconds
uint64_t fremenTimeToNanos(double fremen_time);

}  // namespace fremen_utils

}  // namespace aion
