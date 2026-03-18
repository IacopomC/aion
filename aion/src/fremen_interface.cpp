#include "aion/fremen_interface.h"
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <fremenarray/FremenArrayAction.h>
#include <chrono>

namespace aion {

// Implementation class to hide ROS dependencies
class FremenInterface::Impl {
 public:
  // Full Fremen integration implementation
  
  Impl(const Config& config) : config_(config), fremen_client_(config_.action_server_name, true) {
    ROS_INFO("FremenInterface initializing with full Fremen integration...");
    
    // Check for Fremen action server without blocking
    ROS_INFO("Checking for FremenArray action server at: %s", config_.action_server_name.c_str());
    if (fremen_client_.waitForServer(ros::Duration(1.0))) {
      ROS_INFO("Successfully connected to FremenArray action server");
      connected_ = true;
    } else {
      ROS_WARN("FremenArray action server not yet available - will retry during operation");
      connected_ = false;
    }
  }
  
  bool addState(const std::string& model_id, uint64_t timestamp_ns, 
                const std::vector<int>& states) {
    if (!connected_) {
      // Try to reconnect once
      if (fremen_client_.waitForServer(ros::Duration(0.1))) {
        connected_ = true;
        ROS_INFO("FremenArray connection restored");
      } else {
        ROS_DEBUG("FremenInterface not connected - skipping data");
        return false;
      }
    }
    
    // Use single global Fremen model, ignore model_id
    fremenarray::FremenArrayGoal goal;
    goal.operation = "add";
    goal.time = nanosToFremenTime(timestamp_ns);
    // Convert std::vector<int> to std::vector<int8_t>
    goal.states.clear();
    goal.states.reserve(states.size());
    for (int state : states) {
      goal.states.push_back(static_cast<int8_t>(state));
    }
    
    // PERFORMANCE FIX: Send goal asynchronously (non-blocking)
    fremen_client_.sendGoal(goal);
    
    // Don't wait for result - this was blocking for up to 5 seconds!
    // The Fremen framework will process the data asynchronously
    ROS_DEBUG("Fremen addState sent asynchronously (%zu states)", states.size());
    return true;
  }
  
  bool predict(const std::string& model_id, uint64_t prediction_time_ns,
               std::vector<double>& predicted_states) {
    if (!connected_) {
      // Try to reconnect once, but don't block
      if (fremen_client_.waitForServer(ros::Duration(0.01))) {  // 10ms max
        connected_ = true;
        ROS_INFO("FremenArray connection restored");
      } else {
        ROS_DEBUG("FremenInterface not connected - using fallback");
        // PERFORMANCE FIX: Always provide fallback, never block
        predicted_states.assign(8, 0.125); // Uniform distribution
        return false;
      }
    }
    
    // Use single global Fremen model, ignore model_id
    fremenarray::FremenArrayGoal goal;
    goal.operation = "predict";
    goal.time = nanosToFremenTime(prediction_time_ns);
    goal.order = config_.model_order;
    
    // PERFORMANCE FIX: Send goal and check result NON-BLOCKING
    fremen_client_.sendGoal(goal);
    
    // Check if result is immediately available (non-blocking)
    if (fremen_client_.waitForResult(ros::Duration(0.0))) {
      auto result = fremen_client_.getResult();
      if (!result->probabilities.empty()) {
        // Convert std::vector<float> to std::vector<double>
        predicted_states.clear();
        predicted_states.reserve(result->probabilities.size());
        for (float prob : result->probabilities) {
          predicted_states.push_back(static_cast<double>(prob));
        }
        ROS_DEBUG("Fremen predict result: %zu probabilities", predicted_states.size());
        return true;
      }
    }
    
    // If no immediate result available, use fallback (never block!)
    ROS_DEBUG("Fremen predict not immediately available - using fallback");
    predicted_states.assign(8, 0.125); // Uniform distribution fallback
    return false;
  }
  
  bool buildModel(const std::string& model_id) {
    if (!connected_) {
      ROS_ERROR("FremenInterface not connected");
      return false;
    }
    
    // Fremen builds models automatically during 'add' operations
    // No explicit build operation needed for FremenArray
    return true;
  }
  
  bool isConnected() const { return connected_; }
  
  double nanosToFremenTime(uint64_t timestamp_ns) {
    return static_cast<double>(timestamp_ns) / 1e9;
  }
  
 private:
  Config config_;
  bool connected_ = false;
  actionlib::SimpleActionClient<fremenarray::FremenArrayAction> fremen_client_;
};

FremenInterface::FremenInterface(int model_order) {
  Config config;
  config.model_order = model_order;
  config_ = config;
  impl_ = std::make_unique<Impl>(config);
}

FremenInterface::FremenInterface(const Config& config) 
    : config_(config), impl_(std::make_unique<Impl>(config)) {
}

FremenInterface::~FremenInterface() = default;

bool FremenInterface::addState(const std::string& model_id, uint64_t timestamp_ns,
                              double state_value) {
  return impl_->addState(model_id, timestamp_ns, std::vector<int>{static_cast<int>(state_value)});
}

bool FremenInterface::addState(const std::string& model_id, uint64_t timestamp_ns,
                              const std::vector<int>& states) {
  return impl_->addState(model_id, timestamp_ns, states);
}

bool FremenInterface::addGlobalState(uint64_t timestamp_ns, const std::vector<int>& global_state) {
  return impl_->addState("global", timestamp_ns, global_state);
}

bool FremenInterface::predictGlobalState(uint64_t prediction_time_ns,
                                        std::vector<double>& predicted_states) {
  return impl_->predict("global", prediction_time_ns, predicted_states);
}

bool FremenInterface::initializeGlobalModel(size_t state_size) {
  // Initialize with zeros
  std::vector<int> initial_state(state_size, 0);
  return impl_->addState("global", 0, initial_state);
}

double FremenInterface::predict(const std::string& model_id, uint64_t prediction_time_ns) {
  std::vector<double> predicted_states;
  if (impl_->predict(model_id, prediction_time_ns, predicted_states) && !predicted_states.empty()) {
    return predicted_states[0];
  }
  return 0.0;
}

bool FremenInterface::predict(const std::string& model_id, uint64_t prediction_time_ns,
                             std::vector<double>& predicted_states) {
  return impl_->predict(model_id, prediction_time_ns, predicted_states);
}

bool FremenInterface::buildModel(const std::string& model_id) {
  return impl_->buildModel(model_id);
}

bool FremenInterface::buildModels() {
  // In the global model approach, there's only one model and it's built automatically
  return true;
}

bool FremenInterface::isConnected() const {
  return impl_->isConnected();
}

bool FremenInterface::getModelState(const std::string& model_id, 
                                   std::vector<uint8_t>& model_data) {
  // For now, just return empty state - could be implemented with additional Fremen operations
  model_data.clear();
  return true;
}

bool FremenInterface::setModelState(const std::string& model_id,
                                   const std::vector<uint8_t>& model_data) {
  // For now, just ignore - could be implemented with additional Fremen operations
  return true;
}

namespace fremen_utils {

std::string generateModelId(uint64_t node_id, const std::string& prefix) {
  return prefix + "_" + std::to_string(node_id);
}

double nanosToFremenTime(uint64_t timestamp_ns) {
  return static_cast<double>(timestamp_ns) / 1e9;
}

uint64_t fremenTimeToNanos(double fremen_time) {
  return static_cast<uint64_t>(fremen_time * 1e9);
}

}  // namespace fremen_utils

}  // namespace aion
