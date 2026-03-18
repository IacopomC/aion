#include "aion/temporal_dynamics_module.h"
#include <chrono>
#include <algorithm>
#include <fstream>

namespace aion {

void TemporalDynamicsNode::fremenUpdateTimerCallback(const ros::TimerEvent& event) {
  auto t0 = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::shared_mutex> lock(temporal_data_mutex_);

  ROS_INFO("Periodic Fremen model update started (global model, %zu places)...",
           temporal_place_data_.size());

  // Skip if no connection and insufficient data
  if (!fremen_->isConnected() && temporal_place_data_.size() < static_cast<size_t>(min_places_for_connection_)) {
    ROS_DEBUG("Skipping Fremen update - not connected or insufficient data");
    if (enable_timing_) {
      auto t1 = std::chrono::high_resolution_clock::now();
      double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      logTiming("fremenUpdate", us);
    }
    return;
  }

  // Initialize global model if needed
  if (!global_model_initialized_) {
    initializeGlobalFremenModel();
  }

  // Create global state vector from all current place data
  std::vector<int> global_state = createGlobalStateVector();

  if (global_state.empty()) {
    ROS_DEBUG("No data to send to Fremen - skipping update");
    if (enable_timing_) {
      auto t1 = std::chrono::high_resolution_clock::now();
      double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      logTiming("fremenUpdate", us);
    }
    return;
  }

  const uint64_t current_time = ros::Time::now().toNSec();

  // Send global state to single Fremen model - non-blocking
  if (fremen_->isConnected()) {
    if (!fremen_->addGlobalState(current_time, global_state)) {
      // Continue with local updates even if Fremen fails
    }
  } else {
    ROS_DEBUG("FremenArray not connected - using historical data only");
  }


  // Update historical data for all places
  for (auto& [place_id, place_data] : temporal_place_data_) {
    place_data.addToHistorical();

    // Update total_observations based on accumulated observation_times during periodic updates
    // This prevents real-time visual flickering while maintaining accurate counts
    size_t old_total = place_data.total_observations;
    place_data.total_observations = place_data.observation_times.size();
    place_data.resetCurrentInterval();
    place_data.last_fremen_update = current_time;
  }

  // Compute separated metrics
  if (config_.enable_historical_computation) {
    computeHistoricalMetrics();
  }

  // ROS_INFO("FREMEN DEBUG: Checking prediction computation (enabled: %s)",
  //          config_.enable_prediction_computation ? "true" : "false");
  if (config_.enable_prediction_computation) {
    computePredictionMetrics();
  }

  // Publish temporal data messages
  if (config_.enable_prediction_computation) {
    publishTemporalNodes();
  }

  // Publish unified temporal map if either computation is enabled
  if (config_.enable_historical_computation || config_.enable_prediction_computation) {
    publishTemporalMap();
  }

  ROS_INFO("Global Fremen model update completed for %zu places", temporal_place_data_.size());
  if (enable_timing_) {
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    logTiming("fremenUpdate", us);
  }
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

void TemporalDynamicsNode::logTiming(const std::string& callback_name, double elapsed_us) {
  if (!timing_file_.is_open()) return;
  auto mem = computeMemoryFootprint();
  timing_file_ << ros::Time::now().toNSec() << ","
               << callback_name << ","
               << static_cast<int64_t>(elapsed_us) << ","
               << mem.num_places << ","
               << mem.total_observations << ","
               << mem.estimated_bytes << std::endl;
}

TemporalDynamicsNode::MemoryFootprint TemporalDynamicsNode::computeMemoryFootprint() const {
  // NOTE: caller must already hold temporal_data_mutex_ or accept a racy snapshot
  MemoryFootprint mem;
  mem.num_places = temporal_place_data_.size();

  // Per-place fixed overhead: position(24) + scalars/flags(32)
  // + current_interval_counts + historical_counts + historical_metrics + prediction_metrics
  const size_t per_place_fixed =
      24 /* position */ + 32 /* scalars/flags */ +
      config_.num_orientation_bins * (sizeof(int) * 2 + sizeof(double) * 2); /* counts + probabilities */

  for (const auto& [key, pd] : temporal_place_data_) {
    if (key & 0x8000000000000000ULL) {
      mem.num_bound_nodes++;
    } else {
      mem.num_hash_cells++;
    }
    mem.total_observations += pd.total_observations;

    // Aion per-place: fixed struct + observation_times vector
    mem.estimated_bytes += per_place_fixed
        + pd.observation_times.size() * 8; // observation_times
  }
  // map node overhead: ~48 bytes per std::map node
  mem.estimated_bytes += mem.num_places * 48;
  return mem;
}

std::vector<double> TemporalDynamicsNode::predictFlow(spark_dsg::NodeId place_id,
                                                     uint64_t prediction_time_ns) const {
  std::vector<double> predictions(config_.num_orientation_bins, 0.0);

  // First, check if place exists and has enough data (temporal_data_mutex_ already held by caller)
  auto it = temporal_place_data_.find(place_id);
  if (it == temporal_place_data_.end() || it->second.total_observations < config_.min_observations) {
    return predictions; // Return empty predictions
  }

  // Try global model prediction if available
  bool global_prediction_success = false;
  if (global_model_initialized_) {
    std::vector<double> global_predictions;
    if (fremen_->predictGlobalState(prediction_time_ns, global_predictions)) {
      // Extract place prediction without holding locks for too long
      predictions = extractPlacePrediction(place_id, global_predictions);
      global_prediction_success = true;
    } else {
      ROS_WARN("Global Fremen prediction failed for place %lu", place_id);
    }
  }

  // Fallback to historical averages if global prediction failed
  if (!global_prediction_success) {
    // temporal_data_mutex_ already held by caller
    auto it = temporal_place_data_.find(place_id);
    if (it != temporal_place_data_.end()) {
      const auto& place_data = it->second;

      int total_historical = std::accumulate(place_data.historical_counts.begin(),
                                            place_data.historical_counts.end(), 0);
      if (total_historical > 0) {
        for (size_t bin = 0; bin < config_.num_orientation_bins; ++bin) {
          predictions[bin] = static_cast<double>(place_data.historical_counts[bin]) / total_historical;
        }
      } else {
        // Uniform distribution as last resort
        std::fill(predictions.begin(), predictions.end(), 1.0 / config_.num_orientation_bins);
      }
    }
  }

  return predictions;
}

std::map<spark_dsg::NodeId, double> TemporalDynamicsNode::getEntropyMap() const {
  // Note: temporal_data_mutex_ is already held by caller

  std::map<spark_dsg::NodeId, double> entropy_map;
  for (const auto& [place_id, place_data] : temporal_place_data_) {
    entropy_map[place_id] = getEntropy(place_data);
  }

  return entropy_map;
}

std::map<spark_dsg::NodeId, std::vector<double>> TemporalDynamicsNode::predictAllPlacesFlow(
    uint64_t prediction_time_ns) const {
  std::shared_lock<std::shared_mutex> lock(temporal_data_mutex_);

  std::map<spark_dsg::NodeId, std::vector<double>> all_predictions;

  // Use global model for all predictions
  if (global_model_initialized_) {
    std::vector<double> global_predictions;
    if (fremen_->predictGlobalState(prediction_time_ns, global_predictions)) {
      // Extract individual place predictions from global prediction
      for (const auto& [place_id, place_data] : temporal_place_data_) {
        if (place_data.total_observations < config_.min_observations) {
          continue; // Skip places with insufficient data
        }

        std::vector<double> place_prediction = extractPlacePrediction(place_id, global_predictions);
        all_predictions[place_id] = place_prediction;
      }
    } else {
      ROS_WARN("Global Fremen prediction failed for all places");
    }
  }

  // Fallback to historical averages for places without valid predictions
  for (const auto& [place_id, place_data] : temporal_place_data_) {
    if (place_data.total_observations < config_.min_observations) {
      continue;
    }

    if (all_predictions.find(place_id) == all_predictions.end()) {
      std::vector<double> predictions(config_.num_orientation_bins, 0.0);

      int total_historical = std::accumulate(place_data.historical_counts.begin(),
                                            place_data.historical_counts.end(), 0);
      if (total_historical > 0) {
        for (size_t bin = 0; bin < config_.num_orientation_bins; ++bin) {
          predictions[bin] = static_cast<double>(place_data.historical_counts[bin]) / total_historical;
        }
      } else {
        std::fill(predictions.begin(), predictions.end(), 1.0 / config_.num_orientation_bins);
      }

      all_predictions[place_id] = predictions;
    }
  }

  return all_predictions;
}

void TemporalDynamicsNode::updateTemporalModelsWithFremenPrediction(uint64_t prediction_time_ns) {
  // Update temporal models with Fremen predictions for specified time
  // This works like fremenUpdateTimerCallback() but uses predictions instead of current observations

  // Note: temporal_data_mutex_ is already held by caller

  ROS_INFO("Updating temporal models with Fremen predictions for time %lu (%zu places)...",
           prediction_time_ns, temporal_place_data_.size());

  // Skip if no connection and insufficient data
  if (!fremen_->isConnected() && temporal_place_data_.size() < static_cast<size_t>(min_places_for_connection_)) {
    ROS_DEBUG("Skipping Fremen prediction update - not connected or insufficient data");
    return;
  }

  // Initialize global model if needed
  if (!global_model_initialized_) {
    initializeGlobalFremenModel();
  }

  // Get global Fremen prediction for the specified time
  std::vector<double> global_predictions;
  bool prediction_success = false;

  if (fremen_->isConnected()) {
    prediction_success = fremen_->predictGlobalState(prediction_time_ns, global_predictions);
    if (!prediction_success) {
      ROS_WARN("Failed to get global Fremen prediction for time %lu", prediction_time_ns);
    }
  }

  // Update all place data with predicted values
  for (auto& [place_id, place_data] : temporal_place_data_) {
    if (place_data.total_observations < config_.min_observations) {
      continue; // Skip places with insufficient historical data
    }

    // Get predicted flow for this place
    std::vector<double> place_predictions;
    if (prediction_success && !global_predictions.empty()) {
      // Extract place prediction from global prediction
      place_predictions = extractPlacePrediction(place_id, global_predictions);
    } else {
      // Fallback to individual prediction
      place_predictions = predictFlow(place_id, prediction_time_ns);
    }

    // Update prediction metrics directly
    place_data.prediction_metrics.entropy = computeEntropyFromProbabilities(place_predictions);
    place_data.prediction_metrics.flow_magnitude = computeFlowMagnitude(place_predictions);
    place_data.prediction_metrics.best_angle = computeBestAngle(place_predictions);
    place_data.prediction_metrics.probabilities = place_predictions;
    place_data.prediction_metrics.prediction_time = prediction_time_ns;
    place_data.last_fremen_update = prediction_time_ns;
  }

  // ROS_INFO("Temporal models updated with Fremen predictions for %zu places at time %lu",
  //          temporal_place_data_.size(), prediction_time_ns);
}

std::vector<int> TemporalDynamicsNode::createGlobalStateVector() const {
  // Create stable spatial ordering for ALL temporal entries.
  // For hash cell entries: use their key directly.
  // For node-based (bound) entries: use original_hash_cell for stable ordering.
  // This ensures data remains at the same position in the global vector
  // regardless of whether it's in a hash cell or bound to a node.
  std::vector<std::pair<size_t, size_t>> ordering_key_to_data_key;
  ordering_key_to_data_key.reserve(temporal_place_data_.size());

  for (const auto& [data_key, place_data] : temporal_place_data_) {
    size_t ordering_key;
    if (isNodeBasedEntry(data_key, place_data)) {
      // Bound to a node — use the original hash cell for stable ordering
      ordering_key = place_data.original_hash_cell;
    } else {
      // Hash cell entry — use its key directly
      ordering_key = data_key;
    }
    ordering_key_to_data_key.emplace_back(ordering_key, data_key);
  }

  // Sort by ordering key to ensure consistent ordering
  std::sort(ordering_key_to_data_key.begin(), ordering_key_to_data_key.end());

  // Build state vector in stable spatial order
  std::vector<int> global_state;
  size_t total_size = ordering_key_to_data_key.size() * config_.num_orientation_bins;
  global_state.reserve(total_size);

  for (const auto& [ordering_key, data_key] : ordering_key_to_data_key) {
    auto it = temporal_place_data_.find(data_key);
    if (it != temporal_place_data_.end()) {
      // Use normalized counts
      std::vector<int> normalized = it->second.getNormalizedCounts();
      global_state.insert(global_state.end(), normalized.begin(), normalized.end());
    } else {
      // Spatial location has no data - fill with -1
      for (int i = 0; i < config_.num_orientation_bins; ++i) {
        global_state.push_back(-1);
      }
    }
  }

  return global_state;
}

std::vector<double> TemporalDynamicsNode::extractPlacePrediction(
    size_t query_key, const std::vector<double>& global_predictions) const {

  // Note: Caller should hold temporal_data_mutex_ before calling this function

  std::vector<double> place_prediction(config_.num_orientation_bins, 0.0);

  // Recreate the same ordering used in createGlobalStateVector
  std::vector<std::pair<size_t, size_t>> ordering_key_to_data_key;
  ordering_key_to_data_key.reserve(temporal_place_data_.size());

  for (const auto& [data_key, place_data] : temporal_place_data_) {
    size_t ordering_key;
    if (isNodeBasedEntry(data_key, place_data)) {
      ordering_key = place_data.original_hash_cell;
    } else {
      ordering_key = data_key;
    }
    ordering_key_to_data_key.emplace_back(ordering_key, data_key);
  }

  // Sort to match createGlobalStateVector ordering
  std::sort(ordering_key_to_data_key.begin(), ordering_key_to_data_key.end());

  // Determine the ordering key for the query
  size_t target_ordering_key = query_key;
  auto query_it = temporal_place_data_.find(query_key);
  if (query_it != temporal_place_data_.end() &&
      isNodeBasedEntry(query_key, query_it->second)) {
    target_ordering_key = query_it->second.original_hash_cell;
  }

  // Find the entry in the sorted order
  auto found_it = std::find_if(ordering_key_to_data_key.begin(), ordering_key_to_data_key.end(),
                               [target_ordering_key](const auto& pair) { return pair.first == target_ordering_key; });

  if (found_it != ordering_key_to_data_key.end()) {
    size_t index = std::distance(ordering_key_to_data_key.begin(), found_it);
    size_t start_idx = index * config_.num_orientation_bins;
    size_t end_idx = start_idx + config_.num_orientation_bins;

    if (end_idx <= global_predictions.size()) {
      std::copy(global_predictions.begin() + start_idx,
                global_predictions.begin() + end_idx,
                place_prediction.begin());
    }
  }

  return place_prediction;
}

void TemporalDynamicsNode::initializeGlobalFremenModel() {
  if (global_model_initialized_) {
    return;
  }

  ROS_INFO("Initializing global Fremen model with spatial hash ordering...");

  // Calculate total size based on current places (using stable spatial ordering)
  size_t total_size = temporal_place_data_.size() * config_.num_orientation_bins;

  if (total_size == 0) {
    // ROS_DEBUG("No places yet - deferring global model initialization");
    return;
  }

  // Initialize Fremen with empty global state
  if (fremen_->initializeGlobalModel(total_size)) {
    global_model_initialized_ = true;
    ROS_INFO("Initialized global Fremen model with %zu places (%zu total states) using spatial hash ordering",
             temporal_place_data_.size(), total_size);
  } else {
    ROS_WARN("Failed to initialize global Fremen model");
  }
}

}  // namespace aion
