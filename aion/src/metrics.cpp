#include "aion/temporal_dynamics_module.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace aion {

void TemporalDynamicsNode::computeHistoricalMetrics() {
  // NOTE: temporal_data_mutex_ is already locked by calling function

  size_t processed_count = 0;
  for (auto& [place_id, place_data] : temporal_place_data_) {

    if (place_data.total_observations < config_.min_observations) {
      processed_count++;
      continue; // Skip places with insufficient data
    }

    place_data.historical_metrics.entropy = computeEntropyFromCounts(place_data.historical_counts);

    place_data.historical_metrics.flow_magnitude = computeFlowMagnitude(place_data.historical_counts);

    place_data.historical_metrics.best_angle = computeBestAngle(place_data.historical_counts);

    place_data.historical_metrics.probabilities = computeProbabilitiesFromCounts(place_data.historical_counts);

    place_data.historical_metrics.normalized_counts = computeNormalizedCounts(place_data.historical_counts);

    processed_count++;
  }

}

void TemporalDynamicsNode::computePredictionMetrics() {
  // NOTE: temporal_data_mutex_ is already locked by calling function

  const uint64_t current_time = ros::Time::now().toNSec();

  // Get global predictions if Fremen is available
  std::vector<double> global_predictions;
  bool prediction_success = false;


  if (fremen_->isConnected() && global_model_initialized_) {
    prediction_success = fremen_->predictGlobalState(current_time, global_predictions);
    if (!prediction_success) {
      ROS_DEBUG("Failed to get global Fremen prediction");
    }
  }

  size_t prediction_count = 0;
  for (auto& [place_id, place_data] : temporal_place_data_) {

    if (place_data.total_observations < config_.min_observations) {
      prediction_count++;
      continue; // Skip places with insufficient data
    }

    // Get predictions for this place
    std::vector<double> predictions;
    if (prediction_success && !global_predictions.empty()) {
      predictions = extractPlacePrediction(place_id, global_predictions);
    } else {
      // Fallback to historical probabilities
      predictions = place_data.historical_metrics.probabilities;
    }

    // Compute all prediction metrics
    place_data.prediction_metrics.entropy = computeEntropyFromProbabilities(predictions);
    place_data.prediction_metrics.flow_magnitude = computeFlowMagnitude(predictions);
    place_data.prediction_metrics.best_angle = computeBestAngle(predictions);
    place_data.prediction_metrics.probabilities = predictions;
    place_data.prediction_metrics.prediction_time = current_time;
  }

  ROS_DEBUG("Computed prediction metrics for %zu places", temporal_place_data_.size());
}

void TemporalDynamicsNode::publishTemporalNodes() {
  // NOTE: temporal_data_mutex_ is already locked by calling function

  for (const auto& [place_id, place_data] : temporal_place_data_) {
    if (place_data.total_observations < config_.min_observations) {
      continue; // Skip places with insufficient data
    }

    aion::AionTemporalNode msg = createTemporalNodeMessage(place_id, place_data);
    temporal_nodes_pub_.publish(msg);
  }

  ROS_DEBUG("Published temporal nodes for %zu places", temporal_place_data_.size());
}

void TemporalDynamicsNode::publishTemporalMap() {
  // NOTE: temporal_data_mutex_ is already locked by calling function
  std::shared_lock<std::shared_mutex> places_lock(places_cache_mutex_);

  aion::AionTemporalMap map_msg;
  map_msg.header.stamp = ros::Time::now();
  map_msg.header.frame_id = config_.frame_id;
  map_msg.update_interval_seconds = config_.update_interval_seconds;
  map_msg.num_orientation_bins = config_.num_orientation_bins;

  uint32_t bound_count = 0;
  uint32_t hash_count = 0;

  // Process all temporal data entries
  for (const auto& [place_id, place_data] : temporal_place_data_) {
    if (place_data.total_observations < config_.min_observations) {
      continue; // Skip places with insufficient data
    }

    // Determine position for this entry
    Eigen::Vector3d position;
    bool has_dsg_position = false;

    if (place_data.is_bound_to_node && place_data.bound_node_id != 0) {
      // This is bound to a DSG node - use current DSG position
      auto dsg_pos_it = places_cache_.find(place_data.bound_node_id);
      if (dsg_pos_it != places_cache_.end()) {
        position = dsg_pos_it->second;
        has_dsg_position = true;
        bound_count++;
      } else {
        // DSG node disappeared - use stored position
        position = place_data.position;
        bound_count++;
      }
    } else {
      // Hash cell entry - use stored position
      position = place_data.position;
      hash_count++;
    }

    // Add historical node if enabled
    if (config_.enable_historical_computation) {
      aion::AionHistoricalNode hist_node;
      hist_node.place_id = place_data.is_bound_to_node ? place_data.bound_node_id : place_id;
      hist_node.position.x = position.x();
      hist_node.position.y = position.y();
      hist_node.position.z = position.z();
      hist_node.entropy = place_data.historical_metrics.entropy;
      hist_node.flow_magnitude = place_data.historical_metrics.flow_magnitude;
      hist_node.best_angle = place_data.historical_metrics.best_angle;
      hist_node.probabilities = place_data.historical_metrics.probabilities;
      hist_node.normalized_counts = place_data.historical_metrics.normalized_counts;
      hist_node.total_observations = place_data.total_observations;
      hist_node.is_bound_to_node = place_data.is_bound_to_node;

      map_msg.historical_nodes.push_back(hist_node);
    }

    // Add temporal node if enabled
    if (config_.enable_prediction_computation) {
      aion::AionTemporalNode temp_node;
      temp_node.place_id = place_data.is_bound_to_node ? place_data.bound_node_id : place_id;
      temp_node.position.x = position.x();
      temp_node.position.y = position.y();
      temp_node.position.z = position.z();
      temp_node.entropy = place_data.prediction_metrics.entropy;
      temp_node.flow_magnitude = place_data.prediction_metrics.flow_magnitude;
      temp_node.best_angle = place_data.prediction_metrics.best_angle;
      temp_node.probabilities = place_data.prediction_metrics.probabilities;
      temp_node.prediction_time = place_data.prediction_metrics.prediction_time;
      temp_node.is_bound_to_node = place_data.is_bound_to_node;

      map_msg.temporal_nodes.push_back(temp_node);
    }
  }

  map_msg.total_places = bound_count + hash_count;
  map_msg.bound_places = bound_count;
  map_msg.hash_places = hash_count;

  temporal_map_pub_.publish(map_msg);

  ROS_DEBUG("Published AionTemporalMap: %zu historical nodes, %zu temporal nodes (%u bound, %u hash)",
            map_msg.historical_nodes.size(), map_msg.temporal_nodes.size(), bound_count, hash_count);
}

aion::AionHistoricalNode TemporalDynamicsNode::createHistoricalNodeMessage(
    size_t place_id, const TemporalPlaceData& place_data) {

  aion::AionHistoricalNode msg;
  msg.place_id = place_id;
  msg.position.x = place_data.position.x();
  msg.position.y = place_data.position.y();
  msg.position.z = place_data.position.z();

  msg.best_angle = place_data.historical_metrics.best_angle;
  msg.raw_counts.assign(place_data.historical_counts.begin(), place_data.historical_counts.end());
  msg.probabilities = place_data.historical_metrics.probabilities;
  msg.normalized_counts = place_data.historical_metrics.normalized_counts;
  msg.flow_magnitude = place_data.historical_metrics.flow_magnitude;
  msg.entropy = place_data.historical_metrics.entropy;
  msg.total_observations = place_data.total_observations;
  msg.is_bound_to_node = place_data.is_bound_to_node;
  msg.spatial_hash = place_id; // Using place_id as spatial hash for now

  return msg;
}

aion::AionTemporalNode TemporalDynamicsNode::createTemporalNodeMessage(
    size_t place_id, const TemporalPlaceData& place_data) {

  aion::AionTemporalNode msg;
  msg.place_id = place_id;
  msg.position.x = place_data.position.x();
  msg.position.y = place_data.position.y();
  msg.position.z = place_data.position.z();

  msg.best_angle = place_data.prediction_metrics.best_angle;
  msg.probabilities = place_data.prediction_metrics.probabilities;
  msg.flow_magnitude = place_data.prediction_metrics.flow_magnitude;
  msg.entropy = place_data.prediction_metrics.entropy;
  msg.prediction_time = place_data.prediction_metrics.prediction_time;
  msg.is_bound_to_node = place_data.is_bound_to_node;
  msg.spatial_hash = place_id; // Using place_id as spatial hash for now

  return msg;
}

double TemporalDynamicsNode::computeFlowMagnitude(const std::vector<int>& counts) const {
  return static_cast<double>(std::accumulate(counts.begin(), counts.end(), 0));
}

double TemporalDynamicsNode::computeFlowMagnitude(const std::vector<double>& probabilities) const {
  return std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
}

double TemporalDynamicsNode::computeEntropyFromCounts(const std::vector<int>& counts) const {
  int total_count = std::accumulate(counts.begin(), counts.end(), 0);

  if (total_count < config_.min_observations) {
    return 0.0; // Not enough data for meaningful entropy
  }

  // Calculate Shannon entropy: H = -Σ(p * log2(p))
  double entropy = 0.0;
  for (int count : counts) {
    if (count > 0) {
      double probability = static_cast<double>(count) / total_count;
      entropy -= probability * std::log2(probability);
    }
  }

  // Apply bias correction (Miller-Madow estimator)
  const int bias_correction_threshold = 20;
  if (total_count >= bias_correction_threshold) {
    double bias_correction = (static_cast<double>(config_.num_orientation_bins - 1) /
                             (2.0 * total_count)) * std::log2(2.718281828);
    entropy += bias_correction;
  }

  return std::max(0.0, entropy);
}

double TemporalDynamicsNode::computeEntropyFromProbabilities(const std::vector<double>& probabilities) const {
  double total_prob = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);

  if (total_prob < 0.01) { // Very low total probability
    return 0.0;
  }

  // Calculate Shannon entropy: H = -Σ(p * log2(p))
  double entropy = 0.0;
  for (double prob : probabilities) {
    if (prob > 0) {
      double normalized_prob = prob / total_prob; // Normalize to ensure sum = 1
      entropy -= normalized_prob * std::log2(normalized_prob);
    }
  }

  return std::max(0.0, entropy);
}

double TemporalDynamicsNode::computeBestAngle(const std::vector<int>& counts) const {
  if (counts.empty()) return 0.0;

  // Find the orientation bin with highest count
  size_t dominant_bin = 0;
  int max_count = counts[0];

  for (size_t i = 1; i < counts.size(); ++i) {
    if (counts[i] > max_count) {
      max_count = counts[i];
      dominant_bin = i;
    }
  }

  // Convert bin to angle
  double angle = (dominant_bin * 2.0 * M_PI) / config_.num_orientation_bins;
  return angle;
}

double TemporalDynamicsNode::computeBestAngle(const std::vector<double>& probabilities) const {
  if (probabilities.empty()) return 0.0;

  // Find the orientation bin with highest probability
  size_t dominant_bin = 0;
  double max_prob = probabilities[0];

  for (size_t i = 1; i < probabilities.size(); ++i) {
    if (probabilities[i] > max_prob) {
      max_prob = probabilities[i];
      dominant_bin = i;
    }
  }

  // Convert bin to angle
  double angle = (dominant_bin * 2.0 * M_PI) / config_.num_orientation_bins;
  return angle;
}

std::vector<double> TemporalDynamicsNode::computeProbabilitiesFromCounts(const std::vector<int>& counts) const {
  std::vector<double> probabilities(counts.size(), 0.0);
  int total_count = std::accumulate(counts.begin(), counts.end(), 0);

  if (total_count > 0) {
    for (size_t i = 0; i < counts.size(); ++i) {
      probabilities[i] = static_cast<double>(counts[i]) / total_count;
    }
  }

  return probabilities;
}

std::vector<int> TemporalDynamicsNode::computeNormalizedCounts(const std::vector<int>& counts) const {
  std::vector<int> normalized = counts;
  int max_count = *std::max_element(normalized.begin(), normalized.end());

  if (max_count > 0) {
    // Normalize to 0-100 range (like Fremen)
    for (int& count : normalized) {
      count = (count * 100) / max_count;
    }
  } else {
    // No data - fill with -1
    std::fill(normalized.begin(), normalized.end(), -1);
  }

  return normalized;
}

// ============================================================================
// Metric Access Functions
// ============================================================================

double TemporalDynamicsNode::getEntropy(const TemporalPlaceData& place_data) const {
  if (config_.visualization_data_source == "predictions" &&
      config_.enable_prediction_computation) {
    return place_data.prediction_metrics.entropy;
  } else if (config_.visualization_data_source == "historical" &&
             config_.enable_historical_computation) {
    return place_data.historical_metrics.entropy;
  } else {
    // Fallback: compute entropy on-demand from historical counts
    return computeEntropyFromCounts(place_data.historical_counts);
  }
}

double TemporalDynamicsNode::getFlowMagnitude(const TemporalPlaceData& place_data) const {
  if (config_.visualization_data_source == "predictions" &&
      config_.enable_prediction_computation) {
    return place_data.prediction_metrics.flow_magnitude;
  } else if (config_.visualization_data_source == "historical" &&
             config_.enable_historical_computation) {
    return place_data.historical_metrics.flow_magnitude;
  } else {
    // Fallback: compute from total observations
    return static_cast<double>(place_data.total_observations);
  }
}

double TemporalDynamicsNode::getBestAngle(const TemporalPlaceData& place_data) const {
  if (config_.visualization_data_source == "predictions" &&
      config_.enable_prediction_computation) {
    return place_data.prediction_metrics.best_angle;
  } else if (config_.visualization_data_source == "historical" &&
             config_.enable_historical_computation) {
    return place_data.historical_metrics.best_angle;
  } else {
    // Fallback: compute best angle on-demand from historical counts
    return computeBestAngle(place_data.historical_counts);
  }
}

std::vector<double> TemporalDynamicsNode::getFlowProbabilities(const TemporalPlaceData& place_data) const {
  if (config_.visualization_data_source == "predictions" &&
      config_.enable_prediction_computation) {
    return place_data.prediction_metrics.probabilities;
  } else if (config_.visualization_data_source == "historical" &&
             config_.enable_historical_computation) {
    return place_data.historical_metrics.probabilities;
  } else {
    // Fallback: compute probabilities on-demand from historical counts
    return computeProbabilitiesFromCounts(place_data.historical_counts);
  }
}

}  // namespace aion
