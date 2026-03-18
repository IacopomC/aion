#include "aion/temporal_dynamics_module.h"
#include <chrono>
#include <cmath>

namespace aion {

size_t TemporalDynamicsNode::getSpatialHashIndex(spark_dsg::NodeId place_id, const Eigen::Vector3d& position) const {
  int32_t hash_x = static_cast<int32_t>(std::floor(position.x() / config_.spatial_hash_grid_size));
  int32_t hash_y = static_cast<int32_t>(std::floor(position.y() / config_.spatial_hash_grid_size));

  // Bit-pack two 32-bit signed ints into one 64-bit size_t (collision-free, invertible)
  uint32_t ux = static_cast<uint32_t>(hash_x);
  uint32_t uy = static_cast<uint32_t>(hash_y);
  size_t spatial_index = (static_cast<size_t>(ux) << 32) | static_cast<size_t>(uy);

  return spatial_index;
}

size_t TemporalDynamicsNode::getStableSpatialIndex(size_t spatial_hash) const {
  // With pure spatial grid, the spatial hash IS the stable index
  // No need for complex mapping since grid cells are inherently stable
  return spatial_hash;
}

Eigen::Vector3d TemporalDynamicsNode::getPositionFromHashCell(size_t hash_cell) const {
  // Reverse the bit-packing from getSpatialHashIndex
  uint32_t ux = static_cast<uint32_t>(hash_cell >> 32);
  uint32_t uy = static_cast<uint32_t>(hash_cell & 0xFFFFFFFFULL);
  int32_t hash_x = static_cast<int32_t>(ux);
  int32_t hash_y = static_cast<int32_t>(uy);

  // Convert back to world coordinates (center of hash cell)
  double world_x = (hash_x + 0.5) * config_.spatial_hash_grid_size;
  double world_y = (hash_y + 0.5) * config_.spatial_hash_grid_size;
  double world_z = 0.0; // Hash cells are 2D, use ground level

  return Eigen::Vector3d(world_x, world_y, world_z);
}

// ============================================================================
// GLOBAL STABILITY WINDOW AND BINDING IMPLEMENTATION
// ============================================================================

void TemporalDynamicsNode::globalStabilityTimerCallback(const ros::TimerEvent& event) {
  if (!config_.enable_delayed_binding) {
    return;
  }

  ROS_INFO("Global stability timer triggered - assigning dynamics to stable nodes");
  assignDynamicsToStableNodes();
  handleNodeRemoval();
  publishVisualization();
}

void TemporalDynamicsNode::assignDynamicsToStableNodes() {
  auto start_time = std::chrono::high_resolution_clock::now();

  // GLOBAL MUTEX ORDER: places_cache_mutex_ → binding_mutex_ → temporal_data_mutex_
  std::shared_lock<std::shared_mutex> places_lock(places_cache_mutex_);
  std::lock_guard<std::mutex> binding_lock(binding_mutex_);
  std::lock_guard<std::shared_mutex> temporal_lock(temporal_data_mutex_);

  auto lock_acquired_time = std::chrono::high_resolution_clock::now();

  size_t assignments_made = 0;
  size_t hash_cells_processed = 0;
  size_t nodes_available = places_cache_.size();
  size_t total_observations_transferred = 0;

  // Collect hash cells to process (avoid iterator invalidation)
  std::vector<size_t> hash_cells_to_process;
  for (const auto& [hash_cell, place_data] : temporal_place_data_) {
    // Skip node-based entries and already bound data
    if (!isNodeBasedEntry(hash_cell, place_data) && !place_data.is_bound_to_node) {
      hash_cells_to_process.push_back(hash_cell);
    }
  }

  auto data_collection_time = std::chrono::high_resolution_clock::now();

  ROS_INFO("BINDING START: %zu hash cells to process, %zu nodes available",
           hash_cells_to_process.size(), nodes_available);

  // Process each hash cell safely
  for (size_t hash_cell : hash_cells_to_process) {
    hash_cells_processed++;
    auto it = temporal_place_data_.find(hash_cell);
    if (it == temporal_place_data_.end()) {
      continue; // Already processed or removed
    }

    auto& place_data = it->second;

    // Find the nearest node to this hash cell using efficient spatial search (2D distance for better binding)
    auto spatial_search_start = std::chrono::high_resolution_clock::now();
    spark_dsg::NodeId nearest_node_id = findNearestPlace(place_data.position);
    auto spatial_search_end = std::chrono::high_resolution_clock::now();

    if (nearest_node_id != 0) {
      // Track data being transferred
      total_observations_transferred += place_data.total_observations;

      // Calculate actual distance for monitoring (only for diagnostics)
      double actual_distance = 0.0;
      {
        auto node_pos_it = places_cache_.find(nearest_node_id);
        if (node_pos_it != places_cache_.end()) {
          actual_distance = (place_data.position - node_pos_it->second).norm();
          if (actual_distance > 1.0) {  // Log when distance seems unexpectedly large
            ROS_WARN("DISTANCE ALERT: Hash cell %zu -> Node %lu distance: %.3fm (may indicate spatial issue)",
                     hash_cell, nearest_node_id, actual_distance);
          }
        }
      }

      // Assign this temporal data to the nearest stable node
      size_t node_based_key = static_cast<size_t>(nearest_node_id) | 0x8000000000000000ULL;

      // Create new entry with node-based key
      auto& node_data = temporal_place_data_[node_based_key];
      node_data = place_data; // Full copy
      node_data.bindToNode(nearest_node_id, hash_cell);

      // Update position to current node position (not old hash cell position)
      auto node_pos_it = places_cache_.find(nearest_node_id);
      if (node_pos_it != places_cache_.end()) {
        node_data.position = node_pos_it->second; // Use current node position
      }

      // Update new metrics for the transferred data
      if (config_.enable_historical_computation) {
        node_data.historical_metrics.entropy = computeEntropyFromCounts(node_data.historical_counts);
        node_data.historical_metrics.flow_magnitude = computeFlowMagnitude(node_data.historical_counts);
        node_data.historical_metrics.best_angle = computeBestAngle(node_data.historical_counts);
        node_data.historical_metrics.probabilities = computeProbabilitiesFromCounts(node_data.historical_counts);
        node_data.historical_metrics.normalized_counts = computeNormalizedCounts(node_data.historical_counts);
      }

      // Remove old hash-cell based entry
      temporal_place_data_.erase(hash_cell);

      // Create binding record indexed by NODE ID, not hash cell
      TemporalBinding binding;
      binding.bound_node_id = nearest_node_id;
      binding.bind_time = ros::Time::now().toNSec();
      binding.original_hash_cell = hash_cell;
      binding.is_active = true;

      hash_bindings_[nearest_node_id] = binding;  // Use node ID as key!
      assignments_made++;

      // Quick success log (minimal overhead)
      // ROS_INFO("BOUND: Hash %zu -> Node %lu (%.2fm, %zu obs)",
      //          hash_cell, nearest_node_id, actual_distance, node_data.total_observations);
    }
  }


  if (assignments_made > 0)
  {
    // Force Fremen global model reinitialization since data moved
    global_model_initialized_ = false;
    total_bindings_created_ += assignments_made;
  }
}

void TemporalDynamicsNode::handleNodeRemoval() {
  // This method assumes global_stability_mutex_ is already held by caller

  // GLOBAL MUTEX ORDER: places_cache_mutex_ → binding_mutex_ → temporal_data_mutex_
  std::shared_lock<std::shared_mutex> places_lock(places_cache_mutex_);
  std::lock_guard<std::mutex> binding_lock(binding_mutex_);
  std::lock_guard<std::shared_mutex> temporal_lock(temporal_data_mutex_);

  // Check for nodes that have disappeared and need their temporal data released
  std::vector<spark_dsg::NodeId> missing_nodes;
  for (const auto& [node_id, binding] : hash_bindings_) {
    if (binding.is_active && binding.bound_node_id != 0) {
      // Check if this node still exists in places cache
      if (places_cache_.find(binding.bound_node_id) == places_cache_.end()) {
        missing_nodes.push_back(binding.bound_node_id);
      }
    }
  }

  // Release temporal data from missing nodes back to hash space
  for (spark_dsg::NodeId missing_node : missing_nodes) {
    // ROS_DEBUG("Node %lu disappeared - moving temporal data back to hash space", missing_node);

    // Find the node-based temporal data entry
    size_t node_based_key = static_cast<size_t>(missing_node) | 0x8000000000000000ULL;
    auto node_data_it = temporal_place_data_.find(node_based_key);

    if (node_data_it != temporal_place_data_.end()) {
      // Calculate hash cell for the node's last known position
      size_t target_hash_cell = getSpatialHashIndex(0, node_data_it->second.position);

      // Check if there's already data at the target hash cell
      auto hash_data_it = temporal_place_data_.find(target_hash_cell);

      if (hash_data_it != temporal_place_data_.end()) {
        // Merge the node data with existing hash cell data
        auto& target_data = hash_data_it->second;
        const auto& source_data = node_data_it->second;

        // Merge observation counts and times
        for (size_t i = 0; i < source_data.historical_counts.size(); ++i) {
          target_data.historical_counts[i] += source_data.historical_counts[i];
          target_data.current_interval_counts[i] += source_data.current_interval_counts[i];
        }
        target_data.observation_times.insert(target_data.observation_times.end(),
                                            source_data.observation_times.begin(),
                                            source_data.observation_times.end());
        target_data.total_observations += source_data.total_observations;
        target_data.unbindFromNode();

        // Update position back to hash cell spatial position
        Eigen::Vector3d hash_cell_position = getPositionFromHashCell(target_hash_cell);
        target_data.position = hash_cell_position;

        // Update new metrics for merged data
        if (config_.enable_historical_computation) {
          target_data.historical_metrics.entropy = computeEntropyFromCounts(target_data.historical_counts);
          target_data.historical_metrics.flow_magnitude = computeFlowMagnitude(target_data.historical_counts);
          target_data.historical_metrics.best_angle = computeBestAngle(target_data.historical_counts);
          target_data.historical_metrics.probabilities = computeProbabilitiesFromCounts(target_data.historical_counts);
          target_data.historical_metrics.normalized_counts = computeNormalizedCounts(target_data.historical_counts);
        }

        // ROS_INFO("Merged node %lu data into existing hash cell %zu", missing_node, target_hash_cell);
      } else {
        // Create new hash cell entry
        auto& hash_data = temporal_place_data_[target_hash_cell];
        hash_data = node_data_it->second; // Full copy
        hash_data.unbindFromNode();

        // Update position back to hash cell spatial position
        Eigen::Vector3d hash_cell_position = getPositionFromHashCell(target_hash_cell);
        hash_data.position = hash_cell_position;

        // Update new metrics for the transferred data
        if (config_.enable_historical_computation) {
          hash_data.historical_metrics.entropy = computeEntropyFromCounts(hash_data.historical_counts);
          hash_data.historical_metrics.flow_magnitude = computeFlowMagnitude(hash_data.historical_counts);
          hash_data.historical_metrics.best_angle = computeBestAngle(hash_data.historical_counts);
          hash_data.historical_metrics.probabilities = computeProbabilitiesFromCounts(hash_data.historical_counts);
          hash_data.historical_metrics.normalized_counts = computeNormalizedCounts(hash_data.historical_counts);
        }

        // ROS_INFO("Moved node %lu data to new hash cell %zu", missing_node, target_hash_cell);
      }

      // Remove node-based entry
      temporal_place_data_.erase(node_data_it);

      // Force Fremen global model reinitialization since data moved
      global_model_initialized_ = false;
    }

    // Deactivate the binding - now using node ID as key
    auto binding_it = hash_bindings_.find(missing_node);
    if (binding_it != hash_bindings_.end()) {
      binding_it->second.is_active = false;
      binding_it->second.bound_node_id = 0;
      // Optionally remove the binding entirely
      hash_bindings_.erase(binding_it);
    }
  }

  if (!missing_nodes.empty()) {
    // ROS_INFO("Handled removal of %zu nodes - moved temporal data back to hash space",
    //          missing_nodes.size());
  }
}

size_t TemporalDynamicsNode::getTemporalDataLocation(const Eigen::Vector3d& position) {
  if (!config_.enable_delayed_binding) {
    // Legacy behavior - use spatial hash directly
    return getSpatialHashIndex(0, position);
  }

  // Check if there's a bound node at this position (timer creates bindings when it's time)
  std::lock_guard<std::mutex> binding_lock(binding_mutex_);

  // Find the nearest node to this position using efficient spatial search
  spark_dsg::NodeId nearest_node_id = findNearestPlace(position);

  // If we found a nearby node and it's bound, use its bound location
  if (nearest_node_id != 0) {
    auto binding_it = hash_bindings_.find(nearest_node_id);
    if (binding_it != hash_bindings_.end() && binding_it->second.is_active) {
      // Return the node-based key where data actually lives
      return static_cast<size_t>(nearest_node_id) | 0x8000000000000000ULL;
    }
  }

  // Default to spatial hash for the position
  return getSpatialHashIndex(0, position);
}

bool TemporalDynamicsNode::isNodeBasedEntry(size_t spatial_key, const TemporalPlaceData& place_data) const {
  // Node-based entries have:
  // 1. is_bound_to_node = true
  // 2. bound_node_id != 0
  // 3. spatial_key has high bit set (0x8000000000000000ULL)
  return (place_data.is_bound_to_node &&
          place_data.bound_node_id != 0 &&
          (spatial_key & 0x8000000000000000ULL) != 0);
}

}  // namespace aion
