#include "aion/temporal_dynamics_module.h"
#include <spark_dsg/serialization/graph_binary_serialization.h>
#include <thread>

namespace aion {

void TemporalDynamicsNode::hydraGraphCallback(const hydra_msgs::DsgUpdate::ConstPtr& msg) {
  // Store latest DSG update for periodic processing (rate-limited by place_sync_timer)

  ROS_DEBUG_THROTTLE(throttle_debug_general_, "DSG update received (size: %zu bytes) - storing for periodic sync",
                    msg->layer_contents.size());

  // Store the latest DSG message (thread-safe)
  {
    std::lock_guard<std::mutex> lock(latest_dsg_mutex_);
    latest_dsg_msg_ = msg;
  }

  ROS_DEBUG_THROTTLE(30.0, "DSG update stored - will be processed by place sync timer every %.1f seconds",
                    config_.place_sync_interval_seconds);
}

void TemporalDynamicsNode::processGraphUpdate(const hydra_msgs::DsgUpdate::ConstPtr& msg) {
  // Extract places directly without maintaining full scene graph

  try {
    // Option 1: Fast selective place extraction (if possible)
    std::map<spark_dsg::NodeId, Eigen::Vector3d> extracted_places;
    if (extractPlacesFromBinaryData(msg->layer_contents, extracted_places)) {
      // Success! Update places cache directly
      {
        std::lock_guard<std::shared_mutex> cache_lock(places_cache_mutex_);
        places_cache_ = std::move(extracted_places);
      }
      ROS_DEBUG("Fast extraction: %zu places from binary data", places_cache_.size());
      return;
    }

    // Option 2: Extract places without maintaining full graph
    ROS_DEBUG("Extracting places directly from DSG without full graph maintenance");

    // Directly extract navigation layer without storing the full graph
    auto temp_graph = spark_dsg::io::binary::readGraph(msg->layer_contents);
    if (temp_graph) {
      extractPlacesFromTempGraph(temp_graph);
      ROS_DEBUG("Extracted places from temporary graph with %zu total nodes", temp_graph->numNodes());
    }

  } catch (const std::exception& e) {
    ROS_ERROR("Failed to extract places from DSG: %s", e.what());
  }
}

void TemporalDynamicsNode::extractPlacesFromTempGraph(
    const std::shared_ptr<spark_dsg::DynamicSceneGraph>& temp_graph) {

  if (!temp_graph) {
    ROS_WARN("Temporary graph is null - cannot extract places");
    return;
  }

  ROS_DEBUG_THROTTLE(10.0, "Processing temporary graph with %zu layers", temp_graph->numLayers());

  // Build new places cache from temporary graph (no persistent storage)
  std::map<spark_dsg::NodeId, Eigen::Vector3d> new_places_cache;

  // Use appropriate layer based on configuration
  // Layer 20 = MESH_PLACES (2D navigation places) - contains mesh-based 2D place nodes
  // Layer 3 = PLACES (3D places) - contains 3D place nodes from GVD/navigation
  spark_dsg::LayerId target_layer;
  if (config_.target_layer_id >= 0) {
    target_layer = static_cast<spark_dsg::LayerId>(config_.target_layer_id);
    // ROS_DEBUG_THROTTLE(30.0, "Using override layer %d", config_.target_layer_id);
  } else {
    target_layer = config_.use_navigation_layer ? 20 : 3;
  }

  // Check available layers
  if (temp_graph->hasLayer(3)) {
    const auto& places_layer = temp_graph->getLayer(3);
  }

  if (temp_graph->hasLayer(20)) {
    const auto& mesh_places_layer = temp_graph->getLayer(20);
    // ROS_INFO_THROTTLE(10.0, "Layer 20 (MESH_PLACES): %zu nodes available", mesh_places_layer.numNodes());
  }

  // Also check other potentially relevant layers
  for (const auto& [layer_id, layer] : temp_graph->layers()) {
    if (layer_id != 3 && layer_id != 20) {
      ROS_DEBUG_THROTTLE(60.0, "Layer %zu: %zu nodes", layer_id, layer->numNodes());
    }
  }

  if (!temp_graph->hasLayer(target_layer)) {
    ROS_WARN_THROTTLE(10.0, "Temporary scene graph doesn't have layer %zu yet (available layers: %zu)",
                      target_layer, temp_graph->numLayers());
    // List available layers for debugging
    for (const auto& [layer_id, layer] : temp_graph->layers()) {
      ROS_DEBUG_THROTTLE(30.0, "Available layer: %zu", layer_id);
    }
    return;
  }

  const auto& layer = temp_graph->getLayer(target_layer);
  ROS_DEBUG_THROTTLE(30.0, "Layer %zu has %zu nodes", target_layer, layer.numNodes());

  // Debug: Track basic statistics
  int total_nodes_in_layer = 0;
  int filtered_out_by_z = 0;
  int filtered_out_by_position = 0;
  int nodes_added_to_cache = 0;

  for (const auto& [node_id, node] : layer.nodes()) {
    total_nodes_in_layer++;

    if (!node) {
      ROS_WARN_THROTTLE(30.0, "Null node found in layer %zu", target_layer);
      continue;
    }

    if (!node->attributes().position.allFinite()) {
      filtered_out_by_position++;
      ROS_DEBUG_THROTTLE(30.0, "Node %lu has non-finite position: [%.3f, %.3f, %.3f]",
                        node_id, node->attributes().position.x(),
                        node->attributes().position.y(), node->attributes().position.z());
      continue;
    }

    // Apply Z-height filtering if enabled
    if (config_.filter_by_z_height) {
      double z_height = node->attributes().position.z();
      if (z_height < config_.min_z_height || z_height > config_.max_z_height) {
        filtered_out_by_z++;
        ROS_DEBUG_THROTTLE(30.0, "Node %lu filtered by Z-height: %.3f (limits: %.1f to %.1f)",
                          node_id, z_height, config_.min_z_height, config_.max_z_height);
        continue;
      }
    }

    new_places_cache[node_id] = node->attributes().position;
    nodes_added_to_cache++;

  }

  // Log extraction statistics
  // ROS_INFO_THROTTLE(10.0, "Layer %zu extraction: %d total nodes, %d added to cache, %d filtered by position, %d filtered by Z-height",
  //                   target_layer, total_nodes_in_layer, nodes_added_to_cache, filtered_out_by_position, filtered_out_by_z);

  // Atomic update of places cache
  {
    std::lock_guard<std::shared_mutex> cache_lock(places_cache_mutex_);
    size_t old_size = places_cache_.size();
    places_cache_ = std::move(new_places_cache);

    // Mark spatial index for rebuild if places changed
    if (places_cache_.size() != old_size) {
      std::lock_guard<std::mutex> spatial_lock(spatial_index_mutex_);
      spatial_index_needs_rebuild_ = true;
      ROS_DEBUG("Marked spatial index for rebuild: %zu -> %zu places",
                old_size, places_cache_.size());

    }

    // Note: Position tracking is now handled automatically during binding/unbinding
    // in assignDynamicsToStableNodes() and handleNodeRemoval()
  }

  ROS_INFO_THROTTLE(30.0, "Extracted %zu places from layer %zu",
            places_cache_.size(), target_layer);
}

bool TemporalDynamicsNode::extractPlacesFromBinaryData(
    const std::vector<uint8_t>& binary_data,
    std::map<spark_dsg::NodeId, Eigen::Vector3d>& places) {

  // (TODO): Implement selective binary parsing
  // For now, return false to use full deserialization
  //
  // This would involve:
  // 1. Parsing the binary header to find layer offsets
  // 2. Seeking to the navigation layer (layer 2) section
  // 3. Parsing only the nodes in that layer
  // 4. Extracting position data without deserializing full attributes
  //
  // This is complex and would require understanding Spark DSG's binary format
  // For now, the background thread approach should be sufficient

  ROS_DEBUG_THROTTLE(60.0, "Selective parsing not yet implemented - using full deserialization");
  return false;
}

void TemporalDynamicsNode::processQueuedDsgUpdates() {
  std::queue<hydra_msgs::DsgUpdate::ConstPtr> local_queue;

  // Move all queued updates to local queue
  {
    std::lock_guard<std::mutex> lock(dsg_queue_mutex_);
    local_queue.swap(dsg_update_queue_);
  }

  // Process all updates (typically only the latest one matters)
  while (!local_queue.empty()) {
    auto msg = local_queue.front();
    local_queue.pop();

    // Only process the latest update for efficiency
    if (local_queue.empty()) {
      processGraphUpdate(msg);
    }
  }
}

spark_dsg::NodeId TemporalDynamicsNode::findNearestPlace(const Eigen::Vector3d& position) {
  // Use efficient KD-tree search if available, fallback to linear search
  if (config_.use_efficient_spatial_search && point_spatial_index_) {
    return findNearestPlaceEfficient(position);
  }

  // Original linear search O(n) - kept for compatibility and as fallback
  spark_dsg::NodeId nearest_id = 0;
  double min_distance = config_.detection_association_distance;

  for (const auto& [place_id, place_pos] : places_cache_) {
    double distance = (position - place_pos).norm();
    if (distance < min_distance) {
      min_distance = distance;
      nearest_id = place_id;
    }
  }

  return nearest_id;
}

spark_dsg::NodeId TemporalDynamicsNode::findNearestPlaceEfficient(const Eigen::Vector3d& position) {
  std::lock_guard<std::mutex> spatial_lock(spatial_index_mutex_);

  // Rebuild index if needed
  if (spatial_index_needs_rebuild_ || !point_spatial_index_) {
    const_cast<TemporalDynamicsNode*>(this)->rebuildSpatialIndex();
  }

  if (!point_spatial_index_ || place_nodes_for_index_.empty()) {
    ROS_INFO_THROTTLE(5.0, "Spatial index not available, fallback to linear search");
    std::shared_lock<std::shared_mutex> places_lock(places_cache_mutex_);

    // Fallback to linear search
    spark_dsg::NodeId nearest_id = 0;
    double min_distance = config_.detection_association_distance;

    for (const auto& [place_id, place_pos] : places_cache_) {
      double distance = (position - place_pos).norm();
      if (distance < min_distance) {
        min_distance = distance;
        nearest_id = place_id;
      }
    }
    return nearest_id;
  }

  // Use PointNeighborSearch for O(log n) search
  Eigen::Vector3f query_pos(position.x(), position.y(), position.z());
  float distance_squared;
  size_t closest_index;

  if (point_spatial_index_->search(query_pos, distance_squared, closest_index)) {
    double distance = std::sqrt(distance_squared);
    if (distance <= config_.detection_association_distance &&
        closest_index < place_nodes_for_index_.size()) {
      return place_nodes_for_index_[closest_index];
    }
  }

  return 0; // No suitable place found
}

void TemporalDynamicsNode::rebuildSpatialIndex() {
  // This method assumes spatial_index_mutex_ is already locked by caller

  std::shared_lock<std::shared_mutex> places_lock(places_cache_mutex_);

  if (places_cache_.empty()) {
    point_spatial_index_.reset();
    place_nodes_for_index_.clear();
    spatial_index_needs_rebuild_ = false;

    // Also clear spatial hash map
    std::lock_guard<std::mutex> hash_lock(spatial_hash_mutex_);
    spatial_hash_to_places_.clear();
    return;
  }

  // Instead of creating a SceneGraphLayer, use Hydra's PointNeighborSearch
  // which is simpler and designed for this exact use case
  std::vector<Eigen::Vector3f> place_positions;
  place_nodes_for_index_.clear();
  place_positions.reserve(places_cache_.size());
  place_nodes_for_index_.reserve(places_cache_.size());

  // Also rebuild spatial hash map for O(1) queries
  std::unordered_map<size_t, std::vector<spark_dsg::NodeId>> new_spatial_hash_map;

  // Populate positions and nodes for indexing
  for (const auto& [place_id, place_pos] : places_cache_) {
    place_positions.emplace_back(place_pos.x(), place_pos.y(), place_pos.z());
    place_nodes_for_index_.push_back(place_id);

    // Add to spatial hash map
    size_t hash_cell = getSpatialHashIndex(0, place_pos);
    new_spatial_hash_map[hash_cell].push_back(place_id);
  }

  // Update spatial hash map atomically
  {
    std::lock_guard<std::mutex> hash_lock(spatial_hash_mutex_);
    spatial_hash_to_places_ = std::move(new_spatial_hash_map);
  }

  // Build point-based spatial index instead of scene graph layer
  try {
    point_spatial_index_ = std::make_unique<hydra::PointNeighborSearch>(place_positions);
    spatial_index_needs_rebuild_ = false;

    ROS_DEBUG("Rebuilt spatial index with %zu places (%zu hash cells)",
              places_cache_.size(), spatial_hash_to_places_.size());
  } catch (const std::exception& e) {
    ROS_WARN("Failed to build spatial index: %s. Falling back to linear search", e.what());
    point_spatial_index_.reset();
    place_nodes_for_index_.clear();
  }
}

std::vector<spark_dsg::NodeId> TemporalDynamicsNode::findNearbyPlacesInHashCell(const Eigen::Vector3d& position) const {
  size_t hash_cell = getSpatialHashIndex(0, position);

  std::lock_guard<std::mutex> hash_lock(spatial_hash_mutex_);
  auto it = spatial_hash_to_places_.find(hash_cell);
  if (it != spatial_hash_to_places_.end()) {
    return it->second;
  }

  return {}; // Empty vector if no places in this hash cell
}

void TemporalDynamicsNode::placeSyncTimerCallback(const ros::TimerEvent& event) {
  // Periodic place synchronization from Hydra DSG (rate-limited background processing)

  hydra_msgs::DsgUpdate::ConstPtr msg_to_process;

  // Get the latest DSG message (if any)
  {
    std::lock_guard<std::mutex> lock(latest_dsg_mutex_);
    if (!latest_dsg_msg_) {
      // ROS_DEBUG_THROTTLE(60.0, "No DSG updates received yet - skipping place sync");
      return;
    }
    msg_to_process = latest_dsg_msg_;
    // Note: We don't clear latest_dsg_msg_ here, so we always process the most recent one
  }

  // ROS_DEBUG("Place sync timer: processing DSG update (size: %zu bytes)",
  //           msg_to_process->layer_contents.size());

  // Process the DSG update in background thread to avoid blocking the timer
  std::thread background_thread([this, msg_to_process]() {
    try {
      this->processGraphUpdate(msg_to_process);
      // ROS_DEBUG("Place sync completed successfully");
    } catch (const std::exception& e) {
      ROS_ERROR("Place sync failed: %s", e.what());
    }
  });
  background_thread.detach(); // Fire and forget
}

}  // namespace aion
