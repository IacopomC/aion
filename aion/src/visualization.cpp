#include "aion/temporal_dynamics_module.h"
#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Point.h>
#include <cmath>

namespace aion {

void TemporalDynamicsNode::publishVisualization() {

  // Publish unified temporal markers for RViz visualization
  auto unified_markers = createUnifiedTemporalMarkers();
  temporal_places_pub_.publish(unified_markers);

  // Also publish structured AionTemporalMap data for external systems
  publishTemporalMap();

  // Publish debug visualization for delayed binding
  if (config_.enable_delayed_binding) {
    auto hash_markers = createHashCellMarkers();
    hash_cells_pub_.publish(hash_markers);

    // auto debug_binding_markers = createDebugBindingMarkers();
    // debug_bindings_pub_.publish(debug_binding_markers);
  }

}

visualization_msgs::MarkerArray TemporalDynamicsNode::createUnifiedTemporalMarkers() {
  visualization_msgs::MarkerArray markers;

  size_t marker_id = 0;

  // Clear old markers each time to prevent accumulation
  visualization_msgs::Marker delete_marker;
  delete_marker.header.frame_id = config_.frame_id;
  delete_marker.header.stamp = ros::Time::now();
  delete_marker.action = visualization_msgs::Marker::DELETEALL;
  markers.markers.push_back(delete_marker);

  // Resolve which mode to use.
  // visualization_mode takes precedence over the legacy use_sphere_markers flag.
  const std::string& mode = config_.visualization_mode;
  const bool use_sphere = (mode == "entropy") ||
                          (mode != "bins" && mode != "direction" && config_.use_sphere_markers);

  for (const auto& [place_id, place_pos] : places_cache_) {
    // First check if this node has bound temporal data
    const TemporalPlaceData* place_data_ptr = nullptr;

    if (config_.enable_delayed_binding) {
      // Look for bound temporal data by checking if this node has a binding
      std::lock_guard<std::mutex> binding_lock(binding_mutex_);
      auto binding_it = hash_bindings_.find(place_id);
      if (binding_it != hash_bindings_.end() && binding_it->second.is_active) {
        // This node has bound data - find it using node-based key
        size_t node_based_key = static_cast<size_t>(place_id) | 0x8000000000000000ULL;
        auto node_data_it = temporal_place_data_.find(node_based_key);
        if (node_data_it != temporal_place_data_.end()) {
          place_data_ptr = &node_data_it->second;
        } else {
          ROS_ERROR("VIZ ERROR: Node %lu has binding but no node-based data found (key: %zu)",
                   place_id, node_based_key);
        }
      }
    } else {
      // Legacy mode: delayed binding disabled - check spatial hash location
      size_t spatial_key = getSpatialHashIndex(0, place_pos);
      auto temporal_it = temporal_place_data_.find(spatial_key);
      if (temporal_it != temporal_place_data_.end()) {
        place_data_ptr = &temporal_it->second;
      }
    }

    if (place_data_ptr != nullptr) {
      const auto& place_data = *place_data_ptr;
      if (place_data.total_observations < config_.min_observations) {
        continue;
      }

      const double activity_intensity =
          std::min(1.0, place_data.total_observations / config_.activity_scaling_factor);
      const double confidence =
          std::min(1.0, place_data.total_observations / config_.confidence_scaling_factor);
      const double base_scale = config_.marker_scale;

      if (mode == "bins") {
        // ----------------------------------------------------------------
        // BINS mode: draw one arrow per orientation bin, scaled by its
        // probability.  Low-probability bins (< half the uniform value)
        // are skipped to avoid visual clutter.
        // ----------------------------------------------------------------
        const std::vector<double> probs = getFlowProbabilities(place_data);
        const double min_prob_to_show =
            1.0 / static_cast<double>(config_.num_orientation_bins) * 0.5;
        const double max_arrow_len =
            config_.arrow_scale * (arrow_length_min_multiplier_ +
                                   (arrow_length_max_multiplier_ - arrow_length_min_multiplier_) *
                                       activity_intensity);

        // Find the maximum probability (to normalise per-bin alpha)
        double max_prob = 0.0;
        for (double p : probs) max_prob = std::max(max_prob, p);
        if (max_prob <= 0.0) continue;

        // Base color comes from entropy (same visual language as other modes)
        const std_msgs::ColorRGBA entropy_color = entropyToColor(getEntropy(place_data));

        for (int bin = 0; bin < config_.num_orientation_bins; ++bin) {
          const double prob = (bin < static_cast<int>(probs.size())) ? probs[bin] : 0.0;
          if (prob < min_prob_to_show) continue;

          const double bin_angle =
              (bin * 2.0 * M_PI) / static_cast<double>(config_.num_orientation_bins);
          const double arrow_len = max_arrow_len * prob;

          visualization_msgs::Marker bin_marker;
          bin_marker.header.frame_id = config_.frame_id;
          bin_marker.header.stamp = ros::Time::now();
          bin_marker.id = marker_id++;
          bin_marker.type = visualization_msgs::Marker::ARROW;
          bin_marker.action = visualization_msgs::Marker::ADD;

          // Two-point arrow
          geometry_msgs::Point start_pt, end_pt;
          start_pt.x = place_pos.x();
          start_pt.y = place_pos.y();
          start_pt.z = temporal_marker_z_level_;
          end_pt.x = place_pos.x() + arrow_len * std::cos(bin_angle);
          end_pt.y = place_pos.y() + arrow_len * std::sin(bin_angle);
          end_pt.z = temporal_marker_z_level_;
          bin_marker.points.push_back(start_pt);
          bin_marker.points.push_back(end_pt);

          // Shaft and head scale the shaft diameter by relative probability so
          // the dominant bin stands out with a visibly thicker arrow.
          const double rel_prob = prob / max_prob;  // [0,1]
          bin_marker.scale.x = arrow_shaft_diameter_ * (0.4 + 0.6 * rel_prob);
          bin_marker.scale.y = arrow_head_diameter_ * (0.4 + 0.6 * rel_prob);
          bin_marker.scale.z = arrow_head_length_;

          // Same hue as entropy color; alpha weighted by relative probability
          bin_marker.color = entropy_color;
          bin_marker.color.a = (marker_transparency_min_alpha_ +
                                (marker_transparency_max_alpha_ - marker_transparency_min_alpha_) *
                                    confidence) *
                               rel_prob;

          markers.markers.push_back(bin_marker);
        }

      } else if (use_sphere) {
        // ----------------------------------------------------------------
        // ENTROPY mode: single sphere per place
        // ----------------------------------------------------------------
        visualization_msgs::Marker marker;
        marker.header.frame_id = config_.frame_id;
        marker.header.stamp = ros::Time::now();
        marker.id = marker_id++;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = place_pos.x();
        marker.pose.position.y = place_pos.y();
        marker.pose.position.z = temporal_marker_z_level_;

        const double size_mult = marker_size_min_multiplier_ +
                                 (marker_size_max_multiplier_ - marker_size_min_multiplier_) *
                                     activity_intensity;
        marker.scale.x = base_scale * size_mult;
        marker.scale.y = base_scale * size_mult;
        marker.scale.z = base_scale * size_mult;
        marker.pose.orientation.w = 1.0;

        marker.color = entropyToColor(getEntropy(place_data));
        marker.color.a = marker_transparency_min_alpha_ +
                         (marker_transparency_max_alpha_ - marker_transparency_min_alpha_) *
                             confidence;

        markers.markers.push_back(marker);

      } else {
        // ----------------------------------------------------------------
        // DIRECTION mode (default): single arrow pointing in dominant
        // flow direction.
        // ----------------------------------------------------------------
        visualization_msgs::Marker marker;
        marker.header.frame_id = config_.frame_id;
        marker.header.stamp = ros::Time::now();
        marker.id = marker_id++;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;

        const double dominant_angle = getBestAngle(place_data);
        const double arrow_length =
            config_.arrow_scale *
            (arrow_length_min_multiplier_ +
             (arrow_length_max_multiplier_ - arrow_length_min_multiplier_) * activity_intensity);

        geometry_msgs::Point start_point;
        start_point.x = place_pos.x();
        start_point.y = place_pos.y();
        start_point.z = temporal_marker_z_level_;
        marker.points.push_back(start_point);

        geometry_msgs::Point end_point;
        end_point.x = place_pos.x() + arrow_length * std::cos(dominant_angle);
        end_point.y = place_pos.y() + arrow_length * std::sin(dominant_angle);
        end_point.z = temporal_marker_z_level_;
        marker.points.push_back(end_point);

        marker.scale.x = arrow_shaft_diameter_;
        marker.scale.y = arrow_head_diameter_;
        marker.scale.z = arrow_head_length_;

        marker.color = entropyToColor(getEntropy(place_data));
        marker.color.a = marker_transparency_min_alpha_ +
                         (marker_transparency_max_alpha_ - marker_transparency_min_alpha_) *
                             confidence;

        markers.markers.push_back(marker);
      }

    } else {
      // This place has no temporal data - create small gray sphere
      visualization_msgs::Marker marker;
      marker.header.frame_id = config_.frame_id;
      marker.header.stamp = ros::Time::now();
      marker.id = marker_id++;
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;

      marker.pose.position.x = place_pos.x();
      marker.pose.position.y = place_pos.y();
      marker.pose.position.z = inactive_marker_z_level_;
      marker.pose.orientation.w = 1.0;

      marker.scale.x = config_.marker_scale * inactive_marker_scale_factor_;
      marker.scale.y = config_.marker_scale * inactive_marker_scale_factor_;
      marker.scale.z = config_.marker_scale * inactive_marker_scale_factor_;

      marker.color.r = inactive_marker_color_r_;
      marker.color.g = inactive_marker_color_g_;
      marker.color.b = inactive_marker_color_b_;
      marker.color.a = inactive_marker_alpha_;

      markers.markers.push_back(marker);
    }
  }

  return markers;
}

double TemporalDynamicsNode::getDominantFlowDirection(const std::vector<double>& predictions) const {
  if (predictions.empty()) return 0.0;

  // Find the orientation bin with highest prediction
  size_t dominant_bin = 0;
  double max_prediction = predictions[0];

  for (size_t i = 1; i < predictions.size(); ++i) {
    if (predictions[i] > max_prediction) {
      max_prediction = predictions[i];
      dominant_bin = i;
    }
  }

  // Convert bin to angle
  double angle = (dominant_bin * 2.0 * M_PI) / config_.num_orientation_bins;
  return angle;
}

std_msgs::ColorRGBA TemporalDynamicsNode::entropyToColor(double entropy) const {
  std_msgs::ColorRGBA color;

  double normalized = std::min(1.0, entropy / max_entropy_for_normalization_);

  color.r = normalized;
  color.g = 0.0;
  color.b = 1.0 - normalized;

  return color;
}

visualization_msgs::MarkerArray TemporalDynamicsNode::createHashCellMarkers() {
  visualization_msgs::MarkerArray markers;
  // Note: temporal_data_mutex_ is already held by caller (fremenUpdateTimerCallback)
  std::lock_guard<std::mutex> binding_lock(binding_mutex_);

  size_t marker_id = 0;

  // Clear old markers
  visualization_msgs::Marker clear_marker;
  clear_marker.header.frame_id = config_.frame_id;
  clear_marker.header.stamp = ros::Time::now();
  clear_marker.ns = "debug_hash_cells";
  clear_marker.action = visualization_msgs::Marker::DELETEALL;
  markers.markers.push_back(clear_marker);

  for (const auto& [spatial_key, place_data] : temporal_place_data_) {
    // Skip node-based entries - they have is_bound_to_node=true AND bound_node_id != 0
    // AND their key should be node_id + 1000000000UL
    if (isNodeBasedEntry(spatial_key, place_data)) {
      // This is a node-based entry, not a hash cell - don't show as hash marker
      // ROS_INFO("Skipping node-based entry %zu (bound to node %lu) in hash cell visualization",
      //           spatial_key, place_data.bound_node_id);
      continue;
    }

    visualization_msgs::Marker marker;
    marker.header.frame_id = config_.frame_id;
    marker.header.stamp = ros::Time::now();
    marker.ns = "debug_hash_cells";
    marker.id = marker_id++;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;

    // Position at hash cell location (slightly below main markers)
    marker.pose.position.x = place_data.position.x();
    marker.pose.position.y = place_data.position.y();
    marker.pose.position.z = temporal_marker_z_level_ - 0.5; // Below main markers
    marker.pose.orientation.w = 1.0;

    // Small cube to represent hash cell
    marker.scale.x = config_.spatial_hash_grid_size;
    marker.scale.y = config_.spatial_hash_grid_size;
    marker.scale.z = 0.1;

    // Color coding: Use entropy-based coloring same as temporal dynamics layer
    // This helps verify that colors match between hash cells and temporal nodes
    if (place_data.is_bound_to_node) {
      // This should be rare - a hash cell marked as bound but not matching node-based pattern
      ROS_WARN_THROTTLE(10.0, "Hash cell %zu marked as bound to node %lu but doesn't match node-based key pattern - investigating",
                        spatial_key, place_data.bound_node_id);
      // Still show it but with different color to debug
      marker.color.r = 0.0;
      marker.color.g = 1.0; // Green for problematic entries
      marker.color.b = 0.0;
      marker.color.a = 0.8;
    } else {
      // For unbound hash cells, use entropy-based coloring like temporal nodes
      // Skip cells with insufficient data for meaningful entropy
      if (place_data.total_observations >= config_.min_observations) {
        marker.color = entropyToColor(getEntropy(place_data));
        marker.color.a = 0.7; // Visible for active temporal data
      } else {
        // Insufficient data - use gray color
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.5;
        marker.color.a = 0.4;
      }
    }

    markers.markers.push_back(marker);
  }

  // ROS_DEBUG("Temporal visualization: %lu total markers (%lu bound nodes + %d unbound hash cells)",
  //          markers.markers.size(), places_cache_.size(), unbound_marker_id - 50000);

  return markers;
}

visualization_msgs::MarkerArray TemporalDynamicsNode::createDebugBindingMarkers() {
  visualization_msgs::MarkerArray markers;
  // Note: temporal_data_mutex_ is already held by caller (fremenUpdateTimerCallback)
  // GLOBAL MUTEX ORDER: places_cache_mutex_ → binding_mutex_ → temporal_data_mutex_
  std::shared_lock<std::shared_mutex> places_lock(places_cache_mutex_);
  std::lock_guard<std::mutex> binding_lock(binding_mutex_);

  size_t marker_id = 0;

  // Clear old markers
  visualization_msgs::Marker clear_marker;
  clear_marker.header.frame_id = config_.frame_id;
  clear_marker.header.stamp = ros::Time::now();
  clear_marker.ns = "debug_bindings";
  clear_marker.action = visualization_msgs::Marker::DELETEALL;
  markers.markers.push_back(clear_marker);

  for (const auto& [node_id, binding] : hash_bindings_) {
    if (!binding.is_active) continue;

    // Find the node position and node-based data
    auto node_pos_it = places_cache_.find(binding.bound_node_id);
    if (node_pos_it == places_cache_.end()) continue; // Node not found

    // Find the node-based temporal data to get the original hash position
    size_t node_based_key = static_cast<size_t>(binding.bound_node_id) | 0x8000000000000000ULL;
    auto node_data_it = temporal_place_data_.find(node_based_key);
    if (node_data_it == temporal_place_data_.end()) continue; // No temporal data found

    // Create line connecting original hash position to current node position
    visualization_msgs::Marker line_marker;
    line_marker.header.frame_id = config_.frame_id;
    line_marker.header.stamp = ros::Time::now();
    line_marker.ns = "debug_bindings";
    line_marker.id = marker_id++;
    line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::Marker::ADD;

    // Start point: original hash cell location (stored in binding)
    geometry_msgs::Point start_point;
    // Calculate position from original hash cell using getPositionFromHashCell
    Eigen::Vector3d hash_pos = getPositionFromHashCell(binding.original_hash_cell);
    start_point.x = hash_pos.x();
    start_point.y = hash_pos.y();
    start_point.z = temporal_marker_z_level_ - 0.5; // Hash level

    // End point: bound node location
    geometry_msgs::Point end_point;
    end_point.x = node_pos_it->second.x();
    end_point.y = node_pos_it->second.y();
    end_point.z = temporal_marker_z_level_; // Node level

    line_marker.points.push_back(start_point);
    line_marker.points.push_back(end_point);

    // Line styling
    line_marker.scale.x = 0.05; // Line width
    line_marker.color.r = 0.0;
    line_marker.color.g = 0.0;
    line_marker.color.b = 1.0; // Blue connection
    line_marker.color.a = 0.8;

    markers.markers.push_back(line_marker);

    // Add text label at node location showing binding info
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = config_.frame_id;
    text_marker.header.stamp = ros::Time::now();
    text_marker.ns = "debug_bindings";
    text_marker.id = marker_id++;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;

    text_marker.pose.position = end_point;
    text_marker.pose.position.z += 0.3; // Above node
    text_marker.pose.orientation.w = 1.0;

    text_marker.scale.z = 0.2; // Text size
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;

    // Calculate distance for display
    double binding_distance = std::sqrt(
      (end_point.x - start_point.x) * (end_point.x - start_point.x) +
      (end_point.y - start_point.y) * (end_point.y - start_point.y) +
      (end_point.z - start_point.z) * (end_point.z - start_point.z)
    );

    text_marker.text = "BOUND:" + std::to_string(binding.bound_node_id) +
                      " d=" + std::to_string(binding_distance).substr(0, 4) + "m";

    markers.markers.push_back(text_marker);
  }

  return markers;
}

}  // namespace aion
