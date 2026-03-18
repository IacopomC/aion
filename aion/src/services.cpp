#include "aion/temporal_dynamics_module.h"
#include <spark_dsg/serialization/graph_binary_serialization.h>
#include <nlohmann/json.hpp>
#include <fstream>

namespace aion {

bool TemporalDynamicsNode::predictionServiceCallback(
    aion::GetTemporalPrediction::Request& request,
    aion::GetTemporalPrediction::Response& response) {

  spark_dsg::NodeId query_place_id = request.place_id;

  // If place_id is 0, find nearest place to given position
  if (query_place_id == 0) {
    Eigen::Vector3d query_pos(request.x, request.y, request.z);
    {
      std::shared_lock<std::shared_mutex> lock(places_cache_mutex_);
      query_place_id = findNearestPlace(query_pos);
    }
  }

  if (query_place_id == 0) {
    response.success = false;
    response.message = "No place found for query";
    return true;
  }

  // Get prediction (this will acquire its own locks)
  auto predictions = predictFlow(query_place_id, request.prediction_time);

  // Fill response - acquire temporal data lock only once
  {
    std::shared_lock<std::shared_mutex> temporal_lock(temporal_data_mutex_);
    auto it = temporal_place_data_.find(query_place_id);

    response.success = true;
    response.message = "Prediction successful";
    response.place_id = query_place_id;
    response.flow_probabilities = predictions;

    if (it != temporal_place_data_.end()) {
      response.entropy = getEntropy(it->second);
      response.num_observations = it->second.total_observations;
      response.position_x = it->second.position.x();
      response.position_y = it->second.position.y();
      response.position_z = it->second.position.z();
    } else {
      // Default values if place not found in temporal data
      response.entropy = 0.0;
      response.num_observations = 0;
      response.position_x = 0.0;
      response.position_y = 0.0;
      response.position_z = 0.0;
    }
  }

  // Trigger actual Fremen prediction update (like periodic updates but for specific time)
  ROS_INFO("Prediction service triggered - updating temporal models with Fremen predictions for time %lu", request.prediction_time);

  // Use 0 time for current state, otherwise use specified prediction time
  uint64_t prediction_time_to_use = (request.prediction_time == 0) ? ros::Time::now().toNSec() : request.prediction_time;
  updateTemporalModelsWithFremenPrediction(prediction_time_to_use);

  return true;
}

bool TemporalDynamicsNode::allPlacePredictionsServiceCallback(
    aion::GetAllPlacePredictions::Request& request,
    aion::GetAllPlacePredictions::Response& response) {

  ROS_INFO("Received request for all place predictions for time %lu", request.prediction_time);

  // Get predictions for all places
  auto all_predictions = predictAllPlacesFlow(request.prediction_time);

  response.success = true;
  response.message = "All place predictions successful";
  response.prediction_time = request.prediction_time;

  // Fill response with all place predictions
  std::shared_lock<std::shared_mutex> lock(temporal_data_mutex_);
  for (const auto& [place_id, predictions] : all_predictions) {
    aion::PlacePrediction place_pred;
    place_pred.place_id = place_id;
    place_pred.flow_probabilities = predictions;

    // Add additional place information
    auto it = temporal_place_data_.find(place_id);
    if (it != temporal_place_data_.end()) {
      place_pred.entropy = getEntropy(it->second);
      place_pred.num_observations = it->second.total_observations;
      place_pred.position.x = it->second.position.x();
      place_pred.position.y = it->second.position.y();
      place_pred.position.z = it->second.position.z();
    }

    response.place_predictions.push_back(place_pred);
  }

  // Trigger actual Fremen prediction update (like periodic updates but for specific time)
  ROS_INFO("All predictions service triggered - updating temporal models with Fremen predictions for time %lu", request.prediction_time);

  // Use 0 time for current state, otherwise use specified prediction time
  uint64_t prediction_time_to_use = (request.prediction_time == 0) ? ros::Time::now().toNSec() : request.prediction_time;
  updateTemporalModelsWithFremenPrediction(prediction_time_to_use);

  ROS_INFO("Returning predictions for %zu places", response.place_predictions.size());
  return true;
}

bool TemporalDynamicsNode::resetServiceCallback(
    std_srvs::Trigger::Request& /*request*/,
    std_srvs::Trigger::Response& response) {

  std::lock_guard<std::shared_mutex> lock(temporal_data_mutex_);
  temporal_place_data_.clear();
  total_detections_processed_ = 0;
  total_associations_made_ = 0;
  total_bindings_created_ = 0;

  // Reset global stability and binding tracking
  if (config_.enable_delayed_binding) {
    std::lock_guard<std::mutex> stability_lock(global_stability_mutex_);
    std::lock_guard<std::mutex> binding_lock(binding_mutex_);


    hash_bindings_.clear();

    // Restart the global stability timer
    global_stability_timer_.stop();
    global_stability_timer_ = nh_.createTimer(
        ros::Duration(config_.stability_window_seconds),
        &TemporalDynamicsNode::globalStabilityTimerCallback, this,
        false);  // repeating timer
  }

  response.success = true;
  response.message = "Temporal dynamics data reset successfully";

  ROS_INFO("Temporal dynamics data reset - global stability window restarted");

  return true;
}

bool TemporalDynamicsNode::manualFremenUpdateCallback(
    std_srvs::Trigger::Request& /*request*/,
    std_srvs::Trigger::Response& response) {

  ROS_INFO("Manual Fremen update triggered via service call");

  // Call the same function as the timer
  ros::TimerEvent dummy_event;
  fremenUpdateTimerCallback(dummy_event);

  response.success = true;
  response.message = "Manual Fremen update completed successfully";

  ROS_INFO("Manual Fremen update finished");

  return true;
}

bool TemporalDynamicsNode::exportNavigationDataCallback(
    aion::ExportNavigationData::Request& request,
    aion::ExportNavigationData::Response& response) {

  ROS_INFO("Export navigation data service called");

  try {
    // Create output directory if it doesn't exist
    std::string output_dir = request.output_directory;
    if (output_dir.empty()) {
      output_dir = config_.export_default_output_directory;
    }

    // Create directory (using system call for simplicity)
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
      response.success = false;
      response.message = "Failed to create output directory: " + output_dir;
      return true;
    }

    std::string prefix = request.filename_prefix.empty()
      ? config_.export_default_filename_prefix
      : request.filename_prefix;

    // Export Hydra navigation layer (Layer 20)
    std::string nav_file = output_dir + "/" + prefix + "_navigation.json";
    int num_nodes = 0, num_edges = 0;

    if (!exportHydraNavigationLayer(nav_file, num_nodes, num_edges, request.include_connectivity)) {
      response.success = false;
      response.message = "Failed to export Hydra navigation layer";
      return true;
    }

    response.navigation_file = nav_file;
    response.num_nodes = num_nodes;
    response.num_edges = num_edges;

    // Export Aion temporal dynamics if requested
    std::string temporal_file = output_dir + "/" + prefix + "_temporal.json";
    int num_temporal_places = 0;

    if (request.include_temporal_data) {
      uint64_t prediction_time = request.prediction_time > 0 ?
          static_cast<uint64_t>(request.prediction_time * 1e9) : // Convert seconds to nanoseconds
          static_cast<uint64_t>(ros::Time::now().toNSec());

      if (!exportTemporalDynamics(temporal_file, prediction_time, num_temporal_places)) {
        response.success = false;
        response.message = "Failed to export temporal dynamics";
        return true;
      }

      response.temporal_file = temporal_file;
    }

    response.num_temporal_places = num_temporal_places;

    // Export connectivity if requested
    if (request.include_connectivity && num_edges > 0) {
      response.connectivity_file = output_dir + "/" + prefix + "_connectivity.json";
    }

    response.success = true;
    response.message = "Navigation data exported successfully";

    ROS_INFO("Navigation data exported: %d nodes, %d edges, %d temporal places",
             num_nodes, num_edges, num_temporal_places);

    return true;

  } catch (const std::exception& e) {
    response.success = false;
    response.message = std::string("Exception during export: ") + e.what();
    ROS_ERROR("Export failed: %s", e.what());
    return true;
  }
}

bool TemporalDynamicsNode::exportHydraNavigationLayer(const std::string& filename, int& num_nodes, int& num_edges, bool include_connectivity) {
  // Use latest DSG message to recreate the scene graph for export
  hydra_msgs::DsgUpdate::ConstPtr dsg_msg;
  {
    std::lock_guard<std::mutex> lock(latest_dsg_mutex_);
    dsg_msg = latest_dsg_msg_;
  }

  if (!dsg_msg) {
    ROS_ERROR("No DSG message available for export. Is Hydra publishing DSG data?");
    return false;
  }

  // Recreate scene graph from the latest DSG message
  std::shared_ptr<spark_dsg::DynamicSceneGraph> temp_graph;
  try {
    temp_graph = spark_dsg::io::binary::readGraph(dsg_msg->layer_contents);
    if (!temp_graph) {
      ROS_ERROR("Failed to recreate scene graph from DSG message");
      return false;
    }
  } catch (const std::exception& e) {
    ROS_ERROR("Failed to deserialize scene graph: %s", e.what());
    return false;
  }

  try {
    nlohmann::json export_data;
    export_data["type"] = "hydra_navigation_layer";
    export_data["layer_id"] = 20;
    export_data["frame_id"] = config_.frame_id;
    export_data["timestamp"] = ros::Time::now().toSec();

    nlohmann::json nodes = nlohmann::json::array();
    nlohmann::json edges = nlohmann::json::array();

    num_nodes = 0;
    num_edges = 0;

    // Export Layer 20 (MESH_PLACES) nodes
    if (temp_graph->hasLayer(20)) {
      const auto& layer = temp_graph->getLayer(20);

      for (const auto& node_pair : layer.nodes()) {
        const auto node_id = node_pair.first;
        const auto& attrs = layer.getNode(node_id).attributes();

        nlohmann::json node_data;
        node_data["id"] = static_cast<uint64_t>(node_id);
        node_data["position"] = {
          {"x", attrs.position.x()},
          {"y", attrs.position.y()},
          {"z", attrs.position.z()}
        };

        // Add semantic information if available
        if (auto semantic_attrs = dynamic_cast<const spark_dsg::SemanticNodeAttributes*>(&attrs)) {
          node_data["semantic_label"] = semantic_attrs->semantic_label;
          node_data["color"] = {
            {"r", semantic_attrs->color.r},
            {"g", semantic_attrs->color.g},
            {"b", semantic_attrs->color.b}
          };
        }

        nodes.push_back(node_data);
        num_nodes++;
      }

      // Export edges if connectivity requested
      if (include_connectivity) {
        for (const auto& edge_pair : layer.edges()) {
          const auto& edge_key = edge_pair.first;

          nlohmann::json edge_data;
          edge_data["source"] = static_cast<uint64_t>(edge_key.k1);
          edge_data["target"] = static_cast<uint64_t>(edge_key.k2);

          // Calculate edge weight (Euclidean distance)
          if (layer.hasNode(edge_key.k1) && layer.hasNode(edge_key.k2)) {
            const auto& source_pos = layer.getNode(edge_key.k1).attributes().position;
            const auto& target_pos = layer.getNode(edge_key.k2).attributes().position;
            double distance = (source_pos - target_pos).norm();
            edge_data["weight"] = distance;
          } else {
            edge_data["weight"] = 1.0; // Default weight
          }

          edges.push_back(edge_data);
          num_edges++;
        }
      }
    } else {
      ROS_WARN("Layer 20 (MESH_PLACES) not found in scene graph. Available layers:");
      for (const auto& [layer_id, layer] : temp_graph->layers()) {
        ROS_WARN("  Layer %zu: %zu nodes", layer_id, layer->numNodes());
      }

      // If Layer 20 is not available, try Layer 3 as fallback
      if (temp_graph->hasLayer(3)) {
        ROS_INFO("Using Layer 3 (PLACES) as fallback for navigation export");
        const auto& layer = temp_graph->getLayer(3);

        for (const auto& node_pair : layer.nodes()) {
          const auto node_id = node_pair.first;
          const auto& attrs = layer.getNode(node_id).attributes();

          nlohmann::json node_data;
          node_data["id"] = static_cast<uint64_t>(node_id);
          node_data["position"] = {
            {"x", attrs.position.x()},
            {"y", attrs.position.y()},
            {"z", attrs.position.z()}
          };

          // Add semantic information if available
          if (auto semantic_attrs = dynamic_cast<const spark_dsg::SemanticNodeAttributes*>(&attrs)) {
            node_data["semantic_label"] = semantic_attrs->semantic_label;
            node_data["color"] = {
              {"r", semantic_attrs->color.r},
              {"g", semantic_attrs->color.g},
              {"b", semantic_attrs->color.b}
            };
          }

          nodes.push_back(node_data);
          num_nodes++;
        }

        // Export edges for Layer 3 if requested
        if (include_connectivity) {
          for (const auto& edge_pair : layer.edges()) {
            const auto& edge_key = edge_pair.first;

            nlohmann::json edge_data;
            edge_data["source"] = static_cast<uint64_t>(edge_key.k1);
            edge_data["target"] = static_cast<uint64_t>(edge_key.k2);

            // Calculate edge weight (Euclidean distance)
            if (layer.hasNode(edge_key.k1) && layer.hasNode(edge_key.k2)) {
              const auto& source_pos = layer.getNode(edge_key.k1).attributes().position;
              const auto& target_pos = layer.getNode(edge_key.k2).attributes().position;
              double distance = (source_pos - target_pos).norm();
              edge_data["weight"] = distance;
            } else {
              edge_data["weight"] = 1.0; // Default weight
            }

            edges.push_back(edge_data);
            num_edges++;
          }
        }

        export_data["layer_id"] = 3;  // Update layer ID in metadata
      }
    }

    export_data["nodes"] = nodes;
    export_data["edges"] = edges;
    export_data["num_nodes"] = num_nodes;
    export_data["num_edges"] = num_edges;

    // Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
      ROS_ERROR("Failed to open file for writing: %s", filename.c_str());
      return false;
    }
    file << export_data.dump(2);
    file.close();

    ROS_INFO("Exported %d navigation nodes and %d edges to %s", num_nodes, num_edges, filename.c_str());
    return true;

  } catch (const std::exception& e) {
    ROS_ERROR("Failed to export navigation layer: %s", e.what());
    return false;
  }
}

bool TemporalDynamicsNode::exportTemporalDynamics(const std::string& filename, uint64_t prediction_time, int& num_temporal_places) {
  std::shared_lock<std::shared_mutex> lock(temporal_data_mutex_);

  try {
    nlohmann::json export_data;
    export_data["type"] = "aion_temporal_dynamics";
    export_data["prediction_time"] = prediction_time;
    export_data["timestamp"] = ros::Time::now().toSec();
    export_data["frame_id"] = config_.frame_id;
    export_data["num_orientation_bins"] = config_.num_orientation_bins;

    nlohmann::json temporal_places = nlohmann::json::array();
    num_temporal_places = 0;

    for (const auto& [place_id, place_data] : temporal_place_data_) {
      // Only export places with sufficient data
      if (place_data.total_observations < config_.min_observations) {
        continue;
      }

      nlohmann::json place_json;
      place_json["place_id"] = static_cast<uint64_t>(place_id);
      place_json["position"] = {
        {"x", place_data.position.x()},
        {"y", place_data.position.y()},
        {"z", place_data.position.z()}
      };

      place_json["total_observations"] = place_data.total_observations;
      place_json["is_bound_to_node"] = place_data.is_bound_to_node;

      // Export both historical and prediction metrics
      place_json["historical"] = {
        {"entropy", place_data.historical_metrics.entropy},
        {"flow_magnitude", place_data.historical_metrics.flow_magnitude},
        {"best_angle", place_data.historical_metrics.best_angle},
        {"probabilities", place_data.historical_metrics.probabilities},
        {"raw_counts", place_data.historical_counts}
      };

      // Get Fremen predictions if available
      auto predictions = predictFlow(place_id, prediction_time);
      if (!predictions.empty()) {
        double pred_entropy = computeEntropyFromProbabilities(predictions);
        double pred_flow_mag = computeFlowMagnitude(predictions);
        double pred_best_angle = computeBestAngle(predictions);

        place_json["predictions"] = {
          {"entropy", pred_entropy},
          {"flow_magnitude", pred_flow_mag},
          {"best_angle", pred_best_angle},
          {"probabilities", predictions}
        };
      }

      temporal_places.push_back(place_json);
      num_temporal_places++;
    }

    export_data["temporal_places"] = temporal_places;
    export_data["num_temporal_places"] = num_temporal_places;

    // Write to file
    std::ofstream file(filename);
    file << export_data.dump(2);
    file.close();

    ROS_INFO("Exported %d temporal places to %s", num_temporal_places, filename.c_str());
    return true;

  } catch (const std::exception& e) {
    ROS_ERROR("Failed to export temporal dynamics: %s", e.what());
    return false;
  }
}

aion::AionTemporalMap TemporalDynamicsNode::getCurrentTemporalMap() const {
  // GLOBAL MUTEX ORDER: places_cache_mutex_ → binding_mutex_ → temporal_data_mutex_
  std::shared_lock<std::shared_mutex> places_lock(places_cache_mutex_);
  std::shared_lock<std::shared_mutex> temporal_lock(temporal_data_mutex_);

  aion::AionTemporalMap map_msg;
  map_msg.header.stamp = ros::Time::now();
  map_msg.header.frame_id = config_.frame_id;
  map_msg.update_interval_seconds = config_.update_interval_seconds;
  map_msg.num_orientation_bins = config_.num_orientation_bins;

  uint32_t bound_count = 0;
  uint32_t hash_count = 0;

  // Process all temporal data entries efficiently
  for (const auto& [place_id, place_data] : temporal_place_data_) {
    if (place_data.total_observations < config_.min_observations) {
      continue; // Skip places with insufficient data
    }

    // Determine position for this entry
    Eigen::Vector3d position;
    if (place_data.is_bound_to_node && place_data.bound_node_id != 0) {
      // This is bound to a DSG node - use current DSG position
      auto dsg_pos_it = places_cache_.find(place_data.bound_node_id);
      if (dsg_pos_it != places_cache_.end()) {
        position = dsg_pos_it->second;
      } else {
        position = place_data.position; // Fallback to stored position
      }
      bound_count++;
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

  return map_msg;
}

std::vector<aion::AionHistoricalNode> TemporalDynamicsNode::getHistoricalNodesInRegion(
    const Eigen::Vector3d& center, double radius) const {
  // GLOBAL MUTEX ORDER: places_cache_mutex_ → binding_mutex_ → temporal_data_mutex_
  std::shared_lock<std::shared_mutex> places_lock(places_cache_mutex_);
  std::shared_lock<std::shared_mutex> temporal_lock(temporal_data_mutex_);

  std::vector<aion::AionHistoricalNode> nodes;

  if (!config_.enable_historical_computation) {
    return nodes; // Historical computation disabled
  }

  for (const auto& [place_id, place_data] : temporal_place_data_) {
    if (place_data.total_observations < config_.min_observations) {
      continue; // Skip places with insufficient data
    }

    // Determine position for this entry
    Eigen::Vector3d position;
    if (place_data.is_bound_to_node && place_data.bound_node_id != 0) {
      auto dsg_pos_it = places_cache_.find(place_data.bound_node_id);
      if (dsg_pos_it != places_cache_.end()) {
        position = dsg_pos_it->second;
      } else {
        position = place_data.position;
      }
    } else {
      position = place_data.position;
    }

    // Check if within radius
    if ((position - center).norm() <= radius) {
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

      nodes.push_back(hist_node);
    }
  }

  return nodes;
}

std::optional<aion::AionHistoricalNode> TemporalDynamicsNode::getHistoricalNodeById(spark_dsg::NodeId node_id) const {
  // GLOBAL MUTEX ORDER: places_cache_mutex_ → binding_mutex_ → temporal_data_mutex_
  std::shared_lock<std::shared_mutex> places_lock(places_cache_mutex_);
  std::shared_lock<std::shared_mutex> temporal_lock(temporal_data_mutex_);

  if (!config_.enable_historical_computation) {
    return std::nullopt; // Historical computation disabled
  }

  // Look for bound entry first
  size_t node_based_key = static_cast<size_t>(node_id) | 0x8000000000000000ULL;
  auto bound_it = temporal_place_data_.find(node_based_key);
  if (bound_it != temporal_place_data_.end()) {
    const auto& place_data = bound_it->second;
    if (place_data.total_observations >= config_.min_observations) {
      aion::AionHistoricalNode hist_node;
      hist_node.place_id = node_id;

      // Get current DSG position
      auto dsg_pos_it = places_cache_.find(node_id);
      if (dsg_pos_it != places_cache_.end()) {
        hist_node.position.x = dsg_pos_it->second.x();
        hist_node.position.y = dsg_pos_it->second.y();
        hist_node.position.z = dsg_pos_it->second.z();
      } else {
        hist_node.position.x = place_data.position.x();
        hist_node.position.y = place_data.position.y();
        hist_node.position.z = place_data.position.z();
      }

      hist_node.entropy = place_data.historical_metrics.entropy;
      hist_node.flow_magnitude = place_data.historical_metrics.flow_magnitude;
      hist_node.best_angle = place_data.historical_metrics.best_angle;
      hist_node.probabilities = place_data.historical_metrics.probabilities;
      hist_node.normalized_counts = place_data.historical_metrics.normalized_counts;
      hist_node.total_observations = place_data.total_observations;
      hist_node.is_bound_to_node = true;

      return hist_node;
    }
  }

  return std::nullopt;
}

}  // namespace aion
