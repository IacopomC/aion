#include "aion/temporal_dynamics_module.h"
#include <spark_dsg/serialization/graph_binary_serialization.h>
#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Point.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

namespace aion {

TemporalDynamicsNode::TemporalDynamicsNode() : nh_(), pnh_("~") {

  // Load all parameters from unified config structure - no hardcoded values
  pnh_.param("num_orientation_bins", config_.num_orientation_bins, 8);
  pnh_.param("update_interval_seconds", config_.update_interval_seconds, 5.0); // Reduced from 10.0 to 5.0 for faster updates
  pnh_.param("min_observations", config_.min_observations, 1);
  pnh_.param("detection_association_distance", config_.detection_association_distance, 2.0); // Conservative distance for detection-to-place association
  pnh_.param("fremen_model_order", config_.fremen_model_order, 0);
  pnh_.param("use_navigation_layer", config_.use_navigation_layer, true);
  pnh_.param("target_layer_id", config_.target_layer_id, -1);

  // ROS Topics
  pnh_.param("hydra_dsg_topic", config_.hydra_dsg_topic, std::string("hydra_ros_node/backend/dsg"));
  pnh_.param("people_detection_topic", config_.people_detection_topic, std::string("/people_detections"));
  pnh_.param("temporal_places_topic", config_.temporal_places_topic, std::string("/aion/temporal_dynamics"));
  pnh_.param("hash_cells_topic", config_.hash_cells_topic, std::string("/aion/hash_cells"));
  pnh_.param("debug_bindings_topic", config_.debug_bindings_topic, std::string("/aion/debug/bindings"));
  pnh_.param("temporal_nodes_topic", config_.temporal_nodes_topic, std::string("/aion/temporal_nodes"));
  pnh_.param("historical_nodes_topic", config_.historical_nodes_topic, std::string("/aion/historical_nodes"));
  pnh_.param("temporal_map_topic", config_.temporal_map_topic, std::string("/aion/temporal_map"));

  // ROS Services
  pnh_.param("prediction_service_name", config_.prediction_service_name, std::string("/aion/get_prediction"));
  pnh_.param("all_predictions_service_name", config_.all_predictions_service_name, std::string("/aion/get_all_predictions"));
  pnh_.param("reset_service_name", config_.reset_service_name, std::string("/aion/reset"));
  pnh_.param("manual_fremen_service_name", config_.manual_fremen_service_name, std::string("/aion/update_fremen"));
  pnh_.param("export_navigation_service_name", config_.export_navigation_service_name,
             std::string("/aion/export_navigation_data"));
  pnh_.param("export_default_output_directory", config_.export_default_output_directory,
             std::string("/tmp/aion_navigation_export"));
  pnh_.param("export_default_filename_prefix", config_.export_default_filename_prefix,
             std::string("aion_export"));

  // Fremen Configuration
  pnh_.param("fremen_action_server_name", config_.fremen_action_server_name, std::string("/fremenarray_aion"));
  pnh_.param("fremen_operation_timeout", config_.fremen_operation_timeout, 5.0);

  // Timing Parameters
  pnh_.param("place_sync_interval_seconds", config_.place_sync_interval_seconds, 30.0);
  pnh_.param("min_velocity_threshold", config_.min_velocity_threshold, 0.1);
  pnh_.param("processing_timer_rate", processing_timer_rate_, 2.0);

  // Temporal visualization scaling parameters
  pnh_.param("activity_scaling_factor", config_.activity_scaling_factor, 50.0);
  pnh_.param("confidence_scaling_factor", config_.confidence_scaling_factor, 30.0);

  // Stable ordering parameters (prevents Fremen confusion when new places appear)
  pnh_.param("spatial_hash_grid_size", config_.spatial_hash_grid_size, 1.0);

  if (config_.spatial_hash_grid_size <= 0) {
    config_.spatial_hash_grid_size = std::max(config_.detection_association_distance / 3.0, 0.5);
  }

  // Stability window and delayed binding parameters
  pnh_.param("enable_delayed_binding", config_.enable_delayed_binding, true);
  pnh_.param("stability_window_seconds", config_.stability_window_seconds, 30.0);
  pnh_.param("enable_debug_visualization", config_.enable_debug_visualization, false);

  // Computation control parameters
  pnh_.param("enable_historical_computation", config_.enable_historical_computation, true);
  pnh_.param("enable_prediction_computation", config_.enable_prediction_computation, true);

  // Visualization data source control
  // Load visualization configuration - determines whether to show historical or prediction data
  // To show predictions instead of historical data, set: visualization_data_source: "predictions"
  // in your launch file or config. The same visualization functions handle both data types.
  pnh_.param("visualization_data_source", config_.visualization_data_source, std::string("historical"));

  // Visualization mode: "direction" (dominant arrow), "entropy" (entropy sphere), "bins" (all bin arrows)
  pnh_.param("visualization_mode", config_.visualization_mode, std::string("direction"));

  // Spatial search optimization parameters
  pnh_.param("use_efficient_spatial_search", config_.use_efficient_spatial_search, true);

  // Z-height filtering parameters
  pnh_.param("filter_by_z_height", config_.filter_by_z_height, false);
  pnh_.param("min_z_height", config_.min_z_height, -1.0);
  pnh_.param("max_z_height", config_.max_z_height, 2.0);

  // Performance parameters
  pnh_.param("detection_buffer_size", detection_buffer_size_, 10);

  // Visualization Parameters
  pnh_.param("marker_scale", config_.marker_scale, 0.5);
  pnh_.param("arrow_scale", config_.arrow_scale, 1.0);
  pnh_.param("use_sphere_markers", config_.use_sphere_markers, true);
  pnh_.param("frame_id", config_.frame_id, std::string("map"));

  // Marker visualization ranges
  pnh_.param("marker_size_range/min_multiplier", marker_size_min_multiplier_, 0.7);
  pnh_.param("marker_size_range/max_multiplier", marker_size_max_multiplier_, 1.3);
  pnh_.param("marker_transparency_range/min_alpha", marker_transparency_min_alpha_, 0.8);
  pnh_.param("marker_transparency_range/max_alpha", marker_transparency_max_alpha_, 1.0);
  pnh_.param("arrow_length_range/min_multiplier", arrow_length_min_multiplier_, 0.8);
  pnh_.param("arrow_length_range/max_multiplier", arrow_length_max_multiplier_, 1.2);

  // Inactive place visualization
  pnh_.param("inactive_marker_scale_factor", inactive_marker_scale_factor_, 0.3);
  pnh_.param("inactive_marker_alpha", inactive_marker_alpha_, 0.4);
  pnh_.param("inactive_marker_color/r", inactive_marker_color_r_, 0.5);
  pnh_.param("inactive_marker_color/g", inactive_marker_color_g_, 0.5);
  pnh_.param("inactive_marker_color/b", inactive_marker_color_b_, 0.5);

  // Entropy color mapping
  pnh_.param("entropy_color_mapping/max_entropy_for_normalization", max_entropy_for_normalization_, 3.0);

  // Prediction visualization
  pnh_.param("prediction_markers/green_tint_increase", prediction_green_tint_increase_, 0.3);
  pnh_.param("prediction_markers/transparency_reduction", prediction_transparency_reduction_, 0.1);

  // Arrow dimensions
  pnh_.param("arrow_dimensions/shaft_diameter", arrow_shaft_diameter_, 0.1);
  pnh_.param("arrow_dimensions/head_diameter", arrow_head_diameter_, 0.2);
  pnh_.param("arrow_dimensions/head_length", arrow_head_length_, 0.1);

  // Standardized Z levels - raised to make temporal markers more visible
  pnh_.param("marker_z_levels/temporal_markers", temporal_marker_z_level_, 1.5);
  pnh_.param("marker_z_levels/inactive_markers", inactive_marker_z_level_, 1.0);

  // Debug throttle intervals
  pnh_.param("throttle_intervals/debug_general", throttle_debug_general_, 10.0);
  pnh_.param("throttle_intervals/debug_detailed", throttle_debug_detailed_, 30.0);
  pnh_.param("throttle_intervals/info_statistics", throttle_info_statistics_, 30.0);
  pnh_.param("throttle_intervals/warning_overflow", throttle_warning_overflow_, 5.0);

  // Timing & profiling parameters
  pnh_.param("enable_timing", config_.enable_timing, false);
  pnh_.param("timing_output_file", config_.timing_output_file, std::string(""));
  enable_timing_ = config_.enable_timing;
  if (enable_timing_) {
    timing_output_file_ = config_.timing_output_file.empty()
        ? "/tmp/aion_timing.csv" : config_.timing_output_file;
    timing_file_.open(timing_output_file_, std::ios::out | std::ios::trunc);
    if (timing_file_.is_open()) {
      timing_file_ << "timestamp_ns,callback,elapsed_us,num_places,total_observations,memory_bytes" << std::endl;
      ROS_INFO("Timing enabled — writing to %s", timing_output_file_.c_str());
    } else {
      ROS_WARN("Failed to open timing file %s — timing disabled", timing_output_file_.c_str());
      enable_timing_ = false;
    }
  }

  // Global model parameters
  pnh_.param("global_model/min_places_for_connection", min_places_for_connection_, 3);
  pnh_.param("global_model/prediction_fallback_intensity", prediction_fallback_intensity_, 0.7);
  pnh_.param("global_model/prediction_strength_weight", prediction_strength_weight_, 0.3);

  // Initialize Fremen interface with configurable parameters
  FremenInterface::Config fremen_config;
  fremen_config.model_order = config_.fremen_model_order;
  fremen_config.action_server_name = config_.fremen_action_server_name;
  fremen_config.operation_timeout = config_.fremen_operation_timeout;
  fremen_ = std::make_unique<FremenInterface>(fremen_config);

  // Initialize timing
  last_sync_time_ = std::chrono::steady_clock::now();

  // Initialize global stability tracking (no fields needed)
  {
    std::lock_guard<std::mutex> lock(global_stability_mutex_);
    // global_stability_ struct is empty - timer handles everything
  }

  ROS_INFO("Temporal Dynamics Node initialized with unified markers");
  ROS_INFO("Publishing unified temporal markers to: %s", config_.temporal_places_topic.c_str());

  if (config_.enable_delayed_binding) {
    ROS_INFO("Global delayed binding enabled - stability window: %.1fs",
             config_.stability_window_seconds);
  } else {
    ROS_INFO("Delayed binding disabled - using direct spatial hash binding");
  }
}

TemporalDynamicsNode::~TemporalDynamicsNode() = default;

void TemporalDynamicsNode::run() {
  // Initialize ROS interface
  initializeRosInterface();

  // Start ROS spinning
  ros::spin();
}

void TemporalDynamicsNode::initializeRosInterface() {
  // Subscribers
  hydra_sub_ = nh_.subscribe(config_.hydra_dsg_topic, 10,
    &TemporalDynamicsNode::hydraGraphCallback, this);

  people_sub_ = nh_.subscribe(config_.people_detection_topic, 10,
    &TemporalDynamicsNode::peopleDetectionCallback, this);

  // Publishers - unified temporal markers plus debug visualization
  temporal_places_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
    config_.temporal_places_topic, 10);

  // Publishers for temporal data messages
  temporal_nodes_pub_ = nh_.advertise<aion::AionTemporalNode>(
    config_.temporal_nodes_topic, 10);
  historical_nodes_pub_ = nh_.advertise<aion::AionHistoricalNode>(
    config_.historical_nodes_topic, 10);
  temporal_map_pub_ = nh_.advertise<aion::AionTemporalMap>(
    config_.temporal_map_topic, 10);

  // Debug publishers for understanding delayed binding behavior
  hash_cells_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
    config_.hash_cells_topic, 10);
  debug_bindings_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
    config_.debug_bindings_topic, 10);

  // Note: Using only unified temporal markers that encode all information:
  // - Color = entropy (predictability)
  // - Size = activity magnitude (observation count)
  // - Orientation = dominant flow direction
  // - Transparency = confidence (amount of data)

  // Services
  prediction_service_ = nh_.advertiseService(config_.prediction_service_name,
    &TemporalDynamicsNode::predictionServiceCallback, this);

  all_predictions_service_ = nh_.advertiseService(config_.all_predictions_service_name,
    &TemporalDynamicsNode::allPlacePredictionsServiceCallback, this);

  reset_service_ = nh_.advertiseService(config_.reset_service_name,
    &TemporalDynamicsNode::resetServiceCallback, this);

  // Manual Fremen update service
  manual_fremen_service_ = nh_.advertiseService(config_.manual_fremen_service_name,
    &TemporalDynamicsNode::manualFremenUpdateCallback, this);

  // Export navigation data service for A* analysis
  export_navigation_service_ = nh_.advertiseService(config_.export_navigation_service_name,
    &TemporalDynamicsNode::exportNavigationDataCallback, this);

  // Processing timer (runs main processing loop) - configurable frequency
  processing_timer_ = nh_.createTimer(
    ros::Duration(processing_timer_rate_),
    &TemporalDynamicsNode::processingTimerCallback, this);

  // Fremen update timer - less frequent
  fremen_update_timer_ = nh_.createTimer(
    ros::Duration(config_.update_interval_seconds),
    &TemporalDynamicsNode::fremenUpdateTimerCallback, this);

  // Global stability timer for delayed binding
  if (config_.enable_delayed_binding) {
    global_stability_timer_ = nh_.createTimer(
        ros::Duration(config_.stability_window_seconds),
        &TemporalDynamicsNode::globalStabilityTimerCallback, this,
        false);  // repeating timer - first execution assigns, subsequent check for removal
  }

  // Place synchronization timer - periodic background updates from Hydra DSG
  place_sync_timer_ = nh_.createTimer(
      ros::Duration(config_.place_sync_interval_seconds),
      &TemporalDynamicsNode::placeSyncTimerCallback, this);
}

}  // namespace aion
