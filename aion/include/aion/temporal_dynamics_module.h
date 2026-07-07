#pragma once

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <optional>
#include <visualization_msgs/MarkerArray.h>
#include <hydra_msgs/DsgUpdate.h>
#include <std_srvs/Trigger.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <limits>
#include <memory>
#include <map>
#include <chrono>
#include <fstream>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <queue>

#include "aion/fremen_interface.h"
#include "aion/GetTemporalPrediction.h"
#include "aion/GetAllPlacePredictions.h"
#include "aion/ExportNavigationData.h"
#include "aion/AionTemporalNode.h"
#include "aion/AionHistoricalNode.h"
#include "aion/AionTemporalMap.h"

// Include for efficient spatial indexing
#include "hydra/utils/nearest_neighbor_utilities.h"

namespace aion {

/**
 * @brief Standalone ROS Node for Temporal Dynamics in Hydra Scene Graphs
 * 
 * This node subscribes to Hydra's published scene graph and adds temporal dynamics
 * modeling to place nodes:
 * 
 * Architecture:
 * 1. Subscribe to Hydra's scene graph (hydra_ros_node/backend/dsg)
 * 2. Extract places layer nodes from DsgUpdate messages  
 * 3. Subscribe to people detections and associate with place nodes
 * 4. Build temporal models using Fremen framework at node level
 * 5. Provide service interface for temporal predictions
 * 6. Publish visualization markers for flow and entropy
 * 
 */
class TemporalDynamicsNode {
 public:
  struct Config {
    // Number of orientation bins for flow direction
    int num_orientation_bins = 8;
    
    // Update interval for temporal models (seconds).
    double update_interval_seconds = 10.0;

    // Minimum observations needed for temporal modeling
    int min_observations = 10;

    // Maximum distance to associate detections with temporal places.
    double detection_association_distance = 0.7;

    // Fremen model order (0=basic, higher=more complex).
    int fremen_model_order = 1;
    
    // ROS topic names
    std::string hydra_dsg_topic = "hydra_ros_node/backend/dsg";
    std::string people_detection_topic = "/people_detections";
    std::string temporal_places_topic = "/aion/temporal_dynamics";
    std::string hash_cells_topic = "/aion/hash_cells";
    std::string debug_bindings_topic = "/aion/debug/bindings";
    std::string temporal_nodes_topic = "/aion/temporal_nodes";
    std::string historical_nodes_topic = "/aion/historical_nodes";
    std::string temporal_map_topic = "/aion/temporal_map";
    
    // Use navigation layer (layer 20 mesh places) instead of 3D places (layer 3)
    bool use_navigation_layer = true;
    
    // Override layer selection (-1 = use default based on use_navigation_layer)
    int target_layer_id = -1;
    
    // Service names
    std::string prediction_service_name = "/aion/get_prediction";
    std::string all_predictions_service_name = "/aion/get_all_predictions";
    std::string reset_service_name = "/aion/reset";
    std::string manual_fremen_service_name = "/aion/update_fremen";
    std::string export_navigation_service_name = "/aion/export_navigation_data";

    // Export defaults
    std::string export_default_output_directory = "/tmp/aion_navigation_export";
    std::string export_default_filename_prefix = "aion_export";

    // Fremen interface
    std::string fremen_action_server_name = "/fremenarray_aion";
    double fremen_operation_timeout = 5.0;
    
    // Copy places from main layer interval (seconds)
    double place_sync_interval_seconds = 60.0;
    
    // Minimum velocity to consider as movement (m/s)
    double min_velocity_threshold = 0.1;
    
    // Visualization parameters
    double marker_scale = 0.5;
    double arrow_scale = 1.0;
    bool use_sphere_markers = true; // True=spheres, False=arrows
    std::string frame_id = "map";
    
    // Z-height filtering (to show only one level if environment has multiple floors)
    bool filter_by_z_height = false;
    double min_z_height = -1.0;
    double max_z_height = 2.0;
    
    // Semantic filtering (to exclude ceiling nodes that shouldn't be in navigation layer)
    bool filter_ceiling_nodes = true;
    
    // Configurable sphere scaling factors for temporal dynamics visualization
    double activity_scaling_factor = 50.0;    // Higher = slower sphere growth with activity
    double confidence_scaling_factor = 30.0;  // Higher = slower transparency change with confidence
    
    // Stable ordering for global state vector (prevents Fremen confusion when new places appear)
    double spatial_hash_grid_size = 0.4;  // spatial hash cell size (m)
    // Drop detections farther than this (m) from the robot pose (looked up via
    // TF: frame_id -> robot_frame). Limits the local support window.
    // Infinity = disabled. Matches the offline AionConfig max_detection_range_m.
    double max_detection_range_m = std::numeric_limits<double>::infinity();
    std::string robot_frame = "base_link";  // robot body frame for the range gate
    
    // Stability window for delayed binding (loop closure robustness)
    double stability_window_seconds = 30.0;  // Time to wait before binding dynamics to nodes
    bool enable_delayed_binding = true;    // Enable/disable the delayed binding feature
    bool enable_debug_visualization = false;  // Enable/disable debug markers for hash cells and bindings
    
    // Computation control flags
    bool enable_historical_computation = true;   // Compute metrics from historical data
    bool enable_prediction_computation = true;   // Compute metrics from Fremen predictions
    
    // Visualization data source control
    std::string visualization_data_source = "historical";  // "historical", "predictions", or "legacy"

    // Visualization mode: controls what is shown for active places
    // "direction" = one arrow per place pointing in the dominant flow direction (default)
    // "entropy"   = one sphere per place, colored by entropy, sized by activity
    // "bins"      = all num_orientation_bins arrows per place, each scaled by its probability
    std::string visualization_mode = "direction";
    
    // Spatial search optimization
    bool use_efficient_spatial_search = true;  // Use KD-tree for O(log n) nearest neighbor search vs O(n) linear
    
    // Timing & profiling
    bool enable_timing = false;               // Enable per-iteration timing logging to file
    std::string timing_output_file = "";       // Path for timing CSV (empty = /tmp/aion_timing.csv)
  };

  explicit TemporalDynamicsNode();
  
  virtual ~TemporalDynamicsNode();

  // Initialize and run the node
  void run();

  // Get prediction for a specific place at given time (service interface)
  std::vector<double> predictFlow(spark_dsg::NodeId place_id, 
                                 uint64_t prediction_time_ns) const;

  // Get predictions for all places at given time
  std::map<spark_dsg::NodeId, std::vector<double>> predictAllPlacesFlow(
      uint64_t prediction_time_ns) const;
  
  // Get complete current temporal map (replaces old marker-based visualization)
  aion::AionTemporalMap getCurrentTemporalMap() const;
  
  // Get historical nodes within a spatial region (efficient spatial queries)
  std::vector<aion::AionHistoricalNode> getHistoricalNodesInRegion(
      const Eigen::Vector3d& center, double radius) const;
  
  // Get historical node by DSG node ID (O(log n) lookup)
  std::optional<aion::AionHistoricalNode> getHistoricalNodeById(spark_dsg::NodeId node_id) const;

  // Get entropy map for all temporal places
  std::map<spark_dsg::NodeId, double> getEntropyMap() const;

  // Global stability tracking for delayed binding
  struct GlobalStabilityInfo {
    // No fields needed - timer handles everything
  };

  // Temporal binding information
  struct TemporalBinding {
    spark_dsg::NodeId bound_node_id = 0;  // 0 = unbound (dynamics in hash cell)
    uint64_t bind_time = 0;
    size_t original_hash_cell = 0;
    bool is_active = false;
  };

  // Movement data accumulation per place node or hash cell
  struct TemporalPlaceData {
    // Current interval counts
    std::vector<int> current_interval_counts;  // [orientation_bin] - current accumulation
    
    // Historical data for entropy calculation
    std::vector<int> historical_counts;  // [orientation_bin] - total accumulated
    
    std::vector<uint64_t> observation_times;
    size_t total_observations = 0;
    Eigen::Vector3d position;
    uint64_t last_fremen_update = 0;
    double cached_entropy = 0.0;  // Keep for backward compatibility
    
    // Binding state
    bool is_bound_to_node = false;
    spark_dsg::NodeId bound_node_id = 0;
    size_t original_hash_cell = 0;  // Spatial hash key before binding (for stable FreMEn ordering)
    
    // Separated computed metrics
    struct HistoricalMetrics {
      double entropy = 0.0;
      double flow_magnitude = 0.0;
      double best_angle = 0.0;
      std::vector<double> probabilities;      // Normalized probabilities (sum=1.0)
      std::vector<int> normalized_counts;     // Fremen-style normalized (max=100)
      
      void initialize(int num_bins) {
        probabilities.assign(num_bins, 0.0);
        normalized_counts.assign(num_bins, 0);
      }
    } historical_metrics;
    
    struct PredictionMetrics {
      double entropy = 0.0;
      double flow_magnitude = 0.0; 
      double best_angle = 0.0;
      std::vector<double> probabilities;      // Fremen prediction probabilities
      uint64_t prediction_time = 0;
      
      void initialize(int num_bins) {
        probabilities.assign(num_bins, 0.0);
      }
    } prediction_metrics;
    
    // Initialize with given number of orientation bins
    void initialize(int num_bins) {
      current_interval_counts.assign(num_bins, 0);
      historical_counts.assign(num_bins, 0);
      is_bound_to_node = false;
      bound_node_id = 0;
      original_hash_cell = 0;
      historical_metrics.initialize(num_bins);
      prediction_metrics.initialize(num_bins);
    }
    
    // Reset current interval
    void resetCurrentInterval() {
      std::fill(current_interval_counts.begin(), current_interval_counts.end(), 0);
    }
    
    // Add current interval to historical data
    void addToHistorical() {
      for (size_t i = 0; i < current_interval_counts.size(); ++i) {
        historical_counts[i] += current_interval_counts[i];
      }
    }
    
    // Transfer temporal data to another TemporalPlaceData instance
    void transferTo(TemporalPlaceData& target) {
      target.current_interval_counts = current_interval_counts;
      target.historical_counts = historical_counts;
      target.observation_times = observation_times;
      target.total_observations = total_observations;
      target.last_fremen_update = last_fremen_update;
      target.cached_entropy = cached_entropy;
      // Note: position and binding state are not transferred
    }
    
    // Bind this temporal data to a node
    void bindToNode(spark_dsg::NodeId node_id, size_t source_hash_cell = 0) {
      is_bound_to_node = true;
      bound_node_id = node_id;
      original_hash_cell = source_hash_cell;
    }
    
    // Unbind this temporal data from any node
    void unbindFromNode() {
      is_bound_to_node = false;
      bound_node_id = 0;
      // Note: original_hash_cell is preserved for potential re-binding
    }
    
    // Normalize current counts
    std::vector<int> getNormalizedCounts() const {
      std::vector<int> normalized = current_interval_counts;
      int max_count = *std::max_element(normalized.begin(), normalized.end());
      
      if (max_count > 0) {
        // Normalize to 0-100 range
        for (int& count : normalized) {
          count = (count * 100) / max_count;
        }
      } else {
        // No data - fill with -1
        std::fill(normalized.begin(), normalized.end(), -1);
      }
      
      return normalized;
    }
  };

 protected:
  // Initialize ROS publishers, subscribers, and services
  void initializeRosInterface();
  
  // Main processing timer callback
  void processingTimerCallback(const ros::TimerEvent& event);
  
  // Update temporal models with accumulated movement data
  void updateTemporalModels();
  
  // Periodic timer callback for Fremen model updates
  void fremenUpdateTimerCallback(const ros::TimerEvent& event);
  
  // Write a timing measurement to the CSV file (thread-safe via ROS timer serialization)
  void logTiming(const std::string& callback_name, double elapsed_us);
  
  // Periodic timer callback for global stability and binding
  void globalStabilityTimerCallback(const ros::TimerEvent& event);
  
  // Timer callback for periodic place synchronization from Hydra DSG
  void placeSyncTimerCallback(const ros::TimerEvent& event);
  
  // Process queued DSG updates asynchronously (non-blocking)
  void processQueuedDsgUpdates();
  
  // Create global state vector from all places
  std::vector<int> createGlobalStateVector() const;
  
  // Extract place-specific predictions from global state vector
  std::vector<double> extractPlacePrediction(spark_dsg::NodeId place_id,
                                            const std::vector<double>& global_predictions) const;
  
  // Initialize global Fremen model with all current places
  void initializeGlobalFremenModel();
  
  // Process DSG update (can be deferred for performance)
  void processGraphUpdate(const hydra_msgs::DsgUpdate::ConstPtr& msg);
  
  // Process people detection messages
  void processDetections();
  
  // Associate people detections with place nodes
  void associateDetectionsWithPlaces(const geometry_msgs::PoseArray::ConstPtr& detections);
  
  // Calculate movement direction from pose orientation (assumes orientation = movement direction)
  size_t getOrientationBin(const geometry_msgs::Quaternion& orientation);
  
  // Get current time interval for Fremen modeling
  size_t getCurrentTimeInterval();
  
  // Publish visualization markers for RViz
  void publishVisualization();
  
  // Compute metrics from historical data
  void computeHistoricalMetrics();
  
  // Compute metrics from Fremen predictions
  void computePredictionMetrics();
  
  // Unified metric access based on configuration
  double getEntropy(const TemporalPlaceData& place_data) const;
  double getFlowMagnitude(const TemporalPlaceData& place_data) const;
  double getBestAngle(const TemporalPlaceData& place_data) const;
  std::vector<double> getFlowProbabilities(const TemporalPlaceData& place_data) const;
  
  // Publish temporal map messages
  void publishTemporalNodes();
  void publishTemporalMap();
  
  // Create individual node messages
  aion::AionTemporalNode createTemporalNodeMessage(size_t place_id, const TemporalPlaceData& place_data);
  aion::AionHistoricalNode createHistoricalNodeMessage(size_t place_id, const TemporalPlaceData& place_data);

  // Compute flow metrics from data
  double computeFlowMagnitude(const std::vector<int>& counts) const;
  double computeFlowMagnitude(const std::vector<double>& probabilities) const;
  double computeEntropyFromCounts(const std::vector<int>& counts) const;
  double computeEntropyFromProbabilities(const std::vector<double>& probabilities) const;
  double computeBestAngle(const std::vector<int>& counts) const;
  double computeBestAngle(const std::vector<double>& probabilities) const;
  std::vector<double> computeProbabilitiesFromCounts(const std::vector<int>& counts) const;
  std::vector<int> computeNormalizedCounts(const std::vector<int>& counts) const;
  
  // Update temporal models with Fremen predictions (like fremenUpdateTimerCallback but for specific time)
  void updateTemporalModelsWithFremenPrediction(uint64_t prediction_time_ns);
  
  // Create unified temporal markers (replaces flow, entropy, and places markers)
  visualization_msgs::MarkerArray createUnifiedTemporalMarkers();
  
  // Create debug markers for hash cells and binding visualization
  visualization_msgs::MarkerArray createHashCellMarkers();
  visualization_msgs::MarkerArray createDebugBindingMarkers();
  
  // Get dominant flow direction from predictions (for marker orientation)
  double getDominantFlowDirection(const std::vector<double>& predictions) const;

 private:
  // ROS callback for Hydra scene graph updates
  void hydraGraphCallback(const hydra_msgs::DsgUpdate::ConstPtr& msg);
  
  // ROS callback for people detection messages
  void peopleDetectionCallback(const geometry_msgs::PoseArray::ConstPtr& msg);
  
  // Service callback for temporal predictions
  bool predictionServiceCallback(aion::GetTemporalPrediction::Request& request,
                                aion::GetTemporalPrediction::Response& response);
  
  // Service callback for all places predictions
  bool allPlacePredictionsServiceCallback(aion::GetAllPlacePredictions::Request& request,
                                         aion::GetAllPlacePredictions::Response& response);
    
  // Service callback for reset
  bool resetServiceCallback(std_srvs::Trigger::Request& request,
                           std_srvs::Trigger::Response& response);
                           
  // Service callback for manual Fremen update
  bool manualFremenUpdateCallback(std_srvs::Trigger::Request& request,
                                 std_srvs::Trigger::Response& response);
  
  // Service callback for exporting navigation data for A* analysis  
  bool exportNavigationDataCallback(aion::ExportNavigationData::Request& request,
                                   aion::ExportNavigationData::Response& response);
  
  // Helper methods for export service
  bool exportHydraNavigationLayer(const std::string& filename, int& num_nodes, int& num_edges, bool include_connectivity);
  bool exportTemporalDynamics(const std::string& filename, uint64_t prediction_time, int& num_temporal_places);
  
  // Find nearest place node to given position
  spark_dsg::NodeId findNearestPlace(const Eigen::Vector3d& position);
  
  // Find nearest place using 2D distance (ignores Z-height differences)
  spark_dsg::NodeId findNearestPlace2D(const Eigen::Vector3d& position);
  
  // Efficient O(log n) nearest place search using KD-tree
  spark_dsg::NodeId findNearestPlaceEfficient(const Eigen::Vector3d& position);
  
  // Alternative: O(1) spatial hash based lookup for nearby places
  std::vector<spark_dsg::NodeId> findNearbyPlacesInHashCell(const Eigen::Vector3d& position) const;
  
  // Rebuild spatial index when places cache changes
  void rebuildSpatialIndex();
  
  // Extract places layer from scene graph
  void extractPlacesFromGraph();
  
  // Extract places from temporary graph without storing it
  void extractPlacesFromTempGraph(const std::shared_ptr<spark_dsg::DynamicSceneGraph>& temp_graph);
  
  // PERFORMANCE: Selective binary parsing to extract only places without full deserialization
  bool extractPlacesFromBinaryData(const std::vector<uint8_t>& binary_data,
                                   std::map<spark_dsg::NodeId, Eigen::Vector3d>& places);
  
  // Convert entropy value to RGB color for visualization
  std_msgs::ColorRGBA entropyToColor(double entropy) const;
  
  // Report comprehensive system status (performance and data integrity)
  void reportSystemStatus();
  
  // Spatial hash-based stable ordering for global state vector (prevents Fremen confusion)
  size_t getSpatialHashIndex(spark_dsg::NodeId place_id, const Eigen::Vector3d& position) const;
  
  // Convert hash cell back to 3D position (for unbinding operations)
  Eigen::Vector3d getPositionFromHashCell(size_t hash_cell) const;
  
  // Get stable spatial ordering index for a place (ensures consistent global state vector)
  size_t getStablePlaceIndex(spark_dsg::NodeId place_id) const;

  size_t getStableSpatialIndex(size_t spatial_hash) const;

  // Global stability window and delayed binding methods
  void assignDynamicsToStableNodes();
  void handleNodeRemoval();
  size_t getTemporalDataLocation(const Eigen::Vector3d& position);
  
  // Helper method to distinguish between hash cell entries and node-based entries
  bool isNodeBasedEntry(size_t spatial_key, const TemporalPlaceData& place_data) const;

  Config config_;
  
  // All configurable parameters loaded from YAML - no hardcoded values
  
  // Timing parameters
  double processing_timer_rate_;
  
  // Performance parameters
  int detection_buffer_size_;
  
  // Artificial place generation
  double artificial_place_grid_size_;
  int artificial_place_grid_offset_;
  
  // Marker visualization ranges
  double marker_size_min_multiplier_;
  double marker_size_max_multiplier_;
  double marker_transparency_min_alpha_;
  double marker_transparency_max_alpha_;
  double arrow_length_min_multiplier_;
  double arrow_length_max_multiplier_;
  
  // Inactive place visualization
  double inactive_marker_scale_factor_;
  double inactive_marker_alpha_;
  double inactive_marker_color_r_;
  double inactive_marker_color_g_;
  double inactive_marker_color_b_;
  
  // Entropy color mapping
  double max_entropy_for_normalization_;
  
  // Prediction visualization
  double prediction_green_tint_increase_;
  double prediction_transparency_reduction_;
  
  // Arrow dimensions
  double arrow_shaft_diameter_;
  double arrow_head_diameter_;
  double arrow_head_length_;
  
  // Standardized Z levels
  double temporal_marker_z_level_;
  double inactive_marker_z_level_;
  
  // Debug throttle intervals
  double throttle_debug_general_;
  double throttle_debug_detailed_;
  double throttle_info_statistics_;
  double throttle_warning_overflow_;
  
  // Global model parameters
  int min_places_for_connection_;
  double prediction_fallback_intensity_;
  double prediction_strength_weight_;
  
  // Simulation parameters
  double min_simulated_observations_;
  
  // ROS node handle
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  
  // Scene graph data
  spark_dsg::DynamicSceneGraph::Ptr scene_graph_;
  std::mutex scene_graph_mutex_;
  
  // ROS interface
  ros::Subscriber hydra_sub_;
  ros::Subscriber people_sub_;
  ros::Publisher temporal_places_pub_; // Unified markers encoding entropy, flow, activity
  ros::Publisher hash_cells_pub_; // Debug: Hash cell locations
  ros::Publisher debug_bindings_pub_;   // Debug: Binding connections
  
  // Publishers for temporal data messages
  ros::Publisher temporal_nodes_pub_;    // AionTemporalNode messages (prediction-based)
  ros::Publisher historical_nodes_pub_;  // AionHistoricalNode messages (historical data)
  ros::Publisher temporal_map_pub_;      // AionTemporalMap messages (unified map)
  
  ros::ServiceServer prediction_service_;
  ros::ServiceServer all_predictions_service_;
  ros::ServiceServer reset_service_;
  ros::ServiceServer manual_fremen_service_;
  ros::ServiceServer export_navigation_service_;
  ros::Timer processing_timer_;
  ros::Timer fremen_update_timer_;
  ros::Timer global_stability_timer_;  // Timer for global stability window
  ros::Timer place_sync_timer_;       // Timer for periodic place synchronization
  
  // Fremen interface for temporal modeling
  std::unique_ptr<FremenInterface> fremen_;
  
  // Modern temporal data storage: uses size_t keys for both hash cells and node-based entries
  // Hash cells: spatial hash values
  // Node entries: node_id | 0x8000000000000000ULL
  std::map<size_t, TemporalPlaceData> temporal_place_data_;
  mutable std::shared_mutex temporal_data_mutex_;
  
  // Global model state (simplified - no complex ordering needed with spatial hash)
  bool global_model_initialized_ = false;
  
  // Spatial hash-based stable ordering: spatial hash to global state index mapping
  mutable std::map<size_t, size_t> spatial_hash_to_index_;
  mutable std::mutex place_ordering_mutex_;
  
  // Places layer cache
  std::map<spark_dsg::NodeId, Eigen::Vector3d> places_cache_;
  mutable std::shared_mutex places_cache_mutex_;
  
  // Efficient spatial indexing for place lookup
  std::unique_ptr<hydra::PointNeighborSearch> point_spatial_index_;
  std::vector<spark_dsg::NodeId> place_nodes_for_index_;
  bool spatial_index_needs_rebuild_ = true;
  mutable std::mutex spatial_index_mutex_;
  
  // Additional spatial hash map for O(1) neighbor queries
  std::unordered_map<size_t, std::vector<spark_dsg::NodeId>> spatial_hash_to_places_;
  mutable std::mutex spatial_hash_mutex_;
  
  // Latest DSG message for periodic processing
  hydra_msgs::DsgUpdate::ConstPtr latest_dsg_msg_;
  mutable std::mutex latest_dsg_mutex_;
  
  // Timing for periodic updates
  std::chrono::steady_clock::time_point last_update_time_;
  std::chrono::steady_clock::time_point last_sync_time_;
  
  // Current detection buffer for processing
  std::queue<geometry_msgs::PoseArray::ConstPtr> detection_buffer_;
  std::mutex detection_mutex_;

  // TF (robot pose lookup for the detection range gate). Declared together so
  // tf_listener_ can be initialized from tf_buffer_ in the constructor.
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
    
  // Asynchronous DSG processing queue (non-blocking)
  std::queue<hydra_msgs::DsgUpdate::ConstPtr> dsg_update_queue_;
  std::mutex dsg_queue_mutex_;
  
  // Global stability tracking for delayed binding
  GlobalStabilityInfo global_stability_;
  mutable std::mutex global_stability_mutex_;
  
  // Temporal binding tracking (hash_cell -> binding info)
  std::map<size_t, TemporalBinding> hash_bindings_;
  std::mutex binding_mutex_;
  
  // Statistics
  size_t total_detections_processed_ = 0;
  size_t total_associations_made_ = 0;
  size_t total_bindings_created_ = 0;
  
  // Timing profiling
  bool enable_timing_ = false;
  std::string timing_output_file_;
  std::ofstream timing_file_;
  
  // Memory footprint computation
  struct MemoryFootprint {
    size_t num_places = 0;
    size_t num_bound_nodes = 0;
    size_t num_hash_cells = 0;
    size_t total_observations = 0;
    size_t estimated_bytes = 0;
  };
  MemoryFootprint computeMemoryFootprint() const;
};

}  // namespace aion
