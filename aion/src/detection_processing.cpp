#include "aion/temporal_dynamics_module.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>
#include <chrono>
#include <cmath>

namespace aion {

void TemporalDynamicsNode::peopleDetectionCallback(const geometry_msgs::PoseArray::ConstPtr& msg) {
  ROS_DEBUG_THROTTLE(60.0, "People detection received: %zu poses", msg->poses.size());

  std::lock_guard<std::mutex> lock(detection_mutex_);
  detection_buffer_.push(msg);

  // Limit buffer size to prevent delayed processing - configurable size for real-time performance
  while (detection_buffer_.size() > static_cast<size_t>(detection_buffer_size_)) {
    detection_buffer_.pop();
    ROS_WARN_THROTTLE(throttle_warning_overflow_, "Detection buffer overflow - dropping old detections for real-time performance");
  }

  ROS_DEBUG("Detection buffer size: %zu", detection_buffer_.size());
}

void TemporalDynamicsNode::processingTimerCallback(const ros::TimerEvent& event) {
  auto t0 = std::chrono::high_resolution_clock::now();
  // Process accumulated detections
  processDetections();
  if (enable_timing_) {
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    logTiming("processDetections", us);
  }
}

void TemporalDynamicsNode::processDetections() {
  std::queue<geometry_msgs::PoseArray::ConstPtr> local_buffer;

  // Move detections to local buffer
  {
    std::lock_guard<std::mutex> lock(detection_mutex_);
    local_buffer.swap(detection_buffer_);
  }

  // Process each detection batch
  while (!local_buffer.empty()) {
    auto detections = local_buffer.front();
    local_buffer.pop();

    associateDetectionsWithPlaces(detections);
    total_detections_processed_ += detections->poses.size();
  }
}

void TemporalDynamicsNode::associateDetectionsWithPlaces(
    const geometry_msgs::PoseArray::ConstPtr& detections) {

  auto detection_start = std::chrono::high_resolution_clock::now();

  std::lock_guard<std::shared_mutex> temporal_lock(temporal_data_mutex_);

  const uint64_t current_time = ros::Time::now().toNSec();

  // Robot pose in the map frame for the detection range gate (frame_id ->
  // robot_frame translation). Gate is skipped if disabled or TF unavailable.
  bool have_robot = false;
  double robot_x = 0.0, robot_y = 0.0;
  if (std::isfinite(config_.max_detection_range_m)) {
    try {
      auto tf = tf_buffer_.lookupTransform(
          config_.frame_id, config_.robot_frame, ros::Time(0), ros::Duration(0.1));
      robot_x = tf.transform.translation.x;
      robot_y = tf.transform.translation.y;
      have_robot = true;
    } catch (const tf2::TransformException& ex) {
      ROS_WARN_THROTTLE(5.0, "range-gate TF lookup failed (%s -> %s): %s",
                        config_.frame_id.c_str(), config_.robot_frame.c_str(), ex.what());
    }
  }

  // Track detection processing metrics
  size_t detections_processed = 0;
  size_t detections_associated = 0;
  size_t detections_skipped = 0;
  size_t hash_cell_updates = 0;
  size_t bound_node_updates = 0;

  // Process each detection efficiently
  for (const auto& pose : detections->poses) {
    detections_processed++;
    Eigen::Vector3d detection_pos(pose.position.x, pose.position.y, pose.position.z);

    // Range gate: drop detections farther than max_detection_range_m from the robot.
    if (have_robot) {
      const double dxr = pose.position.x - robot_x;
      const double dyr = pose.position.y - robot_y;
      if (dxr * dxr + dyr * dyr >
          config_.max_detection_range_m * config_.max_detection_range_m) {
        detections_skipped++;
        continue;
      }
    }

    // Find the appropriate location for temporal data (either bound node or hash cell)
    spark_dsg::NodeId target_place_id = 0;
    {
      std::shared_lock<std::shared_mutex> places_lock(places_cache_mutex_);
      target_place_id = findNearestPlace(detection_pos);
    }

    // Determine where to store temporal data based on global stability window
    size_t temporal_location = getTemporalDataLocation(detection_pos);

    if (temporal_location != 0) {
      auto& place_data = temporal_place_data_[temporal_location];
      if (place_data.current_interval_counts.empty()) {
        place_data.initialize(config_.num_orientation_bins);

        // For hash cells, use the grid center position for consistent spatial reference
        if (!(temporal_location & 0x8000000000000000ULL)) {
          // This is a hash cell - use grid center position
          place_data.position = getPositionFromHashCell(temporal_location);
        } else {
          // This is a node-based entry - use the actual detection position
          place_data.position = detection_pos;
        }

        global_model_initialized_ = false;
      }

      place_data.observation_times.push_back(current_time);

      // Track whether this is a hash cell or bound node update
      if (temporal_location & 0x8000000000000000ULL) {
        bound_node_updates++;
      } else {
        hash_cell_updates++;
      }

      // Note: Don't increment total_observations here to prevent real-time visual updates
      // It will be updated during periodic Fremen model updates instead

      // Use pose orientation directly (assumes orientation represents movement direction)
      // This avoids deriving direction from inter-frame velocity estimates
      size_t orientation_bin = getOrientationBin(pose.orientation);
      if (orientation_bin < static_cast<size_t>(config_.num_orientation_bins)) {
        place_data.current_interval_counts[orientation_bin]++;
      }

      total_associations_made_++;
      detections_associated++;

      ROS_DEBUG_THROTTLE(30.0, "Associated detection with temporal location %zu (node %lu)",
                        temporal_location, target_place_id);
    } else {
      // No suitable location found - skip this detection
      detections_skipped++;
      ROS_DEBUG("No suitable temporal location found - skipping detection");
      continue;
    }
  }

  auto detection_end = std::chrono::high_resolution_clock::now();
  auto detection_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(detection_end - detection_start).count();

  // Log detection processing performance and data integrity
  total_detections_processed_ += detections->poses.size();

  if (detections_processed > 0) {
    // ROS_INFO("DETECTION BATCH: %zu processed, %zu associated (%zu hash, %zu bound), %zu skipped (Time=%ldμs)",
    //          detections_processed, detections_associated, hash_cell_updates, bound_node_updates,
    //          detections_skipped, detection_time_ms);
  }

  if (config_.enable_delayed_binding) {
    // ROS_DEBUG("Processed %zu detections: %zu temporal locations, %zu bindings created",
    //           detections->poses.size(), temporal_place_data_.size(),
    //           total_bindings_created_);

    // Periodically report comprehensive system status
    static auto last_status_report = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_report = std::chrono::duration_cast<std::chrono::seconds>(now - last_status_report).count();

    if (time_since_last_report >= 10) {
      last_status_report = now;
    }
  } else {
    ROS_DEBUG("Processed %zu detections: %zu temporal locations (global delayed binding disabled)",
              detections->poses.size(), temporal_place_data_.size());
  }
}

size_t TemporalDynamicsNode::getCurrentTimeInterval() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto tm = *std::localtime(&time_t);

  // Weekly pattern: day_of_week * 24 + hour
  size_t interval = (tm.tm_wday * 24) + tm.tm_hour;
  return interval % (24 * 7); // Ensure within bounds
}

size_t TemporalDynamicsNode::getOrientationBin(const geometry_msgs::Quaternion& orientation) {
  // Convert quaternion to yaw angle
  double yaw = tf2::getYaw(orientation);

  // Convert to degrees and normalize to [0, 360)
  double yaw_degrees = yaw * 180.0 / M_PI;
  if (yaw_degrees < 0) {
    yaw_degrees += 360.0;
  }

  // Calculate bin (with half-bin offset for centering)
  double bin_size_degrees = 360.0 / config_.num_orientation_bins;
  size_t bin_index = static_cast<size_t>((yaw_degrees + bin_size_degrees/2.0) / bin_size_degrees) % config_.num_orientation_bins;

  return bin_index;
}

}  // namespace aion
