#include "aion/temporal_dynamics_module.h"
#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "temporal_dynamics_node");
  
  auto node = std::make_unique<aion::TemporalDynamicsNode>();
  
  ROS_INFO("Starting Temporal Dynamics Node");
  
  try {
    node->run();
  } catch (const std::exception& e) {
    ROS_ERROR("Error in temporal dynamics node: %s", e.what());
  }
  
  return 0;
}
