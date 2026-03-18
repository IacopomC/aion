cmake_minimum_required(VERSION 3.16)
project(aion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  std_msgs
  std_srvs
  visualization_msgs
  tf2
  tf2_geometry_msgs
  message_generation
  hydra_msgs
  actionlib
  fremenarray
)

# External dependencies
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(spark_dsg REQUIRED)
find_package(hydra REQUIRED)
# find_package(config_utilities REQUIRED)

# Generate messages and services
add_message_files(
  FILES
  PlacePrediction.msg
  AionTemporalNode.msg
  AionHistoricalNode.msg
  AionTemporalMap.msg
)

add_service_files(
  FILES
  GetTemporalPrediction.srv
  GetAllPlacePredictions.srv
  ExportNavigationData.srv
  ExportTemporalMap.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS 
    roscpp
    geometry_msgs
    std_msgs
    std_srvs
    visualization_msgs
    tf2
    tf2_geometry_msgs
    message_runtime
    hydra_msgs
    actionlib
    fremenarray
  DEPENDS Eigen3 spark_dsg hydra
)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Library
add_library(${PROJECT_NAME}
  src/temporal_dynamics_module.cpp
  src/graph_processing.cpp
  src/detection_processing.cpp
  src/fremen_integration.cpp
  src/visualization.cpp
  src/services.cpp
  src/binding.cpp
  src/metrics.cpp
  src/fremen_interface.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  spark_dsg::spark_dsg
  hydra::hydra
)

add_dependencies(${PROJECT_NAME} 
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
)

# Standalone node executable
add_executable(temporal_dynamics_node
  src/temporal_dynamics_node.cpp
)

target_link_libraries(temporal_dynamics_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

add_dependencies(temporal_dynamics_node 
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
)

# Install executables
# install(TARGETS temporal_dynamics_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# Install headers
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
# )

# Install library
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

# Install config files
# install(
#   DIRECTORY config/
#   DESTINATION share/${PROJECT_NAME}/config
# )

# Install launch files if they exist
# install(
#   DIRECTORY launch/
#   DESTINATION share/${PROJECT_NAME}/launch
#   OPTIONAL
# )
