#!/usr/bin/env python3

"""
Temporal Path Visualizer

Simple RViz node that loads paths computed by temporal_astar_comparison.py
and visualizes them for comparison.

Usage:
    1. Run export service to get JSON files
    2. Run temporal_astar_comparison.py with --save-paths to compute paths
    3. Run this node to visualize the paths in RViz
    
    rosrun aion temporal_path_visualizer.py \
    --paths /tmp/aion_path_planning/aion_computed_paths.json \
    --navigation /tmp/aion_path_planning/aion_path_navigation.json
"""

import rospy
import json
import argparse
from typing import Dict, List, Optional
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Path

class TemporalPathVisualizer:
    """Simple RViz visualizer for temporal A* comparison results."""
    
    def __init__(self, paths_file: str = None, navigation_file: str = None):
        """Initialize the visualizer."""
        rospy.init_node('temporal_path_visualizer', anonymous=True)
        
        # Publishers
        self.path_markers_pub = rospy.Publisher('temporal_viz/path_markers', MarkerArray, queue_size=10)
        self.standard_path_pub = rospy.Publisher('temporal_viz/standard_path', Path, queue_size=10)
        self.temporal_path_pub = rospy.Publisher('temporal_viz/temporal_path', Path, queue_size=10)
        self.nodes_pub = rospy.Publisher('temporal_viz/nodes', MarkerArray, queue_size=10)
        
        # Data
        self.nodes = {}
        self.comparison_data = None
        
        # Load data
        if navigation_file:
            self.load_navigation_data(navigation_file)
        if paths_file:
            self.load_paths_data(paths_file)
        
        # Parameters
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)  # Hz
        
        rospy.loginfo("Temporal Path Visualizer initialized")
        rospy.loginfo("Topics published:")
        rospy.loginfo("  - temporal_viz/path_markers (MarkerArray) - Path visualization")
        rospy.loginfo("  - temporal_viz/standard_path (Path) - Standard A* path")  
        rospy.loginfo("  - temporal_viz/temporal_path (Path) - Temporal A* path")
        rospy.loginfo("  - temporal_viz/nodes (MarkerArray) - Navigation nodes")
        
    def load_navigation_data(self, filename: str):
        """Load navigation node data for visualization."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            for node_data in data['nodes']:
                node_id = node_data['id']
                self.nodes[node_id] = {
                    'x': node_data['position']['x'],
                    'y': node_data['position']['y'],
                    'z': node_data['position']['z']
                }
                
            rospy.loginfo(f"Loaded {len(self.nodes)} navigation nodes")
            
        except Exception as e:
            rospy.logwarn(f"Failed to load navigation data: {e}")
            
    def load_paths_data(self, filename: str):
        """Load computed path comparison data."""
        try:
            with open(filename, 'r') as f:
                self.comparison_data = json.load(f)
                
            rospy.loginfo("Loaded path comparison data:")
            rospy.loginfo(f"  Standard path: {len(self.comparison_data['standard']['path'])} nodes")
            rospy.loginfo(f"  Temporal path: {len(self.comparison_data['temporal']['path'])} nodes") 
            rospy.loginfo(f"  Cost difference: {self.comparison_data['cost_difference']:.2f}")
            
        except Exception as e:
            rospy.logerr(f"Failed to load paths data: {e}")
            
    def create_path_marker(self, path: List[int], marker_id: int, color: List[float], name: str) -> Marker:
        """Create a line strip marker for a path."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = name
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15  # Line width
        marker.color = ColorRGBA(color[0], color[1], color[2], color[3])
        
        # Add points
        for node_id in path:
            if node_id in self.nodes:
                point = Point()
                point.x = self.nodes[node_id]['x']
                point.y = self.nodes[node_id]['y']
                point.z = self.nodes[node_id]['z'] + 0.1  # Slightly above ground
                marker.points.append(point)
                
        return marker
        
    def create_nav_path(self, path: List[int], frame_id: str = "map") -> Path:
        """Create nav_msgs/Path from node path."""
        nav_path = Path()
        nav_path.header.frame_id = frame_id
        nav_path.header.stamp = rospy.Time.now()
        
        for node_id in path:
            if node_id in self.nodes:
                pose = PoseStamped()
                pose.header = nav_path.header
                pose.pose.position.x = self.nodes[node_id]['x']
                pose.pose.position.y = self.nodes[node_id]['y']
                pose.pose.position.z = self.nodes[node_id]['z']
                pose.pose.orientation.w = 1.0
                nav_path.poses.append(pose)
                
        return nav_path
        
    def create_node_marker(self, node_id: int, marker_id: int, color: List[float], name: str, scale: float = 0.3) -> Marker:
        """Create a sphere marker for a node."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = name
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        if node_id in self.nodes:
            marker.pose.position.x = self.nodes[node_id]['x']
            marker.pose.position.y = self.nodes[node_id]['y']
            marker.pose.position.z = self.nodes[node_id]['z']
            
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = scale
        marker.color = ColorRGBA(color[0], color[1], color[2], color[3])
        
        return marker
        
    def publish_visualization(self):
        """Publish all visualization markers and paths."""
        if not self.comparison_data:
            return
            
        marker_array = MarkerArray()
        
        # Standard path (blue)
        std_path = self.comparison_data['standard']['path']
        if std_path:
            std_marker = self.create_path_marker(std_path, 0, [0.0, 0.0, 1.0, 1.0], "Standard A*")
            marker_array.markers.append(std_marker)
            
            # Publish as nav_msgs/Path too
            std_nav_path = self.create_nav_path(std_path)
            self.standard_path_pub.publish(std_nav_path)
        
        # Temporal path (red)
        temp_path = self.comparison_data['temporal']['path']
        if temp_path:
            temp_marker = self.create_path_marker(temp_path, 1, [1.0, 0.0, 0.0, 1.0], "Temporal A*")
            marker_array.markers.append(temp_marker)
            
            # Publish as nav_msgs/Path too
            temp_nav_path = self.create_nav_path(temp_path)
            self.temporal_path_pub.publish(temp_nav_path)
        
        # Start marker (green)
        if std_path:
            start_marker = self.create_node_marker(std_path[0], 2, [0.0, 1.0, 0.0, 1.0], "START", 0.5)
            marker_array.markers.append(start_marker)
        
        # Goal marker (purple)
        if std_path:
            goal_marker = self.create_node_marker(std_path[-1], 3, [1.0, 0.0, 1.0, 1.0], "GOAL", 0.5)
            marker_array.markers.append(goal_marker)
            
        # Publish all markers
        self.path_markers_pub.publish(marker_array)
        
        # Publish navigation nodes (gray, small)
        if self.nodes:
            node_markers = MarkerArray()
            for i, (node_id, node_data) in enumerate(self.nodes.items()):
                node_marker = Marker()
                node_marker.header.frame_id = "map"
                node_marker.header.stamp = rospy.Time.now()
                node_marker.ns = "navigation_nodes"
                node_marker.id = i
                node_marker.type = Marker.SPHERE
                node_marker.action = Marker.ADD
                node_marker.pose.position.x = node_data['x']
                node_marker.pose.position.y = node_data['y'] 
                node_marker.pose.position.z = node_data['z']
                node_marker.pose.orientation.w = 1.0
                node_marker.scale.x = node_marker.scale.y = node_marker.scale.z = 0.1
                node_marker.color = ColorRGBA(0.5, 0.5, 0.5, 0.3)
                node_markers.markers.append(node_marker)
                
            self.nodes_pub.publish(node_markers)
        
    def run(self):
        """Main run loop."""
        rate = rospy.Rate(self.publish_rate)
        
        while not rospy.is_shutdown():
            self.publish_visualization()
            rate.sleep()

def main():
    parser = argparse.ArgumentParser(description='Visualize temporal A* path comparison results')
    parser.add_argument('--paths', required=True, help='Path comparison JSON file')
    parser.add_argument('--navigation', help='Navigation nodes JSON file (for node visualization)')
    
    args = parser.parse_args()
    
    try:
        visualizer = TemporalPathVisualizer(args.paths, args.navigation)
        visualizer.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
