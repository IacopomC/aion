#!/usr/bin/env python3

"""
Aion Temporal Map Export Service

This script provides a ROS service to export Aion's temporal map data in various formats

Usage:
    rosrun aion aion_map_export_service.py
    rosservice call /aion/export_temporal_map "{min_bounds: {x: -15.0, y: -10.0, z: 0.0}, max_bounds: {x: 15.0, y: 20.0, z: 0.0}, grid_resolution: 0.3, map_type: 'all', file_path: '/tmp/aion', format: 'numpy', prediction_time: -1}"
"""

import rospy
import numpy as np
import json
import os
from typing import List
from aion.msg import AionTemporalMap
from aion.srv import ExportTemporalMap, ExportTemporalMapResponse
from geometry_msgs.msg import Point


class AionMapExportService:
    """
    Service that exports Aion temporal maps to various formats for comparison and analysis.
    """
    
    def __init__(self, service_name: str = '/aion/export_temporal_map',
                 temporal_map_topic: str = '/aion/temporal_map'):
        """
        Initialize the export service.
        
        Args:
            service_name: Name of the ROS service
            temporal_map_topic: Topic containing AionTemporalMap messages
        """
        self.latest_map = None
        self.map_frame_id = "map"  # Default frame
        
        # Subscribe to temporal map updates
        self.map_subscriber = rospy.Subscriber(
            temporal_map_topic,
            AionTemporalMap, 
            self._temporal_map_callback,
            queue_size=1
        )
        
        # Set up service
        self.export_service = rospy.Service(
            service_name, 
            ExportTemporalMap, 
            self._handle_export_request
        )
        
        rospy.loginfo(f"Aion Map Export Service initialized on '{service_name}'")
        rospy.loginfo(f"Subscribing to temporal map topic '{temporal_map_topic}'")
    
    def _temporal_map_callback(self, msg: AionTemporalMap):
        """Callback for receiving temporal map updates."""
        self.latest_map = msg
        self.map_frame_id = msg.header.frame_id
        rospy.logdebug(f"Updated temporal map with {len(msg.temporal_nodes)} temporal nodes "
                      f"and {len(msg.historical_nodes)} historical nodes")
    
    def _handle_export_request(self, request):
        """
        Handle service requests for exporting map data.
        
        Args:
            request: ExportTemporalMap service request
            
        Returns:
            ExportTemporalMapResponse: Success status and saved file paths
        """
        try:
            rospy.loginfo(f"Received export request: type={request.map_type}, format={request.format}, "
                         f"prediction_time={request.prediction_time}")
            
            # Check if we have map data
            if self.latest_map is None:
                raise ValueError("No temporal map data available. Make sure the temporal map is being published.")
            
            # Prepare metadata
            timestamp = rospy.Time.now().to_sec()
            metadata = {
                'timestamp': timestamp,
                'frame_id': self.map_frame_id,
                'update_interval_seconds': self.latest_map.update_interval_seconds,
                'num_orientation_bins': self.latest_map.num_orientation_bins,
                'total_places': self.latest_map.total_places,
                'bound_places': self.latest_map.bound_places,
                'hash_places': self.latest_map.hash_places
            }
            
            saved_files = []
            
            # Determine which map types to export
            if request.map_type == 'all':
                map_types = ['temporal', 'historical']
            elif request.map_type in ['temporal', 'historical']:
                map_types = [request.map_type]
            else:
                raise ValueError(f"Invalid map_type: {request.map_type}. Must be 'temporal', 'historical', or 'all'")
            
            # Export based on format
            for map_type in map_types:
                if request.format == 'json':
                    files = self._export_to_json(map_type, request.file_path, timestamp, 
                                                metadata, request.prediction_time)
                elif request.format == 'numpy':
                    files = self._export_to_numpy(map_type, request.file_path, timestamp,
                                                 metadata, request.min_bounds, request.max_bounds,
                                                 request.grid_resolution, request.prediction_time)
                elif request.format == 'csv':
                    files = self._export_to_csv(map_type, request.file_path, timestamp,
                                               metadata, request.prediction_time)
                else:
                    raise ValueError(f"Invalid format: {request.format}. Must be 'json', 'numpy', or 'csv'")
                
                if files:
                    saved_files.extend(files)
            
            # Create response
            response = ExportTemporalMapResponse()
            response.success = True
            response.message = f"Successfully exported {len(saved_files)} files"
            response.saved_files = saved_files
            
            rospy.loginfo(f"Export completed: {saved_files}")
            return response
            
        except Exception as e:
            rospy.logerr(f"Failed to export map data: {e}")
            response = ExportTemporalMapResponse()
            response.success = False
            response.message = str(e)
            response.saved_files = []
            return response
    
    def _export_to_json(self, map_type: str, base_path: str, timestamp: float, 
                       metadata: dict, prediction_time: int = -1) -> List[str]:
        """Export temporal map to JSON format."""
        filename = f"{base_path}_{timestamp:.0f}_{map_type}.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Get nodes based on type
        if map_type == 'temporal':
            nodes = self.latest_map.temporal_nodes
        elif map_type == 'historical':
            nodes = self.latest_map.historical_nodes
        else:
            raise ValueError(f"Invalid map_type for JSON export: {map_type}")
        
        # Filter temporal nodes by prediction time if specified
        if map_type == 'temporal' and prediction_time >= 0:
            nodes = [node for node in nodes if node.prediction_time == prediction_time]
        
        # Convert to serializable dictionary
        data = {
            'header': {
                'stamp': self.latest_map.header.stamp.to_sec(),
                'frame_id': self.latest_map.header.frame_id
            },
            'metadata': metadata,
            'map_type': map_type,
            'nodes': []
        }
        
        # Convert nodes
        for node in nodes:
            node_data = {
                'place_id': int(node.place_id),
                'position': {
                    'x': float(node.position.x),
                    'y': float(node.position.y), 
                    'z': float(node.position.z)
                },
                'entropy': float(node.entropy),
                'flow_magnitude': float(node.flow_magnitude),
                'best_angle': float(node.best_angle),
                'probabilities': [float(p) for p in node.probabilities],
                'is_bound_to_node': bool(node.is_bound_to_node)
            }
            
            # Add type-specific fields
            if map_type == 'temporal':
                node_data['prediction_time'] = int(node.prediction_time)
                node_data['spatial_hash'] = int(node.spatial_hash)
            elif map_type == 'historical':
                node_data['normalized_counts'] = [int(c) for c in node.normalized_counts]
                node_data['total_observations'] = int(node.total_observations)
            
            data['nodes'].append(node_data)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        rospy.loginfo(f"Exported {len(data['nodes'])} {map_type} nodes to {filename}")
        return [filename]
    
    def _export_to_numpy(self, map_type: str, base_path: str, timestamp: float,
                        metadata: dict, min_bounds: Point, max_bounds: Point,
                        grid_resolution: float, prediction_time: int = -1) -> List[str]:
        """Export temporal map to numpy arrays matching spatial grid format."""
        
        # Set default bounds if not specified
        if (min_bounds.x == 0.0 and min_bounds.y == 0.0 and 
            max_bounds.x == 0.0 and max_bounds.y == 0.0):
            x_min, x_max = -15.0, 15.0
            y_min, y_max = -10.0, 20.0
        else:
            x_min, x_max = min_bounds.x, max_bounds.x
            y_min, y_max = min_bounds.y, max_bounds.y
        
        grid_width = int((x_max - x_min) / grid_resolution)
        grid_height = int((y_max - y_min) / grid_resolution)
        
        # Get nodes based on type
        if map_type == 'temporal':
            nodes = self.latest_map.temporal_nodes
        elif map_type == 'historical':
            nodes = self.latest_map.historical_nodes
        else:
            raise ValueError(f"Invalid map_type for numpy export: {map_type}")
        
        # Filter temporal nodes by prediction time if specified
        if map_type == 'temporal' and prediction_time >= 0:
            nodes = [node for node in nodes if node.prediction_time == prediction_time]
        
        # Initialize grids
        entropy_grid = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
        flow_grid = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
        direction_grid = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
        
        # Fill grids with node data
        for node in nodes:
            # Convert world coordinates to grid indices
            grid_x = int((node.position.x - x_min) / grid_resolution)
            grid_y = int((node.position.y - y_min) / grid_resolution)
            
            # Check bounds
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                entropy_grid[grid_y, grid_x] = node.entropy
                flow_grid[grid_y, grid_x] = node.flow_magnitude
                # Convert radians to degrees for direction
                direction_grid[grid_y, grid_x] = np.degrees(node.best_angle) % 360.0
        
        # Save numpy arrays
        saved_files = []
        base_dir = os.path.dirname(base_path) if os.path.dirname(base_path) else '.'
        os.makedirs(base_dir, exist_ok=True)
        
        for grid_type, grid in [('entropy', entropy_grid), 
                              ('flow', flow_grid), 
                              ('direction', direction_grid)]:
            filename = f"{base_path}_{timestamp:.0f}_{map_type}_{grid_type}.npy"
            np.save(filename, grid)
            saved_files.append(filename)
            rospy.loginfo(f"Saved {map_type} {grid_type} grid to {filename}")
        
        # Save metadata
        grid_metadata = {
            'grid_resolution': grid_resolution,
            'grid_bounds': [x_min, x_max, y_min, y_max],
            'grid_shape': [grid_height, grid_width],
            'origin': [x_min, y_min, 0.0],
            'map_type': map_type,
            **metadata
        }
        
        if prediction_time >= 0:
            grid_metadata['prediction_time'] = prediction_time
        
        metadata_file = f"{base_path}_{timestamp:.0f}_{map_type}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(grid_metadata, f, indent=2)
        saved_files.append(metadata_file)
        
        return saved_files
    
    def _export_to_csv(self, map_type: str, base_path: str, timestamp: float,
                      metadata: dict, prediction_time: int = -1) -> List[str]:
        """Export temporal map to CSV format for analysis."""
        filename = f"{base_path}_{timestamp:.0f}_{map_type}.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Get nodes based on type
        if map_type == 'temporal':
            nodes = self.latest_map.temporal_nodes
        elif map_type == 'historical':
            nodes = self.latest_map.historical_nodes
        else:
            raise ValueError(f"Invalid map_type for CSV export: {map_type}")
        
        # Filter temporal nodes by prediction time if specified
        if map_type == 'temporal' and prediction_time >= 0:
            nodes = [node for node in nodes if node.prediction_time == prediction_time]
        
        # Prepare CSV data
        csv_lines = []
        
        # Create header
        header = ["place_id", "x", "y", "z", "entropy", "flow_magnitude", 
                 "best_angle_deg", "is_bound_to_node"]
        
        # Add type-specific headers
        if map_type == 'temporal':
            header.extend(["prediction_time", "spatial_hash"])
        elif map_type == 'historical':
            header.extend(["total_observations"])
        
        # Add orientation bin headers
        num_bins = self.latest_map.num_orientation_bins if self.latest_map else 0
        for i in range(num_bins):
            header.append(f"prob_bin_{i}")
        
        # Add normalized counts for historical nodes
        if map_type == 'historical':
            for i in range(num_bins):
                header.append(f"count_bin_{i}")
        
        csv_lines.append(",".join(header))
        
        # Add node data
        for node in nodes:
            row = [
                str(node.place_id),
                f"{node.position.x:.3f}",
                f"{node.position.y:.3f}",
                f"{node.position.z:.3f}",
                f"{node.entropy:.4f}",
                f"{node.flow_magnitude:.4f}",
                f"{np.degrees(node.best_angle):.1f}",
                str(node.is_bound_to_node)
            ]
            
            # Add type-specific data
            if map_type == 'temporal':
                row.extend([str(node.prediction_time), str(node.spatial_hash)])
            elif map_type == 'historical':
                row.append(str(node.total_observations))
            
            # Add probability values
            for prob in node.probabilities:
                row.append(f"{prob:.4f}")
            
            # Add normalized counts for historical nodes
            if map_type == 'historical':
                for count in node.normalized_counts:
                    row.append(str(count))
            
            csv_lines.append(",".join(row))
        
        # Write to file
        with open(filename, 'w') as f:
            f.write("\n".join(csv_lines))
        
        rospy.loginfo(f"Exported {len(nodes)} {map_type} nodes to {filename}")
        return [filename]


def main():
    # Initialize ROS node
    rospy.init_node('aion_map_export_service', anonymous=True)
    
    # Create export service
    service_name = rospy.get_param(
        '~service_name',
        rospy.get_param('/temporal_dynamics_node/export_temporal_map_service_name', '/aion/export_temporal_map')
    )
    temporal_map_topic = rospy.get_param(
        '~temporal_map_topic',
        rospy.get_param('/temporal_dynamics_node/temporal_map_topic', '/aion/temporal_map')
    )
    export_service = AionMapExportService(service_name, temporal_map_topic)
    
    rospy.loginfo("Aion Map Export Service is ready")
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Aion Map Export Service")


if __name__ == '__main__':
    main()