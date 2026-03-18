#!/usr/bin/env python3

"""
Aion Temporal Map Export Utility

This script subscribes to Aion's temporal map topic and saves the data
in various formats.

Usage:
    rosrun aion aion_map_exporter.py --format numpy --output /path/to/save
"""

import rospy
import numpy as np
import json
import argparse
import os
from aion.msg import AionTemporalMap


class AionMapExporter:
    """
    Exports Aion temporal maps to JSON, NumPy, or CSV.
    """
    
    def __init__(self, output_dir: str, format_type: str = 'json'):
        """
        Initialize the exporter.
        
        Args:
            output_dir: Directory to save exported files
            format_type: Export format ('json', 'numpy', 'csv')
        """
        self.output_dir = output_dir
        self.format_type = format_type
        self.latest_map = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Only subscribe if we're in auto-save mode (will be set later)
        self.map_subscriber = None
        
        rospy.loginfo(f"Aion Map Exporter initialized. Saving to {output_dir} in {format_type} format")
    
    def temporal_map_callback(self, msg: AionTemporalMap):
        """Callback for receiving temporal map updates."""
        self.latest_map = msg
        rospy.loginfo(f"Received temporal map with {len(msg.temporal_nodes)} temporal nodes "
                     f"and {len(msg.historical_nodes)} historical nodes")
        rospy.logdebug(f"Map timestamp: {msg.header.stamp.to_sec()}, frame: {msg.header.frame_id}")
    
    def export_to_json(self, filename: str = None):
        """Export the latest temporal map to JSON format."""
        if self.latest_map is None:
            rospy.logwarn("No temporal map data available for export")
            return None
            
        if filename is None:
            timestamp = rospy.Time.now().to_sec()
            filename = f"aion_temporal_map_{timestamp:.0f}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert ROS message to serializable dictionary
        data = {
            'header': {
                'stamp': self.latest_map.header.stamp.to_sec(),
                'frame_id': self.latest_map.header.frame_id
            },
            'metadata': {
                'update_interval_seconds': self.latest_map.update_interval_seconds,
                'num_orientation_bins': self.latest_map.num_orientation_bins,
                'total_places': self.latest_map.total_places,
                'bound_places': self.latest_map.bound_places,
                'hash_places': self.latest_map.hash_places
            },
            'temporal_nodes': [],
            'historical_nodes': []
        }
        
        # Convert temporal nodes
        for node in self.latest_map.temporal_nodes:
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
                'prediction_time': int(node.prediction_time),
                'is_bound_to_node': bool(node.is_bound_to_node),
                'spatial_hash': int(node.spatial_hash)
            }
            data['temporal_nodes'].append(node_data)
        
        # Convert historical nodes
        for node in self.latest_map.historical_nodes:
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
                'normalized_counts': [int(c) for c in node.normalized_counts],
                'total_observations': int(node.total_observations),
                'is_bound_to_node': bool(node.is_bound_to_node)
            }
            data['historical_nodes'].append(node_data)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        rospy.loginfo(f"Exported temporal map to {filepath}")
        return filepath
    
    def export_to_numpy(self, grid_resolution: float = 1.0, 
                       grid_bounds: tuple = (-50, 50, -50, 50)):
        """
        Export temporal map to numpy arrays on a spatial grid.
        
        Args:
            grid_resolution: Grid cell size in meters
            grid_bounds: (x_min, x_max, y_min, y_max) in meters
        """
        if self.latest_map is None:
            rospy.logwarn("No temporal map data available for export")
            return None
        
        x_min, x_max, y_min, y_max = grid_bounds
        grid_width = int((x_max - x_min) / grid_resolution)
        grid_height = int((y_max - y_min) / grid_resolution)
        
        timestamp = rospy.Time.now().to_sec()
        saved_files = []
        
        # Process both temporal and historical data
        for data_type, nodes in [('temporal', self.latest_map.temporal_nodes), 
                                ('historical', self.latest_map.historical_nodes)]:
            if not nodes:
                continue
            
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
            for map_type, grid in [('entropy', entropy_grid), 
                                  ('flow', flow_grid), 
                                  ('direction', direction_grid)]:
                filename = f"aion_{data_type}_{map_type}_{timestamp:.0f}.npy"
                filepath = os.path.join(self.output_dir, filename)
                np.save(filepath, grid)
                saved_files.append(filepath)
                rospy.loginfo(f"Saved {data_type} {map_type} grid to {filepath}")
        
        # Save metadata
        metadata = {
            'grid_resolution': grid_resolution,
            'grid_bounds': grid_bounds,
            'grid_shape': [grid_height, grid_width],
            'origin': [x_min, y_min, 0.0],
            'timestamp': timestamp,
            'frame_id': self.latest_map.header.frame_id,
            'num_orientation_bins': self.latest_map.num_orientation_bins
        }
        
        metadata_file = os.path.join(self.output_dir, f"aion_metadata_{timestamp:.0f}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files.append(metadata_file)
        
        return saved_files
    
    def export_to_csv(self, filename: str = None):
        """Export temporal map to CSV format for analysis."""
        if self.latest_map is None:
            rospy.logwarn("No temporal map data available for export")
            return None
        
        if filename is None:
            timestamp = rospy.Time.now().to_sec()
            filename = f"aion_temporal_map_{timestamp:.0f}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare CSV data
        csv_lines = []
        header = ["node_type", "place_id", "x", "y", "z", "entropy", "flow_magnitude", 
                 "best_angle_deg", "total_observations", "is_bound_to_node"]
        
        # Add orientation bin headers
        num_bins = self.latest_map.num_orientation_bins
        for i in range(num_bins):
            header.append(f"prob_bin_{i}")
        
        csv_lines.append(",".join(header))
        
        # Add temporal nodes
        for node in self.latest_map.temporal_nodes:
            row = [
                "temporal",
                str(node.place_id),
                f"{node.position.x:.3f}",
                f"{node.position.y:.3f}",
                f"{node.position.z:.3f}",
                f"{node.entropy:.4f}",
                f"{node.flow_magnitude:.4f}",
                f"{np.degrees(node.best_angle):.1f}",
                "0",  # temporal nodes don't have observation counts
                str(node.is_bound_to_node)
            ]
            # Add probability values
            for prob in node.probabilities:
                row.append(f"{prob:.4f}")
            csv_lines.append(",".join(row))
        
        # Add historical nodes
        for node in self.latest_map.historical_nodes:
            row = [
                "historical",
                str(node.place_id),
                f"{node.position.x:.3f}",
                f"{node.position.y:.3f}",
                f"{node.position.z:.3f}",
                f"{node.entropy:.4f}",
                f"{node.flow_magnitude:.4f}",
                f"{np.degrees(node.best_angle):.1f}",
                str(node.total_observations),
                str(node.is_bound_to_node)
            ]
            # Add probability values
            for prob in node.probabilities:
                row.append(f"{prob:.4f}")
            csv_lines.append(",".join(row))
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write("\n".join(csv_lines))
        
        rospy.loginfo(f"Exported temporal map to {filepath}")
        return filepath
    
    def save_current_map(self):
        """Save the current map in the specified format."""
        if self.format_type == 'json':
            return self.export_to_json()
        elif self.format_type == 'numpy':
            return self.export_to_numpy()
        elif self.format_type == 'csv':
            return self.export_to_csv()
        else:
            rospy.logerr(f"Unknown format: {self.format_type}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Export Aion temporal maps for benchmarking')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--format', '-f', choices=['json', 'numpy', 'csv'], default='json',
                       help='Export format')
    parser.add_argument('--grid_resolution', type=float, default=0.3,
                       help='Grid resolution for numpy export (meters)')
    parser.add_argument('--grid_bounds', nargs=4, type=float, default=[-15, 15, -10, 20],
                       metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'),
                       help='Grid bounds for numpy export (meters)')
    parser.add_argument('--auto_save', action='store_true',
                       help='Automatically save when new map is received')
    parser.add_argument('--save_interval', type=float, default=30.0,
                       help='Auto-save interval in seconds')
    parser.add_argument('--temporal_map_topic', type=str, default=None,
                       help='Aion temporal map topic (default: /temporal_dynamics_node/temporal_map_topic)')
    parser.add_argument('--wait_timeout', type=float, default=10.0,
                       help='Timeout in seconds for one-shot mode')
    
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('aion_map_exporter', anonymous=True)

    temporal_map_topic = args.temporal_map_topic or rospy.get_param(
        '/temporal_dynamics_node/temporal_map_topic',
        '/aion/temporal_map'
    )
    
    # Create exporter
    exporter = AionMapExporter(args.output, args.format)
    
    if args.auto_save:
        # Auto-save mode with timer - need subscriber for continuous updates
        exporter.map_subscriber = rospy.Subscriber(
            temporal_map_topic,
            AionTemporalMap, 
            exporter.temporal_map_callback,
            queue_size=1
        )
        
        def save_timer_callback(event):
            saved_files = exporter.save_current_map()
            if saved_files:
                rospy.loginfo(f"Auto-saved map: {saved_files}")
        
        timer = rospy.Timer(rospy.Duration(args.save_interval), save_timer_callback)
        rospy.loginfo(f"Auto-save enabled with {args.save_interval}s interval")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            timer.shutdown()
    else:
        # Wait for data and save once
        rospy.loginfo(f"Waiting for temporal map data on '{temporal_map_topic}'...")
        
        try:
            # Use rospy.wait_for_message for reliable message waiting
            msg = rospy.wait_for_message(
                temporal_map_topic,
                AionTemporalMap,
                timeout=args.wait_timeout,
            )
            exporter.latest_map = msg
            rospy.loginfo(f"Received temporal map with {len(msg.temporal_nodes)} temporal nodes "
                         f"and {len(msg.historical_nodes)} historical nodes")
        except rospy.ROSException as e:
            rospy.logwarn(f"Failed to receive temporal map: {e}")
            exporter.latest_map = None
        
        if exporter.latest_map is not None:
            rospy.loginfo("Temporal map data received, starting export...")
            if args.format == 'numpy':
                saved_files = exporter.export_to_numpy(args.grid_resolution, args.grid_bounds)
            else:
                saved_files = exporter.save_current_map()
            
            if saved_files:
                rospy.loginfo(f"Successfully exported map: {saved_files}")
            else:
                rospy.logwarn("Failed to export map")
        else:
            rospy.logwarn(f"No map data received after {args.wait_timeout:.1f} seconds")


if __name__ == '__main__':
    main()
