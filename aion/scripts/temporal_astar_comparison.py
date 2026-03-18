#!/usr/bin/env python3

"""
Temporal-Aware A* Pathfinding for Aion

This script compares standard A* pathfinding with temporal-aware A* that considers
Aion's temporal dynamics to avoid crowded or unpredictable areas.

Usage:
    # Export data first
    rosservice call /aion/export_navigation_data "output_directory: '/tmp/aion_path_planning' \
                                                 filename_prefix: 'aion_path' \
                                                 include_connectivity: true \
                                                 include_temporal_data: true \
                                                 prediction_time: 0.0"

    # Then run comparison
    python3 temporal_astar_comparison.py \
        --navigation /tmp/aion_path_planning/aion_path_navigation.json \
        --temporal /tmp/aion_path_planning/aion_path_temporal.json \
        --start-coords 4.5 0.0 0.0 \
        --goal-coords 2.0 2.0 0.0 \
        --save-paths /tmp/aion_path_planning/aion_computed_paths.json
"""

import argparse
import json
import numpy as np
import heapq
import math
from typing import Dict, List, Tuple, Optional, Set
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Node:
    """Represents a navigation node with position and temporal properties."""
    id: int
    x: float
    y: float
    z: float
    # Temporal properties (if available)
    entropy: Optional[float] = None
    flow_magnitude: Optional[float] = None
    best_angle: Optional[float] = None
    temporal_cost: Optional[float] = None

@dataclass
class Edge:
    """Represents an edge between two nodes."""
    source: int
    target: int
    weight: float

@dataclass
class AStarResult:
    """Result of A* pathfinding."""
    path: List[int]
    cost: float
    nodes_explored: int
    computation_time: float

class TemporalAStarComparison:
    """Compare standard A* with temporal-aware A* pathfinding."""
    
    def __init__(self, navigation_file: str, temporal_file: str = None):
        """
        Initialize the comparison with navigation and temporal data.
        
        Args:
            navigation_file: JSON file with Hydra navigation layer data
            temporal_file: JSON file with Aion temporal dynamics (optional)
        """
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        self.load_navigation_data(navigation_file)
        if temporal_file:
            self.load_temporal_data(temporal_file)
        
        print(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")
        
    def load_navigation_data(self, filename: str):
        """Load Hydra navigation layer data."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Load nodes
        for node_data in data['nodes']:
            node = Node(
                id=node_data['id'],
                x=node_data['position']['x'],
                y=node_data['position']['y'], 
                z=node_data['position']['z']
            )
            self.nodes[node.id] = node
        
        # Load edges  
        for edge_data in data['edges']:
            edge = Edge(
                source=edge_data['source'],
                target=edge_data['target'],
                weight=edge_data['weight']
            )
            self.edges.append(edge)
            
            # Build adjacency list
            self.adjacency[edge.source].append((edge.target, edge.weight))
            self.adjacency[edge.target].append((edge.source, edge.weight))  # Assume undirected
        
        print(f"Loaded navigation layer: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def load_temporal_data(self, filename: str):
        """Load Aion temporal dynamics data."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        temporal_places = data.get('temporal_places', [])
        matched_count = 0
        total_places = len(temporal_places)
        
        print(f"Processing {total_places} temporal places...")
        
        # Reduced debug output for cleaner operation
        debug_count = 0
        
        for i, place in enumerate(temporal_places):
            place_position = place['position']
            place_x = place_position['x']
            place_y = place_position['y'] 
            place_z = place_position['z']
            
            # Debug: Check temporal data availability
            has_predictions = 'predictions' in place and place['predictions']
            has_historical = 'historical' in place
            entropy_val = None
            flow_val = None
            
            if has_predictions:
                entropy_val = place['predictions'].get('entropy', 0.0)
                flow_val = place['predictions'].get('flow_magnitude', 0.0)
            elif has_historical:
                entropy_val = place['historical'].get('entropy', 0.0) 
                flow_val = place['historical'].get('flow_magnitude', 0.0)
            
            # Show first few interesting places
            if (debug_count < 3 and (entropy_val and entropy_val > 0.1)) or (flow_val and flow_val > 10):
                print(f"  Place {i}: pos=({place_x:.1f},{place_y:.1f},{place_z:.1f}), entropy={entropy_val:.3f}, flow={flow_val:.1f}")
                debug_count += 1
            
            # Find nearest navigation node to this temporal place
            nearest_node = None
            min_distance = float('inf')
            
            for node_id, node in self.nodes.items():
                distance = math.sqrt(
                    (node.x - place_x) ** 2 +
                    (node.y - place_y) ** 2 +
                    (node.z - place_z) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node
            
            # Only match if distance is reasonable (within 3 meters for more coverage)
            if nearest_node and min_distance < 3.0:
                # Debug: Show successful matches (reduced output)
                show_debug = debug_count < 3 and (entropy_val > 0.1 or flow_val > 10)
                # Use historical or prediction data based on availability
                if has_predictions:
                    pred = place['predictions']
                    new_entropy = pred['entropy']
                    new_flow_magnitude = pred['flow_magnitude']
                    new_best_angle = pred['best_angle']
                elif has_historical:
                    hist = place['historical'] 
                    new_entropy = hist['entropy']
                    new_flow_magnitude = hist['flow_magnitude']
                    new_best_angle = hist['best_angle']
                else:
                    new_entropy = 0.0
                    new_flow_magnitude = 0.0
                    new_best_angle = 0.0
                
                # Compute  temporal cost for dramatic path differences
                entropy_cost = min(new_entropy * 3.0, 10.0)  # Much more aggressive - up to 10x path cost!
                flow_cost = min(new_flow_magnitude * 0.5, 5.0) if new_flow_magnitude > 0 else 0.0
                new_temporal_cost = entropy_cost + flow_cost
                
                # Use maximum temporal cost if node already has data
                if nearest_node.entropy is None or new_temporal_cost > nearest_node.temporal_cost:
                    nearest_node.entropy = new_entropy
                    nearest_node.flow_magnitude = new_flow_magnitude
                    nearest_node.best_angle = new_best_angle
                    nearest_node.temporal_cost = new_temporal_cost
                    
                    if show_debug:
                        print(f"    -> Matched to node {nearest_node.id}, temporal_cost={new_temporal_cost:.3f}")
                        
                matched_count += 1
        
        print(f"Loaded temporal data for {matched_count}/{len(self.nodes)} nodes")
        
        # Debug: Show some example temporal costs
        temporal_nodes = [n for n in self.nodes.values() if hasattr(n, 'entropy') and n.entropy is not None]
        if temporal_nodes:
            print(f"Sample temporal costs (first 10):")
            for i, node in enumerate(temporal_nodes[:10]):
                print(f"  Node {node.id}: entropy={node.entropy:.3f}, flow={node.flow_magnitude:.1f}, temporal_cost={node.temporal_cost:.3f}")
            
            # Show statistics
            entropies = [n.entropy for n in temporal_nodes if n.entropy is not None]
            flows = [n.flow_magnitude for n in temporal_nodes if n.flow_magnitude is not None]
            if entropies:
                print(f"Entropy range: {min(entropies):.3f} to {max(entropies):.3f}")
            if flows:
                print(f"Flow range: {min(flows):.1f} to {max(flows):.1f}")
        else:
            print("No temporal costs calculated!")
    
    def find_nearest_node(self, x: float, y: float, z: float = 0.0) -> Optional[int]:
        """Find the nearest navigation node to given coordinates."""
        if not self.nodes:
            return None
            
        best_node = None
        best_distance = float('inf')
        
        for node_id, node in self.nodes.items():
            distance = math.sqrt((node.x - x)**2 + (node.y - y)**2 + (node.z - z)**2)
            if distance < best_distance:
                best_distance = distance
                best_node = node_id
                
        print(f"Nearest node to ({x:.2f}, {y:.2f}, {z:.2f}): {best_node} (distance: {best_distance:.2f}m)")
        return best_node
    
    def compute_directional_cost(self, from_node_id: int, to_node_id: int) -> float:
        """Compute additional cost based on movement direction vs. flow direction."""
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        
        # If no flow direction data, no additional cost
        if from_node.best_angle is None and to_node.best_angle is None:
            return 0.0
            
        # Calculate movement direction
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        movement_angle = math.atan2(dy, dx)
        
        # Calculate flow alignment penalty
        max_penalty = 0.0
        
        for node, weight in [(from_node, 0.7), (to_node, 0.3)]:  # Weight source more
            if node.best_angle is not None:
                # Angle difference between movement and flow
                angle_diff = abs(movement_angle - node.best_angle)
                # Normalize to [0, π]
                angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                
                # Convert to penalty: 0 (aligned) to 1 (opposite)
                alignment_penalty = angle_diff / math.pi
                
                # Scale by flow magnitude (stronger flow = higher penalty when misaligned)
                if node.flow_magnitude is not None:
                    flow_strength = min(node.flow_magnitude / 10.0, 1.0)
                    penalty = alignment_penalty * flow_strength * weight
                    max_penalty = max(max_penalty, penalty)
        
        return max_penalty
    
    def heuristic(self, node1_id: int, node2_id: int) -> float:
        """Euclidean distance heuristic."""
        n1 = self.nodes[node1_id]
        n2 = self.nodes[node2_id]
        return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2 + (n1.z - n2.z)**2)
    
    def standard_astar(self, start_id: int, goal_id: int) -> AStarResult:
        """Standard A* using only geometric distance."""
        return self._astar(start_id, goal_id, use_temporal=False)
    
    def temporal_astar(self, start_id: int, goal_id: int) -> AStarResult:
        """A* with automatic temporal cost calculation based on entropy, flow, and direction."""
        return self._astar(start_id, goal_id, use_temporal=True)
    
    def _astar(self, start_id: int, goal_id: int, use_temporal: bool = False) -> AStarResult:
        """
        Generic A* implementation.
        
        Args:
            start_id: Starting node ID
            goal_id: Goal node ID
            use_temporal: Whether to include temporal costs (automatically computed)
        """
        import time
        start_time = time.time()
        
        if start_id not in self.nodes or goal_id not in self.nodes:
            return AStarResult(path=[], cost=float('inf'), nodes_explored=0, 
                             computation_time=time.time() - start_time)
        
        # Priority queue: (f_score, node_id)
        open_set = [(0.0, start_id)]
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_id] = 0.0
        nodes_explored = 0
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            nodes_explored += 1
            
            if current == goal_id:
                # Reconstruct path
                path = []
                node = goal_id
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start_id)
                path.reverse()
                
                return AStarResult(
                    path=path,
                    cost=g_score[goal_id],
                    nodes_explored=nodes_explored,
                    computation_time=time.time() - start_time
                )
            
            for neighbor_id, edge_weight in self.adjacency[current]:
                # Base geometric cost
                edge_cost = edge_weight
                
                # Add temporal costs if enabled
                if use_temporal:
                    # Node-based temporal cost (entropy + flow magnitude)
                    current_node = self.nodes[current]
                    neighbor_node = self.nodes[neighbor_id]
                    
                    # Average temporal cost of both nodes
                    node_temporal_cost = 0.0
                    if current_node.temporal_cost is not None:
                        node_temporal_cost += current_node.temporal_cost * 0.5
                    if neighbor_node.temporal_cost is not None:
                        node_temporal_cost += neighbor_node.temporal_cost * 0.5
                    
                    # Directional cost (flow direction vs movement direction)
                    directional_cost = self.compute_directional_cost(current, neighbor_id)
                    
                    # Total AGGRESSIVE temporal penalty (scaled by 3x for dramatic differences)
                    total_temporal_cost = (node_temporal_cost + directional_cost) * edge_weight * 3.0
                    edge_cost += total_temporal_cost
                    
                    # Additional penalty for high entropy nodes - make them REALLY expensive
                    if current_node.entropy is not None and current_node.entropy > 0.5:
                        edge_cost += current_node.entropy * 2.0  # Extra penalty
                    if neighbor_node.entropy is not None and neighbor_node.entropy > 0.5:
                        edge_cost += neighbor_node.entropy * 2.0  # Extra penalty
                
                tentative_g = g_score[current] + edge_cost
                
                if tentative_g < g_score[neighbor_id]:
                    came_from[neighbor_id] = current
                    g_score[neighbor_id] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor_id, goal_id)
                    heapq.heappush(open_set, (f_score, neighbor_id))
        
        # No path found
        return AStarResult(path=[], cost=float('inf'), nodes_explored=nodes_explored,
                         computation_time=time.time() - start_time)
    
    def compare_paths(self, start_id: int, goal_id: int) -> Dict:
        """
        Compare standard A* with temporal-aware A*.
        
        Returns:
            Dictionary with comparison results
        """
        print(f"Comparing paths from {start_id} to {goal_id}")
        
        # Run both algorithms
        standard_result = self.standard_astar(start_id, goal_id)
        temporal_result = self.temporal_astar(start_id, goal_id)
        
        # Calculate metrics
        comparison = {
            'start_id': start_id,
            'goal_id': goal_id,
            'standard': {
                'path': standard_result.path,
                'cost': standard_result.cost,
                'path_length': len(standard_result.path),
                'nodes_explored': standard_result.nodes_explored,
                'computation_time': standard_result.computation_time
            },
            'temporal': {
                'path': temporal_result.path,
                'cost': temporal_result.cost, 
                'path_length': len(temporal_result.path),
                'nodes_explored': temporal_result.nodes_explored,
                'computation_time': temporal_result.computation_time
            }
        }
        
        # Calculate differences
        if standard_result.path and temporal_result.path:
            path_overlap = len(set(standard_result.path) & set(temporal_result.path))
            comparison['path_similarity'] = path_overlap / max(len(standard_result.path), len(temporal_result.path))
            comparison['cost_difference'] = temporal_result.cost - standard_result.cost
            comparison['temporal_avoidance'] = self._calculate_temporal_avoidance(temporal_result.path)
        else:
            comparison['path_similarity'] = 0.0
            comparison['cost_difference'] = 0.0
            comparison['temporal_avoidance'] = 0.0
        
        return comparison
    
    def _calculate_temporal_avoidance(self, path: List[int]) -> float:
        """Calculate how much temporal dynamics the path avoids."""
        if not path:
            return 0.0
        
        total_temporal_cost = 0.0
        valid_nodes = 0
        
        for node_id in path:
            node = self.nodes[node_id]
            if node.temporal_cost is not None:
                total_temporal_cost += node.temporal_cost
                valid_nodes += 1
        
        return total_temporal_cost / max(valid_nodes, 1)
    
    def visualize_comparison(self, comparison: Dict, save_path: str = None):
        """Visualize the path comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Extract node positions
        x_coords = [n.x for n in self.nodes.values()]
        y_coords = [n.y for n in self.nodes.values()]
        
        for ax, title, path_key in [(ax1, "Standard A*", "standard"), (ax2, "Temporal-Aware A*", "temporal")]:
            # Draw all nodes
            ax.scatter(x_coords, y_coords, c='lightgray', s=20, alpha=0.5, label='Navigation nodes')
            
            # Color nodes by temporal cost if available
            temporal_nodes = [n for n in self.nodes.values() if n.temporal_cost is not None]
            if temporal_nodes:
                tx = [n.x for n in temporal_nodes]
                ty = [n.y for n in temporal_nodes] 
                tc = [n.temporal_cost for n in temporal_nodes]
                scatter = ax.scatter(tx, ty, c=tc, s=40, cmap='Reds', alpha=0.7, label='Temporal nodes')
                if ax == ax1:  # Only add colorbar once
                    plt.colorbar(scatter, ax=ax1, label='Temporal Cost')
            
            # Draw edges (sample for visibility)
            edge_sample = self.edges[::max(1, len(self.edges)//100)]  # Sample edges
            for edge in edge_sample:
                if edge.source in self.nodes and edge.target in self.nodes:
                    n1, n2 = self.nodes[edge.source], self.nodes[edge.target]
                    ax.plot([n1.x, n2.x], [n1.y, n2.y], 'k-', alpha=0.1, linewidth=0.5)
            
            # Draw path
            path = comparison[path_key]['path']
            if path:
                path_x = [self.nodes[nid].x for nid in path]
                path_y = [self.nodes[nid].y for nid in path]
                ax.plot(path_x, path_y, 'b-', linewidth=3, label=f'Path (cost: {comparison[path_key]["cost"]:.2f})')
                ax.scatter(path_x[0], path_y[0], c='green', s=100, marker='o', label='Start')
                ax.scatter(path_x[-1], path_y[-1], c='red', s=100, marker='s', label='Goal')
            
            ax.set_title(f"{title}\nCost: {comparison[path_key]['cost']:.2f}, "
                        f"Length: {comparison[path_key]['path_length']}, "
                        f"Explored: {comparison[path_key]['nodes_explored']}")
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def run_multiple_comparisons(self, node_pairs: List[Tuple[int, int]]) -> List[Dict]:
        """Run comparisons for multiple node pairs."""
        results = []
        
        for start_id, goal_id in node_pairs:
            try:
                comparison = self.compare_paths(start_id, goal_id)
                results.append(comparison)
                
                print(f"Start: {start_id}, Goal: {goal_id}")
                print(f"  Standard cost: {comparison['standard']['cost']:.2f}")
                print(f"  Temporal cost: {comparison['temporal']['cost']:.2f}")
                print(f"  Cost difference: {comparison['cost_difference']:.2f}")
                print(f"  Path similarity: {comparison['path_similarity']:.2f}")
                print(f"  Temporal avoidance: {comparison['temporal_avoidance']:.2f}")
                print()
                
            except Exception as e:
                print(f"Error comparing {start_id} -> {goal_id}: {e}")
                    
        return results

def main():
    parser = argparse.ArgumentParser(description='Compare standard and temporal-aware A* pathfinding')
    parser.add_argument('--navigation', required=True, help='Navigation layer JSON file')
    parser.add_argument('--temporal', help='Temporal dynamics JSON file')
    parser.add_argument('--start-id', type=int, help='Start node ID')
    parser.add_argument('--goal-id', type=int, help='Goal node ID')
    parser.add_argument('--start-coords', nargs=3, type=float, metavar=('X', 'Y', 'Z'), help='Start coordinates (x y z)')
    parser.add_argument('--goal-coords', nargs=3, type=float, metavar=('X', 'Y', 'Z'), help='Goal coordinates (x y z)')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--save-plot', help='Save plot to file')
    parser.add_argument('--save-paths', help='Save path results to JSON file')
    parser.add_argument('--multiple', action='store_true', help='Run multiple random comparisons')
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparison_tool = TemporalAStarComparison(args.navigation, args.temporal)
    
    # Determine start and goal nodes
    start_id = args.start_id
    goal_id = args.goal_id
    
    # Convert coordinates to node IDs if provided
    if args.start_coords:
        start_id = comparison_tool.find_nearest_node(*args.start_coords)
        if start_id is None:
            print("Could not find start node from coordinates")
            return
            
    if args.goal_coords:
        goal_id = comparison_tool.find_nearest_node(*args.goal_coords)
        if goal_id is None:
            print("Could not find goal node from coordinates")
            return
    
    if args.multiple or (start_id is None or goal_id is None):
        # Run multiple random comparisons
        node_ids = list(comparison_tool.nodes.keys())
        if len(node_ids) < 2:
            print("Not enough nodes for comparison")
            return
            
        # Select random node pairs
        np.random.seed(42)  # For reproducibility
        num_comparisons = min(5, len(node_ids) // 2)
        node_pairs = []
        
        for _ in range(num_comparisons):
            start, goal = np.random.choice(node_ids, 2, replace=False)
            node_pairs.append((start, goal))
        
        print(f"Running {num_comparisons} random comparisons...")
        results = comparison_tool.run_multiple_comparisons(node_pairs)
        
        # Print summary
        if results:
            avg_cost_diff = np.mean([r['cost_difference'] for r in results])
            avg_similarity = np.mean([r['path_similarity'] for r in results])
            avg_avoidance = np.mean([r['temporal_avoidance'] for r in results])
            
            print("SUMMARY:")
            print(f"Average cost difference: {avg_cost_diff:.2f}")
            print(f"Average path similarity: {avg_similarity:.2f}")
            print(f"Average temporal avoidance: {avg_avoidance:.2f}")
    
    else:
        # Single comparison
        print(f"Comparing paths: {start_id} -> {goal_id}")
        comparison = comparison_tool.compare_paths(start_id, goal_id)
        
        print("COMPARISON RESULTS:")
        print(json.dumps(comparison, indent=2))
        
        # Save paths to JSON if requested
        if args.save_paths:
            with open(args.save_paths, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"Paths saved to {args.save_paths}")
        
        if args.visualize or args.save_plot:
            comparison_tool.visualize_comparison(comparison, args.save_plot)

if __name__ == '__main__':
    main()
