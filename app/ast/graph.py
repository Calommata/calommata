"""Code graph data structure."""

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class CodeGraph:
    """Represents the complete code graph structure."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
        }

    def to_neo4j_format(self) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Convert to Neo4j-compatible format.

        Returns:
            Tuple of (node_queries, relationship_queries)
        """
        node_queries: list[dict[str, Any]] = []
        for node in self.nodes:
            node_data: dict[str, Any] = {
                "id": node["id"],
                "type": node["type"],
                "name": node.get("name"),
                "file_path": node["file_path"],
                "start_byte": node["byte_range"][0],
                "end_byte": node["byte_range"][1],
                "start_line": node["line_range"][0],
                "end_line": node["line_range"][1],
                "source_code": node["source_code"],
            }
            node_queries.append(node_data)

        relationship_queries: list[dict[str, str]] = []
        for edge in self.edges:
            rel_data = {
                "from_id": edge["from"],
                "to_id": edge["to"],
                "type": edge["type"],
            }
            relationship_queries.append(rel_data)

        return node_queries, relationship_queries

    def calculate_in_degrees(self) -> dict[str, int]:
        """Calculate in-degree (number of incoming edges) for each node.

        Returns:
            Dictionary mapping node IDs to their in-degrees
        """
        in_degrees: dict[str, int] = {}

        # Initialize all nodes with 0 in-degree
        for node in self.nodes:
            in_degrees[node["id"]] = 0

        # Count incoming edges
        for edge in self.edges:
            to_id = edge["to"]
            if to_id in in_degrees:
                in_degrees[to_id] += 1

        return in_degrees

    def topological_sort(self) -> list[dict[str, Any]]:
        """Perform topological sort using in-degree (Kahn's algorithm).

        Returns nodes in dependency order: outer nodes (no dependencies) first,
        inner nodes (many dependencies) last.

        Returns:
            List of nodes in topological order (outer to inner)

        Raises:
            ValueError: If graph contains cycles
        """
        in_degrees = self.calculate_in_degrees()
        node_map = {node["id"]: node for node in self.nodes}

        # Queue for nodes with in-degree 0 (outer nodes)
        queue: deque[str] = deque()
        for node_id, degree in in_degrees.items():
            if degree == 0:
                queue.append(node_id)

        result: list[dict[str, Any]] = []
        processed = 0

        # Build adjacency list for outgoing edges
        adjacency: dict[str, list[str]] = {node["id"]: [] for node in self.nodes}
        for edge in self.edges:
            from_id = edge["from"]
            to_id = edge["to"]
            if from_id in adjacency and to_id in node_map:
                adjacency[from_id].append(to_id)

        # Process nodes level by level
        while queue:
            node_id = queue.popleft()
            if node_id in node_map:
                result.append(node_map[node_id])
                processed += 1

                # Decrease in-degree of neighbors
                for neighbor_id in adjacency[node_id]:
                    in_degrees[neighbor_id] -= 1
                    if in_degrees[neighbor_id] == 0:
                        queue.append(neighbor_id)

        # Check for cycles
        if processed < len(self.nodes):
            cycle_nodes = [
                node_id for node_id, degree in in_degrees.items() if degree > 0
            ]
            raise ValueError(
                f"Graph contains cycles. Remaining nodes: {len(cycle_nodes)}"
            )

        return result

    def get_nodes_by_depth(self) -> list[list[dict[str, Any]]]:
        """Group nodes by their depth level (distance from outer nodes).

        Outer nodes (in-degree 0) are at level 0.
        Nodes depending only on outer nodes are at level 1, etc.

        Returns:
            List of node groups, where each group is a list of nodes at that depth
        """
        in_degrees = self.calculate_in_degrees()
        node_map = {node["id"]: node for node in self.nodes}

        # Queue with (node_id, depth) tuples
        queue: deque[tuple[str, int]] = deque()
        for node_id, degree in in_degrees.items():
            if degree == 0:
                queue.append((node_id, 0))

        levels: dict[int, list[dict[str, Any]]] = {}
        visited: set[str] = set()

        # Build adjacency list
        adjacency: dict[str, list[str]] = {node["id"]: [] for node in self.nodes}
        for edge in self.edges:
            from_id = edge["from"]
            to_id = edge["to"]
            if from_id in adjacency and to_id in node_map:
                adjacency[from_id].append(to_id)

        # BFS with depth tracking
        while queue:
            node_id, depth = queue.popleft()

            if node_id in visited or node_id not in node_map:
                continue

            visited.add(node_id)

            if depth not in levels:
                levels[depth] = []
            levels[depth].append(node_map[node_id])

            # Add neighbors with increased depth
            for neighbor_id in adjacency[node_id]:
                in_degrees[neighbor_id] -= 1
                if in_degrees[neighbor_id] == 0 and neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))

        # Convert to sorted list of levels
        max_depth = max(levels.keys()) if levels else 0
        return [levels.get(i, []) for i in range(max_depth + 1)]

    def get_outer_nodes(self) -> list[dict[str, Any]]:
        """Get outer nodes (nodes with in-degree 0).

        These are nodes that are not called or referenced by any other nodes.

        Returns:
            List of outer nodes
        """
        in_degrees = self.calculate_in_degrees()
        return [node for node in self.nodes if in_degrees[node["id"]] == 0]

    def get_inner_nodes(self, threshold: int = 2) -> list[dict[str, Any]]:
        """Get inner nodes (nodes with high in-degree).

        These are nodes that are called or referenced by many other nodes.

        Args:
            threshold: Minimum in-degree to be considered an inner node

        Returns:
            List of inner nodes with their in-degrees
        """
        in_degrees = self.calculate_in_degrees()
        inner_nodes: list[dict[str, Any]] = [
            {**node, "in_degree": in_degrees[node["id"]]}
            for node in self.nodes
            if in_degrees[node["id"]] >= threshold
        ]
        # Sort by in-degree descending
        return sorted(inner_nodes, key=lambda x: x["in_degree"], reverse=True)
