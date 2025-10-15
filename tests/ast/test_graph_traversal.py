"""Tests for graph traversal algorithms."""

import pytest

from app.ast.graph import CodeGraph


@pytest.fixture
def simple_graph():
    """Create a simple graph for testing.

    Structure:
        A (outer) -> B -> C (inner)
        D (outer) -> C
    """
    nodes = [
        {"id": "A", "type": "function", "name": "func_a", "file_path": "test.py"},
        {"id": "B", "type": "function", "name": "func_b", "file_path": "test.py"},
        {"id": "C", "type": "function", "name": "func_c", "file_path": "test.py"},
        {"id": "D", "type": "function", "name": "func_d", "file_path": "test.py"},
    ]
    edges = [
        {"from": "A", "to": "B", "type": "calls"},
        {"from": "B", "to": "C", "type": "calls"},
        {"from": "D", "to": "C", "type": "calls"},
    ]
    return CodeGraph(nodes=nodes, edges=edges)


@pytest.fixture
def complex_graph():
    """Create a more complex graph.

    Structure:
        A -> B -> D
        A -> C -> D
        E (isolated)
    """
    nodes = [
        {"id": "A", "type": "function", "name": "main", "file_path": "main.py"},
        {"id": "B", "type": "function", "name": "helper1", "file_path": "utils.py"},
        {"id": "C", "type": "function", "name": "helper2", "file_path": "utils.py"},
        {"id": "D", "type": "function", "name": "core", "file_path": "core.py"},
        {"id": "E", "type": "function", "name": "isolated", "file_path": "other.py"},
    ]
    edges = [
        {"from": "A", "to": "B", "type": "calls"},
        {"from": "A", "to": "C", "type": "calls"},
        {"from": "B", "to": "D", "type": "calls"},
        {"from": "C", "to": "D", "type": "calls"},
    ]
    return CodeGraph(nodes=nodes, edges=edges)


def test_calculate_in_degrees_simple(simple_graph: CodeGraph):
    """Test in-degree calculation for simple graph."""
    in_degrees = simple_graph.calculate_in_degrees()

    assert in_degrees["A"] == 0  # Outer node
    assert in_degrees["B"] == 1  # Called by A
    assert in_degrees["C"] == 2  # Called by B and D
    assert in_degrees["D"] == 0  # Outer node


def test_calculate_in_degrees_complex(complex_graph: CodeGraph):
    """Test in-degree calculation for complex graph."""
    in_degrees = complex_graph.calculate_in_degrees()

    assert in_degrees["A"] == 0  # Outer node
    assert in_degrees["B"] == 1  # Called by A
    assert in_degrees["C"] == 1  # Called by A
    assert in_degrees["D"] == 2  # Called by B and C
    assert in_degrees["E"] == 0  # Isolated node


def test_get_outer_nodes_simple(simple_graph: CodeGraph):
    """Test getting outer nodes from simple graph."""
    outer_nodes = simple_graph.get_outer_nodes()
    outer_ids = [node["id"] for node in outer_nodes]

    assert len(outer_nodes) == 2
    assert "A" in outer_ids
    assert "D" in outer_ids


def test_get_outer_nodes_complex(complex_graph: CodeGraph):
    """Test getting outer nodes from complex graph."""
    outer_nodes = complex_graph.get_outer_nodes()
    outer_ids = [node["id"] for node in outer_nodes]

    assert len(outer_nodes) == 2
    assert "A" in outer_ids
    assert "E" in outer_ids


def test_get_inner_nodes(simple_graph: CodeGraph):
    """Test getting inner nodes (high in-degree)."""
    inner_nodes = simple_graph.get_inner_nodes(threshold=2)

    assert len(inner_nodes) == 1
    assert inner_nodes[0]["id"] == "C"
    assert inner_nodes[0]["in_degree"] == 2


def test_get_inner_nodes_with_threshold(complex_graph: CodeGraph):
    """Test getting inner nodes with custom threshold."""
    # Threshold 2: only D qualifies
    inner_nodes = complex_graph.get_inner_nodes(threshold=2)
    assert len(inner_nodes) == 1
    assert inner_nodes[0]["id"] == "D"
    assert inner_nodes[0]["in_degree"] == 2

    # Threshold 1: B, C, D qualify
    inner_nodes = complex_graph.get_inner_nodes(threshold=1)
    assert len(inner_nodes) == 3
    inner_ids = [node["id"] for node in inner_nodes]
    assert "B" in inner_ids
    assert "C" in inner_ids
    assert "D" in inner_ids


def test_topological_sort_simple(simple_graph: CodeGraph):
    """Test topological sort on simple graph."""
    sorted_nodes = simple_graph.topological_sort()
    sorted_ids = [node["id"] for node in sorted_nodes]

    # A and D should come before B
    assert sorted_ids.index("A") < sorted_ids.index("B")
    assert sorted_ids.index("D") < sorted_ids.index("C")
    # B should come before C
    assert sorted_ids.index("B") < sorted_ids.index("C")


def test_topological_sort_complex(complex_graph: CodeGraph):
    """Test topological sort on complex graph."""
    sorted_nodes = complex_graph.topological_sort()
    sorted_ids = [node["id"] for node in sorted_nodes]

    # A should come first
    assert sorted_ids.index("A") < sorted_ids.index("B")
    assert sorted_ids.index("A") < sorted_ids.index("C")

    # B and C should come before D
    assert sorted_ids.index("B") < sorted_ids.index("D")
    assert sorted_ids.index("C") < sorted_ids.index("D")

    # E can be anywhere (isolated)
    assert "E" in sorted_ids


def test_get_nodes_by_depth_simple(simple_graph: CodeGraph):
    """Test grouping nodes by depth level."""
    levels = simple_graph.get_nodes_by_depth()

    # Level 0: A and D (outer nodes)
    level_0_ids = [node["id"] for node in levels[0]]
    assert len(level_0_ids) == 2
    assert "A" in level_0_ids
    assert "D" in level_0_ids

    # Level 1: B
    level_1_ids = [node["id"] for node in levels[1]]
    assert "B" in level_1_ids

    # Level 2: C
    level_2_ids = [node["id"] for node in levels[2]]
    assert "C" in level_2_ids


def test_get_nodes_by_depth_complex(complex_graph: CodeGraph):
    """Test depth grouping on complex graph."""
    levels = complex_graph.get_nodes_by_depth()

    # Level 0: A and E (outer nodes)
    level_0_ids = [node["id"] for node in levels[0]]
    assert "A" in level_0_ids
    assert "E" in level_0_ids

    # Level 1: B and C
    level_1_ids = [node["id"] for node in levels[1]]
    assert "B" in level_1_ids
    assert "C" in level_1_ids

    # Level 2: D
    level_2_ids = [node["id"] for node in levels[2]]
    assert "D" in level_2_ids


def test_empty_graph():
    """Test with empty graph."""
    graph = CodeGraph(nodes=[], edges=[])

    assert graph.calculate_in_degrees() == {}
    assert graph.get_outer_nodes() == []
    assert graph.get_inner_nodes() == []
    assert graph.topological_sort() == []
    # Empty graph returns one empty level
    levels = graph.get_nodes_by_depth()
    assert len(levels) <= 1
    if levels:
        assert levels[0] == []


def test_single_node():
    """Test with single isolated node."""
    graph = CodeGraph(
        nodes=[
            {"id": "A", "type": "function", "name": "single", "file_path": "test.py"}
        ],
        edges=[],
    )

    in_degrees = graph.calculate_in_degrees()
    assert in_degrees["A"] == 0

    outer_nodes = graph.get_outer_nodes()
    assert len(outer_nodes) == 1
    assert outer_nodes[0]["id"] == "A"

    sorted_nodes = graph.topological_sort()
    assert len(sorted_nodes) == 1
    assert sorted_nodes[0]["id"] == "A"


def test_cyclic_graph():
    """Test that topological sort detects cycles."""
    # Create a cycle: A -> B -> C -> A
    nodes = [
        {"id": "A", "type": "function", "name": "func_a", "file_path": "test.py"},
        {"id": "B", "type": "function", "name": "func_b", "file_path": "test.py"},
        {"id": "C", "type": "function", "name": "func_c", "file_path": "test.py"},
    ]
    edges = [
        {"from": "A", "to": "B", "type": "calls"},
        {"from": "B", "to": "C", "type": "calls"},
        {"from": "C", "to": "A", "type": "calls"},  # Creates cycle
    ]
    graph = CodeGraph(nodes=nodes, edges=edges)

    with pytest.raises(ValueError, match="Graph contains cycles"):
        graph.topological_sort()
