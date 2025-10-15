"""Tests for graph builder."""

import tempfile
from pathlib import Path

from app.ast.graph_builder import GraphBuilder, build_graph_from_files
from app.ast.models import LanguageType, ParsedNode, ParsedRelation


def test_graph_builder_basic():
    """Test basic graph builder functionality."""
    builder = GraphBuilder()

    # Create test nodes
    node1 = ParsedNode(
        id="file.py:0:10",
        type="function",
        name="func1",
        file_path="file.py",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=1,
        source_code="def func1(): pass",
        parent_id=None,
    )

    node2 = ParsedNode(
        id="file.py:11:20",
        type="function",
        name="func2",
        file_path="file.py",
        start_byte=11,
        end_byte=20,
        start_line=2,
        end_line=2,
        source_code="def func2(): pass",
        parent_id=None,
    )

    # Add nodes
    builder.add_nodes([node1, node2])

    graph = builder.build()

    assert len(graph.nodes) == 2, f"Should have 2 nodes, got {len(graph.nodes)}"
    assert len(graph.edges) == 0, f"Should have 0 edges, got {len(graph.edges)}"


def test_graph_builder_with_relations():
    """Test graph builder with relations."""
    builder = GraphBuilder()

    # Create test nodes
    node1 = ParsedNode(
        id="file.py:0:10",
        type="function",
        name="caller",
        file_path="file.py",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=1,
        source_code="def caller(): pass",
        parent_id=None,
    )

    node2 = ParsedNode(
        id="file.py:11:20",
        type="function",
        name="callee",
        file_path="file.py",
        start_byte=11,
        end_byte=20,
        start_line=2,
        end_line=2,
        source_code="def callee(): pass",
        parent_id=None,
    )

    # Create relation
    relation = ParsedRelation(
        from_id="file.py:0:10", to_id="file.py:11:20", relation_type="calls"
    )

    # Build graph
    builder.add_nodes([node1, node2])
    builder.add_relations([relation])

    graph = builder.build()

    assert len(graph.nodes) == 2, f"Should have 2 nodes, got {len(graph.nodes)}"
    assert len(graph.edges) == 1, f"Should have 1 edge, got {len(graph.edges)}"
    assert graph.edges[0]["type"] == "calls"
    assert graph.edges[0]["from"] == "file.py:0:10"
    assert graph.edges[0]["to"] == "file.py:11:20"


def test_build_graph_from_files():
    """Test building graph from actual files."""
    # Create temporary Python file
    py_source = """
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

def calculate():
    result = add(1, 2)
    return result
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(py_source)
        py_path = f.name

    # Create temporary JavaScript file
    js_source = """
function subtract(a, b) {
    return a - b;
}

class Math {
    divide(x, y) {
        return x / y;
    }
}
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8"
    ) as f:
        f.write(js_source)
        js_path = f.name

    try:
        # Build graph from files with proper typing
        file_list: list[tuple[str, LanguageType]] = [
            (py_path, "python"),
            (js_path, "javascript"),
        ]
        graph = build_graph_from_files(file_list)

        assert len(graph.nodes) > 0, "Should have nodes from parsed files"

        # Check that we have nodes from both files
        file_paths = {node["file_path"] for node in graph.nodes}
        assert len(file_paths) == 2, (
            f"Should have nodes from 2 files, got {len(file_paths)}"
        )

        # Check node types
        node_types = {node["type"] for node in graph.nodes}
        assert "function" in node_types or "class" in node_types, (
            "Should have function or class nodes"
        )

    finally:
        Path(py_path).unlink()
        Path(js_path).unlink()


def test_graph_to_dict():
    """Test graph serialization to dict."""
    builder = GraphBuilder()

    node = ParsedNode(
        id="test.py:0:10",
        type="function",
        name="test",
        file_path="test.py",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=1,
        source_code="def test(): pass",
        parent_id=None,
    )

    builder.add_nodes([node])
    graph = builder.build()

    # Convert to dict
    graph_dict = graph.to_dict()

    assert "nodes" in graph_dict
    assert "edges" in graph_dict
    assert len(graph_dict["nodes"]) == 1
    assert graph_dict["nodes"][0]["name"] == "test"


def test_graph_to_neo4j_format():
    """Test graph conversion to Neo4j format."""
    builder = GraphBuilder()

    node = ParsedNode(
        id="test.py:0:10",
        type="function",
        name="test",
        file_path="test.py",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=1,
        source_code="def test(): pass",
        parent_id=None,
    )

    relation = ParsedRelation(
        from_id="test.py:0:10", to_id="test.py:external:other", relation_type="calls"
    )

    builder.add_nodes([node])
    builder.add_relations([relation])
    graph = builder.build()

    # Convert to Neo4j format - returns tuple of (nodes, relationships)
    neo4j_nodes, neo4j_rels = graph.to_neo4j_format()

    assert len(neo4j_nodes) == 1
    assert len(neo4j_rels) == 1

    # Check Neo4j node format
    neo4j_node = neo4j_nodes[0]
    assert neo4j_node["type"] == "function"
    assert neo4j_node["name"] == "test"
    assert neo4j_node["id"] == "test.py:0:10"

    # Check Neo4j relationship format
    neo4j_rel = neo4j_rels[0]
    assert neo4j_rel["type"] == "calls"
    assert neo4j_rel["from_id"] == "test.py:0:10"
    assert neo4j_rel["to_id"] == "test.py:external:other"
