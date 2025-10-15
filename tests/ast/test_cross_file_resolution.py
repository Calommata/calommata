"""Tests for cross-file symbol resolution in graph building."""

import tempfile
from pathlib import Path

from app.ast.graph_builder import build_graph_from_files
from app.ast.models import LanguageType


def test_cross_file_function_call_resolution():
    """Test that function calls across files are properly resolved."""
    # Create utils.py with utility functions
    utils_source = """
def format_data(data):
    return str(data).upper()

def validate_input(value):
    return value is not None
"""

    # Create main.py that imports and calls utils functions
    main_source = """
from utils import format_data, validate_input

def process(name):
    if validate_input(name):
        return format_data(name)
    return None
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        utils_path = Path(tmpdir) / "utils.py"
        utils_path.write_text(utils_source, encoding="utf-8")

        main_path = Path(tmpdir) / "main.py"
        main_path.write_text(main_source, encoding="utf-8")

        # Build graph from both files
        files: list[tuple[str, LanguageType]] = [
            (str(utils_path), "python"),
            (str(main_path), "python"),
        ]
        graph = build_graph_from_files(files)

        # Calculate in-degrees
        in_degrees = graph.calculate_in_degrees()

        # Find the nodes
        format_data_node = next(
            (n for n in graph.nodes if n.get("name") == "format_data"), None
        )
        validate_input_node = next(
            (n for n in graph.nodes if n.get("name") == "validate_input"), None
        )
        process_node = next(
            (n for n in graph.nodes if n.get("name") == "process"), None
        )

        assert format_data_node is not None, "format_data function should be found"
        assert validate_input_node is not None, (
            "validate_input function should be found"
        )
        assert process_node is not None, "process function should be found"

        # CRITICAL: Check that cross-file calls are resolved
        # format_data and validate_input should have in-degree > 0
        # because process() calls them
        assert in_degrees[format_data_node["id"]] > 0, (
            f"format_data should be called by process, "
            f"but has in-degree {in_degrees[format_data_node['id']]}"
        )
        assert in_degrees[validate_input_node["id"]] > 0, (
            f"validate_input should be called by process, "
            f"but has in-degree {in_degrees[validate_input_node['id']]}"
        )

        # process should have in-degree 0 (entry point)
        assert in_degrees[process_node["id"]] == 0, (
            f"process should be an entry point, "
            f"but has in-degree {in_degrees[process_node['id']]}"
        )


def test_multiple_files_same_function_name():
    """Test resolution when multiple files have functions with the same name.

    Note: Current implementation takes the first match.
    This test documents the current behavior.
    """
    # File 1: utils1.py
    utils1_source = """
def helper():
    return "from utils1"
"""

    # File 2: utils2.py
    utils2_source = """
def helper():
    return "from utils2"
"""

    # File 3: main.py calls helper() - which one?
    main_source = """
from utils1 import helper

def main():
    return helper()
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        utils1_path = Path(tmpdir) / "utils1.py"
        utils1_path.write_text(utils1_source, encoding="utf-8")

        utils2_path = Path(tmpdir) / "utils2.py"
        utils2_path.write_text(utils2_source, encoding="utf-8")

        main_path = Path(tmpdir) / "main.py"
        main_path.write_text(main_source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [
            (str(utils1_path), "python"),
            (str(utils2_path), "python"),
            (str(main_path), "python"),
        ]
        graph = build_graph_from_files(files)

        # Find all helper nodes
        helper_nodes = [n for n in graph.nodes if n.get("name") == "helper"]
        assert len(helper_nodes) == 2, "Should have 2 helper functions"

        # Check that at least one helper has in-degree > 0
        in_degrees = graph.calculate_in_degrees()
        total_helper_indegree = sum(in_degrees.get(n["id"], 0) for n in helper_nodes)

        # Current behavior: first match in name_to_node dict wins
        # So ONE of the helpers should be called
        assert total_helper_indegree > 0, (
            "At least one helper should be called by main()"
        )


def test_cross_file_with_imports():
    """Test that import statements are tracked separately from function calls."""
    utils_source = """
def utility():
    pass
"""

    main_source = """
from utils import utility

def main():
    utility()
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        utils_path = Path(tmpdir) / "utils.py"
        utils_path.write_text(utils_source, encoding="utf-8")

        main_path = Path(tmpdir) / "main.py"
        main_path.write_text(main_source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [
            (str(utils_path), "python"),
            (str(main_path), "python"),
        ]
        graph = build_graph_from_files(files)

        # Should have both import node and function nodes
        node_types = {n["type"] for n in graph.nodes}
        assert "import" in node_types, "Should have import node"
        assert "function" in node_types, "Should have function nodes"

        # Should have both import relation and call relation
        edge_types = {e["type"] for e in graph.edges}
        assert "imports" in edge_types, "Should have import relation"
        assert "calls" in edge_types, "Should have call relation"


def test_three_level_cross_file_depth():
    """Test depth calculation across three files."""
    # Level 0: main.py (entry point)
    main_source = """
from middle import process

def main():
    process()
"""

    # Level 1: middle.py (called by main)
    middle_source = """
from utils import core_function

def process():
    core_function()
"""

    # Level 2: utils.py (called by middle)
    utils_source = """
def core_function():
    pass
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        main_path = Path(tmpdir) / "main.py"
        main_path.write_text(main_source, encoding="utf-8")

        middle_path = Path(tmpdir) / "middle.py"
        middle_path.write_text(middle_source, encoding="utf-8")

        utils_path = Path(tmpdir) / "utils.py"
        utils_path.write_text(utils_source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [
            (str(main_path), "python"),
            (str(middle_path), "python"),
            (str(utils_path), "python"),
        ]
        graph = build_graph_from_files(files)

        # Get nodes by depth
        levels = graph.get_nodes_by_depth()

        # Find function nodes at each level
        main_func = next((n for n in graph.nodes if n.get("name") == "main"), None)
        process_func = next(
            (n for n in graph.nodes if n.get("name") == "process"), None
        )
        core_func = next(
            (n for n in graph.nodes if n.get("name") == "core_function"), None
        )

        assert main_func is not None
        assert process_func is not None
        assert core_func is not None

        # Calculate in-degrees
        in_degrees = graph.calculate_in_degrees()

        # Verify depth structure
        assert in_degrees[main_func["id"]] == 0, "main should be entry point"
        assert in_degrees[process_func["id"]] > 0, "process should be called"
        assert in_degrees[core_func["id"]] > 0, "core_function should be called"

        # Check that we have at least 3 depth levels
        # (may have more due to import nodes)
        assert len(levels) >= 3, (
            f"Should have at least 3 depth levels, got {len(levels)}"
        )
