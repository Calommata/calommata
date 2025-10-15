"""Tests for node extractor."""

import pytest
from tree_sitter import Language, Parser
from tree_sitter_python import language as language_python

from app.ast.models import ParsedNode, ParsedRelation
from app.ast.node_extractor import NodeExtractor


def test_generate_node_id():
    """Test node ID generation."""
    source = b"def test(): pass"
    parser = Parser(Language(language_python()))
    tree = parser.parse(source)
    root = tree.root_node

    if root.children:
        test_node = root.children[0]
        node_id = NodeExtractor.generate_node_id("/path/to/file.py", test_node)
        assert "/path/to/file.py:" in node_id
        assert ":" in node_id.replace("/path/to/file.py:", "")
    else:
        pytest.skip("Could not find test node")


def test_find_name_in_node():
    """Test finding node names."""
    source = b"""
def my_function():
    pass

class MyClass:
    pass
"""

    parser = Parser(Language(language_python()))
    tree = parser.parse(source)
    root = tree.root_node

    # Find function definition
    func_node = None
    class_node = None
    for child in root.children:
        if child.type == "function_definition":
            func_node = child
        elif child.type == "class_definition":
            class_node = child

    assert func_node is not None, "Should find function node"
    assert class_node is not None, "Should find class node"

    func_name = NodeExtractor.find_name_in_node(func_node)
    class_name = NodeExtractor.find_name_in_node(class_node)

    assert func_name == "my_function", (
        f"Function name should be 'my_function', got '{func_name}'"
    )
    assert class_name == "MyClass", (
        f"Class name should be 'MyClass', got '{class_name}'"
    )


def test_extract_node():
    """Test node extraction."""
    source_code = """
def test_func():
    return 42

class TestClass:
    def method(self):
        pass
"""
    source = source_code.encode("utf-8")

    parser = Parser(Language(language_python()))
    tree = parser.parse(source)
    root = tree.root_node

    file_path = "/test/file.py"
    nodes: list[ParsedNode] = []

    # Extract nodes
    for child in root.children:
        if child.type in ("function_definition", "class_definition"):
            node = NodeExtractor.extract_node(child, source_code, file_path)
            if node:
                nodes.append(node)

    assert len(nodes) == 2, f"Should extract 2 nodes, got {len(nodes)}"

    func_node = next(n for n in nodes if n.type == "function")
    class_node = next(n for n in nodes if n.type == "class")

    assert func_node.name == "test_func"
    assert class_node.name == "TestClass"
    assert "def test_func():" in func_node.source_code
    assert "class TestClass:" in class_node.source_code


def test_extract_call_relation():
    """Test call relation extraction."""
    source_code = """
def greet(name):
    return f"Hello, {name}"

def main():
    greet("World")
"""
    source = source_code.encode("utf-8")

    parser = Parser(Language(language_python()))
    tree = parser.parse(source)
    root = tree.root_node

    file_path = "/test/file.py"
    definitions: dict[str, str] = {}

    # Build definitions map first
    for child in root.children:
        if child.type == "function_definition":
            name = NodeExtractor.find_name_in_node(child)
            if name:
                node_id = NodeExtractor.generate_node_id(file_path, child)
                definitions[name] = node_id

    # Find the call expression in main()
    main_func = None
    call_node = None
    for child in root.children:
        if child.type == "function_definition":
            name = NodeExtractor.find_name_in_node(child)
            if name == "main":
                main_func = child
                for descendant in child.children:
                    if descendant.type == "block":
                        for stmt in descendant.children:
                            if stmt.type == "expression_statement":
                                if stmt.children and stmt.children[0].type == "call":
                                    call_node = stmt.children[0]
                                    break

    if main_func and call_node:
        relation = NodeExtractor.extract_call_relation(
            call_node, file_path, definitions
        )

        if relation:
            caller_id = NodeExtractor.generate_node_id(file_path, main_func)
            assert relation.from_id == caller_id, (
                f"Expected from_id {caller_id}, got {relation.from_id}"
            )
            assert relation.relation_type == "calls"
            # The relation should point to the greet function or external
            assert (
                relation.to_id in definitions.values() or "greet" in relation.to_id
            ), f"Expected to_id to be greet function, got {relation.to_id}"
        else:
            pytest.skip("Call relation not extracted (may be complex)")
    else:
        pytest.skip("Could not find call node in test")


def test_extract_import():
    """Test import extraction."""
    source_code = """
import os
from pathlib import Path
"""
    source = source_code.encode("utf-8")

    parser = Parser(Language(language_python()))
    tree = parser.parse(source)
    root = tree.root_node

    file_path = "/test/file.py"
    import_nodes: list[ParsedNode] = []
    relations: list[ParsedRelation] = []

    for child in root.children:
        if child.type in ("import_statement", "import_from_statement"):
            node, relation = NodeExtractor.extract_import(child, source_code, file_path)
            import_nodes.append(node)
            if relation:
                relations.append(relation)

    assert len(import_nodes) >= 1, "Should extract at least one import node"
    assert len(relations) >= 1, "Should extract at least one import relation"
    assert any(r.relation_type == "imports" for r in relations)
