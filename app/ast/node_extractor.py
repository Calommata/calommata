"""Node extraction utilities for AST parsing."""

from tree_sitter import Node

from app.ast.constants import (
    COMPLEX_NAME_TYPES,
    IDENTIFIER_TYPES,
    PARENT_DEFINITION_TYPES,
)


class NodeExtractor:
    """Extracts nodes and relations from tree-sitter AST."""

    @staticmethod
    def generate_node_id(file_path: str, node: Node) -> str:
        """Generate a unique ID for a node."""
        return f"{file_path}:{node.start_byte}:{node.end_byte}"

    @staticmethod
    def find_name_in_node(node: Node) -> str | None:
        """Find identifier name within a node.

        For arrow functions, checks parent variable_declarator.
        """
        # Special case: arrow_function gets name from parent variable_declarator
        if node.type == "arrow_function" and node.parent:
            parent = node.parent
            if parent.type == "variable_declarator":
                # The first child of variable_declarator is the identifier
                for child in parent.children:
                    if child.type in IDENTIFIER_TYPES and child.text:
                        return child.text.decode("utf-8")

        # Standard case: search children
        for child in node.children:
            if child.type in IDENTIFIER_TYPES:
                if child.text:
                    return child.text.decode("utf-8")

            # Recursively search in certain node types
            if child.type in COMPLEX_NAME_TYPES:
                name = NodeExtractor.find_name_in_node(child)
                if name:
                    return name

        return None

    @staticmethod
    def find_parent_definition(node: Node) -> Node | None:
        """Find the parent function or class definition of a node."""
        current = node.parent
        while current:
            if current.type in PARENT_DEFINITION_TYPES:
                return current
            current = current.parent
        return None
