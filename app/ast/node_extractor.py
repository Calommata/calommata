"""Node extraction utilities for AST parsing."""

from tree_sitter import Node

from app.ast.constants import (
    COMPLEX_NAME_TYPES,
    IDENTIFIER_TYPES,
    NODE_TYPE_MAPPING,
    PARENT_DEFINITION_TYPES,
)
from app.ast.models import ParsedNode, ParsedRelation


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

    @classmethod
    def extract_node(cls, node: Node, source_code: str, file_path: str) -> ParsedNode:
        """Extract a node (function, class, etc.)."""
        node_type = NODE_TYPE_MAPPING.get(node.type, "unknown")
        name = cls.find_name_in_node(node)
        node_id = cls.generate_node_id(file_path, node)

        return ParsedNode(
            id=node_id,
            type=node_type,
            name=name,
            file_path=file_path,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            source_code=source_code[node.start_byte : node.end_byte],
        )

    @classmethod
    def extract_call_relation(
        cls, node: Node, file_path: str, definitions: dict[str, str]
    ) -> ParsedRelation | None:
        """Extract a function call relation."""
        called_name = cls.find_name_in_node(node)

        if not called_name:
            return None

        parent_def = cls.find_parent_definition(node)
        if not parent_def:
            return None

        parent_id = cls.generate_node_id(file_path, parent_def)
        target_id = definitions.get(called_name, f"{file_path}:external:{called_name}")

        return ParsedRelation(
            from_id=parent_id,
            to_id=target_id,
            relation_type="calls",
        )

    @classmethod
    def extract_import(
        cls, node: Node, source_code: str, file_path: str
    ) -> tuple[ParsedNode, ParsedRelation | None]:
        """Extract import statement."""
        import_name = cls.find_name_in_node(node)
        node_id = cls.generate_node_id(file_path, node)

        import_node = ParsedNode(
            id=node_id,
            type="import",
            name=import_name,
            file_path=file_path,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            source_code=source_code[node.start_byte : node.end_byte],
        )

        relation = None
        if import_name:
            relation = ParsedRelation(
                from_id=node_id,
                to_id=f"module:{import_name}",
                relation_type="imports",
            )

        return import_node, relation
