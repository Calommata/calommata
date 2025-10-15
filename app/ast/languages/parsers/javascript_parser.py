"""JavaScript-specific AST parser with JavaScript language features."""

from typing import List, Tuple

import tree_sitter_javascript as ts_javascript
from tree_sitter import Language, Node, Parser

from app.ast.base_models import BaseNode, BaseRelation, NodeType
from app.ast.base_parser import BaseASTParser
from app.ast.languages.javascript_models import (
    JavaScriptNode,
    JavaScriptRelation,
    create_javascript_class_node,
    create_javascript_function_node,
)
from app.ast.node_extractor import NodeExtractor


class JavaScriptParser(BaseASTParser):
    """JavaScript-specific AST parser with enhanced JavaScript language support."""

    # JavaScript-specific node types for definitions
    DEFINITION_TYPES = {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "generator_function_declaration",
    }

    # JavaScript-specific call types
    CALL_TYPES = {"call_expression"}

    # JavaScript-specific import types
    IMPORT_TYPES = {"import_statement"}

    def __init__(self):
        """Initialize JavaScript parser."""
        super().__init__("javascript")
        self.parser = Parser(Language(ts_javascript.language()))
        self.extractor = NodeExtractor()

    def get_supported_extensions(self) -> List[str]:
        """Get supported JavaScript file extensions."""
        return [".js", ".jsx", ".mjs", ".cjs"]

    def parse(
        self, source_code: str, file_path: str
    ) -> Tuple[List[BaseNode], List[BaseRelation]]:
        """Parse JavaScript source code with JavaScript-specific features."""
        tree = self.parser.parse(source_code.encode("utf-8"))

        nodes: List[BaseNode] = []
        relations: List[BaseRelation] = []
        definitions: dict[str, str] = {}  # name -> node_id

        def traverse(node: Node) -> None:
            if node.type in self.DEFINITION_TYPES:
                parsed_node = self._extract_javascript_node(
                    node, source_code, file_path
                )
                if parsed_node:
                    nodes.append(parsed_node)
                    if parsed_node.name:
                        definitions[parsed_node.name] = parsed_node.id

            elif node.type in self.CALL_TYPES:
                relation = self._extract_javascript_call_relation(
                    node, file_path, definitions
                )
                if relation:
                    relations.append(relation)

            elif node.type in self.IMPORT_TYPES:
                import_node, import_relation = self._extract_javascript_import(
                    node, source_code, file_path
                )
                nodes.append(import_node)
                if import_relation:
                    relations.append(import_relation)

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return nodes, relations

    def _extract_javascript_node(
        self, node: Node, source_code: str, file_path: str
    ) -> BaseNode | None:
        """Extract JavaScript node with JavaScript-specific features."""
        name = self.extractor.find_name_in_node(node)
        node_id = self.extractor.generate_node_id(file_path, node)

        # Detect JavaScript-specific features
        is_arrow_function = node.type == "arrow_function"
        is_anonymous = not name or name == "anonymous"
        is_generator = self._is_generator_function(node, source_code)
        is_constructor = self._is_constructor_method(node, source_code)

        if node.type in [
            "function_declaration",
            "arrow_function",
            "generator_function_declaration",
        ]:
            return create_javascript_function_node(
                id=node_id,
                name=name or "anonymous",
                file_path=file_path,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                source_code=source_code[node.start_byte : node.end_byte],
                is_arrow_function=is_arrow_function,
                is_anonymous=is_anonymous,
                is_generator=is_generator,
            )
        elif node.type == "class_declaration":
            return create_javascript_class_node(
                id=node_id,
                name=name or "anonymous",
                file_path=file_path,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                source_code=source_code[node.start_byte : node.end_byte],
                is_constructor=is_constructor,
            )

        # For other types, create basic JavaScript node
        return JavaScriptNode(
            id=node_id,
            type=self._map_node_type(node.type),
            name=name,
            file_path=file_path,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            source_code=source_code[node.start_byte : node.end_byte],
            is_arrow_function=is_arrow_function,
            is_anonymous=is_anonymous,
            is_generator=is_generator,
            is_constructor=is_constructor,
        )

    def _extract_javascript_call_relation(
        self, node: Node, file_path: str, definitions: dict[str, str]
    ) -> BaseRelation | None:
        """Extract JavaScript call relation with JavaScript-specific features."""
        called_name = self.extractor.find_name_in_node(node)

        if not called_name:
            return None

        parent_def = self.extractor.find_parent_definition(node)
        if not parent_def:
            return None

        parent_id = self.extractor.generate_node_id(file_path, parent_def)
        target_id = definitions.get(called_name, f"{file_path}:external:{called_name}")

        # Check if this is a callback call
        is_callback_call = self._is_callback_call(node)

        return JavaScriptRelation(
            from_id=parent_id,
            to_id=target_id,
            relation_type="calls",
            is_callback_call=is_callback_call,
        )

    def _extract_javascript_import(
        self, node: Node, source_code: str, file_path: str
    ) -> Tuple[BaseNode, BaseRelation | None]:
        """Extract JavaScript import with JavaScript-specific features."""
        import_name = self.extractor.find_name_in_node(node)
        node_id = self.extractor.generate_node_id(file_path, node)

        import_node = JavaScriptNode(
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
            relation = JavaScriptRelation(
                from_id=node_id,
                to_id=f"module:{import_name}",
                relation_type="imports",
            )

        return import_node, relation

    def _is_generator_function(self, node: Node, source_code: str) -> bool:
        """Check if a function is a generator function."""
        if node.type == "generator_function_declaration":
            return True

        # Check for function* syntax
        node_source = source_code[node.start_byte : node.end_byte]
        return "function*" in node_source

    def _is_constructor_method(self, node: Node, source_code: str) -> bool:
        """Check if a method is a constructor."""
        if node.type == "method_definition":
            name = self.extractor.find_name_in_node(node)
            return name == "constructor"
        return False

    def _is_callback_call(self, node: Node) -> bool:
        """Check if a call is likely a callback (heuristic-based)."""
        # Simple heuristic: check if the function is passed as an argument
        if node.parent and node.parent.type == "arguments":
            return True

        # Check if it's in common callback patterns like map, filter, etc.
        called_name = self.extractor.find_name_in_node(node)
        callback_patterns = ["map", "filter", "forEach", "reduce", "then", "catch"]
        return called_name in callback_patterns if called_name else False

    def _map_node_type(self, tree_sitter_type: str) -> NodeType:
        """Map tree-sitter node type to our node type."""
        mapping: dict[str, NodeType] = {
            "function_declaration": "function",
            "arrow_function": "function",
            "generator_function_declaration": "function",
            "class_declaration": "class",
            "method_definition": "method",
        }
        return mapping.get(tree_sitter_type, "unknown")

    def extract_language_specific_features(
        self, node: BaseNode, source_code: str
    ) -> BaseNode:
        """Extract additional JavaScript-specific features from a node."""
        if isinstance(node, JavaScriptNode):
            # Already has JavaScript features
            return node

        # Convert base node to JavaScript node with features
        javascript_node = JavaScriptNode(
            id=node.id,
            type=node.type,
            name=node.name,
            file_path=node.file_path,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_line,
            end_line=node.end_line,
            source_code=node.source_code,
            parent_id=node.parent_id,
        )

        return javascript_node
