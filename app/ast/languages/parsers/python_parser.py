"""Python-specific AST parser with Python language features."""

from typing import List, Tuple

import tree_sitter_python as ts_python
from tree_sitter import Language, Node, Parser

from app.ast.base_models import BaseNode, BaseRelation
from app.ast.base_parser import BaseASTParser
from app.ast.languages.constants.python_constants import (
    PYTHON_CALL_TYPES,
    PYTHON_DEFINITION_TYPES,
    PYTHON_IMPORT_TYPES,
    PYTHON_NODE_TYPE_MAPPING,
)
from app.ast.languages.python_models import (
    PythonNode,
    PythonRelation,
    create_python_class_node,
    create_python_function_node,
)
from app.ast.node_extractor import NodeExtractor


class PythonParser(BaseASTParser):
    """Python-specific AST parser with enhanced Python language support."""

    def __init__(self):
        """Initialize Python parser."""
        super().__init__("python")
        self.parser = Parser(Language(ts_python.language()))
        self.extractor = NodeExtractor()

    def get_supported_extensions(self) -> List[str]:
        """Get supported Python file extensions."""
        return [".py", ".pyi", ".pyx"]

    def parse(
        self, source_code: str, file_path: str
    ) -> Tuple[List[BaseNode], List[BaseRelation]]:
        """Parse Python source code with Python-specific features."""
        tree = self.parser.parse(source_code.encode("utf-8"))

        nodes: List[BaseNode] = []
        relations: List[BaseRelation] = []
        definitions: dict[str, str] = {}  # name -> node_id

        def traverse(node: Node) -> None:
            if node.type in PYTHON_DEFINITION_TYPES:
                parsed_node = self._extract_python_node(node, source_code, file_path)
                if parsed_node:
                    nodes.append(parsed_node)
                    if parsed_node.name:
                        definitions[parsed_node.name] = parsed_node.id

            elif node.type in PYTHON_CALL_TYPES:
                relation = self._extract_python_call_relation(
                    node, file_path, definitions
                )
                if relation:
                    relations.append(relation)

            elif node.type in PYTHON_IMPORT_TYPES:
                import_node, import_relation = self._extract_python_import(
                    node, source_code, file_path
                )
                nodes.append(import_node)
                if import_relation:
                    relations.append(import_relation)

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return nodes, relations

    def _extract_python_node(
        self, node: Node, source_code: str, file_path: str
    ) -> BaseNode | None:
        """Extract Python node with Python-specific features."""
        name = self.extractor.find_name_in_node(node)
        node_id = self.extractor.generate_node_id(file_path, node)

        # Detect Python-specific features
        is_async = self._is_async_function(node, source_code)
        decorators = self._extract_decorators(node, source_code)

        if node.type in ["function_definition", "async_function_definition"]:
            return create_python_function_node(
                id=node_id,
                name=name or "anonymous",
                file_path=file_path,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                source_code=source_code[node.start_byte : node.end_byte],
                is_async=is_async,
                decorators=decorators,
            )
        elif node.type == "class_definition":
            return create_python_class_node(
                id=node_id,
                name=name or "anonymous",
                file_path=file_path,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                source_code=source_code[node.start_byte : node.end_byte],
                decorators=decorators,
            )

        return None

    def _extract_python_call_relation(
        self, node: Node, file_path: str, definitions: dict[str, str]
    ) -> BaseRelation | None:
        """Extract Python call relation with Python-specific features."""
        called_name = self.extractor.find_name_in_node(node)

        if not called_name:
            return None

        parent_def = self.extractor.find_parent_definition(node)
        if not parent_def:
            return None

        parent_id = self.extractor.generate_node_id(file_path, parent_def)
        target_id = definitions.get(called_name, f"{file_path}:external:{called_name}")

        # Check if this is an async call
        is_async_call = self._is_async_call(node)

        return PythonRelation(
            from_id=parent_id,
            to_id=target_id,
            relation_type="calls",
            is_async_call=is_async_call,
        )

    def _extract_python_import(
        self, node: Node, source_code: str, file_path: str
    ) -> Tuple[BaseNode, BaseRelation | None]:
        """Extract Python import with Python-specific features."""
        import_name = self.extractor.find_name_in_node(node)
        node_id = self.extractor.generate_node_id(file_path, node)

        import_node = PythonNode(
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
            relation = PythonRelation(
                from_id=node_id,
                to_id=f"module:{import_name}",
                relation_type="imports",
            )

        return import_node, relation

    def _is_async_function(self, node: Node, source_code: str) -> bool:
        """Check if a function is async."""
        if node.type == "async_function_definition":
            return True

        # Check for 'async' keyword in source
        node_source = source_code[node.start_byte : node.end_byte]
        return node_source.strip().startswith("async def")

    def _extract_decorators(self, node: Node, source_code: str) -> List[str]:
        """Extract decorators from a function or class."""
        decorators = []

        # Look for decorator nodes in parent or preceding siblings
        if node.parent and node.parent.type == "decorated_definition":
            for child in node.parent.children:
                if child.type == "decorator":
                    decorator_text = source_code[child.start_byte : child.end_byte]
                    # Clean up decorator text (remove @ and whitespace)
                    clean_decorator = decorator_text.strip().lstrip("@")
                    decorators.append(clean_decorator)

        return decorators

    def _is_async_call(self, node: Node) -> bool:
        """Check if a call is awaited (async call)."""
        # Look for 'await' keyword before the call
        if node.parent and node.parent.type == "await":
            return True

        # Check if the call is inside an await expression
        current = node.parent
        while current:
            if current.type == "await":
                return True
            current = current.parent

        return False

    def extract_language_specific_features(
        self, node: BaseNode, source_code: str
    ) -> BaseNode:
        """Extract additional Python-specific features from a node."""
        if isinstance(node, PythonNode):
            # Already has Python features
            return node

        # Convert base node to Python node with features
        # This is a fallback for compatibility
        python_node = PythonNode(
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

        return python_node
