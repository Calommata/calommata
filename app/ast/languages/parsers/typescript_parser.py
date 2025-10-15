"""TypeScript-specific AST parser with TypeScript language features."""

from typing import List, Tuple

import tree_sitter_typescript as ts_typescript
from tree_sitter import Language, Node, Parser

from app.ast.base_models import BaseNode, BaseRelation, NodeType
from app.ast.base_parser import BaseASTParser
from app.ast.languages.typescript_models import (
    TypeScriptNode,
    TypeScriptRelation,
    create_typescript_class_node,
    create_typescript_function_node,
    create_typescript_interface_node,
)
from app.ast.node_extractor import NodeExtractor


class TypeScriptParser(BaseASTParser):
    """TypeScript-specific AST parser with enhanced TypeScript language support."""

    # TypeScript-specific node types for definitions
    DEFINITION_TYPES = {
        "function_declaration",
        "function_signature",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "method_definition",
        "arrow_function",
        "enum_declaration",
        "namespace_declaration",
    }

    # TypeScript-specific call types
    CALL_TYPES = {"call_expression"}

    # TypeScript-specific import types
    IMPORT_TYPES = {"import_statement"}

    def __init__(self):
        """Initialize TypeScript parser."""
        super().__init__("typescript")
        self.parser = Parser(Language(ts_typescript.language_typescript()))
        self.extractor = NodeExtractor()

    def get_supported_extensions(self) -> List[str]:
        """Get supported TypeScript file extensions."""
        return [".ts", ".tsx", ".d.ts"]

    def parse(
        self, source_code: str, file_path: str
    ) -> Tuple[List[BaseNode], List[BaseRelation]]:
        """Parse TypeScript source code with TypeScript-specific features."""
        tree = self.parser.parse(source_code.encode("utf-8"))

        nodes: List[BaseNode] = []
        relations: List[BaseRelation] = []
        definitions: dict[str, str] = {}  # name -> node_id

        def traverse(node: Node) -> None:
            if node.type in self.DEFINITION_TYPES:
                parsed_node = self._extract_typescript_node(
                    node, source_code, file_path
                )
                if parsed_node:
                    nodes.append(parsed_node)
                    if parsed_node.name:
                        definitions[parsed_node.name] = parsed_node.id

            elif node.type in self.CALL_TYPES:
                relation = self._extract_typescript_call_relation(
                    node, file_path, definitions
                )
                if relation:
                    relations.append(relation)

            elif node.type in self.IMPORT_TYPES:
                import_node, import_relation = self._extract_typescript_import(
                    node, source_code, file_path
                )
                nodes.append(import_node)
                if import_relation:
                    relations.append(import_relation)

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return nodes, relations

    def _extract_typescript_node(
        self, node: Node, source_code: str, file_path: str
    ) -> BaseNode | None:
        """Extract TypeScript node with TypeScript-specific features."""
        name = self.extractor.find_name_in_node(node)
        node_id = self.extractor.generate_node_id(file_path, node)

        # Extract TypeScript-specific features
        type_annotation = self._extract_type_annotation(node, source_code)
        generic_parameters = self._extract_generic_parameters(node, source_code)
        access_modifier = self._extract_access_modifier(node, source_code)

        if node.type in [
            "function_declaration",
            "arrow_function",
            "function_signature",
        ]:
            return create_typescript_function_node(
                id=node_id,
                name=name or "anonymous",
                file_path=file_path,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                source_code=source_code[node.start_byte : node.end_byte],
                type_annotation=type_annotation,
                generic_parameters=generic_parameters,
                access_modifier=access_modifier,
            )
        elif node.type == "class_declaration":
            is_abstract = self._is_abstract_class(node, source_code)
            return create_typescript_class_node(
                id=node_id,
                name=name or "anonymous",
                file_path=file_path,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                source_code=source_code[node.start_byte : node.end_byte],
                generic_parameters=generic_parameters,
                access_modifier=access_modifier,
                is_abstract=is_abstract,
            )
        elif node.type == "interface_declaration":
            return create_typescript_interface_node(
                id=node_id,
                name=name or "anonymous",
                file_path=file_path,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                source_code=source_code[node.start_byte : node.end_byte],
                generic_parameters=generic_parameters,
            )

        # For other types, create basic TypeScript node
        return TypeScriptNode(
            id=node_id,
            type=self._map_node_type(node.type),
            name=name,
            file_path=file_path,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            source_code=source_code[node.start_byte : node.end_byte],
            has_type_annotation=bool(type_annotation),
            type_annotation=type_annotation,
            is_generic=bool(generic_parameters),
            generic_parameters=generic_parameters or [],
            access_modifier=access_modifier,
        )

    def _extract_typescript_call_relation(
        self, node: Node, file_path: str, definitions: dict[str, str]
    ) -> BaseRelation | None:
        """Extract TypeScript call relation with TypeScript-specific features."""
        called_name = self.extractor.find_name_in_node(node)

        if not called_name:
            return None

        parent_def = self.extractor.find_parent_definition(node)
        if not parent_def:
            return None

        parent_id = self.extractor.generate_node_id(file_path, parent_def)
        target_id = definitions.get(called_name, f"{file_path}:external:{called_name}")

        # Extract generic type if present - need source code for this
        # generic_type = self._extract_call_generic_type(node, source_code)
        generic_type = None

        return TypeScriptRelation(
            from_id=parent_id,
            to_id=target_id,
            relation_type="calls",
            generic_type=generic_type,
        )

    def _extract_typescript_import(
        self, node: Node, source_code: str, file_path: str
    ) -> Tuple[BaseNode, BaseRelation | None]:
        """Extract TypeScript import with TypeScript-specific features."""
        import_name = self.extractor.find_name_in_node(node)
        node_id = self.extractor.generate_node_id(file_path, node)

        import_node = TypeScriptNode(
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
            relation = TypeScriptRelation(
                from_id=node_id,
                to_id=f"module:{import_name}",
                relation_type="imports",
            )

        return import_node, relation

    def _extract_type_annotation(self, node: Node, source_code: str) -> str | None:
        """Extract type annotation from a node."""
        for child in node.children:
            if child.type == "type_annotation":
                return source_code[child.start_byte : child.end_byte].strip()
        return None

    def _extract_generic_parameters(
        self, node: Node, source_code: str
    ) -> List[str] | None:
        """Extract generic type parameters from a node."""
        generics = []
        for child in node.children:
            if child.type == "type_parameters":
                # Extract individual type parameter names
                for param in child.children:
                    if param.type == "type_parameter":
                        param_name = self.extractor.find_name_in_node(param)
                        if param_name:
                            generics.append(param_name)
        return generics if generics else None

    def _extract_access_modifier(self, node: Node, source_code: str) -> str | None:
        """Extract access modifier (public, private, protected) from a node."""
        for child in node.children:
            if child.type in ["public", "private", "protected"]:
                return child.type
        return None

    def _is_abstract_class(self, node: Node, source_code: str) -> bool:
        """Check if a class is abstract."""
        node_source = source_code[node.start_byte : node.end_byte]
        return "abstract class" in node_source

    def _extract_call_generic_type(self, node: Node, source_code: str) -> str | None:
        """Extract generic type from a call expression."""
        for child in node.children:
            if child.type == "type_arguments":
                return source_code[child.start_byte : child.end_byte].strip()
        return None

    def _map_node_type(self, tree_sitter_type: str) -> NodeType:
        """Map tree-sitter node type to our node type."""
        mapping: dict[str, NodeType] = {
            "function_declaration": "function",
            "arrow_function": "function",
            "class_declaration": "class",
            "interface_declaration": "interface",
            "type_alias_declaration": "type",
            "method_definition": "method",
        }
        return mapping.get(tree_sitter_type, "unknown")

    def extract_language_specific_features(
        self, node: BaseNode, source_code: str
    ) -> BaseNode:
        """Extract additional TypeScript-specific features from a node."""
        if isinstance(node, TypeScriptNode):
            # Already has TypeScript features
            return node

        # Convert base node to TypeScript node with features
        typescript_node = TypeScriptNode(
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

        return typescript_node
