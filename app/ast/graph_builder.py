"""Graph builder for code structure."""

from app.ast.graph import CodeGraph
from app.ast.base_models import BaseNode, BaseRelation, LanguageType
from app.ast.parser import parse_file


class GraphBuilder:
    """Builds a code graph from parsed AST data."""

    def __init__(self):
        """Initialize graph builder."""
        self.parsed_nodes: list[BaseNode] = []
        self.parsed_relations: list[BaseRelation] = []
        self._node_ids: set[str] = set()

    def add_nodes(self, nodes: list[BaseNode]) -> None:
        """Add nodes to the graph.

        Args:
            nodes: List of BaseNode objects
        """
        for node in nodes:
            if node.id not in self._node_ids:
                self.parsed_nodes.append(node)
                self._node_ids.add(node.id)

    def add_relations(self, relations: list[BaseRelation]) -> None:
        """Add relations to the graph.

        Args:
            relations: List of BaseRelation objects
        """
        self.parsed_relations.extend(relations)

    def build(self) -> CodeGraph:
        """Build the code graph from parsed nodes and relations.

        This method performs two important steps:
        1. Convert nodes to dictionary format
        2. Resolve relations by linking external calls to actual function definitions
        """
        # Create lookup dictionary: function/class name -> node ID
        name_to_node: dict[str, str] = {}
        for node in self.parsed_nodes:
            if node.name and node.type in {"function", "class"}:
                name_to_node[node.name] = node.id

        # Convert nodes to dict format
        nodes_dict = [n.to_dict() for n in self.parsed_nodes]

        # Resolve relations
        edges: list[dict[str, str]] = []
        for relation in self.parsed_relations:
            to_id = relation.to_id

            # If this is an external reference, try to resolve it
            if ":external:" in to_id:
                function_name = to_id.split(":")[-1]
                if function_name in name_to_node:
                    to_id = name_to_node[function_name]

            edges.append(
                {
                    "from": relation.from_id,
                    "to": to_id,
                    "type": relation.relation_type,
                }
            )

        return CodeGraph(nodes=nodes_dict, edges=edges)

    def reset(self) -> None:
        """Reset the builder state."""
        self.parsed_nodes = []
        self.parsed_relations = []
        self._node_ids = set()


def build_graph_from_files(file_paths: list[str]) -> CodeGraph:
    """Build a code graph from multiple files with auto-detection.

    Args:
        file_paths: List of file paths (language auto-detected from extensions)

    Returns:
        CodeGraph representing the entire codebase structure

    Example:
        >>> files = ["src/main.py", "src/utils.js", "src/types.ts"]
        >>> graph = build_graph_from_files(files)
    """
    builder = GraphBuilder()

    for file_path in file_paths:
        nodes, relations = parse_file(file_path)
        builder.add_nodes(nodes)
        builder.add_relations(relations)

    return builder.build()


def build_graph_from_files_with_language(
    file_paths: list[tuple[str, LanguageType]],
) -> CodeGraph:
    """Build a code graph from multiple files with explicit language specification.

    Args:
        file_paths: List of (file_path, language) tuples

    Returns:
        CodeGraph representing the entire codebase structure

    Example:
        >>> files = [
        ...     ("src/main.py", "python"),
        ...     ("src/utils.js", "javascript"),
        ... ]
        >>> graph = build_graph_from_files_with_language(files)
    """
    from app.ast.parser_factory import ParserFactory

    builder = GraphBuilder()

    for file_path, language in file_paths:
        parser = ParserFactory.get_parser(language)
        nodes, relations = parser.parse_file(file_path)
        builder.add_nodes(nodes)
        builder.add_relations(relations)

    return builder.build()
