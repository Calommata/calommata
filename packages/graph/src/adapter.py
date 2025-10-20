"""
Parser 결과를 Graph 모델로 변환하는 어댑터
CodeBlock 객체들을 Graph 패키지의 모델로 변환
"""

from pathlib import Path
from typing import Any

from .models import CodeGraph, CodeNode, CodeRelation, NodeType, RelationType


class ParserToGraphAdapter:
    """Parser 결과를 Graph 모델로 변환하는 어댑터"""

    def __init__(self):
        self.node_counter = 0

    def convert_to_graph(
        self,
        parser_results: list[dict[str, Any]] | list[Any],
        project_name: str = "unknown",
        project_path: str = "unknown",
    ) -> CodeGraph:
        """Parser 결과를 CodeGraph로 변환"""

        # 그래프 초기화
        graph = CodeGraph(project_name=project_name, project_path=project_path)

        # Parser 결과가 CodeBlock 객체들인지 dict들인지 확인
        if parser_results and hasattr(parser_results[0], "block_type"):
            # CodeBlock 객체들 처리
            return self._convert_from_code_blocks(parser_results, graph)
        else:
            # 기존 dict 형태 처리
            return self._convert_from_dicts(parser_results, graph)

    def _convert_from_code_blocks(
        self, code_blocks: list[Any], graph: CodeGraph
    ) -> CodeGraph:
        """CodeBlock 객체들을 Graph로 변환"""
        node_map = {}  # name -> node_id 매핑

        # 1단계: 모든 블록을 노드로 변환
        for block in code_blocks:
            node = self._create_node_from_code_block(block)
            graph.add_node(node)
            file_path = getattr(block, "file_path", "unknown.py")
            node_map[f"{file_path}:{block.name}"] = node.id

        # 2단계: 관계 생성 (CodeBlock의 dependencies 활용)
        for block in code_blocks:
            if hasattr(block, "dependencies") and block.dependencies:
                for dep_target in block.dependencies:
                    relation = self._create_relation_from_string_dependency(
                        block, dep_target, node_map
                    )
                    if relation:
                        try:
                            graph.add_relation(relation)
                        except ValueError:
                            # 존재하지 않는 노드에 대한 관계는 무시
                            continue

        # 통계 정보 업데이트
        self._update_graph_statistics(graph)
        return graph

    def _convert_from_dicts(
        self, parser_results: list[dict[str, Any]], graph: CodeGraph
    ) -> CodeGraph:
        """기존 dict 형태 데이터를 Graph로 변환"""
        node_map = {}  # full_name -> node_id 매핑

        # 1단계: 모든 블록을 노드로 변환
        for block_data in parser_results:
            node = self._create_node_from_dict(block_data)
            graph.add_node(node)

            # 전체 이름으로 매핑 저장
            full_name = block_data.get("full_name", f"{node.file_path}:{node.name}")
            node_map[full_name] = node.id

        # 2단계: 관계 생성
        for block_data in parser_results:
            relations = self._extract_relations_from_dict(block_data, node_map)
            for relation in relations:
                try:
                    graph.add_relation(relation)
                except ValueError:
                    # 존재하지 않는 노드에 대한 관계는 무시
                    continue

        # 통계 정보 업데이트
        self._update_graph_statistics(graph)
        return graph

    def _create_node_from_code_block(self, block) -> CodeNode:
        """CodeBlock 객체로부터 CodeNode 생성"""

        # 노드 타입 매핑
        node_type = self._map_block_type_from_enum(block.block_type)

        # file_path는 이제 CodeBlock에 있어야 함
        file_path = getattr(block, "file_path", "unknown.py")
        if file_path == "unknown.py":
            print(f"Warning: CodeBlock {block.name} has no file_path")

        # 고유 ID 생성
        node_id = f"{file_path}:{block.name}:{block.start_line}"

        return CodeNode(
            id=node_id,
            name=block.name,
            node_type=node_type,
            file_path=file_path,
            start_line=block.start_line,
            end_line=block.end_line,
            source_code=getattr(block, "source_code", "") or "",
            docstring=getattr(block, "docstring", None),
            complexity=getattr(block, "complexity", 0),
            scope_level=getattr(block, "scope_level", 0),
            parameters=getattr(block, "parameters", []),
            return_type=getattr(block, "return_type", None),
            decorators=getattr(block, "decorators", []),
            imports=getattr(block, "imports", []) or [],
        )

    def _create_relation_from_dependency(
        self, block, dependency, node_map: dict
    ) -> CodeRelation | None:
        """CodeBlock의 dependency로부터 CodeRelation 생성"""

        file_path = getattr(block, "file_path", "unknown.py")
        source_key = f"{file_path}:{block.name}"
        target_key = f"{file_path}:{dependency.target}"  # 같은 파일 내 가정

        if source_key not in node_map or target_key not in node_map:
            return None

        # 의존성 타입을 관계 타입으로 매핑
        relation_type = self._map_dependency_type(dependency.dependency_type)

        return CodeRelation(
            from_node_id=node_map[source_key],
            to_node_id=node_map[target_key],
            relation_type=relation_type,
            context=f"dependency_type: {dependency.dependency_type.value if hasattr(dependency.dependency_type, 'value') else str(dependency.dependency_type)}",
        )

    def _create_relation_from_string_dependency(
        self, block, target_name: str, node_map: dict
    ) -> CodeRelation | None:
        """CodeBlock의 문자열 dependency로부터 CodeRelation 생성"""

        file_path = getattr(block, "file_path", "unknown.py")
        source_key = f"{file_path}:{block.name}"
        target_key = f"{file_path}:{target_name}"  # 같은 파일 내 가정

        if source_key not in node_map or target_key not in node_map:
            return None

        # 기본 관계 타입 사용
        relation_type = RelationType.REFERENCES

        return CodeRelation(
            from_node_id=node_map[source_key],
            to_node_id=node_map[target_key],
            relation_type=relation_type,
            context="dependency_type: references",
        )

    def _map_block_type_from_enum(self, block_type_enum) -> NodeType:
        """BlockType enum을 NodeType enum으로 매핑"""
        type_mapping = {
            "module": NodeType.MODULE,
            "class": NodeType.CLASS,
            "function": NodeType.FUNCTION,
            "import": NodeType.IMPORT,
            "variable": NodeType.VARIABLE,
        }

        block_type_str = (
            block_type_enum.value
            if hasattr(block_type_enum, "value")
            else str(block_type_enum)
        )
        return type_mapping.get(block_type_str, NodeType.MODULE)

    def _map_dependency_type(self, dep_type) -> RelationType:
        """DependencyType을 RelationType으로 매핑"""
        type_mapping = {
            "calls": RelationType.CALLS,
            "inherits": RelationType.INHERITS,
            "imports": RelationType.IMPORTS,
            "references": RelationType.REFERENCES,
            "defines": RelationType.DEFINES,
            "contains": RelationType.CONTAINS,
        }

        dep_type_str = dep_type.value if hasattr(dep_type, "value") else str(dep_type)
        return type_mapping.get(dep_type_str, RelationType.REFERENCES)

    def _create_node_from_dict(self, block_data: dict[str, Any]) -> CodeNode:
        """블록 데이터로부터 CodeNode 생성"""

        # 노드 타입 매핑
        node_type = self._map_block_type(block_data.get("block_type", "unknown"))

        # 고유 ID 생성
        node_id = self._generate_node_id(block_data)

        return CodeNode(
            id=node_id,
            name=block_data.get("name", "unknown"),
            node_type=node_type,
            file_path=block_data.get("file_path", "unknown.py"),
            start_line=block_data.get("start_line", 0),
            end_line=block_data.get("end_line", 0),
            source_code=block_data.get("source_code", ""),
            docstring=block_data.get("docstring"),
            complexity=self._calculate_complexity(block_data),
            scope_level=block_data.get("scope_level", 0),
            parameters=block_data.get("parameters", []),
            return_type=block_data.get("return_type"),
            decorators=block_data.get("decorators", []),
            imports=block_data.get("imports", []),
        )

    def _map_block_type(self, block_type: str) -> NodeType:
        """블록 타입을 NodeType으로 매핑"""
        mapping = {
            "module": NodeType.MODULE,
            "class": NodeType.CLASS,
            "function": NodeType.FUNCTION,
            "method": NodeType.METHOD,
            "import": NodeType.IMPORT,
            "variable": NodeType.VARIABLE,
            "property": NodeType.PROPERTY,
            "decorator": NodeType.DECORATOR,
        }
        return mapping.get(block_type.lower(), NodeType.FUNCTION)

    def _generate_node_id(self, block_data: dict[str, Any]) -> str:
        """고유 노드 ID 생성"""
        file_path = block_data.get("file_path", "unknown.py")
        file_name = Path(file_path).stem
        block_type = block_data.get("block_type", "unknown")
        name = block_data.get("name", "unknown")
        start_line = block_data.get("start_line", 0)

        return f"{file_name}:{block_type}:{name}:{start_line}"

    def _calculate_complexity(self, block_data: dict[str, Any]) -> int:
        """복잡도 계산"""
        # 기본 복잡도
        complexity = block_data.get("complexity", 0)
        if complexity > 0:
            return complexity

        # 라인 수 기반 계산
        start_line = block_data.get("start_line", 0)
        end_line = block_data.get("end_line", 0)
        line_count = max(end_line - start_line + 1, 1)

        # 의존성 수 고려
        dependencies = block_data.get("dependencies", [])
        dependency_weight = len(dependencies) * 2

        return line_count + dependency_weight

    def _extract_relations_from_dict(
        self, block_data: dict[str, Any], node_map: dict[str, str]
    ) -> list[CodeRelation]:
        """블록 데이터에서 관계 추출"""
        relations = []

        from_node_id = self._generate_node_id(block_data)

        # 부모-자식 관계
        parent_name = block_data.get("parent")
        if parent_name and parent_name in node_map:
            relations.append(
                CodeRelation(
                    from_node_id=node_map[parent_name],
                    to_node_id=from_node_id,
                    relation_type=RelationType.CONTAINS,
                    weight=1.0,
                )
            )

        # Import 관계
        imports = block_data.get("imports", [])
        for imported_module in imports:
            # 외부 import는 별도 처리
            external_id = f"external:{imported_module}"
            relations.append(
                CodeRelation(
                    from_node_id=from_node_id,
                    to_node_id=external_id,
                    relation_type=RelationType.IMPORTS,
                    weight=1.0,
                )
            )

        # 의존성 관계
        dependencies = block_data.get("dependencies", [])
        for dep in dependencies:
            if isinstance(dep, dict):
                target = dep.get("target")
                dep_type = dep.get("type", "depends_on")
                line_number = dep.get("line_number")
                context = dep.get("context")
            else:
                # 문자열 형태의 의존성
                target = str(dep)
                dep_type = "depends_on"
                line_number = None
                context = None

            # 프로젝트 내 노드인지 확인
            if target in node_map:
                relation_type = self._map_dependency_type(dep_type)
                relations.append(
                    CodeRelation(
                        from_node_id=from_node_id,
                        to_node_id=node_map[target],
                        relation_type=relation_type,
                        weight=1.0,
                        line_number=line_number,
                        context=context,
                    )
                )

        return relations

    def _map_dependency_type(self, dep_type: str) -> RelationType:
        """의존성 타입을 RelationType으로 매핑"""
        mapping = {
            "call": RelationType.CALLS,
            "calls": RelationType.CALLS,
            "inherit": RelationType.INHERITS,
            "inherits": RelationType.INHERITS,
            "import": RelationType.IMPORTS,
            "imports": RelationType.IMPORTS,
            "contains": RelationType.CONTAINS,
            "define": RelationType.DEFINES,
            "defines": RelationType.DEFINES,
            "reference": RelationType.REFERENCES,
            "references": RelationType.REFERENCES,
            "depends_on": RelationType.DEPENDS_ON,
            "decorates": RelationType.DECORATES,
            "implements": RelationType.IMPLEMENTS,
            "raises": RelationType.RAISES,
            "returns": RelationType.RETURNS,
        }
        return mapping.get(dep_type.lower(), RelationType.DEPENDS_ON)

    def _update_graph_statistics(self, graph: CodeGraph):
        """그래프 통계 정보 업데이트"""
        if graph.nodes:
            # 파일 수 계산
            file_paths = {node.file_path for node in graph.nodes.values()}
            graph.total_files = len(file_paths)

            # 총 라인 수 계산
            total_lines = 0
            for node in graph.nodes.values():
                if node.end_line > 0:
                    total_lines += node.end_line - node.start_line + 1
            graph.total_lines = total_lines


# 편의 함수들
def convert_parser_results_to_graph(
    parser_results: list[dict[str, Any]], project_name: str, project_path: str
) -> CodeGraph:
    """편의 함수: Parser 결과를 CodeGraph로 변환"""
    adapter = ParserToGraphAdapter()
    return adapter.convert_to_graph(parser_results, project_name, project_path)


def create_node_from_dict(node_data: dict[str, Any]) -> CodeNode:
    """편의 함수: 딕셔너리에서 CodeNode 생성"""
    adapter = ParserToGraphAdapter()
    return adapter._create_node_from_block(node_data)
