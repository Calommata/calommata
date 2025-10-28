"""관계 생성 전문 클래스

노드 간의 관계를 생성하고 관리하는 책임만 담당합니다.
"""

import logging
from typing import Any

from .models import CodeGraph, CodeRelation, RelationType

logger = logging.getLogger(__name__)


class RelationshipBuilder:
    """노드 간 관계 생성 전문 클래스"""

    def build_dependency_relations(
        self,
        blocks: list[Any],
        graph: CodeGraph,
        node_map: dict[str, str],
    ) -> None:
        """의존성 기반 관계 생성

        Args:
            blocks: CodeBlock 리스트
            graph: 대상 CodeGraph
            node_map: 노드 매핑 (key -> node_id)
        """
        for block in blocks:
            if not hasattr(block, "dependencies") or not block.dependencies:
                continue

            for dep_target in block.dependencies:
                relation = self._create_dependency_relation(block, dep_target, node_map)
                if relation:
                    self._add_relation_safe(graph, relation)

    def build_structural_relations(
        self,
        blocks: list[Any],
        graph: CodeGraph,
        node_map: dict[str, str],
    ) -> None:
        """구조적 관계 생성 (CONTAINS)

        클래스 -> 메서드, 모듈 -> 최상위 요소 관계를 생성합니다.

        Args:
            blocks: CodeBlock 리스트
            graph: 대상 CodeGraph
            node_map: 노드 매핑
        """
        for block in blocks:
            file_path = getattr(block, "file_path", "unknown.py")
            block_type = self._get_block_type_value(block)

            if block_type == "class":
                self._create_class_contains_relations(
                    block, blocks, graph, node_map, file_path
                )
            elif block_type == "module":
                self._create_module_contains_relations(
                    block, blocks, graph, node_map, file_path
                )

    def build_relations_from_dict(
        self,
        block_data: dict[str, Any],
        node_map: dict[str, str],
    ) -> list[CodeRelation]:
        """딕셔너리에서 관계 추출

        Args:
            block_data: 블록 데이터
            node_map: 노드 매핑

        Returns:
            생성된 관계 리스트
        """
        relations = []
        from_node_id = self._generate_node_id_from_dict(block_data)

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
            rel = self._create_dependency_from_dict(dep, from_node_id, node_map)
            if rel:
                relations.append(rel)

        return relations

    def _create_dependency_relation(
        self,
        block: Any,
        target_name: str,
        node_map: dict[str, str],
    ) -> CodeRelation | None:
        """의존성에서 관계 생성"""
        file_path = getattr(block, "file_path", "unknown.py")
        source_key = f"{file_path}:{block.name}"

        # 1. 같은 파일 내에서 찾기
        target_key = f"{file_path}:{target_name}"
        if source_key in node_map and target_key in node_map:
            return CodeRelation(
                from_node_id=node_map[source_key],
                to_node_id=node_map[target_key],
                relation_type=RelationType.CALLS,
                context=f"same_file_call: {target_name}",
            )

        # 2. 다른 파일에서 찾기
        for key in node_map.keys():
            if key.endswith(f":{target_name}") and key != target_key:
                return CodeRelation(
                    from_node_id=node_map[source_key],
                    to_node_id=node_map[key],
                    relation_type=RelationType.CALLS,
                    context=f"cross_file_call: {target_name}",
                )

        # 3. Import 관계
        if target_name.startswith("import_"):
            import_name = target_name.replace("import_", "")
            for key in node_map.keys():
                if key.endswith(f":{import_name}") or key.endswith(f":{target_name}"):
                    return CodeRelation(
                        from_node_id=node_map[source_key],
                        to_node_id=node_map[key],
                        relation_type=RelationType.IMPORTS,
                        context=f"import: {import_name}",
                    )

        return None

    def _create_class_contains_relations(
        self,
        block: Any,
        blocks: list[Any],
        graph: CodeGraph,
        node_map: dict[str, str],
        file_path: str,
    ) -> None:
        """클래스 -> 메서드 CONTAINS 관계 생성"""
        for other_block in blocks:
            other_type = self._get_block_type_value(other_block)

            if (
                other_type == "function"
                and getattr(other_block, "file_path", "") == file_path
                and getattr(other_block, "parent") == block
            ):
                source_key = f"{file_path}:{block.name}"
                target_key = f"{file_path}:{other_block.name}"

                if source_key in node_map and target_key in node_map:
                    relation = CodeRelation(
                        from_node_id=node_map[source_key],
                        to_node_id=node_map[target_key],
                        relation_type=RelationType.CONTAINS,
                        context="structural: class contains method",
                    )
                    self._add_relation_safe(graph, relation)

    def _create_module_contains_relations(
        self,
        block: Any,
        blocks: list[Any],
        graph: CodeGraph,
        node_map: dict[str, str],
        file_path: str,
    ) -> None:
        """모듈 -> 최상위 요소 CONTAINS 관계 생성"""
        for other_block in blocks:
            other_type = self._get_block_type_value(other_block)

            if (
                getattr(other_block, "file_path", "") == file_path
                and other_type in ["class", "function"]
                and other_block != block
            ):
                # 최상위 요소인지 확인
                if self._is_top_level_element(other_block, blocks, file_path):
                    source_key = f"{file_path}:{block.name}"
                    target_key = f"{file_path}:{other_block.name}"

                    if source_key in node_map and target_key in node_map:
                        relation = CodeRelation(
                            from_node_id=node_map[source_key],
                            to_node_id=node_map[target_key],
                            relation_type=RelationType.CONTAINS,
                            context="structural: module contains top-level element",
                        )
                        self._add_relation_safe(graph, relation)

    def _is_top_level_element(
        self,
        element: Any,
        blocks: list[Any],
        file_path: str,
    ) -> bool:
        """요소가 최상위인지 확인 (다른 클래스 안에 없음)"""
        for container_block in blocks:
            container_type = self._get_block_type_value(container_block)

            if (
                container_type == "class"
                and container_block != element
                and getattr(container_block, "file_path", "") == file_path
                and getattr(element, "parent") == container_block
            ):
                return False
        return True

    def _create_dependency_from_dict(
        self,
        dep: Any,
        from_node_id: str,
        node_map: dict[str, str],
    ) -> CodeRelation | None:
        """딕셔너리 의존성에서 관계 생성"""
        if isinstance(dep, dict):
            target = dep.get("target")
            dep_type = dep.get("type", "depends_on")
            context = dep.get("context")
        else:
            target = str(dep)
            dep_type = "depends_on"
            context = None

        if target and target in node_map:
            relation_type = self._map_dependency_type(dep_type)
            return CodeRelation(
                from_node_id=from_node_id,
                to_node_id=node_map[target],
                relation_type=relation_type,
                weight=1.0,
                context=context,
            )
        return None

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

    def _get_block_type_value(self, block: Any) -> str:
        """블록의 타입 값을 문자열로 가져오기"""
        block_type = block.block_type
        return block_type.value if hasattr(block_type, "value") else str(block_type)

    def _add_relation_safe(self, graph: CodeGraph, relation: CodeRelation) -> None:
        """안전하게 관계 추가 (예외 무시)"""
        try:
            graph.add_relation(relation)
        except ValueError:
            # 존재하지 않는 노드에 대한 관계는 무시
            pass

    def _generate_node_id_from_dict(self, block_data: dict[str, Any]) -> str:
        """딕셔너리에서 노드 ID 생성"""
        file_path = block_data.get("file_path", "unknown.py")
        name = block_data.get("name", "unknown")
        return f"{file_path}:{name}"
