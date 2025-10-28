"""딕셔너리 기반 관계 생성 전문 클래스

딕셔너리 데이터에서 관계를 생성하는 책임만 담당합니다.
"""

import logging
from typing import Any

from src.graph.db.models import CodeRelation, RelationType

logger = logging.getLogger(__name__)


class DictRelationBuilder:
    """딕셔너리 기반 관계 생성 전문 클래스"""

    def build(
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
            rel = self._create_dependency_relation(dep, from_node_id, node_map)
            if rel:
                relations.append(rel)

        return relations

    @staticmethod
    def _create_dependency_relation(
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
            relation_type = DictRelationBuilder._map_dependency_type(dep_type)
            return CodeRelation(
                from_node_id=from_node_id,
                to_node_id=node_map[target],
                relation_type=relation_type,
                weight=1.0,
                context=context,
            )
        return None

    @staticmethod
    def _map_dependency_type(dep_type: str) -> RelationType:
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

    @staticmethod
    def _generate_node_id_from_dict(block_data: dict[str, Any]) -> str:
        """딕셔너리에서 노드 ID 생성"""
        file_path = block_data.get("file_path", "unknown.py")
        name = block_data.get("name", "unknown")
        return f"{file_path}:{name}"
