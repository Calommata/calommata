"""의존성 기반 관계 생성 전문 클래스

의존성에서 CALLS, IMPORTS 관계를 생성하는 책임만 담당합니다.
"""

import logging
from typing import Any

from .models import CodeGraph, CodeRelation, RelationType

logger = logging.getLogger(__name__)


class DependencyRelationBuilder:
    """의존성 기반 관계 생성 전문 클래스"""

    def build(
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
                relation = self._create_relation(block, dep_target, node_map)
                if relation:
                    self._add_relation_safe(graph, relation)

    def _create_relation(
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

    @staticmethod
    def _add_relation_safe(graph: CodeGraph, relation: CodeRelation) -> None:
        """안전하게 관계 추가 (예외 무시)"""
        try:
            graph.add_relation(relation)
        except ValueError:
            pass
