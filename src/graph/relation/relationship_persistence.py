"""코드 관계 지속성 모듈

관계 저장 및 관리를 담당합니다.
"""

import logging

from neo4j import Driver

from src.graph.db import CodeRelation, Neo4jQueries

logger = logging.getLogger(__name__)


class RelationshipPersistence:
    """코드 관계 저장"""

    DEFAULT_BATCH_SIZE = 500

    def __init__(self, driver: Driver, batch_size: int = DEFAULT_BATCH_SIZE):
        """초기화

        Args:
            driver: Neo4j 드라이버
            batch_size: 배치 크기
        """
        self.driver = driver
        self.batch_size = batch_size
        self._queries = Neo4jQueries()

    def save_relations_batch(self, relations: list[CodeRelation]) -> None:
        """코드 관계들을 배치로 저장

        Args:
            relations: 저장할 관계 리스트
        """
        if not relations:
            logger.warning("⚠️ 저장할 관계가 없습니다")
            return

        try:
            # 관계 타입별로 그룹화
            relations_by_type: dict[str, list[CodeRelation]] = {}
            for rel in relations:
                rel_type_value = (
                    rel.relation_type.value
                    if hasattr(rel.relation_type, "value")
                    else rel.relation_type
                )
                rel_type = str(rel_type_value)
                if rel_type not in relations_by_type:
                    relations_by_type[rel_type] = []
                relations_by_type[rel_type].append(rel)

            # 각 타입별로 배치 저장
            total_saved = 0
            for rel_type, type_relations in relations_by_type.items():
                self._save_relations_of_type(rel_type, type_relations)
                total_saved += len(type_relations)

            logger.info(f"✅ 총 {total_saved}개 코드 관계 저장 완료")

        except Exception as e:
            logger.error(f"❌ 코드 관계 배치 저장 실패: {e}")
            raise

    def _save_relations_of_type(
        self, relation_type: str, relations: list[CodeRelation]
    ) -> None:
        """특정 타입의 관계들을 배치로 저장

        Args:
            relation_type: 관계 타입
            relations: 해당 타입의 관계 리스트
        """
        total_batches = (len(relations) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(relations))
            batch = relations[start_idx:end_idx]

            query = self._queries.create_relation_query(relation_type)

            with self.driver.session() as session:
                for rel in batch:
                    session.run(
                        query,  # type: ignore[arg-type]
                        from_id=rel.from_node_id,
                        to_id=rel.to_node_id,
                        weight=rel.weight,
                        context=rel.context,
                        created_at=rel.created_at.isoformat(),
                    )

            logger.info(
                f"✅ {relation_type} 관계 배치 {batch_idx + 1}/{total_batches} "
                f"저장 완료 ({len(batch)}개)"
            )
