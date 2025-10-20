"""
Neo4j 데이터베이스 지속성 계층

코드 그래프의 저장, 조회, 벡터 인덱스 관리를 담당합니다.
Graph 패키지의 모델 데이터를 Neo4j에 저장하고 검색하는 기능을 제공합니다.
"""

import logging
import os
from typing import Any

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from .models import CodeGraph, CodeNode, CodeRelation

logger = logging.getLogger(__name__)


class Neo4jPersistence:
    """Neo4j 데이터베이스 연결 및 그래프 지속성 관리"""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        """Neo4j 지속성 계층 초기화

        Args:
            uri: Neo4j 데이터베이스 URI (환경변수 NEO4J_URI 사용 가능)
            user: 사용자명 (환경변수 NEO4J_USER 사용 가능)
            password: 패스워드 (환경변수 NEO4J_PASSWORD 사용 가능)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")

        self.driver: Driver | None = None
        self.logger = logger

    def connect(self) -> bool:
        """Neo4j 데이터베이스에 연결

        Returns:
            bool: 연결 성공 여부
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # 연결 테스트
            with self.driver.session() as session:
                session.run("RETURN 1")

            self.logger.info(f"✅ Neo4j 연결 성공: {self.uri}")
            return True

        except (ServiceUnavailable, AuthError) as e:
            self.logger.error(f"❌ Neo4j 연결 실패: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ 예상치 못한 오류: {e}")
            return False

    def close(self) -> None:
        """연결 종료"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j 연결 종료")

    def create_constraints_and_indexes(self) -> bool:
        """데이터베이스 제약 조건 및 인덱스 생성

        Returns:
            bool: 성공 여부
        """
        if not self.driver:
            self.logger.error("데이터베이스에 연결되지 않음")
            return False

        try:
            with self.driver.session() as session:
                # 코드 노드 제약 조건
                constraints = [
                    "CREATE CONSTRAINT code_node_id IF NOT EXISTS FOR (n:CodeNode) REQUIRE n.id IS UNIQUE",
                    "CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE",
                ]

                # 기본 인덱스
                indexes = [
                    "CREATE INDEX code_node_type IF NOT EXISTS FOR (n:CodeNode) ON (n.type)",
                    "CREATE INDEX code_node_file IF NOT EXISTS FOR (n:CodeNode) ON (n.file_path)",
                    "CREATE TEXT INDEX code_node_content IF NOT EXISTS FOR (n:CodeNode) ON (n.source_code)",
                ]

                # 벡터 인덱스 (코드 임베딩용)
                vector_indexes = [
                    """
                    CREATE VECTOR INDEX code_embedding_index IF NOT EXISTS 
                    FOR (n:CodeNode) ON (n.embedding) 
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 384,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                    """
                ]

                for constraint in constraints:
                    try:
                        session.run(constraint)
                        self.logger.info(f"✅ 제약 조건 생성: {constraint[:50]}...")
                    except Exception as e:
                        self.logger.warning(f"⚠️ 제약 조건 생성 실패: {e}")

                for index in indexes + vector_indexes:
                    try:
                        session.run(index)
                        self.logger.info(f"✅ 인덱스 생성: {index[:50]}...")
                    except Exception as e:
                        self.logger.warning(f"⚠️ 인덱스 생성 실패: {e}")

            return True

        except Exception as e:
            self.logger.error(f"❌ 제약 조건/인덱스 생성 실패: {e}")
            return False

    def save_code_graph(
        self, graph: CodeGraph, project_name: str | None = None
    ) -> bool:
        """전체 코드 그래프를 Neo4j에 저장

        Args:
            graph: 저장할 CodeGraph 객체
            project_name: 프로젝트 이름 (None이면 graph.project_name 사용)

        Returns:
            bool: 저장 성공 여부
        """
        if not self.driver:
            self.logger.error("데이터베이스에 연결되지 않음")
            return False

        project_name = project_name or graph.project_name

        try:
            # 1단계: 프로젝트 정보 저장
            if not self._save_project_info(graph, project_name):
                return False

            # 2단계: 노드 저장
            if not self._save_code_nodes(list(graph.nodes.values()), project_name):
                return False

            # 3단계: 관계 저장
            if not self._save_code_relations(graph.relations):
                return False

            self.logger.info(
                f"✅ 그래프 저장 완료: {len(graph.nodes)}개 노드, "
                f"{len(graph.relations)}개 관계"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ 그래프 저장 실패: {e}")
            return False

    def _save_project_info(self, graph: CodeGraph, project_name: str) -> bool:
        """프로젝트 정보 저장 (내부 메서드)"""
        try:
            with self.driver.session() as session:
                project_query = """
                MERGE (p:Project {name: $name})
                SET p.path = $path,
                    p.total_files = $total_files,
                    p.total_lines = $total_lines,
                    p.total_nodes = $total_nodes,
                    p.total_relations = $total_relations,
                    p.analysis_version = $analysis_version,
                    p.created_at = $created_at,
                    p.updated_at = datetime()
                RETURN p
                """

                stats = graph.get_statistics()
                session.run(
                    project_query,
                    name=project_name,
                    path=graph.project_path,
                    total_files=graph.total_files,
                    total_lines=graph.total_lines,
                    total_nodes=stats["total_nodes"],
                    total_relations=stats["total_relations"],
                    analysis_version=graph.analysis_version,
                    created_at=graph.created_at.isoformat(),
                )
                return True

        except Exception as e:
            self.logger.error(f"❌ 프로젝트 정보 저장 실패: {e}")
            return False

    def _save_code_nodes(self, nodes: list[CodeNode], project_name: str) -> bool:
        """코드 노드들을 배치로 저장 (내부 메서드)"""
        try:
            with self.driver.session() as session:
                node_query = """
                UNWIND $nodes AS node_data
                MERGE (n:CodeNode {id: node_data.id})
                SET n.name = node_data.name,
                    n.type = node_data.type,
                    n.file_path = node_data.file_path,
                    n.start_line = node_data.start_line,
                    n.end_line = node_data.end_line,
                    n.source_code = node_data.source_code,
                    n.docstring = node_data.docstring,
                    n.complexity = node_data.complexity,
                    n.scope_level = node_data.scope_level,
                    n.embedding = node_data.embedding,
                    n.embedding_model = node_data.embedding_model,
                    n.created_at = node_data.created_at,
                    n.updated_at = datetime()
                
                WITH n, node_data
                MATCH (p:Project {name: $project_name})
                MERGE (p)-[:CONTAINS]->(n)
                """

                # Neo4j 형식으로 변환
                neo4j_nodes = [node.to_neo4j_node() for node in nodes]

                session.run(node_query, nodes=neo4j_nodes, project_name=project_name)
                self.logger.info(f"✅ {len(nodes)}개 코드 노드 저장 완료")
                return True

        except Exception as e:
            self.logger.error(f"❌ 코드 노드 저장 실패: {e}")
            return False

    def _save_code_relations(self, relations: list[CodeRelation]) -> bool:
        """코드 관계들을 배치로 저장 (내부 메서드)"""
        try:
            with self.driver.session() as session:
                # Cypher에서 동적 관계 생성을 위해 CALL apoc.create.relationship 사용
                # 또는 더 간단한 방식으로 사전 정의된 관계 타입만 사용 가능
                relation_query = """
                UNWIND $relations AS rel_data
                MATCH (from:CodeNode {id: rel_data.from_node_id})
                MATCH (to:CodeNode {id: rel_data.to_node_id})
                CALL apoc.create.relationship(from, rel_data.relation_type, {
                    weight: rel_data.weight,
                    line_number: rel_data.line_number,
                    context: rel_data.context,
                    created_at: rel_data.created_at
                }, to) YIELD rel
                RETURN count(rel)
                """

                # Neo4j 형식으로 변환
                neo4j_relations = [rel.to_neo4j_relation() for rel in relations]

                session.run(relation_query, relations=neo4j_relations)
                self.logger.info(f"✅ {len(relations)}개 코드 관계 저장 완료")
                return True

        except Exception as e:
            # APOC 라이브러리가 없을 수 있으므로 폴백
            self.logger.warning(f"⚠️ APOC 사용 실패, 대체 방식으로 관계 저장: {e}")
            return self._save_code_relations_fallback(relations)

    def _save_code_relations_fallback(self, relations: list[CodeRelation]) -> bool:
        """APOC 없이 관계 저장 (폴백 메서드)"""
        try:
            with self.driver.session() as session:
                for rel in relations:
                    # 사전 정의된 관계 타입으로만 생성
                    rel_type = (
                        rel.relation_type.value
                        if hasattr(rel.relation_type, "value")
                        else str(rel.relation_type)
                    )

                    query = f"""
                    MATCH (from:CodeNode {{id: $from_id}})
                    MATCH (to:CodeNode {{id: $to_id}})
                    CREATE (from)-[:{rel_type} {{
                        weight: $weight,
                        line_number: $line_number,
                        context: $context,
                        created_at: $created_at
                    }}]->(to)
                    """

                    session.run(
                        query,
                        from_id=rel.from_node_id,
                        to_id=rel.to_node_id,
                        weight=rel.weight,
                        line_number=rel.line_number,
                        context=rel.context,
                        created_at=rel.created_at.isoformat(),
                    )

                self.logger.info(f"✅ {len(relations)}개 코드 관계 저장 완료 (폴백)")
                return True

        except Exception as e:
            self.logger.error(f"❌ 관계 저장 폴백 실패: {e}")
            return False

    def update_node_embedding(
        self, node_id: str, embedding: list[float], model: str
    ) -> bool:
        """노드의 임베딩 벡터 업데이트

        Args:
            node_id: 업데이트할 노드 ID
            embedding: 임베딩 벡터
            model: 임베딩 모델명

        Returns:
            bool: 성공 여부
        """
        if not self.driver:
            self.logger.error("데이터베이스에 연결되지 않음")
            return False

        try:
            with self.driver.session() as session:
                query = """
                MATCH (n:CodeNode {id: $node_id})
                SET n.embedding = $embedding,
                    n.embedding_model = $model,
                    n.updated_at = datetime()
                RETURN n
                """

                result = session.run(
                    query, node_id=node_id, embedding=embedding, model=model
                )

                if result.single():
                    self.logger.info(f"✅ 임베딩 업데이트: {node_id}")
                    return True
                else:
                    self.logger.warning(f"⚠️ 노드를 찾을 수 없음: {node_id}")
                    return False

        except Exception as e:
            self.logger.error(f"❌ 임베딩 업데이트 실패: {e}")
            return False

    def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """벡터 유사도 기반 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            limit: 반환할 최대 결과 수
            similarity_threshold: 유사도 임계값

        Returns:
            list[dict]: 유사한 노드들의 정보 리스트
        """
        if not self.driver:
            self.logger.error("데이터베이스에 연결되지 않음")
            return []

        try:
            with self.driver.session() as session:
                query = """
                CALL db.index.vector.queryNodes('code_embedding_index', $limit, $query_embedding)
                YIELD node, score
                WHERE score >= $similarity_threshold
                RETURN node.id AS id,
                       node.name AS name,
                       node.type AS type,
                       node.file_path AS file_path,
                       node.source_code AS source_code,
                       node.docstring AS docstring,
                       score
                ORDER BY score DESC
                """

                result = session.run(
                    query,
                    query_embedding=query_embedding,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                )

                return [record.data() for record in result]

        except Exception as e:
            self.logger.error(f"❌ 벡터 검색 실패: {e}")
            return []

    def get_node_context(self, node_id: str, depth: int = 2) -> dict[str, Any]:
        """노드와 주변 컨텍스트 조회

        Args:
            node_id: 조회할 노드 ID
            depth: 주변 관계 깊이

        Returns:
            dict: 노드 정보와 관련 노드들
        """
        if not self.driver:
            self.logger.error("데이터베이스에 연결되지 않음")
            return {"center_node": None, "related_nodes": [], "relationships": []}

        try:
            with self.driver.session() as session:
                query = """
                MATCH (center:CodeNode {id: $node_id})
                OPTIONAL MATCH path = (center)-[*1..$depth]-(related:CodeNode)
                WITH center, collect(DISTINCT related) AS related_nodes,
                     collect(DISTINCT relationships(path)) AS all_relationships
                
                RETURN center,
                       related_nodes,
                       [r IN all_relationships | {
                           type: type(r),
                           start_node: startNode(r).id,
                           end_node: endNode(r).id,
                           properties: properties(r)
                       }] AS relationships
                """

                result = session.run(query, node_id=node_id, depth=depth)
                record = result.single()

                if record:
                    return {
                        "center_node": dict(record["center"]),
                        "related_nodes": [
                            dict(node) for node in record["related_nodes"]
                        ],
                        "relationships": record["relationships"],
                    }
                else:
                    return {
                        "center_node": None,
                        "related_nodes": [],
                        "relationships": [],
                    }

        except Exception as e:
            self.logger.error(f"❌ 노드 컨텍스트 조회 실패: {e}")
            return {"center_node": None, "related_nodes": [], "relationships": []}

    def get_database_statistics(self) -> dict[str, Any]:
        """데이터베이스 통계 정보 조회

        Returns:
            dict: 노드, 관계, 타입별 통계
        """
        if not self.driver:
            self.logger.error("데이터베이스에 연결되지 않음")
            return {}

        try:
            with self.driver.session() as session:
                # 노드 수 조회
                node_stats = session.run(
                    """
                    MATCH (n:CodeNode)
                    RETURN n.type AS type, count(n) AS count
                    """
                ).data()

                # 관계 수 조회
                rel_stats = session.run(
                    """
                    MATCH ()-[r]->()
                    RETURN type(r) AS type, count(r) AS count
                    """
                ).data()

                # 전체 통계
                total_stats = session.run(
                    """
                    MATCH (n)
                    OPTIONAL MATCH ()-[r]->()
                    RETURN count(DISTINCT n) AS total_nodes,
                           count(r) AS total_relationships
                    """
                ).single()

                return {
                    "total_nodes": total_stats["total_nodes"],
                    "total_relationships": total_stats["total_relationships"],
                    "node_types": {stat["type"]: stat["count"] for stat in node_stats},
                    "relation_types": {
                        stat["type"]: stat["count"] for stat in rel_stats
                    },
                }

        except Exception as e:
            self.logger.error(f"❌ 통계 조회 실패: {e}")
            return {}

    def clear_project_data(self, project_name: str) -> bool:
        """프로젝트 데이터 삭제

        Args:
            project_name: 삭제할 프로젝트명

        Returns:
            bool: 성공 여부
        """
        if not self.driver:
            self.logger.error("데이터베이스에 연결되지 않음")
            return False

        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:Project {name: $project_name})
                OPTIONAL MATCH (p)-[:CONTAINS]->(n:CodeNode)
                DETACH DELETE p, n
                """

                session.run(query, project_name=project_name)
                self.logger.info(f"✅ 프로젝트 데이터 삭제: {project_name}")
                return True

        except Exception as e:
            self.logger.error(f"❌ 프로젝트 데이터 삭제 실패: {e}")
            return False

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()
