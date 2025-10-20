"""
Neo4j 그래프 데이터베이스 핸들러
코드 그래프의 저장, 조회, 벡터 인덱스 관리를 담당
"""

import os
import logging
from typing import Any, Optional
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError


class Neo4jHandler:
    """Neo4j 데이터베이스 연결 및 그래프 작업 처리"""

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
    ):
        """Neo4j 핸들러 초기화"""
        self.uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")

        self.driver: Optional[Driver] = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        """Neo4j 데이터베이스 연결"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # 연결 테스트
            with self.driver.session() as session:
                session.run("RETURN 1")

            self.logger.info(f"Neo4j 연결 성공: {self.uri}")
            return True

        except (ServiceUnavailable, AuthError) as e:
            self.logger.error(f"Neo4j 연결 실패: {e}")
            return False
        except Exception as e:
            self.logger.error(f"예상치 못한 오류: {e}")
            return False

    def close(self):
        """연결 종료"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j 연결 종료")

    def create_constraints_and_indexes(self):
        """제약 조건 및 인덱스 생성"""
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
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                    self.logger.info(f"제약 조건 생성: {constraint}")
                except Exception as e:
                    self.logger.warning(f"제약 조건 생성 실패: {e}")

            for index in indexes + vector_indexes:
                try:
                    session.run(index)
                    self.logger.info(f"인덱스 생성: {index}")
                except Exception as e:
                    self.logger.warning(f"인덱스 생성 실패: {e}")

    def save_project(self, project_data: dict[str, Any]) -> bool:
        """프로젝트 정보 저장"""
        try:
            with self.driver.session() as session:
                # 프로젝트 노드 생성
                project_query = """
                MERGE (p:Project {name: $name})
                SET p.path = $path,
                    p.total_files = $total_files,
                    p.total_lines = $total_lines,
                    p.analysis_version = $analysis_version,
                    p.created_at = $created_at,
                    p.updated_at = $updated_at
                RETURN p
                """

                session.run(project_query, **project_data)
                self.logger.info(f"프로젝트 저장: {project_data['name']}")
                return True

        except Exception as e:
            self.logger.error(f"프로젝트 저장 실패: {e}")
            return False

    def save_code_nodes(self, nodes: list[dict[str, Any]], project_name: str) -> bool:
        """코드 노드들을 배치로 저장"""
        try:
            with self.driver.session() as session:
                # 노드 생성 쿼리
                node_query = """
                UNWIND $nodes AS node
                MERGE (n:CodeNode {id: node.id})
                SET n.name = node.name,
                    n.type = node.type,
                    n.file_path = node.file_path,
                    n.start_line = node.start_line,
                    n.end_line = node.end_line,
                    n.source_code = node.source_code,
                    n.docstring = node.docstring,
                    n.complexity = node.complexity,
                    n.scope_level = node.scope_level,
                    n.embedding = node.embedding,
                    n.embedding_model = node.embedding_model,
                    n.created_at = node.created_at,
                    n.updated_at = node.updated_at
                
                WITH n, node
                MATCH (p:Project {name: $project_name})
                MERGE (p)-[:CONTAINS]->(n)
                """

                # 임베딩이 없는 노드들은 null로 설정
                for node in nodes:
                    if "embedding" not in node or not node["embedding"]:
                        node["embedding"] = None

                session.run(node_query, nodes=nodes, project_name=project_name)
                self.logger.info(f"코드 노드 {len(nodes)}개 저장")
                return True

        except Exception as e:
            self.logger.error(f"코드 노드 저장 실패: {e}")
            return False

    def save_code_relations(self, relations: list[dict[str, Any]]) -> bool:
        """코드 관계들을 배치로 저장"""
        try:
            with self.driver.session() as session:
                # 관계 생성 쿼리
                relation_query = """
                UNWIND $relations AS rel
                MATCH (from:CodeNode {id: rel.from})
                MATCH (to:CodeNode {id: rel.to})
                CALL apoc.create.relationship(from, rel.type, {
                    weight: rel.weight,
                    line_number: rel.line_number,
                    context: rel.context,
                    created_at: rel.created_at
                }, to) YIELD rel as relationship
                RETURN count(relationship)
                """

                session.run(relation_query, relations=relations)
                self.logger.info(f"코드 관계 {len(relations)}개 저장")
                return True

        except Exception as e:
            self.logger.error(f"코드 관계 저장 실패: {e}")
            return False

    def update_node_embedding(
        self, node_id: str, embedding: list[float], model: str
    ) -> bool:
        """노드의 임베딩 벡터 업데이트"""
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
                    return True
                else:
                    self.logger.warning(f"노드를 찾을 수 없음: {node_id}")
                    return False

        except Exception as e:
            self.logger.error(f"임베딩 업데이트 실패: {e}")
            return False

    def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """벡터 유사도 검색"""
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
            self.logger.error(f"벡터 검색 실패: {e}")
            return []

    def get_node_context(self, node_id: str, depth: int = 2) -> dict[str, Any]:
        """노드와 주변 컨텍스트 조회"""
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
            self.logger.error(f"노드 컨텍스트 조회 실패: {e}")
            return {"center_node": None, "related_nodes": [], "relationships": []}

    def get_database_statistics(self) -> dict[str, Any]:
        """데이터베이스 통계 정보"""
        try:
            with self.driver.session() as session:
                # 노드 수 조회
                node_stats = session.run("""
                    MATCH (n:CodeNode)
                    RETURN n.type AS type, count(n) AS count
                """).data()

                # 관계 수 조회
                rel_stats = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) AS type, count(r) AS count
                """).data()

                # 전체 통계
                total_stats = session.run("""
                    MATCH (n)
                    OPTIONAL MATCH ()-[r]->()
                    RETURN count(DISTINCT n) AS total_nodes,
                           count(r) AS total_relationships
                """).single()

                return {
                    "total_nodes": total_stats["total_nodes"],
                    "total_relationships": total_stats["total_relationships"],
                    "node_types": {stat["type"]: stat["count"] for stat in node_stats},
                    "relation_types": {
                        stat["type"]: stat["count"] for stat in rel_stats
                    },
                }

        except Exception as e:
            self.logger.error(f"통계 조회 실패: {e}")
            return {}

    def clear_project_data(self, project_name: str) -> bool:
        """프로젝트 데이터 삭제"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:Project {name: $project_name})
                OPTIONAL MATCH (p)-[:CONTAINS]->(n:CodeNode)
                DETACH DELETE p, n
                """

                session.run(query, project_name=project_name)
                self.logger.info(f"프로젝트 데이터 삭제: {project_name}")
                return True

        except Exception as e:
            self.logger.error(f"프로젝트 데이터 삭제 실패: {e}")
            return False

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()
