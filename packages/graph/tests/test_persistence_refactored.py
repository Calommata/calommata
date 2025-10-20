"""
리팩토링된 Neo4j Persistence 계층 테스트

주요 개선 사항 테스트:
- 배치 처리 기능
- 예외 처리
- 타입 안전성
- 쿼리 분리
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.exceptions import (
    PersistenceError,
    ConnectionError as PersistenceConnectionError,
    NodeNotFoundError,
    InvalidDataError,
)
from src.models import CodeGraph, CodeNode, CodeRelation, NodeType, RelationType
from src.persistence import Neo4jPersistence
from src.queries import Neo4jQueries


class TestNeo4jPersistenceBasics:
    """기본 연결 및 설정 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        persistence = Neo4jPersistence(
            uri="bolt://test:7687",
            user="test_user",
            password="test_pass",
            batch_size=100,
        )

        assert persistence.uri == "bolt://test:7687"
        assert persistence.user == "test_user"
        assert persistence.password == "test_pass"
        assert persistence.batch_size == 100
        assert not persistence.is_connected

    def test_initialization_with_env(self):
        """환경 변수로 초기화 테스트"""
        with patch.dict(
            "os.environ",
            {
                "NEO4J_URI": "bolt://env:7687",
                "NEO4J_USER": "env_user",
                "NEO4J_PASSWORD": "env_pass",
            },
        ):
            persistence = Neo4jPersistence()

            assert persistence.uri == "bolt://env:7687"
            assert persistence.user == "env_user"
            assert persistence.password == "env_pass"

    def test_driver_property_not_connected(self):
        """연결되지 않은 상태에서 driver 접근 시 예외 발생"""
        persistence = Neo4jPersistence()

        with pytest.raises(PersistenceConnectionError, match="연결되지 않음"):
            _ = persistence.driver

    @patch("src.persistence.GraphDatabase")
    def test_connect_success(self, mock_graph_db):
        """연결 성공 테스트"""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        persistence = Neo4jPersistence()
        result = persistence.connect()

        assert result is True
        assert persistence.is_connected
        mock_graph_db.driver.assert_called_once()

    @patch("src.persistence.GraphDatabase")
    def test_connect_failure(self, mock_graph_db):
        """연결 실패 테스트"""
        from neo4j.exceptions import ServiceUnavailable

        mock_graph_db.driver.side_effect = ServiceUnavailable("Connection failed")

        persistence = Neo4jPersistence()

        with pytest.raises(PersistenceConnectionError, match="연결 실패"):
            persistence.connect()

        assert not persistence.is_connected

    def test_close(self):
        """연결 종료 테스트"""
        persistence = Neo4jPersistence()
        persistence._driver = MagicMock()

        persistence.close()

        persistence._driver.close.assert_called_once()
        assert not persistence.is_connected


class TestNeo4jPersistenceBatchProcessing:
    """배치 처리 테스트"""

    def test_batch_size_configuration(self):
        """배치 크기 설정 테스트"""
        persistence = Neo4jPersistence(batch_size=250)
        assert persistence.batch_size == 250

    @patch("src.persistence.GraphDatabase")
    def test_save_nodes_with_batching(self, mock_graph_db):
        """노드 배치 저장 테스트"""
        # 모의 드라이버 설정
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        persistence = Neo4jPersistence(batch_size=2)
        persistence.connect()

        # 테스트 노드 생성 (5개 - 3개 배치로 나뉨)
        nodes = [
            CodeNode(
                id=f"node_{i}",
                name=f"Node{i}",
                node_type=NodeType.FUNCTION,
                file_path="/test.py",
                start_line=i,
                end_line=i + 10,
            )
            for i in range(5)
        ]

        # 배치 저장 실행
        persistence._save_code_nodes_batch(nodes, "test_project")

        # 세션이 3번 생성되었는지 확인 (배치 1, 2, 3)
        assert mock_driver.session.call_count == 3


class TestNeo4jPersistenceGraphOperations:
    """그래프 저장 및 조회 테스트"""

    @patch("src.persistence.GraphDatabase")
    def test_save_code_graph_success(self, mock_graph_db):
        """코드 그래프 저장 성공 테스트"""
        # 모의 드라이버 설정
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        persistence = Neo4jPersistence()
        persistence.connect()

        # 테스트 그래프 생성
        graph = CodeGraph(
            project_name="test_project",
            project_path="/test/path",
            total_files=1,
            total_lines=100,
        )

        # 노드 추가
        node1 = CodeNode(
            id="node1",
            name="TestFunc",
            node_type=NodeType.FUNCTION,
            file_path="/test.py",
            start_line=1,
            end_line=10,
        )
        graph.add_node(node1)

        # 그래프 저장
        result = persistence.save_code_graph(graph)

        assert result is True

    @patch("src.persistence.GraphDatabase")
    def test_save_code_graph_no_project_name(self, mock_graph_db):
        """프로젝트 이름 없이 저장 시 예외 발생"""
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver

        persistence = Neo4jPersistence()
        persistence.connect()

        # 프로젝트 이름 없는 그래프
        graph = CodeGraph(project_name="", project_path="/test")

        with pytest.raises(InvalidDataError, match="프로젝트 이름이 필요"):
            persistence.save_code_graph(graph, project_name=None)


class TestNeo4jPersistenceEmbedding:
    """임베딩 관련 테스트"""

    @patch("src.persistence.GraphDatabase")
    def test_update_node_embedding_success(self, mock_graph_db):
        """임베딩 업데이트 성공 테스트"""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"id": "node1"}

        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_graph_db.driver.return_value = mock_driver

        persistence = Neo4jPersistence()
        persistence.connect()

        embedding = [0.1, 0.2, 0.3]
        result = persistence.update_node_embedding("node1", embedding, "test-model")

        assert result is True

    @patch("src.persistence.GraphDatabase")
    def test_update_node_embedding_not_found(self, mock_graph_db):
        """존재하지 않는 노드 임베딩 업데이트 시 예외"""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = None

        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_graph_db.driver.return_value = mock_driver

        persistence = Neo4jPersistence()
        persistence.connect()

        with pytest.raises(NodeNotFoundError, match="찾을 수 없음"):
            persistence.update_node_embedding("nonexistent", [0.1], "model")


class TestNeo4jPersistenceQueries:
    """쿼리 관련 테스트"""

    def test_queries_class_exists(self):
        """Neo4jQueries 클래스 확인"""
        assert hasattr(Neo4jQueries, "CONSTRAINTS")
        assert hasattr(Neo4jQueries, "INDEXES")
        assert hasattr(Neo4jQueries, "VECTOR_INDEX")
        assert hasattr(Neo4jQueries, "MERGE_PROJECT")

    def test_create_relation_query(self):
        """동적 관계 쿼리 생성 테스트"""
        query = Neo4jQueries.create_relation_query("CALLS")
        assert "CALLS" in query
        assert "MATCH" in query
        assert "MERGE" in query or "CREATE" in query


class TestNeo4jPersistenceContextManager:
    """컨텍스트 매니저 테스트"""

    @patch("src.persistence.GraphDatabase")
    def test_context_manager_success(self, mock_graph_db):
        """컨텍스트 매니저 정상 동작 테스트"""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        with Neo4jPersistence() as persistence:
            assert persistence.is_connected

    @patch("src.persistence.GraphDatabase")
    def test_context_manager_close_on_exit(self, mock_graph_db):
        """컨텍스트 매니저 종료 시 연결 종료 확인"""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        persistence = Neo4jPersistence()

        with persistence:
            pass

        mock_driver.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
