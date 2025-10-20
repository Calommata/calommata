"""Core 패키지 테스트"""

import pytest
import os
from unittest.mock import Mock, patch

from src.config import CoreConfig, Neo4jConfig, EmbeddingConfig
from src.embedder import CodeEmbedder


class TestCoreConfig:
    """설정 테스트"""

    def test_default_config(self):
        """기본 설정 생성 테스트"""
        config = CoreConfig()
        assert config.project_name == "code-analyzer"
        assert config.neo4j.batch_size == 500
        assert config.embedding.normalize is True

    def test_config_from_env(self):
        """환경 변수에서 설정 로드 테스트"""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://test:7687",
                "NEO4J_USER": "testuser",
                "GOOGLE_API_KEY": "test-key",
            },
        ):
            config = CoreConfig.from_env()
            assert config.neo4j.uri == "bolt://test:7687"
            assert config.neo4j.user == "testuser"
            assert config.llm.api_key == "test-key"

    def test_config_to_dict(self):
        """설정을 딕셔너리로 변환 테스트"""
        config = CoreConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "neo4j" in config_dict
        assert "embedding" in config_dict
        assert "retriever" in config_dict
        assert "llm" in config_dict


class TestCodeEmbedder:
    """임베딩 테스트"""

    def test_embedder_initialization(self):
        """임베딩 모델 초기화 테스트"""
        embedder = CodeEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder._embeddings is not None

    def test_embed_code(self):
        """단일 코드 임베딩 테스트"""
        embedder = CodeEmbedder()

        code = """
def hello_world():
    print("Hello, World!")
"""

        embedding = embedder.embed_code(code)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_codes_batch(self):
        """배치 임베딩 테스트"""
        embedder = CodeEmbedder()

        codes = [
            "def func1(): pass",
            "def func2(): pass",
            "class MyClass: pass",
        ]

        embeddings = embedder.embed_codes(codes)

        assert len(embeddings) == len(codes)
        assert all(isinstance(emb, list) for emb in embeddings)

    def test_embedding_dimension(self):
        """임베딩 차원 확인 테스트"""
        embedder = CodeEmbedder()

        dimension = embedder.get_embedding_dimension()

        assert isinstance(dimension, int)
        assert dimension > 0

    def test_preprocess_code(self):
        """코드 전처리 테스트"""
        embedder = CodeEmbedder()

        # 빈 코드
        assert embedder._preprocess_code("") == ""

        # 과도한 빈 줄
        code_with_many_empty_lines = "\n\n\n\ndef func():\n\n\n\n    pass"
        processed = embedder._preprocess_code(code_with_many_empty_lines)

        # 연속된 빈 줄이 2개로 줄어들었는지 확인
        assert processed.count("\n\n\n") == 0

    def test_long_code_truncation(self):
        """긴 코드 잘라내기 테스트"""
        embedder = CodeEmbedder()

        # 2000자 넘는 코드
        long_code = "# " + "x" * 2500
        processed = embedder._preprocess_code(long_code)

        assert len(processed) <= 2020  # 2000 + "# ... (truncated)"


class TestCodeSearchResult:
    """검색 결과 테스트"""

    def test_search_result_creation(self):
        """검색 결과 생성 테스트"""
        from src.retriever import CodeSearchResult

        result = CodeSearchResult(
            node_id="test-id",
            name="test_function",
            node_type="Function",
            file_path="/test/file.py",
            source_code="def test(): pass",
            similarity_score=0.85,
        )

        assert result.name == "test_function"
        assert result.similarity_score == 0.85

    def test_to_context_string(self):
        """컨텍스트 문자열 변환 테스트"""
        from src.retriever import CodeSearchResult

        result = CodeSearchResult(
            node_id="test-id",
            name="test_function",
            node_type="Function",
            file_path="/test/file.py",
            source_code="def test(): pass",
            docstring="Test function",
            similarity_score=0.85,
            related_nodes=[
                {"name": "related1", "type": "Function"},
                {"name": "related2", "type": "Class"},
            ],
        )

        context = result.to_context_string()

        assert "test_function" in context
        assert "Test function" in context
        assert "def test(): pass" in context
        assert "related1" in context
        assert "0.85" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
