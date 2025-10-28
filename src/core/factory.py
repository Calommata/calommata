"""간편한 초기화를 위한 팩토리 함수들"""

import logging

from src.graph import Neo4jPersistence

from src.core.config import CoreConfig
from src.core.embedding.code_embedder import CodeEmbedder
from src.core.code_retriever import CodeRetriever
from src.core.graph import CodeGraphService
from src.core.agent import CodeRAGAgent

logger = logging.getLogger(__name__)


def create_from_config(
    config: CoreConfig | None = None,
) -> tuple[
    Neo4jPersistence,
    CodeEmbedder,
    CodeRetriever,
    CodeGraphService,
    CodeRAGAgent,
]:
    """설정에서 모든 컴포넌트 생성

    Args:
        config: 설정 객체 (None이면 환경 변수에서 로드)

    Returns:
        (persistence, embedder, retriever, graph_service, agent) 튜플
    """
    if config is None:
        config = CoreConfig.from_env()

    logger.info("컴포넌트 초기화 시작")

    # 1. Neo4j 연결
    persistence = Neo4jPersistence(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password,
        batch_size=config.neo4j.batch_size,
    )
    persistence.connect()
    persistence.create_constraints_and_indexes()
    logger.info("✅ Neo4j 연결 완료")

    # 2. 임베딩 모델
    embedder = CodeEmbedder(
        provider=config.embedding.provider,
        model_name=config.embedding.model_name,
        ollama_base_url=config.embedding.ollama_base_url,
        model_kwargs={"device": config.embedding.device},
        encode_kwargs={"normalize_embeddings": config.embedding.normalize},
    )
    logger.info("✅ 임베딩 모델 초기화 완료")

    # 3. 리트리버
    retriever = CodeRetriever(
        persistence=persistence,
        similarity_threshold=config.retriever.similarity_threshold,
        max_results=config.retriever.max_results,
        context_depth=config.retriever.context_depth,
    )
    logger.info("✅ 리트리버 초기화 완료")

    # 4. 그래프 서비스
    graph_service = CodeGraphService(
        persistence=persistence,
        embedder=embedder,
        project_name=config.project_name,
    )
    logger.info("✅ 그래프 서비스 초기화 완료")

    # 5. AI Agent
    agent = CodeRAGAgent(
        embedder=embedder,
        retriever=retriever,
        llm_api_key=config.llm.api_key,
        model_name=config.llm.model_name,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )
    logger.info("✅ AI Agent 초기화 완료")

    logger.info("모든 컴포넌트 초기화 완료!")

    return persistence, embedder, retriever, graph_service, agent
