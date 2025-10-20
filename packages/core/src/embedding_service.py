"""
LangChain 기반 코드 임베딩 서비스
다양한 임베딩 제공자를 통합하여 코드 블록을 벡터로 변환
"""

import os
import logging
from typing import Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    """지원되는 임베딩 제공자"""

    OPENAI = "openai"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


@dataclass
class EmbeddingConfig:
    """임베딩 설정"""

    provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384  # HuggingFace 기본 모델 차원
    chunk_size: int = 512  # 토큰 제한
    api_key: Optional[str] = None
    batch_size: int = 32
    normalize_embeddings: bool = True


class EmbeddingService:
    """LangChain 기반 코드 임베딩 생성 서비스"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.logger = logging.getLogger(__name__)
        self.embeddings: Optional[Embeddings] = None

        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """임베딩 모델 초기화"""
        try:
            if self.config.provider == EmbeddingProvider.OPENAI:
                self.embeddings = self._init_openai_embeddings()
            elif self.config.provider == EmbeddingProvider.GOOGLE:
                self.embeddings = self._init_google_embeddings()
            elif self.config.provider == EmbeddingProvider.HUGGINGFACE:
                self.embeddings = self._init_huggingface_embeddings()
            else:
                raise ValueError(f"지원되지 않는 임베딩 제공자: {self.config.provider}")

            if self.embeddings:
                self.logger.info(
                    f"✅ 임베딩 모델 초기화 완료: {self.config.provider.value} - {self.config.model_name}"
                )

        except Exception as e:
            self.logger.error(f"❌ 임베딩 모델 초기화 실패: {e}")
            self.embeddings = None

    def _init_openai_embeddings(self) -> Optional[Embeddings]:
        """OpenAI 임베딩 초기화"""
        try:
            from langchain_openai import OpenAIEmbeddings

            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.error("OpenAI API 키가 설정되지 않았습니다")
                return None

            return OpenAIEmbeddings(
                model=self.config.model_name,
                openai_api_key=api_key,
                dimensions=self.config.dimensions
                if "text-embedding-3" in self.config.model_name
                else None,
                chunk_size=self.config.chunk_size,
            )

        except ImportError:
            self.logger.error("langchain-openai가 설치되지 않았습니다")
            return None

    def _init_google_embeddings(self) -> Optional[Embeddings]:
        """Google 임베딩 초기화"""
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            api_key = (
                self.config.api_key
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("GEMINI_API_KEY")
            )
            if not api_key:
                self.logger.error("Google API 키가 설정되지 않았습니다")
                return None

            return GoogleGenerativeAIEmbeddings(
                model=self.config.model_name, google_api_key=api_key
            )

        except ImportError:
            self.logger.error("langchain-google-genai가 설치되지 않았습니다")
            return None

    def _init_huggingface_embeddings(self) -> Optional[Embeddings]:
        """HuggingFace 임베딩 초기화"""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": self.config.normalize_embeddings,
                    "batch_size": self.config.batch_size,
                },
            )

        except ImportError:
            self.logger.error("langchain-community가 설치되지 않았습니다")
            return None

    def create_code_embedding(
        self, source_code: str, docstring: str = None
    ) -> Optional[List[float]]:
        """코드 블록에 대한 임베딩 생성"""
        if not self.embeddings:
            self.logger.error("임베딩 모델이 초기화되지 않았습니다")
            return None

        try:
            # 임베딩할 텍스트 준비
            text_to_embed = self._prepare_code_text(source_code, docstring)

            # 임베딩 생성
            embedding = self.embeddings.embed_query(text_to_embed)
            return embedding

        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            return None

    def create_query_embedding(self, query: str) -> Optional[List[float]]:
        """검색 쿼리에 대한 임베딩 생성"""
        if not self.embeddings:
            return None

        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            self.logger.error(f"쿼리 임베딩 생성 실패: {e}")
            return None

    def batch_create_embeddings(
        self, code_blocks: List[Dict[str, str]]
    ) -> List[Optional[List[float]]]:
        """여러 코드 블록에 대한 임베딩을 배치로 생성"""
        if not self.embeddings:
            return [None] * len(code_blocks)

        try:
            # 텍스트 준비
            texts = []
            for block in code_blocks:
                source_code = block.get("source_code", "")
                docstring = block.get("docstring", "")
                text = self._prepare_code_text(source_code, docstring)
                texts.append(text)

            # 배치 임베딩 생성
            embeddings = self.embeddings.embed_documents(texts)

            self.logger.info(f"배치 임베딩 생성 완료: {len(embeddings)}개")
            return embeddings

        except Exception as e:
            self.logger.error(f"배치 임베딩 생성 실패: {e}")
            return [None] * len(code_blocks)

    def _prepare_code_text(self, source_code: str, docstring: str = None) -> str:
        """코드를 임베딩하기 위한 텍스트로 준비"""
        parts = []

        # docstring이 있으면 먼저 추가
        if docstring and docstring.strip():
            parts.append(f"Documentation: {docstring.strip()}")

        # 코드 추가
        if source_code and source_code.strip():
            # 코드에서 불필요한 공백 제거
            cleaned_code = "\n".join(line.rstrip() for line in source_code.split("\n"))
            parts.append(f"Code: {cleaned_code}")

        text = "\n\n".join(parts)

        # 토큰 제한 확인 (대략적으로 4자 = 1토큰으로 계산)
        if len(text) > self.config.chunk_size * 4:
            text = text[: self.config.chunk_size * 4]
            self.logger.warning("텍스트가 너무 길어 잘렸습니다")

        return text

    def calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """두 임베딩 간의 코사인 유사도 계산"""
        try:
            import numpy as np

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # 코사인 유사도 계산
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            self.logger.error(f"유사도 계산 실패: {e}")
            return 0.0

    def get_embedding_info(self) -> Dict[str, Union[str, int, bool]]:
        """현재 임베딩 설정 정보"""
        return {
            "provider": self.config.provider.value,
            "model_name": self.config.model_name,
            "dimensions": self.config.dimensions,
            "chunk_size": self.config.chunk_size,
            "batch_size": self.config.batch_size,
            "normalize_embeddings": self.config.normalize_embeddings,
            "available": self.embeddings is not None,
        }

    def is_available(self) -> bool:
        """임베딩 서비스 사용 가능 여부"""
        return self.embeddings is not None
