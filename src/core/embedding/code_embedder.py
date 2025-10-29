"""코드 임베딩 모듈

로컬 LLM 모델 및 Hugging Face 모델을 사용하여 코드를 벡터로 변환합니다.
Ollama를 통한 로컬 LLM과 HuggingFace 모델을 모두 지원합니다.
"""

import logging
from typing import Any, Literal

from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class CodeEmbedder:
    """코드 임베딩 생성기

    로컬 LLM (Ollama) 또는 Hugging Face 모델을 사용하여
    코드 조각을 벡터 표현으로 변환합니다.

    지원하는 모델:
    로컬 모델 (Ollama):
    - nomic-embed-text (기본값, 코드에 특화)
    - all-minilm
    - mxbai-embed-large

    HuggingFace 모델:
    - microsoft/codebert-base
    - sentence-transformers/all-MiniLM-L6-v2
    - BAAI/bge-small-en-v1.5
    """

    def __init__(
        self,
        provider: Literal["ollama", "huggingface"],
        model_name: str,
        ollama_base_url: str,
        model_kwargs: dict[str, Any],
        encode_kwargs: dict[str, Any],
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.ollama_base_url = ollama_base_url
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs

        # 임베딩 모델 초기화
        try:
            logger.info(f"임베딩 모델 로딩 중: {self.provider}/{self.model_name}")

            if provider == "ollama":
                self._embeddings = OllamaEmbeddings(
                    model=self.model_name,
                    base_url=self.ollama_base_url,
                )
                logger.info(f"✅ Ollama 임베딩 모델 로딩 완료: {self.model_name}")

            elif provider == "huggingface":
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs=self.model_kwargs,
                    encode_kwargs=self.encode_kwargs,
                )
                logger.info(f"✅ HuggingFace 임베딩 모델 로딩 완료: {self.model_name}")

            else:
                raise ValueError(f"지원하지 않는 제공자: {provider}")

        except Exception as e:
            logger.error(f"❌ 임베딩 모델 로딩 실패: {e}")
            raise

    def embed_code(self, code: str) -> list[float]:
        """단일 코드를 임베딩으로 변환

        Args:
            code: 임베딩할 코드 문자열

        Returns:
            임베딩 벡터 (리스트)

        Raises:
            ValueError: 임베딩 모델이 초기화되지 않은 경우
        """
        if self._embeddings is None:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다")

        try:
            # 코드 전처리
            preprocessed_code = self._preprocess_code(code)

            # 임베딩 생성
            embedding = self._embeddings.embed_query(preprocessed_code)

            logger.debug(f"코드 임베딩 완료 (차원: {len(embedding)})")
            return embedding

        except Exception as e:
            logger.error(f"❌ 코드 임베딩 실패: {e}")
            raise

    def embed_codes(self, codes: list[str]) -> list[list[float]]:
        """여러 코드를 배치로 임베딩 변환

        Args:
            codes: 임베딩할 코드 문자열 리스트

        Returns:
            임베딩 벡터들의 리스트

        Raises:
            ValueError: 임베딩 모델이 초기화되지 않은 경우
        """
        if self._embeddings is None:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다")

        try:
            # 코드 전처리
            preprocessed_codes = [self._preprocess_code(code) for code in codes]

            # 배치 임베딩 생성
            embeddings = self._embeddings.embed_documents(preprocessed_codes)

            logger.info(f"✅ {len(codes)}개 코드 임베딩 완료")
            return embeddings

        except Exception as e:
            logger.error(f"❌ 배치 임베딩 실패: {e}")
            raise

    def _preprocess_code(self, code: str) -> str:
        """코드 전처리

        임베딩 품질 향상을 위해 코드를 전처리합니다.
        - 과도한 공백 제거
        - 주석 보존 (중요한 컨텍스트)

        Args:
            code: 원본 코드

        Returns:
            전처리된 코드
        """
        if not code or not code.strip():
            return ""

        # 기본적인 정리
        lines = code.split("\n")

        # 빈 줄이 연속으로 3개 이상이면 2개로 줄임
        processed_lines: list[str] = []
        empty_count = 0

        for line in lines:
            if not line.strip():
                empty_count += 1
                if empty_count <= 2:
                    processed_lines.append(line)
            else:
                empty_count = 0
                processed_lines.append(line)

        processed_code = "\n".join(processed_lines)
        return processed_code

    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원 반환

        Returns:
            임베딩 벡터 차원
        """
        if self._embeddings is None:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다")

        # 테스트 임베딩으로 차원 확인
        test_embedding = self._embeddings.embed_query("test")
        return len(test_embedding)

    def get_model_info(self) -> dict[str, Any]:
        """모델 정보 반환

        Returns:
            모델 메타데이터
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "ollama_base_url": self.ollama_base_url
            if self.provider == "ollama"
            else None,
            "model_kwargs": self.model_kwargs
            if self.provider == "huggingface"
            else None,
            "encode_kwargs": self.encode_kwargs
            if self.provider == "huggingface"
            else None,
            "embedding_dimension": self.get_embedding_dimension()
            if self._embeddings
            else None,
        }
