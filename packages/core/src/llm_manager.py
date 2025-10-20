"""
LangChain 기반 LLM 통합 관리자
다양한 LLM 제공자를 통합 관리하고 프롬프트 템플릿을 제공
"""

import os
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import logging

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM 제공자별 import (조건부)
try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """지원되는 LLM 제공자"""

    OPENAI = "openai"
    GOOGLE_GEMINI = "google_gemini"
    LOCAL = "local"


class TaskType(Enum):
    """분석 작업 타입"""

    CODE_ANALYSIS = "code_analysis"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    SIMILARITY_SEARCH = "similarity_search"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"


@dataclass
class LLMConfig:
    """LLM 설정"""

    provider: LLMProvider = LLMProvider.GOOGLE_GEMINI
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60
    api_key: Optional[str] = None
    max_retries: int = 3
    streaming: bool = False


class LLMManager:
    """LangChain 기반 LLM 통합 관리자"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm: Optional[BaseLanguageModel] = None
        self.output_parser = StrOutputParser()
        self.prompt_templates = self._init_prompt_templates()

        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """LLM 초기화"""
        try:
            if self.config.provider == LLMProvider.OPENAI:
                self.llm = self._init_openai()
            elif self.config.provider == LLMProvider.GOOGLE_GEMINI:
                self.llm = self._init_google_gemini()
            else:
                raise ValueError(f"지원되지 않는 LLM 제공자: {self.config.provider}")

            if self.llm:
                logger.info(
                    f"✅ LLM 초기화 완료: {self.config.provider.value} - {self.config.model_name}"
                )

        except Exception as e:
            logger.error(f"❌ LLM 초기화 실패: {e}")
            self.llm = None

    def _init_openai(self) -> Optional[ChatOpenAI]:
        """OpenAI ChatGPT 초기화"""
        if not OPENAI_AVAILABLE:
            logger.error("langchain-openai가 설치되지 않았습니다")
            return None

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다")
            return None

        return ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            streaming=self.config.streaming,
            api_key=api_key,
        )

    def _init_google_gemini(self) -> Optional[ChatGoogleGenerativeAI]:
        """Google Gemini 초기화"""
        if not GOOGLE_AVAILABLE:
            logger.error("langchain-google-genai가 설치되지 않았습니다")
            return None

        api_key = (
            self.config.api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            logger.error("Google API 키가 설정되지 않았습니다")
            return None

        return ChatGoogleGenerativeAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            streaming=self.config.streaming,
            google_api_key=api_key,
        )

    def _init_prompt_templates(self) -> Dict[TaskType, ChatPromptTemplate]:
        """작업별 프롬프트 템플릿 초기화"""
        templates = {}

        # 코드 분석 템플릿
        templates[TaskType.CODE_ANALYSIS] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 전문적인 코드 분석가입니다. 
            주어진 코드를 분석하여 다음 관점에서 평가해주세요:
            1. 코드 구조와 아키텍처
            2. 코드 품질과 가독성  
            3. 잠재적 이슈나 개선점
            4. 성능 고려사항
            
            분석은 구체적이고 실용적이어야 합니다.""",
                ),
                (
                    "human",
                    """다음 코드를 분석해주세요:

파일 경로: {file_path}
코드 타입: {code_type}

```{language}
{code}
```

{additional_context}""",
                ),
            ]
        )

        # 코드 리뷰 템플릿
        templates[TaskType.CODE_REVIEW] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 시니어 개발자로서 코드 리뷰를 수행합니다.
            다음 기준으로 코드를 리뷰해주세요:
            1. 코딩 표준 준수
            2. 보안 취약점
            3. 성능 최적화 가능성
            4. 테스트 가능성
            5. 유지보수성
            
            건설적이고 구체적인 피드백을 제공해주세요.""",
                ),
                (
                    "human",
                    """다음 코드를 리뷰해주세요:

파일: {file_path}
함수/클래스: {name}

```{language}
{code}
```

특별히 검토할 부분: {focus_areas}""",
                ),
            ]
        )

        # 문서화 템플릿
        templates[TaskType.DOCUMENTATION] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 기술 문서 작성 전문가입니다.
            주어진 코드에 대한 명확하고 유용한 문서를 작성해주세요.
            - 목적과 기능 설명
            - 매개변수와 반환값
            - 사용 예시
            - 주의사항이나 제한사항""",
                ),
                (
                    "human",
                    """다음 코드에 대한 문서를 작성해주세요:

```{language}
{code}
```

문서 타입: {doc_type}
상세 수준: {detail_level}""",
                ),
            ]
        )

        # 리팩토링 제안 템플릿
        templates[TaskType.REFACTORING] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 코드 리팩토링 전문가입니다.
            주어진 코드를 분석하고 다음과 같은 개선 제안을 해주세요:
            1. 코드 구조 개선
            2. 성능 최적화
            3. 가독성 향상
            4. 재사용성 증대
            
            구체적인 before/after 코드 예시와 함께 설명해주세요.""",
                ),
                (
                    "human",
                    """다음 코드의 리팩토링을 제안해주세요:

현재 코드:
```{language}
{code}
```

리팩토링 목표: {refactoring_goals}
제약사항: {constraints}""",
                ),
            ]
        )

        # 유사 코드 검색 설명 템플릭
        templates[TaskType.SIMILARITY_SEARCH] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 코드 분석 전문가입니다.
            주어진 유사 코드 검색 결과를 분석하고 다음을 설명해주세요:
            1. 코드들 간의 유사점과 차이점
            2. 공통 패턴이나 아키텍처
            3. 각각의 장단점
            4. 학습할 수 있는 인사이트""",
                ),
                (
                    "human",
                    """다음 유사 코드 검색 결과를 분석해주세요:

검색 쿼리: {query}

유사 코드들:
{similar_codes}

분석 관점: {analysis_focus}""",
                ),
            ]
        )

        # 아키텍처 분석 템플릿
        templates[TaskType.ARCHITECTURE_ANALYSIS] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 소프트웨어 아키텍트입니다.
            주어진 프로젝트의 코드 구조를 분석하고 다음을 평가해주세요:
            1. 전체 아키텍처 패턴
            2. 모듈 간 의존성
            3. 설계 원칙 준수 여부
            4. 확장성과 유지보수성
            5. 개선 제안사항""",
                ),
                (
                    "human",
                    """다음 프로젝트 구조를 분석해주세요:

프로젝트: {project_name}
파일 수: {file_count}
주요 구성요소: {components}

코드 그래프 정보:
{graph_info}

분석 초점: {focus_area}""",
                ),
            ]
        )

        return templates

    async def analyze_code(self, task_type: TaskType, **kwargs) -> Optional[str]:
        """코드 분석 실행"""
        if not self.llm:
            logger.error("LLM이 초기화되지 않았습니다")
            return None

        try:
            # 프롬프트 템플릿 선택
            template = self.prompt_templates.get(task_type)
            if not template:
                logger.error(f"지원되지 않는 작업 타입: {task_type}")
                return None

            # 체인 구성
            chain = template | self.llm | self.output_parser

            # 실행
            result = await chain.ainvoke(kwargs)

            logger.info(f"✅ {task_type.value} 분석 완료")
            return result

        except Exception as e:
            logger.error(f"❌ 코드 분석 실패 ({task_type.value}): {e}")
            return None

    def create_custom_chain(
        self, system_message: str, human_template: str
    ) -> Optional[Any]:
        """커스텀 분석 체인 생성"""
        if not self.llm:
            return None

        template = ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", human_template)]
        )

        return template | self.llm | self.output_parser

    async def batch_analyze(
        self, task_type: TaskType, inputs: List[Dict[str, Any]]
    ) -> List[Optional[str]]:
        """배치 분석 실행"""
        if not self.llm:
            return [None] * len(inputs)

        template = self.prompt_templates.get(task_type)
        if not template:
            return [None] * len(inputs)

        # 병렬 실행을 위한 체인 구성
        chain = template | self.llm | self.output_parser

        results = []
        for input_data in inputs:
            try:
                result = await chain.ainvoke(input_data)
                results.append(result)
            except Exception as e:
                logger.error(f"배치 분석 중 오류: {e}")
                results.append(None)

        return results

    def get_supported_tasks(self) -> List[TaskType]:
        """지원되는 작업 타입 목록"""
        return list(self.prompt_templates.keys())

    def is_available(self) -> bool:
        """LLM 사용 가능 여부"""
        return self.llm is not None

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "provider": self.config.provider.value,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "available": self.is_available(),
        }


class CodeAnalysisPrompts:
    """코드 분석용 프롬프트 유틸리티"""

    @staticmethod
    def create_context_summary(nodes: List[Dict[str, Any]]) -> str:
        """코드 노드들로부터 컨텍스트 요약 생성"""
        if not nodes:
            return "분석할 코드가 없습니다."

        summary_parts = []

        # 노드 타입별 통계
        type_counts = {}
        for node in nodes:
            node_type = node.get("node_type", "Unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        summary_parts.append("코드 구성요소:")
        for node_type, count in type_counts.items():
            summary_parts.append(f"- {node_type}: {count}개")

        # 파일 경로별 분포
        file_paths = set(node.get("file_path", "") for node in nodes)
        summary_parts.append(f"\n관련 파일: {len(file_paths)}개")
        for path in sorted(file_paths):
            if path:
                summary_parts.append(f"- {path}")

        return "\n".join(summary_parts)

    @staticmethod
    def create_code_snippet(node: Dict[str, Any], max_lines: int = 20) -> str:
        """코드 노드에서 스니펫 생성"""
        source_code = node.get("source_code", "")
        if not source_code:
            return f"# {node.get('name', 'Unknown')}\n# 소스 코드 없음"

        lines = source_code.split("\n")
        if len(lines) > max_lines:
            # 앞부분과 뒷부분 일부만 표시
            half = max_lines // 2
            snippet_lines = lines[:half] + ["..."] + lines[-half:]
        else:
            snippet_lines = lines

        return "\n".join(snippet_lines)
