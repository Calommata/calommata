"""컨텍스트 최적화 모듈

LLM에 전달할 컨텍스트를 최적화하여 토큰 사용을 줄이고 품질을 높입니다.
"""

import logging
from typing import Any

from .constants import DEFAULT_RELATED_NODES_LIMIT

logger = logging.getLogger(__name__)


class ContextOptimizer:
    """LLM 컨텍스트 최적화 클래스"""

    def __init__(
        self,
        max_code_length: int = 500,
        max_total_context: int = 4000,
        related_nodes_limit: int = DEFAULT_RELATED_NODES_LIMIT,
    ):
        """초기화

        Args:
            max_code_length: 단일 코드 스니펫 최대 길이
            max_total_context: 전체 컨텍스트 최대 길이
            related_nodes_limit: 관련 노드 최대 개수
        """
        self.max_code_length = max_code_length
        self.max_total_context = max_total_context
        self.related_nodes_limit = related_nodes_limit

    def optimize_code_snippet(self, code: str) -> dict[str, Any]:
        """코드 스니펫 최적화

        Args:
            code: 원본 코드

        Returns:
            최적화된 코드 정보 (code, summary, is_truncated)
        """
        if len(code) <= self.max_code_length:
            return {
                "code": code,
                "summary": self._generate_summary(code),
                "is_truncated": False,
            }

        # 중요한 부분만 추출
        lines = code.split("\n")
        important_lines = self._extract_important_lines(lines)

        truncated_code = "\n".join(important_lines[: self.max_code_length // 50])

        return {
            "code": truncated_code,
            "summary": self._generate_summary(code),
            "is_truncated": True,
            "original_line_count": len(lines),
            "truncated_line_count": len(truncated_code.split("\n")),
        }

    def format_search_result_compact(self, result: Any) -> str:
        """검색 결과를 압축된 형태로 포맷팅

        Args:
            result: CodeSearchResult 객체

        Returns:
            압축된 컨텍스트 문자열
        """
        # 코드 최적화
        optimized = self.optimize_code_snippet(result.source_code)

        # 기본 정보
        context = f"### {result.node_type}: `{result.name}`\n"
        context += f"**File:** `{result.file_path}`\n"
        context += f"**Similarity:** {result.similarity_score:.1%}\n\n"  # 요약 (코드가 잘린 경우)
        if optimized["is_truncated"]:
            context += f"**Summary:** {optimized['summary']}\n"
            context += f"*(Showing {optimized['truncated_line_count']} of {optimized['original_line_count']} lines)*\n\n"

        # 코드 (압축됨)
        context += f"```python\n{optimized['code']}\n```\n\n"

        # 관련 컴포넌트 (제한적으로)
        if result.related_nodes:
            context += "**Related:** "
            related_names = [
                f"{node.get('type', '?')}:`{node.get('name', '?')}`"
                for node in result.related_nodes[: self.related_nodes_limit]
            ]
            context += ", ".join(related_names) + "\n"

        return context

    def build_optimized_context(
        self,
        search_results: list[Any],
        max_results: int = 3,
    ) -> str:
        """최적화된 전체 컨텍스트 생성

        Args:
            search_results: 검색 결과 리스트
            max_results: 포함할 최대 결과 수

        Returns:
            최적화된 컨텍스트 문자열
        """
        if not search_results:
            return "No relevant code found."

        # 유사도 순으로 정렬
        sorted_results = sorted(
            search_results, key=lambda x: x.similarity_score, reverse=True
        )

        context_parts = []
        total_length = 0

        for i, result in enumerate(sorted_results[:max_results], 1):
            result_context = self.format_search_result_compact(result)

            # 길이 체크
            if total_length + len(result_context) > self.max_total_context:
                logger.warning(
                    f"컨텍스트 길이 초과. {i - 1}개 결과만 포함. "
                    f"(현재: {total_length}, 제한: {self.max_total_context})"
                )
                break

            context_parts.append(f"## Result {i}\n{result_context}")
            total_length += len(result_context)

        final_context = "\n\n---\n\n".join(context_parts)

        logger.info(
            f"최적화된 컨텍스트 생성: {len(context_parts)}개 결과, {total_length} 문자"
        )

        return final_context

    def _extract_important_lines(self, lines: list[str]) -> list[str]:
        """중요한 라인 추출

        Args:
            lines: 코드 라인 리스트

        Returns:
            중요한 라인들
        """
        important = []

        for line in lines:
            stripped = line.strip()

            # 빈 줄 건너뛰기
            if not stripped:
                continue

            # 주석 (docstring 포함)
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                important.append(line)
                continue

            # 함수/클래스 정의
            if stripped.startswith("def ") or stripped.startswith("class "):
                important.append(line)
                continue

            # 데코레이터
            if stripped.startswith("@"):
                important.append(line)
                continue

            # import 문
            if stripped.startswith("import ") or stripped.startswith("from "):
                important.append(line)
                continue

            # 타입 힌트가 있는 할당
            if ": " in stripped and "=" in stripped:
                important.append(line)
                continue

            # return 문
            if stripped.startswith("return "):
                important.append(line)
                continue

        return important

    def _generate_summary(self, code: str) -> str:
        """코드 요약 생성

        Args:
            code: 원본 코드

        Returns:
            간단한 요약
        """
        lines = code.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        # 함수/클래스 개수 세기
        functions = sum(
            1 for line in non_empty_lines if line.strip().startswith("def ")
        )
        classes = sum(
            1 for line in non_empty_lines if line.strip().startswith("class ")
        )

        parts = []
        if classes > 0:
            parts.append(f"{classes} class{'es' if classes > 1 else ''}")
        if functions > 0:
            parts.append(f"{functions} function{'s' if functions > 1 else ''}")

        if parts:
            return ", ".join(parts)
        else:
            return f"{len(non_empty_lines)} lines of code"
