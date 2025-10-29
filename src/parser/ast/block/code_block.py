from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BlockType(Enum):
    """코드 블록 타입 열거형"""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    IMPORT = "import"
    VARIABLE = "variable"


@dataclass
class CodeBlock:
    """코드의 분할 가능한 블록 (함수, 클래스, 모듈 등)

    코드 분석의 기본 단위로, 함수, 클래스, 모듈 등을 나타냅니다.

    Attributes:
        block_type: 블록 타입 (BlockType enum)
        name: 블록의 이름 (함수명, 클래스명 등)
        file_path: 파일의 절대 경로
        parent: 부모 블록 (중첩된 함수/클래스의 경우)
        children: 자식 블록 (클래스 내 메서드 등)
        imports: 이 블록이 import하는 모듈들
        source_code: 블록의 원본 소스 코드
        dependencies: 의존하는 이름들의 리스트
        scope_level: 스코프 깊이 (0: 모듈, 1: 클래스, 2: 함수 등)
        complexity: 복잡도 점수
    """

    block_type: BlockType
    name: str
    file_path: str = ""
    parent: CodeBlock | None = None
    children: list[CodeBlock] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    source_code: str | None = None
    dependencies: list[str] = field(default_factory=list)
    scope_level: int = 0
    complexity: int = 0

    def __post_init__(self) -> None:
        # 복잡도 및 스코프 레벨 계산
        self.complexity = self._calculate_complexity()
        self.scope_level = self._calculate_scope_level()

    def get_full_name(self) -> str:
        """모듈 경로 포함 전체 이름 반환

        예: "MyClass.my_method"

        Returns:
            전체 경로를 포함한 이름
        """
        if self.parent and self.parent.name and self.parent.name != "module":
            return f"{self.parent.get_full_name()}.{self.name}"
        return self.name

    def _calculate_complexity(self) -> int:
        """블록의 복잡도 계산

        복잡도는 의존성 개수 * 2로 계산됩니다.

        Returns:
            계산된 복잡도 값
        """
        dep_count = len(self.dependencies) if self.dependencies else 0
        return dep_count * 2

    def _calculate_scope_level(self) -> int:
        """스코프 레벨 계산

        부모 블록의 깊이를 계산합니다.
        - 0: 모듈 최상위
        - 1: 클래스 또는 함수
        - 2: 클래스 내 함수 등

        Returns:
            스코프 레벨
        """
        level = 0
        current = self.parent
        while current:
            level += 1
            current = current.parent
        return level

    def is_method(self) -> bool:
        """메서드인지 확인 (클래스 내부의 함수)

        Returns:
            메서드이면 True, 아니면 False
        """
        return (
            self.block_type == BlockType.FUNCTION
            and self.parent is not None
            and self.parent.block_type == BlockType.CLASS
        )
