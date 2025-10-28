"""코드 블록 데이터 모델 정의"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BlockType(Enum):
    """코드 블록 타입 열거형"""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    IMPORT = "import"
    VARIABLE = "variable"


class DependencyType(Enum):
    """의존성 관계 타입 열거형

    Attributes:
        CALLS: 함수 호출 관계
        INHERITS: 클래스 상속 관계
        IMPORTS: 모듈 import 관계
        REFERENCES: 변수/속성 참조 관계
        DEFINES: 정의 관계
        CONTAINS: 포함 관계 (클래스 -> 메서드)
    """

    CALLS = "calls"
    INHERITS = "inherits"
    IMPORTS = "imports"
    REFERENCES = "references"
    DEFINES = "defines"
    CONTAINS = "contains"


@dataclass
class Dependency:
    """의존성 관계를 나타내는 클래스

    Attributes:
        target: 의존성 대상 (함수, 클래스, 모듈 이름 등)
        dependency_type: 의존성 관계 타입
        line_number: 의존성이 나타나는 라인 번호
    """

    target: str
    dependency_type: DependencyType
    line_number: int | None = None


@dataclass
class CodeBlock:
    """코드의 분할 가능한 블록 (함수, 클래스, 모듈 등)

    코드 분석의 기본 단위로, 함수, 클래스, 모듈 등을 나타냅니다.

    Attributes:
        block_type: 블록 타입 ("module", "class", "function", "import")
        name: 블록의 이름 (함수명, 클래스명 등)
        start_line: 시작 라인 번호
        end_line: 종료 라인 번호
        file_path: 파일의 절대 경로
        parent: 부모 블록 (중첩된 함수/클래스의 경우)
        children: 자식 블록 (클래스 내 메서드 등)
        imports: 이 블록이 import하는 모듈들
        source_code: 블록의 원본 소스 코드
        dependencies: 의존하는 이름들의 리스트
        scope_level: 스코프 깊이 (0: 모듈, 1: 클래스, 2: 함수 등)
        complexity: 복잡도 점수
        typed_dependencies: 타입별로 분류된 의존성들
    """

    block_type: str
    name: str
    start_line: int
    end_line: int
    file_path: str = ""
    parent: CodeBlock | None = None
    children: list[CodeBlock] | None = None
    imports: list[str] | None = None
    source_code: str | None = None
    dependencies: list[str] | None = None
    scope_level: int = 0
    complexity: int = 0
    typed_dependencies: list[Dependency] | None = None

    def __post_init__(self) -> None:
        """데이터 클래스 초기화 후 처리"""
        if self.children is None:
            self.children = []
        if self.imports is None:
            self.imports = []
        if self.dependencies is None:
            self.dependencies = []
        if self.typed_dependencies is None:
            self.typed_dependencies = []

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

        복잡도는 라인 수 + 의존성 개수 * 2 로 계산됩니다.

        Returns:
            계산된 복잡도 값
        """
        base_complexity = self.end_line - self.start_line + 1
        dep_count = len(self.dependencies) if self.dependencies else 0
        dependency_complexity = dep_count * 2
        return base_complexity + dependency_complexity

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

    def add_dependency(
        self, target: str, dep_type: DependencyType, line_number: int | None = None
    ) -> None:
        """타입별 의존성 추가

        Args:
            target: 의존성 대상 이름
            dep_type: 의존성 타입
            line_number: 의존성이 나타나는 라인 번호 (선택사항)
        """
        if not self.typed_dependencies:
            self.typed_dependencies = []

        dep = Dependency(target, dep_type, line_number)
        self.typed_dependencies.append(dep)

        # 기존 dependencies 리스트도 업데이트
        if self.dependencies is None:
            self.dependencies = []
        if target not in self.dependencies:
            self.dependencies.append(target)

    def get_dependencies_by_type(self, dep_type: DependencyType) -> list[str]:
        """특정 타입의 의존성들만 반환

        Args:
            dep_type: 필터링할 의존성 타입

        Returns:
            해당 타입의 의존성 대상들의 리스트
        """
        if not self.typed_dependencies:
            return []

        return [
            dep.target
            for dep in self.typed_dependencies
            if dep.dependency_type == dep_type
        ]

    def is_method(self) -> bool:
        """메서드인지 확인 (클래스 내부의 함수)

        Returns:
            메서드이면 True, 아니면 False
        """
        return (
            self.block_type == "function"
            and self.parent is not None
            and self.parent.block_type == "class"
        )

    def is_static_method(self) -> bool:
        """정적 메서드인지 확인

        Returns:
            정적 메서드이면 True, 아니면 False
        """
        if not self.is_method():
            return False

        if self.source_code is None:
            return False

        return "@staticmethod" in self.source_code

    def get_method_names(self) -> list[str]:
        """클래스의 모든 메서드명 반환

        Returns:
            메서드명들의 리스트 (클래스가 아니면 빈 리스트)
        """
        if self.block_type != "class":
            return []

        if not self.children:
            return []

        return [child.name for child in self.children if child.block_type == "function"]

    def to_dict(self) -> dict:
        """딕셔너리로 변환 (JSON 직렬화용)

        Returns:
            블록의 모든 정보를 포함한 딕셔너리
        """
        return {
            "name": self.name,
            "full_name": self.get_full_name(),
            "type": self.block_type,
            "file_path": self.file_path,
            "lines": f"{self.start_line}-{self.end_line}",
            "scope_level": self.scope_level,
            "complexity": self.complexity,
            "is_method": self.is_method(),
            "is_static": self.is_static_method(),
            "imports": self.imports or [],
            "dependencies": self.dependencies or [],
            "typed_dependencies": [
                {
                    "target": dep.target,
                    "type": dep.dependency_type.value,
                    "line": dep.line_number,
                }
                for dep in (self.typed_dependencies or [])
            ],
            "source_code": self.source_code or "",
        }
