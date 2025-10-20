from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class BlockType(Enum):
    """코드 블록 타입"""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    IMPORT = "import"
    VARIABLE = "variable"


class DependencyType(Enum):
    """의존성 관계 타입"""

    CALLS = "calls"  # 함수 호출 관계
    INHERITS = "inherits"  # 클래스 상속 관계
    IMPORTS = "imports"  # 모듈 import 관계
    REFERENCES = "references"  # 변수/속성 참조 관계
    DEFINES = "defines"  # 정의 관계
    CONTAINS = "contains"  # 포함 관계 (클래스 -> 메서드)


@dataclass
class Dependency:
    """의존성 관계를 나타내는 클래스"""

    target: str
    dependency_type: DependencyType
    line_number: int | None = None


@dataclass
class CodeBlock:
    """코드의 분할 가능한 블록 (함수, 클래스, 모듈 등)"""

    block_type: str  # "module", "class", "function", "import"
    name: str
    start_line: int
    end_line: int
    file_path: str = ""  # 파일 경로 추가
    parent: CodeBlock | None = None
    children: list["CodeBlock"] | None = None
    imports: list[str] | None = None  # ["os", "sys", "module.submodule"]
    source_code: str | None = None  # 소스 코드 저장
    dependencies: list[str] | None = None  # 클래스 의존성 ["ClassName"]

    # 확장된 속성들
    scope_level: int = 0  # 스코프 깊이 (0: 모듈, 1: 클래스, 2: 함수 등)
    complexity: int = 0  # 복잡도 (라인 수, 의존성 개수 등을 기반으로)
    typed_dependencies: list[Dependency] | None = None  # 타입별 의존성
    docstring: str | None = None  # 문서 문자열

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.imports is None:
            self.imports = []
        if self.dependencies is None:
            self.dependencies = []
        if self.typed_dependencies is None:
            self.typed_dependencies = []

        # 복잡도 계산
        self.complexity = self._calculate_complexity()

        # 스코프 레벨 계산
        self.scope_level = self._calculate_scope_level()

    def get_full_name(self) -> str:
        """모듈 경로 포함 전체 이름"""
        if self.parent and self.parent.name and self.parent.name != "module":
            return f"{self.parent.get_full_name()}.{self.name}"
        return self.name

    def _calculate_complexity(self) -> int:
        """블록의 복잡도 계산"""
        base_complexity = self.end_line - self.start_line + 1
        dependency_complexity = len(self.dependencies) * 2
        return base_complexity + dependency_complexity

    def _calculate_scope_level(self) -> int:
        """스코프 레벨 계산"""
        level = 0
        current = self.parent
        while current:
            level += 1
            current = current.parent
        return level

    def add_dependency(
        self, target: str, dep_type: DependencyType, line_number: int | None = None
    ):
        """타입별 의존성 추가"""
        dep = Dependency(target, dep_type, line_number)
        self.typed_dependencies.append(dep)

        # 기존 dependencies 리스트도 업데이트
        if target not in self.dependencies:
            self.dependencies.append(target)

    def get_dependencies_by_type(self, dep_type: DependencyType) -> list[str]:
        """특정 타입의 의존성들만 반환"""
        return [
            dep.target
            for dep in self.typed_dependencies
            if dep.dependency_type == dep_type
        ]

    def is_method(self) -> bool:
        """메서드인지 확인 (클래스 내부의 함수)"""
        return (
            self.block_type == "function"
            and self.parent
            and self.parent.block_type == "class"
        )

    def is_static_method(self) -> bool:
        """정적 메서드인지 확인"""
        return (
            self.is_method()
            and self.source_code
            and "@staticmethod" in self.source_code
        )

    def get_method_names(self) -> list[str]:
        """클래스의 모든 메서드명 반환"""
        if self.block_type != "class":
            return []

        return [child.name for child in self.children if child.block_type == "function"]

    def to_dict(self) -> dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
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
            "imports": self.imports,
            "dependencies": self.dependencies,
            "typed_dependencies": [
                {
                    "target": dep.target,
                    "type": dep.dependency_type.value,
                    "line": dep.line_number,
                }
                for dep in self.typed_dependencies
            ],
            "docstring": self.docstring,
            "source_code": self.source_code or "",
        }
