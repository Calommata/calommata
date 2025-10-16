from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CodeBlock:
    """코드의 분할 가능한 블록 (함수, 클래스, 모듈 등)"""

    block_type: str  # "module", "class", "function", "import"
    name: str
    start_line: int
    end_line: int
    parent: CodeBlock | None = None
    children: list["CodeBlock"] | None = None
    imports: list[str] | None = None  # ["os", "sys", "module.submodule"]
    source_code: str | None = None  # 소스 코드 저장
    dependencies: list[str] | None = None  # 클래스 의존성 ["ClassName"]

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.imports is None:
            self.imports = []
        if self.dependencies is None:
            self.dependencies = []

    def get_full_name(self) -> str:
        """모듈 경로 포함 전체 이름"""
        if self.parent and self.parent.name:
            return f"{self.parent.get_full_name()}.{self.name}"
        return self.name
