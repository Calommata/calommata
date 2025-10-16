from dataclasses import dataclass, field
from code_block import CodeBlock
from typing import Any


@dataclass
class ImportGraph:
    """Import 관계를 나타내는 그래프"""

    nodes: dict[str, CodeBlock] = field(default_factory=lambda: dict())
    edges: dict[str, set[str]] = field(default_factory=lambda: dict())  # from -> to

    def add_block(self, block: CodeBlock):
        """블록을 노드로 추가"""
        self.nodes[block.get_full_name()] = block
        if block.get_full_name() not in self.edges:
            self.edges[block.get_full_name()] = set()

    def add_dependency(self, from_block: str, to_module: str):
        """의존성 추가"""
        if from_block not in self.edges:
            self.edges[from_block] = set()
        self.edges[from_block].add(to_module)

    def get_dependencies(self, block_name: str) -> set[str]:
        """특정 블록의 의존성 조회"""
        return self.edges.get(block_name, set())

    def get_dependents(self, module_name: str) -> set[str]:
        """특정 모듈을 사용하는 블록들"""
        dependents: set[str] = set()
        for from_block, deps in self.edges.items():
            if module_name in deps:
                dependents.add(from_block)
        return dependents

    def find_circular_dependencies(self) -> list[list[str]]:
        """순환 의존성 감지"""
        cycles: list[list[str]] = []
        visited: set[str] = set()

        def dfs(node: str, path: list[str], rec_stack: set[str]):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path + [neighbor], rec_stack)
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            rec_stack.remove(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node, [node], set())

        return cycles

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 포맷으로 변환 (JSON 직렬화용)"""
        return {
            "nodes": {
                k: {
                    "type": v.block_type,
                    "lines": f"{v.start_line}-{v.end_line}",
                    "source_code": v.source_code or "",
                    "imports": v.imports or [],
                    "dependencies": v.dependencies or [],
                }
                for k, v in self.nodes.items()
            },
            "edges": {k: list(v) for k, v in self.edges.items()},
        }
