"""노드 변환 전문 클래스

CodeBlock을 CodeNode로 변환하는 책임만 담당합니다.
"""

import logging
from typing import Any

from src.parser import BlockType
from src.graph.db import CodeNode, NodeType

logger = logging.getLogger(__name__)


class NodeConverter:
    """CodeBlock -> CodeNode 변환 전문 클래스"""

    def convert_block_to_node(self, block: Any) -> CodeNode:
        """CodeBlock 객체를 CodeNode로 변환

        Args:
            block: CodeBlock 객체

        Returns:
            변환된 CodeNode
        """
        node_type = self._map_block_type(block.block_type)
        file_path = getattr(block, "file_path", "unknown.py")
        node_id = self._generate_node_id(file_path, block.name)

        return CodeNode(
            id=node_id,
            name=block.name,
            node_type=node_type,
            file_path=file_path,
            source_code=getattr(block, "source_code", "") or "",
            complexity=getattr(block, "complexity", 0),
            scope_level=getattr(block, "scope_level", 0),
            parameters=getattr(block, "parameters", []),
            return_type=getattr(block, "return_type", None),
            decorators=getattr(block, "decorators", []),
            imports=getattr(block, "imports", []) or [],
        )

    def convert_dict_to_node(self, block_data: dict[str, Any]) -> CodeNode:
        """딕셔너리를 CodeNode로 변환

        Args:
            block_data: 블록 데이터 딕셔너리

        Returns:
            변환된 CodeNode
        """
        node_type = self._map_block_type_string(block_data.get("block_type", "unknown"))
        node_id = self._generate_node_id_from_dict(block_data)

        return CodeNode(
            id=node_id,
            name=block_data.get("name", "unknown"),
            node_type=node_type,
            file_path=block_data.get("file_path", "unknown.py"),
            source_code=block_data.get("source_code", ""),
            complexity=self._calculate_complexity(block_data),
            scope_level=block_data.get("scope_level", 0),
            parameters=block_data.get("parameters", []),
            return_type=block_data.get("return_type"),
            decorators=block_data.get("decorators", []),
            imports=block_data.get("imports", []),
        )

    def _map_block_type(self, block_type: BlockType) -> NodeType:
        """BlockType enum을 NodeType enum으로 매핑

        Args:
            block_type: BlockType enum

        Returns:
            대응하는 NodeType
        """
        block_type_value = (
            block_type.value if hasattr(block_type, "value") else str(block_type)
        )

        type_mapping = {
            "module": NodeType.MODULE,
            "class": NodeType.CLASS,
            "function": NodeType.FUNCTION,
            "import": NodeType.IMPORT,
            "variable": NodeType.VARIABLE,
        }
        return type_mapping.get(block_type_value, NodeType.MODULE)

    def _map_block_type_string(self, block_type: str) -> NodeType:
        """문자열 블록 타입을 NodeType으로 매핑"""
        mapping = {
            "module": NodeType.MODULE,
            "class": NodeType.CLASS,
            "function": NodeType.FUNCTION,
            "method": NodeType.METHOD,
            "import": NodeType.IMPORT,
            "variable": NodeType.VARIABLE,
            "property": NodeType.PROPERTY,
            "decorator": NodeType.DECORATOR,
        }
        return mapping.get(block_type.lower(), NodeType.FUNCTION)

    def _generate_node_id(self, file_path: str, name: str) -> str:
        """노드 ID 생성

        Args:
            file_path: 파일 경로
            name: 노드 이름

        Returns:
            고유 노드 ID
        """
        return f"{file_path}:{name}"

    def _generate_node_id_from_dict(self, block_data: dict[str, Any]) -> str:
        """딕셔너리에서 노드 ID 생성"""
        file_path = block_data.get("file_path", "unknown.py")
        name = block_data.get("name", "unknown")
        return self._generate_node_id(file_path, name)

    def _calculate_complexity(self, block_data: dict[str, Any]) -> int:
        """복잡도 계산

        Args:
            block_data: 블록 데이터

        Returns:
            계산된 복잡도
        """
        complexity = block_data.get("complexity", 0)
        if complexity > 0:
            return complexity

        # 의존성 개수로 복잡도 추정
        dependencies = block_data.get("dependencies", [])
        return len(dependencies) * 2
