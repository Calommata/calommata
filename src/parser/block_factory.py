"""블록 생성 전담 모듈"""

import logging
from tree_sitter import Node

from .code_block import CodeBlock, BlockType

logger = logging.getLogger(__name__)


class BlockFactory:
    """CodeBlock 생성 책임만 담당

    TSQuery 캡처 결과로부터 CodeBlock을 생성합니다.
    """

    @staticmethod
    def create_module_block(source_code: str, file_path: str = "") -> CodeBlock:
        """모듈 블록 생성

        Args:
            source_code: 모듈의 전체 소스 코드
            file_path: 파일 경로

        Returns:
            모듈 CodeBlock
        """
        return CodeBlock(
            block_type=BlockType.MODULE,
            name="module",
            file_path=file_path,
            parent=None,
            source_code=source_code,
        )

    @staticmethod
    def create_import_block(
        module_names: list[str],
        source_code: str,
        file_path: str,
        parent: CodeBlock | None,
    ) -> CodeBlock:
        """import 블록 생성

        Args:
            module_names: import하는 모듈 이름들
            source_code: import 문의 소스 코드
            file_path: 파일 경로
            parent: 부모 블록

        Returns:
            import CodeBlock
        """
        block_name = f"import_{module_names[0]}" if module_names else "import_unknown"

        return CodeBlock(
            block_type=BlockType.IMPORT,
            name=block_name,
            file_path=file_path,
            parent=parent,
            source_code=source_code,
            imports=module_names,
        )

    @staticmethod
    def create_class_block(
        class_name: str,
        source_code: str,
        file_path: str,
        parent: CodeBlock | None,
        dependencies: list[str] | None = None,
    ) -> CodeBlock:
        """클래스 블록 생성

        Args:
            class_name: 클래스 이름
            source_code: 클래스의 소스 코드
            file_path: 파일 경로
            parent: 부모 블록
            dependencies: 의존성 목록 (선택사항)

        Returns:
            클래스 CodeBlock
        """
        return CodeBlock(
            block_type=BlockType.CLASS,
            name=class_name,
            file_path=file_path,
            parent=parent,
            source_code=source_code,
            dependencies=dependencies or [],
        )

    @staticmethod
    def create_function_block(
        func_name: str,
        source_code: str,
        file_path: str,
        parent: CodeBlock | None,
        dependencies: list[str] | None = None,
    ) -> CodeBlock:
        """함수/메서드 블록 생성

        Args:
            func_name: 함수 이름
            source_code: 함수의 소스 코드
            file_path: 파일 경로
            parent: 부모 블록
            dependencies: 의존성 목록 (선택사항)

        Returns:
            함수 CodeBlock
        """
        return CodeBlock(
            block_type=BlockType.FUNCTION,
            name=func_name,
            file_path=file_path,
            parent=parent,
            source_code=source_code,
            dependencies=dependencies or [],
        )


class BlockParentResolver:
    """블록의 부모 관계 해결 책임만 담당

    블록 트리 구조에서 각 블록의 부모를 결정합니다.
    """

    @staticmethod
    def find_parent_block(
        blocks: list[CodeBlock],
        current_name: str = "",
    ) -> CodeBlock | None:
        """부모 블록 찾기

        단순화된 로직:
        - 마지막에 추가된 CLASS 블록을 찾아서 부모로 사용 (method인 경우)
        - 없으면 MODULE을 부모로 사용

        Args:
            blocks: 현재까지 생성된 블록 리스트
            current_name: 현재 블록 이름 (미사용, 호환성 유지)

        Returns:
            부모 블록 또는 None
        """
        # 역순으로 검색하여 가장 최근의 클래스 블록 찾기
        for block in reversed(blocks):
            if block.block_type == BlockType.CLASS:
                return block

        # 클래스가 없으면 모듈 블록 반환
        for block in blocks:
            if block.block_type == BlockType.MODULE:
                return block

        return None


class NodeTextExtractor:
    """AST 노드에서 텍스트 추출 책임만 담당"""

    @staticmethod
    def get_node_text(node: Node) -> str:
        """노드에서 텍스트 추출

        Args:
            node: Tree-sitter 노드

        Returns:
            노드의 텍스트 내용
        """
        return node.text.decode("utf-8") if node.text else ""
