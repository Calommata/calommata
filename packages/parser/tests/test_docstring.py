"""
Docstring 추출 테스트
"""

from pathlib import Path

from src.graph_builder import CodeAnalyzer

# 현재 디렉토리 reference
current_dir = Path(__file__).parent.parent


def test_docstring_extraction():
    """docstring 추출 기능 테스트"""
    analyzer = CodeAnalyzer()
    example_path = current_dir / "example_code"

    blocks = analyzer.analyze_directory(str(example_path))

    print("\n📚 Docstring 추출 결과:")
    print("=" * 60)

    docstring_blocks = [b for b in blocks if b.docstring]

    for block in docstring_blocks:
        print(f"블록: {block.name} ({block.block_type})")
        print(f"파일: {Path(block.file_path).name}")
        print(f"Docstring: {repr(block.docstring)}")
        print("-" * 40)

    print(f"\n✅ Docstring이 있는 블록: {len(docstring_blocks)}/{len(blocks)}개")


if __name__ == "__main__":
    test_docstring_extraction()
