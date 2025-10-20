"""
Parser 패키지 단독 테스트
Tree-sitter를 사용한 코드 분석 테스트
"""

import sys
from pathlib import Path

# 현재 패키지 경로 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_parser():
    """Parser 기능 테스트"""
    print("🚀 Parser 패키지 테스트")
    print("=" * 40)

    try:
        # CodeAnalyzer 초기화
        from main.graph_builder import CodeAnalyzer

        analyzer = CodeAnalyzer()
        print("✅ CodeAnalyzer 초기화 성공")

        # 예제 코드 분석
        example_path = current_dir / "example_code"
        if example_path.exists():
            print(f"📁 분석 경로: {example_path}")
            blocks = analyzer.analyze_directory(str(example_path))

            if blocks:
                print(f"✅ 코드 블록 {len(blocks)}개 분석 완료")

                # 블록 타입별 분류
                block_types = {}
                for block in blocks:
                    block_type = block.block_type
                    if block_type not in block_types:
                        block_types[block_type] = []
                    block_types[block_type].append(block)

                # 결과 출력
                print("\n📊 분석 결과:")
                for block_type, type_blocks in block_types.items():
                    print(f"   • {block_type}: {len(type_blocks)}개")

                    # 각 타입별로 처음 3개 예시 출력
                    for block in type_blocks[:3]:
                        print(
                            f"     - {block.name} ({block.start_line}-{block.end_line}줄)"
                        )

                return blocks
            else:
                print("❌ 분석 결과가 없습니다")
                return None
        else:
            print(f"❌ 예제 경로가 존재하지 않습니다: {example_path}")
            return None

    except Exception as e:
        print(f"❌ Parser 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    result = test_parser()

    if result:
        print("\n🎉 Parser 테스트 완료!")
        print("   다음 단계: Core 패키지와 통합")
    else:
        print("\n❌ Parser 테스트 실패")


if __name__ == "__main__":
    main()
