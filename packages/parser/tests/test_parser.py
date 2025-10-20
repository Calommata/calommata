"""Parser 패키지 pytest 테스트

Tree-sitter를 사용한 코드 분석의 통합 테스트 스위트입니다.
CodeAnalyzer, CodeBlock, AST 추출기 등의 기능을 검증합니다.

Test Coverage:
- Parser 초기화 및 기본 동작
- 파일 분석 및 디렉토리 분석
- CodeBlock 모델 및 의존성
- docstring 추출
- 통합 워크플로우
"""

from pathlib import Path

import pytest

from src.code_block import BlockType, CodeBlock, DependencyType
from src.graph_builder import CodeAnalyzer

# 현재 디렉토리 reference
current_dir = Path(__file__).parent.parent


class TestCodeAnalyzer:
    """CodeAnalyzer 클래스 테스트"""

    @pytest.fixture
    def analyzer(self):
        """CodeAnalyzer 인스턴스 픽스처"""
        return CodeAnalyzer()

    @pytest.fixture
    def example_code_path(self):
        """예제 코드 경로 픽스처"""
        return current_dir / "example_code"

    def test_analyzer_initialization(self, analyzer):
        """분석기 초기화 테스트"""
        assert analyzer is not None
        assert analyzer.parser is not None
        assert analyzer.extractor is not None
        assert analyzer.analyzed_blocks == []

    def test_analyze_directory_exists(self, analyzer, example_code_path):
        """디렉토리 분석 테스트 - 경로 존재"""
        if not example_code_path.exists():
            pytest.skip(f"Example code path does not exist: {example_code_path}")

        blocks = analyzer.analyze_directory(str(example_code_path))
        assert len(blocks) > 0
        assert all(isinstance(block, CodeBlock) for block in blocks)

    def test_analyze_directory_nonexistent(self, analyzer):
        """디렉토리 분석 테스트 - 존재하지 않는 경로"""
        blocks = analyzer.analyze_directory("/nonexistent/path")
        assert blocks == []

    def test_file_path_in_blocks(self, analyzer, example_code_path):
        """블록에 file_path가 올바르게 설정되는지 테스트"""
        if not example_code_path.exists():
            pytest.skip(f"Example code path does not exist: {example_code_path}")

        blocks = analyzer.analyze_directory(str(example_code_path))

        for block in blocks:
            assert hasattr(block, "file_path")
            assert block.file_path != ""
            # file_path가 절대 경로인지 확인
            assert Path(block.file_path).is_absolute()


class TestCodeBlock:
    """CodeBlock 클래스 테스트"""

    @pytest.fixture
    def sample_block(self):
        """샘플 CodeBlock 픽스처"""
        return CodeBlock(
            block_type="function",
            name="test_function",
            start_line=10,
            end_line=20,
            file_path="/path/to/test.py",
            source_code="def test_function():\n    pass",
            dependencies=["dependency1", "dependency2"],
        )

    def test_block_creation(self, sample_block):
        """블록 생성 테스트"""
        assert sample_block.block_type == "function"
        assert sample_block.name == "test_function"
        assert sample_block.start_line == 10
        assert sample_block.end_line == 20
        assert sample_block.file_path == "/path/to/test.py"
        assert len(sample_block.dependencies) == 2

    def test_block_complexity_calculation(self, sample_block):
        """복잡도 계산 테스트"""
        # 라인 수 + 의존성 개수 * 2
        expected_complexity = (20 - 10 + 1) + (2 * 2)
        assert sample_block.complexity == expected_complexity

    def test_get_full_name_no_parent(self, sample_block):
        """부모가 없는 경우 full_name 테스트"""
        assert sample_block.get_full_name() == "test_function"

    def test_get_full_name_with_parent(self, sample_block):
        """부모가 있는 경우 full_name 테스트"""
        parent_block = CodeBlock(
            block_type="class",
            name="TestClass",
            start_line=1,
            end_line=30,
            file_path="/path/to/test.py",
        )
        sample_block.parent = parent_block
        assert sample_block.get_full_name() == "TestClass.test_function"

    def test_is_method(self, sample_block):
        """메서드 판별 테스트"""
        # 부모가 없으면 메서드가 아님
        assert not sample_block.is_method()

        # 클래스 부모가 있으면 메서드
        class_parent = CodeBlock(
            block_type="class",
            name="TestClass",
            start_line=1,
            end_line=30,
            file_path="/path/to/test.py",
        )
        sample_block.parent = class_parent
        assert sample_block.is_method()

    def test_add_dependency(self, sample_block):
        """타입별 의존성 추가 테스트"""
        sample_block.add_dependency("new_dependency", DependencyType.CALLS, 15)

        # typed_dependencies에 추가되었는지 확인
        assert len(sample_block.typed_dependencies) == 1
        dep = sample_block.typed_dependencies[0]
        assert dep.target == "new_dependency"
        assert dep.dependency_type == DependencyType.CALLS
        assert dep.line_number == 15

        # 기존 dependencies 리스트에도 추가되었는지 확인
        assert "new_dependency" in sample_block.dependencies

    def test_get_dependencies_by_type(self, sample_block):
        """타입별 의존성 조회 테스트"""
        sample_block.add_dependency("call_dep", DependencyType.CALLS)
        sample_block.add_dependency("import_dep", DependencyType.IMPORTS)
        sample_block.add_dependency("another_call", DependencyType.CALLS)

        call_deps = sample_block.get_dependencies_by_type(DependencyType.CALLS)
        assert len(call_deps) == 2
        assert "call_dep" in call_deps
        assert "another_call" in call_deps

        import_deps = sample_block.get_dependencies_by_type(DependencyType.IMPORTS)
        assert len(import_deps) == 1
        assert "import_dep" in import_deps

    def test_to_dict(self, sample_block):
        """딕셔너리 변환 테스트"""
        dict_result = sample_block.to_dict()

        assert dict_result["name"] == "test_function"
        assert dict_result["type"] == "function"
        assert dict_result["file_path"] == "/path/to/test.py"
        assert dict_result["lines"] == "10-20"
        assert dict_result["complexity"] > 0
        assert "dependencies" in dict_result
        assert "typed_dependencies" in dict_result


class TestIntegration:
    """통합 테스트"""

    def test_full_analysis_workflow(self):
        """전체 분석 워크플로 테스트"""
        analyzer = CodeAnalyzer()
        example_path = current_dir / "example_code"

        if not example_path.exists():
            pytest.skip(f"Example code path does not exist: {example_path}")

        # 1. 디렉토리 분석
        blocks = analyzer.analyze_directory(str(example_path))
        assert len(blocks) > 0

        # 2. 블록 타입 검증
        block_types = {block.block_type for block in blocks}
        expected_types = {"module", "import", "class", "function"}
        assert expected_types.issubset(block_types)

        # 3. 각 블록의 필수 속성 검증
        for block in blocks:
            assert block.name is not None
            assert block.start_line >= 0
            assert block.end_line >= block.start_line
            assert block.file_path != ""
            assert block.complexity >= 0
            assert block.scope_level >= 0

        # 4. 의존성 관계 검증
        functions = [b for b in blocks if b.block_type == "function"]
        for func in functions:
            if func.dependencies:
                assert all(isinstance(dep, str) for dep in func.dependencies)

        # 5. 분석기 상태 검증
        all_blocks = analyzer.get_all_blocks()
        assert len(all_blocks) == len(blocks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
