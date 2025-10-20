"""Graph 패키지 Adapter 테스트

ParserToGraphAdapter의 기능을 검증하는 테스트들입니다.
CodeBlock을 CodeGraph로 변환하는 과정을 테스트합니다.
"""

import pytest

from src.adapter import ParserToGraphAdapter
from src.models import CodeGraph, CodeNode, NodeType, RelationType


class MockCodeBlock:
    """테스트용 Mock CodeBlock"""

    def __init__(
        self,
        block_type: str,
        name: str,
        start_line: int,
        end_line: int,
        file_path: str = "test.py",
        dependencies=None,
        docstring: str | None = None,
        complexity: int = 0,
        scope_level: int = 0,
    ):
        self.block_type = block_type
        self.name = name
        self.start_line = start_line
        self.end_line = end_line
        self.file_path = file_path
        self.dependencies = dependencies or []
        self.docstring = docstring
        self.complexity = complexity
        self.scope_level = scope_level
        self.source_code = f"def {name}(): pass"
        self.imports = []
        self.parameters = []
        self.return_type = None
        self.decorators = []


class TestParserToGraphAdapter:
    """ParserToGraphAdapter 테스트"""

    @pytest.fixture
    def adapter(self):
        """어댑터 인스턴스 픽스처"""
        return ParserToGraphAdapter()

    @pytest.fixture
    def sample_blocks(self):
        """샘플 CodeBlock들"""
        return [
            MockCodeBlock("module", "test_module", 0, 100, "test.py"),
            MockCodeBlock("class", "TestClass", 10, 50, "test.py", complexity=5),
            MockCodeBlock(
                "function", "test_func", 60, 80, "test.py", complexity=3, scope_level=0
            ),
        ]

    def test_adapter_initialization(self, adapter):
        """어댑터 초기화 테스트"""
        assert adapter is not None
        assert adapter.node_counter == 0

    def test_convert_to_graph_with_code_blocks(self, adapter, sample_blocks):
        """CodeBlock 리스트를 그래프로 변환"""
        graph = adapter.convert_to_graph(
            sample_blocks, project_name="test_project", project_path="/test"
        )

        assert isinstance(graph, CodeGraph)
        assert graph.project_name == "test_project"
        assert graph.project_path == "/test"
        assert len(graph.nodes) > 0

    def test_node_creation_from_code_block(self, adapter):
        """CodeBlock에서 노드 생성"""
        block = MockCodeBlock("function", "my_func", 10, 20, "module.py")
        node = adapter._create_node_from_code_block(block)

        assert isinstance(node, CodeNode)
        assert node.name == "my_func"
        # Enum 또는 문자열로 저장될 수 있으므로 둘 다 확인
        node_type_value = (
            node.node_type if isinstance(node.node_type, str) else node.node_type.value
        )
        assert node_type_value == NodeType.FUNCTION.value
        assert node.start_line == 10
        assert node.end_line == 20

    def test_block_type_mapping(self, adapter):
        """블록 타입 매핑 테스트"""
        test_cases = [
            ("module", NodeType.MODULE),
            ("class", NodeType.CLASS),
            ("function", NodeType.FUNCTION),
            ("import", NodeType.IMPORT),
            ("variable", NodeType.VARIABLE),
        ]

        for block_type_str, expected_node_type in test_cases:
            node_type = adapter._map_block_type_from_enum(block_type_str)
            assert node_type == expected_node_type

    def test_dependency_type_mapping(self, adapter):
        """의존성 타입 매핑 테스트"""
        test_cases = [
            ("calls", RelationType.CALLS),
            ("inherits", RelationType.INHERITS),
            ("imports", RelationType.IMPORTS),
            ("references", RelationType.REFERENCES),
            ("defines", RelationType.DEFINES),
            ("contains", RelationType.CONTAINS),
        ]

        for dep_type_str, expected_rel_type in test_cases:
            rel_type = adapter._map_dependency_type(dep_type_str)
            assert rel_type == expected_rel_type

    def test_node_id_generation(self, adapter):
        """노드 ID 생성 테스트"""
        block_data = {
            "file_path": "/path/to/file.py",
            "block_type": "function",
            "name": "my_func",
            "start_line": 10,
        }

        node_id = adapter._generate_node_id(block_data)
        assert "file" in node_id
        assert "function" in node_id
        assert "my_func" in node_id
        assert "10" in node_id

    def test_complexity_calculation(self, adapter):
        """복잡도 계산 테스트"""
        # 경우 1: 이미 복잡도가 있는 경우
        block_data_with_complexity = {
            "complexity": 10,
            "start_line": 0,
            "end_line": 0,
            "dependencies": [],
        }
        complexity = adapter._calculate_complexity(block_data_with_complexity)
        assert complexity == 10

        # 경우 2: 라인 수 기반 계산
        block_data_without_complexity = {
            "complexity": 0,
            "start_line": 10,
            "end_line": 20,
            "dependencies": ["dep1", "dep2"],
        }
        complexity = adapter._calculate_complexity(block_data_without_complexity)
        assert complexity > 0  # 라인 수 + 의존성 가중치

    def test_statistics_update(self, adapter):
        """그래프 통계 업데이트 테스트"""
        blocks = [
            MockCodeBlock("module", "mod1", 0, 100, "file1.py"),
            MockCodeBlock("class", "Class1", 10, 50, "file1.py"),
            MockCodeBlock("function", "func1", 60, 80, "file2.py"),
        ]

        graph = adapter.convert_to_graph(blocks)

        assert graph.total_files == 2  # file1.py, file2.py
        assert graph.total_nodes == 3
        assert graph.total_lines > 0

    def test_convert_with_dependencies(self, adapter):
        """의존성을 포함한 변환 테스트"""
        blocks = [
            MockCodeBlock("class", "ClassA", 10, 30, "test.py"),
            MockCodeBlock(
                "class", "ClassB", 40, 60, "test.py", dependencies=["ClassA"]
            ),
        ]

        graph = adapter.convert_to_graph(blocks)

        # 노드 생성 확인
        assert len(graph.nodes) == 2

        # 관계 생성 확인 (ClassB가 ClassA에 의존)
        assert len(graph.relations) > 0


class TestAdapterEdgeCases:
    """어댑터 엣지 케이스 테스트"""

    @pytest.fixture
    def adapter(self):
        """어댑터 인스턴스 픽스처"""
        return ParserToGraphAdapter()

    def test_empty_blocks_list(self, adapter):
        """빈 블록 리스트 처리"""
        graph = adapter.convert_to_graph([], project_name="empty", project_path="/")
        assert isinstance(graph, CodeGraph)
        assert len(graph.nodes) == 0
        assert len(graph.relations) == 0

    def test_block_with_no_file_path(self, adapter):
        """파일 경로가 없는 블록 처리"""
        block = MockCodeBlock("function", "func", 0, 10)
        block.file_path = ""

        node = adapter._create_node_from_code_block(block)
        assert node is not None
        assert node.file_path == ""

    def test_single_node_graph(self, adapter):
        """단일 노드 그래프"""
        blocks = [MockCodeBlock("module", "single", 0, 10, "single.py")]
        graph = adapter.convert_to_graph(blocks)

        assert len(graph.nodes) == 1
        assert len(graph.relations) == 0

    def test_complex_dependency_chain(self, adapter):
        """복잡한 의존성 체인"""
        blocks = [
            MockCodeBlock("class", "A", 1, 10, "file.py"),
            MockCodeBlock("class", "B", 11, 20, "file.py", dependencies=["A"]),
            MockCodeBlock("class", "C", 21, 30, "file.py", dependencies=["B"]),
            MockCodeBlock("class", "D", 31, 40, "file.py", dependencies=["A", "C"]),
        ]

        graph = adapter.convert_to_graph(blocks)

        # 의존성 체인이 올바르게 생성되는지 확인
        assert len(graph.nodes) == 4
        # 관계가 생성되었는지 확인
        assert len(graph.relations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
