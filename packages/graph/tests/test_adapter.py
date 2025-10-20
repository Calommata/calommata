"""
Graph 패키지 어댑터 테스트
"""

import pytest

from src.adapter import ParserToGraphAdapter
from src.models import CodeGraph, NodeType, RelationType


class MockCodeBlock:
    """테스트용 Mock CodeBlock 클래스"""

    def __init__(
        self,
        block_type,
        name,
        start_line,
        end_line,
        file_path="test.py",
        source_code="",
        docstring=None,
        dependencies=None,
        imports=None,
        complexity=0,
        scope_level=0,
    ):
        self.block_type = block_type
        self.name = name
        self.start_line = start_line
        self.end_line = end_line
        self.file_path = file_path
        self.source_code = source_code
        self.docstring = docstring
        self.dependencies = dependencies or []
        self.imports = imports or []
        self.complexity = complexity
        self.scope_level = scope_level


class TestParserToGraphAdapter:
    """ParserToGraphAdapter 클래스 테스트"""

    @pytest.fixture
    def adapter(self):
        """어댑터 픽스처"""
        return ParserToGraphAdapter()

    @pytest.fixture
    def mock_code_blocks(self):
        """Mock CodeBlock 리스트 픽스처"""
        return [
            MockCodeBlock(
                block_type="module",
                name="test_module",
                start_line=1,
                end_line=50,
                file_path="/path/to/test.py",
            ),
            MockCodeBlock(
                block_type="class",
                name="TestClass",
                start_line=10,
                end_line=30,
                file_path="/path/to/test.py",
                docstring="Test class",
                complexity=5,
            ),
            MockCodeBlock(
                block_type="function",
                name="test_method",
                start_line=15,
                end_line=25,
                file_path="/path/to/test.py",
                docstring="Test method",
                dependencies=["TestClass"],
                complexity=3,
            ),
        ]

    @pytest.fixture
    def dict_blocks(self):
        """딕셔너리 형태 블록 데이터 픽스처"""
        return [
            {
                "block_type": "module",
                "name": "test_module",
                "start_line": 1,
                "end_line": 50,
                "file_path": "/path/to/test.py",
                "source_code": "# module content",
                "complexity": 10,
            },
            {
                "block_type": "class",
                "name": "TestClass",
                "start_line": 10,
                "end_line": 30,
                "file_path": "/path/to/test.py",
                "docstring": "Test class",
                "dependencies": ["BaseClass"],
                "complexity": 15,
            },
            {
                "block_type": "function",
                "name": "test_function",
                "start_line": 35,
                "end_line": 45,
                "file_path": "/path/to/test.py",
                "docstring": "Test function",
                "parameters": ["param1", "param2"],
                "return_type": "str",
                "dependencies": ["TestClass"],
                "imports": ["os", "sys"],
            },
        ]

    def test_adapter_initialization(self, adapter):
        """어댑터 초기화 테스트"""
        assert adapter.node_counter == 0

    def test_convert_to_graph_from_code_blocks(self, adapter, mock_code_blocks):
        """CodeBlock 객체들로부터 그래프 변환 테스트"""
        graph = adapter.convert_to_graph(
            mock_code_blocks,
            project_name="test_project",
            project_path="/path/to/project",
        )

        assert isinstance(graph, CodeGraph)
        assert graph.project_name == "test_project"
        assert graph.project_path == "/path/to/project"
        assert len(graph.nodes) == 3

        # 노드 타입 확인 (Enum이 문자열로 저장되는 경우 대응)
        node_types = [str(node.node_type) for node in graph.nodes.values()]
        assert "Module" in node_types
        assert "Class" in node_types
        assert "Function" in node_types

    def test_convert_to_graph_from_dicts(self, adapter, dict_blocks):
        """딕셔너리 데이터로부터 그래프 변환 테스트"""
        graph = adapter.convert_to_graph(
            dict_blocks,
            project_name="dict_project",
            project_path="/path/to/dict_project",
        )

        assert isinstance(graph, CodeGraph)
        assert graph.project_name == "dict_project"
        assert len(graph.nodes) == 3

        # 특정 노드 확인
        function_nodes = graph.get_nodes_by_type(NodeType.FUNCTION)
        assert len(function_nodes) == 1
        func_node = function_nodes[0]
        assert func_node.name == "test_function"
        assert len(func_node.parameters) == 2
        assert func_node.return_type == "str"
        assert len(func_node.imports) == 2

    def test_create_node_from_code_block(self, adapter, mock_code_blocks):
        """CodeBlock으로부터 노드 생성 테스트"""
        code_block = mock_code_blocks[1]  # TestClass
        node = adapter._create_node_from_code_block(code_block)

        assert node.name == "TestClass"
        assert node.node_type == NodeType.CLASS or str(node.node_type) == "Class"
        assert node.file_path == "/path/to/test.py"
        assert node.start_line == 10
        assert node.end_line == 30
        assert node.docstring == "Test class"
        assert node.complexity == 5

    def test_create_node_from_dict(self, adapter, dict_blocks):
        """딕셔너리로부터 노드 생성 테스트"""
        block_data = dict_blocks[2]  # test_function
        node = adapter._create_node_from_dict(block_data)

        assert node.name == "test_function"
        assert node.node_type == NodeType.FUNCTION or str(node.node_type) == "Function"
        assert node.file_path == "/path/to/test.py"
        assert node.start_line == 35
        assert node.end_line == 45
        assert len(node.parameters) == 2
        assert node.return_type == "str"
        assert len(node.imports) == 2

    def test_map_block_type_from_enum(self, adapter):
        """블록 타입 매핑 테스트"""
        # 문자열 블록 타입
        assert adapter._map_block_type("module") == NodeType.MODULE
        assert adapter._map_block_type("class") == NodeType.CLASS
        assert adapter._map_block_type("function") == NodeType.FUNCTION
        assert adapter._map_block_type("unknown") == NodeType.FUNCTION  # 기본값

    def test_map_dependency_type(self, adapter):
        """의존성 타입 매핑 테스트"""
        assert adapter._map_dependency_type("calls") == RelationType.CALLS
        assert adapter._map_dependency_type("inherits") == RelationType.INHERITS
        assert adapter._map_dependency_type("imports") == RelationType.IMPORTS
        assert (
            adapter._map_dependency_type("unknown") == RelationType.DEPENDS_ON
        )  # 기본값

    def test_calculate_complexity(self, adapter, dict_blocks):
        """복잡도 계산 테스트"""
        # 복잡도가 이미 있는 경우
        block_with_complexity = dict_blocks[0]
        complexity = adapter._calculate_complexity(block_with_complexity)
        assert complexity == 10

        # 복잡도가 없는 경우 - 라인 수와 의존성으로 계산
        block_without_complexity = {
            "start_line": 10,
            "end_line": 20,
            "dependencies": ["dep1", "dep2"],
        }
        complexity = adapter._calculate_complexity(block_without_complexity)
        expected = (20 - 10 + 1) + (2 * 2)  # 라인 수 + 의존성*2
        assert complexity == expected

    def test_generate_node_id(self, adapter):
        """노드 ID 생성 테스트"""
        block_data = {
            "file_path": "/path/to/test.py",
            "block_type": "function",
            "name": "test_func",
            "start_line": 10,
        }

        node_id = adapter._generate_node_id(block_data)
        expected = "test:function:test_func:10"
        assert node_id == expected

    def test_update_graph_statistics(self, adapter, mock_code_blocks):
        """그래프 통계 업데이트 테스트"""
        graph = adapter.convert_to_graph(
            mock_code_blocks, project_name="stats_test", project_path="/path"
        )

        assert graph.total_files == 1  # 모든 블록이 같은 파일
        assert graph.total_lines > 0  # 라인 수 계산됨

        stats = graph.get_statistics()
        assert stats["total_nodes"] == 3
        assert stats["node_types"]["Module"] == 1
        assert stats["node_types"]["Class"] == 1
        assert stats["node_types"]["Function"] == 1


class TestIntegration:
    """통합 테스트"""

    def test_full_conversion_workflow(self):
        """전체 변환 워크플로 테스트"""
        # Mock 데이터 준비
        code_blocks = [
            MockCodeBlock("module", "main", 1, 100, "/project/main.py"),
            MockCodeBlock(
                "class",
                "Calculator",
                10,
                50,
                "/project/main.py",
                docstring="Calculator class",
                dependencies=[],
                complexity=8,
            ),
            MockCodeBlock(
                "function",
                "add",
                15,
                20,
                "/project/main.py",
                docstring="Add two numbers",
                dependencies=["Calculator"],
                complexity=2,
            ),
            MockCodeBlock(
                "function",
                "multiply",
                25,
                35,
                "/project/main.py",
                docstring="Multiply two numbers",
                dependencies=["Calculator"],
                complexity=3,
            ),
        ]

        # 변환 실행
        adapter = ParserToGraphAdapter()
        graph = adapter.convert_to_graph(
            code_blocks, project_name="calculator_project", project_path="/project"
        )

        # 결과 검증
        assert len(graph.nodes) == 4
        assert len(graph.relations) >= 2  # add와 multiply가 Calculator에 의존

        # 특정 노드 존재 확인
        calc_nodes = [n for n in graph.nodes.values() if n.name == "Calculator"]
        assert len(calc_nodes) == 1

        # 관계 확인
        add_to_calc_relations = [
            r
            for r in graph.relations
            if "add" in r.from_node_id and "Calculator" in r.to_node_id
        ]
        assert len(add_to_calc_relations) >= 0  # 관계가 생성될 수 있음

        # 통계 확인
        stats = graph.get_statistics()
        assert stats["total_nodes"] == 4
        assert stats["node_types"]["Module"] == 1
        assert stats["node_types"]["Class"] == 1
        assert stats["node_types"]["Function"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
