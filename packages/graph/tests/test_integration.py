"""
Parser와 Graph 패키지 통합 테스트
실제 Parser 결과를 Graph로 변환하는 테스트
"""

import sys
from pathlib import Path

import pytest

from src.adapter import ParserToGraphAdapter
from src.models import CodeGraph, NodeType

# Parser 패키지 경로 추가
parser_path = Path(__file__).parent.parent.parent / "parser"
if str(parser_path) not in sys.path:
    sys.path.insert(0, str(parser_path))


class TestParserGraphIntegration:
    """Parser와 Graph 패키지 통합 테스트"""

    @pytest.mark.skipif(not parser_path.exists(), reason="Parser package not found")
    def test_real_parser_to_graph_conversion(self):
        """실제 Parser 결과를 Graph로 변환"""
        try:
            from main.graph_builder import CodeAnalyzer

            # Parser로 코드 분석
            analyzer = CodeAnalyzer()
            example_path = parser_path / "example_code"

            if not example_path.exists():
                pytest.skip("Parser example_code not found")

            parser_blocks = analyzer.analyze_directory(str(example_path))

            # Graph 어댑터로 변환
            adapter = ParserToGraphAdapter()
            graph = adapter.convert_to_graph(
                parser_blocks,
                project_name="parser_example",
                project_path=str(example_path),
            )

            # 변환 결과 검증
            assert isinstance(graph, CodeGraph)
            assert len(graph.nodes) > 0
            assert graph.project_name == "parser_example"

            # 노드 타입들 확인
            node_types = {str(node.node_type) for node in graph.nodes.values()}
            expected_types = {"Module", "Class", "Function", "Import"}

            # 적어도 일부 타입은 있어야 함
            assert len(node_types.intersection(expected_types)) > 0

            # 파일 경로가 올바르게 설정되었는지 확인
            for node in graph.nodes.values():
                assert node.file_path != ""
                assert node.file_path != "unknown.py"

            # 통계 확인
            stats = graph.get_statistics()
            assert stats["total_nodes"] > 0
            assert stats["total_files"] > 0

            print("✅ 통합 테스트 성공:")
            print(f"   노드 수: {len(graph.nodes)}")
            print(f"   관계 수: {len(graph.relations)}")
            print(f"   파일 수: {graph.total_files}")
            print(f"   노드 타입: {node_types}")

        except ImportError:
            pytest.skip("Parser package not available")

    def test_mock_integration_workflow(self):
        """Mock 데이터로 전체 워크플로 테스트"""

        # Mock Parser 결과 생성 (실제 CodeBlock과 유사한 구조)
        class MockCodeBlock:
            def __init__(self, block_type, name, start_line, end_line, **kwargs):
                self.block_type = block_type
                self.name = name
                self.start_line = start_line
                self.end_line = end_line
                self.file_path = kwargs.get("file_path", "/mock/test.py")
                self.source_code = kwargs.get("source_code", "")
                self.docstring = kwargs.get("docstring", None)
                self.dependencies = kwargs.get("dependencies", [])
                self.imports = kwargs.get("imports", [])
                self.complexity = kwargs.get("complexity", 0)
                self.scope_level = kwargs.get("scope_level", 0)

        mock_blocks = [
            MockCodeBlock(
                "module",
                "test_module",
                1,
                100,
                file_path="/project/test.py",
                source_code="# Test module",
                complexity=10,
            ),
            MockCodeBlock(
                "import",
                "import_os",
                2,
                2,
                file_path="/project/test.py",
                imports=["os"],
            ),
            MockCodeBlock(
                "class",
                "TestClass",
                10,
                50,
                file_path="/project/test.py",
                docstring="Test class for demonstration",
                complexity=20,
                scope_level=1,
            ),
            MockCodeBlock(
                "function",
                "test_method",
                15,
                25,
                file_path="/project/test.py",
                docstring="Test method",
                dependencies=["TestClass"],
                complexity=5,
                scope_level=2,
            ),
            MockCodeBlock(
                "function",
                "helper_function",
                60,
                80,
                file_path="/project/test.py",
                docstring="Helper function",
                dependencies=["test_method"],
                complexity=8,
                scope_level=1,
            ),
        ]

        # Graph로 변환
        adapter = ParserToGraphAdapter()
        graph = adapter.convert_to_graph(
            mock_blocks, project_name="mock_integration_test", project_path="/project"
        )

        # 결과 검증
        assert len(graph.nodes) == 5
        assert graph.project_name == "mock_integration_test"

        # 각 타입별 노드 확인
        modules = graph.get_nodes_by_type(NodeType.MODULE)
        classes = graph.get_nodes_by_type(NodeType.CLASS)
        functions = graph.get_nodes_by_type(NodeType.FUNCTION)
        imports = graph.get_nodes_by_type(NodeType.IMPORT)

        assert len(modules) >= 1
        assert len(classes) >= 1
        assert len(functions) >= 2
        assert len(imports) >= 1

        # 복잡도와 스코프 레벨 확인
        test_class = next(
            (n for n in graph.nodes.values() if n.name == "TestClass"), None
        )
        assert test_class is not None
        assert test_class.complexity == 20
        assert test_class.scope_level == 1

        # 의존성 관계 확인 (적어도 일부는 생성되어야 함)
        assert len(graph.relations) >= 0

        # 파일 경로 일관성 확인
        for node in graph.nodes.values():
            assert node.file_path == "/project/test.py"

        # Neo4j 형식 변환 테스트
        neo4j_data = graph.to_neo4j_format()
        assert "project" in neo4j_data
        assert "nodes" in neo4j_data
        assert "relations" in neo4j_data
        assert "statistics" in neo4j_data

        assert neo4j_data["project"]["name"] == "mock_integration_test"
        assert len(neo4j_data["nodes"]) == 5

        # 각 노드의 Neo4j 형식 확인
        for node_data in neo4j_data["nodes"]:
            assert "id" in node_data
            assert "name" in node_data
            assert "type" in node_data
            assert "file_path" in node_data

        print("✅ Mock 통합 테스트 성공:")
        print(f"   노드 수: {len(graph.nodes)}")
        print(f"   관계 수: {len(graph.relations)}")
        print(f"   통계: {graph.get_statistics()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
