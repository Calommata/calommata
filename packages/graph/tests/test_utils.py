"""Graph 패키지 Utilities 테스트

GraphValidator, GraphExporter, GraphAnalyzer의 기능을 검증하는 테스트들입니다.
"""

import pytest

from src.models import CodeGraph, CodeNode, CodeRelation, NodeType, RelationType
from src.utils import GraphAnalyzer, GraphExporter, GraphValidator, validate_graph


@pytest.fixture
def sample_graph():
    """샘플 그래프 픽스처"""
    graph = CodeGraph(project_name="test_project", project_path="/test")

    # 노드 생성
    node1 = CodeNode(
        id="node_1",
        name="Module",
        node_type=NodeType.MODULE,
        file_path="module.py",
        start_line=0,
        end_line=100,
    )
    node2 = CodeNode(
        id="node_2",
        name="TestClass",
        node_type=NodeType.CLASS,
        file_path="module.py",
        start_line=10,
        end_line=50,
        complexity=5,
    )
    node3 = CodeNode(
        id="node_3",
        name="test_method",
        node_type=NodeType.FUNCTION,
        file_path="module.py",
        start_line=15,
        end_line=30,
        complexity=3,
    )

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    # 관계 생성
    rel1 = CodeRelation(
        from_node_id="node_1",
        to_node_id="node_2",
        relation_type=RelationType.CONTAINS,
    )
    rel2 = CodeRelation(
        from_node_id="node_2",
        to_node_id="node_3",
        relation_type=RelationType.CONTAINS,
    )

    graph.add_relation(rel1)
    graph.add_relation(rel2)

    return graph


class TestGraphValidator:
    """GraphValidator 테스트"""

    def test_validator_initialization(self, sample_graph):
        """검증기 초기화"""
        validator = GraphValidator(sample_graph)
        assert validator is not None
        assert validator.graph == sample_graph
        assert len(validator.errors) == 0
        assert len(validator.warnings) == 0

    def test_valid_graph_validation(self, sample_graph):
        """유효한 그래프 검증"""
        validator = GraphValidator(sample_graph)
        result = validator.validate()

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_invalid_relation_validation(self, sample_graph):
        """유효하지 않은 관계 검증"""
        # 존재하지 않는 노드를 참조하는 관계 추가
        invalid_rel = CodeRelation(
            from_node_id="node_1",
            to_node_id="nonexistent_node",
            relation_type=RelationType.CALLS,
        )
        # add_relation은 유효성 검사를 하지 않으므로 직접 추가
        sample_graph.relations.append(invalid_rel)

        validator = GraphValidator(sample_graph)
        result = validator.validate()

        # 오류가 생겨야 함
        assert len(result["errors"]) > 0

    def test_validation_statistics(self, sample_graph):
        """검증 통계"""
        validator = GraphValidator(sample_graph)
        result = validator.validate()

        stats = result["statistics"]
        assert stats["total_nodes"] == 3
        assert stats["total_relations"] == 2
        assert "node_types" in stats
        assert "relation_types" in stats

    def test_convenience_function(self, sample_graph):
        """편의 함수 테스트"""
        result = validate_graph(sample_graph)
        assert "is_valid" in result
        assert "errors" in result
        assert "statistics" in result


class TestGraphExporter:
    """GraphExporter 테스트"""

    def test_exporter_initialization(self, sample_graph):
        """내보내기 초기화"""
        exporter = GraphExporter(sample_graph)
        assert exporter is not None
        assert exporter.graph == sample_graph

    def test_to_json(self, sample_graph):
        """JSON 변환"""
        exporter = GraphExporter(sample_graph)
        json_data = exporter.to_json()

        assert json_data is not None
        assert "project" in json_data
        assert "nodes" in json_data
        assert "relations" in json_data
        assert "statistics" in json_data

    def test_to_json_with_indent(self, sample_graph):
        """들여쓰기가 있는 JSON"""
        exporter = GraphExporter(sample_graph)
        json_data_indented = exporter.to_json(indent=4)
        json_data_compact = exporter.to_json(indent=0)

        assert len(json_data_indented) > len(json_data_compact)

    def test_json_saves_correctly(self, sample_graph, tmp_path):
        """JSON 파일 저장"""
        exporter = GraphExporter(sample_graph)
        file_path = tmp_path / "test_graph.json"

        exporter.save_json(str(file_path))

        assert file_path.exists()
        content = file_path.read_text()
        assert len(content) > 0
        assert "test_project" in content

    def test_to_dot(self, sample_graph):
        """DOT 형식 변환"""
        exporter = GraphExporter(sample_graph)
        dot_data = exporter.to_dot()

        assert dot_data is not None
        assert "digraph" in dot_data
        assert "node" in dot_data
        assert "TestClass" in dot_data or "node_2" in dot_data

    def test_dot_saves_correctly(self, sample_graph, tmp_path):
        """DOT 파일 저장"""
        exporter = GraphExporter(sample_graph)
        file_path = tmp_path / "test_graph.dot"

        exporter.save_dot(str(file_path))

        assert file_path.exists()
        content = file_path.read_text()
        assert "digraph" in content


class TestGraphAnalyzer:
    """GraphAnalyzer 테스트"""

    def test_analyzer_initialization(self, sample_graph):
        """분석기 초기화"""
        analyzer = GraphAnalyzer(sample_graph)
        assert analyzer is not None
        assert analyzer.graph == sample_graph

    def test_find_circular_dependencies(self, sample_graph):
        """순환 의존성 탐지"""
        analyzer = GraphAnalyzer(sample_graph)
        cycles = analyzer.find_circular_dependencies()

        # 샘플 그래프에는 순환 의존성이 없어야 함
        assert len(cycles) == 0

    def test_find_circular_dependencies_with_cycles(self):
        """순환 의존성이 있는 경우"""
        graph = CodeGraph(project_name="test", project_path="/test")

        # 순환 의존성 생성: A -> B -> C -> A
        node_a = CodeNode(
            id="a",
            name="A",
            node_type=NodeType.CLASS,
            file_path="test.py",
            start_line=1,
            end_line=10,
        )
        node_b = CodeNode(
            id="b",
            name="B",
            node_type=NodeType.CLASS,
            file_path="test.py",
            start_line=11,
            end_line=20,
        )
        node_c = CodeNode(
            id="c",
            name="C",
            node_type=NodeType.CLASS,
            file_path="test.py",
            start_line=21,
            end_line=30,
        )

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)

        # 순환 관계 생성
        graph.relations.append(
            CodeRelation(
                from_node_id="a",
                to_node_id="b",
                relation_type=RelationType.DEPENDS_ON,
            )
        )
        graph.relations.append(
            CodeRelation(
                from_node_id="b",
                to_node_id="c",
                relation_type=RelationType.DEPENDS_ON,
            )
        )
        graph.relations.append(
            CodeRelation(
                from_node_id="c",
                to_node_id="a",
                relation_type=RelationType.DEPENDS_ON,
            )
        )

        analyzer = GraphAnalyzer(graph)
        cycles = analyzer.find_circular_dependencies()

        # 순환 의존성이 탐지되어야 함
        assert len(cycles) > 0

    def test_get_dependency_depth(self, sample_graph):
        """의존성 깊이 계산"""
        analyzer = GraphAnalyzer(sample_graph)
        depth = analyzer.get_dependency_depth("node_1")

        assert isinstance(depth, int)
        assert depth >= 0

    def test_find_most_connected_nodes(self, sample_graph):
        """가장 연결된 노드 찾기"""
        analyzer = GraphAnalyzer(sample_graph)
        connected_nodes = analyzer.find_most_connected_nodes(top_n=5)

        assert isinstance(connected_nodes, list)
        # 최소한 일부 노드가 반환되어야 함
        assert len(connected_nodes) <= 5

        # 각 항목이 올바른 구조를 가지는지 확인
        for node_info in connected_nodes:
            assert "node_id" in node_info
            assert "name" in node_info
            assert "type" in node_info
            assert "total_connections" in node_info

    def test_get_file_complexity_ranking(self, sample_graph):
        """파일별 복잡도 순위"""
        analyzer = GraphAnalyzer(sample_graph)
        ranking = analyzer.get_file_complexity_ranking()

        assert isinstance(ranking, list)
        assert len(ranking) > 0

        # 첫 번째 항목이 올바른 구조를 가지는지 확인
        first_file = ranking[0]
        assert "file_path" in first_file
        assert "total_complexity" in first_file
        assert "average_complexity" in first_file
        assert "node_count" in first_file
        assert "total_lines" in first_file


class TestGraphAnalyzerWithComplexGraph:
    """복잡한 그래프를 이용한 분석기 테스트"""

    @pytest.fixture
    def complex_graph(self):
        """복잡한 그래프"""
        graph = CodeGraph(project_name="complex", project_path="/complex")

        # 여러 파일의 노드 생성
        files = ["module1.py", "module2.py", "module3.py"]
        for i, file_path in enumerate(files):
            for j in range(3):
                node = CodeNode(
                    id=f"node_{i}_{j}",
                    name=f"Component_{i}_{j}",
                    node_type=NodeType.CLASS,
                    file_path=file_path,
                    start_line=j * 10,
                    end_line=(j + 1) * 10,
                    complexity=(i + 1) * (j + 1),
                )
                graph.add_node(node)

        # 몇 가지 관계 추가
        graph.relations.append(
            CodeRelation(
                from_node_id="node_0_0",
                to_node_id="node_0_1",
                relation_type=RelationType.DEPENDS_ON,
            )
        )
        graph.relations.append(
            CodeRelation(
                from_node_id="node_0_1",
                to_node_id="node_1_0",
                relation_type=RelationType.DEPENDS_ON,
            )
        )

        return graph

    def test_complex_graph_statistics(self, complex_graph):
        """복잡한 그래프 통계"""
        validator = GraphValidator(complex_graph)
        result = validator.validate()

        stats = result["statistics"]
        assert stats["total_nodes"] == 9  # 3 files * 3 nodes each
        assert stats["total_relations"] == 2

    def test_complex_graph_file_ranking(self, complex_graph):
        """복잡한 그래프의 파일 복잡도 순위"""
        analyzer = GraphAnalyzer(complex_graph)
        ranking = analyzer.get_file_complexity_ranking()

        # 모든 파일이 순위에 포함되어야 함
        assert len(ranking) == 3

        # 파일이 복잡도로 정렬되어 있는지 확인
        for i in range(len(ranking) - 1):
            assert (
                ranking[i]["average_complexity"] >= ranking[i + 1]["average_complexity"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
