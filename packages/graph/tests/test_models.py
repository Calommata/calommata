"""
Graph 패키지 모델 테스트
"""

import pytest

from src.models import (
    CodeGraph,
    CodeNode,
    CodeRelation,
    Dependency,
    NodeType,
    RelationType,
)


class TestDependency:
    """Dependency 모델 테스트"""

    def test_dependency_creation(self):
        """Dependency 생성 테스트"""
        dep = Dependency(
            target="test_target",
            dependency_type="calls",
            line_number=10,
            context="function call",
        )

        assert dep.target == "test_target"
        assert dep.dependency_type == "calls"
        assert dep.line_number == 10
        assert dep.context == "function call"

    def test_dependency_str(self):
        """Dependency __str__ 테스트"""
        dep = Dependency(target="test_func", dependency_type="calls")
        assert str(dep) == "calls: test_func"


class TestCodeNode:
    """CodeNode 모델 테스트"""

    @pytest.fixture
    def sample_node(self):
        """샘플 CodeNode 픽스처"""
        return CodeNode(
            id="test_node_1",
            name="test_function",
            node_type=NodeType.FUNCTION,
            file_path="/path/to/test.py",
            start_line=10,
            end_line=20,
            source_code="def test_function():\n    pass",
            docstring="Test function",
            complexity=5,
            scope_level=1,
            parameters=["param1", "param2"],
            return_type="str",
            decorators=["@decorator"],
            imports=["os", "sys"],
        )

    def test_node_creation(self, sample_node):
        """노드 생성 테스트"""
        assert sample_node.id == "test_node_1"
        assert sample_node.name == "test_function"
        # Enum이 문자열로 저장되는 경우 대응
        assert (
            sample_node.node_type == NodeType.FUNCTION
            or str(sample_node.node_type) == "Function"
        )
        assert sample_node.file_path == "/path/to/test.py"
        assert len(sample_node.parameters) == 2
        assert len(sample_node.decorators) == 1
        assert len(sample_node.imports) == 2

    def test_get_full_name(self, sample_node):
        """전체 이름 생성 테스트"""
        expected = "/path/to/test.py:Function:test_function"
        assert sample_node.get_full_name() == expected

    def test_add_dependency(self, sample_node):
        """의존성 추가 테스트"""
        initial_count = len(sample_node.dependencies)

        sample_node.add_dependency(
            target="dependency_target",
            dep_type="calls",
            line_number=15,
            context="function call",
        )

        assert len(sample_node.dependencies) == initial_count + 1
        new_dep = sample_node.dependencies[-1]
        assert new_dep.target == "dependency_target"
        assert new_dep.dependency_type == "calls"
        assert new_dep.line_number == 15

    def test_get_dependencies_by_type(self, sample_node):
        """타입별 의존성 조회 테스트"""
        sample_node.add_dependency("target1", "calls")
        sample_node.add_dependency("target2", "imports")
        sample_node.add_dependency("target3", "calls")

        call_deps = sample_node.get_dependencies_by_type("calls")
        import_deps = sample_node.get_dependencies_by_type("imports")

        assert len(call_deps) == 2
        assert len(import_deps) == 1
        assert call_deps[0].target == "target1"
        assert import_deps[0].target == "target2"

    def test_to_neo4j_node(self, sample_node):
        """Neo4j 노드 변환 테스트"""
        neo4j_data = sample_node.to_neo4j_node()

        assert neo4j_data["id"] == "test_node_1"
        assert neo4j_data["name"] == "test_function"
        assert neo4j_data["type"] == "Function"
        assert neo4j_data["complexity"] == 5
        assert "created_at" in neo4j_data
        assert "updated_at" in neo4j_data


class TestCodeRelation:
    """CodeRelation 모델 테스트"""

    @pytest.fixture
    def sample_relation(self):
        """샘플 CodeRelation 픽스처"""
        return CodeRelation(
            from_node_id="node_1",
            to_node_id="node_2",
            relation_type=RelationType.CALLS,
            weight=1.5,
            line_number=15,
            context="function call",
        )

    def test_relation_creation(self, sample_relation):
        """관계 생성 테스트"""
        assert sample_relation.from_node_id == "node_1"
        assert sample_relation.to_node_id == "node_2"
        # Enum이 문자열로 저장되는 경우 대응
        assert (
            sample_relation.relation_type == RelationType.CALLS
            or str(sample_relation.relation_type) == "CALLS"
        )
        assert sample_relation.weight == 1.5
        assert sample_relation.line_number == 15
        assert sample_relation.context == "function call"

    def test_to_neo4j_relation(self, sample_relation):
        """Neo4j 관계 변환 테스트"""
        neo4j_data = sample_relation.to_neo4j_relation()

        assert neo4j_data["type"] == "CALLS"
        assert neo4j_data["weight"] == 1.5
        assert neo4j_data["line_number"] == 15
        assert neo4j_data["context"] == "function call"
        assert "created_at" in neo4j_data


class TestCodeGraph:
    """CodeGraph 모델 테스트"""

    @pytest.fixture
    def sample_graph(self):
        """샘플 CodeGraph 픽스처"""
        return CodeGraph(
            project_name="test_project",
            project_path="/path/to/project",
            total_files=3,
            total_lines=100,
        )

    @pytest.fixture
    def sample_nodes(self):
        """샘플 노드들 픽스처"""
        return [
            CodeNode(
                id="node_1",
                name="class_a",
                node_type=NodeType.CLASS,
                file_path="/path/to/test.py",
                start_line=1,
                end_line=10,
            ),
            CodeNode(
                id="node_2",
                name="method_b",
                node_type=NodeType.FUNCTION,
                file_path="/path/to/test.py",
                start_line=5,
                end_line=8,
            ),
        ]

    def test_graph_creation(self, sample_graph):
        """그래프 생성 테스트"""
        assert sample_graph.project_name == "test_project"
        assert sample_graph.project_path == "/path/to/project"
        assert sample_graph.total_files == 3
        assert sample_graph.total_lines == 100
        assert len(sample_graph.nodes) == 0
        assert len(sample_graph.relations) == 0

    def test_add_node(self, sample_graph, sample_nodes):
        """노드 추가 테스트"""
        initial_count = len(sample_graph.nodes)

        sample_graph.add_node(sample_nodes[0])

        assert len(sample_graph.nodes) == initial_count + 1
        assert "node_1" in sample_graph.nodes
        assert sample_graph.nodes["node_1"] == sample_nodes[0]

    def test_add_relation(self, sample_graph, sample_nodes):
        """관계 추가 테스트"""
        # 먼저 노드들 추가
        for node in sample_nodes:
            sample_graph.add_node(node)

        relation = CodeRelation(
            from_node_id="node_1",
            to_node_id="node_2",
            relation_type=RelationType.CONTAINS,
        )

        initial_count = len(sample_graph.relations)
        sample_graph.add_relation(relation)

        assert len(sample_graph.relations) == initial_count + 1
        assert sample_graph.relations[0] == relation

    def test_add_relation_invalid_nodes(self, sample_graph):
        """존재하지 않는 노드 관계 추가 테스트"""
        relation = CodeRelation(
            from_node_id="nonexistent_1",
            to_node_id="nonexistent_2",
            relation_type=RelationType.CALLS,
        )

        with pytest.raises(ValueError):
            sample_graph.add_relation(relation)

    def test_get_node_by_id(self, sample_graph, sample_nodes):
        """ID로 노드 조회 테스트"""
        sample_graph.add_node(sample_nodes[0])

        found_node = sample_graph.get_node_by_id("node_1")
        not_found = sample_graph.get_node_by_id("nonexistent")

        assert found_node == sample_nodes[0]
        assert not_found is None

    def test_get_nodes_by_type(self, sample_graph, sample_nodes):
        """타입별 노드 조회 테스트"""
        for node in sample_nodes:
            sample_graph.add_node(node)

        class_nodes = sample_graph.get_nodes_by_type(NodeType.CLASS)
        function_nodes = sample_graph.get_nodes_by_type(NodeType.FUNCTION)

        assert len(class_nodes) == 1
        assert len(function_nodes) == 1
        assert class_nodes[0].name == "class_a"
        assert function_nodes[0].name == "method_b"

    def test_get_relations_from_node(self, sample_graph, sample_nodes):
        """노드에서 출발하는 관계 조회 테스트"""
        # 노드들과 관계 추가
        for node in sample_nodes:
            sample_graph.add_node(node)

        relation = CodeRelation(
            from_node_id="node_1",
            to_node_id="node_2",
            relation_type=RelationType.CONTAINS,
        )
        sample_graph.add_relation(relation)

        from_relations = sample_graph.get_relations_from_node("node_1")
        empty_relations = sample_graph.get_relations_from_node("node_2")

        assert len(from_relations) == 1
        assert len(empty_relations) == 0
        assert from_relations[0] == relation

    def test_get_statistics(self, sample_graph, sample_nodes):
        """그래프 통계 테스트"""
        for node in sample_nodes:
            sample_graph.add_node(node)

        relation = CodeRelation(
            from_node_id="node_1",
            to_node_id="node_2",
            relation_type=RelationType.CONTAINS,
        )
        sample_graph.add_relation(relation)

        stats = sample_graph.get_statistics()

        assert stats["total_nodes"] == 2
        assert stats["total_relations"] == 1
        assert stats["node_types"]["Class"] == 1
        assert stats["node_types"]["Function"] == 1
        assert stats["relation_types"]["CONTAINS"] == 1
        assert stats["total_files"] == 3
        assert stats["total_lines"] == 100

    def test_to_neo4j_format(self, sample_graph, sample_nodes):
        """Neo4j 형식 변환 테스트"""
        for node in sample_nodes:
            sample_graph.add_node(node)

        relation = CodeRelation(
            from_node_id="node_1",
            to_node_id="node_2",
            relation_type=RelationType.CONTAINS,
        )
        sample_graph.add_relation(relation)

        neo4j_data = sample_graph.to_neo4j_format()

        assert neo4j_data["project"]["name"] == "test_project"
        assert len(neo4j_data["nodes"]) == 2
        assert len(neo4j_data["relations"]) == 1
        assert "statistics" in neo4j_data

        # 관계 구조 확인
        rel_data = neo4j_data["relations"][0]
        assert rel_data["from"] == "node_1"
        assert rel_data["to"] == "node_2"
        assert rel_data["type"] == "CONTAINS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
