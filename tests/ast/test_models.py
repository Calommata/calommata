"""Tests for data models."""

from app.ast.models import ParsedNode, ParsedRelation, LanguageType


def test_parsed_node_creation():
    """Test ParsedNode creation."""
    node = ParsedNode(
        id="test.py:0:10",
        type="function",
        name="test_func",
        file_path="test.py",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=2,
        source_code="def test_func():\n    pass",
        parent_id=None,
    )

    assert node.id == "test.py:0:10"
    assert node.type == "function"
    assert node.name == "test_func"
    assert node.file_path == "test.py"
    assert node.start_byte == 0
    assert node.end_byte == 10
    assert node.start_line == 1
    assert node.end_line == 2
    assert "def test_func()" in node.source_code
    assert node.parent_id is None


def test_parsed_node_with_parent():
    """Test ParsedNode with parent."""
    parent_node = ParsedNode(
        id="test.py:0:50",
        type="class",
        name="TestClass",
        file_path="test.py",
        start_byte=0,
        end_byte=50,
        start_line=1,
        end_line=5,
        source_code="class TestClass:\n    pass",
        parent_id=None,
    )

    child_node = ParsedNode(
        id="test.py:10:30",
        type="method",
        name="test_method",
        file_path="test.py",
        start_byte=10,
        end_byte=30,
        start_line=2,
        end_line=3,
        source_code="    def test_method(self):\n        pass",
        parent_id=parent_node.id,
    )

    assert child_node.parent_id == "test.py:0:50"
    assert child_node.parent_id == parent_node.id


def test_parsed_relation_creation():
    """Test ParsedRelation creation."""
    relation = ParsedRelation(
        from_id="test.py:0:10", to_id="test.py:11:20", relation_type="calls"
    )

    assert relation.from_id == "test.py:0:10"
    assert relation.to_id == "test.py:11:20"
    assert relation.relation_type == "calls"


def test_relation_types():
    """Test different relation types."""
    calls_relation = ParsedRelation(
        from_id="a.py:0:10", to_id="b.py:0:10", relation_type="calls"
    )

    imports_relation = ParsedRelation(
        from_id="a.py:0:10", to_id="b.py:0:10", relation_type="imports"
    )

    assert calls_relation.relation_type == "calls"
    assert imports_relation.relation_type == "imports"


def test_language_type():
    """Test LanguageType literal type."""
    # These should be valid LanguageType values
    python: LanguageType = "python"
    javascript: LanguageType = "javascript"
    typescript: LanguageType = "typescript"

    assert python == "python"
    assert javascript == "javascript"
    assert typescript == "typescript"


def test_parsed_node_to_dict():
    """Test ParsedNode serialization."""
    node = ParsedNode(
        id="test.py:0:10",
        type="function",
        name="test_func",
        file_path="test.py",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=2,
        source_code="def test_func():\n    pass",
        parent_id=None,
    )

    # Use the to_dict method
    node_dict = node.to_dict()

    assert node_dict["id"] == "test.py:0:10"
    assert node_dict["type"] == "function"
    assert node_dict["name"] == "test_func"
    assert node_dict["file_path"] == "test.py"
    assert node_dict["byte_range"] == [0, 10]
    assert node_dict["line_range"] == [1, 2]


def test_parsed_relation_to_dict():
    """Test ParsedRelation serialization."""
    relation = ParsedRelation(
        from_id="test.py:0:10", to_id="test.py:11:20", relation_type="calls"
    )

    # Use the to_dict method
    relation_dict = relation.to_dict()

    assert relation_dict["from"] == "test.py:0:10"
    assert relation_dict["to"] == "test.py:11:20"
    assert relation_dict["type"] == "calls"
