"""Tests for AST parser."""

import tempfile
from pathlib import Path

from app.ast.parser import parse_file


def test_python_parser():
    """Test Python AST parsing."""
    source = """
def greet(name):
    return f"Hello, {name}!"

class Person:
    def __init__(self, name):
        self.name = name
    
    def say_hello(self):
        return greet(self.name)
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(source)
        temp_path = f.name

    try:
        nodes, _relations = parse_file(temp_path, "python")

        # Should find function and class definitions
        assert len(nodes) > 0, "Should find at least one node"

        # Check node types
        node_types = {node.type for node in nodes}
        assert "function" in node_types or "class" in node_types, (
            "Should find function or class"
        )

        # Check that nodes have names
        named_nodes = [n for n in nodes if n.name]
        assert len(named_nodes) > 0, "Should find named nodes"

    finally:
        Path(temp_path).unlink()


def test_javascript_parser():
    """Test JavaScript AST parsing."""
    source = """
function add(a, b) {
    return a + b;
}

class Calculator {
    multiply(x, y) {
        return x * y;
    }
}
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8"
    ) as f:
        f.write(source)
        temp_path = f.name

    try:
        nodes, _relations = parse_file(temp_path, "javascript")

        assert len(nodes) > 0, "Should find at least one node"

        node_types = {node.type for node in nodes}
        assert "function" in node_types or "class" in node_types, (
            "Should find function or class"
        )

    finally:
        Path(temp_path).unlink()


def test_typescript_parser():
    """Test TypeScript AST parsing."""
    source = """
interface User {
    name: string;
    age: number;
}

function getUser(): User {
    return { name: "John", age: 30 };
}

class UserManager {
    users: User[] = [];
    
    addUser(user: User): void {
        this.users.push(user);
    }
}
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ts", delete=False, encoding="utf-8"
    ) as f:
        f.write(source)
        temp_path = f.name

    try:
        nodes, _relationsrelations = parse_file(temp_path, "typescript")

        assert len(nodes) > 0, "Should find at least one node"

        node_types = {node.type for node in nodes}
        assert len(node_types) > 0, "Should find various node types"

    finally:
        Path(temp_path).unlink()
