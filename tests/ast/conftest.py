"""Pytest configuration and fixtures for AST tests."""

import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return project_root


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return """
def greet(name):
    return f"Hello, {name}!"

class Person:
    def __init__(self, name):
        self.name = name
    
    def say_hello(self):
        return greet(self.name)
"""


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing."""
    return """
function add(a, b) {
    return a + b;
}

class Calculator {
    multiply(x, y) {
        return x * y;
    }
}
"""


@pytest.fixture
def sample_typescript_code():
    """Sample TypeScript code for testing."""
    return """
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
