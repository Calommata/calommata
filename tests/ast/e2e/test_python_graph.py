"""End-to-end tests for Python code graph generation."""

import tempfile
from pathlib import Path

from app.ast.graph_builder import build_graph_from_files
from app.ast.models import LanguageType


def test_python_simple_functions():
    """Test Python file with simple functions."""
    source = """
def greet(name):
    return f"Hello, {name}"

def main():
    message = greet("World")
    print(message)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.py"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "python")]
        graph = build_graph_from_files(files)

        # Check nodes
        assert len(graph.nodes) >= 2, "Should have at least 2 function nodes"

        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "greet" in function_names, "Should have greet function"
        assert "main" in function_names, "Should have main function"

        # Check edges (main calls greet)
        assert len(graph.edges) > 0, "Should have call relationships"

        # Check in-degrees
        in_degrees = graph.calculate_in_degrees()
        greet_node = next(n for n in graph.nodes if n.get("name") == "greet")
        main_node = next(n for n in graph.nodes if n.get("name") == "main")

        assert in_degrees[greet_node["id"]] > 0, "greet should be called by main"
        assert in_degrees[main_node["id"]] == 0, "main should be entry point"


def test_python_class_and_methods():
    """Test Python file with class and methods."""
    source = """
class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

def calculate():
    calc = Calculator()
    result = calc.add(1, 2)
    return result
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "calculator.py"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "python")]
        graph = build_graph_from_files(files)

        # Check nodes
        node_types = {n["type"] for n in graph.nodes}
        assert "class" in node_types, "Should have class node"
        assert "function" in node_types, "Should have function nodes"

        # Check class name
        class_names = {n["name"] for n in graph.nodes if n["type"] == "class"}
        assert "Calculator" in class_names, "Should have Calculator class"

        # Check method names
        method_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "add" in method_names, "Should have add method"
        assert "multiply" in method_names, "Should have multiply method"
        assert "calculate" in method_names, "Should have calculate function"


def test_python_imports():
    """Test Python imports and cross-file references."""
    utils_source = """
def format_string(text):
    return text.upper()

def validate(value):
    return value is not None and len(value) > 0
"""

    main_source = """
from utils import format_string, validate

def process_input(user_input):
    if validate(user_input):
        return format_string(user_input)
    return None

def main():
    result = process_input("hello")
    print(result)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        utils_path = Path(tmpdir) / "utils.py"
        utils_path.write_text(utils_source, encoding="utf-8")

        main_path = Path(tmpdir) / "main.py"
        main_path.write_text(main_source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [
            (str(utils_path), "python"),
            (str(main_path), "python"),
        ]
        graph = build_graph_from_files(files)

        # Check nodes from both files
        all_names = {n.get("name") for n in graph.nodes}
        assert "format_string" in all_names, "Should have format_string from utils"
        assert "validate" in all_names, "Should have validate from utils"
        assert "process_input" in all_names, "Should have process_input from main"
        assert "main" in all_names, "Should have main from main"

        # Check import nodes
        import_nodes = [n for n in graph.nodes if n["type"] == "import"]
        assert len(import_nodes) > 0, "Should have import nodes"

        # Check cross-file calls are resolved
        in_degrees = graph.calculate_in_degrees()

        format_node = next(n for n in graph.nodes if n.get("name") == "format_string")
        validate_node = next(n for n in graph.nodes if n.get("name") == "validate")

        assert in_degrees[format_node["id"]] > 0, "format_string should be called"
        assert in_degrees[validate_node["id"]] > 0, "validate should be called"


def test_python_nested_calls():
    """Test Python with nested function calls (multi-level depth)."""
    source = """
def level_3():
    return "deepest"

def level_2():
    return level_3()

def level_1():
    return level_2()

def entry_point():
    return level_1()
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "nested.py"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "python")]
        graph = build_graph_from_files(files)

        # Check all functions exist
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert function_names == {"level_3", "level_2", "level_1", "entry_point"}

        # Check depth levels
        levels = graph.get_nodes_by_depth()
        assert len(levels) >= 4, "Should have at least 4 depth levels"

        # Check topological order
        sorted_nodes = graph.topological_sort()
        sorted_names = [n["name"] for n in sorted_nodes if n["type"] == "function"]

        # entry_point should come before level_1
        assert sorted_names.index("entry_point") < sorted_names.index("level_1")
        # level_1 should come before level_2
        assert sorted_names.index("level_1") < sorted_names.index("level_2")
        # level_2 should come before level_3
        assert sorted_names.index("level_2") < sorted_names.index("level_3")


def test_python_decorator():
    """Test Python functions with decorators."""
    source = """
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator
def decorated_function():
    return "decorated"

def main():
    result = decorated_function()
    return result
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "decorated.py"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "python")]
        graph = build_graph_from_files(files)

        # Check nodes
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "decorator" in function_names
        assert "wrapper" in function_names
        assert "decorated_function" in function_names
        assert "main" in function_names

        # Graph should have edges
        assert len(graph.edges) > 0, "Should have call relationships"


def test_python_lambda_and_comprehension():
    """Test Python with lambda and list comprehension."""
    source = """
def filter_numbers(numbers):
    filtered = [x for x in numbers if x > 0]
    return filtered

def transform_data(data):
    mapped = list(map(lambda x: x * 2, data))
    return mapped

def main():
    nums = [1, -2, 3, -4, 5]
    positive = filter_numbers(nums)
    doubled = transform_data(positive)
    return doubled
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "functional.py"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "python")]
        graph = build_graph_from_files(files)

        # Check main functions exist
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "filter_numbers" in function_names
        assert "transform_data" in function_names
        assert "main" in function_names

        # Check call graph
        in_degrees = graph.calculate_in_degrees()
        main_node = next(n for n in graph.nodes if n.get("name") == "main")
        assert in_degrees[main_node["id"]] == 0, "main should be entry point"


def test_python_async_functions():
    """Test Python async/await functions."""
    source = """
async def fetch_data():
    return "data"

async def process_data():
    data = await fetch_data()
    return data

def main():
    import asyncio
    result = asyncio.run(process_data())
    return result
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "async_code.py"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "python")]
        graph = build_graph_from_files(files)

        # Check async functions are recognized
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "fetch_data" in function_names
        assert "process_data" in function_names
        assert "main" in function_names

        # Check that process_data calls fetch_data
        in_degrees = graph.calculate_in_degrees()
        fetch_node = next(n for n in graph.nodes if n.get("name") == "fetch_data")
        assert in_degrees[fetch_node["id"]] > 0, "fetch_data should be called"


def test_python_complex_project():
    """Test a complex Python project with multiple files and dependencies."""
    # models.py
    models_source = """
class User:
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        return self.name

class Product:
    def __init__(self, title, price):
        self.title = title
        self.price = price
"""

    # utils.py
    utils_source = """
def validate_email(email):
    return "@" in email

def format_price(price):
    return f"${price:.2f}"
"""

    # services.py
    services_source = """
from models import User, Product
from utils import validate_email, format_price

def create_user(name, email):
    if validate_email(email):
        return User(name)
    return None

def display_product(product):
    formatted = format_price(product.price)
    return f"{product.title}: {formatted}"
"""

    # main.py
    main_source = """
from services import create_user, display_product
from models import Product

def main():
    user = create_user("Alice", "alice@example.com")
    product = Product("Laptop", 999.99)
    description = display_product(product)
    return user, description
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        models_path = Path(tmpdir) / "models.py"
        models_path.write_text(models_source, encoding="utf-8")

        utils_path = Path(tmpdir) / "utils.py"
        utils_path.write_text(utils_source, encoding="utf-8")

        services_path = Path(tmpdir) / "services.py"
        services_path.write_text(services_source, encoding="utf-8")

        main_path = Path(tmpdir) / "main.py"
        main_path.write_text(main_source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [
            (str(models_path), "python"),
            (str(utils_path), "python"),
            (str(services_path), "python"),
            (str(main_path), "python"),
        ]
        graph = build_graph_from_files(files)

        # Verify all components are present
        all_names = {n.get("name") for n in graph.nodes}

        # Classes
        assert "User" in all_names
        assert "Product" in all_names

        # Utils
        assert "validate_email" in all_names
        assert "format_price" in all_names

        # Services
        assert "create_user" in all_names
        assert "display_product" in all_names

        # Main
        assert "main" in all_names

        # Check graph structure
        outer_nodes = graph.get_outer_nodes()
        outer_names = {n["name"] for n in outer_nodes if n["type"] != "import"}
        assert "main" in outer_names, "main should be an outer node"

        # Check inner nodes (heavily used utilities)
        inner_nodes = graph.get_inner_nodes(threshold=1)
        inner_names = {n["name"] for n in inner_nodes}

        # validate_email and format_price should be inner nodes
        # (called by services)
        assert "validate_email" in inner_names or "format_price" in inner_names

        # Check depth levels
        levels = graph.get_nodes_by_depth()
        assert len(levels) >= 3, "Should have at least 3 depth levels"

        # Verify topological ordering makes sense
        sorted_nodes = graph.topological_sort()
        function_nodes = [n for n in sorted_nodes if n["type"] == "function"]

        # Should be able to process in dependency order
        assert len(function_nodes) > 0
