"""End-to-end tests for JavaScript code graph generation."""

import tempfile
from pathlib import Path

from app.ast.graph_builder import build_graph_from_files
from app.ast.models import LanguageType


def test_javascript_simple_functions():
    """Test JavaScript file with simple functions."""
    source = """
function greet(name) {
    return `Hello, ${name}`;
}

function main() {
    const message = greet("World");
    console.log(message);
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.js"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "javascript")]
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


def test_javascript_arrow_functions():
    """Test JavaScript arrow functions."""
    source = """
const add = (a, b) => a + b;

const multiply = (x, y) => {
    return x * y;
};

const calculate = () => {
    const sum = add(1, 2);
    const product = multiply(3, 4);
    return sum + product;
};
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "arrow.js"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "javascript")]
        graph = build_graph_from_files(files)

        # Check nodes - arrow functions now have names extracted
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "add" in function_names, "Should have add function"
        assert "multiply" in function_names, "Should have multiply function"
        assert "calculate" in function_names, "Should have calculate function"

        # Check that calculate calls add and multiply
        in_degrees = graph.calculate_in_degrees()
        add_node = next(n for n in graph.nodes if n.get("name") == "add")
        multiply_node = next(n for n in graph.nodes if n.get("name") == "multiply")
        calculate_node = next(n for n in graph.nodes if n.get("name") == "calculate")

        assert in_degrees[add_node["id"]] > 0, "add should be called by calculate"
        assert in_degrees[multiply_node["id"]] > 0, (
            "multiply should be called by calculate"
        )
        assert in_degrees[calculate_node["id"]] == 0, "calculate should be entry point"

        # Check edges exist
        assert len(graph.edges) >= 2, "Should have at least 2 call relationships"


def test_javascript_class():
    """Test JavaScript ES6 class."""
    source = """
class Calculator {
    add(a, b) {
        return a + b;
    }
    
    subtract(a, b) {
        return a - b;
    }
    
    multiply(a, b) {
        return a * b;
    }
}

function performCalculation() {
    const calc = new Calculator();
    const result = calc.add(10, 5);
    return result;
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "calculator.js"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "javascript")]
        graph = build_graph_from_files(files)

        # Check nodes
        node_types = {n["type"] for n in graph.nodes}
        assert "class" in node_types, "Should have class node"
        assert "function" in node_types, "Should have function nodes"

        # Check class name
        class_names = {n["name"] for n in graph.nodes if n["type"] == "class"}
        assert "Calculator" in class_names, "Should have Calculator class"

        # Check that we have the outer function
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "performCalculation" in function_names, (
            "Should have performCalculation function"
        )

        # We should have at least some nodes from the class
        assert len(graph.nodes) >= 2, "Should have class and function nodes"


def test_javascript_imports():
    """Test JavaScript imports and cross-file references."""
    utils_source = """
export function formatString(text) {
    return text.toUpperCase();
}

export function validate(value) {
    return value !== null && value !== undefined && value.length > 0;
}
"""

    main_source = """
import { formatString, validate } from './utils';

function processInput(userInput) {
    if (validate(userInput)) {
        return formatString(userInput);
    }
    return null;
}

function main() {
    const result = processInput("hello");
    console.log(result);
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        utils_path = Path(tmpdir) / "utils.js"
        utils_path.write_text(utils_source, encoding="utf-8")

        main_path = Path(tmpdir) / "main.js"
        main_path.write_text(main_source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [
            (str(utils_path), "javascript"),
            (str(main_path), "javascript"),
        ]
        graph = build_graph_from_files(files)

        # Check nodes from both files
        all_names = {n.get("name") for n in graph.nodes}
        assert "formatString" in all_names, "Should have formatString from utils"
        assert "validate" in all_names, "Should have validate from utils"
        assert "processInput" in all_names, "Should have processInput from main"
        assert "main" in all_names, "Should have main from main"

        # Check cross-file calls are resolved
        in_degrees = graph.calculate_in_degrees()

        format_node = next(n for n in graph.nodes if n.get("name") == "formatString")
        validate_node = next(n for n in graph.nodes if n.get("name") == "validate")

        assert in_degrees[format_node["id"]] > 0, "formatString should be called"
        assert in_degrees[validate_node["id"]] > 0, "validate should be called"


def test_javascript_nested_calls():
    """Test JavaScript with nested function calls."""
    source = """
function level3() {
    return "deepest";
}

function level2() {
    return level3();
}

function level1() {
    return level2();
}

function entryPoint() {
    return level1();
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "nested.js"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "javascript")]
        graph = build_graph_from_files(files)

        # Check all functions exist
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert function_names == {"level3", "level2", "level1", "entryPoint"}

        # Check depth levels
        levels = graph.get_nodes_by_depth()
        assert len(levels) >= 4, "Should have at least 4 depth levels"

        # Check topological order
        sorted_nodes = graph.topological_sort()
        sorted_names = [n["name"] for n in sorted_nodes if n["type"] == "function"]

        assert sorted_names.index("entryPoint") < sorted_names.index("level1")
        assert sorted_names.index("level1") < sorted_names.index("level2")
        assert sorted_names.index("level2") < sorted_names.index("level3")


def test_javascript_async_functions():
    """Test JavaScript async/await functions."""
    source = """
async function fetchData() {
    return "data";
}

async function processData() {
    const data = await fetchData();
    return data;
}

function main() {
    processData().then(result => {
        console.log(result);
    });
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "async_code.js"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "javascript")]
        graph = build_graph_from_files(files)

        # Check async functions are recognized
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "fetchData" in function_names
        assert "processData" in function_names
        assert "main" in function_names

        # Check that processData calls fetchData
        in_degrees = graph.calculate_in_degrees()
        fetch_node = next(n for n in graph.nodes if n.get("name") == "fetchData")
        assert in_degrees[fetch_node["id"]] > 0, "fetchData should be called"


def test_javascript_callback_pattern():
    """Test JavaScript callback pattern."""
    source = """
function fetchUser(id, callback) {
    const user = { id: id, name: "User" };
    callback(user);
}

function handleUser(user) {
    console.log(user.name);
}

function main() {
    fetchUser(1, handleUser);
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "callbacks.js"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "javascript")]
        graph = build_graph_from_files(files)

        # Check functions exist
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "fetchUser" in function_names
        assert "handleUser" in function_names
        assert "main" in function_names

        # main should call fetchUser
        in_degrees = graph.calculate_in_degrees()
        fetch_node = next(n for n in graph.nodes if n.get("name") == "fetchUser")
        main_node = next(n for n in graph.nodes if n.get("name") == "main")

        assert in_degrees[fetch_node["id"]] > 0, "fetchUser should be called"
        assert in_degrees[main_node["id"]] == 0, "main should be entry point"


def test_javascript_complex_project():
    """Test a complex JavaScript project with multiple files."""
    # models.js
    models_source = """
export class User {
    constructor(name, email) {
        this.name = name;
        this.email = email;
    }
    
    getName() {
        return this.name;
    }
}

export class Product {
    constructor(title, price) {
        this.title = title;
        this.price = price;
    }
    
    getPrice() {
        return this.price;
    }
}
"""

    # utils.js
    utils_source = """
export function validateEmail(email) {
    return email.includes("@");
}

export function formatPrice(price) {
    return `$${price.toFixed(2)}`;
}
"""

    # services.js
    services_source = """
import { User, Product } from './models';
import { validateEmail, formatPrice } from './utils';

export function createUser(name, email) {
    if (validateEmail(email)) {
        return new User(name, email);
    }
    return null;
}

export function displayProduct(product) {
    const formatted = formatPrice(product.getPrice());
    return `${product.title}: ${formatted}`;
}
"""

    # main.js
    main_source = """
import { createUser, displayProduct } from './services';
import { Product } from './models';

function main() {
    const user = createUser("Alice", "alice@example.com");
    const product = new Product("Laptop", 999.99);
    const description = displayProduct(product);
    console.log(description);
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        models_path = Path(tmpdir) / "models.js"
        models_path.write_text(models_source, encoding="utf-8")

        utils_path = Path(tmpdir) / "utils.js"
        utils_path.write_text(utils_source, encoding="utf-8")

        services_path = Path(tmpdir) / "services.js"
        services_path.write_text(services_source, encoding="utf-8")

        main_path = Path(tmpdir) / "main.js"
        main_path.write_text(main_source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [
            (str(models_path), "javascript"),
            (str(utils_path), "javascript"),
            (str(services_path), "javascript"),
            (str(main_path), "javascript"),
        ]
        graph = build_graph_from_files(files)

        # Verify all components are present
        all_names = {n.get("name") for n in graph.nodes}

        # Classes
        assert "User" in all_names
        assert "Product" in all_names

        # Utils
        assert "validateEmail" in all_names
        assert "formatPrice" in all_names

        # Services
        assert "createUser" in all_names
        assert "displayProduct" in all_names

        # Main
        assert "main" in all_names

        # Check graph structure
        outer_nodes = graph.get_outer_nodes()
        outer_names = {n["name"] for n in outer_nodes if n["type"] != "import"}
        assert "main" in outer_names, "main should be an outer node"

        # Check inner nodes
        inner_nodes = graph.get_inner_nodes(threshold=1)
        inner_names = {n["name"] for n in inner_nodes}

        # Utils should be inner nodes
        assert "validateEmail" in inner_names or "formatPrice" in inner_names

        # Check depth levels
        levels = graph.get_nodes_by_depth()
        assert len(levels) >= 2, "Should have at least 2 depth levels"
