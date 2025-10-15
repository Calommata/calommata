"""End-to-end tests for TypeScript code graph generation."""

import tempfile
from pathlib import Path

from app.ast.graph_builder import build_graph_from_files
from app.ast.models import LanguageType


def test_typescript_simple_functions():
    """Test TypeScript file with simple typed functions."""
    source = """
function greet(name: string): string {
    return `Hello, ${name}`;
}

function main(): void {
    const message: string = greet("World");
    console.log(message);
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.ts"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "typescript")]
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


def test_typescript_interfaces_and_types():
    """Test TypeScript interfaces and type definitions."""
    source = """
interface User {
    name: string;
    email: string;
}

type Product = {
    title: string;
    price: number;
};

function createUser(name: string, email: string): User {
    return { name, email };
}

function createProduct(title: string, price: number): Product {
    return { title, price };
}

function main(): void {
    const user = createUser("Alice", "alice@example.com");
    const product = createProduct("Laptop", 999.99);
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "types.ts"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "typescript")]
        graph = build_graph_from_files(files)

        # Check function nodes
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "createUser" in function_names
        assert "createProduct" in function_names
        assert "main" in function_names

        # Check that main calls the factory functions
        in_degrees = graph.calculate_in_degrees()
        create_user_node = next(n for n in graph.nodes if n.get("name") == "createUser")
        create_product_node = next(
            n for n in graph.nodes if n.get("name") == "createProduct"
        )

        assert in_degrees[create_user_node["id"]] > 0, "createUser should be called"
        assert in_degrees[create_product_node["id"]] > 0, (
            "createProduct should be called"
        )


def test_typescript_class():
    """Test TypeScript class with typed methods."""
    source = """
class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
    
    subtract(a: number, b: number): number {
        return a - b;
    }
    
    multiply(a: number, b: number): number {
        return a * b;
    }
}

function performCalculation(): number {
    const calc = new Calculator();
    const result = calc.add(10, 5);
    return result;
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "calculator.ts"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "typescript")]
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


def test_typescript_imports():
    """Test TypeScript imports and cross-file references."""
    utils_source = """
export function formatString(text: string): string {
    return text.toUpperCase();
}

export function validate(value: string | null): boolean {
    return value !== null && value !== undefined && value.length > 0;
}
"""

    main_source = """
import { formatString, validate } from './utils';

function processInput(userInput: string | null): string | null {
    if (validate(userInput)) {
        return formatString(userInput!);
    }
    return null;
}

function main(): void {
    const result = processInput("hello");
    console.log(result);
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        utils_path = Path(tmpdir) / "utils.ts"
        utils_path.write_text(utils_source, encoding="utf-8")

        main_path = Path(tmpdir) / "main.ts"
        main_path.write_text(main_source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [
            (str(utils_path), "typescript"),
            (str(main_path), "typescript"),
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


def test_typescript_generics():
    """Test TypeScript with generic functions."""
    source = """
function identity<T>(arg: T): T {
    return arg;
}

function map<T, U>(array: T[], fn: (item: T) => U): U[] {
    return array.map(fn);
}

function main(): void {
    const num = identity<number>(42);
    const str = identity<string>("hello");
    
    const numbers = [1, 2, 3];
    const doubled = map(numbers, x => x * 2);
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "generics.ts"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "typescript")]
        graph = build_graph_from_files(files)

        # Check generic functions are recognized
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "identity" in function_names
        assert "map" in function_names
        assert "main" in function_names

        # Check that main calls the generic functions
        in_degrees = graph.calculate_in_degrees()
        identity_node = next(n for n in graph.nodes if n.get("name") == "identity")
        map_node = next(n for n in graph.nodes if n.get("name") == "map")

        assert in_degrees[identity_node["id"]] > 0, "identity should be called"
        assert in_degrees[map_node["id"]] > 0, "map should be called"


def test_typescript_arrow_functions():
    """Test TypeScript arrow functions with type annotations."""
    source = """
const add = (a: number, b: number): number => a + b;

const multiply = (x: number, y: number): number => {
    return x * y;
};

const calculate = (): number => {
    const sum = add(1, 2);
    const product = multiply(3, 4);
    return sum + product;
};
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "arrow.ts"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "typescript")]
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


def test_typescript_async_functions():
    """Test TypeScript async/await with types."""
    source = """
async function fetchData(): Promise<string> {
    return "data";
}

async function processData(): Promise<string> {
    const data = await fetchData();
    return data.toUpperCase();
}

function main(): void {
    processData().then((result: string) => {
        console.log(result);
    });
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "async_code.ts"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "typescript")]
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


def test_typescript_nested_calls():
    """Test TypeScript with nested function calls."""
    source = """
function level3(): string {
    return "deepest";
}

function level2(): string {
    return level3();
}

function level1(): string {
    return level2();
}

function entryPoint(): string {
    return level1();
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "nested.ts"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "typescript")]
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


def test_typescript_complex_project():
    """Test a complex TypeScript project with multiple files."""
    # models.ts
    models_source = """
export interface IUser {
    name: string;
    email: string;
}

export class User implements IUser {
    constructor(public name: string, public email: string) {}
    
    getName(): string {
        return this.name;
    }
}

export class Product {
    constructor(public title: string, public price: number) {}
    
    getPrice(): number {
        return this.price;
    }
}
"""

    # utils.ts
    utils_source = """
export function validateEmail(email: string): boolean {
    return email.includes("@");
}

export function formatPrice(price: number): string {
    return `$${price.toFixed(2)}`;
}
"""

    # services.ts
    services_source = """
import { User, Product } from './models';
import { validateEmail, formatPrice } from './utils';

export function createUser(name: string, email: string): User | null {
    if (validateEmail(email)) {
        return new User(name, email);
    }
    return null;
}

export function displayProduct(product: Product): string {
    const formatted = formatPrice(product.getPrice());
    return `${product.title}: ${formatted}`;
}
"""

    # main.ts
    main_source = """
import { createUser, displayProduct } from './services';
import { Product } from './models';

function main(): void {
    const user = createUser("Alice", "alice@example.com");
    const product = new Product("Laptop", 999.99);
    const description = displayProduct(product);
    console.log(description);
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        models_path = Path(tmpdir) / "models.ts"
        models_path.write_text(models_source, encoding="utf-8")

        utils_path = Path(tmpdir) / "utils.ts"
        utils_path.write_text(utils_source, encoding="utf-8")

        services_path = Path(tmpdir) / "services.ts"
        services_path.write_text(services_source, encoding="utf-8")

        main_path = Path(tmpdir) / "main.ts"
        main_path.write_text(main_source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [
            (str(models_path), "typescript"),
            (str(utils_path), "typescript"),
            (str(services_path), "typescript"),
            (str(main_path), "typescript"),
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


def test_typescript_enum_and_namespace():
    """Test TypeScript enums and namespaces."""
    source = """
enum Status {
    Active,
    Inactive,
    Pending
}

namespace Utils {
    export function getStatus(value: number): Status {
        return value as Status;
    }
}

function processStatus(status: Status): string {
    return Status[status];
}

function main(): void {
    const status = Utils.getStatus(0);
    const name = processStatus(status);
    console.log(name);
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "enums.ts"
        file_path.write_text(source, encoding="utf-8")

        files: list[tuple[str, LanguageType]] = [(str(file_path), "typescript")]
        graph = build_graph_from_files(files)

        # Check function nodes
        function_names = {n["name"] for n in graph.nodes if n["type"] == "function"}
        assert "getStatus" in function_names or "processStatus" in function_names
        assert "main" in function_names

        # Graph should have structure
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0
