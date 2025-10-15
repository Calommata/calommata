"""Simple test to verify the new language-specific parser structure works."""

import tempfile
from pathlib import Path

from app.ast import parse_file, detect_language, get_supported_languages
from app.ast.languages.python_models import PythonNode


def test_python_parsing():
    """Test Python parsing with decorator and async features."""
    python_code = """
@dataclass
class User:
    name: str
    age: int

@lru_cache(maxsize=128)
def get_user_info(user_id: int) -> str:
    return f"User {user_id}"

async def fetch_data():
    await asyncio.sleep(1)
    return "data"

def main():
    user = User("Alice", 25)
    info = get_user_info(123)
    data = asyncio.run(fetch_data())
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(python_code)
        f.flush()

        # Test language detection
        detected = detect_language(f.name)
        print(f"Detected language: {detected}")
        assert detected == "python"

        # Parse the file
        nodes, relations = parse_file(f.name)
        print(f"Found {len(nodes)} nodes and {len(relations)} relations")

        # Check for Python-specific features
        python_nodes = [n for n in nodes if isinstance(n, PythonNode)]
        print(f"Python nodes: {len(python_nodes)}")

        # Check for async functions
        async_functions = [n for n in python_nodes if getattr(n, "is_async", False)]
        print(f"Async functions: {len(async_functions)}")

        # Check for decorated functions/classes
        decorated_nodes = [
            n for n in python_nodes if getattr(n, "has_decorators", False)
        ]
        print(f"Decorated nodes: {len(decorated_nodes)}")

        for node in decorated_nodes:
            decorators = getattr(node, "decorators", [])
            print(f"  {node.name}: decorators = {decorators}")

    # Clean up
    Path(f.name).unlink()


def test_typescript_parsing():
    """Test TypeScript parsing with interface and generics."""
    typescript_code = """
interface User<T> {
    name: string;
    data: T;
}

class UserManager<T> implements User<T> {
    private users: User<T>[] = [];
    
    addUser(user: User<T>): void {
        this.users.push(user);
    }
    
    getUser<K>(id: K): User<T> | null {
        return this.users.find(u => u.id === id) || null;
    }
}

function createUser<T>(name: string, data: T): User<T> {
    return { name, data };
}
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(typescript_code)
        f.flush()

        detected = detect_language(f.name)
        print(f"\nDetected language: {detected}")
        assert detected == "typescript"

        nodes, relations = parse_file(f.name)
        print(f"Found {len(nodes)} nodes and {len(relations)} relations")

        # Check for interface nodes
        interfaces = [n for n in nodes if n.type == "interface"]
        print(f"Interfaces: {len(interfaces)}")

        # Check for generic nodes
        generic_nodes = [n for n in nodes if getattr(n, "is_generic", False)]
        print(f"Generic nodes: {len(generic_nodes)}")

        for node in generic_nodes:
            params = getattr(node, "generic_parameters", [])
            print(f"  {node.name}: generics = {params}")

    Path(f.name).unlink()


def test_javascript_parsing():
    """Test JavaScript parsing with arrow functions and callbacks."""
    javascript_code = """
class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        return a + b;
    }
}

const multiply = (x, y) => x * y;

function* fibonacci() {
    let [a, b] = [0, 1];
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}

function processArray(arr, callback) {
    return arr.map(callback).filter(x => x > 0);
}

const result = processArray([1, -2, 3], x => x * 2);
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(javascript_code)
        f.flush()

        detected = detect_language(f.name)
        print(f"\nDetected language: {detected}")
        assert detected == "javascript"

        nodes, relations = parse_file(f.name)
        print(f"Found {len(nodes)} nodes and {len(relations)} relations")

        # Check for arrow functions
        arrow_functions = [n for n in nodes if getattr(n, "is_arrow_function", False)]
        print(f"Arrow functions: {len(arrow_functions)}")

        # Check for generator functions
        generators = [n for n in nodes if getattr(n, "is_generator", False)]
        print(f"Generator functions: {len(generators)}")

        for node in nodes:
            if hasattr(node, "name") and node.name:
                print(f"  {node.type}: {node.name}")

    Path(f.name).unlink()


if __name__ == "__main__":
    print("Testing new language-specific parser structure...")
    print(f"Supported languages: {get_supported_languages()}")

    test_python_parsing()
    test_typescript_parsing()
    test_javascript_parsing()

    print("\nAll tests completed successfully! 🎉")
