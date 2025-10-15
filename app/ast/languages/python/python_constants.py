"""Python-specific AST constants and mappings."""

# Python-specific node types for definitions
PYTHON_DEFINITION_TYPES: set[str] = {
    "function_definition",
    "async_function_definition",
    "class_definition",
    "decorated_definition",
}

# Python-specific call types
PYTHON_CALL_TYPES: set[str] = {
    "call",
}

# Python-specific import types
PYTHON_IMPORT_TYPES: set[str] = {
    "import_statement",
    "import_from_statement",
}

# Python-specific node type mappings
PYTHON_NODE_TYPE_MAPPING: dict[str, str] = {
    "function_definition": "function",
    "async_function_definition": "function",
    "class_definition": "class",
    "decorated_definition": "function",  # Usually decorates functions
}

# Python-specific parent definition types for call relation extraction
PYTHON_PARENT_DEFINITION_TYPES: set[str] = {
    "function_definition",
    "async_function_definition",
    "class_definition",
    "decorated_definition",
}

# Python-specific identifier types
PYTHON_IDENTIFIER_TYPES: set[str] = {
    "identifier",
}

# Python-specific complex name types
PYTHON_COMPLEX_NAME_TYPES: set[str] = {
    "dotted_name",
    "attribute",
}

# Python-specific decorator keywords
PYTHON_DECORATOR_KEYWORDS: set[str] = {
    "@property",
    "@staticmethod",
    "@classmethod",
    "@abstractmethod",
    "@dataclass",
    "@lru_cache",
}

# Python async/await keywords
PYTHON_ASYNC_KEYWORDS: set[str] = {
    "async",
    "await",
    "async def",
}

# Python comprehension types
PYTHON_COMPREHENSION_TYPES: set[str] = {
    "list_comprehension",
    "dict_comprehension",
    "set_comprehension",
    "generator_expression",
}
