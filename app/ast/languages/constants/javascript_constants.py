"""JavaScript-specific AST constants and mappings."""

from typing import Set

# JavaScript-specific node types for definitions
JAVASCRIPT_DEFINITION_TYPES: Set[str] = {
    "function_declaration",
    "class_declaration",
    "method_definition",
    "arrow_function",
    "generator_function_declaration",
}

# JavaScript-specific call types
JAVASCRIPT_CALL_TYPES: Set[str] = {
    "call_expression",
}

# JavaScript-specific import types
JAVASCRIPT_IMPORT_TYPES: Set[str] = {
    "import_statement",
}

# JavaScript-specific node type mappings
JAVASCRIPT_NODE_TYPE_MAPPING: dict[str, str] = {
    "function_declaration": "function",
    "arrow_function": "function",
    "generator_function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
}

# JavaScript-specific parent definition types
JAVASCRIPT_PARENT_DEFINITION_TYPES: Set[str] = {
    "function_declaration",
    "class_declaration",
    "method_definition",
    "arrow_function",
    "generator_function_declaration",
}

# JavaScript-specific identifier types
JAVASCRIPT_IDENTIFIER_TYPES: Set[str] = {
    "identifier",
    "property_identifier",
}

# JavaScript-specific complex name types
JAVASCRIPT_COMPLEX_NAME_TYPES: Set[str] = {
    "member_expression",
}

# JavaScript callback function patterns
JAVASCRIPT_CALLBACK_PATTERNS: Set[str] = {
    "map",
    "filter",
    "forEach",
    "reduce",
    "then",
    "catch",
    "finally",
    "addEventListener",
    "setTimeout",
    "setInterval",
}

# JavaScript generator keywords
JAVASCRIPT_GENERATOR_KEYWORDS: Set[str] = {
    "function*",
    "yield",
    "yield*",
}

# JavaScript closure detection patterns
JAVASCRIPT_CLOSURE_PATTERNS: Set[str] = {
    "return function",
    "function(",
    "() =>",
}
