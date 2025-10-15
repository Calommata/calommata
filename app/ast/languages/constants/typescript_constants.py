"""TypeScript-specific AST constants and mappings."""

from typing import Set

# TypeScript-specific node types for definitions
TYPESCRIPT_DEFINITION_TYPES: Set[str] = {
    "function_declaration",
    "function_signature",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "method_definition",
    "arrow_function",
    "enum_declaration",
    "namespace_declaration",
}

# TypeScript-specific call types
TYPESCRIPT_CALL_TYPES: Set[str] = {
    "call_expression",
}

# TypeScript-specific import types
TYPESCRIPT_IMPORT_TYPES: Set[str] = {
    "import_statement",
}

# TypeScript-specific node type mappings
TYPESCRIPT_NODE_TYPE_MAPPING: dict[str, str] = {
    "function_declaration": "function",
    "function_signature": "function",
    "arrow_function": "function",
    "class_declaration": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "method_definition": "method",
    "enum_declaration": "enum",
    "namespace_declaration": "namespace",
}

# TypeScript-specific parent definition types
TYPESCRIPT_PARENT_DEFINITION_TYPES: Set[str] = {
    "function_declaration",
    "function_signature",
    "class_declaration",
    "interface_declaration",
    "method_definition",
    "arrow_function",
    "namespace_declaration",
}

# TypeScript-specific identifier types
TYPESCRIPT_IDENTIFIER_TYPES: Set[str] = {
    "identifier",
    "type_identifier",
    "property_identifier",
}

# TypeScript-specific complex name types
TYPESCRIPT_COMPLEX_NAME_TYPES: Set[str] = {
    "member_expression",
    "qualified_name",
}

# TypeScript access modifiers
TYPESCRIPT_ACCESS_MODIFIERS: Set[str] = {
    "public",
    "private",
    "protected",
    "readonly",
}

# TypeScript type annotation keywords
TYPESCRIPT_TYPE_KEYWORDS: Set[str] = {
    "string",
    "number",
    "boolean",
    "any",
    "void",
    "never",
    "unknown",
    "object",
}

# TypeScript generic and constraint keywords
TYPESCRIPT_GENERIC_KEYWORDS: Set[str] = {
    "extends",
    "keyof",
    "typeof",
    "infer",
}
