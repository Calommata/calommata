"""Python 언어용 TSQuery 정의"""

# Python 함수 정의 쿼리
FUNCTION_QUERY = """
(function_definition
  name: (identifier) @function.name
  body: (block) @function.body)
"""

# Python 클래스 정의 쿼리
CLASS_QUERY = """
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list
    (identifier) @class.superclass)*
  body: (block) @class.body)
"""

# Import 문 쿼리
IMPORT_QUERY = """
(import_statement
  name: (dotted_name) @import.module)

(import_from_statement
  module_name: (dotted_name) @import.module)
"""

# 함수 호출 쿼리
FUNCTION_CALL_QUERY = """
(call
  function: (identifier) @call.function)

(call
  function: (attribute
    attribute: (identifier) @call.method))
"""

# 변수 정의 쿼리
VARIABLE_USAGE_QUERY = """
(assignment
  left: (identifier) @variable.name)
"""

# 타입 힌트 쿼리
TYPE_HINT_QUERY = """
(typed_parameter
  type: (type
    (identifier) @type.name))

(function_definition
  return_type: (type
    (identifier) @type.name))
"""

# 모든 쿼리를 딕셔너리로 정리
PYTHON_QUERIES = {
    "functions": FUNCTION_QUERY,
    "classes": CLASS_QUERY,
    "imports": IMPORT_QUERY,
    "function_calls": FUNCTION_CALL_QUERY,
    "variable_usage": VARIABLE_USAGE_QUERY,
    "type_hints": TYPE_HINT_QUERY,
}
