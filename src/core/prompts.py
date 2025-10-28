"""개선된 프롬프트 템플릿 정의

LLM 컨텍스트를 최적화하고 더 나은 답변을 유도하는 프롬프트입니다.
"""

# 시스템 프롬프트: LLM의 역할과 행동 지침
SYSTEM_PROMPT = """You are an expert code analyst and software architect with deep knowledge of Python.

Your role is to:
1. Analyze code structure, dependencies, and relationships
2. Explain complex code patterns in clear, concise terms
3. Identify potential issues or improvement opportunities
4. Provide actionable insights based on code context

When answering:
- Start with a direct answer to the user's question
- Use code examples only when necessary (keep them minimal)
- Focus on high-level concepts before diving into details
- Highlight important relationships between components
- If multiple code snippets are relevant, prioritize by relevance score

Output format:
- Use clear headings (##) to organize your response
- Keep code blocks short and focused
- Use bullet points for lists
- Cite specific file paths when referencing code"""

# 사용자 프롬프트 템플릿: 구조화된 컨텍스트 제공
USER_PROMPT_TEMPLATE = """## User Question
{query}

## Retrieved Code Context
{context}

## Instructions
Analyze the code context above and provide a comprehensive answer to the user's question. Focus on the most relevant code snippets (those with highest similarity scores) and explain their relationships."""

# 컨텍스트 최적화를 위한 코드 요약 템플릿
CODE_SUMMARY_TEMPLATE = """### {node_type}: `{name}`
**File:** `{file_path}`
**Similarity:** {similarity:.1%}

{summary}

{related_info}"""

# 관계 정보 템플릿
RELATIONSHIP_TEMPLATE = """**Related Components:**
{relationships}"""

# 의존성 정보 템플릿
DEPENDENCY_TEMPLATE = """**Dependencies:**
{dependencies}"""
