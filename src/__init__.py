"""Code Analyzer - LangChain과 Neo4j를 활용한 코드 분석 및 GraphRAG 시스템

주요 기능:
- AST 기반 코드 구조 분석
- 로컬 LLM (Ollama)을 통한 코드 임베딩
- Neo4j를 활용한 코드 그래프 저장 및 검색
- Gemini 2.0 Flash를 활용한 고품질 코드 분석
- LangGraph 기반 AI Agent 워크플로우

설계 원칙:
- Python 3.13 모범 사례 준수
- 모듈식 아키텍처로 유지보수성 확보
- MVP 우선 개발로 핵심 기능에 집중
- 디펜시브 로직 최소화로 성능 최적화

사용 예시:
```python
from code_analyzer.core import create_from_config, CoreConfig

# 환경 설정
config = CoreConfig.from_env()

# 모든 컴포넌트 초기화
persistence, embedder, retriever, graph_service, agent = create_from_config(config)

# 프로젝트 분석
graph = graph_service.analyze_and_store_project("/path/to/project")

# 코드 검색
results = agent.query("특정 함수의 구현을 찾아주세요")
```
"""

from .core import (
    CodeRAGAgent,
    CodeEmbedder,
    CodeRetriever,
    GraphService,
    CoreConfig,
    create_from_config,
    create_agent_only,
)

from .graph import (
    CodeGraph,
    CodeNode,
    CodeRelation,
    Neo4jPersistence,
    ParserToGraphAdapter,
)

from .parser import (
    CodeAnalyzer,
    CodeBlock,
    ASTExtractor,
    BaseParser,
)

__version__ = "0.2.0"
__author__ = "Code Analyzer Team"
__description__ = "LLM-powered code analysis and GraphRAG service"

__all__ = [
    # Core functionality
    "CodeRAGAgent",
    "CodeEmbedder",
    "CodeRetriever",
    "GraphService",
    "CoreConfig",
    "create_from_config",
    "create_agent_only",
    # Graph components
    "CodeGraph",
    "CodeNode",
    "CodeRelation",
    "Neo4jPersistence",
    "ParserToGraphAdapter",
    # Parser components
    "CodeAnalyzer",
    "CodeBlock",
    "ASTExtractor",
    "BaseParser",
]
