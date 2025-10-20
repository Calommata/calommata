# Core Package Documentation

## 📋 패키지 개요

Core 패키지는 전체 시스템의 중앙 집중식 서비스들을 제공합니다. Neo4j 데이터베이스 관리, AI 기반 코드 분석, 임베딩 서비스, 그리고 GraphRAG(Retrieval-Augmented Generation) 기능을 통합하여 지능형 코드 분석 플랫폼을 구현합니다.

## 🏗️ 아키텍처

```
Core Package
├── src/
│   ├── neo4j_handler.py       # Neo4j 데이터베이스 관리
│   ├── embedding_service.py   # 코드 임베딩 서비스
│   ├── code_vectorizer.py     # 벡터화 유틸리티
│   └── graph_rag.py          # GraphRAG 서비스
├── integration_test.py        # 통합 테스트
├── parser_graph_test.py      # Parser-Graph 연동 테스트
├── simple_test.py           # 단순 기능 테스트
├── demo.py                  # 데모 스크립트
└── main.py                  # 메인 실행 파일
```

## 🔧 주요 컴포넌트

### Neo4jHandler (neo4j_handler.py)
Neo4j 그래프 데이터베이스와의 모든 상호작용을 관리합니다.

```python
class Neo4jHandler:
    def __init__(self, uri: str, user: str, password: str):
        """Neo4j 연결 초기화"""
        
    async def connect(self) -> bool:
        """데이터베이스 연결"""
        
    async def close(self) -> None:
        """연결 종료"""
        
    async def create_node(self, node: CodeNode) -> bool:
        """노드 생성"""
        
    async def create_relationship(self, relation: CodeRelation) -> bool:
        """관계 생성"""
        
    async def save_graph(self, graph: CodeGraph) -> bool:
        """전체 그래프 저장"""
        
    async def find_similar_nodes(self, 
                                embedding: list[float], 
                                limit: int = 5) -> list[dict]:
        """벡터 유사도 기반 노드 검색"""
```

#### 주요 기능
- **연결 관리**: Neo4j 데이터베이스 연결 및 인증
- **CRUD 작업**: 노드/관계 생성, 조회, 업데이트, 삭제
- **벡터 검색**: 임베딩 기반 유사 코드 검색
- **배치 처리**: 대량 데이터 효율적 저장

### EmbeddingService (embedding_service.py)
코드를 벡터로 변환하는 임베딩 서비스입니다.

```python
class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """임베딩 모델 초기화"""
        
    def embed_code(self, code: str) -> list[float]:
        """코드를 벡터로 변환"""
        
    def embed_text(self, text: str) -> list[float]:
        """텍스트를 벡터로 변환"""
        
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩 처리"""
        
    def calculate_similarity(self, 
                           embedding1: list[float], 
                           embedding2: list[float]) -> float:
        """임베딩 간 유사도 계산"""
```

#### 특징
- **로컬 모델**: all-MiniLM-L6-v2 사용 (384차원)
- **배치 처리**: 다중 텍스트 동시 처리
- **정규화**: 코사인 유사도 계산을 위한 벡터 정규화
- **효율성**: GPU 지원 자동 감지

### CodeVectorizer (code_vectorizer.py)
코드 노드를 벡터화하는 특화된 서비스입니다.

```python
class CodeVectorizer:
    def __init__(self, embedding_service: EmbeddingService):
        """벡터화 서비스 초기화"""
        
    def vectorize_node(self, node: CodeNode) -> list[float]:
        """노드를 벡터로 변환"""
        
    def vectorize_nodes(self, nodes: list[CodeNode]) -> dict[str, list[float]]:
        """다중 노드 배치 벡터화"""
        
    def create_combined_text(self, node: CodeNode) -> str:
        """노드 정보를 결합한 텍스트 생성"""
```

#### 벡터화 전략
```python
def create_combined_text(self, node: CodeNode) -> str:
    """효과적인 임베딩을 위한 텍스트 조합"""
    parts = []
    
    # 1. 기본 정보
    parts.append(f"{node.type}: {node.name}")
    
    # 2. 문서화 문자열
    if node.docstring:
        parts.append(f"Documentation: {node.docstring}")
    
    # 3. 소스 코드 (제한적)
    if node.source_code and len(node.source_code) < 500:
        parts.append(f"Code: {node.source_code}")
    
    return " | ".join(parts)
```

### GraphRAGService (graph_rag.py)
AI 기반 코드 분석 및 질의응답 서비스입니다.

```python
class GraphRAGService:
    def __init__(self, 
                 neo4j_handler: Neo4jHandler,
                 embedding_service: EmbeddingService,
                 api_key: str):
        """GraphRAG 서비스 초기화"""
        
    async def analyze_code(self, query: str) -> str:
        """코드 분석 질의"""
        
    async def find_similar_code(self, 
                              code_snippet: str, 
                              limit: int = 5) -> list[dict]:
        """유사 코드 검색"""
        
    async def get_code_recommendations(self, 
                                     context: str) -> list[str]:
        """코드 개선 추천"""
        
    def _build_context_from_nodes(self, nodes: list[dict]) -> str:
        """노드들로부터 컨텍스트 구성"""
```

#### AI 통합 기능
- **Google Gemini 2.5 Flash**: 고성능 코드 분석
- **컨텍스트 구성**: 관련 코드 노드들로 풍부한 컨텍스트 생성
- **의미적 검색**: 임베딩 기반 유사 코드 발견
- **지능형 추천**: AI 기반 코드 개선 제안

## 📊 통합 워크플로우

### 전체 시스템 파이프라인
```python
class IntegratedCodeAnalyzer:
    """전체 시스템을 통합한 분석기"""
    
    def __init__(self):
        # 1. 서비스 초기화
        self.parser = CodeAnalyzer()
        self.graph_adapter = GraphAdapter()
        self.neo4j = Neo4jHandler(uri, user, password)
        self.embedding = EmbeddingService()
        self.vectorizer = CodeVectorizer(self.embedding)
        
        # 2. AI 서비스 (환경변수 필요)
        if os.getenv("GEMINI_API_KEY"):
            self.graph_rag = GraphRAGService(
                self.neo4j, 
                self.embedding, 
                os.getenv("GEMINI_API_KEY")
            )
    
    async def analyze_project(self, project_path: str) -> dict:
        """프로젝트 전체 분석"""
        # 1단계: 코드 파싱
        blocks = self.parser.analyze_directory(project_path)
        
        # 2단계: 그래프 변환
        graph = self.graph_adapter.convert_to_graph(blocks)
        
        # 3단계: 벡터화
        embeddings = self.vectorizer.vectorize_nodes(graph.nodes)
        
        # 4단계: 데이터베이스 저장
        await self.neo4j.save_graph(graph)
        
        # 5단계: AI 분석 (선택적)
        ai_insights = None
        if hasattr(self, 'graph_rag'):
            ai_insights = await self.graph_rag.analyze_code(
                f"Analyze this project with {len(graph.nodes)} components"
            )
        
        return {
            "blocks_found": len(blocks),
            "nodes_created": len(graph.nodes),
            "relations_created": len(graph.relations),
            "embeddings_generated": len(embeddings),
            "ai_insights": ai_insights
        }
```

## 🔍 사용 예시

### 기본 시스템 초기화
```python
import os
from core.src.neo4j_handler import Neo4jHandler
from core.src.embedding_service import EmbeddingService
from core.src.graph_rag import GraphRAGService

async def setup_system():
    # Neo4j 연결
    neo4j = Neo4jHandler(
        uri="bolt://localhost:7687",
        user="neo4j", 
        password="password"
    )
    
    await neo4j.connect()
    
    # 임베딩 서비스
    embedding = EmbeddingService()
    
    # AI 서비스 (API 키 필요)
    if os.getenv("GEMINI_API_KEY"):
        rag = GraphRAGService(neo4j, embedding, os.getenv("GEMINI_API_KEY"))
        
        # AI 분석 실행
        result = await rag.analyze_code(
            "What are the main components of this codebase?"
        )
        print(f"AI 분석 결과: {result}")
    
    await neo4j.close()
```

### 코드 유사도 검색
```python
async def find_similar_functions():
    # 검색하고자 하는 코드
    query_code = """
    def calculate_total(items):
        return sum(item.price for item in items)
    """
    
    # 임베딩 생성
    embedding = EmbeddingService()
    query_vector = embedding.embed_code(query_code)
    
    # 유사 코드 검색
    neo4j = Neo4jHandler(uri, user, password)
    await neo4j.connect()
    
    similar_nodes = await neo4j.find_similar_nodes(
        query_vector, 
        limit=5
    )
    
    print("유사한 코드 발견:")
    for node in similar_nodes:
        print(f"- {node['name']} (유사도: {node['similarity']:.3f})")
```

### GraphRAG 질의응답
```python
async def ask_about_codebase():
    # GraphRAG 서비스 초기화
    rag = GraphRAGService(neo4j, embedding, api_key)
    
    # 질문들
    questions = [
        "What are the main classes in this project?",
        "Which functions have the highest complexity?",
        "How are the components connected?",
        "What would you recommend to improve this code?"
    ]
    
    for question in questions:
        answer = await rag.analyze_code(question)
        print(f"Q: {question}")
        print(f"A: {answer}\n")
```

## ⚙️ 설정 및 환경

### 필수 환경 변수
```bash
# Neo4j 설정
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# AI 서비스 API 키
GEMINI_API_KEY=your_gemini_api_key

# 임베딩 모델 설정 (선택적)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_CACHE_DIR=./models
```

### Neo4j 데이터베이스 설정
```cypher
-- 벡터 인덱스 생성
CREATE VECTOR INDEX code_embeddings
FOR (n:CodeNode) 
ON n.embedding 
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}}

-- 텍스트 인덱스 생성  
CREATE TEXT INDEX node_names FOR (n:CodeNode) ON n.name
CREATE TEXT INDEX node_types FOR (n:CodeNode) ON n.type
```

### 의존성 설치
```toml
# pyproject.toml
[project]
dependencies = [
    "neo4j>=5.0.0",
    "google-generativeai>=0.3.0", 
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "code-analyzer-parser",
    "code-analyzer-graph"
]

[tool.uv.sources]
code-analyzer-parser = { path = "../parser", editable = true }
code-analyzer-graph = { path = "../graph", editable = true }
```

## 🧪 테스트 및 검증

### 통합 테스트 실행
```bash
cd packages/core
uv run python integration_test.py
```

### 예상 결과
```
🚀 통합 코드 분석 시스템 테스트
========================================

📁 Parser 분석:
   ✅ 34개 코드 블록 발견

🔄 Graph 변환:  
   ✅ 33개 노드 생성
   ✅ 15개 관계 생성

🔮 임베딩 생성:
   ✅ 33개 벡터 생성 (384차원)

🗄️ Neo4j 연결:
   ✅ 데이터베이스 연결 성공
   ⚠️ 실제 저장은 구현 중

🤖 AI 서비스:
   ⚠️ GEMINI_API_KEY 환경변수 필요

🎉 통합 테스트 완료!
```

### 성능 벤치마크
```python
import time
import psutil

async def benchmark_system():
    """시스템 성능 측정"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    analyzer = IntegratedCodeAnalyzer()
    result = await analyzer.analyze_project("./example_project")
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    print(f"📊 성능 리포트:")
    print(f"   • 처리 시간: {end_time - start_time:.2f}초")
    print(f"   • 메모리 사용: {end_memory - start_memory:.1f}MB 증가")
    print(f"   • 노드 처리 속도: {result['nodes_created']/(end_time - start_time):.1f} nodes/sec")
```

## 🚨 제한사항 및 해결책

### 현재 제한사항

1. **Neo4j 실제 저장 미완성**
```python
# 현재: 연결만 테스트
async def save_graph(self, graph: CodeGraph) -> bool:
    # TODO: 실제 노드/관계 저장 구현
    return True

# 해결 예정: 배치 저장 구현
async def save_graph(self, graph: CodeGraph) -> bool:
    async with self.driver.session() as session:
        # 노드 배치 생성
        await self._create_nodes_batch(session, graph.nodes)
        # 관계 배치 생성  
        await self._create_relations_batch(session, graph.relations)
        return True
```

2. **API 키 의존성**
```python
# 현재: 환경변수 없으면 AI 기능 비활성화
if os.getenv("GEMINI_API_KEY"):
    self.graph_rag = GraphRAGService(...)

# 개선 방향: 폴백 메커니즘
def initialize_ai_service(self):
    if os.getenv("GEMINI_API_KEY"):
        return GraphRAGService(...)
    elif os.getenv("OPENAI_API_KEY"):
        return OpenAIGraphRAGService(...)
    else:
        return DummyGraphRAGService()  # 기본 분석 제공
```

3. **대용량 데이터 처리**
```python
# 메모리 사용량 최적화 필요
# 스트리밍 처리 도입 예정
class StreamingCodeAnalyzer:
    async def analyze_large_project(self, 
                                  project_path: str,
                                  batch_size: int = 100):
        """대용량 프로젝트 스트리밍 분석"""
        async for batch in self._stream_files(project_path, batch_size):
            yield await self._process_batch(batch)
```

## 🔮 로드맵

### v0.2.0 - 완전한 데이터 지속성
- [ ] Neo4j 실제 CRUD 작업 구현
- [ ] 벡터 인덱스 자동 생성
- [ ] 배치 처리 최적화

### v0.3.0 - AI 기능 확장  
- [ ] 다중 AI 모델 지원 (OpenAI, Claude 등)
- [ ] 코드 리뷰 자동화
- [ ] 리팩토링 제안 기능

### v0.4.0 - 고급 분석 기능
- [ ] 코드 품질 점수 계산
- [ ] 취약점 탐지
- [ ] 성능 병목 지점 식별

### v1.0.0 - 운영 환경 준비
- [ ] 클러스터링 지원
- [ ] 모니터링 및 로깅
- [ ] API 서버 구현

## 📚 API 참조

### Neo4jHandler
```python
class Neo4jHandler:
    async def connect(self) -> bool: ...
    async def close(self) -> None: ...
    async def save_graph(self, graph: CodeGraph) -> bool: ...
    async def find_similar_nodes(self, embedding: list[float], limit: int = 5) -> list[dict]: ...
    async def query_cypher(self, query: str, parameters: dict = None) -> list[dict]: ...
```

### EmbeddingService  
```python
class EmbeddingService:
    def embed_code(self, code: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    def calculate_similarity(self, emb1: list[float], emb2: list[float]) -> float: ...
```

### GraphRAGService
```python
class GraphRAGService:
    async def analyze_code(self, query: str) -> str: ...
    async def find_similar_code(self, code_snippet: str, limit: int = 5) -> list[dict]: ...
    async def get_code_recommendations(self, context: str) -> list[str]: ...
```

## 💡 베스트 프랙티스

### 시스템 초기화
1. **순서 준수**: Neo4j → Embedding → AI 서비스 순으로 초기화
2. **연결 테스트**: 각 서비스 초기화 후 연결 상태 확인
3. **오류 처리**: 서비스 실패 시 graceful degradation

### 성능 최적화
1. **배치 처리**: 단일 작업보다 배치 처리 우선
2. **캐싱 전략**: 자주 사용되는 임베딩은 캐시 활용
3. **연결 풀링**: Neo4j 연결 풀을 통한 동시성 향상

### 보안 고려사항
1. **API 키 관리**: 환경변수 또는 보안 볼트 사용
2. **데이터베이스 인증**: 강력한 패스워드 및 SSL 연결
3. **입력 검증**: 사용자 입력에 대한 철저한 검증

---

---

## 🔄 최근 개선사항 (v0.2.0)

### LangChain/LangGraph 통합
- **LLM 관리**: 다양한 LLM 제공자를 통합 관리 (OpenAI, Google Gemini)
- **임베딩 서비스**: LangChain Embeddings로 통합 (OpenAI, Google, HuggingFace)
- **워크플로우 엔진**: LangGraph 기반 지능형 코드 분석 파이프라인
- **프롬프트 템플릿**: 작업별 최적화된 프롬프트 시스템

### 아키텍처 개선
```
Old: 직접 API 호출 → 단순 분석
New: LangChain → LangGraph → 워크플로우 기반 분석
```

### 새로운 워크플로우
1. **SIMPLE_SEARCH**: 기본 벡터 검색
2. **CONTEXTUAL_ANALYSIS**: 컨텍스트 기반 분석 
3. **ARCHITECTURE_REVIEW**: 아키텍처 평가
4. **CODE_SIMILARITY**: 유사 코드 패턴 분석
5. **REFACTORING_SUGGESTIONS**: 리팩토링 제안

### 기술 스택 업그레이드
- **LangChain 0.3.0+**: 통합 LLM 프레임워크
- **LangGraph 0.2.30+**: 상태 기반 워크플로우
- **LangSmith**: 모니터링 및 디버깅
- **다중 제공자 지원**: OpenAI, Google, HuggingFace

## 🧪 현대적 테스트 시스템

### 통합 테스트 실행
```bash
cd packages/core
uv run python integration_test.py
```

### 예상 결과 (v0.2.0)
```
🚀 LangChain/LangGraph 기반 현대적 코드 분석 시스템
======================================================================
📦 구성: Parser + Graph + LangChain + LangGraph + HuggingFace + Gemini

✅ Parser 초기화 완료
✅ Graph 어댑터 초기화 완료
✅ Core 서비스 초기화 완료
✅ 코드 블록 34개 분석 완료
✅ 그래프 변환 완료: 노드 33개, 관계 15개
✅ Neo4j 연결 성공
✅ 임베딩 생성 완료: 3/3개
✅ LLM 매니저 사용 가능
✅ LLM 코드 분석 성공
✅ GraphRAG 워크플로우 테스트 완료

🎉 현대적 시스템 통합 완료!
   ✅ Parser: Tree-sitter 기반 코드 분석
   ✅ Graph: Pydantic v2 데이터 모델
   ✅ Embedding: LangChain HuggingFace
   ✅ LLM: LangChain Gemini
   ✅ RAG: LangGraph 워크플로우
   ✅ Integration: 전체 파이프라인 연동
```

---

**패키지 버전**: v0.2.0  
**마지막 업데이트**: 2025-10-21  
**이전 패키지**: [Graph Package](graph.md) | **메인 문서**: [AGENTS.md](../AGENTS.md)