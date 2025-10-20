# Code Analyzer - AI Agent Project

## 🎯 프로젝트 개요

Code Analyzer는 Python 코드를 분석하여 그래프 구조로 변환하고, Neo4j 데이터베이스에 저장하여 GraphRAG 시스템을 구축하는 AI 기반 코드 분석 도구입니다.

## 🏗️ 시스템 아키텍처

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Parser    │───▶│    Graph    │───▶│    Core     │
│ (Tree-sitter)│    │  (Models)   │    │(Neo4j+RAG) │
└─────────────┘    └─────────────┘    └─────────────┘
      │                     │                  │
      ▼                     ▼                  ▼
 코드 파싱              그래프 변환         AI 검색
 AST 추출              데이터 모델         임베딩 생성
 블록 생성              관계 매핑           GraphRAG
```

## 📦 패키지 구조

### 🔍 Parser Package
- **위치**: `packages/parser/`
- **역할**: Python 코드 파싱 및 AST 추출
- **기술 스택**: Tree-sitter, Python AST
- **주요 기능**:
  - Python 파일 분석
  - 함수, 클래스, 모듈 추출
  - 의존성 관계 파악
  - 코드 블록 생성

### 🕸️ Graph Package
- **위치**: `packages/graph/`
- **역할**: 데이터 모델 및 그래프 변환
- **기술 스택**: Pydantic v2, Python 3.13+
- **주요 기능**:
  - CodeNode, CodeRelation 모델
  - Parser 결과를 Graph 모델로 변환
  - Neo4j 형식 데이터 준비
  - 그래프 검증 및 분석

### 💾 Core Package
- **위치**: `packages/core/`
- **역할**: 데이터베이스 관리 및 AI 서비스
- **기술 스택**: Neo4j, LangChain, LangGraph
- **주요 기능**:
  - Neo4j 데이터베이스 연결 및 관리
  - 코드 임베딩 생성 (로컬 모델)
  - GraphRAG 검색 및 추천
  - AI 기반 코드 분석

## 🔄 데이터 플로우

```
1. 📂 Python Files
        ↓
2. 🔍 Parser (Tree-sitter)
        ↓
3. 📋 CodeBlock Objects
        ↓
4. 🕸️ Graph Adapter
        ↓
5. 🏗️ CodeGraph Model
        ↓
6. 💾 Neo4j Database
        ↓
7. 🤖 Embedding Service
        ↓
8. 🔍 GraphRAG Search
        ↓
9. 🤖 AI Analysis (Gemini)
```

## 🚀 주요 기능

### 코드 분석
- **Python 파일 자동 스캔**: 프로젝트 디렉토리 전체 분석
- **AST 기반 파싱**: Tree-sitter를 활용한 정확한 구문 분석
- **의존성 추출**: 함수 호출, 클래스 상속, import 관계 파악
- **메타데이터 수집**: 복잡도, 스코프, 문서화 수준 분석

### 그래프 데이터베이스
- **Neo4j 통합**: 코드 구조를 그래프로 저장
- **관계 매핑**: 코드 간의 다양한 관계 표현
- **벡터 인덱스**: 의미적 검색을 위한 임베딩 저장
- **스키마 관리**: 자동 제약조건 및 인덱스 생성

### AI 기반 검색
- **의미적 검색**: 자연어 쿼리로 코드 검색
- **GraphRAG**: 그래프 기반 컨텍스트 생성
- **코드 추천**: 유사한 코드 패턴 제안
- **AI 분석**: Google Gemini를 활용한 코드 리뷰

## 🛠️ 기술 스택

### 언어 & 런타임
- **Python 3.13+**: 최신 타입 힌트 및 성능 개선 활용
- **UV Package Manager**: 빠른 의존성 관리

### 코드 분석
- **Tree-sitter 0.25+**: 고성능 파서 생성기
- **Tree-sitter-python**: Python 언어 바인딩

### 데이터 모델링
- **Pydantic v2**: 타입 안전한 데이터 검증
- **Python Dataclasses**: 경량 데이터 구조

### 데이터베이스
- **Neo4j 6.0+**: 그래프 데이터베이스
- **Vector Index**: 임베딩 기반 검색

### AI & ML
- **Sentence Transformers**: 로컬 임베딩 모델
- **LangChain & LangGraph**: LLM 관리

## 📋 설치 및 실행

### 환경 설정
```bash
# 각 패키지별 의존성 설치
cd packages/parser && uv sync
cd packages/graph && uv sync  
cd packages/core && uv sync
```

### 환경 변수
```bash
# Neo4j 설정
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# AI 서비스 (선택적)
export GEMINI_API_KEY="your_gemini_key"
export OPENAI_API_KEY="your_openai_key"
```

### 실행
```bash
# 전체 시스템 통합 테스트
cd packages/core
uv run python integration_test.py

# 개별 패키지 테스트
cd packages/parser && uv run python test_parser.py
cd packages/core && uv run python simple_test.py
```

## 🧪 테스트 현황

### ✅ 완료된 테스트
- **Parser**: 34개 코드 블록 분석 성공
- **Graph**: 33개 노드, 15개 관계 변환 성공
- **Core**: 임베딩 생성 및 서비스 초기화 성공
- **Integration**: Parser → Graph → Core 파이프라인 작동

### ⚠️ 개발 중
- Neo4j 실제 데이터 저장 기능
- GraphRAG 검색 최적화
- 대용량 프로젝트 처리 성능

## 🔮 향후 계획

### Phase 1: Core 기능 완성
- [ ] Neo4j 데이터 저장/조회 완전 구현
- [ ] GraphRAG 검색 정확도 향상
- [ ] 성능 최적화 및 메모리 관리

### Phase 2: 기능 확장
- [ ] 다중 언어 지원 (JavaScript, TypeScript)
- [ ] 웹 인터페이스 개발
- [ ] 실시간 코드 분석

### Phase 3: AI 고도화  
- [ ] 코드 품질 자동 평가
- [ ] 리팩토링 제안 시스템
- [ ] 자동 문서화 생성

## 📊 시스템 메트릭

### 현재 성능
- **파싱 속도**: ~34 블록/초
- **그래프 변환**: 즉시 처리
- **임베딩 생성**: ~3초/배치 (로컬 모델)
- **메모리 사용량**: ~200MB (중간 규모 프로젝트)

### 확장성
- **지원 파일 수**: 제한 없음
- **최대 노드 수**: Neo4j 제한에 따름
- **동시 처리**: 단일 스레드 (향후 병렬화 예정)

## 🏷️ 태그 및 분류

**카테고리**: Code Analysis, Graph Database, AI/ML, Developer Tools
**기술 키워드**: Python, Neo4j, Tree-sitter, GraphRAG, Embeddings
**사용 사례**: Code Review, Documentation, Refactoring, Learning