# Python Code Analyzer 📊

Tree-sitter를 사용하여 Python 코드의 구조와 의존성을 분석하고 Neo4j 그래프 데이터베이스에 저장하는 도구입니다.

## ✨ 주요 기능

- **🔍 코드 블록 분석**: 모듈, 클래스, 함수, import 문 자동 추출
- **🔗 의존성 분석**: 클래스 상속, 함수 호출, import 관계 추적
- **📊 Interactive 시각화**: 클릭 가능한 HTML 그래프 생성
- **💾 JSON 출력**: 구조화된 분석 결과 저장

## 🚀 사용법

### 기본 분석 (HTML 출력)
```bash
cd packages/parser
python main/main.py
```

### 출력 파일
- `output.json`: 분석 결과 JSON 형식
- `graph_visualization.html`: Interactive 시각화 (브라우저에서 열기)

### Neo4j GraphRAG 통합
Neo4j 그래프 데이터베이스와 임베딩 기반 GraphRAG는 core 패키지에서 처리됩니다:

```bash
cd packages/core

# 환경변수 설정
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USER="neo4j" 
export NEO4J_PASSWORD="your_password"
export OPENAI_API_KEY="your_openai_key"  # 임베딩용 (선택)

# 전체 파이프라인 실행
python main.py
```

## 📁 패키지 구조

```
packages/
├── parser/       # Python 코드 파싱 (순수 분석 기능)
│   ├── main/
│   │   ├── ast_extractor.py      # AST 추출
│   │   ├── code_block.py         # 코드 블록 모델
│   │   ├── graph_builder.py      # 코드 분석기
│   │   └── main.py              # 기본 실행
│   └── example_code/
│       ├── api.py               # 테스트용 API 코드
│       ├── database.py          # 테스트용 DB 코드
│       └── user.py              # 테스트용 User 코드
├── graph/        # 그래프 데이터 모델 및 어댑터
│   ├── src/
│   │   ├── models.py            # 그래프 데이터 모델
│   │   └── adapter.py           # 파서-그래프 어댑터
│   └── __init__.py
└── core/         # Neo4j 및 GraphRAG 통합 (메인 엔진)
    ├── src/
    │   ├── neo4j_handler.py     # Neo4j 데이터베이스 관리
    │   ├── embedding_service.py # 코드 임베딩 서비스
    │   ├── code_vectorizer.py   # 코드 벡터화
    │   └── graph_rag.py         # GraphRAG 서비스
    └── main.py                  # 통합 실행 스크립트
```

## 🔧 분석 결과 예시

### 추출되는 코드 블록 타입
- **Module**: 파일 단위 모듈
- **Class**: 클래스 정의 (상속 관계 포함)
- **Function**: 함수/메서드 정의
- **Import**: import/from 문

### 의존성 관계 타입
- **CALLS**: 함수 호출 관계
- **INHERITS**: 클래스 상속 관계  
- **IMPORTS**: 모듈 import 관계
- **CONTAINS**: 포함 관계 (클래스 → 메서드)

## 📊 분석 통계

현재 예제 코드 분석 결과:
- **총 34개 노드** (코드 블록)
- **총 47개 의존성** 관계
- **3개 파일** 분석

### 파일별 분석 결과
| 파일 | 블록 수 | 주요 구성 요소 |
|------|---------|----------------|
| api.py | 11개 | APIHandler, ResponseFormatter 클래스 |
| database.py | 12개 | DatabaseConnection, QueryBuilder 클래스 |
| user.py | 11개 | UserManager, AuthenticationService 클래스 |

## 🎯 주요 분석 기능

### 1. 클래스 의존성 분석
```python
# 타입 힌트 기반 의존성 추출
class APIHandler:
    def __init__(self, auth_service: AuthenticationService):  # ← 의존성 감지
        self.auth_service = auth_service
```

### 2. 함수 호출 관계 분석
```python
def handle_request(self, request: dict) -> dict:
    if method == "login":
        return self._handle_login(request)  # ← 호출 관계 감지
```

### 3. Import 관계 추적
```python
from user import AuthenticationService  # ← Import 의존성 감지
```

## 🔍 Interactive 시각화 기능

HTML 출력 파일(`graph_visualization.html`)에서 제공하는 기능:

- **📱 반응형 레이아웃**: 노드 목록 + 그래프 + 상세 패널
- **🎨 색상 구분**: 노드 타입별 색상 (모듈, 클래스, 함수, import)
- **🖱️ 인터랙티브**: 노드 클릭으로 상세 정보 표시
- **📊 통계**: 실시간 노드/엣지 개수 표시
- **🔍 탐색**: 줌, 팬, 노드 드래그 지원

## 🛠️ 개발자 정보

### 구현된 주요 개선사항 (2024-10-20)
- ✅ Tree-sitter 기반 정확한 AST 분석
- ✅ 타입별 코드 블록 분류 시스템
- ✅ 의존성 관계 타입 분류
- ✅ Interactive HTML 시각화
- ✅ 코드 리팩토링 및 최적화

### 기술 스택
- **Parser**: Tree-sitter (Python)
- **Visualization**: Vis.js + HTML/CSS/JavaScript
- **Data Format**: JSON
- **Language**: Python 3.13+

## 🧪 테스트

Parser 패키지는 **44개의 pytest 테스트**를 포함하며, **88% 이상의 코드 커버리지**를 달성합니다.

### 테스트 실행

```bash
cd packages/parser

# 모든 테스트 실행
uv run pytest tests/ -v

# 커버리지 보고서 생성
uv run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

### 테스트 통계

- **총 테스트 케이스**: 44개
- **테스트 성공률**: 100% ✅
- **코드 커버리지**: 88%
- **주요 테스트 영역**:
  - ✅ BaseParser (파서 초기화, 코드 파싱, 에러 처리)
  - ✅ ASTExtractor (블록 추출, docstring 추출, 의존성 분석)
  - ✅ CodeBlock (모델 생성, 복잡도 계산, 의존성 관리)
  - ✅ CodeAnalyzer (파일/디렉토리 분석, 통합 워크플로우)

### 테스트 파일

- `tests/test_parser.py` - CodeAnalyzer, CodeBlock 통합 테스트
- `tests/test_ast_extractor.py` - AST 추출기 테스트 (13개 케이스)
- `tests/test_base_parser.py` - 파서 기본 기능 및 에러 처리 (17개 케이스)
- `tests/test_docstring.py` - docstring 추출 검증

### 커버리지 세부 사항

```
src/__init__.py         3      0   100%
src/ast_extractor.py    158    17    89%
src/base_parser.py      24     3    88%
src/code_block.py       96     10    90%
src/graph_builder.py    65     12    82%
───────────────────────────────────
TOTAL                   346    42    88%
```

---

## ✨ 최근 개선사항 (v0.2.0)

### 🔧 리펙토링

- **타입 힌트 완성**: 모든 함수에 정확한 타입 힌트 적용 (Python 3.13+ 문법)
- **에러 처리 강화**: 구체적인 예외 처리 및 로깅 추가
- **문서화 개선**: 모든 클래스와 메서드에 상세한 docstring 추가
- **코드 품질**: 불필요한 import 제거, 로깅 시스템 추가

### 🧪 테스트 강화

- **44개의 포괄적인 테스트**: 기본 기능부터 엣지 케이스까지 검증
- **88% 커버리지**: 핵심 기능의 높은 커버리지 달성
- **pytest 통합**: 체계적인 테스트 구조 및 픽스처 활용
- **CI/CD 준비**: pytest.ini 및 설정 파일 완비

### 📊 성능 지표

- **테스트 실행 시간**: 0.26초 (44개 테스트)
- **파싱 속도**: 34개 블록/초 (변경 없음)
- **코드 커버리지**: 88% (목표: 85% 달성 ✅)

