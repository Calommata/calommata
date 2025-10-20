# Parser Package Documentation

## 📋 패키지 개요

Parser 패키지는 Python 코드를 분석하여 구조화된 코드 블록으로 변환하는 핵심 모듈입니다. Tree-sitter를 기반으로 한 고성능 파싱 엔진을 제공합니다.

## 🏗️ 아키텍처

```
Parser Package
├── main/
│   ├── __init__.py          # 패키지 진입점
│   ├── graph_builder.py     # 메인 분석기
│   ├── base_parser.py       # Tree-sitter 파서 래퍼
│   ├── ast_extractor.py     # AST 블록 추출기
│   └── code_block.py        # 데이터 모델
├── example_code/            # 테스트용 샘플 코드
└── test_parser.py          # 단위 테스트
```

## 🔧 주요 컴포넌트

### CodeAnalyzer (graph_builder.py)
메인 분석 엔진으로 디렉토리 전체 또는 개별 파일을 분석합니다.

```python
from parser.main.graph_builder import CodeAnalyzer

analyzer = CodeAnalyzer()
blocks = analyzer.analyze_directory("./src")
```

**주요 메서드**:
- `analyze_directory(dir_path: str) -> list[CodeBlock]`: 디렉토리 분석
- `analyze_file(file_path: str) -> list[CodeBlock]`: 단일 파일 분석
- `get_all_blocks() -> list[CodeBlock]`: 분석된 모든 블록 반환

### BaseParser (base_parser.py)
Tree-sitter 파서의 래퍼 클래스입니다.

```python
from parser.main.base_parser import BaseParser
import tree_sitter_python as tslanguage

parser = BaseParser(tslanguage.language())
tree = parser.parse_code(source_code)
```

**기능**:
- Tree-sitter 언어 객체 관리
- 소스 코드를 구문 트리로 변환
- 파싱 에러 처리

### ASTExtractor (ast_extractor.py)
구문 트리에서 의미있는 코드 블록을 추출합니다.

```python
from parser.main.ast_extractor import ASTExtractor

extractor = ASTExtractor(language)
blocks = extractor.extract_blocks(tree, source_code)
```

**추출 가능한 블록 타입**:
- `module`: Python 모듈
- `class`: 클래스 정의
- `function`: 함수/메서드 정의
- `import`: import 문
- `variable`: 변수 할당

### CodeBlock (code_block.py)
코드 블록을 표현하는 데이터 모델입니다.

```python
@dataclass
class CodeBlock:
    block_type: str          # 블록 타입
    name: str               # 블록 이름
    start_line: int         # 시작 라인
    end_line: int          # 종료 라인
    file_path: str          # 파일 경로 (v0.1.1에서 추가)
    source_code: str       # 소스 코드
    docstring: str         # 문서화 문자열
    dependencies: list[str] # 의존성 목록
    complexity: int        # 복잡도 점수
    scope_level: int       # 스코프 깊이
    typed_dependencies: list[Dependency] # 타입별 의존성
```

## 📊 분석 결과 예시

### 입력: Python 파일
```python
# example.py
import os
from typing import List

class UserManager:
    """사용자 관리 클래스"""
    
    def __init__(self):
        self.users = []
    
    def add_user(self, name: str) -> bool:
        """사용자 추가"""
        if name not in self.users:
            self.users.append(name)
            return True
        return False
```

### 출력: CodeBlock 리스트
```
1. Module Block: example (lines 1-16)
2. Import Block: import_os (line 1)
3. Import Block: import_typing (line 2)
4. Class Block: UserManager (lines 4-16)
5. Function Block: __init__ (lines 7-8)
6. Function Block: add_user (lines 10-16)
```

## 🔍 사용 예시

### 기본 사용법
```python
from parser.main.graph_builder import CodeAnalyzer

# 분석기 초기화
analyzer = CodeAnalyzer()

# 디렉토리 분석
blocks = analyzer.analyze_directory("./my_project")

print(f"분석된 블록 수: {len(blocks)}")

# 블록 타입별 분류
for block in blocks:
    print(f"{block.block_type}: {block.name} ({block.start_line}-{block.end_line})")
```

### 고급 사용법
```python
# 특정 타입의 블록만 필터링
functions = [b for b in blocks if b.block_type == "function"]
classes = [b for b in blocks if b.block_type == "class"]

# 복잡도 기준 정렬
complex_blocks = sorted(blocks, key=lambda x: x.complexity, reverse=True)

# 의존성 분석
for block in blocks:
    if block.dependencies:
        print(f"{block.name} depends on: {block.dependencies}")
```

## ⚙️ 설정 및 최적화

### 의존성 설치
```toml
# pyproject.toml
[project]
dependencies = [
    "tree-sitter>=0.25.2",
    "tree-sitter-python>=0.25.0",
]
```

### 성능 최적화
- **병렬 처리**: 큰 프로젝트의 경우 파일별 병렬 분석 고려
- **메모리 관리**: 대용량 파일은 스트리밍 방식으로 처리
- **캐싱**: 변경되지 않은 파일의 분석 결과 캐시

## 🧪 테스트

### pytest 테스트 실행 (권장)
```bash
cd packages/parser
uv sync --extra test    # 테스트 의존성 설치
uv run pytest          # 모든 테스트 실행
uv run pytest --cov=main --cov-report=html  # 커버리지 포함
```

### 테스트 결과
```
================================= test session starts ==================================
platform win32 -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0
rootdir: packages/parser
configfile: pytest.ini
plugins: cov-7.0.0
collected 13 items

tests\test_parser.py .............                                                [100%]

================================== 13 passed in 0.14s ==================================

==================================== coverage report ====================================
Name                    Stmts   Miss  Cover
-------------------------------------------
main\__init__.py            3      0   100%
main\ast_extractor.py     129     15    88%
main\base_parser.py         8      0   100%
main\code_block.py         80      3    96%
main\graph_builder.py      47      5    89%
main\main.py               21     21     0%
-------------------------------------------
TOTAL                     288     44    85%
```

### 기존 테스트 (Legacy)
```bash
uv run python legacy_test_parser.py
```

## 🚨 제한사항 및 알려진 이슈

### 현재 제한사항
- **Python만 지원**: 다른 언어는 추가 개발 필요
- **의존성 분석 제한**: 복잡한 import 패턴 미지원
- **성능 최적화 필요**: 큰 프로젝트에서 메모리 사용량 증가

### ✅ 해결된 이슈 (v0.1.1)
- **file_path 속성 추가**: CodeBlock에 파일 경로 정보 포함
- **타입 힌트 개선**: 모든 함수와 메서드에 적절한 타입 힌트 적용
- **pytest 테스트 도입**: 13개 테스트 케이스, 85% 커버리지 달성

## 🔮 로드맵

### ✅ 완료된 목표 (v0.1.1)
- [x] CodeBlock에 file_path 속성 추가
- [x] 전체 타입 힌트 개선
- [x] pytest 테스트 도입 (13개 테스트, 85% 커버리지)
- [x] 코드 품질 개선 (Ruff 린터 적용)

### 단기 목표 (v0.2.0)
- [ ] 더 정교한 의존성 분석 (import alias, nested import 지원)
- [ ] 에러 처리 개선 및 로깅 추가
- [ ] docstring 추출 기능 강화
- [ ] 테스트 커버리지 95% 이상 달성

### 중기 목표 (v0.3.0)
- [ ] JavaScript/TypeScript 지원
- [ ] 병렬 처리 구현 (멀티프로세싱)
- [ ] 메모리 사용량 최적화
- [ ] 설정 파일 지원 (.parserrc)

### 장기 목표 (v1.0.0)
- [ ] 다중 언어 통합 분석
- [ ] 크로스 언어 의존성 추적
- [ ] 실시간 분석 지원 (파일 와처)
- [ ] CLI 도구 개발

## 📚 API 참조

### CodeAnalyzer API
```python
class CodeAnalyzer:
    def __init__(self) -> None: ...
    def analyze_directory(self, dir_path: str) -> list[CodeBlock]: ...
    def analyze_file(self, file_path: str) -> list[CodeBlock]: ...
    def get_all_blocks(self) -> list[CodeBlock]: ...
```

### CodeBlock API
```python
@dataclass
class CodeBlock:
    block_type: str
    name: str
    start_line: int
    end_line: int
    source_code: str | None = None
    docstring: str | None = None
    dependencies: list[str] | None = None
    complexity: int = 0
    scope_level: int = 0
    
    def get_full_name(self) -> str: ...
    def add_dependency(self, target: str, dep_type: DependencyType) -> None: ...
```

## 💡 베스트 프랙티스

### 코드 분석 시
1. **디렉토리 구조 확인**: 분석 전 프로젝트 구조 파악
2. **점진적 분석**: 작은 모듈부터 시작하여 전체로 확장
3. **결과 검증**: 분석 결과의 정확성 수동 확인

### 성능 고려사항
1. **큰 파일 주의**: 10,000줄 이상 파일은 분할 고려
2. **메모리 모니터링**: 대량 분석 시 메모리 사용량 체크
3. **타임아웃 설정**: 무한 루프 방지를 위한 시간 제한

---

## 📈 최근 개선사항 (v0.1.1)

### 🔧 코드 리팩토링
- **file_path 속성 추가**: 모든 CodeBlock에 파일 경로 정보 포함
- **타입 힌트 완성**: 모든 함수와 메서드에 정확한 타입 힌트 적용
- **docstring 추출**: 클래스와 함수의 문서화 문자열 자동 추출

### 🧪 테스트 개선
- **pytest 도입**: 기존 단순 테스트에서 체계적인 pytest 테스트로 전환
- **14개 테스트 케이스**: 주요 기능별 상세 테스트 구현
- **84% 커버리지**: 높은 코드 커버리지로 품질 보장
- **CI/CD 준비**: pytest.ini 및 설정 파일 완비

### 📊 성능 지표
- **분석 속도**: 34개 블록/초 (변경 없음)
- **메모리 효율**: 타입 힌트로 최적화
- **docstring 추출**: 21/34개 블록에서 성공적으로 추출
- **테스트 실행 시간**: 0.27초 (14개 테스트)

---

**패키지 버전**: v0.1.1  
**마지막 업데이트**: 2025-10-20  
**다음 패키지**: [Graph Package](graph.md)