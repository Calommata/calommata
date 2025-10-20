# CLI Package Documentation

## 📋 패키지 개요

CLI 패키지는 Code Analyzer 시스템의 명령줄 인터페이스를 제공합니다. 사용자가 터미널에서 직접 코드 분석, 그래프 생성, AI 질의 등의 작업을 수행할 수 있도록 합니다.

## 🏗️ 아키텍처

```
CLI Package
├── src/
│   ├── __init__.py
│   ├── main.py           # CLI 진입점
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── analyze.py    # 코드 분석 명령
│   │   ├── query.py      # GraphRAG 질의 명령
│   │   ├── export.py     # 결과 내보내기 명령
│   │   └── setup.py      # 시스템 설정 명령
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py     # 설정 관리
│   │   ├── output.py     # 출력 포맷팅
│   │   └── progress.py   # 진행률 표시
│   └── templates/
│       ├── config.yaml   # 기본 설정 템플릿
│       └── report.html   # 리포트 템플릿
└── pyproject.toml        # 패키지 설정
```

## 🔧 주요 기능

### 코드 분석 (analyze 명령)
```bash
# 디렉토리 분석
code-analyzer analyze ./my-project

# 특정 파일 분석
code-analyzer analyze ./src/main.py

# 상세 분석 (AI 포함)
code-analyzer analyze ./project --ai --verbose

# 결과를 Neo4j에 저장
code-analyzer analyze ./project --save-to-neo4j
```

### GraphRAG 질의 (query 명령)
```bash
# 코드베이스에 대한 질문
code-analyzer query "What are the main components?"

# 유사 코드 검색
code-analyzer query --similar "def calculate_total"

# 코드 리뷰 요청
code-analyzer query --review "./src/utils.py"
```

### 결과 내보내기 (export 명령)
```bash
# JSON 형태로 내보내기
code-analyzer export --format json --output results.json

# HTML 리포트 생성
code-analyzer export --format html --output report.html

# 그래프 시각화
code-analyzer export --format graphviz --output graph.dot
```

### 시스템 설정 (setup 명령)
```bash
# 초기 설정
code-analyzer setup init

# Neo4j 연결 설정
code-analyzer setup neo4j --uri bolt://localhost:7687

# AI API 키 설정
code-analyzer setup ai --provider gemini --api-key YOUR_KEY
```

## 📋 명령어 상세

### analyze 명령
```bash
code-analyzer analyze [PATH] [OPTIONS]

Options:
  --recursive, -r         하위 디렉토리 포함
  --include PATTERN       포함할 파일 패턴
  --exclude PATTERN       제외할 파일 패턴
  --ai                    AI 분석 포함
  --save-to-neo4j        Neo4j에 결과 저장
  --verbose, -v          상세 출력
  --quiet, -q            최소 출력
  --output, -o FILE      결과를 파일로 저장
  --format FORMAT        출력 형식 (json|yaml|table)
```

### query 명령
```bash
code-analyzer query [QUESTION] [OPTIONS]

Options:
  --similar CODE         유사 코드 검색
  --review FILE          코드 리뷰 요청
  --context N            컨텍스트 노드 수 (기본값: 5)
  --model MODEL          AI 모델 선택
  --temperature TEMP     AI 응답 온도 (0.0-1.0)
  --max-tokens N         최대 토큰 수
```

### export 명령
```bash
code-analyzer export [OPTIONS]

Options:
  --format FORMAT        출력 형식 (json|html|csv|graphviz)
  --output, -o FILE      출력 파일명
  --template TMPL        사용자 정의 템플릿
  --include-source       소스 코드 포함
  --include-embeddings   임베딩 벡터 포함
```

## 🔍 사용 예시

### 프로젝트 분석 시나리오
```bash
# 1. 초기 설정
code-analyzer setup init

# 2. 프로젝트 분석
code-analyzer analyze ./my-python-project \
  --recursive \
  --exclude "*.pyc,__pycache__" \
  --ai \
  --save-to-neo4j \
  --verbose

# 3. 분석 결과 질의
code-analyzer query "What are the most complex functions?"

# 4. HTML 리포트 생성
code-analyzer export --format html --output analysis-report.html
```

### 코드 리뷰 시나리오
```bash
# 특정 파일 리뷰
code-analyzer query --review ./src/database.py

# 유사한 코드 검색
code-analyzer query --similar "async def connect_database"

# 개선점 질의
code-analyzer query "How can I improve the performance of this code?"
```

## ⚙️ 설정 관리

### 설정 파일 (~/.code-analyzer/config.yaml)
```yaml
# 기본 설정
default:
  include_patterns: ["*.py"]
  exclude_patterns: ["*.pyc", "__pycache__", ".git"]
  max_file_size: 10MB
  
# Neo4j 설정
neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j" 
  password: "password"
  database: "code_analyzer"

# AI 서비스 설정
ai:
  provider: "gemini"  # gemini, openai, claude
  model: "gemini-2.5-flash"
  temperature: 0.1
  max_tokens: 4000
  
# 임베딩 설정
embedding:
  model: "all-MiniLM-L6-v2"
  cache_dir: "~/.code-analyzer/embeddings"
  batch_size: 32

# 출력 설정
output:
  format: "table"  # table, json, yaml
  colors: true
  progress_bar: true
  timestamp: true
```

### 환경변수 지원
```bash
# API 키
export GEMINI_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"

# Neo4j 연결
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

# 임베딩 모델 캐시
export EMBEDDING_CACHE_DIR="./models"
```

## 🎨 출력 형식

### 테이블 형식 (기본)
```
📊 코드 분석 결과
========================================
파일: ./src/main.py
라인: 1-50
블록: 3개 (1 class, 2 functions)

┌─────────────┬──────────┬─────────┬─────────────┐
│ 이름        │ 타입     │ 복잡도  │ 라인        │
├─────────────┼──────────┼─────────┼─────────────┤
│ UserManager │ class    │ 8       │ 5-30        │
│ __init__    │ function │ 2       │ 7-10        │
│ add_user    │ function │ 4       │ 12-25       │
└─────────────┴──────────┴─────────┴─────────────┘
```

### JSON 형식
```json
{
  "analysis_timestamp": "2025-01-20T10:30:00Z",
  "project_path": "./src",
  "summary": {
    "total_files": 15,
    "total_blocks": 89,
    "total_nodes": 85,
    "total_relations": 42
  },
  "files": [
    {
      "path": "./src/main.py",
      "blocks": [
        {
          "name": "UserManager",
          "type": "class",
          "start_line": 5,
          "end_line": 30,
          "complexity": 8,
          "docstring": "사용자 관리 클래스"
        }
      ]
    }
  ]
}
```

### HTML 리포트
```html
<!DOCTYPE html>
<html>
<head>
    <title>Code Analysis Report</title>
    <style>
        /* 반응형 대시보드 스타일 */
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>프로젝트 분석 리포트</h1>
        
        <div class="summary-cards">
            <div class="card">
                <h3>총 파일 수</h3>
                <span class="metric">15</span>
            </div>
            <!-- 더 많은 메트릭 카드들 -->
        </div>
        
        <div class="graphs">
            <!-- 복잡도 분포 차트 -->
            <!-- 의존성 그래프 -->
        </div>
        
        <div class="details">
            <!-- 상세 코드 블록 정보 -->
        </div>
    </div>
</body>
</html>
```

## 🧪 테스트 및 검증

### CLI 테스트 스크립트
```bash
#!/bin/bash
# test_cli.sh

echo "🧪 CLI 패키지 테스트 시작"

# 기본 명령 테스트
echo "1. 도움말 테스트"
code-analyzer --help

echo "2. 분석 명령 테스트"
code-analyzer analyze ./example_code --quiet

echo "3. 설정 테스트"
code-analyzer setup init

echo "4. 내보내기 테스트"
code-analyzer export --format json --output test_result.json

echo "✅ CLI 테스트 완료"
```

### 통합 테스트
```python
import subprocess
import json
import tempfile

def test_cli_integration():
    """CLI 통합 테스트"""
    
    # 임시 디렉토리에서 테스트
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. 분석 실행
        result = subprocess.run([
            "code-analyzer", "analyze", "./example_code",
            "--output", f"{tmp_dir}/result.json",
            "--format", "json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        
        # 2. 결과 검증
        with open(f"{tmp_dir}/result.json") as f:
            data = json.load(f)
            assert "summary" in data
            assert data["summary"]["total_blocks"] > 0
        
        print("✅ CLI 통합 테스트 통과")
```

## 🚨 에러 처리

### 일반적인 에러와 해결책

1. **연결 에러**
```bash
❌ Neo4j 연결 실패: bolt://localhost:7687
💡 해결책: Neo4j 서버가 실행 중인지 확인하세요.

# 연결 테스트
code-analyzer setup neo4j --test
```

2. **API 키 에러**
```bash
❌ AI 서비스 인증 실패
💡 해결책: API 키를 확인하세요.

# API 키 설정
code-analyzer setup ai --provider gemini --api-key YOUR_KEY
```

3. **권한 에러**
```bash
❌ 파일 읽기 권한 없음: ./private/
💡 해결책: 파일 권한을 확인하거나 --exclude 옵션을 사용하세요.

# 권한 문제가 있는 디렉토리 제외
code-analyzer analyze . --exclude "private/,secrets/"
```

## 🔮 로드맵

### v0.2.0 - 고급 CLI 기능
- [ ] 대화형 모드 (--interactive)
- [ ] 자동 완성 스크립트 생성
- [ ] 설정 검증 및 진단 도구

### v0.3.0 - 통합 도구
- [ ] VS Code 확장 연동
- [ ] Git 훅 통합
- [ ] CI/CD 파이프라인 지원

### v1.0.0 - 엔터프라이즈 기능
- [ ] 멀티 프로젝트 관리
- [ ] 팀 협업 기능
- [ ] 분석 결과 비교 도구

## 📚 API 참조

### 주요 클래스 및 함수
```python
# src/main.py
def main() -> int:
    """CLI 메인 진입점"""
    
class CLIApplication:
    def __init__(self, config: Config): ...
    def run(self, args: list[str]) -> int: ...

# src/commands/analyze.py  
class AnalyzeCommand:
    async def execute(self, args: AnalyzeArgs) -> int: ...
    
# src/utils/config.py
class Config:
    @classmethod
    def load(cls, config_path: str = None) -> 'Config': ...
    def save(self, config_path: str = None) -> None: ...
```

## 💡 베스트 프랙티스

### 명령어 사용법
1. **프로젝트 루트에서 실행**: 상대 경로 문제 방지
2. **설정 파일 활용**: 반복적인 옵션 지정 대신 설정 파일 사용
3. **배치 처리**: 큰 프로젝트는 여러 번에 나누어 분석

### 성능 최적화
1. **제외 패턴 활용**: 불필요한 파일은 분석에서 제외
2. **병렬 처리**: `--parallel` 옵션으로 처리 속도 향상
3. **캐싱 활용**: 임베딩 결과 캐시로 재분석 시간 단축

### 결과 활용
1. **정기적 분석**: 코드 품질 모니터링을 위한 정기 실행
2. **리포트 공유**: HTML 형식으로 팀원들과 결과 공유
3. **추세 분석**: 시간에 따른 코드 복잡도 변화 추적

---

**패키지 버전**: v0.1.0 (예정)  
**마지막 업데이트**: 2025-10-20  
**연관 패키지**: [Parser](parser.md) | [Graph](graph.md) | [Core](core.md)