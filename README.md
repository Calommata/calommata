# Code Analyzer with GraphRAG 🚀

Tree-sitter 기반 Python 코드 분석 및 Neo4j GraphRAG 시스템

## ✨ 주요 기능

- **🔍 정확한 코드 분석**: Tree-sitter를 활용한 AST 기반 코드 파싱
- **📊 그래프 구조화**: 코드 블록과 의존성을 그래프로 모델링
- **🤖 코드 임베딩**: OpenAI 또는 로컬 모델을 활용한 코드 벡터화
- **🔍 의미적 검색**: 자연어 쿼리로 관련 코드 블록 검색
- **💡 스마트 추천**: 유사/관련/컨텍스트 기반 코드 추천
- **🔗 의존성 분석**: 코드 간 관계 구조 시각화
- **📈 GraphRAG**: Neo4j 그래프 데이터베이스 기반 검색 증강 생성

## 🏗️ 아키텍처

```
packages/
├── parser/       # 코드 파싱 엔진
│   ├── main/
│   │   ├── ast_extractor.py      # Tree-sitter AST 추출
│   │   ├── code_block.py         # 코드 블록 모델
│   │   ├── graph_builder.py      # 코드 분석기
│   │   └── main.py              # 파싱 전용 실행
│   └── example_code/            # 테스트 코드
│
├── graph/        # 그래프 데이터 모델
│   ├── src/
│   │   ├── models.py            # Pydantic 그래프 모델
│   │   └── adapter.py           # 파서-그래프 어댑터
│   └── __init__.py
│
└── core/         # GraphRAG 메인 엔진
    ├── src/
    │   ├── neo4j_handler.py     # Neo4j 데이터베이스 관리
    │   ├── embedding_service.py # 코드 임베딩 생성
    │   ├── code_vectorizer.py   # 벡터화 처리
    │   └── graph_rag.py         # GraphRAG 서비스
    ├── main.py                  # 전체 파이프라인 실행
    └── demo.py                  # GraphRAG 데모
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# Neo4j 환경변수 설정
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# OpenAI API (선택사항 - 로컬 임베딩도 지원)
export OPENAI_API_KEY="your_openai_key"
```

### 2. 의존성 설치
```bash
# Core 패키지 (메인 엔진)
cd packages/core
uv add neo4j pydantic openai sentence-transformers numpy

# Parser 패키지 (코드 분석)
cd packages/parser
pip install tree-sitter pydantic
```

### 3. 전체 파이프라인 실행
```bash
cd packages/core
python main.py
```

### 4. GraphRAG 데모 실행
```bash
cd packages/core
python demo.py
```

## 📊 처리 파이프라인

1. **코드 파싱** (Parser 패키지)
   - Tree-sitter로 Python AST 추출
   - 클래스, 함수, 모듈, import 블록 식별
   - 의존성 관계 분석

2. **그래프 변환** (Graph 패키지)
   - 파싱 결과를 Pydantic 모델로 변환
   - Neo4j 호환 형식으로 직렬화

3. **데이터 저장** (Core 패키지)
   - Neo4j 그래프 데이터베이스에 저장
   - 벡터 인덱스 및 제약 조건 생성

4. **코드 벡터화** (Core 패키지)
   - OpenAI 또는 sentence-transformers로 임베딩 생성
   - Neo4j 벡터 인덱스에 저장

5. **GraphRAG 서비스** (Core 패키지)
   - 자연어 쿼리 → 임베딩 변환
   - 벡터 유사도 검색
   - 그래프 구조 기반 컨텍스트 생성

## 🔍 GraphRAG 기능

### 자연어 코드 검색
```python
# 예시 쿼리들
"database connection and query execution"
"user authentication and login" 
"API request handling"
"error handling and exceptions"
```

### 코드 추천
- **유사 코드**: 임베딩 기반 의미적 유사성
- **관련 코드**: 그래프 구조 기반 의존성
- **컨텍스트 기반**: 같은 파일/모듈 내 관련 요소

### 의존성 분석
- **직접 의존성**: 호출, 상속, import 관계
- **역방향 의존성**: 이 코드를 사용하는 다른 코드
- **컨텍스트 분석**: 같은 스코프 내 관련 요소

## 🎯 사용 사례

### 1. 코드 리뷰 및 분석
```bash
# 전체 프로젝트 분석
python core/main.py

# 특정 기능 검색
"find authentication related code"
```

### 2. 기술 부채 식별
```bash
# 복잡한 의존성 구조 분석
python core/demo.py -> 의존성 분석 메뉴
```

### 3. 코드 리팩토링 지원
```bash
# 유사한 패턴의 코드 찾기
"find similar database access patterns"
```

### 4. 개발자 온보딩
```bash
# 프로젝트 구조 이해
python core/demo.py -> 코드 검색 메뉴
```

## 📈 성능 특징

### 분석 속도
- **Tree-sitter**: 빠르고 정확한 AST 파싱
- **배치 처리**: 대량 코드 블록 병렬 임베딩
- **인덱싱**: Neo4j 벡터 인덱스로 빠른 검색

### 확장성
- **모듈식 아키텍처**: 각 패키지 독립적 사용 가능
- **다중 임베딩 모델**: OpenAI/로컬/커스텀 모델 지원
- **그래프 확장**: 새로운 노드/관계 타입 쉽게 추가

### 정확도
- **구문 인식**: Tree-sitter의 정확한 AST 분석
- **의미적 검색**: 코드 의미 기반 임베딩
- **컨텍스트 보강**: 그래프 구조로 관련성 강화

## 🛠️ 개발 정보

### 기술 스택
- **파싱**: Tree-sitter (Python)
- **데이터베이스**: Neo4j 6.0.2+
- **임베딩**: OpenAI API / sentence-transformers
- **모델링**: Pydantic v2
- **언어**: Python 3.13+

### 확장 계획
- [ ] 다른 언어 지원 (JavaScript, TypeScript 등)
- [ ] 웹 인터페이스 개발
- [ ] 실시간 코드 분석
- [ ] CI/CD 통합
- [ ] 코드 품질 메트릭

## 📝 예시 결과

### 분석 통계
```
📊 코드 분석 및 GraphRAG 처리 완료
=================================================
🎯 프로젝트: example_code
📂 경로: /path/to/project

📈 Neo4j 데이터베이스 통계:
  • 총 노드: 34개
  • 총 관계: 47개

🏷️ 노드 타입별 분포:
  • Module: 3개
  • Class: 6개  
  • Function: 15개
  • Method: 8개
  • Import: 2개

🤖 벡터화 통계:
  • 벡터화된 노드: 34개
  • 진행률: 100.0%
  • 임베딩 모델: text-embedding-3-small
```

### 검색 결과 예시
```
🔍 검색 중: 'database connection and query execution'

📝 'database connection and query execution'에 대해 3개의 매칭 코드를 2개 파일에서 발견했습니다. 주요 타입: Class

1. DatabaseConnection (Class)
   📁 database.py
   📏 줄: 10-25
   🎯 유사도: 0.892
   🔗 관련 함수: connect, execute_query, close

2. QueryBuilder (Class)  
   📁 database.py
   📏 줄: 45-80
   🎯 유사도: 0.856
   🔗 관련 함수: build_select, build_insert, build_update
```

## 🤝 기여하기

1. Fork 프로젝트
2. Feature 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

---

*Python Code Analyzer with GraphRAG - Tree-sitter + Neo4j + AI로 코드를 이해하다* 🚀
