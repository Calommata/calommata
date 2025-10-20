# Graph Package Documentation

## 📋 패키지 개요

Graph 패키지는 Parser에서 추출된 코드 블록을 그래프 구조로 변환하는 모듈입니다. 의존성 관계를 명확히 하고 Neo4j와 같은 그래프 데이터베이스에 저장할 수 있는 형태로 변환합니다.

## 🏗️ 아키텍처

```
Graph Package
├── src/
│   ├── models.py       # 그래프 데이터 모델
│   ├── adapter.py      # 데이터 변환 어댑터
│   └── utils.py        # 유틸리티 함수
└── __init__.py         # 패키지 초기화
```

## 🔧 주요 컴포넌트

### 데이터 모델 (models.py)

#### CodeNode
그래프의 노드를 표현하는 모델입니다.

```python
@dataclass
class CodeNode:
    id: str                    # 고유 식별자
    name: str                  # 노드 이름
    type: str                  # 노드 타입 (class, function, etc.)
    source_code: str | None    # 소스 코드
    docstring: str | None      # 문서화 문자열
    complexity: int            # 복잡도 점수
    start_line: int           # 시작 라인 번호
    end_line: int             # 종료 라인 번호
    scope_level: int          # 스코프 깊이
    file_path: str | None     # 파일 경로
    
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CodeNode': ...
```

#### CodeRelation
노드 간의 관계를 표현하는 모델입니다.

```python
@dataclass
class CodeRelation:
    id: str                    # 관계 고유 식별자
    source_id: str            # 시작 노드 ID
    target_id: str            # 대상 노드 ID
    relation_type: str        # 관계 타입
    weight: float             # 관계 가중치
    properties: dict[str, Any] # 추가 속성
    
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CodeRelation': ...
```

#### CodeGraph
전체 그래프를 표현하는 컨테이너 모델입니다.

```python
@dataclass
class CodeGraph:
    nodes: list[CodeNode]           # 노드 리스트
    relations: list[CodeRelation]   # 관계 리스트
    metadata: dict[str, Any]        # 메타데이터
    
    def add_node(self, node: CodeNode) -> None: ...
    def add_relation(self, relation: CodeRelation) -> None: ...
    def get_node_by_id(self, node_id: str) -> CodeNode | None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CodeGraph': ...
```

### 데이터 어댑터 (adapter.py)

#### GraphAdapter
CodeBlock을 CodeGraph로 변환하는 핵심 클래스입니다.

```python
class GraphAdapter:
    def __init__(self):
        self.node_id_counter = 0
        self.relation_id_counter = 0
    
    def convert_to_graph(self, blocks: list[CodeBlock] | list[dict]) -> CodeGraph:
        """CodeBlock 리스트를 CodeGraph로 변환"""
        
    def _convert_from_code_blocks(self, blocks: list[CodeBlock]) -> CodeGraph:
        """CodeBlock 객체들을 변환"""
        
    def _convert_from_dicts(self, blocks: list[dict]) -> CodeGraph:
        """딕셔너리 형태의 블록들을 변환"""
```

### 관계 타입 정의

```python
class RelationType:
    CONTAINS = "CONTAINS"           # 포함 관계 (클래스 → 메서드)
    CALLS = "CALLS"                 # 호출 관계 (함수 → 함수)
    IMPORTS = "IMPORTS"             # 임포트 관계 (모듈 → 모듈)
    INHERITS = "INHERITS"           # 상속 관계 (클래스 → 클래스)
    DEPENDS_ON = "DEPENDS_ON"       # 의존 관계 (일반적인 의존성)
    DEFINES = "DEFINES"             # 정의 관계 (변수, 상수 등)
```

## 📊 변환 프로세스

### 1. 입력: CodeBlock 리스트
```python
# Parser에서 생성된 블록들
blocks = [
    CodeBlock(
        block_type="class",
        name="UserManager", 
        start_line=4,
        end_line=16,
        # ... 기타 속성들
    ),
    # ... 더 많은 블록들
]
```

### 2. 변환 과정
```python
from graph.src.adapter import GraphAdapter

adapter = GraphAdapter()
graph = adapter.convert_to_graph(blocks)

print(f"노드 수: {len(graph.nodes)}")
print(f"관계 수: {len(graph.relations)}")
```

### 3. 결과: CodeGraph
```
노드 생성:
- node_1: module "api" (type: module)
- node_2: class "UserManager" (type: class)  
- node_3: function "__init__" (type: function)
- node_4: function "add_user" (type: function)

관계 생성:
- relation_1: node_1 CONTAINS node_2
- relation_2: node_2 CONTAINS node_3
- relation_3: node_2 CONTAINS node_4
```

## 🔍 사용 예시

### 기본 변환
```python
from graph.src.adapter import GraphAdapter

# 어댑터 초기화
adapter = GraphAdapter()

# CodeBlock을 CodeGraph로 변환
graph = adapter.convert_to_graph(code_blocks)

# 결과 확인
print(f"총 {len(graph.nodes)}개 노드 생성")
print(f"총 {len(graph.relations)}개 관계 생성")

# 노드 정보 출력
for node in graph.nodes:
    print(f"- {node.name} ({node.type})")
```

### 특정 노드 검색
```python
# 특정 이름의 노드 찾기
user_manager = None
for node in graph.nodes:
    if node.name == "UserManager":
        user_manager = node
        break

if user_manager:
    print(f"발견: {user_manager.name} ({user_manager.type})")
    print(f"복잡도: {user_manager.complexity}")
    print(f"위치: {user_manager.start_line}-{user_manager.end_line}")
```

### 관계 분석
```python
# 특정 노드의 관계 찾기
node_id = "node_2"
incoming_relations = [r for r in graph.relations if r.target_id == node_id]
outgoing_relations = [r for r in graph.relations if r.source_id == node_id]

print(f"들어오는 관계: {len(incoming_relations)}개")
print(f"나가는 관계: {len(outgoing_relations)}개")

# 관계 타입별 분류
from collections import Counter
relation_types = [r.relation_type for r in graph.relations]
type_counts = Counter(relation_types)

for rel_type, count in type_counts.items():
    print(f"{rel_type}: {count}개")
```

## ⚙️ 설정 및 커스터마이징

### 노드 ID 생성 규칙
```python
# 기본 ID 생성 패턴
node_id = f"node_{self.node_id_counter}"
relation_id = f"relation_{self.relation_id_counter}"

# 커스텀 ID 생성기
class CustomGraphAdapter(GraphAdapter):
    def _generate_node_id(self, block: CodeBlock) -> str:
        return f"{block.block_type}_{block.name}_{block.start_line}"
```

### 관계 가중치 계산
```python
def _calculate_weight(self, source: CodeBlock, target: CodeBlock) -> float:
    """관계 가중치 계산 로직"""
    base_weight = 1.0
    
    # 복잡도 기반 가중치 조정
    complexity_factor = (source.complexity + target.complexity) / 20.0
    
    # 스코프 레벨 기반 조정
    scope_factor = abs(source.scope_level - target.scope_level) * 0.1
    
    return base_weight + complexity_factor - scope_factor
```

## 🧪 테스트 및 검증

### 변환 정확성 테스트
```python
def test_conversion_accuracy():
    """변환 정확성 검증"""
    blocks = sample_code_blocks()
    adapter = GraphAdapter()
    graph = adapter.convert_to_graph(blocks)
    
    # 노드 수 검증
    expected_nodes = count_expected_nodes(blocks)
    assert len(graph.nodes) == expected_nodes
    
    # 관계 수 검증
    expected_relations = count_expected_relations(blocks)
    assert len(graph.relations) == expected_relations
    
    print("✅ 변환 정확성 테스트 통과")
```

### 성능 테스트
```python
import time

def test_performance():
    """대용량 데이터 처리 성능 테스트"""
    large_blocks = generate_large_dataset(1000)  # 1000개 블록
    
    adapter = GraphAdapter()
    start_time = time.time()
    
    graph = adapter.convert_to_graph(large_blocks)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"📊 성능 테스트 결과:")
    print(f"   • 처리 시간: {processing_time:.2f}초")
    print(f"   • 초당 블록 처리: {len(large_blocks)/processing_time:.1f}개")
    print(f"   • 생성된 노드: {len(graph.nodes)}개")
    print(f"   • 생성된 관계: {len(graph.relations)}개")
```

## 🚨 제한사항 및 해결책

### ✅ 해결된 이슈 (v0.2.1)
- **file_path 정보 처리**: Parser 패키지 개선으로 완전히 해결
- **타입 힌트 완성**: 모든 함수와 메서드에 정확한 타입 힌트 적용
- **Pydantic v2 호환성**: 최신 Pydantic 사용 및 ConfigDict 적용
- **pytest 테스트 도입**: 31개 테스트 케이스, 83% 커버리지 달성

### 현재 제한사항

1. **복잡한 관계 추론 제한**
```python
# 현재: 단순한 포함/의존 관계만 처리
# 향후: 더 정교한 관계 분석 필요
```

2. **메모리 사용량**
```python
# 대용량 프로젝트 시 메모리 사용량 증가
# 해결책: 스트리밍 방식 도입 예정
```

3. **utils.py 커버리지**
```python
# 현재: utils.py 모듈 18% 커버리지
# 개선 필요: 유틸리티 함수들의 테스트 추가
```

### 해결 방향

#### 점진적 관계 구축
```python
class IncrementalGraphBuilder:
    def __init__(self):
        self.partial_graph = CodeGraph([], [], {})
    
    def add_blocks_batch(self, blocks: list[CodeBlock]) -> None:
        """배치 단위로 블록 추가"""
        for block in blocks:
            self._add_single_block(block)
    
    def finalize_relations(self) -> None:
        """모든 블록 추가 후 관계 최종 구축"""
        self._build_cross_references()
```

## 🔮 로드맵

### ✅ 완료된 목표 (v0.2.1)
- [x] file_path 속성 완전 지원
- [x] Pydantic v2 업그레이드 및 호환성 확보
- [x] pytest 테스트 도입 (31개 테스트, 83% 커버리지)
- [x] 타입 힌트 완성 및 코드 품질 개선
- [x] Parser-Graph 통합 테스트 구현

### v0.2.2 - 테스트 및 품질 개선
- [ ] utils.py 테스트 커버리지 90% 이상 달성
- [ ] 성능 벤치마크 테스트 추가
- [ ] 문서화 자동 생성 도구 연동

### v0.3.0 - 향상된 관계 분석
- [ ] 함수 호출 관계 추출 정확도 향상
- [ ] 클래스 상속 관계 감지 고도화
- [ ] 크로스 파일 의존성 분석

### v0.4.0 - 성능 최적화
- [ ] 스트리밍 변환 지원
- [ ] 병렬 처리 구현
- [ ] 메모리 사용량 최적화

### v1.0.0 - 고급 그래프 기능
- [ ] 그래프 질의 언어 지원
- [ ] 시각화 도구 통합
- [ ] 실시간 업데이트 지원

## 📚 API 참조

### GraphAdapter
```python
class GraphAdapter:
    def __init__(self) -> None: ...
    def convert_to_graph(self, blocks: list[CodeBlock] | list[dict]) -> CodeGraph: ...
    def _create_node_from_code_block(self, block: CodeBlock) -> CodeNode: ...
    def _build_relations(self, blocks: list[CodeBlock], nodes: list[CodeNode]) -> list[CodeRelation]: ...
```

### CodeGraph
```python
class CodeGraph:
    def add_node(self, node: CodeNode) -> None: ...
    def add_relation(self, relation: CodeRelation) -> None: ...
    def get_node_by_id(self, node_id: str) -> CodeNode | None: ...
    def get_nodes_by_type(self, node_type: str) -> list[CodeNode]: ...
    def get_relations_by_type(self, relation_type: str) -> list[CodeRelation]: ...
```

## 💡 베스트 프랙티스

### 변환 프로세스
1. **입력 검증**: CodeBlock 리스트의 무결성 확인
2. **점진적 구축**: 대용량 데이터는 배치 단위로 처리
3. **관계 검증**: 생성된 관계의 논리적 타당성 확인

### 성능 고려사항
1. **메모리 관리**: 불필요한 객체 참조 제거
2. **배치 처리**: 대량 데이터는 청크 단위로 분할
3. **인덱싱**: 자주 조회되는 노드는 인덱스 구축

### 확장성 고려
1. **플러그인 패턴**: 새로운 관계 타입 쉽게 추가
2. **인터페이스 분리**: 어댑터 로직을 인터페이스로 추상화
3. **설정 가능**: 변환 규칙을 외부에서 설정 가능하게

---

## 📈 최근 개선사항 (v0.2.1)

### 🔧 코드 리팩토링
- **file_path 완전 지원**: Parser 패키지 개선으로 모든 노드에 정확한 파일 경로 포함
- **Pydantic v2 업그레이드**: 최신 Pydantic 사용 및 ConfigDict 적용으로 성능 향상
- **타입 힌트 완성**: 모든 함수와 메서드에 정확한 타입 힌트 적용
- **Enum 호환성**: 문자열과 Enum 타입 모두 지원하는 유연한 비교 로직

### 🧪 테스트 구축
- **pytest 도입**: 기존 테스트 없던 상태에서 체계적인 pytest 테스트로 전환
- **31개 테스트 케이스**: 모델, 어댑터, 통합 테스트 포함
- **83% 커버리지**: adapter.py와 models.py에서 높은 커버리지 달성
- **통합 테스트**: Parser와 Graph 패키지 간 실제 연동 테스트

### 📊 성능 지표
- **변환 속도**: 대용량 프로젝트도 빠른 변환 (최적화 예정)
- **메모리 효율**: Pydantic v2로 메모리 사용량 개선
- **Neo4j 호환성**: 완전한 Neo4j 형식 변환 지원
- **테스트 실행 시간**: 0.55초 (31개 테스트)

---

**패키지 버전**: v0.2.1  
**마지막 업데이트**: 2025-10-21  
**이전 패키지**: [Parser Package](parser.md) | **다음 패키지**: [Core Package](core.md)