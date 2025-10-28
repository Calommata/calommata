"""코드 분석을 위한 그래프 데이터 구조

Neo4j 통합을 위한 노드와 관계 모델 정의
Pydantic v2 기반 데이터 검증 및 직렬화
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """노드 타입 정의"""

    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    VARIABLE = "Variable"
    IMPORT = "Import"
    PROPERTY = "Property"
    DECORATOR = "Decorator"


class RelationType(Enum):
    """관계 타입 정의"""

    CALLS = "CALLS"
    INHERITS = "INHERITS"
    IMPORTS = "IMPORTS"
    CONTAINS = "CONTAINS"
    DEFINES = "DEFINES"
    REFERENCES = "REFERENCES"
    DEPENDS_ON = "DEPENDS_ON"
    DECORATES = "DECORATES"
    IMPLEMENTS = "IMPLEMENTS"
    RAISES = "RAISES"
    RETURNS = "RETURNS"


class Dependency(BaseModel):
    """의존성 정보 모델"""

    target: str = Field(..., description="의존 대상")
    dependency_type: str = Field(
        ..., description="의존성 타입 (import, call, inherit 등)"
    )
    context: str | None = Field(default=None, description="의존성 컨텍스트")

    def __str__(self) -> str:
        return f"{self.dependency_type}: {self.target}"


class CodeNode(BaseModel):
    """코드 노드 모델 - Neo4j 노드로 변환될 기본 단위"""

    # 기본 식별자
    id: str = Field(..., description="고유 식별자")
    name: str = Field(..., description="노드 이름")
    node_type: NodeType = Field(..., description="노드 타입")

    # 위치 정보
    file_path: str = Field(..., description="파일 경로")

    # 코드 내용
    source_code: str = Field(default="", description="소스 코드")

    # 메타데이터
    complexity: int = Field(default=0, description="복잡도 점수")
    scope_level: int = Field(default=0, description="스코프 깊이")
    parameters: list[str] = Field(
        default_factory=list, description="함수/메서드 매개변수"
    )
    return_type: str | None = Field(default=None, description="반환 타입")
    decorators: list[str] = Field(default_factory=list, description="데코레이터 목록")

    # 의존성 정보
    dependencies: list[Dependency] = Field(
        default_factory=list, description="의존성 목록"
    )
    imports: list[str] = Field(default_factory=list, description="import 목록")

    # 임베딩 관련
    embedding_vector: list[float] | None = Field(
        default=None, description="코드 임베딩 벡터"
    )
    embedding_model: str | None = Field(default=None, description="사용된 임베딩 모델")

    # 타임스탬프
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.now, description="수정 시간")

    model_config = ConfigDict(use_enum_values=True)

    def get_full_name(self) -> str:
        """노드의 전체 이름 반환 (file_path:type:name)

        Returns:
            전체 경로를 포함한 노드 이름

        Example:
            /path/to/file.py:Function:my_func
        """
        # node_type이 이미 문자열이면 그대로, enum이면 .value 사용
        node_type_str = (
            self.node_type.value
            if isinstance(self.node_type, NodeType)
            else self.node_type
        )
        return f"{self.file_path}:{node_type_str}:{self.name}"

    def add_dependency(
        self,
        target: str,
        dep_type: str,
        context: str | None = None,
    ) -> None:
        """의존성 추가

        Args:
            target: 의존 대상 이름
            dep_type: 의존성 타입
            context: 의존성 컨텍스트 (선택사항)
        """
        dependency = Dependency(
            target=target,
            dependency_type=dep_type,
            context=context,
        )
        self.dependencies.append(dependency)
        self.updated_at = datetime.now()

    def get_dependencies_by_type(self, dep_type: str) -> list[Dependency]:
        """타입별 의존성 조회"""
        return [dep for dep in self.dependencies if dep.dependency_type == dep_type]

    def to_neo4j_node(self) -> dict[str, Any]:
        """Neo4j 노드 생성을 위한 딕셔너리 변환"""
        # node_type이 이미 문자열이면 그대로, enum이면 .value 사용
        node_type_str = (
            self.node_type.value
            if isinstance(self.node_type, NodeType)
            else self.node_type
        )
        return {
            "id": self.id,
            "name": self.name,
            "type": node_type_str,
            "file_path": self.file_path,
            "source_code": self.source_code,
            "complexity": self.complexity,
            "scope_level": self.scope_level,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "decorators": self.decorators,
            "imports": self.imports,
            "embedding": self.embedding_vector,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class CodeRelation(BaseModel):
    """코드 관계 모델 - Neo4j 관계로 변환될 기본 단위"""

    # 관계 식별자
    from_node_id: str = Field(..., description="시작 노드 ID")
    to_node_id: str = Field(..., description="끝 노드 ID")
    relation_type: RelationType = Field(..., description="관계 타입")

    # 관계 메타데이터
    weight: float = Field(default=1.0, description="관계 가중치")
    context: str | None = Field(default=None, description="관계 컨텍스트")

    # 타임스탬프
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")

    model_config = ConfigDict(use_enum_values=True)

    def to_neo4j_relation(self) -> dict[str, Any]:
        """Neo4j 관계 생성을 위한 딕셔너리 변환"""
        # relation_type이 이미 문자열이면 그대로, enum이면 .value 사용
        relation_type_str = (
            self.relation_type.value
            if isinstance(self.relation_type, RelationType)
            else self.relation_type
        )
        return {
            "type": relation_type_str,
            "weight": self.weight,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
        }


class CodeGraph(BaseModel):
    """코드 그래프 모델 - 전체 프로젝트의 그래프 구조"""

    # 기본 정보
    project_name: str = Field(..., description="프로젝트 이름")
    project_path: str = Field(..., description="프로젝트 경로")

    # 그래프 데이터
    nodes: dict[str, CodeNode] = Field(default_factory=dict, description="노드들")
    relations: list[CodeRelation] = Field(default_factory=list, description="관계들")

    # 메타데이터
    total_files: int = Field(default=0, description="총 파일 수")
    total_lines: int = Field(default=0, description="총 라인 수")
    analysis_version: str = Field(default="1.0.0", description="분석 버전")

    # 타임스탬프
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.now, description="수정 시간")

    @property
    def total_nodes(self) -> int:
        """총 노드 수"""
        return len(self.nodes)

    def add_node(self, node: CodeNode) -> None:
        """노드 추가"""
        self.nodes[node.id] = node
        self.updated_at = datetime.now()

    def add_relation(self, relation: CodeRelation) -> None:
        """관계 추가"""
        # 노드 존재 확인
        if relation.from_node_id not in self.nodes:
            raise ValueError(f"From node {relation.from_node_id} not found")
        if relation.to_node_id not in self.nodes:
            raise ValueError(f"To node {relation.to_node_id} not found")

        self.relations.append(relation)
        self.updated_at = datetime.now()

    def get_node_by_id(self, node_id: str) -> CodeNode | None:
        """ID로 노드 조회"""
        return self.nodes.get(node_id)

    def get_relations_from_node(self, node_id: str) -> list[CodeRelation]:
        """특정 노드에서 시작하는 관계들 조회"""
        return [r for r in self.relations if r.from_node_id == node_id]

    def get_relations_to_node(self, node_id: str) -> list[CodeRelation]:
        """특정 노드로 들어오는 관계들 조회"""
        return [r for r in self.relations if r.to_node_id == node_id]

    def get_nodes_by_type(self, node_type: NodeType) -> list[CodeNode]:
        """타입별 노드 조회"""
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def get_relations_by_type(self, relation_type: RelationType) -> list[CodeRelation]:
        """타입별 관계 조회"""
        return [r for r in self.relations if r.relation_type == relation_type]

    def get_statistics(self) -> dict[str, Any]:
        """그래프 통계 정보"""
        node_type_counts = {}
        for node_type in NodeType:
            node_type_counts[node_type.value] = len(self.get_nodes_by_type(node_type))

        relation_type_counts = {}
        for relation_type in RelationType:
            relation_type_counts[relation_type.value] = len(
                self.get_relations_by_type(relation_type)
            )

        return {
            "total_nodes": len(self.nodes),
            "total_relations": len(self.relations),
            "node_types": node_type_counts,
            "relation_types": relation_type_counts,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "analysis_version": self.analysis_version,
        }

    def to_neo4j_format(self) -> dict[str, Any]:
        """Neo4j 가져오기를 위한 형식으로 변환"""
        return {
            "project": {
                "name": self.project_name,
                "path": self.project_path,
                "total_files": self.total_files,
                "total_lines": self.total_lines,
                "analysis_version": self.analysis_version,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            },
            "nodes": [node.to_neo4j_node() for node in self.nodes.values()],
            "relations": [
                {
                    "from": rel.from_node_id,
                    "to": rel.to_node_id,
                    **rel.to_neo4j_relation(),
                }
                for rel in self.relations
            ],
            "statistics": self.get_statistics(),
        }
