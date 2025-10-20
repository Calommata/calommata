"""Neo4j Cypher 쿼리 정의

모든 데이터베이스 쿼리를 중앙화하여 관리합니다.
"""


class Neo4jQueries:
    """Neo4j Cypher 쿼리 모음"""

    # ========== 제약 조건 및 인덱스 ==========

    CONSTRAINTS = [
        "CREATE CONSTRAINT code_node_id IF NOT EXISTS FOR (n:CodeNode) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE",
    ]

    INDEXES = [
        "CREATE INDEX code_node_type IF NOT EXISTS FOR (n:CodeNode) ON (n.type)",
        "CREATE INDEX code_node_file IF NOT EXISTS FOR (n:CodeNode) ON (n.file_path)",
        "CREATE TEXT INDEX code_node_content IF NOT EXISTS FOR (n:CodeNode) ON (n.source_code)",
    ]

    # HuggingFace 모델용 벡터 인덱스 (384차원 - sentence-transformers/all-MiniLM-L6-v2)
    VECTOR_INDEX = """
        CREATE VECTOR INDEX code_embedding_index IF NOT EXISTS 
        FOR (n:CodeNode) ON (n.embedding) 
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }
        }
    """

    # 다른 모델용 벡터 인덱스 (768차원)
    VECTOR_INDEX_LARGE = """
        CREATE VECTOR INDEX code_embedding_index_large IF NOT EXISTS 
        FOR (n:CodeNode) ON (n.embedding) 
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }
        }
    """

    # ========== 프로젝트 관리 ==========

    MERGE_PROJECT = """
        MERGE (p:Project {name: $name})
        SET p.path = $path,
            p.total_files = $total_files,
            p.total_lines = $total_lines,
            p.total_nodes = $total_nodes,
            p.total_relations = $total_relations,
            p.analysis_version = $analysis_version,
            p.created_at = $created_at,
            p.updated_at = datetime()
        RETURN p
    """

    DELETE_PROJECT = """
        MATCH (p:Project {name: $project_name})
        OPTIONAL MATCH (p)-[:CONTAINS]->(n:CodeNode)
        DETACH DELETE p, n
    """

    # ========== 노드 관리 ==========

    MERGE_CODE_NODES_BATCH = """
        UNWIND $nodes AS node_data
        MERGE (n:CodeNode {id: node_data.id})
        SET n.name = node_data.name,
            n.type = node_data.type,
            n.file_path = node_data.file_path,
            n.start_line = node_data.start_line,
            n.end_line = node_data.end_line,
            n.source_code = node_data.source_code,
            n.docstring = node_data.docstring,
            n.complexity = node_data.complexity,
            n.scope_level = node_data.scope_level,
            n.embedding = node_data.embedding,
            n.embedding_model = node_data.embedding_model,
            n.project_name = $project_name,
            n.created_at = node_data.created_at,
            n.updated_at = datetime()
        
        WITH n, node_data
        MATCH (p:Project {name: $project_name})
        MERGE (p)-[:CONTAINS]->(n)
    """

    UPDATE_NODE_EMBEDDING = """
        MATCH (n:CodeNode {id: $node_id})
        SET n.embedding = $embedding,
            n.embedding_model = $model,
            n.updated_at = datetime()
        RETURN n
    """

    # ========== 관계 관리 ==========

    # 동적 관계 타입을 위한 템플릿
    CREATE_RELATION_TEMPLATE = """
        MATCH (from:CodeNode {{id: $from_id}})
        MATCH (to:CodeNode {{id: $to_id}})
        MERGE (from)-[r:{relation_type} {{
            weight: $weight,
            line_number: $line_number,
            context: $context,
            created_at: $created_at
        }}]->(to)
        RETURN r
    """

    # ========== 검색 및 조회 ==========

    VECTOR_SEARCH = """
        CALL db.index.vector.queryNodes('code_embedding_index', $limit, $query_embedding)
        YIELD node, score
        WHERE score >= $similarity_threshold 
          AND (node.project_name = $project_name OR $project_name IS NULL)
        RETURN node.id AS id,
               node.name AS name,
               node.type AS type,
               node.file_path AS file_path,
               node.source_code AS source_code,
               node.docstring AS docstring,
               score
        ORDER BY score DESC
    """

    GET_NODE_CONTEXT = """
        MATCH (center:CodeNode {id: $node_id})
        OPTIONAL MATCH path = (center)-[*1..2]-(related:CodeNode)
        WITH center, collect(DISTINCT related) AS related_nodes,
             collect(DISTINCT relationships(path)) AS all_relationships
        
        RETURN center,
               related_nodes,
               [r IN all_relationships WHERE r IS NOT NULL | {
                   type: type(r),
                   start_node: startNode(r).id,
                   end_node: endNode(r).id,
                   properties: properties(r)
               }] AS relationships
    """

    # ========== 통계 ==========

    GET_NODE_STATS = """
        MATCH (n:CodeNode)
        RETURN n.type AS type, count(n) AS count
    """

    GET_RELATION_STATS = """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
    """

    GET_TOTAL_STATS = """
        MATCH (n)
        OPTIONAL MATCH ()-[r]->()
        RETURN count(DISTINCT n) AS total_nodes,
               count(r) AS total_relationships
    """

    # ========== 유틸리티 ==========

    TEST_CONNECTION = "RETURN 1"

    @classmethod
    def create_relation_query(cls, relation_type: str) -> str:
        """동적 관계 타입을 위한 쿼리 생성

        Args:
            relation_type: 관계 타입 이름

        Returns:
            str: 포맷된 Cypher 쿼리
        """
        return cls.CREATE_RELATION_TEMPLATE.format(relation_type=relation_type)
