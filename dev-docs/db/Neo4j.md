1. 노드 제약 정의
```
CREATE CONSTRAINT code_file_id IF NOT EXISTS FOR (cf:CodeFile) REQUIRE cf.id IS UNIQUE;
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (doc:Document) REQUIRE doc.id IS UNIQUE;
CREATE CONSTRAINT code_id IF NOT EXISTS FOR (cd:Code) REQUIRE cd.id IS UNIQUE;
CREATE CONSTRAINT code_file_path IF NOT EXISTS FOR (cf:CodeFile) REQUIRE cf.path IS UNIQUE;
CREATE CONSTRAINT document_path IF NOT EXISTS FOR (doc:Document) REQUIRE doc.path IS UNIQUE;
```

2. 코드 파일 노드
```
CREATE (cf:CodeFile {
  id: "cf_001",
  path: "/src/components/Header.tsx",
  git_hash: "abc123"
})
```

3. 코드 데이터 노드
```
CREATE (cd:Code {
  id: "cd_001",
  chunk_index: 0,
  type: "function",
  name: "Header",
  qdrant_id: "code_001"
})
```

4. 문서 노드
```
CREATE (doc:Document {
  id: "doc_001", 
  path: "/docs/api/authentication.md",
  qdrant_id: "doc_001"
})
```

5. 관계 정의
- 코드 간 의존성  
  - `CREATE (cd1:Code)-[:USES]->(cd2:Code)`

- 문서 간 연관성  
  - `CREATE (doc1:Document)-[:RELATED_TO]->(doc2:Document)`

- 코드-문서 간 연관성
  - `CREATE (cd:Code)-[:REFLECT_BY]->(doc:Document)`

- 코드-코드 파일 간 연관성
  - `CREATE (cf:CodeFile)-[:INCLUDES]->(cd:Code)`