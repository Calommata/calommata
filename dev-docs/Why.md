## 서버
- 데이터베이스 보안을 위해 CLI 가 아닌 서버 형태로 제작
- DockerFile & Docker Compose 가 준비되어서 원활한 배포가 가능해야 함
- 서버는 CLI 가 보내주는 파일 정보를 읽고 그 파일과 작업 상태에 따른 작업을 해야 함
- LangChain/LangGraph + FastAPI 로 제작
  - Qdrant + Neo4j 관련 생태계가 안정적임 (NestJS 는 직접 구현해야할 부분이 많음)
  - Spring AI + Spring 도 안정적이지만, 이왕 옮긴다면 LangChain/LangGraph + FastAPI 가 생태계 + 생산성에서 낫다고 판단
  - 추후 오픈소스화 되었을 때 사용자가 쉽게 사용할지를 생각했을 때도 FastAPI + LangChain/LangGraph 가 낫다고 판단

## CLI
- Python 으로 작성되며, 프로젝트의 Git 명령어로 branch, diff 를 얻음
- Rust 도 생각하였으나, 서버와 CLI 의 언어를 통일하여 생산성과 기타 이점을 얻는 것이 최선이라 판단

## 데이터베이스
- Neo4j + Qdrant
- 코드들의 연관 관계는 복잡한 경우가 많기 때문에, Neo4j 에 저장하는 것이 유리
- 비정형 임베딩 데이터는 Qdrant 에 저장하고, 변경된 파일을 git 으로 인식해 Neo4j 에서 탐색 후 변경 전 코드의 데이터와 연관 코드를 Context 에 투입하여 AI 정확도 개선 