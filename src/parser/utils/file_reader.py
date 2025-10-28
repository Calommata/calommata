import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileReader:
    @staticmethod
    def read_file(file_path: str) -> str:
        """파일 내용 읽기

        Args:
            file_path: 읽을 파일 경로

        Returns:
            파일의 내용

        Raises:
            FileNotFoundError: 파일이 없는 경우
            IOError: 파일 읽기 실패 시
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Read file: {file_path} ({len(content)} bytes)")
            return content
        except IOError as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise

    @staticmethod
    def find_python_files(dir_path: str) -> list[Path]:
        """디렉토리에서 Python 파일들 찾기

        재귀적으로 디렉토리를 탐색하여 모든 .py 파일을 찾습니다.

        Args:
            dir_path: 검색할 디렉토리 경로

        Returns:
            발견된 Python 파일들의 Path 객체 리스트
        """
        path = Path(dir_path)
        if not path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        if not path.is_dir():
            logger.warning(f"Path is not a directory: {dir_path}")
            return []

        python_files = list(path.glob("**/*.py"))
        logger.debug(f"Found {len(python_files)} Python files in {dir_path}")
        return python_files
