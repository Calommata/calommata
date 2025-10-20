"""
pytest 설정 파일
"""

import sys
from pathlib import Path

# 패키지 루트 디렉토리를 sys.path에 추가
parser_root = Path(__file__).parent.parent
if str(parser_root) not in sys.path:
    sys.path.insert(0, str(parser_root))
