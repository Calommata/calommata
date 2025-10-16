"""API 모듈 - 웹 API 엔드포인트"""

import json
from user import AuthenticationService


class APIHandler:
    """API 요청을 처리하는 클래스"""

    def __init__(self, auth_service: AuthenticationService):
        self.auth_service = auth_service

    def handle_request(self, request: dict) -> dict:
        """API 요청 처리"""
        method = request.get("method")
        if method == "login":
            return self._handle_login(request)
        elif method == "get_user":
            return self._handle_get_user(request)
        return {"error": "Unknown method"}

    def _handle_login(self, request: dict) -> dict:
        """로그인 요청 처리"""
        username = request.get("username")
        password = request.get("password")

        if self.auth_service.authenticate(username, password):
            return {"status": "success", "message": "Login successful"}
        return {"status": "fail", "message": "Invalid credentials"}

    def _handle_get_user(self, request: dict) -> dict:
        """사용자 정보 조회 요청 처리"""
        user_id = request.get("user_id")
        user = self.auth_service.user_manager.get_user(user_id)
        return {"status": "success", "data": user}


class ResponseFormatter:
    """API 응답을 포맷팅하는 클래스"""

    @staticmethod
    def format_json(data: dict) -> str:
        """데이터를 JSON 문자열로 변환"""
        return json.dumps(data, indent=2)

    @staticmethod
    def format_error(error_message: str, error_code: int = 500) -> dict:
        """에러 응답 생성"""
        return {"error": error_message, "code": error_code}
