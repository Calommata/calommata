"""복잡한 코드 구조 E2E 테스트

상속, 데코레이터, 예외 처리, 다중 모듈 등이 포함된
복잡한 파이썬 프로젝트를 분석하고 GraphRAG 검색 테스트
"""

import logging
import os
import tempfile
from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv

from src.core import CoreConfig, create_from_config

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_complex_project(base_path: Path) -> None:
    """복잡한 프로젝트 구조 생성"""

    # models.py - 기본 모델들
    models_py = base_path / "models.py"
    models_py.write_text(
        dedent("""
        '''데이터 모델 정의'''
        from abc import ABC, abstractmethod
        from dataclasses import dataclass
        from typing import Optional, List, Dict, Any
        from enum import Enum
        
        
        class UserRole(Enum):
            '''사용자 역할 정의'''
            ADMIN = "admin"
            USER = "user"
            GUEST = "guest"
        
        
        @dataclass
        class User:
            '''사용자 모델'''
            id: int
            name: str
            email: str
            role: UserRole = UserRole.USER
            
            def is_admin(self) -> bool:
                '''관리자 권한 확인'''
                return self.role == UserRole.ADMIN
            
            def get_display_name(self) -> str:
                '''표시용 이름 반환'''
                return f"{self.name} ({self.role.value})"
        
        
        class DatabaseEntity(ABC):
            '''데이터베이스 엔티티 추상 클래스'''
            
            def __init__(self, id: Optional[int] = None):
                self.id = id
                self.created_at = None
                self.updated_at = None
            
            @abstractmethod
            def validate(self) -> bool:
                '''데이터 검증'''
                pass
            
            @abstractmethod
            def to_dict(self) -> Dict[str, Any]:
                '''딕셔너리 변환'''
                pass
        
        
        class Product(DatabaseEntity):
            '''상품 모델'''
            
            def __init__(self, name: str, price: float, category: str, **kwargs):
                super().__init__(**kwargs)
                self.name = name
                self.price = price
                self.category = category
                self.stock = 0
            
            def validate(self) -> bool:
                '''상품 데이터 검증'''
                return (
                    self.name and len(self.name.strip()) > 0
                    and self.price > 0
                    and self.category
                )
            
            def to_dict(self) -> Dict[str, Any]:
                '''딕셔너리 변환'''
                return {
                    'id': self.id,
                    'name': self.name,
                    'price': self.price,
                    'category': self.category,
                    'stock': self.stock
                }
            
            def update_stock(self, quantity: int) -> None:
                '''재고 업데이트'''
                if self.stock + quantity < 0:
                    raise ValueError("재고가 부족합니다")
                self.stock += quantity
    """),
        encoding="utf-8",
    )

    # services.py - 비즈니스 로직
    services_py = base_path / "services.py"
    services_py.write_text(
        dedent("""
        '''비즈니스 로직 서비스'''
        from typing import List, Optional, Dict, Any
        from functools import wraps
        import logging
        
        from models import User, Product, UserRole
        from exceptions import (
            UserNotFoundError, 
            ProductNotFoundError, 
            InsufficientPermissionError,
            ValidationError
        )
        
        logger = logging.getLogger(__name__)
        
        
        def require_admin(func):
            '''관리자 권한 필요 데코레이터'''
            @wraps(func)
            def wrapper(self, user: User, *args, **kwargs):
                if not user.is_admin():
                    raise InsufficientPermissionError("관리자 권한이 필요합니다")
                return func(self, user, *args, **kwargs)
            return wrapper
        
        
        def log_operation(operation_name: str):
            '''작업 로깅 데코레이터'''
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    logger.info(f"작업 시작: {operation_name}")
                    try:
                        result = func(*args, **kwargs)
                        logger.info(f"작업 완료: {operation_name}")
                        return result
                    except Exception as e:
                        logger.error(f"작업 실패: {operation_name} - {e}")
                        raise
                return wrapper
            return decorator
        
        
        class UserService:
            '''사용자 관리 서비스'''
            
            def __init__(self):
                self._users: Dict[int, User] = {}
                self._next_id = 1
            
            @log_operation("사용자 생성")
            def create_user(self, name: str, email: str, role: UserRole = UserRole.USER) -> User:
                '''사용자 생성'''
                if not name or not email:
                    raise ValidationError("이름과 이메일은 필수입니다")
                
                user = User(
                    id=self._next_id,
                    name=name,
                    email=email,
                    role=role
                )
                self._users[user.id] = user
                self._next_id += 1
                return user
            
            def get_user(self, user_id: int) -> User:
                '''사용자 조회'''
                if user_id not in self._users:
                    raise UserNotFoundError(f"사용자를 찾을 수 없습니다: {user_id}")
                return self._users[user_id]
            
            def list_users(self) -> List[User]:
                '''모든 사용자 목록'''
                return list(self._users.values())
            
            @require_admin
            def delete_user(self, admin: User, user_id: int) -> bool:
                '''사용자 삭제 (관리자 전용)'''
                if user_id not in self._users:
                    raise UserNotFoundError(f"사용자를 찾을 수 없습니다: {user_id}")
                del self._users[user_id]
                return True
        
        
        class ProductService:
            '''상품 관리 서비스'''
            
            def __init__(self):
                self._products: Dict[int, Product] = {}
                self._next_id = 1
            
            @log_operation("상품 생성")
            def create_product(self, name: str, price: float, category: str) -> Product:
                '''상품 생성'''
                product = Product(
                    id=self._next_id,
                    name=name,
                    price=price,
                    category=category
                )
                
                if not product.validate():
                    raise ValidationError("상품 데이터가 유효하지 않습니다")
                
                self._products[product.id] = product
                self._next_id += 1
                return product
            
            def get_product(self, product_id: int) -> Product:
                '''상품 조회'''
                if product_id not in self._products:
                    raise ProductNotFoundError(f"상품을 찾을 수 없습니다: {product_id}")
                return self._products[product_id]
            
            def search_products(self, category: Optional[str] = None, 
                              min_price: Optional[float] = None,
                              max_price: Optional[float] = None) -> List[Product]:
                '''상품 검색'''
                results = list(self._products.values())
                
                if category:
                    results = [p for p in results if p.category == category]
                
                if min_price is not None:
                    results = [p for p in results if p.price >= min_price]
                
                if max_price is not None:
                    results = [p for p in results if p.price <= max_price]
                
                return results
            
            @require_admin  
            def update_product_stock(self, admin: User, product_id: int, quantity: int) -> Product:
                '''재고 업데이트 (관리자 전용)'''
                product = self.get_product(product_id)
                product.update_stock(quantity)
                return product
        
        
        class OrderService:
            '''주문 관리 서비스'''
            
            def __init__(self, user_service: UserService, product_service: ProductService):
                self.user_service = user_service
                self.product_service = product_service
                self._orders: Dict[int, Dict[str, Any]] = {}
                self._next_id = 1
            
            @log_operation("주문 생성")
            def create_order(self, user_id: int, product_id: int, quantity: int) -> Dict[str, Any]:
                '''주문 생성'''
                user = self.user_service.get_user(user_id)
                product = self.product_service.get_product(product_id)
                
                if product.stock < quantity:
                    raise ValidationError("재고가 부족합니다")
                
                total_price = product.price * quantity
                order = {
                    'id': self._next_id,
                    'user_id': user_id,
                    'product_id': product_id,
                    'quantity': quantity,
                    'total_price': total_price,
                    'status': 'pending'
                }
                
                # 재고 차감
                product.update_stock(-quantity)
                
                self._orders[order['id']] = order
                self._next_id += 1
                return order
            
            def get_order(self, order_id: int) -> Dict[str, Any]:
                '''주문 조회'''
                if order_id not in self._orders:
                    raise ValueError(f"주문을 찾을 수 없습니다: {order_id}")
                return self._orders[order_id]
            
            def get_user_orders(self, user_id: int) -> List[Dict[str, Any]]:
                '''사용자별 주문 목록'''
                return [order for order in self._orders.values() 
                       if order['user_id'] == user_id]
    """),
        encoding="utf-8",
    )

    # exceptions.py - 커스텀 예외들
    exceptions_py = base_path / "exceptions.py"
    exceptions_py.write_text(
        dedent("""
        '''커스텀 예외 정의'''
        
        
        class BusinessLogicError(Exception):
            '''비즈니스 로직 기본 예외'''
            
            def __init__(self, message: str, code: str = None):
                super().__init__(message)
                self.message = message
                self.code = code
        
        
        class ValidationError(BusinessLogicError):
            '''데이터 검증 오류'''
            
            def __init__(self, message: str, field: str = None):
                super().__init__(message, "VALIDATION_ERROR")
                self.field = field
        
        
        class UserNotFoundError(BusinessLogicError):
            '''사용자를 찾을 수 없음'''
            
            def __init__(self, message: str, user_id: int = None):
                super().__init__(message, "USER_NOT_FOUND")
                self.user_id = user_id
        
        
        class ProductNotFoundError(BusinessLogicError):
            '''상품을 찾을 수 없음'''
            
            def __init__(self, message: str, product_id: int = None):
                super().__init__(message, "PRODUCT_NOT_FOUND")
                self.product_id = product_id
        
        
        class InsufficientPermissionError(BusinessLogicError):
            '''권한 부족'''
            
            def __init__(self, message: str, required_role: str = None):
                super().__init__(message, "INSUFFICIENT_PERMISSION")
                self.required_role = required_role
        
        
        class InventoryError(BusinessLogicError):
            '''재고 관련 오류'''
            
            def __init__(self, message: str, product_id: int = None, available_stock: int = None):
                super().__init__(message, "INVENTORY_ERROR")
                self.product_id = product_id
                self.available_stock = available_stock
    """),
        encoding="utf-8",
    )

    # utils.py - 유틸리티 함수들
    utils_py = base_path / "utils.py"
    utils_py.write_text(
        dedent("""
        '''유틸리티 함수들'''
        import re
        from typing import Any, Dict, List, Optional
        from datetime import datetime, timedelta
        import hashlib
        import json
        
        
        def validate_email(email: str) -> bool:
            '''이메일 형식 검증'''
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, email))
        
        
        def hash_password(password: str, salt: str = None) -> str:
            '''비밀번호 해시화'''
            if salt is None:
                salt = "default_salt"
            
            combined = f"{password}{salt}"
            return hashlib.sha256(combined.encode()).hexdigest()
        
        
        def format_currency(amount: float, currency: str = "KRW") -> str:
            '''통화 포맷팅'''
            if currency == "KRW":
                return f"{amount:,.0f}원"
            elif currency == "USD":
                return f"${amount:,.2f}"
            else:
                return f"{amount:,.2f} {currency}"
        
        
        def calculate_discount(original_price: float, discount_rate: float) -> float:
            '''할인가 계산'''
            if not 0 <= discount_rate <= 1:
                raise ValueError("할인율은 0과 1 사이여야 합니다")
            
            return original_price * (1 - discount_rate)
        
        
        def generate_report_data(orders: List[Dict[str, Any]]) -> Dict[str, Any]:
            '''주문 데이터 기반 리포트 생성'''
            if not orders:
                return {
                    'total_orders': 0,
                    'total_revenue': 0,
                    'average_order_value': 0,
                    'top_products': []
                }
            
            total_orders = len(orders)
            total_revenue = sum(order.get('total_price', 0) for order in orders)
            average_order_value = total_revenue / total_orders if total_orders > 0 else 0
            
            # 상품별 주문 집계
            product_sales = {}
            for order in orders:
                product_id = order.get('product_id')
                quantity = order.get('quantity', 0)
                
                if product_id in product_sales:
                    product_sales[product_id] += quantity
                else:
                    product_sales[product_id] = quantity
            
            # 상위 상품 정렬
            top_products = sorted(
                product_sales.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            return {
                'total_orders': total_orders,
                'total_revenue': total_revenue,
                'average_order_value': average_order_value,
                'top_products': [{'product_id': pid, 'quantity': qty} for pid, qty in top_products]
            }
        
        
        class DataProcessor:
            '''데이터 처리 유틸리티 클래스'''
            
            @staticmethod
            def normalize_text(text: str) -> str:
                '''텍스트 정규화'''
                if not text:
                    return ""
                
                # 소문자 변환, 공백 정리
                normalized = re.sub(r'\s+', ' ', text.lower().strip())
                return normalized
            
            @staticmethod
            def extract_keywords(text: str, min_length: int = 3) -> List[str]:
                '''키워드 추출'''
                if not text:
                    return []
                
                # 알파벳과 한글만 추출
                words = re.findall(r'[a-zA-Z가-힣]+', text)
                keywords = [word.lower() for word in words if len(word) >= min_length]
                
                return list(set(keywords))  # 중복 제거
            
            @classmethod
            def merge_dictionaries(cls, *dicts: Dict[str, Any]) -> Dict[str, Any]:
                '''딕셔너리 병합'''
                result = {}
                for d in dicts:
                    if isinstance(d, dict):
                        result.update(d)
                return result
        
        
        def backup_data(data: Any, filename: str) -> bool:
            '''데이터 백업'''
            try:
                with open(f"backup_{filename}", 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                print(f"백업 실패: {e}")
                return False
    """),
        encoding="utf-8",
    )

    # main.py - 메인 애플리케이션
    main_py = base_path / "main.py"
    main_py.write_text(
        dedent("""
        '''메인 애플리케이션'''
        from typing import Dict, Any
        import logging
        
        from models import User, UserRole
        from services import UserService, ProductService, OrderService
        from exceptions import BusinessLogicError, ValidationError
        from utils import format_currency, generate_report_data, DataProcessor
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        
        class ECommerceApp:
            '''전자상거래 애플리케이션'''
            
            def __init__(self):
                self.user_service = UserService()
                self.product_service = ProductService()
                self.order_service = OrderService(self.user_service, self.product_service)
                self.data_processor = DataProcessor()
                
                # 기본 데이터 초기화
                self._initialize_default_data()
            
            def _initialize_default_data(self) -> None:
                '''기본 데이터 초기화'''
                try:
                    # 관리자 사용자 생성
                    admin = self.user_service.create_user(
                        "관리자", 
                        "admin@example.com", 
                        UserRole.ADMIN
                    )
                    
                    # 일반 사용자들 생성
                    self.user_service.create_user("김철수", "kim@example.com")
                    self.user_service.create_user("이영희", "lee@example.com")
                    
                    # 상품들 생성
                    laptop = self.product_service.create_product("노트북", 1500000, "전자제품")
                    mouse = self.product_service.create_product("마우스", 50000, "전자제품")
                    book = self.product_service.create_product("프로그래밍 책", 35000, "도서")
                    
                    # 재고 설정 (관리자 권한으로)
                    self.product_service.update_product_stock(admin, laptop.id, 10)
                    self.product_service.update_product_stock(admin, mouse.id, 50)
                    self.product_service.update_product_stock(admin, book.id, 20)
                    
                    logger.info("기본 데이터 초기화 완료")
                    
                except BusinessLogicError as e:
                    logger.error(f"기본 데이터 초기화 실패: {e}")
            
            def create_sample_orders(self) -> None:
                '''샘플 주문 생성'''
                try:
                    # 사용자들의 주문 생성
                    users = self.user_service.list_users()
                    products = self.product_service.search_products()
                    
                    if len(users) >= 2 and len(products) >= 2:
                        # 첫 번째 사용자 주문
                        user1 = users[1]  # 관리자가 아닌 첫 번째 사용자
                        self.order_service.create_order(user1.id, products[0].id, 1)
                        self.order_service.create_order(user1.id, products[1].id, 2)
                        
                        # 두 번째 사용자 주문  
                        if len(users) > 2:
                            user2 = users[2]
                            self.order_service.create_order(user2.id, products[0].id, 1)
                    
                    logger.info("샘플 주문 생성 완료")
                    
                except BusinessLogicError as e:
                    logger.error(f"샘플 주문 생성 실패: {e}")
            
            def generate_sales_report(self) -> Dict[str, Any]:
                '''매출 리포트 생성'''
                try:
                    # 모든 주문 조회
                    all_orders = []
                    users = self.user_service.list_users()
                    
                    for user in users:
                        user_orders = self.order_service.get_user_orders(user.id)
                        all_orders.extend(user_orders)
                    
                    # 리포트 데이터 생성
                    report_data = generate_report_data(all_orders)
                    
                    # 포맷팅된 리포트
                    formatted_report = {
                        'summary': {
                            'total_orders': report_data['total_orders'],
                            'total_revenue': format_currency(report_data['total_revenue']),
                            'average_order_value': format_currency(report_data['average_order_value'])
                        },
                        'top_products': report_data['top_products']
                    }
                    
                    logger.info("매출 리포트 생성 완료")
                    return formatted_report
                    
                except Exception as e:
                    logger.error(f"매출 리포트 생성 실패: {e}")
                    return {}
            
            def search_products_with_filters(self, query: str = None, 
                                           category: str = None,
                                           price_range: tuple = None) -> list:
                '''필터링 상품 검색'''
                try:
                    min_price, max_price = price_range if price_range else (None, None)
                    
                    products = self.product_service.search_products(
                        category=category,
                        min_price=min_price,
                        max_price=max_price
                    )
                    
                    # 텍스트 검색 필터링
                    if query:
                        normalized_query = self.data_processor.normalize_text(query)
                        filtered_products = []
                        
                        for product in products:
                            product_text = f"{product.name} {product.category}"
                            normalized_text = self.data_processor.normalize_text(product_text)
                            
                            if normalized_query in normalized_text:
                                filtered_products.append(product)
                        
                        products = filtered_products
                    
                    return products
                    
                except Exception as e:
                    logger.error(f"상품 검색 실패: {e}")
                    return []
            
            def run_demo(self) -> None:
                '''데모 실행'''
                logger.info("=== 전자상거래 애플리케이션 데모 시작 ===")
                
                # 샘플 주문 생성
                self.create_sample_orders()
                
                # 사용자 목록 출력
                users = self.user_service.list_users()
                logger.info(f"총 {len(users)}명의 사용자")
                
                # 상품 검색 테스트
                laptop_products = self.search_products_with_filters(
                    query="노트북", 
                    category="전자제품"
                )
                logger.info(f"노트북 검색 결과: {len(laptop_products)}개")
                
                # 매출 리포트 생성
                report = self.generate_sales_report()
                logger.info(f"매출 리포트: {report}")
                
                logger.info("=== 데모 완료 ===")
        
        
        def main():
            '''메인 함수'''
            try:
                app = ECommerceApp()
                app.run_demo()
                return app
                
            except Exception as e:
                logger.error(f"애플리케이션 실행 실패: {e}")
                raise
        
        
        if __name__ == "__main__":
            main()
    """),
        encoding="utf-8",
    )


def test_complex_code_analysis():
    """복잡한 코드 구조 분석 테스트"""

    logger.info("복잡한 코드 구조 E2E 테스트 시작")

    # 설정 생성
    config = CoreConfig()
    config.embedding.provider = "huggingface"
    config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    config.project_name = "complex-test-project"

    # 환경변수에서 API 키 확인
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("❌ GOOGLE_API_KEY 환경변수를 설정해주세요")
        return

    config.llm.api_key = api_key
    config.llm.model_name = "gemini-2.0-flash-lite"  # 더 많은 요청 가능
    config.llm.temperature = 0.8
    config.llm.max_tokens = 4096  # 더 긴 응답 허용 (늘림)

    # 컴포넌트 초기화
    persistence, embedder, retriever, graph_service, agent = create_from_config(config)

    try:
        # 임시 디렉토리에 복잡한 프로젝트 생성
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            create_complex_project(tmp_path)

            # 이전 테스트 데이터 정리
            logger.info("이전 테스트 데이터 정리 중...")
            persistence.clear_project_data(config.project_name)

            # 코드 분석 및 저장
            logger.info(f"복잡한 코드 분석 시작: {tmp_path}")
            graph = graph_service.analyze_and_store_project(
                str(tmp_path), create_embeddings=True
            )

            # 분석 결과 통계
            logger.info(
                f"분석 완료: {len(graph.nodes)}개 노드, {len(graph.relations)}개 관계"
            )

            # 노드 타입별 통계
            node_types = {}
            for node in graph.nodes.values():
                node_type = (
                    node.node_type.value
                    if hasattr(node.node_type, "value")
                    else str(node.node_type)
                )
                node_types[node_type] = node_types.get(node_type, 0) + 1

            logger.info(f"노드 타입별 통계: {node_types}")

            # 다양한 GraphRAG 질의 테스트
            test_queries = [
                "UserService 클래스의 create_user 메서드는 어떤 역할을 하나요?",
                "require_admin 데코레이터는 어떻게 작동하나요?",
                "Product 클래스가 DatabaseEntity를 상속받는 이유는 무엇인가요?",
                "주문 생성 시 재고 관리는 어떻게 처리되나요?",
                "사용자 권한 시스템은 어떻게 구현되어 있나요?",
            ]

            for i, query in enumerate(test_queries, 1):
                logger.info(f"\n=== 질의 {i}/5: {query} ===")

                # 검색 단계 먼저 확인
                search_results = agent.get_search_results(query)
                logger.info(f"검색 결과: {len(search_results)}개")

                # 상위 3개 결과 출력
                for idx, result in enumerate(search_results[:3]):
                    result_type = (
                        result.node_type.value
                        if hasattr(result.node_type, "value")
                        else str(result.node_type)
                    )
                    logger.info(f"  {idx + 1}. {result_type}: {result.name}")

                # GraphRAG 답변 생성
                answer = agent.query(query)
                logger.info(f"답변 길이: {len(answer)}자")
                logger.info("=" * 50)
                logger.info("전체 답변:")
                logger.info(answer)
                logger.info("=" * 50)

                # 잠시 대기 (API 제한 고려)
                import time

                time.sleep(1)

            # 최종 통계
            stats = graph_service.get_statistics()
            logger.info("\n=== 최종 프로젝트 통계 ===")
            logger.info(f"전체 통계: {stats}")

            logger.info("✅ 복잡한 코드 구조 E2E 테스트 성공!")

    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        raise

    finally:
        # 정리
        logger.info("테스트 데이터 정리 중...")
        try:
            persistence.clear_project_data(config.project_name)
            persistence.close()
        except Exception as e:
            logger.warning(f"정리 중 오류: {e}")


if __name__ == "__main__":
    test_complex_code_analysis()
