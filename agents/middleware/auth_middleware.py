import logging
import re
import time
from typing import Optional

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from agents.common.error_messages import get_error_message
from agents.common.http_utils import add_cors_headers
from agents.common.response import RestResponse
from agents.exceptions import ErrorCode
from agents.services import open_service
from agents.utils.jwt_utils import verify_token

security = HTTPBearer()
logger = logging.getLogger(__name__)

class AuthConfig:
    """Authentication configuration"""
    PUBLIC_PATHS = [
        "/api/auth/login", "/api/auth/register",
        "/api/auth/wallet/nonce", "/api/auth/wallet/login",
        "/api/auth/refresh", "/api/auth/reset-password",
        "/api/auth/verify-email", "/docs", "/redoc",
        "/api/health", "/openapi.json", "/api/upload/file",
        "/api/images/generate", "/api/agents/public",
        "/api/categories", "/", "/api/health/detailed"
    ]
    PUBLIC_PREFIXES = ["/api/files/", "/api/categories/", "/mcp", "/messages/"]
    OPEN_API_PATHS = [
        r"^/api/agents/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/dialogue$",
        r"^/api/open/agents/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/dialogue$",
        r"^/api/mcp/.*$",
        r"^/mcp/.*$",
        r"^/api/tools/.*$",
        r"^/A2A/.*$",
    ]

class AuthResponse:
    """Authentication response helper"""
    @staticmethod
    def error(error_code: ErrorCode) -> JSONResponse:
        return add_cors_headers(JSONResponse(
            status_code=200,
            content=RestResponse(
                code=error_code,
                msg=get_error_message(error_code)
            ).model_dump(exclude_none=True)
        ))

class AuthError(Exception):
    def __init__(self, error_code: ErrorCode):
        self.error_code = error_code

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path

        # Check public paths
        if (path in AuthConfig.PUBLIC_PATHS or
            any(path.startswith(prefix) for prefix in AuthConfig.PUBLIC_PREFIXES)):
            return await call_next(request)

        try:
            # Handle Open API paths
            if any(re.match(pattern, path) for pattern in AuthConfig.OPEN_API_PATHS):
                # First try to authenticate using Open Platform token
                if await self._authenticate_open_api_token(request):
                    return await call_next(request)
                    
                # If token authentication fails, try signature authentication
                if await self._authenticate_open_api(request):
                    return await call_next(request)

            # Try JWT authentication
            await self._authenticate_jwt(request)
            return await call_next(request)
        except AuthError as e:
            return AuthResponse.error(e.error_code)
        except Exception as e:
            logger.error("Authentication error", exc_info=True)
            return AuthResponse.error(ErrorCode.TOKEN_INVALID)

    async def _authenticate_open_api_token(self, request: Request) -> bool:
        """Authenticate using Open API token"""
        try:
            # Get X-API-Token header
            token = request.headers.get("X-API-Token")
            if not token:
                return False
                
            # Verify token and get credentials
            try:
                async with request.app.state.db() as session:
                    user_info = await open_service.verify_token_and_get_credentials(token, session)
                    if not user_info:
                        return False
                        
                    request.state.user = user_info
                    return True
            except Exception as e:
                logger.error(f"Database operation failed: {str(e)}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"API token authentication failed: {str(e)}", exc_info=True)
            return False

    async def _authenticate_open_api(self, request: Request) -> bool:
        """Authenticate using Open API credentials"""
        try:
            headers = request.headers
            access_key = headers.get("X-Access-Key")
            timestamp = headers.get("X-Timestamp")
            signature = headers.get("X-Signature")

            if not all([access_key, timestamp, signature]):
                return False

            if not self._verify_timestamp(timestamp):
                return False

            try:
                async with request.app.state.db() as session:
                    credentials = await open_service.get_credentials(access_key, session)
                    if not credentials:
                        logger.warning(f"Invalid access_key: {access_key}")
                        return False

                    if not open_service.verify_signature(
                        access_key, credentials.secret_key, timestamp, signature
                    ):
                        logger.warning(f"Invalid signature for access_key: {access_key}")
                        return False
                    user_info = await open_service.verify_token_and_get_credentials(credentials.token, session)
                    if not user_info:
                        return False
                    user_info["type"] = "api_key"
                    request.state.user = user_info
                    return True
            except Exception as e:
                logger.error(f"Database operation failed: {str(e)}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"Open API authentication failed: {str(e)}", exc_info=True)
            return False

    async def _authenticate_jwt(self, request: Request):
        """Authenticate using JWT token"""
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise AuthError(ErrorCode.TOKEN_MISSING)

            token = auth_header.split(" ")[1] if auth_header.startswith("Bearer ") else auth_header
            payload = verify_token(token)
            if not payload:
                raise AuthError(ErrorCode.TOKEN_EXPIRED)

            request.state.user = payload
        except (IndexError, AttributeError) as e:
            logger.error(f"JWT token parsing error: {str(e)}")
            raise AuthError(ErrorCode.TOKEN_INVALID)

    def _verify_timestamp(self, timestamp: str, max_age: int = 300) -> bool:
        """Verify if timestamp is within allowed range"""
        try:
            current_time = int(time.time())
            request_time = int(timestamp)
            return abs(current_time - request_time) <= max_age
        except ValueError:
            return False

async def get_current_user(request: Request) -> dict:
    """Get authenticated user from request"""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=401,
            detail=get_error_message(ErrorCode.UNAUTHORIZED)
        )
    return user

async def get_optional_current_user(request: Request) -> Optional[dict]:
    """Get optional user from request"""
    return getattr(request.state, "user", None)