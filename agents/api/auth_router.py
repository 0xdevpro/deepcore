import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from agents.models.db import get_db
from agents.protocol.schemas import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse, NonceResponse, \
    WalletLoginRequest, WalletLoginResponse, RefreshTokenRequest, TokenResponse
from agents.services import auth_service
from agents.common.response import RestResponse
from agents.exceptions import CustomAgentException, ErrorCode
from agents.common.error_messages import get_error_message

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/login", response_model=RestResponse[LoginResponse], summary="User login")
async def login(request: LoginRequest, session: AsyncSession = Depends(get_db)):
    """
    User login endpoint

    - **username**: Username or email for login
    - **password**: Password for login
    """
    try:
        result = await auth_service.login(request, session)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error in user login: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in user login: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/register", response_model=RestResponse[RegisterResponse], summary="User registration")
async def register(request: RegisterRequest, session: AsyncSession = Depends(get_db)):
    """
    User registration endpoint

    - **username**: Username for registration
    - **email**: Email address
    - **password**: Password for registration
    """
    try:
        result = await auth_service.register(request, session)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error in user registration: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in user registration: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/wallet/nonce", response_model=RestResponse[NonceResponse])
async def get_nonce(
        wallet_address: str,
        session: AsyncSession = Depends(get_db)
):
    """
    Get nonce for wallet signature

    - **wallet_address**: Ethereum wallet address
    """
    try:
        result = await auth_service.get_wallet_nonce(wallet_address, session)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error getting nonce for wallet: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting nonce: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/wallet/login", response_model=RestResponse[WalletLoginResponse])
async def wallet_login(
        request: WalletLoginRequest,
        session: AsyncSession = Depends(get_db)
):
    """
    Login with wallet signature

    - **wallet_address**: Ethereum wallet address
    - **signature**: Signed message
    """
    try:
        result = await auth_service.wallet_login(request, session)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error in wallet login: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in wallet login: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/refresh", response_model=RestResponse[TokenResponse])
async def refresh_token(
    request: RefreshTokenRequest,
    session: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token
    
    - **refresh_token**: Valid refresh token
    """
    try:
        result = await auth_service.refresh_token(request.refresh_token, session)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error refreshing token: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error refreshing token: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )
