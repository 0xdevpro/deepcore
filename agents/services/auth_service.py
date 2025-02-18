import uuid
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import re
import json
import eth_account
import logging

from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.models import User
from agents.protocol.schemas import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse, NonceResponse, \
    WalletLoginRequest, WalletLoginResponse, TokenResponse
from agents.utils.jwt_utils import generate_token, generate_token_pair, verify_refresh_token, generate_access_token
from agents.utils.web3_utils import generate_nonce, get_message_to_sign, verify_signature
from agents.common.redis_utils import redis_utils

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
NONCE_EXPIRY_MINUTES = 1  # Nonce expires after 5 minutes
NONCE_KEY_PREFIX = "wallet_nonce:"  # Redis key prefix for nonce storage


def get_nonce_key(wallet_address: str) -> str:
    """Generate Redis key for storing nonce"""
    return f"{NONCE_KEY_PREFIX}{wallet_address}"


async def login(request: LoginRequest, session: AsyncSession) -> LoginResponse:
    """
    Handle user login with username or email
    """
    try:
        result = await session.execute(
            select(User).where(
                (User.username == request.username) |
                (User.email == request.username)
            )
        )
        user = result.scalar_one_or_none()

        if not user:
            raise CustomAgentException(
                ErrorCode.INVALID_CREDENTIALS,
                "Invalid username/email or password"
            )

        if not user.check_password(request.password):
            raise CustomAgentException(
                ErrorCode.INVALID_CREDENTIALS,
                "Invalid username/email or password"
            )

        # Generate token pair
        access_token, refresh_token = generate_token_pair(
            user_id=str(user.id),
            username=user.username,
            tenant_id=user.tenant_id
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": user.to_dict()
        }
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in login: {str(e)}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.INTERNAL_ERROR,
            f"Login failed: {str(e)}"
        )


async def register(request: RegisterRequest, session: AsyncSession) -> RegisterResponse:
    """
    Handle user registration
    """
    if not EMAIL_REGEX.match(request.email):
        raise CustomAgentException(message="Invalid email format")

    if len(request.email) > 120:
        raise CustomAgentException(message="Email is too long")

    # Check if username already exists
    result = await session.execute(
        select(User).where(User.username == request.username)
    )
    if result.scalar_one_or_none():
        raise CustomAgentException(message="Username already exists")

    # Check if email already exists
    result = await session.execute(
        select(User).where(User.email == request.email)
    )
    if result.scalar_one_or_none():
        raise CustomAgentException(message="Email already exists")

    # Generate tenant_id
    tenant_id = str(uuid.uuid4())

    # Create new user with tenant_id
    user = User(
        username=request.username,
        email=request.email,
        tenant_id=tenant_id  # Add tenant_id
    )
    user.set_password(request.password)

    session.add(user)
    await session.commit()
    await session.refresh(user)

    return {
        "message": "User registered successfully",
        "user": user.to_dict()
    }


async def get_wallet_nonce(wallet_address: str, session: AsyncSession) -> NonceResponse:
    """
    Get or generate nonce for wallet signature with expiry time using Redis
    """
    # Generate new nonce and message
    nonce = generate_nonce()
    message = get_message_to_sign(wallet_address, nonce)

    # Store nonce in Redis with expiry
    nonce_data = {
        "nonce": nonce,
        "created_at": datetime.utcnow().isoformat()
    }
    redis_utils.set_value(
        get_nonce_key(wallet_address),
        json.dumps(nonce_data),
        ex=NONCE_EXPIRY_MINUTES * 60
    )

    # Check if user exists
    result = await session.execute(
        select(User).where(User.wallet_address == wallet_address)
    )
    user = result.scalar_one_or_none()

    if not user:
        # Create temporary user entry with generated username
        temp_username = f"wallet_{wallet_address[-8:]}"

        # Check if the generated username exists
        username_result = await session.execute(
            select(User).where(User.username == temp_username)
        )
        if username_result.scalar_one_or_none():
            temp_username = f"wallet_{wallet_address[-8:]}_{uuid.uuid4().hex[:4]}"

        # Create new user with tenant_id
        tenant_id = str(uuid.uuid4())
        user = User(
            username=temp_username,
            wallet_address=wallet_address,
            tenant_id=tenant_id
        )
        session.add(user)
        await session.commit()

    return {
        "nonce": nonce,
        "message": message,
        "expires_in": NONCE_EXPIRY_MINUTES * 60
    }


async def wallet_login(request: WalletLoginRequest, session: AsyncSession) -> WalletLoginResponse:
    """
    Handle wallet login/registration with nonce verification
    """
    try:
        # Verify signature
        if not request.signature:
            raise CustomAgentException(message="Signature is required")

        # Get stored nonce data from Redis
        nonce_key = get_nonce_key(request.wallet_address)
        stored_nonce_data = redis_utils.get_value(nonce_key)

        if not stored_nonce_data:
            raise CustomAgentException(message="Nonce not found or expired. Please request a new one.")

        nonce_data = json.loads(stored_nonce_data)
        nonce = nonce_data["nonce"]

        # Verify signature
        message = get_message_to_sign(request.wallet_address, nonce)
        if not verify_signature(message, request.signature, request.wallet_address):
            raise CustomAgentException(message="Invalid signature")

        # Delete used nonce from Redis
        redis_utils.delete_key(nonce_key)

        # Get or create user
        user = await get_or_create_wallet_user(request.wallet_address, session)

        # Set is_new_user flag
        is_new_user = not user.create_time

        # Update create_time if this is first login
        if is_new_user:
            user.create_time = datetime.utcnow()
            await session.commit()

        # Generate token pair
        access_token, refresh_token = generate_token_pair(
            user_id=str(user.id),
            username=user.username,
            tenant_id=user.tenant_id
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": user.to_dict(),
            "is_new_user": is_new_user
        }

    except Exception as e:
        logger.error(f"Error in wallet login: {e}", exc_info=True)
        raise e


async def get_or_create_wallet_user(wallet_address: str, session: AsyncSession) -> User:
    """
    Get existing user by wallet address or create a new one
    """
    # Check if user exists
    result = await session.execute(
        select(User).where(User.wallet_address == wallet_address)
    )
    user = result.scalar_one_or_none()

    if not user:
        # Create temporary user entry with generated username
        temp_username = f"wallet_{wallet_address[-8:]}"

        # Check if the generated username exists
        username_result = await session.execute(
            select(User).where(User.username == temp_username)
        )
        if username_result.scalar_one_or_none():
            temp_username = f"wallet_{wallet_address[-8:]}_{uuid.uuid4().hex[:4]}"

        # Create new user with tenant_id
        tenant_id = str(uuid.uuid4())
        user = User(
            username=temp_username,
            wallet_address=wallet_address,
            tenant_id=tenant_id,
            create_time=datetime.utcnow()
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

    return user


async def refresh_token(refresh_token: str, session: AsyncSession) -> TokenResponse:
    """
    Refresh access token using refresh token
    """
    # Verify refresh token
    user_id = verify_refresh_token(refresh_token)
    if not user_id:
        raise CustomAgentException(message="Invalid or expired refresh token")
    
    # Get user info
    result = await session.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise CustomAgentException(message="User not found")
    
    # Generate new token pair
    access_token, new_refresh_token = generate_token_pair(
        user_id=str(user.id),
        username=user.username,
        tenant_id=user.tenant_id
    )
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token
    }
