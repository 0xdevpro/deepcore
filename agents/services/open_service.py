import hashlib
import hmac
import uuid
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.models import OpenPlatformKey
from agents.common.encryption_utils import encryption_utils

# Remove JWT and token secret configuration
# TOKEN_SECRET = "open_platform_token_secret"  # In production, this should be stored in environment variables or config files
TOKEN_EXPIRE_HOURS = 24 * 365 * 10  # Token validity period set to 10 years, effectively permanent

def generate_key_pair():
    """Generate a new access key and secret key pair for open platform"""
    access_key = f"ak_{uuid.uuid4().hex[:16]}"
    secret_key = f"sk_{uuid.uuid4().hex}"
    return access_key, secret_key

def generate_signature(access_key: str, secret_key: str, timestamp: str) -> str:
    """Generate signature for open platform API request"""
    message = f"{access_key}{timestamp}"
    return hmac.new(
        secret_key.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()

def verify_signature(access_key: str, secret_key: str, timestamp: str, signature: str) -> bool:
    """Verify open platform API signature"""
    expected_signature = generate_signature(access_key, secret_key, timestamp)
    return hmac.compare_digest(signature, expected_signature)

async def get_or_create_credentials(user: dict, session: AsyncSession) -> dict:
    """Get existing credentials or create new ones if they don't exist"""
    user_id = user.get("user_id")
    if not user_id:
        raise CustomAgentException(
            error_code=ErrorCode.INVALID_REQUEST,
            message="User ID is required"
        )

    # Try to get existing credentials
    query = select(OpenPlatformKey).where(
        OpenPlatformKey.user_id == user_id,
        OpenPlatformKey.is_deleted == False
    )
    result = await session.execute(query)
    credentials = result.scalar_one_or_none()
    
    if credentials:
        return {
            "access_key": credentials.access_key,
            "secret_key": credentials.secret_key
        }
    
    # Create new credentials if none exist
    access_key, secret_key = generate_key_pair()
    credentials = OpenPlatformKey(
        name=f"User {user_id} Open Platform Key",
        access_key=access_key,
        secret_key=secret_key,
        user_id=user_id,
        created_at=datetime.utcnow()
    )
    
    session.add(credentials)
    await session.commit()
    
    return {
        "access_key": access_key,
        "secret_key": secret_key
    }

async def get_credentials(access_key: str, session: AsyncSession) -> Optional[OpenPlatformKey]:
    """Get credentials by access key"""
    query = select(OpenPlatformKey).where(
        OpenPlatformKey.access_key == access_key,
        OpenPlatformKey.is_deleted == False
    )
    result = await session.execute(query)
    return result.scalar_one_or_none()

async def generate_token(access_key: str, session: AsyncSession) -> Dict[str, Any]:
    """Generate a token for open platform API access"""
    credentials = await get_credentials(access_key, session)
    if not credentials:
        raise CustomAgentException(
            error_code=ErrorCode.INVALID_REQUEST,
            message="Invalid access key"
        )
    
    # Create token expiration (set to 10 years, effectively permanent)
    expiration = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    exp_timestamp = int(expiration.timestamp())
    
    # Create token payload as a string
    token_data = f"{access_key}:{credentials.user_id}:{exp_timestamp}:{int(time.time())}"
    
    # Encrypt token using encryption_utils
    token = encryption_utils.encrypt(token_data)
    if not token:
        raise CustomAgentException(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to generate token"
        )
    
    # Store token in database (no need for additional encryption)
    stmt = (
        update(OpenPlatformKey)
        .where(OpenPlatformKey.access_key == access_key)
        .values(token=token, token_created_at=datetime.utcnow())
    )
    await session.execute(stmt)
    await session.commit()
    
    return {
        "token": token,
        "expires_at": exp_timestamp
    }

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify open platform API token"""
    try:
        # Decrypt token
        decrypted_token = encryption_utils.decrypt(token)
        if not decrypted_token:
            return None
        
        # Parse token data
        parts = decrypted_token.split(":")
        if len(parts) != 4:
            return None
        
        access_key, user_id, exp_str, iat_str = parts
        
        # Check expiration
        try:
            exp = int(exp_str)
            if exp < int(time.time()):
                return None
        except ValueError:
            return None
        
        # Return payload similar to JWT format for compatibility
        return {
            "access_key": access_key,
            "user_id": int(user_id),
            "exp": exp,
            "iat": int(iat_str)
        }
    except Exception:
        return None

async def reset_token(access_key: str, session: AsyncSession) -> Dict[str, Any]:
    """Reset token for open platform API access"""
    # Verify access_key exists
    credentials = await get_credentials(access_key, session)
    if not credentials:
        raise CustomAgentException(
            error_code=ErrorCode.INVALID_REQUEST,
            message="Invalid access key"
        )
    
    # Generate new token
    return await generate_token(access_key, session)

async def get_token(access_key: str, session: AsyncSession) -> Optional[str]:
    """Get stored token for access key"""
    credentials = await get_credentials(access_key, session)
    if not credentials or not credentials.token:
        return None
    
    # Token is already encrypted, no need for additional decryption
    return credentials.token

async def verify_stored_token(access_key: str, token: str, session: AsyncSession) -> bool:
    """Verify if provided token matches the stored token"""
    stored_token = await get_token(access_key, session)
    if not stored_token:
        return False
    
    return stored_token == token 
