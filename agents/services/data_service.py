import logging
from enum import Enum
from typing import Dict, Any, AsyncGenerator, List, Optional

import aiohttp
import httpx
from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from agents.common.config import SETTINGS
from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.db import get_db

logger = logging.getLogger(__name__)

class AnalyzeTokenInfoDto(BaseModel):
    """
    Data transfer object for token analysis request
    """
    chain: str  # Platform of the token, must be 'sol' or 'eth'
    ca: str     # Token contract address

class ChainEnum(str, Enum):
    SOLANA = "SOLANA"
    BASE = "BASE"
    ETH = "ETH"
    SUI = "SUI"
    BSC = "BSC"

class CommandEnum(str, Enum):
    TOP100_15M = "top100_15m"
    TOP100_30M = "top100_30m"
    TOP100_1H = "top100_1h"
    TOP100_4H = "top100_4h"
    TOP100_8H = "top100_8h"
    TOP100_12H = "top100_12h"
    TOP100_24H = "top100_24h"
    LOW100_15M = "low100_15m"
    LOW100_30M = "low100_30m"
    LOW100_1H = "low100_1h"
    LOW100_4H = "low100_4h"
    LOW100_8H = "low100_8h"
    LOW100_12H = "low100_12h"
    LOW100_24H = "low100_24h"

class TransAmountStatisticsDto(BaseModel):
    """
    Data transfer object for transaction amount statistics request
    """
    chain: ChainEnum
    cmd: CommandEnum

class DeepThinkDto(BaseModel):
    """
    Data transfer object for deep analysis request
    """
    q: str  # Query string for deep analysis

class DataService:
    def __init__(self, session: AsyncSession = Depends(get_db)):
        self.db_session = session
        # Configure base URL and API key
        self.api_base = SETTINGS.DATA_API_BASE  # Get API base URL from configuration
        self.api_key = SETTINGS.DATA_API_KEY  # Get API key from configuration

    async def get_xpro_hot(self, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """
        Get Xpro Hot data
        :param page: Page number
        :param page_size: Items per page
        :return: API response data
        """
        url = f"{self.api_base}/p/data/xpro/hot"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        params = {
            "page": page,
            "page_size": page_size
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=params, timeout=httpx.Timeout(30.0))
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                logger.error(f"Request error for xpro_hot: {str(e)}", exc_info=True)
                raise Exception(f"Failed to fetch xpro_hot data: {str(e)}")
    
    async def get_xpro_ca(self, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """
        Get Xpro Ca data
        :param page: Page number
        :param page_size: Items per page
        :return: API response data
        """
        url = f"{self.api_base}/p/data/xpro/ca"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        params = {
            "page": page,
            "page_size": page_size
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=params, timeout=httpx.Timeout(30.0))
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                logger.error(f"Request error for xpro_ca: {str(e)}", exc_info=True)
                raise Exception(f"Failed to fetch xpro_ca data: {str(e)}")
                
    async def analyze_token(self, token_info: AnalyzeTokenInfoDto) -> AsyncGenerator[str, None]:
        """
        Analyze token information with streaming response
        
        :param token_info: Token information including chain and contract address
        :return: Async generator yielding analysis results
        """
        url = f"{self.api_base}/p/agent/stream/analyze_token"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        payload = token_info.dict()
        
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=httpx.Timeout(60.0)  # Longer timeout for analysis
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        yield chunk
            except httpx.RequestError as e:
                logger.error(f"Request error for analyze_token: {str(e)}", exc_info=True)
                yield f"error: {str(e)}"

    async def get_trans_amount_statistics(self, params: TransAmountStatisticsDto) -> List[Dict[str, Any]]:
        """
        Get transaction amount TopN or LowN statistics
        
        :param params: Parameters including chain and command
        :return: List of transaction statistics
        """
        url = f"{self.api_base}/crypto/trans/amount/toplow"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        # Convert Enum values to strings
        query_params = {
            "chain": params.chain.value,
            "cmd": params.cmd.value
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url, 
                    headers=headers, 
                    params=query_params,
                    timeout=httpx.Timeout(30.0)
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                logger.error(f"Request error for get_trans_amount_statistics: {str(e)}", exc_info=True)
                raise CustomAgentException(
                    error_code=ErrorCode.API_CALL_ERROR,
                    message=f"Failed to fetch transaction statistics: {str(e)}"
                )
                
    async def deep_think(self, params: DeepThinkDto) -> AsyncGenerator[str, None]:
        """
        Deep analysis with streaming response
        
        :param params: Query parameters including the question for analysis
        :return: Async generator yielding analysis results
        """
        url = f"{self.api_base}/p/agent/stream/deep-think"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        payload = {"q": params.q}
        
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=httpx.Timeout(120.0)  # Longer timeout for deep analysis
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        yield chunk
            except httpx.RequestError as e:
                logger.error(f"Request error for deep_think: {str(e)}", exc_info=True)
                yield f"error: {str(e)}"


async def get_cryptocurrency_latest(id: int) -> Optional[Any]:
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': SETTINGS.C_PRO_API_KEY,
    }

    params = {
        'id': id,
        'convert': "USD"
    }

    logger.info(f"Calling {SETTINGS.CRYPTOCURRENCY_LATEST_URL} with params: {params}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SETTINGS.CRYPTOCURRENCY_LATEST_URL, headers=headers, params=params) as response:
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        logging.error(e)

    return None