import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from agents.agent.executor.deep_thinking_executor import DeepThinkDto
from agents.common.error_messages import get_error_message
from agents.common.response import RestResponse
from agents.exceptions import CustomAgentException, ErrorCode
from agents.middleware.auth_middleware import get_optional_current_user
from agents.models.db import get_db
from agents.services.data_service import DataService, AnalyzeTokenInfoDto, TransAmountStatisticsDto, ChainEnum, \
    CommandEnum, get_cryptocurrency_latest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/data/xpro/hot", summary="Xpro Hot")
async def xpro_hot(
        page: int = Query(1, description="Page number"),
        page_size: int = Query(10, description="Items per page"),
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Get Xpro Hot data
    
    Parameters:
        - page: Page number, default is 1
        - page_size: Items per page, default is 10
    
    Returns:
        - List of Xpro Hot data
    """
    try:
        data_service = DataService(session)
        result = await data_service.get_xpro_hot(page, page_size)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error getting xpro hot data: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting xpro hot data: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/data/xpro/ca", summary="Xpro Ca")
async def xpro_ca(
        page: int = Query(1, description="Page number"),
        page_size: int = Query(10, description="Items per page"),
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Get Xpro Ca data
    
    Parameters:
        - page: Page number, default is 1
        - page_size: Items per page, default is 10
    
    Returns:
        - List of Xpro Ca data
    """
    try:
        data_service = DataService(session)
        result = await data_service.get_xpro_ca(page, page_size)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error getting xpro ca data: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting xpro ca data: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/agent/stream/analyze_token", summary="Analyze Token")
async def analyze_token(
        token_info: AnalyzeTokenInfoDto = Body(..., description="Token information for analysis"),
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Analyze token information
    
    This endpoint provides a token analysis engine, useful for finding KOLs, developers, and call channels.
    The response is streamed as Server-Sent Events (SSE).
    
    Parameters:
        - token_info: Token information including chain and contract address
    
    Returns:
        - Streaming response with analysis results
    """
    try:
        data_service = DataService(session)
        return StreamingResponse(
            content=data_service.analyze_token(token_info),
            media_type="text/event-stream"
        )
    except CustomAgentException as e:
        logger.error(f"Error analyzing token: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error analyzing token: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/crypto/trans/amount/toplow", summary="Get Transaction Amount TopN or LowN")
async def get_trans_amount_statistics(
        chain: ChainEnum = Query(..., description="The blockchain chain"),
        cmd: CommandEnum = Query(..., description="The command to fetch specific statistics"),
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Get transaction amount TopN or LowN statistics
    
    This endpoint provides transaction amount statistics for different blockchains and time periods.
    
    Parameters:
        - chain: The blockchain chain (SOLANA, BASE, ETH, SUI, BSC)
        - cmd: The command to fetch specific statistics (e.g., top100_15m, low100_24h)
    
    Returns:
        - List of transaction statistics
    """
    try:
        data_service = DataService(session)
        params = TransAmountStatisticsDto(chain=chain, cmd=cmd)
        result = await data_service.get_trans_amount_statistics(params)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error fetching transaction statistics: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error fetching transaction statistics: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/agent/stream/deep-think", summary="Deep Analysis")
async def deep_think(
        query: DeepThinkDto = Body(..., description="Query for deep analysis"),
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Deep analysis with streaming response
    
    This endpoint provides in-depth analysis based on your query, with results streamed as they're generated.
    The response is streamed as Server-Sent Events (SSE).
    
    Parameters:
        - query: Contains the question (q) to analyze
    
    Returns:
        - Streaming response with analysis results
    """
    try:
        data_service = DataService(session)
        return StreamingResponse(
            content=data_service.deep_think(query),
            media_type="text/event-stream"
        )
    except CustomAgentException as e:
        logger.error(f"Error performing deep analysis: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in deep analysis: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/data/cryptocurrency/latest")
async def cryptocurrency_latest(
        id: int,
):
    try:
        result = {}
        latest = await get_cryptocurrency_latest(id)
        result.update({'latest': latest})
        return RestResponse(data=result)
    except Exception as e:
        logger.error(f"Unexpected error getting cryptocurrency latest data: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )
