import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from agents.common.response import RestResponse
from agents.middleware.auth_middleware import get_current_user
from agents.models.db import get_db
from agents.protocol.schemas import DepositRequest, ProfileInfo
from agents.services.profiles_service import get_profile_info, get_agent_usage_stats, bg_check_tx

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/profile/deposit")
async def deposit(tx_info: DepositRequest,
                  background_tasks: BackgroundTasks,
                  user: dict = Depends(get_current_user)):
    background_tasks.add_task(bg_check_tx, user, tx_info)
    return RestResponse()


@router.get("/profile/info", response_model=RestResponse[ProfileInfo])
async def info(user: dict = Depends(get_current_user),
               session: AsyncSession = Depends(get_db)):
    data = await get_profile_info(user, session)
    return RestResponse(data=data)


@router.get(
    "/profile/agent_usage_stats",
    summary="Get agent usage statistics with pagination",
    description="Get all agent usage statistics for the current user with pagination. Returns total, page, page_size, and items. 'page' is the page number (starting from 1), 'page_size' is the number of items per page."
)
async def agent_usage_stats(
    user: dict = Depends(get_current_user),
    page: int = 1,  # Page number, starting from 1
    page_size: int = 10  # Number of items per page, default 10, max 100
):
    """
    Get all agent usage statistics for the current user with pagination.
    - **page**: Page number, starting from 1
    - **page_size**: Number of items per page, default 10, max 100
    The response contains total, page, page_size, and items.
    """
    stats = get_agent_usage_stats(user["user_id"], page, page_size)
    return RestResponse(data=stats)
