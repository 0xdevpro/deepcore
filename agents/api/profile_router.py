import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from agents.common.response import RestResponse
from agents.middleware.auth_middleware import get_current_user
from agents.models.db import get_db
from agents.protocol.schemas import DepositRequest, ProfileInfo
from agents.services.profiles_service import get_profile_info, get_agent_usage_stats

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/profile/deposit")
async def deposit(tx_info: DepositRequest, background_tasks: BackgroundTasks):
    return RestResponse()


@router.get("/profile/info", response_model=RestResponse[ProfileInfo])
async def info(user: dict = Depends(get_current_user),
               session: AsyncSession = Depends(get_db)):
    data = await get_profile_info(user, session)
    return RestResponse(data=data)


@router.get("/profile/agent_usage_stats")
async def agent_usage_stats(user: dict = Depends(get_current_user)):
    stats = get_agent_usage_stats(user["user_id"])
    return RestResponse(data=stats)
