import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from agents.common.response import RestResponse
from agents.middleware.auth_middleware import get_current_user
from agents.protocol.schemas import DepositRequest, ProfileInfo
from agents.services.profiles_service import get_profile_info

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/profile/deposit")
async def deposit(tx_info: DepositRequest, background_tasks: BackgroundTasks):
    return RestResponse()


@router.get("/profile/info", response_model=RestResponse[ProfileInfo])
async def info(user: dict = Depends(get_current_user)):
    data = get_profile_info(user["tenant_id"])
    return RestResponse(data=data)
