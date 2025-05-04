from decimal import Decimal

from bson import Decimal128
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agents.models.mongo_db import profiles_col
from agents.protocol.schemas import ProfileInfo, DepositInfo
from agents.services import get_or_create_credentials


class SpendChangeRequest(BaseModel):
    tenant_id: str
    amount: Decimal
    requests_count: int = Field(default=1)


def spend_balance(request: SpendChangeRequest):
    profiles_col.update_one(
        {"tenant_id": request.tenant_id},
        {
            "$inc": {
                "balance": Decimal128(str(-request.amount)),
                "total_spend": Decimal128(str(request.amount)),
                "total_requests_count": request.requests_count
            }
        },
        upsert=True
    )


async def get_profile_info(user: dict, session: AsyncSession) -> ProfileInfo:
    tenant_id = user["tenant_id"]
    ret = ProfileInfo(tenant_id=tenant_id)

    ret.api_key = (await get_or_create_credentials(user, session)).get("token", None)

    doc = profiles_col.find_one({"tenant_id": tenant_id})
    if doc:
        ret.balance = doc.get("balance", Decimal128("0.0")).to_decimal()
        ret.total_spend = doc.get("total_spend", Decimal128("0.0")).to_decimal()
        ret.total_requests_count = doc.get("total_requests_count", 0)

        deposit_history = doc.get("deposit_history", [])
        for deposit in deposit_history:
            if isinstance(deposit, dict):
                ret.deposit_history.append(DepositInfo(**deposit))
    return ret


async def deposit(tenant_id: str, deposit_info: DepositInfo):
    data = deposit_info.model_dump()
    data.update({"amount": Decimal128(str(data.get("amount", Decimal("0.0"))))})
    profiles_col.update_one(
        {"tenant_id": tenant_id},
        {
            "$push": {"deposit_history": data},
            "$inc": {"balance": data.get("amount")}
        }
    )
