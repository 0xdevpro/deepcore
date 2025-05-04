import datetime
import logging
from decimal import Decimal

from bson import Decimal128
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agents.models.mongo_db import profiles_col, agent_usage_stats_col, agent_usage_logs_col
from agents.protocol.schemas import ProfileInfo, DepositInfo, DepositRequest
from agents.services import get_or_create_credentials
from agents.common.config import SETTINGS

logger = logging.getLogger(__name__)


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
    ret.wallet_address = user.get("wallet_address", "")
    # Set master_address from config
    ret.master_address = SETTINGS.MASTER_ADDRESS

    doc = profiles_col.find_one({"tenant_id": tenant_id})
    if doc:
        ret.balance = doc.get("balance", Decimal128("0.0")).to_decimal()
        ret.total_spend = doc.get("total_spend", Decimal128("0.0")).to_decimal()
        ret.total_requests_count = doc.get("total_requests_count", 0)

        deposit_history = doc.get("deposit_history", [])
        for deposit in deposit_history:
            if isinstance(deposit, dict):
                # Convert amount from Decimal128 to Decimal if needed
                if isinstance(deposit.get("amount"), Decimal128):
                    deposit["amount"] = deposit["amount"].to_decimal()
                ret.deposit_history.append(DepositInfo(**deposit))
    return ret


def get_balance(user: dict) -> Decimal:
    """Query the user's balance by tenant_id."""
    tenant_id = user["tenant_id"]
    doc = profiles_col.find_one({"tenant_id": tenant_id})
    if doc:
        return doc.get("balance", Decimal128("0.0")).to_decimal()
    return Decimal("0.0")


def record_agent_usage(agent_id: str, user: dict, price: float, query: str, response: str, agent_name: str = None):
    """
    Update agent usage statistics (user+agent dimension) and insert detailed usage log.
    Store agent_name for easier display and update if changed.
    """
    price = float(price)  # Ensure price is always float
    update_fields = {
        "$inc": {"requests": 1, "cost": price},
        "$set": {"last_used_time": datetime.datetime.utcnow()}
    }
    if agent_name:
        update_fields["$set"]["agent_name"] = agent_name
    agent_usage_stats_col.update_one(
        {"agent_id": agent_id, "user_id": user.get("user_id")},
        update_fields,
        upsert=True
    )
    log_doc = {
        "agent_id": agent_id,
        "user_id": user.get("user_id"),
        "tenant_id": user.get("tenant_id"),
        "request_time": datetime.datetime.utcnow(),
        "cost": price,
        "query": query,
        "response": response
    }
    if agent_name:
        log_doc["agent_name"] = agent_name
    agent_usage_logs_col.insert_one(log_doc)


def get_agent_usage_stats(user_id: str):
    """
    Query agent usage statistics for a given user_id.
    Returns a list of usage stats for all agents used by this user.
    """
    stats = list(agent_usage_stats_col.find({"user_id": user_id}))
    # Convert ObjectId to str
    for stat in stats:
        if "_id" in stat:
            stat["_id"] = str(stat["_id"])
    return stats


async def deposit(tenant_id: str, deposit_info: DepositInfo):
    data = deposit_info.model_dump()
    data.update({"amount": Decimal128(str(data.get("amount", Decimal("0.0"))))})
    profiles_col.update_one(
        {"tenant_id": tenant_id},
        {
            "$push": {"deposit_history": data},
            "$inc": {"balance": data.get("amount")}
        },
        upsert=True
    )


async def bg_check_tx(user: dict, deposit_request: DepositRequest):
    # TODO scan check tx status
    logger.info(f"tx req {deposit_request.model_dump()} user {user}")
    pass
