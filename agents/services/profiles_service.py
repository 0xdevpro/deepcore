import asyncio
import datetime
import logging
from decimal import Decimal

from bson import Decimal128
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agents.common.config import SETTINGS
from agents.common.solana_client import solana_get_transaction
from agents.models.mongo_db import profiles_col, agent_usage_stats_col, agent_usage_logs_col
from agents.protocol.schemas import ProfileInfo, DepositInfo, DepositRequest
from agents.services import get_or_create_credentials

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
        deposit_history = sorted(
            deposit_history,
            key=lambda x: x.get("transaction_ts", 0),
            reverse=True
        )
        for item in deposit_history:
            if isinstance(item, dict):
                if item.get("tx_hash", None):
                    # Convert amount from Decimal128 to Decimal if needed
                    if isinstance(item.get("amount"), Decimal128):
                        item["amount"] = item["amount"].to_decimal()
                    ret.deposit_history.append(DepositInfo(**item))
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


def get_agent_usage_stats(user_id: str, page: int = 1, page_size: int = 10):
    """
    Query agent usage statistics for a given user_id, with pagination.
    Returns a dict with total, page, page_size, items.
    """
    skip = (page - 1) * page_size
    cursor = agent_usage_stats_col.find({"user_id": user_id}).skip(skip).limit(page_size)
    stats = list(cursor)
    total = agent_usage_stats_col.count_documents({"user_id": user_id})
    for stat in stats:
        if "_id" in stat:
            stat["_id"] = str(stat["_id"])
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": stats
    }


async def deposit(tenant_id: str, deposit_info: DepositInfo):
    if deposit_info.tx_hash and len(deposit_info.tx_hash) > 6:
        pre = profiles_col.find_one({"tenant_id": tenant_id})
        pre_deposit_history = pre.get("deposit_history", [])
        for item in pre_deposit_history:
            if isinstance(item, dict):
                if deposit_info.tx_hash == item.get("tx_hash", ""):
                    logger.error(f"BAD tx hash repeat!!! {deposit_info.tx_hash}")
                    return

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
    logger.info(f"tx req {deposit_request.model_dump()} user {user}")
    if SETTINGS.MASTER_ADDRESS != deposit_request.to_wallet:
        logger.error(f"BAD master address {deposit_request.to_wallet}")
        return

    for i in range(60 * 10):
        try:
            tx = solana_get_transaction(deposit_request.tx_hash)
            account_keys = tx.value.transaction.transaction.message.account_keys or None
            log_messages = tx.value.transaction.meta.log_messages or []
            if (
                    len(account_keys) != 4
                    or str(account_keys[0]) != deposit_request.from_wallet
                    or str(account_keys[1]) != deposit_request.to_wallet
                    # sol program
                    or str(account_keys[2]) != "11111111111111111111111111111111"
                    # solana trans program
                    or str(account_keys[3]) != "ComputeBudget111111111111111111111111111111"
            ):
                logger.error(f"BAD tx account !!! {account_keys}")
                break
            min_expected = int(Decimal(deposit_request.amount) * Decimal(1_000_000_000))
            delta = tx.value.transaction.meta.post_balances[1] - tx.value.transaction.meta.pre_balances[1]
            if delta < min_expected:
                logger.error(f"BAD tx amount !!! delta:{delta} min_expected:{min_expected}")
                break
            if (
                    not tx.value.transaction.meta.err
                    and "Program 11111111111111111111111111111111 success" in log_messages
            ):
                eposit_info = DepositInfo(**deposit_request.model_dump())
                eposit_info.status = "PAID"
                await deposit(user["tenant_id"], eposit_info)
                return

        except Exception as e:
            logger.info(f"tx req {deposit_request.model_dump()} error {e}")
        finally:
            await asyncio.sleep(1)

    eposit_info = DepositInfo(**deposit_request.model_dump())
    eposit_info.status = "FAIL"
    eposit_info.amount = Decimal("0.0")
    await deposit(user["tenant_id"], eposit_info)
