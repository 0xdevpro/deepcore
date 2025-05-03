from decimal import Decimal

from bson import Decimal128
from pydantic import BaseModel, Field

from agents.models.mongo_db import wallets_col, WalletInfo


class WalletSpendChangeRequest(BaseModel):
    tenant_id: str
    amount: Decimal
    requests_count: int = Field(default=1)


def spend_wallet_balance(request: WalletSpendChangeRequest):
    wallets_col.update_one(
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


def get_wallet_info(tenant_id: str) -> WalletInfo:
    doc = wallets_col.find_one({"tenant_id": tenant_id})
    if doc:
        return WalletInfo(
            tenant_id=doc["tenant_id"],
            balance=doc.get("balance", Decimal128("0.0")).to_decimal(),
            total_spend=doc.get("total_spend", Decimal128("0.0")).to_decimal(),
            total_requests_count=doc.get("total_requests_count", 0)
        )
    return WalletInfo(tenant_id=tenant_id)
