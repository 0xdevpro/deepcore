import logging

from solana.rpc.api import Client
from solders.solders import Signature, GetTransactionResp

from agents.common.config import SETTINGS

logger = logging.getLogger(__name__)

solana_client = Client(SETTINGS.SOLANA_RPC_URL)


def solana_get_transaction(tx_hash: str) -> GetTransactionResp | None:
    try:
        return solana_client.get_transaction(Signature.from_string(tx_hash),
                                             commitment="confirmed",
                                             max_supported_transaction_version=0)
    except Exception as e:
        logger.error(f"Failed to get transaction: {e}")
    return None
