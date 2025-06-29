from agents.agent.entity.inner.finish import FinishOutput
from agents.agent.entity.inner.wallet_output import WalletOutput


async def sign_with_wallet(unsignedTransaction: str, hitMessage: str = "Please sign the transaction."):
    """
    Sign a transaction with the wallet.
    Args:
        unsignedTransaction (str): The unsigned transaction to be signed.
        hitMessage (str): A friendly prompt automatically generated by the model for signing the transaction based on context. Defaults to "Please sign the transaction."
    Returns:
        str: The signed transaction.
    """
    # return "signed"
    yield WalletOutput({"type": "sign", "unsignedTransaction": unsignedTransaction, "msg": hitMessage})
    yield FinishOutput()

async def get_public_key(hitMessage: str = "Please enter the transaction account", chain_id: str = None):
    """
    Get the public key from the wallet.
    Args:
        hitMessage (str): A friendly prompt automatically generated by the model for getting the public key based on context. Defaults to "Please enter the transaction account".
        chain_id (str): The ID of the Web3 blockchain network.
    Returns:
        str: The public key.
    """
    # return "publicKey"
    yield WalletOutput({"type": "public_key", "msg": hitMessage, "chain_id": chain_id})
    yield FinishOutput()
