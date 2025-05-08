import json
import logging
from typing import Dict

from agents.agent.entity.inner.custom_output import CustomOutput
from agents.agent.entity.inner.finish import FinishOutput
from agents.common.config import SETTINGS
from agents.utils.http_client import AsyncHttpClient

logger = logging.getLogger(__name__)

# Base API URL
BASE_URL = SETTINGS.DATA_API_BASE
API_KEY = SETTINGS.DATA_API_KEY

class TokenAnalyzer:
    """Token analysis tool class for cryptocurrency tokens"""
    
    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"x-api-key": self.api_key}
    
    async def search_token(self, query: str) -> Dict:
        """
        Search for cryptocurrency tokens by name or address
        
        Args:
            query: Token address or name
            
        Returns:
            Dict: Search results
        """
        path = "/p/data/crypto_token_search"
        json_data = {"q": query}
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response
    
    async def get_token_history(self, token_address: str, chain: str = "solana") -> Dict:
        """
        Get cryptocurrency token history data
        
        Args:
            token_address: Token address
            chain: Blockchain network, can be 'solana', 'ether', or 'ton', defaults to 'solana'
            
        Returns:
            Dict: Token history data
        """
        path = "/p/data/crypto_token_history"
        json_data = {
            "token": token_address,
            "chain": chain
        }
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response
    
    async def get_token_quote(self, token_address: str, chain: str = "solana") -> Dict:
        """
        Get cryptocurrency token quote information
        
        Args:
            token_address: Token address
            chain: Blockchain network, can be 'solana', 'ether', or 'ton', defaults to 'solana'
            
        Returns:
            Dict: Token quote information
        """
        path = "/p/data/crypto_token_quote"
        json_data = {
            "token": token_address,
            "chain": chain
        }
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response
    
    async def get_token_holders(self, token_address: str, chain: str = "solana") -> Dict:
        """
        Get cryptocurrency token holders information
        
        Args:
            token_address: Token address
            chain: Blockchain network, can be 'solana', 'ether', or 'ton', defaults to 'solana'
            
        Returns:
            Dict: Token holders information
        """
        path = "/p/data/crypto_token_holders"
        json_data = {
            "token": token_address,
            "chain": chain
        }
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response
    
    async def get_token_summary(self, token_address: str, chain: str = "solana") -> Dict:
        """
        Get cryptocurrency token summary information
        
        Args:
            token_address: Token address
            chain: Blockchain network, can be 'solana', 'ether', or 'ton', defaults to 'solana'
            
        Returns:
            Dict: Token summary information
        """
        path = "/p/data/crypto_token_summary"
        json_data = {
            "token": token_address,
            "chain": chain
        }
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response
    
    async def get_token_insights(self, token_address: str, token_symbol: str, chain: str = "sol") -> Dict:
        """
        Get cryptocurrency token insights analysis
        
        Args:
            token_address: Token address
            token_symbol: Token symbol
            chain: Blockchain network, can be 'sol', 'eth', 'bnb', or 'base', defaults to 'sol'
            
        Returns:
            Dict: Token insights analysis
        """
        path = "/p/data/crypto_token_insights"
        json_data = {
            "token_address": token_address,
            "token_symbol": token_symbol,
            "chain": chain
        }
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response
    
    async def search_x_posts(self, query: str, max_results: int = 10, cache_enabled: bool = True) -> Dict:
        """
        Search Twitter (X) posts related to a query
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            cache_enabled: Whether to use cached results
            
        Returns:
            Dict: Twitter search results
        """
        path = "/ps/data/tweets_search"
        body = {
            "query": query,
            "max_results": max_results,
            "cache_enabled": str(cache_enabled).lower()
        }
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=body,
                stream=False
            ):
                return response
    
    # Pro API methods
    async def search_token_pro(self, query: str) -> Dict:
        """
        [Pro] Search for cryptocurrency tokens by name or address
        
        Args:
            query: Token address or name
            
        Returns:
            Dict: Enhanced search results
        """
        path = "/p/data/crypto_pro_token_search"
        json_data = {"q": query}
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response
    
    async def analyze_solana_token_twitter(self, token_address: str, chain: str = "solana") -> Dict:
        """
        Analyze Twitter data related to a cryptocurrency token
        
        Args:
            token_address: Token address
            chain: Blockchain network, can be 'solana', 'ether', or 'ton', defaults to 'solana'
            
        Returns:
            Dict: Twitter analysis results for the token
        """
        path = "/p/data/analyze_solana_token_twitter"
        json_data = {
            "token": token_address,
            "chain": chain
        }
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response
    
    async def get_token_overview_pro(self, token_address: str, chain: str = "solana") -> Dict:
        """
        [Pro] Get comprehensive cryptocurrency token overview
        
        Args:
            token_address: Token address
            chain: Blockchain network, can be 'sol', 'eth', 'bnb', or 'base', defaults to 'solana'
            
        Returns:
            Dict: Comprehensive token overview
        """
        path = "/p/data/crypto_pro_token_overview"
        # Convert 'solana' to 'sol' if needed
        if chain == "solana":
            chain = "sol"
        elif chain == "ether":
            chain = "eth"
            
        json_data = {
            "token_address": token_address,
            "chain": chain
        }
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response
    
    async def get_token_markets_pro(self, token_address: str, chain: str = "solana") -> Dict:
        """
        [Pro] Get cryptocurrency token markets information
        
        Args:
            token_address: Token address
            chain: Blockchain network, can be 'sol', 'eth', 'bnb', or 'base', defaults to 'solana'
            
        Returns:
            Dict: Detailed token markets information
        """
        path = "/p/data/crypto_pro_token_markets"
        # Convert 'solana' to 'sol' if needed
        if chain == "solana":
            chain = "sol"
        elif chain == "ether":
            chain = "eth"
            
        json_data = {
            "token_address": token_address,
            "chain": chain
        }
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response
    
    async def get_token_holders_pro(self, token_address: str, chain: str = "solana") -> Dict:
        """
        [Pro] Get detailed cryptocurrency token holders information
        
        Args:
            token_address: Token address
            chain: Blockchain network, can be 'sol', 'eth', 'bnb', or 'base', defaults to 'solana'
            
        Returns:
            Dict: Detailed token holders information
        """
        path = "/p/data/crypto_pro_token_holders"
        # Convert 'solana' to 'sol' if needed
        if chain == "solana":
            chain = "sol"
        elif chain == "ether":
            chain = "eth"
            
        json_data = {
            "token_address": token_address,
            "chain": chain
        }
        
        async with AsyncHttpClient(headers=self.headers) as client:
            async for response in client.request(
                method="POST",
                base_url=self.base_url,
                path=path,
                json_data=json_data,
                stream=False
            ):
                return response

# Agent tool functions
async def analyze_token(token_query: str, analysis_type: str = "search", chain: str = "solana"):
    """
    Analyze cryptocurrency token tool function
    
    Args:
        token_query (str): Token address or name
        analysis_type (str): Analysis type, can be 'search', 'history', 'quote', 'holders', 'summary', or 'insights'
        chain (str): Blockchain network
        
    Returns:
        Dict: Analysis result
    """
    analyzer = TokenAnalyzer()
    result = None
    
    try:
        if analysis_type == "search":
            result = await analyzer.search_token(token_query)
        elif analysis_type == "history":
            result = await analyzer.get_token_history(token_query, chain)
        elif analysis_type == "quote":
            result = await analyzer.get_token_quote(token_query, chain)
        elif analysis_type == "holders":
            result = await analyzer.get_token_holders(token_query, chain)
        elif analysis_type == "summary":
            result = await analyzer.get_token_summary(token_query, chain)
        elif analysis_type == "insights":
            # For insights analysis, we assume token_query is a JSON string containing token_address and token_symbol
            token_data = json.loads(token_query)
            result = await analyzer.get_token_insights(
                token_data.get("token_address", ""),
                token_data.get("token_symbol", ""),
                chain
            )
        else:
            result = {"error": f"Unsupported analysis type: {analysis_type}"}
    except Exception as e:
        logger.error(f"Token analysis failed: {str(e)}")
        result = {"error": f"Token analysis failed: {str(e)}"}
    
    yield CustomOutput(result, "token_analysis")
    yield FinishOutput()

async def comprehensive_token_analysis(token_query: str):
    """
    Perform comprehensive analysis on a cryptocurrency token, requiring only token address or name
    Yields results for each API call as they complete
    
    Args:
        token_query (str): Token address or name
        
    Returns:
        Multiple yielded results for each API call
    """
    analyzer = TokenAnalyzer()
    
    try:
        # First get token basic information through search API
        search_result = await analyzer.search_token(token_query)
        
        # Check if search result is valid
        if not search_result or not search_result.get("tokens") or len(search_result["tokens"]) == 0:
            raise Exception(f"Token not found: {token_query}")
        
        # Get the first search result as the target token
        token_info = search_result["tokens"][0]
        token_address = token_info.get("token_contract")
        token_symbol = token_info.get("token_symbol")
        chain_name = token_info.get("chain_name", "solana")
        
        # Yield search result
        yield CustomOutput({"type": "token_info", "data": token_info})

        # Request history API
        try:
            history = await analyzer.get_token_history(token_address, chain_name)
            yield CustomOutput({"type": "token_history", "data": history})
        except Exception as e:
            logger.warning(f"Failed to get token history: {str(e)}")

        # Request quote API
        try:
            quote = await analyzer.get_token_quote(token_address, chain_name)
            yield CustomOutput({"type": "token_quote", "data": quote})
        except Exception as e:
            logger.warning(f"Failed to get token quote: {str(e)}")

        # Request holders API
        try:
            holders = await analyzer.get_token_holders(token_address, chain_name)
            yield CustomOutput({"type": "token_holders", "data": holders})
        except Exception as e:
            logger.warning(f"Failed to get token holders: {str(e)}")

        # Request summary API
        try:
            summary = await analyzer.get_token_summary(token_address, chain_name)
            yield CustomOutput({"type": "token_summary", "data": summary})
        except Exception as e:
            logger.warning(f"Failed to get token summary: {str(e)}")

        # Request insights API
        try:
            chain_for_insights = "sol" if chain_name == "solana" else ("eth" if chain_name == "ether" else chain_name)
            insights = await analyzer.get_token_insights(token_address, token_symbol, chain_for_insights)
            yield CustomOutput({"type": "token_insights", "data": insights})
        except Exception as e:
            logger.warning(f"Failed to get token insights: {str(e)}")
            
        # Request X (Twitter) posts related to token
        try:
            # Create search query combining token symbol and name for better results
            # search_query = f"${token_symbol}" if token_symbol else token_info.get("token_name", "")
            x_posts = await analyzer.search_x_posts(token_query, max_results=20)
            yield CustomOutput({"type": "xposts", "text": x_posts})
        except Exception as e:
            logger.warning(f"Failed to get X posts for token: {str(e)}")

    except Exception as e:
        logger.error(f"Comprehensive token analysis failed: {str(e)}")
        raise Exception(f"Comprehensive token analysis failed: {str(e)}")
    
    yield FinishOutput()

async def comprehensive_pro_token_analysis(token_query: str):
    """
    Perform PRO comprehensive analysis on a cryptocurrency token, requiring only token address or name
    Yields enhanced professional results for each API call as they complete
    
    Args:
        token_query (str): Token address or name
        
    Returns:
        Multiple yielded results with enhanced professional data
    """
    analyzer = TokenAnalyzer()
    
    try:
        # First get token basic information through pro search API
        search_result = await analyzer.search_token_pro(token_query)
        
        # Check if search result is valid
        if not search_result or len(search_result) == 0:
            raise Exception(f"Token not found: {token_query}")
        
        # Get the first search result as the target token
        token_info = search_result[0]
        token_address = token_info.get("tokenContractAddress")
        token_symbol = token_info.get("tokenSymbol", "")
        chain_name = token_info.get("chainName", "Solana").lower()
        
        # Ensure chain name is compatible with Pro API
        chain_for_pro = _transfer_chain_name(chain_name)
        
        # Yield pro search result
        yield CustomOutput({"type": "token_info_pro", "data": token_info})

        # Request overview API (Pro)
        try:
            overview = await analyzer.get_token_overview_pro(token_address, chain_for_pro)
            yield CustomOutput({"type": "token_overview_pro", "data": overview})
        except Exception as e:
            logger.warning(f"Failed to get pro token overview: {str(e)}")

        # Request markets API (Pro)
        try:
            markets = await analyzer.get_token_markets_pro(token_address, chain_for_pro)
            yield CustomOutput({"type": "token_markets_pro", "data": markets})
        except Exception as e:
            logger.warning(f"Failed to get pro token markets: {str(e)}")

        # Request holders API (Pro)
        try:
            holders = await analyzer.get_token_holders_pro(token_address, chain_for_pro)
            yield CustomOutput({"type": "token_holders_pro", "data": holders})
        except Exception as e:
            logger.warning(f"Failed to get pro token holders: {str(e)}")

        # Request insights API
        try:
            insights = await analyzer.get_token_insights(token_address, token_symbol, chain_for_pro)
            yield CustomOutput({"type": "token_insights", "data": insights})
        except Exception as e:
            logger.warning(f"Failed to get token insights: {str(e)}")
            
        # Request Solana token Twitter analysis
        try:
            # Use the original chain name for compatibility
            token_twitter_analysis = await analyzer.analyze_solana_token_twitter(token_address, chain=chain_for_pro)
            yield CustomOutput({"type": "token_twitter_analysis", "data": token_twitter_analysis})
        except Exception as e:
            logger.warning(f"Failed to get token Twitter analysis: {str(e)}")

        # Request X (Twitter) posts related to token
        try:
            # Use cashtag format for Twitter search
            # search_query = f"${token_symbol}" if token_symbol else token_info.get("tokenName", "")
            x_posts = await analyzer.search_x_posts(token_query, max_results=30)
            yield CustomOutput({"type": "xposts", "text": x_posts})
        except Exception as e:
            logger.warning(f"Failed to get X posts for token: {str(e)}")


        yield CustomOutput({"type": "tokenin_out", "text": token_address})

    except Exception as e:
        logger.error(f"Pro token analysis failed: {str(e)}")
        raise Exception(f"Pro token analysis failed: {str(e)}")
    
    yield FinishOutput()

def _transfer_chain_name(chain_name):
    if chain_name == "ethereum" or chain_name == "eth":
        return "ether"
    elif chain_name == "BNB Chain".lower():
        return "bnb"
    elif chain_name == "zkSync Era".lower():
        return "zksync"
    elif chain_name == "Avalanche C".lower():
        return "avalanche"

    return chain_name


async def analyze_token_pro(token_query: str, analysis_type: str = "search", chain: str = "solana"):
    """
    Analyze cryptocurrency token with Pro API tools
    
    Args:
        token_query (str): Token address or name
        analysis_type (str): Analysis type, can be 'search', 'overview', 'markets', or 'holders'
        chain (str): Blockchain network
        
    Returns:
        Dict: Enhanced professional analysis result
    """
    analyzer = TokenAnalyzer()
    result = None
    
    # Convert chain format if needed
    chain_for_pro = "sol" if chain == "solana" else ("eth" if chain == "ether" or chain == "ethereum" else chain)
    
    try:
        if analysis_type == "search":
            result = await analyzer.search_token_pro(token_query)
        elif analysis_type == "overview":
            # For non-search operations, we need the token address
            # First search for the token to get its address
            search_result = await analyzer.search_token_pro(token_query)
            if not search_result or len(search_result) == 0:
                raise Exception(f"Token not found: {token_query}")
            
            token_address = search_result[0].get("tokenContractAddress")
            result = await analyzer.get_token_overview_pro(token_address, chain_for_pro)
        elif analysis_type == "markets":
            # First search for the token to get its address
            search_result = await analyzer.search_token_pro(token_query)
            if not search_result or len(search_result) == 0:
                raise Exception(f"Token not found: {token_query}")
            
            token_address = search_result[0].get("tokenContractAddress")
            result = await analyzer.get_token_markets_pro(token_address, chain_for_pro)
        elif analysis_type == "holders":
            # First search for the token to get its address
            search_result = await analyzer.search_token_pro(token_query)
            if not search_result or len(search_result) == 0:
                raise Exception(f"Token not found: {token_query}")
            
            token_address = search_result[0].get("tokenContractAddress")
            result = await analyzer.get_token_holders_pro(token_address, chain_for_pro)
        elif analysis_type == "twitter" or analysis_type == "x":
            # First search for the token to get its symbol
            search_result = await analyzer.search_token_pro(token_query)
            if not search_result or len(search_result) == 0:
                raise Exception(f"Token not found: {token_query}")
            
            token_symbol = search_result[0].get("tokenSymbol")
            search_query = f"${token_symbol}" if token_symbol else search_result[0].get("tokenName", token_query)
            result = await analyzer.search_x_posts(search_query, max_results=30)
        else:
            result = {"error": f"Unsupported Pro analysis type: {analysis_type}"}
    except Exception as e:
        logger.error(f"Pro token analysis failed: {str(e)}")
        result = {"error": f"Pro token analysis failed: {str(e)}"}
    
    yield CustomOutput(result, "token_analysis_pro")
    yield FinishOutput()