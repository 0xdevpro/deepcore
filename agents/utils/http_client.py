import logging
from typing import Dict, Any, AsyncGenerator, Union, List
from typing import Optional

import aiohttp
from aiohttp import ClientTimeout
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class AsyncHttpClient:
    """Asynchronous HTTP client that supports both streaming and non-streaming requests"""
    
    def __init__(
        self,
        base_url: str = "",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Initialize HTTP client
        
        Args:
            base_url: Base URL for API requests
            headers: Request headers
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.headers = headers or {}
        self.timeout = ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def init_session(self):
        """Initialize aiohttp session"""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout
            )

    async def close(self):
        """Close the session"""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_full_url(self, base_url: str, path: str) -> str:
        """Get complete URL by combining base URL and path"""
        if path.startswith(('http://', 'https://')):
            return path
        return f"{base_url}/{path.lstrip('/')}"

    def _apply_auth_config(self, url: str, headers: dict, auth_config: Optional[Dict | List]) -> tuple[str, dict]:
        """Apply authentication configuration to request
        
        Args:
            url: Request URL
            headers: Request headers
            auth_config: Authentication configuration, can be:
                - Single dict: {"location": "header"/"param", "key": "key_name", "value": "key_value"}
                - List of dicts: [{"location": "header", ...}, {"location": "param", ...}]
                
        Returns:
            Tuple of (modified_url, modified_headers)
        """
        if not auth_config:
            return url, headers
        
        # Convert single dict to list for uniform processing
        configs = auth_config if isinstance(auth_config, list) else [auth_config]
        
        for config in configs:
            if config.get('location') == 'header':
                headers[config['key']] = config['value']
            elif config.get('location') == 'param':
                separator = '&' if '?' in url else '?'
                url = f"{url}{separator}{config['key']}={config['value']}"
        
        return url, headers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def request(
        self,
        method: str,
        base_url: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        auth_config: Optional[Dict | List] = None,
        stream: bool = False
    ) -> Union[Dict, str, AsyncGenerator[str, None]]:
        """
        Send HTTP request
        
        Args:
            method: HTTP method
            path: Request path
            params: URL parameters
            json_data: JSON request body
            data: Form data
            headers: Additional request headers
            auth_config: Authentication configuration in format {"location": "header"/"param", "key": "key_name", "value": "key_value"}
            stream: Whether to use streaming response
            
        Returns:
            If stream=False: Response data as dict or string
            If stream=True: Async generator for streaming response
        """
        await self.init_session()
        if not self._session:
            raise RuntimeError("Session not initialized")

        merged_headers = {**self.headers}
        if headers:
            merged_headers.update(headers)

        url = self._get_full_url(base_url, path)
        url, merged_headers = self._apply_auth_config(url, merged_headers, auth_config)
        logger.info(f"Sending HTTP request: {method} {url} params={params} json={json_data} data={data}")
        try:
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                data=data,
                headers=merged_headers
            ) as response:
                if stream:
                    async for data in self._handle_stream_response(response):
                        yield data
                else:
                    yield await self._handle_normal_response(response)
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {str(e)}", exc_info=True)
            raise e

    async def _handle_normal_response(self, response: aiohttp.ClientResponse) -> Union[Dict, str]:
        """Handle normal (non-streaming) response"""
        try:
            response.raise_for_status()
        except aiohttp.ClientResponseError as e:
            # Try to get error message from response body
            content_type = response.headers.get('Content-Type', '')
            try:
                if 'application/json' in content_type:
                    error_body = await response.json()
                    error_msg = str(error_body)
                else:
                    error_body = await response.text()
                    error_msg = error_body
            except Exception:
                error_msg = str(e)
            
            # Create a new exception with the error message from response body
            raise aiohttp.ClientResponseError(
                response.request_info,
                response.history,
                status=response.status,
                message=error_msg
            )
        
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            return await response.json()
        return await response.text()

    async def _handle_stream_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[str, None]:
        """Handle streaming response"""
        async for chunk in response.content.iter_any():
            yield chunk.decode('utf-8')

# Create a default client instance
async_client = AsyncHttpClient()

async def debug_tool_api(
    tool_info: Dict,
    input_params: Dict,
    user_headers: Optional[Dict[str, str]] = None
) -> Union[Dict, str, AsyncGenerator[str, None]]:
    """
    Debug a tool API by sending a request with user-provided parameters
    
    Args:
        tool_info: Tool information including origin, path, method, and auth_config
        input_params: User input parameters for the API call
        user_headers: Additional headers to include in the request
        
    Returns:
        API response which can be a dict, string, or async generator for streaming responses
    """
    method = tool_info.get('method', 'GET')
    origin = tool_info.get('origin', '')
    path = tool_info.get('path', '')
    auth_config = tool_info.get('auth_config')
    is_stream = tool_info.get('is_stream', False)
    
    # Process parameters based on tool parameters structure
    params = {}
    headers = user_headers or {}
    json_data = None
    form_data = None
    
    tool_parameters = tool_info.get('parameters', {})
    
    # Process query parameters
    if 'query' in tool_parameters and isinstance(tool_parameters['query'], list):
        for param in tool_parameters['query']:
            param_name = param.get('name')
            if param_name and param_name in input_params:
                params[param_name] = input_params[param_name]
    
    # Process header parameters
    if 'header' in tool_parameters and isinstance(tool_parameters['header'], list):
        for param in tool_parameters['header']:
            param_name = param.get('name')
            if param_name and param_name in input_params:
                headers[param_name] = input_params[param_name]
    
    # Process body parameters
    if 'body' in tool_parameters and input_params.get('body'):
        json_data = input_params.get('body')
    
    # Create HTTP client
    async with AsyncHttpClient(headers=headers) as client:
        logger.info(f"Debugging tool API: {method} {origin}/{path}")
        logger.info(f"With parameters: {params}, body: {json_data}")
        
        # Use the request method to make the API call
        async for response in client.request(
            method=method,
            base_url=origin,
            path=path,
            params=params,
            json_data=json_data,
            data=form_data,
            auth_config=auth_config,
            stream=is_stream
        ):
            yield response