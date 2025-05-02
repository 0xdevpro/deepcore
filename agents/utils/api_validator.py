from typing import Dict, List, Optional
from agents.exceptions import CustomAgentException, ErrorCode

class APIValidator:
    """Utility class for validating API data structure"""
    
    @staticmethod
    def validate_api_tool(api_tool: Dict) -> None:
        """
        Validate the structure and required fields of an API tool
        
        Args:
            api_tool: Dictionary containing API tool information
            
        Raises:
            CustomAgentException: If validation fails
        """
        # Check required fields
        required_fields = ['name', 'description', 'path', 'method', 'origin']
        for field in required_fields:
            if not api_tool.get(field):
                raise CustomAgentException(
                    ErrorCode.INVALID_PARAMETERS,
                    f"Missing required field: {field}"
                )
        
        # Validate parameters if they exist
        parameters = api_tool.get('parameters', {})
        if parameters:
            APIValidator._validate_parameters(parameters)
            
        # Validate auth_config if it exists
        auth_config = api_tool.get('auth_config')
        if auth_config:
            APIValidator._validate_auth_config(auth_config)
    
    @staticmethod
    def _validate_auth_config(auth_config: Dict) -> None:
        """
        Validate the structure of authentication configuration
        
        Args:
            auth_config: Dictionary containing authentication configuration
            
        Raises:
            CustomAgentException: If validation fails
        """
        # Check required fields
        required_fields = ['location', 'key', 'value']
        for field in required_fields:
            if not auth_config.get(field):
                raise CustomAgentException(
                    ErrorCode.INVALID_PARAMETERS,
                    f"Missing required field in auth_config: {field}"
                )
        
        # Validate location value
        location = auth_config.get('location')
        if location not in ['header', 'param']:
            raise CustomAgentException(
                ErrorCode.INVALID_PARAMETERS,
                f"Invalid auth_config location: {location}. Must be either 'header' or 'param'"
            )
    
    @staticmethod
    def _validate_parameters(parameters: Dict) -> None:
        """
        Validate the structure of API parameters
        
        Args:
            parameters: Dictionary containing parameter information
            
        Raises:
            CustomAgentException: If validation fails
        """
        param_types = ['header', 'query', 'path', 'body']
        
        for param_type in param_types:
            if param_type in parameters and parameters[param_type]:
                if param_type == 'body':
                    # For body parameters, we just check if it's not None
                    if parameters[param_type] is None:
                        raise CustomAgentException(
                            ErrorCode.INVALID_PARAMETERS,
                            f"Body parameter cannot be None"
                        )
                else:
                    # For other parameter types, validate each parameter
                    APIValidator._validate_parameter_list(parameters[param_type], param_type)
    
    @staticmethod
    def _validate_parameter_list(params: List[Dict], param_type: str) -> None:
        """
        Validate a list of parameters
        
        Args:
            params: List of parameter dictionaries
            param_type: Type of parameters (header, query, path)
            
        Raises:
            CustomAgentException: If validation fails
        """
        for param in params:
            if not param.get('name'):
                raise CustomAgentException(
                    ErrorCode.INVALID_PARAMETERS,
                    f"Missing name in {param_type} parameter"
                )
            if not param.get('type'):
                raise CustomAgentException(
                    ErrorCode.INVALID_PARAMETERS,
                    f"Missing type in {param_type} parameter: {param.get('name')}"
                )
            if not param.get('description'):
                raise CustomAgentException(
                    ErrorCode.INVALID_PARAMETERS,
                    f"Missing description in {param_type} parameter: {param.get('name')}"
                )

if __name__ == '__main__':
    # Test with valid auth_config
    api_tool = {
        'name': 'test_api',
        'description': 'Test API',
        'path': '/test',
        'method': 'GET',
        'origin': 'http://example.com',
        'parameters': {
            'header': [
                {
                    'name': 'Authorization',
                    'type': 'string',
                    'description': 'Bearer token'
                }
            ]
        },
        'auth_config': {
            'location': 'header',
            'key': 'X-API-Key',
            'value': 'your_api_key'
        }
    }

    APIValidator.validate_api_tool(api_tool)
    
    # Test with invalid auth_config
    try:
        invalid_api_tool = {
            'name': 'test_api',
            'description': 'Test API',
            'path': '/test',
            'method': 'GET',
            'origin': 'http://example.com',
            'auth_config': {
                'location': 'invalid',
                'key': 'X-API-Key',
                'value': 'your_api_key'
            }
        }
        APIValidator.validate_api_tool(invalid_api_tool)
    except CustomAgentException as e:
        print(f"Expected error: {str(e)}")