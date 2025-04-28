import logging
from typing import Dict, Optional

from fastapi import Depends
from langchain_core.language_models import BaseChatModel
from langchain_xai import ChatXAI
from sqlalchemy.ext.asyncio import AsyncSession

from agents.models.db import get_db
from agents.models.entity import ModelInfo
from agents.services.model_service import get_model_with_key

logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory class for managing LLM instances"""
    
    _instances: Dict[int, BaseChatModel] = {}
    
    @classmethod
    async def get_llm(cls, identifier: int | str, user: dict = None, session: AsyncSession = Depends(get_db)) \
            -> Optional[BaseChatModel]:
        """
        Get LLM instance by name or ID
        
        Args:
            identifier: Model ID
            user: Optional user info for authentication
            
        Returns:
            Model instance if found, None otherwise
        """
        # Check cache first
        if identifier in cls._instances:
            return cls._instances[identifier]
            
        try:
            # Try to get model info from database
            model_dto, api_key = await get_model_with_key(identifier, user, session)
            if not model_dto:
                logger.warning(f"Model not found in database: {identifier}")
                return None
            model_info = ModelInfo(**model_dto.model_dump())
            
            # Create new model instance
            model = ChatXAI(
                xai_api_key=api_key,
                xai_api_base=model_info.endpoint,
                model_name=model_info.model_name
            )
                
            # Cache the instance
            cls._instances[identifier] = model
            return model
            
        except Exception as e:
            logger.error(f"Error getting model: {e}", exc_info=True)
            return None
            
    @classmethod
    def clear_cache(cls):
        """Clear all cached model instances"""
        cls._instances.clear() 