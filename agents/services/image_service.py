import logging
import os
import httpx
import base64
from typing import Optional, Dict, Any
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from agents.common.config import SETTINGS
from agents.models.db import get_db
from agents.models.models import FileStorage

logger = logging.getLogger(__name__)

class ImageService:
    def __init__(self, session: AsyncSession = Depends(get_db)):
        self.api_key = SETTINGS.OPENAI_API_KEY
        self.api_base = "https://api.openai.com/v1"
        self.db_session = session
        
    def _enhance_prompt(self, content: str) -> str:
        """Enhance user input prompt for better image generation results"""
        enhanced_prompt = f"""
        Create a highly detailed and visually striking image of {content}. 
        The image should feature:
        - Professional quality and artistic composition
        - Rich, vibrant colors and careful attention to lighting
        - Clear focal points and balanced visual elements
        - Photorealistic details and textures where appropriate
        - High attention to detail and visual clarity
        - Balanced composition following the rule of thirds
        - Dramatic lighting and shadows for depth
        """
        return enhanced_prompt.strip()
    
    async def save_image(self, image_data: bytes, prompt: str, size: str) -> str:
        """Save generated image to database"""
        import uuid
        file_uuid = str(uuid.uuid4())
        new_file = FileStorage(
            file_name=f"generated_image_{size}.png",
            file_uuid=file_uuid,
            file_content=image_data,
            size=len(image_data),
            metadata={
                "prompt": prompt,
                "size": size,
                "type": "generated_image"
            }
        )
        self.db_session.add(new_file)
        await self.db_session.commit()
        return file_uuid
    
    async def generate_image(self, content: str, size: str = "1024x1024", quality: str = "standard") -> Dict[str, Any]:
        """
        Generate image using DALL-E 3
        :param content: User's image description
        :param size: Image size, options: "1024x1024", "1792x1024", "1024x1792"
        :param quality: Image quality, options: "standard" or "hd"
        :return: Response containing image data and file ID
        """
        url = f"{self.api_base}/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Enhance prompt
        enhanced_prompt = self._enhance_prompt(content)
        
        payload = {
            "model": "dall-e-3",
            "prompt": enhanced_prompt,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": "b64_json"  # Changed to b64_json
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=httpx.Timeout(120.0))
                response.raise_for_status()
                result = response.json()
                
                # Convert base64 to bytes
                image_data = base64.b64decode(result['data'][0]['b64_json'])
                
                # Save to database
                file_id = await self.save_image(image_data, content, size)
                
                return {
                    "file_id": file_id,
                    "url": f"/api/files/{file_id}",
                    "metadata": {
                        "prompt": content,
                        "size": size,
                        "quality": quality
                    }
                }
                
            except httpx.RequestError as e:
                logger.error(f"Request error: {str(e)}", exc_info=True)
                raise Exception(f"Failed to generate image: {str(e)}") 