from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from agents.common.response import RestResponse
from ..common.error_messages import get_error_message
from ..exceptions import ErrorCode
from ..models.db import get_db
from ..services.image_service import ImageService

router = APIRouter()

class ImageGenerateRequest(BaseModel):
    content: str
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = "standard"

@router.post("/api/images/generate", response_model=RestResponse[Dict[str, Any]])
async def generate_image(
    request: ImageGenerateRequest,
    session: AsyncSession = Depends(get_db)
):
    """
    Image Generation API
    Request body format:
    {
        "content": "Image description",
        "size": "1024x1024",  // Optional, default: 1024x1024
        "quality": "standard" // Optional, options: standard/hd, default: standard
    }
    Response format:
    {
        "data": {
            "file_id": "uuid",
            "url": "/api/files/uuid",
            "metadata": {
                "prompt": "original prompt",
                "size": "image size",
                "quality": "image quality"
            }
        }
    }
    """
    try:
        # Validate size parameter
        valid_sizes = ['1024x1024', '1792x1024', '1024x1792']
        if request.size not in valid_sizes:
            return RestResponse(
                code=1,
                msg=f"Invalid size parameter. Available options: {', '.join(valid_sizes)}"
            )
            
        # Validate quality parameter
        valid_qualities = ['standard', 'hd']
        if request.quality not in valid_qualities:
            return RestResponse(
                code=1,
                msg=f"Invalid quality parameter. Available options: {', '.join(valid_qualities)}"
            )
        
        image_service = ImageService(session)
        result = await image_service.generate_image(
            content=request.content,
            size=request.size,
            quality=request.quality
        )
        return RestResponse(data=result)
        
    except Exception as e:
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )
