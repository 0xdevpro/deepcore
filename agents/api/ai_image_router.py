import logging
from typing import Optional

from fastapi import APIRouter, Depends, Body, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from agents.common.error_messages import get_error_message
from agents.common.response import RestResponse
from agents.exceptions import CustomAgentException, ErrorCode
from agents.middleware.auth_middleware import get_optional_current_user
from agents.models.db import get_db
from agents.services.ai_image_service import (
    AIImageService, CreateAIImageTaskDTO, AIImageTaskQueryDTO,
    AITemplateQueryDTO
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ai_image/create_task", summary="Create AI Image Task")
async def create_ai_image_task(
        background_tasks: BackgroundTasks,
        task_info: CreateAIImageTaskDTO = Body(..., description="Task information for creating AI image task"),
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Create a new AI image task
    
    Parameters:
        - task_info: Task information including x_link and template_img_url
        - user: Current user information (for tenant_id)
    
    Returns:
        - Task creation result
    """
    try:
        if not user:
            raise CustomAgentException(
                error_code=ErrorCode.UNAUTHORIZED,
                message="User authentication required"
            )

        tenant_id = user.get("tenant_id")
        if not tenant_id:
            raise CustomAgentException(
                error_code=ErrorCode.INVALID_PARAMETERS,
                message="Tenant ID not found in user information"
            )

        ai_image_service = AIImageService(session)
        await ai_image_service.create_ai_image_task(task_info, tenant_id, background_tasks, user)
        return RestResponse(data=True)
    except CustomAgentException as e:
        logger.error(f"Error creating AI image task: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error creating AI image task: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/ai_image/query_task_list", summary="Query AI Image Task List")
async def query_ai_image_task_list(
        query_params: AIImageTaskQueryDTO = Body(..., description="Query parameters for AI image tasks"),
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Query AI image task list
    
    Parameters:
        - query_params: Query parameters including page and page_size
        - user: Current user information (for tenant_id)
    
    Returns:
        - List of AI image tasks
    """
    try:
        if not user:
            raise CustomAgentException(
                error_code=ErrorCode.UNAUTHORIZED,
                message="User authentication required"
            )

        tenant_id = user.get("tenant_id")
        if not tenant_id:
            raise CustomAgentException(
                error_code=ErrorCode.INVALID_PARAMS,
                message="Tenant ID not found in user information"
            )

        ai_image_service = AIImageService(session)
        result = await ai_image_service.query_ai_image_task_list(query_params, tenant_id)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error querying AI image task list: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error querying AI image task list: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/ai_template/list", summary="Query AI Template List")
async def query_template_list(
        query_params: AITemplateQueryDTO = Body(..., description="Query parameters for AI templates"),
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Query AI template list with pagination
    
    Parameters:
        - query_params: Query parameters including page, page_size and status
        - user: Current user information
    
    Returns:
        - List of templates with basic information
    """
    try:
        if not user:
            raise CustomAgentException(
                error_code=ErrorCode.UNAUTHORIZED,
                message="User authentication required"
            )

        ai_image_service = AIImageService(session)
        result = await ai_image_service.query_template_list(query_params)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error querying template list: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error querying template list: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )
