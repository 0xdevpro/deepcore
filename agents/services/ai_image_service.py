import json
import logging
from typing import Dict, Any, List, Optional

import httpx
import pymongo
from fastapi import Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.common.config import SETTINGS
from agents.common.http_utils import url_to_base64, fetch_image_as_base64
from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.db import get_db
from agents.models.models import AIImageTemplate
from agents.models.mongo_db import AigcImgTask, aigc_img_tasks_col
from agents.services.aigc_image_service import backgroud_run_aigc_img_task
from agents.services.twitter_service import get_twitter_user_by_username
from agents.services.profiles_service import get_balance, SpendChangeRequest, spend_balance, record_agent_usage
from decimal import Decimal

logger = logging.getLogger(__name__)


class CreateAIImageTaskDTO(BaseModel):
    """
    Data transfer object for creating AI image task
    Mode 1: Template + Custom Prompt + Image
    Mode 2: Template + X Link
    """
    template_id: str
    mode: int  # 1: Custom mode, 2: X Link mode
    # Mode 1 fields
    prompt: Optional[str] = None
    image_url: Optional[str] = None
    # Mode 2 fields
    x_link: Optional[str] = None

    def get_twitter_username(self) -> Optional[str]:
        """
        Extract Twitter username from x_link
        Example: https://x.com/ViewsOfChris -> ViewsOfChris
        """
        if not self.x_link:
            return None
        # Remove trailing slash if exists
        x_link = self.x_link.rstrip('/')
        # Split by slash and get the last part
        return x_link.split('/')[-1]

    def validate_fields(self):
        """Validate fields based on mode"""
        if self.mode == 1:
            if not self.image_url:
                raise CustomAgentException(
                    error_code=ErrorCode.INVALID_PARAMETERS,
                    message="Mode 1 requires image_url"
                )
        elif self.mode == 2:
            if not self.x_link:
                raise CustomAgentException(
                    error_code=ErrorCode.INVALID_PARAMETERS,
                    message="Mode 2 requires x_link"
                )
        else:
            raise CustomAgentException(
                error_code=ErrorCode.INVALID_PARAMETERS,
                message="Invalid mode, must be 1 or 2"
            )


class AIImageTaskQueryDTO(BaseModel):
    """
    Data transfer object for querying AI image tasks
    """
    page: int = 1
    page_size: int = 20
    type: Optional[int] = None  # 1-Custom mode, 2-X Link mode


class AITemplateQueryDTO(BaseModel):
    """Query parameters for AI templates"""
    page: int = 1
    page_size: int = 20
    type: Optional[int] = None  # 1-Custom mode, 2-X Link mode


class AITemplateListResponse(BaseModel):
    """Response model for template list"""
    id: str
    name: str
    preview_url: str
    description: Optional[str]
    type: int


class AIImageService:
    def __init__(self, session: AsyncSession = Depends(get_db)):
        self.db_session = session
        # Configure base URL and API key
        self.api_base = SETTINGS.DATA_API_BASE
        self.api_key = SETTINGS.DATA_API_KEY

    async def create_ai_image_task(self,
                                   task_req: CreateAIImageTaskDTO,
                                   tenant_id: str,
                                   background_tasks: BackgroundTasks,
                                   user: dict = None) -> Dict[str, Any]:
        """
        Create a new AI image task
        
        :param task_req: Task information including template_id and mode-specific fields
        :param tenant_id: Tenant ID from user information
        :param user: User info dict (required for balance check)
        :return: API response data
        """
        # Balance check and deduction
        agent_id = "ai_image_agent"
        agent_name = "AI Image Generator"
        price = 0.002
        if user and SETTINGS.AGENT_BALANCE_CHECK_ENABLED:
            balance = get_balance(user)
            if balance < Decimal(str(price)):
                raise CustomAgentException(
                    error_code=ErrorCode.INSUFFICIENT_BALANCE,
                    message="Insufficient balance, please recharge before using the AI image service."
                )
        # Validate fields based on mode
        task_req.validate_fields()

        # Get template information
        template = await self.get_template(task_req.template_id)
        if not template:
            raise CustomAgentException(
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                message="Template not found"
            )

        task = AigcImgTask(
            tenant_id=tenant_id,
            mode=task_req.mode,
        )

        base64_img_list = []
        if template.template_url:
            base64_img_list.append(await fetch_image_as_base64(template.template_url))
        if task_req.image_url:
            base64_img_list.append(await url_to_base64(task_req.image_url))

        if task_req.mode == 1:
            prompt_tpl = template.prompt
            assert prompt_tpl
            task.prompt = prompt_tpl.format(
                custom= task_req.prompt or "",
            )
        else:  # mode 2
            prompt_tpl = template.prompt
            assert prompt_tpl
            twitter_username = task_req.get_twitter_username()
            twitter_user_info = get_twitter_user_by_username(twitter_username)
            if twitter_user_info:
                twitter_user_info.recent_posts = []
                base64_img_list.append(await url_to_base64(twitter_user_info.profile_image_url))
                json_twitter_user_info = json.dumps(twitter_user_info.model_dump(), ensure_ascii=False)
            else:
                json_twitter_user_info = ""

            task.prompt = prompt_tpl.format(
                json_twitter_user_info=json_twitter_user_info
            )
            task.base64_image_list = base64_img_list

        background_tasks.add_task(backgroud_run_aigc_img_task, task)

        # Deduct balance and record usage after task creation
        if user:
            spend_balance(SpendChangeRequest(
                tenant_id=user["tenant_id"],
                amount=Decimal(str(price)),
                requests_count=1
            ))
            record_agent_usage(
                agent_id=agent_id,
                user=user,
                price=price,
                query=json.dumps(task_req.dict(), ensure_ascii=False),
                response="AI image task created",
                agent_name=agent_name
            )

    async def query_ai_image_task_list(self, query_params: AIImageTaskQueryDTO, tenant_id: str) -> Dict[str, Any]:
        """
        Query AI image task list from MongoDB

        :param query_params: Query parameters including page, page_size and type
        :param tenant_id: Tenant ID from user information
        :return: Task list with pagination information
        """
        try:
            # Build query filter
            query_filter = {"tenant_id": tenant_id}
            if query_params.type is not None:
                query_filter["mode"] = query_params.type

            # Calculate skip and limit for pagination
            skip = (query_params.page - 1) * query_params.page_size
            limit = query_params.page_size

            # Query tasks with pagination
            tasks = list(aigc_img_tasks_col.find(query_filter)
                         .sort("timestamp", pymongo.DESCENDING)
                         .skip(skip)
                         .limit(limit))

            # Get total count for pagination
            total = aigc_img_tasks_col.count_documents(query_filter)

            # Format response
            return {
                "total": total,
                "page": query_params.page,
                "page_size": query_params.page_size,
                "items": [
                    {
                        "task_id": task["task_id"],
                        "mode": task["mode"],
                        "status": task["status"],
                        "prompt": task["prompt"],
                        "result_img_url": task.get("result_img_url", ""),
                        "timestamp": task["timestamp"],
                        "gen_cost_s": task.get("gen_cost_s", 0),
                        "process_msg": task.get("process_msg", [])
                    }
                    for task in tasks
                ]
            }
        except Exception as e:
            logger.error(f"Error querying AI image task list: {e}", exc_info=True)
            raise CustomAgentException(ErrorCode.API_CALL_ERROR, str(e))

    async def query_template_list(self, query_params: AITemplateQueryDTO) -> List[Dict]:
        """
        Query active AI image template list
        
        :param query_params: Query parameters including page, page_size and type
        :return: List of active templates with basic information
        """
        try:
            # Start with base query for active templates
            query = select(AIImageTemplate).where(AIImageTemplate.status == 1)

            # Add type filter if specified
            if query_params.type is not None:
                query = query.where(AIImageTemplate.type == query_params.type)

            # Add pagination
            offset = (query_params.page - 1) * query_params.page_size
            query = query.offset(offset).limit(query_params.page_size)

            # Execute query
            result = await self.db_session.execute(query)
            templates = result.scalars().all()

            # Format response
            return [
                AITemplateListResponse(
                    id=template.id,
                    name=template.name,
                    preview_url=template.preview_url,
                    description=template.description,
                    type=template.type
                ).dict()
                for template in templates
            ]
        except Exception as e:
            logger.error(f"Error querying template list: {e}", exc_info=True)
            raise CustomAgentException(ErrorCode.API_CALL_ERROR, str(e))

    async def get_template(self, template_id: str) -> Optional[AIImageTemplate]:
        """
        Get single template by ID
        
        :param template_id: Template ID to retrieve
        :return: Template information or None if not found
        """
        try:
            result = await self.db_session.execute(
                select(AIImageTemplate).where(
                    AIImageTemplate.id == template_id,
                    AIImageTemplate.status == 1
                )
            )
            template = result.scalars().first()

            if not template:
                return None

            return template
        except Exception as e:
            logger.error(f"Error getting template: {e}", exc_info=True)
            raise CustomAgentException(ErrorCode.API_CALL_ERROR, str(e))
