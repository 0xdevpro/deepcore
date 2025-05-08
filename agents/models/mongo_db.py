from enum import StrEnum
from typing import Optional, List

import pymongo
from pydantic import BaseModel, Field

from agents.common.config import SETTINGS

mongo_client = pymongo.MongoClient(SETTINGS.MONGO_STRING)
mongo_db = mongo_client["deepcore"]

aigc_img_tasks_col = mongo_db["aigc_img_tasks"]
twitter_user_col = mongo_db["twitter_user"]
profiles_col = mongo_db["profiles"]

# Add agent usage statistics and logs collections
agent_usage_stats_col = mongo_db["agent_usage_stats"]
agent_usage_logs_col = mongo_db["agent_usage_logs"]


class TwitterPost(BaseModel):
    content: Optional[str] = ""
    created_at: Optional[str] = ""
    media_image_urls: List[str]
    x_id: Optional[str]
    x_post_url: Optional[str]


class TwitterUser(BaseModel):
    username: str
    user_id: str
    name: str
    description: str = ""
    verified_type: str = "none"
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    listed_count: int = 0
    like_count: int = 0
    media_count: int = 0
    update_at: int = -1
    profile_image_url: str = ""
    recent_posts: List[TwitterPost] = []


def save_twitter_user(user: TwitterUser):
    twitter_user_col.replace_one({"user_id": user.user_id}, user.model_dump(), upsert=True)


def find_by_username(username: str) -> TwitterUser:
    ret = twitter_user_col.find_one({"username": username})
    if ret:
        return TwitterUser(**ret)
    return None


class AigcImgTaskStatus(StrEnum):
    TODO = "TODO"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class AigcImgTask(BaseModel):
    tenant_id: str
    mode: int
    prompt: Optional[str] = ""
    base64_image_list: List[str] = Field(default_factory=list,
                                         description="List of base64 encoded images, use url_to_base64 function to convert them to base64")

    task_id: Optional[str] = ""
    tid: Optional[str] = ""
    timestamp: Optional[int] = Field(default=None, description="Create Timestamp")
    status: Optional[str] = Field(default=None, description="AigcImgTaskStatus")
    result_img_url: Optional[str] = ""

    gen_cost_s: Optional[int] = 0
    gen_timestamp: Optional[int] = 0
    process_msg: List[str] = []


def save_aigc_img_task(task: AigcImgTask):
    aigc_img_tasks_col.replace_one({"task_id": task.task_id}, task.model_dump(), upsert=True)
