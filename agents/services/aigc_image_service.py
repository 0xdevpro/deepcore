import logging
import time
import uuid

import aiohttp

from agents.common.config import SETTINGS
from agents.common.otel import Otel
from agents.common.s3_client import download_and_upload_image
from agents.models.mongo_db import save_aigc_img_task, AigcImgTask, AigcImgTaskStatus

logger = logging.getLogger(__name__)


async def backgroud_run_aigc_img_task(task: AigcImgTask):
    if not task.task_id:
        task.task_id = str(uuid.uuid4().hex)
    if not task.timestamp:
        task.timestamp = int(time.time())
    if not task.status:
        task.status = AigcImgTaskStatus.TODO
    if not task.tid:
        task.tid = Otel.get_cur_tid()

    save_aigc_img_task(task)

    img_url = await _aigc_gen_img(task)
    if img_url:
        logging.info("aigc_img_task finish")
        task.result_img_url = img_url
        task.process_msg.append(f"aigc_img_task success")
        task.status = AigcImgTaskStatus.DONE
    else:
        task.process_msg.append(f"aigc_img_task failed")
        task.status = AigcImgTaskStatus.FAILED

    task.gen_timestamp = int(time.time())
    task.gen_cost_s = task.gen_timestamp - task.timestamp

    save_aigc_img_task(task)


async def _aigc_gen_img(task: AigcImgTask):
    try:
        content = [
            {"type": "text", "text": task.prompt},
        ]
        if task.base64_image_list:
            for base64_image in task.base64_image_list:
                content.append({"type": "image_url", "image_url": {"url": base64_image}})

        data = {
            "model": "gpt-4o-image-vip",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }

        headers = {
            "Authorization": f"Bearer {SETTINGS.IMGAI_API_KEY}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(SETTINGS.IMGAI_URL, json=data, headers=headers,
                                    timeout=1200) as response:
                if response.status != 200:
                    logging.warning(f'aigc_gen_img: {task.task_id}, http status: {response.status}')
                    return None
                response_text = await response.text()
                logging.info(f'aigc_gen_img: {task.task_id}, response: {response_text}')
                result = await response.json()
                if "error" in result:
                    return None
                if "choices" in result and isinstance(result["choices"], list):
                    for choice in result["choices"]:
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"]["content"]
                            import re
                            matches = re.findall(r"!\[.*?\]\((https?://[^\s]+)\)", content)
                            for image_url in matches:
                                if image_url:
                                    ret_img = await download_and_upload_image(image_url, "deepweb3")
                                    if ret_img:
                                        logging.info(f"aigc_gen_img {task.task_id} successful!")
                                        return ret_img
    except Exception as e:
        logging.error(f"aigc_gen_img {task.task_id} error: {e}", exc_info=True)
    return None
