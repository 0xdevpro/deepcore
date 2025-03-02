import json
import logging

from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.agent.llm.default_llm import openai

gen_aegnt_prompt = """\
Here's a refined version of your prompt that explicitly instructs the model to determine the language based on the user's input:  

---

**You are an expert in generating agent configurations. Your task is to generate an agent configuration based on the user's input while ensuring language consistency.**  

### **Requirements**  
1. Detect the language of the user input.  
2. Generate the agent configuration using the same language as the user input.  

**User Input:**  
{input}  

**Now, generate the agent configuration in the detected language:**
"""

logger = logging.getLogger(__name__)

class AgentConfigurations(BaseModel):
    """Agent configurations"""
    name: str = Field(description="Name of the agent. 4 ~ 8 characters")
    description: str = Field(description="Description of the agent. within 200 characters")
    role_settings: str = Field(
        description="LLM role settings for the agent. 200 ~ 1000 characters, output with markdown format, contains ## Role ## Skills(3～5) ## Limit(3～5)")
    welcome_message: str = Field(description="welcome message for the agent. within 50 characters")


async def gen_agent(input: str):
    structured_model = openai.get_model().with_structured_output(AgentConfigurations)

    prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(gen_aegnt_prompt)])

    chain = prompt | structured_model

    logger.info(f"gen_agent {chain.to_json()}")
    async for chunk in chain.astream({'input': input}):
        if isinstance(chunk, AgentConfigurations):
            yield f'event: message\ndata: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n'
        else:
            yield f'event: message\ndata: {str(chunk)}\n\n'
