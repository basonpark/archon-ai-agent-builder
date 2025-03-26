from __future__ import annotations as _annotations

import logfire
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import the env_loader module
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from iterations.env_loader import loaded as env_loaded

import asyncio
import json
from typing import List, Dict, Any, Optional, Literal
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import get_env_var
from archon.agent_prompts import prompt_refiner_prompt

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
llm = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

model = AnthropicModel(llm, api_key=api_key) if provider == "Anthropic" else OpenAIModel(llm, base_url=base_url, api_key=api_key)

logfire.configure(send_to_logfire='if-token-present')

prompt_refiner_agent = Agent(
    model,
    system_prompt=prompt_refiner_prompt
)