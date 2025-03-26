from __future__ import annotations
from typing import Literal, TypedDict, List, Dict, Any, Optional
import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import the env_loader module
sys.path.append(str(Path(__file__).parent.parent.parent))
from iterations.env_loader import loaded as env_loaded

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

import archon_graph

# No need for load_dotenv() since we're using the env_loader module

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

@st.cache_resource
def get_thread_id():
    return str(uuid.uuid4())

thread_id = get_thread_id()

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # First message from user
    if len(st.session_state.messages) == 1:
        async for msg in archon_graph.agentic_flow.astream(
                {"latest_user_message": user_input}, config, stream_mode="custom"
            ):
                yield msg
    # Continue the conversation
    else:
        async for msg in archon_graph.agentic_flow.astream(
            archon_graph.Command(resume=user_input), config, stream_mode="custom"
        ):
            yield msg


async def main():
    st.title("Archon - Agent Builder")
    st.write("Describe to me an AI agent you want to build and I'll code it for you with Pydantic AI.")
    st.write("Example: Build me an AI agent that can search the web with the Brave API.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_type = message["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message["content"])    

    # Chat input for the user
    user_input = st.chat_input("What do you want to build today?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append({"type": "human", "content": user_input})
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            # Run the async generator to fetch responses
            async for chunk in run_agent_with_streaming(user_input):
                response_content += chunk
                # Update the placeholder with the current response content
                message_placeholder.markdown(response_content)
        
        st.session_state.messages.append({"type": "ai", "content": response_content})


if __name__ == "__main__":
    asyncio.run(main())
