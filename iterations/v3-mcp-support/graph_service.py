import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import the env_loader module
sys.path.append(str(Path(__file__).parent.parent.parent))
from iterations.env_loader import loaded as env_loaded

import json
import time
import httpx
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from archon.archon_graph import agentic_flow
from langgraph.types import Command
from utils.utils import write_to_log

app = FastAPI()

class InvokeRequest(BaseModel):
    message: str
    thread_id: str
    is_first_message: bool = False
    config: Optional[Dict[str, Any]] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}    

@app.post("/invoke")
async def invoke_agent(request: InvokeRequest):
    """Process a message through the agentic flow and return the complete response.

    The agent streams the response but this API endpoint waits for the full output
    before returning so it's a synchronous operation for MCP.
    Another endpoint will be made later to fully stream the response from the API.
    
    Args:
        request: The InvokeRequest containing message and thread info
        
    Returns:
        dict: Contains the complete response from the agent
    """
    try:
        config = request.config or {
            "configurable": {
                "thread_id": request.thread_id
            }
        }

        response = ""
        if request.is_first_message:
            write_to_log(f"Processing first message for thread {request.thread_id}")
            async for msg in agentic_flow.astream(
                {"latest_user_message": request.message}, 
                config,
                stream_mode="custom"
            ):
                response += str(msg)
        else:
            write_to_log(f"Processing continuation for thread {request.thread_id}")
            async for msg in agentic_flow.astream(
                Command(resume=request.message),
                config,
                stream_mode="custom"
            ):
                response += str(msg)

        write_to_log(f"Final response for thread {request.thread_id}: {response}")
        return {"response": response}
        
    except Exception as e:
        write_to_log(f"Error processing message for thread {request.thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)
