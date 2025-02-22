"""
FastAPI endpoints for the Hermes AI System.
"""

from fastapi import FastAPI, HTTPException
from uuid import uuid4
from typing import Dict, Any, Optional

from ..core.orchestrator import orchestrator
from ..core.memory import memory_manager
from ..sephirot.tiferet import Tiferet

app = FastAPI(
    title="Hermes API",
    description="Hermes AI System - Tree of Life Architecture",
    version="5.0",
)


@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"status": "active", "system": "Hermes AI"}


@app.post("/session")
async def start_session():
    """Start a new session."""
    session_id = str(uuid4())
    return {"session_id": session_id}


@app.post("/process/{session_id}")
async def process_input(session_id: str, input_text: str):
    """Process input through the system."""
    try:
        result = await orchestrator.orchestrate(session_id, input_text)
        await memory_manager.store(session_id, input_text, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/{session_id}")
async def deep_research(
    session_id: str,
    input_text: str,
    depth: Optional[int] = None,
    breadth: Optional[int] = None,
):
    """Perform deep research."""
    try:
        result = await orchestrator.orchestrate(session_id, input_text, depth, breadth)
        await memory_manager.store(session_id, input_text, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{session_id}")
async def retrieve_stored_memory(session_id: str):
    """Retrieve stored memories."""
    return await memory_manager.retrieve(session_id)


@app.post("/harmonize")
async def harmonize_response(input_text: str):
    """Harmonize input through Tiferet."""
    tiferet = Tiferet()
    return await tiferet.process(None, input_text)
