#!/usr/bin/env python3
"""
Hermes AI System (Final Integrated Version)
---------------------------------------------
This version combines robust session management, persistent memory,
external API integration (DeepSeek‑R1), advanced NLP (via LangChain),
sacred geometry processing, and modular orchestration with Ray.
It also includes immediate improvements for security and resilience:
  - API Key and configuration management via environment variables.
  - Basic input sanitization.
  - Enhanced error handling with try/except blocks.
  - Externalized configuration settings.
  - Safe Gradio UI launch handling.
  
This code is designed to run cross‑platform (Windows, macOS, Linux).
"""

import os
import random
import logging
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, List, Tuple, Union

import ray
import requests
import numpy as np
import matplotlib.pyplot as plt

import chromadb
from fastapi import FastAPI
import gradio as gr

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import threading  # Added for thread-safety

# ----------------------------
# Configuration via Environment Variables
# ----------------------------
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY")
CHROMA_PATH: str = os.environ.get("CHROMA_PATH", "./memory_storage")
TIMEOUT_SECONDS: int = int(os.environ.get("TIMEOUT_SECONDS", "5"))

# ----------------------------
# Initialize Ray for Parallel Execution
# ----------------------------
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HermesCore")

# ----------------------------
# FastAPI Initialization
# ----------------------------
app = FastAPI(title="Hermes API", description="Hermes AI System", version="3.9")

@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Hermes API is live."}

# ----------------------------
# Persistent Memory Storage (ChromaDB)
# ----------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
memory_collection = chroma_client.get_or_create_collection("hermes_memory")
feedback_collection = chroma_client.get_or_create_collection("netzach_feedback")

# In‑memory session storage with thread safety
session_data: Dict[str, List[Tuple[str, Any]]] = {}
session_data_lock = threading.Lock()  # Lock for thread-safe access

def store_memory(session_id: str, user_input: str, response: Any) -> None:
    """Stores conversation history for long-term AI memory."""
    try:
        memory_collection.add(
            documents=[f"{session_id}: {user_input} -> {response}"],
            ids=[str(datetime.now().timestamp())]
        )
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
    with session_data_lock:
        session_data.setdefault(session_id, []).append((user_input, response))

def retrieve_memory(session_id: str) -> List[Tuple[str, Any]]:
    """Retrieves stored memory for context-aware learning within a session (last 5 interactions)."""
    with session_data_lock:
        return session_data.get(session_id, [])[-5:]

# ----------------------------
# DeepSeek‑R1 API Integration (With Timeout Handling & Failover)
# ----------------------------
def query_deepseek(input_text: str) -> str:
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    try:
        response = requests.post(
            "https://deepseek-r1.com/api",
            json={"query": input_text},
            headers=headers,
            timeout=TIMEOUT_SECONDS
        )
        if response.status_code == 200:
            return response.json().get("result", "DeepSeek-R1 provided no useful response.")
        else:
            logger.warning(f"DeepSeek-R1 API returned {response.status_code}.")
            return "DeepSeek-R1 API unavailable, using internal logic."
    except requests.exceptions.Timeout:
        logger.warning("DeepSeek-R1 API timeout. Using internal reasoning.")
        return "API timeout. Using internal reasoning."
    except requests.exceptions.RequestException as e:
        logger.warning(f"DeepSeek-R1 API error: {e}")
        return "API offline, defaulting to internal logic."

# ----------------------------
# Advanced NLP: LangChain LLMChain for Fallback Reasoning
# ----------------------------
def create_llm_chain() -> LLMChain:
    prompt_template = PromptTemplate(
        input_variables=["input_text"],
        template="Provide a logical and creative answer to: {input_text}"
    )
    llm = OpenAI(temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain

llm_chain: LLMChain = create_llm_chain()  # Global chain instance

# ----------------------------
# Keter (Central Orchestrator)
# ----------------------------
@ray.remote
class Keter:
    def __init__(self) -> None:
        self.modules: Dict[str, Any] = {
            "Magí": Magi.remote(),
            "Chesed": Chesed.remote(),
            "Binah": Binah.remote(),
            "Netzach": Netzach.remote()
        }
        logger.info("Keter Orchestrator initialized.")

    def orchestrate(self, session_id: str, input_data: str) -> Dict[str, Any]:
        """Dispatches the input to each module with individual error handling."""
        if not input_data or not input_data.strip():
            return {"error": "Invalid input: Empty query."}
        
        # Retrieve past conversation once to pass to all modules
        past_convo = retrieve_memory(session_id)
        futures: Dict[str, Any] = {}
        for name, module in self.modules.items():
            futures[name] = module.process.remote(session_id, input_data, past_convo)
        
        insights: Dict[str, Any] = {}
        for name, future in futures.items():
            try:
                result = ray.get(future, timeout=TIMEOUT_SECONDS)
                insights[name] = result
            except Exception as e:
                logger.error(f"Module {name} failed with error: {e}")
                insights[name] = f"Error in {name} module."
        store_memory(session_id, input_data, insights)
        return insights

# ----------------------------
# Magí (Creative & Symbolic Intelligence)
# ----------------------------
@ray.remote
class Magi:
    def __init__(self) -> None:
        self.numerology = Numerology.remote()
        self.symbols: Dict[str, str] = {
            "Golden Ratio": "1.618",
            "Fibonacci Sequence": "Mathematical progression",
            "Metatron’s Cube": "Sacred interconnected intelligence",
            "Sri Yantra": "Recursion in sacred geometry",
            "Torus": "Continuous feedback loop"
        }

    def process(self, session_id: str, input_data: str, past_convo: List[Tuple[str, Any]]) -> str:
        numbers = [int(char) for char in input_data if char.isdigit()]
        if numbers:
            try:
                return ray.get(self.numerology.interpret.remote(sum(numbers)), timeout=TIMEOUT_SECONDS)
            except Exception as e:
                logger.error(f"Error in numerology: {e}")
                return "Numerology processing error."
        return random.choice(list(self.symbols.keys()))

# ----------------------------
# Chesed (Emotional Intelligence)
# ----------------------------
@ray.remote
class Chesed:
    def process(self, session_id: str, input_data: str, past_convo: List[Tuple[str, Any]]) -> str:
        if past_convo:
            last_message = past_convo[-1][0]
            return f"I noticed you mentioned '{last_message}'. Let's expand on that with empathy."
        return "Empathetic response generated."

# ----------------------------
# Binah (Logical Reasoning & Decision-Making)
# ----------------------------
@ray.remote
class Binah:
    def process(self, session_id: str, input_data: str, past_convo: List[Tuple[str, Any]]) -> str:
        deepseek_result = query_deepseek(input_data)
        # Use LangChain LLM chain as fallback if DeepSeek is not helpful
        if "no useful response" in deepseek_result.lower() or "offline" in deepseek_result.lower():
            try:
                chain_result = llm_chain.run(input_text=input_data)
                if chain_result and chain_result.strip():
                    return chain_result
            except Exception as e:
                logger.warning(f"LLM chain error: {e}")
            if past_convo:
                return f"Building on our discussion: {past_convo[-1][1]}"
            return "Fallback logical reasoning."
        return deepseek_result

# ----------------------------
# Netzach (Optimization & Self-Learning)
# ----------------------------
@ray.remote
class Netzach:
    def process(self, session_id: str, input_data: str, past_convo: List[Tuple[str, Any]]) -> str:
        past_feedback = feedback_collection.query(query_texts=[input_data], n_results=5)
        # TODO: Validate the structure of past_feedback if needed.
        refined_strategy = (
            past_feedback[0] if isinstance(past_feedback, list) and len(past_feedback) > 0
            else "New optimization strategy."
        )
        feedback_collection.add(
            documents=[f"{input_data} | Suggestion: {refined_strategy}"],
            ids=[str(datetime.now().timestamp())]
        )
        return refined_strategy

# ----------------------------
# Numerology Module
# ----------------------------
@ray.remote
class Numerology:
    def interpret(self, number: int) -> str:
        reduced = sum(int(digit) for digit in str(number))
        return f"Numerology Insight: {reduced} holds deep significance."

# ----------------------------
# Fibonacci Spiral Generator (Sacred Geometry)
# ----------------------------
def generate_fibonacci_spiral() -> str:
    theta = np.linspace(0, 4 * np.pi, 1000)
    r = np.sqrt(theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color="gold", lw=2)
    plt.title("Fibonacci Spiral", fontsize=16)
    plt.axis("off")
    plt.savefig("fibonacci_spiral.png")
    plt.close()
    return "Fibonacci Spiral generated."

# ----------------------------
# Lunar Phase Tracker (Cosmic Synchronization)
# ----------------------------
def get_lunar_phase() -> str:
    phases = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous", 
              "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
    return phases[(datetime.now().day % 8)]

# ----------------------------
# Gradio UI for Local Testing
# ----------------------------
def hermes_ui(session_id: str, input_text: str) -> Dict[str, Any]:
    keter_instance = Keter.remote()
    insights = ray.get(keter_instance.orchestrate.remote(session_id, input_text), timeout=TIMEOUT_SECONDS)
    memory = retrieve_memory(session_id)
    return {"session_id": session_id, "insights": insights, "memory": memory}

# Updated Gradio interface with labeled inputs
iface = gr.Interface(
    fn=hermes_ui,
    inputs=[
        gr.Textbox(label="Session ID", placeholder="Enter Session ID"),
        gr.Textbox(label="Your Query", placeholder="Enter your query here")
    ],
    outputs="json",
    title="Hermes AI System"
)

def safe_launch() -> None:
    """Launches Gradio UI safely to avoid port conflicts."""
    try:
        iface.launch()
    except Exception as e:
        logger.warning(f"Gradio UI launch error: {e}")

# Launch Gradio in a separate thread to avoid blocking the FastAPI server.
gradio_thread = threading.Thread(target=safe_launch, daemon=True)
gradio_thread.start()

# ----------------------------
# API Endpoints
# ----------------------------
@app.post("/start_session")
def start_session() -> Dict[str, str]:
    """Creates a new session for conversation tracking."""
    session_id = str(uuid4())
    with session_data_lock:
        session_data[session_id] = []
    return {"session_id": session_id}

@app.post("/process")
def process_input(session_id: str, input_text: str) -> Dict[str, Any]:
    """Processes input within a session for contextual responses."""
    if not input_text or not input_text.strip():
        return {"error": "Input text cannot be empty.", "session_id": session_id}
    keter = Keter.remote()
    try:
        response = ray.get(keter.orchestrate.remote(session_id, input_text), timeout=TIMEOUT_SECONDS)
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        response = {"error": "Processing error occurred."}
    store_memory(session_id, input_text, response)
    return {"response": response, "session_id": session_id, "lunar_phase": get_lunar_phase()}

@app.get("/retrieve_memory")
def retrieve_stored_memory(session_id: str) -> Dict[str, Any]:
    """Retrieves conversation history for an ongoing session."""
    return {"memory": retrieve_memory(session_id)}

# ----------------------------
# Execution
# ----------------------------
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.warning(f"API launch error: {e}")
