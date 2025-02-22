#!/usr/bin/env python3
"""
Hermes AI System – Enhanced Version with Async Endpoints, Observability,
Circuit Breakers, Graph Memory Placeholder, Self‑Optimization, and Multi‑Agent Coordination.

This version integrates:
  - Asynchronous FastAPI endpoints for non‑blocking operation.
  - OpenTelemetry for tracing and observability (exporters to be configured later).
  - Circuit breaker wrappers for external API calls.
  - Graph‑based long‑term memory stubs (Yesod‑like).
  - A dummy reinforcement learning (RL) agent for self‑optimization (Netzach).
  - A multi‑agent negotiation stub (Da’at) for future dynamic module coordination.
  - A local‑only LLM mode flag for true autonomy.

All data is processed securely and (if configured) can use external services while preserving privacy.
"""

import asyncio
import logging
# ============================ Standard & Third‑Party Imports ============================
import os
import random
import threading
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple
from uuid import uuid4

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import ray
import requests
from fastapi import FastAPI, HTTPException
# ----- OpenTelemetry Imports -----
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (BatchSpanProcessor,
                                            ConsoleSpanExporter)
from pydantic_settings import BaseSettings

from hermes.core.config import settings

if settings.enable_gpu:
    # GPU-specific logic here
    pass

# -------------------- Configuration Management --------------------
class Settings(BaseSettings):
    openai_api_key: str = ""
    firecrawl_api_key: str = "fc-abced1f1cf4949339b826851c9d9d1a5"

    allow_remote: bool = True
    allow_openai: bool = False
    allow_external: bool = True
    local_only_llm: bool = True

    chroma_remote: bool = False
    chroma_url: str = "https://your-chroma-cloud-instance.com"
    chroma_path: str = "./memory_storage"

    timeout_seconds: int = 5
    research_depth: int = 2
    research_breadth: int = 3

    o3_model: str = "o3-mini-high"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# -------------------- Initialize Ray --------------------
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

# -------------------- Logging Configuration --------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HermesCore")

# -------------------- OpenTelemetry Setup --------------------
resource = Resource(attributes={"service.name": "HermesAI"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)
# (For production, configure exporters to send data to Prometheus, Loki, Tempo, Grafana, etc.)


# -------------------- Circuit Breaker Decorator --------------------
def circuit_breaker(fallback: Callable):
    """A simple circuit breaker decorator. In production, consider a library like pybreaker."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Circuit breaker activated for {func.__name__}: {e}")
                return fallback(*args, **kwargs)

        return wrapper

    return decorator


# -------------------- Persistent Memory Storage (ChromaDB) --------------------
if settings.chroma_remote:
    chroma_client = chromadb.PersistentClient(url=settings.chroma_url)
else:
    chroma_client = chromadb.PersistentClient(path=settings.chroma_path)

memory_collection = chroma_client.get_or_create_collection("hermes_memory")
feedback_collection = chroma_client.get_or_create_collection("netzach_feedback")

session_data: Dict[str, List[Tuple[str, Any]]] = {}
session_data_lock = threading.Lock()


def store_memory(session_id: str, user_input: str, response: Any) -> None:
    """Stores conversation history and research reports in both ChromaDB and in‑memory."""
    with tracer.start_as_current_span("store_memory"):
        try:
            memory_collection.add(
                documents=[f"{session_id}: {user_input} -> {response}"],
                ids=[str(datetime.now().timestamp())],
            )
        except Exception as e:
            logger.error(f"Error storing memory in ChromaDB: {e}")
        with session_data_lock:
            session_data.setdefault(session_id, []).append((user_input, response))


def retrieve_memory(session_id: str) -> List[Tuple[str, Any]]:
    """Retrieves the last 5 interactions for a given session."""
    with session_data_lock:
        return session_data.get(session_id, [])[-5:]


# -------------------- Graph Memory (Yesod‑like) Placeholder --------------------
graph_memory: Dict[str, List[Dict[str, Any]]] = {}
graph_memory_lock = threading.Lock()


def store_graph_memory(session_id: str, node: Dict[str, Any]) -> None:
    with graph_memory_lock:
        graph_memory.setdefault(session_id, []).append(node)
        logger.info(f"Graph memory updated for session {session_id}")


def retrieve_graph_memory(session_id: str) -> List[Dict[str, Any]]:
    with graph_memory_lock:
        return graph_memory.get(session_id, [])


# -------------------- External API Integrations with Circuit Breakers --------------------
@circuit_breaker(lambda query: "External calls disabled; using internal logic.")
def query_firecrawl(query: str) -> str:
    api_key = settings.firecrawl_api_key
    if not api_key:
        return "Firecrawl API key missing; using internal logic."
    headers = {"Authorization": f"Bearer {api_key}"}
    url = "https://firecrawl.com/api/v1/search"  # Updated endpoint
    payload = {"query": query}
    response = requests.post(
        url, json=payload, headers=headers, timeout=settings.timeout_seconds
    )
    if response.status_code == 200:
        return response.json().get("result", "No result returned.")
    else:
        logger.warning(f"Firecrawl API returned status {response.status_code}")
        return "Firecrawl API error."


@circuit_breaker(lambda prompt, model=None: "OpenAI API call failed.")
def query_openai(prompt: str, model: str = None) -> str:
    import openai

    openai.api_key = settings.openai_api_key
    use_model = model if model else settings.o3_model
    response = openai.ChatCompletion.create(
        model=use_model,
        messages=[{"role": "user", "content": prompt}],
        timeout=settings.timeout_seconds,
    )
    return response.choices[0].message.content


@circuit_breaker(
    lambda input_text: "DeepSeek‑R1 API request failed; using internal logic."
)
def query_deepseek(input_text: str) -> str:
    headers = {"Authorization": f"Bearer {settings.deepseek_api_key}"}
    response = requests.post(
        "https://api.deepseek.com/v1",
        json={"query": input_text, "model": "deepseek-reasoner"},
        headers=headers,
        timeout=settings.timeout_seconds,
    )
    if response.status_code == 200:
        return response.json().get("result", "DeepSeek‑R1 returned no useful response.")
    else:
        logger.warning(f"DeepSeek‑R1 API returned {response.status_code}.")
        return "DeepSeek‑R1 API unavailable, using internal logic."


# -------------------- Local-Only LLM Mode and Dynamic LLM Switching --------------------
import ollama


class OllamaWrapper:
    def __init__(self, model: str):
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]


def get_llm():
    test_message = "ping"
    if settings.local_only_llm:
        logger.info("Using local-only LLM mode.")
        try:
            test_response = ollama.chat(
                model="mistral", messages=[{"role": "user", "content": test_message}]
            )
            if "message" in test_response and "content" in test_response["message"]:
                return OllamaWrapper(model="mistral")
        except Exception as e:
            logger.warning(f"Local mistral via Ollama not available: {e}")
    else:
        try:
            test_response = ollama.chat(
                model="deepseek-r1",
                messages=[{"role": "user", "content": test_message}],
            )
            if "message" in test_response and "content" in test_response["message"]:
                logger.info("Using DeepSeek‑R1 via Ollama as primary LLM.")
                return OllamaWrapper(model="deepseek-r1")
        except Exception as e:
            logger.warning(f"DeepSeek‑R1 via Ollama not available: {e}")
        try:
            test_response = ollama.chat(
                model="mistral", messages=[{"role": "user", "content": test_message}]
            )
            if "message" in test_response and "content" in test_response["message"]:
                logger.info("Using Mistral via Ollama as fallback LLM.")
                return OllamaWrapper(model="mistral")
        except Exception as e:
            logger.warning(f"Mistral via Ollama not available: {e}")
        if settings.allow_openai and settings.openai_api_key:
            logger.info("Using OpenAI GPT‑4/o3‑mini‑high (remote) as fallback LLM.")
            return lambda prompt: query_openai(prompt, model=settings.o3_model)
    logger.error("No LLM available.")
    return lambda prompt: "No LLM available at the moment."


llm = get_llm()

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

prompt_template = PromptTemplate(
    input_variables=["query"],
    template="You are Hermes, an advanced AI. Answer: {query}",
)


def llm_run(query: str) -> str:
    return llm(query)


llm_chain = prompt_template | RunnableLambda(llm_run)


# -------------------- Dummy RL Agent for Self‑Optimization (Netzach) --------------------
class RLAgent:
    def __init__(self):
        self.logic_weight = 0.5  # 0.0 = pure logic, 1.0 = pure creativity

    def update(self, feedback: Dict[str, Any]):
        self.logic_weight = max(
            0.3, min(0.7, self.logic_weight + random.uniform(-0.05, 0.05))
        )
        logger.info(f"RLAgent updated logic weight to {self.logic_weight}")


rl_agent = RLAgent()


# -------------------- Multi‑Agent Negotiation Stub (Da’at) --------------------
class Daat:
    @staticmethod
    def negotiate(responses: Dict[str, str]) -> str:
        best_response = max(responses.values(), key=lambda x: len(x.split()))
        return best_response


# -------------------- Ray Remote Module Definitions --------------------
@ray.remote
class Apre:
    def process(
        self, session_id: str, input_data: str, past_convo: List[Tuple[str, Any]]
    ) -> str:
        with tracer.start_as_current_span("Apre.process"):
            result = query_deepseek(input_data)
            if any(
                x in result.lower() for x in ["no useful response", "offline", "failed"]
            ):
                try:
                    chain_result = llm_chain.invoke(query=input_data)
                    if chain_result and chain_result.strip():
                        return chain_result
                except Exception as e:
                    logger.warning(f"LLM chain error in Apré: {e}")
                if past_convo:
                    return f"Building on our discussion: {past_convo[-1][1]}"
                return "Fallback logical reasoning."
            return result


@ray.remote
class Magi:
    def __init__(self) -> None:
        self.symbols = {
            "Golden Ratio": "1.618",
            "Fibonacci Sequence": "Mathematical progression",
            "Metatron’s Cube": "Sacred interconnected intelligence",
            "Sri Yantra": "Recursion in sacred geometry",
            "Torus": "Continuous feedback loop",
        }

    def process(
        self, session_id: str, input_data: str, past_convo: List[Tuple[str, Any]]
    ) -> str:
        with tracer.start_as_current_span("Magi.process"):
            numbers = [int(char) for char in input_data if char.isdigit()]
            numerology_insight = ""
            if numbers:
                try:
                    numerology_insight = (
                        f"Numerology Insight: {sum(numbers)} holds deep significance."
                    )
                except Exception as e:
                    logger.error(f"Error in numerology processing: {e}")
                    numerology_insight = "Numerology processing error."
            past_feedback = feedback_collection.query(
                query_texts=[input_data], n_results=5
            )
            optimization_suggestion = (
                past_feedback[0]
                if isinstance(past_feedback, list) and past_feedback
                else "New optimization strategy."
            )
            feedback_collection.add(
                documents=[f"{input_data} | Suggestion: {optimization_suggestion}"],
                ids=[str(datetime.now().timestamp())],
            )
            if numerology_insight:
                combined = f"{numerology_insight} | {optimization_suggestion}"
            else:
                creative_symbol = random.choice(list(self.symbols.keys()))
                combined = f"{creative_symbol} | {optimization_suggestion}"
            return combined


@ray.remote
class Chesed:
    def process(
        self, session_id: str, input_data: str, past_convo: List[Tuple[str, Any]]
    ) -> str:
        with tracer.start_as_current_span("Chesed.process"):
            if past_convo:
                last_message = past_convo[-1][0]
                return f"I noticed you mentioned '{last_message}'. Let's expand on that with empathy."
            return "Empathetic response generated."


@ray.remote
class Chochmah:
    def process(
        self, session_id: str, input_data: str, depth: int, breadth: int
    ) -> str:
        with tracer.start_as_current_span("Chochmah.process"):
            logger.info(
                f"[Chochmah] Starting deep research for session {session_id} with depth {depth} and breadth {breadth}."
            )
            initial_response = llm_chain.invoke(query=input_data)
            learnings = [initial_response.strip()]
            logger.info("[Chochmah] Initial research response obtained.")
            current_depth = depth
            while current_depth > 0:
                followup_prompt = (
                    "Based on the following insights, generate a follow-up research question to explore further:\n"
                    + "\n".join(learnings)
                )
                followup_query = llm(followup_prompt)
                logger.info(f"[Chochmah] Generated follow-up query: {followup_query}")
                new_response = llm_chain.invoke(query=followup_query)
                if new_response and new_response.strip():
                    learnings.append(new_response.strip())
                else:
                    learnings.append("No additional insights.")
                current_depth -= 1
            report_lines = [
                "# Deep Research Report (Chochmah)",
                f"**Session ID:** {session_id}",
                f"**Initial Query:** {input_data}",
                "## Findings:",
            ]
            for idx, item in enumerate(learnings, 1):
                report_lines.append(f"### Insight {idx}")
                report_lines.append(item)
                report_lines.append("")
            report_lines.append("## End of Report")
            report = "\n".join(report_lines)
            logger.info("[Chochmah] Deep research completed and report generated.")
            store_memory(session_id, f"Deep Research (Chochmah): {input_data}", report)
            return report


class Tiferet:
    """
    Tiferet acts as a harmonization filter between Apré (logic) and Magí (creativity),
    ensuring a balanced output that blends structured reasoning with intuitive insight.
    """

    def __init__(self):
        self.apre = Apre()  # Local instance for internal blending.
        self.magi = Magi()
        self.default_weight = rl_agent.logic_weight  # Dynamic weight from RLAgent

    def process(self, session_id: str, input_data: str) -> str:
        with tracer.start_as_current_span("Tiferet.process"):
            try:
                logic_response = ray.get(
                    self.apre.process.remote(session_id, input_data, []),
                    timeout=settings.timeout_seconds,
                )
                creative_response = ray.get(
                    self.magi.process.remote(session_id, input_data, []),
                    timeout=settings.timeout_seconds,
                )
            except Exception as e:
                logger.error(f"Error in Tiferet module: {e}")
                return "Harmonization error occurred."
            self.default_weight = rl_agent.logic_weight
            balance_factor = self._calculate_harmony_weight(input_data)
            return self._harmonize_output(
                logic_response, creative_response, balance_factor
            )

    def _calculate_harmony_weight(self, input_text: str) -> float:
        words = input_text.split()
        word_count = len(words)
        complexity_factor = min(word_count / 50.0, 1.0)
        dynamic_weight = (
            self.default_weight * (1 - complexity_factor) + complexity_factor * 0.75
        )
        return dynamic_weight

    def _harmonize_output(self, logic: str, creative: str, weight: float) -> str:
        if weight < 0.3:
            return f"[Structured Insight] {logic}"
        elif weight > 0.7:
            return f"[Creative Vision] {creative}"
        else:
            logic_words = logic.split()
            creative_words = creative.split()
            num_logic = int(len(logic_words) * (1 - weight))
            num_creative = int(len(creative_words) * weight)
            blended = " ".join(logic_words[:num_logic] + creative_words[:num_creative])
            return f"[Balanced Thought] {blended}"


@ray.remote
class TiferetRemote(Tiferet):
    pass


@ray.remote
class Keter:
    def __init__(self) -> None:
        self.modules: Dict[str, Any] = {
            "Apré": Apre.remote(),
            "Magí": Magi.remote(),
            "Chesed": Chesed.remote(),
            "Tiferet": TiferetRemote.remote(),
            "Chochmah": Chochmah.remote(),
        }
        logger.info("Keter Orchestrator initialized (local‑only).")

    def orchestrate(
        self, session_id: str, input_data: str, depth: int = None, breadth: int = None
    ) -> Dict[str, Any]:
        with tracer.start_as_current_span("Keter.orchestrate"):
            if not input_data or not input_data.strip():
                return {"error": "Invalid input: Empty query."}
            past_convo = retrieve_memory(session_id)
            depth = depth if depth is not None else settings.research_depth
            breadth = breadth if breadth is not None else settings.research_breadth

            insights: Dict[str, Any] = {}

            # Run core modules: Apré, Magí, and Chesed.
            modules_to_run = {
                "Apré": self.modules["Apré"],
                "Magí": self.modules["Magí"],
                "Chesed": self.modules["Chesed"],
            }
            futures = {
                name: module.process.remote(session_id, input_data, past_convo)
                for name, module in modules_to_run.items()
            }
            for name, future in futures.items():
                try:
                    result = ray.get(future, timeout=settings.timeout_seconds)
                    insights[name] = result
                except Exception as e:
                    logger.error(f"Module {name} failed with error: {e}")
                    insights[name] = f"Error in {name} module."

            try:
                harmonized_response = ray.get(
                    self.modules["Tiferet"].process.remote(session_id, input_data),
                    timeout=settings.timeout_seconds,
                )
                insights["Tiferet"] = harmonized_response
            except Exception as e:
                logger.error(f"Tiferet module failed: {e}")
                insights["Tiferet"] = "Harmonization error occurred."

            try:
                research_result = ray.get(
                    self.modules["Chochmah"].process.remote(
                        session_id, input_data, depth, breadth
                    ),
                    timeout=(settings.timeout_seconds * (depth + 1)),
                )
                insights["Chochmah"] = research_result
            except Exception as e:
                logger.error(f"Chochmah module failed: {e}")
                insights["Chochmah"] = "Deep research processing error occurred."

            final_response = Daat.negotiate(insights)
            store_memory(session_id, input_data, insights)
            store_graph_memory(session_id, {"input": input_data, "insights": insights})
            return {"insights": insights, "final": final_response}


# -------------------- Additional Creative & Utility Features --------------------
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


def get_lunar_phase() -> str:
    phases = [
        "New Moon",
        "Waxing Crescent",
        "First Quarter",
        "Waxing Gibbous",
        "Full Moon",
        "Waning Gibbous",
        "Last Quarter",
        "Waning Crescent",
    ]
    return phases[(datetime.now().day % 8)]


# -------------------- FastAPI Endpoints (Async) --------------------
app = FastAPI(title="Hermes API", description="Hermes AI System", version="5.0")


@app.get("/")
async def read_root() -> Dict[str, str]:
    with tracer.start_as_current_span("read_root"):
        return {"message": "Hermes API is live."}


@app.post("/start_session")
async def start_session() -> Dict[str, str]:
    with tracer.start_as_current_span("start_session"):
        session_id = str(uuid4())
        with session_data_lock:
            session_data[session_id] = []
        return {"session_id": session_id}


@app.post("/process")
async def process_input(session_id: str, input_text: str) -> Dict[str, Any]:
    with tracer.start_as_current_span("process_input"):
        if not input_text or not input_text.strip():
            return {"error": "Input text cannot be empty.", "session_id": session_id}
        keter = Keter.remote()
        try:
            loop = asyncio.get_running_loop()
            response_future = loop.run_in_executor(
                None, ray.get, keter.orchestrate.remote(session_id, input_text)
            )
            response = await asyncio.wait_for(
                response_future, timeout=settings.timeout_seconds
            )
        except Exception as e:
            logger.error(f"Error processing input in /process: {e}")
            response = {"error": "Processing error occurred."}
        store_memory(session_id, input_text, response)
        return {
            "response": response,
            "session_id": session_id,
            "lunar_phase": get_lunar_phase(),
        }


@app.post("/deep_research")
async def deep_research(
    session_id: str, input_text: str, depth: int = None, breadth: int = None
) -> Dict[str, Any]:
    with tracer.start_as_current_span("deep_research"):
        if not input_text or not input_text.strip():
            return {"error": "Input text cannot be empty.", "session_id": session_id}
        depth = depth if depth is not None else settings.research_depth
        breadth = breadth if breadth is not None else settings.research_breadth
        chochmah_actor = Chochmah.remote()
        try:
            loop = asyncio.get_running_loop()
            report_future = loop.run_in_executor(
                None,
                ray.get,
                chochmah_actor.process.remote(session_id, input_text, depth, breadth),
            )
            report = await asyncio.wait_for(
                report_future, timeout=(settings.timeout_seconds * (depth + 1))
            )
        except Exception as e:
            logger.error(f"Error in deep research endpoint: {e}")
            report = "Deep research processing error occurred."
        store_memory(session_id, f"Deep Research (Chochmah): {input_text}", report)
        return {"report": report, "session_id": session_id}


@app.get("/retrieve_memory")
async def retrieve_stored_memory(session_id: str) -> Dict[str, Any]:
    with tracer.start_as_current_span("retrieve_memory"):
        return {"memory": retrieve_memory(session_id)}


@app.post("/harmonize")
async def harmonize_response(input_text: str) -> Dict[str, str]:
    with tracer.start_as_current_span("harmonize_response"):
        tiferet_instance = Tiferet()  # Local instance for testing.
        output = tiferet_instance.process("test_session", input_text)
        return {"harmonized_output": output}


# -------------------- Main Entrypoint --------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Running Hermes in Headless Mode (FastAPI)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
