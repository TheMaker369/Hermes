"""
Memory management for Hermes using ChromaDB and Ray.
"""

import threading
from typing import Any, Dict, List, Optional

import chromadb

from ..config import settings


class MemoryManager:
    def __init__(self):
        self.chroma_client = (
            chromadb.PersistentClient(url=settings.chroma_url)
            if settings.chroma_remote
            else chromadb.PersistentClient(path=settings.chroma_path)
        )
        self.graph_memory: Dict[str, List[Dict[str, Any]]] = {}
        self.graph_memory_lock = threading.Lock()

    async def store(self, session_id: str, user_input: str, response: Any) -> None:
        """Store memory in both ChromaDB and graph memory."""
        # ChromaDB storage
        collection = self.chroma_client.get_or_create_collection(
            f"session_{session_id}"
        )
        collection.add(
            documents=[str(response)],
            metadatas=[{"type": "response", "input": user_input}],
            ids=[f"{session_id}_{len(collection.get()['ids'])}"],
        )

        # Graph memory storage
        with self.graph_memory_lock:
            if session_id not in self.graph_memory:
                self.graph_memory[session_id] = []
            self.graph_memory[session_id].append(
                {
                    "input": user_input,
                    "response": response,
                    "timestamp": settings.get_timestamp(),
                }
            )

    async def retrieve(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve recent memories."""
        collection = self.chroma_client.get_or_create_collection(
            f"session_{session_id}"
        )
        results = collection.get()

        # Combine with graph memory
        with self.graph_memory_lock:
            graph_results = self.graph_memory.get(session_id, [])[-limit:]

        return {"vector_memory": results, "graph_memory": graph_results}


memory_manager = MemoryManager()
