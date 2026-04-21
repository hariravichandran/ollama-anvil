"""Ollama LLM client, model management, context compression, embeddings, RAG."""

from anvil.llm.aliases import AliasManager
from anvil.llm.capabilities import ModelCapabilities, detect_capabilities
from anvil.llm.chunker import Chunk, chunk_file, chunk_text
from anvil.llm.client import OllamaClient
from anvil.llm.context import ContextCompressor
from anvil.llm.embeddings import EmbeddingStore
from anvil.llm.health import HealthChecker, HealthResult
from anvil.llm.profiles import Profile, ProfileManager
from anvil.llm.rag import RAGPipeline, RAGResult
from anvil.llm.resource import ModelFitResult, ResourceManager, ResourceStatus

__all__ = [
    "OllamaClient",
    "ContextCompressor",
    "EmbeddingStore",
    "ModelCapabilities",
    "detect_capabilities",
    "Chunk",
    "chunk_text",
    "chunk_file",
    "RAGPipeline",
    "RAGResult",
    "HealthChecker",
    "HealthResult",
    "Profile",
    "ProfileManager",
    "AliasManager",
    "ResourceManager",
    "ResourceStatus",
    "ModelFitResult",
]
