"""Embedding abstraction layer."""
from .interface import (
    EmbeddingProvider,
    get_embedding_provider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OllamaEmbeddingProvider,
    MockEmbeddingProvider,
)

__all__ = [
    "EmbeddingProvider",
    "get_embedding_provider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "MockEmbeddingProvider",
]

