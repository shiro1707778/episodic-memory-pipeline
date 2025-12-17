"""
Episodic Memory Pipeline

A local-first personal memory system for AI assistants.
"""

__version__ = "0.1.0"

from memory.models import Episode, Fact, Summary, MemoryType
from memory.ingestion import IngestionPipeline
from memory.retrieval import MemoryRetriever
from memory.consolidation import ConsolidationService

__all__ = [
    "Episode",
    "Fact", 
    "Summary",
    "MemoryType",
    "IngestionPipeline",
    "MemoryRetriever",
    "ConsolidationService",
]

