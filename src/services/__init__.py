"""
Services layer for the episodic memory pipeline.

This module contains business logic that is independent of the CLI.
Services return plain dataclasses/dicts and do not import Rich/Typer.

Usage:
    from src.services import IngestionService, RetrievalService
    
    service = IngestionService(components)
    result = service.ingest_text("Some memory")
"""
from .ingestion import IngestionService
from .retrieval import RetrievalService
from .evaluation import EvaluationService
from .diagnostics import DiagnosticsService

__all__ = [
    "IngestionService",
    "RetrievalService",
    "EvaluationService",
    "DiagnosticsService",
]

