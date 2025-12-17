"""
Episode extractor.

Extracts structured episodic memory from raw text using LLM.
"""
import json
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from ..llm import LLMProvider
from ..models import Episode, MemoryType
from ..prompts import PromptTemplates
from ..utils import as_list, as_str, as_float


@dataclass
class ExtractionResult:
    """Result of episode extraction."""
    episode: Episode
    extraction_confidence: float
    raw_llm_response: str


class EpisodeExtractor:
    """
    Extracts structured episodic memories from text.
    
    Uses LLM to:
    - Clean and normalize content
    - Identify memory type
    - Extract topics and entities
    - Determine importance
    - Infer temporal offsets ("yesterday", "last week")
    """
    
    # Time offset mappings
    TIME_OFFSETS = {
        "none": timedelta(0),
        "yesterday": timedelta(days=-1),
        "last_week": timedelta(days=-7),
        "last_month": timedelta(days=-30),
        "few_days_ago": timedelta(days=-3),
        "earlier_today": timedelta(hours=-6),
    }
    
    def __init__(self, llm: LLMProvider):
        """
        Initialize extractor.
        
        Args:
            llm: LLM provider for extraction
        """
        self.llm = llm
    
    def extract(
        self,
        text: str,
        memory_type_hint: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        source: str = "chat",
        session_id: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract episode from text.
        
        Args:
            text: Raw input text
            memory_type_hint: Hint from classifier (may be overridden)
            timestamp: When the input was received
            source: Source of input (chat, note, import)
            session_id: Session identifier for grouping
            
        Returns:
            ExtractionResult with Episode and metadata
        """
        timestamp = timestamp or datetime.utcnow()
        
        prompt = PromptTemplates.EPISODE_EXTRACTION.format(
            text=text,
            timestamp=timestamp.isoformat()
        )
        
        try:
            response = self.llm.complete(prompt)
            result = json.loads(response)
            
            # Parse memory type (sanitize string value)
            memory_type_str = as_str(result.get("memory_type"), default=memory_type_hint or "episodic")
            try:
                memory_type = MemoryType(memory_type_str)
            except ValueError:
                memory_type = MemoryType.EPISODIC
            
            # Calculate occurred_at from offset
            offset_str = as_str(result.get("occurred_at_offset"), default="none")
            offset = self.TIME_OFFSETS.get(offset_str, timedelta(0))
            occurred_at = timestamp + offset
            
            # Sanitize all LLM output fields
            topics = as_list(result.get("topics"))
            entities = as_list(result.get("entities"))
            importance = as_float(result.get("importance"), default=0.5)
            content = as_str(result.get("content"), default=text) or text
            
            # Create episode with sanitized values
            episode = Episode(
                raw_input=text,
                content=content,
                memory_type=memory_type,
                topics=topics,
                entities=entities,
                importance=importance,
                confidence=1.0,  # Will be set by pipeline
                occurred_at=occurred_at,
                source=source,
                session_id=session_id,
            )
            
            return ExtractionResult(
                episode=episode,
                extraction_confidence=0.9,  # LLM extraction assumed reliable
                raw_llm_response=response
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: create basic episode from raw text
            episode = Episode(
                raw_input=text,
                content=text,
                memory_type=MemoryType(memory_type_hint) if memory_type_hint else MemoryType.EPISODIC,
                topics=self._extract_basic_topics(text),
                entities=[],
                importance=0.5,
                confidence=0.5,
                occurred_at=timestamp,
                source=source,
                session_id=session_id,
            )
            
            return ExtractionResult(
                episode=episode,
                extraction_confidence=0.3,
                raw_llm_response=str(e)
            )
    
    def _extract_basic_topics(self, text: str) -> list[str]:
        """
        Extract basic topics using heuristics (fallback).
        
        This is used when LLM extraction fails.
        """
        topics = []
        text_lower = text.lower()
        
        # Common topic keywords
        topic_keywords = {
            "work": ["work", "job", "office", "meeting", "project", "deadline"],
            "learning": ["learn", "study", "course", "lesson", "practice"],
            "health": ["health", "exercise", "workout", "diet", "sleep", "doctor"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel"],
            "family": ["family", "mom", "dad", "parent", "sibling", "child"],
            "hobby": ["hobby", "game", "read", "music", "movie", "art"],
            "finance": ["money", "budget", "save", "invest", "cost", "pay"],
            "social": ["friend", "party", "dinner", "meet", "talk"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        
        return topics[:3]  # Limit to top 3
    
    def extract_batch(
        self,
        texts: list[str],
        timestamp: Optional[datetime] = None,
        source: str = "chat"
    ) -> list[ExtractionResult]:
        """Extract episodes from multiple texts."""
        timestamp = timestamp or datetime.utcnow()
        return [
            self.extract(text, timestamp=timestamp, source=source)
            for text in texts
        ]

