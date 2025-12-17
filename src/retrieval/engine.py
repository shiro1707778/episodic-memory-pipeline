"""
Retrieval engine - unified interface for memory queries.

Combines semantic and narrative retrieval with LLM-based answer synthesis.
"""
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from ..models import Episode, Fact, Summary
from ..storage import Database, VectorStore
from ..embeddings import EmbeddingProvider
from ..llm import LLMProvider
from ..prompts import PromptTemplates
from .semantic import SemanticRetriever, SemanticResult
from .narrative import NarrativeRetriever, NarrativeResult


@dataclass
class QueryResult:
    """Result of a memory query."""
    answer: str
    confidence: float
    episodes: list[Episode]
    facts: list[Fact]
    summaries: list[Summary]
    query_type: str  # "semantic" or "narrative"
    gaps: list[str]  # Information that would help but is missing


class RetrievalEngine:
    """
    Unified retrieval engine that:
    1. Analyzes the query to determine best strategy
    2. Retrieves relevant memories
    3. Synthesizes an answer using LLM
    
    This is the main interface for querying the memory system.
    """
    
    def __init__(
        self,
        database: Database,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
    ):
        """
        Initialize retrieval engine.
        
        Args:
            database: Database storage
            vector_store: Vector storage
            embedding_provider: Embedding model
            llm: LLM for analysis and synthesis
        """
        self.database = database
        self.llm = llm
        
        self.semantic = SemanticRetriever(
            database, vector_store, embedding_provider
        )
        self.narrative = NarrativeRetriever(
            database, vector_store, embedding_provider
        )
    
    def query(
        self,
        query: str,
        synthesize: bool = True,
    ) -> QueryResult:
        """
        Process a natural language query.
        
        Args:
            query: User's query
            synthesize: Whether to generate LLM answer
            
        Returns:
            QueryResult with answer and supporting memories
        """
        # Analyze query to determine strategy
        query_analysis = self._analyze_query(query)
        
        query_type = query_analysis.get("query_type", "semantic")
        time_filter = query_analysis.get("time_filter", {})
        topic_filters = query_analysis.get("topic_filters", [])
        reformulated = query_analysis.get("reformulated_query", query)
        
        # Parse time filters
        since = None
        until = None
        if time_filter.get("since"):
            try:
                since = datetime.fromisoformat(time_filter["since"])
            except (ValueError, TypeError):
                pass
        if time_filter.get("until"):
            try:
                until = datetime.fromisoformat(time_filter["until"])
            except (ValueError, TypeError):
                pass
        
        # Retrieve based on query type
        if query_type == "narrative":
            result = self._narrative_retrieval(
                reformulated, 
                topic_filters[0] if topic_filters else None,
                since, until
            )
        else:  # semantic or hybrid
            result = self._semantic_retrieval(
                reformulated,
                topic_filters[0] if topic_filters else None,
                since, until
            )
        
        # Synthesize answer if requested
        if synthesize:
            synthesis = self._synthesize_answer(query, result)
            answer = synthesis.get("answer", "I don't have enough information to answer this.")
            confidence = synthesis.get("confidence", 0.5)
            gaps = synthesis.get("gaps", [])
        else:
            answer = ""
            confidence = 0.0
            gaps = []
        
        return QueryResult(
            answer=answer,
            confidence=confidence,
            episodes=result.episodes,
            facts=result.facts if hasattr(result, 'facts') else [],
            summaries=result.summaries if hasattr(result, 'summaries') else [],
            query_type=query_type,
            gaps=gaps
        )
    
    def _analyze_query(self, query: str) -> dict:
        """Analyze query to determine retrieval strategy."""
        # Get known topics for context
        topics = self.database.get_topics()
        topic_names = [t["name"] for t in topics]
        
        prompt = PromptTemplates.QUERY_ANALYSIS.format(
            query=query,
            known_topics=", ".join(topic_names[:20]) if topic_names else "none",
            recent_activity="recent memory activity"
        )
        
        try:
            response = self.llm.complete(prompt)
            return json.loads(response)
        except (json.JSONDecodeError, Exception):
            # Default to semantic search
            return {
                "query_type": "semantic",
                "time_relevance": "all_time",
                "time_filter": {},
                "search_concepts": [query],
                "topic_filters": [],
                "reformulated_query": query
            }
    
    def _semantic_retrieval(
        self,
        query: str,
        topic: Optional[str],
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> SemanticResult:
        """Perform semantic retrieval."""
        return self.semantic.search(
            query,
            topic_filter=topic,
            time_filter_since=since,
            time_filter_until=until,
        )
    
    def _narrative_retrieval(
        self,
        query: str,
        topic: Optional[str],
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> NarrativeResult:
        """Perform narrative retrieval."""
        if topic:
            return self.narrative.recall(topic, since=since, until=until)
        else:
            return self.narrative.recall_by_query(query, since=since, until=until)
    
    def _synthesize_answer(
        self,
        query: str,
        result,
    ) -> dict:
        """Synthesize an answer from retrieved memories."""
        # Format memories for prompt
        episodes_text = PromptTemplates.format_episodes_for_prompt(
            result.episodes if result.episodes else []
        )
        
        facts_text = PromptTemplates.format_facts_for_prompt(
            result.facts if hasattr(result, 'facts') and result.facts else []
        )
        
        summaries_text = PromptTemplates.format_summaries_for_prompt(
            result.summaries if hasattr(result, 'summaries') and result.summaries else []
        )
        
        prompt = PromptTemplates.ANSWER_SYNTHESIS.format(
            query=query,
            summaries=summaries_text,
            facts=facts_text,
            episodes=episodes_text
        )
        
        try:
            response = self.llm.complete(prompt)
            return json.loads(response)
        except (json.JSONDecodeError, Exception) as e:
            return {
                "answer": f"Retrieved {len(result.episodes)} episodes but couldn't synthesize answer: {e}",
                "confidence": 0.3,
                "key_sources": [],
                "gaps": ["synthesis failed"]
            }
    
    def recall_narrative(
        self,
        topic_or_query: str,
        is_topic: bool = False,
    ) -> QueryResult:
        """
        Recall a narrative (story/journey) about a topic.
        
        This is optimized for "Tell me about..." style queries.
        """
        if is_topic:
            result = self.narrative.recall(topic_or_query)
        else:
            result = self.narrative.recall_by_query(topic_or_query)
        
        # Generate narrative synthesis
        narrative = self._synthesize_narrative(topic_or_query, result)
        
        return QueryResult(
            answer=narrative.get("narrative", "No narrative available."),
            confidence=0.8 if result.episodes else 0.2,
            episodes=result.episodes,
            facts=result.facts,
            summaries=result.summaries,
            query_type="narrative",
            gaps=[]
        )
    
    def _synthesize_narrative(
        self,
        topic: str,
        result: NarrativeResult,
    ) -> dict:
        """Generate a narrative synthesis."""
        episodes_text = PromptTemplates.format_episodes_for_prompt(result.episodes)
        facts_text = PromptTemplates.format_facts_for_prompt(result.facts)
        summaries_text = PromptTemplates.format_summaries_for_prompt(result.summaries)
        
        prompt = PromptTemplates.NARRATIVE_SYNTHESIS.format(
            topic=topic,
            query=f"Tell me about {topic}",
            episodes=episodes_text,
            facts=facts_text,
            summaries=summaries_text
        )
        
        try:
            response = self.llm.complete(prompt)
            return json.loads(response)
        except (json.JSONDecodeError, Exception):
            return {
                "narrative": f"Found {len(result.episodes)} memories about {topic}.",
                "timeline": [],
                "key_moments": [],
                "current_status": "Unknown"
            }
    
    def quick_lookup(self, query: str) -> list[Fact]:
        """
        Quick fact lookup without synthesis.
        
        For simple "What is my X?" type queries.
        """
        result = self.semantic.search(
            query,
            search_episodes=False,
            search_summaries=False,
            search_facts=True,
            top_k=5
        )
        return result.facts
    
    def get_context(self, topic: str, max_items: int = 5) -> dict:
        """
        Get quick context about a topic.
        
        Returns recent episodes, key facts, and latest summary.
        """
        episodes = self.database.get_episodes(topic=topic, limit=max_items)
        facts = self.database.get_facts(topic=topic, limit=max_items)
        summary = self.database.get_latest_summary(topic)
        
        return {
            "topic": topic,
            "recent_episodes": episodes,
            "facts": facts,
            "summary": summary,
            "episode_count": len(episodes),
            "fact_count": len(facts),
        }

