"""
Semantic retrieval - vector similarity based memory lookup.

For queries like "What am I learning right now?" or "What do I know about X?"
"""
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from ..models import Episode, Fact, Summary
from ..storage import Database, VectorStore
from ..embeddings import EmbeddingProvider


@dataclass 
class SemanticResult:
    """Result of semantic retrieval."""
    episodes: list[Episode]
    facts: list[Fact]
    summaries: list[Summary]
    query_embedding_time: float
    search_time: float


class SemanticRetriever:
    """
    Semantic retrieval using vector similarity.
    
    Combines results from episodes, facts, and summaries
    with optional metadata filtering.
    """
    
    def __init__(
        self,
        database: Database,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize semantic retriever.
        
        Args:
            database: Database for metadata
            vector_store: Vector store for similarity search
            embedding_provider: Embedding model
            top_k: Number of results per type
            similarity_threshold: Minimum similarity score
        """
        self.database = database
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.threshold = similarity_threshold
    
    def search(
        self,
        query: str,
        search_episodes: bool = True,
        search_facts: bool = True,
        search_summaries: bool = True,
        topic_filter: Optional[str] = None,
        time_filter_since: Optional[datetime] = None,
        time_filter_until: Optional[datetime] = None,
        top_k: Optional[int] = None,
    ) -> SemanticResult:
        """
        Perform semantic search across memory types.
        
        Args:
            query: Search query
            search_episodes: Include episodes in search
            search_facts: Include facts in search
            search_summaries: Include summaries in search
            topic_filter: Filter by topic
            time_filter_since: Filter episodes after this time
            time_filter_until: Filter episodes before this time
            top_k: Override default top_k
            
        Returns:
            SemanticResult with ranked results
        """
        import time
        
        k = top_k or self.top_k
        
        # Embed query
        t0 = time.time()
        query_embedding = self.embedding_provider.embed_text(query)
        embed_time = time.time() - t0
        
        episodes = []
        facts = []
        summaries = []
        
        t0 = time.time()
        
        # Search episodes
        if search_episodes:
            episode_results = self._search_episodes(
                query_embedding, k, topic_filter, 
                time_filter_since, time_filter_until
            )
            episodes = episode_results
        
        # Search facts
        if search_facts:
            fact_results = self._search_facts(
                query_embedding, k, topic_filter
            )
            facts = fact_results
        
        # Search summaries
        if search_summaries:
            summary_results = self._search_summaries(
                query_embedding, k, topic_filter
            )
            summaries = summary_results
        
        search_time = time.time() - t0
        
        return SemanticResult(
            episodes=episodes,
            facts=facts,
            summaries=summaries,
            query_embedding_time=embed_time,
            search_time=search_time
        )
    
    def _search_episodes(
        self,
        query_embedding,
        k: int,
        topic_filter: Optional[str],
        time_since: Optional[datetime],
        time_until: Optional[datetime],
    ) -> list[Episode]:
        """Search episodes with optional filtering."""
        
        # If filtering, get valid IDs first
        if topic_filter or time_since or time_until:
            db_episodes = self.database.get_episodes(
                topic=topic_filter,
                since=time_since,
                until=time_until,
                limit=k * 3  # Get more to account for vector filtering
            )
            valid_ids = {ep.id for ep in db_episodes}
            
            if not valid_ids:
                return []
            
            results = self.vector_store.search_with_filter(
                "episodes",
                query_embedding,
                valid_ids,
                k=k,
                threshold=self.threshold
            )
        else:
            results = self.vector_store.search(
                "episodes",
                query_embedding,
                k=k,
                threshold=self.threshold
            )
        
        # Fetch full episodes
        episodes = []
        for record_id, score in results:
            episode = self.database.get_episode(record_id)
            if episode:
                episodes.append(episode)
        
        return episodes
    
    def _search_facts(
        self,
        query_embedding,
        k: int,
        topic_filter: Optional[str],
    ) -> list[Fact]:
        """Search facts with optional filtering."""
        
        if topic_filter:
            db_facts = self.database.get_facts(topic=topic_filter)
            valid_ids = {f.id for f in db_facts}
            
            if not valid_ids:
                return []
            
            results = self.vector_store.search_with_filter(
                "facts",
                query_embedding,
                valid_ids,
                k=k,
                threshold=self.threshold
            )
        else:
            results = self.vector_store.search(
                "facts",
                query_embedding,
                k=k,
                threshold=self.threshold
            )
        
        facts = []
        for record_id, score in results:
            fact = self.database.get_fact(record_id)
            if fact:
                facts.append(fact)
        
        return facts
    
    def _search_summaries(
        self,
        query_embedding,
        k: int,
        topic_filter: Optional[str],
    ) -> list[Summary]:
        """Search summaries with optional filtering."""
        
        if topic_filter:
            db_summaries = self.database.get_summaries(topic=topic_filter)
            valid_ids = {s.id for s in db_summaries}
            
            if not valid_ids:
                return []
            
            results = self.vector_store.search_with_filter(
                "summaries",
                query_embedding,
                valid_ids,
                k=k,
                threshold=self.threshold
            )
        else:
            results = self.vector_store.search(
                "summaries",
                query_embedding,
                k=k,
                threshold=self.threshold
            )
        
        summaries = []
        for record_id, score in results:
            summary = self.database.get_summary(record_id)
            if summary:
                summaries.append(summary)
        
        return summaries
    
    def find_related_memories(
        self,
        episode: Episode,
        exclude_self: bool = True,
    ) -> SemanticResult:
        """
        Find memories related to a given episode.
        
        Useful for showing context or finding contradictions.
        """
        # Use episode's embedding text as query
        query_text = episode.to_embedding_text()
        result = self.search(query_text)
        
        if exclude_self:
            result.episodes = [ep for ep in result.episodes if ep.id != episode.id]
        
        return result

