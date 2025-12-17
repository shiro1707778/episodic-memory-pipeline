"""
SQLite database layer for structured memory storage.

Handles persistence of Episodes, Facts, Summaries, and their relationships.
"""
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import contextmanager

from ..models import Episode, Fact, Summary


class Database:
    """SQLite database manager for the memory pipeline."""
    
    def __init__(self, db_path: Path):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_directory()
        self._initialize_schema()
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _initialize_schema(self):
        """Initialize database schema from SQL file."""
        schema_path = Path(__file__).parent.parent.parent / "schema.sql"
        
        if schema_path.exists():
            with open(schema_path) as f:
                schema_sql = f.read()
            
            with self._connection() as conn:
                conn.executescript(schema_sql)
    
    # =========================================================================
    # Episode Operations
    # =========================================================================
    
    def save_episode(self, episode: Episode) -> str:
        """
        Save an episode to the database.
        
        Args:
            episode: Episode to save
            
        Returns:
            Episode ID
        """
        row = episode.to_db_row()
        
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO episodes (
                    id, created_at, occurred_at, raw_input, content,
                    memory_type, topics, entities, confidence, importance,
                    source, session_id, is_active, consolidated, embedding_id
                ) VALUES (
                    :id, :created_at, :occurred_at, :raw_input, :content,
                    :memory_type, :topics, :entities, :confidence, :importance,
                    :source, :session_id, :is_active, :consolidated, :embedding_id
                )
            """, row)
            
            # Update topic counts
            for topic in episode.topics:
                conn.execute("""
                    INSERT INTO topics (name, episode_count)
                    VALUES (?, 1)
                    ON CONFLICT(name) DO UPDATE SET
                        episode_count = episode_count + 1
                """, (topic,))
        
        return episode.id
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM episodes WHERE id = ?",
                (episode_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return Episode.from_db_row(dict(row))
            return None
    
    def get_episodes(
        self,
        topic: Optional[str] = None,
        memory_type: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        consolidated: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Episode]:
        """
        Query episodes with filters.
        
        Args:
            topic: Filter by topic (uses JSON contains)
            memory_type: Filter by memory type
            since: Episodes occurring after this time
            until: Episodes occurring before this time
            consolidated: Filter by consolidation status
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of matching episodes
        """
        conditions = ["is_active = TRUE"]
        params = []
        
        if topic:
            conditions.append("topics LIKE ?")
            params.append(f'%"{topic}"%')
        
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        
        if since:
            conditions.append("occurred_at >= ?")
            params.append(since.isoformat())
        
        if until:
            conditions.append("occurred_at <= ?")
            params.append(until.isoformat())
        
        if consolidated is not None:
            conditions.append("consolidated = ?")
            params.append(consolidated)
        
        query = f"""
            SELECT * FROM episodes
            WHERE {' AND '.join(conditions)}
            ORDER BY occurred_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return [Episode.from_db_row(dict(row)) for row in cursor.fetchall()]
    
    def mark_episodes_consolidated(self, episode_ids: list[str]):
        """Mark episodes as consolidated."""
        if not episode_ids:
            return
        
        placeholders = ",".join("?" * len(episode_ids))
        with self._connection() as conn:
            conn.execute(
                f"UPDATE episodes SET consolidated = TRUE WHERE id IN ({placeholders})",
                episode_ids
            )
    
    def get_unconsolidated_episodes(
        self,
        topic: Optional[str] = None,
        limit: int = 100
    ) -> list[Episode]:
        """Get episodes that haven't been consolidated yet."""
        return self.get_episodes(
            topic=topic,
            consolidated=False,
            limit=limit
        )
    
    # =========================================================================
    # Fact Operations
    # =========================================================================
    
    def save_fact(self, fact: Fact, source_episode_ids: list[str] = None) -> str:
        """
        Save a fact and link to source episodes.
        
        Args:
            fact: Fact to save
            source_episode_ids: IDs of episodes that support this fact
            
        Returns:
            Fact ID
        """
        row = fact.to_db_row()
        
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO facts (
                    id, created_at, updated_at, content, category,
                    topic, entities, confidence, valid_from, valid_until,
                    is_active, superseded_by, embedding_id
                ) VALUES (
                    :id, :created_at, :updated_at, :content, :category,
                    :topic, :entities, :confidence, :valid_from, :valid_until,
                    :is_active, :superseded_by, :embedding_id
                )
            """, row)
            
            # Link to source episodes
            if source_episode_ids:
                for episode_id in source_episode_ids:
                    conn.execute("""
                        INSERT OR IGNORE INTO episode_facts (episode_id, fact_id, relationship)
                        VALUES (?, ?, 'source')
                    """, (episode_id, fact.id))
        
        return fact.id
    
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM facts WHERE id = ?",
                (fact_id,)
            )
            row = cursor.fetchone()
            
            if row:
                fact = Fact.from_db_row(dict(row))
                # Load source episode IDs
                cursor = conn.execute(
                    "SELECT episode_id FROM episode_facts WHERE fact_id = ?",
                    (fact_id,)
                )
                fact.source_episode_ids = [r["episode_id"] for r in cursor.fetchall()]
                return fact
            return None
    
    def get_facts(
        self,
        topic: Optional[str] = None,
        category: Optional[str] = None,
        current_only: bool = True,
        limit: int = 100
    ) -> list[Fact]:
        """Query facts with filters."""
        conditions = ["is_active = TRUE"]
        params = []
        
        if topic:
            conditions.append("topic = ?")
            params.append(topic)
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if current_only:
            conditions.append("(valid_until IS NULL OR valid_until > ?)")
            params.append(datetime.utcnow().isoformat())
            conditions.append("superseded_by IS NULL")
        
        query = f"""
            SELECT * FROM facts
            WHERE {' AND '.join(conditions)}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        
        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return [Fact.from_db_row(dict(row)) for row in cursor.fetchall()]
    
    def find_similar_facts(self, content: str, topic: str) -> list[Fact]:
        """Find potentially duplicate or related facts by content similarity."""
        # Simple text-based similarity check (vector search handles semantic)
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM facts
                WHERE topic = ? AND is_active = TRUE
                ORDER BY updated_at DESC
                LIMIT 10
            """, (topic,))
            return [Fact.from_db_row(dict(row)) for row in cursor.fetchall()]
    
    def supersede_fact(self, old_fact_id: str, new_fact: Fact):
        """Mark an old fact as superseded by a new one."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE facts SET superseded_by = ?, is_active = FALSE WHERE id = ?",
                (new_fact.id, old_fact_id)
            )
    
    # =========================================================================
    # Summary Operations
    # =========================================================================
    
    def save_summary(
        self,
        summary: Summary,
        source_episode_ids: list[str] = None,
        key_episode_ids: list[str] = None
    ) -> str:
        """
        Save a summary and link to source episodes.
        
        Args:
            summary: Summary to save
            source_episode_ids: All episodes that contributed
            key_episode_ids: Subset of episodes that are key events
            
        Returns:
            Summary ID
        """
        row = summary.to_db_row()
        
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO summaries (
                    id, created_at, updated_at, content, topic,
                    time_start, time_end, episode_count, key_events,
                    parent_summary_id, summary_level, is_active, embedding_id
                ) VALUES (
                    :id, :created_at, :updated_at, :content, :topic,
                    :time_start, :time_end, :episode_count, :key_events,
                    :parent_summary_id, :summary_level, :is_active, :embedding_id
                )
            """, row)
            
            # Link to source episodes
            if source_episode_ids:
                key_set = set(key_episode_ids or [])
                for episode_id in source_episode_ids:
                    conn.execute("""
                        INSERT OR IGNORE INTO episode_summaries 
                        (episode_id, summary_id, is_key_event)
                        VALUES (?, ?, ?)
                    """, (episode_id, summary.id, episode_id in key_set))
            
            # Update topic's last consolidation time
            conn.execute("""
                UPDATE topics SET last_consolidation = ?
                WHERE name = ?
            """, (datetime.utcnow().isoformat(), summary.topic))
        
        return summary.id
    
    def get_summary(self, summary_id: str) -> Optional[Summary]:
        """Retrieve a summary by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM summaries WHERE id = ?",
                (summary_id,)
            )
            row = cursor.fetchone()
            
            if row:
                summary = Summary.from_db_row(dict(row))
                cursor = conn.execute(
                    "SELECT episode_id FROM episode_summaries WHERE summary_id = ?",
                    (summary_id,)
                )
                summary.source_episode_ids = [r["episode_id"] for r in cursor.fetchall()]
                return summary
            return None
    
    def get_summaries(
        self,
        topic: Optional[str] = None,
        level: Optional[int] = None,
        since: Optional[datetime] = None,
        limit: int = 50
    ) -> list[Summary]:
        """Query summaries with filters."""
        conditions = ["is_active = TRUE"]
        params = []
        
        if topic:
            conditions.append("topic = ?")
            params.append(topic)
        
        if level:
            conditions.append("summary_level = ?")
            params.append(level)
        
        if since:
            conditions.append("time_end >= ?")
            params.append(since.isoformat())
        
        query = f"""
            SELECT * FROM summaries
            WHERE {' AND '.join(conditions)}
            ORDER BY time_end DESC
            LIMIT ?
        """
        params.append(limit)
        
        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return [Summary.from_db_row(dict(row)) for row in cursor.fetchall()]
    
    def get_latest_summary(self, topic: str) -> Optional[Summary]:
        """Get the most recent summary for a topic."""
        summaries = self.get_summaries(topic=topic, limit=1)
        return summaries[0] if summaries else None
    
    # =========================================================================
    # Topic Operations
    # =========================================================================
    
    def get_topics(self) -> list[dict]:
        """Get all registered topics with stats."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT name, description, episode_count, last_consolidation
                FROM topics
                ORDER BY episode_count DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_topics_needing_consolidation(
        self,
        min_episodes: int = 5,
        max_age_days: int = 7
    ) -> list[str]:
        """
        Find topics that need consolidation.
        
        Criteria:
        - Has at least min_episodes unconsolidated episodes
        - OR hasn't been consolidated in max_age_days
        """
        cutoff = datetime.utcnow()
        
        with self._connection() as conn:
            # Topics with enough unconsolidated episodes
            cursor = conn.execute("""
                SELECT DISTINCT json_each.value as topic
                FROM episodes, json_each(episodes.topics)
                WHERE consolidated = FALSE AND is_active = TRUE
                GROUP BY json_each.value
                HAVING COUNT(*) >= ?
            """, (min_episodes,))
            
            topics = {row["topic"] for row in cursor.fetchall()}
            
            # Topics that haven't been consolidated recently
            cursor = conn.execute("""
                SELECT name FROM topics
                WHERE last_consolidation IS NULL
                   OR last_consolidation < datetime('now', ? || ' days')
            """, (f"-{max_age_days}",))
            
            topics.update(row["name"] for row in cursor.fetchall())
            
            return list(topics)
    
    # =========================================================================
    # Utility Operations
    # =========================================================================
    
    def update_embedding_id(self, table: str, record_id: str, embedding_id: int):
        """Update the embedding_id for a record."""
        valid_tables = {"episodes", "facts", "summaries"}
        if table not in valid_tables:
            raise ValueError(f"Invalid table: {table}")
        
        with self._connection() as conn:
            conn.execute(
                f"UPDATE {table} SET embedding_id = ? WHERE id = ?",
                (embedding_id, record_id)
            )
    
    def get_statistics(self) -> dict:
        """Get database statistics."""
        with self._connection() as conn:
            stats = {}
            
            cursor = conn.execute("SELECT COUNT(*) FROM episodes WHERE is_active = TRUE")
            stats["total_episodes"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM episodes WHERE consolidated = FALSE AND is_active = TRUE")
            stats["unconsolidated_episodes"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM facts WHERE is_active = TRUE")
            stats["total_facts"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM summaries WHERE is_active = TRUE")
            stats["total_summaries"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM topics")
            stats["total_topics"] = cursor.fetchone()[0]
            
            return stats

