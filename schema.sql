-- Episodic Memory Pipeline Database Schema
-- SQLite with JSON1 extension for flexible metadata

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ============================================================================
-- CORE MEMORY TABLES
-- ============================================================================

-- Episodes: Raw episodic memories (events, conversations, notes)
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,                          -- UUID
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    occurred_at TIMESTAMP NOT NULL,               -- When the event actually happened
    
    -- Content
    raw_input TEXT NOT NULL,                      -- Original input text
    content TEXT NOT NULL,                        -- Processed/cleaned content
    
    -- Classification
    memory_type TEXT NOT NULL CHECK (memory_type IN (
        'episodic',      -- Event that happened
        'fact',          -- Factual statement
        'goal',          -- User's goal or intention
        'preference',    -- User's preference
        'reflection'     -- Meta-cognitive reflection
    )),
    
    -- Metadata for filtering
    topics TEXT NOT NULL DEFAULT '[]',            -- JSON array of topic tags
    entities TEXT NOT NULL DEFAULT '[]',          -- JSON array of named entities
    confidence REAL NOT NULL DEFAULT 1.0,         -- Extraction confidence (0-1)
    importance REAL NOT NULL DEFAULT 0.5,         -- Computed importance score (0-1)
    
    -- Context
    source TEXT DEFAULT 'chat',                   -- chat, note, import, etc.
    session_id TEXT,                              -- Group related inputs
    
    -- State
    is_active BOOLEAN NOT NULL DEFAULT TRUE,      -- Soft delete / archival
    consolidated BOOLEAN NOT NULL DEFAULT FALSE,  -- Has been processed by consolidation
    
    -- Vector reference (FAISS index uses this ID)
    embedding_id INTEGER                          -- Index in FAISS
);

CREATE INDEX IF NOT EXISTS idx_episodes_occurred_at ON episodes(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_memory_type ON episodes(memory_type);
CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated);
CREATE INDEX IF NOT EXISTS idx_episodes_is_active ON episodes(is_active);

-- ============================================================================
-- SEMANTIC MEMORY: Facts
-- ============================================================================

-- Facts: Distilled, stable knowledge extracted from episodes
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,                          -- UUID
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Content
    content TEXT NOT NULL,                        -- The factual statement
    category TEXT NOT NULL CHECK (category IN (
        'personal',      -- About the user
        'preference',    -- User preferences
        'relationship',  -- People/entities user knows
        'knowledge',     -- Things user knows
        'context',       -- Situational context
        'goal'           -- Long-term goals
    )),
    
    -- Metadata
    topic TEXT NOT NULL,                          -- Primary topic
    entities TEXT NOT NULL DEFAULT '[]',          -- JSON array of entities
    
    -- Confidence and validity
    confidence REAL NOT NULL DEFAULT 0.8,         -- How certain is this fact
    valid_from TIMESTAMP,                         -- When fact became true
    valid_until TIMESTAMP,                        -- When fact stopped being true (NULL = current)
    
    -- State
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_by TEXT REFERENCES facts(id),      -- If updated by newer fact
    
    -- Vector reference
    embedding_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_facts_topic ON facts(topic);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_facts_is_active ON facts(is_active);

-- ============================================================================
-- CONSOLIDATED MEMORY: Summaries
-- ============================================================================

-- Summaries: Narrative consolidations of episodes by topic/time
CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,                          -- UUID
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Content
    content TEXT NOT NULL,                        -- The summary narrative
    
    -- Scope
    topic TEXT NOT NULL,                          -- What topic this summarizes
    time_start TIMESTAMP NOT NULL,                -- Coverage start
    time_end TIMESTAMP NOT NULL,                  -- Coverage end
    
    -- Metadata
    episode_count INTEGER NOT NULL DEFAULT 0,     -- How many episodes contributed
    key_events TEXT NOT NULL DEFAULT '[]',        -- JSON array of key event snippets
    
    -- Hierarchy
    parent_summary_id TEXT REFERENCES summaries(id),  -- For multi-level summaries
    summary_level INTEGER NOT NULL DEFAULT 1,     -- 1=weekly, 2=monthly, 3=quarterly
    
    -- State
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Vector reference
    embedding_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_summaries_topic ON summaries(topic);
CREATE INDEX IF NOT EXISTS idx_summaries_time ON summaries(time_start, time_end);
CREATE INDEX IF NOT EXISTS idx_summaries_level ON summaries(summary_level);

-- ============================================================================
-- PROVENANCE: Linking memories to sources
-- ============================================================================

-- Link facts to their source episodes
CREATE TABLE IF NOT EXISTS episode_facts (
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- How this episode supports this fact
    relationship TEXT NOT NULL CHECK (relationship IN (
        'source',        -- Episode is the source of this fact
        'confirms',      -- Episode confirms existing fact
        'contradicts',   -- Episode contradicts this fact
        'updates'        -- Episode updates this fact
    )),
    
    confidence REAL NOT NULL DEFAULT 1.0,
    
    PRIMARY KEY (episode_id, fact_id)
);

CREATE INDEX IF NOT EXISTS idx_episode_facts_fact ON episode_facts(fact_id);

-- Link summaries to their source episodes
CREATE TABLE IF NOT EXISTS episode_summaries (
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    summary_id TEXT NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Whether this episode is a key contributor
    is_key_event BOOLEAN NOT NULL DEFAULT FALSE,
    
    PRIMARY KEY (episode_id, summary_id)
);

CREATE INDEX IF NOT EXISTS idx_episode_summaries_summary ON episode_summaries(summary_id);

-- Link summaries to extracted facts
CREATE TABLE IF NOT EXISTS summary_facts (
    summary_id TEXT NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
    fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (summary_id, fact_id)
);

-- ============================================================================
-- OPERATIONAL TABLES
-- ============================================================================

-- Track consolidation runs
CREATE TABLE IF NOT EXISTS consolidation_runs (
    id TEXT PRIMARY KEY,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    
    topic TEXT,                                   -- NULL = all topics
    episodes_processed INTEGER DEFAULT 0,
    summaries_created INTEGER DEFAULT 0,
    facts_extracted INTEGER DEFAULT 0,
    
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    error_message TEXT
);

-- Topic registry for consistent topic management
CREATE TABLE IF NOT EXISTS topics (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    episode_count INTEGER NOT NULL DEFAULT 0,
    last_consolidation TIMESTAMP
);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Active episodes with parsed metadata
CREATE VIEW IF NOT EXISTS v_active_episodes AS
SELECT 
    e.*,
    json_extract(e.topics, '$') as topics_array,
    json_extract(e.entities, '$') as entities_array
FROM episodes e
WHERE e.is_active = TRUE;

-- Facts with source episode count
CREATE VIEW IF NOT EXISTS v_facts_with_sources AS
SELECT 
    f.*,
    COUNT(ef.episode_id) as source_count,
    MAX(e.occurred_at) as last_supporting_event
FROM facts f
LEFT JOIN episode_facts ef ON f.id = ef.fact_id
LEFT JOIN episodes e ON ef.episode_id = e.id
WHERE f.is_active = TRUE
GROUP BY f.id;

-- Summary with episode details
CREATE VIEW IF NOT EXISTS v_summaries_detailed AS
SELECT 
    s.*,
    COUNT(es.episode_id) as linked_episode_count,
    GROUP_CONCAT(e.id) as episode_ids
FROM summaries s
LEFT JOIN episode_summaries es ON s.id = es.summary_id
LEFT JOIN episodes e ON es.episode_id = e.id
WHERE s.is_active = TRUE
GROUP BY s.id;

