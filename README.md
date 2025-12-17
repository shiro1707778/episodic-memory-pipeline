# Episodic Memory Pipeline

A local-first personal episodic memory system for AI assistants. This system stores event-based memories, consolidates them into semantic summaries, and supports time-aware narrative retrieval.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EPISODIC MEMORY PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Raw Input  │───▶│  Ingestion   │───▶│   Episode    │                  │
│  │  (chat/note) │    │   Pipeline   │    │   Storage    │                  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│                             │                    │                          │
│                             ▼                    │                          │
│                      ┌──────────────┐           │                          │
│                      │  Memory-     │           │                          │
│                      │  Worthiness  │           │                          │
│                      │  Classifier  │           │                          │
│                      └──────────────┘           │                          │
│                                                 ▼                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        STORAGE LAYER                                  │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│  │  │    SQLite      │  │     FAISS      │  │   Provenance   │         │  │
│  │  │  (structured)  │  │   (vectors)    │  │    (links)     │         │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                  │                                         │
│         ┌────────────────────────┼────────────────────────┐               │
│         ▼                        ▼                        ▼               │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐            │
│  │   Episodes   │    │  Consolidation   │    │   Retrieval  │            │
│  │  (raw events)│◀───│    Process       │───▶│    Engine    │            │
│  └──────────────┘    └──────────────────┘    └──────────────┘            │
│         │                    │ │                     │                    │
│         │                    ▼ ▼                     │                    │
│         │            ┌───────────────┐               │                    │
│         │            │  Summaries &  │               │                    │
│         └───────────▶│    Facts      │◀──────────────┘                    │
│                      └───────────────┘                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Design Philosophy

1. **Episodic memory ≠ vector blobs**: Each memory is a structured event with context, time, and meaning.
2. **Time and provenance matter**: Every fact and summary links back to its source episodes.
3. **Memory must be curated, not accumulated**: Not everything is worth remembering. We filter aggressively.
4. **Retrieval should feel like recalling a journey**: Narrative coherence over raw similarity scores.

## Storage Choice: SQLite + FAISS

**Why SQLite over Postgres?**

- Local-first, no server dependencies
- Single-file portability (backup = copy file)
- Sufficient for single-user workloads (thousands of memories)
- JSON1 extension for flexible metadata
- Zero configuration

**Why FAISS for vectors?**

- Mature, fast, local-only
- Supports multiple index types for scaling
- Works well alongside SQLite
- No external service required

## Core Concepts

### Episode (Episodic Memory)

A timestamped event capturing what happened, when, and in what context.

```
"On Tuesday at 3pm, I told my assistant I'm learning Korean for a trip to Seoul in March."
```

### Fact (Semantic Memory)

A distilled, stable piece of knowledge extracted from episodes.

```
"User is learning Korean. User has a trip to Seoul planned for March 2024."
```

### Summary (Consolidated Narrative)

A topic-level summary that weaves together multiple episodes into coherent narrative.

```
"User's Korean language learning journey: Started in January 2024 motivated by upcoming Seoul trip..."
```

## Installation

```bash
cd episodic-memory-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Embedding configuration (default: local with BGE-M3)
EMBEDDING_PROVIDER=local            # 'local', 'openai', 'ollama', or 'mock'
EMBEDDING_MODEL=BAAI/bge-m3         # SentenceTransformers model name
EMBEDDING_DEVICE=cpu                # 'cpu', 'cuda', or 'mps' (Mac)
# OPENAI_API_KEY=sk-...             # only if EMBEDDING_PROVIDER=openai

# LLM configuration
LLM_PROVIDER=ollama                 # 'openai' or 'ollama' for local inference
OLLAMA_MODEL=qwen2.5:7b-instruct    # recommended local model
OLLAMA_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.2                 # low for determinism

# Storage paths
DATABASE_PATH=./data/memory.db
VECTOR_INDEX_PATH=./data/vectors.faiss
```

## Local Embeddings (Default)

The pipeline uses **local embeddings by default** with the `BAAI/bge-m3` model. No API key is required.

### First Run Model Download

On first run, SentenceTransformers will download the BGE-M3 model (~1GB). This happens automatically:

```bash
python cli.py ingest "My first memory"
# [INFO] Loading embedding model 'BAAI/bge-m3' on device 'cpu'
# (download happens here if needed)
```

### Device Selection

For faster embeddings on GPU or Apple Silicon:

```bash
# Apple Silicon (Metal)
export EMBEDDING_DEVICE=mps

# NVIDIA GPU
export EMBEDDING_DEVICE=cuda

# CPU (default)
export EMBEDDING_DEVICE=cpu
```

### Alternative Embedding Models

Smaller, faster models are available:

```bash
# Smaller, faster model (384 dimensions)
export EMBEDDING_MODEL=all-MiniLM-L6-v2

# Medium model, good balance (768 dimensions)
export EMBEDDING_MODEL=all-mpnet-base-v2
```

### Why BGE-M3?

**BAAI/bge-m3** is the recommended embedding model because:

- **High quality**: State-of-the-art multilingual embeddings
- **Dense retrieval optimized**: Designed for semantic search
- **Multilingual**: Excellent for personal memory with mixed languages
- **1024 dimensions**: Good balance of quality vs storage

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed model comparison.

## Local-First Setup with Qwen (Recommended)

For fully local operation without API dependencies, use Ollama with Qwen:

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows - download from https://ollama.com/download
```

### 2. Pull the Qwen Model

```bash
ollama pull qwen2.5:7b-instruct
```

### 3. Start Ollama Server

```bash
ollama serve
```

### 4. Configure Environment

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5:7b-instruct
export EMBEDDING_PROVIDER=local  # optional: use local embeddings too
```

### Why Qwen?

**Qwen 2.5 7B Instruct** is the recommended local model for this pipeline because:

- **Excellent instruction following**: Reliably produces structured JSON output
- **Strong multilingual support**: Important for personal memory systems (names, phrases in other languages)
- **Good reasoning**: Handles memory classification and extraction tasks well
- **Efficient**: 7B parameters runs well on consumer hardware (8GB+ VRAM or 16GB+ RAM)
- **Apache 2.0 license**: Free for commercial and personal use

Alternative models that work well:
- `llama3.2:latest` - Good general-purpose alternative
- `mistral:7b-instruct` - Fast, good for simple extraction
- `phi3:medium` - Smaller, faster, decent quality

## Usage

### Ingestion

```bash
python cli.py ingest "I started learning Korean today. My goal is to be conversational by March for my Seoul trip."
```

### Consolidation

```bash
python cli.py consolidate --topic "language_learning"
python cli.py consolidate --all  # consolidate all topics
```

### Retrieval

```bash
# Semantic lookup
python cli.py query "What am I learning right now?"

# Narrative recall
python cli.py recall "What have I said about learning Korean?"
```

### Interactive Demo

```bash
python cli.py demo
```

### Evaluation

Run evaluation metrics on the pipeline:

```bash
# Run the diary evaluation scenario
python cli.py eval --scenario diary

# With custom precision@k
python cli.py eval --scenario diary --k 10

# Verbose output with conflict details
python cli.py eval --scenario diary --verbose
```

Available scenarios:
- `diary` - Personal diary entries over one week with multiple topics

### Doctor Command

Before running evaluations or debugging issues, use the `doctor` command to inspect system configuration:

```bash
python cli.py doctor
```

This command reports:
- **Bootstrap status**: Whether FAISS/SentenceTransformers init-order safeguards are active
- **LLM provider**: Active provider, model, temperature
- **Embedding provider**: Active provider, model, device, dimension
- **Vector store**: FAISS index type, dimension consistency
- **Evaluation readiness**: Warnings about mock providers and their impact
- **Suggested fixes**: Copy-pasteable shell commands to fix configuration issues

#### Dry-Run Mode

Use `--dry` for a safe, lightweight inspection that does NOT initialize any models, FAISS indices, or database connections:

```bash
python cli.py doctor --dry
```

Dry-run mode is useful when:

- You want to check configuration before installing dependencies
- You're debugging environment variable issues
- You want a quick config check without loading heavy models
- You're working in CI/CD or review environments

Example use cases:

- Debugging why evaluation metrics are zero or skipped
- Verifying provider selection before code reviews
- Understanding why retrieval results seem random (mock embeddings!)
- Getting copy-paste commands to fix misconfigurations

```bash
# Check with mock providers (full initialization)
python cli.py --mock doctor

# Check with real providers (default, full initialization)
python cli.py doctor

# Safe config-only inspection (no initialization)
python cli.py doctor --dry
```

## macOS: FAISS + SentenceTransformers Note

On macOS, there's a known interaction issue between FAISS (native C++ library) and SentenceTransformers (PyTorch + tokenizers). If these libraries are initialized in the wrong order, you may see segfaults during Python cleanup.

**This is handled automatically.** The bootstrap module (`src/bootstrap.py`) ensures proper initialization order:
1. Set `TOKENIZERS_PARALLELISM=false` before any imports
2. Load SentenceTransformers embedding model BEFORE importing FAISS
3. Only then import FAISS-related modules

**For CLI users:** No action needed. Just use `python cli.py` normally.

**For library users:** Import from `src.bootstrap` instead of importing modules directly:

```python
from src.bootstrap import get_components

components = get_components()
# components.database, components.embedding_provider, etc.
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for more technical details.

## Running Tests

```bash
# Run fast tests (default, no model downloads)
pytest

# Run all tests including slow embedding tests
pytest --run-slow

# Run only slow tests (embedding model tests)
pytest -m slow

# Run specific test file
pytest tests/test_pipeline.py -v
```

**Note:** Slow tests (`@pytest.mark.slow`) require downloading SentenceTransformer models. They're skipped by default for faster CI/local development.

## Project Structure

```
episodic-memory-pipeline/
├── README.md
├── ARCHITECTURE.md        # Design decisions & evaluation
├── requirements.txt
├── pytest.ini             # Test configuration
├── config.py              # Configuration management
├── schema.sql             # Database schema
├── cli.py                 # Command-line interface
├── src/
│   ├── bootstrap.py       # Initialization ordering (FAISS/ST workaround)
│   ├── models/            # Data models
│   │   ├── episode.py
│   │   ├── fact.py
│   │   └── summary.py
│   ├── embeddings/        # Embedding abstraction
│   │   └── interface.py
│   ├── llm/               # LLM abstraction (OpenAI, Ollama, mock)
│   │   └── interface.py
│   ├── storage/           # Persistence layer
│   │   ├── database.py
│   │   └── vector_store.py
│   ├── ingestion/         # Input processing
│   │   ├── classifier.py
│   │   ├── extractor.py
│   │   └── pipeline.py
│   ├── consolidation/     # Memory consolidation
│   │   ├── summarizer.py
│   │   └── fact_extractor.py
│   ├── retrieval/         # Query handling
│   │   ├── semantic.py
│   │   └── narrative.py
│   ├── evaluation/        # Metrics and evaluation
│   │   ├── metrics.py
│   │   └── runner.py
│   └── prompts/           # LLM prompt templates
│       └── templates.py
├── data/                  # SQLite + FAISS storage
└── tests/
    ├── conftest.py        # Test fixtures and markers
    └── test_*.py          # Test files
```

## License

MIT
