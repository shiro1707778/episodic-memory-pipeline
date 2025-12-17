# Episodic Memory Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Cognitive-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Maintained-success?style=flat-square)

**A local-first cognitive architecture for AI agents.**

This is not just a vector database wrapper. It is a system that mimics human memory consolidation by separating **Episodic Memory** (raw, timestamped events) from **Semantic Memory** (consolidated, stable facts). It features defense-in-depth LLM sanitization, provenance tracking, and is optimized for multilingual (CJK) contexts using **Qwen-2.5** and **BGE-M3**.

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

1.  **Episodic memory ≠ vector blobs**: Each memory is a structured event with context, time, and meaning.
2.  **Time and provenance matter**: Every fact and summary links back to its source episodes. Hallucination prevention starts with lineage.
3.  **Memory must be curated, not accumulated**: Not everything is worth remembering. We filter aggressively via a "Memory Worthiness" gate.
4.  **Retrieval should feel like recalling a journey**: Narrative coherence over raw similarity scores.

## Storage Choice: SQLite + FAISS

**Why SQLite over Postgres?**
- Local-first, no server dependencies.
- Single-file portability (backup = copy file).
- JSON1 extension for flexible metadata.
- Zero configuration required.

**Why FAISS for vectors?**
- Mature, fast, local-only C++ library.
- Supports multiple index types for scaling.
- Works well alongside SQLite for hybrid retrieval.

## Core Concepts

### 1. Episode (Episodic Memory)
A timestamped event capturing what happened, when, and in what context.
> *"On Tuesday at 3pm, I told my assistant I'm learning Korean for a trip to Seoul in March."*

### 2. Fact (Semantic Memory)
A distilled, stable piece of knowledge extracted from episodes.
> *"User is learning Korean. User has a trip to Seoul planned for March 2024."*

### 3. Summary (Consolidated Narrative)
A topic-level summary that weaves together multiple episodes into a coherent narrative.
> *"User's Korean language learning journey: Started in January 2024 motivated by upcoming Seoul trip..."*

## Installation

```bash
git clone https://github.com/wheevu/episodic-memory-pipeline.git
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
```

### Device Selection
For faster embeddings on GPU or Apple Silicon:
```bash
# Apple Silicon (Metal)
export EMBEDDING_DEVICE=mps

# NVIDIA GPU
export EMBEDDING_DEVICE=cuda
```

### Why BGE-M3?
**BAAI/bge-m3** is the recommended embedding model because:
- **High quality**: State-of-the-art multilingual embeddings.
- **Dense retrieval optimized**: Designed for semantic search.
- **Multilingual**: Excellent for personal memory with mixed languages (Vietnamese/Korean/English).
- **1024 dimensions**: Good balance of quality vs storage.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed model comparison.

## Local-First Setup with Qwen (Recommended)

For fully local operation without API dependencies, use Ollama with Qwen:

1.  **Install Ollama:** [ollama.com](https://ollama.com)
2.  **Pull the Model:** `ollama pull qwen2.5:7b-instruct`
3.  **Start Server:** `ollama serve`
4.  **Configure Env:**
    ```bash
    export LLM_PROVIDER=ollama
    export OLLAMA_MODEL=qwen2.5:7b-instruct
    ```

**Why Qwen 2.5?**
It offers superior instruction following for JSON output and excellent multilingual support (Asian languages) compared to Llama 3 or Mistral, making it ideal for robust information extraction.

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
# Semantic lookup (Fact-based)
python cli.py query "What am I learning right now?"

# Narrative recall (Episode-based)
python cli.py recall "What have I said about learning Korean?"
```

### Interactive Demo
```bash
python cli.py demo
```

### Evaluation
Run custom metrics (Conflict Rate, Compression Ratio) on the pipeline:
```bash
# Run the diary evaluation scenario
python cli.py eval --scenario diary --verbose
```

## Reliability & Diagnostics

### The `doctor` Command
Before running evaluations or debugging issues, use the built-in diagnostic tool to inspect system configuration:

```bash
python cli.py doctor
```

This command reports:
- **Bootstrap status**: Verifies the FAISS/SentenceTransformers init-order safeguards (prevents macOS segfaults).
- **Provider Status**: Active provider, model, temperature, and device mapping.
- **Vector Store**: FAISS index type and dimension consistency.
- **Suggested fixes**: Copy-pasteable shell commands to fix configuration issues.

#### Dry-Run Mode
Use `--dry` for a safe, lightweight inspection that does NOT initialize any models (CI/CD friendly):
```bash
python cli.py doctor --dry
```

## Running Tests

```bash
# Run fast tests (default)
pytest

# Run all tests including slow embedding model downloads
pytest --run-slow
```

## Project Structure

```
episodic-memory-pipeline/
├── README.md
├── ARCHITECTURE.md        # Design decisions & evaluation
├── requirements.txt
├── config.py              # Configuration management
├── schema.sql             # Database schema
├── cli.py                 # Command-line interface
├── src/
│   ├── bootstrap.py       # Initialization ordering (FAISS/ST workaround)
│   ├── models/            # Data models
│   ├── embeddings/        # Embedding abstraction
│   ├── llm/               # LLM abstraction
│   ├── storage/           # Persistence layer (SQLite + FAISS)
│   ├── ingestion/         # Input processing
│   ├── consolidation/     # Memory consolidation logic
│   ├── retrieval/         # Query handling
│   ├── evaluation/        # Metrics and evaluation
│   └── prompts/           # LLM prompt templates
└── tests/
```

## License

MIT
