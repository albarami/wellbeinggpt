# Wellbeing Data Foundation

A zero-hallucination, evidence-only Arabic wellbeing assistant built on a canonical data infrastructure.

## Overview

This project implements a production-quality MVP that:

1. **Ingests** Arabic wellbeing framework documents (`.docx`) into a structured canonical database
2. **Retrieves** relevant evidence using hybrid search (entity + vector + graph)
3. **Answers** questions using Azure GPT-5 with strict evidence-only constraints
4. **Refuses** to answer when evidence is insufficient (no hallucination guarantee)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Muḥāsibī Reasoning Middleware                │
│  LISTEN → PURPOSE → PATH → RETRIEVE → ACCOUNT → INTERPRET →    │
│                     REFLECT → FINALIZE                          │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴─────────────────────────────────┐
│                     Evidence Retrieval Layer                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Entity-first│  │ Vector Top-K│  │ Graph Expansion     │   │
│  │ Lookup      │  │ (pgvector)  │  │ (edges table)       │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴─────────────────────────────────┐
│                  Canonical Structured Store                    │
│  ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌─────────────┐   │
│  │ Pillars │→ │CoreValues │→ │SubValues │→ │ Evidence    │   │
│  └─────────┘  └───────────┘  └──────────┘  └─────────────┘   │
│                      PostgreSQL + pgvector                     │
└───────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Azure OpenAI API access (GPT-5 deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/albarami/wellbeinggpt.git
cd wellbeinggpt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Run database migrations
# (instructions will be added after Phase 2)

# Start the API server
uvicorn apps.api.main:app --reload
```

### Running Tests

```bash
pytest -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/docx` | Ingest Arabic framework document |
| GET | `/ingest/runs/{run_id}` | Get ingestion run status |
| POST | `/ask` | Ask a question (returns answer + citations) |
| POST | `/search/vector` | Vector similarity search |
| POST | `/search/graph` | Graph traversal search |

## Project Structure

```
apps/api/
  main.py                         # FastAPI application
  routes/
    ingest.py                     # Ingestion endpoints
    ask.py                        # Query endpoints
  core/
    muhasibi_state_machine.py     # Reasoning middleware
    schemas.py                    # Pydantic models
  ingest/
    docx_reader.py                # DOCX parsing
    rule_extractor.py             # State machine extractor
    evidence_parser.py            # Quran/Hadith parsing
    validator.py                  # Validation gates
    loader.py                     # DB loader
    chunker.py                    # Chunk generation
    embedder.py                   # Embedding generation
  retrieve/
    normalize_ar.py               # Arabic normalization
    entity_resolver.py            # Entity matching
    sql_retriever.py              # SQL-based retrieval
    vector_retriever.py           # Vector search
    graph_retriever.py            # Graph expansion
    merge_rank.py                 # Result merging
  llm/
    gpt5_client_azure.py          # Azure OpenAI client
    prompts/                      # Prompt templates
  guardrails/
    citation_enforcer.py          # Citation validation
    claim_checker.py              # Claim-to-evidence check
    refusal_policy.py             # Refusal logic
tests/
  test_docx_extract.py
  test_evidence_parse.py
  test_retrieval.py
  test_guardrails.py
  test_end_to_end_ask.py
scripts/
  demo.py                         # Demo script
db/
  schema.sql                      # Database schema
  migrations/                     # Migration files
```

## Non-Negotiable Rules

1. **No evidence → no answer**: Returns `not_found=true` with refusal message
2. **Every answer has citations**: Mapped to `chunk_id` + `source_anchor`
3. **LLM never searches**: Only interprets pre-retrieved evidence packets
4. **Programmatic guardrails**: Reject uncited or unsupported outputs

## License

[TBD]

## Contributing

See [PLANNING.md](PLANNING.md) for architecture details and [TASK.md](TASK.md) for current tasks.

