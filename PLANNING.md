# Wellbeing Data Foundation — Planning Document

## Project Overview

Build a **zero-hallucination, evidence-only wellbeing assistant** that:
1. Ingests Arabic wellbeing framework documents into a canonical structured database
2. Provides hybrid retrieval (entity + vector + graph) returning citeable evidence packets
3. Answers questions via Azure GPT-5 using ONLY retrieved evidence with mandatory citations
4. Refuses to answer when evidence is insufficient (`not_found=true`)

## Architecture (Three-Layer Retrieval)

```
Layer A — Canonical Structured Store (Truth Layer)
    └── PostgreSQL: Pillars → Core Values → Sub-values + Evidence + Source metadata

Layer B — Retrieval Layer (Evidence Layer)
    ├── Entity-first lookup (exact match on names)
    ├── Vector search (pgvector, Arabic embeddings)
    └── Graph expansion (edges table in Postgres; Neo4j optional later)

Layer C — GPT-5 Answering Layer (Interpretation Layer)
    ├── Muḥāsibī reasoning middleware (state machine)
    ├── Evidence-only prompts with structured JSON output
    └── External guardrails (CitationEnforcer, claim-to-evidence checker)
```

## Data Scope

- **5 Pillars**: Spiritual, Emotional, Intellectual, Physical, Social (ركائز الحياة الطيبة)
- **~15 Core Values** (القيم الكلية/الأمهات)
- **~60+ Sub-values** (القيم الجزئية/الأحفاد)
- **Hundreds of citations** (Quran verses, Hadith)
- **Arabic primary** (English nullable until translation workflow exists)

## Non-Negotiable Constraints

1. **No evidence → no answer**: If evidence is insufficient, return `not_found=true`
2. **Non-LLM retrieval and extraction**: LLM never "searches"; it only interprets evidence packets
3. **Every answer must include citations** mapped to `chunk_id` + `source_anchor`
4. **Hard programmatic guardrails** must reject uncited/unsupported outputs
5. **Do not proceed to next phase** until all tests pass and GitHub push is confirmed
6. **Never exceed 500 lines per file**; split modules immediately
7. **Pytest required** for all new features; include Expected + Edge + Failure cases

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (Python) |
| Database | PostgreSQL + pgvector |
| Graph | Edges table in Postgres (Neo4j optional later) |
| Embeddings | Multilingual model compatible with Arabic (configurable) |
| LLM | Azure OpenAI GPT-5 (config via env vars) |
| Secrets | `.env` local; Key Vault for production |

## Data Contracts (Do Not Change)

### Evidence Packet Schema
```json
{
  "chunk_id": "CH_000123",
  "entity_type": "pillar|core_value|sub_value",
  "entity_id": "SV001",
  "chunk_type": "definition|evidence|commentary",
  "text_ar": "...",
  "source_doc_id": "DOC_2025_10",
  "source_anchor": "para_412",
  "refs": [{"type":"quran|hadith|book","ref":"..."}]
}
```

### Final Response Schema
```json
{
  "listen_summary_ar": "string",
  "purpose": {"ultimate_goal_ar":"string","constraints_ar":["string"]},
  "path_plan_ar": ["string"],
  "answer_ar": "string",
  "citations": [{"chunk_id":"CH_000123","source_anchor":"para_412","ref":"..."}],
  "entities": [{"type":"pillar|core_value|sub_value","id":"SV001","name_ar":"..."}],
  "difficulty": "easy|medium|hard",
  "not_found": true,
  "confidence": "high|medium|low"
}
```

## Muḥāsibī Reasoning Middleware (State Machine)

| State | Type | Description |
|-------|------|-------------|
| LISTEN | non-LLM | Normalize Arabic, detect entities, produce `listen_summary_ar` |
| PURPOSE | GPT-5 | Output ultimate goal + constraints (must include evidence-only rules) |
| PATH | GPT-5 | Output short plan steps with prioritization |
| RETRIEVE | non-LLM | Produce evidence packets bundle |
| ACCOUNT | non-LLM | Enforce citation coverage, reject unsupported claims |
| INTERPRET | GPT-5 | Answer using only evidence packets with citations |
| REFLECT | GPT-5 | Consequence-aware reflection (no new claims; label general advice as "إرشاد عام") |
| FINALIZE | non-LLM | Validate schema + citations + claim checks |

## Phase Plan

- **Phase 0**: Repo + environment + CI + push verification
- **Phase 1**: Ingestion (DOCX → canonical JSON → Postgres)
- **Phase 2**: DB schema + chunk table + loader
- **Phase 3**: Hybrid retrieval (entity + vector + graph)
- **Phase 4**: Muḥāsibī middleware + strong verification + demo
- **Phase 5**: Azure provider abstraction

## References

- Arabic Framework: الإطار المرجعي اكتوبر 2025
- Data Infrastructure Strategy v4
- Azure OpenAI API documentation

