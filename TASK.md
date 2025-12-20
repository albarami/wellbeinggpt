# Task Tracking

## Current Sprint: Wellbeing Data Foundation MVP

### Active Tasks

| ID | Task | Status | Date Started | Date Completed |
|----|------|--------|--------------|----------------|
| D1-1 | Ingest missing sections from image-based pages (OCR from DOCX media) and re-run ingestion | ✅ Complete | 2025-12-12 | 2025-12-13 |
| E1-1 | Build deterministic end-to-end evaluation harness (datasets, modes, scoring, reports, pytest gates) | ✅ Complete | 2025-12-14 | 2025-12-14 |
| S2-1 | Scholar-level depth upgrade (Scholar Notes Pack + typed edge-grounding + scholar reasoning + eval gates + tuning) | ✅ Complete | 2025-12-14 | 2025-12-15 |
| S2-2 | Stage 2 depth: raise Deep rubric/claims + lift Gold QA rubric via light-deep routing (no safety regressions) | ✅ Complete | 2025-12-15 | 2025-12-15 |
| S3-1 | Framework semantic edge miner: extract grounded cross-pillar SCHOLAR_LINK edges from framework chunks (no new sources) | ✅ Complete | 2025-12-16 | 2025-12-16 |
| S3-2 | Scale framework semantic edges + multi-span stitching + richer bridge notes v2 (no new sources) | ✅ Complete | 2025-12-16 | 2025-12-16 |
| S3-3 | Add argument-chain trace + boundary mining upgrades + stakeholder_acceptance_v2_hard (framework-only) | ✅ Complete | 2025-12-16 | 2025-12-16 |
| S3-4 | Add NATURAL_SCHOLAR_CHAT ("natural_chat") mode: narrative interpreter prompt + runtime switch + stakeholder eval wiring (no safety regressions) | ✅ Complete | 2025-12-16 | 2025-12-16 |
| UI1-1 | Build production Scholar Chat + Evidence + Graph UI (Next.js in apps/web) + backend /ask/ui wrapper + grounded/capped graph hooks + replay/feedback loop | 🔄 In Progress | 2025-12-16 |  |
| DOC1-2 | Restore + enhance `QURANIC_BEHAVIOR_CLASSIFICATION_MATRIX.md` using Bouzidani paper (fitrah/basira, objective ethics, 3-world model, action/niyyah taxonomy, measurement instruments) | ✅ Complete | 2025-12-20 | 2025-12-20 |
| DOC1-3 | Add “لا…ولكن/لكنهم” negation-contrast guidance + analytic crosswalk appendix + tighten periodicity wording + enforce behavior-concept assertions | ✅ Complete | 2025-12-20 | 2025-12-20 |
| DOC1-4 | Add worked examples + stronger warnings for research-grade axes in `QURANIC_BEHAVIOR_CLASSIFICATION_MATRIX.md` | ✅ Complete | 2025-12-20 | 2025-12-20 |

### Discovered During Work

| ID | Task | Status | Date Started | Date Completed |
|----|------|--------|--------------|----------------|
| D1-1 | Ingest missing sections from image-based pages (OCR from DOCX media) and re-run ingestion | ✅ Complete | 2025-12-12 | 2025-12-13 |
| D1-2 | Replace Azure Search dependency with local BM25 vector retrieval (no external services) and update tests to run with no skips | ✅ Complete | 2025-12-14 | 2025-12-14 |
| D1-3 | Generate proof artifacts (coverage/vector/graph/ask/A-B) and add local embeddings (no Azure) so coverage is fully provable | ✅ Complete | 2025-12-14 | 2025-12-14 |
| E1-1a | Define eval JSONL schema (claims with support policy, canonical citations) + deterministic logger | ✅ Complete | 2025-12-14 | 2025-12-14 |
| E1-1b | Add canonical citation span support (precomputed chunk sentence spans; validate not discover) | ✅ Complete | 2025-12-14 | 2025-12-14 |
| E1-1c | Deterministic dataset generator (gold/cross-pillar/negative/injection/mixed) + golden slice overrides | ✅ Complete | 2025-12-14 | 2025-12-14 |
| E1-1d | Deterministic runner with comparable modes (LLM_ONLY_UNGROUNDED, LLM_ONLY_SAFE, RAG_ONLY, RAG_PLUS_GRAPH, FULL_SYSTEM) | ✅ Complete | 2025-12-14 | 2025-12-14 |
| E1-1e | Scorers + hard gates + report generation + make targets (incl. policy audit) | ✅ Complete | 2025-12-14 | 2025-12-14 |

---

## Completed Tasks

| ID | Task | Status | Date Started | Date Completed |
|----|------|--------|--------------|----------------|
| P0-1 | Initialize repo + governance files + git push | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P0-2 | Add pytest smoke test | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P1-1 | Implement DOCX parser + state-machine extractor | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P1-2 | Implement Quran/Hadith ref normalization | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P1-3 | Add Phase 1 tests | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P2-1 | Create Postgres schema + migrations + loader | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P2-2 | Add chunk table | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P2-3 | Add Phase 2 DB tests | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P3-1 | Implement hybrid retrieval layer | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P3-2 | Add Phase 3 retrieval tests | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P4-1 | Implement Muḥāsibī state machine | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P4-2 | Implement strong verification guardrails | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P4-3 | Add GPT-5 prompt files | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P4-4 | Add Phase 4 tests | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P4-5 | Create demo script | ✅ Complete | 2025-12-12 | 2025-12-12 |
| P5-1 | Implement Azure provider abstraction | ✅ Complete | 2025-12-12 | 2025-12-12 |

---

## Implementation Summary

### Phase 0: Repository Setup ✅
- PLANNING.md, TASK.md, README.md created
- Git initialized with remote: https://github.com/albarami/wellbeinggpt.git
- Smoke test added to prevent "no tests collected" failures

### Phase 1: Ingestion Pipeline ✅
- DOCX parser with stable source anchors (para_index + text hash fallback)
- State-machine extractor for Arabic framework structure
- Quran/Hadith reference parsing with ref_raw + ref_norm
- Validation gates with warnings and errors

### Phase 2: Database Layer ✅
- PostgreSQL schema with pillars, core_values, sub_values, evidence, chunks
- Canonical JSON export/import
- Provenance fields for edges (created_method, status, score)

### Phase 3: Hybrid Retrieval ✅
- Entity resolver with Arabic normalization
- SQL, vector, and graph retrievers
- Merge/rank policy producing evidence packets

### Phase 4: Muḥāsibī Middleware + Guardrails ✅
- 8-state machine: LISTEN→PURPOSE→PATH→RETRIEVE→ACCOUNT→INTERPRET→REFLECT→FINALIZE
- CitationEnforcer, Evidence-ID verifier, claim-to-evidence checker
- GPT-5 prompts with structured JSON output requirements
- Demo script showing in-corpus and out-of-corpus behavior

### Phase 5: Azure Provider Abstraction ✅
- Azure OpenAI Responses API (preferred)
- Azure Chat Completions API (fallback)
- Configuration via environment variables (no hardcoding)
- Mock provider for testing

---

## Test Summary

- **131 tests passed**, 5 skipped (integration tests requiring live DB/DOCX)
- All phase gates cleared with `pytest -v`

---

## Notes

- All tasks completed with tests passing ✅
- Git commits pushed after each phase completion ✅
- 500-line-per-file limit followed ✅
