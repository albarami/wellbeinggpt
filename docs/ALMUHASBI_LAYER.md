# Al-Muḥāsibī (المحاسبي) Reasoning Layer

## Technical Architecture Documentation

**Version:** 1.0
**Last Updated:** December 2024
**Author:** WellbeingGPT Engineering Team

---

## Table of Contents

1. [Overview](#overview)
2. [Etymology and Design Philosophy](#etymology-and-design-philosophy)
3. [Architecture](#architecture)
4. [The 8-State Pipeline](#the-8-state-pipeline)
5. [Core Components](#core-components)
6. [Safety Mechanisms](#safety-mechanisms)
7. [LLM Integration](#llm-integration)
8. [API Integration](#api-integration)
9. [Traceability](#traceability)
10. [Testing Strategy](#testing-strategy)
11. [Configuration](#configuration)
12. [File Reference](#file-reference)

---

## Overview

The **Al-Muḥāsibī Layer** is a deterministic state machine middleware that implements an 8-state reasoning pipeline for evidence-only question answering in Arabic. It serves as the core reasoning engine for the WellbeingGPT system, ensuring that every response is backed by verifiable evidence from the canonical knowledge base.

### Key Characteristics

- **Zero-Hallucination Guarantee**: Every claim must be backed by evidence; unevidenced questions are gracefully refused
- **Deterministic Pipeline**: Reproducible state transitions with predictable behavior
- **Arabic-First Design**: Native support for Arabic text normalization and morphology-aware processing
- **Fail-Closed Safety**: When in doubt, refuse rather than risk misinformation
- **LLM-Optional**: Critical paths work without LLM availability through deterministic fallbacks

---

## Etymology and Design Philosophy

The name **Muḥāsibī** (محاسبي) derives from the Arabic root **ح-س-ب** (ḥ-s-b), meaning "to reckon" or "to account." In Islamic tradition, **muḥāsaba** (محاسبة) refers to self-accountability and spiritual self-examination—the practice of holding oneself accountable before being held accountable.

This naming reflects the system's core philosophy:

1. **Accountability**: Every answer must account for its sources
2. **Self-Examination**: The system examines its own reasoning at each state
3. **Reckoning**: Evidence is "reckoned" against claims before any response
4. **Spiritual Alignment**: Respects the Islamic intellectual tradition of careful verification

The design draws inspiration from classical Islamic scholarly methodology (تحقيق العلم), where claims require chains of evidence (إسناد) and scholars refuse to speak without knowledge.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Al-Muḥāsibī Reasoning Middleware                     │
│                                                                          │
│   ┌────────┐   ┌─────────┐   ┌──────┐   ┌──────────┐   ┌─────────┐     │
│   │ LISTEN │ → │ PURPOSE │ → │ PATH │ → │ RETRIEVE │ → │ ACCOUNT │     │
│   └────────┘   └─────────┘   └──────┘   └──────────┘   └─────────┘     │
│                                                               │          │
│   ┌──────────┐   ┌─────────┐   ┌──────────┐                  │          │
│   │ FINALIZE │ ← │ REFLECT │ ← │INTERPRET │ ←────────────────┘          │
│   └──────────┘   └─────────┘   └──────────┘                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Supporting Systems                                │
│                                                                          │
│   ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐    │
│   │ Entity Resolver │  │ Hybrid Retriever │  │ Muḥāsibī LLM Client│    │
│   │ (Arabic NLP)    │  │ (Entity+Vec+Graph)│  │ (GPT-5 Structured) │    │
│   └─────────────────┘  └──────────────────┘  └────────────────────┘    │
│                                                                          │
│   ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐    │
│   │ Guardrails      │  │ Arabic Normalizer│  │ Structure Answerer │    │
│   │ (Citation Check)│  │ (Morphology)     │  │ (Deterministic)    │    │
│   └─────────────────┘  └──────────────────┘  └────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The 8-State Pipeline

### State 1: LISTEN (الاستماع)

**Purpose**: Understand the question through Arabic-first meaning extraction.

**Operations**:
- Normalize Arabic text (remove diacritics, normalize hamza/alef variants)
- Extract Arabic keywords using morphology-aware tokenization
- Detect entities (pillars, core values, sub-values) via the Entity Resolver
- Classify intent (optional LLM, with deterministic fallback)
- Generate Arabic listen summary

**Outputs**:
- `normalized_question`: Cleaned Arabic text
- `question_keywords`: Extracted meaningful terms
- `detected_entities`: Resolved framework entities
- `intent`: Classified question intent
- `listen_summary_ar`: Human-readable summary

**Deterministic Intent Detection**:
```python
# Example: Detecting "list pillars" intent
if ("ركائز" in q or "اركان" in q) and ("الخمس" in q or "خمسة" in q):
    intent_type = "list_pillars"
```

---

### State 2: PURPOSE (الغاية)

**Purpose**: Establish the ultimate goal and constraints for answering.

**Operations**:
- Generate structured purpose via LLM (or use deterministic fallback)
- Enforce required constraints: `evidence_only`, `cite_every_claim`, `refuse_if_missing`
- Extract initial path plan if provided by LLM

**Outputs**:
- `purpose.ultimate_goal_ar`: The goal in Arabic
- `purpose.constraints_ar`: List of constraints (always includes required three)

**Required Constraints** (Non-negotiable):
```python
REQUIRED_CONSTRAINTS = [
    "evidence_only",      # Only use retrieved evidence
    "cite_every_claim",   # Every claim needs a citation
    "refuse_if_missing",  # Refuse if evidence is insufficient
]
```

---

### State 3: PATH (المسار)

**Purpose**: Plan the approach based on question difficulty.

**Operations**:
- Assess difficulty based on detected entities count
- Generate step-by-step plan in Arabic

**Difficulty Assessment**:
| Entity Count | Difficulty | Reasoning |
|--------------|------------|-----------|
| 0 entities   | HARD       | No anchor points; requires broad search |
| 1 entity     | MEDIUM     | Single anchor; focused retrieval |
| 2+ entities  | EASY       | Multiple anchors; precise retrieval |

**Default Path Plan**:
```arabic
1. استخراج الكيانات المذكورة في السؤال
2. استرجاع التعريفات والأدلة من قاعدة البيانات
3. التحقق من تغطية الأدلة للسؤال
4. صياغة الإجابة مع الاستشهادات
```

---

### State 4: RETRIEVE (الاسترجاع)

**Purpose**: Gather evidence packets from the knowledge base.

**Operations**:
1. Execute hybrid retrieval (entity + vector + graph)
2. If insufficient results, use LLM for Arabic query rewriting
3. Execute additional retrieval with rewritten queries
4. Merge and deduplicate evidence packets

**Evidence Packet Structure**:
```python
{
    "chunk_id": "uuid",
    "chunk_type": "definition" | "evidence",
    "text_ar": "Arabic text content",
    "source_anchor": "Source reference",
    "entity_type": "pillar" | "core_value" | "sub_value",
    "entity_id": "uuid",
    "refs": [{"ref": "Quran 2:45", ...}]
}
```

**Query Rewriting** (LLM-assisted, evidence-only):
The LLM generates alternative Arabic phrasings to improve retrieval recall. It does NOT answer the question—only suggests search terms.

---

### State 5: ACCOUNT (المحاسبة)

**Purpose**: Validate evidence coverage and enforce scope boundaries.

**Operations**:
1. Check evidence existence
2. Validate relevance (lexical matching between question and evidence)
3. Detect out-of-scope questions (fiqh rulings, general knowledge)
4. Provide graceful reframing suggestions for out-of-scope questions

**Out-of-Scope Detection**:

The system refuses **fiqh ruling questions** (فتوى) that ask for Islamic legal verdicts:

```python
fiqh_markers = ["ما حكم", "حكم", "يجوز", "حلال", "حرام", ...]
worship_terms = ["صيام", "صلاة", "زكاة", "حج", ...]

# If both present, question is fiqh-related
if contains_fiqh_markers AND contains_worship_terms:
    refuse_with_reframing_suggestion()
```

**Reframing Example**:
```
Question: "ما حكم صيام يوم الجمعة؟" (What is the ruling on Friday fasting?)
Refusal: "لا أستطيع إصدار فتوى/حكم شرعي من هذا النظام."
Suggestion: "بديل داخل النطاق: ما أثر الصيام كعبادة على تزكية النفس والطاعة ضمن إطار الحياة الطيبة؟"
```

---

### State 6: INTERPRET (التفسير)

**Purpose**: Generate an answer from evidence packets.

**Operations**:
1. **Deterministic Structure Answering**: For list intents, generate answer directly from database structure (no LLM needed)
2. **LLM Interpretation**: For complex questions, use mode-aware interpretation
3. **Deterministic Fallback**: If LLM fails, synthesize answer from definitions and evidence
4. **Guardrails Validation**: Verify citations are valid
5. **Citation Building**: Construct citation objects from evidence packets

**Answer Modes**:
| Mode | Description | Use Case |
|------|-------------|----------|
| `answer` | Direct, informative response | Default Q&A |
| `debate` | Present multiple perspectives | Comparative analysis |
| `socratic` | Guide through questions | Educational contexts |
| `judge` | Evaluate claims against evidence | Verification requests |

**Deterministic Structure Answering**:
For questions like "ما هي ركائز الحياة الطيبة الخمس؟", the system:
1. Detects `list_pillars` intent
2. Queries database directly for pillar names
3. Builds response with citations to heading chunks
4. Returns 100% accurate, citeable answer without LLM

---

### State 7: REFLECT (التأمل)

**Purpose**: Add consequence-aware reflection (currently conservative/placeholder).

**Current Behavior**: Minimal processing; placeholder for future enhancement.

**Future Potential**:
- Add spiritual/ethical implications
- Connect to broader wellbeing framework
- Provide actionable guidance

---

### State 8: FINALIZE (الإنهاء)

**Purpose**: Validate schema and build final response.

**Output Schema** (`FinalResponse`):
```python
@dataclass
class FinalResponse:
    listen_summary_ar: str      # LISTEN output
    purpose: Purpose            # PURPOSE output
    path_plan_ar: list[str]     # PATH output
    answer_ar: str              # INTERPRET output
    citations: list[Citation]   # Evidence citations
    entities: list[EntityRef]   # Detected entities
    difficulty: Difficulty      # Question difficulty
    not_found: bool             # True if no evidence
    confidence: Confidence      # HIGH/MEDIUM/LOW
```

---

## Core Components

### StateContext

The `StateContext` dataclass carries all state between pipeline stages:

```python
@dataclass
class StateContext:
    # Input
    question: str
    language: str = "ar"
    mode: str = "answer"  # answer|debate|socratic|judge

    # LISTEN outputs
    normalized_question: str
    listen_summary_ar: str
    detected_entities: list[dict]
    question_keywords: list[str]
    intent: Optional[dict]

    # PURPOSE outputs
    purpose: Optional[Purpose]

    # PATH outputs
    path_plan_ar: list[str]
    difficulty: Difficulty

    # RETRIEVE outputs
    evidence_packets: list[dict]
    has_definition: bool
    has_evidence: bool

    # ACCOUNT outputs
    citation_valid: bool
    account_issues: list[str]
    refusal_suggestion_ar: Optional[str]

    # INTERPRET outputs
    answer_ar: str
    citations: list[Citation]
    entities: list[EntityRef]
    not_found: bool
    confidence: Confidence

    # REFLECT outputs
    reflection_added: bool

    # Error handling & timing
    error: Optional[str]
    retry_count: int
    state_timings: dict[str, float]
    trace: list[dict]
```

### MuhasibiMiddleware

The main orchestrator class:

```python
class MuhasibiMiddleware:
    def __init__(
        self,
        entity_resolver=None,   # Arabic entity detection
        retriever=None,         # Hybrid evidence retrieval
        llm_client=None,        # GPT-5 structured outputs
        guardrails=None,        # Citation validation
    ):
        ...

    async def process(self, question: str, language: str, mode: str) -> FinalResponse:
        """Single-pass pipeline execution."""
        ...

    async def process_with_trace(self, ...) -> tuple[FinalResponse, list[dict]]:
        """Pipeline execution with state transition tracing."""
        ...
```

---

## Safety Mechanisms

### 1. Evidence-Only Constraint

Every claim in the answer must derive from retrieved evidence packets. The LLM is explicitly instructed:

```
NEVER add information not present in the evidence packets.
ONLY synthesize and explain what the evidence contains.
```

### 2. Citation Enforcement

If `not_found=False` (we claim to have an answer), the response MUST have citations:

```python
if not ctx.not_found:
    if not ctx.citations and ctx.evidence_packets:
        # Build citations from evidence packets
        ctx.citations = [Citation(...) for p in chosen_packets]
```

### 3. Automatic Refusal

When evidence is insufficient, the system refuses gracefully:

```python
if not ctx.evidence_packets:
    ctx.not_found = True
    ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
```

### 4. Relevance Gating

Even with evidence, the system validates relevance:

```python
# If no question terms match evidence content
if matched_terms == 0:
    ctx.not_found = True
    ctx.account_issues.append("السؤال خارج نطاق البيانات المتاحة")
```

### 5. Fiqh Ruling Detection

Islamic jurisprudence questions are detected and refused with helpful alternatives:

```python
if _is_fiqh_ruling_question(question):
    ctx.not_found = True
    ctx.refusal_suggestion_ar = "بديل داخل النطاق: ..."
```

### 6. Guardrails Validation

Post-generation validation ensures answer quality:

```python
if self.guardrails:
    result = self.guardrails.validate(
        answer_ar=ctx.answer_ar,
        citations=ctx.citations,
        evidence_packets=ctx.evidence_packets,
    )
    if not result.passed:
        ctx.not_found = True  # Fail closed
```

### 7. Deterministic Fallback

Critical paths work without LLM:

```python
# If no LLM but evidence exists
if not ctx.answer_ar and ctx.evidence_packets:
    # Build answer from definitions + evidence directly
    parts = []
    if definitions:
        parts.append(f"التعريف:\n{definition_text}")
    if evidence:
        parts.append(f"التأصيل/الدليل:\n{evidence_text}")
    ctx.answer_ar = "\n\n".join(parts)
```

---

## LLM Integration

### MuhasibiLLMClient

Wraps the LLM provider for structured JSON outputs:

```python
class MuhasibiLLMClient:
    async def purpose_path(question) -> PurposePathResult
    async def interpret(question, evidence, entities, mode) -> InterpretResult
    async def query_rewrite_ar(question, entities, keywords) -> dict
    async def classify_intent_ar(question, entities, keywords) -> dict
```

### JSON Schemas

All LLM outputs use strict JSON schemas:

**Purpose/Path Schema**:
```json
{
    "purpose": {
        "ultimate_goal_ar": "string",
        "constraints_ar": ["string"]
    },
    "path_plan_ar": ["string"],
    "difficulty": "easy|medium|hard"
}
```

**Interpreter Schema**:
```json
{
    "answer_ar": "string",
    "citations": [{"chunk_id": "string", "source_anchor": "string", "ref": "string|null"}],
    "entities": [{"type": "string", "id": "string", "name_ar": "string"}],
    "not_found": "boolean",
    "confidence": "high|medium|low"
}
```

### Mode-Aware Prompts

Different prompts for different interaction modes:
- `interpreter.md` - Default answer mode
- `interpreter_debate_ar.md` - Debate/comparative mode
- `interpreter_socratic_ar.md` - Socratic questioning mode
- `interpreter_judge_ar.md` - Claim verification mode

---

## API Integration

### Endpoints

**POST `/ask`** - Question answering with full Muḥāsibī pipeline:
```python
class AskRequest(BaseModel):
    question: str
    language: str = "ar"
    mode: str = "answer"  # answer|debate|socratic|judge
    engine: str = "hybrid"
```

**POST `/ask/trace`** - Same as above with state transition trace:
```python
class AskTraceResponse(BaseModel):
    response: FinalResponse
    trace: list[TraceEvent]
```

### Error Handling

The API implements fail-closed behavior:

```python
def _fail_closed_if_invalid(resp: FinalResponse) -> FinalResponse:
    """Enterprise stability: If answer is invalid, return safe refusal."""
    if not resp.not_found and not resp.answer_ar.strip():
        return FinalResponse(
            not_found=True,
            answer_ar="لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال.",
            ...
        )
    return resp
```

---

## Traceability

### Safe Trace Generation

The trace system exposes high-level state transitions without revealing internal reasoning:

```python
def summarize_state(state_name: str, ctx) -> dict:
    """Build safe trace snapshot."""
    base = {
        "state": state_name,
        "mode": ctx.mode,
        "language": ctx.language,
        "elapsed_s": elapsed_time,
    }

    # State-specific additions (counts, not content)
    if state_name == "LISTEN":
        base["detected_entities_count"] = len(ctx.detected_entities)
        base["keywords_count"] = len(ctx.question_keywords)
    elif state_name == "RETRIEVE":
        base["evidence_packets_count"] = len(ctx.evidence_packets)
    ...

    return base
```

### What IS Exposed

- State names and transition order
- Timing per state
- Entity/keyword counts (not content)
- Confidence levels
- Issue flags

### What is NOT Exposed

- Internal chain-of-thought
- LLM reasoning traces
- Evidence packet full text
- Intermediate prompts

---

## Testing Strategy

### Unit Tests

- `test_muhasibi_trace.py` - Trace generation safety
- `test_modes_prompt_selection.py` - Mode-specific prompt selection
- `test_query_rewrite_llm.py` - Query rewriting functionality

### Integration Tests

- `test_end_to_end_ask.py` - Full pipeline behavior
  - In-corpus question flow
  - Out-of-corpus refusal behavior
  - State transitions
  - Entity resolution

### Test Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| State Machine | 10 | LISTEN, PURPOSE, PATH, ACCOUNT, INTERPRET |
| State Transitions | 3 | Pipeline order, error handling |
| End-to-End | 2 | In-corpus, out-of-corpus scenarios |
| Trace Safety | 2 | No internal exposure |
| Mode Selection | 4 | All four modes |

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-5

# Feature Flags
ENABLE_INTENT_CLASSIFIER=1  # Enable LLM intent classification

# Database
DATABASE_URL=postgresql://...
```

### Middleware Initialization

```python
middleware = MuhasibiMiddleware(
    entity_resolver=EntityResolver(session),
    retriever=HybridRetriever(session),
    llm_client=MuhasibiLLMClient(llm_provider),
    guardrails=CitationGuardrails(),
)
```

---

## File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `muhasibi_state_machine.py` | Main orchestrator, state definitions | ~450 |
| `muhasibi_listen.py` | LISTEN state implementation | ~110 |
| `muhasibi_account.py` | ACCOUNT state, relevance gating | ~140 |
| `muhasibi_interpret.py` | INTERPRET/REFLECT states | ~230 |
| `muhasibi_structure_answer.py` | Deterministic list answering | ~110 |
| `muhasibi_trace.py` | Safe trace generation | ~70 |
| `muhasibi_llm_client.py` | LLM integration | ~380 |

---

## Summary

The Al-Muḥāsibī Layer represents a principled approach to AI-assisted Islamic knowledge systems:

1. **Evidence-First**: No answer without evidence
2. **Traceable**: Every response has citations
3. **Accountable**: Named after the principle of accountability
4. **Graceful**: Refuses politely with helpful alternatives
5. **Robust**: Works with or without LLM availability
6. **Respectful**: Honors Islamic scholarly tradition of verification

The 8-state pipeline ensures that questions flow through structured understanding, planning, retrieval, validation, interpretation, and reflection—mirroring the thoughtful process a human scholar would follow when answering questions about Islamic wellbeing.

---

*This document describes the Al-Muḥāsibī Reasoning Layer as implemented in WellbeingGPT v1.0.*
