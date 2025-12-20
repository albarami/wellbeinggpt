# Muḥāsibī A/B Evaluation Report

**Generated**: 2025-12-16 22:18:56 UTC
**Breakthrough Mode**: Disabled

## Summary by Mode

| Mode | N | unsup_must_cite | cite_errors | PASS_FULL | used_edges | arg_chains | quarantine | p50 ms | p95 ms |
|------|--:|----------------:|------------:|----------:|-----------:|-----------:|-----------:|-------:|-------:|
| LLM_ONLY_UNGROUNDED | 5 | 0.000 | 0 | 0.0% | 0.00 | 0.00 | 0 | 4371 | 4555 |
| RAG_ONLY | 5 | 0.000 | 0 | 0.0% | 0.00 | 0.00 | 0 | 254 | 272 |
| RAG_ONLY_INTEGRITY | 5 | 0.000 | 0 | 0.0% | 0.00 | 0.00 | 0 | 255 | 256 |
| RAG_PLUS_GRAPH | 5 | 0.000 | 0 | 0.0% | 0.00 | 0.00 | 0 | 631 | 633 |
| RAG_PLUS_GRAPH_INTEGRITY | 5 | 0.000 | 0 | 0.0% | 0.00 | 0.00 | 0 | 635 | 636 |
| FULL_SYSTEM | 5 | 0.000 | 0 | 0.0% | 7.20 | 7.20 | 0 | 8333 | 12721 |

## Style Metrics

| Mode | Quote Count | Paragraph Count | Reasoning Leak Rate |
|------|------------:|----------------:|--------------------:|
| LLM_ONLY_UNGROUNDED | 0.60 | 7.00 | 0.0000 |
| RAG_ONLY | 0.00 | 3.00 | 0.0000 |
| RAG_ONLY_INTEGRITY | 0.00 | 3.00 | 0.0000 |
| RAG_PLUS_GRAPH | 0.00 | 3.00 | 0.0000 |
| RAG_PLUS_GRAPH_INTEGRITY | 0.00 | 3.00 | 0.0000 |
| FULL_SYSTEM | 6.00 | 8.20 | 0.0000 |

## Attribution Analysis (Clean Deltas)

| Comparison | unsup_must_cite | cite_errors | quarantine | PASS_FULL | used_edges | arg_chains | p50_ms | p95_ms |
|------------|-----------------|-------------|------------|-----------|------------|------------|--------|--------|
| RAG_ONLY_INT - RAG_ONLY | +0.000 (+0.0%) | - | +0 | +0.000 (+0.0%) | +0.000 (+0.0%) | +0.000 (+0.0%) | +-14ms | - |
| RAG_GRAPH_INT - RAG_GRAPH | +0.000 (+0.0%) | - | +0 | +0.000 (+0.0%) | +0.000 (+0.0%) | +0.000 (+0.0%) | +2ms | - |
| FULL - RAG_ONLY | +0.000 (+0.0%) | - | +0 | +0.000 (+0.0%) | +7.200 (+100.0%) | +7.200 (+100.0%) | +10053ms | - |
| FULL - RAG_PLUS_GRAPH | +0.000 (+0.0%) | - | +0 | +0.000 (+0.0%) | +7.200 (+100.0%) | +7.200 (+100.0%) | +9690ms | - |
| FULL - RAG_GRAPH_INT (Muḥāsibī) | +0.000 (+0.0%) | - | +0 | +0.000 (+0.0%) | +7.200 (+100.0%) | +7.200 (+100.0%) | +9688ms | - |

## Value-Add per Millisecond

| Effect | Metric | Delta | Latency Delta | Value/ms |
|--------|--------|-------|---------------|----------|
| Muḥāsibī | mean_used_edges | +7.200 (+100.0%) | +9688ms | 0.000743 |
| Muḥāsibī | mean_argument_chains | +7.200 (+100.0%) | +9688ms | 0.000743 |
| Total | mean_used_edges | +7.200 (+100.0%) | +10053ms | 0.000716 |
| Total | mean_argument_chains | +7.200 (+100.0%) | +10053ms | 0.000716 |

## Top 10 Quality Improvements (Baseline → Full)

| ID | Question | Edges | Citations | Quality Delta |
|----|----------|-------|-----------|---------------|
| gold-0003 | عرّف الحياة العاطفية كما ورد في الإطار، واذكر نصًا مُستشهدًا... | 0→12 | 8→17 | +33 |
| gold-0001 | عرّف الحياة الروحية كما ورد في الإطار، واذكر نصًا مُستشهدًا ... | 0→12 | 8→16 | +32 |
| gold-0005 | عرّف الحياة الفكرية كما ورد في الإطار، واذكر نصًا مُستشهدًا ... | 0→12 | 9→16 | +31 |
| gold-0002 | قدّم مثالًا تطبيقيًا لكيفية ممارسة الحياة الروحية في موقف وا... | 0→0 | 8→10 | +2 |
| gold-0004 | قدّم مثالًا تطبيقيًا لكيفية ممارسة الحياة العاطفية في موقف و... | 0→0 | 8→10 | +2 |

## Interpretation

- **Integrity effect**: Reduction in bad citations from quarantine (RAG_ONLY_INTEGRITY - RAG_ONLY)
- **Graph effect**: Improvement from graph expansion (RAG_PLUS_GRAPH - RAG_ONLY)
- **Muḥāsibī effect**: Pure reasoning value-add from contracts, binding, critic loop (FULL - RAG_PLUS_GRAPH_INTEGRITY)
- **Value/ms**: Delta per millisecond latency increase - higher is better ROI

## Safety Gates

- Unsupported MUST_CITE rate in FULL_SYSTEM: 0.0000
- Reasoning block leak rate: 0.0000