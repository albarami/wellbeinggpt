# 160Q Bakeoff - Selective Reranker Mode

**Date:** 2025-12-21 01:50

**Configuration:**
- RERANKER_ENABLED=false
- RERANKER_SELECTIVE_MODE=true
- Model: checkpoints/reranker_phase2

## Overall Metrics

| Metric | Value |
|--------|-------|
| Total Questions | 160 |
| **PASS_FULL Rate** | **97.5%** |
| PASS_PARTIAL | 0 |
| FAIL | 4 |
| **Unexpected Fail Rate** | **0.0%** |
| **Citation Present Rate** | **97.5%** |
| **Mean Used Edges (Cross-Pillar)** | **7.3** |
| Reranker Used Count | 5 |
| Reranker Used Rate | 3.1% |

## Results by Question Type

| Type | Total | PASS_FULL | PASS_PARTIAL | FAIL | Reranker Used |
|------|-------|-----------|--------------|------|---------------|
| boundaries | 30 | 30 (100%) | 0 | 0 | 0 (0%) |
| cross_pillar | 50 | 50 (100%) | 0 | 0 | 1 (2%) |
| global_synthesis | 40 | 40 (100%) | 0 | 0 | 4 (10%) |
| injection | 5 | 2 (40%) | 0 | 3 | 0 (0%) |
| natural_chat | 30 | 30 (100%) | 0 | 0 | 0 (0%) |
| out_of_scope | 5 | 4 (80%) | 0 | 1 | 0 (0%) |

## Reranker Decision Distribution

| Decision | Count |
|----------|-------|
| unknown | 160 |
