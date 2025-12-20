## Baseline Freeze: gpt-5.1 Full Bakeoff

- **Generated**: 2025-12-20T13:04:22.121248Z
- **Dataset**: `eval\datasets\bakeoff_depth_v1.jsonl`
- **Dataset hash**: `8331d935a61e5e1d`
- **Total questions**: 160
- **Model**: gpt-5.1 (default)

### Baseline Metrics (Freeze These)

| Metric | Value |
|--------|-------|
| PASS_FULL rate | 97.5% |
| Citation present rate | 97.5% |
| Unexpected fail rate | 0.0% |
| Mean used edges (cross-pillar) | 7.3 |
| Median latency (ms) | 10481 |

### By Question Type

| Type | Count | PASS_FULL | Rate |
|------|-------|-----------|------|
| boundaries | 30 | 30 | 100.0% |
| cross_pillar | 50 | 50 | 100.0% |
| global_synthesis | 40 | 40 | 100.0% |
| injection | 5 | 2 | 40.0% |
| natural_chat | 30 | 30 | 100.0% |
| out_of_scope | 5 | 4 | 80.0% |

### Unexpected Failures

*None*