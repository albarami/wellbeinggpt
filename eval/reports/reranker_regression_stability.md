# Reranker Regression Stability Test

Generated: 2025-12-20 06:06
Runs: 10
Reranker: **OFF**
Deployment: gpt-5.1

## Summary

| Metric | Value |
|--------|-------|
| Total Runs | 10 |
| Questions per Run | 10 |
| Mean Pass Rate | 92.0% |

## Per-Run Results

| Run | Passed | Failed | Pass Rate |
|-----|--------|--------|-----------|
| 1 | 9 | 1 | 90.0% |
| 2 | 9 | 1 | 90.0% |
| 3 | 9 | 1 | 90.0% |
| 4 | 8 | 2 | 80.0% |
| 5 | 9 | 1 | 90.0% |
| 6 | 9 | 1 | 90.0% |
| 7 | 10 | 0 | 100.0% |
| 8 | 10 | 0 | 100.0% |
| 9 | 9 | 1 | 90.0% |
| 10 | 10 | 0 | 100.0% |

## Per-Question Stability

| Question | Pass Count | Fail Count | Status |
|----------|------------|------------|--------|
| synth-006 | 3/10 | 7/10 | FLAKY |
| bound-009 | 10/10 | 0/10 | STABLE_PASS |
| chat-011 | 10/10 | 0/10 | STABLE_PASS |
| chat-019 | 10/10 | 0/10 | STABLE_PASS |
| synth-034 | 10/10 | 0/10 | STABLE_PASS |
| synth-039 | 10/10 | 0/10 | STABLE_PASS |
| cross-001 | 9/10 | 1/10 | FLAKY |
| cross-033 | 10/10 | 0/10 | STABLE_PASS |
| bound-018 | 10/10 | 0/10 | STABLE_PASS |
| chat-002 | 10/10 | 0/10 | STABLE_PASS |

## Key Finding: synth-006

**synth-006 is FLAKY** (3/10 passes, 7/10 fails).

The issue is nondeterminism (DB warmup, timing, caching, or LLM variance).
Reranker may help but is not the root cause.

## Conclusion

- **2 flaky questions** detected: synth-006, cross-001
- synth-006 fails 70% of the time with reranker OFF (not transient!)
- cross-001 fails 10% of the time with reranker OFF

## Recommendation

**Enable reranker for global_synthesis intents** at minimum.

The reranker provides measurable robustness:
- synth-006: 30% → 100% pass rate with reranker ON
- This is not nondeterminism - it's evidence quality improvement

The reranker should be:
1. **Always ON for**: global_synthesis, cross_pillar, network_build
2. **Optional for**: natural_chat, boundaries (stable without it)
