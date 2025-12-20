# Reranker A/B Bakeoff Test

Generated: 2025-12-20 08:11
Dataset: bakeoff_depth_v1.jsonl (160 questions)
Deployment: gpt-5.1

## Summary

| Metric | Reranker OFF | Reranker ON | Delta |
|--------|--------------|-------------|-------|
| PASS_FULL Rate | 97.5% | 93.8% | -3.7% |
| Pass Rate (any) | 97.5% | 94.4% | -3.1% |
| Citation Present Rate | 97.5% | 94.4% | -3.1% |
| Mean Citations | 8.5 | 8.3 | -0.2 |
| Mean Used Edges | 4.6 | 4.5 | -0.1 |
| Mean Argument Chains | 4.6 | 4.5 | -0.1 |
| Boundary Rate | 43.8% | 41.2% | -2.5% |
| Unexpected Fail Rate | 2.5% | 5.6% | +3.1% |

## Detailed Counts

| Condition | Total | Passed | PASS_FULL | Failed |
|-----------|-------|--------|-----------|--------|
| Reranker OFF | 160 | 156 | 156 | 4 |
| Reranker ON | 160 | 151 | 150 | 9 |

## Interpretation

**Reranker DECREASES PASS_FULL rate by 3.7%** - unexpected regression.

## Recommendation

**Keep reranker OFF by default, enable selectively for synthesis intents.**

### Key Finding: Reranker has mixed effects

1. **On 160-question bakeoff**: Reranker HURTS overall PASS_FULL (-3.7%)
2. **On 10-question regression (stability test)**: Reranker HELPS synth-006 (30% → 100%)

### Root Cause Hypothesis

The reranker may be:
- **Over-filtering** good evidence for some questions (causing more failures)
- **Rescuing** synthesis questions that need better evidence ranking

### Recommended Action

1. **Disable reranker globally** (`RERANKER_ENABLED=false`)
2. **Enable reranker per-intent** for:
   - `global_synthesis` (where synth-006 type questions benefit)
   - `cross_pillar` with low initial evidence scores
3. **Train on more data** (50k+ pairs with hard negatives) before full production deployment
4. **Tune threshold** - the current reranker may be too aggressive in filtering

### Evidence

| Scenario | Reranker OFF | Reranker ON | 
|----------|--------------|-------------|
| Full bakeoff PASS_FULL | 97.5% | 93.8% |
| synth-006 regression (10 runs) | 30% | 100% |

The reranker helps synthesis but hurts overall breadth.
