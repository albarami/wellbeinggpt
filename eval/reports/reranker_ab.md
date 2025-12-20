# Reranker A/B Test Report

Generated: 2025-12-20 (Updated after full bakeoff)
Deployment: gpt-5.1
Reranker Model: `checkpoints/reranker/final` (8×A100 trained, 5,600 pairs)

## Summary

The reranker was trained on 8× NVIDIA A100-SXM4-80GB GPUs in 69 seconds using DataParallel.

### Full Bakeoff Results (160 questions)

| Metric | Reranker OFF | Reranker ON | Delta |
|--------|--------------|-------------|-------|
| **PASS_FULL Rate** | 97.5% | 93.8% | **-3.7%** |
| Pass Rate (any) | 97.5% | 94.4% | -3.1% |
| Mean Citations | 8.5 | 8.3 | -0.2 |
| Mean Used Edges | 4.6 | 4.5 | -0.1 |
| Unexpected Fail Rate | 2.5% | 5.6% | +3.1% |

### Regression Stability Results (10 runs, synth-006)

| Condition | synth-006 Pass Rate |
|-----------|---------------------|
| Reranker OFF | 30% (3/10) |
| Reranker ON | 100% (10/10) |

## Key Finding: Mixed Effects

The reranker has **opposite effects** on different question types:

1. **Hurts overall bakeoff** (-3.7% PASS_FULL)
2. **Helps synthesis questions** (synth-006: 30% → 100%)

### Root Cause

The reranker may be:
- Over-filtering good evidence for simple questions
- Rescuing synthesis questions that need better evidence ranking

## Recommendation

**Disable reranker globally. Enable per-intent for synthesis.**

```env
# Production default
RERANKER_ENABLED=false
```

Enable selectively for:
- `global_synthesis` intent
- `cross_pillar` with low evidence scores

## Next Steps

1. **Train on more data** (50k+ pairs with hard negatives)
2. **Tune threshold** - current reranker may be too aggressive
3. **Implement intent-based routing** for reranker activation
