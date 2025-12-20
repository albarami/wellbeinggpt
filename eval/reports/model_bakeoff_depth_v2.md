## System Depth Bakeoff (Production Grounding)

- **Generated**: 2025-12-18T17:06:57.244641Z
- **Dataset**: `D:/wellbeingqa/eval/datasets/bakeoff_depth_v1.jsonl`
- **Dataset sha256**: `8331d935a61e5e1ded4b12d672221e3df57e23c7479338f274c1489c6c0a1aca`

### Grounding Methodology

Uses **production grounding signals** (same as runtime):
- **Term coverage**: answer terms appearing in cited evidence (min 50% required)
- **Contract outcome**: PASS_FULL / PASS_PARTIAL / FAIL
- **Expected failures**: injection + OOS questions (correctly abstained)

### Hard Gates

- **unexpected_fail_rate <= 5%**: Only fails on non-expected questions
- **low_coverage_rate <= 10%**: Answers with term coverage < 50%

### Per-model Summary

| Model | Eligible | Composite | Depth | Cross | Nat | Ground | PASS_FULL | Unexpected Fail | Term Cov |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5-chat | DQ | 0.0 | 81.9 | 63.1 | 86.4 | 22.5 | 95.0% | 2.5% | 49.1% |
| gpt-5.1 | DQ | 0.0 | 82.7 | 63.1 | 86.3 | 25.9 | 95.6% | 1.9% | 51.6% |
| gpt-5.2 | DQ | 0.0 | 81.2 | 59.8 | 86.8 | 23.3 | 91.9% | 5.6% | 50.2% |

### Winners

- **Best (all DQ)**: gpt-5.1 (depth=82.7)
