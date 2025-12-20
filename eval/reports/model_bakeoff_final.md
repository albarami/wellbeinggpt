## System Depth Bakeoff - Final Results

- **Generated**: 2025-12-18T17:08:35.584682Z
- **Dataset**: `D:/wellbeingqa/eval/datasets/bakeoff_depth_v1.jsonl`
- **Dataset sha256**: `8331d935a61e5e1ded4b12d672221e3df57e23c7479338f274c1489c6c0a1aca`
- **Questions**: 160

### Methodology

**Grounding Signal**: `contract_outcome` from API (production's final grounding verdict)

**Hard Gates**:
- `unexpected_fail_rate <= 5%` (safety: injection/OOS correctly fail)
- `citation_present_rate >= 90%` (non-abstained answers have evidence)

**Scoring**: 45% Grounding + 35% Structure + 20% Naturalness

### Results

| Model | Eligible | Composite | Grounding | Structure | Natural | PASS_FULL | Unexpected Fail |
|---|:---:|---:|---:|---:|---:|---:|---:|
| gpt-5-chat | OK | 76.3 | 80.0 | 65.8 | 86.4 | 95.0% | 2.5% |
| gpt-5.1 | OK | 77.5 | 82.5 | 66.0 | 86.3 | 95.6% | 1.9% |
| gpt-5.2 | DQ | 0.0 | 67.5 | 63.9 | 86.8 | 91.9% | 5.6% |

### Failure Analysis

| Model | Total Fail | Expected (inj/oos) | Unexpected |
|---|---:|---:|---:|
| gpt-5-chat | 8 | 4 | 4 |
| gpt-5.1 | 7 | 4 | 3 |
| gpt-5.2 | 13 | 4 | 9 |

### Winner

**gpt-5.1** with composite score 77.5

| Metric | Value |
|---|---|
| PASS_FULL rate | 95.6% |
| Unexpected fail rate | 1.9% |
| Mean edges | 5.3 |
| Quote compliance | 100.0% |
