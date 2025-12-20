## System Depth Bakeoff (Corrected Scoring)

- **Generated**: 2025-12-18T16:59:34.206016Z
- **Dataset**: `D:/wellbeingqa/eval/datasets/bakeoff_depth_v1.jsonl`
- **Dataset sha256**: `8331d935a61e5e1ded4b12d672221e3df57e23c7479338f274c1489c6c0a1aca`

### Scoring Correction

The original scoring used post-hoc claim extraction which caused ~52% false unsupported rate.
This corrected scoring uses **contract_outcome** from the API (the authoritative grounding check).

### Hard Gates

- **citation_validity_errors == 0**: Actual span validity
- **contract_fail_rate <= 5%**: System's own grounding check (allows OOS questions to fail)

### Per-model Summary

| Model | Eligible | Composite | Depth | Cross | Nat | PASS_FULL | PASS_PARTIAL | FAIL | Edges | Chains |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5-chat | ✅ | 84.4 | 89.0 | 76.1 | 86.4 | 95.0% | 0.0% | 5.0% | 5.3 | 5.3 |
| gpt-5.1 | ✅ | 84.7 | 89.7 | 76.1 | 86.3 | 95.6% | 0.0% | 4.4% | 5.3 | 5.3 |
| gpt-5.2 | ❌ | 0.0 | 87.5 | 73.4 | 86.8 | 91.9% | 0.0% | 8.1% | 5.0 | 5.0 |

### Winners

- **Overall**: gpt-5.1 (composite=84.7)
- **Depth**: gpt-5.1 (89.7)
- **Cross-pillar**: gpt-5-chat (76.1)
- **Naturalness**: gpt-5-chat (86.4)
