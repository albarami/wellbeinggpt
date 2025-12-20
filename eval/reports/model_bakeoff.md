# Model Bakeoff Report

Generated: 2025-12-17T21:25:20.637782+00:00
Models: gpt-5-chat, gpt-5.1, gpt-5.2
Datasets: stress_12
Modes: answer

## Summary (Aggregated Across All Datasets/Modes)

| Rank | Model | Weighted Score | Depth | Connections | Naturalness | Speed | Status |
|------|-------|----------------|-------|-------------|-------------|-------|--------|
| 1 | gpt-5-chat | -999.0 | 0.0 | 0.0 | 0.0 | 0.0 | DISQUALIFIED |
| 2 | gpt-5.1 | -999.0 | 0.0 | 0.0 | 0.0 | 0.0 | DISQUALIFIED |
| 3 | gpt-5.2 | -999.0 | 0.0 | 0.0 | 0.0 | 0.0 | DISQUALIFIED |

## Winner Per Dimension

- **depth**: No valid models
- **connections**: No valid models
- **naturalness**: No valid models
- **speed**: No valid models
- **weighted_score**: No valid models

## Detailed Results Per Dataset

### stress_12

#### Mode: answer

| Model | Score | Rubric | Citations | Edges | Pillars | Latency p50 | Status |
|-------|-------|--------|-----------|-------|---------|-------------|--------|
| gpt-5-chat | -999.0 | 0.0 | 0 | 0 | 0 | 0ms | DQ: Can't reconnect until invalid  |
| gpt-5.1 | -999.0 | 0.0 | 0 | 0 | 0 | 0ms | DQ: Can't reconnect until invalid  |
| gpt-5.2 | -999.0 | 0.0 | 0 | 0 | 0 | 0ms | DQ: Can't reconnect until invalid  |

## Raw Metrics

```json
[
  {
    "model": "gpt-5-chat",
    "dataset": "stress_12",
    "mode": "answer",
    "weighted_score": -999,
    "disqualified": true,
    "metrics": {
      "rubric_score": 0.0,
      "citation_validity_errors": 0,
      "unsupported_must_cite_rate": 0.0,
      "used_edges_count": 0,
      "distinct_pillars": 0,
      "pass_full_rate": 0.0,
      "latency_p50_ms": 0.0,
      "latency_p95_ms": 0.0
    }
  },
  {
    "model": "gpt-5.1",
    "dataset": "stress_12",
    "mode": "answer",
    "weighted_score": -999,
    "disqualified": true,
    "metrics": {
      "rubric_score": 0.0,
      "citation_validity_errors": 0,
      "unsupported_must_cite_rate": 0.0,
      "used_edges_count": 0,
      "distinct_pillars": 0,
      "pass_full_rate": 0.0,
      "latency_p50_ms": 0.0,
      "latency_p95_ms": 0.0
    }
  },
  {
    "model": "gpt-5.2",
    "dataset": "stress_12",
    "mode": "answer",
    "weighted_score": -999,
    "disqualified": true,
    "metrics": {
      "rubric_score": 0.0,
      "citation_validity_errors": 0,
      "unsupported_must_cite_rate": 0.0,
      "used_edges_count": 0,
      "distinct_pillars": 0,
      "pass_full_rate": 0.0,
      "latency_p50_ms": 0.0,
      "latency_p95_ms": 0.0
    }
  }
]
```