"""Quickly test /ask/ui responsiveness for a synthesis question.

Prints numeric diagnostics only (avoids Windows console encoding issues).
"""

import time

import requests


def main() -> None:
    payload = {
        "question": "كيف يؤدي الإطار إلى ازدهار الإنسان؟",
        "mode": "answer",
        "engine": "muhasibi",
        "model_deployment": "gpt-5-chat",
    }

    t0 = time.perf_counter()
    # Use IPv4 explicitly (Windows localhost can resolve in surprising ways).
    r = requests.post("http://127.0.0.1:8000/ask/ui", json=payload, timeout=240)
    dt = time.perf_counter() - t0

    print("status:", r.status_code)
    print("elapsed_s:", round(dt, 2))
    if r.status_code != 200:
        print("body_len:", len(r.text))
        return

    data = r.json()
    spans = data.get("citations_spans") or []
    word_counts = [len((s.get("quote") or "").split()) for s in spans if isinstance(s, dict)]
    max_words = max(word_counts) if word_counts else 0
    mean_words = (sum(word_counts) / len(word_counts)) if word_counts else 0.0
    resolved = sum(1 for s in spans if isinstance(s, dict) and s.get("span_start") is not None and s.get("span_end") is not None)

    print("citations_spans:", len(spans))
    print("resolved_spans:", resolved)
    print("quote_words_max:", max_words)
    print("quote_words_mean:", round(mean_words, 2))


if __name__ == "__main__":
    main()


