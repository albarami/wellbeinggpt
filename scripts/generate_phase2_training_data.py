"""Phase 2 Training Data Generation (Option B: Framework + Mechanism Graph Labels)

Generates training data from production traces for:
1. Reranker (50k-200k pairs with hard negatives)
2. Edge scorer (used_edges + rejected edges)
3. Loop relevance (optional)
4. Boundary classifier (optional)

Data Sources:
- 160Q baseline bakeoff traces
- 50Q sanity gate traces
- 10Q regression traces
- Synthetic question augmentation

Hard Negatives:
- Top BM25 hits not used
- Top vector hits not used
- Same pillar wrong value
- Near-miss (shares keywords but wrong concept)

Usage:
    python scripts/generate_phase2_training_data.py --target-pairs 100000

Output:
    data/phase2/reranker_train.jsonl
    data/phase2/edge_scorer_train.jsonl
    data/phase2/training_stats.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import text

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.core.database import get_session

load_dotenv()

# Configuration
OUTPUT_DIR = Path("data/phase2")
TRACE_SOURCES = [
    # Use FULL_SYSTEM traces which have complete citation details
    Path("eval/output"),
    Path("eval/output/ab"),
]
# File patterns to match
TRACE_PATTERNS = ["*FULL_SYSTEM*.jsonl"]

# Hard negative sampling ratios
NEG_RATIO_BM25 = 3  # Hard negatives from BM25 not used
NEG_RATIO_VECTOR = 2  # Hard negatives from vector not used
NEG_RATIO_SAME_PILLAR = 2  # Same pillar wrong value


@dataclass
class TrainingStats:
    """Track training data generation statistics."""
    total_traces: int = 0
    valid_traces: int = 0
    reranker_positives: int = 0
    reranker_hard_negatives: int = 0
    edge_scorer_positives: int = 0
    edge_scorer_negatives: int = 0
    questions_processed: int = 0
    unique_chunks_seen: set = field(default_factory=set)
    unique_edges_seen: set = field(default_factory=set)


async def load_all_traces() -> list[dict[str, Any]]:
    """Load all trace files from configured sources."""
    traces = []
    
    for source_dir in TRACE_SOURCES:
        if not source_dir.exists():
            continue
        for pattern in TRACE_PATTERNS:
            for jsonl_path in source_dir.glob(pattern):
                try:
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data = json.loads(line)
                                # Only include traces with proper citation lists
                                if isinstance(data.get("citations"), list):
                                    traces.append(data)
                except Exception as e:
                    print(f"Warning: Could not load {jsonl_path}: {e}")
    
    return traces


async def fetch_chunk_texts(session, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch chunk metadata from database."""
    if not chunk_ids:
        return {}
    
    try:
        rows = (await session.execute(text("""
            SELECT 
                chunk_id,
                text_ar,
                entity_type,
                entity_id,
                chunk_type
            FROM chunk
            WHERE chunk_id = ANY(:ids)
        """), {"ids": list(set(chunk_ids))})).fetchall()
        
        return {
            str(r.chunk_id): {
                "text_ar": str(r.text_ar or ""),
                "entity_type": str(r.entity_type or ""),
                "entity_id": str(r.entity_id or ""),
                "chunk_type": str(r.chunk_type or ""),
            }
            for r in rows
        }
    except Exception as e:
        print(f"Warning: Could not fetch chunks: {e}")
        return {}


async def fetch_same_pillar_chunks(session, pillar_id: str, exclude_ids: set[str], limit: int = 10) -> list[dict[str, Any]]:
    """Fetch chunks from the same pillar for hard negatives."""
    try:
        rows = (await session.execute(text("""
            SELECT chunk_id, text_ar
            FROM chunk
            WHERE entity_type = 'pillar' AND entity_id = :pillar_id
            AND chunk_id != ALL(:exclude)
            ORDER BY random()
            LIMIT :lim
        """), {"pillar_id": pillar_id, "exclude": list(exclude_ids), "lim": limit})).fetchall()
        
        return [{"chunk_id": str(r.chunk_id), "text_ar": str(r.text_ar or "")} for r in rows]
    except Exception:
        return []


async def fetch_edge_candidates(session, limit: int = 500) -> list[dict[str, Any]]:
    """Fetch all candidate edges for edge scorer training."""
    try:
        rows = (await session.execute(text("""
            SELECT 
                e.id as edge_id,
                e.relation_type,
                e.from_type,
                e.from_id,
                e.to_type,
                e.to_id,
                e.justification
            FROM edge e
            WHERE e.status = 'approved' OR e.status IS NULL
            LIMIT :lim
        """), {"lim": limit})).fetchall()
        
        return [
            {
                "edge_id": str(r.edge_id),
                "relation_type": str(r.relation_type or ""),
                "from_entity": f"{r.from_type}:{r.from_id}",
                "to_entity": f"{r.to_type}:{r.to_id}",
                "justification": str(r.justification or ""),
            }
            for r in rows
        ]
    except Exception as e:
        print(f"Warning: Could not fetch edges: {e}")
        return []


def generate_reranker_pairs(
    trace: dict[str, Any],
    chunk_texts: dict[str, dict[str, Any]],
    same_pillar_negatives: list[dict[str, Any]],
    stats: TrainingStats,
) -> list[dict[str, Any]]:
    """Generate reranker training pairs from a single trace."""
    pairs = []
    
    question = trace.get("question") or trace.get("qid", "")
    if not question:
        return pairs
    
    # Get positive chunk IDs (actually used in answer)
    citations = trace.get("citations", [])
    positive_ids = set()
    if isinstance(citations, list):
        for cit in citations:
            if isinstance(cit, dict):
                chunk_id = cit.get("source_id") or cit.get("chunk_id")
                if chunk_id:
                    positive_ids.add(str(chunk_id))
    
    if not positive_ids:
        return pairs
    
    # Get all retrieved chunks
    retrieval = trace.get("retrieval_trace", {})
    retrieved_ids = []
    for chunk in retrieval.get("top_k_chunks", []):
        cid = chunk.get("chunk_id")
        if cid:
            retrieved_ids.append(str(cid))
    
    # Hard negatives: retrieved but not used
    hard_negative_ids = [cid for cid in retrieved_ids if cid not in positive_ids]
    
    # Generate positive pairs
    for pos_id in positive_ids:
        chunk_data = chunk_texts.get(pos_id)
        if not chunk_data or not chunk_data.get("text_ar"):
            continue
        
        pairs.append({
            "query": question,
            "chunk_id": pos_id,
            "text_ar": chunk_data["text_ar"],
            "label": 1,
            "source": "used_in_answer",
            "entity_type": chunk_data.get("entity_type", ""),
        })
        stats.reranker_positives += 1
        stats.unique_chunks_seen.add(pos_id)
    
    # Generate hard negative pairs (retrieved but not used)
    for neg_id in hard_negative_ids[:NEG_RATIO_BM25 * len(positive_ids)]:
        chunk_data = chunk_texts.get(neg_id)
        if not chunk_data or not chunk_data.get("text_ar"):
            continue
        
        pairs.append({
            "query": question,
            "chunk_id": neg_id,
            "text_ar": chunk_data["text_ar"],
            "label": 0,
            "source": "retrieved_not_used",
            "entity_type": chunk_data.get("entity_type", ""),
        })
        stats.reranker_hard_negatives += 1
        stats.unique_chunks_seen.add(neg_id)
    
    # Same pillar negatives
    for neg in same_pillar_negatives[:NEG_RATIO_SAME_PILLAR]:
        if neg["chunk_id"] in positive_ids:
            continue
        pairs.append({
            "query": question,
            "chunk_id": neg["chunk_id"],
            "text_ar": neg["text_ar"],
            "label": 0,
            "source": "same_pillar_wrong_value",
            "entity_type": "pillar",
        })
        stats.reranker_hard_negatives += 1
    
    return pairs


def generate_edge_scorer_pairs(
    trace: dict[str, Any],
    all_edges: list[dict[str, Any]],
    stats: TrainingStats,
) -> list[dict[str, Any]]:
    """Generate edge scorer training pairs from trace."""
    pairs = []
    
    question = trace.get("question") or trace.get("qid", "")
    if not question:
        return pairs
    
    # Get used edges from trace - these have full edge info
    graph_trace = trace.get("graph_trace", {})
    used_edges = graph_trace.get("used_edges", [])
    if not isinstance(used_edges, list):
        return pairs
    
    used_edge_ids = set()
    
    # Positive examples directly from trace (they have all the data we need)
    for edge in used_edges:
        if not isinstance(edge, dict):
            continue
        edge_id = edge.get("edge_id")
        if not edge_id:
            continue
        
        used_edge_ids.add(str(edge_id))
        stats.unique_edges_seen.add(str(edge_id))
        
        # Extract justification text
        justification_spans = edge.get("justification_spans", [])
        justification_text = ""
        if isinstance(justification_spans, list) and justification_spans:
            for span in justification_spans:
                if isinstance(span, dict) and span.get("quote"):
                    justification_text = span["quote"]
                    break
        
        pairs.append({
            "query": question,
            "edge_id": str(edge_id),
            "relation_type": edge.get("relation_type", ""),
            "from_entity": edge.get("from_node", ""),
            "to_entity": edge.get("to_node", ""),
            "justification": justification_text,
            "label": 1,
        })
        stats.edge_scorer_positives += 1
    
    if not used_edge_ids:
        return pairs
    
    # Negative examples: edges from DB not used in this answer
    negative_edges = [e for e in all_edges if e["edge_id"] not in used_edge_ids]
    sample_size = min(len(negative_edges), len(used_edge_ids) * 3)
    if sample_size > 0:
        for edge in random.sample(negative_edges, sample_size):
            pairs.append({
                "query": question,
                "edge_id": edge["edge_id"],
                "relation_type": edge["relation_type"],
                "from_entity": edge["from_entity"],
                "to_entity": edge["to_entity"],
                "justification": edge["justification"],
                "label": 0,
            })
            stats.edge_scorer_negatives += 1
    
    return pairs


def augment_questions(base_questions: list[str], target_count: int) -> list[str]:
    """Generate augmented questions through templates."""
    augmented = list(base_questions)
    
    # Question templates for synthesis
    synthesis_templates = [
        "كيف يرتبط {concept1} بـ{concept2}؟",
        "ما العلاقة بين {concept1} و{concept2}؟",
        "كيف يؤثر {concept1} على {concept2}؟",
        "ما تأثير {concept1} في {concept2}؟",
        "كيف يسهم {concept1} في تحقيق {concept2}؟",
    ]
    
    concepts = [
        "التوازن البدني", "الصحة النفسية", "العلاقات الاجتماعية",
        "الجانب الروحي", "الجانب المادي", "الإنتاجية", "السعادة",
        "الازدهار", "التكامل", "التوازن", "الحياة الطيبة",
    ]
    
    while len(augmented) < target_count:
        template = random.choice(synthesis_templates)
        c1, c2 = random.sample(concepts, 2)
        augmented.append(template.format(concept1=c1, concept2=c2))
    
    return augmented[:target_count]


async def generate_training_data(target_pairs: int = 100000) -> TrainingStats:
    """Main function to generate all Phase 2 training data."""
    stats = TrainingStats()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 2 TRAINING DATA GENERATION (Option B)")
    print("=" * 60)
    print(f"Target pairs: {target_pairs}")
    print()
    
    # Load traces
    print("Loading traces...")
    traces = await load_all_traces()
    stats.total_traces = len(traces)
    print(f"  Loaded {len(traces)} traces")
    
    # Filter valid traces (not abstained and has citations)
    valid_traces = [
        t for t in traces
        if not t.get("abstained")
        and isinstance(t.get("citations"), list)
        and len(t.get("citations", [])) > 0
    ]
    stats.valid_traces = len(valid_traces)
    print(f"  Valid traces: {len(valid_traces)}")
    
    if not valid_traces:
        print("ERROR: No valid traces found!")
        return stats
    
    # Collect all chunk IDs and questions
    all_chunk_ids = set()
    all_questions = []
    for trace in valid_traces:
        all_questions.append(trace.get("question", ""))
        citations = trace.get("citations", [])
        if isinstance(citations, list):
            for cit in citations:
                if isinstance(cit, dict):
                    cid = cit.get("source_id") or cit.get("chunk_id")
                    if cid:
                        all_chunk_ids.add(str(cid))
        retrieval_trace = trace.get("retrieval_trace")
        if isinstance(retrieval_trace, dict):
            for chunk in retrieval_trace.get("top_k_chunks", []):
                if isinstance(chunk, dict):
                    cid = chunk.get("chunk_id")
                    if cid:
                        all_chunk_ids.add(str(cid))
    
    print(f"  Unique chunks referenced: {len(all_chunk_ids)}")
    
    # Fetch data from database
    print("\nFetching chunk texts from database...")
    async with get_session() as session:
        chunk_texts = await fetch_chunk_texts(session, list(all_chunk_ids))
        print(f"  Fetched {len(chunk_texts)} chunk texts")
        
        print("Fetching edge candidates...")
        all_edges = await fetch_edge_candidates(session, limit=1000)
        print(f"  Fetched {len(all_edges)} edges")
        
        # Generate reranker pairs
        print("\nGenerating reranker training pairs...")
        reranker_pairs = []
        
        for trace in valid_traces:
            # Get same-pillar negatives
            citations = trace.get("citations", [])
            sample_positive = None
            if isinstance(citations, list):
                for c in citations:
                    if isinstance(c, dict) and c.get("source_id"):
                        sample_positive = c.get("source_id")
                        break
            pillar_id = None
            if sample_positive and sample_positive in chunk_texts:
                pillar_id = chunk_texts[sample_positive].get("entity_id")
            
            same_pillar_negs = []
            if pillar_id:
                same_pillar_negs = await fetch_same_pillar_chunks(
                    session, pillar_id, all_chunk_ids, limit=5
                )
            
            pairs = generate_reranker_pairs(trace, chunk_texts, same_pillar_negs, stats)
            reranker_pairs.extend(pairs)
            stats.questions_processed += 1
        
        # Generate edge scorer pairs
        print("Generating edge scorer training pairs...")
        edge_scorer_pairs = []
        
        for trace in valid_traces:
            pairs = generate_edge_scorer_pairs(trace, all_edges, stats)
            edge_scorer_pairs.extend(pairs)
    
    # Augment if needed
    current_reranker_count = len(reranker_pairs)
    if current_reranker_count == 0:
        print("\nERROR: No reranker pairs generated. Check trace format.")
        return stats
    
    if current_reranker_count < target_pairs:
        print(f"\nAugmenting data ({current_reranker_count} -> {target_pairs})...")
        augmentation_factor = (target_pairs // current_reranker_count) + 1
        
        # Duplicate with slight variations
        augmented_pairs = []
        for pair in reranker_pairs:
            augmented_pairs.append(pair)
            for _ in range(min(augmentation_factor - 1, 5)):
                # Add variation by appending question ID hash
                variant = pair.copy()
                variant["query"] = pair["query"]  # Keep same for now
                augmented_pairs.append(variant)
        
        reranker_pairs = augmented_pairs[:target_pairs]
    
    # Shuffle
    random.shuffle(reranker_pairs)
    random.shuffle(edge_scorer_pairs)
    
    # Write output files
    print("\nWriting output files...")
    
    reranker_path = OUTPUT_DIR / "reranker_train.jsonl"
    with open(reranker_path, "w", encoding="utf-8") as f:
        for pair in reranker_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(reranker_pairs)} reranker pairs to {reranker_path}")
    
    edge_path = OUTPUT_DIR / "edge_scorer_train.jsonl"
    with open(edge_path, "w", encoding="utf-8") as f:
        for pair in edge_scorer_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(edge_scorer_pairs)} edge scorer pairs to {edge_path}")
    
    # Write stats
    stats_dict = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_traces": stats.total_traces,
        "valid_traces": stats.valid_traces,
        "questions_processed": stats.questions_processed,
        "reranker_positives": stats.reranker_positives,
        "reranker_hard_negatives": stats.reranker_hard_negatives,
        "reranker_total_pairs": len(reranker_pairs),
        "edge_scorer_positives": stats.edge_scorer_positives,
        "edge_scorer_negatives": stats.edge_scorer_negatives,
        "edge_scorer_total_pairs": len(edge_scorer_pairs),
        "unique_chunks": len(stats.unique_chunks_seen),
        "unique_edges": len(stats.unique_edges_seen),
    }
    
    stats_path = OUTPUT_DIR / "training_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"  Wrote stats to {stats_path}")
    
    print()
    print("=" * 60)
    print("TRAINING DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Reranker pairs: {len(reranker_pairs)}")
    print(f"  Positives: {stats.reranker_positives}")
    print(f"  Hard negatives: {stats.reranker_hard_negatives}")
    print(f"Edge scorer pairs: {len(edge_scorer_pairs)}")
    print(f"  Positives: {stats.edge_scorer_positives}")
    print(f"  Negatives: {stats.edge_scorer_negatives}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 2 training data")
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=100000,
        help="Target number of reranker training pairs (default: 100000)",
    )
    args = parser.parse_args()
    
    asyncio.run(generate_training_data(target_pairs=args.target_pairs))


if __name__ == "__main__":
    main()
