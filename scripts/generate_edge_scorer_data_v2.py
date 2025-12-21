"""
Generate Edge Scorer Training Data - Phase B (5k pairs).

Target:
- 1,000-1,500 positives (used_edges from PASS_FULL)
- 3,500-4,000 negatives with:
  - Hard negatives ≥50% (candidate edges not selected)
  - Wrong relation_type negatives
  - Confusable text negatives
  - Random negatives (minority)

Split by question_id to prevent leakage.
"""

import asyncio
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "phase2"

# Target counts
TARGET_POSITIVES = 1500
TARGET_NEGATIVES = 4000
HARD_NEGATIVE_RATIO = 0.55  # ≥50%

# Relation types for wrong-type negatives
RELATION_TYPES = [
    "ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON",
    "INHIBITS", "TENSION_WITH", "RESOLVES_WITH", "RELATED_TO"
]


def load_traces() -> list[dict]:
    """Load all available traces from eval outputs."""
    traces = []
    
    # Sources of traces - look for FULL_SYSTEM files which have rich graph_trace
    trace_dir = PROJECT_ROOT / "eval" / "output"
    
    if not trace_dir.exists():
        logger.warning(f"Trace dir not found: {trace_dir}")
        return traces
    
    # First pass: FULL_SYSTEM files (best source - use argument_chains)
    for jsonl_file in trace_dir.glob("*FULL_SYSTEM*.jsonl"):
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        gt = data.get("graph_trace", {})
                        if not isinstance(gt, dict):
                            continue
                        
                        # Use argument_chains - these have rich edge data
                        chains = gt.get("argument_chains", [])
                        if chains and isinstance(chains, list):
                            # Filter to dict items only
                            valid_chains = [c for c in chains if isinstance(c, dict)]
                            if valid_chains:
                                data["_used_edges"] = valid_chains
                                data["_source"] = str(jsonl_file.name)
                                traces.append(data)
                                continue
                        
                        # Fallback: used_edges if present
                        used_edges = gt.get("used_edges", [])
                        if used_edges and isinstance(used_edges, list):
                            valid_edges = [e for e in used_edges if isinstance(e, dict)]
                            if valid_edges:
                                data["_used_edges"] = valid_edges
                                data["_source"] = str(jsonl_file.name)
                                traces.append(data)
                                
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Error reading {jsonl_file}: {e}")
    
    logger.info(f"Loaded {len(traces)} traces with edges from FULL_SYSTEM files")
    return traces


def get_db_connection():
    """Get database connection using psycopg2."""
    import psycopg2
    from dotenv import load_dotenv
    import re
    
    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "")
    
    # Parse connection string
    # Format: postgresql+asyncpg://user:password@host:port/dbname
    # or postgresql://user:password@host:port/dbname
    match = re.match(r'postgresql(?:\+\w+)?://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', db_url)
    if match:
        user, password, host, port, database = match.groups()
        return psycopg2.connect(
            host=host,
            port=int(port),
            database=database,
            user=user,
            password=password,
        )
    else:
        raise ValueError(f"Could not parse DATABASE_URL: {db_url[:50]}...")


async def fetch_entity_labels() -> dict[str, str]:
    """Fetch entity labels from database."""
    labels = {}
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Pillars (id, name_ar)
        cursor.execute("SELECT id, name_ar FROM pillar")
        for row in cursor.fetchall():
            labels[f"pillar:{row[0]}"] = row[1] or row[0]
        
        # Core values
        cursor.execute("SELECT id, name_ar FROM core_value")
        for row in cursor.fetchall():
            labels[f"core_value:{row[0]}"] = row[1] or row[0]
        
        # Sub values
        cursor.execute("SELECT id, name_ar FROM sub_value")
        for row in cursor.fetchall():
            labels[f"sub_value:{row[0]}"] = row[1] or row[0]
        
        cursor.close()
        conn.close()
        logger.info(f"Fetched {len(labels)} entity labels")
    except Exception as e:
        logger.warning(f"Could not fetch entity labels: {e}")
    
    return labels


async def fetch_all_edges() -> list[dict]:
    """Fetch all edges from database as candidate pool."""
    edges = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Schema uses from_type/from_id/to_type/to_id
        cursor.execute("""
            SELECT 
                e.id as edge_id,
                e.from_type, e.from_id,
                e.to_type, e.to_id,
                e.relation_type,
                (SELECT string_agg(js.quote, ' ') FROM edge_justification_span js WHERE js.edge_id = e.id) as justification
            FROM edge e
            WHERE e.status = 'approved' OR e.status IS NULL
            LIMIT 5000
        """)
        
        for row in cursor.fetchall():
            edges.append({
                "edge_id": str(row[0]),
                "from_node": f"{row[1]}:{row[2]}",
                "to_node": f"{row[3]}:{row[4]}",
                "relation_type": row[5] or "",
                "justification": row[6] or "",
            })
        
        cursor.close()
        conn.close()
        logger.info(f"Fetched {len(edges)} edges from database")
    except Exception as e:
        logger.warning(f"Could not fetch edges: {e}")
    
    return edges


def extract_used_edges(trace: dict) -> list[dict]:
    """Extract used edges from a trace."""
    # Use pre-extracted edges if available
    if "_used_edges" in trace:
        return trace["_used_edges"]
    
    used_edges = []
    
    # Try graph_trace first
    graph_trace = trace.get("graph_trace", {})
    if isinstance(graph_trace, dict):
        # Try multiple field names
        for key in ["used_edges", "edges"]:
            for edge in graph_trace.get(key, []):
                if isinstance(edge, dict):
                    used_edges.append(edge)
            if used_edges:
                break
    
    # Fallback to direct used_edges
    if not used_edges:
        for edge in trace.get("used_edges", []):
            if isinstance(edge, dict):
                used_edges.append(edge)
    
    return used_edges


def generate_positive(
    edge: dict,
    query: str,
    entity_labels: dict[str, str],
) -> dict:
    """Generate a positive training example from a used edge or argument_chain."""
    from_node = edge.get("from_node", "")
    to_node = edge.get("to_node", "")
    relation_type = edge.get("relation_type", "RELATED_TO")
    
    # Get labels
    from_text = entity_labels.get(from_node, from_node)
    to_text = entity_labels.get(to_node, to_node)
    
    # Get justification from multiple possible sources
    justification = ""
    
    # Try evidence_spans (from argument_chains)
    spans = edge.get("evidence_spans", [])
    if spans and isinstance(spans, list):
        for s in spans[:2]:
            if isinstance(s, dict):
                justification += " " + str(s.get("quote", ""))
    
    # Fallback to justification_spans
    if not justification.strip():
        spans = edge.get("justification_spans", [])
        if spans and isinstance(spans, list):
            for s in spans[:2]:
                if isinstance(s, dict):
                    justification += " " + str(s.get("quote", ""))
    
    # Fallback to claim_ar (from argument_chains)
    if not justification.strip():
        justification = edge.get("claim_ar", "")
    
    justification = justification.strip()
    
    return {
        "query": query,
        "edge_id": edge.get("edge_id", ""),
        "from_node": from_node,
        "to_node": to_node,
        "from_text": from_text,
        "to_text": to_text,
        "relation_type": relation_type,
        "justification": justification,
        "label": 1,
        "negative_type": None,
    }


def generate_hard_negative(
    candidate_edge: dict,
    query: str,
    entity_labels: dict[str, str],
) -> dict:
    """Generate a hard negative from a candidate edge that wasn't selected."""
    from_node = candidate_edge.get("from_node", "")
    to_node = candidate_edge.get("to_node", "")
    relation_type = candidate_edge.get("relation_type", "RELATED_TO")
    
    from_text = entity_labels.get(from_node, from_node)
    to_text = entity_labels.get(to_node, to_node)
    justification = candidate_edge.get("justification", "")
    
    return {
        "query": query,
        "edge_id": candidate_edge.get("edge_id", ""),
        "from_node": from_node,
        "to_node": to_node,
        "from_text": from_text,
        "to_text": to_text,
        "relation_type": relation_type,
        "justification": justification,
        "label": 0,
        "negative_type": "hard_negative",
    }


def generate_wrong_relation_negative(
    positive: dict,
    entity_labels: dict[str, str],
) -> dict:
    """Generate a negative with same nodes but wrong relation type."""
    # Pick a different relation type
    wrong_types = [r for r in RELATION_TYPES if r != positive["relation_type"]]
    wrong_relation = random.choice(wrong_types) if wrong_types else "RELATED_TO"
    
    return {
        "query": positive["query"],
        "edge_id": positive["edge_id"] + "_wrong_rel",
        "from_node": positive["from_node"],
        "to_node": positive["to_node"],
        "from_text": positive["from_text"],
        "to_text": positive["to_text"],
        "relation_type": wrong_relation,
        "justification": positive["justification"],
        "label": 0,
        "negative_type": "wrong_relation",
    }


def generate_confusable_negative(
    positive: dict,
    all_positives: list[dict],
    entity_labels: dict[str, str],
) -> dict | None:
    """Generate a confusable negative: similar justification but wrong nodes."""
    # Find another positive with similar relation type
    candidates = [p for p in all_positives 
                  if p["relation_type"] == positive["relation_type"]
                  and p["from_node"] != positive["from_node"]]
    
    if not candidates:
        return None
    
    other = random.choice(candidates)
    
    return {
        "query": positive["query"],
        "edge_id": positive["edge_id"] + "_confusable",
        "from_node": other["from_node"],  # Wrong nodes
        "to_node": other["to_node"],
        "from_text": other["from_text"],
        "to_text": other["to_text"],
        "relation_type": positive["relation_type"],
        "justification": positive["justification"],  # Keep justification
        "label": 0,
        "negative_type": "confusable",
    }


def generate_random_negative(
    query: str,
    all_edges: list[dict],
    entity_labels: dict[str, str],
) -> dict | None:
    """Generate a random negative from the edge pool."""
    if not all_edges:
        return None
    
    edge = random.choice(all_edges)
    from_node = edge.get("from_node", "")
    to_node = edge.get("to_node", "")
    
    return {
        "query": query,
        "edge_id": edge.get("edge_id", "") + "_random",
        "from_node": from_node,
        "to_node": to_node,
        "from_text": entity_labels.get(from_node, from_node),
        "to_text": entity_labels.get(to_node, to_node),
        "relation_type": edge.get("relation_type", "RELATED_TO"),
        "justification": edge.get("justification", ""),
        "label": 0,
        "negative_type": "random",
    }


async def generate_training_data():
    """Generate the full training dataset."""
    logger.info("=" * 60)
    logger.info("EDGE SCORER DATA GENERATION - Phase B (5k pairs)")
    logger.info("=" * 60)
    
    # Load resources
    traces = load_traces()
    entity_labels = await fetch_entity_labels()
    all_edges = await fetch_all_edges()
    
    # Generate positives from database edges (approved edges are "good")
    # Each edge can be paired with multiple synthetic queries
    positives = []
    
    # Load sample questions for pairing with edges
    sample_queries = []
    datasets = [
        PROJECT_ROOT / "eval" / "datasets" / "bakeoff_depth_v1.jsonl",
        PROJECT_ROOT / "eval" / "datasets" / "bakeoff_sanity_gate_v1.jsonl",
        PROJECT_ROOT / "eval" / "datasets" / "regression_unexpected_fails.jsonl",
    ]
    for ds_path in datasets:
        if ds_path.exists():
            with open(ds_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        q = data.get("question") or data.get("question_ar", "")
                        if q:
                            sample_queries.append(q)
                    except:
                        pass
    
    logger.info(f"Loaded {len(sample_queries)} sample queries for pairing")
    
    if not sample_queries:
        sample_queries = ["ما هي العلاقة بين الركائز المختلفة؟"]
    
    # Create positives from database edges (approved edges are good)
    for edge in all_edges:
        # Pair each edge with a random relevant query
        query = random.choice(sample_queries)
        
        from_node = edge.get("from_node", "")
        to_node = edge.get("to_node", "")
        relation_type = edge.get("relation_type", "RELATED_TO")
        
        from_text = entity_labels.get(from_node, from_node)
        to_text = entity_labels.get(to_node, to_node)
        justification = edge.get("justification", "") or ""
        
        if from_node and to_node:
            positives.append({
                "query": query,
                "edge_id": edge.get("edge_id", ""),
                "from_node": from_node,
                "to_node": to_node,
                "from_text": from_text,
                "to_text": to_text,
                "relation_type": relation_type,
                "justification": justification,
                "label": 1,
                "negative_type": None,
            })
    
    # Also add positives from traces (if any)
    for trace in traces:
        query = trace.get("question") or trace.get("question_ar", "")
        if not query:
            continue
        
        used_edges = extract_used_edges(trace)
        for edge in used_edges:
            pos = generate_positive(edge, query, entity_labels)
            if pos["from_node"] and pos["to_node"]:
                positives.append(pos)
    
    logger.info(f"Extracted {len(positives)} raw positives")
    
    # Deduplicate positives by (from_node, to_node, relation_type)
    seen_pos = set()
    unique_positives = []
    for pos in positives:
        key = (pos["from_node"], pos["to_node"], pos["relation_type"])
        if key not in seen_pos:
            seen_pos.add(key)
            unique_positives.append(pos)
    
    positives = unique_positives[:TARGET_POSITIVES]
    logger.info(f"Deduplicated to {len(positives)} unique positives")
    
    # Generate negatives
    negatives = []
    used_edge_keys = {(p["from_node"], p["to_node"]) for p in positives}
    
    # 1. Hard negatives - valid edges paired with WRONG queries
    # An edge that's good for one topic is a hard negative for a different topic
    hard_neg_count = int(TARGET_NEGATIVES * HARD_NEGATIVE_RATIO)
    queries = list(set(p["query"] for p in positives))
    
    # Create hard negatives by mismatching edges and queries
    # Use multiple wrong queries per positive to get enough hard negatives
    neg_per_pos = max(2, hard_neg_count // len(positives) + 1)
    
    for pos in positives:
        if len(negatives) >= hard_neg_count:
            break
        
        # Find queries different from the positive's query
        other_queries = [q for q in queries if q != pos["query"]]
        if not other_queries:
            continue
        
        # Generate multiple hard negatives per positive
        for _ in range(min(neg_per_pos, len(other_queries))):
            if len(negatives) >= hard_neg_count:
                break
            
            wrong_query = random.choice(other_queries)
            other_queries.remove(wrong_query)  # Don't reuse same query
            
            # Same edge but wrong query = hard negative
            neg = {
                "query": wrong_query,
                "edge_id": pos["edge_id"] + f"_hard_{len(negatives)}",
                "from_node": pos["from_node"],
                "to_node": pos["to_node"],
                "from_text": pos["from_text"],
                "to_text": pos["to_text"],
                "relation_type": pos["relation_type"],
                "justification": pos["justification"],
                "label": 0,
                "negative_type": "hard_negative",
            }
            negatives.append(neg)
    
    hard_neg_actual = len(negatives)
    logger.info(f"Generated {hard_neg_actual} hard negatives")
    
    # 2. Wrong relation type negatives (~20%)
    wrong_rel_target = int(TARGET_NEGATIVES * 0.20)
    for _ in range(wrong_rel_target):
        if len(negatives) >= TARGET_NEGATIVES:
            break
        pos = random.choice(positives)
        neg = generate_wrong_relation_negative(pos, entity_labels)
        negatives.append(neg)
    
    wrong_rel_count = len(negatives) - hard_neg_actual
    logger.info(f"Generated {wrong_rel_count} wrong-relation negatives")
    
    # 3. Confusable negatives (~15%)
    confusable_target = int(TARGET_NEGATIVES * 0.15)
    confusable_count = 0
    for _ in range(confusable_target):
        if len(negatives) >= TARGET_NEGATIVES:
            break
        pos = random.choice(positives)
        neg = generate_confusable_negative(pos, positives, entity_labels)
        if neg:
            negatives.append(neg)
            confusable_count += 1
    
    logger.info(f"Generated {confusable_count} confusable negatives")
    
    # 4. Random negatives (fill remainder)
    random_count = 0
    while len(negatives) < TARGET_NEGATIVES:
        query = random.choice(queries) if queries else ""
        neg = generate_random_negative(query, all_edges, entity_labels)
        if neg:
            negatives.append(neg)
            random_count += 1
    
    logger.info(f"Generated {random_count} random negatives")
    
    # Combine and shuffle
    all_data = positives + negatives
    random.shuffle(all_data)
    
    # Split by question (to prevent leakage)
    question_ids = list(set(d["query"][:50] for d in all_data))  # Use query prefix as ID
    random.shuffle(question_ids)
    split_point = int(len(question_ids) * 0.8)
    train_questions = set(question_ids[:split_point])
    
    train_data = [d for d in all_data if d["query"][:50] in train_questions]
    val_data = [d for d in all_data if d["query"][:50] not in train_questions]
    
    logger.info(f"\nDataset split:")
    logger.info(f"  Train: {len(train_data)} examples")
    logger.info(f"  Val: {len(val_data)} examples")
    
    # Count by type
    train_pos = sum(1 for d in train_data if d["label"] == 1)
    train_neg = len(train_data) - train_pos
    val_pos = sum(1 for d in val_data if d["label"] == 1)
    val_neg = len(val_data) - val_pos
    
    logger.info(f"  Train: {train_pos} pos / {train_neg} neg")
    logger.info(f"  Val: {val_pos} pos / {val_neg} neg")
    
    # Count negative types
    neg_types = defaultdict(int)
    for d in all_data:
        if d["label"] == 0:
            neg_types[d.get("negative_type", "unknown")] += 1
    
    logger.info(f"\nNegative breakdown:")
    for neg_type, count in sorted(neg_types.items()):
        pct = 100.0 * count / len(negatives) if negatives else 0
        logger.info(f"  {neg_type}: {count} ({pct:.1f}%)")
    
    hard_neg_pct = neg_types.get("hard_negative", 0) / len(negatives) if negatives else 0
    logger.info(f"\nHard negative ratio: {hard_neg_pct:.1%} (target: ≥{HARD_NEGATIVE_RATIO:.0%})")
    
    # Save files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = OUTPUT_DIR / "edge_scorer_train_v2.jsonl"
    val_path = OUTPUT_DIR / "edge_scorer_val_v2.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for d in train_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for d in val_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    # Save stats
    stats = {
        "total": len(all_data),
        "positives": len(positives),
        "negatives": len(negatives),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "hard_negative_ratio": hard_neg_pct,
        "negative_breakdown": dict(neg_types),
    }
    
    with open(OUTPUT_DIR / "edge_scorer_stats_v2.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nSaved:")
    logger.info(f"  {train_path}")
    logger.info(f"  {val_path}")
    logger.info(f"  {OUTPUT_DIR / 'edge_scorer_stats_v2.json'}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA GENERATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(generate_training_data())
