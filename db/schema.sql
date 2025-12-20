-- Wellbeing Data Foundation Database Schema
-- PostgreSQL with pgvector extension

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- NOTE: pgvector ("vector" extension) is not available on all Windows Postgres installs.
-- This schema is compatible without pgvector. For enterprise vector search, use Azure AI Search.

-- =============================================================================
-- Source Documents and Ingestion
-- =============================================================================

CREATE TABLE source_document (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_name VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 hash
    framework_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE ingestion_run (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_doc_id UUID REFERENCES source_document(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    entities_extracted INTEGER DEFAULT 0,
    evidence_extracted INTEGER DEFAULT 0,
    validation_errors JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- =============================================================================
-- Core Entities: Pillars, Core Values, Sub-Values
-- =============================================================================

CREATE TABLE pillar (
    id VARCHAR(50) PRIMARY KEY,  -- e.g., P001
    name_ar VARCHAR(255) NOT NULL,
    name_en VARCHAR(255),  -- Nullable per MVP policy
    description_ar TEXT,
    source_doc_id UUID REFERENCES source_document(id),
    source_anchor JSONB NOT NULL,  -- {anchor_type, anchor_id, anchor_range}
    ingestion_run_id UUID REFERENCES ingestion_run(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(name_ar, source_doc_id)
);

CREATE TABLE core_value (
    id VARCHAR(50) PRIMARY KEY,  -- e.g., CV001
    pillar_id VARCHAR(50) REFERENCES pillar(id),
    name_ar VARCHAR(255) NOT NULL,
    name_en VARCHAR(255),
    definition_ar TEXT,
    source_doc_id UUID REFERENCES source_document(id),
    source_anchor JSONB NOT NULL,
    ingestion_run_id UUID REFERENCES ingestion_run(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(name_ar, pillar_id)
);

CREATE TABLE sub_value (
    id VARCHAR(50) PRIMARY KEY,  -- e.g., SV001
    core_value_id VARCHAR(50) REFERENCES core_value(id),
    name_ar VARCHAR(255) NOT NULL,
    name_en VARCHAR(255),
    definition_ar TEXT,
    source_doc_id UUID REFERENCES source_document(id),
    source_anchor JSONB NOT NULL,
    ingestion_run_id UUID REFERENCES ingestion_run(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(name_ar, core_value_id)
);

-- =============================================================================
-- Evidence: Quran, Hadith, and other references
-- =============================================================================

CREATE TABLE evidence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL,  -- pillar | core_value | sub_value
    entity_id VARCHAR(50) NOT NULL,
    evidence_type VARCHAR(50) NOT NULL,  -- quran | hadith | book
    ref_raw TEXT NOT NULL,  -- Original reference string
    ref_norm VARCHAR(255),  -- Normalized canonical form
    text_ar TEXT NOT NULL,
    -- Structured fields for Quran
    surah_name_ar VARCHAR(100),
    surah_number INTEGER,
    ayah_number INTEGER,
    -- Structured fields for Hadith
    hadith_collection VARCHAR(100),
    hadith_number INTEGER,
    -- Parse status
    parse_status VARCHAR(50) DEFAULT 'success',  -- success | failed | needs_review
    -- Provenance
    source_doc_id UUID REFERENCES source_document(id),
    source_anchor JSONB NOT NULL,
    ingestion_run_id UUID REFERENCES ingestion_run(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_evidence_entity ON evidence(entity_type, entity_id);
CREATE INDEX idx_evidence_type ON evidence(evidence_type);
CREATE INDEX idx_evidence_parse_status ON evidence(parse_status);

-- =============================================================================
-- Text Blocks (definitions, commentaries)
-- =============================================================================

CREATE TABLE text_block (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(50) NOT NULL,
    block_type VARCHAR(50) NOT NULL,  -- definition | commentary
    text_ar TEXT NOT NULL,
    text_en TEXT,  -- Nullable per MVP policy
    source_doc_id UUID REFERENCES source_document(id),
    source_anchor JSONB NOT NULL,
    ingestion_run_id UUID REFERENCES ingestion_run(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_text_block_entity ON text_block(entity_type, entity_id);

-- =============================================================================
-- Chunks for Vector Search (First-class table per D.2(3))
-- =============================================================================

CREATE TABLE chunk (
    chunk_id VARCHAR(50) PRIMARY KEY,  -- e.g., CH_000123
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(50) NOT NULL,
    chunk_type VARCHAR(50) NOT NULL,  -- definition | evidence | commentary
    text_ar TEXT NOT NULL,
    text_en TEXT,  -- Nullable per MVP policy
    source_doc_id UUID REFERENCES source_document(id),
    source_anchor VARCHAR(255) NOT NULL,
    token_count_estimate INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_chunk_entity ON chunk(entity_type, entity_id);
CREATE INDEX idx_chunk_type ON chunk(chunk_type);

-- =============================================================================
-- Argument-grade scholar graph primitives (Phase 3)
-- =============================================================================

CREATE TABLE IF NOT EXISTS argument_claim (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text_ar TEXT NOT NULL,
    entity_type VARCHAR(50),   -- optional: pillar|core_value|sub_value
    entity_id VARCHAR(50),     -- optional: entity id
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_argument_claim_entity ON argument_claim(entity_type, entity_id);

CREATE TABLE IF NOT EXISTS argument_evidence_span (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id VARCHAR(50) REFERENCES chunk(chunk_id),
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    quote TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_argument_span_chunk ON argument_evidence_span(chunk_id);

CREATE TABLE IF NOT EXISTS argument_argument (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_claim_id UUID REFERENCES argument_claim(id),
    to_claim_id UUID REFERENCES argument_claim(id),
    relation_type VARCHAR(50) NOT NULL, -- ENTAILS|TENSION_WITH|RECONCILED_BY|CONDITIONAL_ON|ENABLES|REINFORCES
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_argument_argument_from ON argument_argument(from_claim_id);
CREATE INDEX IF NOT EXISTS idx_argument_argument_to ON argument_argument(to_claim_id);

CREATE TABLE IF NOT EXISTS argument_conflict (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_a_id UUID REFERENCES argument_claim(id),
    claim_b_id UUID REFERENCES argument_claim(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS argument_resolution (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_id UUID REFERENCES argument_conflict(id),
    resolution_claim_id UUID REFERENCES argument_claim(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- Embeddings (vector search)
-- =============================================================================

CREATE TABLE embedding (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id VARCHAR(50) REFERENCES chunk(chunk_id),
    -- Stored for audit/debug only in no-pgvector environments.
    -- When using Azure AI Search, vectors are indexed there.
    vector DOUBLE PRECISION[],
    model VARCHAR(100) NOT NULL,
    dims INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_embedding_chunk ON embedding(chunk_id);
-- Ensure upsert works by chunk_id
CREATE UNIQUE INDEX uq_embedding_chunk_id ON embedding(chunk_id);

-- =============================================================================
-- Knowledge Graph Edges (with provenance per D.2(7))
-- =============================================================================

CREATE TABLE edge (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_type VARCHAR(50) NOT NULL,
    from_id VARCHAR(50) NOT NULL,
    rel_type VARCHAR(50) NOT NULL,  -- CONTAINS | SUPPORTED_BY | RELATES_TO | CROSS_REFERENCES
    -- Semantic scholar relation type (strict controlled vocabulary).
    -- IMPORTANT: `rel_type` remains structural/topological; `relation_type` is semantic and optional.
    relation_type VARCHAR(50),
    to_type VARCHAR(50) NOT NULL,
    to_id VARCHAR(50) NOT NULL,
    -- Provenance fields (mandatory per D.2(7))
    created_method VARCHAR(50) NOT NULL,  -- rule_exact_match | rule_lemma | embedding_candidate | human_approved
    created_by VARCHAR(100) NOT NULL,
    justification TEXT,  -- e.g., matched term, similarity score
    -- Deterministic optional score for semantic edges derived from evidence density.
    strength_score FLOAT,
    status VARCHAR(50) DEFAULT 'candidate',  -- candidate | approved | rejected
    score FLOAT,  -- For embedding candidates
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    -- NOTE:
    -- Do NOT enforce structural uniqueness across all edges, because semantic edges
    -- (relation_type IS NOT NULL) must allow multiple semantic relations between the same
    -- endpoints (e.g., ENABLES and REINFORCES).
);

CREATE INDEX idx_edge_from ON edge(from_type, from_id);
CREATE INDEX idx_edge_to ON edge(to_type, to_id);
CREATE INDEX idx_edge_rel ON edge(rel_type);
CREATE INDEX idx_edge_relation_type ON edge(relation_type);
CREATE INDEX idx_edge_status ON edge(status);

-- Semantic uniqueness: allow multiple semantic relation types between same endpoints,
-- while preventing duplicates for the same relation_type.
CREATE UNIQUE INDEX IF NOT EXISTS uq_edge_semantic
ON edge(from_type, from_id, rel_type, relation_type, to_type, to_id)
WHERE relation_type IS NOT NULL;

-- Structural uniqueness: keep old behavior only for non-semantic edges.
CREATE UNIQUE INDEX IF NOT EXISTS uq_edge_structural
ON edge(from_type, from_id, rel_type, to_type, to_id)
WHERE relation_type IS NULL;

-- Backward-compatible schema upgrades (for existing DBs running an older schema.sql).
-- These are safe to re-run due to IF NOT EXISTS.
ALTER TABLE edge ADD COLUMN IF NOT EXISTS relation_type VARCHAR(50);
ALTER TABLE edge ADD COLUMN IF NOT EXISTS strength_score FLOAT;
-- Drop legacy structural unique constraint that blocked multiple semantic relation types.
ALTER TABLE edge DROP CONSTRAINT IF EXISTS edge_from_type_from_id_rel_type_to_type_to_id_key;

-- =============================================================================
-- Edge-level grounding: justification spans for edges (multi-span supported)
-- =============================================================================

CREATE TABLE IF NOT EXISTS edge_justification_span (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    edge_id UUID REFERENCES edge(id) ON DELETE CASCADE,
    chunk_id VARCHAR(50) REFERENCES chunk(chunk_id),
    span_start INTEGER NOT NULL DEFAULT 0,
    span_end INTEGER NOT NULL DEFAULT 0,
    quote TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(edge_id, chunk_id, span_start, span_end)
);

CREATE INDEX IF NOT EXISTS idx_edge_just_span_edge ON edge_justification_span(edge_id);
CREATE INDEX IF NOT EXISTS idx_edge_just_span_chunk ON edge_justification_span(chunk_id);

-- =============================================================================
-- Chunk References (links chunks to evidence)
-- =============================================================================

CREATE TABLE chunk_ref (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id VARCHAR(50) REFERENCES chunk(chunk_id),
    ref_type VARCHAR(50) NOT NULL,  -- quran | hadith | book
    -- Some sources include full ayah/hadith text; keep this unbounded to avoid truncation.
    ref TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_chunk_ref_chunk ON chunk_ref(chunk_id);

-- =============================================================================
-- Production ask observability (UI support)
-- =============================================================================

-- Append-only ask run storage for replay + feedback loops.
-- NOTE: payload fields are capped at write-time to prevent unbounded growth.
CREATE TABLE IF NOT EXISTS ask_run (
    request_id UUID PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    question TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'ar',
    mode VARCHAR(50) NOT NULL,
    engine VARCHAR(50) NOT NULL,
    latency_ms INTEGER DEFAULT 0,
    contract_outcome VARCHAR(20),
    contract_reasons JSONB DEFAULT '[]'::jsonb,
    abstain_reason TEXT,
    final_response JSONB NOT NULL,
    graph_trace JSONB NOT NULL,
    citations_spans JSONB NOT NULL,
    muhasibi_trace JSONB DEFAULT '[]'::jsonb,
    truncated_fields JSONB DEFAULT '{}'::jsonb,
    original_counts JSONB DEFAULT '{}'::jsonb,
    debug_summary JSONB DEFAULT '{}'::jsonb
);

-- Backward-compatible upgrade for existing DBs where ask_run was created before this column existed.
ALTER TABLE ask_run ADD COLUMN IF NOT EXISTS muhasibi_trace JSONB DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_ask_run_created_at ON ask_run(created_at);
CREATE INDEX IF NOT EXISTS idx_ask_run_contract_outcome ON ask_run(contract_outcome);

CREATE TABLE IF NOT EXISTS ask_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID REFERENCES ask_run(request_id) ON DELETE CASCADE,
    rating INTEGER,
    tags TEXT[],
    comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ask_feedback_request_id ON ask_feedback(request_id);

-- =============================================================================
-- World Model: Mechanism Nodes (thin wrapper over existing hierarchy)
-- =============================================================================

CREATE TABLE IF NOT EXISTS mechanism_node (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    -- Anchor to existing ontology via ref_kind + ref_id (no duplicate universe)
    ref_kind VARCHAR(50) NOT NULL,  -- pillar|core_value|sub_value|mechanism|outcome
    ref_id VARCHAR(50) NOT NULL,    -- actual ID from referenced table (P001, CV001, SV001, etc.)
    label_ar TEXT NOT NULL,         -- cached display label (derived from referenced entity)
    source_id UUID REFERENCES source_document(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Ensure no duplicate wrappers for same entity
    UNIQUE(ref_kind, ref_id)
);

CREATE INDEX IF NOT EXISTS idx_mechanism_node_ref ON mechanism_node(ref_kind, ref_id);

-- =============================================================================
-- World Model: Mechanism Edges with Signed Polarity
-- =============================================================================

CREATE TABLE IF NOT EXISTS mechanism_edge (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_node UUID REFERENCES mechanism_node(id) ON DELETE CASCADE,
    to_node UUID REFERENCES mechanism_node(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) NOT NULL,  -- ENABLES|REINFORCES|CONDITIONAL_ON|INHIBITS|TENSION_WITH|RESOLVES_WITH|COMPLEMENTS
    -- Polarity as signed integer: +1 (positive/enabling) or -1 (negative/inhibiting)
    polarity SMALLINT NOT NULL DEFAULT 1 CHECK (polarity IN (-1, 1)),
    -- Evidence-based confidence (span count + diversity), not hardcoded
    confidence FLOAT DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mechanism_edge_from ON mechanism_edge(from_node);
CREATE INDEX IF NOT EXISTS idx_mechanism_edge_to ON mechanism_edge(to_node);
CREATE INDEX IF NOT EXISTS idx_mechanism_edge_relation ON mechanism_edge(relation_type);
-- Semantic uniqueness: prevent duplicates for the same relation_type between the same nodes.
-- Note: if your DB already has duplicates, you must deduplicate before creating this index.
CREATE UNIQUE INDEX IF NOT EXISTS uq_mechanism_edge_semantic
ON mechanism_edge(from_node, to_node, relation_type);

-- =============================================================================
-- World Model: Edge-level Spans (hard gate: no edge without span)
-- =============================================================================

CREATE TABLE IF NOT EXISTS mechanism_edge_span (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    edge_id UUID REFERENCES mechanism_edge(id) ON DELETE CASCADE,
    chunk_id VARCHAR(50) REFERENCES chunk(chunk_id),
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    quote TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(edge_id, chunk_id, span_start, span_end)
);

CREATE INDEX IF NOT EXISTS idx_mechanism_edge_span_edge ON mechanism_edge_span(edge_id);
CREATE INDEX IF NOT EXISTS idx_mechanism_edge_span_chunk ON mechanism_edge_span(chunk_id);

-- =============================================================================
-- World Model: Feedback Loops (no stored summary - generate on-the-fly)
-- =============================================================================

CREATE TABLE IF NOT EXISTS feedback_loop (
    loop_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    edge_ids UUID[] NOT NULL,
    -- Loop type derived from polarity product (reinforcing if +1, balancing if -1)
    loop_type VARCHAR(20) NOT NULL CHECK (loop_type IN ('reinforcing', 'balancing')),
    -- NO loop_summary_ar column - summaries generated on-the-fly from edge spans
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_loop_type ON feedback_loop(loop_type);

-- =============================================================================
-- Views for common queries
-- =============================================================================

-- Hierarchy view: Pillar -> Core Value -> Sub Value
CREATE VIEW v_value_hierarchy AS
SELECT
    p.id AS pillar_id,
    p.name_ar AS pillar_name_ar,
    cv.id AS core_value_id,
    cv.name_ar AS core_value_name_ar,
    sv.id AS sub_value_id,
    sv.name_ar AS sub_value_name_ar
FROM pillar p
LEFT JOIN core_value cv ON cv.pillar_id = p.id
LEFT JOIN sub_value sv ON sv.core_value_id = cv.id;

-- Evidence counts per entity
CREATE VIEW v_evidence_counts AS
SELECT
    entity_type,
    entity_id,
    evidence_type,
    COUNT(*) AS evidence_count
FROM evidence
GROUP BY entity_type, entity_id, evidence_type;

