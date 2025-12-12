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
    to_type VARCHAR(50) NOT NULL,
    to_id VARCHAR(50) NOT NULL,
    -- Provenance fields (mandatory per D.2(7))
    created_method VARCHAR(50) NOT NULL,  -- rule_exact_match | rule_lemma | embedding_candidate | human_approved
    created_by VARCHAR(100) NOT NULL,
    justification TEXT,  -- e.g., matched term, similarity score
    status VARCHAR(50) DEFAULT 'candidate',  -- candidate | approved | rejected
    score FLOAT,  -- For embedding candidates
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(from_type, from_id, rel_type, to_type, to_id)
);

CREATE INDEX idx_edge_from ON edge(from_type, from_id);
CREATE INDEX idx_edge_to ON edge(to_type, to_id);
CREATE INDEX idx_edge_rel ON edge(rel_type);
CREATE INDEX idx_edge_status ON edge(status);

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

