-- =============================================================================
-- MIGRATION: create_cover_test_storage.sql
-- RemiCare Cover Test Cloud Storage & Relational Metadata
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. TABLE: cover_test_sessions
CREATE TABLE IF NOT EXISTS cover_test_sessions (
    session_id            UUID PRIMARY KEY,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at          TIMESTAMPTZ,
    cycle_count           INTEGER NOT NULL DEFAULT 3,
    sampling_rate_hz      REAL NOT NULL DEFAULT 15.0,
    source_device         VARCHAR(50) NOT NULL DEFAULT 'WEBCAM',
    tracker               VARCHAR(50) NOT NULL DEFAULT 'MEDIAPIPE_IRIS',
    raw_schema_version    VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    storage_root          TEXT NOT NULL,
    processing_status     VARCHAR(30) NOT NULL DEFAULT 'SESSION_CREATED',
    error_message         TEXT,
    client_metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_sessions_status CHECK (
        processing_status IN (
            'SESSION_CREATED',
            'COLLECTING',
            'UPLOADING',
            'PROCESSING',
            'COMPLETED',
            'PARTIAL_SUCCESS',
            'FAILED'
        )
    ),
    CONSTRAINT chk_sessions_sampling_rate CHECK (sampling_rate_hz > 0)
);

CREATE INDEX IF NOT EXISTS idx_cover_test_sessions_status ON cover_test_sessions (processing_status);
CREATE INDEX IF NOT EXISTS idx_cover_test_sessions_created_at ON cover_test_sessions (created_at DESC);

-- 2. TABLE: cover_test_cycles
CREATE TABLE IF NOT EXISTS cover_test_cycles (
    cycle_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id            UUID NOT NULL REFERENCES cover_test_sessions(session_id) ON DELETE CASCADE,
    cycle_number          INTEGER NOT NULL,
    covered_eye           VARCHAR(20) NOT NULL,
    tracked_eye           VARCHAR(20) NOT NULL,
    sample_count          INTEGER NOT NULL DEFAULT 0,
    valid_sample_count    INTEGER NOT NULL DEFAULT 0,
    valid_ratio           REAL NOT NULL DEFAULT 0.0,
    mean_tracking_quality REAL NOT NULL DEFAULT 0.0,
    cycle_status          VARCHAR(20) NOT NULL DEFAULT 'COMPLETED',
    raw_storage_path      TEXT NOT NULL,
    duration_ms           INTEGER,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_session_cycle UNIQUE (session_id, cycle_number),
    CONSTRAINT chk_cycles_number CHECK (cycle_number >= 1),
    CONSTRAINT chk_cycles_covered_eye CHECK (covered_eye IN ('LEFT', 'RIGHT', 'ALTERNATING')),
    CONSTRAINT chk_cycles_tracked_eye CHECK (tracked_eye IN ('LEFT', 'RIGHT', 'BOTH')),
    CONSTRAINT chk_cycles_sample_counts CHECK (valid_sample_count <= sample_count AND sample_count >= 0),
    CONSTRAINT chk_cycles_quality CHECK (mean_tracking_quality BETWEEN 0.0 AND 1.0)
);

CREATE INDEX IF NOT EXISTS idx_cover_test_cycles_session ON cover_test_cycles (session_id);

-- 3. TABLE: cover_test_images
CREATE TABLE IF NOT EXISTS cover_test_images (
    image_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id            UUID NOT NULL REFERENCES cover_test_sessions(session_id) ON DELETE CASCADE,
    cycle_id              UUID NOT NULL REFERENCES cover_test_cycles(cycle_id) ON DELETE CASCADE,
    cycle_number          INTEGER NOT NULL,
    eye                   VARCHAR(10) NOT NULL,
    capture_event         VARCHAR(30) NOT NULL,
    timestamp             TIMESTAMPTZ NOT NULL,
    storage_path          TEXT NOT NULL,
    mime_type             VARCHAR(30) NOT NULL DEFAULT 'image/jpeg',
    width                 INTEGER NOT NULL,
    height                INTEGER NOT NULL,
    file_size             INTEGER NOT NULL,
    upload_status         VARCHAR(20) NOT NULL DEFAULT 'UPLOADED',
    crop_region           JSONB,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_cycle_eye UNIQUE (cycle_id, eye),
    CONSTRAINT uq_session_cycle_eye UNIQUE (session_id, cycle_number, eye),
    CONSTRAINT chk_images_eye CHECK (eye IN ('LEFT', 'RIGHT')),
    CONSTRAINT chk_images_capture_event CHECK (capture_event IN ('UNCOVER_LEFT', 'UNCOVER_RIGHT')),
    CONSTRAINT chk_images_dimensions CHECK (width > 0 AND height > 0 AND file_size > 0),
    CONSTRAINT chk_images_upload_status CHECK (upload_status IN ('PENDING', 'UPLOADED', 'FAILED'))
);

CREATE INDEX IF NOT EXISTS idx_cover_test_images_session ON cover_test_images (session_id);
CREATE INDEX IF NOT EXISTS idx_cover_test_images_cycle ON cover_test_images (cycle_id);

-- 4. TABLE: cover_test_results
CREATE TABLE IF NOT EXISTS cover_test_results (
    result_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id             UUID NOT NULL REFERENCES cover_test_sessions(session_id) ON DELETE CASCADE,
    model_name             VARCHAR(100) NOT NULL DEFAULT 'korean_shared_model',
    model_version          VARCHAR(50) NOT NULL DEFAULT 'shared-v1.0.0',
    feature_schema_version VARCHAR(50) NOT NULL DEFAULT 'shared-v1.0.0',
    status                 VARCHAR(30) NOT NULL DEFAULT 'TRANSFER_EXPERIMENT',
    input_compatible       BOOLEAN NOT NULL DEFAULT TRUE,
    prediction             VARCHAR(30) NOT NULL,
    class_probabilities    JSONB NOT NULL,
    domain_shift_warning   BOOLEAN NOT NULL DEFAULT TRUE,
    features_snapshot      JSONB,
    comparison_models      JSONB NOT NULL DEFAULT '[]'::jsonb,
    notice                 TEXT NOT NULL DEFAULT 'Research transfer experiment only — not a medical diagnosis.',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_session_model_version UNIQUE (session_id, model_name, model_version),
    CONSTRAINT chk_results_prediction CHECK (prediction IN ('NORMAL', 'STRABISMUS', 'INCONCLUSIVE'))
);

CREATE INDEX IF NOT EXISTS idx_cover_test_results_session ON cover_test_results (session_id);
