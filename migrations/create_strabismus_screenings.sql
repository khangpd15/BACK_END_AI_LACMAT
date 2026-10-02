-- Migration: Create strabismus_screenings table for RemiCare Strabismus AI
-- Stores metadata only; zero raw/biometric image storage

CREATE TABLE IF NOT EXISTS strabismus_screenings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    status VARCHAR(50) NOT NULL,
    strabismus_probability REAL,
    confidence REAL,
    quality_score REAL,
    threshold REAL DEFAULT 0.20 NOT NULL,
    model_version VARCHAR(100) DEFAULT 'remicare-bilateral-resnet18-v1' NOT NULL,
    inference_latency_ms REAL,
    image_saved BOOLEAN DEFAULT FALSE NOT NULL,
    failure_reason VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_strabismus_screenings_created_at ON strabismus_screenings (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_strabismus_screenings_status ON strabismus_screenings (status);
