-- =============================================================================
-- MIGRATION: add_cloudinary_and_fps_fields.sql
-- RemiCare Strabismus AI - 10-15 FPS Primary Model & Cloudinary Storage Integration
-- =============================================================================

-- Add Cloudinary and 10-15 FPS fields to cover_test_results
ALTER TABLE cover_test_results ADD COLUMN IF NOT EXISTS image_url TEXT;
ALTER TABLE cover_test_results ADD COLUMN IF NOT EXISTS cloudinary_public_id VARCHAR(255);
ALTER TABLE cover_test_results ADD COLUMN IF NOT EXISTS confidence REAL;
ALTER TABLE cover_test_results ADD COLUMN IF NOT EXISTS model_source VARCHAR(50) DEFAULT 'fps_10_15_model';

-- Update default model_name and version for new records
ALTER TABLE cover_test_results ALTER COLUMN model_name SET DEFAULT 'remicare-fps-10-15';
ALTER TABLE cover_test_results ALTER COLUMN model_version SET DEFAULT '10-15fps-v1.1.0';
ALTER TABLE cover_test_results ALTER COLUMN status SET DEFAULT 'COMPLETED';
ALTER TABLE cover_test_results ALTER COLUMN notice SET DEFAULT 'Screening result derived from 10-15 FPS model consensus aggregation.';

-- Add index on model_source for efficient filtering
CREATE INDEX IF NOT EXISTS idx_cover_test_results_model_source ON cover_test_results (model_source);
CREATE INDEX IF NOT EXISTS idx_cover_test_results_prediction ON cover_test_results (prediction);
