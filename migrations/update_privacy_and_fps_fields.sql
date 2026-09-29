-- =============================================================================
-- MIGRATION: update_cover_test_privacy_and_fps_fields.sql
-- RemiCare Strabismus AI - Privacy Hardening & 10-15 FPS Primary Model Integration
-- 1. Protect Customer Privacy: Drop all image storage (PII / Biometrics)
-- 2. Configure 10-15 FPS Consensus Model as primary screening engine
-- =============================================================================

-- Step 1: Remove customer biometric image fields from cover_test_results
ALTER TABLE cover_test_results DROP COLUMN IF EXISTS image_url;
ALTER TABLE cover_test_results DROP COLUMN IF EXISTS cloudinary_public_id;

-- Step 2: Remove cover_test_images table to permanently prevent storing customer facial/eye images
DROP TABLE IF EXISTS cover_test_images CASCADE;

-- Step 3: Add 10-15 FPS Model tracking fields to cover_test_results
ALTER TABLE cover_test_results ADD COLUMN IF NOT EXISTS confidence REAL;
ALTER TABLE cover_test_results ADD COLUMN IF NOT EXISTS model_source VARCHAR(50) DEFAULT 'fps_10_15_model';

-- Step 4: Update default model_name and version for 10-15 FPS consensus pipeline
ALTER TABLE cover_test_results ALTER COLUMN model_name SET DEFAULT 'remicare-fps-10-15';
ALTER TABLE cover_test_results ALTER COLUMN model_version SET DEFAULT '10-15fps-v1.1.0';
ALTER TABLE cover_test_results ALTER COLUMN status SET DEFAULT 'COMPLETED';
ALTER TABLE cover_test_results ALTER COLUMN notice SET DEFAULT 'Screening result derived from 10-15 FPS model consensus aggregation.';

-- Step 5: Add index on model_source for efficient filtering between 10-15 FPS model and research baseline
CREATE INDEX IF NOT EXISTS idx_cover_test_results_model_source ON cover_test_results (model_source);
CREATE INDEX IF NOT EXISTS idx_cover_test_results_prediction ON cover_test_results (prediction);
