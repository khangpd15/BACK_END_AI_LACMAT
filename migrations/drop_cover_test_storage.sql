-- =============================================================================
-- REVERSIBLE DOWN MIGRATION: drop_cover_test_storage.sql
-- WARNING: Drops all Cover Test storage tables and indexes
-- =============================================================================

DROP TABLE IF EXISTS cover_test_results CASCADE;
DROP TABLE IF EXISTS cover_test_images CASCADE;
DROP TABLE IF EXISTS cover_test_cycles CASCADE;
DROP TABLE IF EXISTS cover_test_sessions CASCADE;
