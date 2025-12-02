-- ============================================================================
-- RAGNAROK V8.0 APEX - SCHEMA MIGRATION (ATOMIC)
-- ============================================================================
-- Run this in Supabase SQL Editor: https://app.supabase.com → SQL Editor
--
-- Features:
--   - Transaction-wrapped for atomicity (all or nothing)
--   - Safety checks before modifications
--   - Distributed lock function for idempotency
--   - Performance indexes for hot path queries
--   - Optional DLQ table for persistence
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. SAFETY CHECK - Verify required tables exist
-- ============================================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'leads') THEN
        RAISE EXCEPTION 'Table "leads" does not exist.';
    END IF;
    IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'discovery_leads') THEN
        RAISE EXCEPTION 'Table "discovery_leads" does not exist.';
    END IF;
END $$;

-- ============================================================================
-- 2. DISCOVERY LEADS UPGRADE
-- ============================================================================
ALTER TABLE discovery_leads
ADD COLUMN IF NOT EXISTS teaser_video_url TEXT,
ADD COLUMN IF NOT EXISTS teaser_cost_usd NUMERIC,
ADD COLUMN IF NOT EXISTS teaser_hook TEXT,
ADD COLUMN IF NOT EXISTS email_sent BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS engaged_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS engagement_started_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS error_message TEXT,
ADD COLUMN IF NOT EXISTS correlation_id TEXT;

-- ============================================================================
-- 3. LEADS UPGRADE
-- ============================================================================
ALTER TABLE leads
ADD COLUMN IF NOT EXISTS processing_started_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS failed_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS correlation_id TEXT;

-- ============================================================================
-- 4. DISTRIBUTED LOCK FUNCTION
-- ============================================================================
-- This function implements optimistic locking for job processing.
-- Returns TRUE if lock acquired, FALSE if already processing.
-- Used by the V8 orchestrator to prevent duplicate job execution.

CREATE OR REPLACE FUNCTION try_acquire_job_lock(
    p_job_id UUID,
    p_job_type TEXT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_rows_updated INT;
BEGIN
    IF p_job_type = 'payment' THEN
        UPDATE leads
        SET status = 'processing', processing_started_at = NOW()
        WHERE id = p_job_id AND status = 'paid_client';
        GET DIAGNOSTICS v_rows_updated = ROW_COUNT;
    ELSIF p_job_type = 'hunter' THEN
        UPDATE discovery_leads
        SET status = 'engaging', engagement_started_at = NOW()
        WHERE id = p_job_id AND status = 'new';
        GET DIAGNOSTICS v_rows_updated = ROW_COUNT;
    ELSE
        RETURN FALSE;
    END IF;
    RETURN v_rows_updated > 0;
END;
$$;

GRANT EXECUTE ON FUNCTION try_acquire_job_lock(UUID, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION try_acquire_job_lock(UUID, TEXT) TO authenticated;

-- ============================================================================
-- 5. PERFORMANCE INDEXES
-- ============================================================================
-- These indexes optimize the polling queries used by watchers

-- Payment watcher: SELECT * FROM leads WHERE status = 'paid_client'
CREATE INDEX IF NOT EXISTS idx_leads_status_paid
ON leads(status) WHERE status = 'paid_client';

-- Hunter watcher: SELECT * FROM discovery_leads WHERE score >= 8 AND status = 'new'
CREATE INDEX IF NOT EXISTS idx_discovery_leads_hot
ON discovery_leads(qualification_score, status)
WHERE status = 'new' AND qualification_score >= 8;

-- ============================================================================
-- 6. DLQ TABLE (Optional but recommended for persistent DLQ)
-- ============================================================================
CREATE TABLE IF NOT EXISTS dlq_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue TEXT NOT NULL,
    payload JSONB NOT NULL,
    error TEXT NOT NULL,
    category TEXT NOT NULL,
    attempt_count INT DEFAULT 0,
    next_retry TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    fingerprint TEXT NOT NULL,
    is_poison BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dlq_processing
ON dlq_messages(queue, next_retry) WHERE is_poison = FALSE;

COMMIT;

-- ============================================================================
-- VERIFICATION (Run after commit to confirm success)
-- ============================================================================
SELECT 'RAGNAROK V8.0 APEX MIGRATION COMPLETE' as status;
SELECT proname FROM pg_proc WHERE proname = 'try_acquire_job_lock';
SELECT indexname FROM pg_indexes WHERE indexname LIKE 'idx_%leads%';
