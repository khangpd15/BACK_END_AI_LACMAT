"""Unit test suite for Keep-Alive, Self-Ping, and Database Diagnostics."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.db.database import (
    check_db_connection,
    format_async_db_url,
    get_db_connect_args,
)
from app.services.keep_alive import KeepAliveService, get_keep_alive_service

client = TestClient(app)


def test_ping_endpoint():
    """Verify /ping endpoint responds with 200, alive status, and timestamp."""
    resp = client.get("/ping")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "alive"
    assert data["service"] == "remicare-strabismus-ai"
    assert "timestamp" in data


def test_health_endpoint_includes_keep_alive():
    """Verify /health endpoint includes keepAlive stats and model metadata."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "keepAlive" in data
    assert "isRunning" in data["keepAlive"]
    assert "intervalSeconds" in data["keepAlive"]


def test_health_endpoint_with_check_db():
    """Verify /health?check_db=true checks DB and reports database connection status."""
    resp = client.get("/health?check_db=true")
    assert resp.status_code == 200
    data = resp.json()
    assert "database" in data
    assert "status" in data["database"]
    assert "connected" in data["database"]


def test_db_health_endpoint():
    """Verify /health/db endpoint executes SELECT 1 and returns connection diagnostics."""
    resp = client.get("/health/db")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "connected" in data
    assert "latency_ms" in data


def test_get_db_connect_args_pooler():
    """Verify get_db_connect_args disables statement cache for Supabase Supavisor pooler."""
    # Port 6543
    args1 = get_db_connect_args("postgresql://user:pass@host:6543/postgres")
    assert args1.get("statement_cache_size") == 0
    assert args1.get("prepared_statement_cache_size") == 0

    # pooler.supabase.com
    args2 = get_db_connect_args("postgresql://user.ref:pass@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres")
    assert args2.get("statement_cache_size") == 0
    assert args2.get("prepared_statement_cache_size") == 0

    # Standard Postgres URL should not disable statement cache
    args3 = get_db_connect_args("postgresql://user:pass@localhost:5432/testdb")
    assert "statement_cache_size" not in args3


def test_get_db_connect_args_direct_supabase_warning(caplog):
    """Verify direct Supabase hostname triggers explicit warning log regarding Render IPv6 Errno 101."""
    import logging
    with caplog.at_level(logging.WARNING):
        get_db_connect_args("postgresql://postgres:pass@db.abcdefgh.supabase.co:5432/postgres")
        assert any("Errno 101" in record.message or "SUPABASE DIRECT CONNECTION" in record.message for record in caplog.records)


def test_keep_alive_service_resolve_url():
    """Verify target URL resolution precedence."""
    svc = KeepAliveService(custom_url="https://custom.onrender.com/")
    assert svc.resolve_target_url() == "https://custom.onrender.com"

    svc_empty = KeepAliveService(custom_url="")
    # When no url is set and env vars are empty
    with patch("app.services.keep_alive.SELF_PING_URL", ""), patch("app.services.keep_alive.RENDER_EXTERNAL_URL", ""):
        assert svc_empty.resolve_target_url() is None


def test_keep_alive_service_ping_cycle():
    """Verify keep_alive ping_cycle executes HTTP and DB pings."""
    import anyio

    async def _test():
        svc = KeepAliveService(custom_url="http://localhost:8000", ping_db=True)
        with patch.object(svc, "ping_http", new_callable=AsyncMock) as mock_http, \
             patch.object(svc, "ping_database", new_callable=AsyncMock) as mock_db:

            mock_http.return_value = {"success": True, "status_code": 200}
            mock_db.return_value = {"status": "healthy", "connected": True}

            res = await svc.ping_cycle()
            assert res["http"] == {"success": True, "status_code": 200}
            assert res["database"] == {"status": "healthy", "connected": True}
            mock_http.assert_awaited_once_with("http://localhost:8000")
            mock_db.assert_awaited_once()

    anyio.run(_test)


def test_keep_alive_service_status():
    """Verify get_status returns expected dictionary schema."""
    svc = KeepAliveService(custom_url="https://test.onrender.com", interval_seconds=300)
    status = svc.get_status()
    assert status["enabled"] is True
    assert status["intervalSeconds"] == 300
    assert status["targetUrl"] == "https://test.onrender.com"
    assert "httpPings" in status
    assert "databasePings" in status

