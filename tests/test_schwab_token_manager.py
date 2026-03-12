"""
Tests for SchwabTokenManager — proactive alerts, expiry stamping, status().
All file I/O is mocked via tmp_path.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_manager(tmp_path, token_overrides=None, meta_overrides=None, notifier=None):
    """Return a SchwabTokenManager wired to tmp_path files."""
    from src.exchange.schwab_token_manager import (
        SchwabTokenManager, REFRESH_TOKEN_TTL,
    )

    token = {
        "access_token":  "AT_test",
        "refresh_token": "RT_test",
        "token_type":    "Bearer",
        "expires_in":    1800,
        "scope":         "api",
        "refresh_token_expires_at": (
            datetime.now(timezone.utc) + timedelta(seconds=REFRESH_TOKEN_TTL)
        ).isoformat(),
        "last_refreshed_at": datetime.now(timezone.utc).isoformat(),
    }
    if token_overrides:
        token.update(token_overrides)

    meta = {"refresh_token_issued_at": time.time()}
    if meta_overrides:
        meta.update(meta_overrides)

    token_path = tmp_path / ".schwab_token.json"
    meta_path  = tmp_path / ".schwab_token_meta.json"
    token_path.write_text(json.dumps(token))
    meta_path.write_text(json.dumps(meta))

    with (
        patch("src.exchange.schwab_token_manager.TOKEN_PATH",      token_path),
        patch("src.exchange.schwab_token_manager.TOKEN_META_PATH",  meta_path),
    ):
        mgr = SchwabTokenManager(notifier=notifier)
    # store paths so tests can reload files
    mgr._token_path = token_path
    mgr._meta_path  = meta_path
    return mgr, token_path, meta_path


# ── 1. status() returns 'ok' when token is healthy ───────────────────────────
def test_status_ok(tmp_path):
    from src.exchange.schwab_token_manager import REFRESH_TOKEN_TTL
    mgr, tp, _ = _make_manager(tmp_path)
    with patch("src.exchange.schwab_token_manager.TOKEN_PATH", tp):
        s = mgr.status()
    assert s["health"] == "ok"
    assert s["refresh_token_days_left"] > 3
    assert s["needs_reauth"] is False


# ── 2. status() returns 'warning' at 2 days remaining ────────────────────────
def test_status_warning(tmp_path):
    expires_at = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    mgr, tp, _ = _make_manager(tmp_path, token_overrides={"refresh_token_expires_at": expires_at})
    with patch("src.exchange.schwab_token_manager.TOKEN_PATH", tp):
        s = mgr.status()
    assert s["health"] == "warning"
    assert s["needs_reauth"] is False


# ── 3. status() returns 'critical' at 12 hours remaining ─────────────────────
def test_status_critical(tmp_path):
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
    mgr, tp, _ = _make_manager(tmp_path, token_overrides={"refresh_token_expires_at": expires_at})
    with patch("src.exchange.schwab_token_manager.TOKEN_PATH", tp):
        s = mgr.status()
    assert s["health"] == "critical"
    assert s["needs_reauth"] is True


# ── 4. status() returns 'expired' when token is past expiry ──────────────────
def test_status_expired(tmp_path):
    expires_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    mgr, tp, _ = _make_manager(tmp_path, token_overrides={"refresh_token_expires_at": expires_at})
    with patch("src.exchange.schwab_token_manager.TOKEN_PATH", tp):
        s = mgr.status()
    assert s["health"] == "expired"
    assert s["needs_reauth"] is True


# ── 5. do_refresh() stamps last_refreshed_at and preserves refresh_token_expires_at ──
def test_do_refresh_stamps_timestamps(tmp_path):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token":  "NEW_AT",
        "refresh_token": "RT_test",   # same RT — no rotation
        "token_type": "Bearer", "expires_in": 1800, "scope": "api",
    }

    mgr, tp, mp = _make_manager(tmp_path)
    with (
        patch("src.exchange.schwab_token_manager.TOKEN_PATH",     tp),
        patch("src.exchange.schwab_token_manager.TOKEN_META_PATH", mp),
        patch("requests.post", return_value=mock_resp),
    ):
        result = mgr.do_refresh()

    assert result is True
    saved = json.loads(tp.read_text())
    assert saved["access_token"] == "NEW_AT"
    assert "last_refreshed_at" in saved
    assert "refresh_token_expires_at" in saved   # preserved from original


# ── 6. do_refresh() resets expiry when Schwab rotates the refresh token ───────
def test_do_refresh_new_refresh_token_resets_expiry(tmp_path):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token":  "NEW_AT",
        "refresh_token": "RT_BRAND_NEW",   # rotated
        "token_type": "Bearer", "expires_in": 1800, "scope": "api",
    }

    mgr, tp, mp = _make_manager(tmp_path)
    with (
        patch("src.exchange.schwab_token_manager.TOKEN_PATH",     tp),
        patch("src.exchange.schwab_token_manager.TOKEN_META_PATH", mp),
        patch("requests.post", return_value=mock_resp),
    ):
        mgr.do_refresh()

    saved = json.loads(tp.read_text())
    # New expiry should be ~7 days from now
    new_exp = datetime.fromisoformat(saved["refresh_token_expires_at"])
    days_left = (new_exp - datetime.now(timezone.utc)).total_seconds() / 86400
    assert 6.9 < days_left <= 7.1


# ── 7. _check_refresh_token_health() sends Telegram at < 1 day ───────────────
def test_alert_sent_critical(tmp_path):
    notifier = MagicMock()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=10)).isoformat()
    mgr, tp, _ = _make_manager(tmp_path,
        token_overrides={"refresh_token_expires_at": expires_at},
        notifier=notifier)

    with patch("src.exchange.schwab_token_manager.TOKEN_PATH", tp):
        mgr._check_refresh_token_health()

    notifier.send.assert_called_once()
    msg = notifier.send.call_args[0][0]
    assert "expires in" in msg.lower() or "expire" in msg.lower()


# ── 8. _check_refresh_token_health() sends alert at < 3 days ─────────────────
def test_alert_sent_warning(tmp_path):
    notifier = MagicMock()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    mgr, tp, _ = _make_manager(tmp_path,
        token_overrides={"refresh_token_expires_at": expires_at},
        notifier=notifier)

    with patch("src.exchange.schwab_token_manager.TOKEN_PATH", tp):
        mgr._check_refresh_token_health()

    notifier.send.assert_called_once()


# ── 9. _check_refresh_token_health() respects cooldown ───────────────────────
def test_alert_cooldown_respected(tmp_path):
    notifier = MagicMock()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=10)).isoformat()
    mgr, tp, _ = _make_manager(tmp_path,
        token_overrides={"refresh_token_expires_at": expires_at},
        notifier=notifier)

    # First alert fires
    mgr._last_alert_ts = time.time()   # simulate recent alert
    with patch("src.exchange.schwab_token_manager.TOKEN_PATH", tp):
        mgr._check_refresh_token_health()

    # Should NOT have sent because cooldown not elapsed
    notifier.send.assert_not_called()


# ── 10. _check_refresh_token_health() sends forced alert when expired ─────────
def test_alert_forced_when_expired(tmp_path):
    notifier = MagicMock()
    expires_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    mgr, tp, _ = _make_manager(tmp_path,
        token_overrides={"refresh_token_expires_at": expires_at},
        notifier=notifier)

    # Even with recent alert timestamp, force=True overrides cooldown
    mgr._last_alert_ts = time.time()
    with patch("src.exchange.schwab_token_manager.TOKEN_PATH", tp):
        mgr._check_refresh_token_health()

    notifier.send.assert_called_once()
    msg = notifier.send.call_args[0][0]
    assert "expired" in msg.lower() or "blind" in msg.lower()


# ── 11. status() includes access_token timing ────────────────────────────────
def test_status_includes_access_token_age(tmp_path):
    last_refresh = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    mgr, tp, _ = _make_manager(tmp_path,
        token_overrides={"last_refreshed_at": last_refresh})
    with patch("src.exchange.schwab_token_manager.TOKEN_PATH", tp):
        s = mgr.status()
    assert s["access_token_age_s"] is not None
    assert s["access_token_age_s"] > 290     # ~5 min
    assert s["access_token_expires_in_s"] is not None


# ── 12. no notifier → no crash ────────────────────────────────────────────────
def test_no_notifier_no_crash(tmp_path):
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    mgr, tp, _ = _make_manager(tmp_path,
        token_overrides={"refresh_token_expires_at": expires_at},
        notifier=None)
    # Should not raise
    with patch("src.exchange.schwab_token_manager.TOKEN_PATH", tp):
        mgr._check_refresh_token_health()
