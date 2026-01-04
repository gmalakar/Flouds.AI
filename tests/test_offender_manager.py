import os
import time

from app.modules.offender_manager import OffenderManager


def test_offender_blocks_after_threshold_env(monkeypatch):
    # Use environment variables to control thresholds (no DB needed)
    monkeypatch.setenv("FLOUDS_BLOCK_MAX_ATTEMPS", "1")
    monkeypatch.setenv("FLOUDS_BLOCK_WINDOW_SECONDS", "10")
    monkeypatch.setenv("FLOUDS_BLOCK_SECONDS", "2")

    mgr = OffenderManager()
    ip = "127.0.0.1"

    # initially not blocked
    blocked, _ = mgr.is_blocked(ip)
    assert blocked is False

    # first failed attempt -> not blocked yet
    blocked_now, reason = mgr.register_attempt(ip, tenant="")
    assert blocked_now is False
    assert reason == ""

    # second failed attempt should trigger block because max_attempts=1
    blocked_now, reason = mgr.register_attempt(ip, tenant="")
    assert blocked_now is True
    assert "Blocked until" in reason

    # is_blocked should return True
    blocked, until = mgr.is_blocked(ip)
    assert blocked is True
    assert until > time.time()

    # wait for block to expire
    time.sleep(2.1)
    blocked, _ = mgr.is_blocked(ip)
    assert blocked is False


def test_offender_resets_count_after_window(monkeypatch):
    monkeypatch.setenv("FLOUDS_BLOCK_MAX_ATTEMPS", "2")
    monkeypatch.setenv("FLOUDS_BLOCK_WINDOW_SECONDS", "1")
    monkeypatch.setenv("FLOUDS_BLOCK_SECONDS", "5")

    mgr = OffenderManager()
    ip = "10.0.0.1"

    # two quick attempts -> not blocked (threshold is >2)
    b1, _ = mgr.register_attempt(ip, tenant="")
    assert b1 is False
    b2, _ = mgr.register_attempt(ip, tenant="")
    assert b2 is False

    # sleep past the window so count resets
    time.sleep(1.1)

    # new attempt should be treated as first attempt again
    b3, _ = mgr.register_attempt(ip, tenant="")
    assert b3 is False
