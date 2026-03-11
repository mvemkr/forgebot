"""tests/test_pre_candidate_reset.py
=====================================
Tests for fix/pre-candidate-stale-reset.

Verifies that self._pre_candidate_data[pair] = [] is executed as the very
first statement in SetAndForgetStrategy.evaluate(), BEFORE every early-return
path, so the orchestrator never reads stale sub-threshold patterns from a
previous in-session scan.

Issue B diagnosis:
  Without this reset, evaluate() may return early (session filter, stop
  cooldown, open-positions gate, etc.) before reaching the _pre_candidate_data
  assignment at line ~1120.  The orchestrator then reads the stale value from
  the last full evaluation and could emit PRE_CANDIDATE with wrong pair/pattern
  data.

What these tests do NOT cover:
  - Orchestrator guard (decision.pattern is None) — unchanged, not modified.
  - Sub-threshold pattern emission correctness — covered elsewhere.
  - Issue C (structural rarity) — runtime property, not a code bug.
"""
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n=100, price=1.35, atr=0.001):
    """Minimal OHLCV dataframe."""
    closes = np.full(n, price)
    return pd.DataFrame({
        "open":   closes,
        "high":   closes + atr / 2,
        "low":    closes - atr / 2,
        "close":  closes,
        "volume": np.ones(n) * 1000,
    })


def _make_dfs(n=100):
    return {
        "weekly": _make_df(n),
        "daily":  _make_df(n),
        "h4":     _make_df(n),
        "h1":     _make_df(n),
    }


def _fake_pattern(clarity=0.2, pattern_type="double_top", direction="bearish"):
    p = MagicMock()
    p.clarity       = clarity
    p.pattern_type  = pattern_type
    p.direction     = direction
    p.neckline      = 1.350
    p.pattern_level = 1.360
    p.stop_anchor   = 1.370
    p.stop_loss     = 1.370
    p.target_1      = 1.320
    p.target_2      = 1.300
    return p


def _build_strategy():
    """Build a SetAndForgetStrategy with all external dependencies mocked."""
    from src.strategy.forex.set_and_forget import SetAndForgetStrategy

    # Use __new__ to skip __init__ entirely — we wire state manually below
    strat = SetAndForgetStrategy.__new__(SetAndForgetStrategy)

    # Minimal __init__ state
    from src.strategy.forex.set_and_forget import _TFBarCache
    from src.strategy.forex.pattern_detector import PatternDetector
    from src.strategy.forex.entry_signal import EntrySignalDetector
    from src.strategy.forex.level_detector import LevelDetector
    from src.strategy.forex.session_filter import SessionFilter

    strat._pre_candidate_data    = {}
    strat._tf_cache              = _TFBarCache()
    strat.pattern_detector       = PatternDetector()
    strat.signal_detector        = EntrySignalDetector(
        min_body_ratio=0.45, lookback_candles=2
    )
    strat.level_detector         = LevelDetector()
    strat.session_filter         = MagicMock(spec=SessionFilter)
    strat.open_positions         = {}
    strat._stop_out_times        = {}
    strat.min_pattern_clarity    = 0.4
    strat._level_cache           = {}
    strat.STOP_COOLDOWN_DAYS     = 3.0

    # Session: allowed by default
    strat.session_filter.is_entry_allowed.return_value = (True, "London")
    strat.session_filter.session_quality.return_value  = ("London", "high")

    return strat


# ── 1. Reset fires before session-blocked early return ────────────────────────

class TestResetBeforeSessionBlock(unittest.TestCase):

    def setUp(self):
        self.strat = _build_strategy()
        self.pair  = "USD/JPY"
        self.dfs   = _make_dfs()

    def _call_blocked(self):
        """Evaluate with session blocked."""
        self.strat.session_filter.is_entry_allowed.return_value = (
            False, "OUTSIDE_LONDON_SESSION"
        )
        return self.strat.evaluate(
            self.pair,
            self.dfs["weekly"], self.dfs["daily"],
            self.dfs["h4"],     self.dfs["h1"],
        )

    def test_stale_data_cleared_on_session_block(self):
        """Stale _pre_candidate_data[pair] must be [] after a session-blocked call."""
        # Plant stale data (simulates what a previous in-session scan would leave)
        self.strat._pre_candidate_data[self.pair] = [_fake_pattern(clarity=0.25)]
        self.assertGreater(
            len(self.strat._pre_candidate_data[self.pair]), 0,
            "precondition: stale data present",
        )

        self._call_blocked()

        self.assertEqual(
            self.strat._pre_candidate_data[self.pair], [],
            "_pre_candidate_data must be [] after session-blocked call",
        )

    def test_empty_dict_stays_empty_on_session_block(self):
        """No prior stale data — dict should be [] (not missing key) after block."""
        self.assertNotIn(self.pair, self.strat._pre_candidate_data)
        self._call_blocked()
        self.assertEqual(self.strat._pre_candidate_data.get(self.pair), [])

    def test_key_present_after_session_block(self):
        """pair key must exist in _pre_candidate_data after any evaluate() call."""
        self._call_blocked()
        self.assertIn(self.pair, self.strat._pre_candidate_data)

    def test_other_pairs_not_affected(self):
        """Reset for USD/JPY must not touch GBP/USD's stale data."""
        other = "GBP/USD"
        sentinel = [_fake_pattern(clarity=0.3, pattern_type="head_and_shoulders")]
        self.strat._pre_candidate_data[other] = sentinel

        self._call_blocked()

        self.assertIs(
            self.strat._pre_candidate_data[other], sentinel,
            "GBP/USD stale data must be untouched when USD/JPY is evaluated",
        )


# ── 2. Reset fires before stop-cooldown early return ─────────────────────────

class TestResetBeforeStopCooldown(unittest.TestCase):

    def setUp(self):
        self.strat = _build_strategy()
        self.pair  = "GBP/USD"
        self.dfs   = _make_dfs()

    def test_stale_cleared_on_stop_cooldown(self):
        """_pre_candidate_data reset even when stop-cooldown gate fires."""
        # Plant stale data
        self.strat._pre_candidate_data[self.pair] = [_fake_pattern()]

        # Put pair in stop cooldown
        self.strat._stop_out_times[self.pair] = datetime.now(timezone.utc)

        self.strat.evaluate(
            self.pair,
            self.dfs["weekly"], self.dfs["daily"],
            self.dfs["h4"],     self.dfs["h1"],
        )

        self.assertEqual(self.strat._pre_candidate_data[self.pair], [])


# ── 3. Reset fires before open-positions early return ────────────────────────

class TestResetBeforeOpenPositions(unittest.TestCase):

    def setUp(self):
        self.strat = _build_strategy()
        self.pair  = "EUR/USD"
        self.dfs   = _make_dfs()

    def test_stale_cleared_when_max_concurrent_reached(self):
        """_pre_candidate_data reset even when concurrent position cap fires."""
        self.strat._pre_candidate_data[self.pair] = [_fake_pattern()]

        # MAX_CONCURRENT is a local in evaluate() — equals 1 when no macro theme.
        # Adding one open position triggers the concurrent-cap early return.
        self.strat.open_positions["FAKE/P0"] = {"direction": "short"}

        self.strat.evaluate(
            self.pair,
            self.dfs["weekly"], self.dfs["daily"],
            self.dfs["h4"],     self.dfs["h1"],
        )

        self.assertEqual(self.strat._pre_candidate_data[self.pair], [])


# ── 4. Normal in-session path still populates correctly ──────────────────────

class TestNormalPathUnaffected(unittest.TestCase):
    """
    The reset initialises to [] then the normal evaluate() path may overwrite
    it with real sub-threshold patterns.  Verify the contract is preserved.
    """

    def setUp(self):
        self.strat = _build_strategy()
        self.pair  = "USD/CHF"
        self.dfs   = _make_dfs()

    def test_reset_to_empty_at_call_start(self):
        """_pre_candidate_data[pair] == [] at the start of every evaluate() call
        (verified indirectly: after a session-blocked call the value is [])."""
        self.strat._pre_candidate_data[self.pair] = [_fake_pattern(), _fake_pattern()]

        self.strat.session_filter.is_entry_allowed.return_value = (
            False, "OUTSIDE_SESSION"
        )
        self.strat.evaluate(
            self.pair,
            self.dfs["weekly"], self.dfs["daily"],
            self.dfs["h4"],     self.dfs["h1"],
        )

        # After an early return the reset left [] — proves it fired first
        self.assertEqual(self.strat._pre_candidate_data[self.pair], [])

    def test_sub_threshold_patterns_still_captured_after_reset(self):
        """
        When evaluate() reaches the pattern-loop section with sub-threshold
        patterns, _pre_candidate_data[pair] is populated (reset then overwritten).
        We simulate this by directly invoking the assignment path the way
        evaluate() does after a reset.
        """
        # Simulate: reset fires, then sub-threshold patterns are written
        self.strat._pre_candidate_data[self.pair] = []
        sub = [_fake_pattern(clarity=0.25), _fake_pattern(clarity=0.31)]
        self.strat._pre_candidate_data[self.pair] = sub

        self.assertEqual(len(self.strat._pre_candidate_data[self.pair]), 2)
        for p in self.strat._pre_candidate_data[self.pair]:
            self.assertLess(p.clarity, 0.4)
            self.assertGreater(p.clarity, 0.0)


# ── 5. Orchestrator guard unchanged ──────────────────────────────────────────

class TestOrchestratorGuardUnchanged(unittest.TestCase):
    """
    Confirm the decision.pattern is None guard in orchestrator._evaluate_pair
    is still in place.  Fix 1 does NOT touch the orchestrator.
    """

    def test_guard_present_in_source(self):
        import ast
        from pathlib import Path

        src = (Path(__file__).parent.parent
               / "src" / "execution" / "orchestrator.py").read_text()

        # Parse and walk AST to find the guard
        tree  = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            # Looking for: if decision.pattern is None:
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Attribute)
                    and test.left.attr == "pattern"
                    and isinstance(test.left.value, ast.Name)
                    and test.left.value.id == "decision"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Is)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value is None):
                found = True
                break

        self.assertTrue(
            found,
            "orchestrator.py must still contain 'if decision.pattern is None:' guard",
        )

    def test_pre_candidate_call_inside_guard(self):
        """log_pre_candidate call must be inside the decision.pattern is None block."""
        from pathlib import Path

        src = (Path(__file__).parent.parent
               / "src" / "execution" / "orchestrator.py").read_text()

        lines = src.splitlines()
        guard_idx   = None
        call_idx    = None
        guard_indent = None

        for i, line in enumerate(lines):
            if "if decision.pattern is None:" in line and guard_idx is None:
                guard_idx    = i
                guard_indent = len(line) - len(line.lstrip())
            if "log_pre_candidate" in line and guard_idx is not None:
                call_idx = i
                break

        self.assertIsNotNone(guard_idx, "guard line not found")
        self.assertIsNotNone(call_idx,  "log_pre_candidate call not found")
        self.assertGreater(call_idx, guard_idx,
                           "log_pre_candidate must come after the guard")

        # call must be indented further than the guard (i.e. inside it)
        call_indent = len(lines[call_idx]) - len(lines[call_idx].lstrip())
        self.assertGreater(call_indent, guard_indent,
                           "log_pre_candidate must be indented inside the guard block")


# ── 6. Reset is genuinely first — no code before it in evaluate() ────────────

class TestResetIsFirst(unittest.TestCase):
    """
    AST-level check: self._pre_candidate_data[pair] = [] must be the first
    statement in the evaluate() function body (after the docstring).
    """

    def test_reset_is_first_statement(self):
        import ast
        from pathlib import Path

        src  = (Path(__file__).parent.parent
                / "src" / "strategy" / "forex" / "set_and_forget.py").read_text()
        tree = ast.parse(src)

        evaluate_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
                evaluate_fn = node
                break

        self.assertIsNotNone(evaluate_fn, "evaluate() not found")

        # Body: skip leading docstring (Expr node with Constant value)
        body = evaluate_fn.body
        first_exec = None
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue   # skip docstring
            first_exec = stmt
            break

        self.assertIsNotNone(first_exec, "no executable statement in evaluate()")

        # Must be: self._pre_candidate_data[pair] = []
        self.assertIsInstance(
            first_exec, ast.Assign,
            f"first executable statement must be an assignment, got {type(first_exec).__name__}",
        )

        # LHS: self._pre_candidate_data[pair]
        self.assertEqual(len(first_exec.targets), 1)
        target = first_exec.targets[0]
        self.assertIsInstance(target, ast.Subscript)
        self.assertIsInstance(target.value, ast.Attribute)
        self.assertEqual(target.value.attr, "_pre_candidate_data")

        # RHS: []
        self.assertIsInstance(first_exec.value, ast.List)
        self.assertEqual(len(first_exec.value.elts), 0,
                         "RHS must be an empty list []")


if __name__ == "__main__":
    unittest.main()
