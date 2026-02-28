"""
Entry Signal Detector — The Engulfing Candle

THE most important module in the system. This is the patience filter.

From the strategy research:
  "Price reaching a key level is NOT an entry.
   The entry is: price at key level + bearish/bullish engulfing candle."

Rules:
  - Bearish engulfing: current body CLOSES below previous candle's full body
  - Bullish engulfing: current body CLOSES above previous candle's full body
  - Body must be meaningful (not a doji — body > wick ratio threshold)
  - No entry without the candle. No exceptions. No FOMO.

Additional confirmation:
  - Marubozu (large body, small wicks) = stronger signal
  - Tweezer top/bottom (two candles with near-equal highs/lows) = valid
  - Pin bar with strong body in direction = valid
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from . import strategy_config as _cfg   # module-ref so apply_levers() patches propagate


@dataclass
class EntrySignal:
    signal_type: str       # 'bearish_engulfing', 'bullish_engulfing', 'pin_bar_bear', etc.
    direction: str         # 'long' or 'short'
    strength: float        # 0.0–1.0 (1.0 = perfect textbook candle)
    candle_index: int      # which candle fired (usually -1 = latest closed)
    close: float           # close price of signal candle
    body_size: float       # absolute body size
    notes: str = ""

    @property
    def is_strong(self) -> bool:
        return self.strength >= 0.65

    @property
    def is_valid(self) -> bool:
        return self.strength >= 0.40


def is_indecision(candle: pd.Series) -> bool:
    """
    Returns True when the candle is a doji / spinning top / long-legged doji.

    Definition (objective, deterministic):
      body / range  ≤  INDECISION_MAX_BODY_RATIO   (tiny body)
      AND upper_wick / range  ≥  INDECISION_MIN_WICK_RATIO
      AND lower_wick / range  ≥  INDECISION_MIN_WICK_RATIO

    This catches: classic doji, long-legged doji, spinning top.
    It does NOT catch: hammer (one-sided wick), shooting star (one-sided wick).

    Exception rule: when this candle is the middle bar of a Morning/Evening Star,
    it is allowed to be indecision — that's the whole point of the pattern.
    Call sites must handle that exception explicitly.
    """
    rng = float(candle["high"] - candle["low"])
    if rng <= 1e-10:
        return True   # flat/zero-range bar → treat as indecision

    body       = abs(float(candle["close"]) - float(candle["open"]))
    body_high  = max(float(candle["open"]), float(candle["close"]))
    body_low   = min(float(candle["open"]), float(candle["close"]))
    upper_wick = float(candle["high"]) - body_high
    lower_wick = body_low - float(candle["low"])

    body_ratio  = body / rng
    upper_ratio = upper_wick / rng
    lower_ratio = lower_wick / rng

    if body_ratio > _cfg.INDECISION_MAX_BODY_RATIO:
        return False
    return (upper_ratio >= _cfg.INDECISION_MIN_WICK_RATIO
            and lower_ratio >= _cfg.INDECISION_MIN_WICK_RATIO)


class EntrySignalDetector:
    """
    Detects engulfing candles and other entry confirmation signals.

    Parameters
    ----------
    min_body_ratio : float
        Minimum ratio of body to total candle range (filter dojis).
        Default 0.5 = body must be ≥50% of high–low range.
    lookback_candles : int
        How many recent candles to scan for signals (default 2 = last 2 closed candles).
    """

    def __init__(
        self,
        min_body_ratio: float = 0.50,
        lookback_candles: int = 2,
    ):
        self.min_body_ratio = min_body_ratio
        self.lookback = lookback_candles

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(self, df: pd.DataFrame) -> Optional[EntrySignal]:
        """
        Scan recent candles for entry signals.
        Returns the strongest signal found, or None.

        df: OHLC dataframe, latest candle last.
        NOTE: The latest candle (index -1) should be CLOSED, not currently forming.
        """
        if len(df) < 3:
            return None

        signals = []

        for i in range(-self.lookback, 0):
            try:
                current = df.iloc[i]
                prev    = df.iloc[i - 1]
            except IndexError:
                continue

            # Bearish engulfing
            be = self._bearish_engulfing(current, prev)
            if be:
                signals.append(EntrySignal(
                    signal_type='bearish_engulfing',
                    direction='short',
                    strength=be,
                    candle_index=i,
                    close=current['close'],
                    body_size=abs(current['open'] - current['close']),
                    notes=f"Bearish engulfing at {current['close']:.5f}",
                ))

            # Bullish engulfing
            bue = self._bullish_engulfing(current, prev)
            if bue:
                signals.append(EntrySignal(
                    signal_type='bullish_engulfing',
                    direction='long',
                    strength=bue,
                    candle_index=i,
                    close=current['close'],
                    body_size=abs(current['open'] - current['close']),
                    notes=f"Bullish engulfing at {current['close']:.5f}",
                ))

            # Pin bar — active only when ENTRY_TRIGGER_MODE == "engulf_or_pin".
            # Tight spec in _pin_bar() prevents noise: wick ≥ 2× body, close
            # in outer 30% of range in trade direction.
            if _cfg.ENTRY_TRIGGER_MODE == "engulf_or_pin":
                pb = self._pin_bar(current, direction="")
                if pb:
                    signals.append(pb)

        if not signals:
            return None

        # Return strongest signal
        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals[0]

    def has_signal(
        self,
        df: pd.DataFrame,
        direction: str,
        level: float = 0.0,
    ) -> tuple[bool, Optional[EntrySignal], str]:
        """
        Check if there's a valid entry signal in the given direction.
        Returns (found, signal_or_None, reason_code).

        level : key price level (neckline / round number) used by at-level
                confirmation modes.  Pass 0.0 (default) to skip level gate.

        reason_code when False:
          NO_ENGULF            — engulfing not found (engulf_only)
          NO_ENGULF|NO_PIN     — neither (engulf_or_pin)
          NO_ENGULF|NO_STAR    — neither (engulf_or_star_at_level)
          NO_ENGULF|NO_STRICTPIN — neither (engulf_or_strict_pin_at_level)
          NO_TRIGGER           — signal found but wrong direction/strength
        """
        if len(df) < 3:
            return False, None, "NO_TRIGGER"

        mode = _cfg.ENTRY_TRIGGER_MODE
        tried_pin        = (mode == "engulf_or_pin")
        tried_at_level   = mode in (
            "engulf_or_star_at_level",
            "engulf_or_strict_pin_at_level",
            "engulf_or_star_or_strict_pin_at_level",
        )
        found_engulf = False
        found_alt    = False          # pin | star | strict_pin
        best: Optional[EntrySignal] = None

        for i in range(-self.lookback, 0):
            try:
                current = df.iloc[i]
                prev    = df.iloc[i - 1]
            except IndexError:
                continue

            # ── Engulfing ──────────────────────────────────────────────
            if direction == "short":
                s = self._bearish_engulfing(current, prev)
                if s:
                    found_engulf = True
                    sig = EntrySignal('bearish_engulfing', 'short', s, i,
                                      current['close'],
                                      abs(current['open'] - current['close']),
                                      f"Bearish engulfing at {current['close']:.5f}")
                    if best is None or sig.strength > best.strength:
                        best = sig
            else:
                s = self._bullish_engulfing(current, prev)
                if s:
                    found_engulf = True
                    sig = EntrySignal('bullish_engulfing', 'long', s, i,
                                      current['close'],
                                      abs(current['open'] - current['close']),
                                      f"Bullish engulfing at {current['close']:.5f}")
                    if best is None or sig.strength > best.strength:
                        best = sig

            # ── Loose pin bar (engulf_or_pin mode) ────────────────────
            if tried_pin:
                pb = self._pin_bar(current, direction)
                if pb:
                    found_alt = True
                    if best is None or pb.strength > best.strength:
                        best = pb

        # ── At-level confirmations (star / strict pin) ─────────────────
        # These look at the last 2-3 bars as a unit, so called once outside loop.
        if tried_at_level and level > 0:
            from .candle_confirmations import detect_at_level_confirmation
            al_sig = detect_at_level_confirmation(df, direction, level, mode)
            if al_sig:
                found_alt = True
                if best is None or al_sig.strength > best.strength:
                    best = al_sig

        if best and best.direction == direction and best.is_valid:
            # ── Indecision / doji filter ─────────────────────────────
            # Block if the trigger candle itself is a doji / spinning top.
            # Exception: morning_star / evening_star check bar3 internally —
            # those signals already guarantee bar3 is decisive.
            if _cfg.INDECISION_FILTER_ENABLED:
                _is_star_signal = best.signal_type in ("morning_star", "evening_star")
                if not _is_star_signal:
                    try:
                        trigger_bar = df.iloc[best.candle_index]
                        if is_indecision(trigger_bar):
                            return False, None, "INDECISION_DOJI"
                    except (IndexError, KeyError):
                        pass   # safety: if lookup fails, allow entry

            return True, best, ""

        # ── Diagnostic reason ──────────────────────────────────────────
        if tried_pin:
            reason = "NO_ENGULF|NO_PIN"   if (not found_engulf and not found_alt)  else "NO_TRIGGER"
        elif mode == "engulf_or_star_at_level":
            reason = "NO_ENGULF|NO_STAR"  if (not found_engulf and not found_alt)  else "NO_TRIGGER"
        elif mode == "engulf_or_strict_pin_at_level":
            reason = "NO_ENGULF|NO_STRICTPIN" if (not found_engulf and not found_alt) else "NO_TRIGGER"
        elif mode == "engulf_or_star_or_strict_pin_at_level":
            reason = "NO_ENGULF|NO_STAR|NO_STRICTPIN" if (not found_engulf and not found_alt) else "NO_TRIGGER"
        else:
            reason = "NO_ENGULF" if not found_engulf else "NO_TRIGGER"
        return False, None, reason

    # ------------------------------------------------------------------ #
    # Individual signal detectors
    # ------------------------------------------------------------------ #

    def _bearish_engulfing(self, current: pd.Series, prev: pd.Series) -> float:
        """
        Returns strength (0.0 = not engulfing, >0 = engulfing).

        Classic bearish engulfing:
        - Current candle is bearish (close < open)
        - Current body fully engulfs previous body
        - Previous candle was bullish (close > open) — ideal
        """
        curr_open  = current['open']
        curr_close = current['close']
        curr_high  = current['high']
        curr_low   = current['low']

        prev_open  = prev['open']
        prev_close = prev['close']

        # Must be bearish
        if curr_close >= curr_open:
            return 0.0

        curr_body = curr_open - curr_close
        curr_range = curr_high - curr_low

        # Filter doji
        if curr_range > 0 and curr_body / curr_range < self.min_body_ratio:
            return 0.0

        # Body must engulf previous candle's body
        prev_body_high = max(prev_open, prev_close)
        prev_body_low  = min(prev_open, prev_close)

        if curr_open < prev_body_high or curr_close > prev_body_low:
            return 0.0

        # Strength: how much does it over-engulf?
        engulf_ratio = curr_body / max(abs(prev_open - prev_close), 0.00001)
        strength = min(1.0, engulf_ratio / 2)

        # Bonus: previous candle was bullish (cleaner reversal signal)
        if prev_close > prev_open:
            strength = min(1.0, strength + 0.15)

        # Bonus: small wicks (marubozu-like)
        upper_wick = curr_high - curr_open
        wick_ratio = upper_wick / curr_range if curr_range > 0 else 1
        if wick_ratio < 0.15:
            strength = min(1.0, strength + 0.10)

        return strength

    def _bullish_engulfing(self, current: pd.Series, prev: pd.Series) -> float:
        """Bullish mirror of _bearish_engulfing."""
        curr_open  = current['open']
        curr_close = current['close']
        curr_high  = current['high']
        curr_low   = current['low']

        prev_open  = prev['open']
        prev_close = prev['close']

        # Must be bullish
        if curr_close <= curr_open:
            return 0.0

        curr_body = curr_close - curr_open
        curr_range = curr_high - curr_low

        if curr_range > 0 and curr_body / curr_range < self.min_body_ratio:
            return 0.0

        prev_body_high = max(prev_open, prev_close)
        prev_body_low  = min(prev_open, prev_close)

        if curr_close < prev_body_high or curr_open > prev_body_low:
            return 0.0

        engulf_ratio = curr_body / max(abs(prev_open - prev_close), 0.00001)
        strength = min(1.0, engulf_ratio / 2)

        if prev_close < prev_open:
            strength = min(1.0, strength + 0.15)

        lower_wick = curr_open - curr_low
        wick_ratio = lower_wick / curr_range if curr_range > 0 else 1
        if wick_ratio < 0.15:
            strength = min(1.0, strength + 0.10)

        return strength

    def _pin_bar(self, current: pd.Series, direction: str = "") -> Optional[EntrySignal]:
        """
        Pin bar / rejection wick — tight spec (all three rules must pass):
          1. Rejection wick ≥ PIN_BAR_MIN_WICK_BODY_RATIO × body (default 2×)
          2. Close in outer PIN_BAR_CLOSE_OUTER_PCT of candle range in
             the trade direction (bearish = lower 30%, bullish = upper 30%)
          3. Directional: upper wick only for shorts, lower wick only for longs

        Why tight: prevents noise wicks (NFP spikes, overnight gaps) from
        triggering entries. Alex's pin bars are textbook — wick dominant,
        body clearly at the far end, clean close-back.
        """
        from .strategy_config import PIN_BAR_MIN_WICK_BODY_RATIO, PIN_BAR_CLOSE_OUTER_PCT

        o = current['open']
        h = current['high']
        l = current['low']
        c = current['close']

        candle_range = h - l
        if candle_range < 1e-8:
            return None

        body       = abs(c - o)
        body_high  = max(o, c)
        body_low   = min(o, c)
        upper_wick = h - body_high
        lower_wick = body_low - l

        # Minimum body floor: 0.5% of candle range to avoid divide-by-near-zero
        # (doji candles with 0-pip bodies pass wick≥2×body trivially — exclude them)
        body_floor = max(body, candle_range * 0.005)

        # ── Bearish pin (SHORT): dominant upper wick ────────────────────
        # Upper wick drove into resistance then price closed back down.
        # Close must be in the LOWER 30% of the candle range.
        if direction in ("", "short"):
            if (upper_wick >= PIN_BAR_MIN_WICK_BODY_RATIO * body_floor
                    and upper_wick > lower_wick              # upper dominates
                    and c <= l + PIN_BAR_CLOSE_OUTER_PCT * candle_range):
                strength = min(1.0, (upper_wick / candle_range) - 0.05) * 0.85
                return EntrySignal(
                    signal_type='pin_bar_bearish',
                    direction='short',
                    strength=strength,
                    candle_index=-1,
                    close=c,
                    body_size=body,
                    notes=(f"Bearish pin — wick {upper_wick/candle_range:.0%} of range, "
                           f"wick/body={upper_wick/body_floor:.1f}×, close lower {(c-l)/candle_range:.0%}"),
                )

        # ── Bullish pin (LONG): dominant lower wick ─────────────────────
        # Lower wick drove into support then price closed back up.
        # Close must be in the UPPER 30% of the candle range.
        if direction in ("", "long"):
            if (lower_wick >= PIN_BAR_MIN_WICK_BODY_RATIO * body_floor
                    and lower_wick > upper_wick              # lower dominates
                    and c >= h - PIN_BAR_CLOSE_OUTER_PCT * candle_range):
                strength = min(1.0, (lower_wick / candle_range) - 0.05) * 0.85
                return EntrySignal(
                    signal_type='pin_bar_bullish',
                    direction='long',
                    strength=strength,
                    candle_index=-1,
                    close=c,
                    body_size=body,
                    notes=(f"Bullish pin — wick {lower_wick/candle_range:.0%} of range, "
                           f"wick/body={lower_wick/body_floor:.1f}×, close upper {(h-c)/candle_range:.0%}"),
                )

        return None


if __name__ == "__main__":
    # Quick test
    import pandas as pd

    # Bearish engulfing example
    data = {
        'open':  [1.3000, 1.3010, 1.3020, 1.3040, 1.3060],
        'high':  [1.3020, 1.3030, 1.3050, 1.3070, 1.3070],
        'low':   [1.2990, 1.3000, 1.3010, 1.3030, 1.2990],
        'close': [1.3010, 1.3020, 1.3040, 1.3060, 1.3000],  # last candle: big down candle
    }
    df = pd.DataFrame(data)
    detector = EntrySignalDetector()
    signal = detector.detect(df)
    print(f"Signal: {signal}")
