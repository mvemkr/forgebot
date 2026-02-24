#!/usr/bin/env python3
"""
Forge vs Alex — Visual comparison chart
Runs chart_forge_vs_alex.py first to build the CSV, then renders the image.
Output: logs/forge_vs_alex_chart.png
"""
import json, subprocess, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

LOGS = Path(__file__).parent.parent / "logs"

# ── Step 1: regenerate CSV ────────────────────────────────────────────────────
subprocess.run([sys.executable, str(Path(__file__).parent / "chart_forge_vs_alex.py")],
               check=True, capture_output=True)

# ── Step 2: load latest backtest ─────────────────────────────────────────────
with open(LOGS / "backtest_results.jsonl") as f:
    last = json.loads(f.readlines()[-1])

results = last["results"]
trades  = last["trades"]

def pipcount(entry, exit_p, direction, pair):
    pip = 0.01 if "JPY" in pair else 0.0001
    d = (exit_p - entry) if direction == "long" else (entry - exit_p)
    return round(d / pip, 1)

# ── Alex's weekly reference data ──────────────────────────────────────────────
ALEX_SERIES = [
    ("Wk1",  "GBP/JPY","short", 136.9, "2024-07-15", "win"),
    ("Wk2",  "USD/JPY","short", 171.0, "2024-07-16", "win"),
    ("Wk3",  "USD/CHF","short", 136.0, "2024-08-01", "win"),
    ("Wk4",  "EUR/USD","short", None,  None,          "loss"),
    ("Wk5",  None,     None,    None,  None,          "skip"),
    ("Wk6",  "GBP/CHF","short", 153.0, "2024-07-25", "win"),
    ("Wk7",  "GBP/CHF","short", 153.0, "2024-07-25", "win"),
    ("Wk8",  "USD/JPY","short", 171.0, "2024-07-16", "win"),
    ("Wk9",  None,     None,    None,  None,          "skip"),
    ("Wk10", "USD/CAD","short", 80.0,  "2024-08-06", "win"),
    ("Wk11", "USD/CAD","short", None,  "2024-08-06", "loss"),
    ("Wk12a",None,     None,    None,  None,          "skip"),
    ("Wk12b","GBP/CHF","short", 153.0, "2024-10-16", "win"),
    ("Wk13", "USD/CHF","long",  142.0, None,          "win"),
]

ALEX_DATES = {
    ("GBP/JPY","short","2024-07-15"),
    ("USD/JPY","short","2024-07-16"),
    ("USD/CHF","short","2024-08-01"),
    ("GBP/CHF","short","2024-07-25"),
    ("USD/CAD","short","2024-08-06"),
    ("EUR/USD","long", "2024-07-17"),
    ("EUR/USD","short","2024-08-20"),
    ("GBP/USD","short","2024-08-06"),
    ("GBP/USD","short","2024-09-11"),
    ("GBP/USD","short","2024-10-22"),
    ("GBP/CHF","short","2024-10-16"),
    ("USD/CHF","short","2024-07-31"),
}

bot_by_key = {
    (t["pair"], t["direction"], t["entry_ts"][:10]): t
    for t in trades
}

# Build Alex comparison rows
alex_rows = []
for wk, pair, direction, alex_pips, bot_date, outcome in ALEX_SERIES:
    bot_t    = bot_by_key.get((pair, direction, bot_date)) if bot_date else None
    bot_pips = pipcount(bot_t["entry"], bot_t["exit"],
                        bot_t["direction"], bot_t["pair"]) if bot_t else None
    if bot_t:
        if   bot_t["reason"] == "open_at_end": status = "open"
        elif bot_t["pnl"] == 0:                status = "BE"
        elif bot_t["pnl"] > 0:                 status = "win"
        else:                                   status = "stop"
    else:
        status = "MISSED" if wk == "Wk13" else "SKIP"
    alex_rows.append({
        "wk": wk, "pair": pair or "—", "dir": direction,
        "alex_pips": alex_pips, "bot_pips": bot_pips,
        "outcome": outcome, "status": status, "bot_pnl": bot_t["pnl"] if bot_t else 0,
    })

# Build bot-only rows
bot_only = []
for t in sorted(trades, key=lambda x: x["entry_ts"]):
    key = (t["pair"], t["direction"], t["entry_ts"][:10])
    if key in ALEX_DATES:
        continue
    pips = pipcount(t["entry"], t["exit"], t["direction"], t["pair"])
    if   t["pnl"] > 0:  result = "WIN"
    elif t["pnl"] < 0:  result = "LOSS"
    else:                result = "BE"
    bot_only.append({
        "date":  t["entry_ts"][:10],
        "pair":  t["pair"],
        "dir":   t["direction"],
        "pips":  pips,
        "pnl":   t["pnl"],
        "result":result,
        "pattern": t.get("pattern","?")[:26],
    })

# ── Colours ───────────────────────────────────────────────────────────────────
C_ALEX_WIN  = "#2ecc71"
C_ALEX_LOSS = "#e74c3c"
C_ALEX_SKIP = "#95a5a6"
C_BOT_WIN   = "#27ae60"
C_BOT_LOSS  = "#c0392b"
C_BOT_BE    = "#7f8c8d"
C_BOT_OPEN  = "#3498db"
C_BG        = "#0f1117"
C_PANEL     = "#1a1d27"
C_TEXT      = "#e8eaf0"
C_SUBTEXT   = "#8b9bb4"
C_GRID      = "#2a2d3a"
C_HEADER    = "#252836"
C_ACCENT    = "#f39c12"

# ── Layout ────────────────────────────────────────────────────────────────────
n_alex   = len(alex_rows)
n_bot    = len(bot_only)
fig_h    = 3.0 + n_alex * 0.52 + 1.2 + n_bot * 0.46

plt.rcParams["text.usetex"]     = False
plt.rcParams["text.parse_math"] = False   # treat $ as literal, not math delimiter

fig = plt.figure(figsize=(16, fig_h), facecolor=C_BG)
gs  = gridspec.GridSpec(3, 1, figure=fig,
                        height_ratios=[n_alex * 0.52 + 1.0, 0.6, n_bot * 0.46 + 0.8],
                        hspace=0.04)

ax_alex  = fig.add_subplot(gs[0])
ax_div   = fig.add_subplot(gs[1])
ax_bot   = fig.add_subplot(gs[2])

for ax in (ax_alex, ax_div, ax_bot):
    ax.set_facecolor(C_PANEL)
    ax.set_xlim(0, 1)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Alex vs Forge comparison
# ═══════════════════════════════════════════════════════════════════════════════
ROW_H   = 1.0 / (n_alex + 1.5)
COL_WK  = 0.04
COL_PAIR= 0.12
COL_DIR = 0.22
COL_ALEX= 0.32   # alex pips bar starts
BAR_MID = 0.50   # centre spine
COL_BOT = 0.68   # bot pips bar starts
COL_STA = 0.84
COL_PNL = 0.94
MAX_PIPS= 850

def pip_bar_w(pips, max_pips=MAX_PIPS, max_w=0.16):
    if pips is None: return 0
    return min(abs(pips) / max_pips, 1.0) * max_w

# Header row
hy = 1.0 - ROW_H * 0.55
ax_alex.text(0.01,  hy, "WEEK",   color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_alex.transAxes)
ax_alex.text(COL_PAIR, hy, "PAIR",  color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_alex.transAxes)
ax_alex.text(COL_DIR,  hy, "DIR",   color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_alex.transAxes)
ax_alex.text(COL_ALEX, hy, "◀  ALEX PIPS",  color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_alex.transAxes)
ax_alex.text(COL_BOT,  hy, "BOT PIPS  ▶",  color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_alex.transAxes)
ax_alex.text(COL_STA,  hy, "STATUS", color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_alex.transAxes)
ax_alex.text(COL_PNL,  hy, "BOT P&L", color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", ha="right", transform=ax_alex.transAxes)

# Centre spine line
ax_alex.axvline(BAR_MID, color=C_GRID, linewidth=0.8, alpha=0.6)

# Title
ax_alex.text(0.5, 1.0 - ROW_H * 0.15,
             ">>> FORGE vs ALEX  |  Jul 15 – Oct 31 2024",
             color=C_TEXT, fontsize=13, fontweight="bold",
             ha="center", va="center", transform=ax_alex.transAxes)

for i, row in enumerate(alex_rows):
    y = 1.0 - ROW_H * (i + 1.5)

    # Row background alternating
    bg_c = "#1e2130" if i % 2 == 0 else C_PANEL
    ax_alex.add_patch(mpatches.FancyBboxPatch(
        (0, y - ROW_H * 0.48), 1, ROW_H * 0.96,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=bg_c, transform=ax_alex.transAxes, clip_on=False))

    ym = y  # text mid

    # Wk label
    wk_c = C_TEXT if row["outcome"] != "skip" else C_SUBTEXT
    ax_alex.text(0.01, ym, row["wk"], color=wk_c, fontsize=8.5,
                 fontweight="bold", va="center", transform=ax_alex.transAxes)

    if row["outcome"] == "skip":
        ax_alex.text(0.5, ym, "— no trade —", color=C_SUBTEXT, fontsize=8,
                     ha="center", va="center", transform=ax_alex.transAxes)
        continue

    # Pair + dir
    pair_c = C_TEXT if row["outcome"] == "win" else (C_ALEX_LOSS if row["outcome"] == "loss" else C_SUBTEXT)
    ax_alex.text(COL_PAIR, ym, row["pair"], color=pair_c, fontsize=8.5, va="center", transform=ax_alex.transAxes)
    dir_txt = "▼ SHORT" if row["dir"] == "short" else "▲ LONG"
    dir_c   = "#e74c3c" if row["dir"] == "short" else "#2ecc71"
    ax_alex.text(COL_DIR, ym, dir_txt, color=dir_c, fontsize=7.5, va="center", transform=ax_alex.transAxes)

    # ALEX pip bar (grows LEFT from centre)
    if row["alex_pips"] and row["outcome"] == "win":
        bw = pip_bar_w(row["alex_pips"])
        ax_alex.add_patch(mpatches.FancyBboxPatch(
            (BAR_MID - bw, ym - ROW_H * 0.30), bw, ROW_H * 0.60,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=C_ALEX_WIN, alpha=0.85, transform=ax_alex.transAxes))
        ax_alex.text(BAR_MID - bw - 0.005, ym,
                     f"{row['alex_pips']:.0f}p", color=C_ALEX_WIN,
                     fontsize=7.5, ha="right", va="center", transform=ax_alex.transAxes)
    elif row["outcome"] == "loss":
        ax_alex.text(BAR_MID - 0.02, ym, "LOSS", color=C_ALEX_LOSS,
                     fontsize=7.5, ha="right", va="center", transform=ax_alex.transAxes)

    # BOT pip bar (grows RIGHT from centre)
    bot_pips = row["bot_pips"]
    status   = row["status"]
    if status == "MISSED":
        ax_alex.text(COL_BOT + 0.01, ym, "[ MISSED ]", color=C_ALEX_LOSS,
                     fontsize=8, va="center", transform=ax_alex.transAxes)
    elif status in ("open","win","BE","stop") and bot_pips is not None:
        bw   = pip_bar_w(bot_pips)
        sign = bot_pips >= 0
        bc   = C_BOT_WIN if sign else C_BOT_LOSS
        if status == "BE": bc = C_BOT_BE
        if status == "open": bc = C_BOT_OPEN
        ax_alex.add_patch(mpatches.FancyBboxPatch(
            (BAR_MID, ym - ROW_H * 0.30), bw, ROW_H * 0.60,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=bc, alpha=0.85, transform=ax_alex.transAxes))
        pip_lbl = f"{bot_pips:+.0f}p"
        ax_alex.text(BAR_MID + bw + 0.005, ym, pip_lbl, color=bc,
                     fontsize=7.5, ha="left", va="center", transform=ax_alex.transAxes)

    # Status badge
    STATUS_C = {
        "open":"#3498db","win":"#2ecc71","BE":"#7f8c8d",
        "stop":"#e74c3c","SKIP":"#555","MISSED":"#e74c3c"
    }
    sc = STATUS_C.get(status, C_TEXT)
    ax_alex.text(COL_STA, ym, status.upper(), color=sc,
                 fontsize=7.5, fontweight="bold", va="center", transform=ax_alex.transAxes)

    # Bot P&L
    pnl = row["bot_pnl"]
    if pnl is not None and pnl != 0:
        pnl_c = C_BOT_WIN if pnl > 0 else C_BOT_LOSS
        ax_alex.text(COL_PNL, ym, f"${pnl:+,.0f}", color=pnl_c,
                     fontsize=8, fontweight="bold", ha="right",
                     va="center", transform=ax_alex.transAxes)

# Summary bar
sy = ROW_H * 0.35
ax_alex.add_patch(mpatches.FancyBboxPatch(
    (0, 0), 1, sy,
    boxstyle="square,pad=0", linewidth=0,
    facecolor=C_HEADER, transform=ax_alex.transAxes))
caught = sum(1 for r in alex_rows if r["status"] in ("open","win","BE","stop") and r["outcome"] == "win")
total_alex_wins = sum(1 for r in alex_rows if r["outcome"] == "win")
net_pnl = results["net_pnl"]
ret_pct = results["return_pct"]
ax_alex.text(0.01, sy/2,
             f"Alex trades caught: {caught}/{total_alex_wins}   |   "
             f"Window P&L: ${net_pnl:+,.2f}  ({ret_pct:+.1f}%)   |   "
             f"8K → ${8000+net_pnl:,.0f}",
             color=C_ACCENT, fontsize=9, fontweight="bold",
             va="center", transform=ax_alex.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
#  DIVIDER
# ═══════════════════════════════════════════════════════════════════════════════
ax_div.set_facecolor(C_HEADER)
ax_div.text(0.5, 0.5, "BOT-ONLY TRADES  (not in Alex's series)",
            color=C_ACCENT, fontsize=10, fontweight="bold",
            ha="center", va="center", transform=ax_div.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Bot-only trades
# ═══════════════════════════════════════════════════════════════════════════════
BCOL  = {"WIN": C_BOT_WIN, "LOSS": C_BOT_LOSS, "BE": C_BOT_BE}
ROW_B = 1.0 / (n_bot + 1.5)
BAR_MAX_W = 0.20
MAX_ABS_PIPS = max((abs(r["pips"]) for r in bot_only), default=1)

# Header
bhy = 1.0 - ROW_B * 0.6
ax_bot.text(0.01,  bhy, "DATE",    color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_bot.transAxes)
ax_bot.text(0.12,  bhy, "PAIR",    color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_bot.transAxes)
ax_bot.text(0.22,  bhy, "DIR",     color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_bot.transAxes)
ax_bot.text(0.32,  bhy, "PATTERN", color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_bot.transAxes)
ax_bot.text(0.62,  bhy, "PIPS",    color=C_SUBTEXT, fontsize=8, fontweight="bold", va="center", transform=ax_bot.transAxes)
ax_bot.text(0.99,  bhy, "P&L",     color=C_SUBTEXT, fontsize=8, fontweight="bold", ha="right",  va="center", transform=ax_bot.transAxes)

for i, row in enumerate(bot_only):
    y  = 1.0 - ROW_B * (i + 1.5)
    bg = "#1e2130" if i % 2 == 0 else C_PANEL
    ax_bot.add_patch(mpatches.FancyBboxPatch(
        (0, y - ROW_B * 0.48), 1, ROW_B * 0.96,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=bg, transform=ax_bot.transAxes, clip_on=False))

    rc = BCOL[row["result"]]
    ax_bot.text(0.01, y, row["date"][5:],  color=C_SUBTEXT, fontsize=7.5, va="center", transform=ax_bot.transAxes)
    ax_bot.text(0.12, y, row["pair"],       color=C_TEXT,    fontsize=8,   va="center", transform=ax_bot.transAxes)
    ddir = "▼ SHORT" if row["dir"] == "short" else "▲ LONG"
    dc   = "#e74c3c" if row["dir"] == "short" else "#2ecc71"
    ax_bot.text(0.22, y, ddir,              color=dc,        fontsize=7.5, va="center", transform=ax_bot.transAxes)
    ax_bot.text(0.32, y, row["pattern"],    color=C_SUBTEXT, fontsize=7,   va="center", transform=ax_bot.transAxes)

    # Pip bar
    bw = min(abs(row["pips"]) / MAX_ABS_PIPS, 1.0) * BAR_MAX_W
    ax_bot.add_patch(mpatches.FancyBboxPatch(
        (0.62, y - ROW_B * 0.30), bw, ROW_B * 0.60,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=rc, alpha=0.8, transform=ax_bot.transAxes))
    ax_bot.text(0.62 + bw + 0.005, y, f"{row['pips']:+.0f}p",
                color=rc, fontsize=7.5, va="center", transform=ax_bot.transAxes)

    pnl_c = C_BOT_WIN if row["pnl"] > 0 else (C_BOT_LOSS if row["pnl"] < 0 else C_BOT_BE)
    ax_bot.text(0.99, y, f"${row['pnl']:+,.0f}", color=pnl_c,
                fontsize=8, fontweight="bold", ha="right", va="center",
                transform=ax_bot.transAxes)

# Bot-only summary
n_wins  = sum(1 for r in bot_only if r["result"] == "WIN")
n_loss  = sum(1 for r in bot_only if r["result"] == "LOSS")
n_be    = sum(1 for r in bot_only if r["result"] == "BE")
net_bot = sum(r["pnl"] for r in bot_only)

bsy = ROW_B * 0.35
ax_bot.add_patch(mpatches.FancyBboxPatch(
    (0, 0), 1, bsy, boxstyle="square,pad=0", linewidth=0,
    facecolor=C_HEADER, transform=ax_bot.transAxes))
ax_bot.text(0.01, bsy/2,
            f"Bot-only:  {n_wins}W  {n_loss}L  {n_be}BE   |   Net: ${net_bot:+,.2f}",
            color=C_SUBTEXT, fontsize=8.5, va="center", transform=ax_bot.transAxes)

# ── Save ──────────────────────────────────────────────────────────────────────
out = LOGS / "forge_vs_alex_chart.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"Saved → {out}")
