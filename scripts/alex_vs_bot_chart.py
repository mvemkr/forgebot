"""
Alex vs Bot — Side-by-side trade comparison for Jul–Oct 2024 window.
"""
import json, sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BG      = "#0a0e1a"
SURFACE = "#141b2d"
TEXT    = "#e0e0e0"
SUBTEXT = "#8a9bb5"
GREEN   = "#26a69a"
RED     = "#ef5350"
AMBER   = "#ffb74d"
BLUE    = "#42a5f5"

# ── Alex's trades (from full transcript) ────────────────────────────────────
ALEX_TRADES = [
    {"week": 1, "pair": "GBP/JPY", "dir": "short", "entry": "~192.50",
     "pips": "+230",  "r": "+3.4R", "result": "WIN",  "note": "4H H&S at 192-193 round level"},
    {"week": 2, "pair": "USD/JPY", "dir": "short", "entry": "~157.50",
     "pips": "+635",  "r": "+6.0R", "result": "WIN",  "note": "H&S neckline retest · $437→$3,074"},
    {"week": 3, "pair": "USD/CHF", "dir": "short", "entry": "~0.899",
     "pips": "~+180", "r": "~+2R",  "result": "WIN",  "note": "Round level short (transcript ref)"},
    {"week": 4, "pair": "LOSSES",  "dir": "mixed", "entry": "N/A",
     "pips": "N/A",   "r": "~-3R",  "result": "LOSS", "note": "Felt entitled · multiple losses"},
    {"week": 7, "pair": "GBP/CHF", "dir": "short", "entry": "~1.135",
     "pips": "+20",   "r": "~0R",   "result": "BE",   "note": "Double top · broker slippage ate profit"},
    {"week": 8, "pair": "NZD/JPY+", "dir": "short", "entry": "~90.00",
     "pips": "+500+", "r": "+10R+", "result": "WIN",  "note": "JPY theme stack (NZD+GBP+AUD/JPY) · $20k→$90k"},
]

# ── Bot trades ───────────────────────────────────────────────────────────────
RESULTS_LOG = pathlib.Path(__file__).parents[1] / "logs" / "backtest_results.jsonl"

def load_bot_trades():
    with open(RESULTS_LOG) as f:
        runs = [json.loads(l) for l in f]
    alex_runs = [r for r in runs
                 if "2024-07-01" in r.get("window_start", "")
                 and "news_filter" not in r.get("notes", "")]
    last = alex_runs[-1]
    return last["trades"], last["results"]

bot_trades_raw, bot_results = load_bot_trades()

def calc_pips(pair, direction, entry, exit_p):
    if exit_p is None: return None
    mult = 100 if "JPY" in pair else 10000
    raw  = (exit_p - entry) * mult
    return raw if direction == "long" else -raw

def result_label(t):
    r = t.get("reason", "")
    if "be_stop" in r or "breakeven" in r: return "BE"
    if "stop" in r:  return "STOP"
    if "open" in r:  return "OPEN"
    return "WIN" if t.get("pnl", 0) > 0 else "LOSS"

bot_trades = []
for t in bot_trades_raw:
    lbl = result_label(t)
    bot_trades.append({
        "pair":    t["pair"],
        "dir":     t["direction"],
        "entry":   t["entry"],
        "exit":    t.get("exit"),
        "pips":    calc_pips(t["pair"], t["direction"], t["entry"], t.get("exit")),
        "r":       t.get("r", 0),
        "pnl":     t.get("pnl", 0),
        "pattern": t.get("pattern", ""),
        "dt":      t.get("entry_ts", "")[:10],
        "label":   lbl,
    })

# Alex pair+dir set for matching
alex_pd = {(a["pair"].split("+")[0], a["dir"]) for a in ALEX_TRADES if a["dir"] != "mixed"}

def rc(label):
    return {"WIN":GREEN,"STOP":RED,"LOSS":RED,"BE":AMBER,"OPEN":BLUE}.get(label, TEXT)

def arc(r):
    return {"WIN":GREEN,"LOSS":RED,"BE":AMBER}.get(r, TEXT)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(22, 24),
                         gridspec_kw={"height_ratios":[1,1,0.22],"hspace":0.04},
                         facecolor=BG)
fig.suptitle("ALEX vs BOT  ·  Jul–Oct 2024 Challenge Window",
             fontsize=20, fontweight="bold", color=TEXT, y=0.995)

def setup_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a3550")

# ─── PANEL 1: BOT ────────────────────────────────────────────────────────────
ax = axes[0]
setup_ax(ax)
ax.text(0.5, 0.97, "[BOT] TRADES  (backtest · Jul–Oct 2024 · all 13)",
        ha="center", va="top", fontsize=13, fontweight="bold", color=TEXT, transform=ax.transAxes)

BCOLS  = [0.01, 0.09, 0.17, 0.24, 0.33, 0.42, 0.49, 0.57, 0.65, 0.74, 0.87]
BHDRS  = ["Date","Pair","Dir","Entry","Exit","Pips","R","P&L","Result","Pattern","Alex?"]
BALIGN = ["left","left","left","right","right","right","right","right","center","left","center"]

yh = 0.89
for i,(h,x) in enumerate(zip(BHDRS,BCOLS)):
    ax.text(x, yh, h, ha=BALIGN[i], va="top", fontsize=8.5,
            fontweight="bold", color=SUBTEXT, transform=ax.transAxes)
ax.plot([0.01,0.99],[yh-0.03,yh-0.03], color="#2a3550", lw=0.8, transform=ax.transAxes)

rh = 0.073
for idx, t in enumerate(bot_trades):
    y = yh - 0.06 - idx*rh
    key = (t["pair"], t["dir"])
    matched = key in alex_pd
    bg = "#1e2e1e" if matched else ("#1a2540" if idx%2==0 else SURFACE)
    ax.add_patch(mpatches.FancyBboxPatch((0.005, y-0.005), 0.99, rh-0.01,
                 boxstyle="square,pad=0", facecolor=bg, alpha=0.6,
                 transform=ax.transAxes, zorder=0))

    pip_s = f"{t['pips']:+.0f}" if t["pips"] is not None else "open"
    r_s   = f"{t['r']:+.1f}R"
    pnl_s = f"${t['pnl']:+,.0f}"
    m_s   = "[Y]" if matched else "—"
    m_c   = GREEN if matched else SUBTEXT

    vals  = [t["dt"], t["pair"], t["dir"].upper(),
             f"{t['entry']:.5f}", f"{t['exit']:.5f}" if t["exit"] else "—",
             pip_s, r_s, pnl_s, t["label"], t["pattern"].replace("_"," "), m_s]
    vcols = [TEXT, TEXT,
             GREEN if t["dir"]=="long" else RED,
             TEXT, TEXT,
             GREEN if (t["pips"] or 0)>0 else RED,
             GREEN if t["r"]>0 else (AMBER if abs(t["r"])<0.1 else RED),
             GREEN if t["pnl"]>0 else (AMBER if abs(t["pnl"])<1 else RED),
             rc(t["label"]), SUBTEXT, m_c]

    for i,(v,x) in enumerate(zip(vals,BCOLS)):
        ax.text(x, y+rh*0.45, v, ha=BALIGN[i], va="center",
                fontsize=8 if i<10 else 9, color=vcols[i],
                fontweight="bold" if i==8 else "normal",
                transform=ax.transAxes)

ys = yh - 0.06 - len(bot_trades)*rh - 0.025
ret = bot_results["return_pct"]
wins = sum(1 for t in bot_trades if t["pnl"]>0)
ax.text(0.5, ys, f"NET: {ret:+.1f}%  ·  {len(bot_trades)} trades  ·  {wins}/{len(bot_trades)} wins  ·  "
        f"matched {sum(1 for t in bot_trades if (t['pair'],t['dir']) in alex_pd)} of Alex's trades",
        ha="center", va="top", fontsize=10, fontweight="bold",
        color=GREEN if ret>0 else RED, transform=ax.transAxes)

# ─── PANEL 2: ALEX ──────────────────────────────────────────────────────────
ax2 = axes[1]
setup_ax(ax2)
ax2.text(0.5, 0.97, "[ALEX]'S TRADES  (from transcript · Jul–Oct 2024 · key trades)",
         ha="center", va="top", fontsize=13, fontweight="bold", color=TEXT, transform=ax2.transAxes)

ACOLS  = [0.01, 0.07, 0.16, 0.24, 0.33, 0.41, 0.49, 0.59, 0.72]
AHDRS  = ["Week","Pair","Dir","~Entry","Pips","R","Result","Bot Took It?","Notes"]
AALIGN = ["left","left","left","right","right","right","center","center","left"]

ax2.text(0.95, 0.91, "[!] Entries/pips are approximate\nwhere transcript lacks exact price",
         ha="right", va="top", fontsize=7.5, color=SUBTEXT,
         style="italic", transform=ax2.transAxes)

yh2 = 0.88
for i,(h,x) in enumerate(zip(AHDRS,ACOLS)):
    ax2.text(x, yh2, h, ha=AALIGN[i], va="top", fontsize=8.5,
             fontweight="bold", color=SUBTEXT, transform=ax2.transAxes)
ax2.plot([0.01,0.99],[yh2-0.03,yh2-0.03], color="#2a3550", lw=0.8, transform=ax2.transAxes)

rh2 = 0.135
for idx, a in enumerate(ALEX_TRADES):
    y2 = yh2 - 0.06 - idx*rh2
    bot_match = any(t["pair"]==a["pair"].split("+")[0] and t["dir"]==a["dir"]
                    for t in bot_trades)
    bg2 = "#1e2e1e" if bot_match else ("#2e1e1e" if a["result"]=="LOSS" else SURFACE)
    ax2.add_patch(mpatches.FancyBboxPatch((0.005, y2-0.005), 0.99, rh2-0.01,
                  boxstyle="square,pad=0", facecolor=bg2, alpha=0.6,
                  transform=ax2.transAxes, zorder=0))

    bt_s = "[MATCH]" if bot_match else ("—" if a["dir"]=="mixed" else "[MISSED]")
    bt_c = GREEN if bot_match else (SUBTEXT if a["dir"]=="mixed" else RED)

    vals2  = [f"Wk {a['week']}", a["pair"], a["dir"].upper() if a["dir"]!="mixed" else "MIXED",
              a["entry"], a["pips"], a["r"], a["result"], bt_s, a["note"]]
    vcols2 = [SUBTEXT, TEXT,
              GREEN if a["dir"]=="long" else (RED if a["dir"]=="short" else SUBTEXT),
              TEXT,
              GREEN if "+" in str(a["pips"]) else (RED if "N/A" in str(a["pips"]) else AMBER),
              GREEN if "+" in str(a["r"]) else RED,
              arc(a["result"]), bt_c, SUBTEXT]

    for i,(v,x) in enumerate(zip(vals2,ACOLS)):
        ax2.text(x, y2+rh2*0.45, v, ha=AALIGN[i], va="center",
                 fontsize=8.5 if i<8 else 7.5, color=vcols2[i],
                 fontweight="bold" if i==6 else "normal",
                 transform=ax2.transAxes)

# ─── PANEL 3: Summary callouts ───────────────────────────────────────────────
ax3 = axes[2]
setup_ax(ax3)

bullets = [
    (GREEN, "[Y] MATCHED:",
     "USD/JPY SHORT · GBP/CHF SHORT · NZD/JPY SHORT  (bot entry levels differ but same pair/direction/level type)"),
    (RED,   "[X] BOT MISSED:",
     "GBP/JPY SHORT Wk1 (+3.4R, Alex's first win) · USD/CHF SHORT Wk3 · full JPY stack scale (bot took NZD/JPY, missed simultaneous GBP/JPY+AUD/JPY)"),
    (AMBER, "[EXTRA] BOT EXTRA:",
     "AUD/NZD · AUD/USD · EUR/GBP · AUD/CAD · EUR/USD · NZD/USD — 6 trades Alex never took · mostly 0.01-increment level noise"),
]

y3 = 0.88
for col, label, desc in bullets:
    ax3.text(0.01, y3, label, ha="left", va="top", fontsize=8.5,
             fontweight="bold", color=col, transform=ax3.transAxes)
    ax3.text(0.16, y3, desc, ha="left", va="top", fontsize=8,
             color=TEXT, transform=ax3.transAxes)
    y3 -= 0.30

out = "/tmp/alex_vs_bot.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
plt.close(fig)
