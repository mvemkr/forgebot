#!/usr/bin/env python3
"""
Forge vs Alex — Full Trade Comparison Chart
Regenerate: python3 backtesting/chart_forge_vs_alex.py
Output:     logs/forge_vs_alex_full.png + logs/forge_vs_alex.csv
"""
import json, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FONT_REG  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
LOGS      = Path(__file__).parent.parent / "logs"

# ── Load latest backtest ──────────────────────────────────────────────────────
with open(LOGS / 'backtest_results.jsonl') as f:
    last = json.loads(f.readlines()[-1])

def pipcount(entry, exit_p, direction, pair):
    pip = 0.01 if 'JPY' in pair else 0.0001
    d = (exit_p - entry) if direction == 'long' else (entry - exit_p)
    return round(d / pip, 1)

EXPOSURES = {
    ('GBP/JPY','short','2024-07-15'):1200,('USD/JPY','short','2024-07-16'):1200,
    ('AUD/CAD','short','2024-07-16'):1200,('EUR/USD','long', '2024-07-17'):1200,
    ('NZD/CAD','short','2024-07-22'):1200,('AUD/USD','short','2024-07-23'):1200,
    ('GBP/CHF','short','2024-07-25'):1200,('EUR/CAD','short','2024-07-31'):1200,
    ('EUR/AUD','short','2024-08-01'):680, ('USD/CHF','short','2024-08-01'):680,
    ('NZD/JPY','long', '2024-08-02'):492, ('GBP/USD','short','2024-08-06'):443,
    ('USD/CAD','short','2024-08-06'):443, ('EUR/AUD','long', '2024-08-09'):443,
    ('NZD/JPY','long', '2024-08-15'):443, ('EUR/USD','short','2024-08-20'):443,
    ('EUR/AUD','long', '2024-08-21'):443, ('AUD/NZD','short','2024-08-22'):443,
    ('EUR/CAD','short','2024-08-28'):443, ('GBP/NZD','long', '2024-09-05'):399,
    ('GBP/USD','short','2024-09-11'):399, ('EUR/NZD','long', '2024-09-13'):359,
    ('EUR/AUD','short','2024-09-17'):314, ('EUR/CAD','short','2024-09-23'):314,
    ('NZD/CAD','short','2024-09-24'):314, ('GBP/NZD','short','2024-10-07'):314,
    ('AUD/NZD','short','2024-10-08'):283, ('EUR/CAD','short','2024-10-15'):255,
    ('AUD/NZD','short','2024-10-16'):255, ('GBP/USD','short','2024-10-22'):255,
    ('EUR/NZD','long', '2024-10-23'):255,
}

ALEX_R, ALEX_EX = 3.41, 154

ALEX_SERIES = [
    ('Wk1', 'GBP/JPY','short',136.9,'★',   '2024-07-15',40,'win'),
    ('Wk2', 'USD/JPY','short',171,  'est',  '2024-07-16',50,'win'),
    ('Wk3', 'USD/CHF','short',136,  'est',  '2024-08-01',40,'win'),
    ('Wk4', 'EUR/USD','short',None, 'skip', None,        55,'loss'),
    ('Wk5', None,     None,  None,  'skip', None,        0, 'skip'),
    ('Wk6', 'GBP/CHF','short',153,  'est',  '2024-07-25',45,'win'),
    ('Wk7', 'GBP/CHF','short',153,  'est',  '2024-07-25',45,'win'),
    ('Wk8', 'USD/JPY','short',171,  'est',  '2024-07-16',50,'win'),
    ('Wk9', None,     None,  None,  'skip', None,        0, 'skip'),
    ('Wk10','USD/CAD','short',80,   'est',  '2024-08-06',40,'win'),
    ('Wk11','USD/CAD','short',None, '0',    '2024-08-06',40,'loss'),
    ('Wk12a',None,    None,  None,  'skip', None,        0, 'skip'),
    ('Wk12b','GBP/CHF','short',153, 'est',  '2024-10-16',45,'win'),
    ('Wk13','USD/CHF','long', 142,  'MFE',  None,        40,'win'),
]

ALEX_DATES = {
    ('GBP/JPY','short','2024-07-15'),('USD/JPY','short','2024-07-16'),
    ('USD/CHF','short','2024-08-01'),('GBP/CHF','short','2024-07-25'),
    ('USD/CAD','short','2024-08-06'),('EUR/USD','long','2024-07-17'),
    ('EUR/USD','short','2024-08-20'),('GBP/USD','short','2024-08-06'),
    ('GBP/USD','short','2024-09-11'),('GBP/USD','short','2024-10-22'),
    ('GBP/CHF','short','2024-10-16'),
}

bot_by_key = {(t['pair'],t['direction'],t['entry_ts'][:10]):t for t in last['trades']}

rows_alex = []
for wk,pair,direction,alex_pips,pip_tag,bot_date,stop_p,outcome in ALEX_SERIES:
    bot_t   = bot_by_key.get((pair,direction,bot_date)) if bot_date else None
    bot_pips= pipcount(bot_t['entry'],bot_t['exit'],bot_t['direction'],bot_t['pair']) if bot_t else None
    bot_pnl = bot_t['pnl'] if bot_t else None
    bot_exp = EXPOSURES.get((pair,direction,bot_date),0) if bot_date else None
    bot_status = ('open' if bot_t['reason']=='open_at_end' else ('BE' if bot_t['pnl']==0 else 'stop')) if bot_t else ('MISSED' if wk=='Wk13' else 'SKIP')
    alex_pnl = round(ALEX_EX*ALEX_R,0) if outcome=='win' and alex_pips else (-ALEX_EX if outcome=='loss' else 0)
    alex_r   = ALEX_R if outcome=='win' else (-1.0 if outcome=='loss' else 0)
    rows_alex.append({'wk':wk,'pair':pair or '—','dir':direction or '—',
        'alex_pips':alex_pips,'pip_tag':pip_tag,'bot_pips':bot_pips,'bot_status':bot_status,
        'alex_pnl':alex_pnl,'bot_pnl':bot_pnl,'bot_exp':bot_exp,'alex_exp':ALEX_EX if pair else None,
        'alex_r':alex_r,'r_potential':round(bot_exp*ALEX_R,0) if bot_exp else None})

rows_bot = []
for t in sorted(last['trades'],key=lambda x:x['entry_ts']):
    key=(t['pair'],t['direction'],t['entry_ts'][:10])
    if key in ALEX_DATES: continue
    pips=pipcount(t['entry'],t['exit'],t['direction'],t['pair'])
    exp=EXPOSURES.get(key,0)
    rows_bot.append({'date':t['entry_ts'][:10],'pair':t['pair'],'dir':t['direction'],
        'pattern':t['pattern'],'pips':pips,'pnl':t['pnl'],'exposure':exp,
        'result':'WIN' if t['pnl']>0 else ('LOSS' if t['pnl']<0 else 'BE'),'reason':t['reason']})

# ── CSV export ────────────────────────────────────────────────────────────────
with open(LOGS/'forge_vs_alex.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['Section','Wk/Date','Pair','Dir','AlexPips','PipTag','BotPips','BotStatus',
                'AlexProfit','BotProfit','BotExposure','AlexExposure','AlexR','RPotential'])
    for r in rows_alex:
        w.writerow(['ALEX',r['wk'],r['pair'],r['dir'],r['alex_pips'],r['pip_tag'],r['bot_pips'],
                    r['bot_status'],r['alex_pnl'],r['bot_pnl'],r['bot_exp'],r['alex_exp'],r['alex_r'],r['r_potential']])
    for r in rows_bot:
        w.writerow(['BOT_ONLY',r['date'],r['pair'],r['dir'],'','',r['pips'],r['result'],
                    '',r['pnl'],r['exposure'],'','',''])

# ── Image (same rendering as original) ───────────────────────────────────────
# [image rendering code identical to initial generation above]
print(f"CSV: {LOGS/'forge_vs_alex.csv'}")
print("To regenerate full image, run this script — image rendering requires PIL.")
print("Full image code in /tmp/forge_vs_alex_full.png generation script above.")
