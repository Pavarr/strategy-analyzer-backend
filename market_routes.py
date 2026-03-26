# ═══════════════════════════════════════════════════════════════
# market_routes.py
# ═══════════════════════════════════════════════════════════════


import asyncio
from datetime import date, timedelta
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx

router = APIRouter(prefix="/market", tags=["market"])

FRED_API_KEY = "33ef3a5eeaa4ec2bfba86142f960cec0"

# ── FRED series: key → (series_id, limit) ───────────────────────
# limit = 560 ≈ 2 anni lavorativi (serve per correlazioni rolling 365gg)
FRED_SERIES = {
    "sp500":  ("SP500",            560),
    "vix":    ("VIXCLS",           560),
    "gold":   ("GOLDPMGBD228NLBM", 560),
    "y10":    ("DGS10",            560),
    "y2":     ("DGS2",             560),
    "hy_oas": ("BAMLH0A0HYM2",     560),
    "m2":     ("M2SL",             120),
}

# ── Yahoo Finance symbols (server-side, no CORS) ────────────────
YF_SYMBOLS = {
    "dxy":  ("DX-Y.NYB", "2y"),   # DXY ICE — il vero dollar index
    "gold_yf": ("GC%3DF",  "2y"), # Gold futures fallback
}


async def fetch_fred_history(client, series_id: str, limit: int) -> list:
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}"
        f"&sort_order=desc&limit={limit}&file_type=json"
    )
    try:
        r = await client.get(url, timeout=15)
        r.raise_for_status()
        obs = [
            {"date": o["date"], "value": float(o["value"])}
            for o in r.json().get("observations", [])
            if o["value"] not in (".", "")
        ]
        return list(reversed(obs))
    except Exception as e:
        print(f"[FRED {series_id}] {e}")
        return []


async def fetch_yf_chart(client, symbol: str, range_str: str) -> list:
    """Yahoo Finance → lista cronologica [{date, value}]."""
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?range={range_str}&interval=1d&includePrePost=false"
    )
    try:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        r.raise_for_status()
        result = r.json().get("chart", {}).get("result", [None])[0]
        if not result:
            return []
        timestamps = result.get("timestamp", [])
        closes     = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        out = []
        for ts, v in zip(timestamps, closes):
            if v is not None:
                d = date.fromtimestamp(ts).isoformat()
                out.append({"date": d, "value": round(float(v), 4)})
        return out
    except Exception as e:
        print(f"[YF {symbol}] {e}")
        return []


def period_changes(hist: list) -> dict:
    values = [o["value"] for o in hist]
    n = len(values)
    def pct(offset):
        if n <= offset:
            return None
        cur, prev = values[-1], values[-(offset + 1)]
        return round((cur - prev) / prev * 100, 2) if prev else None
    return {"1d": pct(1), "3d": pct(3), "7d": pct(7), "15d": pct(15), "30d": pct(30)}


def moving_average(values: list, window: int):
    w = min(window, len(values))
    return round(sum(values[-w:]) / w, 4) if w else None


def direction(hist: list, offset: int = 1) -> str | None:
    if len(hist) <= offset:
        return None
    cur, prev = hist[-1]["value"], hist[-(offset + 1)]["value"]
    if abs(cur - prev) < 1e-9:
        return "flat"
    return "up" if cur > prev else "down"


def hist_slice(hist: list, n: int = 120) -> list:
    return [{"date": o["date"], "value": o["value"]} for o in hist[-n:]]


@router.get("/data")
async def get_market_data():
    async with httpx.AsyncClient() as client:
        fred_tasks = {k: fetch_fred_history(client, sid, lim)
                      for k, (sid, lim) in FRED_SERIES.items()}
        yf_tasks   = {k: fetch_yf_chart(client, sym, rng)
                      for k, (sym, rng) in YF_SYMBOLS.items()}
        all_tasks  = {**fred_tasks, **yf_tasks}
        results    = dict(zip(all_tasks.keys(), await asyncio.gather(*all_tasks.values())))

    sp_hist   = results["sp500"]
    vix_hist  = results["vix"]
    gold_hist = results["gold"]
    y10_hist  = results["y10"]
    y2_hist   = results["y2"]
    oas_hist  = results["hy_oas"]
    m2_hist   = results["m2"]
    dxy_hist  = results["dxy"]          # Yahoo DX-Y.NYB — vero DXY ~100
    gold_yf   = results.get("gold_yf", [])

    # Gold: se FRED ha <31 punti usa Yahoo
    if len(gold_hist) < 31 and gold_yf:
        gold_hist = gold_yf

    def last(hist):
        return hist[-1]["value"] if hist else None

    # ── MA e direzioni ────────────────────────────────────────────
    sp_values  = [o["value"] for o in sp_hist]
    dxy_values = [o["value"] for o in dxy_hist]
    spy_ma200  = moving_average(sp_values, 200)
    dxy_ma20   = moving_average(dxy_values, 20)

    spy_ma200_yday = moving_average(sp_values[:-1], 200) if len(sp_values) > 1 else None
    ma200_dir = (None if not (spy_ma200 and spy_ma200_yday)
                 else "flat" if abs(spy_ma200 - spy_ma200_yday) < 1e-6
                 else "up" if spy_ma200 > spy_ma200_yday else "down")

    # ── Yield spread storico ──────────────────────────────────────
    y10_map = {o["date"]: o["value"] for o in y10_hist}
    y2_map  = {o["date"]: o["value"] for o in y2_hist}
    spread_hist = [
        {"date": d, "value": round(y10_map[d] - y2_map[d], 3)}
        for d in sorted(set(y10_map) & set(y2_map))
    ]

    # ── Changes e directions ──────────────────────────────────────
    changes = {
        "spy":    period_changes(sp_hist),
        "vix":    period_changes(vix_hist),
        "dxy":    period_changes(dxy_hist),
        "gold":   period_changes(gold_hist),
        "y10":    period_changes(y10_hist),
        "y2":     period_changes(y2_hist),
        "hy_oas": period_changes(oas_hist),
        "m2":     period_changes(m2_hist),
    }
    directions = {
        "spy":    direction(sp_hist),
        "ma200":  ma200_dir,
        "vix":    direction(vix_hist),
        "dxy":    direction(dxy_hist),
        "gold":   direction(gold_hist),
        "y10":    direction(y10_hist),
        "y2":     direction(y2_hist),
        "hy_oas": direction(oas_hist),
        "m2":     direction(m2_hist),
    }

    return JSONResponse({
        # Prezzi correnti
        "spy":           last(sp_hist),
        "vix":           last(vix_hist),
        "dxy":           last(dxy_hist),
        "gold":          last(gold_hist),
        "y10":           last(y10_hist),
        "y2":            last(y2_hist),
        "credit_spread": last(oas_hist),
        "m2":            last(m2_hist),

        # MA
        "spy_ma200": spy_ma200,
        "dxy_ma20":  dxy_ma20,

        "changes":    changes,
        "directions": directions,

        # ── Storici completi per grafici e correlazioni (365gg) ──
        # Ogni asset ha la sua serie completa — il frontend userà
        # queste per l'asset chart e per il correlation chart
        "histories": {
            "spy":    hist_slice(sp_hist,     365),
            "vix":    hist_slice(vix_hist,    365),
            "dxy":    hist_slice(dxy_hist,    365),
            "gold":   hist_slice(gold_hist,   365),
            "y10":    hist_slice(y10_hist,    365),
            "y2":     hist_slice(y2_hist,     365),
            "hy_oas": hist_slice(oas_hist,    365),
            "spread": hist_slice(spread_hist, 365),
        },
    })
