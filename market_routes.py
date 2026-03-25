# ═══════════════════════════════════════════════════════════════
# market_routes.py
# ═══════════════════════════════════════════════════════════════

import asyncio
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx

router = APIRouter(prefix="/market", tags=["market"])

FRED_API_KEY = "33ef3a5eeaa4ec2bfba86142f960cec0"

# key → (series_id, limit)
FRED_SERIES = {
    "sp500":  ("SP500",            310),  # 200 gg lavorativi + margine per MA200
    "vix":    ("VIXCLS",           250),
    "dxy":    ("DTWEXBGS",         250),
    "gold":   ("GOLDPMGBD228NLBM", 250),
    "y10":    ("DGS10",            250),
    "y2":     ("DGS2",             250),
    "hy_oas": ("BAMLH0A0HYM2",     250),
    "m2":     ("M2SL",             60),   # M2 Money Supply (weekly/monthly)
}


async def fetch_fred_history(client, series_id: str, limit: int) -> list:
    """Storico FRED in ordine cronologico."""
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}"
        f"&sort_order=desc&limit={limit}&file_type=json"
    )
    try:
        r = await client.get(url, timeout=12)
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


async def fetch_yf_chart(client, symbol: str, range_str: str) -> dict | None:
    """Yahoo Finance server-side."""
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?range={range_str}&interval=1d&includePrePost=false"
    )
    try:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        result = r.json().get("chart", {}).get("result", [None])[0]
        if not result:
            return None
        meta       = result.get("meta", {})
        timestamps = result.get("timestamp", [])
        closes     = [v for v in result.get("indicators", {}).get("quote", [{}])[0].get("close", []) if v is not None]
        price      = meta.get("regularMarketPrice") or (closes[-1] if closes else None)
        prev       = meta.get("chartPreviousClose") or meta.get("previousClose")
        dates      = [str(ts) for ts in timestamps]
        return {
            "price":      price,
            "change_pct": round((price - prev) / prev * 100, 2) if price and prev else None,
            "closes":     closes,
            "dates":      dates,
        }
    except Exception as e:
        print(f"[YF {symbol}] {e}")
        return None


def period_changes(hist: list) -> dict:
    """Change% a 1, 7, 15, 30 giorni lavorativi."""
    values = [o["value"] for o in hist]
    n = len(values)

    def pct(offset: int):
        if n <= offset:
            return None
        cur, prev = values[-1], values[-(offset + 1)]
        return round((cur - prev) / prev * 100, 2) if prev else None

    return {"1d": pct(1), "7d": pct(7), "15d": pct(15), "30d": pct(30)}


def moving_average(values: list, window: int) -> float | None:
    w = min(window, len(values))
    return round(sum(values[-w:]) / w, 4) if w else None


def direction(hist: list, offset: int = 1) -> str | None:
    """'up' / 'down' / 'flat' rispetto a N periodi fa."""
    if len(hist) <= offset:
        return None
    cur, prev = hist[-1]["value"], hist[-(offset + 1)]["value"]
    if abs(cur - prev) < 1e-6:
        return "flat"
    return "up" if cur > prev else "down"


@router.get("/data")
async def get_market_data():
    async with httpx.AsyncClient() as client:
        tasks = {
            **{k: fetch_fred_history(client, sid, lim)
               for k, (sid, lim) in FRED_SERIES.items()},
            # Gold Yahoo come fallback/integrazione storico
            "yf_gold": fetch_yf_chart(client, "GC%3DF", "2y"),
        }
        results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values())))

    sp_hist   = results["sp500"]
    vix_hist  = results["vix"]
    dxy_hist  = results["dxy"]
    gold_hist = results["gold"]
    y10_hist  = results["y10"]
    y2_hist   = results["y2"]
    oas_hist  = results["hy_oas"]
    m2_hist   = results["m2"]
    yf_gold   = results.get("yf_gold")

    # ── Se FRED gold ha meno di 31 punti, costruiamo storico da Yahoo ──
    if len(gold_hist) < 31 and yf_gold and yf_gold["closes"]:
        closes = yf_gold["closes"]
        dates  = yf_gold.get("dates", [str(i) for i in range(len(closes))])
        gold_hist = [{"date": d, "value": v} for d, v in zip(dates, closes)]

    def last(hist):
        return hist[-1]["value"] if hist else None

    # ── MA200 (solo FRED SP500) ───────────────────────────────────
    sp_values = [o["value"] for o in sp_hist]
    spy_ma200 = moving_average(sp_values, 200)

    # MA200 di ieri = media degli ultimi 200 punti escludendo l'ultimo
    spy_ma200_yesterday = moving_average(sp_values[:-1], 200) if len(sp_values) > 1 else None
    ma200_direction = None
    if spy_ma200 and spy_ma200_yesterday:
        if abs(spy_ma200 - spy_ma200_yesterday) < 1e-6:
            ma200_direction = "flat"
        else:
            ma200_direction = "up" if spy_ma200 > spy_ma200_yesterday else "down"

    # ── DXY MA20 ─────────────────────────────────────────────────
    dxy_values = [o["value"] for o in dxy_hist]
    dxy_ma20   = moving_average(dxy_values, 20)

    # ── Yield spread 10Y-2Y: storico ─────────────────────────────
    # Allinea le due serie per data
    y10_map = {o["date"]: o["value"] for o in y10_hist}
    y2_map  = {o["date"]: o["value"] for o in y2_hist}
    common_dates = sorted(set(y10_map) & set(y2_map))
    spread_hist = [
        {"date": d, "value": round(y10_map[d] - y2_map[d], 3)}
        for d in common_dates
    ]

    # ── Liquidità M2: direzione ───────────────────────────────────
    m2_direction  = direction(m2_hist, 1)
    m2_change_pct = period_changes(m2_hist) if m2_hist else {}

    # ── Variazioni periodiche ─────────────────────────────────────
    changes = {
        "spy":    period_changes(sp_hist),
        "vix":    period_changes(vix_hist),
        "dxy":    period_changes(dxy_hist),
        "gold":   period_changes(gold_hist),
        "y10":    period_changes(y10_hist),
        "y2":     period_changes(y2_hist),
        "hy_oas": period_changes(oas_hist),
        "m2":     m2_change_pct,
    }

    # ── Direzioni (su/giù vs ieri) ────────────────────────────────
    directions = {
        "spy":    direction(sp_hist),
        "ma200":  ma200_direction,
        "vix":    direction(vix_hist),
        "dxy":    direction(dxy_hist),
        "gold":   direction(gold_hist),
        "y10":    direction(y10_hist),
        "y2":     direction(y2_hist),
        "hy_oas": direction(oas_hist),
        "m2":     m2_direction,
    }

    def hist_slice(hist, n=120):
        return [{"date": o["date"], "value": o["value"]} for o in hist[-n:]]

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

        # Medie mobili
        "spy_ma200": spy_ma200,
        "dxy_ma20":  dxy_ma20,

        # Variazioni % per periodo
        "changes": changes,

        # Direzioni vs ieri: "up" / "down" / "flat" / null
        "directions": directions,

        # Storici per grafici
        "spy_history":    hist_slice(sp_hist, 120),
        "vix_history":    hist_slice(vix_hist, 60),
        "y10_history":    hist_slice(y10_hist, 120),
        "y2_history":     hist_slice(y2_hist,  120),
        "spread_history": hist_slice(spread_hist, 120),
        "m2_history":     hist_slice(m2_hist, 60),
    })
