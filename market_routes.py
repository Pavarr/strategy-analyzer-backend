# ═══════════════════════════════════════════════════════════════
# market_routes.py  —  FastAPI backend per Market Regime Dashboard
#
# COME INTEGRARLO:
#   1. Copia nella root del progetto backend (stessa dir di main.py)
#   2. In main.py:
#        from market_routes import router as market_router
#        app.include_router(market_router)
# ═══════════════════════════════════════════════════════════════

import asyncio
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx

router = APIRouter(prefix="/market", tags=["market"])

FRED_API_KEY = "33ef3a5eeaa4ec2bfba86142f960cec0"

# ── Config serie FRED: key → (series_id, limit_storico, label) ──
# SP500 ne serve 310 per MA200 corretta (200 gg lavorativi)
# Gli altri: 250 per coprire ~30 gg lavorativi con margine
FRED_SERIES_CONFIG = {
    "sp500":  ("SP500",           310, "S&P 500"),
    "vix":    ("VIXCLS",          250, "VIX"),
    "dxy":    ("DTWEXBGS",        250, "DXY"),
    "gold":   ("GOLDPMGBD228NLBM",250, "Gold"),
    "y10":    ("DGS10",           250, "US 10Y"),
    "y2":     ("DGS2",            250, "US 2Y"),
    "hy_oas": ("BAMLH0A0HYM2",    250, "HY OAS"),
}


async def fetch_fred_history(client, series_id: str, limit: int) -> list:
    """Storico FRED in ordine cronologico (dal più vecchio al più recente)."""
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
    """Yahoo Finance server-side — nessun CORS."""
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
        meta   = result.get("meta", {})
        closes = [v for v in result.get("indicators", {}).get("quote", [{}])[0].get("close", []) if v is not None]
        price  = meta.get("regularMarketPrice") or (closes[-1] if closes else None)
        prev   = meta.get("chartPreviousClose") or meta.get("previousClose")
        return {
            "price":     price,
            "change_pct": round((price - prev) / prev * 100, 2) if price and prev else None,
            "closes":    closes,
        }
    except Exception as e:
        print(f"[YF {symbol}] {e}")
        return None


def period_changes(hist: list) -> dict:
    """
    Calcola variazioni % a 1, 7, 15, 30 giorni lavorativi
    dallo storico cronologico FRED.
    """
    values = [o["value"] for o in hist]
    n = len(values)

    def pct(offset: int):
        if n <= offset:
            return None
        cur  = values[-1]
        prev = values[-(offset + 1)]
        return round((cur - prev) / prev * 100, 2) if prev else None

    return {"1d": pct(1), "7d": pct(7), "15d": pct(15), "30d": pct(30)}


def moving_average(values: list, window: int):
    w = min(window, len(values))
    if not w:
        return None
    return round(sum(values[-w:]) / w, 4)


@router.get("/data")
async def get_market_data():
    async with httpx.AsyncClient() as client:
        tasks = {
            **{k: fetch_fred_history(client, sid, lim)
               for k, (sid, lim, _) in FRED_SERIES_CONFIG.items()},
            "yf_gold": fetch_yf_chart(client, "GC%3DF", "5d"),
        }
        results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values())))

    sp_hist   = results["sp500"]
    vix_hist  = results["vix"]
    dxy_hist  = results["dxy"]
    gold_hist = results["gold"]
    y10_hist  = results["y10"]
    y2_hist   = results["y2"]
    oas_hist  = results["hy_oas"]
    yf_gold   = results.get("yf_gold")

    def last(hist):
        return hist[-1]["value"] if hist else None

    # ── MA200 usa ESCLUSIVAMENTE storico FRED SP500 ──────────────
    # (evita mismatch tra indice S&P ~6500 e ETF SPY ~560)
    sp_values  = [o["value"] for o in sp_hist]
    dxy_values = [o["value"] for o in dxy_hist]
    spy_ma200  = moving_average(sp_values, 200)
    dxy_ma20   = moving_average(dxy_values, 20)

    # ── Variazioni periodiche ─────────────────────────────────────
    changes = {
        "spy":    period_changes(sp_hist),
        "vix":    period_changes(vix_hist),
        "dxy":    period_changes(dxy_hist),
        "gold":   period_changes(gold_hist),
        "y10":    period_changes(y10_hist),
        "y2":     period_changes(y2_hist),
        "hy_oas": period_changes(oas_hist),
    }

    return JSONResponse({
        "spy":           last(sp_hist),
        "vix":           last(vix_hist),
        "dxy":           last(dxy_hist),
        "gold":          last(gold_hist) or (yf_gold["price"] if yf_gold else None),
        "y10":           last(y10_hist),
        "y2":            last(y2_hist),
        "credit_spread": last(oas_hist),

        "spy_ma200": spy_ma200,
        "dxy_ma20":  dxy_ma20,

        # changes.spy = { "1d": x, "7d": x, "15d": x, "30d": x }
        "changes": changes,

        # Storici per grafici
        "spy_history": [{"date": o["date"], "value": o["value"]} for o in sp_hist[-120:]],
        "vix_history": [{"date": o["date"], "value": o["value"]} for o in vix_hist[-60:]],
    })
