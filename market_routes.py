# ═══════════════════════════════════════════════════════════════
# market_routes.py  —  aggiungere al backend FastAPI su Render
#
# COME INTEGRARLO:
#   1. Copia questo file nella root del tuo progetto backend
#   2. In main.py aggiungi:
#        from market_routes import router as market_router
#        app.include_router(market_router)
#   3. In requirements.txt aggiungi (se non presenti):
#        httpx
#        fastapi
# ═══════════════════════════════════════════════════════════════

import asyncio
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx

router = APIRouter(prefix="/market", tags=["market"])

FRED_API_KEY = "33ef3a5eeaa4ec2bfba86142f960cec0"

# ── Serie FRED da recuperare ─────────────────────────────────────
FRED_SERIES = {
    "sp500":  "SP500",
    "vix":    "VIXCLS",
    "dxy":    "DTWEXBGS",
    "gold":   "GOLDAMGBD228NLBM",
    "y10":    "DGS10",
    "y2":     "DGS2",
    "hy_oas": "BAMLH0A0HYM2",
}

FRED_HISTORY = {
    "sp500": ("SP500",    300),   # per MA200
    "vix":   ("VIXCLS",  60),    # per grafico
    "dxy":   ("DTWEXBGS", 30),   # per MA20
}


async def fetch_fred_latest(client: httpx.AsyncClient, series_id: str) -> dict | None:
    """Ultima osservazione valida di una serie FRED."""
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}"
        "&sort_order=desc&limit=5&file_type=json"
    )
    try:
        r = await client.get(url, timeout=10)
        r.raise_for_status()
        obs = [o for o in r.json().get("observations", [])
               if o["value"] not in (".", "")]
        if not obs:
            return None
        return {"value": float(obs[0]["value"]), "date": obs[0]["date"]}
    except Exception as e:
        print(f"[FRED {series_id}] {e}")
        return None


async def fetch_fred_history(client: httpx.AsyncClient, series_id: str, limit: int) -> list:
    """Serie storica FRED in ordine cronologico."""
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}"
        f"&sort_order=desc&limit={limit}&file_type=json"
    )
    try:
        r = await client.get(url, timeout=10)
        r.raise_for_status()
        obs = [
            {"date": o["date"], "value": float(o["value"])}
            for o in r.json().get("observations", [])
            if o["value"] not in (".", "")
        ]
        return list(reversed(obs))   # ordine cronologico
    except Exception as e:
        print(f"[FRED history {series_id}] {e}")
        return []


async def fetch_yf_chart(client: httpx.AsyncClient, symbol: str, range_str: str) -> dict | None:
    """Dati storici Yahoo Finance (chiamata server-side, nessun CORS)."""
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?range={range_str}&interval=1d&includePrePost=false"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = await client.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        result = r.json().get("chart", {}).get("result", [None])[0]
        if not result:
            return None
        meta       = result.get("meta", {})
        timestamps = result.get("timestamp", [])
        closes     = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        price      = meta.get("regularMarketPrice")
        prev       = meta.get("chartPreviousClose") or meta.get("previousClose")
        change_pct = ((price - prev) / prev * 100) if price and prev else None
        return {
            "price":      price,
            "change_pct": change_pct,
            "timestamps": timestamps,
            "closes":     [v for v in closes if v is not None],
        }
    except Exception as e:
        print(f"[YF {symbol}] {e}")
        return None


@router.get("/data")
async def get_market_data():
    """
    Endpoint principale: aggrega tutti i dati di mercato.
    Chiamato dal frontend marketregime.html ogni N minuti.
    """
    async with httpx.AsyncClient() as client:

        # Fetch tutto in parallelo
        tasks = {
            # Prezzi correnti FRED
            **{k: fetch_fred_latest(client, sid) for k, sid in FRED_SERIES.items()},
            # Storici FRED per MA e grafici
            **{
                f"hist_{k}": fetch_fred_history(client, sid, lim)
                for k, (sid, lim) in FRED_HISTORY.items()
            },
            # SPY storico da Yahoo come fallback/integrazione
            "yf_spy": fetch_yf_chart(client, "SPY", "1y"),
        }

        results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values())))

    # ── Prezzi correnti ──────────────────────────────────────────
    def val(key):
        r = results.get(key)
        return r["value"] if r else None

    sp500_hist = results.get("hist_sp500", [])
    vix_hist   = results.get("hist_vix",   [])
    dxy_hist   = results.get("hist_dxy",   [])
    yf_spy     = results.get("yf_spy")

    # Usa Yahoo per S&P se FRED non disponibile (weekend/festivi)
    spy_price  = val("sp500") or (yf_spy["price"] if yf_spy else None)

    # Change% da storico (oggi vs ieri)
    def change_from_hist(hist):
        if len(hist) >= 2:
            c, p = hist[-1]["value"], hist[-2]["value"]
            return round((c - p) / p * 100, 2) if p else None
        return None

    # MA200 da storico SP500 (preferisce Yahoo che è più lungo)
    sp_closes = [v for v in (yf_spy["closes"] if yf_spy else [o["value"] for o in sp500_hist])]
    ma200 = round(sum(sp_closes[-200:]) / min(len(sp_closes), 200), 2) if sp_closes else None

    # MA20 DXY
    dxy_closes = [o["value"] for o in dxy_hist]
    dxy_ma20   = round(sum(dxy_closes[-20:]) / min(len(dxy_closes), 20), 2) if dxy_closes else None

    return JSONResponse({
        # Prezzi
        "spy":           spy_price,
        "spy_change":    change_from_hist(sp500_hist) or (yf_spy["change_pct"] if yf_spy else None),
        "spy_ma200":     ma200,
        "vix":           val("vix"),
        "vix_change":    change_from_hist(vix_hist),
        "dxy":           val("dxy"),
        "dxy_ma20":      dxy_ma20,
        "gold":          val("gold"),
        "y10":           val("y10"),
        "y2":            val("y2"),
        "credit_spread": val("hy_oas"),

        # Storici per grafici
        "spy_history": [
            {"date": o["date"], "value": o["value"]} for o in sp500_hist[-120:]
        ] if sp500_hist else (
            [{"date": str(ts), "value": v}
             for ts, v in zip(yf_spy["timestamps"][-120:], yf_spy["closes"][-120:])]
            if yf_spy else []
        ),
        "vix_history": [
            {"date": o["date"], "value": o["value"]} for o in vix_hist
        ],
    })
