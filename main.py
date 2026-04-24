from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from strategy_parser_v2 import StrategyParser

app = FastAPI(title="Strategy Analyzer API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
parsers_storage = {}
strategies_storage = {}

@app.get("/")
async def root():
    return {
        "message": "Strategy Analyzer API v2",
        "endpoints": {
            "/upload": "POST - Upload strategy HTML",
            "/strategies/{name}": "GET - Get strategy data",
            "/strategies": "GET - List all strategies",
            "/strategies/{name}": "DELETE - Delete strategy",
            "/recalculate/{name}": "GET - Recalculate with new parameters"
        }
    }

@app.post("/upload")
async def upload_strategy(file: UploadFile = File(...)):
    """Upload and parse MetaTrader HTML file"""
    
    if not file.filename.endswith('.html'):
        raise HTTPException(status_code=400, detail="File must be .html")
    
    try:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.html') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        parser = StrategyParser(temp_path, original_filename=file.filename)
        result = parser.generate_output(risk_free_rate=0, trade_type='all')
        
        strategy_name = result['strategy_name']
        parsers_storage[strategy_name] = parser
        strategies_storage[strategy_name] = result
        
        os.unlink(temp_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies/{name}")
async def get_strategy(
    name: str,
    trade_type: str = Query('all', regex='^(all|long|short)$')
):
    if name not in parsers_storage:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    try:
        parser = parsers_storage[name]
        result = parser.generate_output(risk_free_rate=0, trade_type=trade_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recalculate/{name}")
async def recalculate_metrics(
    name: str,
    rf_rate: float = Query(0, ge=0, le=10),
    balance: float = Query(None),
    lot_multiplier: float = Query(None),
    trade_type: str = Query('all', regex='^(all|long|short)$')
):
    if name not in parsers_storage:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    try:
        parser = parsers_storage[name]
        parser.reset_trades()
        
        if lot_multiplier is not None and parser.original_lot_size:
            ratio = lot_multiplier / parser.original_lot_size
            parser.apply_lot_multiplier(ratio)
        
        result = parser.generate_output(
            risk_free_rate=rf_rate,
            custom_balance=balance,
            trade_type=trade_type
        )
        strategies_storage[name] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/strategies/{name}")
async def delete_strategy(name: str):
    if name not in strategies_storage:
        raise HTTPException(status_code=404, detail="Strategy not found")
    del strategies_storage[name]
    del parsers_storage[name]
    return {"message": f"Strategy '{name}' deleted"}

@app.get("/strategies")
async def list_strategies():
    return {"strategies": list(strategies_storage.keys())}


# =============================================
# MT5 LIVE ACCOUNTS ENDPOINTS
# =============================================
from pydantic import BaseModel
from typing import Optional, List
import httpx
from datetime import datetime, timedelta

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

def supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json"
    }

class MT5Data(BaseModel):
    api_key: str
    account_number: str
    account_name: Optional[str] = ""
    balance: float
    equity: float
    drawdown_pct: float
    open_trades: Optional[list] = []
    closed_trades: Optional[list] = []
    stats: Optional[dict] = {}
    currency: Optional[str] = "USD"
    platform: Optional[str] = "MT5"

@app.post("/mt5/update")
async def mt5_update(data: MT5Data):
    """Riceve i dati dall'EA MT5 e li salva su Supabase"""

    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={"api_key": f"eq.{data.api_key}", "select": "id"}
        )
        profiles = res.json()
        if not profiles:
            raise HTTPException(status_code=401, detail="API key non valida")

        user_id = profiles[0]["id"]

        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/mt5_accounts",
            headers=supabase_headers(),
            params={
                "user_id": f"eq.{user_id}",
                "account_number": f"eq.{data.account_number}",
                "select": "id"
            }
        )
        existing = res.json()

        payload = {
            "user_id": user_id,
            "account_number": data.account_number,
            "account_name": data.account_name,
            "balance": data.balance,
            "equity": data.equity,
            "drawdown_pct": data.drawdown_pct,
            "open_trades": data.open_trades,
            "closed_trades": data.closed_trades,
            "stats": data.stats,
            "currency": data.currency,
            "platform": data.platform,
            "last_update": datetime.utcnow().isoformat()
        }

        if existing:
            record_id = existing[0]["id"]
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/mt5_accounts",
                headers=supabase_headers(),
                params={"id": f"eq.{record_id}"},
                json=payload
            )
        else:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/mt5_accounts",
                headers=supabase_headers(),
                json=payload
            )

    # Snapshot storico — arrotonda al minuto
    now_rounded = datetime.utcnow().replace(second=0, microsecond=0)
    snapshot_payload = {
        "user_id": user_id,
        "account_number": data.account_number,
        "currency": data.currency,
        "platform": data.platform,
        "balance": data.balance,
        "equity": data.equity,
        "recorded_at": now_rounded.isoformat(),
    }
    async with httpx.AsyncClient() as client2:
        await client2.post(
            f"{SUPABASE_URL}/rest/v1/mt5_snapshots",
            headers=supabase_headers(),
            json=snapshot_payload
        )

    # Trade chiusi (no duplicati grazie a unique constraint)
    if data.closed_trades:
        async with httpx.AsyncClient() as client3:
            for trade in data.closed_trades:
                try:
                    trade_payload = {
                        "user_id": user_id,
                        "account_number": data.account_number,
                        "account_name": data.account_name,
                        "currency": data.currency,
                        "ticket": trade.get("ticket"),
                        "symbol": trade.get("symbol"),
                        "type": trade.get("type"),
                        "lots": trade.get("lots"),
                        "price": trade.get("price"),
                        "profit": trade.get("profit"),
                        "magic": trade.get("magic", 0),
                        "close_time": trade.get("time"),
                        "open_time": trade.get("open_time"),
                    }
                    await client3.post(
                        f"{SUPABASE_URL}/rest/v1/mt5_trade_history",
                        headers={**supabase_headers(), "Prefer": "resolution=ignore-duplicates"},
                        json=trade_payload
                    )
                except Exception:
                    pass

    return {"status": "ok"}


@app.get("/mt5/accounts")
async def mt5_get_accounts(api_key: str = Query(...)):
    """Restituisce tutti i conti MT5 dell'utente"""

    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={"api_key": f"eq.{api_key}", "select": "id"}
        )
        profiles = res.json()
        if not profiles:
            raise HTTPException(status_code=401, detail="API key non valida")

        user_id = profiles[0]["id"]

        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/mt5_accounts",
            headers=supabase_headers(),
            params={"user_id": f"eq.{user_id}"}
        )
        accounts = res.json()

    return {"accounts": accounts}


@app.get("/mt5/snapshots")
async def mt5_get_snapshots(
    api_key: str = Query(...),
    date_from: str = Query(None)
):
    """Restituisce gli snapshot storici dell'utente."""

    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={"api_key": f"eq.{api_key}", "select": "id"}
        )
        profiles = res.json()
        if not profiles:
            raise HTTPException(status_code=401, detail="API key non valida")

        user_id = profiles[0]["id"]

        if not date_from:
            date_from = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")

        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/mt5_snapshots",
            headers=supabase_headers(),
            params={
                "user_id": f"eq.{user_id}",
                "order": "recorded_at.asc",
                "select": "account_number,currency,balance,equity,recorded_at",
                "limit": "50000",
                "recorded_at": f"gte.{date_from}"
            }
        )
        snapshots = res.json()

    return {"snapshots": snapshots}


@app.get("/mt5/trade-history")
async def mt5_trade_history(
    api_key: str = Query(...),
    date_from: str = Query(None),
    date_to: str = Query(None)
):
    """Restituisce lo storico trade chiusi dell'utente con filtri opzionali"""

    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={"api_key": f"eq.{api_key}", "select": "id"}
        )
        profiles = res.json()
        if not profiles:
            raise HTTPException(status_code=401, detail="API key non valida")

        user_id = profiles[0]["id"]

        params = [
            ("user_id", f"eq.{user_id}"),
            ("order", "close_time.desc"),
            ("limit", "10000"),
        ]
        if date_from:
            params.append(("close_time", f"gte.{date_from}"))
        if date_to:
            params.append(("close_time", f"lte.{date_to}"))

        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/mt5_trade_history",
            headers=supabase_headers(),
            params=params
        )
        trades = res.json()

    return {"trades": trades}


@app.get("/capital-events")
async def get_capital_events(api_key: str = Query(...)):
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={"api_key": f"eq.{api_key}", "select": "id"}
        )
        profiles = res.json()
        if not profiles:
            raise HTTPException(status_code=401, detail="API key non valida")
        user_id = profiles[0]["id"]

        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/capital_events",
            headers=supabase_headers(),
            params={"user_id": f"eq.{user_id}", "order": "event_date.asc", "limit": "1000"}
        )
    return {"events": res.json()}


class CapitalEventData(BaseModel):
    account_number: str
    currency: str
    amount: float
    event_date: Optional[str] = None
    description: Optional[str] = ""

@app.post("/capital-events")
async def add_capital_event(data: CapitalEventData, api_key: str = Query(...)):
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={"api_key": f"eq.{api_key}", "select": "id"}
        )
        profiles = res.json()
        if not profiles:
            raise HTTPException(status_code=401, detail="API key non valida")
        user_id = profiles[0]["id"]

        payload = {
            "user_id": user_id,
            "account_number": data.account_number,
            "currency": data.currency,
            "amount": data.amount,
            "description": data.description,
        }
        if data.event_date:
            payload["event_date"] = data.event_date

        await client.post(
            f"{SUPABASE_URL}/rest/v1/capital_events",
            headers=supabase_headers(),
            json=payload
        )
    return {"status": "ok"}


@app.delete("/capital-events/{event_id}")
async def delete_capital_event(event_id: str, api_key: str = Query(...)):
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={"api_key": f"eq.{api_key}", "select": "id"}
        )
        profiles = res.json()
        if not profiles:
            raise HTTPException(status_code=401, detail="API key non valida")

        await client.delete(
            f"{SUPABASE_URL}/rest/v1/capital_events",
            headers=supabase_headers(),
            params={"id": f"eq.{event_id}"}
        )
    return {"status": "ok"}


# =============================================
# PROP FIRM MATEMATICA ENDPOINT
# =============================================

class PropFirmData(BaseModel):
    api_key: str
    account_number: str
    account_name: Optional[str] = ""
    currency: Optional[str] = "USD"
    balance: float
    equity: float
    drawdown_pct: Optional[float] = 0
    open_trades: Optional[list] = []

@app.post("/mt5/propfirm/update")
async def propfirm_update(data: PropFirmData):
    """
    Riceve i dati dall'EA PropFirmSender e li salva in propfirm_snapshots.
    Separato da mt5_snapshots — questi conti non appaiono in LivePortafogli.
    """
    async with httpx.AsyncClient() as client:
        # Valida API key e recupera user_id
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={"api_key": f"eq.{data.api_key}", "select": "id"}
        )
        profiles = res.json()
        if not profiles:
            raise HTTPException(status_code=401, detail="API key non valida")

        user_id = profiles[0]["id"]

        # Inserisci snapshot in propfirm_snapshots
        now_rounded = datetime.utcnow().replace(second=0, microsecond=0)
        snapshot_payload = {
            "user_id":        user_id,
            "account_number": data.account_number,
            "account_name":   data.account_name,
            "currency":       data.currency,
            "balance":        data.balance,
            "equity":         data.equity,
            "drawdown_pct":   data.drawdown_pct,
            "open_trades":    data.open_trades,
            "recorded_at":    now_rounded.isoformat(),
        }

        await client.post(
            f"{SUPABASE_URL}/rest/v1/propfirm_snapshots",
            headers=supabase_headers(),
            json=snapshot_payload
        )

    return {
        "status": "ok",
        "account": data.account_number,
        "balance": data.balance,
        "equity":  data.equity
    }


from market_routes import router as market_router
app.include_router(market_router)
