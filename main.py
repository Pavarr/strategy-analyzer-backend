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
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.html') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Parse with original filename
        parser = StrategyParser(temp_path, original_filename=file.filename)
        result = parser.generate_output(risk_free_rate=0, trade_type='all')
        
        strategy_name = result['strategy_name']
        
        # Store parser and result
        parsers_storage[strategy_name] = parser
        strategies_storage[strategy_name] = result
        
        # Cleanup
        os.unlink(temp_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies/{name}")
async def get_strategy(
    name: str,
    trade_type: str = Query('all', regex='^(all|long|short)$')
):
    """Get strategy data with optional trade type filter"""
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
    """Recalculate metrics with new parameters"""
    if name not in parsers_storage:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    try:
        parser = parsers_storage[name]
        
        # Reset to original
        parser.reset_trades()
        
        # Apply lot multiplier if provided (lot_multiplier è il NUOVO lottaggio assoluto)
        if lot_multiplier is not None and parser.original_lot_size:
            # Calcola ratio
            ratio = lot_multiplier / parser.original_lot_size
            parser.apply_lot_multiplier(ratio)
        
        result = parser.generate_output(
            risk_free_rate=rf_rate, 
            custom_balance=balance,
            trade_type=trade_type
        )
        
        # Update storage
        strategies_storage[name] = result
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/strategies/{name}")
async def delete_strategy(name: str):
    """Delete strategy from memory"""
    if name not in strategies_storage:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    del strategies_storage[name]
    del parsers_storage[name]
    
    return {"message": f"Strategy '{name}' deleted"}

@app.get("/strategies")
async def list_strategies():
    """List all uploaded strategies"""
    return {"strategies": list(strategies_storage.keys())}


# =============================================
# MT5 LIVE ACCOUNTS ENDPOINTS
# =============================================
from pydantic import BaseModel
from typing import Optional, List
import httpx
from datetime import datetime

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

        # 1. Trova l'utente tramite api_key
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={"api_key": f"eq.{data.api_key}", "select": "id"}
        )
        profiles = res.json()
        if not profiles:
            raise HTTPException(status_code=401, detail="API key non valida")

        user_id = profiles[0]["id"]

        # 2. Cerca se esiste già un record per questo conto
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
            # 3a. Aggiorna record esistente
            record_id = existing[0]["id"]
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/mt5_accounts",
                headers=supabase_headers(),
                params={"id": f"eq.{record_id}"},
                json=payload
            )
        else:
            # 3b. Crea nuovo record
            await client.post(
                f"{SUPABASE_URL}/rest/v1/mt5_accounts",
                headers=supabase_headers(),
                json=payload
            )

    # Salva snapshot storico
    snapshot_payload = {
        "user_id": user_id,
        "account_number": data.account_number,
        "currency": data.currency,
        "platform": data.platform,
        "balance": data.balance,
        "equity": data.equity,
    }
    async with httpx.AsyncClient() as client2:
        await client2.post(
            f"{SUPABASE_URL}/rest/v1/mt5_snapshots",
            headers=supabase_headers(),
            json=snapshot_payload
        )

    return {"status": "ok"}


@app.get("/mt5/accounts")
async def mt5_get_accounts(api_key: str = Query(...)):
    """Restituisce tutti i conti MT5 dell'utente"""

    async with httpx.AsyncClient() as client:

        # 1. Trova user_id tramite api_key
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=supabase_headers(),
            params={"api_key": f"eq.{api_key}", "select": "id"}
        )
        profiles = res.json()
        if not profiles:
            raise HTTPException(status_code=401, detail="API key non valida")

        user_id = profiles[0]["id"]

        # 2. Recupera tutti i conti
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/mt5_accounts",
            headers=supabase_headers(),
            params={"user_id": f"eq.{user_id}"}
        )
        accounts = res.json()

    return {"accounts": accounts}


@app.get("/mt5/snapshots")
async def mt5_get_snapshots(api_key: str = Query(...)):
    """Restituisce gli snapshot storici di tutti i conti dell'utente"""

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
            f"{SUPABASE_URL}/rest/v1/mt5_snapshots",
            headers=supabase_headers(),
            params={
                "user_id": f"eq.{user_id}",
                "order": "recorded_at.asc",
                "select": "account_number,currency,balance,equity,recorded_at",
                "limit": "10000"
            }
        )
        snapshots = res.json()

    return {"snapshots": snapshots}
