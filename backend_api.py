from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from pathlib import Path
from strategy_parser_v2 import StrategyParser

app = FastAPI(title="Strategy Analyzer API")

# CORS configuration - permetti tutto
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permetti tutti i domini
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage in memoria per strategie (reset al riavvio)
strategies_storage = {}
parsers_storage = {}

# Directory temporanea per file HTML caricati
TEMP_DIR = Path(tempfile.gettempdir()) / "strategy_analyzer"
TEMP_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "strategies_count": len(strategies_storage),
        "strategies": list(strategies_storage.keys())
    }


@app.post("/upload")
async def upload_strategy(file: UploadFile = File(...)):
    """
    Upload e analizza file HTML strategia MetaTrader
    """
    try:
        # Verifica che sia un file HTML
        if not file.filename.endswith('.html'):
            raise HTTPException(status_code=400, detail="File must be .html")
        
        # Salva file temporaneamente
        temp_path = TEMP_DIR / file.filename
        with open(temp_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Parse strategia
        parser = StrategyParser(str(temp_path))
        result = parser.generate_output(risk_free_rate=0)
        
        strategy_name = result['strategy_name']
        
        # Salva in memoria
        strategies_storage[strategy_name] = result
        parsers_storage[strategy_name] = parser
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/strategies")
async def list_strategies():
    """
    Lista tutte le strategie caricate
    """
    return {
        "strategies": list(strategies_storage.keys()),
        "count": len(strategies_storage)
    }


@app.get("/strategies/{strategy_name}")
async def get_strategy(strategy_name: str):
    """
    Ottieni dati di una strategia specifica
    """
    if strategy_name not in strategies_storage:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    return strategies_storage[strategy_name]


@app.delete("/strategies/{strategy_name}")
async def delete_strategy(strategy_name: str):
    """
    Elimina una strategia
    """
    if strategy_name not in strategies_storage:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Rimuovi da storage
    del strategies_storage[strategy_name]
    del parsers_storage[strategy_name]
    
    # Rimuovi file temporaneo se esiste
    temp_path = TEMP_DIR / f"{strategy_name}.html"
    if temp_path.exists():
        temp_path.unlink()
    
    return {"message": f"Strategy '{strategy_name}' deleted successfully"}


@app.get("/recalculate/{strategy_name}")
async def recalculate_strategy(
    strategy_name: str,
    rf_rate: float = 0,
    balance: float = None,
    lot_multiplier: float = None
):
    """
    Ricalcola metriche con nuovi parametri
    
    Args:
        strategy_name: Nome della strategia
        rf_rate: Risk-free rate (%)
        balance: Nuovo balance iniziale (opzionale)
        lot_multiplier: Moltiplicatore lottaggio (opzionale)
    """
    if strategy_name not in parsers_storage:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    try:
        parser = parsers_storage[strategy_name]
        
        # Caso 1: Solo balance modificato
        if balance is not None and lot_multiplier is None:
            result = parser.generate_output(risk_free_rate=rf_rate, custom_balance=balance)
        
        # Caso 2: Solo lottaggio modificato
        elif balance is None and lot_multiplier is not None and lot_multiplier != 1:
            # Verifica che lottaggio sia uniforme
            lot_check = parser.check_uniform_lot_size()
            if not lot_check['is_uniform']:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot change lot size: trades have different lot sizes"
                )
            result = parser.recalculate_with_lot_multiplier(lot_multiplier)
            
            # Applica anche rf_rate se diverso da 0
            if rf_rate != 0:
                result = parser.generate_output(risk_free_rate=rf_rate)
        
        # Caso 3: Entrambi modificati
        elif balance is not None and lot_multiplier is not None and lot_multiplier != 1:
            # Prima applica lottaggio
            lot_check = parser.check_uniform_lot_size()
            if not lot_check['is_uniform']:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot change lot size: trades have different lot sizes"
                )
            
            # Moltiplica profit/comm/swap
            for trade in parser.trades:
                trade['profit'] = trade['profit'] * lot_multiplier
                trade['commission'] = trade['commission'] * lot_multiplier
                trade['swap'] = trade['swap'] * lot_multiplier
                trade['volume'] = trade['volume'] * lot_multiplier
            
            # Ricalcola balance
            parser._recalculate_balance()
            
            # Poi applica custom balance e rf_rate
            result = parser.generate_output(risk_free_rate=rf_rate, custom_balance=balance)
        
        # Caso 4: Solo rf_rate modificato
        else:
            result = parser.generate_output(risk_free_rate=rf_rate, custom_balance=balance)
        
        # Aggiorna storage
        strategies_storage[strategy_name] = result
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recalculation error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "temp_dir": str(TEMP_DIR),
        "strategies_loaded": len(strategies_storage)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
