from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from strategy_parser import StrategyParser

app = FastAPI(title="Strategy Analyzer API")

# CORS per permettere al frontend di comunicare
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione: specifica domini esatti
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage in memoria delle strategie (in produzione: usa database)
strategies_storage = {}

@app.get("/")
def root():
    return {
        "message": "Strategy Analyzer API",
        "endpoints": {
            "/upload": "POST - Upload HTML strategy file",
            "/strategies": "GET - List all strategies",
            "/strategy/{name}": "GET - Get specific strategy data"
        }
    }

@app.post("/api/upload")
async def upload_strategy(file: UploadFile = File(...)):
    """
    Upload e parsing di un file HTML di strategia MetaTrader
    """
    # Validazione file
    if not file.filename.endswith('.html'):
        raise HTTPException(status_code=400, detail="File deve essere .html")
    
    try:
        # Crea file temporaneo
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.html') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Parsing con strategy_parser
        parser = StrategyParser(temp_path)
        result = parser.generate_output()
        
        # Usa il nome del file originale invece del path temporaneo
        result['strategy_name'] = file.filename.replace('.html', '')
        
        # Salva in memoria
        strategy_name = result['strategy_name']
        strategies_storage[strategy_name] = result
        
        # Rimuovi file temporaneo
        os.unlink(temp_path)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Strategia '{strategy_name}' caricata con successo",
            "data": result
        })
        
    except Exception as e:
        # Cleanup in caso di errore
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Errore processing file: {str(e)}")

@app.get("/api/strategies")
def list_strategies():
    """
    Lista tutte le strategie caricate
    """
    return JSONResponse(content={
        "success": True,
        "strategies": [
            {
                "name": name,
                "total_trades": data['metrics']['total_trades'],
                "win_rate": data['metrics']['win_rate_pct'],
                "total_profit": data['metrics']['total_profit'],
                "max_dd": data['metrics']['max_drawdown_pct']
            }
            for name, data in strategies_storage.items()
        ]
    })

@app.get("/api/strategy/{strategy_name}")
def get_strategy(strategy_name: str):
    """
    Ottieni dati completi di una strategia specifica
    """
    if strategy_name not in strategies_storage:
        raise HTTPException(status_code=404, detail=f"Strategia '{strategy_name}' non trovata")
    
    return JSONResponse(content={
        "success": True,
        "data": strategies_storage[strategy_name]
    })

@app.delete("/api/strategy/{strategy_name}")
def delete_strategy(strategy_name: str):
    """
    Elimina una strategia
    """
    if strategy_name not in strategies_storage:
        raise HTTPException(status_code=404, detail=f"Strategia '{strategy_name}' non trovata")
    
    del strategies_storage[strategy_name]
    
    return JSONResponse(content={
        "success": True,
        "message": f"Strategia '{strategy_name}' eliminata"
    })

if __name__ == "__main__":
    print("🚀 Starting Strategy Analyzer API...")
    print("📊 API disponibile su: http://localhost:8000")
    print("📚 Documentazione: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
