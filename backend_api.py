from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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
            "/strategy/{name}": "GET - Get specific strategy data",
            "/recalculate/{name}": "GET - Recalculate with custom risk-free rate"
        }
    }

@app.post("/api/upload")
async def upload_strategy(file: UploadFile = File(...), risk_free_rate: float = Query(0, description="Risk-free rate in %")):
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
        result = parser.generate_output(risk_free_rate=risk_free_rate)
        
        # Usa il nome del file originale invece del path temporaneo
        result['strategy_name'] = file.filename.replace('.html', '')
        
        # Salva in memoria
        strategy_name = result['strategy_name']
        
        # Salva anche il file path temporaneo per ricalcoli futuri
        # In realtà salviamo i dati raw per permettere ricalcoli
        strategies_storage[strategy_name] = {
            'data': result,
            'file_content': content  # Salva contenuto per ricalcoli
        }
        
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
                "total_trades": data['data']['metrics']['total_trades'],
                "win_rate": data['data']['metrics']['win_rate_pct'],
                "total_profit": data['data']['metrics']['total_profit'],
                "max_dd": data['data']['metrics']['max_drawdown_pct']
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
        "data": strategies_storage[strategy_name]['data']
    })

@app.get("/api/recalculate/{strategy_name}")
def recalculate_strategy(strategy_name: str, risk_free_rate: float = Query(0, description="Risk-free rate in %")):
    """
    Ricalcola una strategia esistente con un nuovo risk-free rate
    """
    if strategy_name not in strategies_storage:
        raise HTTPException(status_code=404, detail=f"Strategia '{strategy_name}' non trovata")
    
    try:
        # Ricrea file temporaneo dal contenuto salvato
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.html') as temp_file:
            temp_file.write(strategies_storage[strategy_name]['file_content'])
            temp_path = temp_file.name
        
        # Ricalcola con nuovo risk-free rate
        parser = StrategyParser(temp_path)
        result = parser.generate_output(risk_free_rate=risk_free_rate)
        result['strategy_name'] = strategy_name
        
        # Aggiorna storage (mantieni file_content)
        strategies_storage[strategy_name]['data'] = result
        
        # Cleanup
        os.unlink(temp_path)
        
        return JSONResponse(content={
            "success": True,
            "data": result
        })
        
    except Exception as e:
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Errore ricalcolo: {str(e)}")

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
