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
        
        # Parse
        parser = StrategyParser(temp_path)
        result = parser.generate_output(risk_free_rate=0)
        
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
async def get_strategy(name: str):
    """Get strategy data"""
    if name not in parsers_storage:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    try:
        parser = parsers_storage[name]
        result = parser.generate_output(risk_free_rate=0)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recalculate/{name}")
async def recalculate_metrics(
    name: str,
    rf_rate: float = Query(0, ge=0, le=10),
    balance: float = Query(None),
    lot_multiplier: float = Query(None)
):
    """Recalculate metrics with new parameters"""
    if name not in parsers_storage:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    try:
        parser = parsers_storage[name]
        
        # Apply lot multiplier if provided
        if lot_multiplier is not None and lot_multiplier != 1.0:
            for trade in parser.trades:
                if 'original_profit' not in trade:
                    trade['original_profit'] = trade['profit']
                    trade['original_commission'] = trade['commission']
                    trade['original_swap'] = trade['swap']
                
                trade['profit'] = trade['original_profit'] * lot_multiplier
                trade['commission'] = trade['original_commission'] * lot_multiplier
                trade['swap'] = trade['original_swap'] * lot_multiplier
        
        result = parser.generate_output(risk_free_rate=rf_rate, custom_balance=balance)
        
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
