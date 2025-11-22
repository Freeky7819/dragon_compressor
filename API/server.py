# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import uvicorn
import torch
from dragon.interface import Dragon

# Initialization (loads only once on startup)
app = FastAPI(title="Dragon Compressor API", version="1.0")
print("🐲 Loading Dragon model into memory...")
compressor = Dragon() 

class CompressRequest(BaseModel):
    text: Union[str, List[str]]
    ratio: int = 16

@app.post("/compress")
async def compress(request: CompressRequest):
    try:
        # Dragon accepts string or list
        result = compressor.compress(request.text, ratio=request.ratio)
        
        # Convert tensors to lists for JSON
        return {
            "vectors": result['compressed_vectors'].tolist(),
            "positions": result['positions'].tolist(),
            "ratio": request.ratio,
            "device": str(compressor.device)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "active", "model": "Dragon Pro 1:16"}

if __name__ == "__main__":
    # Starts server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)