#!/usr/bin/env python3
"""
🧠 TEST BRAIN CONTROLLER
Simple test to verify the brain controller can start
"""

import sys
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import threading
import time

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "VIPER MCP Brain Controller - TEST MODE"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Brain controller is working!"}

def start_server():
    """Start the server"""
    print("🚀 Starting test brain controller on port 8080...")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")

if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n🛑 Test brain controller stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
