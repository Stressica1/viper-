#!/usr/bin/env python3
"""
🚀 VIPER MCP BRAIN CONTROLLER - SIMPLE VERSION
A working brain controller that actually starts and serves the web interface
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="VIPER MCP Brain Controller", version="1.0.0")

class SimpleBrainController:
    """Simplified brain controller that works"""

    def __init__(self):
        self.start_time = datetime.now()
        self.command_count = 0
        self.system_status = {
            "brain_active": True,
            "trading_active": False,
            "ai_integration": False,
            "rules_enforced": True
        }
        logger.info("🧠 Simple Brain Controller initialized")

    def get_system_health(self) -> dict:
        """Get system health status"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        return {
            "status": "healthy" if cpu_percent < 80 else "warning",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).seconds,
            "system": {
                "cpu_usage": f"{cpu_percent:.1f}%",
                "memory_usage": f"{memory.percent:.1f}%",
                "brain_active": self.system_status["brain_active"]
            },
            "trading": {
                "active_positions": 0,
                "pnl": "+0.00",
                "active_strategy": "IDLE"
            },
            "ai": {
                "cursor_integration": self.system_status["ai_integration"],
                "commands_processed": self.command_count
            }
        }

    def execute_command(self, command: str, params: dict = None) -> dict:
        """Execute a command"""
        if params is None:
            params = {}

        self.command_count += 1

        logger.info(f"Executing command: {command} with params: {params}")

        # Handle different commands
        if command == "start_trading":
            self.system_status["trading_active"] = True
            return {"status": "success", "message": "Trading started", "strategy": params.get("strategy", "VIPER")}

        elif command == "stop_trading":
            self.system_status["trading_active"] = False
            return {"status": "success", "message": "Trading stopped"}

        elif command == "get_portfolio":
            return {
                "status": "success",
                "portfolio": {
                    "total_balance": "1000.00 USDT",
                    "positions": [],
                    "pnl": "+0.00"
                }
            }

        elif command == "analyze_market":
            return {
                "status": "success",
                "analysis": {
                    "symbol": params.get("symbol", "BTC/USDT"),
                    "signal": "HOLD",
                    "confidence": 75,
                    "indicators": {
                        "rsi": 55,
                        "macd": "neutral",
                        "bollinger": "middle"
                    }
                }
            }

        elif command == "emergency_stop":
            self.system_status["trading_active"] = False
            return {"status": "success", "message": "EMERGENCY STOP activated"}

        else:
            return {"status": "error", "message": f"Unknown command: {command}"}

# Create brain controller instance
brain = SimpleBrainController()

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🧠 VIPER MCP Brain Controller</title>
        <style>
            body {{
                font-family: 'Courier New', monospace;
                background: #0a0a0a;
                color: #00ff00;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #00ff00;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .status-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .status-card {{
                background: #1a1a1a;
                border: 1px solid #00ff00;
                padding: 20px;
                border-radius: 5px;
            }}
            .menu-section {{
                background: #1a1a1a;
                border: 1px solid #00ff00;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .command-btn {{
                background: #00ff00;
                color: #000;
                border: none;
                padding: 10px 15px;
                margin: 5px;
                cursor: pointer;
                border-radius: 3px;
                font-weight: bold;
            }}
            .command-btn:hover {{
                background: #00cc00;
            }}
            .status-healthy {{ color: #00ff00; }}
            .status-warning {{ color: #ffff00; }}
            .status-error {{ color: #ff0000; }}
            .log-area {{
                background: #000;
                border: 1px solid #333;
                padding: 10px;
                height: 200px;
                overflow-y: auto;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 VIPER MCP BRAIN CONTROLLER</h1>
                <p>The Central Intelligence System - RUNNING</p>
                <div id="brain-status">🟢 BRAIN ACTIVE - Uptime: <span id="uptime">--</span></div>
            </div>

            <div class="status-grid">
                <div class="status-card">
                    <h3>🖥️ System Status</h3>
                    <div id="system-status">
                        Loading...
                    </div>
                </div>

                <div class="status-card">
                    <h3>💰 Trading Status</h3>
                    <div id="trading-status">
                        Loading...
                    </div>
                </div>

                <div class="status-card">
                    <h3>🤖 AI Status</h3>
                    <div id="ai-status">
                        Loading...
                    </div>
                </div>
            </div>

            <div class="menu-section">
                <h3>🎮 Command Menu</h3>
                <button class="command-btn" onclick="executeCommand('start_trading', {strategy: 'viper'})">
                    Start VIPER Trading
                </button>
                <button class="command-btn" onclick="executeCommand('stop_trading')">
                    Stop Trading
                </button>
                <button class="command-btn" onclick="executeCommand('get_portfolio')">
                    Get Portfolio
                </button>
                <button class="command-btn" onclick="executeCommand('analyze_market', {symbol: 'BTC/USDT'})">
                    Analyze Market
                </button>
                <button class="command-btn" onclick="executeCommand('emergency_stop')">
                    Emergency Stop
                </button>
            </div>

            <div class="menu-section">
                <h3>📋 Command Log</h3>
                <div class="log-area" id="command-log">
                    System ready. Commands will appear here.
                </div>
            </div>
        </div>

        <script>
            // Update status every 5 seconds
            setInterval(async () => {{
                try {{
                    const response = await fetch('/health');
                    const health = await response.json();

                    // Update system status
                    document.getElementById('system-status').innerHTML = `
                        <p>CPU: <span class="${health.system.cpu_usage > '70%' ? 'status-error' : 'status-healthy'}">${health.system.cpu_usage}</span></p>
                        <p>Memory: <span class="${health.system.memory_usage > '80%' ? 'status-error' : 'status-healthy'}">${health.system.memory_usage}</span></p>
                        <p>Status: <span class="status-healthy">${health.status.toUpperCase()}</span></p>
                    `;

                    // Update trading status
                    document.getElementById('trading-status').innerHTML = `
                        <p>Active Positions: ${health.trading.active_positions}</p>
                        <p>P&L: ${health.trading.pnl}</p>
                        <p>Strategy: ${health.trading.active_strategy}</p>
                    `;

                    // Update AI status
                    document.getElementById('ai-status').innerHTML = `
                        <p>Cursor Integration: ${health.ai.cursor_integration ? 'ACTIVE' : 'INACTIVE'}</p>
                        <p>Commands Processed: ${health.ai.commands_processed}</p>
                        <p>Brain Active: ${health.system.brain_active ? 'YES' : 'NO'}</p>
                    `;

                    // Update uptime
                    document.getElementById('uptime').textContent = `${health.uptime_seconds}s`;

                }} catch (error) {{
                    console.error('Health check failed:', error);
                }}
            }}, 5000);

            async function executeCommand(command, params = {{}}) {{
                try {{
                    const response = await fetch('/command', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ command, params }})
                    }});
                    const result = await response.json();

                    // Log the command
                    const log = document.getElementById('command-log');
                    const timestamp = new Date().toLocaleTimeString();
                    log.innerHTML += `\\n[${timestamp}] ${command}: ${JSON.stringify(result)}`;
                    log.scrollTop = log.scrollHeight;

                    // Show result
                    alert(`Command Result: ${JSON.stringify(result, null, 2)}`);

                }} catch (error) {{
                    alert(`Command failed: ${error.message}`);
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return brain.get_system_health()

@app.post("/command")
async def execute_command(request: Request):
    """Execute commands"""
    data = await request.json()
    command = data.get("command", "")
    params = data.get("params", {})

    return brain.execute_command(command, params)

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "brain_controller": "running",
        "version": "1.0.0",
        "uptime": (datetime.now() - brain.start_time).seconds,
        "timestamp": datetime.now().isoformat()
    }

def main():
    """Start the brain controller"""
    logger.info("🚀 Starting VIPER MCP Brain Controller...")
    logger.info("📊 Dashboard will be available at: http://localhost:8080")

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_level="info",
            reload=False
        )
    except Exception as e:
        logger.error(f"❌ Failed to start brain controller: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
