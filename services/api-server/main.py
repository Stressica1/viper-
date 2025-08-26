#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - API Server
FastAPI web server providing dashboard and REST API endpoints

Port: 8000
Endpoints:
- GET / - Web dashboard
- GET /health - Health check
- GET /api/metrics - Performance metrics
- POST /api/backtest/start - Start backtest
- GET /api/risk/status - Risk assessment
"""

import os
import uvicorn
import requests
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
from pathlib import Path

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'api-server')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="VIPER Trading Bot",
    version="2.0.0",
    description="Ultra High-Performance Algorithmic Trading Platform",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Static files directory
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main web dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚀 VIPER Trading Bot Dashboard</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                text-align: center;
                padding: 50px 0;
            }
            .header h1 {
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .services-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .service-card {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: transform 0.3s ease;
            }
            .service-card:hover {
                transform: translateY(-5px);
            }
            .service-card h3 {
                margin-top: 0;
                color: #ffd700;
            }
            .status {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: bold;
            }
            .status.running { background: #4CAF50; }
            .status.stopped { background: #f44336; }
            .metrics {
                background: rgba(0, 0, 0, 0.2);
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
            }
            .metric {
                display: inline-block;
                margin: 10px;
                text-align: center;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #ffd700;
            }
            .metric-label {
                font-size: 0.9em;
                opacity: 0.8;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 VIPER Trading Bot</h1>
                <p>Ultra High-Performance Algorithmic Trading Platform</p>
            </div>

            <div class="metrics">
                <h2>📊 System Metrics</h2>
                <div class="metric">
                    <div class="metric-value" id="uptime">Loading...</div>
                    <div class="metric-label">Uptime</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="services">8/8</div>
                    <div class="metric-label">Services Online</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="win-rate">67%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
            </div>

            <div class="services-grid">
                <div class="service-card">
                    <h3>🌐 API Server</h3>
                    <p>Web dashboard & REST API endpoints</p>
                    <span class="status running">Running</span>
                    <br><small>Port: 8000</small>
                </div>

                <div class="service-card">
                    <h3>🧪 Ultra Backtester</h3>
                    <p>Strategy testing & validation engine</p>
                    <span class="status running">Running</span>
                    <br><small>Port: 8001</small>
                </div>

                <div class="service-card">
                    <h3>🎯 Strategy Optimizer</h3>
                    <p>Parameter tuning & optimization</p>
                    <span class="status running">Running</span>
                    <br><small>Port: 8004</small>
                </div>

                <div class="service-card">
                    <h3>🔥 Live Trading Engine</h3>
                    <p>Automated trading execution</p>
                    <span class="status running">Running</span>
                    <br><small>Port: 8007</small>
                </div>

                <div class="service-card">
                    <h3>💾 Data Manager</h3>
                    <p>Market data synchronization</p>
                    <span class="status running">Running</span>
                    <br><small>Port: 8003</small>
                </div>

                <div class="service-card">
                    <h3>🔗 Exchange Connector</h3>
                    <p>Bitget API integration</p>
                    <span class="status running">Running</span>
                    <br><small>Port: 8005</small>
                </div>

                <div class="service-card">
                    <h3>🚨 Risk Manager</h3>
                    <p>Safety & position control</p>
                    <span class="status running">Running</span>
                    <br><small>Port: 8002</small>
                </div>

                <div class="service-card">
                    <h3>📊 Monitoring Service</h3>
                    <p>System analytics & alerts</p>
                    <span class="status running">Running</span>
                    <br><small>Port: 8006</small>
                </div>
            </div>
        </div>

        <script>
            // Simple uptime counter
            let startTime = Date.now();
            function updateUptime() {
                let elapsed = Math.floor((Date.now() - startTime) / 1000);
                let hours = Math.floor(elapsed / 3600);
                let minutes = Math.floor((elapsed % 3600) / 60);
                let seconds = elapsed % 60;
                document.getElementById('uptime').textContent =
                    hours.toString().padStart(2, '0') + ':' +
                    minutes.toString().padStart(2, '0') + ':' +
                    seconds.toString().padStart(2, '0');
            }
            setInterval(updateUptime, 1000);
            updateUptime();
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "api-server",
        "version": "2.0.0",
        "uptime": "running"
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get system performance metrics"""
    return {
        "win_rate": 67.0,
        "total_trades": 1250,
        "active_positions": 3,
        "daily_pnl": 2.34,
        "risk_score": 85,
        "services_online": 8,
        "last_update": "2024-01-01T12:00:00Z"
    }

@app.post("/api/backtest/start")
async def start_backtest(request: Request):
    """Start a backtest with given parameters"""
    data = await request.json()
    logger.info(f"Starting backtest with parameters: {data}")

    try:
        # Make request to ultra-backtester service
        ultra_backtester_url = os.getenv('ULTRA_BACKTESTER_URL', 'http://ultra-backtester:8000')

        response = requests.post(
            f"{ultra_backtester_url}/api/backtest/start",
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"Backtest started successfully: {result}")
            return result
        else:
            logger.error(f"Failed to start backtest: {response.status_code} - {response.text}")
            return {
                "status": "error",
                "message": f"Failed to start backtest: {response.status_code}",
                "details": response.text
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to ultra-backtester service: {e}")
        return {
            "status": "error",
            "message": "Unable to connect to backtester service",
            "details": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error starting backtest: {e}")
        return {
            "status": "error",
            "message": "Unexpected error occurred",
            "details": str(e)
        }

@app.get("/api/risk/status")
async def get_risk_status():
    """Get current risk management status"""
    return {
        "overall_risk": "low",
        "daily_loss_limit": 0.03,
        "current_daily_loss": 0.012,
        "max_position_size": 0.1,
        "active_stops": 3,
        "risk_score": 85
    }

@app.get("/api/services/status")
async def get_services_status():
    """Get status of all microservices"""
    services = {
        "api-server": {"status": "running", "port": 8000},
        "ultra-backtester": {"status": "running", "port": 8001},
        "strategy-optimizer": {"status": "running", "port": 8004},
        "live-trading-engine": {"status": "running", "port": 8007},
        "data-manager": {"status": "running", "port": 8003},
        "exchange-connector": {"status": "running", "port": 8005},
        "risk-manager": {"status": "running", "port": 8002},
        "monitoring-service": {"status": "running", "port": 8006}
    }
    return services

if __name__ == "__main__":
    port = int(os.getenv("API_SERVER_PORT", 8000))
    logger.info(f"Starting VIPER API Server on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
