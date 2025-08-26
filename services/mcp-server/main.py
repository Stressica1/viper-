#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - MCP Server
Model Context Protocol server for AI-powered trading operations

Features:
- Standardized AI integration interface
- Secure API key management
- Trading operations via MCP protocol
- Real-time market data access
- Risk management controls
- GitHub Project Management Integration
"""

import os
import json
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse
import uvicorn
import redis
import requests
import httpx
import base64

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'mcp-server')
VAULT_URL = os.getenv('VAULT_URL', 'http://credential-vault:8008')
VAULT_ACCESS_TOKEN = os.getenv('VAULT_ACCESS_TOKEN', '')

# GitHub Configuration
GITHUB_PAT = os.getenv('GITHUB_PAT', '')
GITHUB_OWNER = os.getenv('GITHUB_OWNER', 'user')  # Replace with actual GitHub username/org
GITHUB_REPO = os.getenv('GITHUB_REPO', 'Bitget-New')  # Replace with actual repo name

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger(__name__)

class MCPServer:
    """MCP Server for VIPER trading operations"""

    def __init__(self):
        self.redis_client = None
        self.app = FastAPI(title="VIPER MCP Server", version="1.0.0")
        self.service_urls = {
            'api-server': 'http://api-server:8000',
            'ultra-backtester': 'http://ultra-backtester:8000',
            'risk-manager': 'http://risk-manager:8000',
            'data-manager': 'http://data-manager:8000',
            'exchange-connector': 'http://exchange-connector:8000',
            'signal-processor': 'http://signal-processor:8000',
            'alert-system': 'http://alert-system:8000',
            'order-lifecycle-manager': 'http://order-lifecycle-manager:8000',
            'position-synchronizer': 'http://position-synchronizer:8000'
        }

        self.setup_routes()

    def setup_routes(self):
        """Setup MCP protocol routes"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "service": "mcp-server",
                "version": "1.0.0",
                "capabilities": self.get_capabilities()
            }

        @self.app.get("/capabilities")
        async def get_capabilities():
            """Get MCP server capabilities"""
            return self.get_capabilities()

        @self.app.post("/trading/start")
        async def start_trading(request: Request):
            """Start automated trading via MCP"""
            data = await request.json()
            return await self.handle_trading_operation("start", data)

        @self.app.post("/trading/stop")
        async def stop_trading(request: Request):
            """Stop automated trading via MCP"""
            data = await request.json()
            return await self.handle_trading_operation("stop", data)

        @self.app.post("/backtest/run")
        async def run_backtest(request: Request):
            """Run backtest via MCP"""
            data = await request.json()
            return await self.handle_backtest_operation(data)

        @self.app.get("/market/data")
        async def get_market_data(symbol: str = "BTC/USDT:USDT"):
            """Get real-time market data via MCP"""
            return await self.handle_market_data_request(symbol)

        @self.app.get("/portfolio/status")
        async def get_portfolio_status():
            """Get current portfolio status via MCP"""
            return await self.handle_portfolio_status()

        @self.app.post("/risk/assess")
        async def assess_risk(request: Request):
            """Assess trading risk via MCP"""
            data = await request.json()
            return await self.handle_risk_assessment(data)

        # GitHub Integration Endpoints
        @self.app.post("/github/create-task")
        async def create_github_task(request: Request):
            """Create a GitHub task/issue via MCP"""
            data = await request.json()
            return await self.create_github_issue(data)

        @self.app.get("/github/tasks")
        async def list_github_tasks(status: str = "open"):
            """List GitHub tasks/issues via MCP"""
            return await self.list_github_issues(status)

        @self.app.post("/github/update-task")
        async def update_github_task(request: Request):
            """Update a GitHub task/issue via MCP"""
            data = await request.json()
            return await self.update_github_issue(data)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time MCP communication"""
            await websocket.accept()
            while True:
                try:
                    data = await websocket.receive_json()
                    response = await self.handle_websocket_message(data)
                    await websocket.send_json(response)
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break

        @self.app.get("/sse")
        async def sse_endpoint():
            """Server-Sent Events endpoint for MCP streaming"""
            async def generate():
                while True:
                    # Stream real-time trading updates
                    update = await self.get_real_time_update()
                    yield f"data: {json.dumps(update)}\n\n"
                    await asyncio.sleep(1)

            return StreamingResponse(generate(), media_type="text/event-stream")

    def get_capabilities(self) -> Dict[str, Any]:
        """Get MCP server capabilities"""
        return {
            "trading": {
                "start_trading": "Start automated trading",
                "stop_trading": "Stop automated trading",
                "get_portfolio": "Get portfolio status",
                "assess_risk": "Assess trading risk"
            },
            "backtesting": {
                "run_backtest": "Execute backtest with parameters",
                "get_results": "Get backtest results"
            },
            "market_data": {
                "get_ticker": "Get real-time ticker data",
                "get_ohlcv": "Get OHLCV data"
            },
            "risk_management": {
                "check_limits": "Check risk limits",
                "get_positions": "Get current positions",
                "calculate_sizing": "Calculate position sizing"
            },
            "monitoring": {
                "get_metrics": "Get system metrics",
                "get_alerts": "Get active alerts",
                "system_status": "Get system health status"
            }
        }

    async def handle_trading_operation(self, operation: str, data: Dict) -> Dict[str, Any]:
        """Handle trading operations via MCP"""
        try:
            if operation == "start":
                # Start live trading engine
                response = await self.call_service(
                    'api-server',
                    '/api/trading/start',
                    method='POST',
                    data=data
                )
                return {
                    "status": "success",
                    "operation": "start_trading",
                    "result": response,
                    "message": "Trading started successfully"
                }

            elif operation == "stop":
                # Stop live trading engine
                response = await self.call_service(
                    'api-server',
                    '/api/trading/stop',
                    method='POST',
                    data=data
                )
                return {
                    "status": "success",
                    "operation": "stop_trading",
                    "result": response,
                    "message": "Trading stopped successfully"
                }

        except Exception as e:
            logger.error(f"Trading operation error: {e}")
            return {
                "status": "error",
                "operation": operation,
                "error": str(e)
            }

    async def handle_backtest_operation(self, data: Dict) -> Dict[str, Any]:
        """Handle backtest operations via MCP"""
        try:
            # Call ultra-backtester service
            response = await self.call_service(
                'api-server',
                '/api/backtest/start',
                method='POST',
                data=data
            )

            return {
                "status": "success",
                "operation": "run_backtest",
                "result": response,
                "message": "Backtest started successfully"
            }

        except Exception as e:
            logger.error(f"Backtest operation error: {e}")
            return {
                "status": "error",
                "operation": "run_backtest",
                "error": str(e)
            }

    async def handle_market_data_request(self, symbol: str) -> Dict[str, Any]:
        """Handle market data requests via MCP"""
        try:
            # Get ticker data
            ticker_response = await self.call_service(
                'data-manager',
                f'/api/ticker/{symbol}',
                method='GET'
            )

            # Get OHLCV data
            ohlcv_response = await self.call_service(
                'data-manager',
                f'/api/ohlcv/{symbol}',
                method='GET',
                params={'timeframe': '1h', 'limit': 100}
            )

            return {
                "status": "success",
                "symbol": symbol,
                "ticker": ticker_response,
                "ohlcv": ohlcv_response,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Market data request error: {e}")
            return {
                "status": "error",
                "symbol": symbol,
                "error": str(e)
            }

    async def handle_portfolio_status(self) -> Dict[str, Any]:
        """Handle portfolio status requests via MCP"""
        try:
            # Get account balance
            balance_response = await self.call_service(
                'exchange-connector',
                '/api/balance',
                method='GET'
            )

            # Get positions
            positions_response = await self.call_service(
                'exchange-connector',
                '/api/positions',
                method='GET'
            )

            # Get risk status
            risk_response = await self.call_service(
                'risk-manager',
                '/api/risk/status',
                method='GET'
            )

            return {
                "status": "success",
                "balance": balance_response,
                "positions": positions_response.get('positions', []),
                "risk_status": risk_response,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Portfolio status error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def handle_risk_assessment(self, data: Dict) -> Dict[str, Any]:
        """Handle risk assessment requests via MCP"""
        try:
            response = await self.call_service(
                'risk-manager',
                '/api/position/check',
                method='POST',
                data=data
            )

            return {
                "status": "success",
                "assessment": response,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def handle_websocket_message(self, data: Dict) -> Dict[str, Any]:
        """Handle WebSocket messages"""
        message_type = data.get('type', 'unknown')

        if message_type == 'subscribe_trades':
            # Subscribe to real-time trading updates
            return await self.handle_trading_subscription(data)

        elif message_type == 'get_signals':
            # Get trading signals
            return await self.handle_signal_request(data)

        elif message_type == 'execute_trade':
            # Execute trade
            return await self.handle_trade_execution(data)

        else:
            return {
                "status": "error",
                "message": f"Unknown message type: {message_type}"
            }

    async def handle_trading_subscription(self, data: Dict) -> Dict[str, Any]:
        """Handle trading subscription"""
        # This would set up real-time trading updates
        return {
            "status": "success",
            "message": "Subscribed to trading updates",
            "subscription_id": str(uuid.uuid4())
        }

    async def handle_signal_request(self, data: Dict) -> Dict[str, Any]:
        """Handle signal requests"""
        try:
            response = await self.call_service(
                'signal-processor',
                '/api/signals',
                method='GET',
                params=data
            )

            return {
                "status": "success",
                "signals": response
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def handle_trade_execution(self, data: Dict) -> Dict[str, Any]:
        """Handle trade execution"""
        try:
            response = await self.call_service(
                'order-lifecycle-manager',
                '/api/orders',
                method='POST',
                data=data
            )

            return {
                "status": "success",
                "execution": response
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def get_real_time_update(self) -> Dict[str, Any]:
        """Get real-time system update for SSE"""
        try:
            # Get quick status from multiple services
            status_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "system_update"
            }

            # Add basic health status
            status_data["healthy"] = True
            status_data["services"] = len(self.service_urls)

            return status_data

        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "type": "error",
                "error": str(e)
            }

    async def call_service(self, service_name: str, endpoint: str,
                          method: str = 'GET', data: Dict = None,
                          params: Dict = None) -> Dict[str, Any]:
        """Call a VIPER microservice"""
        try:
            base_url = self.service_urls.get(service_name)
            if not base_url:
                raise HTTPException(status_code=404, detail=f"Service {service_name} not found")

            url = f"{base_url}{endpoint}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                if method.upper() == 'GET':
                    response = await client.get(url, params=params)
                elif method.upper() == 'POST':
                    response = await client.post(url, json=data)
                elif method.upper() == 'PUT':
                    response = await client.put(url, json=data)
                elif method.upper() == 'DELETE':
                    response = await client.delete(url, params=params)
                else:
                    raise HTTPException(status_code=405, detail=f"Method {method} not allowed")

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Service {service_name} returned {response.status_code}: {response.text}")
                    return {"error": f"HTTP {response.status_code}", "details": response.text}

        except Exception as e:
            logger.error(f"Service call error for {service_name}: {e}")
            raise HTTPException(status_code=503, detail=f"Service {service_name} unavailable")

    # GitHub Integration Methods
    async def create_github_issue(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a GitHub issue via MCP"""
        try:
            if not GITHUB_PAT:
                return {"error": "GitHub PAT not configured", "status": "error"}

            headers = {
                "Authorization": f"token {GITHUB_PAT}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json"
            }

            issue_data = {
                "title": data.get("title", "New Task"),
                "body": data.get("body", ""),
                "labels": data.get("labels", ["task"]),
                "assignees": data.get("assignees", [])
            }

            url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/issues"
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=issue_data, headers=headers)

            if response.status_code == 201:
                issue = response.json()
                logger.info(f"GitHub issue created: #{issue['number']} - {issue['title']}")
                return {
                    "status": "success",
                    "operation": "create_github_issue",
                    "issue": {
                        "number": issue["number"],
                        "title": issue["title"],
                        "url": issue["html_url"],
                        "state": issue["state"]
                    }
                }
            else:
                error_msg = response.text
                logger.error(f"GitHub API error: {response.status_code} - {error_msg}")
                return {"status": "error", "error": error_msg}

        except Exception as e:
            logger.error(f"GitHub issue creation error: {e}")
            return {"status": "error", "error": str(e)}

    async def list_github_issues(self, status: str = "open") -> Dict[str, Any]:
        """List GitHub issues via MCP"""
        try:
            if not GITHUB_PAT:
                return {"error": "GitHub PAT not configured", "status": "error"}

            headers = {
                "Authorization": f"token {GITHUB_PAT}",
                "Accept": "application/vnd.github.v3+json"
            }

            url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/issues"
            params = {"state": status, "per_page": 10}

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                issues = response.json()
                return {
                    "status": "success",
                    "operation": "list_github_issues",
                    "issues": [
                        {
                            "number": issue["number"],
                            "title": issue["title"],
                            "state": issue["state"],
                            "url": issue["html_url"],
                            "created_at": issue["created_at"]
                        }
                        for issue in issues
                    ]
                }
            else:
                return {"status": "error", "error": response.text}

        except Exception as e:
            logger.error(f"GitHub issues listing error: {e}")
            return {"status": "error", "error": str(e)}

    async def update_github_issue(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a GitHub issue via MCP"""
        try:
            if not GITHUB_PAT:
                return {"error": "GitHub PAT not configured", "status": "error"}

            issue_number = data.get("issue_number")
            if not issue_number:
                return {"error": "Issue number required", "status": "error"}

            headers = {
                "Authorization": f"token {GITHUB_PAT}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json"
            }

            update_data = {}
            if "title" in data:
                update_data["title"] = data["title"]
            if "body" in data:
                update_data["body"] = data["body"]
            if "state" in data:
                update_data["state"] = data["state"]
            if "labels" in data:
                update_data["labels"] = data["labels"]

            url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/issues/{issue_number}"
            async with httpx.AsyncClient() as client:
                response = await client.patch(url, json=update_data, headers=headers)

            if response.status_code == 200:
                issue = response.json()
                logger.info(f"GitHub issue updated: #{issue['number']} - {issue['title']}")
                return {
                    "status": "success",
                    "operation": "update_github_issue",
                    "issue": {
                        "number": issue["number"],
                        "title": issue["title"],
                        "url": issue["html_url"],
                        "state": issue["state"]
                    }
                }
            else:
                return {"status": "error", "error": response.text}

        except Exception as e:
            logger.error(f"GitHub issue update error: {e}")
            return {"status": "error", "error": str(e)}

def main():
    """Main function to start MCP server"""
    server = MCPServer()

    # Get port from environment
    port = int(os.getenv('MCP_SERVER_PORT', '8015'))

    logger.info(f"🚀 Starting VIPER MCP Server on port {port}")
    logger.info("📋 MCP Capabilities:")
    capabilities = server.get_capabilities()
    for category, functions in capabilities.items():
        logger.info(f"  {category.upper()}: {len(functions)} functions")

    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )

if __name__ == "__main__":
    main()
