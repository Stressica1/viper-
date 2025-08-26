#!/usr/bin/env python3
"""
🚀 VIPER MCP Client Library
Python client for interacting with VIPER Trading Bot via MCP protocol

Usage:
    from viper_mcp_client import VIPERMCPClient

    client = VIPERMCPClient()
    result = client.start_trading({"symbol": "BTC/USDT:USDT"})
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

class VIPERMCPClient:
    """
    VIPER MCP Client for AI agents and external applications

    Provides a simple, Pythonic interface to all VIPER trading operations
    via the Model Context Protocol (MCP).
    """

    def __init__(self, base_url: str = "http://localhost:8015", timeout: int = 30):
        """
        Initialize MCP client

        Args:
            base_url: MCP server URL (default: http://localhost:8015)
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self._last_health_check = 0
        self._health_cache_duration = 60  # Cache health for 60 seconds

    def _make_request(self, method: str, endpoint: str,
                     data: Optional[Dict] = None,
                     params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to MCP server

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data for POST/PUT
            params: Query parameters for GET

        Returns:
            Response data as dictionary

        Raises:
            ConnectionError: If server is unreachable
            TimeoutError: If request times out
            ValueError: If response is invalid
        """
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=self.timeout)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data, timeout=self.timeout)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, params=params, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            if response.content:
                return response.json()
            else:
                return {"status": "success"}

        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to VIPER MCP server at {url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            error_data = {"status": "error", "http_error": str(e)}
            try:
                error_details = e.response.json()
                error_data.update(error_details)
            except:
                error_data["response_text"] = e.response.text
            return error_data
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get MCP server capabilities and available functions

        Returns:
            Dictionary of available capabilities organized by category
        """
        return self._make_request('GET', '/capabilities')

    def get_health(self) -> Dict[str, Any]:
        """
        Get MCP server health status

        Returns:
            Health status with service information
        """
        return self._make_request('GET', '/health')

    def is_healthy(self) -> bool:
        """
        Check if MCP server is healthy and responding

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            health = self.get_health()
            return health.get('status') == 'healthy'
        except:
            return False

    # ============================================================================
    # TRADING OPERATIONS
    # ============================================================================

    def start_trading(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start automated trading

        Args:
            config: Trading configuration with parameters like:
                - symbol: Trading pair (e.g., "BTC/USDT:USDT")
                - strategy: Strategy name (e.g., "VIPER")
                - risk_per_trade: Risk per trade (0.01 = 1%)
                - amount: Position size
                - max_positions: Maximum concurrent positions

        Returns:
            Operation result with status and details
        """
        return self._make_request('POST', '/trading/start', data=config)

    def stop_trading(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Stop automated trading

        Args:
            config: Optional stop configuration

        Returns:
            Operation result with status and details
        """
        return self._make_request('POST', '/trading/stop', data=config or {})

    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get current portfolio status

        Returns:
            Portfolio information including:
            - Balance (total, free, used)
            - Positions (current holdings)
            - Risk metrics (daily P&L, exposure)
            - Performance statistics
        """
        return self._make_request('GET', '/portfolio/status')

    # ============================================================================
    # BACKTESTING OPERATIONS
    # ============================================================================

    def run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute backtest with given parameters

        Args:
            params: Backtest parameters including:
                - symbol: Trading pair
                - start_date: Start date (YYYY-MM-DD)
                - end_date: End date (YYYY-MM-DD)
                - initial_balance: Starting balance
                - risk_per_trade: Risk per trade
                - strategy: Strategy name
                - timeframe: Chart timeframe

        Returns:
            Backtest results with performance metrics
        """
        return self._make_request('POST', '/backtest/run', data=params)

    # ============================================================================
    # MARKET DATA OPERATIONS
    # ============================================================================

    def get_market_data(self, symbol: str = "BTC/USDT:USDT") -> Dict[str, Any]:
        """
        Get comprehensive market data for symbol

        Args:
            symbol: Trading pair symbol

        Returns:
            Market data including:
            - Real-time ticker
            - OHLCV data
            - Order book snapshot
            - Recent trades
        """
        return self._make_request('GET', '/market/data', params={"symbol": symbol})

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time ticker data

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data with price, volume, spread, etc.
        """
        return self._make_request('GET', f'/market/ticker/{symbol}')

    def get_ohlcv(self, symbol: str, timeframe: str = "1h",
                  limit: int = 100) -> Dict[str, Any]:
        """
        Get OHLCV (candlestick) data

        Args:
            symbol: Trading pair symbol
            timeframe: Chart timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to retrieve

        Returns:
            OHLCV data array with timestamp, open, high, low, close, volume
        """
        return self._make_request('GET', f'/market/ohlcv/{symbol}',
                                params={"timeframe": timeframe, "limit": limit})

    # ============================================================================
    # RISK MANAGEMENT OPERATIONS
    # ============================================================================

    def assess_risk(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk for a potential trade

        Args:
            trade_data: Trade parameters including:
                - symbol: Trading pair
                - amount: Position size
                - price: Entry price
                - side: Buy or sell

        Returns:
            Risk assessment with:
            - Risk score and level
            - Position size recommendation
            - Capital utilization impact
            - Risk limits compliance
        """
        return self._make_request('POST', '/risk/assess', data=trade_data)

    # ============================================================================
    # MONITORING OPERATIONS
    # ============================================================================

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system performance metrics

        Returns:
            System metrics including:
            - Service health status
            - Performance counters
            - Resource utilization
            - Error rates
        """
        return self._make_request('GET', '/system/metrics')

    def get_alerts(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get recent system alerts

        Args:
            limit: Maximum number of alerts to retrieve

        Returns:
            Recent alerts with timestamps and severity levels
        """
        return self._make_request('GET', '/alerts', params={"limit": limit})

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def wait_for_health(self, max_wait: int = 60) -> bool:
        """
        Wait for MCP server to become healthy

        Args:
            max_wait: Maximum wait time in seconds

        Returns:
            True if server becomes healthy, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < max_wait:
            if self.is_healthy():
                return True
            time.sleep(2)

        return False

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status overview

        Returns:
            Combined status from all VIPER services
        """
        try:
            # Get health from all major services
            services = {
                "mcp_server": self.get_health(),
                "portfolio": self.get_portfolio_status(),
                "market_data": self.get_market_data()
            }

            # Calculate overall status
            healthy_services = sum(1 for s in services.values()
                                 if isinstance(s, dict) and s.get('status') == 'healthy')

            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "healthy" if healthy_services >= 2 else "degraded",
                "services": services,
                "healthy_count": healthy_services,
                "total_services": len(services)
            }

        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }

# ============================================================================
# CONVENIENCE CLASSES
# ============================================================================

class VIPERTradingAgent:
    """
    High-level trading agent using MCP client

    Provides AI-friendly interface for automated trading operations.
    """

    def __init__(self, mcp_client: Optional[VIPERMCPClient] = None):
        self.mcp = mcp_client or VIPERMCPClient()
        self.is_active = False
        self.portfolio_cache = {}
        self.last_update = 0
        self.cache_duration = 30  # Cache for 30 seconds

    def start(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start the trading agent

        Args:
            config: Trading configuration

        Returns:
            Startup result
        """
        if not self.mcp.is_healthy():
            return {"status": "error", "message": "MCP server not healthy"}

        result = self.mcp.start_trading(config)
        if result.get("status") == "success":
            self.is_active = True

        return result

    def stop(self) -> Dict[str, Any]:
        """
        Stop the trading agent

        Returns:
            Stop result
        """
        result = self.mcp.stop_trading()
        if result.get("status") == "success":
            self.is_active = False

        return result

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status

        Returns:
            Agent status with portfolio and market information
        """
        if time.time() - self.last_update > self.cache_duration:
            self.portfolio_cache = self.mcp.get_portfolio_status()
            self.last_update = time.time()

        return {
            "agent_active": self.is_active,
            "mcp_healthy": self.mcp.is_healthy(),
            "portfolio": self.portfolio_cache,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_opportunity(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze trading opportunity

        Args:
            symbol: Trading pair to analyze

        Returns:
            Analysis result with recommendation
        """
        market_data = self.mcp.get_market_data(symbol)
        portfolio = self.mcp.get_portfolio_status()

        # Simple analysis (can be enhanced with AI)
        price = market_data.get("ticker", {}).get("price", 0)

        analysis = {
            "symbol": symbol,
            "current_price": price,
            "market_data": market_data,
            "portfolio_balance": portfolio.get("balance", {}).get("free", 0),
            "recommendation": "HOLD",  # Default recommendation
            "confidence": 0.5
        }

        # Basic strategy logic
        if price > 0:
            # This is where AI analysis would go
            analysis["recommendation"] = "BUY" if price < 45000 else "SELL"
            analysis["confidence"] = 0.7

        return analysis

    def execute_trade(self, symbol: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade based on analysis

        Args:
            symbol: Trading pair
            analysis: Analysis result from analyze_opportunity

        Returns:
            Trade execution result
        """
        if not self.is_active:
            return {"status": "error", "message": "Trading agent not active"}

        recommendation = analysis.get("recommendation", "HOLD")
        if recommendation == "HOLD":
            return {"status": "success", "message": "No trade executed"}

        # Assess risk before trading
        risk_assessment = self.mcp.assess_risk({
            "symbol": symbol,
            "amount": 0.001,  # Fixed amount for demo
            "price": analysis["current_price"]
        })

        if not risk_assessment.get("allowed", False):
            return {
                "status": "error",
                "message": "Trade blocked by risk management",
                "risk_details": risk_assessment
            }

        # Execute trade
        trade_config = {
            "symbol": symbol,
            "strategy": "AI_AGENT",
            "amount": 0.001,
            "risk_per_trade": 0.02
        }

        return self.mcp.start_trading(trade_config)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example usage of VIPER MCP client"""

    # Initialize client
    client = VIPERMCPClient()

    # Check if server is healthy
    if not client.is_healthy():
        print("❌ MCP server not available")
        return

    print("✅ VIPER MCP server is healthy")

    # Get capabilities
    capabilities = client.get_capabilities()
    print(f"📋 Available capabilities: {list(capabilities.keys())}")

    # Get market data
    market_data = client.get_market_data("BTC/USDT:USDT")
    print(f"📊 BTC Price: ${market_data.get('ticker', {}).get('price', 'N/A')}")

    # Get portfolio status
    portfolio = client.get_portfolio_status()
    balance = portfolio.get("balance", {}).get("free", 0)
    print(f"💰 Portfolio Balance: ${balance:.2f}")

    # Example trading agent
    agent = VIPERTradingAgent(client)

    # Analyze opportunity
    analysis = agent.analyze_opportunity("BTC/USDT:USDT")
    print(f"🎯 Analysis: {analysis['recommendation']} (confidence: {analysis['confidence']:.1f})")

    # Start trading (uncomment to actually trade)
    # result = agent.start({
    #     "symbol": "BTC/USDT:USDT",
    #     "strategy": "AI_AGENT",
    #     "risk_per_trade": 0.02
    # })
    # print(f"🚀 Trading started: {result}")

if __name__ == "__main__":
    example_usage()
