# 🚀 VIPER Trading Bot - MCP Integration Guide
## Model Context Protocol Implementation

## 📋 Overview

The VIPER Trading Bot now includes full Model Context Protocol (MCP) support, enabling seamless integration with AI agents and providing standardized access to all trading operations.

## 🤖 MCP Server Features

### **Core Capabilities**

#### **Trading Operations**
- **Start/Stop Trading**: Control automated trading execution
- **Portfolio Management**: Real-time portfolio status and P&L tracking
- **Risk Assessment**: Comprehensive risk analysis and position sizing

#### **Backtesting Engine**
- **Strategy Testing**: Execute backtests with custom parameters
- **Performance Analysis**: Get detailed backtest results and metrics
- **Optimization**: Run parameter optimization studies

#### **Market Data Access**
- **Real-time Tickers**: Live price feeds and market data
- **OHLCV Data**: Historical price data for analysis
- **Order Book**: Market depth and liquidity information

#### **Risk Management**
- **Position Limits**: Check position size compliance
- **Risk Limits**: Monitor daily loss limits and exposure
- **Auto-stops**: Emergency stop mechanisms

#### **System Monitoring**
- **Health Checks**: Service status and system health
- **Metrics**: Performance metrics and system analytics
- **Alerts**: Real-time alert notifications

## 🛠️ Setup Instructions

### **1. Start MCP Server**
```bash
# Start all services including MCP server
docker-compose up -d

# Check MCP server health
curl http://localhost:8015/health
```

### **2. Configure MCP Client**
The MCP server is already configured in `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "viper_mcp": {
      "url": "http://localhost:8015/sse",
      "description": "VIPER Trading Bot MCP Server"
    }
  }
}
```

### **3. Test MCP Connection**
```bash
# Get MCP capabilities
curl http://localhost:8015/capabilities

# Get system health
curl http://localhost:8015/health
```

## 📡 API Reference

### **REST Endpoints**

#### **Trading Operations**
```
POST /trading/start    - Start automated trading
POST /trading/stop     - Stop automated trading
GET  /portfolio/status - Get portfolio status
POST /risk/assess      - Assess trading risk
```

#### **Backtesting**
```
POST /backtest/run     - Execute backtest
GET  /backtest/results - Get backtest results
```

#### **Market Data**
```
GET  /market/data?symbol=BTC/USDT:USDT - Get market data
GET  /market/ticker/{symbol}           - Get ticker data
GET  /market/ohlcv/{symbol}            - Get OHLCV data
```

#### **System**
```
GET  /health          - Health check
GET  /capabilities    - Get capabilities
GET  /sse            - Server-Sent Events stream
WS   /ws             - WebSocket connection
```

### **WebSocket Messages**

#### **Subscribe to Updates**
```json
{
  "type": "subscribe_trades",
  "symbol": "BTC/USDT:USDT"
}
```

#### **Request Signals**
```json
{
  "type": "get_signals",
  "symbol": "BTC/USDT:USDT",
  "timeframe": "1h"
}
```

#### **Execute Trade**
```json
{
  "type": "execute_trade",
  "symbol": "BTC/USDT:USDT",
  "side": "buy",
  "amount": 0.001,
  "price": 45000
}
```

## 🔧 MCP Client Library

### **Python MCP Client**

```python
import requests
import json
from typing import Dict, Any

class VIPERMCPClient:
    """VIPER MCP Client for AI agents"""

    def __init__(self, mcp_url: str = "http://localhost:8015"):
        self.base_url = mcp_url

    def get_capabilities(self) -> Dict[str, Any]:
        """Get MCP server capabilities"""
        response = requests.get(f"{self.base_url}/capabilities")
        return response.json()

    def start_trading(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start automated trading"""
        response = requests.post(f"{self.base_url}/trading/start", json=config)
        return response.json()

    def stop_trading(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Stop automated trading"""
        response = requests.post(f"{self.base_url}/trading/stop", json=config)
        return response.json()

    def run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest with parameters"""
        response = requests.post(f"{self.base_url}/backtest/run", json=params)
        return response.json()

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data"""
        response = requests.get(f"{self.base_url}/market/data", params={"symbol": symbol})
        return response.json()

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        response = requests.get(f"{self.base_url}/portfolio/status")
        return response.json()

    def assess_risk(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess trading risk"""
        response = requests.post(f"{self.base_url}/risk/assess", json=trade_data)
        return response.json()

# Usage example
client = VIPERMCPClient()

# Get capabilities
capabilities = client.get_capabilities()
print("Available functions:", list(capabilities.keys()))

# Start trading
result = client.start_trading({
    "symbol": "BTC/USDT:USDT",
    "strategy": "VIPER",
    "risk_per_trade": 0.02
})
print("Trading started:", result)

# Get market data
market_data = client.get_market_data("BTC/USDT:USDT")
print("BTC Price:", market_data.get("ticker", {}).get("price"))

# Assess risk
risk = client.assess_risk({
    "symbol": "BTC/USDT:USDT",
    "amount": 0.001,
    "price": 45000
})
print("Risk assessment:", risk)
```

## 🚀 Integration Examples

### **AI Agent Trading Example**

```python
# AI Agent using VIPER MCP for automated trading

import asyncio
from viper_mcp_client import VIPERMCPClient

class TradingAgent:
    def __init__(self):
        self.mcp = VIPERMCPClient()
        self.is_trading = False

    async def analyze_market(self):
        """Analyze market conditions"""
        data = self.mcp.get_market_data("BTC/USDT:USDT")
        price = data["ticker"]["price"]

        # AI analysis logic here
        signal = "BUY" if price > 43000 else "SELL"
        return signal, price

    async def execute_strategy(self):
        """Execute trading strategy"""
        signal, price = await self.analyze_market()

        if signal == "BUY" and not self.is_trading:
            # Assess risk first
            risk = self.mcp.assess_risk({
                "symbol": "BTC/USDT:USDT",
                "amount": 0.001,
                "price": price
            })

            if risk["allowed"]:
                # Execute trade
                result = self.mcp.start_trading({
                    "symbol": "BTC/USDT:USDT",
                    "strategy": "AI_VIPER",
                    "amount": 0.001
                })

                self.is_trading = True
                print(f"Trade executed: {result}")

    async def monitor_portfolio(self):
        """Monitor portfolio performance"""
        while True:
            status = self.mcp.get_portfolio_status()
            print(f"Portfolio: ${status['balance']['total']:.2f}")
            await asyncio.sleep(60)  # Check every minute

    async def run(self):
        """Run the trading agent"""
        await asyncio.gather(
            self.execute_strategy(),
            self.monitor_portfolio()
        )

# Start AI trading agent
agent = TradingAgent()
asyncio.run(agent.run())
```

### **Backtesting Example**

```python
# AI-driven backtesting using MCP

from viper_mcp_client import VIPERMCPClient

def optimize_strategy():
    """Use AI to optimize trading strategy"""
    mcp = VIPERMCPClient()

    # Define parameter ranges for optimization
    risk_levels = [0.01, 0.02, 0.03]
    timeframes = ["1h", "4h", "1d"]
    thresholds = [75, 85, 95]

    best_result = None
    best_score = -float('inf')

    for risk in risk_levels:
        for timeframe in timeframes:
            for threshold in thresholds:

                # Run backtest with current parameters
                result = mcp.run_backtest({
                    "symbol": "BTC/USDT:USDT",
                    "start_date": "2023-01-01",
                    "end_date": "2024-01-01",
                    "risk_per_trade": risk,
                    "timeframe": timeframe,
                    "viper_threshold": threshold
                })

                # Evaluate results (AI scoring logic)
                score = calculate_score(result)

                if score > best_score:
                    best_score = score
                    best_result = {
                        "parameters": {
                            "risk": risk,
                            "timeframe": timeframe,
                            "threshold": threshold
                        },
                        "result": result,
                        "score": score
                    }

    return best_result

def calculate_score(backtest_result):
    """AI scoring function for backtest results"""
    win_rate = backtest_result.get("win_rate", 0)
    total_return = backtest_result.get("total_return", 0)
    max_drawdown = backtest_result.get("max_drawdown", 100)

    # Custom scoring algorithm
    score = (win_rate * 0.4) + (total_return * 0.4) - (max_drawdown * 0.2)
    return score

# Run optimization
best_config = optimize_strategy()
print(f"Best configuration: {best_config['parameters']}")
print(f"Score: {best_config['score']:.2f}")
```

## 🔒 Security Best Practices

### **Authentication**
- MCP server uses vault-based authentication
- Access tokens required for all operations
- Rate limiting on all endpoints

### **Data Validation**
- Comprehensive input validation
- Sanitization of all user inputs
- Type checking and bounds validation

### **Error Handling**
- Graceful error responses
- Detailed logging without sensitive data
- Fallback mechanisms for service failures

## 📊 Monitoring & Logging

### **MCP Server Logs**
```bash
# View MCP server logs
docker-compose logs mcp-server

# Follow logs in real-time
docker-compose logs -f mcp-server
```

### **Performance Metrics**
```bash
# Get MCP server metrics
curl http://localhost:8015/metrics

# System health dashboard
open http://localhost:8000  # API Server dashboard
open http://localhost:3000  # Grafana monitoring
```

## 🎯 Use Cases

### **AI-Powered Trading**
- Automated signal generation and execution
- Real-time market analysis and decision making
- Adaptive strategy optimization

### **Portfolio Management**
- Risk assessment and position sizing
- Automated rebalancing and diversification
- Performance monitoring and reporting

### **Backtesting & Research**
- Historical strategy validation
- Parameter optimization studies
- Scenario analysis and stress testing

### **Market Analysis**
- Real-time data processing and analysis
- Pattern recognition and prediction
- Multi-asset correlation analysis

## 🚨 Troubleshooting

### **Common Issues**

#### **MCP Server Not Starting**
```bash
# Check container status
docker-compose ps mcp-server

# View logs
docker-compose logs mcp-server

# Restart service
docker-compose restart mcp-server
```

#### **Connection Refused**
```bash
# Check if service is running
curl http://localhost:8015/health

# Verify port configuration
docker-compose config | grep -A 10 mcp-server
```

#### **Authentication Errors**
```bash
# Check vault token configuration
echo $VAULT_ACCESS_TOKEN_MCP_SERVER

# Verify vault service is running
docker-compose ps credential-vault
```

## 📈 Performance Optimization

### **Caching Strategies**
- Redis-based response caching
- Market data prefetching
- Connection pooling

### **Rate Limiting**
- Configurable request limits
- Burst handling capabilities
- Graceful degradation

### **Scalability**
- Horizontal scaling support
- Load balancing configuration
- Resource optimization

## 🔄 Future Enhancements

### **Planned Features**
- **Advanced AI Integration**: Enhanced ML model support
- **Multi-Exchange Support**: Additional exchange integrations
- **Real-time Alerts**: Push notification system
- **Advanced Analytics**: Machine learning insights
- **API Rate Optimization**: Smart rate limit management

---

## 🎉 Getting Started

1. **Deploy the System**:
   ```bash
   docker-compose up -d
   ```

2. **Configure MCP**:
   ```bash
   # MCP is already configured in .cursor/mcp.json
   # Start using the VIPERMCPClient in your AI applications
   ```

3. **Test Integration**:
   ```python
   from viper_mcp_client import VIPERMCPClient

   client = VIPERMCPClient()
   capabilities = client.get_capabilities()
   print("VIPER MCP Ready:", capabilities)
   ```

**The VIPER Trading Bot with MCP integration provides a powerful, standardized interface for AI-driven trading operations! 🚀**
