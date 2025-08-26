# 🚀 VIPER Trading System - MCP Server

**Live Trading & Scanning Capabilities**

This MCP server provides comprehensive trading functionality for the VIPER algorithmic trading system, enabling AI-powered live trading, market scanning, and portfolio management.

## ✨ Features

### 🔥 Live Trading
- **Real-time Strategy Execution** - VIPER, Momentum, Mean Reversion strategies
- **Risk Management** - Automatic position sizing and stop-losses
- **Portfolio Tracking** - Real-time P&L and position monitoring
- **Emergency Controls** - Instant position closure capabilities

### 🔍 Market Scanning
- **Multi-Symbol Scanning** - BTC, ETH, SOL, and custom pairs
- **Real-time Opportunity Detection** - Volume, volatility, and trend analysis
- **Technical Indicators** - RSI, MACD, Bollinger Bands, Moving Averages
- **Custom Criteria** - Configurable scanning parameters

### 📊 Analysis & Backtesting
- **Technical Analysis** - Comprehensive indicator calculations
- **Strategy Backtesting** - Historical performance analysis
- **Risk Metrics** - Portfolio risk assessment and limits
- **Market Data** - Real-time OHLCV and ticker data

## 🛠️ Available Tools

### Live Trading Tools
- `start_live_trading` - Start automated trading with specified strategy
- `stop_live_trading` - Stop trading with emergency stop option
- `execute_trade` - Execute individual buy/sell orders
- `get_portfolio` - Get current portfolio status and positions

### Market Scanning Tools
- `start_market_scan` - Start real-time opportunity scanning
- `stop_market_scan` - Stop scanning operations
- `get_market_data` - Retrieve real-time market data
- `analyze_pair` - Technical analysis with indicators

### Analysis Tools
- `get_risk_metrics` - Comprehensive risk assessment
- `backtest_strategy` - Historical strategy performance
- `analyze_pair` - Detailed technical analysis

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd mcp-trading-server
npm install
```

### 2. Configure Environment
```bash
# Copy environment template
cp ../.env .env

# Edit with your Bitget API credentials
BITGET_API_KEY=your_api_key
BITGET_API_SECRET=your_api_secret
BITGET_API_PASSWORD=your_api_password
```

### 3. Start the Server
```bash
npm start
```

## 📖 Usage Examples

### Start Live Trading
```javascript
// Start VIPER strategy trading
const result = await client.callTool("start_live_trading", {
  strategy: "viper",
  symbol: "BTC/USDT:USDT",
  risk_per_trade: 0.02,
  max_positions: 15
});
```

### Market Scanning
```javascript
// Start scanning for opportunities
const result = await client.callTool("start_market_scan", {
  symbols: ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
  scan_interval: 30,
  criteria: {
    min_volume: 1000000,
    min_volatility: 0.02,
    trend_filter: "bullish"
  }
});
```

### Technical Analysis
```javascript
// Analyze a trading pair
const result = await client.callTool("analyze_pair", {
  symbol: "BTC/USDT:USDT",
  indicators: ["rsi", "macd", "bollinger", "volume"]
});
```

### Execute Trade
```javascript
// Execute a market buy order
const result = await client.callTool("execute_trade", {
  symbol: "BTC/USDT:USDT",
  side: "buy",
  order_type: "market",
  amount: 0.001
});
```

### Portfolio Management
```javascript
// Get current portfolio status
const result = await client.callTool("get_portfolio", {
  include_pnl: true
});
```

## 🔧 Configuration

### Environment Variables
```env
# Bitget API Credentials
BITGET_API_KEY=your_api_key
BITGET_API_SECRET=your_api_secret
BITGET_API_PASSWORD=your_api_password

# Trading Parameters
RISK_PER_TRADE=0.02
MAX_POSITIONS=15
MAX_LEVERAGE=25

# Scanning Parameters
SCAN_INTERVAL=30
MIN_VOLUME=1000000
MIN_VOLATILITY=0.02
```

### Strategy Configuration

#### VIPER Strategy
- **Threshold**: 85% confidence required
- **Cooldown**: 300 seconds between trades
- **Indicators**: RSI, MACD, Bollinger Bands
- **Risk Management**: 2% per trade, 15 position limit

#### Momentum Strategy
- **Entry**: RSI < 30 (oversold)
- **Exit**: RSI > 70 (overbought)
- **Filters**: Volume and volatility thresholds

## 📊 Risk Management

### Built-in Safety Features
- **Position Limits** - Maximum 15 concurrent positions
- **Risk Controls** - 2% maximum risk per trade
- **Emergency Stop** - Instant position closure capability
- **Daily Loss Limits** - 3% maximum daily loss
- **Leverage Limits** - Configurable maximum leverage

### Monitoring
- **Real-time P&L** - Live profit/loss tracking
- **Risk Metrics** - Portfolio risk assessment
- **Position Tracking** - Active position monitoring
- **Performance Analytics** - Trading performance metrics

## 🔌 Integration

### MCP Client Integration
```javascript
import { McpClient } from "@modelcontextprotocol/client";

const client = new McpClient({
  transport: {
    type: "process",
    process: tradingServerProcess
  }
});

// Use trading tools
const portfolio = await client.callTool("get_portfolio");
const analysis = await client.callTool("analyze_pair", {
  symbol: "BTC/USDT:USDT"
});
```

### VIPER System Integration
- **WebSocket Streams** - Real-time market data
- **REST API** - Direct trading execution
- **Portfolio Sync** - Automatic position synchronization
- **Risk Integration** - Centralized risk management

## 📈 Performance Metrics

- **Scan Speed**: < 2 seconds per symbol scan
- **Trade Execution**: < 1 second order placement
- **Data Latency**: < 100ms market data updates
- **Risk Calculation**: Real-time portfolio assessment
- **Strategy Execution**: 30-second intervals

## 🛡️ Security

- **API Key Encryption** - Secure credential storage
- **Request Validation** - Input sanitization and validation
- **Rate Limiting** - Exchange API rate limit management
- **Error Handling** - Comprehensive error recovery
- **Audit Logging** - Complete transaction logging

## 🚨 Error Handling

The server includes comprehensive error handling for:
- Network connectivity issues
- API rate limit exceeded
- Insufficient balance errors
- Invalid order parameters
- Exchange maintenance periods

## 📝 License

This MCP server is part of the VIPER Trading System and follows the same licensing terms.

## 🤝 Contributing

Contributions to improve the trading functionality are welcome. Please ensure all changes maintain the security and risk management standards.

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Always test thoroughly before using in live trading environments. Cryptocurrency trading involves substantial risk of loss.
