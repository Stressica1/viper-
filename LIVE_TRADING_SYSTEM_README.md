# 🚀 VIPER Trading System - Complete Live Trading & Scanning

**Enterprise-Grade Algorithmic Trading Platform with MCP Integration**

[![System Status](https://img.shields.io/badge/Status-Production--Ready-green)](https://github.com/Stressica1/NEW-PHEMEX)
[![MCP Ready](https://img.shields.io/badge/MCP-Integrated-blue)](https://modelcontextprotocol.io)
[![Docker Ready](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com)

## 🔥 What Makes This System Special

### 🤖 **AI-Powered Trading**
- **VIPER Strategy**: Advanced algorithmic trading with 85% confidence threshold
- **Momentum Strategy**: RSI-based trading with oversold/overbought detection
- **Real-time Analysis**: Technical indicators (RSI, MACD, Bollinger Bands)
- **Risk Management**: 2% per trade limit with automated stop-losses

### 🔍 **Market Scanning Engine**
- **Multi-Symbol Scanning**: BTC, ETH, SOL, and custom pairs
- **Opportunity Detection**: Volume, volatility, and trend analysis
- **Real-time Alerts**: Automated opportunity notifications
- **Custom Criteria**: Configurable scanning parameters

### 🏗️ **Microservices Architecture**
- **14 Services**: Complete event-driven architecture
- **Docker Orchestration**: Production-ready containerization
- **Health Monitoring**: Automated service health checks
- **Scalable Design**: Horizontal scaling capabilities

### 📊 **Enterprise Monitoring**
- **Prometheus Metrics**: Real-time performance monitoring
- **Grafana Dashboards**: Comprehensive visualization
- **ELK Stack**: Centralized logging and analytics
- **Alert System**: Multi-channel notifications

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Required software
- Docker Desktop (latest)
- Node.js 18+ & npm
- Python 3.11+
- 8GB+ RAM recommended
- 50GB+ disk space
```

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/Stressica1/NEW-PHEMEX.git
cd NEW-PHEMEX

# Configure environment (add your Bitget API keys)
cp .env.example .env
# Edit .env with your credentials:
# BITGET_API_KEY=your_api_key
# BITGET_API_SECRET=your_api_secret
# BITGET_API_PASSWORD=your_password
```

### 2. Start Complete System
```bash
# One-command system startup
python scripts/connect_trading_system.py start

# This starts:
# ✅ 8+ Microservices (API, Risk Manager, Exchange Connector, etc.)
# ✅ MCP Trading Server (Live trading & scanning)
# ✅ Monitoring Stack (Prometheus, Grafana)
# ✅ Health Checks & Auto-recovery
```

### 3. Access Your Trading System
```bash
# Main Dashboard
open http://localhost:8000

# Trading Analytics
open http://localhost:3001  # Grafana (admin/viper_admin)

# System Monitoring
open http://localhost:9090  # Prometheus
open http://localhost:5601  # Kibana Logs
```

## 🎯 Live Trading Commands

### Start Live Trading
```bash
# Start VIPER strategy trading
python scripts/start_microservices.py start
# Then configure trading parameters via the web dashboard
```

### Market Scanning
```bash
# Start real-time market scanning
python scripts/start_microservices.py start
# Access scanning controls via API or dashboard
```

### MCP Trading Server
```javascript
// Start the MCP server for AI-powered trading
cd mcp-trading-server
npm start

// Available tools:
// - start_live_trading: Start automated trading
// - start_market_scan: Begin opportunity scanning
// - execute_trade: Manual trade execution
// - get_portfolio: Portfolio status
// - get_risk_metrics: Risk analysis
```

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    🌐 EXTERNAL APIs                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Bitget Exchange API                     │   │
│  │  • Real-time market data streams                        │   │
│  │  • Order execution & position management               │   │
│  │  • Account balance & P&L tracking                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🚀 VIPER TRADING ENGINE                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 MCP Trading Server                      │   │
│  │  • Live trading execution                               │   │
│  │  • Market scanning & analysis                          │   │
│  │  • Strategy implementation                              │   │
│  │  • Risk management & controls                          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🏗️ MICROSERVICES LAYER                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  API Server (8000) | Risk Manager (8002) | Data Mgr   │   │
│  │  Exchange Connector | Monitoring Service | Redis       │   │
│  │  Ultra Backtester   | Strategy Optimizer | Alert Sys   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    📊 MONITORING & LOGGING                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Prometheus (9090) | Grafana (3001) | Kibana (5601)   │   │
│  │  ELK Stack         | Real-time Metrics | Alert Rules   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Available Commands

### System Management
```bash
# Start complete system
python scripts/connect_trading_system.py start

# Stop complete system
python scripts/connect_trading_system.py stop

# Check system status
python scripts/connect_trading_system.py status

# Run integration tests
python scripts/test_live_trading_integration.py
```

### Individual Services
```bash
# Microservices management
python scripts/start_microservices.py start
python scripts/start_microservices.py stop
python scripts/start_microservices.py status
python scripts/start_microservices.py health

# Build services
python scripts/start_microservices.py build --parallel
```

### MCP Servers
```bash
# Trading MCP Server
cd mcp-trading-server
npm start

# GitHub MCP Server
cd mcp-github-project-manager
node build/index.js
```

## 🎮 Trading Strategies

### VIPER Strategy
- **Entry**: 85%+ confidence from technical analysis
- **Exit**: Profit target or stop-loss activation
- **Risk**: 2% maximum per trade
- **Timeframe**: 30-second execution intervals

### Momentum Strategy
- **Entry**: RSI < 30 (oversold) or RSI > 70 (overbought)
- **Exit**: Opposite RSI signals
- **Risk**: 1% maximum per trade
- **Indicators**: RSI(14), MACD, Bollinger Bands

## 📈 Risk Management

### Safety Features
- **Position Limits**: Maximum 15 concurrent positions
- **Daily Loss Limit**: 3% maximum daily drawdown
- **Emergency Stop**: Instant position closure capability
- **Circuit Breakers**: Automatic trading suspension
- **Leverage Controls**: Configurable maximum leverage

### Monitoring
- **Real-time P&L**: Live profit/loss tracking
- **Risk Metrics**: Portfolio risk assessment
- **Performance Analytics**: Strategy performance tracking
- **Alert System**: Multi-channel risk notifications

## 🔧 Configuration

### Environment Variables
```env
# Bitget API (REQUIRED)
BITGET_API_KEY=your_api_key
BITGET_API_SECRET=your_api_secret
BITGET_API_PASSWORD=your_password

# Trading Parameters
RISK_PER_TRADE=0.02
MAX_POSITIONS=15
MAX_LEVERAGE=25

# VIPER Strategy
VIPER_THRESHOLD=85
SIGNAL_COOLDOWN=300

# Scanning Parameters
SCAN_INTERVAL=30
MIN_VOLUME=1000000
MIN_VOLATILITY=0.02
```

### Docker Services
```yaml
# Key services and ports
api-server: 8000          # Web dashboard & API
risk-manager: 8002        # Risk management
data-manager: 8003        # Market data
exchange-connector: 8005  # Bitget integration
monitoring-service: 8006  # System monitoring
ultra-backtester: 8001    # Strategy testing
strategy-optimizer: 8004  # Parameter optimization

# Infrastructure
redis: 6379               # Message broker
prometheus: 9090          # Metrics collection
grafana: 3001             # Dashboards
kibana: 5601              # Log visualization
```

## 🧪 Testing & Validation

### Integration Tests
```bash
# Run comprehensive system tests
python scripts/test_live_trading_integration.py

# Tests include:
# ✅ Microservices health checks
# ✅ API endpoint validation
# ✅ Network connectivity
# ✅ Docker container status
# ✅ Environment configuration
# ✅ MCP server functionality
```

### Health Checks
```bash
# Check all service health
python scripts/start_microservices.py health

# Individual service health
curl http://localhost:8000/health  # API Server
curl http://localhost:8002/health  # Risk Manager
curl http://localhost:8005/health  # Exchange Connector
```

## 🚨 Emergency Procedures

### Emergency Stop
```bash
# Stop all trading immediately
python scripts/start_microservices.py stop

# Emergency position closure
# (Available in MCP Trading Server)
```

### System Recovery
```bash
# Restart complete system
python scripts/connect_trading_system.py stop
python scripts/connect_trading_system.py start

# Check system health
python scripts/test_live_trading_integration.py
```

## 📚 Documentation

### API Documentation
- **REST API**: http://localhost:8000/docs
- **Health Endpoints**: `/health` on each service
- **Metrics**: http://localhost:9090/metrics

### Architecture Documentation
- **System Design**: `docs/TECHNICAL_DOC.md`
- **API Setup**: `docs/API_SETUP_README.md`
- **Risk Management**: `docs/RISK_MANAGEMENT_IMPLEMENTATION.md`
- **User Guide**: `docs/USER_GUIDE.md`

### Service Documentation
- **MCP Trading Server**: `mcp-trading-server/README.md`
- **Microservices**: `services/*/README.md`
- **Docker Setup**: `infrastructure/docker-compose.yml`

## 🤝 Contributing

### Development Setup
```bash
# Clone and setup
git clone https://github.com/Stressica1/NEW-PHEMEX.git
cd NEW-PHEMEX

# Install dependencies
pip install -r requirements.txt
npm install  # For MCP servers

# Start development environment
python scripts/connect_trading_system.py start
```

### Code Standards
- **Python**: PEP 8 with type hints
- **Node.js**: ESLint configuration
- **Docker**: Multi-stage builds
- **Documentation**: Comprehensive READMEs

## 📄 License

This VIPER Trading System is proprietary software. See LICENSE file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Always test thoroughly before live trading
- Never risk more than you can afford to lose
- Understand the risks of algorithmic trading
- Consult financial advisors for investment decisions
- Past performance does not guarantee future results

**Use at your own risk. The authors are not responsible for any financial losses.**

---

## 🎯 Ready to Start Trading?

```bash
# 1. Configure your API keys in .env
# 2. Start the system
python scripts/connect_trading_system.py start

# 3. Access dashboard
open http://localhost:8000

# 4. Start live trading (via MCP or dashboard)
# 5. Monitor performance on Grafana
open http://localhost:3001
```

**Happy Trading! 🚀📈**
