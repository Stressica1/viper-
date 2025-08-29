# 🚀 VIPER Live Trading Bot - Docker & MCP Enforced System

**⚠️ LIVE TRADING SYSTEM ONLY - NO MOCK DATA OR DEMO MODE**

A high-performance automated trading system with mandatory Docker and MCP (Model Context Protocol) enforcement for live cryptocurrency trading.

## 🔒 System Requirements (MANDATORY)

**CRITICAL:** This system operates in LIVE TRADING mode only. All components require:

- **Docker & Docker Compose** - Mandatory for all operations
- **MCP Server** - Required for system coordination  
- **Valid Bitget API credentials** - Real trading credentials only
- **Redis** - For data caching and coordination
- **GitHub PAT** - For MCP integration

## 🚨 Live Trading Warning

**THIS SYSTEM EXECUTES REAL TRADES WITH REAL MONEY**

- No simulation or paper trading mode
- All trades are executed on live markets
- Losses can occur - use proper risk management
- Ensure you understand the risks before running

## 🚀 Quick Start

### 1. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your REAL credentials
# CRITICAL: Use real Bitget API credentials, not placeholders
```

### 2. Start with Mandatory Enforcement

```bash
# Start the system with full enforcement
python start_live_trading_mandatory.py
```

Or use Docker directly:

```bash
# Start Docker services
docker compose up -d

# Verify MCP server is running
curl http://localhost:8015/health

# Start live trading
python final_live_trading_launcher.py
```

## 🔧 System Architecture

The system uses a microservices architecture with Docker enforcement:

- **MCP Server** (Port 8015) - System coordination
- **Live Trading Engine** (Port 8007) - Trade execution
- **Risk Manager** (Port 8002) - Position and risk control
- **Exchange Connector** (Port 8005) - Bitget API integration
- **Market Data Manager** (Port 8003) - Real-time data feeds
- **Redis** (Port 6379) - Data caching and messaging

## ⚙️ Configuration

Key environment variables (all in `.env`):

```bash
# Trading Mode (ENFORCED)
USE_MOCK_DATA=false
FORCE_LIVE_TRADING=true
MANDATORY_DOCKER=true
MANDATORY_MCP=true

# Bitget API (REQUIRED)
BITGET_API_KEY=your_real_api_key
BITGET_API_SECRET=your_real_api_secret  
BITGET_API_PASSWORD=your_real_api_password

# Risk Management
RISK_PER_TRADE=0.02
MAX_LEVERAGE=50
MAX_POSITIONS=15
DAILY_LOSS_LIMIT=0.03
```

## 🛡️ Safety Features

- **Emergency Stop System** - Immediate position closure
- **Risk Limits** - Automatic position sizing and limits
- **Docker Health Checks** - Service monitoring and restart
- **MCP Validation** - Ensures system coordination
- **Real-time Monitoring** - Live performance tracking

## 📊 Monitoring

Access the monitoring dashboard:

- **Grafana:** http://localhost:3000 
- **Prometheus:** http://localhost:9090
- **MCP Server:** http://localhost:8015

## 🚫 What Was Removed

This version has all demo/mock functionality removed:

- All demo_*.py files
- All test_*.py files  
- Mock data generation functions
- Simulation modes
- Development validation scripts

## 🔧 Troubleshooting

### Docker Issues
```bash
# Check Docker status
docker compose ps

# Restart services  
docker compose down && docker compose up -d

# View logs
docker compose logs -f
```

### MCP Server Issues
```bash
# Check MCP server health
curl http://localhost:8015/health

# Restart MCP server
docker compose restart mcp-server
```

## ⚠️ Important Notes

- **No Demo Mode:** This system only operates with real money
- **Docker Required:** All operations require Docker services
- **MCP Required:** System coordination requires MCP server
- **API Credentials:** Must use valid Bitget production credentials
- **Risk Management:** Built-in limits protect against large losses

To contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run tests to ensure everything works
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Requirements

- Python 3.7+
- pip

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Status

🚧 This project is currently under development. More features and documentation coming soon!
