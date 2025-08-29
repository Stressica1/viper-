# 🤖 VIPER Trading Bot - AI Setup Guide

This comprehensive guide helps AI systems set up and deploy the VIPER trading bot with all dependencies and configurations.

## 📋 Table of Contents

- [🏗️ System Architecture](#-system-architecture)
- [📦 Dependencies](#-dependencies) 
- [🔧 Installation](#-installation)
- [⚙️ Configuration](#-configuration)
- [🐳 Docker Setup](#-docker-setup)
- [🚀 Deployment](#-deployment)
- [📊 Monitoring](#-monitoring)
- [🔍 Troubleshooting](#-troubleshooting)
- [🧪 Testing](#-testing)
- [🛠️ Development](#-development)

## 🏗️ System Architecture

The VIPER trading bot uses a modular architecture organized as follows:

```
viper-/
├── src/viper/              # Main source code
│   ├── core/              # Core trading systems
│   ├── execution/         # Trade execution engines  
│   ├── strategies/        # Trading strategies & optimization
│   ├── risk/              # Risk management systems
│   └── utils/             # Utility modules
├── scripts/               # Executable scripts
├── config/                # Configuration files
├── docs/                  # Documentation
├── tools/                 # Development tools
├── services/              # Microservices
└── deployments/           # Deployment configurations
```

### Core Components

- **Trading Engine**: Located in `src/viper/core/`
- **Risk Management**: Located in `src/viper/risk/`
- **Strategy Engine**: Located in `src/viper/strategies/`
- **Execution System**: Located in `src/viper/execution/`
- **MCP Integration**: Model Context Protocol for AI coordination

## 📦 Dependencies

### System Requirements

- **Python**: 3.8+ (recommended: 3.10+)
- **Docker**: Latest stable version
- **Docker Compose**: v2.0+
- **Redis**: 6.0+ (via Docker)
- **Git**: For repository management

### Python Dependencies

All Python dependencies are managed in `requirements.txt`:

```bash
# Core Framework
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.0.0

# Trading & Exchange
ccxt>=4.1.63

# Data Processing  
pandas>=2.1.4
numpy>=1.24.3

# Database & Caching
redis>=5.0.1

# Rich Terminal Output
rich>=13.0.0

# Complete list in requirements.txt
```

## 🔧 Installation

### Quick Setup (Automated)

For AI systems and rapid deployment, use the automated setup script:

```bash
# Clone repository
git clone https://github.com/Stressica1/viper-.git
cd viper-

# Run automated setup
python scripts/quick_setup.py
```

This script will automatically:
- Set up Python virtual environment
- Install all dependencies 
- Create configuration files
- Start Docker services
- Validate the complete setup

### Manual Setup

#### 1. Clone Repository

```bash
git clone https://github.com/Stressica1/viper-.git
cd viper-
```

#### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv viper_env

# Activate virtual environment
# On Linux/macOS:
source viper_env/bin/activate
# On Windows:
viper_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin python3-dev build-essential
sudo usermod -aG docker $USER
# Logout and login again for docker group changes
```

#### CentOS/RHEL
```bash
sudo yum install -y docker docker-compose python3-devel gcc
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

#### macOS
```bash
# Install Docker Desktop from https://docker.com/products/docker-desktop
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.10
```

#### Windows
```bash
# Install Docker Desktop from https://docker.com/products/docker-desktop
# Install Python from https://python.org/downloads/
# Use Windows Subsystem for Linux (WSL2) for best experience
```

## ⚙️ Configuration

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit environment file
nano .env  # or vim .env
```

### 2. Essential Environment Variables

```bash
# Trading Configuration
FORCE_LIVE_TRADING=true
USE_MOCK_DATA=false
MANDATORY_DOCKER=true
MANDATORY_MCP=true

# Exchange API Credentials (Bitget)
BITGET_API_KEY=your_actual_api_key
BITGET_API_SECRET=your_actual_api_secret
BITGET_API_PASSWORD=your_actual_api_password

# Risk Management
RISK_PER_TRADE=0.02           # 2% risk per trade
MAX_LEVERAGE=50               # Maximum leverage
MAX_POSITIONS=15              # Maximum concurrent positions
DAILY_LOSS_LIMIT=0.03         # 3% daily loss limit
POSITION_SIZE_USDT=100        # Position size in USDT

# System Configuration
REDIS_URL=redis://localhost:6379
MCP_SERVER_URL=http://localhost:8015
GITHUB_TOKEN=your_github_token  # For MCP integration

# Monitoring
ENABLE_MONITORING=true
GRAFANA_ADMIN_PASSWORD=secure_password
```

### 3. Advanced Configuration Files

#### config/optimal_mcp_config.py
```python
# MCP (Model Context Protocol) Configuration
MCP_CONFIG = {
    "server_host": "0.0.0.0",
    "server_port": 8015,
    "max_connections": 100,
    "timeout_seconds": 30,
    "retry_attempts": 3
}
```

#### config/vault/trading_config.json
```json
{
    "risk_management": {
        "max_risk_per_trade": 0.02,
        "max_portfolio_risk": 0.10,
        "stop_loss_percentage": 0.05,
        "take_profit_ratio": 2.0
    },
    "strategy_settings": {
        "default_timeframe": "1m",
        "indicators": ["RSI", "MACD", "EMA"],
        "signal_threshold": 0.7
    }
}
```

## 🐳 Docker Setup

### 1. Docker Services

The system uses Docker Compose for service orchestration:

```yaml
# docker-compose.yml (simplified)
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    
  mcp-server:
    build: ./services/mcp-server
    ports:
      - "8015:8015"
    depends_on:
      - redis
      
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
```

### 2. Start Docker Services

```bash
# Start all services
docker compose up -d

# Verify services are running
docker compose ps

# Check service logs
docker compose logs -f mcp-server
```

### 3. Service Health Checks

```bash
# Check Redis
redis-cli ping

# Check MCP Server
curl http://localhost:8015/health

# Check Prometheus
curl http://localhost:9090/-/healthy

# Check Grafana
curl http://localhost:3000/api/health
```

## 🚀 Deployment

### 1. Development Mode

```bash
# Quick development start
python scripts/run_live_system.py

# Or with specific components
python scripts/start_basic_services.py
```

### 2. Production Deployment

```bash
# Use mandatory enforcement launcher
python scripts/start_live_trading_mandatory.py

# This launcher:
# ✅ Validates environment configuration
# ✅ Ensures Docker services are running
# ✅ Confirms MCP server is operational
# ✅ Enforces live trading mode only
# ✅ Blocks execution if requirements not met
```

### 3. Complete System Launch

```bash
# Launch complete integrated system
python scripts/launch_complete_system.py

# This includes:
# - All microservices
# - Monitoring stack
# - Risk management
# - Strategy engines
# - MCP coordination
```

### 4. Validation Commands

```bash
# Validate repository structure
python tools/repository_rules.py --validate

# Check system health
python tools/system_health_check.py

# Run comprehensive diagnostics
python scripts/master_diagnostic_scanner.py
```

## 📊 Monitoring

### 1. Access Monitoring Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MCP Server**: http://localhost:8015
- **Trading Dashboard**: http://localhost:8007/dashboard

### 2. Log Files

```bash
# Main system logs
tail -f logs/viper_system.log

# Trading activity
tail -f logs/trading_activity.log

# Error logs
tail -f logs/error.log

# Docker logs
docker compose logs -f --tail=100
```

### 3. Metrics and Alerts

The system provides comprehensive monitoring:

- **Trading Metrics**: P&L, win rate, position count
- **System Metrics**: CPU, memory, network usage  
- **Exchange Metrics**: API latency, order status
- **Risk Metrics**: Exposure, drawdown, volatility

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# If you get import errors, ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/viper-/src"

# Or add to your script:
import sys
sys.path.append('/path/to/viper-/src')
```

#### 2. Docker Service Issues

```bash
# Restart Docker services
docker compose down
docker compose up -d

# Check Docker daemon
sudo systemctl status docker
sudo systemctl start docker

# Clean Docker system
docker system prune -a
```

#### 3. Redis Connection Issues

```bash
# Test Redis connection
redis-cli -h localhost -p 6379 ping

# Restart Redis
docker compose restart redis

# Check Redis logs
docker compose logs redis
```

#### 4. MCP Server Issues

```bash
# Check MCP server health
curl -v http://localhost:8015/health

# Restart MCP server
docker compose restart mcp-server

# Check MCP logs
docker compose logs mcp-server
```

#### 5. API Connection Issues

```bash
# Test exchange connectivity
python -c "
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()
exchange = ccxt.bitget({
    'apiKey': os.getenv('BITGET_API_KEY'),
    'secret': os.getenv('BITGET_API_SECRET'),
    'password': os.getenv('BITGET_API_PASSWORD'),
    'sandbox': False
})
print(exchange.fetch_balance())
"
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python scripts/run_live_system.py --verbose --debug

# Use diagnostic tools
python tools/comprehensive_bug_detector.py
```

## 🧪 Testing

### 1. Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_strategies.py
pytest tests/test_risk_management.py
pytest tests/test_execution.py
```

### 2. Integration Tests

```bash
# Test MCP integration
python scripts/test_mcp_fix_system.py

# Test system integration
python src/viper/core/integration_test_complete.py

# Comprehensive system validation
python src/viper/core/comprehensive_verification_system.py
```

### 3. Strategy Backtesting

```bash
# Run backtest on specific strategy
python scripts/run_backtesting_optimizer.py

# Comprehensive strategy analysis
python scripts/run_comprehensive_strategy_analysis.py

# Massive backtest across multiple timeframes
python scripts/run_massive_backtest.py
```

## 🛠️ Development

### 1. Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/
mypy src/
```

### 2. Adding New Strategies

Create new strategies in `src/viper/strategies/`:

```python
# src/viper/strategies/my_strategy.py
from src.viper.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "My Custom Strategy"
    
    def generate_signals(self, data):
        # Your strategy logic here
        pass
```

### 3. Repository Maintenance

```bash
# Validate repository structure
python tools/repository_rules.py --validate

# Clean up any violations
python tools/clean_root_enforcer.py --execute

# Organize new files
python tools/repo_organizer.py --scan --execute
```

### 4. CI/CD Integration

The repository includes GitHub Actions workflows:

- **Structure Validation**: Ensures repository organization
- **Code Quality**: Linting, formatting, type checking
- **Security Scanning**: Dependency and secret scanning
- **Testing**: Unit tests, integration tests, backtesting

### 5. MCP Development

For MCP (Model Context Protocol) development:

```bash
# Start MCP development server
python services/mcp-server/main.py --dev

# Test MCP integration
python scripts/github_mcp_trading_tasks.py

# MCP brain controller
python src/viper/core/mcp_brain_controller.py
```

## 🔐 Security Best Practices

### 1. API Key Management

- Never commit API keys to version control
- Use `.env` files for local development
- Use secure vaults for production (config/vault/)
- Rotate API keys regularly

### 2. Network Security

- Use Docker network isolation
- Configure firewall rules
- Enable SSL/TLS for external connections
- Monitor network traffic

### 3. System Security

- Regular security updates
- Monitor logs for suspicious activity
- Use strong passwords for services
- Enable two-factor authentication where possible

## 📚 Additional Resources

### Documentation

- [Trading Strategies Guide](./TRADING_STRATEGIES.md)
- [Risk Management Guide](./RISK_MANAGEMENT.md)
- [API Reference](./API_REFERENCE.md)
- [Deployment Guide](./DEPLOYMENT.md)

### Support

- **Issues**: Create GitHub issues for bugs/features
- **Discussions**: Use GitHub discussions for questions
- **Wiki**: Check the repository wiki for additional docs

### External Resources

- [CCXT Documentation](https://ccxt.readthedocs.io/)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Bitget API Documentation](https://bitgetlimited.github.io/apidoc/en/mix/)

---

## 🎯 Quick Start Checklist

- [ ] Clone repository
- [ ] Setup Python environment (`python -m venv viper_env`)
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Configure environment (`.env` file)
- [ ] Start Docker services (`docker compose up -d`)
- [ ] Validate setup (`python tools/repository_rules.py --validate`)
- [ ] Run health check (`python scripts/system_health_check.py`)
- [ ] Launch system (`python scripts/start_live_trading_mandatory.py`)

**⚠️ WARNING**: This system executes real trades with real money. Ensure you understand the risks and have proper risk management in place before running in production.

---

*Last updated: $(date)*
*Repository: viper- by @Stressica1*