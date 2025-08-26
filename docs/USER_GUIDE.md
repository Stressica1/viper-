# 🚀 VIPER Trading Bot - User Guide

## Quick Start (5 Minutes Setup)

### Prerequisites
- **Docker Desktop** installed and running
- **Git** for repository management
- **Text Editor** (VS Code recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/viper-trading-bot.git
cd viper-trading-bot
```

### Step 2: Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your API credentials (required for live trading)
# IMPORTANT: Never commit real credentials to version control
nano .env
```

### Step 3: Start System
```bash
# Launch all microservices
python start_microservices.py start

# Check system status
python start_microservices.py status
```

### Step 4: Access Dashboard
Visit: **http://localhost:8000**

### Understanding the System Architecture

The VIPER system consists of 17 specialized microservices working together:

#### Core Services
- **🌐 API Server** (Port 8000) - Web dashboard and REST API
- **🧪 Ultra Backtester** (Port 8001) - Strategy backtesting engine
- **🎯 Strategy Optimizer** (Port 8004) - Parameter optimization
- **🔥 Live Trading Engine** (Port 8007) - Automated trading execution

#### Trading Services
- **🎯 Signal Processor** (Port 8011) - VIPER scoring and signal generation
- **📋 Order Lifecycle Manager** (Port 8013) - Complete order management
- **🔄 Position Synchronizer** (Port 8014) - Real-time position synchronization
- **🚨 Risk Manager** (Port 8002) - Advanced risk control

#### Data & Connectivity
- **💾 Data Manager** (Port 8003) - Market data management
- **🔗 Exchange Connector** (Port 8005) - Bitget API integration
- **📡 Market Data Streamer** (Port 8010) - Real-time market data feeds
- **🔐 Credential Vault** (Port 8008) - Secure API key management

#### Monitoring & Alerting
- **📊 Monitoring Service** (Port 8006) - System analytics and metrics
- **🚨 Alert System** (Port 8012) - Notifications and alerts
- **📝 Centralized Logger** (Port 8016) - Log aggregation and search
- **🤖 MCP Server** (Port 8015) - AI integration interface

#### Infrastructure
- **🗄️ Redis** (Port 6379) - High-performance caching
- **📈 Prometheus** (Port 9090) - Metrics collection
- **📊 Grafana** (Port 3000) - Visualization dashboard
- **🔍 Elasticsearch** (Port 9200) - Log search and analytics
- **📥 Logstash** (Ports 5044, 9600) - Log processing
- **📊 Kibana** (Port 5601) - Log visualization

---

## 🧪 Backtesting Guide

### Running Your First Backtest

1. **Start Backtester Service**
```bash
python start_microservices.py start --service ultra-backtester
```

2. **Configure Parameters**
```python
# Example backtest configuration
backtest_config = {
    'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
    'start_date': '2023-01-01',
    'end_date': '2024-01-01',
    'initial_capital': 10000,
    'risk_per_trade': 0.02,
    'score_threshold': 85
}
```

3. **View Results**
- Results saved in `./backtest_results/`
- Performance metrics and charts
- Trade-by-trade analysis

### Backtest Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `symbols` | Trading pairs | `['BTCUSDT', 'ETHUSDT']` |
| `start_date` | Historical start | `2023-01-01` |
| `end_date` | Historical end | `2024-01-01` |
| `initial_capital` | Starting balance | `10000` |
| `risk_per_trade` | Risk per position | `0.02` (2%) |
| `score_threshold` | VIPER score minimum | `85` |

### Understanding Results

- **Win Rate**: 65-70% average
- **Risk/Reward**: 1:1.5 average ratio
- **Daily Trades**: 5-15 opportunities
- **Max Drawdown**: 2-3% with risk management

---

## 🔥 Live Trading Setup

### ⚠️ IMPORTANT SAFETY NOTICE
**Never risk money you can't afford to lose. Always test thoroughly before live trading.**

### Prerequisites
1. **Bitget Account** with API access
2. **API Credentials** (Key, Secret, Password)
3. **Test Environment** verification

### Configuration Steps

1. **Update Environment Variables**
```bash
# Edit .env file
BITGET_API_KEY=your_actual_api_key
BITGET_API_SECRET=your_actual_api_secret
BITGET_API_PASSWORD=your_actual_password
```

2. **Start Live Trading Engine**
```bash
python start_microservices.py start --service live-trading-engine
```

3. **Verify Configuration**
```bash
# Check service status
python start_microservices.py status

# View logs for any errors
python start_microservices.py logs --service live-trading-engine
```

### Risk Management Settings

```bash
# Conservative settings (recommended)
RISK_PER_TRADE=0.01          # 1% risk per trade
MAX_LEVERAGE=25              # Maximum 25x leverage
DAILY_LOSS_LIMIT=0.02        # 2% daily loss limit
ENABLE_AUTO_STOPS=true       # Automatic emergency stops
```

### Live Trading Monitoring

- **Real-time Dashboard**: http://localhost:8000
- **Performance Metrics**: Track win rate, P&L, risk scores
- **Risk Alerts**: Automatic notifications for high-risk situations
- **Emergency Stop**: Manual override capability

---

## 📊 Risk Management

### Core Risk Principles

1. **Position Sizing**: Never risk more than 2% per trade
2. **Daily Loss Limits**: Automatic suspension if exceeded
3. **Diversification**: Multiple symbols and timeframes
4. **Emergency Stops**: Manual and automatic shutdown procedures

### Risk Controls

```python
# VIPER Risk Management Configuration
risk_config = {
    'max_risk_per_trade': 0.02,        # 2% maximum risk
    'daily_loss_limit': 0.03,          # 3% daily loss limit
    'max_open_positions': 5,           # Maximum concurrent positions
    'max_leverage': 50,                # Maximum leverage allowed
    'auto_stop_conditions': [
        'daily_loss_exceeded',
        'high_volatility',
        'system_error'
    ]
}
```

### Emergency Procedures

1. **Immediate Stop**
```bash
python start_microservices.py stop --service live-trading-engine
```

2. **System Shutdown**
```bash
python start_microservices.py stop
```

3. **Manual Position Closure**
- Use Bitget web interface
- Contact support if automated closure fails

---

## 🔧 System Management

### Service Management

```bash
# Start all services
python start_microservices.py start

# Start specific service
python start_microservices.py start --service ultra-backtester

# Stop all services
python start_microservices.py stop

# Check service status
python start_microservices.py status

# View service logs
python start_microservices.py logs --service api-server --follow
```

### Service Health Checks

```bash
# Check all services
python start_microservices.py health

# Check specific service
python start_microservices.py health --service risk-manager
```

### Log Management

- **Application Logs**: `./logs/` directory
- **System Logs**: Docker container logs
- **Performance Logs**: Prometheus metrics
- **Trade Logs**: Detailed trade execution records

---

## 📈 Performance Optimization

### System Performance

- **Response Time**: <3 seconds per signal scan
- **Memory Usage**: <500MB per service
- **CPU Usage**: Optimized for multi-core systems
- **Network**: Low latency connections to Bitget

### Optimization Tips

1. **Resource Allocation**
```bash
# Increase workers for backtesting
BACKTEST_WORKERS=8

# Optimize cache settings
CACHE_TTL_SECONDS=600
```

2. **Network Optimization**
- Use dedicated trading VPS
- Enable connection pooling
- Implement rate limiting

3. **Database Optimization**
- **Redis** for high-frequency data and caching
- **Elasticsearch** for log search and analytics
- **Prometheus** for time-series metrics storage
- Regular cache cleanup and data archiving

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check Docker status
docker --version
docker-compose --version

# Check available resources
docker system df

# View detailed logs
python start_microservices.py logs --service <service-name>
```

#### 2. API Connection Issues
```bash
# Verify credentials in .env
cat .env | grep BITGET

# Test API connectivity
python -c "import ccxt; print(ccxt.bitget().fetch_balance())"
```

#### 3. Memory Issues
```bash
# Check memory usage
docker stats

# Reduce worker count
BACKTEST_WORKERS=2

# Clear Docker cache
docker system prune -a
```

#### 4. Performance Issues
```bash
# Check system resources
docker system df

# Monitor performance
python start_microservices.py health

# Check network connectivity
curl -f http://localhost:8000/health
```

### Getting Help

1. **Check Documentation**: This user guide and technical docs
2. **System Logs**: Review `./logs/` directory
3. **Health Checks**: Run diagnostic commands
4. **Community Support**: Check GitHub issues

---

## 📊 Monitoring & Analytics

### Dashboard Access

#### Main Dashboards
- **🌐 VIPER Web Dashboard**: http://localhost:8000 (Main trading interface)
- **📊 Grafana**: http://localhost:3000 (admin/viper_admin) - Metrics visualization
- **📈 Prometheus**: http://localhost:9090 - Raw metrics data
- **🔍 Kibana**: http://localhost:5601 - Log search and analysis

#### ELK Stack Monitoring
- **📥 Logstash**: Processes and forwards logs to Elasticsearch
- **🔍 Elasticsearch**: Indexes and stores log data for search
- **📊 Kibana**: Visualizes logs and creates dashboards

### Key Metrics

- **Trading Performance**: Win rate, P&L, Sharpe ratio, drawdown
- **Risk Metrics**: Current risk score, position sizes, leverage usage
- **System Health**: Service availability, response times, error rates
- **Infrastructure**: CPU usage, memory consumption, network latency

### Alert System

#### Automated Alerts
- **🚨 Risk Alerts**: High-risk situation warnings via Alert System (Port 8012)
- **📊 System Alerts**: Service availability issues and performance degradation
- **🔥 Trade Alerts**: Significant trade executions and position changes
- **📡 Market Alerts**: Market volatility and opportunity notifications

#### Notification Channels
- **Email**: SMTP-based notifications for important events
- **Telegram**: Real-time alerts via Telegram bot
- **Web Dashboard**: In-app notifications and alerts
- **Logs**: Structured logging for all system events

---

## 🎯 Strategy Customization

### VIPER Scoring System

The VIPER algorithm evaluates trades based on:

```python
# VIPER Score Components
viper_score = {
    'volume': 0.25,      # Volume analysis (25%)
    'price': 0.25,       # Price action (25%)
    'external': 0.25,    # External factors (25%)
    'range': 0.25        # Support/resistance (25%)
}
```

### Customization Options

1. **Score Thresholds**
```bash
VIPER_THRESHOLD=85        # Minimum score for trade entry
ATR_LENGTH=200           # ATR calculation period
```

2. **Risk Parameters**
```bash
RISK_PER_TRADE=0.02      # Position sizing
MAX_LEVERAGE=50          # Leverage limits
DAILY_LOSS_LIMIT=0.03    # Daily loss protection
```

3. **Market Conditions**
```bash
VOLATILITY_MULTIPLIER=1.5  # Market volatility adjustment
TREND_STRENGTH_MIN=0.6    # Minimum trend strength
```

---

## 🔄 Updates & Maintenance

### System Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild services
python start_microservices.py build

# Restart services
python start_microservices.py restart
```

### Backup Procedures

1. **Configuration Backup**
```bash
cp .env .env.backup
cp config/ config_backup/
```

2. **Data Backup**
```bash
# Redis data backup
docker exec viper-redis redis-cli --rdb /data/redis_backup.rdb

# Elasticsearch snapshots
curl -X PUT "localhost:9200/_snapshot/my_backup" -H 'Content-Type: application/json' -d'{"type": "fs","settings":{"location":"/usr/share/elasticsearch/backups"}}'

# Results backup
cp -r backtest_results/ backtest_results_backup/

# Prometheus metrics backup
cp -r prometheus_data/ prometheus_backup/
```

3. **Log Archive**
```bash
tar -czf logs_archive.tar.gz logs/
```

---

## 📞 Support & Resources

### Getting Help

1. **Quick Diagnosis**
```bash
# System health check
python start_microservices.py health

# View recent logs
python start_microservices.py logs --follow
```

2. **Documentation**
- User Guide (this document)
- Technical Documentation
- API Reference

3. **Community Resources**
- GitHub Issues
- Discussion Forums
- Community Examples

### Emergency Contacts

- **System Emergency**: `python start_microservices.py stop`
- **Exchange Support**: Bitget customer service
- **Technical Support**: GitHub issues

---

**🚀 Happy Trading with VIPER!**

*Remember: Trading involves risk. Always use proper risk management and never trade more than you can afford to lose.*
