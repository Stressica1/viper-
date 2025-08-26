# 🚀 VIPER Trading Bot - Complete Setup Guide

This guide will help you set up and run the complete VIPER trading system with all microservices.

## 📋 Prerequisites

Before starting, ensure you have:

- **Docker and Docker Compose** installed on your system
- **Git** for cloning the repository
- **At least 8GB RAM** and **4 CPU cores** recommended
- **Bitget API credentials** (for live trading)

### Installing Docker

**Windows/Mac:**
```bash
# Download and install Docker Desktop from:
https://www.docker.com/products/docker-desktop
```

**Linux:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd Bitget-New

# Copy environment template
cp .env.template .env
```

### 2. Configure Environment

Edit the `.env` file with your settings:

```bash
# Required: Add your Bitget API credentials
BITGET_API_KEY=your_actual_api_key
BITGET_API_SECRET=your_actual_secret
BITGET_API_PASSWORD=your_actual_password

# Optional: Adjust other settings as needed
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### 3. Start the System

```bash
# Start all services
python start_microservices.py start

# Or use Docker Compose directly
docker-compose up -d
```

### 4. Access the System

#### Main Interfaces
- **🌐 VIPER Web Dashboard:** http://localhost:8000 (Main trading interface)
- **📊 Grafana:** http://localhost:3000 (admin/viper_admin) - Metrics visualization
- **📈 Prometheus:** http://localhost:9090 - Raw metrics data
- **🔍 Kibana:** http://localhost:5601 - Log search and analysis
- **🔍 Elasticsearch:** http://localhost:9200 - REST API for search

#### Service Health Checks
- **API Server Health:** http://localhost:8000/health
- **MCP Server Health:** http://localhost:8015/health
- **All Services Status:** Check via `python start_microservices.py status`

## 🏗️ System Architecture

VIPER consists of 17 specialized microservices working together:

### Core Services

1. **🌐 API Server** (Port 8000)
   - Web dashboard and REST API
   - Service coordination and status monitoring

2. **🧪 Ultra Backtester** (Port 8001)
   - Historical data backtesting
   - Performance metrics calculation
   - Monte Carlo simulations

3. **🎯 Strategy Optimizer** (Port 8004)
   - Parameter optimization using genetic algorithms
   - Grid search and walk-forward analysis

4. **🔥 Live Trading Engine** (Port 8007)
   - Real-time automated trading execution
   - Integration with Bitget API

### Trading Services

5. **🎯 Signal Processor** (Port 8011)
   - VIPER scoring algorithm implementation
   - Real-time signal generation and filtering
   - Technical indicator calculations

6. **📋 Order Lifecycle Manager** (Port 8013)
   - Complete order management from creation to execution
   - Order tracking and status updates
   - Error handling and retry logic

7. **🔄 Position Synchronizer** (Port 8014)
   - Real-time position synchronization
   - Exchange position reconciliation
   - Position drift detection and correction

8. **🚨 Risk Manager** (Port 8002)
   - Real-time risk monitoring and control
   - Position limits and auto-stop mechanisms
   - Risk assessment and alerts

### Data & Connectivity

9. **💾 Data Manager** (Port 8003)
   - Market data synchronization
   - Redis caching and persistence
   - Historical data management

10. **🔗 Exchange Connector** (Port 8005)
    - Bitget API client with rate limiting
    - Order management and position tracking
    - Secure credential management

11. **📡 Market Data Streamer** (Port 8010)
    - Real-time market data feeds
    - WebSocket connection management
    - Data preprocessing and normalization

12. **🔐 Credential Vault** (Port 8008)
    - Secure API key storage and management
    - Access token generation and validation
    - Encrypted credential storage

### Monitoring & Alerting

13. **📊 Monitoring Service** (Port 8006)
    - System metrics collection and analysis
    - Performance monitoring and alerting
    - Health check coordination

14. **🚨 Alert System** (Port 8012)
    - Multi-channel notification system
    - Email and Telegram alert delivery
    - Configurable alert rules and thresholds

15. **📝 Centralized Logger** (Port 8016)
    - Log aggregation and processing
    - Structured logging across all services
    - Log forwarding to Elasticsearch

16. **🤖 MCP Server** (Port 8015)
    - Model Context Protocol implementation
    - AI agent integration interface
    - Standardized API for automation

### Infrastructure Components

17. **🗄️ Redis** (Port 6379)
    - High-performance caching and messaging
    - Session storage and pub/sub communication
    - Data persistence and caching

18. **📈 Prometheus** (Port 9090)
    - Metrics collection and time-series storage
    - Service monitoring and alerting
    - Performance data aggregation

19. **📊 Grafana** (Port 3000)
    - Visualization and dashboard creation
    - Metrics visualization and analysis
    - Custom dashboard development

20. **🔍 Elasticsearch** (Port 9200)
    - Log search and analytics
    - Full-text search capabilities
    - Log aggregation and indexing

21. **📥 Logstash** (Ports 5044, 9600)
    - Log processing and transformation
    - Data pipeline management
    - Log forwarding to Elasticsearch

22. **📊 Kibana** (Port 5601)
    - Log visualization and dashboard
    - Search interface for logs
    - Analytics and reporting tools

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# API Configuration
BITGET_API_KEY=your_key
BITGET_API_SECRET=your_secret
BITGET_API_PASSWORD=your_password

# Service Ports
API_SERVER_PORT=8000
ULTRA_BACKTESTER_PORT=8001
# ... etc

# Risk Management
RISK_PER_TRADE=0.02
DAILY_LOSS_LIMIT=0.03
MAX_LEVERAGE=50

# Performance Tuning
BACKTEST_WORKERS=4
CACHE_TTL_SECONDS=300
```

### Bitget API Setup

1. Go to [Bitget API Management](https://www.bitget.com/en/account/newapi)
2. Create a new API key
3. Enable the following permissions:
   - **Read Info** ✅
   - **Spot & Margin Trading** ✅ (if using spot/margin)
   - **Futures Trading** ✅ (if using futures)
   - **Read Info** ✅
4. Set IP restrictions for security
5. Copy the API Key, Secret, and Password to your `.env` file

## 🎯 Usage Guide

### Starting Services

```bash
# Start all services
python start_microservices.py start

# Start specific service
python start_microservices.py start api-server

# Check status
python start_microservices.py status

# View logs
python start_microservices.py logs

# Stop services
python start_microservices.py stop
```

### Running Backtests

```bash
# Via API
curl -X POST http://localhost:8001/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "initial_balance": 10000
  }'
```

### Strategy Optimization

```bash
# Via API
curl -X POST http://localhost:8004/api/optimization/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "method": "genetic",
    "parameter_ranges": {
      "sma_20_period": [10, 15, 20, 25, 30],
      "sma_50_period": [40, 45, 50, 55, 60]
    }
  }'
```

### Risk Management

```bash
# Check risk status
curl http://localhost:8002/api/risk/status

# Calculate position size
curl -X POST http://localhost:8002/api/position/size \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USDT:USDT",
    "price": 45000,
    "balance": 10000,
    "risk_per_trade": 0.02
  }'
```

## 📊 Monitoring

### Health Checks

```bash
# Check all services
curl http://localhost:8006/api/services/health

# Check specific service
curl http://localhost:8006/api/services/api-server/health

# System metrics
curl http://localhost:8006/api/metrics
```

### Dashboard Access

#### Monitoring Dashboards
- **🌐 Main Dashboard:** http://localhost:8000
- **📊 Grafana:** http://localhost:3000 (admin/viper_admin) - System metrics
- **📈 Prometheus:** http://localhost:9090 - Metrics collection
- **🔍 Kibana:** http://localhost:5601 - Log analysis and search
- **🔍 Elasticsearch:** http://localhost:9200 - Log storage and search API

#### ELK Stack Integration
- **📥 Logstash** processes logs from all services
- **🔍 Elasticsearch** indexes and stores log data
- **📊 Kibana** provides visualization and search interface
- **📝 Centralized Logger** aggregates logs from all microservices

## 🔍 Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check Docker logs
docker-compose logs

# Check specific service
docker-compose logs api-server

# Check all service health
python start_microservices.py health

# View infrastructure status
docker-compose ps
```

**API connection issues:**
```bash
# Test Bitget API connectivity
curl -X GET "http://localhost:8005/api/ticker/BTC/USDT:USDT"

# Test exchange connector health
curl http://localhost:8005/health

# Check credential vault
curl http://localhost:8008/health
```

**MCP Server issues:**
```bash
# Check MCP server health
curl http://localhost:8015/health

# Test MCP capabilities
curl http://localhost:8015/capabilities

# Check MCP server logs
docker-compose logs mcp-server
```

**ELK Stack issues:**
```bash
# Check Elasticsearch health
curl http://localhost:9200/_cluster/health

# Check Kibana status
curl http://localhost:5601/api/status

# Check Logstash pipeline
curl http://localhost:9600/_node/stats
```

**Memory issues:**
```bash
# Check resource usage
docker stats

# Monitor specific services
docker stats api-server ultra-backtester

# Restart services
docker-compose restart
```

**Port conflicts:**
```bash
# Check used ports
netstat -tulpn | grep LISTEN

# Check Docker port mappings
docker-compose ps

# Change ports in .env file
```

### Logs and Debugging

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api-server

# Check Redis
docker exec -it viper-redis redis-cli

# Check Elasticsearch indices
curl http://localhost:9200/_cat/indices

# Search logs in Kibana
open http://localhost:5601

# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=up

# View Grafana dashboards
open http://localhost:3000

# Check service health via MCP
curl http://localhost:8015/health

# Monitor all services status
python start_microservices.py status
```

#### Log Analysis with ELK Stack

```bash
# Search for errors across all services
curl -X GET "localhost:9200/viper-logs-*/_search?q=level:ERROR"

# View logs for specific service
curl -X GET "localhost:9200/viper-logs-*/_search?q=service:api-server"

# Check log volume by service
curl -X GET "localhost:9200/viper-logs-*/_search" -H 'Content-Type: application/json' -d'
{
  "size": 0,
  "aggs": {
    "services": {
      "terms": {"field": "service.keyword"}
    }
  }
}'
```

## 🔒 Security Considerations

### Production Deployment

1. **Use strong passwords** for all services
2. **Enable IP restrictions** on API keys
3. **Use HTTPS** in production
4. **Regular backup** of data and configurations
5. **Monitor logs** for suspicious activity

### API Key Security

- Never commit API keys to version control
- Use environment variables for sensitive data
- Rotate API keys regularly
- Enable 2FA on exchange accounts

## 📈 Performance Optimization

### Resource Allocation

```yaml
# docker-compose.yml adjustments
services:
  ultra-backtester:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
```

### Scaling Services

```bash
# Scale specific services
docker-compose up -d --scale ultra-backtester=3

# Check resource usage
docker stats
```

## 🔄 Updates and Maintenance

### Updating the System

```bash
# Pull latest changes
git pull origin main

# Rebuild services
docker-compose build

# Restart services
docker-compose up -d
```

### Backup and Recovery

```bash
# Backup Redis data
docker exec viper-redis redis-cli --rdb /data/redis_backup_$(date +%Y%m%d).rdb

# Create Elasticsearch snapshot
curl -X PUT "localhost:9200/_snapshot/viper_backup_$(date +%Y%m%d)" -H 'Content-Type: application/json' -d'{"indices": "viper-*","ignore_unavailable": true,"include_global_state": false}'

# Backup Prometheus metrics data
cp -r prometheus_data/ prometheus_backup_$(date +%Y%m%d)/

# Backup Grafana dashboards and configurations
cp -r grafana_data/ grafana_backup_$(date +%Y%m%d)/

# Backup configurations
cp .env .env.backup.$(date +%Y%m%d)

# Backup all logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/

# Complete system backup
tar -czf viper_full_backup_$(date +%Y%m%d).tar.gz \
  backtest_results/ \
  data/ \
  logs/ \
  config/ \
  .env

# Restore from backup
docker-compose down

# Restore Redis data
docker volume rm viper-redis_data
docker-compose up -d redis
docker exec viper-redis redis-cli --rdb /data/redis_backup.rdb

# Restore full system
docker-compose up -d
```

## 📚 Additional Resources

- [User Guide](docs/USER_GUIDE.md)
- [Technical Documentation](docs/TECHNICAL_DOC.md)
- [API Documentation](http://localhost:8000/docs)
- [Bitget API Documentation](https://bitgetlimited.github.io/apidoc/en/mix/)

## 🆘 Support

If you encounter issues:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review the logs using `docker-compose logs`
3. Check the [GitHub issues](https://github.com/your-repo/issues)
4. Consult the [documentation](docs/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
