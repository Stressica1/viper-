# 🚀 VIPER TRADING SYSTEM - DEPLOYMENT GUIDE
## Complete Production Deployment Manual

---

## 📋 TABLE OF CONTENTS
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Docker Deployment](#docker-deployment)
5. [Configuration](#configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)
10. [Emergency Procedures](#emergency-procedures)

---

## 🎯 SYSTEM OVERVIEW

### Architecture
The VIPER Trading System is a **microservices-based algorithmic trading platform** with:
- **29 Docker services** for complete functionality
- **Enterprise-grade security** with credential vault
- **Real-time monitoring** with Grafana/Prometheus
- **MCP GitHub integration** for automated task management
- **Multi-exchange support** (Bitget, Jordan Mainnet)

### Key Components
- **API Server**: FastAPI dashboard and REST endpoints
- **Ultra Backtester**: High-performance backtesting engine
- **Live Trading Engine**: Production trading with 50x leverage
- **Risk Manager**: Advanced position control and safety
- **MCP Server**: AI/ML integration and task orchestration
- **Credential Vault**: Secure secrets management
- **Monitoring Stack**: Comprehensive observability

---

## 📋 PREREQUISITES

### Hardware Requirements
```bash
# Minimum Requirements
CPU: 4 cores (8 recommended)
RAM: 8GB (16GB recommended)
Storage: 50GB SSD
Network: 100Mbps stable connection

# Recommended for Production
CPU: 8+ cores
RAM: 32GB+
Storage: 200GB+ NVMe SSD
Network: 1Gbps dedicated connection
```

### Software Requirements
```bash
# Operating System
Ubuntu 20.04+ or CentOS 8+
Docker Engine 24.0+
Docker Compose 2.20+

# Python Dependencies
Python 3.12+
pip 23.0+
virtualenv

# System Tools
git 2.30+
curl 7.68+
jq 1.6+
```

### Network Requirements
- **Inbound**: 8000-8021 (API services)
- **Outbound**: Full internet access for exchanges
- **Internal**: Docker network communication

---

## 🔧 ENVIRONMENT SETUP

### 1. System Preparation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install required packages
sudo apt install -y python3 python3-pip git curl jq htop iotop
```

### 2. Repository Setup
```bash
# Clone repository
git clone https://github.com/tradecomp/viper-.git
cd viper-

# Create required directories
mkdir -p logs config backtest_results performance_results

# Set proper permissions
chmod 755 scripts/*.py
chmod 644 config/*.json
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env

# Required variables:
BITGET_API_KEY=your_api_key_here
BITGET_API_SECRET=your_api_secret_here
BITGET_API_PASSWORD=your_api_password_here
GITHUB_PAT=github_pat_your_token_here
```

---

## 🐳 DOCKER DEPLOYMENT

### Production Deployment
```bash
# Full production deployment
docker-compose --env-file .env.docker -f docker/docker-compose.yml -f docker/docker-compose.override.yml up -d

# Check deployment status
docker-compose ps

# View logs
docker-compose logs -f api-server
```

### Service-Specific Deployment
```bash
# Deploy only essential services
docker-compose --env-file .env.docker up -d redis credential-vault api-server

# Deploy trading services
docker-compose --env-file .env.docker up -d exchange-connector risk-manager live-trading-engine

# Deploy monitoring stack
docker-compose --env-file .env.docker up -d prometheus grafana elasticsearch logstash kibana
```

### Scaling Services
```bash
# Scale backtesting workers
docker-compose up -d --scale ultra-backtester=4

# Scale monitoring services
docker-compose up -d --scale prometheus=2
```

---

## ⚙️ CONFIGURATION

### Environment Variables
```bash
# Trading Configuration
MAX_LEVERAGE=50
RISK_PER_TRADE=0.02
MAX_POSITIONS=15
DAILY_LOSS_LIMIT=0.03

# System Configuration
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379
API_SERVER_PORT=8000

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Docker Secrets
```bash
# Create Docker secrets
echo "your_bitget_api_key" | docker secret create bitget_api_key -
echo "your_github_pat" | docker secret create github_pat -

# Use secrets in compose
secrets:
  bitget_credentials:
    external: true
  github_pat:
    external: true
```

### Performance Tuning
```bash
# Redis optimization
redis:
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru

# Worker scaling
ultra-backtester:
  deploy:
    replicas: 4
    resources:
      limits:
        memory: 2G
        cpus: '2.0'
```

---

## 📊 MONITORING SETUP

### Grafana Dashboard Configuration
```bash
# Access Grafana
open http://localhost:3000
# Default credentials: admin / viper_secure_2025

# Import dashboards
# 1. Trading Performance Dashboard
# 2. System Health Dashboard
# 3. Risk Management Dashboard
# 4. MCP Activity Dashboard
```

### Prometheus Metrics
```yaml
# Key metrics to monitor
trading:
  active_positions: gauge
  total_pnl: counter
  win_rate: histogram
  max_drawdown: gauge

system:
  cpu_usage: gauge
  memory_usage: gauge
  disk_usage: gauge
  network_io: counter
```

### Alert Configuration
```yaml
# Critical alerts
- alert: HighDrawdown
  expr: trading_max_drawdown > 0.15
  for: 5m
  labels:
    severity: critical

- alert: SystemDown
  expr: up == 0
  for: 2m
  labels:
    severity: critical
```

---

## 🚀 PERFORMANCE OPTIMIZATION

### Automated Optimization
```bash
# Run performance optimization
python scripts/performance_optimization_tool.py

# Results will be saved to:
# - config/enhanced_risk_config.json (updated parameters)
# - performance_results/optimization_report_*.json
```

### Manual Parameter Tuning
```bash
# Adjust risk parameters
MAX_RISK_PER_TRADE=0.015
MAX_DAILY_LOSS=0.025
STOP_LOSS_PCT=0.018

# Adjust leverage settings
MAX_LEVERAGE=40
POSITION_SCALING=true

# Adjust technical indicators
FAST_MA_LENGTH=18
SLOW_MA_LENGTH=45
TREND_MA_LENGTH=180
```

### System Scaling
```bash
# Vertical scaling
docker-compose up -d --scale live-trading-engine=2

# Horizontal scaling
# Add more worker nodes for backtesting
docker-compose up -d --scale ultra-backtester=8
```

---

## 🔧 TROUBLESHOOTING

### Common Issues

#### 1. Container Startup Failures
```bash
# Check container logs
docker-compose logs <service_name>

# Check container status
docker-compose ps

# Restart specific service
docker-compose restart <service_name>
```

#### 2. Database Connection Issues
```bash
# Check Redis connectivity
docker-compose exec redis redis-cli ping

# Restart Redis
docker-compose restart redis

# Check Redis logs
docker-compose logs redis
```

#### 3. API Connection Problems
```bash
# Test exchange connectivity
docker-compose exec exchange-connector python -c "
import ccxt
exchange = ccxt.bitget()
print('Exchange status:', exchange.loadMarkets())
"
```

#### 4. Memory Issues
```bash
# Check memory usage
docker stats

# Adjust memory limits in docker-compose.override.yml
services:
  ultra-backtester:
    deploy:
      resources:
        limits:
          memory: 4G
```

#### 5. Network Connectivity
```bash
# Check network connectivity
docker network ls
docker network inspect viper_viper-network

# Restart network
docker-compose down
docker-compose up -d
```

---

## 🛠️ MAINTENANCE

### Regular Maintenance Tasks
```bash
# Daily maintenance
# 1. Check system health
docker-compose ps
docker stats

# 2. Review logs for errors
docker-compose logs --tail=100

# 3. Monitor performance metrics
# Access Grafana dashboards

# 4. Check disk usage
df -h
du -sh logs/ config/ backtest_results/
```

### Weekly Maintenance
```bash
# Update system
docker-compose pull
docker-compose up -d

# Clean old logs
find logs/ -name "*.log" -mtime +7 -delete

# Optimize database
docker-compose exec redis redis-cli BGREWRITEAOF

# Performance check
python scripts/performance_optimization_tool.py
```

### Monthly Maintenance
```bash
# Security updates
sudo apt update && sudo apt upgrade -y

# Backup configuration
cp -r config/ backups/config_$(date +%Y%m%d)/

# Review and rotate API keys
# Update Docker secrets if needed

# Performance audit
python scripts/system_health_check.py
```

---

## 🚨 EMERGENCY PROCEDURES

### Emergency Shutdown
```bash
# Immediate shutdown
docker-compose down

# Emergency stop trading
docker-compose exec live-trading-engine python -c "
from emergency_stop_system import EmergencyStopSystem
stopper = EmergencyStopSystem()
stopper.emergency_stop()
"

# Kill all trading processes
docker-compose kill live-trading-engine
```

### Data Recovery
```bash
# Restore from backup
cp -r backups/config_latest/* config/
cp -r backups/logs_latest/* logs/

# Restart services
docker-compose up -d

# Verify data integrity
python scripts/system_health_check.py
```

### Incident Response
1. **Assess Situation**: Check logs and monitoring dashboards
2. **Contain Damage**: Stop trading if necessary
3. **Investigate Root Cause**: Review logs and system state
4. **Implement Fix**: Apply patches or configuration changes
5. **Test Recovery**: Validate system functionality
6. **Document Incident**: Update incident log and improve procedures

---

## 📞 SUPPORT & RESOURCES

### Documentation
- **API Documentation**: `docs/api/`
- **Configuration Guide**: `docs/CONFIGURATION.md`
- **Troubleshooting Guide**: `docs/TROUBLESHOOTING.md`

### Monitoring Endpoints
- **API Server**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601

### Emergency Contacts
- **System Alerts**: Check Grafana alerts
- **GitHub Issues**: https://github.com/tradecomp/viper-/issues
- **Logs**: `logs/` directory
- **Backups**: `backups/` directory

---

## ✅ DEPLOYMENT CHECKLIST

### Pre-Deployment
- [x] System requirements met
- [x] Docker and Docker Compose installed
- [x] Repository cloned and configured
- [x] Environment variables set
- [x] API credentials configured
- [x] GitHub PAT configured

### Deployment
- [x] Docker images built successfully
- [x] Services started without errors
- [x] Database connections established
- [x] API endpoints responding
- [x] Monitoring dashboards accessible

### Post-Deployment
- [x] Performance optimization completed
- [x] Monitoring alerts configured
- [x] Backup strategy implemented
- [x] Documentation updated
- [x] Emergency procedures documented

---

## 🎯 FINAL NOTES

### Production Readiness
✅ **System Status**: PRODUCTION READY
✅ **Security**: ENTERPRISE GRADE
✅ **Monitoring**: COMPREHENSIVE
✅ **Documentation**: COMPLETE
✅ **Support**: MCP GITHUB INTEGRATION ACTIVE

### Key Metrics
- **Uptime Target**: 99.9%
- **Response Time**: <100ms API calls
- **Risk Control**: Automated position management
- **Performance**: Optimized for 50x leverage trading

### Next Steps
1. **Deploy to Production**: Use provided commands
2. **Monitor Performance**: Access Grafana dashboards
3. **Configure Alerts**: Set up notification channels
4. **Schedule Maintenance**: Automate regular tasks
5. **Scale as Needed**: Add resources based on load

---

*Deployment Guide Version: 2.0.0*
*Last Updated: 2025-08-29*
*Maintained by: VIPER Development Team*
