# ⚙️ VIPER TRADING SYSTEM - CONFIGURATION GUIDE
## Complete Configuration Manual

---

## 📋 TABLE OF CONTENTS
1. [Configuration Overview](#configuration-overview)
2. [Environment Variables](#environment-variables)
3. [Docker Configuration](#docker-configuration)
4. [Trading Parameters](#trading-parameters)
5. [Risk Management](#risk-management)
6. [Technical Indicators](#technical-indicators)
7. [Monitoring & Alerts](#monitoring--alerts)
8. [Performance Tuning](#performance-tuning)
9. [Security Configuration](#security-configuration)
10. [Backup & Recovery](#backup--recovery)

---

## 🎯 CONFIGURATION OVERVIEW

The VIPER Trading System uses a **layered configuration approach**:

### Configuration Hierarchy
```
1. Environment Variables (.env) - Highest priority
2. Docker Secrets - Secure credential storage
3. Configuration Files (config/*.json) - System parameters
4. Default Values - Fallback settings
```

### Configuration Files
```
config/
├── enhanced_risk_config.json      # Risk management parameters
├── enhanced_technical_config.json # Technical analysis settings
├── exchange_credentials.json      # Exchange API credentials
├── performance_monitoring_config.json # Monitoring settings
└── enhanced_ai_ml_config.json     # AI/ML parameters
```

---

## 🔧 ENVIRONMENT VARIABLES

### Core System Variables
```bash
# System Configuration
LOG_LEVEL=INFO
DOCKER_MODE=true
REDIS_URL=redis://redis:6379

# Service Ports
API_SERVER_PORT=8000
ULTRA_BACKTESTER_PORT=8001
RISK_MANAGER_PORT=8002
LIVE_TRADING_ENGINE_PORT=8007
CREDENTIAL_VAULT_PORT=8008
```

### Trading Configuration
```bash
# Risk Parameters
MAX_LEVERAGE=50
RISK_PER_TRADE=0.02
MAX_POSITIONS=15
DAILY_LOSS_LIMIT=0.03
ENABLE_AUTO_STOPS=true

# Trading Behavior
TRADING_MODE=PRODUCTION
SCAN_INTERVAL=30
MAX_TRADES_PER_HOUR=20
MIN_VOLUME_THRESHOLD=10000
```

### Exchange Configuration
```bash
# Bitget API (Primary Exchange)
BITGET_API_KEY=your_api_key_here
BITGET_API_SECRET=your_api_secret_here
BITGET_API_PASSWORD=your_api_password_here
EXCHANGE_TYPE=BITGET_V2

# Jordan Mainnet (Secondary Exchange)
JORDAN_MAINNET_KEY=your_jordan_key_here
JORDAN_MAINNET_SECRET=your_jordan_secret_here
JORDAN_MAINNET_PASSPHRASE=your_jordan_passphrase_here
```

### GitHub MCP Integration
```bash
# GitHub Configuration
GITHUB_PAT=github_pat_your_token_here
GITHUB_OWNER=tradecomp
GITHUB_REPO=viper
GITHUB_BRANCH=main

# MCP Server Configuration
MCP_HOST=0.0.0.0
MCP_PORT=8015
MCP_WORKERS=4
MCP_MAX_CONNECTIONS=100
```

### Monitoring Configuration
```bash
# Prometheus & Grafana
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=viper_secure_2025

# Elasticsearch Stack
ELASTICSEARCH_PORT=9200
LOGSTASH_PORT=5044
KIBANA_PORT=5601

# Alert Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
FROM_EMAIL=viper@tradingbot.com
TO_EMAILS=alerts@tradecomp.com
```

---

## 🐳 DOCKER CONFIGURATION

### Docker Compose Structure
```yaml
# Base Configuration
version: '3.8'
services:
  # Core Services
  redis:               # Caching and session storage
  credential-vault:    # Secure secrets management
  api-server:          # REST API and dashboard

  # Trading Services
  exchange-connector:      # Exchange API client
  risk-manager:           # Position control and safety
  live-trading-engine:    # Production trading

  # Analysis Services
  ultra-backtester:       # Backtesting engine
  strategy-optimizer:     # Parameter optimization
  viper-scoring-service:  # Signal scoring

  # Monitoring Services
  prometheus:            # Metrics collection
  grafana:              # Dashboard and visualization
  elasticsearch:        # Log search and analytics
  logstash:             # Log processing
  kibana:               # Log visualization

  # Advanced Services
  mcp-server:                  # AI/ML integration
  market-data-streamer:        # Real-time data feed
  signal-processor:            # Trading signal generation
  alert-system:                # Notification system
  order-lifecycle-manager:     # Complete order management
  position-synchronizer:       # Real-time position sync
```

### Resource Configuration
```yaml
# Production Resource Limits
services:
  live-trading-engine:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'

  ultra-backtester:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'
```

### Network Configuration
```yaml
# Secure Network Isolation
networks:
  viper-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: viper_bridge

  # External networks for specific services
  monitoring:
    external: true
```

---

## 📊 TRADING PARAMETERS

### Risk Management Parameters
```json
{
  "risk_management": {
    "max_risk_per_trade": 0.02,
    "max_positions": 15,
    "max_daily_loss": 0.03,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.06,
    "trailing_stop_pct": 0.015,
    "max_leverage": 50,
    "min_position_size": 0.001,
    "max_position_size_pct": 0.1,
    "enable_auto_stops": true,
    "emergency_stop_enabled": true
  }
}
```

### Trading Strategy Parameters
```json
{
  "trading_strategy": {
    "min_viper_score": 70.0,
    "scan_interval": 30,
    "max_trades_per_hour": 20,
    "min_volume_threshold": 10000,
    "max_spread_threshold": 0.001,
    "pairs_batch_size": 20,
    "enable_market_orders": true,
    "enable_limit_orders": false,
    "slippage_tolerance": 0.001,
    "confidence_threshold": 0.7
  }
}
```

### Leverage and Position Management
```json
{
  "position_management": {
    "leverage": 50,
    "leverage_scaling": true,
    "max_concurrent_positions": 15,
    "position_size_scaling": "fixed_percentage",
    "fixed_position_size": 0.02,
    "percentage_of_balance": 0.1,
    "min_position_size": 0.0001,
    "max_position_size": 1.0,
    "enable_position_averaging": false,
    "max_additional_positions": 2
  }
}
```

---

## 🛡️ RISK MANAGEMENT

### Stop Loss Configuration
```json
{
  "stop_loss": {
    "enabled": true,
    "type": "percentage",
    "percentage": 0.02,
    "fixed_amount": 10.0,
    "trailing_stop": {
      "enabled": true,
      "activation_pct": 0.01,
      "trail_pct": 0.005
    },
    "emergency_stop": {
      "enabled": true,
      "trigger_conditions": [
        "max_daily_loss",
        "max_drawdown",
        "api_errors",
        "network_timeout"
      ],
      "auto_close_positions": true,
      "notification_enabled": true
    }
  }
}
```

### Daily Loss Limits
```json
{
  "daily_limits": {
    "max_daily_loss_pct": 0.03,
    "max_daily_loss_amount": 100.0,
    "max_trades_per_day": 100,
    "max_volume_per_day": 100000,
    "trading_hours": {
      "enabled": true,
      "start_time": "00:00",
      "end_time": "23:59",
      "timezone": "UTC"
    },
    "weekend_trading": false,
    "holiday_trading": false
  }
}
```

### Circuit Breakers
```json
{
  "circuit_breakers": {
    "enabled": true,
    "max_consecutive_losses": 3,
    "cool_down_period": 300,
    "volatility_breaker": {
      "enabled": true,
      "threshold": 0.05,
      "period": 3600
    },
    "drawdown_breaker": {
      "enabled": true,
      "threshold": 0.10,
      "reset_period": 86400
    }
  }
}
```

---

## 📈 TECHNICAL INDICATORS

### Moving Average Configuration
```json
{
  "moving_averages": {
    "fast_ma": {
      "period": 21,
      "type": "SMA",
      "source": "close"
    },
    "slow_ma": {
      "period": 50,
      "type": "EMA",
      "source": "close"
    },
    "trend_ma": {
      "period": 200,
      "type": "SMA",
      "source": "close"
    },
    "signal_ma": {
      "period": 9,
      "type": "SMA",
      "source": "close"
    }
  }
}
```

### RSI Configuration
```json
{
  "rsi": {
    "enabled": true,
    "period": 14,
    "overbought_level": 70,
    "oversold_level": 30,
    "signal_strength": 0.8,
    "divergence_detection": true,
    "centerline_crosses": true
  }
}
```

### Bollinger Bands
```json
{
  "bollinger_bands": {
    "enabled": true,
    "period": 20,
    "standard_deviations": 2.0,
    "source": "close",
    "squeeze_detection": true,
    "bandwidth_threshold": 0.1,
    "mean_reversion_signals": true
  }
}
```

### MACD Configuration
```json
{
  "macd": {
    "enabled": true,
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9,
    "histogram_smoothing": 3,
    "signal_threshold": 0.001,
    "divergence_signals": true,
    "zero_line_crosses": true
  }
}
```

---

## 📊 MONITORING & ALERTS

### Prometheus Metrics Configuration
```yaml
# Key Metrics
scrape_configs:
  - job_name: 'viper-trading'
    static_configs:
      - targets: ['api-server:8000', 'live-trading-engine:8007']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'viper-system'
    static_configs:
      - targets: ['monitoring-service:8006']
    scrape_interval: 30s
```

### Alert Rules
```yaml
groups:
  - name: trading_alerts
    rules:
      - alert: HighDrawdown
        expr: trading_max_drawdown > 0.15
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High drawdown detected"

      - alert: LowWinRate
        expr: trading_win_rate < 0.4
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low win rate detected"
```

### Log Configuration
```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
      "console": {
        "enabled": true,
        "level": "INFO"
      },
      "file": {
        "enabled": true,
        "filename": "logs/viper.log",
        "max_bytes": 10485760,
        "backup_count": 5
      },
      "elasticsearch": {
        "enabled": true,
        "hosts": ["elasticsearch:9200"],
        "index": "viper-logs"
      }
    }
  }
}
```

---

## ⚡ PERFORMANCE TUNING

### Database Optimization
```yaml
# Redis Configuration
redis:
  maxmemory: 512mb
  maxmemory-policy: allkeys-lru
  tcp-keepalive: 300
  timeout: 300

# Connection Pool Settings
connection_pool:
  max_connections: 20
  max_idle_time: 300
  retry_on_failure: true
  retry_delay: 1.0
```

### API Rate Limiting
```json
{
  "rate_limiting": {
    "enabled": true,
    "requests_per_second": 10,
    "burst_limit": 20,
    "backoff_factor": 1.5,
    "max_retries": 3,
    "retry_delay": 1.0,
    "circuit_breaker": {
      "failure_threshold": 5,
      "recovery_timeout": 60,
      "expected_exception": ["ConnectionError", "TimeoutError"]
    }
  }
}
```

### Memory Optimization
```json
{
  "memory_management": {
    "gc_threshold": 1000,
    "max_cache_size": 1000,
    "cache_ttl": 300,
    "memory_monitoring": true,
    "memory_threshold": 0.8,
    "cleanup_interval": 300,
    "object_pooling": true,
    "lazy_loading": true
  }
}
```

---

## 🔒 SECURITY CONFIGURATION

### Credential Management
```yaml
# Docker Secrets
secrets:
  bitget_credentials:
    file: ./.env.docker
    environment: "BITGET_API_KEY,BITGET_API_SECRET,BITGET_API_PASSWORD"

  github_pat:
    file: ./.env.docker
    environment: "GITHUB_PAT"

  vault_master_key:
    file: ./.env.docker
    environment: "VAULT_MASTER_KEY"
```

### Access Control
```json
{
  "access_control": {
    "api_authentication": {
      "enabled": true,
      "token_expiration": 3600,
      "refresh_token_enabled": true,
      "max_login_attempts": 5,
      "lockout_duration": 900
    },
    "ip_whitelisting": {
      "enabled": true,
      "allowed_ips": ["192.168.1.0/24", "10.0.0.0/8"],
      "block_duration": 3600
    },
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 60,
      "burst_limit": 10
    }
  }
}
```

### Encryption Configuration
```json
{
  "encryption": {
    "algorithm": "AES-256-GCM",
    "key_rotation": true,
    "key_rotation_interval": 2592000,
    "backup_encryption": true,
    "data_at_rest_encryption": true,
    "tls_enabled": true,
    "certificate_validation": true
  }
}
```

---

## 💾 BACKUP & RECOVERY

### Automated Backup Configuration
```json
{
  "backup": {
    "enabled": true,
    "schedule": "0 2 * * *",
    "retention_days": 30,
    "compression": true,
    "encryption": true,
    "locations": {
      "local": "/opt/viper/backups",
      "remote": "s3://viper-backups",
      "cloud": "gcs://viper-backups"
    },
    "components": {
      "database": true,
      "logs": true,
      "config": true,
      "models": true,
      "performance_data": true
    }
  }
}
```

### Recovery Procedures
```json
{
  "recovery": {
    "automated_recovery": true,
    "recovery_time_objective": 300,
    "recovery_point_objective": 3600,
    "failover_configuration": {
      "enabled": true,
      "primary_region": "us-east-1",
      "secondary_region": "us-west-2",
      "dns_failover": true,
      "load_balancer_failover": true
    },
    "data_recovery": {
      "point_in_time_recovery": true,
      "cross_region_replication": true,
      "automated_testing": true
    }
  }
}
```

---

## 🚀 ADVANCED CONFIGURATION

### AI/ML Configuration
```json
{
  "ai_ml": {
    "enabled": true,
    "model_update_interval": 86400,
    "feature_engineering": {
      "enabled": true,
      "lag_features": [1, 3, 5, 10],
      "rolling_features": [5, 10, 20],
      "technical_indicators": true,
      "market_data_features": true
    },
    "model_configuration": {
      "algorithm": "ensemble",
      "ensemble_method": "voting",
      "models": ["random_forest", "gradient_boosting", "neural_network"],
      "hyperparameter_tuning": true,
      "cross_validation": true
    }
  }
}
```

### Custom Strategy Configuration
```json
{
  "custom_strategies": {
    "enabled": true,
    "strategy_directory": "/opt/viper/strategies",
    "auto_discovery": true,
    "backtest_before_live": true,
    "risk_assessment_required": true,
    "performance_monitoring": true,
    "max_custom_strategies": 10,
    "strategy_timeout": 300
  }
}
```

---

## 📋 CONFIGURATION VALIDATION

### Validation Script
```bash
# Validate configuration
python scripts/validate_configuration.py

# Check for common issues
python scripts/configuration_checker.py

# Generate configuration report
python scripts/config_report_generator.py
```

### Configuration Testing
```bash
# Test trading configuration
python scripts/test_trading_config.py

# Test risk management
python scripts/test_risk_management.py

# Test monitoring setup
python scripts/test_monitoring_setup.py
```

---

## 🎯 CONFIGURATION BEST PRACTICES

### 1. Environment Separation
```bash
# Use different configurations for different environments
cp .env .env.production
cp .env .env.staging
cp .env .env.development

# Environment-specific settings
if [ "$ENVIRONMENT" = "production" ]; then
    MAX_LEVERAGE=25
    RISK_PER_TRADE=0.01
else
    MAX_LEVERAGE=50
    RISK_PER_TRADE=0.02
fi
```

### 2. Parameter Optimization
```bash
# Regular parameter optimization
python scripts/performance_optimization_tool.py

# Automated parameter updates
# Schedule this to run weekly
0 2 * * 1 python scripts/performance_optimization_tool.py
```

### 3. Configuration Monitoring
```bash
# Monitor configuration changes
python scripts/config_change_monitor.py

# Alert on configuration drift
python scripts/config_drift_detector.py
```

### 4. Documentation Maintenance
```bash
# Keep documentation updated
python scripts/update_documentation.py

# Validate documentation accuracy
python scripts/validate_documentation.py
```

---

## 📞 SUPPORT & RESOURCES

### Configuration Help
- **Configuration Validator**: `scripts/validate_configuration.py`
- **Parameter Optimizer**: `scripts/performance_optimization_tool.py`
- **Configuration Report**: `scripts/config_report_generator.py`

### Emergency Configuration
```bash
# Emergency configuration reset
python scripts/emergency_config_reset.py

# Safe mode configuration
python scripts/safe_mode_config.py

# Minimal configuration for testing
python scripts/minimal_config.py
```

---

*Configuration Guide Version: 2.0.0*
*Last Updated: 2025-08-29*
*Maintained by: VIPER Configuration Team*
