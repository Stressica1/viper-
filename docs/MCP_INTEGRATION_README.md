# 🔗 VIPER MCP LIVE TRADING INTEGRATION

Complete GitHub MCP integration for automated live trading operations.

## 🚀 Quick Start

### 1. Start the Complete MCP System
```bash
python mcp_trading_integration.py --mode start
```

### 2. Monitor in Real-time
```bash
python mcp_trading_monitor.py --mode interactive
```

### 3. Launch Individual Components
```bash
# Start MCP trading system
python start_mcp_live_trading.py

# Monitor performance
python mcp_performance_tracker.py
```

## 📊 Available Commands

### System Control
- `python mcp_trading_integration.py --mode start` - Start complete system
- `python mcp_trading_integration.py --mode status` - System status
- `python mcp_trading_integration.py --mode emergency` - Emergency shutdown

### Trading Operations
- `python start_mcp_live_trading.py --mode auto` - Auto-start trading
- `python start_mcp_live_trading.py --mode interactive` - Interactive mode

### Monitoring
- `python mcp_trading_monitor.py --mode monitor` - Real-time monitoring
- `python mcp_trading_monitor.py --mode interactive` - Interactive monitoring

## 🎯 MCP Task Management

### Create Trading Task
```python
from mcp_live_trading_connector import create_live_trading_task

config = {
    'trading_pairs': ['BTCUSDT', 'ETHUSDT'],
    'max_positions': 5,
    'risk_per_trade': 0.02,
    'leverage': 25
}

task_id = await create_live_trading_task(config)
```

### Start/Stop Tasks
```python
from mcp_live_trading_connector import start_live_trading_task, stop_live_trading_task

await start_live_trading_task(task_id)
await stop_live_trading_task(task_id)
```

## ⚙️ Configuration

### Trading Tasks (`mcp_trading_tasks.json`)
```json
{
  "mcp_trading_tasks": {
    "default_config": {
      "trading_pairs": ["BTCUSDT", "ETHUSDT"],
      "max_positions": 5,
      "risk_per_trade": 0.02,
      "leverage": 25
    },
    "scheduled_tasks": [
      {
        "id": "daily_scalping",
        "name": "Daily Scalping Strategy",
        "schedule": "0 9 * * 1-5",
        "active": true
      }
    ]
  }
}
```

### Environment Variables
```bash
# GitHub Integration
export GITHUB_TOKEN=your_github_token

# Trading Parameters
export MAX_POSITIONS=15
export RISK_PER_TRADE=0.02
export LEVERAGE=50
export TRADING_PAIRS="BTCUSDT,ETHUSDT"

# Risk Management
export MIN_BALANCE_THRESHOLD=100.0
```

## 🔧 Components Overview

### 1. MCP Live Trading Connector (`mcp_live_trading_connector.py`)
- Core trading engine with WebSocket support
- Task scheduling and management
- Emergency controls and safety features

### 2. Performance Tracker (`mcp_performance_tracker.py`)
- Real-time performance monitoring
- Automated GitHub reporting
- Risk metrics calculation

### 3. Trading Monitor (`mcp_trading_monitor.py`)
- Real-time system monitoring
- Interactive control interface
- Status dashboard

### 4. Complete Integration (`mcp_trading_integration.py`)
- Unified system launcher
- Component orchestration
- Health monitoring

## 📈 Performance Features

### Automated Reporting
- Daily performance summaries
- GitHub issue creation for alerts
- Performance data export/import

### Risk Management
- Emergency stop protocols
- Daily loss limits
- Drawdown monitoring

### Metrics Tracked
- Win rate and profit factor
- Sharpe ratio and volatility
- Maximum drawdown
- Total P&L and commissions

## 🚨 Safety Features

### Emergency Controls
```python
# Emergency stop all operations
from mcp_live_trading_connector import MCPLiveTradingConnector

connector = MCPLiveTradingConnector()
await connector.emergency_stop_all()
```

### Risk Thresholds
- Daily loss limits
- Maximum drawdown alerts
- Win rate monitoring
- Component health checks

## 🔗 WebSocket API

### Real-time Monitoring
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8765');

// Send commands
ws.send(JSON.stringify({
  'type': 'get_status'
}));

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Status:', data);
};
```

### Available Commands
- `get_status` - Get system status
- `create_task` - Create trading task
- `start_task` - Start trading task
- `stop_task` - Stop trading task
- `emergency_stop` - Emergency shutdown

## 📋 Cron Scheduling

### Task Scheduling
```bash
# Daily at 9 AM (weekdays)
0 9 * * 1-5

# Every 4 hours
0 */4 * * *

# Monday at 10 AM
0 10 * * 1
```

## 🎯 Trading Strategies

### Pre-configured Tasks
1. **Daily Scalping** - High-frequency with tight stops
2. **Swing Trading** - Medium-term positions
3. **Trend Following** - Long-term trend strategies

### Custom Tasks
Create custom trading tasks by modifying `mcp_trading_tasks.json` with your strategy parameters.

## 🔍 Troubleshooting

### Common Issues
1. **Connection Failed** - Check WebSocket port 8765
2. **GitHub Token** - Verify GITHUB_TOKEN environment variable
3. **Trading Pairs** - Ensure valid pair symbols

### Logs
- `mcp_integration.log` - Main integration logs
- `mcp_live_trading.log` - Trading operations
- `performance_export.json` - Performance data

## 🚀 Advanced Usage

### Custom Performance Metrics
```python
from mcp_performance_tracker import MCPPerformanceTracker

tracker = MCPPerformanceTracker()
# Add custom metrics
await tracker.github_mcp.log_system_performance({
    'custom_metric': 'value'
})
```

### Integration with External Systems
```python
# Webhook notifications
# API integrations
# Custom alerting systems
```

---

## 📞 Support

For issues or questions:
1. Check the logs in the project directory
2. Review GitHub issues for known problems
3. Verify configuration in `mcp_trading_tasks.json`
4. Test individual components before full integration

---

*VIPER Ultimate Trading System - MCP Integration v1.0.0*
