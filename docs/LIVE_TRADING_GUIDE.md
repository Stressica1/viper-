# 🚀 VIPER LIVE TRADING BOT - QUICK START GUIDE

## ⚠️ IMPORTANT: LIVE TRADING INVOLVES REAL MONEY RISK

**This system will execute real trades on cryptocurrency exchanges. Only proceed if you:**
- Understand the risks of automated trading
- Have sufficient funds for trading
- Are prepared for potential losses
- Have tested with paper trading first

---

## 📋 PREREQUISITES

### 1. Exchange Account Setup
- **Bitget Account**: Required (primary exchange)
- **API Keys**: Trading permissions enabled
- **Testnet Testing**: Recommended before live trading

### 2. System Requirements
- **Python 3.8+**: Latest version recommended
- **Stable Internet**: Reliable connection required
- **System Resources**: 4GB RAM minimum, 2 CPU cores
- **GitHub Account**: For MCP integration (optional but recommended)

---

## 🔧 SETUP INSTRUCTIONS

### Step 1: Configure Exchange Credentials

1. **Get Bitget API Credentials:**
   - Visit: https://www.bitget.com/account/newapi
   - Create new API key with trading permissions
   - **Important**: Enable spot trading and futures trading
   - Note down: API Key, Secret Key, and Passphrase

2. **Update Credentials File:**
   ```bash
   # Edit the credentials file
   nano config/exchange_credentials.json
   ```

   Replace the placeholder values:
   ```json
   {
     "bitget": {
       "api_key": "YOUR_ACTUAL_API_KEY",
       "secret_key": "YOUR_ACTUAL_SECRET_KEY",
       "passphrase": "YOUR_ACTUAL_PASSPHRASE",
       "enabled": true,
       "testnet": false
     }
   }
   ```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Additional dependencies for live trading
pip install python-dotenv psutil ccxt
```

### Step 3: Test System Setup

```bash
# Test the live trading launcher (without trading)
python launch_live_trading.py --test-only

# Check system diagnostics
python scripts/system_health_check.py
```

---

## 🚀 LAUNCH LIVE TRADING BOT

### Quick Launch (Recommended)

```bash
# Launch with full system validation
python launch_live_trading.py
```

### Alternative Launch Methods

```bash
# Direct async trader launch
python viper_async_trader.py

# Live trading manager
python scripts/live_trading_manager.py

# Complete system launcher
python scripts/start_live_trading_complete.py
```

### Launch Options

```bash
# Launch with specific configuration
python launch_live_trading.py --config production

# Launch with risk limits
python launch_live_trading.py --max-drawdown 0.05

# Launch with GitHub MCP disabled
python launch_live_trading.py --no-mcp
```

---

## 📊 SYSTEM STATUS CHECKLIST

Before launching, ensure:

- ✅ **Credentials**: Valid API keys configured
- ✅ **Connection**: Exchange connectivity confirmed
- ✅ **Balance**: Sufficient funds available
- ✅ **Risk**: Parameters set appropriately
- ✅ **Health**: System resources adequate
- ✅ **Backup**: Emergency stop configured

---

## 🎯 WHAT HAPPENS WHEN YOU LAUNCH

### 1. Pre-Launch Validation
- 🔐 **Credentials Check**: Validates API keys
- 🔌 **Exchange Test**: Confirms connectivity
- 🛡️ **Risk Setup**: Configures safety parameters
- 🔗 **MCP Init**: GitHub integration (if enabled)
- 🏥 **Health Check**: System resource validation

### 2. Launch Sequence
```
🚀 VIPER LIVE TRADING BOT LAUNCHER
==============================================
🔥 Advanced automated trading system
🎯 High-performance execution
🛡️ Risk-managed operations
==============================================

🤖 System Status:
   • Credentials: ✅ Valid
   • Exchange: ✅ Connected
   • Risk Management: ✅ Active
   • GitHub MCP: ✅ Active
==============================================

🎯 Starting live trading operations...
📊 Press Ctrl+C to stop safely
```

### 3. Active Trading
- **Concurrent Processing**: Multiple trading strategies
- **Real-time Monitoring**: Position tracking
- **Risk Management**: Automatic position control
- **Performance Logging**: GitHub MCP integration
- **Emergency Stops**: Circuit breaker systems

---

## 🛑 STOPPING THE BOT

### Safe Shutdown (Recommended)

```bash
# Press Ctrl+C in the terminal running the bot
# The system will gracefully close all positions and shutdown
```

### Emergency Stop

```bash
# Kill all trading processes
pkill -f viper

# Or use the emergency stop script
python scripts/emergency_stop_system.py
```

---

## 📈 MONITORING & LOGGING

### Live Monitoring
- **Console Output**: Real-time trading activity
- **Log Files**: `live_trading.log` and `async_trader.log`
- **GitHub Issues**: Automated performance reports
- **System Health**: Resource usage monitoring

### Performance Tracking
```bash
# View live performance
python scripts/strategy_metrics_dashboard.py

# Check system health
python scripts/system_health_check.py

# View trading logs
tail -f live_trading.log
```

---

## ⚙️ CONFIGURATION OPTIONS

### Risk Management
```python
# In launch_live_trading.py
risk_config = {
    'max_drawdown': 0.05,      # 5% max drawdown
    'max_position_size': 0.02, # 2% max position size
    'daily_loss_limit': 0.03,  # 3% daily loss limit
    'max_open_positions': 5,   # Maximum concurrent positions
    'emergency_stop_enabled': True
}
```

### Trading Parameters
```python
# Position sizing and leverage
position_size_pct = 0.02      # 2% of portfolio per trade
leverage = 5                  # 5x leverage (if using futures)
min_trade_size = 10           # Minimum trade size in USD
max_trade_size = 1000         # Maximum trade size in USD
```

---

## 🚨 EMERGENCY PROCEDURES

### If Something Goes Wrong

1. **Immediate Stop**: Press `Ctrl+C` or run emergency stop
2. **Check Logs**: Review `live_trading.log` for errors
3. **Verify Positions**: Check exchange account manually
4. **Contact Support**: If automated systems fail

### Recovery Steps

```bash
# 1. Stop all processes
pkill -f viper

# 2. Check system status
python scripts/system_health_check.py

# 3. Manual position verification
# Check your exchange account directly

# 4. Restart with conservative settings
python launch_live_trading.py --conservative
```

---

## 📊 PERFORMANCE EXPECTATIONS

### Realistic Goals
- **Win Rate**: 55-65% (realistic for automated systems)
- **Risk/Reward**: 1:1.5 to 1:2 ratio
- **Max Drawdown**: 5-10% (with proper risk management)
- **Monthly Returns**: 5-15% (depending on market conditions)

### Important Notes
- **Past Performance ≠ Future Results**
- **Markets are unpredictable**
- **Always use stop losses**
- **Start small, scale gradually**
- **Monitor regularly**

---

## 🆘 SUPPORT & TROUBLESHOOTING

### Common Issues

**❌ "Credentials not configured"**
- Solution: Update `config/exchange_credentials.json`

**❌ "Exchange connection failed"**
- Check internet connection
- Verify API keys have correct permissions
- Try testnet first

**❌ "High resource usage"**
- Reduce concurrent workers
- Check system requirements
- Close other applications

**❌ "GitHub MCP not available"**
- This is optional, trading will work without it
- Check GitHub token if needed

### Getting Help
1. Check logs in `live_trading.log`
2. Review system diagnostics
3. Test with smaller position sizes first
4. Use testnet for practice

---

## 🎯 FINAL CHECKLIST

- [ ] Exchange credentials configured correctly
- [ ] API keys have trading permissions
- [ ] Sufficient funds in account
- [ ] Risk parameters set appropriately
- [ ] System health verified
- [ ] Emergency stop tested
- [ ] Backup systems ready

**Ready to launch? Run:**
```bash
python launch_live_trading.py
```

**Remember: This is live trading with real money. Trade responsibly! 🚀**

---

*VIPER Live Trading System v2.0.0*
*Advanced automated cryptocurrency trading with AI/ML optimization*
