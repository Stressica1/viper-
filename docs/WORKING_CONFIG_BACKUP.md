# 🚀 VIPER TRADING SYSTEM - WORKING CONFIGURATION BACKUP
# ✅ SUCCESSFUL HEDGE MODE + 50x LEVERAGE SETUP

## 📅 BACKUP CREATED: $(date)
## 🎯 STATUS: LIVE TRADING ACTIVE WITH SUCCESSFUL TRADES

---

## 🔐 WORKING BITGET API CONFIGURATION

### API Credentials (Environment Variables)
```bash
BITGET_API_KEY=[BITGET_API_KEY]
BITGET_API_SECRET=your_bitget_api_secret_here
BITGET_API_PASSWORD=22672267
```

### Exchange Configuration (Hedge Mode)
```python
exchange = ccxt.bitget({
    'apiKey': api_key,
    'secret': api_secret,
    'password': api_password,
    'options': {
        'defaultType': 'swap',  # Perpetual swaps
        'adjustForTimeDifference': True,
    },
    'sandbox': False,  # Live trading
})
```

---

## 🏦 WORKING ORDER EXECUTION (HEDGE MODE)

### Successful Order Parameters
```python
# For BUY/LONG positions
order = exchange.create_order(
    symbol,
    'market',
    'buy',
    size,
    None,
    params={'tradeSide': 'open'}  # Critical for hedge mode
)

# For SELL/SHORT positions
order = exchange.create_order(
    symbol,
    'market',
    'sell',
    size,
    None,
    params={'tradeSide': 'open'}  # Critical for hedge mode
)
```

---

## 🎯 WORKING POSITION SIZING

### Current Balance: $96.04 USDT
### Position Size Calculation:
```python
# For 0.001 BTC minimum contract size
min_contract_size = 0.001  # 0.001 BTC minimum
max_trade_value = balance * 0.05  # 5% of balance per trade
contract_value = min_contract_size * price

max_contracts = max_trade_value / contract_value
position_size = max(min_contract_size, min(max_contracts * min_contract_size, balance * 0.1 / price))
```

---

## 🔧 WORKING DOCKER COMPOSE CONFIGURATION

### Live Trading Setup:
```yaml
version: '3.8'
services:
  live-trading-engine:
    build:
      context: ./services/live-trading-engine
      dockerfile: Dockerfile
    ports:
      - "${LIVE_TRADING_ENGINE_PORT:-8007}:8000"
    environment:
      - BITGET_API_KEY=${BITGET_API_KEY}
      - BITGET_API_SECRET=${BITGET_API_SECRET}
      - BITGET_API_PASSWORD=${BITGET_API_PASSWORD}
      - MAX_LEVERAGE=${MAX_LEVERAGE:-50}
      - RISK_PER_TRADE=${RISK_PER_TRADE:-0.02}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    volumes:
      - ./logs:/app/logs
```

---

## 📊 WORKING ENVIRONMENT VARIABLES

### Complete .env Configuration:
```bash
# VIPER Trading Bot - Secure Environment Configuration
VAULT_MASTER_KEY=e501753005a89adc89b419457a5aa7b12bb5c01aa570177c50b2

# Access tokens for all services (comma-separated)
VAULT_ACCESS_TOKENS=c0b25118b40ac34959df9ea2ffc04089eb8138932c1a44e8c920f01547b1e4a6,0052ead6c6466fb4067cb9afc9b1d53630987ca740b9f7406339a64e1bf86d4e,181400690d5aa52ab882557cef482fa156a82751e7491c8db850ec83a63b63e2,7f0b0c08d4e33c8c8fa675ae9ab5e32c9b81e6ebf3f96d71401ca703b6a596e5,2984968205df8465525d5d60fc9899baf16c8832d576023433d10b65fd3befb1,8b059281b8d59e08b498b541f9164464d20745fc23ce044ab0e16312edfa0c92,685feb571934e99864f2e5e95fd2c3b69eb09a308feb173cef093a1bf9e33955,3d1124faef727bff64f2e5e95fd2c3b69eb09a308feb173cef093a1bf9e33955

VAULT_URL=http://credential-vault:8008

# Individual service access tokens
VAULT_ACCESS_TOKEN_API_SERVER=c0b25118b40ac34959df9ea2ffc04089eb8138932c1a44e8c920f01547b1e4a6
VAULT_ACCESS_TOKEN_ULTRA_BACKTESTER=0052ead6c6466fb4067cb9afc9b1d53630987ca740b9f7406339a64e1bf86d4e
VAULT_ACCESS_TOKEN_STRATEGY_OPTIMIZER=181400690d5aa52ab882557cef482fa156a82751e7491c8db850ec83a63b63e2
VAULT_ACCESS_TOKEN_LIVE_TRADING_ENGINE=7f0b0c08d4e33c8c8fa675ae9ab5e32c9b81e6ebf3f96d71401ca703b6a596e5
VAULT_ACCESS_TOKEN_DATA_MANAGER=2984968205df8465525d5d60fc9899baf16c8832d576023433d10b65fd3befb1
VAULT_ACCESS_TOKEN_EXCHANGE_CONNECTOR=8b059281b8d59e08b498b541f9164464d20745fc23ce044ab0e16312edfa0c92
VAULT_ACCESS_TOKEN_RISK_MANAGER=685feb571934e99864f2e5e95fd2c3b69eb09a308feb173cef093a1bf9e33955
VAULT_ACCESS_TOKEN_MONITORING_SERVICE=3d1124faef727bff64f2e5e95fd2c3b69eb09a308feb173cef093a1bf9e33955

# LIVE TRADING CREDENTIALS - REMOVE AFTER SETUP
BITGET_API_KEY=[BITGET_API_KEY]
BITGET_API_SECRET=your_bitget_api_secret_here
BITGET_API_PASSWORD=22672267

# Service Ports
API_SERVER_PORT=8000
ULTRA_BACKTESTER_PORT=8001
RISK_MANAGER_PORT=8002
DATA_MANAGER_PORT=8003
STRATEGY_OPTIMIZER_PORT=8004
EXCHANGE_CONNECTOR_PORT=8005
MONITORING_SERVICE_PORT=8006
LIVE_TRADING_ENGINE_PORT=8007
CREDENTIAL_VAULT_PORT=8008

# Risk Management
RISK_PER_TRADE=0.02
MAX_LEVERAGE=50
DAILY_LOSS_LIMIT=0.03
MAX_POSITION_SIZE_PERCENT=0.1

# Trading Parameters
VIPER_THRESHOLD=85
BACKTEST_WORKERS=4

# Service URLs
LIVE_TRADING_ENGINE_URL=http://live-trading-engine:8000
EXCHANGE_CONNECTOR_URL=http://exchange-connector:8000
RISK_MANAGER_URL=http://risk-manager:8000

# Docker Configuration
DOCKER_MODE=true
COMPOSE_PROJECT_NAME=viper-trading
```

---

## ✅ VERIFICATION OF WORKING SYSTEM

### Last Successful Trade:
- **Order ID**: 1344187646598586369
- **Type**: SELL 0.001 BTC
- **Price**: $110,249.90
- **Status**: ✅ EXECUTED
- **Timestamp**: 2025-08-26 10:47:09

### System Health:
- ✅ Bitget API Connection: ACTIVE
- ✅ Swaps Wallet Balance: $96.04 USDT
- ✅ Live Trading Engine: RUNNING
- ✅ Market Data Streaming: ACTIVE
- ✅ VIPER Signals: ACTIVE
- ✅ Redis Cache: OPERATIONAL
- ✅ Hedge Mode: ENABLED

---

## 🚨 CRITICAL WORKING COMPONENTS

1. **Hedge Mode Parameter**: `params={'tradeSide': 'open'}` - REQUIRED for Bitget hedge mode
2. **Position Size**: 0.001 BTC minimum for swaps
3. **Exchange Type**: 'swap' for perpetual contracts
4. **Order Type**: 'market' for immediate execution
5. **Leverage**: Set to 50x in environment variables

---

## 📋 REPRODUCTION STEPS

To recreate this working configuration:

1. **Set Environment Variables**:
   ```bash
   export BITGET_API_KEY="[BITGET_API_KEY]"
   export BITGET_API_SECRET="your_bitget_api_secret_here"
   export BITGET_API_PASSWORD="22672267"
   export MAX_LEVERAGE=50
   ```

2. **Start Services**:
   ```bash
   docker-compose -f docker-compose.live.yml up -d
   ```

3. **Verify Connection**:
   ```bash
   docker logs viper-trading-live-trading-engine-1 --tail 10
   ```

---

## 🔍 DIAGNOSTIC INFORMATION

### Successful Order Execution Log:
```
2025-08-26 10:47:09,508 - __main__ - INFO - 🚀 EXECUTING SELL ORDER: 0.001000 BTC on BTC/USDT:USDT
2025-08-26 10:47:09,692 - __main__ - INFO - ✅ SELL order executed successfully: ID 1344187646598586369
2025-08-26 10:47:09,693 - __main__ - INFO - ✅ Trade data stored in Redis
```

### Market Data Working:
```
2025-08-26 10:47:09,259 - __main__ - INFO - 📊 REAL DATA: Current price: $110249.90
```

---

## 💡 KEY SUCCESS FACTORS

1. **Hedge Mode Enabled** in Bitget account settings
2. **Correct tradeSide parameter** for opening positions
3. **Minimum contract size** respected (0.001 BTC)
4. **Proper exchange initialization** with swap type
5. **Environment variables** correctly configured
6. **Redis connection** for trade storage

---

## 🚀 QUICK START (From this backup)

```bash
# 1. Load environment variables
cp .env .env.backup
source .env

# 2. Start live trading
docker-compose -f docker-compose.live.yml up -d

# 3. Check logs
docker logs viper-trading-live-trading-engine-1 --tail 20

# 4. Access dashboard
open http://localhost:8000
```

**🎉 WORKING CONFIGURATION SUCCESSFULLY BACKED UP!**

This backup contains all the working parameters for successful live trading with Bitget hedge mode and 50x leverage support.
