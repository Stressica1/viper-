# 🚀 VIPER Standalone Trading Component

## Complete Scan → Score → Trade → TP/SL Flow in One Script

The VIPER Standalone Trading Component is a comprehensive, self-contained trading system that integrates all essential trading workflows into a single Python script.

## ✨ Features

### 🔍 **Market Scanning**
- Monitors multiple cryptocurrency trading pairs simultaneously
- Fetches real-time market data from Bitget exchange
- Analyzes volume, price movements, and volatility
- Configurable scanning intervals (default: 30 seconds)

### 📊 **VIPER Scoring Algorithm**
- **Volume Score (30%)**: Evaluates liquidity and market interest
- **Price Score (35%)**: Measures momentum strength and direction  
- **External Score (20%)**: Considers execution costs (spreads)
- **Range Score (15%)**: Assesses volatility within optimal bounds
- Generates scores from 0-100 (higher = better opportunity)

### 💰 **Automated Trade Execution**
- Executes trades only for signals above configurable threshold (default: 85)
- Supports both LONG and SHORT positions
- Implements proper position sizing based on risk management
- Uses market orders for immediate execution

### 🛡️ **Risk Management**
- **Position Limits**: Maximum concurrent positions (default: 5)
- **Risk per Trade**: Configurable percentage of account balance (default: 2%)
- **Stop Loss**: Automatic loss prevention (default: 2% risk)
- **Take Profit**: Profit-taking levels (default: 4% gain)
- **Daily Loss Limits**: Circuit breakers for protection

### 📈 **Take Profit (TP) & Stop Loss (SL) Management**
- Real-time position monitoring
- Automatic execution of TP/SL orders
- Dynamic P&L tracking and reporting
- Graceful position closure on triggers

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Install required Python packages
pip install ccxt requests python-dotenv

# Or install all project dependencies
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the project root with your Bitget API credentials:

```env
# Required Bitget API credentials
BITGET_API_KEY=your_api_key_here
BITGET_API_SECRET=your_api_secret_here  
BITGET_API_PASSWORD=your_api_password_here

# Optional trading parameters (defaults shown)
MAX_POSITIONS=15
RISK_PER_TRADE=0.02
VIPER_THRESHOLD=85.0
SCAN_INTERVAL=30
STOP_LOSS_PERCENT=0.02
TAKE_PROFIT_PERCENT=0.04
DAILY_LOSS_LIMIT=0.05
```

### API Setup
1. Visit [Bitget API Management](https://www.bitget.com/en/account/newapi)
2. Create a new API key with trading permissions
3. Enable IP restrictions for security
4. Add your credentials to the `.env` file

## 🚀 Usage

### Basic Usage
```bash
# Run the standalone trader
python standalone_viper_trader.py
```

### Test Mode
```bash
# Run comprehensive tests without live trading
python test_standalone_trader.py
```

## 📊 Trading Workflow

### 1. Market Scanning Phase
```
🔍 Scanning 9 trading pairs...
📊 BTC/USDT:USDT: VIPER Score 87.3 → LONG Signal
📊 ETH/USDT:USDT: VIPER Score 82.1 → No Signal (below threshold)
🎯 Found 1 trading opportunities
```

### 2. Signal Processing Phase
```
🎯 Processing LONG signal for BTC/USDT:USDT
   VIPER Score: 87.3
   Confidence: 87.3%
   Entry Price: $50,234.56
   Stop Loss: $49,729.47
   Take Profit: $52,244.34
```

### 3. Trade Execution Phase
```
🎯 Executing LONG trade for BTC/USDT:USDT
   Size: 0.0398 BTC, Price: $50,234.56
   Stop Loss: $49,729.47, Take Profit: $52,244.34
✅ Trade executed successfully for BTC/USDT:USDT
   Order ID: 1234567890
```

### 4. Position Monitoring Phase
```
📊 Monitoring 1 active positions...
📈 BTC/USDT:USDT (BUY): $50,756.23 | P&L: 1.04%
🎯 Take Profit triggered for BTC/USDT:USDT
✅ Position closed for BTC/USDT:USDT
   Final P&L: 4.12%
   Reason: take_profit
```

## 🎛️ Configuration Options

### Trading Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_POSITIONS` | 15 | Maximum concurrent positions |
| `RISK_PER_TRADE` | 0.02 (2%) | Risk per individual trade |
| `VIPER_THRESHOLD` | 50.0 | Minimum score for trade signals (reduced due to stricter scoring) |
| `SCAN_INTERVAL` | 30 | Seconds between market scans |
| `STOP_LOSS_PERCENT` | 0.02 (2%) | Stop loss distance |
| `TAKE_PROFIT_PERCENT` | 0.04 (4%) | Take profit target |
| `DAILY_LOSS_LIMIT` | 0.05 (5%) | Maximum daily drawdown |

### Monitored Trading Pairs
Default pairs (easily customizable in code):
- BTC/USDT:USDT
- ETH/USDT:USDT  
- SOL/USDT:USDT
- BNB/USDT:USDT
- ADA/USDT:USDT
- DOT/USDT:USDT
- MATIC/USDT:USDT
- AVAX/USDT:USDT
- LINK/USDT:USDT

## 📈 VIPER Scoring Methodology

### Volume Score (25% weight)
- Evaluates trading volume relative to 1M threshold
- Higher volume indicates better liquidity
- Formula: `min((volume / 1_000_000) * 25, 100)`

### Price Score (30% weight)  
- Measures price momentum strength
- Considers absolute percentage change
- Formula: `min(abs(price_change) * 20, 100)`

### External Score (30% weight)
- **ENHANCED**: Execution cost awareness to prevent $3+ losses
- Calculates real execution cost including spread and market impact
- Zero score if execution cost ≥ $3.00 (prevents losing trades)
- Low score (30) if execution cost ≥ $2.00 
- Medium score (60) if execution cost ≥ $1.00
- High score for low-cost scenarios with improved spread sensitivity
- Formula: Execution cost-based scoring + `max(100 - (spread * 5000), 50)` for low costs

### Range Score (15% weight)
- Evaluates volatility within reasonable bounds
- Formula: `min(volatility * 10, 100) if volatility > 0.5%`

### Signal Generation Criteria
- **UPDATED**: Minimum VIPER score: 50 (reduced from 85 due to stricter execution cost scoring)
- **NEW**: Maximum execution cost: $3.00 (prevents losing trades)
- Minimum momentum: ±1.0% price change
- LONG signals: Price change > +1.0%
- SHORT signals: Price change < -1.0%
- **ENHANCED**: Smart order routing (LIMIT vs MARKET based on execution cost)
- **ENHANCED**: Dynamic stop/take-profit levels adjusted for execution costs

## 🛡️ Safety Features

### Position Limits
- Maximum concurrent positions enforced
- Prevents over-leverage and concentration risk
- Configurable limits based on account size

### Risk Controls
- Position sizing based on stop-loss distance
- Account balance validation before trades
- Emergency stop functionality (Ctrl+C)

### Monitoring & Alerts
- Real-time P&L tracking
- Comprehensive logging to file and console
- Position status updates every scan cycle

## 📝 Logging

The system creates detailed logs in `viper_trader.log`:
```
2024-01-15 10:30:15 | INFO     | 🚀 Starting VIPER Standalone Trading System...
2024-01-15 10:30:16 | INFO     | 📊 Monitoring 9 trading pairs
2024-01-15 10:30:16 | INFO     | ⏰ Scan interval: 30s
2024-01-15 10:30:17 | INFO     | 🔍 Scanning 9 trading pairs...
2024-01-15 10:30:18 | INFO     | 📊 BTC/USDT:USDT: VIPER Score 87.3 → LONG Signal
2024-01-15 10:30:19 | INFO     | ✅ Trade executed successfully for BTC/USDT:USDT
```

## 🚨 Important Warnings

### ⚠️ Financial Risk
- **This is live trading software that uses real money**
- Start with small amounts and test thoroughly
- Never risk more than you can afford to lose
- Past performance does not guarantee future results

### 🔐 Security
- Keep API credentials secure and never share them
- Use API key restrictions (IP whitelist, trading only)
- Enable 2FA on your exchange account
- Monitor your account regularly

### 🧪 Testing
- Always test with small amounts first
- Use Bitget's sandbox environment if available
- Verify all calculations and logic before live use
- Monitor positions actively when starting

## 🛠️ Customization

### Adding New Trading Pairs
Edit the `trading_pairs` list in the `StandaloneVIPERTrader.__init__()` method:
```python
self.trading_pairs = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
    'YOUR_PAIR/USDT:USDT'  # Add your pairs here
]
```

### Modifying VIPER Scoring
Adjust the scoring algorithm in the `calculate_viper_score()` method:
```python
# Weighted VIPER score - modify weights as needed
viper_score = (
    volume_score * 0.30 +     # Volume weight
    price_score * 0.35 +      # Momentum weight  
    external_score * 0.20 +   # Execution cost weight
    range_score * 0.15        # Volatility weight
)
```

### Custom Signal Logic
Modify signal generation in the `generate_signal()` method:
```python
# Custom momentum thresholds
if price_change > 2.0:  # Increase threshold
    signal = "LONG"
elif price_change < -2.0:
    signal = "SHORT"
```

## 📞 Support

For issues or questions:
1. Check the log files for error details
2. Run the test script to validate functionality
3. Verify API credentials and permissions
4. Review the configuration parameters

## 📄 License

This software is provided as-is for educational and research purposes. Use at your own risk.

---

**Remember: Trading cryptocurrencies carries significant financial risk. Always do your own research and never invest more than you can afford to lose.**