# 🚀 VIPER Trading Bot - Complete Installation Guide

**Get your algorithmic trading system running in 5 minutes!**

This guide will walk you through the complete installation process for the VIPER Trading Bot. After following these steps, you'll be able to plug in your API keys and start trading immediately.

---

## 📋 Table of Contents

1. [Prerequisites](#-prerequisites)
2. [Quick Installation (Recommended)](#-quick-installation-recommended)
3. [Manual Installation](#-manual-installation)
4. [API Key Setup](#-api-key-setup)
5. [System Verification](#-system-verification)
6. [First Run](#-first-run)
7. [Troubleshooting](#-troubleshooting)
8. [Next Steps](#-next-steps)

---

## 🔧 Prerequisites

Before installing VIPER, ensure you have the following installed on your system:

### Required Software

| Software | Version | Download Link | Purpose |
|----------|---------|---------------|---------|
| **Docker Desktop** | Latest | [Docker.com](https://www.docker.com/products/docker-desktop/) | Container runtime |
| **Python** | 3.11+ | [Python.org](https://www.python.org/downloads/) | Core runtime |
| **Git** | Latest | [Git-scm.com](https://git-scm.com/downloads) | Repository management |

### System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space minimum
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Network**: Stable internet connection for live trading

### Bitget Trading Account (For Live Trading)

1. Create account at [Bitget.com](https://www.bitget.com/)
2. Complete KYC verification
3. Enable 2FA security
4. Create API keys (we'll do this together later)

---

## ⚡ Quick Installation (Recommended)

**This is the fastest way to get VIPER running!**

### Step 1: Download and Setup

```bash
# Clone the repository
git clone https://github.com/Stressica1/viper-.git
cd viper-

# Run the automated setup script
python setup.py
```

The setup script will:
- ✅ Install all Python dependencies
- ✅ Set up Docker containers
- ✅ Configure environment files
- ✅ Validate your system
- ✅ Guide you through API key setup

### Step 2: Add Your API Keys

The setup script will prompt you for your Bitget API credentials. If you don't have them yet, see [API Key Setup](#-api-key-setup) below.

### Step 3: Start Trading

```bash
# Start all services
python scripts/start_microservices.py start

# Open your browser to http://localhost:8000
```

**That's it! 🎉 Your VIPER system is now running!**

---

## 🔨 Manual Installation

If you prefer to install manually or the quick installation doesn't work:

### Step 1: Clone Repository

```bash
git clone https://github.com/Stressica1/viper-.git
cd viper-
```

### Step 2: Install Python Dependencies

```bash
# Install core dependencies
pip install -e .

# Install optional MCP dependencies (recommended)
pip install -e ".[mcp]"

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Step 3: Set Up Environment

```bash
# Copy the environment template
cp .env.template .env

# Edit the .env file with your settings
nano .env  # or use your preferred editor
```

### Step 4: Configure Docker

```bash
# Start Docker Desktop first, then:
docker-compose -f infrastructure/docker-compose.yml pull
```

### Step 5: Configure API Keys

```bash
# Run the API configuration wizard
python scripts/configure_api.py
```

---

## 🔐 API Key Setup

### Creating Bitget API Keys

1. **Login to Bitget**
   - Go to [Bitget API Management](https://www.bitget.com/en/account/newapi)
   - Log in to your account

2. **Create New API Key**
   - Click "Create API Key"
   - Choose "System-generated API Key" (recommended)

3. **Set Permissions**
   - ✅ **Read Info** - Required for account data
   - ✅ **Spot & Margin Trading** - For spot trading
   - ✅ **Futures Trading** - For futures trading
   - ❌ **Transfer** - Not recommended for security
   - ❌ **Withdraw** - Not recommended for security

4. **Security Settings**
   - **IP Restriction**: Add your IP address for security
   - **API Password**: Set a strong password (you'll need this)

5. **Save Your Credentials**
   - **API Key**: Starts with `bg_`
   - **API Secret**: Long string of characters
   - **API Password**: The password you set

### Adding Keys to VIPER

You have three options to add your API keys:

#### Option 1: Interactive Configuration (Recommended)

```bash
python scripts/configure_api.py
```

This will securely prompt you for your credentials and configure everything automatically.

#### Option 2: Environment File

Edit the `.env` file and replace these lines:

```bash
BITGET_API_KEY=your_bitget_api_key_here
BITGET_API_SECRET=your_bitget_api_secret_here
BITGET_API_PASSWORD=your_bitget_api_password_here
```

With your actual credentials:

```bash
BITGET_API_KEY=bg_your_actual_key_here
BITGET_API_SECRET=your_actual_secret_here
BITGET_API_PASSWORD=your_actual_password_here
```

#### Option 3: Secure Credential Manager

```bash
python scripts/setup_credentials.py setup
```

---

## ✅ System Verification

After installation, verify everything is working:

### Check Dependencies

```bash
# Verify Python dependencies
python -c "import fastapi, uvicorn, redis, ccxt, pandas, numpy, aiohttp; print('✅ All dependencies installed')"

# Check Docker
docker --version
docker-compose --version
```

### Validate Configuration

```bash
# Run system validation
python scripts/quick_validation.py

# Check environment setup
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('✅ Environment loaded' if os.getenv('BITGET_API_KEY') else '❌ API keys not set')"
```

### Test API Connection

```bash
# Test Bitget API connection (only if you have real API keys)
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

if not os.getenv('BITGET_API_KEY', '').startswith('your_'):
    import ccxt
    exchange = ccxt.bitget({
        'apiKey': os.getenv('BITGET_API_KEY'),
        'secret': os.getenv('BITGET_API_SECRET'),
        'password': os.getenv('BITGET_API_PASSWORD'),
        'sandbox': True  # Use sandbox for testing
    })
    try:
        balance = exchange.fetch_balance()
        print('✅ API connection successful')
    except Exception as e:
        print(f'❌ API connection failed: {e}')
else:
    print('⚠️ Using placeholder API keys - live trading disabled')
"
```

---

## 🚀 First Run

### Start the System

```bash
# Start all microservices
python scripts/start_microservices.py start

# Check status
python scripts/start_microservices.py status
```

You should see output like:
```
✅ API Server (Port 8000) - Running
✅ Ultra Backtester (Port 8001) - Running  
✅ Risk Manager (Port 8002) - Running
✅ Data Manager (Port 8003) - Running
... and more services
```

### Access the Dashboard

1. Open your web browser
2. Go to: **http://localhost:8000**
3. You should see the VIPER Trading Dashboard

### Run Your First Backtest

1. In the dashboard, go to "Backtesting"
2. Select a symbol (e.g., BTC/USDT)
3. Choose date range (last 30 days)
4. Click "Start Backtest"
5. View results in real-time!

### Test MCP Integration

```bash
# Test MCP server
curl http://localhost:8015/health

# Test with Python client
python -c "
from src.clients.viper_mcp_client import VIPERMCPClient
client = VIPERMCPClient()
result = client.get_portfolio_status()
print('✅ MCP integration working' if result else '❌ MCP not responding')
"
```

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### Docker Issues

**Problem**: "Docker daemon not running"
```bash
# Solution: Start Docker Desktop
# Windows: Start Docker Desktop application
# macOS: Start Docker Desktop from Applications
# Linux: sudo systemctl start docker
```

**Problem**: "Port already in use"
```bash
# Solution: Stop conflicting services
python scripts/start_microservices.py stop
# Wait 30 seconds, then try again
python scripts/start_microservices.py start
```

#### Python Issues

**Problem**: "Module not found"
```bash
# Solution: Reinstall dependencies
pip install --upgrade pip
pip install -e .
pip install -e ".[mcp]"
```

**Problem**: "Permission denied"
```bash
# Solution: Fix permissions
chmod +x scripts/*.py
# On Windows, run CMD as administrator
```

#### API Connection Issues

**Problem**: "Invalid API key"
- ✅ Double-check your API key format (should start with `bg_`)
- ✅ Verify permissions are set correctly on Bitget
- ✅ Check IP restrictions if enabled
- ✅ Try regenerating API keys

**Problem**: "Connection timeout"
- ✅ Check your internet connection
- ✅ Verify firewall settings
- ✅ Try using sandbox mode first

#### Service Startup Issues

**Problem**: Services won't start
```bash
# Check logs
docker-compose -f infrastructure/docker-compose.yml logs

# Reset everything
python scripts/start_microservices.py stop
docker system prune -f
python scripts/start_microservices.py start
```

### Getting Help

If you're still having issues:

1. **Check Logs**: Look in `logs/` directory for detailed error messages
2. **System Status**: Run `python scripts/start_microservices.py status`
3. **Validation**: Run `python scripts/quick_validation.py`
4. **Documentation**: Check `docs/` folder for detailed guides
5. **Issues**: Create a GitHub issue with your error logs

---

## 🎯 Next Steps

Congratulations! You now have VIPER running. Here's what to do next:

### 1. **Explore the Dashboard**
- View real-time market data
- Run backtests on different symbols
- Monitor system performance
- Check risk management settings

### 2. **Paper Trading (Recommended First)**
- Test strategies with paper trading
- Verify all systems work correctly
- Get comfortable with the interface
- Fine-tune risk parameters

### 3. **Go Live (When Ready)**
- Start with small position sizes
- Monitor closely for the first few trades
- Gradually increase position sizes
- Set up alerts and monitoring

### 4. **Advanced Features**
- Configure custom strategies
- Set up automated alerts
- Use MCP AI integration
- Implement custom risk rules

### 5. **Performance Optimization**
- Scale services based on usage
- Optimize parameters for your trading style
- Set up automated backups
- Configure monitoring and alerting

---

## 📚 Additional Resources

- **[User Guide](docs/USER_GUIDE.md)** - Complete usage documentation
- **[API Setup Guide](docs/API_SETUP_README.md)** - Detailed API configuration
- **[Technical Documentation](docs/TECHNICAL_DOC.md)** - System architecture
- **[MCP Integration Guide](docs/MCP_INTEGRATION_GUIDE.md)** - AI integration
- **[Risk Management](docs/RISK_MANAGEMENT_IMPLEMENTATION.md)** - Safety guidelines

---

## ⚠️ Important Safety Notes

- **Start Small**: Begin with small position sizes
- **Paper Trade First**: Test thoroughly before live trading
- **Risk Management**: Never risk more than you can afford to lose
- **API Security**: Keep your API keys secure and private
- **Regular Monitoring**: Check your positions regularly
- **Emergency Stops**: Know how to stop all trading immediately

---

**🎉 Happy Trading with VIPER!**

*The world's most advanced algorithmic trading platform, now made simple.*

---

**Need Help?** Create an issue on GitHub or check the documentation in the `docs/` folder.