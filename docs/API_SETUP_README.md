# 🚀 VIPER Trading Bot - API Setup Guide

## 🔐 Secure API Credential Configuration

This guide shows you how to set up your Bitget API credentials for live trading with the VIPER system.

---

## 📋 Quick Setup (3 Steps)

### Step 1: Get Your Bitget API Credentials

1. **Go to Bitget API Management**
   ```
   https://www.bitget.com/en/account/newapi
   ```

2. **Create New API Key**
   - Click "Create API Key"
   - Choose "System Generated" for security

3. **Configure Permissions**
   - ✅ **Read Info** - Required for account data
   - ✅ **Spot & Margin Trading** - For spot trading
   - ✅ **Futures Trading** - For futures trading (if needed)
   - ✅ **Enable Reading** - For market data

4. **Security Settings**
   - ✅ **IP Restriction** - Add your IP for security
   - ✅ **Enable 2FA** on your Bitget account
   - ❌ **Disable Withdrawals** - Not needed for trading

5. **Save Credentials**
   - **API Key**: `bg_xxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - **API Secret**: `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - **API Password**: `xxxxxx`

---

### Step 2: Run Configuration Wizard

**Start Docker Desktop first**, then run:

```bash
# Interactive setup wizard
python configure_api.py
```

**What it will ask:**
```
🔑 Bitget API Configuration
1️⃣  API Key: [enter your API key]
2️⃣  API Secret: [enter your API secret]
3️⃣  API Password: [enter your API password]
```

---

### Step 3: Start Live Trading

```bash
# Start all services with your credentials
python start_microservices.py start

# Check system status
python start_microservices.py status

# Open dashboard
# Visit: http://localhost:8000
```

---

## 🔧 Manual Configuration (Alternative)

If you prefer manual setup:

### Option A: Edit .env File Directly

```bash
# Edit the .env file
nano .env

# Replace these lines:
BITGET_API_KEY=YOUR_ACTUAL_API_KEY_HERE
BITGET_API_SECRET=YOUR_ACTUAL_API_SECRET_HERE
BITGET_API_PASSWORD=YOUR_ACTUAL_API_PASSWORD_HERE

# With your real credentials:
BITGET_API_KEY=bg_your_actual_key_here
BITGET_API_SECRET=your_actual_secret_here
BITGET_API_PASSWORD=your_actual_password_here
```

### Option B: Use Credential Manager

```bash
# Advanced credential management
python setup_credentials.py setup
```

---

## 📊 What Gets Configured

When you run the configuration wizard, it automatically updates:

### 1. Main Configuration
- ✅ **`.env`** - Main environment file
- ✅ **`docker-compose.yml`** - Docker services
- ✅ **`docker-compose.live.yml`** - Live trading setup

### 2. Service-Specific Configs
- ✅ **`services/api-server/credentials.env`**
- ✅ **`services/exchange-connector/credentials.env`**
- ✅ **`services/live-trading-engine/credentials.env`**
- ✅ **`services/data-manager/credentials.env`**
- ✅ **`services/risk-manager/credentials.env`**
- ✅ **`services/monitoring-service/credentials.env`**

### 3. Security Features
- ✅ **Encrypted backup** (`.credentials.json`)
- ✅ **Automatic validation** of credential format
- ✅ **Secure storage** with system-based encryption

---

## 🔒 Security Best Practices

### API Key Security
- **Never share** your API credentials
- **Use IP restrictions** on all keys
- **Enable 2FA** on your exchange account
- **Regular rotation** of API keys (every 30 days)

### File Security
- **`.env` files** are excluded from git commits
- **Credentials are encrypted** for backup storage
- **Service configs** are automatically created
- **Read-only mode** available for production

### Network Security
- **Internal Docker network** for service communication
- **No external exposure** of sensitive services
- **HTTPS only** for web dashboard
- **Rate limiting** on API endpoints

---

## 🧪 Testing Your Setup

### Test API Connectivity

```bash
# Test exchange connector
curl http://localhost:8005/api/ticker/BTC/USDT:USDT

# Test account balance
curl http://localhost:8005/api/balance

# Test risk manager
curl http://localhost:8002/api/risk/status
```

### Test Live Trading (Safe Mode)

```bash
# Start in demo mode first
python start_microservices.py start --demo

# Check dashboard
open http://localhost:8000

# Monitor logs
python start_microservices.py logs --service live-trading-engine
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. "Invalid API Key"
```bash
❌ Error: API key appears to be placeholder
✅ Fix: Use your real Bitget API key (starts with 'bg_')
```

#### 2. "Docker Not Running"
```bash
❌ Error: Cannot connect to Docker
✅ Fix: Start Docker Desktop and try again
```

#### 3. "Permission Denied"
```bash
❌ Error: Cannot write to .env file
✅ Fix: Remove read-only attribute
Set-ItemProperty .env -Name IsReadOnly -Value $false
```

#### 4. "Service Won't Start"
```bash
# Check logs
python start_microservices.py logs --service [service-name]

# Restart specific service
python start_microservices.py restart --service [service-name]
```

---

## 🔄 Updating Credentials

### Change API Keys

```bash
# Run configuration wizard again
python configure_api.py

# Or use credential manager
python setup_credentials.py update
```

### Remove All Credentials

```bash
# Reset to placeholders
python setup_credentials.py reset

# Remove completely
python setup_credentials.py remove
```

---

## 📋 Configuration Checklist

### Before Live Trading

- [ ] **Docker Desktop running**
- [ ] **Bitget API key created**
- [ ] **IP restrictions enabled**
- [ ] **Credentials configured** (`python configure_api.py`)
- [ ] **System tested** (`python start_microservices.py status`)
- [ ] **Dashboard accessible** (`http://localhost:8000`)

### Production Ready

- [ ] **Risk limits set** (2% per trade max)
- [ ] **Daily loss limits** (3% max daily loss)
- [ ] **Emergency stop enabled**
- [ ] **Monitoring alerts configured**
- [ ] **Backup procedures tested**

---

## 🎯 Next Steps

1. **Complete API setup** using the wizard
2. **Start the system** and verify all services
3. **Test with small amounts** first
4. **Monitor performance** via dashboard
5. **Set up alerts** for system notifications

---

## 📞 Support

**Need Help?**

1. **Check this guide** first
2. **Run diagnostic**: `python start_microservices.py health`
3. **Check logs**: `python start_microservices.py logs`
4. **Verify credentials**: `python setup_credentials.py status`

**Emergency Contacts:**
- **System Emergency**: `python start_microservices.py stop`
- **Exchange Support**: Bitget customer service
- **Technical Issues**: Check GitHub issues

---

**🚀 Happy Trading with VIPER!**

*Remember: Trading involves risk. Never trade more than you can afford to lose. Always test thoroughly before live trading.*
