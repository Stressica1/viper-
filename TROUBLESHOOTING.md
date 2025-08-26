# 🚨 VIPER Troubleshooting Guide

**Quick solutions to common issues**

## 🔧 Quick Diagnostics

**First step for any issue:**
```bash
python scripts/quick_validation.py
```

This will tell you exactly what's wrong and how to fix it.

---

## 🐳 Docker Issues

### "Docker daemon not running"
**Solution:**
- **Windows/Mac**: Start Docker Desktop application
- **Linux**: `sudo systemctl start docker`

### "Port already in use"  
**Solution:**
```bash
python scripts/start_microservices.py stop
# Wait 30 seconds
python scripts/start_microservices.py start
```

### "Docker not found"
**Solution:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## 🐍 Python Issues

### "Module not found" 
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
# OR
pip install -e .
```

### "Permission denied"
**Solution:**
```bash
# Linux/Mac
chmod +x scripts/*.py

# Windows: Run CMD as Administrator
```

### Python version error
**Solution:** Install Python 3.11+ from [python.org](https://python.org)

---

## 🔐 API Issues

### "Invalid API key"
**Check:**
- ✅ API key starts with `bg_`
- ✅ Permissions enabled (Read Info, Spot Trading, Futures Trading)
- ✅ IP restrictions allow your IP
- ✅ 2FA is enabled on Bitget account

**Fix:**
```bash
python scripts/configure_api.py
```

### "Connection timeout"
**Check:**
- ✅ Internet connection
- ✅ Firewall settings
- ✅ Try sandbox mode first

---

## ⚙️ Service Issues

### Services won't start
**Solution:**
```bash
# Check logs
docker-compose -f infrastructure/docker-compose.yml logs

# Reset everything
python scripts/start_microservices.py stop
docker system prune -f
python scripts/start_microservices.py start
```

### Dashboard not accessible
**Check:**
- ✅ Go to: http://localhost:8000 (not https)
- ✅ Port 8000 not blocked by firewall
- ✅ Services are running: `python scripts/start_microservices.py status`

---

## 💾 Installation Issues

### Setup script fails
**Try:**
```bash
# Manual installation
pip install -r requirements.txt
cp .env.template .env
python scripts/configure_api.py
```

### Dependencies missing
**Solution:**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 🌐 Network Issues

### Can't reach exchanges
**Check:**
- ✅ Internet connection
- ✅ VPN/proxy settings
- ✅ Corporate firewall

### WebSocket errors
**Try:**
- Restart services
- Check firewall settings
- Use different network

---

## 📊 Trading Issues

### No trading signals
**Check:**
- ✅ Market is open
- ✅ Symbol is active (BTC/USDT, ETH/USDT work well)
- ✅ VIPER threshold settings in dashboard

### Backtest errors
**Try:**
- Use shorter date ranges
- Check symbol format (BTC/USDT:USDT for futures)
- Verify historical data availability

---

## 🚨 Emergency Stops

### Stop all trading immediately
```bash
# Stop all services
python scripts/start_microservices.py stop

# Or force stop
docker-compose -f infrastructure/docker-compose.yml down --remove-orphans
```

### Check positions manually
- Log into Bitget website
- Check open positions
- Close positions manually if needed

---

## 📞 Get More Help

1. **Run diagnostics**: `python scripts/quick_validation.py`
2. **Check logs**: Look in `logs/` directory
3. **System status**: `python scripts/start_microservices.py status`
4. **Documentation**: Browse `docs/` folder
5. **GitHub Issues**: Create issue with error logs

---

## 🛠️ Log Files

**Check these logs for detailed error information:**

```bash
# Application logs
ls logs/

# Docker logs
docker-compose -f infrastructure/docker-compose.yml logs

# Service-specific logs
docker logs viper-api-server
docker logs viper-exchange-connector
```

---

## ✅ System Health Check

**Use this checklist to verify everything is working:**

- [ ] Python 3.11+ installed
- [ ] Docker Desktop running  
- [ ] Dependencies installed: `pip list`
- [ ] Environment configured: `.env` file exists
- [ ] API keys added (for live trading)
- [ ] Services running: `python scripts/start_microservices.py status`
- [ ] Dashboard accessible: http://localhost:8000
- [ ] Validation passes: `python scripts/quick_validation.py`

---

**Still having issues?** 

Run the validation script and create a GitHub issue with the output:
```bash
python scripts/quick_validation.py > validation_report.txt
```