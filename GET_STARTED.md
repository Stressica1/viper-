# 🚀 Get Started with VIPER in 2 Minutes

**The fastest way to get your algorithmic trading bot running!**

## ⚡ Super Quick Start

```bash
# 1. Clone and enter directory
git clone https://github.com/Stressica1/viper-.git
cd viper-

# 2. Run automated setup
python install_viper.py

# 3. Start trading system  
python scripts/start_microservices.py start

# 4. Open dashboard
# Go to: http://localhost:8000
```

**That's it! 🎉 VIPER is now running!**

---

## 🔐 Add Your API Keys (For Live Trading)

After the system is running, add your Bitget API credentials:

```bash
# Run the configuration wizard
python scripts/configure_api.py
```

**Get Bitget API Keys:**
1. Go to [Bitget API Management](https://www.bitget.com/en/account/newapi)
2. Create API key with trading permissions
3. Copy: API Key, Secret, and Password
4. Run the wizard above and enter your credentials

---

## 🎯 What You Get

- **📊 Real-time Dashboard** - http://localhost:8000
- **🧪 Backtesting Engine** - Test strategies on historical data  
- **🔥 Live Trading** - Automated trading with risk management
- **📈 Performance Analytics** - Track profits and losses
- **🚨 Risk Management** - Automatic position sizing and stops
- **🤖 AI Integration** - MCP support for AI assistants

---

## 🛠️ Troubleshooting

**Problem?** Run the diagnostic tool:
```bash
python scripts/quick_validation.py
```

**Need Docker?** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Python Issues?** Make sure you have Python 3.11+

**Still stuck?** Check [INSTALLATION.md](INSTALLATION.md) for detailed guide

---

## 📚 Next Steps

- **Test First**: Run backtests before live trading
- **Start Small**: Begin with small position sizes
- **Monitor**: Watch your first few trades closely
- **Learn**: Read the documentation in `docs/`

---

**⚠️ Trading Warning**: Never trade with money you can't afford to lose. Test thoroughly first!

**🎉 Happy Trading!** You now have a professional-grade algorithmic trading system.