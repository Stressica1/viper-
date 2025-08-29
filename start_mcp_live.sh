#!/bin/bash

# MCP Live Trading Quick Start Script
# This script helps you set up and start the MCP live trading system

echo "🎯 MCP Live Trading Setup"
echo "=========================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Please copy .env.template to .env and configure your Bitget API credentials"
    echo "   cp .env.template .env"
    exit 1
fi

# Check if API credentials are configured
if grep -q "your_api_key_here" .env; then
    echo "⚠️  API credentials not configured!"
    echo "   Please edit .env file with your actual Bitget API credentials"
    echo "   nano .env"
    exit 1
fi

echo "✅ Configuration files found"

# Install required packages
echo ""
echo "📦 Installing required packages..."
pip install ccxt python-dotenv pandas numpy

if [ $? -ne 0 ]; then
    echo "❌ Failed to install packages"
    exit 1
fi

echo "✅ Packages installed successfully"

# Test connection
echo ""
echo "🔗 Testing Bitget API connection..."
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

try:
    import ccxt
    exchange = ccxt.bitget({
        'apiKey': os.getenv('BITGET_API_KEY'),
        'secret': os.getenv('BITGET_SECRET_KEY'),
        'password': os.getenv('BITGET_PASSPHRASE'),
        'enableRateLimit': True
    })

    # Test connection
    balance = exchange.fetch_balance()
    print('✅ Bitget API connection successful')
    print(f'   USDT Balance: {balance[\"USDT\"][\"free\"]:.2f}')

except Exception as e:
    print(f'❌ API connection failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ API connection test failed"
    exit 1
fi

echo ""
echo "🎯 MCP Live Trading System Ready!"
echo "==================================="
echo ""
echo "Choose your trading mode:"
echo "1) Test Mode (Safe - no real trades)"
echo "2) Live Trading (REAL TRADES - Use with caution!)"
echo "3) Show Configuration"
echo "4) Exit"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🧪 Starting MCP in TEST MODE..."
        echo "   This will simulate trading without real orders"
        echo "   Press Ctrl+C to stop"
        echo ""
        python mcp_live_trader.py --test-mode
        ;;
    2)
        echo ""
        echo "⚠️  LIVE TRADING MODE SELECTED"
        echo "   This will execute REAL trades on Bitget!"
        echo ""
        read -p "Are you sure you want to start live trading? (yes/no): " confirm

        if [ "$confirm" = "yes" ] || [ "$confirm" = "y" ]; then
            echo ""
            echo "🚀 Starting MCP LIVE TRADING..."
            echo "   Press Ctrl+C to stop"
            echo ""
            python mcp_live_trader.py --start-live
        else
            echo "Live trading cancelled"
        fi
        ;;
    3)
        echo ""
        python mcp_live_trader.py --show-performance
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run again."
        exit 1
        ;;
esac
