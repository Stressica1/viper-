#!/usr/bin/env python3
"""
Test the exact same order execution that fails in live trading engine
"""

import ccxt
import os

# Use same credentials as live trading engine
api_key = os.getenv('BITGET_API_KEY', 'bg_d20a392139710bc38b8ab39e970114eb')
api_secret = os.getenv('BITGET_API_SECRET', 'your_bitget_api_secret_here')
api_password = os.getenv('BITGET_API_PASSWORD', 'your_bitget_api_password_here')

print("🔍 Testing order execution with live trading engine credentials...")
print(f"API Key: {api_key[:10]}...")
print(f"API Secret: {api_secret[:10]}...")
print(f"API Password: {api_password}")

# Initialize exchange with EXACT same config as live trading engine
exchange = ccxt.bitget({
    'apiKey': api_key,
    'secret': api_secret,
    'password': api_password,
    'options': {
        'defaultType': 'swap',
        'adjustForTimeDifference': True,
    },
    'sandbox': False,
})

try:
    print("📡 Loading markets...")
    exchange.load_markets()
    print("✅ Exchange loaded successfully")

    print("🚀 Testing SELL order with 0.001 BTC...")
    result = exchange.create_order('BTC/USDT:USDT', 'market', 'sell', 0.001, None)
    print(f"✅ Order placed successfully: {result.get('id', 'N/A')}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
