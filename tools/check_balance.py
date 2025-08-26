#!/usr/bin/env python3
"""
Check Bitget Swaps Wallet Balance and Positions
"""

import ccxt
import os

# Initialize Bitget exchange
exchange = ccxt.bitget({
    'apiKey': 'bg_d20a392139710bc38b8ab39e970114eb',
    'secret': '23ed4a7fe10b9c947d41a15223647f1b263f0d932b7d5e9e7bdfac01d3b84b36',
    'password': '22672267',
    'options': {
        'defaultType': 'swap',
        'adjustForTimeDifference': True,
    },
    'sandbox': False,
})

try:
    print("🔄 Fetching Swaps Wallet Balance...")

    # Get balance
    balance = exchange.fetch_balance()
    print("\n💰 Swaps Wallet Balance:")
    usdt_balance = balance.get('USDT', {})
    free_usdt = usdt_balance.get('free', 0)
    total_usdt = usdt_balance.get('total', 0)

    print(f"USDT Free: ${free_usdt:.2f}")
    print(f"USDT Total: ${total_usdt:.2f}")

    print("\n📊 Checking Open Positions...")

    # Get positions
    positions = exchange.fetch_positions()
    open_positions = [pos for pos in positions if pos['contracts'] > 0]

    if open_positions:
        print(f"📈 Open Positions: {len(open_positions)}")
        for pos in open_positions:
            print(f"  {pos['symbol']}: {pos['contracts']} contracts ({pos['side']})")
    else:
        print("📉 No open positions")

    # Calculate appropriate position size for swaps
    if free_usdt > 0:
        print("\n🎯 Recommended Position Sizing:")
        print(".2f")
        print("  Min Contract Size: 0.001 BTC")
        print(".6f")

except Exception as e:
    print(f"❌ Error: {e}")
