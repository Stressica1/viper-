#!/usr/bin/env python3
import ccxt
from dotenv import load_dotenv
import os

load_dotenv()

print('🔍 Testing Bitget connection and pair data...')

try:
    # Test exchange connection
    exchange = ccxt.bitget({
        'apiKey': os.getenv('BITGET_API_KEY'),
        'secret': os.getenv('BITGET_API_SECRET'),
        'password': os.getenv('BITGET_API_PASSWORD'),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })

    # Test connection
    exchange.load_markets()
    print('✅ Exchange connected successfully')

    # Get all swap markets
    markets = exchange.markets
    swap_pairs = [symbol for symbol, market in markets.items()
                 if market.get('active', False) and market.get('type') == 'swap'
                 and market.get('quote') == 'USDT']

    print(f'📊 Found {len(swap_pairs)} USDT swap pairs')

    # Test first 5 pairs
    print('\n🔍 Testing first 5 pairs:')
    for i, symbol in enumerate(swap_pairs[:5], 1):
        try:
            ticker = exchange.fetch_ticker(symbol)
            volume = ticker.get('quoteVolume', 0)
            leverage = markets[symbol].get('leverage', {}).get('max', 1)
            spread = abs(ticker.get('ask', 0) - ticker.get('bid', 0)) / ticker.get('bid', 1) if ticker.get('bid', 1) > 0 else 1

            print(f'{i}. {symbol}:')
            print(f'   Volume: ${volume:,.0f}')
            print(f'   Leverage: {leverage}x')
            print(f'   Spread: {spread:.4f}')
            print(f'   Price: ${ticker.get("last", 0):.4f}')

            # Check filtering criteria
            vol_ok = volume >= 100000
            lev_ok = leverage >= 3
            spread_ok = spread <= 0.001

            print(f'   Volume OK (≥$100K): {vol_ok}')
            print(f'   Leverage OK (≥3x): {lev_ok}')
            print(f'   Spread OK (≤0.1%): {spread_ok}')
            print(f'   QUALIFIED: {vol_ok and lev_ok and spread_ok}')
            print()

        except Exception as e:
            print(f'{i}. {symbol}: ERROR - {e}')
            print()

except Exception as e:
    print(f'❌ Error: {e}')
