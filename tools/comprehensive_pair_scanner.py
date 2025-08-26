#!/usr/bin/env python3
"""
🚀 VIPER Trading System - Comprehensive 50x Leverage Pair Scanner
MCP-style documentation and analysis of all Bitget trading pairs
"""

import ccxt
import json
import os
import time
from typing import Dict, List, Tuple
from datetime import datetime

class ComprehensivePairScanner:
    """
    Comprehensive scanner for all Bitget trading pairs with 25x leverage support.
    Provides MCP-style documentation and analysis.
    """

    def __init__(self):
        """Initialize scanner with working API credentials"""
        self.exchange = ccxt.bitget({
            'apiKey': os.getenv('BITGET_API_KEY', ''),
            'secret': os.getenv('BITGET_API_SECRET', ''),
            'password': os.getenv('BITGET_API_PASSWORD', ''),
            'options': {
                'defaultType': 'swap',  # Focus on perpetual swaps for 50x leverage
                'adjustForTimeDifference': True,
            },
            'sandbox': False,
        })

        self.scan_results = {
            'timestamp': datetime.now().isoformat(),
            'total_pairs': 0,
            'pairs_with_25x_leverage': [],
            'pairs_without_25x_leverage': [],
            'error_pairs': [],
            'market_analysis': {},
            'leverage_distribution': {},
            'trading_volume_analysis': {},
            'recommendations': []
        }

    def initialize_connection(self) -> bool:
        """
        Initialize Bitget API connection and load markets.
        Returns True if successful, False otherwise.
        """
        try:
            print("🔄 Initializing Bitget API connection...")
            self.exchange.load_markets()
            print(f"✅ Successfully loaded {len(self.exchange.symbols)} markets")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize connection: {e}")
            return False

    def scan_pair_details(self, symbol: str) -> Dict:
        """
        Scan detailed information for a specific trading pair.
        Returns comprehensive market data.
        """
        try:
            # Get market info
            market = self.exchange.market(symbol)

            # Get ticker data
            ticker = self.exchange.fetch_ticker(symbol)

            # Get order book
            order_book = self.exchange.fetch_order_book(symbol, limit=5)

            return {
                'symbol': symbol,
                'base': market.get('base', 'Unknown'),
                'quote': market.get('quote', 'Unknown'),
                'type': market.get('type', 'spot'),
                'active': market.get('active', False),
                'contract_size': market.get('contractSize', 1),
                'min_amount': market.get('limits', {}).get('amount', {}).get('min', 0),
                'max_amount': market.get('limits', {}).get('amount', {}).get('max', None),
                'price_precision': market.get('precision', {}).get('price', 0),
                'amount_precision': market.get('precision', {}).get('amount', 0),
                'current_price': ticker.get('last', 0),
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'volume_24h': ticker.get('baseVolume', 0),
                'spread': (ticker.get('ask', 0) - ticker.get('bid', 0)) if ticker.get('ask') and ticker.get('bid') else 0,
                'order_book_depth': len(order_book.get('bids', [])) + len(order_book.get('asks', [])),
                'leverage_available': self.check_leverage_availability(symbol),
                'trading_signals': self.generate_trading_signals(ticker, market)
            }

        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'leverage_available': False
            }

    def check_leverage_availability(self, symbol: str) -> bool:
        """
        Check if a symbol supports 25x leverage.
        For perpetual swaps, this is typically available by default.
        """
        try:
            # For Bitget perpetual swaps, 25x leverage is typically available
            # We can verify by checking market type and contract details
            market = self.exchange.market(symbol)

            if market.get('type') == 'swap' and market.get('active'):
                # Get leverage info if available
                leverage_info = self.exchange.fetch_leverage_tiers(symbol)
                if leverage_info:
                    # Check if 25x is in the available tiers
                    for tier in leverage_info:
                        if tier.get('maxLeverage', 0) >= 25:
                            return True
                else:
                    # If no specific info, assume 25x is available for active swaps
                    return True

            return False

        except Exception as e:
            print(f"⚠️ Error checking leverage for {symbol}: {e}")
            return False

    def generate_trading_signals(self, ticker: Dict, market: Dict) -> Dict:
        """
        Generate basic trading signals based on market data.
        """
        try:
            current_price = ticker.get('last', 0)
            if current_price <= 0:
                return {'signal': 'NO_DATA', 'confidence': 0}

            # Simple momentum signal based on price movement
            open_price = ticker.get('open', current_price)
            price_change = (current_price - open_price) / open_price if open_price > 0 else 0

            # Volume analysis
            volume = ticker.get('baseVolume', 0)

            if price_change > 0.02 and volume > 1000:  # 2% up with decent volume
                return {
                    'signal': 'BUY',
                    'confidence': min(0.8, abs(price_change) * 10),
                    'reason': '.2%'
                }
            elif price_change < -0.02 and volume > 1000:  # 2% down with decent volume
                return {
                    'signal': 'SELL',
                    'confidence': min(0.8, abs(price_change) * 10),
                    'reason': '.2%'
                }
            else:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.5,
                    'reason': 'Low volatility or volume'
                }

        except Exception as e:
            return {'signal': 'ERROR', 'confidence': 0, 'error': str(e)}

    def scan_all_pairs(self) -> Dict:
        """
        Main scanning function that processes all available pairs.
        Returns comprehensive analysis results.
        """
        print("🔍 Starting comprehensive pair scan...")
        print("=" * 60)

        if not self.initialize_connection():
            return {'error': 'Failed to initialize connection'}

        total_pairs = len([s for s in self.exchange.symbols if ':USDT' in s])
        print(f"📊 Found {total_pairs} USDT perpetual swap pairs to analyze")

        pairs_scanned = 0
        pairs_25x = 0

        # Focus on USDT perpetual swaps (most relevant for 25x leverage)
        swap_pairs = [s for s in self.exchange.symbols if ':USDT' in s]

        for symbol in swap_pairs:
            try:
                print(f"📋 Scanning {symbol} ({pairs_scanned + 1}/{len(swap_pairs)})")

                pair_data = self.scan_pair_details(symbol)
                self.scan_results['total_pairs'] += 1

                if pair_data.get('error'):
                    self.scan_results['error_pairs'].append(pair_data)
                    print(f"  ❌ Error: {pair_data['error']}")
                elif pair_data.get('leverage_available'):
                    self.scan_results['pairs_with_25x_leverage'].append(pair_data)
                    pairs_25x += 1
                    print(f"  ✅ 25x Leverage Available - Price: ${pair_data.get('current_price', 0):.4f}")
                else:
                    self.scan_results['pairs_without_25x_leverage'].append(pair_data)
                    print(f"  ❌ No 25x Leverage - Price: ${pair_data.get('current_price', 0):.4f}")

                pairs_scanned += 1

                # Add small delay to be respectful to API
                time.sleep(0.1)

            except Exception as e:
                print(f"❌ Error scanning {symbol}: {e}")
                self.scan_results['error_pairs'].append({
                    'symbol': symbol,
                    'error': str(e)
                })

        self.analyze_results()
        return self.scan_results

    def analyze_results(self):
        """
        Analyze scan results and generate recommendations.
        """
        print("\n📊 ANALYSIS COMPLETE")
        print("=" * 40)

        pairs_25x = self.scan_results['pairs_with_25x_leverage']
        pairs_no_25x = self.scan_results['pairs_without_25x_leverage']

        print(f"✅ Pairs with 25x leverage: {len(pairs_25x)}")
        print(f"❌ Pairs without 25x leverage: {len(pairs_no_25x)}")
        print(f"⚠️ Error pairs: {len(self.scan_results['error_pairs'])}")

        # Analyze by trading signals
        buy_signals = [p for p in pairs_25x if p.get('trading_signals', {}).get('signal') == 'BUY']
        sell_signals = [p for p in pairs_25x if p.get('trading_signals', {}).get('signal') == 'SELL']

        print(f"📈 BUY signals: {len(buy_signals)}")
        print(f"📉 SELL signals: {len(sell_signals)}")

        # Generate recommendations
        recommendations = []

        if buy_signals:
            top_buy = sorted(buy_signals, key=lambda x: x.get('trading_signals', {}).get('confidence', 0), reverse=True)[:5]
            recommendations.append(f"🚀 TOP BUY OPPORTUNITIES: {[p['symbol'] for p in top_buy]}")

        if sell_signals:
            top_sell = sorted(sell_signals, key=lambda x: x.get('trading_signals', {}).get('confidence', 0), reverse=True)[:5]
            recommendations.append(f"📉 TOP SELL OPPORTUNITIES: {[p['symbol'] for p in top_sell]}")

        # Volume analysis
        high_volume = [p for p in pairs_25x if p.get('volume_24h', 0) > 10000]
        recommendations.append(f"💰 HIGH VOLUME PAIRS: {[p['symbol'] for p in high_volume[:5]]}")

        self.scan_results['recommendations'] = recommendations

        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")

    def generate_mcp_documentation(self) -> str:
        """
        Generate MCP-style documentation for the scanned results.
        """
        results = self.scan_results

        doc = f"""# 🚀 VIPER TRADING SYSTEM - 25X LEVERAGE PAIR SCAN

## 📊 SCAN SUMMARY
- **Scan Date**: {results['timestamp']}
- **Total Pairs Scanned**: {results['total_pairs']}
- **Pairs with 25x Leverage**: {len(results['pairs_with_25x_leverage'])}
- **Pairs without 25x Leverage**: {len(results['pairs_without_25x_leverage'])}
- **Error Pairs**: {len(results['error_pairs'])}

## ✅ 25X LEVERAGE PAIRS

"""

        for pair in results['pairs_with_25x_leverage'][:20]:  # Top 20
            doc += f"""### {pair['symbol']}
- **Price**: ${pair.get('current_price', 0):.4f}
- **24h Volume**: {pair.get('volume_24h', 0):,.0f}
- **Spread**: ${pair.get('spread', 0):.4f}
- **Trading Signal**: {pair.get('trading_signals', {}).get('signal', 'UNKNOWN')}
- **Min Amount**: {pair.get('min_amount', 0)}
- **Contract Size**: {pair.get('contract_size', 1)}

"""

        doc += """## 📋 IMPLEMENTATION GUIDE

### For Live Trading Engine:

```python
# Add these pairs to your trading universe
VIPER_50X_PAIRS = [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'BNB/USDT:USDT',
    # ... add from scan results
]

def get_50x_pairs():
    return [p['symbol'] for p in pairs_with_50x_leverage]
```

### For Risk Manager:

```python
# Configure position limits per pair
PAIR_POSITION_LIMITS = {
    'BTC/USDT:USDT': 0.001,  # 0.001 BTC max per trade
    'ETH/USDT:USDT': 0.01,   # 0.01 ETH max per trade
    # ... configure based on scan results
}
```

## 🎯 TRADING RECOMMENDATIONS

"""

        for rec in results['recommendations']:
            doc += f"- {rec}\n"

        doc += """

## 🔧 CONFIGURATION FOR 50X PAIRS

### Environment Variables:
```bash
# Enable 50x pairs scanning
ENABLE_50X_PAIRS=true
MAX_50X_PAIRS=50

# Risk settings per pair
BTC_MAX_POSITION=0.001
ETH_MAX_POSITION=0.01
BNB_MAX_POSITION=1.0
```

### Docker Configuration:
```yaml
environment:
  - ENABLE_50X_PAIRS=true
  - MAX_50X_PAIRS=50
  - PAIR_SCAN_INTERVAL=3600  # Scan every hour
```

## 📈 PERFORMANCE ANALYSIS

### By Trading Signal:
- **BUY Signals**: Pairs showing upward momentum
- **SELL Signals**: Pairs showing downward momentum
- **HOLD Signals**: Low volatility pairs

### By Volume:
- **High Volume**: >10,000 contracts/24h
- **Medium Volume**: 1,000-10,000 contracts/24h
- **Low Volume**: <1,000 contracts/24h

## 🚨 RISK CONSIDERATIONS

1. **Volume Check**: Always verify 24h volume >1,000 before trading
2. **Spread Analysis**: Avoid pairs with spreads >1% of price
3. **Liquidity**: Prefer pairs with deep order books
4. **Position Sizing**: Use smaller sizes for low-volume pairs

## 🔄 SCAN AUTOMATION

### Cron Job for Regular Scanning:
```bash
# Scan every 4 hours
0 */4 * * * /usr/local/bin/python /app/comprehensive_pair_scanner.py
```

### Real-time Monitoring:
```python
def monitor_50x_pairs():
    scanner = ComprehensivePairScanner()
    results = scanner.scan_all_pairs()

    # Update trading pairs
    update_trading_universe(results['pairs_with_50x_leverage'])

    # Send alerts for high-confidence signals
    send_trading_alerts(results['recommendations'])
```

## 📊 ANALYTICS DASHBOARD

### Key Metrics to Monitor:
- Total 50x pairs available
- Pairs by trading signal
- Average volume per pair
- Spread distribution
- Error rate by pair

### Alerts to Configure:
- New pair added to 50x list
- High-confidence trading signals
- Volume spikes on low-volume pairs
- Spread widening alerts

## 🎉 CONCLUSION

This comprehensive scan provides a complete database of all Bitget perpetual swap pairs with 25x leverage support, along with trading signals and risk analysis. The system is now equipped to trade across the full universe of available 25x leverage pairs with proper risk management.

**Total 25x Leverage Pairs Available**: {len(results['pairs_with_25x_leverage'])}

**Ready for automated trading across all viable pairs!** 🚀
"""

        return doc

def main():
    """Main execution function"""
    scanner = ComprehensivePairScanner()
    results = scanner.scan_all_pairs()

    if 'error' not in results:
        # Generate MCP documentation
        mcp_doc = scanner.generate_mcp_documentation()

        # Save results
        with open('25x_leverage_pairs_scan.json', 'w') as f:
            json.dump(results, f, indent=2)

        with open('25x_leverage_pairs_MCP.md', 'w') as f:
            f.write(mcp_doc)

        print("\n📄 Results saved:")
        print("  📊 JSON data: 25x_leverage_pairs_scan.json")
        print("  📋 MCP docs: 25x_leverage_pairs_MCP.md")
        # Print summary
        print("\n🎯 SUMMARY:")
        print(f"  ✅ 25x Leverage Pairs: {len(results['pairs_with_25x_leverage'])}")
        print(f"  📈 BUY Signals: {len([p for p in results['pairs_with_25x_leverage'] if p.get('trading_signals', {}).get('signal') == 'BUY'])}")
        print(f"  📉 SELL Signals: {len([p for p in results['pairs_with_25x_leverage'] if p.get('trading_signals', {}).get('signal') == 'SELL'])}")

    else:
        print(f"❌ Scan failed: {results['error']}")

if __name__ == "__main__":
    main()
