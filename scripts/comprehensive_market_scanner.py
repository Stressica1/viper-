#!/usr/bin/env python3
"""
🚀 VIPER Trading System - Comprehensive Market Scanner
Scans ALL available trading pairs with intelligent batching and rate limiting

Features:
- Dynamic pair discovery from Bitget
- Intelligent batch processing
- Rate limiting to prevent API bans
- Real-time signal generation across all pairs
- Performance monitoring and optimization
"""

import os
import sys
import time
import json
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Load environment variables
SCAN_ALL_PAIRS = os.getenv('SCAN_ALL_PAIRS', 'true').lower() == 'true'
MAX_PAIRS_LIMIT = int(os.getenv('MAX_PAIRS_LIMIT', '500'))
SCAN_INTERVAL_SECONDS = int(os.getenv('SCAN_INTERVAL_SECONDS', '15'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))

class ComprehensiveMarketScanner:
    """Scans ALL trading pairs with intelligent batching and rate limiting"""

    def __init__(self):
        self.base_url = "https://api.bitget.com"
        self.is_running = False
        self.all_symbols = []
        self.signals_generated = 0
        self.scan_count = 0
        self.last_scan_time = None
        self.api_call_count = 0
        self.error_count = 0

        print("🚀 Comprehensive Market Scanner Initialized")
        print(f"📊 Scan All Pairs: {SCAN_ALL_PAIRS}")
        print(f"📊 Max Pairs Limit: {MAX_PAIRS_LIMIT}")
        print(f"📊 Scan Interval: {SCAN_INTERVAL_SECONDS}s")
        print(f"📊 Batch Size: {BATCH_SIZE}")
        print("=" * 80)

    def fetch_all_trading_pairs(self) -> List[str]:
        """Fetch ALL available USDT trading pairs from Bitget"""
        try:
            print("🔍 Discovering ALL available trading pairs...")

            # Use Bitget's public API to get all instruments
            spot_instruments_url = f"{self.base_url}/api/v2/spot/public/symbols"

            response = requests.get(spot_instruments_url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get('code') == '00000' and data.get('data'):
                all_pairs = []
                for symbol_data in data['data']:
                    if (symbol_data.get('quoteCoin') == 'USDT' and
                        symbol_data.get('status') == 'online' and
                        symbol_data.get('minTradeAmount', 0) > 0):

                        symbol_name = f"{symbol_data.get('baseCoin')}/USDT:USDT"
                        all_pairs.append(symbol_name)

                print(f"✅ Found {len(all_pairs)} USDT trading pairs on Bitget")

                # Sort by base coin for consistent ordering
                all_pairs.sort()

                # Apply pair limit if specified
                if MAX_PAIRS_LIMIT > 0 and len(all_pairs) > MAX_PAIRS_LIMIT:
                    print(f"📊 Limiting to top {MAX_PAIRS_LIMIT} pairs")
                    all_pairs = all_pairs[:MAX_PAIRS_LIMIT]

                return all_pairs

            print("❌ Failed to fetch trading pairs, using fallback")
            return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT']

        except Exception as e:
            print(f"❌ Error fetching trading pairs: {e}")
            return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT']

    def fetch_market_data_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch market data for a batch of symbols with rate limiting"""
        market_data = {}

        for symbol in symbols:
            try:
                # Rate limiting: small delay between API calls
                time.sleep(0.05)  # 50ms delay to prevent rate limiting

                ticker_url = f"{self.base_url}/api/v2/spot/market/tickers"
                response = requests.get(ticker_url, timeout=5)
                response.raise_for_status()

                self.api_call_count += 1
                data = response.json()

                if data.get('code') == '00000' and data.get('data'):
                    for ticker in data['data']:
                        if ticker.get('symbol') == symbol.replace('/', '').replace(':USDT', ''):
                            market_data[symbol] = {
                                'symbol': symbol,
                                'price': float(ticker.get('last', 0)),
                                'high': float(ticker.get('high24h', 0)),
                                'low': float(ticker.get('low24h', 0)),
                                'volume': float(ticker.get('volume24h', 0)),
                                'price_change': float(ticker.get('change', 0)),
                                'timestamp': datetime.now().isoformat()
                            }
                            break

            except Exception as e:
                self.error_count += 1
                print(f"❌ Error fetching data for {symbol}: {e}")

        return market_data

    def calculate_viper_score(self, market_data: Dict) -> float:
        """Calculate VIPER score for comprehensive analysis"""
        try:
            volume = market_data.get('volume', 0)
            price_change = market_data.get('price_change', 0)
            high = market_data.get('high', 1)
            low = market_data.get('low', 0)
            current_price = market_data.get('price', 1)

            # Enhanced VIPER scoring algorithm
            volume_score = min(volume / 500000, 100)  # Adjusted for broader market
            price_score = abs(price_change) * 200     # Increased sensitivity
            range_score = ((high - low) / current_price) * 200  # Volatility factor

            # Market momentum factor
            momentum_score = 0
            if price_change > 1.0:
                momentum_score = 100
            elif price_change > 0.5:
                momentum_score = 75
            elif price_change > 0:
                momentum_score = 50
            elif price_change > -0.5:
                momentum_score = 25

            # Combine all factors with weighted average
            viper_score = (
                volume_score * 0.25 +      # 25% - Volume importance
                price_score * 0.30 +       # 30% - Price movement
                range_score * 0.25 +       # 25% - Volatility
                momentum_score * 0.20      # 20% - Momentum
            )

            return min(viper_score, 100)

        except Exception as e:
            print(f"❌ Error calculating VIPER score: {e}")
            return 0

    def generate_signal(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal with enhanced criteria"""
        viper_score = self.calculate_viper_score(market_data)

        if viper_score >= 75:  # Lower threshold for broader coverage
            price_change = market_data.get('price_change', 0)
            current_price = market_data.get('price', 0)

            # Enhanced signal generation
            signal = None
            confidence = min(viper_score / 100, 1.0)

            # Strong momentum signals
            if price_change > 1.5:  # Very strong upward
                signal = "LONG"
            elif price_change > 0.8:  # Strong upward
                signal = "LONG"
            elif price_change < -1.5:  # Very strong downward
                signal = "SHORT"
            elif price_change < -0.8:  # Strong downward
                signal = "SHORT"

            # Additional criteria for broader coverage
            elif viper_score >= 85:  # High VIPER score regardless of momentum
                if price_change > 0:
                    signal = "LONG"
                elif price_change < 0:
                    signal = "SHORT"

            if signal:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'viper_score': viper_score,
                    'confidence': confidence,
                    'price': current_price,
                    'price_change': price_change,
                    'volume': market_data.get('volume', 0),
                    'timestamp': datetime.now().isoformat(),
                    'risk_per_trade': 0.02,  # 2% risk per trade
                    'stop_loss': current_price * (0.98 if signal == "LONG" else 1.02),
                    'take_profit': current_price * (1.03 if signal == "LONG" else 0.97),
                    'strategy': 'COMPREHENSIVE_VIPER'
                }

        return None

    def display_market_overview(self, market_data: Dict[str, Dict]):
        """Display comprehensive market overview"""
        print(f"\n📊 COMPREHENSIVE MARKET OVERVIEW - {len(market_data)} PAIRS")
        print("=" * 120)

        # Sort by VIPER score for best opportunities
        sorted_pairs = sorted(market_data.items(),
                            key=lambda x: self.calculate_viper_score(x[1]),
                            reverse=True)

        print(f"{'Symbol':<18} | {'Price':<12} | {'24h Change':<12} | {'Volume':<12} | {'VIPER Score':<12} | {'Signal':<8}")
        print("-" * 120)

        displayed_count = 0
        signal_count = 0

        for symbol, data in sorted_pairs[:25]:  # Show top 25 pairs
            displayed_count += 1
            viper_score = self.calculate_viper_score(data)
            price = data.get('price', 0)
            change = data.get('price_change', 0)
            volume = data.get('volume', 0)

            # Color coding
            if viper_score >= 85:
                viper_color = "🟢"
                signal_indicator = "🚨"
                signal_count += 1
            elif viper_score >= 70:
                viper_color = "🟡"
                signal_indicator = "⚠️"
            else:
                viper_color = "🔴"
                signal_indicator = "   "

            change_color = "🟢" if change >= 0 else "🔴"
            change_str = f"{change_color} {change:+.2f}%"

            print(f"{symbol:<18} | ${price:<11.4f} | {change_str:<12} | "
                  f"{volume:<11.0f} | {viper_color} {viper_score:<11.1f} | {signal_indicator}")

        print("=" * 120)
        print(f"📈 Showing top {displayed_count} pairs | 🚨 {signal_count} potential signals detected")

    def process_batch(self, batch_symbols: List[str]) -> Dict[str, Any]:
        """Process a batch of symbols"""
        batch_results = {
            'market_data': {},
            'signals': [],
            'api_calls': 0,
            'errors': 0
        }

        try:
            # Fetch market data for batch
            market_data = self.fetch_market_data_batch(batch_symbols)
            batch_results['market_data'] = market_data
            batch_results['api_calls'] = len(batch_symbols)

            # Generate signals for batch
            for symbol in batch_symbols:
                if symbol in market_data:
                    signal = self.generate_signal(symbol, market_data[symbol])
                    if signal:
                        batch_results['signals'].append(signal)
                        self.signals_generated += 1

        except Exception as e:
            batch_results['errors'] = 1
            print(f"❌ Batch processing error: {e}")

        return batch_results

    def run_comprehensive_scan(self):
        """Run comprehensive market scanning with batching"""
        print("🚀 STARTING COMPREHENSIVE MARKET SCAN...")
        print("🔍 Scanning ALL available trading pairs with intelligent batching...")
        # Fetch all available pairs
        self.all_symbols = self.fetch_all_trading_pairs()
        print(f"📊 Monitoring {len(self.all_symbols)} trading pairs")

        print("🎯 VIPER Strategy Configuration:")
        print("   • VIPER Threshold: 75+ (optimized for broader coverage)")
        print("   • Risk per Trade: 2%")
        print("   • Signal Types: LONG/SHORT based on momentum & VIPER score")
        print("   • Update Interval: 15 seconds")
        print("   • Batch Processing: 50 pairs per batch")
        print("-" * 80)

        self.is_running = True

        try:
            while self.is_running:
                self.scan_count += 1
                scan_start_time = time.time()

                print(f"\n🔍 COMPREHENSIVE SCAN #{self.scan_count} - {datetime.now().strftime('%H:%M:%S')}")

                # Process symbols in batches
                all_market_data = {}
                all_signals = []

                # Create batches
                batches = [self.all_symbols[i:i + BATCH_SIZE]
                          for i in range(0, len(self.all_symbols), BATCH_SIZE)]

                print(f"📦 Processing {len(batches)} batches of up to {BATCH_SIZE} pairs each")

                # Process batches with parallel execution for better performance
                with ThreadPoolExecutor(max_workers=3) as executor:
                    batch_futures = [executor.submit(self.process_batch, batch)
                                   for batch in batches]

                    for future in as_completed(batch_futures):
                        batch_results = future.result()
                        all_market_data.update(batch_results['market_data'])
                        all_signals.extend(batch_results['signals'])

                # Display comprehensive market overview
                if all_market_data:
                    self.display_market_overview(all_market_data)

                # Display signals if any were generated
                if all_signals:
                    print(f"\n🚨 TRADING SIGNALS GENERATED ({len(all_signals)} signals):")
                    print("=" * 80)

                    for i, signal in enumerate(all_signals[:10], 1):  # Show top 10
                        print(f"🚨 Signal #{i}: {signal['symbol']} {signal['signal']}")
                        print(f"   VIPER Score: {signal['viper_score']:.1f}/100 | "
                              f"Confidence: {signal['confidence']:.2f}")
                        print(f"   Price: ${signal['price']:.4f} | "
                              f"Change: {signal['price_change']:+.2f}%")
                        print(f"   Stop Loss: ${signal['stop_loss']:.4f} | "
                              f"Take Profit: ${signal['take_profit']:.4f}")
                        print()

                # Performance metrics
                scan_duration = time.time() - scan_start_time
                pairs_per_second = len(self.all_symbols) / scan_duration if scan_duration > 0 else 0

                print("
📈 SCAN PERFORMANCE:"                print(f"   Pairs Scanned: {len(self.all_symbols)}")
                print(f"   Signals Generated: {len(all_signals)}")
                print(f"   Scan Duration: {scan_duration:.2f}s")
                print(f"   Pairs/Second: {pairs_per_second:.1f}")
                print(f"   API Calls: {self.api_call_count}")
                print(f"   Error Count: {self.error_count}")
                print(f"   Signal Rate: {(len(all_signals)/len(self.all_symbols)*100):.2f}%")

                # Wait before next scan
                next_scan_in = max(0, SCAN_INTERVAL_SECONDS - scan_duration)
                if next_scan_in > 0:
                    print(f"⏰ Next comprehensive scan in {next_scan_in:.0f} seconds...")
                    time.sleep(next_scan_in)

        except KeyboardInterrupt:
            print("
👋 Comprehensive scan stopped by user"        except Exception as e:
            print(f"\n❌ Scan error: {e}")
        finally:
            self.is_running = False
            self.display_final_statistics()

    def display_final_statistics(self):
        """Display final session statistics"""
        print("
📊 COMPREHENSIVE SCAN SESSION SUMMARY"        print("=" * 60)
        print(f"Total Scans Completed: {self.scan_count}")
        print(f"Trading Pairs Monitored: {len(self.all_symbols)}")
        print(f"Signals Generated: {self.signals_generated}")
        print(f"API Calls Made: {self.api_call_count}")
        print(f"Errors Encountered: {self.error_count}")

        if self.scan_count > 0:
            avg_signals_per_scan = self.signals_generated / self.scan_count
            print(f"Average Signals Per Scan: {avg_signals_per_scan:.1f}")

        if self.api_call_count > 0:
            error_rate = (self.error_count / self.api_call_count) * 100
            print(f"API Error Rate: {error_rate:.2f}%")

        print("
🎯 VIPER Strategy Performance:"        print("   • Comprehensive market coverage across all pairs"        print("   • Intelligent signal generation with VIPER scoring"        print("   • Real-time market analysis with rate limiting"        print("   • Batch processing for optimal performance"        print("   • Risk-controlled position sizing and management"
def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🚀 VIPER TRADING SYSTEM - COMPREHENSIVE MARKET SCANNER                       ║
║ 🔥 All Pairs Scanning | 📊 Complete Market Coverage | 🎯 Mass Signal Generation║
║ 🧠 Intelligent Batching | 📈 Real-time Performance | 🚨 Opportunity Detection ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    if not SCAN_ALL_PAIRS:
        print("❌ Comprehensive scanning is disabled. Enable SCAN_ALL_PAIRS=true")
        sys.exit(1)

    scanner = ComprehensiveMarketScanner()
    scanner.run_comprehensive_scan()

if __name__ == "__main__":
    main()
