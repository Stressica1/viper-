#!/usr/bin/env python3
"""
🚀 VIPER Trading System - Direct Swap Trader for All Pairs
Execute swap trades across all available Bitget pairs using direct API integration

Features:
- Direct Bitget API integration (no MCP dependency)
- Comprehensive pair scanning and analysis
- Automated swap execution for all pairs
- 50x leverage trading with risk management
- Real-time monitoring and logging
- Emergency stop mechanisms
"""

import os
import sys
import time
import json
import ccxt
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from enum import Enum

# Load environment variables
BITGET_API_KEY = os.getenv('BITGET_API_KEY', '')
BITGET_API_SECRET = os.getenv('BITGET_API_SECRET', '')
BITGET_API_PASSWORD = os.getenv('BITGET_API_PASSWORD', '')

class TradeSignal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"

class DirectSwapTrader:
    """
    Direct swap trader for all Bitget pairs with 50x leverage
    """

    def __init__(self):
        """Initialize direct swap trader"""
        self.is_running = False
        self.trades_executed = 0
        self.active_positions = {}
        self.pair_signals = {}

        # Initialize exchange connection
        self.exchange = ccxt.bitget({
            'apiKey': BITGET_API_KEY,
            'secret': BITGET_API_SECRET,
            'password': BITGET_API_PASSWORD,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
            },
            'sandbox': False,
        })

        # Load all available swap pairs
        self.all_pairs = []
        self.load_all_pairs()

        # Trading parameters
        self.risk_per_trade = 0.02  # 2% per trade
        self.max_leverage = 50  # 50x leverage
        self.max_positions = 15  # Maximum concurrent positions
        self.min_volume_threshold = 1000000  # Minimum 24h volume
        self.min_order_size = 0.001  # Minimum order size in base currency

        print(f"📊 Loaded {len(self.all_pairs)} swap pairs")
        print(f"🎯 Risk per trade: {self.risk_per_trade*100}%")

    def load_all_pairs(self) -> None:
        """Load all available swap pairs from Bitget"""
        try:
            markets = self.exchange.loadMarkets()
            self.all_pairs = [
                symbol for symbol in markets.keys()
                if markets[symbol]['active'] and
                markets[symbol]['type'] == 'swap' and
                markets[symbol]['quote'] == 'USDT'
            ]
            print(f"✅ Loaded {len(self.all_pairs)} active swap pairs")
        except Exception as e:
            self.all_pairs = []

    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': float(ticker['last']),
                'high': float(ticker['high']),
                'low': float(ticker['low']),
                'volume': float(ticker['quoteVolume']),
                'price_change': float(ticker['percentage']),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return None

    def calculate_position_size(self, symbol: str, price: float, balance: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get contract size and leverage info
            market = self.exchange.market(symbol)
            contract_size = market.get('contractSize', 1)

            # Calculate position size based on risk
            risk_amount = balance * self.risk_per_trade
            position_size_usd = risk_amount * self.max_leverage

            # Convert to base currency
            position_size_base = position_size_usd / price / contract_size

            # Ensure minimum order size
            return max(position_size_base, self.min_order_size)

        except Exception as e:
            return self.min_order_size

    def calculate_viper_score(self, market_data: Dict) -> float:
        """Calculate VIPER score for trading signal"""
        try:
            price_change = market_data.get('price_change', 0)
            volume = market_data.get('volume', 0)
            high_low_range = market_data.get('high', 1) - market_data.get('low', 0)

            # VIPER scoring algorithm
            volume_score = min(volume / 1000000, 100)  # Volume factor
            price_score = abs(price_change) * 100  # Price movement factor
            range_score = (high_low_range / market_data.get('price', 1)) * 100  # Range factor

            # Combine scores with weights
            viper_score = (volume_score * 0.4) + (price_score * 0.3) + (range_score * 0.3)

            return min(viper_score, 100)

        except Exception as e:
            return 0

    def generate_signal(self, symbol: str, viper_score: float, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal based on VIPER score"""
        try:
            # Check if we already have a position
            if symbol in self.active_positions:
                return None

            # Generate signal based on VIPER score and market conditions
            signal = None
            confidence = 0

            if viper_score >= 85:  # High threshold for quality signals
                price_change = market_data.get('price_change', 0)

                if price_change > 0.5:  # Strong upward movement
                    signal = TradeSignal.LONG
                    confidence = min(viper_score / 100, 1.0)
                elif price_change < -0.5:  # Strong downward movement
                    signal = TradeSignal.SHORT
                    confidence = min(viper_score / 100, 1.0)

            if signal and confidence > 0.8:  # Only trade very high-confidence signals
                return {
                    'symbol': symbol,
                    'signal': signal.value,
                    'viper_score': viper_score,
                    'confidence': confidence,
                    'price': market_data.get('price', 0),
                    'price_change': market_data.get('price_change', 0),
                    'volume': market_data.get('volume', 0),
                    'timestamp': datetime.now().isoformat(),
                    'risk_per_trade': self.risk_per_trade,
                    'leverage': min(self.max_leverage, 25)  # Conservative leverage
                }

            return None

        except Exception as e:
            print(f"❌ Error generating signal for {symbol}: {e}")
            return None

    def execute_swap_trade(self, signal: Dict) -> bool:
        """Execute swap trade directly via Bitget API"""
        try:
            symbol = signal['symbol']
            side = 'buy' if signal['signal'] == 'LONG' else 'sell'

            # Get account balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']

            if usdt_balance < 10:  # Minimum balance check
                print(f"⚠️  Insufficient balance for {symbol}: ${usdt_balance}")
                return False

            # Calculate position size
            position_size = self.calculate_position_size(symbol, signal['price'], usdt_balance)

            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'market',
                'amount': position_size,
                'leverage': signal['leverage']
            }

            print(f"📊 Executing {signal['signal']} order for {symbol}")

            # Execute order (in demo mode for safety)
            if not BITGET_API_KEY:
                print("🎯 DEMO MODE: Would execute trade (no API keys provided)")
                self.trades_executed += 1
                self.active_positions[symbol] = {
                    'signal': signal,
                    'entry_time': datetime.now(),
                    'entry_price': signal['price'],
                    'position_size': position_size
                }
                return True

            # Real order execution
            order = self.exchange.create_order(**order_params)

            if order and order.get('id'):
                self.trades_executed += 1
                self.active_positions[symbol] = {
                    'signal': signal,
                    'entry_time': datetime.now(),
                    'entry_price': signal['price'],
                    'position_size': position_size,
                    'order_id': order['id']
                }
                print(f"✅ Swap trade executed: {symbol} {signal['signal']} at ${signal['price']}")
                return True
            else:
                return False

        except Exception as e:
            return False

    def monitor_positions(self) -> None:
        """Monitor active positions and close if needed"""
        try:
            for symbol, position in list(self.active_positions.items()):
                market_data = self.get_market_data(symbol)
                if not market_data:
                    continue

                entry_price = position['entry_price']
                current_price = market_data['price']
                signal = position['signal']['signal']

                # Simple exit conditions
                if signal == 'LONG' and current_price > entry_price * 1.02:  # 2% profit
                    self.close_position(symbol, "Take Profit")
                elif signal == 'LONG' and current_price < entry_price * 0.98:  # 2% loss
                    self.close_position(symbol, "Stop Loss")
                elif signal == 'SHORT' and current_price < entry_price * 0.98:  # 2% profit
                    self.close_position(symbol, "Take Profit")
                elif signal == 'SHORT' and current_price > entry_price * 1.02:  # 2% loss
                    self.close_position(symbol, "Stop Loss")

        except Exception as e:

    def close_position(self, symbol: str, reason: str) -> None:
        """Close a position"""
        try:
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                signal = position['signal']['signal']

                # Determine close side (opposite of entry)
                close_side = 'sell' if signal == 'LONG' else 'buy'

                if not BITGET_API_KEY:
                    print(f"🎯 DEMO MODE: Would close {symbol} position ({reason})")
                    del self.active_positions[symbol]
                    return

                # Close position
                close_params = {
                    'symbol': symbol,
                    'side': close_side,
                    'type': 'market',
                    'amount': position['position_size']
                }

                order = self.exchange.create_order(**close_params)
                if order and order.get('id'):
                    del self.active_positions[symbol]
                else:

        except Exception as e:

    def start_swap_trading(self) -> None:
        """Start direct swap trading for all pairs"""
        print("\n🚀 STARTING DIRECT SWAP TRADING FOR ALL PAIRS...")
        print("🔥 Scanning and trading all available swap pairs")
        print("⚡ Using 50x leverage with comprehensive risk management")

        if not self.all_pairs:
            return

        self.is_running = True
        scan_count = 0

        try:
            while self.is_running:
                scan_count += 1
                print(f"\n🔍 Market Scan #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")

                # Limit concurrent positions
                if len(self.active_positions) >= self.max_positions:
                    print(f"📊 Max positions reached ({self.max_positions}). Monitoring existing positions...")
                    self.monitor_positions()
                    time.sleep(30)
                    continue

                # Scan all pairs for trading opportunities
                opportunities_found = 0

                for symbol in self.all_pairs[:100]:  # Limit to first 100 pairs for comprehensive scanning
                    try:
                        # Get market data
                        market_data = self.get_market_data(symbol)

                        if market_data and market_data['volume'] > self.min_volume_threshold:
                            # Calculate VIPER score
                            viper_score = self.calculate_viper_score(market_data)

                            # Display market state for high-volume pairs
                            if viper_score > 70:  # Only show interesting pairs
                                print(f"  {symbol:<20} | Price: ${market_data['price']:<10.4f} | "
                                      f"Change: {market_data['price_change']:<+6.2f}% | "
                                      f"VIPER: {viper_score:<5.1f} | "
                                      f"Volume: {market_data['volume']:<10.0f}")

                            # Generate signal
                            signal = self.generate_signal(symbol, viper_score, market_data)
                            if signal:
                                opportunities_found += 1
                                print(f"    🎯 OPPORTUNITY: {signal['signal']} signal with {signal['confidence']:.2f} confidence")

                                # Execute trade
                                if self.execute_swap_trade(signal):

                    except Exception as e:

                if opportunities_found == 0:
                    print("  📊 No high-confidence trading opportunities found in this scan")

                # Monitor existing positions
                if self.active_positions:
                    print(f"\n📊 Active Positions: {len(self.active_positions)}")
                    for symbol, position in self.active_positions.items():
                        entry_price = position['entry_price']
                        current_data = self.get_market_data(symbol)
                        if current_data:
                            pnl_pct = ((current_data['price'] - entry_price) / entry_price) * 100
                            signal = position['signal']['signal']
                            pnl_display = f"{pnl_pct:+.2f}%" if signal == 'LONG' else f"{-pnl_pct:+.2f}%"
                            print(f"   {symbol}: {signal} @ ${entry_price:.4f} | Current: ${current_data['price']:.4f} | P&L: {pnl_display}")

                    self.monitor_positions()

                # Wait before next scan
                print("⏰ Waiting 60 seconds for next comprehensive scan...")
                time.sleep(60)

        except KeyboardInterrupt:
            print("\n\n🛑 Direct Swap Trading stopped by user")
        except Exception as e:
        finally:
            self.is_running = False
            self.emergency_stop()
            print(f"   Trades executed: {self.trades_executed}")
            print(f"   Active positions: {len(self.active_positions)}")

    def emergency_stop(self) -> None:
        """Emergency stop - close all positions"""
        print("\n🚨 EMERGENCY STOP - Closing all positions...")
        for symbol in list(self.active_positions.keys()):
            self.close_position(symbol, "Emergency Stop")

    def stop(self) -> None:
        """Stop the trading system"""
        self.is_running = False

def main():
    """Main entry point"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🚀 VIPER DIRECT SWAP TRADER - ALL PAIRS TRADING                             ║
║ 🔥 Automated Swap Trading | 📊 50x Leverage | 🎯 Direct API Integration     ║
║ ⚡ Real-time Scanning | 🧠 VIPER Signals | 📈 Risk Management               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Check API credentials
    if not all([BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASSWORD]):
        print("⚠️  Warning: API credentials not found in environment variables")
        print("   Running in DEMO MODE (no real trades will be executed)")
        print("   Trading signals will be generated and displayed\n")

    trader = DirectSwapTrader()

    try:
        trader.start_swap_trading()
    except KeyboardInterrupt:
        print("\n\n👋 Direct Swap Trader terminated gracefully")
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()
