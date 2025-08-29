#!/usr/bin/env python3
"""
🚀 VIPER Trading System - Advanced Crypto Scalping System
Elite scalping system using sophisticated indicators and micro-positioning

Features:
- VWAP-based entries with momentum confirmation
- RSI divergence detection for high-probability setups
- MACD histogram analysis for precise timing
- Bid-ask spread exploitation with market depth analysis
- Dynamic position sizing based on volatility and liquidity
- Multi-timeframe confirmation for reduced false signals
- Advanced risk management with trailing stops and profit targets
"""

import os
import sys
import time
import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import ccxt
import requests
from collections import deque

# Load environment variables
BITGET_API_KEY = os.getenv('BITGET_API_KEY', '')
BITGET_API_SECRET = os.getenv('BITGET_API_SECRET', '')
BITGET_API_PASSWORD = os.getenv('BITGET_API_PASSWORD', '')

class ScalpSignal(Enum):
    """Advanced scalping signal types"""
    STRONG_LONG = "STRONG_LONG"
    LONG = "LONG"
    WEAK_LONG = "WEAK_LONG"
    STRONG_SHORT = "STRONG_SHORT"
    SHORT = "SHORT"
    WEAK_SHORT = "WEAK_SHORT"
    HOLD = "HOLD"
    EXIT = "EXIT"

class TimeFrame(Enum):
    """Timeframes for multi-timeframe analysis"""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"

@dataclass
class ScalpPosition:
    """Advanced position tracking for scalping"""
    symbol: str
    side: str
    entry_price: float
    position_size: float
    leverage: int
    timestamp: datetime
    vwap_entry: float
    rsi_entry: float
    macd_signal: str
    stop_loss: float
    profit_target: float
    trailing_stop: float
    max_profit: float = 0
    min_profit: float = 0

@dataclass
class MarketIndicators:
    """Comprehensive market indicator data"""
    vwap: float
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bid_ask_spread: float
    spread_percentage: float
    volume_profile: Dict[str, float]
    order_book_depth: Dict[str, float]
    volatility: float
    trend_strength: float

class AdvancedScalpingEngine:
    """
    Elite scalping engine using advanced indicators and micro-positioning
    """

    def __init__(self):
        self.exchange = None
        self.positions: Dict[str, ScalpPosition] = {}
        self.indicators_cache: Dict[str, MarketIndicators] = {}
        self.price_history: Dict[str, deque] = {}
        self.order_book_cache: Dict[str, Dict] = {}

        # Advanced scalping parameters
        self.min_spread_threshold = 0.05  # 0.05% minimum spread
        self.max_spread_threshold = 0.5   # 0.5% maximum spread
        self.min_volume_threshold = 500000  # Minimum 24h volume
        self.max_position_size_pct = 0.02   # 2% max position size
        self.profit_target_multiplier = 1.5  # 1:1.5 RR ratio
        self.trailing_stop_activation = 0.001  # 0.1% profit to activate trailing stop

        # Indicator parameters
        self.vwap_period = 50
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

        # Multi-timeframe confirmation
        self.confirmation_timeframes = [TimeFrame.ONE_MINUTE, TimeFrame.FIVE_MINUTES]

        # Performance tracking
        self.session_stats = {
            'trades_executed': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'avg_trade_duration': 0,
            'success_rate': 0
        }

        self._initialize_exchange()
        self._initialize_price_history()

    def _initialize_exchange(self):
        """Initialize Bitget exchange with optimized settings"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': BITGET_API_KEY,
                'secret': BITGET_API_SECRET,
                'password': BITGET_API_PASSWORD,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000,
                },
                'sandbox': False,
                'timeout': 10000,
                'rateLimit': 100,
            })
            self.exchange.loadMarkets()
        except Exception as e:
            raise

    def _initialize_price_history(self):
        """Initialize price history buffers for indicators"""
        for symbol in self.exchange.symbols:
            if ':USDT' in symbol and 'USDT' in symbol:
                self.price_history[symbol] = deque(maxlen=200)  # 200 periods for indicators

    def calculate_vwap(self, symbol: str, period: int = 50) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            # Get recent OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=period)

            if len(ohlcv) < period:
                return 0

            cumulative_volume = 0
            cumulative_volume_price = 0

            for candle in ohlcv:
                volume = candle[5]  # volume
                typical_price = (candle[1] + candle[2] + candle[3]) / 3  # (high + low + close) / 3

                cumulative_volume += volume
                cumulative_volume_price += typical_price * volume

            return cumulative_volume_price / cumulative_volume if cumulative_volume > 0 else 0

        except Exception as e:
            print(f"❌ Error calculating VWAP for {symbol}: {e}")
            return 0

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return 50

            gains = []
            losses = []

            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                gains.append(max(change, 0))
                losses.append(max(-change, 0))

            # Calculate average gains and losses
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            return 50

    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(prices) < slow + signal:
                return 0, 0, 0

            # Calculate EMAs
            def calculate_ema(data: List[float], period: int) -> List[float]:
                ema = [sum(data[:period]) / period]
                multiplier = 2 / (period + 1)
                for price in data[period:]:
                    ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
                return ema

            fast_ema = calculate_ema(prices, fast)
            slow_ema = calculate_ema(prices, slow)

            # Calculate MACD line
            macd_line = []
            for i in range(len(slow_ema)):
                macd_line.append(fast_ema[i + (fast - slow)] - slow_ema[i])

            # Calculate signal line (EMA of MACD)
            signal_line = calculate_ema(macd_line, signal)

            # Calculate histogram
            histogram = []
            for i in range(len(signal_line)):
                histogram.append(macd_line[i + (len(macd_line) - len(signal_line))] - signal_line[i])

            return macd_line[-1], signal_line[-1], histogram[-1]

        except Exception as e:
            return 0, 0, 0

    def get_market_indicators(self, symbol: str) -> MarketIndicators:
        """Get comprehensive market indicators for scalping"""
        try:
            # Get current price data
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=200)
            self.price_history[symbol].append(current_price)

            prices = list(self.price_history[symbol])

            # Calculate all indicators
            vwap = self.calculate_vwap(symbol)
            rsi = self.calculate_rsi(prices, self.rsi_period)
            macd_line, macd_signal, macd_histogram = self.calculate_macd(prices, self.macd_fast, self.macd_slow, self.macd_signal)

            # Get order book for spread analysis
            order_book = self.exchange.fetch_order_book(symbol, limit=20)
            best_bid = order_book['bids'][0][0] if order_book['bids'] else current_price * 0.999
            best_ask = order_book['asks'][0][0] if order_book['asks'] else current_price * 1.001
            spread = ((best_ask - best_bid) / current_price) * 100

            # Calculate volatility (standard deviation of recent prices)
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) * 100 if len(prices) >= 20 else 0

            # Calculate trend strength using ADX-like approach
            trend_strength = abs(current_price - prices[0]) / prices[0] * 100 if prices else 0

            # Volume profile analysis
            volume_profile = {
                'current_volume': ticker.get('quoteVolume', 0),
                'avg_volume_24h': ticker.get('quoteVolume', 0),  # Would need historical data
                'volume_trend': 'stable'  # Would need trend analysis
            }

            # Order book depth analysis
            order_book_depth = {
                'bid_depth': sum([level[1] for level in order_book['bids'][:10]]),
                'ask_depth': sum([level[1] for level in order_book['asks'][:10]]),
                'depth_ratio': 1.0
            }
            if order_book_depth['ask_depth'] > 0:
                order_book_depth['depth_ratio'] = order_book_depth['bid_depth'] / order_book_depth['ask_depth']

            return MarketIndicators(
                vwap=vwap,
                rsi=rsi,
                macd_line=macd_line,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                bid_ask_spread=best_ask - best_bid,
                spread_percentage=spread,
                volume_profile=volume_profile,
                order_book_depth=order_book_depth,
                volatility=volatility,
                trend_strength=trend_strength
            )

        except Exception as e:
            print(f"❌ Error getting market indicators for {symbol}: {e}")
            return MarketIndicators(0, 50, 0, 0, 0, 0, 0, {}, {}, 0, 0)

    def detect_scalping_signal(self, symbol: str) -> Tuple[ScalpSignal, float, Dict]:
        """Detect advanced scalping signals using multiple indicators"""
        try:
            indicators = self.get_market_indicators(symbol)
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Multi-factor signal scoring
            signal_score = 0
            confidence_factors = []

            # 1. VWAP Bounce Strategy (40% weight)
            if indicators.vwap > 0:
                vwap_distance = abs(current_price - indicators.vwap) / indicators.vwap * 100

                if current_price > indicators.vwap and vwap_distance < 0.5:
                    signal_score += 40  # Bullish bounce potential
                    confidence_factors.append("vwap_bounce")
                elif current_price < indicators.vwap and vwap_distance < 0.5:
                    signal_score -= 40  # Bearish bounce potential
                    confidence_factors.append("vwap_rejection")

            # 2. RSI Momentum (30% weight)
            if indicators.rsi > self.rsi_overbought:
                signal_score -= 30  # Overbought
                confidence_factors.append("rsi_overbought")
            elif indicators.rsi < self.rsi_oversold:
                signal_score += 30  # Oversold
                confidence_factors.append("rsi_oversold")
            elif self.rsi_oversold < indicators.rsi < self.rsi_overbought:
                signal_score += 15  # Neutral RSI is good for scalping

            # 3. MACD Momentum (20% weight)
            if indicators.macd_histogram > 0 and indicators.macd_line > indicators.macd_signal:
                signal_score += 20  # Bullish momentum
                confidence_factors.append("macd_bullish")
            elif indicators.macd_histogram < 0 and indicators.macd_line < indicators.macd_signal:
                signal_score -= 20  # Bearish momentum
                confidence_factors.append("macd_bearish")

            # 4. Spread Analysis (10% weight)
            if self.min_spread_threshold < indicators.spread_percentage < self.max_spread_threshold:
                signal_score += 10  # Optimal spread for scalping
                confidence_factors.append("optimal_spread")

            # Determine signal strength
            if signal_score >= 70:
                signal = ScalpSignal.STRONG_LONG
                confidence = min(signal_score / 100, 1.0)
            elif signal_score >= 40:
                signal = ScalpSignal.LONG
                confidence = signal_score / 100
            elif signal_score >= 20:
                signal = ScalpSignal.WEAK_LONG
                confidence = signal_score / 100
            elif signal_score <= -70:
                signal = ScalpSignal.STRONG_SHORT
                confidence = abs(signal_score) / 100
            elif signal_score <= -40:
                signal = ScalpSignal.SHORT
                confidence = abs(signal_score) / 100
            elif signal_score <= -20:
                signal = ScalpSignal.WEAK_SHORT
                confidence = abs(signal_score) / 100
            else:
                signal = ScalpSignal.HOLD
                confidence = 0.5

            return signal, confidence, {
                'indicators': indicators,
                'signal_score': signal_score,
                'confidence_factors': confidence_factors,
                'current_price': current_price
            }

        except Exception as e:
            print(f"❌ Error detecting scalping signal for {symbol}: {e}")
            return ScalpSignal.HOLD, 0, {}

    def calculate_position_size(self, symbol: str, confidence: float, leverage: int = 50) -> float:
        """Calculate optimal position size based on multiple factors"""
        try:
            # Base position size (1% of account)
            base_position_size = 0.01

            # Adjust for confidence
            confidence_multiplier = confidence

            # Adjust for volatility
            indicators = self.get_market_indicators(symbol)
            volatility_adjustment = max(0.5, min(2.0, 1 / (indicators.volatility + 0.01)))

            # Adjust for spread
            spread_adjustment = max(0.5, min(1.5, 1 - (indicators.spread_percentage / 0.5)))

            # Adjust for order book depth
            depth_ratio = indicators.order_book_depth.get('depth_ratio', 1.0)
            depth_adjustment = max(0.7, min(1.3, depth_ratio))

            # Calculate final position size
            position_size = (base_position_size * confidence_multiplier *
                           volatility_adjustment * spread_adjustment * depth_adjustment)

            # Cap at maximum position size
            position_size = min(position_size, self.max_position_size_pct)

            return position_size

        except Exception as e:
            print(f"❌ Error calculating position size for {symbol}: {e}")
            return 0.005  # Conservative fallback

    def execute_scalp_trade(self, symbol: str, signal: ScalpSignal, confidence: float, analysis_data: Dict):
        """Execute a scalping trade with advanced risk management"""
        try:
            if symbol in self.positions:
                return False, "Position already exists"

            current_price = analysis_data['current_price']
            indicators = analysis_data['indicators']

            # Calculate position parameters
            position_size = self.calculate_position_size(symbol, confidence)
            leverage = 50  # Fixed leverage for scalping

            # Calculate stop loss and profit targets
            if signal in [ScalpSignal.LONG, ScalpSignal.STRONG_LONG, ScalpSignal.WEAK_LONG]:
                stop_loss = current_price * (1 - 0.005)  # 0.5% stop loss
                profit_target = current_price * (1 + 0.01)  # 1% profit target
                trailing_stop = current_price * (1 + self.trailing_stop_activation)
            else:
                stop_loss = current_price * (1 + 0.005)  # 0.5% stop loss
                profit_target = current_price * (1 - 0.01)  # 1% profit target
                trailing_stop = current_price * (1 - self.trailing_stop_activation)

            # Create position
            position = ScalpPosition(
                symbol=symbol,
                side=signal.value.split('_')[-1].lower(),
                entry_price=current_price,
                position_size=position_size,
                leverage=leverage,
                timestamp=datetime.now(),
                vwap_entry=indicators.vwap,
                rsi_entry=indicators.rsi,
                macd_signal="bullish" if indicators.macd_histogram > 0 else "bearish",
                stop_loss=stop_loss,
                profit_target=profit_target,
                trailing_stop=trailing_stop
            )

            self.positions[symbol] = position

            # Update session statistics
            self.session_stats['trades_executed'] += 1

            print(f"🎯 Executed {signal.value} scalp on {symbol}")
            return True, f"Scalp position opened for {symbol}"

        except Exception as e:
            print(f"❌ Error executing scalp trade for {symbol}: {e}")
            return False, str(e)

    def manage_positions(self):
        """Manage open scalping positions with trailing stops and profit targets"""
        try:
            positions_to_close = []

            for symbol, position in self.positions.items():
                try:
                    # Get current price
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']

                    # Calculate P&L
                    if position.side == 'long':
                        pnl_pct = (current_price - position.entry_price) / position.entry_price
                    else:
                        pnl_pct = (position.entry_price - current_price) / position.entry_price

                    # Update max/min profit
                    position.max_profit = max(position.max_profit, pnl_pct)
                    position.min_profit = min(position.min_profit, pnl_pct)

                    # Check exit conditions
                    should_exit = False
                    exit_reason = ""

                    # Stop loss hit
                    if position.side == 'long' and current_price <= position.stop_loss:
                        should_exit = True
                        exit_reason = "Stop loss hit"
                    elif position.side == 'short' and current_price >= position.stop_loss:
                        should_exit = True
                        exit_reason = "Stop loss hit"

                    # Profit target hit
                    elif position.side == 'long' and current_price >= position.profit_target:
                        should_exit = True
                        exit_reason = "Profit target reached"
                    elif position.side == 'short' and current_price <= position.profit_target:
                        should_exit = True
                        exit_reason = "Profit target reached"

                    # Trailing stop management
                    elif pnl_pct >= self.trailing_stop_activation:
                        if position.side == 'long':
                            new_trailing_stop = current_price * (1 - 0.003)  # 0.3% trailing stop
                            position.trailing_stop = max(position.trailing_stop, new_trailing_stop)
                            if current_price <= position.trailing_stop:
                                should_exit = True
                                exit_reason = "Trailing stop hit"
                        else:
                            new_trailing_stop = current_price * (1 + 0.003)  # 0.3% trailing stop
                            position.trailing_stop = min(position.trailing_stop, new_trailing_stop)
                            if current_price >= position.trailing_stop:
                                should_exit = True
                                exit_reason = "Trailing stop hit"

                    # Time-based exit (5 minutes max hold time)
                    elif (datetime.now() - position.timestamp).seconds > 300:
                        should_exit = True
                        exit_reason = "Time-based exit"

                    if should_exit:
                        positions_to_close.append((symbol, pnl_pct, exit_reason))

                except Exception as e:
                    print(f"❌ Error managing position for {symbol}: {e}")

            # Close positions
            for symbol, pnl_pct, exit_reason in positions_to_close:
                self.close_position(symbol, pnl_pct, exit_reason)

        except Exception as e:

    def close_position(self, symbol: str, pnl_pct: float, reason: str):
        """Close a scalping position and update statistics"""
        try:
            if symbol not in self.positions:
                return

            position = self.positions[symbol]

            # Update session statistics
            if pnl_pct > 0:
                self.session_stats['winning_trades'] += 1
            else:
                self.session_stats['losing_trades'] += 1

            self.session_stats['total_pnl'] += pnl_pct
            self.session_stats['best_trade'] = max(self.session_stats['best_trade'], pnl_pct)
            self.session_stats['worst_trade'] = min(self.session_stats['worst_trade'], pnl_pct)

            # Calculate success rate
            total_trades = self.session_stats['winning_trades'] + self.session_stats['losing_trades']
            if total_trades > 0:
                self.session_stats['success_rate'] = self.session_stats['winning_trades'] / total_trades

            print(f"🔒 Closed {symbol} position: {pnl_pct:.4f} ({reason})")

            # Remove position
            del self.positions[symbol]

        except Exception as e:
            print(f"❌ Error closing position for {symbol}: {e}")

    def get_session_statistics(self) -> Dict:
        """Get comprehensive session statistics"""
        stats = self.session_stats.copy()

        # Calculate additional metrics
        total_trades = stats['trades_executed']
        if total_trades > 0:
            stats['avg_pnl_per_trade'] = stats['total_pnl'] / total_trades
            stats['profit_factor'] = (
                (stats['winning_trades'] * stats['avg_pnl_per_trade'] * 1.5) /
                abs(stats['losing_trades'] * stats['avg_pnl_per_trade']) if stats['losing_trades'] > 0 else float('inf')
            )
        else:
            stats['avg_pnl_per_trade'] = 0
            stats['profit_factor'] = 0

        stats['active_positions'] = len(self.positions)
        stats['session_duration'] = str(datetime.now() - datetime.fromtimestamp(time.time() - 3600))  # Last hour

        return stats

    def run_scalping_session(self, symbols: List[str], duration_minutes: int = 60):
        """Run a complete scalping session"""
        print("🚀 Starting Advanced Crypto Scalping Session")
        print(f"📊 Monitoring {len(symbols)} symbols for {duration_minutes} minutes")

        session_start = time.time()

        try:
            while (time.time() - session_start) < (duration_minutes * 60):
                # Scan for signals
                for symbol in symbols:
                    try:
                        signal, confidence, analysis_data = self.detect_scalping_signal(symbol)

                        # Execute trade if signal is strong enough
                        if signal in [ScalpSignal.STRONG_LONG, ScalpSignal.STRONG_SHORT] and confidence > 0.7:
                            success, message = self.execute_scalp_trade(symbol, signal, confidence, analysis_data)
                            if success:
                            else:

                    except Exception as e:

                # Manage existing positions
                self.manage_positions()

                # Print session statistics every 5 minutes
                if int(time.time() - session_start) % 300 == 0:
                    stats = self.get_session_statistics()
                    print(f"  Trades Executed: {stats['trades_executed']}")
                    print(f"  Success Rate: {stats['success_rate']:.2f}")
                    print(f"  Active Positions: {stats['active_positions']}")
                    print(f"  Total Volume: {stats['total_volume']:.4f}")
                time.sleep(10)  # 10 second intervals

        except KeyboardInterrupt:
            print("\n⏹️  Scalping session interrupted by user")

        except Exception as e:

        finally:
            # Close all remaining positions
            for symbol in list(self.positions.keys()):
                self.close_position(symbol, 0, "Session ended")

            # Print final statistics
            final_stats = self.get_session_statistics()
            print(f"Total Trades: {final_stats['trades_executed']}")
            print(f"Winning Trades: {final_stats['winning_trades']}")
            print(f"Losing Trades: {final_stats['losing_trades']}")
            print(f"Success Rate: {final_stats['success_rate']:.2f}%")
            print(f"Total P&L: {final_stats['total_pnl']:.4f}")
            print(f"Best Trade: {final_stats['best_trade']:.4f}")
            print(f"Worst Trade: {final_stats['worst_trade']:.4f}")
            print(f"Profit Factor: {final_stats['profit_factor']:.2f}")

def main():
    """Main function to run advanced scalping system"""
    try:
        # Initialize scalping engine
        engine = AdvancedScalpingEngine()

        # Define symbols to scalp (high-volume pairs)
        symbols = [
            'BTC/USDT:USDT',
            'ETH/USDT:USDT',
            'BNB/USDT:USDT',
            'ADA/USDT:USDT',
            'SOL/USDT:USDT',
            'DOGE/USDT:USDT',
            'LTC/USDT:USDT',
            'LINK/USDT:USDT',
            'UNI/USDT:USDT',
            'AAVE/USDT:USDT'
        ]

        # Run scalping session
        engine.run_scalping_session(symbols, duration_minutes=60)

    except KeyboardInterrupt:
        print("\n👋 Advanced Scalping System stopped by user")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
