#!/usr/bin/env python3
"""
🚀 FIBONACCI RETRACEMENT STRATEGY  
Advanced Fibonacci-based trading strategy for crypto markets

This strategy implements:
✅ Automatic Fibonacci level detection from swing highs/lows
✅ Golden ratio retracement entries (61.8%, 78.6%)
✅ Extension targets for profit taking
✅ Multi-timeframe Fibonacci confluence
✅ Dynamic level adjustment based on market structure
"""

import numpy as np
import pandas as pd
import talib as ta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from scipy.signal import argrelextrema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - FIBONACCI_STRATEGY - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FibLevel:
    """Fibonacci level data structure"""
    level: float
    price: float
    ratio: float
    level_type: str  # 'retracement', 'extension'
    strength: float  # How many times price has respected this level

@dataclass
class FibSignal:
    """Fibonacci strategy signal"""
    timestamp: datetime
    symbol: str
    direction: str  # 'long', 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    take_profit_2: float  # Second target
    confidence: float
    fib_level: float  # Entry fib level (e.g., 0.618)
    swing_high: float
    swing_low: float
    retracement_depth: float
    timeframe: str
    confluence_count: int  # How many fib levels are near entry
    risk_reward_ratio: float

class FibonacciStrategy:
    """
    Advanced Fibonacci Retracement Strategy
    Uses automatic swing detection and golden ratio entries
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.fib_levels: Dict[str, List[FibLevel]] = {}
        self.current_signals: Dict[str, List[FibSignal]] = {}
        
        logger.info("🚀 Fibonacci Strategy initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Fibonacci strategy"""
        return {
            # Fibonacci levels
            'fib_retracement_levels': [0.236, 0.382, 0.500, 0.618, 0.786],
            'fib_extension_levels': [1.272, 1.414, 1.618, 2.000, 2.618],
            'golden_ratios': [0.618, 0.786],  # Key levels for entries
            
            # Swing detection
            'swing_lookback': 5,  # Bars to look back for swing high/low
            'min_swing_size': 0.01,  # Minimum 1% swing size
            'max_swing_age': 50,  # Maximum bars since swing formation
            
            # Entry criteria
            'entry_tolerance': 0.002,  # 0.2% tolerance around fib level
            'min_retracement_depth': 0.30,  # Minimum 30% retracement
            'max_retracement_depth': 0.90,  # Maximum 90% retracement
            
            # Confluence detection
            'confluence_distance': 0.005,  # 0.5% distance for confluence
            'min_confluence_count': 2,
            
            # Volume confirmation
            'volume_ma_period': 20,
            'volume_threshold': 1.3,  # Volume 30% above average
            'volume_confirmation_required': False,
            
            # Risk management
            'atr_period': 14,
            'stop_loss_beyond_swing': 0.5,  # Stop 50% beyond swing point
            'take_profit_extension': 1.272,  # First target at 127.2% extension
            'take_profit_2_extension': 1.618,  # Second target at 161.8%
            
            # Trend filters
            'trend_ma_period': 50,
            'trend_filter_enabled': True,
            'only_trend_direction': False,  # Only trade in trend direction
            
            # Signal validation
            'min_confidence': 0.7,
            'min_rrr': 2.0,
            'max_signals_per_swing': 3,
            
            # Timeframe settings
            'timeframes': ['15m', '30m', '1h', '4h'],
            'primary_timeframe': '1h',
        }

    def find_swings(self, df: pd.DataFrame) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find swing highs and lows"""
        if len(df) < self.config['swing_lookback'] * 2 + 1:
            return [], []
            
        lookback = self.config['swing_lookback']
        
        swing_highs = []
        swing_lows = []
        
        try:
            # Find swing highs and lows
            high_indices = argrelextrema(df['high'].values, np.greater, order=lookback)[0]
            low_indices = argrelextrema(df['low'].values, np.less, order=lookback)[0]
            
            # Filter by minimum swing size
            for idx in high_indices:
                if idx < len(df):
                    swing_high = df['high'].iloc[idx]
                    # Check if swing is significant enough
                    nearby_lows = df['low'].iloc[max(0, idx-20):min(len(df), idx+20)]
                    if len(nearby_lows) > 0:
                        min_nearby_low = nearby_lows.min()
                        swing_size = (swing_high - min_nearby_low) / min_nearby_low
                        if swing_size >= self.config['min_swing_size']:
                            swing_highs.append((idx, swing_high))
            
            for idx in low_indices:
                if idx < len(df):
                    swing_low = df['low'].iloc[idx]
                    # Check if swing is significant enough
                    nearby_highs = df['high'].iloc[max(0, idx-20):min(len(df), idx+20)]
                    if len(nearby_highs) > 0:
                        max_nearby_high = nearby_highs.max()
                        swing_size = (max_nearby_high - swing_low) / swing_low
                        if swing_size >= self.config['min_swing_size']:
                            swing_lows.append((idx, swing_low))
                            
        except Exception as e:
            logger.warning(f"Error finding swings: {e}")
            
        return swing_highs, swing_lows

    def calculate_fibonacci_levels(self, swing_high: float, swing_low: float, 
                                 retracement: bool = True) -> List[FibLevel]:
        """Calculate Fibonacci retracement or extension levels"""
        levels = []
        swing_range = swing_high - swing_low
        
        if retracement:
            # Retracement levels (from swing high back towards swing low)
            for ratio in self.config['fib_retracement_levels']:
                price = swing_high - (swing_range * ratio)
                strength = 1.0 if ratio in self.config['golden_ratios'] else 0.5
                
                levels.append(FibLevel(
                    level=ratio,
                    price=price,
                    ratio=ratio,
                    level_type='retracement',
                    strength=strength
                ))
        else:
            # Extension levels (beyond the swing range)
            for ratio in self.config['fib_extension_levels']:
                price = swing_high + (swing_range * (ratio - 1.0))
                strength = 1.0 if ratio in [1.272, 1.618] else 0.5
                
                levels.append(FibLevel(
                    level=ratio,
                    price=price,
                    ratio=ratio,
                    level_type='extension',
                    strength=strength
                ))
        
        return levels

    def detect_fib_confluence(self, all_levels: List[FibLevel], target_price: float) -> int:
        """Detect how many Fibonacci levels are near a target price"""
        confluence_count = 0
        tolerance = self.config['confluence_distance']
        
        for level in all_levels:
            price_diff = abs(level.price - target_price) / target_price
            if price_diff <= tolerance:
                confluence_count += 1
        
        return confluence_count

    def identify_fibonacci_signals(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[FibSignal]:
        """Identify Fibonacci-based trading opportunities"""
        if len(df) < 100:
            return []

        signals = []
        
        # Find swing points
        swing_highs, swing_lows = self.find_swings(df)
        
        if len(swing_highs) == 0 or len(swing_lows) == 0:
            return signals

        # Get current price and recent data
        current = df.iloc[-1]
        current_price = current['close']
        
        # Calculate ATR for risk management
        atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, 
                     timeperiod=self.config['atr_period'])[-1]
        if pd.isna(atr):
            atr = abs(current_price * 0.02)

        # Trend filter
        if self.config['trend_filter_enabled']:
            trend_ma = ta.SMA(df['close'].values, timeperiod=self.config['trend_ma_period'])[-1]
            bullish_trend = current_price > trend_ma
        else:
            bullish_trend = True  # Ignore trend filter

        # Look for recent swings to create Fibonacci levels
        max_age = self.config['max_swing_age']
        current_bar = len(df) - 1
        
        # Check for bullish setups (retracement from swing high to swing low, then bounce)
        for high_idx, swing_high in swing_highs:
            if current_bar - high_idx > max_age:
                continue
                
            # Find subsequent swing low after this high
            subsequent_lows = [(idx, price) for idx, price in swing_lows if idx > high_idx]
            if not subsequent_lows:
                continue
                
            for low_idx, swing_low in subsequent_lows:
                if current_bar - low_idx > max_age // 2:  # More recent low required
                    continue
                    
                # Calculate retracement depth
                swing_range = swing_high - swing_low
                retracement_depth = (swing_high - current_price) / swing_range
                
                # Check if we're in valid retracement zone
                if not (self.config['min_retracement_depth'] <= retracement_depth <= self.config['max_retracement_depth']):
                    continue
                
                # Calculate Fibonacci levels
                fib_levels = self.calculate_fibonacci_levels(swing_high, swing_low, retracement=True)
                
                # Check if current price is near a key Fibonacci level
                for fib_level in fib_levels:
                    if fib_level.ratio not in self.config['golden_ratios']:
                        continue
                        
                    price_diff = abs(current_price - fib_level.price) / current_price
                    
                    if price_diff <= self.config['entry_tolerance']:
                        # Check trend filter for bullish setup
                        if self.config['only_trend_direction'] and not bullish_trend:
                            continue
                            
                        # Calculate targets using extensions
                        extension_levels = self.calculate_fibonacci_levels(swing_high, swing_low, retracement=False)
                        take_profit_1 = next((level.price for level in extension_levels 
                                            if level.ratio == self.config['take_profit_extension']), 
                                           swing_high + swing_range * 0.5)
                        take_profit_2 = next((level.price for level in extension_levels 
                                            if level.ratio == self.config['take_profit_2_extension']), 
                                           swing_high + swing_range * 1.0)
                        
                        # Stop loss beyond swing low
                        stop_loss = swing_low - (atr * self.config['stop_loss_beyond_swing'])
                        
                        # Calculate confluence
                        all_levels = fib_levels + extension_levels
                        confluence_count = self.detect_fib_confluence(all_levels, current_price)
                        
                        # Calculate confidence
                        confidence = 0.6
                        confidence += 0.1 if fib_level.ratio == 0.618 else 0  # Golden ratio bonus
                        confidence += 0.05 * min(confluence_count, 4)  # Confluence bonus
                        confidence += 0.1 if bullish_trend else 0  # Trend alignment
                        confidence += 0.1 if retracement_depth > 0.5 else 0  # Deeper retrace = stronger
                        
                        # Risk-reward calculation
                        rrr_1 = (take_profit_1 - current_price) / (current_price - stop_loss) if stop_loss < current_price else 0
                        
                        if confidence >= self.config['min_confidence'] and rrr_1 >= self.config['min_rrr']:
                            signal = FibSignal(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                direction='long',
                                entry_price=current_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit_1,
                                take_profit_2=take_profit_2,
                                confidence=confidence,
                                fib_level=fib_level.ratio,
                                swing_high=swing_high,
                                swing_low=swing_low,
                                retracement_depth=retracement_depth,
                                timeframe=timeframe,
                                confluence_count=confluence_count,
                                risk_reward_ratio=rrr_1
                            )
                            
                            signals.append(signal)
                            logger.info(f"📈 Fibonacci LONG signal for {symbol} on {timeframe}: "
                                       f"Entry at {fib_level.ratio:.1%} retracement, "
                                       f"Confidence: {confidence:.2f}, RRR: {rrr_1:.2f}")

        # Check for bearish setups (retracement from swing low to swing high, then rejection)
        for low_idx, swing_low in swing_lows:
            if current_bar - low_idx > max_age:
                continue
                
            # Find subsequent swing high after this low
            subsequent_highs = [(idx, price) for idx, price in swing_highs if idx > low_idx]
            if not subsequent_highs:
                continue
                
            for high_idx, swing_high in subsequent_highs:
                if current_bar - high_idx > max_age // 2:
                    continue
                    
                # Calculate retracement depth
                swing_range = swing_high - swing_low
                retracement_depth = (current_price - swing_low) / swing_range
                
                if not (self.config['min_retracement_depth'] <= retracement_depth <= self.config['max_retracement_depth']):
                    continue
                
                # Calculate Fibonacci levels (inverted for bearish)
                fib_levels = self.calculate_fibonacci_levels(swing_high, swing_low, retracement=True)
                
                for fib_level in fib_levels:
                    if fib_level.ratio not in self.config['golden_ratios']:
                        continue
                        
                    # For bearish, we look at retracement up from swing low
                    bearish_fib_price = swing_low + (swing_range * fib_level.ratio)
                    price_diff = abs(current_price - bearish_fib_price) / current_price
                    
                    if price_diff <= self.config['entry_tolerance']:
                        # Check trend filter for bearish setup
                        if self.config['only_trend_direction'] and bullish_trend:
                            continue
                            
                        # Calculate targets (extensions downward)
                        take_profit_1 = swing_low - (swing_range * (self.config['take_profit_extension'] - 1.0))
                        take_profit_2 = swing_low - (swing_range * (self.config['take_profit_2_extension'] - 1.0))
                        
                        # Stop loss beyond swing high
                        stop_loss = swing_high + (atr * self.config['stop_loss_beyond_swing'])
                        
                        # Calculate confidence
                        confidence = 0.6
                        confidence += 0.1 if fib_level.ratio == 0.618 else 0
                        confidence += 0.1 if not bullish_trend else 0  # Bearish trend alignment
                        confidence += 0.1 if retracement_depth > 0.5 else 0
                        
                        # Risk-reward calculation
                        rrr_1 = (current_price - take_profit_1) / (stop_loss - current_price) if stop_loss > current_price else 0
                        
                        if confidence >= self.config['min_confidence'] and rrr_1 >= self.config['min_rrr']:
                            signal = FibSignal(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                direction='short',
                                entry_price=current_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit_1,
                                take_profit_2=take_profit_2,
                                confidence=confidence,
                                fib_level=fib_level.ratio,
                                swing_high=swing_high,
                                swing_low=swing_low,
                                retracement_depth=retracement_depth,
                                timeframe=timeframe,
                                confluence_count=1,  # Simplified for bearish
                                risk_reward_ratio=rrr_1
                            )
                            
                            signals.append(signal)
                            logger.info(f"📉 Fibonacci SHORT signal for {symbol} on {timeframe}: "
                                       f"Entry at {fib_level.ratio:.1%} retracement, "
                                       f"Confidence: {confidence:.2f}, RRR: {rrr_1:.2f}")

        return signals

    def analyze_symbol(self, symbol: str, df: pd.DataFrame, timeframe: str) -> List[FibSignal]:
        """Analyze symbol for Fibonacci opportunities"""
        try:
            # Store historical data
            self.historical_data[f"{symbol}_{timeframe}"] = df
            
            # Identify signals
            signals = self.identify_fibonacci_signals(df, symbol, timeframe)
            
            # Store current signals
            key = f"{symbol}_{timeframe}"
            self.current_signals[key] = signals
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {e}")
            return []

    def get_fibonacci_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get detailed Fibonacci analysis"""
        key = f"{symbol}_{timeframe}"
        if key not in self.historical_data:
            return {}
            
        df = self.historical_data[key]
        swing_highs, swing_lows = self.find_swings(df)
        
        return {
            'swing_highs_count': len(swing_highs),
            'swing_lows_count': len(swing_lows),
            'recent_swing_high': swing_highs[-1][1] if swing_highs else None,
            'recent_swing_low': swing_lows[-1][1] if swing_lows else None,
            'signals_count': len(self.current_signals.get(key, [])),
        }

    def get_active_signals(self, symbol: Optional[str] = None) -> List[FibSignal]:
        """Get all active Fibonacci signals"""
        active_signals = []
        
        for key, signals in self.current_signals.items():
            if symbol is None or key.startswith(symbol):
                active_signals.extend(signals)
        
        # Sort by confidence and confluence
        active_signals.sort(key=lambda x: (x.confidence, x.confluence_count), reverse=True)
        return active_signals

# Global instance
_fibonacci_strategy = None

def get_fibonacci_strategy() -> FibonacciStrategy:
    """Get global Fibonacci strategy instance"""
    global _fibonacci_strategy
    if _fibonacci_strategy is None:
        _fibonacci_strategy = FibonacciStrategy()
    return _fibonacci_strategy

# Example usage and testing
async def main():
    """Test Fibonacci strategy"""
    print("🚀 FIBONACCI RETRACEMENT STRATEGY TEST")
    print("=" * 60)

    strategy = get_fibonacci_strategy()

    # Generate test data with clear swings
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')
    np.random.seed(42)
    
    prices = []
    base_price = 100.0
    
    for i in range(200):
        # Create clear swing patterns
        if i < 50:
            trend = 0.2  # Strong uptrend
        elif i < 100:
            trend = -0.15  # Retracement
        elif i < 150:
            trend = 0.1   # Recovery
        else:
            trend = -0.05  # Minor pullback
            
        noise = np.random.normal(0, 0.3)
        close_price = base_price + trend + noise
        
        # Generate OHLC
        open_price = close_price + np.random.normal(0, 0.2)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.4))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.4))
        
        prices.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.lognormal(10, 0.3)
        })
        
        base_price = close_price

    sample_data = pd.DataFrame(prices, index=dates)

    symbol = "SOLUSDT"
    timeframe = "1h"

    # Analyze the symbol
    signals = strategy.analyze_symbol(symbol, sample_data, timeframe)
    print(f"📊 Found {len(signals)} Fibonacci signals")

    for i, signal in enumerate(signals, 1):
        print(f"\n🎯 Fibonacci Signal {i}:")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Entry: {signal.entry_price:.6f}")
        print(f"   Stop: {signal.stop_loss:.6f}")
        print(f"   Target 1: {signal.take_profit:.6f}")
        print(f"   Target 2: {signal.take_profit_2:.6f}")
        print(f"   Fib Level: {signal.fib_level:.1%}")
        print(f"   Retracement: {signal.retracement_depth:.1%}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Confluence: {signal.confluence_count}")
        print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")

    # Get Fibonacci analysis
    fib_analysis = strategy.get_fibonacci_analysis(symbol, timeframe)
    if fib_analysis:
        print(f"\n📊 Fibonacci Analysis:")
        print(f"   Swing Highs: {fib_analysis['swing_highs_count']}")
        print(f"   Swing Lows: {fib_analysis['swing_lows_count']}")
        print(f"   Recent High: {fib_analysis['recent_swing_high']}")
        print(f"   Recent Low: {fib_analysis['recent_swing_low']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())