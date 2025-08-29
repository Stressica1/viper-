#!/usr/bin/env python3
"""
🚀 SCALPING GRID STRATEGY
Advanced grid trading strategy optimized for crypto scalping on lower timeframes

This strategy implements:
✅ Dynamic grid level calculation based on volatility
✅ Range detection and grid placement
✅ Micro-trend identification within ranges
✅ Volume-weighted grid spacing
✅ Automatic grid rebalancing
"""

import numpy as np
import pandas as pd
import talib as ta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SCALPING_GRID - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GridLevel:
    """Individual grid level"""
    price: float
    level_type: str  # 'buy', 'sell'
    filled: bool
    size: float
    target_price: float
    timestamp: datetime

@dataclass
class GridSignal:
    """Grid trading signal"""
    timestamp: datetime
    symbol: str
    direction: str  # 'long', 'short'
    entry_price: float
    exit_price: float
    grid_size: float
    confidence: float
    range_detected: bool
    grid_level: int  # Which grid level (1=first, 2=second, etc.)
    volume_strength: float
    volatility_level: float
    timeframe: str
    expected_profit_pct: float

class ScalpingGridStrategy:
    """
    Advanced Scalping Grid Strategy
    Designed for lower timeframes with high-frequency opportunities
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.active_grids: Dict[str, List[GridLevel]] = {}
        self.current_signals: Dict[str, List[GridSignal]] = {}
        
        logger.info("🚀 Scalping Grid Strategy initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for scalping grid strategy"""
        return {
            # Grid settings
            'grid_levels': 5,  # Number of grid levels above/below center
            'base_grid_spacing_pct': 0.003,  # 0.3% base spacing between levels
            'grid_size_pct': 0.001,  # 0.1% position size per grid level
            'max_grid_spread': 0.02,  # Maximum 2% spread for full grid
            
            # Range detection
            'range_detection_period': 50,
            'range_threshold': 0.01,  # 1% price movement to define range
            'min_range_duration': 20,  # Minimum bars in range
            'range_confirmation_bars': 5,
            
            # Volatility adjustment
            'volatility_period': 20,
            'volatility_multiplier': 1.5,  # Adjust grid spacing by volatility
            'min_volatility': 0.005,  # 0.5% minimum volatility
            'max_volatility': 0.05,   # 5% maximum volatility
            
            # Volume analysis
            'volume_ma_period': 14,
            'volume_threshold': 0.8,  # Minimum volume level
            'volume_surge_multiplier': 2.0,
            
            # Entry conditions
            'rsi_period': 14,
            'rsi_neutral_zone': [40, 60],  # RSI range for grid trading
            'trend_filter_period': 20,
            'max_trend_strength': 0.005,  # Maximum trend for grid trading
            
            # Risk management
            'stop_loss_pct': 0.015,  # 1.5% stop loss
            'take_profit_pct': 0.008,  # 0.8% take profit per level
            'max_open_grids': 10,
            'grid_rebalance_threshold': 0.01,  # 1% price move triggers rebalance
            
            # Timeframe optimization
            'timeframes': ['1m', '5m', '15m'],
            'scalping_timeframe': '1m',
            'min_confidence': 0.6,
            
            # Market conditions
            'market_hours_only': False,
            'min_spread_pct': 0.0005,  # 0.05% minimum spread
            'max_spread_pct': 0.002,   # 0.2% maximum spread
        }

    def detect_range_market(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if market is in a ranging condition suitable for grid trading"""
        if len(df) < self.config['range_detection_period']:
            return {'is_ranging': False, 'range_top': 0, 'range_bottom': 0, 'confidence': 0}
        
        period = self.config['range_detection_period']
        recent_data = df.tail(period)
        
        # Calculate range metrics
        high_price = recent_data['high'].max()
        low_price = recent_data['low'].min()
        price_range = (high_price - low_price) / low_price
        
        # Check if range is suitable for grid trading
        range_suitable = self.config['range_threshold'] <= price_range <= self.config['max_grid_spread']
        
        # Calculate trend strength
        trend_ma = ta.SMA(df['close'].values, timeperiod=self.config['trend_filter_period'])[-period:]
        trend_slope = np.polyfit(range(len(trend_ma)), trend_ma, 1)[0]
        trend_strength = abs(trend_slope / trend_ma[-1]) if len(trend_ma) > 0 else 0
        
        # Range confidence factors
        confidence = 0.5
        
        # Range size appropriate
        if range_suitable:
            confidence += 0.2
            
        # Low trend strength (good for ranging)
        if trend_strength < self.config['max_trend_strength']:
            confidence += 0.2
            
        # Price touching range boundaries
        current_price = df['close'].iloc[-1]
        range_center = (high_price + low_price) / 2
        distance_from_center = abs(current_price - range_center) / range_center
        
        if distance_from_center > 0.3:  # Near boundaries
            confidence += 0.1
            
        # Volume consistency (not too volatile)
        if 'volume' in df.columns:
            volume_cv = recent_data['volume'].std() / recent_data['volume'].mean()
            if volume_cv < 1.0:  # Consistent volume
                confidence += 0.1
        
        is_ranging = (range_suitable and 
                     trend_strength < self.config['max_trend_strength'] and 
                     confidence > 0.6)
        
        return {
            'is_ranging': is_ranging,
            'range_top': high_price,
            'range_bottom': low_price,
            'range_center': range_center,
            'confidence': confidence,
            'trend_strength': trend_strength,
            'price_range_pct': price_range
        }

    def calculate_grid_levels(self, center_price: float, range_info: Dict[str, Any]) -> List[float]:
        """Calculate optimal grid levels based on market conditions"""
        # Get volatility-adjusted spacing
        volatility = range_info.get('price_range_pct', self.config['base_grid_spacing_pct'])
        
        # Adjust spacing based on volatility
        adjusted_spacing = self.config['base_grid_spacing_pct'] * (1 + volatility * self.config['volatility_multiplier'])
        adjusted_spacing = max(self.config['min_volatility'], 
                              min(self.config['max_volatility'], adjusted_spacing))
        
        grid_levels = []
        num_levels = self.config['grid_levels']
        
        # Create levels above and below center price
        for i in range(-num_levels, num_levels + 1):
            if i == 0:
                continue  # Skip center price
                
            level_price = center_price * (1 + i * adjusted_spacing)
            
            # Ensure levels are within reasonable range boundaries
            if (range_info['range_bottom'] <= level_price <= range_info['range_top']):
                grid_levels.append(level_price)
        
        return sorted(grid_levels)

    def identify_grid_signals(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[GridSignal]:
        """Identify scalping grid trading opportunities"""
        if len(df) < 50:
            return []

        signals = []
        
        # Detect range market conditions
        range_info = self.detect_range_market(df)
        
        if not range_info['is_ranging']:
            return signals  # Only trade in ranging markets
            
        current = df.iloc[-1]
        current_price = current['close']
        
        # Calculate RSI for entry timing
        rsi = ta.RSI(df['close'].values, timeperiod=self.config['rsi_period'])[-1]
        if pd.isna(rsi):
            return signals
            
        # Check if RSI is in neutral zone (good for grid trading)
        rsi_neutral = (self.config['rsi_neutral_zone'][0] <= rsi <= self.config['rsi_neutral_zone'][1])
        if not rsi_neutral:
            return signals
            
        # Volume check
        volume_strength = 1.0
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(self.config['volume_ma_period']).mean().iloc[-1]
            volume_strength = current['volume'] / volume_ma if volume_ma > 0 else 1.0
            
        if volume_strength < self.config['volume_threshold']:
            return signals
            
        # Calculate grid levels
        grid_levels = self.calculate_grid_levels(current_price, range_info)
        
        if len(grid_levels) < 2:
            return signals
            
        # Find nearest grid levels above and below current price
        levels_below = [level for level in grid_levels if level < current_price]
        levels_above = [level for level in grid_levels if level > current_price]
        
        # Generate buy signals (at levels below current price)
        if levels_below:
            nearest_buy_level = max(levels_below)
            distance_to_buy = abs(current_price - nearest_buy_level) / current_price
            
            # Only signal if we're close to a grid level
            if distance_to_buy <= self.config['base_grid_spacing_pct'] * 0.5:
                # Calculate exit price (next level up)
                if levels_above:
                    exit_price = min(levels_above)
                else:
                    exit_price = current_price * (1 + self.config['take_profit_pct'])
                
                expected_profit = (exit_price - nearest_buy_level) / nearest_buy_level
                
                # Calculate confidence
                confidence = 0.6
                confidence += 0.1 if range_info['confidence'] > 0.7 else 0
                confidence += 0.1 if volume_strength > 1.2 else 0
                confidence += 0.1 if 45 <= rsi <= 55 else 0  # Very neutral RSI
                confidence += 0.1 if expected_profit > self.config['take_profit_pct'] else 0
                
                if confidence >= self.config['min_confidence']:
                    signal = GridSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        direction='long',
                        entry_price=nearest_buy_level,
                        exit_price=exit_price,
                        grid_size=self.config['grid_size_pct'],
                        confidence=confidence,
                        range_detected=True,
                        grid_level=len([l for l in levels_below if l <= nearest_buy_level]),
                        volume_strength=volume_strength,
                        volatility_level=range_info['price_range_pct'],
                        timeframe=timeframe,
                        expected_profit_pct=expected_profit * 100
                    )
                    
                    signals.append(signal)
                    logger.info(f"📈 Grid BUY signal for {symbol} on {timeframe}: "
                               f"Entry: {nearest_buy_level:.6f}, Exit: {exit_price:.6f}, "
                               f"Profit: {expected_profit*100:.2f}%, Confidence: {confidence:.2f}")

        # Generate sell signals (at levels above current price)
        if levels_above:
            nearest_sell_level = min(levels_above)
            distance_to_sell = abs(nearest_sell_level - current_price) / current_price
            
            if distance_to_sell <= self.config['base_grid_spacing_pct'] * 0.5:
                # Calculate exit price (next level down)
                if levels_below:
                    exit_price = max(levels_below)
                else:
                    exit_price = current_price * (1 - self.config['take_profit_pct'])
                
                expected_profit = (nearest_sell_level - exit_price) / nearest_sell_level
                
                # Calculate confidence
                confidence = 0.6
                confidence += 0.1 if range_info['confidence'] > 0.7 else 0
                confidence += 0.1 if volume_strength > 1.2 else 0
                confidence += 0.1 if 45 <= rsi <= 55 else 0
                confidence += 0.1 if expected_profit > self.config['take_profit_pct'] else 0
                
                if confidence >= self.config['min_confidence']:
                    signal = GridSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        direction='short',
                        entry_price=nearest_sell_level,
                        exit_price=exit_price,
                        grid_size=self.config['grid_size_pct'],
                        confidence=confidence,
                        range_detected=True,
                        grid_level=len([l for l in levels_above if l >= nearest_sell_level]),
                        volume_strength=volume_strength,
                        volatility_level=range_info['price_range_pct'],
                        timeframe=timeframe,
                        expected_profit_pct=expected_profit * 100
                    )
                    
                    signals.append(signal)
                    logger.info(f"📉 Grid SELL signal for {symbol} on {timeframe}: "
                               f"Entry: {nearest_sell_level:.6f}, Exit: {exit_price:.6f}, "
                               f"Profit: {expected_profit*100:.2f}%, Confidence: {confidence:.2f}")

        return signals

    def analyze_symbol(self, symbol: str, df: pd.DataFrame, timeframe: str) -> List[GridSignal]:
        """Analyze symbol for scalping grid opportunities"""
        try:
            # Store historical data
            self.historical_data[f"{symbol}_{timeframe}"] = df
            
            # Identify signals
            signals = self.identify_grid_signals(df, symbol, timeframe)
            
            # Store current signals
            key = f"{symbol}_{timeframe}"
            self.current_signals[key] = signals
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {e}")
            return []

    def get_grid_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get detailed grid analysis"""
        key = f"{symbol}_{timeframe}"
        if key not in self.historical_data:
            return {}
            
        df = self.historical_data[key]
        range_info = self.detect_range_market(df)
        
        current = df.iloc[-1]
        rsi = ta.RSI(df['close'].values, timeperiod=self.config['rsi_period'])[-1]
        
        return {
            'is_ranging': range_info['is_ranging'],
            'range_confidence': range_info['confidence'],
            'range_size_pct': range_info['price_range_pct'] * 100,
            'current_rsi': rsi,
            'rsi_in_neutral_zone': (self.config['rsi_neutral_zone'][0] <= rsi <= self.config['rsi_neutral_zone'][1]),
            'trend_strength': range_info['trend_strength'],
            'signals_count': len(self.current_signals.get(key, [])),
            'range_top': range_info.get('range_top'),
            'range_bottom': range_info.get('range_bottom'),
            'current_price': current['close']
        }

    def get_active_signals(self, symbol: Optional[str] = None) -> List[GridSignal]:
        """Get all active grid signals"""
        active_signals = []
        
        for key, signals in self.current_signals.items():
            if symbol is None or key.startswith(symbol):
                active_signals.extend(signals)
        
        # Sort by expected profit and confidence
        active_signals.sort(key=lambda x: (x.expected_profit_pct, x.confidence), reverse=True)
        return active_signals

# Global instance
_scalping_grid_strategy = None

def get_scalping_grid_strategy() -> ScalpingGridStrategy:
    """Get global scalping grid strategy instance"""
    global _scalping_grid_strategy
    if _scalping_grid_strategy is None:
        _scalping_grid_strategy = ScalpingGridStrategy()
    return _scalping_grid_strategy

# Example usage and testing
async def main():
    """Test scalping grid strategy"""
    print("🚀 SCALPING GRID STRATEGY TEST")
    print("=" * 60)

    strategy = get_scalping_grid_strategy()

    # Generate ranging market data
    dates = pd.date_range('2024-01-01', periods=500, freq='1min')
    np.random.seed(42)
    
    prices = []
    volumes = []
    base_price = 100.0
    
    # Create ranging market with some noise
    for i in range(500):
        # Oscillate around center with some noise
        cycle_position = np.sin(i / 50) * 2  # 2% range
        noise = np.random.normal(0, 0.3)
        close_price = base_price + cycle_position + noise
        
        # Generate OHLC
        open_price = close_price + np.random.normal(0, 0.1)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.2))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
        
        # Steady volume for ranging market
        volume = np.random.lognormal(9, 0.2)
        
        prices.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        })
        volumes.append(volume)

    sample_data = pd.DataFrame({
        'open': [p['open'] for p in prices],
        'high': [p['high'] for p in prices],
        'low': [p['low'] for p in prices],
        'close': [p['close'] for p in prices],
        'volume': volumes
    }, index=dates)

    symbol = "DOTUSDT"
    timeframe = "1m"

    # Analyze the symbol
    signals = strategy.analyze_symbol(symbol, sample_data, timeframe)
    print(f"📊 Found {len(signals)} scalping grid signals")

    for i, signal in enumerate(signals[:5], 1):  # Show first 5 signals
        print(f"\n🎯 Grid Signal {i}:")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Entry: {signal.entry_price:.6f}")
        print(f"   Exit: {signal.exit_price:.6f}")
        print(f"   Expected Profit: {signal.expected_profit_pct:.3f}%")
        print(f"   Grid Level: {signal.grid_level}")
        print(f"   Volume Strength: {signal.volume_strength:.2f}x")
        print(f"   Confidence: {signal.confidence:.2f}")

    # Get grid analysis
    analysis = strategy.get_grid_analysis(symbol, timeframe)
    if analysis:
        print(f"\n📊 Grid Analysis:")
        print(f"   Is Ranging: {analysis['is_ranging']}")
        print(f"   Range Confidence: {analysis['range_confidence']:.2f}")
        print(f"   Range Size: {analysis['range_size_pct']:.2f}%")
        print(f"   Current RSI: {analysis['current_rsi']:.1f}")
        print(f"   RSI in Neutral Zone: {analysis['rsi_in_neutral_zone']}")
        print(f"   Trend Strength: {analysis['trend_strength']:.4f}")
        if analysis.get('range_top'):
            print(f"   Range: {analysis['range_bottom']:.6f} - {analysis['range_top']:.6f}")
            print(f"   Current Price: {analysis['current_price']:.6f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())