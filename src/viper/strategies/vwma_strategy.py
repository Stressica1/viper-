#!/usr/bin/env python3
"""
🚀 VOLUME WEIGHTED MOVING AVERAGE (VWMA) STRATEGY
Advanced volume-based trend following strategy for crypto markets

This strategy implements:
✅ Volume-Weighted Moving Average calculations
✅ Volume-Price Trend Analysis
✅ Multi-timeframe VWMA confluences
✅ Volume surge detection for breakouts
✅ Dynamic position sizing based on volume strength
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
    format='%(asctime)s - VWMA_STRATEGY - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VWMASignal:
    """VWMA strategy signal"""
    timestamp: datetime
    symbol: str
    direction: str  # 'long', 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    vwma_fast: float
    vwma_slow: float
    volume_strength: float
    price_volume_trend: float
    timeframe: str
    position_size_multiplier: float
    risk_reward_ratio: float

class VWMAStrategy:
    """
    Volume-Weighted Moving Average Strategy
    Uses volume confirmation for trend following in crypto markets
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_signals: Dict[str, List[VWMASignal]] = {}
        
        logger.info("🚀 VWMA Strategy initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for VWMA strategy"""
        return {
            # VWMA settings
            'vwma_fast_period': 10,
            'vwma_slow_period': 21,
            'vwma_signal_period': 5,  # For signal smoothing
            
            # Volume analysis
            'volume_ma_period': 20,
            'volume_surge_multiplier': 2.0,  # Volume must be 2x average
            'volume_trend_period': 14,
            'min_volume_strength': 1.5,  # Minimum volume strength for signals
            
            # Price-Volume Trend (PVT)
            'pvt_smoothing_period': 9,
            'pvt_signal_threshold': 0.001,  # 0.1% threshold for PVT signals
            
            # Trend confirmation
            'trend_confirmation_bars': 3,
            'min_trend_strength': 0.002,  # 0.2% minimum price movement
            
            # Risk management
            'atr_period': 14,
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_atr_multiplier': 3.5,
            'trailing_stop_enabled': True,
            'trailing_stop_atr_multiplier': 1.5,
            
            # Position sizing
            'base_position_size': 1.0,
            'max_position_multiplier': 2.0,
            'volume_position_scaling': True,
            
            # Entry filters
            'min_confidence': 0.65,
            'min_rrr': 2.0,
            'max_spread_pct': 0.1,  # Max 0.1% spread
            
            # Timeframe settings
            'timeframes': ['5m', '15m', '30m', '1h'],
            'primary_timeframe': '15m',
        }

    def calculate_vwma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume-Weighted Moving Average"""
        if 'volume' not in df.columns:
            # Fallback to regular MA if no volume
            df['vwma_fast'] = ta.SMA(df['close'].values, timeperiod=self.config['vwma_fast_period'])
            df['vwma_slow'] = ta.SMA(df['close'].values, timeperiod=self.config['vwma_slow_period'])
            return df

        fast_period = self.config['vwma_fast_period']
        slow_period = self.config['vwma_slow_period']
        
        # Calculate VWMA manually
        def vwma(prices, volumes, period):
            if len(prices) < period:
                return pd.Series([np.nan] * len(prices), index=prices.index)
                
            vwma_values = []
            for i in range(len(prices)):
                if i < period - 1:
                    vwma_values.append(np.nan)
                else:
                    price_slice = prices.iloc[i-period+1:i+1]
                    volume_slice = volumes.iloc[i-period+1:i+1]
                    
                    if volume_slice.sum() > 0:
                        vwma_val = (price_slice * volume_slice).sum() / volume_slice.sum()
                    else:
                        vwma_val = price_slice.mean()
                    
                    vwma_values.append(vwma_val)
                    
            return pd.Series(vwma_values, index=prices.index)

        df['vwma_fast'] = vwma(df['close'], df['volume'], fast_period)
        df['vwma_slow'] = vwma(df['close'], df['volume'], slow_period)
        
        # VWMA signal line (smoothed difference)
        vwma_diff = df['vwma_fast'] - df['vwma_slow']
        df['vwma_signal'] = ta.SMA(vwma_diff.values, timeperiod=self.config['vwma_signal_period'])
        
        return df

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        if 'volume' not in df.columns:
            df['volume_ma'] = 1
            df['volume_ratio'] = 1
            df['volume_trend'] = 0
            df['price_volume_trend'] = 0
            return df

        # Volume moving average and ratio
        volume_period = self.config['volume_ma_period']
        df['volume_ma'] = df['volume'].rolling(window=volume_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volume trend (increasing/decreasing volume)
        trend_period = self.config['volume_trend_period']
        df['volume_trend'] = df['volume'].rolling(window=trend_period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == trend_period else 0,
            raw=False
        )
        
        # Price-Volume Trend (PVT)
        df['price_change_pct'] = df['close'].pct_change()
        df['price_volume_trend'] = (df['price_change_pct'] * df['volume']).cumsum()
        
        # Smooth PVT
        pvt_period = self.config['pvt_smoothing_period']
        df['pvt_smooth'] = ta.SMA(df['price_volume_trend'].values, timeperiod=pvt_period)
        
        return df

    def identify_vwma_signals(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[VWMASignal]:
        """Identify VWMA trading signals"""
        if len(df) < 50:
            return []

        signals = []
        
        # Get recent data for analysis
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        # Check for valid data
        if pd.isna(current['vwma_fast']) or pd.isna(current['vwma_slow']):
            return signals

        # Volume strength check
        volume_strength = current['volume_ratio']
        if volume_strength < self.config['min_volume_strength']:
            return signals

        # ATR for risk management
        atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, 
                     timeperiod=self.config['atr_period'])[-1]
        if pd.isna(atr):
            atr = abs(current['close'] * 0.02)

        # VWMA crossover signals
        vwma_fast_curr = current['vwma_fast']
        vwma_slow_curr = current['vwma_slow']
        vwma_fast_prev = previous['vwma_fast']
        vwma_slow_prev = previous['vwma_slow']
        
        # Check for crossovers
        bullish_cross = (vwma_fast_curr > vwma_slow_curr and vwma_fast_prev <= vwma_slow_prev)
        bearish_cross = (vwma_fast_curr < vwma_slow_curr and vwma_fast_prev >= vwma_slow_prev)
        
        # Trend confirmation
        price_trend = (current['close'] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        trend_confirmed = abs(price_trend) > self.config['min_trend_strength']
        
        # PVT confirmation
        pvt_trend = current['price_volume_trend'] - previous['price_volume_trend']
        pvt_confirmed = abs(pvt_trend) > self.config['pvt_signal_threshold']

        signal_direction = None
        confidence = 0.5
        
        if bullish_cross and trend_confirmed and price_trend > 0:
            signal_direction = 'long'
            entry_price = current['close']
            stop_loss = entry_price - (atr * self.config['stop_loss_atr_multiplier'])
            take_profit = entry_price + (atr * self.config['take_profit_atr_multiplier'])
            
            # Confidence factors for long
            confidence += 0.15 if pvt_trend > 0 else 0
            confidence += 0.1 if volume_strength > 2.0 else 0
            confidence += 0.1 if current['volume_trend'] > 0 else 0
            confidence += 0.15 if vwma_fast_curr > vwma_fast_prev else 0
            
        elif bearish_cross and trend_confirmed and price_trend < 0:
            signal_direction = 'short'
            entry_price = current['close']
            stop_loss = entry_price + (atr * self.config['stop_loss_atr_multiplier'])
            take_profit = entry_price - (atr * self.config['take_profit_atr_multiplier'])
            
            # Confidence factors for short
            confidence += 0.15 if pvt_trend < 0 else 0
            confidence += 0.1 if volume_strength > 2.0 else 0
            confidence += 0.1 if current['volume_trend'] > 0 else 0  # High volume on breakdown
            confidence += 0.15 if vwma_fast_curr < vwma_fast_prev else 0

        if signal_direction is None:
            return signals

        # Volume surge bonus
        if volume_strength > self.config['volume_surge_multiplier']:
            confidence += 0.1
            
        # Calculate risk-reward ratio
        if signal_direction == 'long':
            rrr = (take_profit - entry_price) / (entry_price - stop_loss) if stop_loss < entry_price else 0
        else:
            rrr = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else 0

        # Position sizing based on volume strength
        position_multiplier = 1.0
        if self.config['volume_position_scaling']:
            # Scale position size based on volume strength (capped at max multiplier)
            vol_scaling = min(volume_strength / 2.0, self.config['max_position_multiplier'])
            position_multiplier = max(0.5, vol_scaling)

        # Filter by minimum requirements
        if confidence >= self.config['min_confidence'] and rrr >= self.config['min_rrr']:
            
            signal = VWMASignal(
                timestamp=datetime.now(),
                symbol=symbol,
                direction=signal_direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                vwma_fast=vwma_fast_curr,
                vwma_slow=vwma_slow_curr,
                volume_strength=volume_strength,
                price_volume_trend=pvt_trend,
                timeframe=timeframe,
                position_size_multiplier=position_multiplier,
                risk_reward_ratio=rrr
            )
            
            signals.append(signal)
            logger.info(f"📈 VWMA {signal_direction.upper()} signal for {symbol} on {timeframe}: "
                       f"Entry: {entry_price:.6f}, Vol Strength: {volume_strength:.2f}, "
                       f"Confidence: {confidence:.2f}, RRR: {rrr:.2f}")

        return signals

    def analyze_symbol(self, symbol: str, df: pd.DataFrame, timeframe: str) -> List[VWMASignal]:
        """Analyze symbol for VWMA signals"""
        try:
            # Calculate all indicators
            df = self.calculate_vwma(df)
            df = self.calculate_volume_indicators(df)
            
            # Store historical data
            self.historical_data[f"{symbol}_{timeframe}"] = df
            
            # Identify signals
            signals = self.identify_vwma_signals(df, symbol, timeframe)
            
            # Store current signals
            key = f"{symbol}_{timeframe}"
            self.current_signals[key] = signals
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {e}")
            return []

    def get_volume_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get detailed volume analysis for a symbol"""
        key = f"{symbol}_{timeframe}"
        if key not in self.historical_data:
            return {}
            
        df = self.historical_data[key]
        if len(df) == 0:
            return {}
            
        current = df.iloc[-1]
        
        return {
            'volume_strength': current.get('volume_ratio', 1.0),
            'volume_trend': 'increasing' if current.get('volume_trend', 0) > 0 else 'decreasing',
            'pvt_trend': current.get('price_volume_trend', 0),
            'volume_surge': current.get('volume_ratio', 1.0) > self.config['volume_surge_multiplier'],
            'vwma_direction': 'bullish' if current.get('vwma_fast', 0) > current.get('vwma_slow', 0) else 'bearish'
        }

    def get_active_signals(self, symbol: Optional[str] = None) -> List[VWMASignal]:
        """Get all active VWMA signals"""
        active_signals = []
        
        for key, signals in self.current_signals.items():
            if symbol is None or key.startswith(symbol):
                active_signals.extend(signals)
        
        # Sort by confidence and volume strength
        active_signals.sort(key=lambda x: (x.confidence, x.volume_strength), reverse=True)
        return active_signals

# Global instance
_vwma_strategy = None

def get_vwma_strategy() -> VWMAStrategy:
    """Get global VWMA strategy instance"""
    global _vwma_strategy
    if _vwma_strategy is None:
        _vwma_strategy = VWMAStrategy()
    return _vwma_strategy

# Example usage and testing
async def main():
    """Test VWMA strategy"""
    print("🚀 VOLUME-WEIGHTED MOVING AVERAGE STRATEGY TEST")
    print("=" * 60)

    strategy = get_vwma_strategy()

    # Generate test data with volume patterns
    dates = pd.date_range('2024-01-01', periods=150, freq='15min')
    np.random.seed(42)
    
    prices = []
    volumes = []
    base_price = 100.0
    
    for i in range(150):
        # Create volume surge at trend changes
        if i in [50, 100]:  # Trend change points
            volume_multiplier = 3.0
        elif 45 <= i <= 55 or 95 <= i <= 105:  # Around trend changes
            volume_multiplier = 2.0
        else:
            volume_multiplier = 1.0
            
        # Price movement with some trending
        if i < 50:
            trend = 0.1  # Uptrend
        elif i < 100:
            trend = -0.05  # Downtrend
        else:
            trend = 0.08  # Uptrend
            
        noise = np.random.normal(0, 0.5)
        close_price = base_price + trend + noise
        
        # OHLC from close
        open_price = close_price + np.random.normal(0, 0.2)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.3))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.3))
        
        # Volume with patterns
        base_volume = np.random.lognormal(10, 0.3)
        volume = base_volume * volume_multiplier
        
        prices.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        })
        volumes.append(volume)
        
        base_price = close_price

    sample_data = pd.DataFrame({
        'open': [p['open'] for p in prices],
        'high': [p['high'] for p in prices],
        'low': [p['low'] for p in prices],
        'close': [p['close'] for p in prices],
        'volume': volumes
    }, index=dates)

    symbol = "ADAUSDT"
    timeframe = "15m"

    # Analyze the symbol
    signals = strategy.analyze_symbol(symbol, sample_data, timeframe)
    print(f"📊 Found {len(signals)} VWMA signals")

    for i, signal in enumerate(signals, 1):
        print(f"\n🎯 VWMA Signal {i}:")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Entry: {signal.entry_price:.6f}")
        print(f"   Stop: {signal.stop_loss:.6f}")
        print(f"   Target: {signal.take_profit:.6f}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Volume Strength: {signal.volume_strength:.2f}x")
        print(f"   Position Size: {signal.position_size_multiplier:.2f}x")
        print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")

    # Get volume analysis
    vol_analysis = strategy.get_volume_analysis(symbol, timeframe)
    if vol_analysis:
        print(f"\n📊 Volume Analysis:")
        print(f"   Volume Strength: {vol_analysis['volume_strength']:.2f}x")
        print(f"   Volume Trend: {vol_analysis['volume_trend']}")
        print(f"   Volume Surge: {vol_analysis['volume_surge']}")
        print(f"   VWMA Direction: {vol_analysis['vwma_direction']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())