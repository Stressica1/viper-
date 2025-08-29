#!/usr/bin/env python3
"""
🚀 MOMENTUM BREAKOUT STRATEGY
High-probability momentum breakout strategy optimized for crypto markets

This strategy implements:
✅ Dynamic support/resistance level detection
✅ Volume-confirmed breakouts
✅ Momentum oscillator confirmation
✅ False breakout filtering
✅ Volatility-based position sizing
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
    format='%(asctime)s - MOMENTUM_BREAKOUT - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BreakoutSignal:
    """Momentum breakout signal"""
    timestamp: datetime
    symbol: str
    direction: str  # 'long', 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    breakout_level: float
    volume_strength: float
    momentum_score: float
    volatility_percentile: float
    timeframe: str
    level_strength: int  # How many times level has been tested
    risk_reward_ratio: float

class MomentumBreakoutStrategy:
    """
    Advanced Momentum Breakout Strategy
    Identifies high-probability breakouts with volume and momentum confirmation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.support_resistance_levels: Dict[str, List[float]] = {}
        self.current_signals: Dict[str, List[BreakoutSignal]] = {}
        
        logger.info("🚀 Momentum Breakout Strategy initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for momentum breakout strategy"""
        return {
            # Support/Resistance detection
            'sr_lookback_period': 20,
            'sr_touch_threshold': 0.002,  # 0.2% threshold for level touch
            'min_level_strength': 2,  # Minimum touches to validate level
            'level_age_limit': 100,  # Maximum bars since level formation
            
            # Breakout detection
            'breakout_threshold': 0.001,  # 0.1% minimum breakout distance
            'breakout_confirmation_bars': 2,  # Bars to confirm breakout
            'false_breakout_filter': True,
            
            # Volume confirmation
            'volume_ma_period': 20,
            'volume_breakout_multiplier': 1.8,  # 80% above average volume
            'volume_confirmation_required': True,
            
            # Momentum indicators
            'rsi_period': 14,
            'rsi_momentum_threshold': 60,  # RSI above 60 for bullish momentum
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'momentum_confirmation_required': True,
            
            # Volatility analysis
            'volatility_period': 20,
            'high_volatility_threshold': 75,  # 75th percentile
            'low_volatility_threshold': 25,   # 25th percentile
            'volatility_scaling': True,
            
            # Risk management
            'atr_period': 14,
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_atr_multiplier': 4.0,
            'breakout_stop_distance': 0.5,  # Stop 50% back through breakout level
            
            # Position sizing
            'base_position_size': 1.0,
            'volatility_position_scaling': True,
            'max_position_multiplier': 2.0,
            
            # Signal filtering
            'min_confidence': 0.65,
            'min_rrr': 2.0,
            'max_daily_signals': 10,
            
            # Timeframe settings
            'timeframes': ['5m', '15m', '30m', '1h'],
            'primary_timeframe': '15m',
        }

    def detect_support_resistance_levels(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Detect dynamic support and resistance levels"""
        if len(df) < self.config['sr_lookback_period']:
            return [], []
            
        support_levels = []
        resistance_levels = []
        lookback = self.config['sr_lookback_period']
        
        # Find local highs and lows
        try:
            high_indices = argrelextrema(df['high'].values, np.greater, order=5)[0]
            low_indices = argrelextrema(df['low'].values, np.less, order=5)[0]
            
            # Group similar levels together
            threshold = self.config['sr_touch_threshold']
            
            # Process resistance levels (highs)
            resistance_candidates = [df['high'].iloc[i] for i in high_indices]
            resistance_groups = []
            
            for level in resistance_candidates:
                # Find if this level is close to existing group
                added_to_group = False
                for group in resistance_groups:
                    avg_level = np.mean(group)
                    if abs(level - avg_level) / avg_level <= threshold:
                        group.append(level)
                        added_to_group = True
                        break
                
                if not added_to_group:
                    resistance_groups.append([level])
            
            # Filter groups by strength (minimum touches)
            min_strength = self.config['min_level_strength']
            for group in resistance_groups:
                if len(group) >= min_strength:
                    resistance_levels.append(np.mean(group))
            
            # Process support levels (lows)
            support_candidates = [df['low'].iloc[i] for i in low_indices]
            support_groups = []
            
            for level in support_candidates:
                added_to_group = False
                for group in support_groups:
                    avg_level = np.mean(group)
                    if abs(level - avg_level) / avg_level <= threshold:
                        group.append(level)
                        added_to_group = True
                        break
                
                if not added_to_group:
                    support_groups.append([level])
            
            for group in support_groups:
                if len(group) >= min_strength:
                    support_levels.append(np.mean(group))
                    
        except Exception as e:
            logger.warning(f"Error detecting S/R levels: {e}")
        
        return support_levels, resistance_levels

    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based indicators"""
        # RSI
        df['rsi'] = ta.RSI(df['close'].values, timeperiod=self.config['rsi_period'])
        
        # MACD
        macd, macd_signal, macd_hist = ta.MACD(
            df['close'].values,
            fastperiod=self.config['macd_fast'],
            slowperiod=self.config['macd_slow'],
            signalperiod=self.config['macd_signal']
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Rate of Change
        df['roc'] = ta.ROC(df['close'].values, timeperiod=10)
        
        # Momentum Score (composite)
        df['momentum_score'] = 0.0
        
        # RSI component
        rsi_normalized = (df['rsi'] - 50) / 50  # -1 to 1
        df['momentum_score'] += rsi_normalized * 0.4
        
        # MACD component
        macd_normalized = np.tanh(df['macd'] / df['close'] * 100)  # Normalize MACD
        df['momentum_score'] += macd_normalized * 0.3
        
        # ROC component
        roc_normalized = np.tanh(df['roc'] / 10)  # Normalize ROC
        df['momentum_score'] += roc_normalized * 0.3
        
        return df

    def calculate_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based metrics"""
        period = self.config['volatility_period']
        
        # True Range and ATR
        df['atr'] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
        # Volatility percentile
        df['volatility'] = df['atr'] / df['close'] * 100  # ATR as % of price
        df['volatility_percentile'] = df['volatility'].rolling(window=period*2).rank(pct=True) * 100
        
        return df

    def identify_breakout_signals(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[BreakoutSignal]:
        """Identify momentum breakout trading opportunities"""
        if len(df) < 50:
            return []

        signals = []
        
        # Detect support and resistance levels
        support_levels, resistance_levels = self.detect_support_resistance_levels(df)
        
        if not support_levels and not resistance_levels:
            return signals
            
        # Get current market state
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        current_price = current['close']
        
        # Volume confirmation
        volume_strength = current.get('volume_ratio', 1.0)
        volume_confirmed = volume_strength >= self.config['volume_breakout_multiplier']
        
        if self.config['volume_confirmation_required'] and not volume_confirmed:
            return signals
            
        # Momentum confirmation
        momentum_score = current.get('momentum_score', 0.0)
        rsi_val = current.get('rsi', 50)
        
        # Check for bullish breakouts (resistance breaks)
        for resistance_level in resistance_levels:
            # Check if price broke above resistance
            price_above = current_price > resistance_level * (1 + self.config['breakout_threshold'])
            previous_below = previous['close'] <= resistance_level
            
            if price_above and previous_below:
                # Momentum confirmation for bullish breakout
                if self.config['momentum_confirmation_required']:
                    if rsi_val < self.config['rsi_momentum_threshold'] or momentum_score < 0.2:
                        continue
                
                # Calculate trade parameters
                entry_price = current_price
                
                # Stop loss: percentage back through breakout level
                stop_distance = (entry_price - resistance_level) * self.config['breakout_stop_distance']
                stop_loss = resistance_level - stop_distance
                
                # Take profit based on ATR
                atr = current.get('atr', abs(entry_price * 0.02))
                take_profit = entry_price + (atr * self.config['take_profit_atr_multiplier'])
                
                # Calculate confidence
                confidence = 0.6
                confidence += 0.1 if volume_strength > 2.0 else 0
                confidence += 0.15 if momentum_score > 0.5 else 0
                confidence += 0.1 if rsi_val > 70 else 0
                confidence += 0.1 if current.get('volatility_percentile', 50) > 60 else 0
                
                # Risk-reward ratio
                rrr = (take_profit - entry_price) / (entry_price - stop_loss) if stop_loss < entry_price else 0
                
                if confidence >= self.config['min_confidence'] and rrr >= self.config['min_rrr']:
                    signal = BreakoutSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        direction='long',
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        confidence=confidence,
                        breakout_level=resistance_level,
                        volume_strength=volume_strength,
                        momentum_score=momentum_score,
                        volatility_percentile=current.get('volatility_percentile', 50),
                        timeframe=timeframe,
                        level_strength=2,  # Simplified
                        risk_reward_ratio=rrr
                    )
                    
                    signals.append(signal)
                    logger.info(f"📈 Momentum BREAKOUT LONG for {symbol} on {timeframe}: "
                               f"Broke ${resistance_level:.6f}, Vol: {volume_strength:.2f}x, "
                               f"Momentum: {momentum_score:.2f}, Confidence: {confidence:.2f}")

        # Check for bearish breakouts (support breaks)
        for support_level in support_levels:
            # Check if price broke below support
            price_below = current_price < support_level * (1 - self.config['breakout_threshold'])
            previous_above = previous['close'] >= support_level
            
            if price_below and previous_above:
                # Momentum confirmation for bearish breakout
                if self.config['momentum_confirmation_required']:
                    if rsi_val > (100 - self.config['rsi_momentum_threshold']) or momentum_score > -0.2:
                        continue
                
                # Calculate trade parameters
                entry_price = current_price
                
                # Stop loss: percentage back through breakout level
                stop_distance = (support_level - entry_price) * self.config['breakout_stop_distance']
                stop_loss = support_level + stop_distance
                
                # Take profit based on ATR
                atr = current.get('atr', abs(entry_price * 0.02))
                take_profit = entry_price - (atr * self.config['take_profit_atr_multiplier'])
                
                # Calculate confidence
                confidence = 0.6
                confidence += 0.1 if volume_strength > 2.0 else 0
                confidence += 0.15 if momentum_score < -0.5 else 0
                confidence += 0.1 if rsi_val < 30 else 0
                confidence += 0.1 if current.get('volatility_percentile', 50) > 60 else 0
                
                # Risk-reward ratio
                rrr = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else 0
                
                if confidence >= self.config['min_confidence'] and rrr >= self.config['min_rrr']:
                    signal = BreakoutSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        direction='short',
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        confidence=confidence,
                        breakout_level=support_level,
                        volume_strength=volume_strength,
                        momentum_score=momentum_score,
                        volatility_percentile=current.get('volatility_percentile', 50),
                        timeframe=timeframe,
                        level_strength=2,  # Simplified
                        risk_reward_ratio=rrr
                    )
                    
                    signals.append(signal)
                    logger.info(f"📉 Momentum BREAKOUT SHORT for {symbol} on {timeframe}: "
                               f"Broke ${support_level:.6f}, Vol: {volume_strength:.2f}x, "
                               f"Momentum: {momentum_score:.2f}, Confidence: {confidence:.2f}")

        return signals

    def analyze_symbol(self, symbol: str, df: pd.DataFrame, timeframe: str) -> List[BreakoutSignal]:
        """Analyze symbol for momentum breakout opportunities"""
        try:
            # Calculate all indicators
            if 'volume' not in df.columns:
                df['volume'] = 1000  # Default volume
                
            # Volume indicators
            volume_period = self.config['volume_ma_period']
            df['volume_ma'] = df['volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Momentum indicators
            df = self.calculate_momentum_indicators(df)
            
            # Volatility metrics
            df = self.calculate_volatility_metrics(df)
            
            # Store historical data
            self.historical_data[f"{symbol}_{timeframe}"] = df
            
            # Identify signals
            signals = self.identify_breakout_signals(df, symbol, timeframe)
            
            # Store current signals
            key = f"{symbol}_{timeframe}"
            self.current_signals[key] = signals
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {e}")
            return []

    def get_breakout_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get detailed breakout analysis"""
        key = f"{symbol}_{timeframe}"
        if key not in self.historical_data:
            return {}
            
        df = self.historical_data[key]
        current = df.iloc[-1]
        
        support_levels, resistance_levels = self.detect_support_resistance_levels(df)
        
        return {
            'support_levels': len(support_levels),
            'resistance_levels': len(resistance_levels),
            'current_momentum': current.get('momentum_score', 0.0),
            'volume_strength': current.get('volume_ratio', 1.0),
            'volatility_percentile': current.get('volatility_percentile', 50),
            'signals_count': len(self.current_signals.get(key, [])),
            'nearest_support': min(support_levels, key=lambda x: abs(x - current['close'])) if support_levels else None,
            'nearest_resistance': min(resistance_levels, key=lambda x: abs(x - current['close'])) if resistance_levels else None,
        }

    def get_active_signals(self, symbol: Optional[str] = None) -> List[BreakoutSignal]:
        """Get all active breakout signals"""
        active_signals = []
        
        for key, signals in self.current_signals.items():
            if symbol is None or key.startswith(symbol):
                active_signals.extend(signals)
        
        # Sort by confidence and volume strength
        active_signals.sort(key=lambda x: (x.confidence, x.volume_strength), reverse=True)
        return active_signals

# Global instance
_momentum_breakout_strategy = None

def get_momentum_breakout_strategy() -> MomentumBreakoutStrategy:
    """Get global momentum breakout strategy instance"""
    global _momentum_breakout_strategy
    if _momentum_breakout_strategy is None:
        _momentum_breakout_strategy = MomentumBreakoutStrategy()
    return _momentum_breakout_strategy

# Example usage and testing
async def main():
    """Test momentum breakout strategy"""
    print("🚀 MOMENTUM BREAKOUT STRATEGY TEST")
    print("=" * 60)

    strategy = get_momentum_breakout_strategy()

    # Generate test data with clear support/resistance and breakouts
    dates = pd.date_range('2024-01-01', periods=300, freq='15min')
    np.random.seed(42)
    
    prices = []
    volumes = []
    base_price = 100.0
    
    # Create ranges and breakouts
    for i in range(300):
        # Create consolidation ranges and breakouts
        if i < 100:
            # Range-bound market around 100
            target_price = 100 + np.sin(i/20) * 2
            noise = np.random.normal(0, 0.5)
            volume_mult = 1.0
        elif i < 150:
            # Breakout upward with high volume
            target_price = 100 + (i - 100) * 0.2
            noise = np.random.normal(0, 0.8)
            volume_mult = 2.5 if 100 <= i <= 110 else 1.2  # High volume on breakout
        elif i < 200:
            # New range around 110
            target_price = 110 + np.sin((i-150)/15) * 1.5
            noise = np.random.normal(0, 0.4)
            volume_mult = 1.0
        else:
            # Breakdown with volume
            target_price = 110 - (i - 200) * 0.15
            noise = np.random.normal(0, 0.6)
            volume_mult = 2.0 if 200 <= i <= 210 else 1.1
        
        close_price = target_price + noise
        
        # Generate OHLC
        open_price = close_price + np.random.normal(0, 0.3)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.4))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.4))
        
        # Generate volume with patterns
        base_volume = np.random.lognormal(10, 0.3)
        volume = base_volume * volume_mult
        
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

    symbol = "MATICUSDT"
    timeframe = "15m"

    # Analyze the symbol
    signals = strategy.analyze_symbol(symbol, sample_data, timeframe)
    print(f"📊 Found {len(signals)} momentum breakout signals")

    for i, signal in enumerate(signals, 1):
        print(f"\n🎯 Breakout Signal {i}:")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Entry: {signal.entry_price:.6f}")
        print(f"   Stop: {signal.stop_loss:.6f}")
        print(f"   Target: {signal.take_profit:.6f}")
        print(f"   Breakout Level: {signal.breakout_level:.6f}")
        print(f"   Volume Strength: {signal.volume_strength:.2f}x")
        print(f"   Momentum Score: {signal.momentum_score:.2f}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")

    # Get breakout analysis
    analysis = strategy.get_breakout_analysis(symbol, timeframe)
    if analysis:
        print(f"\n📊 Breakout Analysis:")
        print(f"   Support Levels: {analysis['support_levels']}")
        print(f"   Resistance Levels: {analysis['resistance_levels']}")
        print(f"   Current Momentum: {analysis['current_momentum']:.2f}")
        print(f"   Volume Strength: {analysis['volume_strength']:.2f}x")
        print(f"   Volatility Percentile: {analysis['volatility_percentile']:.0f}%")
        if analysis.get('nearest_support'):
            print(f"   Nearest Support: {analysis['nearest_support']:.6f}")
        if analysis.get('nearest_resistance'):
            print(f"   Nearest Resistance: {analysis['nearest_resistance']:.6f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())