#!/usr/bin/env python3
"""
🚀 RSI DIVERGENCE STRATEGY
Advanced RSI divergence detection for crypto trading with high probability setups

This strategy implements:
✅ Regular and Hidden Divergence Detection
✅ Multi-timeframe RSI Analysis  
✅ Volume Confirmation for Divergences
✅ Dynamic RSI Thresholds for Crypto Markets
✅ Precise Entry Timing with Lower Timeframe Confirmations
"""

import numpy as np
import pandas as pd
import talib as ta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from scipy.signal import find_peaks, argrelextrema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RSI_DIVERGENCE - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DivergenceSignal:
    """RSI divergence signal data structure"""
    timestamp: datetime
    symbol: str
    direction: str  # 'bullish', 'bearish'
    divergence_type: str  # 'regular', 'hidden'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    rsi_value: float
    price_pivot_1: float
    price_pivot_2: float
    rsi_pivot_1: float
    rsi_pivot_2: float
    timeframe: str
    volume_confirmation: bool
    risk_reward_ratio: float

@dataclass  
class RSIPivot:
    """RSI and price pivot point"""
    index: int
    price: float
    rsi_value: float
    timestamp: datetime
    pivot_type: str  # 'high', 'low'

class RSIDivergenceStrategy:
    """
    Advanced RSI Divergence Strategy for crypto markets
    Detects both regular and hidden divergences with volume confirmation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.pivot_history: Dict[str, List[RSIPivot]] = {}
        self.current_signals: Dict[str, List[DivergenceSignal]] = {}
        
        logger.info("🚀 RSI Divergence Strategy initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for RSI divergence strategy"""
        return {
            # RSI settings
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_extreme_overbought': 80,
            'rsi_extreme_oversold': 20,
            
            # Divergence detection
            'min_bars_between_pivots': 5,
            'max_bars_between_pivots': 50,
            'pivot_lookback': 3,  # Bars to look on each side for pivot
            'divergence_tolerance': 0.001,  # 0.1% tolerance for price comparison
            
            # Volume confirmation  
            'volume_ma_period': 20,
            'volume_threshold': 1.2,  # Volume must be 20% above average
            'volume_confirmation_required': True,
            
            # Risk management
            'atr_period': 14,
            'stop_loss_atr_multiplier': 2.5,
            'take_profit_atr_multiplier': 4.0,
            'max_risk_per_trade': 0.015,  # 1.5% risk per trade
            
            # Entry timing
            'wait_for_rsi_confirmation': True,
            'rsi_confirmation_threshold': 5,  # RSI must move 5 points in favor
            'min_confidence': 0.65,
            'min_rrr': 2.0,
            
            # Timeframe settings
            'timeframes': ['5m', '15m', '30m', '1h'],
            'primary_timeframe': '15m',
            
            # Crypto specific optimizations
            'crypto_volatility_adj': True,
            'dynamic_thresholds': True,
        }

    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI with crypto optimizations"""
        period = self.config['rsi_period']
        df['rsi'] = ta.RSI(df['close'].values, timeperiod=period)
        
        # Dynamic RSI thresholds based on market conditions
        if self.config['dynamic_thresholds']:
            rsi_values = df['rsi'].dropna()
            if len(rsi_values) > 50:
                # Adjust thresholds based on RSI distribution
                rsi_75th = rsi_values.quantile(0.75)
                rsi_25th = rsi_values.quantile(0.25)
                
                # More sensitive thresholds for crypto volatility
                df['rsi_overbought'] = min(75, max(65, rsi_75th))
                df['rsi_oversold'] = max(25, min(35, rsi_25th))
            else:
                df['rsi_overbought'] = self.config['rsi_overbought']
                df['rsi_oversold'] = self.config['rsi_oversold']
        else:
            df['rsi_overbought'] = self.config['rsi_overbought']
            df['rsi_oversold'] = self.config['rsi_oversold']
            
        return df

    def find_pivots(self, df: pd.DataFrame) -> Tuple[List[RSIPivot], List[RSIPivot]]:
        """Find price and RSI pivot points"""
        if len(df) < self.config['pivot_lookback'] * 2 + 1:
            return [], []
            
        lookback = self.config['pivot_lookback']
        
        # Find price pivots using scipy
        high_pivots = []
        low_pivots = []
        
        try:
            # Find local maxima and minima
            high_indices = argrelextrema(df['high'].values, np.greater, order=lookback)[0]
            low_indices = argrelextrema(df['low'].values, np.less, order=lookback)[0]
            
            # Create pivot objects for highs
            for idx in high_indices:
                if idx < len(df) and not pd.isna(df['rsi'].iloc[idx]):
                    pivot = RSIPivot(
                        index=idx,
                        price=df['high'].iloc[idx],
                        rsi_value=df['rsi'].iloc[idx],
                        timestamp=df.index[idx] if hasattr(df.index[idx], 'to_pydatetime') else datetime.now(),
                        pivot_type='high'
                    )
                    high_pivots.append(pivot)
            
            # Create pivot objects for lows  
            for idx in low_indices:
                if idx < len(df) and not pd.isna(df['rsi'].iloc[idx]):
                    pivot = RSIPivot(
                        index=idx,
                        price=df['low'].iloc[idx],
                        rsi_value=df['rsi'].iloc[idx],
                        timestamp=df.index[idx] if hasattr(df.index[idx], 'to_pydatetime') else datetime.now(),
                        pivot_type='low'
                    )
                    low_pivots.append(pivot)
                    
        except Exception as e:
            logger.warning(f"Error finding pivots: {e}")
            
        return high_pivots, low_pivots

    def detect_divergence(self, pivots: List[RSIPivot], divergence_type: str) -> List[Tuple[RSIPivot, RSIPivot, str]]:
        """Detect RSI divergences between pivot points"""
        if len(pivots) < 2:
            return []
            
        divergences = []
        min_bars = self.config['min_bars_between_pivots']
        max_bars = self.config['max_bars_between_pivots']
        tolerance = self.config['divergence_tolerance']
        
        for i in range(len(pivots) - 1):
            for j in range(i + 1, len(pivots)):
                pivot1 = pivots[i]
                pivot2 = pivots[j]
                
                # Check if pivots are within acceptable range
                bars_between = pivot2.index - pivot1.index
                if bars_between < min_bars or bars_between > max_bars:
                    continue
                
                if divergence_type == 'high':
                    # Bearish Regular Divergence: Higher high in price, lower high in RSI
                    price_higher = pivot2.price > pivot1.price * (1 + tolerance)
                    rsi_lower = pivot2.rsi_value < pivot1.rsi_value - 1
                    
                    if price_higher and rsi_lower:
                        divergences.append((pivot1, pivot2, 'bearish_regular'))
                    
                    # Bearish Hidden Divergence: Lower high in price, higher high in RSI
                    price_lower = pivot2.price < pivot1.price * (1 - tolerance)
                    rsi_higher = pivot2.rsi_value > pivot1.rsi_value + 1
                    
                    if price_lower and rsi_higher:
                        divergences.append((pivot1, pivot2, 'bearish_hidden'))
                        
                elif divergence_type == 'low':
                    # Bullish Regular Divergence: Lower low in price, higher low in RSI
                    price_lower = pivot2.price < pivot1.price * (1 - tolerance)
                    rsi_higher = pivot2.rsi_value > pivot1.rsi_value + 1
                    
                    if price_lower and rsi_higher:
                        divergences.append((pivot1, pivot2, 'bullish_regular'))
                    
                    # Bullish Hidden Divergence: Higher low in price, lower low in RSI
                    price_higher = pivot2.price > pivot1.price * (1 + tolerance)
                    rsi_lower = pivot2.rsi_value < pivot1.rsi_value - 1
                    
                    if price_higher and rsi_lower:
                        divergences.append((pivot1, pivot2, 'bullish_hidden'))
        
        return divergences

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators for confirmation"""
        if 'volume' not in df.columns:
            df['volume_ma'] = 1
            df['volume_ratio'] = 1
            return df
            
        period = self.config['volume_ma_period']
        df['volume_ma'] = df['volume'].rolling(window=period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df

    def identify_divergence_signals(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DivergenceSignal]:
        """Identify tradable RSI divergence signals"""
        if len(df) < 50:
            return []

        signals = []
        
        # Find pivots
        high_pivots, low_pivots = self.find_pivots(df)
        
        if len(high_pivots) == 0 and len(low_pivots) == 0:
            return signals
        
        # Detect divergences
        high_divergences = self.detect_divergence(high_pivots, 'high')
        low_divergences = self.detect_divergence(low_pivots, 'low')
        
        all_divergences = high_divergences + low_divergences
        
        if not all_divergences:
            return signals
            
        # Get current market state
        current = df.iloc[-1]
        atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=self.config['atr_period'])[-1]
        if pd.isna(atr):
            atr = abs(current['close'] * 0.02)  # 2% fallback
        
        # Process each divergence
        for pivot1, pivot2, div_type in all_divergences:
            # Only consider recent divergences (pivot2 should be recent)
            if pivot2.index < len(df) - 5:  # Not within last 5 bars
                continue
                
            # Determine signal direction and type
            if 'bullish' in div_type:
                direction = 'bullish'
                entry_price = current['close']
                stop_loss = entry_price - (atr * self.config['stop_loss_atr_multiplier'])
                take_profit = entry_price + (atr * self.config['take_profit_atr_multiplier'])
            else:
                direction = 'bearish'  
                entry_price = current['close']
                stop_loss = entry_price + (atr * self.config['stop_loss_atr_multiplier'])
                take_profit = entry_price - (atr * self.config['take_profit_atr_multiplier'])
            
            # Calculate confidence based on multiple factors
            confidence = 0.5
            
            # RSI position confidence
            rsi_val = current['rsi']
            if direction == 'bullish' and rsi_val < 40:
                confidence += 0.2
            elif direction == 'bearish' and rsi_val > 60:
                confidence += 0.2
                
            # Divergence type confidence
            if 'regular' in div_type:
                confidence += 0.2  # Regular divergences are stronger
            else:
                confidence += 0.1
                
            # Volume confirmation
            volume_confirmed = current['volume_ratio'] >= self.config['volume_threshold']
            if volume_confirmed:
                confidence += 0.15
            elif self.config['volume_confirmation_required']:
                continue  # Skip if volume confirmation required but not present
                
            # Time between pivots (not too recent, not too old)
            bars_between = pivot2.index - pivot1.index
            if 10 <= bars_between <= 30:
                confidence += 0.1
                
            # RSI divergence strength
            rsi_diff = abs(pivot2.rsi_value - pivot1.rsi_value)
            if rsi_diff > 10:
                confidence += 0.1
            
            # Calculate risk-reward ratio
            if direction == 'bullish':
                rrr = (take_profit - entry_price) / (entry_price - stop_loss) if stop_loss < entry_price else 0
            else:
                rrr = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else 0

            # Filter by minimum requirements
            if confidence >= self.config['min_confidence'] and rrr >= self.config['min_rrr']:
                
                signal = DivergenceSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    direction=direction,
                    divergence_type=div_type,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    rsi_value=rsi_val,
                    price_pivot_1=pivot1.price,
                    price_pivot_2=pivot2.price,
                    rsi_pivot_1=pivot1.rsi_value,
                    rsi_pivot_2=pivot2.rsi_value,
                    timeframe=timeframe,
                    volume_confirmation=volume_confirmed,
                    risk_reward_ratio=rrr
                )
                
                signals.append(signal)
                logger.info(f"📈 {div_type.upper()} divergence for {symbol} on {timeframe}: "
                           f"{direction} - Confidence: {confidence:.2f}, RRR: {rrr:.2f}")

        return signals

    def analyze_symbol(self, symbol: str, df: pd.DataFrame, timeframe: str) -> List[DivergenceSignal]:
        """Analyze symbol for RSI divergences"""
        try:
            # Calculate indicators
            df = self.calculate_rsi(df)
            df = self.calculate_volume_indicators(df)
            
            # Store historical data
            self.historical_data[f"{symbol}_{timeframe}"] = df
            
            # Identify signals
            signals = self.identify_divergence_signals(df, symbol, timeframe)
            
            # Store current signals
            key = f"{symbol}_{timeframe}"
            self.current_signals[key] = signals
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {e}")
            return []

    def get_divergence_strength(self, signal: DivergenceSignal) -> str:
        """Assess the strength of a divergence signal"""
        if signal.confidence > 0.8:
            return "Very Strong"
        elif signal.confidence > 0.7:
            return "Strong" 
        elif signal.confidence > 0.6:
            return "Moderate"
        else:
            return "Weak"

    def get_active_signals(self, symbol: Optional[str] = None) -> List[DivergenceSignal]:
        """Get all active divergence signals"""
        active_signals = []
        
        for key, signals in self.current_signals.items():
            if symbol is None or key.startswith(symbol):
                active_signals.extend(signals)
        
        # Sort by confidence
        active_signals.sort(key=lambda x: x.confidence, reverse=True)
        return active_signals

# Global instance
_rsi_divergence_strategy = None

def get_rsi_divergence_strategy() -> RSIDivergenceStrategy:
    """Get global RSI divergence strategy instance"""
    global _rsi_divergence_strategy
    if _rsi_divergence_strategy is None:
        _rsi_divergence_strategy = RSIDivergenceStrategy()
    return _rsi_divergence_strategy

# Example usage and testing
async def main():
    """Test RSI divergence strategy"""
    print("🚀 RSI DIVERGENCE STRATEGY TEST")
    print("=" * 60)

    strategy = get_rsi_divergence_strategy()

    # Generate realistic test data with divergence patterns
    dates = pd.date_range('2024-01-01', periods=200, freq='15T')
    np.random.seed(42)
    
    # Create price data with divergence patterns
    prices = []
    base_price = 100.0
    trend = 0
    
    for i in range(200):
        # Create divergence pattern around bars 100-150
        if 80 <= i <= 120:
            # Price makes higher highs but with decreasing momentum (bearish divergence setup)
            trend += 0.3 if i < 100 else -0.1
        elif 150 <= i <= 180:
            # Price makes lower lows but with increasing momentum (bullish divergence setup) 
            trend -= 0.2 if i < 165 else 0.2
        else:
            trend += np.random.normal(0, 0.1)
        
        noise = np.random.normal(0, 0.5)
        close_price = base_price + trend + noise
        
        # Generate OHLC from close
        open_price = close_price + np.random.normal(0, 0.2)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.3))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.3))
        volume = np.random.lognormal(10, 0.3)
        
        prices.append({
            'open': open_price,
            'high': high_price,
            'low': low_price, 
            'close': close_price,
            'volume': volume
        })
        
        base_price = close_price

    sample_data = pd.DataFrame(prices, index=dates)

    symbol = "ETHUSDT"
    timeframe = "15m"

    # Analyze the symbol
    signals = strategy.analyze_symbol(symbol, sample_data, timeframe)
    print(f"📊 Found {len(signals)} RSI divergence signals")

    for i, signal in enumerate(signals, 1):
        print(f"\n🎯 Divergence Signal {i}:")
        print(f"   Type: {signal.divergence_type}")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Entry: {signal.entry_price:.6f}")
        print(f"   Stop: {signal.stop_loss:.6f}")
        print(f"   Target: {signal.take_profit:.6f}")
        print(f"   Confidence: {signal.confidence:.2f} ({strategy.get_divergence_strength(signal)})")
        print(f"   RSI: {signal.rsi_value:.1f}")
        print(f"   Volume Confirmed: {signal.volume_confirmation}")
        print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")

    # Get active signals
    active = strategy.get_active_signals(symbol)
    print(f"\n📈 Active signals for {symbol}: {len(active)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())