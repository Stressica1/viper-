#!/usr/bin/env python3
"""
🚀 BOLLINGER BAND MEAN REVERSION STRATEGY
Proven mean reversion strategy using dynamic Bollinger Bands for crypto trading

This strategy implements:
✅ Dynamic Bollinger Band calculations with configurable periods
✅ Mean reversion signals with volume confirmation
✅ Risk management with ATR-based stops
✅ Multi-timeframe support (1m, 5m, 15m, 30m)
✅ Crypto market optimization with volatility adjustment
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
    format='%(asctime)s - BOLLINGER_MEAN_REVERSION - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BollingerSignal:
    """Bollinger Band mean reversion signal"""
    timestamp: datetime
    symbol: str
    direction: str  # 'long', 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    bb_position: float  # How far from mean (0=center, 1=upper, -1=lower)
    volume_strength: float
    timeframe: str
    risk_reward_ratio: float

@dataclass
class BollingerBands:
    """Bollinger Band calculations"""
    upper: float
    middle: float
    lower: float
    width: float
    position: float  # Current price position within bands
    squeeze: bool  # Is the band width contracting?

class BollingerMeanReversionStrategy:
    """
    Advanced Bollinger Band Mean Reversion Strategy
    Optimized for crypto markets with lower timeframe focus
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_signals: Dict[str, List[BollingerSignal]] = {}
        
        logger.info("🚀 Bollinger Mean Reversion Strategy initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration optimized for crypto"""
        return {
            # Bollinger Band settings
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'bb_squeeze_threshold': 0.1,  # Band width threshold for squeeze detection
            
            # Mean reversion thresholds
            'entry_threshold_upper': 0.8,  # Enter short when price is 80% to upper band
            'entry_threshold_lower': 0.2,  # Enter long when price is 20% to lower band
            'mean_return_threshold': 0.1,  # Exit when returning 10% to mean
            
            # Volume confirmation
            'volume_ma_period': 14,
            'volume_multiplier': 1.2,  # Volume must be 20% above average
            
            # Risk management
            'atr_period': 14,
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_atr_multiplier': 3.0,
            'max_risk_per_trade': 0.02,  # 2% risk per trade
            
            # Timeframe specific settings
            'timeframes': ['1m', '5m', '15m', '30m'],
            'min_confidence': 0.6,
            'min_rrr': 1.5,  # Minimum risk-reward ratio
            
            # Crypto specific
            'crypto_volatility_adjustment': True,
            'dynamic_std_dev': True,  # Adjust std dev based on volatility
        }

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands with crypto optimizations"""
        if len(df) < self.config['bb_period']:
            return df

        # Standard Bollinger Bands
        period = self.config['bb_period']
        std_dev = self.config['bb_std_dev']
        
        # Dynamic standard deviation for crypto volatility
        if self.config['dynamic_std_dev']:
            # Adjust std dev based on recent volatility
            volatility = df['close'].rolling(window=period).std()
            current_vol = volatility.iloc[-1] if len(volatility) > 0 else std_dev
            market_vol = volatility.rolling(window=period*2).mean().iloc[-1] if len(volatility) > period*2 else std_dev
            
            if current_vol > market_vol * 1.5:
                std_dev *= 1.2  # Increase bands during high volatility
            elif current_vol < market_vol * 0.7:
                std_dev *= 0.8  # Decrease bands during low volatility

        # Calculate bands using TA-Lib
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            df['close'].values, 
            timeperiod=period, 
            nbdevup=std_dev, 
            nbdevdn=std_dev
        )
        
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Bollinger Band squeeze detection
        bb_width_ma = df['bb_width'].rolling(window=20).mean()
        df['bb_squeeze'] = df['bb_width'] < bb_width_ma * self.config['bb_squeeze_threshold']
        
        return df

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        if 'volume' not in df.columns:
            logger.warning("No volume data available")
            df['volume_ma'] = 0
            df['volume_ratio'] = 1
            return df
            
        period = self.config['volume_ma_period']
        df['volume_ma'] = df['volume'].rolling(window=period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df

    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range for risk management"""
        atr_period = self.config['atr_period']
        df['atr'] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=atr_period)
        return df

    def identify_mean_reversion_signals(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[BollingerSignal]:
        """Identify mean reversion trading opportunities"""
        if len(df) < 50:  # Need sufficient data
            return []

        signals = []
        
        # Get the latest data point
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        # Check for valid Bollinger Band data
        if pd.isna(current['bb_upper']) or pd.isna(current['bb_position']):
            return signals

        # Volume confirmation
        volume_confirmed = current['volume_ratio'] >= self.config['volume_multiplier']
        if not volume_confirmed:
            return signals

        # Mean reversion logic
        bb_position = current['bb_position']
        confidence = 0.5
        
        # LONG signal: Price near lower band, expect bounce to mean
        if (bb_position <= self.config['entry_threshold_lower'] and 
            previous['bb_position'] > bb_position):  # Price moving deeper into oversold
            
            direction = 'long'
            entry_price = current['close']
            atr = current['atr'] if not pd.isna(current['atr']) else abs(entry_price * 0.02)
            
            stop_loss = entry_price - (atr * self.config['stop_loss_atr_multiplier'])
            take_profit = entry_price + (atr * self.config['take_profit_atr_multiplier'])
            
            # Adjust take profit to middle band if closer
            if current['bb_middle'] > entry_price:
                take_profit = min(take_profit, current['bb_middle'])
            
            confidence = 0.6 + (0.4 * (self.config['entry_threshold_lower'] - bb_position))
            
        # SHORT signal: Price near upper band, expect drop to mean  
        elif (bb_position >= self.config['entry_threshold_upper'] and 
              previous['bb_position'] < bb_position):  # Price moving deeper into overbought
            
            direction = 'short'
            entry_price = current['close']
            atr = current['atr'] if not pd.isna(current['atr']) else abs(entry_price * 0.02)
            
            stop_loss = entry_price + (atr * self.config['stop_loss_atr_multiplier'])
            take_profit = entry_price - (atr * self.config['take_profit_atr_multiplier'])
            
            # Adjust take profit to middle band if closer
            if current['bb_middle'] < entry_price:
                take_profit = max(take_profit, current['bb_middle'])
                
            confidence = 0.6 + (0.4 * (bb_position - self.config['entry_threshold_upper']))
        
        else:
            return signals

        # Calculate risk-reward ratio
        if direction == 'long':
            rrr = (take_profit - entry_price) / (entry_price - stop_loss) if stop_loss < entry_price else 0
        else:
            rrr = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else 0

        # Filter by minimum requirements
        if confidence >= self.config['min_confidence'] and rrr >= self.config['min_rrr']:
            
            signal = BollingerSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                bb_position=bb_position,
                volume_strength=current['volume_ratio'],
                timeframe=timeframe,
                risk_reward_ratio=rrr
            )
            
            signals.append(signal)
            logger.info(f"📈 {direction.upper()} signal for {symbol} on {timeframe}: "
                       f"Entry: {entry_price:.6f}, Confidence: {confidence:.2f}, RRR: {rrr:.2f}")

        return signals

    def analyze_symbol(self, symbol: str, df: pd.DataFrame, timeframe: str) -> List[BollingerSignal]:
        """Analyze a symbol for mean reversion opportunities"""
        try:
            # Calculate all indicators
            df = self.calculate_bollinger_bands(df)
            df = self.calculate_volume_indicators(df)  
            df = self.calculate_atr(df)
            
            # Store historical data
            self.historical_data[f"{symbol}_{timeframe}"] = df
            
            # Identify signals
            signals = self.identify_mean_reversion_signals(df, symbol, timeframe)
            
            # Store current signals
            key = f"{symbol}_{timeframe}"
            self.current_signals[key] = signals
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {e}")
            return []

    def get_market_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market analysis for a symbol"""
        analysis = {
            'symbol': symbol,
            'overall_bias': 'neutral',
            'confidence': 0.0,
            'timeframe_signals': {},
            'risk_assessment': 'medium'
        }
        
        total_signals = 0
        total_confidence = 0.0
        long_signals = 0
        short_signals = 0
        
        for timeframe in self.config['timeframes']:
            key = f"{symbol}_{timeframe}"
            if key in self.current_signals:
                signals = self.current_signals[key]
                analysis['timeframe_signals'][timeframe] = len(signals)
                
                for signal in signals:
                    total_signals += 1
                    total_confidence += signal.confidence
                    if signal.direction == 'long':
                        long_signals += 1
                    else:
                        short_signals += 1
        
        if total_signals > 0:
            analysis['confidence'] = total_confidence / total_signals
            
            if long_signals > short_signals:
                analysis['overall_bias'] = 'bullish'
            elif short_signals > long_signals:
                analysis['overall_bias'] = 'bearish'
            
            # Risk assessment based on signal strength
            if analysis['confidence'] > 0.8:
                analysis['risk_assessment'] = 'low'
            elif analysis['confidence'] < 0.6:
                analysis['risk_assessment'] = 'high'
        
        return analysis

    def get_active_signals(self, symbol: Optional[str] = None) -> List[BollingerSignal]:
        """Get all active signals, optionally filtered by symbol"""
        active_signals = []
        
        for key, signals in self.current_signals.items():
            if symbol is None or key.startswith(symbol):
                active_signals.extend(signals)
        
        # Sort by confidence
        active_signals.sort(key=lambda x: x.confidence, reverse=True)
        return active_signals

# Global instance
_bollinger_strategy = None

def get_bollinger_strategy() -> BollingerMeanReversionStrategy:
    """Get global Bollinger mean reversion strategy instance"""
    global _bollinger_strategy
    if _bollinger_strategy is None:
        _bollinger_strategy = BollingerMeanReversionStrategy()
    return _bollinger_strategy

# Example usage and testing
async def main():
    """Test Bollinger mean reversion strategy"""
    print("🚀 BOLLINGER MEAN REVERSION STRATEGY TEST")
    print("=" * 60)

    strategy = get_bollinger_strategy()

    # Example market data (replace with real data)
    dates = pd.date_range('2024-01-01', periods=100, freq='5T')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    base_price = 100.0
    prices = []
    volumes = []
    
    for i in range(100):
        # Add some mean reversion behavior
        if i > 0:
            prev_price = prices[-1]['close']
            mean_price = base_price + np.sin(i/10) * 5  # Oscillating around base
            drift = (mean_price - prev_price) * 0.1  # Mean reversion force
            change = np.random.normal(drift, 1.0)  # Random walk with mean reversion
        else:
            change = 0
            
        open_price = base_price + change if i == 0 else prices[-1]['close']
        change2 = np.random.normal(0, 0.5)
        close_price = open_price + change2
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.3))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.3))
        volume = np.random.lognormal(10, 0.5)
        
        prices.append({
            'open': open_price,
            'high': high_price, 
            'low': low_price,
            'close': close_price
        })
        volumes.append(volume)
        
        base_price = close_price

    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': [p['open'] for p in prices],
        'high': [p['high'] for p in prices],
        'low': [p['low'] for p in prices],
        'close': [p['close'] for p in prices],
        'volume': volumes
    })

    symbol = "BTCUSDT"
    timeframe = "5m"

    # Analyze the symbol
    signals = strategy.analyze_symbol(symbol, sample_data, timeframe)
    print(f"📊 Found {len(signals)} mean reversion signals")

    for i, signal in enumerate(signals[:3], 1):  # Show first 3 signals
        print(f"\n🎯 Signal {i}:")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Entry: {signal.entry_price:.6f}")
        print(f"   Stop: {signal.stop_loss:.6f}")  
        print(f"   Target: {signal.take_profit:.6f}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")

    # Get market analysis
    analysis = strategy.get_market_analysis(symbol)
    print(f"\n📈 Market Analysis:")
    print(f"   Overall Bias: {analysis['overall_bias']}")
    print(f"   Confidence: {analysis['confidence']:.2f}")
    print(f"   Risk Assessment: {analysis['risk_assessment']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())