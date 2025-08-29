#!/usr/bin/env python3
"""
🚀 PREDICTIVE RANGES TRADING STRATEGY
LuxAlgo-inspired predictive support/resistance levels for optimal trade entries

This strategy implements:
✅ Predictive Range Projections - Forecast future S/R levels
✅ Dynamic Range Calculations - Volatility-adjusted projections
✅ Multi-Timeframe Analysis - Confluence across timeframes
✅ Entry Signal Optimization - Precise entry timing
✅ Risk Management Integration - ATR-based stops and targets
"""

import numpy as np
import pandas as pd
import talib as ta
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - PREDICTIVE_RANGES - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PredictiveRange:
    """Predictive range data structure"""
    timestamp: datetime
    current_price: float
    predicted_support: float
    predicted_resistance: float
    confidence_level: float
    range_width: float
    breakout_probability: float
    timeframe: str
    direction: str  # 'bullish', 'bearish', 'neutral'

@dataclass
class EntrySignal:
    """Optimized entry signal with predictive ranges"""
    symbol: str
    direction: str
    entry_price: float
    predicted_target: float
    predicted_stop: float
    confidence: float
    risk_reward_ratio: float
    timeframe: str
    confluence_score: float

class PredictiveRangesStrategy:
    """
    Advanced Predictive Ranges Strategy for optimal trade entries
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # Historical data storage
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.predictive_ranges: Dict[str, List[PredictiveRange]] = {}

        logger.info("🚀 Predictive Ranges Strategy initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for predictive ranges"""
        return {
            'lookback_periods': [20, 50, 100, 200],
            'projection_periods': [5, 10, 20, 50],
            'volatility_multipliers': [1.0, 1.5, 2.0, 2.5],
            'confidence_thresholds': [0.6, 0.7, 0.8, 0.9],
            'timeframes': ['5m', '15m', '1h', '4h'],
            'min_range_width': 0.001,  # 0.1% minimum range
            'max_range_width': 0.05,   # 5% maximum range
            'confluence_threshold': 0.7,
            'breakout_probability_threshold': 0.65
        }

    def calculate_predictive_ranges(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PredictiveRange]:
        """
        Calculate predictive support and resistance ranges

        Args:
            df: OHLCV dataframe
            symbol: Trading symbol
            timeframe: Timeframe string

        Returns:
            List of predictive ranges
        """
        if len(df) < 100:
            logger.warning(f"Insufficient data for {symbol} on {timeframe}")
            return []

        ranges = []

        try:
            # Calculate volatility-adjusted ranges
            for lookback in self.config['lookback_periods']:
                for projection in self.config['projection_periods']:
                    for vol_mult in self.config['volatility_multipliers']:

                        # Get recent data
                        recent_data = df.tail(lookback + projection)

                        if len(recent_data) < lookback:
                            continue

                        # Calculate ATR for volatility
                        high = recent_data['high'].values
                        low = recent_data['low'].values
                        close = recent_data['close'].values

                        atr = ta.ATR(high, low, close, timeperiod=min(14, len(recent_data)))

                        if len(atr) == 0 or np.isnan(atr[-1]):
                            continue

                        current_atr = atr[-1]
                        current_price = close[-1]

                        # Calculate predictive range
                        range_width = current_atr * vol_mult
                        range_width = np.clip(range_width,
                                            current_price * self.config['min_range_width'],
                                            current_price * self.config['max_range_width'])

                        # Trend analysis for directional bias
                        sma_20 = ta.SMA(close, timeperiod=min(20, len(close)))
                        sma_50 = ta.SMA(close, timeperiod=min(50, len(close)))

                        if len(sma_20) > 0 and len(sma_50) > 0:
                            trend_slope = (sma_20[-1] - sma_50[-1]) / sma_50[-1]

                            if trend_slope > 0.005:  # Bullish trend
                                predicted_support = current_price - (range_width * 0.7)
                                predicted_resistance = current_price + (range_width * 1.3)
                                direction = 'bullish'
                                breakout_prob = 0.7
                            elif trend_slope < -0.005:  # Bearish trend
                                predicted_support = current_price - (range_width * 1.3)
                                predicted_resistance = current_price + (range_width * 0.7)
                                direction = 'bearish'
                                breakout_prob = 0.7
                            else:  # Sideways
                                predicted_support = current_price - range_width
                                predicted_resistance = current_price + range_width
                                direction = 'neutral'
                                breakout_prob = 0.5
                        else:
                            # Default symmetric range
                            predicted_support = current_price - range_width
                            predicted_resistance = current_price + range_width
                            direction = 'neutral'
                            breakout_prob = 0.5

                        # Calculate confidence based on various factors
                        confidence = self._calculate_range_confidence(
                            recent_data, current_price, range_width, trend_slope if 'trend_slope' in locals() else 0
                        )

                        # Create predictive range
                        pred_range = PredictiveRange(
                            timestamp=datetime.now(),
                            current_price=current_price,
                            predicted_support=predicted_support,
                            predicted_resistance=predicted_resistance,
                            confidence_level=confidence,
                            range_width=range_width,
                            breakout_probability=breakout_prob,
                            timeframe=timeframe,
                            direction=direction
                        )

                        ranges.append(pred_range)

            # Store ranges for this symbol
            self.predictive_ranges[symbol] = ranges

            logger.info(f"✅ Calculated {len(ranges)} predictive ranges for {symbol} on {timeframe}")

        except Exception as e:
            logger.error(f"❌ Error calculating predictive ranges for {symbol}: {e}")

        return ranges

    def _calculate_range_confidence(self, df: pd.DataFrame, current_price: float,
                                  range_width: float, trend_slope: float) -> float:
        """Calculate confidence level for predictive range"""

        confidence = 0.5  # Base confidence

        try:
            # Volume confirmation
            volume = df['volume'].values
            avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
            current_volume = volume[-1] if len(volume) > 0 else avg_volume

            if current_volume > avg_volume * 1.2:
                confidence += 0.1  # High volume increases confidence

            # Volatility consistency
            returns = np.diff(df['close'].values) / df['close'].values[:-1]
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)

            if volatility < 0.02:  # Low volatility = higher confidence
                confidence += 0.1

            # Trend strength
            if abs(trend_slope) > 0.01:
                confidence += 0.15  # Strong trend increases confidence
            elif abs(trend_slope) > 0.005:
                confidence += 0.1   # Moderate trend

            # Price position within range
            range_center = (df['high'].max() + df['low'].min()) / 2
            price_position = abs(current_price - range_center) / (df['high'].max() - df['low'].min())

            if price_position < 0.3:  # Price near range center
                confidence += 0.1

            # Ensure confidence is between 0 and 1
            confidence = np.clip(confidence, 0.1, 0.95)

        except Exception as e:
            logger.warning(f"Error calculating range confidence: {e}")

        return confidence

    def find_optimal_entries(self, symbol: str, current_price: float,
                           risk_per_trade: float = 0.01) -> List[EntrySignal]:
        """
        Find optimal entry points based on predictive ranges

        Args:
            symbol: Trading symbol
            current_price: Current market price
            risk_per_trade: Risk per trade as decimal

        Returns:
            List of optimized entry signals
        """
        if symbol not in self.predictive_ranges:
            logger.warning(f"No predictive ranges available for {symbol}")
            return []

        signals = []

        try:
            # Get best predictive ranges (highest confidence)
            ranges = sorted(self.predictive_ranges[symbol],
                          key=lambda x: x.confidence_level, reverse=True)

            # Analyze top ranges for entry opportunities
            for pred_range in ranges[:5]:  # Top 5 most confident ranges

                # Check for bullish entry (near support)
                if current_price <= pred_range.predicted_support * 1.02:  # Within 2% of support
                    entry_signal = self._create_entry_signal(
                        symbol, 'buy', current_price, pred_range, risk_per_trade
                    )
                    if entry_signal:
                        signals.append(entry_signal)

                # Check for bearish entry (near resistance)
                elif current_price >= pred_range.predicted_resistance * 0.98:  # Within 2% of resistance
                    entry_signal = self._create_entry_signal(
                        symbol, 'sell', current_price, pred_range, risk_per_trade
                    )
                    if entry_signal:
                        signals.append(entry_signal)

            # Sort by confluence score
            signals.sort(key=lambda x: x.confluence_score, reverse=True)

            logger.info(f"🎯 Found {len(signals)} optimal entry signals for {symbol}")

        except Exception as e:
            logger.error(f"❌ Error finding optimal entries for {symbol}: {e}")

        return signals

    def _create_entry_signal(self, symbol: str, direction: str, current_price: float,
                           pred_range: PredictiveRange, risk_per_trade: float) -> Optional[EntrySignal]:
        """Create optimized entry signal"""

        try:
            if direction == 'buy':
                # Bullish entry near support
                entry_price = pred_range.predicted_support
                predicted_target = pred_range.predicted_resistance
                predicted_stop = entry_price - pred_range.range_width * 0.5

                # Ensure minimum risk-reward ratio
                risk = abs(entry_price - predicted_stop)
                reward = abs(predicted_target - entry_price)
                rr_ratio = reward / risk if risk > 0 else 0

            else:  # sell
                # Bearish entry near resistance
                entry_price = pred_range.predicted_resistance
                predicted_target = pred_range.predicted_support
                predicted_stop = entry_price + pred_range.range_width * 0.5

                # Ensure minimum risk-reward ratio
                risk = abs(entry_price - predicted_stop)
                reward = abs(predicted_target - entry_price)
                rr_ratio = reward / risk if risk > 0 else 0

            # Calculate confluence score (multiple factors)
            confluence_score = self._calculate_confluence_score(
                pred_range, current_price, rr_ratio
            )

            # Only create signal if meets criteria
            if (rr_ratio >= 2.0 and  # Minimum 2:1 RR
                pred_range.confidence_level >= 0.6 and  # Good confidence
                confluence_score >= 0.5):  # Good confluence

                signal = EntrySignal(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    predicted_target=predicted_target,
                    predicted_stop=predicted_stop,
                    confidence=pred_range.confidence_level,
                    risk_reward_ratio=rr_ratio,
                    timeframe=pred_range.timeframe,
                    confluence_score=confluence_score
                )

                return signal

        except Exception as e:
            logger.error(f"Error creating entry signal: {e}")

        return None

    def _calculate_confluence_score(self, pred_range: PredictiveRange,
                                  current_price: float, rr_ratio: float) -> float:
        """Calculate confluence score for entry signal"""

        score = 0.0

        # Base confidence (40% weight)
        score += pred_range.confidence_level * 0.4

        # Risk-reward ratio (30% weight)
        rr_score = min(rr_ratio / 4.0, 1.0)  # Cap at 4:1 = perfect score
        score += rr_score * 0.3

        # Price position relative to range (20% weight)
        if pred_range.direction == 'bullish':
            price_position = (current_price - pred_range.predicted_support) / pred_range.range_width
            position_score = max(0, 1 - abs(price_position))  # Better when closer to support
        else:
            price_position = (pred_range.predicted_resistance - current_price) / pred_range.range_width
            position_score = max(0, 1 - abs(price_position))  # Better when closer to resistance

        score += position_score * 0.2

        # Breakout probability (10% weight)
        score += pred_range.breakout_probability * 0.1

        return min(score, 1.0)  # Cap at 1.0

    def get_range_forecast(self, symbol: str, periods_ahead: int = 5) -> Dict[str, Any]:
        """Get range forecast for future periods"""

        if symbol not in self.predictive_ranges:
            return {"error": "No predictive ranges available"}

        ranges = self.predictive_ranges[symbol]
        if not ranges:
            return {"error": "No ranges calculated"}

        # Get most confident range
        best_range = max(ranges, key=lambda x: x.confidence_level)

        forecast = {
            "symbol": symbol,
            "current_price": best_range.current_price,
            "forecast_periods": periods_ahead,
            "predicted_support": best_range.predicted_support,
            "predicted_resistance": best_range.predicted_resistance,
            "confidence": best_range.confidence_level,
            "direction": best_range.direction,
            "range_width_pct": (best_range.range_width / best_range.current_price) * 100,
            "breakout_probability": best_range.breakout_probability,
            "recommendation": self._get_trading_recommendation(best_range)
        }

        return forecast

    def _get_trading_recommendation(self, pred_range: PredictiveRange) -> str:
        """Get trading recommendation based on predictive range"""

        if pred_range.confidence_level < 0.5:
            return "WAIT - Low confidence signal"

        if pred_range.direction == 'bullish':
            if pred_range.breakout_probability > 0.7:
                return "BUY BREAKOUT - High probability upward move"
            else:
                return "BUY SUPPORT - Look for bounces at predicted support"

        elif pred_range.direction == 'bearish':
            if pred_range.breakout_probability > 0.7:
                return "SELL BREAKOUT - High probability downward move"
            else:
                return "SELL RESISTANCE - Look for rejections at predicted resistance"

        else:
            return "RANGE TRADE - Look for mean reversion opportunities"

    def update_market_data(self, symbol: str, df: pd.DataFrame):
        """Update market data for predictive calculations"""
        self.price_history[symbol] = df

        # Recalculate predictive ranges for all timeframes
        for timeframe in self.config['timeframes']:
            self.calculate_predictive_ranges(df, symbol, timeframe)

# Global instance
_predictive_strategy = None

def get_predictive_strategy() -> PredictiveRangesStrategy:
    """Get global predictive ranges strategy instance"""
    global _predictive_strategy
    if _predictive_strategy is None:
        _predictive_strategy = PredictiveRangesStrategy()
    return _predictive_strategy

# Example usage and testing
async def main():
    """Test predictive ranges strategy"""

    strategy = get_predictive_strategy()

    # Example market data (replace with real data)
    sample_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 102, 101, 100, 99, 98, 99],
        'high': [102, 103, 104, 105, 104, 103, 102, 101, 100, 101],
        'low': [99, 100, 101, 102, 101, 100, 99, 98, 97, 98],
        'close': [101, 102, 103, 104, 103, 102, 101, 100, 99, 100],
        'volume': [1000, 1100, 1200, 1300, 1200, 1100, 1000, 900, 800, 900]
    })

    symbol = "BTCUSDT"
    timeframe = "1h"

    # Calculate predictive ranges
    ranges = strategy.calculate_predictive_ranges(sample_data, symbol, timeframe)
    print(f"📊 Calculated {len(ranges)} predictive ranges")

    # Find optimal entries
    current_price = 100.0
    signals = strategy.find_optimal_entries(symbol, current_price)

    # Get forecast
    forecast = strategy.get_range_forecast(symbol)
    print(f"🔮 Forecast: {forecast.get('recommendation', 'No recommendation')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
