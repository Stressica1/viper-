#!/usr/bin/env python3
"""
# Target ADVANCED TREND DIRECTION DETECTOR
Stable trend detection with ATR, moving averages, and Fibonacci levels
Features:
- Multi-timeframe trend confirmation
- ATR-based dynamic levels
- Fibonacci retracement zones
- Trend stability filters
- Configurable parameters for testing
"""

import os
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import ccxt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"

class TrendStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

@dataclass
class TrendSignal:
    direction: TrendDirection
    strength: TrendStrength
    confidence: float
    atr_value: float
    support_level: float
    resistance_level: float
    fib_levels: Dict[str, float]
    ma_alignment: bool
    trend_age: int  # bars since trend started
    timestamp: datetime

@dataclass
class TrendConfig:
    # Moving Average Parameters
    fast_ma_length: int = 21
    slow_ma_length: int = 50
    trend_ma_length: int = 200
    
    # ATR Parameters
    atr_length: int = 14
    atr_multiplier: float = 2.0
    
    # Trend Stability
    min_trend_bars: int = 5  # Minimum bars to confirm trend
    trend_change_threshold: float = 0.02  # 2% move needed to change trend
    
    # Fibonacci Parameters
    fib_lookback: int = 100  # Bars to look back for swing highs/lows
    
    # Timeframes for multi-timeframe analysis
    timeframes: List[str] = None
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['4h', '1h', '15m']

class AdvancedTrendDetector:
    """Advanced trend detection with ATR, MA, and Fibonacci confluence"""
    
    def __init__(self, config: TrendConfig = None):
        self.config = config or TrendConfig()
        self.exchange = None
        self.price_history = {}  # Store OHLCV data per symbol/timeframe
        self.trend_history = {}  # Store trend signals per symbol
        self.last_signals = {}   # Last signal per symbol to prevent flipping
        
        logger.info("# Target Advanced Trend Detector initialized")
        logger.info(f"# Chart Config: MA({self.config.fast_ma_length},{self.config.slow_ma_length},{self.config.trend_ma_length}) "
                   f"ATR({self.config.atr_length}x{self.config.atr_multiplier}) "
                   f"MinBars({self.config.min_trend_bars})")

    async def initialize_exchange(self) -> bool:
        """Initialize exchange connection"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_API_SECRET'),
                'password': os.getenv('BITGET_API_PASSWORD'),
                'options': {'defaultType': 'swap'},
                'sandbox': False
            })
            
            self.exchange.load_markets()
            logger.info("# Check Exchange connected for trend analysis")
            return True
            
        except Exception as e:
            logger.error(f"# X Failed to connect exchange: {e}")
            return False

    async def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data and convert to DataFrame"""
        try:
            if not self.exchange:
                return None
                
            # Use synchronous fetch for OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                return None

            # CCXT returns OHLCV data as [timestamp, open, high, low, close, volume]
            try:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # Convert price columns to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)

                return df
            except Exception as df_error:
                logger.error(f"# X DataFrame construction failed: {df_error}")
                logger.error(f"   OHLCV data type: {type(ohlcv)}")
                logger.error(f"   OHLCV length: {len(ohlcv) if ohlcv else 0}")
                if ohlcv:
                    logger.error(f"   First row type: {type(ohlcv[0]) if ohlcv else 'N/A'}")
                    logger.error(f"   First row: {ohlcv[0] if len(ohlcv) > 0 else 'N/A'}")
                return None
            
        except Exception as e:
            logger.error(f"# X Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return None

    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range"""
        period = period or self.config.atr_length
        
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()
        
        return atr

    def calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate multiple moving averages"""
        return {
            'fast_ma': df['close'].rolling(window=self.config.fast_ma_length).mean(),
            'slow_ma': df['close'].rolling(window=self.config.slow_ma_length).mean(),
            'trend_ma': df['close'].rolling(window=self.config.trend_ma_length).mean()
        }

    def find_swing_points(self, df: pd.DataFrame, lookback: int = None) -> Tuple[List[float], List[float]]:
        """Find swing highs and lows for Fibonacci levels"""
        lookback = lookback or self.config.fib_lookback
        
        if len(df) < lookback:
            return [], []
            
        recent_data = df.tail(lookback)
        
        # Find swing highs (local maxima)
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_data) - 2):
            high = recent_data.iloc[i]['high']
            low = recent_data.iloc[i]['low']
            
            # Check if it's a swing high
            if (high > recent_data.iloc[i-1]['high'] and
                high > recent_data.iloc[i-2]['high'] and
                high > recent_data.iloc[i+1]['high'] and 
                high > recent_data.iloc[i+2]['high']):
                swing_highs.append(high)
            
            # Check if it's a swing low
            if (low < recent_data.iloc[i-1]['low'] and
                low < recent_data.iloc[i-2]['low'] and
                low < recent_data.iloc[i+1]['low'] and 
                low < recent_data.iloc[i+2]['low']):
                swing_lows.append(low)
        
        return swing_highs, swing_lows

    def calculate_fibonacci_levels(self, swing_highs: List[float], swing_lows: List[float], 
                                 current_price: float) -> Dict[str, float]:
        """Calculate key Fibonacci retracement levels"""
        if not swing_highs or not swing_lows:
            return {}
        
        # Get most recent significant swing high and low
        recent_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
        recent_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
        
        if recent_high == recent_low:
            return {}
        
        # Calculate Fibonacci levels
        diff = recent_high - recent_low
        
        fib_levels = {
            'fib_0': recent_low,
            'fib_236': recent_low + diff * 0.236,
            'fib_382': recent_low + diff * 0.382,
            'fib_500': recent_low + diff * 0.500,
            'fib_618': recent_low + diff * 0.618,
            'fib_786': recent_low + diff * 0.786,
            'fib_100': recent_high,
            'fib_1236': recent_high + diff * 0.236,  # Extension levels
            'fib_1618': recent_high + diff * 0.618
        }
        
        return fib_levels

    def analyze_ma_alignment(self, mas: Dict[str, pd.Series], current_idx: int) -> Tuple[bool, TrendDirection]:
        """Analyze moving average alignment for trend direction"""
        try:
            fast = mas['fast_ma'].iloc[current_idx]
            slow = mas['slow_ma'].iloc[current_idx]
            trend = mas['trend_ma'].iloc[current_idx]
            
            if pd.isna(fast) or pd.isna(slow) or pd.isna(trend):
                return False, TrendDirection.NEUTRAL
            
            # Perfect bullish alignment
            if fast > slow > trend:
                return True, TrendDirection.STRONG_BULLISH
            
            # Basic bullish alignment
            elif fast > slow and slow > trend * 0.999:  # Small tolerance for noise
                return True, TrendDirection.BULLISH
            
            # Perfect bearish alignment
            elif fast < slow < trend:
                return True, TrendDirection.STRONG_BEARISH
            
            # Basic bearish alignment
            elif fast < slow and slow < trend * 1.001:  # Small tolerance for noise
                return True, TrendDirection.BEARISH
            
            else:
                return False, TrendDirection.NEUTRAL
                
        except (IndexError, KeyError):
            return False, TrendDirection.NEUTRAL

    def check_fibonacci_confluence(self, current_price: float, fib_levels: Dict[str, float]) -> float:
        """Check if price is near key Fibonacci levels for confluence"""
        if not fib_levels:
            return 0.0
        
        confluence_score = 0.0
        tolerance = 0.01  # 1% tolerance around Fib levels
        
        key_levels = ['fib_382', 'fib_500', 'fib_618', 'fib_786']
        
        for level_name in key_levels:
            if level_name in fib_levels:
                level_price = fib_levels[level_name]
                if level_price > 0:
                    distance = abs(current_price - level_price) / level_price
                    if distance <= tolerance:
                        # Closer to level = higher confluence
                        confluence_score += (tolerance - distance) / tolerance
        
        return min(confluence_score, 1.0)  # Cap at 1.0

    def calculate_trend_strength(self, df: pd.DataFrame, mas: Dict[str, pd.Series], 
                               atr: pd.Series, current_idx: int) -> TrendStrength:
        """Calculate trend strength based on multiple factors"""
        try:
            current_price = df.iloc[current_idx]['close']
            fast_ma = mas['fast_ma'].iloc[current_idx]
            slow_ma = mas['slow_ma'].iloc[current_idx]
            atr_value = atr.iloc[current_idx]
            
            if pd.isna(fast_ma) or pd.isna(slow_ma) or pd.isna(atr_value):
                return TrendStrength.VERY_WEAK
            
            # Calculate MA separation as % of ATR
            ma_separation = abs(fast_ma - slow_ma) / atr_value if atr_value > 0 else 0
            
            # Calculate price momentum
            price_change = (current_price - df.iloc[max(0, current_idx - 5)]['close']) / current_price
            momentum_strength = abs(price_change) * 100
            
            # Combine factors
            strength_score = ma_separation * 0.6 + momentum_strength * 0.4
            
            if strength_score >= 3.0:
                return TrendStrength.VERY_STRONG
            elif strength_score >= 2.0:
                return TrendStrength.STRONG
            elif strength_score >= 1.0:
                return TrendStrength.MODERATE
            elif strength_score >= 0.5:
                return TrendStrength.WEAK
            else:
                return TrendStrength.VERY_WEAK
                
        except (IndexError, KeyError, ZeroDivisionError):
            return TrendStrength.VERY_WEAK

    def is_trend_stable(self, symbol: str, new_direction: TrendDirection) -> bool:
        """Check if trend change is stable enough to avoid whipsaws"""
        if symbol not in self.last_signals:
            return True
        
        last_signal = self.last_signals[symbol]
        
        # If same direction, it's stable
        if last_signal.direction == new_direction:
            return True
        
        # Check if enough time has passed
        time_diff = (datetime.now() - last_signal.timestamp).total_seconds() / 3600  # hours
        min_time_hours = self.config.min_trend_bars * 0.25  # Assuming 15m bars
        
        if time_diff < min_time_hours:
            return False
        
        return True

    async def detect_trend(self, symbol: str, timeframe: str = '1h') -> Optional[TrendSignal]:
        """Main trend detection function"""
        try:
            # Get market data
            df = await self.get_ohlcv_data(symbol, timeframe, limit=300)
            if df is None or len(df) < self.config.trend_ma_length:
                return None
            
            current_idx = -1  # Latest bar
            current_price = df.iloc[current_idx]['close']
            
            # Calculate indicators
            atr = self.calculate_atr(df)
            mas = self.calculate_moving_averages(df)
            
            # Find swing points and calculate Fibonacci levels
            swing_highs, swing_lows = self.find_swing_points(df)
            fib_levels = self.calculate_fibonacci_levels(swing_highs, swing_lows, current_price)
            
            # Analyze trend direction
            ma_aligned, trend_direction = self.analyze_ma_alignment(mas, current_idx)
            
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength(df, mas, atr, current_idx)
            
            # Check Fibonacci confluence
            fib_confluence = self.check_fibonacci_confluence(current_price, fib_levels)
            
            # Calculate support/resistance levels using ATR
            atr_value = atr.iloc[current_idx] if not pd.isna(atr.iloc[current_idx]) else 0
            support_level = current_price - (atr_value * self.config.atr_multiplier)
            resistance_level = current_price + (atr_value * self.config.atr_multiplier)
            
            # Calculate confidence
            confidence = 0.0
            if ma_aligned:
                confidence += 0.4
            if trend_strength.value >= 3:
                confidence += 0.3
            if fib_confluence > 0.5:
                confidence += 0.2
            if atr_value > 0:
                confidence += 0.1
            
            # Check trend stability
            if not self.is_trend_stable(symbol, trend_direction):
                # If not stable, keep previous direction but lower confidence
                if symbol in self.last_signals:
                    trend_direction = self.last_signals[symbol].direction
                    confidence *= 0.5
            
            # Create trend signal
            signal = TrendSignal(
                direction=trend_direction,
                strength=trend_strength,
                confidence=confidence,
                atr_value=atr_value,
                support_level=support_level,
                resistance_level=resistance_level,
                fib_levels=fib_levels,
                ma_alignment=ma_aligned,
                trend_age=0,  # Would need to track this over time
                timestamp=datetime.now()
            )
            
            # Store signal
            self.last_signals[symbol] = signal
            
            logger.info(f"# Target {symbol} Trend: {trend_direction.value} ({trend_strength.value}/5) "
                       f"Conf:{confidence:.2f} ATR:{atr_value:.6f} "
                       f"S/R:{support_level:.6f}/{resistance_level:.6f} "
                       f"FibConf:{fib_confluence:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"# X Error detecting trend for {symbol}: {e}")
            return None

    async def multi_timeframe_analysis(self, symbol: str) -> Dict[str, TrendSignal]:
        """Analyze trend across multiple timeframes"""
        signals = {}
        
        for timeframe in self.config.timeframes:
            signal = await self.detect_trend(symbol, timeframe)
            if signal:
                signals[timeframe] = signal
        
        return signals

    def get_consensus_trend(self, mtf_signals: Dict[str, TrendSignal]) -> Optional[TrendSignal]:
        """Get consensus trend from multiple timeframes"""
        if not mtf_signals:
            return None
        
        # Weight by timeframe importance (higher timeframes = more weight)
        timeframe_weights = {'4h': 0.5, '1h': 0.3, '15m': 0.2}
        
        direction_scores = {
            TrendDirection.STRONG_BULLISH: 0,
            TrendDirection.BULLISH: 0,
            TrendDirection.NEUTRAL: 0,
            TrendDirection.BEARISH: 0,
            TrendDirection.STRONG_BEARISH: 0
        }
        
        total_confidence = 0
        total_weight = 0
        
        for tf, signal in mtf_signals.items():
            weight = timeframe_weights.get(tf, 0.1)
            direction_scores[signal.direction] += weight * signal.confidence
            total_confidence += signal.confidence * weight
            total_weight += weight
        
        # Find dominant direction
        dominant_direction = max(direction_scores, key=direction_scores.get)
        consensus_confidence = total_confidence / total_weight if total_weight > 0 else 0
        
        # Use the highest timeframe signal as base, but adjust direction and confidence
        base_signal = list(mtf_signals.values())[0]
        base_signal.direction = dominant_direction
        base_signal.confidence = consensus_confidence
        
        return base_signal

    def validate_trade_direction_with_trend(self, symbol: str, proposed_side: str, 
                                           mtf_signals: Optional[Dict[str, TrendSignal]] = None) -> Tuple[bool, str, float]:
        """
        Validate if a proposed trade direction aligns with sound trend analysis
        
        Args:
            symbol: Trading symbol
            proposed_side: 'buy' or 'sell'
            mtf_signals: Multi-timeframe signals (if None, will fetch fresh)
            
        Returns:
            Tuple of (is_valid, reason, confidence_score)
        """
        try:
            # Get multi-timeframe signals if not provided
            if mtf_signals is None:
                # Use the last available MTF analysis or perform fresh analysis
                if hasattr(self, '_last_mtf_signals') and symbol in self._last_mtf_signals:
                    mtf_signals = self._last_mtf_signals[symbol]
                else:
                    logger.warning(f"# Warning No MTF signals available for {symbol} trade validation")
                    return False, "No trend data available", 0.0
            
            if not mtf_signals:
                return False, "No trend signals available", 0.0
                
            # Get consensus trend
            consensus_signal = self.get_consensus_trend(mtf_signals)
            if not consensus_signal:
                return False, "Unable to determine trend consensus", 0.0
            
            # Check minimum confidence threshold for trading
            min_confidence = 0.6
            if consensus_signal.confidence < min_confidence:
                return False, f"Trend confidence too low ({consensus_signal.confidence:.2f} < {min_confidence})", consensus_signal.confidence
            
            # Check trend strength - avoid trading in very weak trends
            if consensus_signal.strength == TrendStrength.VERY_WEAK:
                return False, "Trend strength too weak for reliable trading", consensus_signal.confidence
                
            # Validate trade direction alignment
            trend_direction = consensus_signal.direction
            
            # Define valid combinations
            valid_bullish_directions = [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]
            valid_bearish_directions = [TrendDirection.STRONG_BEARISH, TrendDirection.BEARISH]
            
            if proposed_side == 'buy':
                if trend_direction in valid_bullish_directions:
                    confidence_bonus = 0.2 if trend_direction == TrendDirection.STRONG_BULLISH else 0.1
                    final_confidence = min(1.0, consensus_signal.confidence + confidence_bonus)
                    return True, f"✓ Buy aligns with {trend_direction.value} trend", final_confidence
                elif trend_direction == TrendDirection.NEUTRAL:
                    # Allow trades in neutral trend but with reduced confidence
                    if consensus_signal.confidence >= 0.7:  # Higher threshold for neutral
                        return True, "⚠ Buy in neutral trend (higher risk)", consensus_signal.confidence * 0.8
                    else:
                        return False, "Neutral trend with insufficient confidence for buy", consensus_signal.confidence
                else:
                    return False, f"✗ Buy conflicts with {trend_direction.value} trend", consensus_signal.confidence
                    
            elif proposed_side == 'sell':
                if trend_direction in valid_bearish_directions:
                    confidence_bonus = 0.2 if trend_direction == TrendDirection.STRONG_BEARISH else 0.1
                    final_confidence = min(1.0, consensus_signal.confidence + confidence_bonus)
                    return True, f"✓ Sell aligns with {trend_direction.value} trend", final_confidence
                elif trend_direction == TrendDirection.NEUTRAL:
                    # Allow trades in neutral trend but with reduced confidence
                    if consensus_signal.confidence >= 0.7:  # Higher threshold for neutral
                        return True, "⚠ Sell in neutral trend (higher risk)", consensus_signal.confidence * 0.8
                    else:
                        return False, "Neutral trend with insufficient confidence for sell", consensus_signal.confidence
                else:
                    return False, f"✗ Sell conflicts with {trend_direction.value} trend", consensus_signal.confidence
            else:
                return False, f"Invalid trade side: {proposed_side}", 0.0
                
        except Exception as e:
            logger.error(f"# X Error validating trade direction for {symbol}: {e}")
            return False, f"Validation error: {str(e)}", 0.0

    def check_multi_timeframe_trend_agreement(self, mtf_signals: Dict[str, TrendSignal], 
                                            min_agreement_threshold: float = 0.7) -> Tuple[bool, str]:
        """
        Check if multiple timeframes show sufficient agreement in trend direction
        
        Args:
            mtf_signals: Multi-timeframe trend signals
            min_agreement_threshold: Minimum agreement ratio (0.7 = 70% agreement)
            
        Returns:
            Tuple of (has_agreement, description)
        """
        try:
            if not mtf_signals or len(mtf_signals) < 2:
                return False, "Insufficient timeframes for agreement check"
                
            # Count direction votes with broader agreement logic
            direction_votes = {
                TrendDirection.STRONG_BULLISH: 0,
                TrendDirection.BULLISH: 0,
                TrendDirection.NEUTRAL: 0,
                TrendDirection.BEARISH: 0,
                TrendDirection.STRONG_BEARISH: 0
            }
            
            # Also track broader directional agreement (bullish vs bearish vs neutral)
            broad_direction_votes = {
                'bullish': 0,  # STRONG_BULLISH + BULLISH
                'bearish': 0,  # STRONG_BEARISH + BEARISH
                'neutral': 0   # NEUTRAL
            }
            
            # Weight by timeframe importance and signal confidence
            timeframe_weights = {'4h': 0.5, '1h': 0.3, '15m': 0.2}
            total_weight = 0
            
            for tf, signal in mtf_signals.items():
                weight = timeframe_weights.get(tf, 0.1) * signal.confidence
                direction_votes[signal.direction] += weight
                
                # Count broad directional agreement
                if signal.direction in [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]:
                    broad_direction_votes['bullish'] += weight
                elif signal.direction in [TrendDirection.STRONG_BEARISH, TrendDirection.BEARISH]:
                    broad_direction_votes['bearish'] += weight
                else:
                    broad_direction_votes['neutral'] += weight
                    
                total_weight += weight
            
            if total_weight == 0:
                return False, "No valid signals for agreement calculation"
                
            # Check both specific and broad directional agreement
            max_specific_votes = max(direction_votes.values())
            max_broad_votes = max(broad_direction_votes.values())
            
            specific_agreement = max_specific_votes / total_weight if total_weight > 0 else 0
            broad_agreement = max_broad_votes / total_weight if total_weight > 0 else 0
            
            # Use broad agreement if it's stronger (allows STRONG_BULLISH + BULLISH to agree)
            agreement_ratio = max(specific_agreement, broad_agreement)
            
            dominant_direction = max(direction_votes, key=direction_votes.get)
            
            if agreement_ratio >= min_agreement_threshold:
                agreement_type = "specific" if specific_agreement >= broad_agreement else "broad directional"
                return True, f"Strong {agreement_type} agreement ({agreement_ratio:.1%}) on {dominant_direction.value} trend"
            else:
                return False, f"Weak agreement ({agreement_ratio:.1%}) - conflicting timeframe signals"
                
        except Exception as e:
            logger.error(f"# X Error checking MTF agreement: {e}")
            return False, f"Agreement check error: {str(e)}"

    async def enhanced_multi_timeframe_analysis(self, symbol: str) -> Tuple[Dict[str, TrendSignal], bool, str]:
        """
        Enhanced multi-timeframe analysis with agreement validation
        
        Returns:
            Tuple of (mtf_signals, has_sound_trend, trend_summary)
        """
        try:
            # Get multi-timeframe signals - use cached if available for testing
            if hasattr(self, '_last_mtf_signals') and symbol in self._last_mtf_signals:
                mtf_signals = self._last_mtf_signals[symbol]
                logger.info(f"# Test Using cached MTF signals for {symbol}")
            else:
                # Get multi-timeframe signals normally
                mtf_signals = await self.multi_timeframe_analysis(symbol)
            
            if not mtf_signals:
                return {}, False, "No trend signals available"
            
            # Store for later use
            if not hasattr(self, '_last_mtf_signals'):
                self._last_mtf_signals = {}
            self._last_mtf_signals[symbol] = mtf_signals
            
            # Check timeframe agreement
            has_agreement, agreement_desc = self.check_multi_timeframe_trend_agreement(mtf_signals)
            
            # Get consensus
            consensus = self.get_consensus_trend(mtf_signals)
            
            if consensus and has_agreement and consensus.confidence >= 0.6:
                trend_summary = f"✓ Sound trend: {consensus.direction.value} " + \
                              f"(Confidence: {consensus.confidence:.2f}, {agreement_desc})"
                return mtf_signals, True, trend_summary
            else:
                reasons = []
                if not has_agreement:
                    reasons.append(agreement_desc)
                if not consensus:
                    reasons.append("no consensus")
                elif consensus.confidence < 0.6:
                    reasons.append(f"low confidence ({consensus.confidence:.2f})")
                    
                trend_summary = f"⚠ Unsound trend: {'; '.join(reasons)}"
                return mtf_signals, False, trend_summary
                
        except Exception as e:
            logger.error(f"# X Error in enhanced MTF analysis for {symbol}: {e}")
            return {}, False, f"Analysis error: {str(e)}"

async def test_trend_detector():
    """Test the trend detector with different configurations"""
    
    # Test different configurations
    test_configs = [
        TrendConfig(fast_ma_length=13, slow_ma_length=34, atr_multiplier=1.5),  # Aggressive
        TrendConfig(fast_ma_length=21, slow_ma_length=50, atr_multiplier=2.0),  # Balanced
        TrendConfig(fast_ma_length=34, slow_ma_length=89, atr_multiplier=2.5),  # Conservative
    ]
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    for i, config in enumerate(test_configs):
        logger.info(f"\n🧪 Testing Configuration {i+1}: MA({config.fast_ma_length},{config.slow_ma_length}) ATR({config.atr_multiplier})")
        
        detector = AdvancedTrendDetector(config)
        
        if not await detector.initialize_exchange():
            continue
        
        for symbol in symbols:
            # Single timeframe analysis
            signal = await detector.detect_trend(symbol)
            
            if signal:
                logger.info(f"# Chart {symbol}: {signal.direction.value} "
                           f"Strength:{signal.strength.value}/5 "
                           f"Confidence:{signal.confidence:.2f}")
            
            # Multi-timeframe analysis
            mtf_signals = await detector.multi_timeframe_analysis(symbol)
            consensus = detector.get_consensus_trend(mtf_signals)
            
            if consensus:
                logger.info(f"# Target {symbol} Consensus: {consensus.direction.value} "
                           f"Confidence:{consensus.confidence:.2f}")
        
        detector.exchange.close()

if __name__ == "__main__":
    asyncio.run(test_trend_detector())
