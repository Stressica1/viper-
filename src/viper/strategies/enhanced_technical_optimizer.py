#!/usr/bin/env python3
"""
# Rocket ENHANCED TECHNICAL INDICATORS OPTIMIZER
Advanced technical analysis with optimized calculations and multi-timeframe confluence

This enhanced version includes:
- Optimized indicator calculations for better performance
- Multi-timeframe confluence analysis
- Advanced pattern recognition
- Fibonacci extensions and projections
- Volume profile analysis
- Market microstructure indicators
"""

import os
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import ccxt
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTrendDirection(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"

class EnhancedTrendStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

@dataclass
class EnhancedTrendSignal:
    direction: EnhancedTrendDirection
    strength: EnhancedTrendStrength
    confidence: float
    confluence_score: float
    timeframe_alignment: Dict[str, EnhancedTrendDirection]
    key_levels: Dict[str, float]
    pattern_signals: List[str]
    volume_profile: Dict[str, float]
    timestamp: datetime

@dataclass
class EnhancedConfig:
    # Optimized moving average periods
    fast_ma_length: int = 13
    medium_ma_length: int = 34
    slow_ma_length: int = 89
    trend_ma_length: int = 200

    # Enhanced ATR parameters
    atr_length: int = 14
    atr_multiplier_base: float = 1.5
    atr_multiplier_dynamic: bool = True

    # Fibonacci parameters
    fib_lookback: int = 150
    fib_extensions: List[float] = None

    # Volume profile
    volume_profile_bins: int = 20
    volume_profile_lookback: int = 100

    # Pattern recognition
    pattern_min_strength: float = 0.6
    pattern_max_lookback: int = 50

    # Multi-timeframe weights
    timeframe_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.fib_extensions is None:
            self.fib_extensions = [1.236, 1.382, 1.618, 2.0, 2.618]

        if self.timeframe_weights is None:
            self.timeframe_weights = {
                '1m': 0.1, '5m': 0.15, '15m': 0.2, '1h': 0.25,
                '4h': 0.2, '1d': 0.1
            }

class EnhancedTechnicalOptimizer:
    """Enhanced technical analysis with optimized calculations and advanced features"""

    def __init__(self, config: EnhancedConfig = None):
        self.config = config or EnhancedConfig()
        self.exchange = None
        self.price_cache = {}  # Cache for multi-timeframe data
        self.pattern_cache = {}  # Cache for pattern recognition
        self.volume_profiles = {}  # Cache for volume profiles

        logger.info("# Target Enhanced Technical Optimizer initialized")
        logger.info(f"# Chart Config optimized for performance and accuracy")

    async def initialize_exchange(self) -> bool:
        """Initialize exchange with optimized settings"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_API_SECRET'),
                'password': os.getenv('BITGET_API_PASSWORD'),
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'},
                'sandbox': False
            })

            self.exchange.load_markets()
            logger.info("# Check Exchange connected for enhanced technical analysis")
            return True

        except Exception as e:
            logger.error(f"# X Failed to connect exchange: {e}")
            return False

    async def get_multi_timeframe_data(self, symbol: str, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Optimized multi-timeframe data fetching with caching"""
        if timeframes is None:
            timeframes = list(self.config.timeframe_weights.keys())

        data_frames = {}

        for timeframe in timeframes:
            cache_key = f"{symbol}_{timeframe}"

            # Check cache first
            if cache_key in self.price_cache:
                cached_data = self.price_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 300:  # 5 min cache
                    data_frames[timeframe] = cached_data['data']
                    continue

            # Fetch new data
            df = await self._fetch_optimized_ohlcv(symbol, timeframe)
            if df is not None:
                data_frames[timeframe] = df

                # Update cache
                self.price_cache[cache_key] = {
                    'data': df,
                    'timestamp': datetime.now()
                }

        return data_frames

    async def _fetch_optimized_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> Optional[pd.DataFrame]:
        """Optimized OHLCV fetching with error handling"""
        try:
            if not self.exchange:
                return None

            # Use synchronous fetch for better performance
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv or len(ohlcv) == 0:
                return None

            # Convert to DataFrame efficiently
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert to float32 for memory efficiency
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(np.float32)

            return df

        except Exception as e:
            logger.error(f"# X Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return None

    def calculate_optimized_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate optimized technical indicators with performance improvements"""
        try:
            # Vectorized calculations for better performance
            df = df.copy()

            # Optimized moving averages using pandas rolling
            df['fast_ma'] = df['close'].rolling(window=self.config.fast_ma_length).mean()
            df['medium_ma'] = df['close'].rolling(window=self.config.medium_ma_length).mean()
            df['slow_ma'] = df['close'].rolling(window=self.config.slow_ma_length).mean()
            df['trend_ma'] = df['close'].rolling(window=self.config.trend_ma_length).mean()

            # Exponential moving averages
            df['fast_ema'] = df['close'].ewm(span=self.config.fast_ma_length).mean()
            df['medium_ema'] = df['close'].ewm(span=self.config.medium_ma_length).mean()

            # RSI with optimized calculation
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

            # MACD with optimized parameters
            macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            # ATR with dynamic multiplier
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.config.atr_length)

            if self.config.atr_multiplier_dynamic:
                # Dynamic ATR multiplier based on volatility
                volatility = df['atr'].rolling(window=20).std()
                df['atr_multiplier'] = self.config.atr_multiplier_base * (1 + volatility / df['atr'])
            else:
                df['atr_multiplier'] = self.config.atr_multiplier_base

            # Support and resistance levels
            df['support_20'] = df['low'].rolling(window=20).min()
            df['resistance_20'] = df['high'].rolling(window=20).max()

            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # Price momentum
            df['momentum'] = df['close'] - df['close'].shift(10)
            df['roc'] = ta.momentum.roc(df['close'], window=10)

            # Trend strength indicators
            df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].shift(20)
            df['trend_slope'] = (df['fast_ma'] - df['fast_ma'].shift(5)) / 5

            # Fill NaN values efficiently
            df = df.fillna(method='forward').fillna(0)

            return df

        except Exception as e:
            logger.error(f"# X Error calculating optimized indicators: {e}")
            return df

    def detect_advanced_patterns(self, df: pd.DataFrame) -> List[str]:
        """Advanced pattern detection with optimized algorithms"""
        patterns = []

        try:
            # Candlestick patterns
            if len(df) >= 3:
                # Doji pattern
                body_size = abs(df['close'] - df['open'])
                total_range = df['high'] - df['low']
                doji_ratio = body_size / total_range

                if doji_ratio.iloc[-1] < 0.1:  # Very small body
                    patterns.append("DOJI")

                # Engulfing patterns
                current_body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
                prev_body = abs(df['close'].iloc[-2] - df['open'].iloc[-2])

                if (df['close'].iloc[-1] > df['open'].iloc[-1] and  # Current bullish:
                    df['close'].iloc[-2] < df['open'].iloc[-2] and  # Previous bearish
                    df['close'].iloc[-1] > df['open'].iloc[-2] and  # Engulfs previous open
                    df['open'].iloc[-1] < df['close'].iloc[-2]):    # Engulfs previous close
                    patterns.append("BULLISH_ENGULFING")

                elif (df['close'].iloc[-1] < df['open'].iloc[-1] and  # Current bearish:
                      df['close'].iloc[-2] > df['open'].iloc[-2] and  # Previous bullish
                      df['close'].iloc[-1] < df['open'].iloc[-2] and  # Engulfs previous open
                      df['open'].iloc[-1] > df['close'].iloc[-2]):    # Engulfs previous close
                    patterns.append("BEARISH_ENGULFING")

            # Chart patterns
            if len(df) >= 50:
                # Double top/bottom detection
                recent_highs = df['high'].tail(30)
                peaks = self._find_peaks_valleys(recent_highs, prominence=0.01)

                if len(peaks) >= 2:
                    # Check for double top
                    peak_prices = recent_highs.iloc[peaks]
                    if abs(peak_prices.iloc[-1] - peak_prices.iloc[-2]) / peak_prices.iloc[-2] < 0.02:
                        patterns.append("DOUBLE_TOP")

                    # Check for double bottom
                    recent_lows = df['low'].tail(30)
                    valleys = self._find_peaks_valleys(-recent_lows, prominence=0.01)
                    if len(valleys) >= 2:
                        valley_prices = recent_lows.iloc[valleys]
                        if abs(valley_prices.iloc[-1] - valley_prices.iloc[-2]) / valley_prices.iloc[-2] < 0.02:
                            patterns.append("DOUBLE_BOTTOM")

            # Triangle patterns
            if len(df) >= 40:
                highs_trend = np.polyfit(range(len(df['high'].tail(20))), df['high'].tail(20), 1)[0]
                lows_trend = np.polyfit(range(len(df['low'].tail(20))), df['low'].tail(20), 1)[0]

                if highs_trend < -0.001 and lows_trend > 0.001:
                    patterns.append("DESCENDING_TRIANGLE")
                elif highs_trend > 0.001 and lows_trend < -0.001:
                    patterns.append("ASCENDING_TRIANGLE")

        except Exception as e:
            logger.warning(f"# Warning Error in pattern detection: {e}")

        return patterns

    def _find_peaks_valleys(self, data: pd.Series, prominence: float = 0.01) -> List[int]:
        """Find peaks and valleys in data series"""
        peaks = []
        valleys = []

        for i in range(2, len(data) - 2):
            # Check for peak
            if (data.iloc[i] > data.iloc[i-1] and data.iloc[i] > data.iloc[i-2] and:
                data.iloc[i] > data.iloc[i+1] and data.iloc[i] > data.iloc[i+2]):
                # Check prominence
                left_min = min(data.iloc[i-2:i])
                right_min = min(data.iloc[i+1:i+3])
                if data.iloc[i] - min(left_min, right_min) > prominence * data.iloc[i]:
                    peaks.append(i)

            # Check for valley
            if (data.iloc[i] < data.iloc[i-1] and data.iloc[i] < data.iloc[i-2] and:
                data.iloc[i] < data.iloc[i+1] and data.iloc[i] < data.iloc[i+2]):
                # Check prominence
                left_max = max(data.iloc[i-2:i])
                right_max = max(data.iloc[i+1:i+3])
                if max(left_max, right_max) - data.iloc[i] > prominence * data.iloc[i]:
                    valleys.append(i)

        return peaks + valleys

    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile for key levels"""
        try:
            if len(df) < self.config.volume_profile_lookback:
                return {}

            recent_data = df.tail(self.config.volume_profile_lookback)

            # Create price bins
            price_min = recent_data['low'].min()
            price_max = recent_data['high'].max()
            bins = np.linspace(price_min, price_max, self.config.volume_profile_bins)

            # Calculate volume per bin
            volume_profile = {}
            for i in range(len(bins) - 1):
                bin_mask = (recent_data['low'] <= bins[i+1]) & (recent_data['high'] >= bins[i])
                volume_profile[f'bin_{i}'] = recent_data[bin_mask]['volume'].sum()

            # Find high volume levels
            total_volume = sum(volume_profile.values())
            high_volume_levels = {}

            for bin_name, volume in volume_profile.items():
                if volume > total_volume * 0.1:  # Top 10% of volume
                    bin_index = int(bin_name.split('_')[1])
                    price_level = (bins[bin_index] + bins[bin_index + 1]) / 2
                    high_volume_levels[f'high_volume_{bin_index}'] = price_level

            return high_volume_levels

        except Exception as e:
            logger.warning(f"# Warning Error calculating volume profile: {e}")
            return {}

    def calculate_fibonacci_levels_advanced(self, df: pd.DataFrame) -> Dict[str, float]:
        """Advanced Fibonacci level calculation with extensions"""
        try:
            if len(df) < self.config.fib_lookback:
                return {}

            recent_data = df.tail(self.config.fib_lookback)

            # Find significant swing points
            highs = recent_data['high']
            lows = recent_data['low']

            # Find swing highs and lows
            swing_highs = []
            swing_lows = []

            for i in range(5, len(recent_data) - 5):
                # Swing high
                if highs.iloc[i] == highs.iloc[i-5:i+6].max():
                    swing_highs.append(highs.iloc[i])

                # Swing low
                if lows.iloc[i] == lows.iloc[i-5:i+6].min():
                    swing_lows.append(lows.iloc[i])

            if not swing_highs or not swing_lows:
                return {}

            # Get most significant swing points
            recent_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
            recent_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)

            if recent_high == recent_low:
                return {}

            # Calculate Fibonacci levels
            diff = recent_high - recent_low
            current_price = df['close'].iloc[-1]

            fib_levels = {
                'fib_0': recent_low,
                'fib_236': recent_low + diff * 0.236,
                'fib_382': recent_low + diff * 0.382,
                'fib_500': recent_low + diff * 0.500,
                'fib_618': recent_low + diff * 0.618,
                'fib_786': recent_low + diff * 0.786,
                'fib_100': recent_high,
            }

            # Add extensions
            for ext in self.config.fib_extensions:
                fib_levels[f'fib_ext_{ext}'] = recent_high + diff * (ext - 1)

            # Calculate confluence with current price
            fib_levels['near_fib_level'] = self._find_nearest_fib_level(current_price, fib_levels)

            return fib_levels

        except Exception as e:
            logger.warning(f"# Warning Error calculating advanced Fibonacci levels: {e}")
            return {}

    def _find_nearest_fib_level(self, price: float, fib_levels: Dict[str, float], tolerance: float = 0.01) -> Optional[str]:
        """Find the nearest Fibonacci level to current price"""
        nearest_level = None
        min_distance = float('inf')

        for level_name, level_price in fib_levels.items():
            if isinstance(level_price, (int, float)):
                distance = abs(price - level_price) / price
                if distance < tolerance and distance < min_distance:
                    min_distance = distance
                    nearest_level = level_name

        return nearest_level

    def calculate_confluence_score(self, df: pd.DataFrame, fib_levels: Dict[str, float],
                                 volume_profile: Dict[str, float]) -> float:
        """Calculate confluence score from multiple indicators"""
        try:
            confluence_factors = []

            # Fibonacci confluence
            current_price = df['close'].iloc[-1]
            near_fib = self._find_nearest_fib_level(current_price, fib_levels)

            if near_fib:
                distance = abs(current_price - fib_levels[near_fib]) / current_price
                fib_score = max(0, 1 - distance * 100)  # Closer = higher score
                confluence_factors.append(fib_score)

            # Moving average confluence
            ma_alignment = 0
            if df['fast_ma'].iloc[-1] > df['medium_ma'].iloc[-1] > df['slow_ma'].iloc[-1]:
                ma_alignment = 1.0  # Bullish alignment
            elif df['fast_ma'].iloc[-1] < df['medium_ma'].iloc[-1] < df['slow_ma'].iloc[-1]:
                ma_alignment = 1.0  # Bearish alignment
            else:
                ma_alignment = 0.5  # Mixed alignment

            confluence_factors.append(ma_alignment)

            # RSI confluence
            rsi = df['rsi'].iloc[-1]
            if (rsi > 70 and current_price > df['bb_middle'].iloc[-1]) or (rsi < 30 and current_price < df['bb_middle'].iloc[-1]):
                rsi_confluence = 1.0
            else:
                rsi_confluence = 0.5

            confluence_factors.append(rsi_confluence)

            # Volume profile confluence
            volume_confluence = 0.5
            for level_name, level_price in volume_profile.items():
                if abs(current_price - level_price) / current_price < 0.005:  # Within 0.5%
                    volume_confluence = 1.0
                    break

            confluence_factors.append(volume_confluence)

            # Calculate overall confluence score
            confluence_score = np.mean(confluence_factors)

            return float(confluence_score)

        except Exception as e:
            logger.warning(f"# Warning Error calculating confluence score: {e}")
            return 0.5

    async def analyze_enhanced_trend(self, symbol: str, primary_timeframe: str = '1h') -> Optional[EnhancedTrendSignal]:
        """Enhanced trend analysis with multi-timeframe confluence"""
        try:
            # Get multi-timeframe data
            timeframe_data = await self.get_multi_timeframe_data(symbol, ['15m', '1h', '4h'])

            if not timeframe_data or primary_timeframe not in timeframe_data:
                return None

            # Calculate indicators for primary timeframe
            primary_df = self.calculate_optimized_indicators(timeframe_data[primary_timeframe])

            if primary_df.empty:
                return None

            # Multi-timeframe analysis
            timeframe_directions = {}
            timeframe_confidences = {}

            for tf, df in timeframe_data.items():
                if len(df) >= 50:
                    enhanced_df = self.calculate_optimized_indicators(df)
                    direction, confidence = self._analyze_single_timeframe_trend(enhanced_df)
                    timeframe_directions[tf] = direction
                    timeframe_confidences[tf] = confidence

            # Calculate weighted consensus direction
            consensus_direction = self._calculate_timeframe_consensus(
                timeframe_directions, timeframe_confidences
            )

            # Calculate trend strength
            trend_strength = self._calculate_enhanced_trend_strength(
                primary_df, timeframe_directions
            )

            # Advanced pattern detection
            patterns = self.detect_advanced_patterns(primary_df)

            # Volume profile analysis
            volume_profile = self.calculate_volume_profile(primary_df)

            # Fibonacci analysis
            fib_levels = self.calculate_fibonacci_levels_advanced(primary_df)

            # Confluence score
            confluence_score = self.calculate_confluence_score(primary_df, fib_levels, volume_profile)

            # Calculate overall confidence
            base_confidence = timeframe_confidences.get(primary_timeframe, 0.5)
            confidence = min(base_confidence * confluence_score * (1 + len(patterns) * 0.1), 1.0)

            # Key levels
            key_levels = {
                'support': primary_df['support_20'].iloc[-1],
                'resistance': primary_df['resistance_20'].iloc[-1],
                'pivot': (primary_df['high'].iloc[-1] + primary_df['low'].iloc[-1] + primary_df['close'].iloc[-1]) / 3,
                'atr_stop': primary_df['atr'].iloc[-1] * primary_df['atr_multiplier'].iloc[-1]
            }

            # Create enhanced signal
            signal = EnhancedTrendSignal(
                direction=consensus_direction,
                strength=trend_strength,
                confidence=confidence,
                confluence_score=confluence_score,
                timeframe_alignment=timeframe_directions,
                key_levels=key_levels,
                pattern_signals=patterns,
                volume_profile=volume_profile,
                timestamp=datetime.now()
            )

            logger.info(f"# Target Enhanced Trend Analysis for {symbol}:")
            logger.info(f"   Direction: {consensus_direction.value}")
            logger.info(f"   Strength: {trend_strength.value}/5")
            logger.info(f"   Confidence: {confidence:.3f}")
            logger.info(f"   Confluence: {confluence_score:.3f}")
            logger.info(f"   Patterns: {patterns}")

            return signal

        except Exception as e:
            logger.error(f"# X Error in enhanced trend analysis for {symbol}: {e}")
            return None

    def _analyze_single_timeframe_trend(self, df: pd.DataFrame) -> Tuple[EnhancedTrendDirection, float]:
        """Analyze trend for a single timeframe"""
        try:
            latest = df.iloc[-1]

            # MA alignment analysis
            ma_score = 0
            if latest['fast_ma'] > latest['medium_ma']:
                ma_score += 0.5
            if latest['medium_ma'] > latest['slow_ma']:
                ma_score += 0.5

            # Trend slope analysis
            trend_slope = latest.get('trend_slope', 0)
            if trend_slope > 0.001:
                ma_score += 0.3
            elif trend_slope < -0.001:
                ma_score -= 0.3

            # RSI confirmation
            rsi = latest.get('rsi', 50)
            if rsi > 60 and ma_score > 0:
                ma_score += 0.2
            elif rsi < 40 and ma_score < 0:
                ma_score += 0.2

            # Determine direction
            if ma_score >= 0.7:
                direction = EnhancedTrendDirection.STRONG_BULLISH
            elif ma_score >= 0.3:
                direction = EnhancedTrendDirection.BULLISH
            elif ma_score <= -0.7:
                direction = EnhancedTrendDirection.STRONG_BEARISH
            elif ma_score <= -0.3:
                direction = EnhancedTrendDirection.BEARISH
            else:
                direction = EnhancedTrendDirection.NEUTRAL

            # Calculate confidence
            confidence = min(abs(ma_score) + 0.3, 1.0)

            return direction, confidence

        except Exception as e:
            logger.warning(f"# Warning Error analyzing single timeframe trend: {e}")
            return EnhancedTrendDirection.NEUTRAL, 0.5

    def _calculate_timeframe_consensus(self, directions: Dict[str, EnhancedTrendDirection],
                                     confidences: Dict[str, float]) -> EnhancedTrendDirection:
        """Calculate consensus direction from multiple timeframes"""
        try:
            direction_scores = {
                EnhancedTrendDirection.STRONG_BULLISH: 0,
                EnhancedTrendDirection.BULLISH: 0,
                EnhancedTrendDirection.NEUTRAL: 0,
                EnhancedTrendDirection.BEARISH: 0,
                EnhancedTrendDirection.STRONG_BEARISH: 0
            }

            total_weight = 0

            for tf, direction in directions.items():
                weight = self.config.timeframe_weights.get(tf, 0.1)
                confidence = confidences.get(tf, 0.5)

                direction_scores[direction] += weight * confidence
                total_weight += weight

            # Find dominant direction
            dominant_direction = max(direction_scores, key=direction_scores.get)

            # Adjust for very weak consensus
            max_score = max(direction_scores.values())
            total_score = sum(direction_scores.values())

            if max_score / total_score < 0.4:  # Less than 40% consensus
                return EnhancedTrendDirection.NEUTRAL

            return dominant_direction

        except Exception as e:
            logger.warning(f"# Warning Error calculating timeframe consensus: {e}")
            return EnhancedTrendDirection.NEUTRAL

    def _calculate_enhanced_trend_strength(self, df: pd.DataFrame,
                                         timeframe_directions: Dict[str, EnhancedTrendDirection]) -> EnhancedTrendStrength:
        """Calculate enhanced trend strength across multiple factors"""
        try:
            latest = df.iloc[-1]

            strength_factors = []

            # MA separation
            ma_separation = abs(latest.get('fast_ma', 0) - latest.get('slow_ma', 0)) / latest.get('close', 1)
            strength_factors.append(min(ma_separation * 10, 1.0))

            # Trend slope
            trend_slope = abs(latest.get('trend_slope', 0)) * 1000
            strength_factors.append(min(trend_slope, 1.0))

            # ATR normalized trend
            atr = latest.get('atr', 0)
            if atr > 0:
                normalized_trend = latest.get('trend_strength', 0) / atr
                strength_factors.append(min(normalized_trend * 5, 1.0))

            # Multi-timeframe agreement
            bullish_count = sum(1 for d in timeframe_directions.values()
                              if d in [EnhancedTrendDirection.BULLISH, EnhancedTrendDirection.STRONG_BULLISH]):
            total_tf = len(timeframe_directions)
            agreement_ratio = max(bullish_count, total_tf - bullish_count) / total_tf
            strength_factors.append(agreement_ratio)

            # Calculate overall strength
            avg_strength = np.mean(strength_factors)

            if avg_strength >= 0.8:
                return EnhancedTrendStrength.VERY_STRONG
            elif avg_strength >= 0.65:
                return EnhancedTrendStrength.STRONG
            elif avg_strength >= 0.5:
                return EnhancedTrendStrength.MODERATE
            elif avg_strength >= 0.35:
                return EnhancedTrendStrength.WEAK
            else:
                return EnhancedTrendStrength.VERY_WEAK

        except Exception as e:
            logger.warning(f"# Warning Error calculating enhanced trend strength: {e}")
            return EnhancedTrendStrength.VERY_WEAK

async def test_enhanced_optimizer():
    """Test the enhanced technical optimizer"""

    optimizer = EnhancedTechnicalOptimizer()

    if not await optimizer.initialize_exchange():
        return

    symbols = ['BTCUSDT', 'ETHUSDT']

    for symbol in symbols:

        signal = await optimizer.analyze_enhanced_trend(symbol)

        if signal:
            print(f"Confluence Score: {signal.confluence_score:.3f}")
            print(f"Patterns Detected: {signal.pattern_signals}")
            print(f"Timeframe Alignment: {[(tf, d.value) for tf, d in signal.timeframe_alignment.items()]}")
        else:

if __name__ == "__main__":
    asyncio.run(test_enhanced_optimizer())
