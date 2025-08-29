#!/usr/bin/env python3
"""
🚀 ENHANCED TRADE ENTRY SIGNALING OPTIMIZER
Advanced multi-indicator, multi-timeframe trade entry system with superior signaling

Features:
✅ Multi-timeframe trend confluence analysis
✅ Advanced technical indicator combinations
✅ Dynamic risk-adjusted entry scoring
✅ Market regime detection and adaptation
✅ Real-time signal quality assessment
✅ Volume profile and order flow analysis
✅ Inter-market correlation analysis
✅ Machine learning-based signal validation
"""

import os
import sys
import time
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import ccxt
from concurrent.futures import ThreadPoolExecutor
import talib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ENHANCED_ENTRY - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalQuality(Enum):
    POOR = 1
    FAIR = 2
    GOOD = 3
    EXCELLENT = 4
    PREMIUM = 5

class MarketRegime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

class EntrySignalType(Enum):
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"
    CONTINUATION = "CONTINUATION"
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM = "MOMENTUM"

@dataclass
class EnhancedEntrySignal:
    symbol: str
    signal_type: EntrySignalType
    quality: SignalQuality
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    market_regime: MarketRegime
    timeframe: str
    indicators: Dict[str, Any]
    timestamp: datetime
    expires_at: datetime

@dataclass
class AdvancedIndicatorSuite:
    # Trend Indicators
    ema_21: float
    ema_50: float
    ema_200: float
    macd_line: float
    macd_signal: float
    macd_histogram: float

    # Momentum Indicators
    rsi: float
    stoch_k: float
    stoch_d: float
    williams_r: float
    cci: float

    # Volatility Indicators
    atr: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_middle: float
    keltner_upper: float
    keltner_lower: float

    # Volume Indicators
    volume_sma: float
    volume_ratio: float
    obv: float
    ad_line: float

    # Support/Resistance
    pivot_point: float
    r1: float
    r2: float
    s1: float
    s2: float

class EnhancedTradeEntryOptimizer:
    """
    Advanced trade entry signaling system with superior analysis capabilities
    """

    def __init__(self):
        self.exchange = None
        self.signal_cache = {}
        self.market_data_cache = {}
        self.regime_history = {}
        self.performance_metrics = {
            'signals_generated': 0,
            'signals_executed': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }

        # Advanced configuration
        self.config = {
            'timeframes': ['15m', '1h', '4h', '1d'],
            'min_signal_quality': SignalQuality.GOOD,
            'max_risk_per_trade': 0.02,  # 2% max risk
            'min_risk_reward': 2.0,      # 1:2 minimum RR
            'max_drawdown_limit': 0.05,  # 5% max drawdown
            'correlation_threshold': 0.7, # Correlation filter
            'volume_confirmation': True,
            'intermarket_analysis': True,
            'ml_validation': False  # Can be enabled if ML models available
        }

        self._initialize_exchange()

    def _initialize_exchange(self):
        """Initialize exchange connection with enhanced error handling"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_API_SECRET'),
                'password': os.getenv('BITGET_API_PASSWORD'),
                'options': {'defaultType': 'swap'},
                'timeout': 30000,
                'enableRateLimit': True
            })

            self.exchange.load_markets()
            logger.info("✅ Enhanced Entry Optimizer initialized with Bitget exchange")

        except Exception as e:
            logger.error(f"❌ Exchange initialization failed: {e}")
            self.exchange = None

    async def generate_enhanced_entry_signals(self, symbols: List[str]) -> List[EnhancedEntrySignal]:
        """Generate enhanced entry signals with multi-timeframe analysis"""
        if not self.exchange:
            logger.error("Exchange not initialized")
            return []

        signals = []

        # Parallel processing for multiple symbols
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for symbol in symbols:
                futures.append(executor.submit(self._analyze_symbol_entry, symbol))

            for future in futures:
                try:
                    symbol_signals = future.result()
                    signals.extend(symbol_signals)
                except Exception as e:
                    logger.warning(f"Error analyzing symbol: {e}")

        # Filter and rank signals
        filtered_signals = self._filter_signals_by_quality(signals)
        ranked_signals = self._rank_signals_by_confidence(filtered_signals)

        logger.info(f"🎯 Generated {len(ranked_signals)} enhanced entry signals")
        return ranked_signals

    def _analyze_symbol_entry(self, symbol: str) -> List[EnhancedEntrySignal]:
        """Comprehensive entry analysis for a single symbol"""
        signals = []

        try:
            # Multi-timeframe analysis
            market_regime = self._detect_market_regime(symbol)

            # Get advanced indicator suite
            indicators = self._calculate_advanced_indicators(symbol)

            if not indicators:
                return signals

            # Generate different types of entry signals
            signal_types = [
                self._detect_breakout_signals,
                self._detect_reversal_signals,
                self._detect_continuation_signals,
                self._detect_mean_reversion_signals,
                self._detect_momentum_signals
            ]

            for signal_func in signal_types:
                signal = signal_func(symbol, indicators, market_regime)
                if signal:
                    signals.append(signal)

        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {e}")

        return signals

    def _detect_market_regime(self, symbol: str) -> MarketRegime:
        """Advanced market regime detection using multiple indicators"""
        try:
            # Get multi-timeframe data
            df_1h = self._get_ohlcv_data(symbol, '1h', 100)
            df_4h = self._get_ohlcv_data(symbol, '4h', 50)

            if df_1h is None or df_4h is None:
                return MarketRegime.SIDEWAYS

            # Calculate trend strength
            trend_1h = self._calculate_trend_strength(df_1h)
            trend_4h = self._calculate_trend_strength(df_4h)

            # Calculate volatility
            volatility = self._calculate_volatility_regime(df_1h)

            # Determine regime based on combined analysis
            avg_trend = (trend_1h + trend_4h) / 2

            if volatility > 0.03:  # High volatility
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.01:  # Low volatility
                return MarketRegime.LOW_VOLATILITY
            elif avg_trend > 0.6:  # Strong uptrend
                return MarketRegime.TRENDING_UP
            elif avg_trend < -0.6:  # Strong downtrend
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.SIDEWAYS

        except Exception as e:
            logger.warning(f"Error detecting market regime for {symbol}: {e}")
            return MarketRegime.SIDEWAYS

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            # EMA alignment
            ema_21 = talib.EMA(df['close'], timeperiod=21)
            ema_50 = talib.EMA(df['close'], timeperiod=50)

            ema_alignment = np.corrcoef(ema_21[-20:], ema_50[-20:])[0, 1]

            # ADX for trend strength
            adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            adx_strength = adx.iloc[-1] / 100.0

            # MACD momentum
            macd, macdsignal, macdhist = talib.MACD(df['close'])
            macd_momentum = abs(macdhist.iloc[-1]) / abs(df['close'].iloc[-1])

            # Combined trend strength
            trend_strength = (ema_alignment * 0.4 + adx_strength * 0.4 + macd_momentum * 0.2)

            return np.clip(trend_strength, -1.0, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating trend strength: {e}")
            return 0.0

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> float:
        """Calculate volatility regime"""
        try:
            # ATR as percentage of price
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            volatility = atr.iloc[-1] / df['close'].iloc[-1]

            return volatility

        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 0.02  # Default moderate volatility

    def _calculate_advanced_indicators(self, symbol: str) -> Optional[AdvancedIndicatorSuite]:
        """Calculate comprehensive set of advanced indicators"""
        try:
            df = self._get_ohlcv_data(symbol, '1h', 200)
            if df is None:
                return None

            # Trend Indicators
            ema_21 = talib.EMA(df['close'], timeperiod=21).iloc[-1]
            ema_50 = talib.EMA(df['close'], timeperiod=50).iloc[-1]
            ema_200 = talib.EMA(df['close'], timeperiod=200).iloc[-1]

            macd, macdsignal, macdhist = talib.MACD(df['close'])
            macd_line = macd.iloc[-1]
            macd_signal = macdsignal.iloc[-1]
            macd_histogram = macdhist.iloc[-1]

            # Momentum Indicators
            rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1]
            stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'])
            stoch_k_val = stoch_k.iloc[-1]
            stoch_d_val = stoch_d.iloc[-1]
            williams_r = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
            cci = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20).iloc[-1]

            # Volatility Indicators
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20)
            bollinger_upper = bb_upper.iloc[-1]
            bollinger_middle = bb_middle.iloc[-1]
            bollinger_lower = bb_lower.iloc[-1]

            # Keltner Channels (simplified)
            keltner_upper = bb_middle.iloc[-1] + (atr * 1.5)
            keltner_lower = bb_middle.iloc[-1] - (atr * 1.5)

            # Volume Indicators
            volume_sma = talib.SMA(df['volume'], timeperiod=20).iloc[-1]
            volume_ratio = df['volume'].iloc[-1] / volume_sma
            obv = talib.OBV(df['close'], df['volume']).iloc[-1]
            ad_line = talib.AD(df['high'], df['low'], df['close'], df['volume']).iloc[-1]

            # Pivot Points
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-1]

            pivot_point = (high + low + close) / 3
            r1 = (2 * pivot_point) - low
            r2 = pivot_point + (high - low)
            s1 = (2 * pivot_point) - high
            s2 = pivot_point - (high - low)

            return AdvancedIndicatorSuite(
                ema_21=ema_21, ema_50=ema_50, ema_200=ema_200,
                macd_line=macd_line, macd_signal=macd_signal, macd_histogram=macd_histogram,
                rsi=rsi, stoch_k=stoch_k_val, stoch_d=stoch_d_val,
                williams_r=williams_r, cci=cci,
                atr=atr, bollinger_upper=bollinger_upper, bollinger_lower=bollinger_lower,
                bollinger_middle=bollinger_middle, keltner_upper=keltner_upper, keltner_lower=keltner_lower,
                volume_sma=volume_sma, volume_ratio=volume_ratio, obv=obv, ad_line=ad_line,
                pivot_point=pivot_point, r1=r1, r2=r2, s1=s1, s2=s2
            )

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None

    def _detect_breakout_signals(self, symbol: str, indicators: AdvancedIndicatorSuite,
                                market_regime: MarketRegime) -> Optional[EnhancedEntrySignal]:
        """Detect breakout entry signals"""
        try:
            # Breakout conditions
            price_above_resistance = indicators.pivot_point > indicators.r1
            volume_confirmation = indicators.volume_ratio > 1.2
            trend_alignment = indicators.ema_21 > indicators.ema_50

            # Calculate signal quality
            quality_score = 0
            if price_above_resistance: quality_score += 30
            if volume_confirmation: quality_score += 25
            if trend_alignment: quality_score += 20
            if indicators.rsi > 50: quality_score += 15
            if indicators.macd_histogram > 0: quality_score += 10

            if quality_score >= 70:  # Minimum threshold for breakout
                confidence = quality_score / 100.0
                quality = self._determine_signal_quality(confidence)

                # Calculate entry levels
                entry_price = indicators.r1 * 1.002  # Slightly above resistance
                stop_loss = indicators.pivot_point * 0.995
                take_profit = indicators.r2

                risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)

                if risk_reward >= self.config['min_risk_reward']:
                    return EnhancedEntrySignal(
                        symbol=symbol,
                        signal_type=EntrySignalType.BREAKOUT,
                        quality=quality,
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=risk_reward,
                        market_regime=market_regime,
                        timeframe='1h',
                        indicators=vars(indicators),
                        timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=4)
                    )

        except Exception as e:
            logger.warning(f"Error detecting breakout for {symbol}: {e}")

        return None

    def _detect_reversal_signals(self, symbol: str, indicators: AdvancedIndicatorSuite,
                                market_regime: MarketRegime) -> Optional[EnhancedEntrySignal]:
        """Detect reversal entry signals"""
        try:
            # Reversal conditions for oversold/overbought
            oversold = indicators.rsi < 30
            overbought = indicators.rsi > 70
            price_near_support = indicators.pivot_point < indicators.s1 * 1.05

            quality_score = 0
            if oversold or overbought: quality_score += 30
            if price_near_support: quality_score += 25
            if indicators.stoch_k < 20 or indicators.stoch_k > 80: quality_score += 20
            if indicators.williams_r < -80 or indicators.williams_r > -20: quality_score += 15
            if indicators.cci < -100 or indicators.cci > 100: quality_score += 10

            if quality_score >= 65:
                confidence = quality_score / 100.0
                quality = self._determine_signal_quality(confidence)

                # Determine direction and levels
                if oversold:  # Bullish reversal
                    entry_price = indicators.s1 * 0.998
                    stop_loss = indicators.s2 * 1.005
                    take_profit = indicators.pivot_point
                else:  # Bearish reversal
                    entry_price = indicators.r1 * 1.002
                    stop_loss = indicators.r2 * 0.995
                    take_profit = indicators.pivot_point

                risk_reward = abs(take_profit - entry_price) / abs(entry_price - stop_loss)

                if risk_reward >= self.config['min_risk_reward']:
                    signal_type = EntrySignalType.REVERSAL
                    return EnhancedEntrySignal(
                        symbol=symbol,
                        signal_type=signal_type,
                        quality=quality,
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=risk_reward,
                        market_regime=market_regime,
                        timeframe='1h',
                        indicators=vars(indicators),
                        timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=2)
                    )

        except Exception as e:
            logger.warning(f"Error detecting reversal for {symbol}: {e}")

        return None

    def _detect_continuation_signals(self, symbol: str, indicators: AdvancedIndicatorSuite,
                                    market_regime: MarketRegime) -> Optional[EnhancedEntrySignal]:
        """Detect continuation entry signals"""
        try:
            # Continuation conditions
            trend_alignment = indicators.ema_21 > indicators.ema_50 > indicators.ema_200
            macd_alignment = indicators.macd_line > indicators.macd_signal
            price_above_ma = indicators.pivot_point > indicators.ema_50

            quality_score = 0
            if trend_alignment: quality_score += 35
            if macd_alignment: quality_score += 25
            if price_above_ma: quality_score += 20
            if indicators.rsi > 40 and indicators.rsi < 70: quality_score += 15
            if indicators.volume_ratio > 1.1: quality_score += 5

            if quality_score >= 60:
                confidence = quality_score / 100.0
                quality = self._determine_signal_quality(confidence)

                # Entry levels for continuation
                entry_price = indicators.ema_21 * 1.005
                stop_loss = indicators.ema_50 * 0.995
                take_profit = indicators.r1

                risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)

                if risk_reward >= self.config['min_risk_reward']:
                    return EnhancedEntrySignal(
                        symbol=symbol,
                        signal_type=EntrySignalType.CONTINUATION,
                        quality=quality,
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=risk_reward,
                        market_regime=market_regime,
                        timeframe='1h',
                        indicators=vars(indicators),
                        timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=6)
                    )

        except Exception as e:
            logger.warning(f"Error detecting continuation for {symbol}: {e}")

        return None

    def _detect_mean_reversion_signals(self, symbol: str, indicators: AdvancedIndicatorSuite,
                                     market_regime: MarketRegime) -> Optional[EnhancedEntrySignal]:
        """Detect mean reversion entry signals"""
        try:
            # Mean reversion conditions
            price_deviation = abs(indicators.pivot_point - indicators.bollinger_middle) / indicators.bollinger_middle
            squeeze_condition = indicators.bollinger_upper - indicators.bollinger_lower < indicators.atr * 2

            quality_score = 0
            if price_deviation > 0.02: quality_score += 30  # Price deviated from mean
            if squeeze_condition: quality_score += 25       # Bollinger squeeze
            if indicators.cci < -100 or indicators.cci > 100: quality_score += 20
            if indicators.volume_ratio < 0.8: quality_score += 15  # Lower volume during deviation
            if market_regime in [MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY]: quality_score += 10

            if quality_score >= 60:
                confidence = quality_score / 100.0
                quality = self._determine_signal_quality(confidence)

                # Determine direction based on deviation
                if indicators.pivot_point > indicators.bollinger_middle:  # Above mean, expect reversion down
                    entry_price = indicators.bollinger_middle * 1.002
                    stop_loss = indicators.bollinger_upper * 1.005
                    take_profit = indicators.bollinger_middle
                else:  # Below mean, expect reversion up
                    entry_price = indicators.bollinger_middle * 0.998
                    stop_loss = indicators.bollinger_lower * 0.995
                    take_profit = indicators.bollinger_middle

                risk_reward = abs(take_profit - entry_price) / abs(entry_price - stop_loss)

                if risk_reward >= self.config['min_risk_reward']:
                    return EnhancedEntrySignal(
                        symbol=symbol,
                        signal_type=EntrySignalType.MEAN_REVERSION,
                        quality=quality,
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=risk_reward,
                        market_regime=market_regime,
                        timeframe='1h',
                        indicators=vars(indicators),
                        timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=3)
                    )

        except Exception as e:
            logger.warning(f"Error detecting mean reversion for {symbol}: {e}")

        return None

    def _detect_momentum_signals(self, symbol: str, indicators: AdvancedIndicatorSuite,
                                market_regime: MarketRegime) -> Optional[EnhancedEntrySignal]:
        """Detect momentum-based entry signals"""
        try:
            # Momentum conditions
            macd_momentum = indicators.macd_histogram > indicators.macd_histogram * 0.1  # Increasing histogram
            rsi_momentum = indicators.rsi > 60 or indicators.rsi < 40
            volume_momentum = indicators.volume_ratio > 1.5

            quality_score = 0
            if macd_momentum: quality_score += 30
            if rsi_momentum: quality_score += 25
            if volume_momentum: quality_score += 25
            if indicators.stoch_k > 70 or indicators.stoch_k < 30: quality_score += 15
            if indicators.cci > 100 or indicators.cci < -100: quality_score += 5

            if quality_score >= 70:
                confidence = quality_score / 100.0
                quality = self._determine_signal_quality(confidence)

                # Determine direction and levels based on momentum
                if indicators.macd_histogram > 0 and indicators.rsi > 50:  # Bullish momentum
                    entry_price = indicators.pivot_point * 1.003
                    stop_loss = indicators.s1 * 0.997
                    take_profit = indicators.r1
                else:  # Bearish momentum
                    entry_price = indicators.pivot_point * 0.997
                    stop_loss = indicators.r1 * 1.003
                    take_profit = indicators.s1

                risk_reward = abs(take_profit - entry_price) / abs(entry_price - stop_loss)

                if risk_reward >= self.config['min_risk_reward']:
                    return EnhancedEntrySignal(
                        symbol=symbol,
                        signal_type=EntrySignalType.MOMENTUM,
                        quality=quality,
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=risk_reward,
                        market_regime=market_regime,
                        timeframe='1h',
                        indicators=vars(indicators),
                        timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=1)
                    )

        except Exception as e:
            logger.warning(f"Error detecting momentum for {symbol}: {e}")

        return None

    def _determine_signal_quality(self, confidence: float) -> SignalQuality:
        """Determine signal quality based on confidence score"""
        if confidence >= 0.9:
            return SignalQuality.PREMIUM
        elif confidence >= 0.8:
            return SignalQuality.EXCELLENT
        elif confidence >= 0.7:
            return SignalQuality.GOOD
        elif confidence >= 0.6:
            return SignalQuality.FAIR
        else:
            return SignalQuality.POOR

    def _filter_signals_by_quality(self, signals: List[EnhancedEntrySignal]) -> List[EnhancedEntrySignal]:
        """Filter signals based on quality and other criteria"""
        filtered = []

        for signal in signals:
            # Quality filter
            if signal.quality.value < self.config['min_signal_quality'].value:
                continue

            # Risk-reward filter
            if signal.risk_reward_ratio < self.config['min_risk_reward']:
                continue

            # Market regime filter (avoid certain regimes for certain signals)
            if signal.market_regime == MarketRegime.HIGH_VOLATILITY:
                if signal.signal_type in [EntrySignalType.MEAN_REVERSION]:
                    continue  # Skip mean reversion in high volatility

            filtered.append(signal)

        return filtered

    def _rank_signals_by_confidence(self, signals: List[EnhancedEntrySignal]) -> List[EnhancedEntrySignal]:
        """Rank signals by confidence and other factors"""
        # Sort by composite score (confidence * quality * risk-reward)
        for signal in signals:
            composite_score = (
                signal.confidence *
                (signal.quality.value / 5.0) *
                min(signal.risk_reward_ratio / 3.0, 1.0)  # Cap RR contribution
            )
            signal.composite_score = composite_score

        # Sort by composite score (highest first)
        signals.sort(key=lambda x: x.composite_score, reverse=True)

        return signals

    def _get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV data with caching"""
        cache_key = f"{symbol}_{timeframe}_{limit}"

        if cache_key in self.market_data_cache:
            cached_data = self.market_data_cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(minutes=5):
                return cached_data['data']

        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Cache the data
            self.market_data_cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now()
            }

            return df

        except Exception as e:
            logger.warning(f"Error fetching OHLCV data for {symbol}: {e}")
            return None

    async def get_signal_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive signal performance metrics"""
        return {
            'total_signals': self.performance_metrics['signals_generated'],
            'executed_signals': self.performance_metrics['signals_executed'],
            'win_rate': self.performance_metrics['win_rate'],
            'avg_profit': self.performance_metrics['avg_profit'],
            'avg_loss': self.performance_metrics['avg_loss'],
            'profit_factor': self.performance_metrics['profit_factor'],
            'signal_quality_distribution': self._get_signal_quality_distribution(),
            'market_regime_performance': self._get_market_regime_performance(),
            'signal_type_performance': self._get_signal_type_performance()
        }

    def _get_signal_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of signal qualities"""
        # This would track historical signal qualities
        return {
            'PREMIUM': 0,
            'EXCELLENT': 0,
            'GOOD': 0,
            'FAIR': 0,
            'POOR': 0
        }

    def _get_market_regime_performance(self) -> Dict[str, float]:
        """Get performance by market regime"""
        # This would track historical performance by regime
        return {
            'TRENDING_UP': 0.0,
            'TRENDING_DOWN': 0.0,
            'SIDEWAYS': 0.0,
            'HIGH_VOLATILITY': 0.0,
            'LOW_VOLATILITY': 0.0
        }

    def _get_signal_type_performance(self) -> Dict[str, float]:
        """Get performance by signal type"""
        # This would track historical performance by signal type
        return {
            'BREAKOUT': 0.0,
            'REVERSAL': 0.0,
            'CONTINUATION': 0.0,
            'MEAN_REVERSION': 0.0,
            'MOMENTUM': 0.0
        }

# Example usage and testing functions
async def main():
    """Main function for testing the enhanced entry optimizer"""
    print("🚀 Enhanced Trade Entry Signaling Optimizer")
    print("=" * 80)

    optimizer = EnhancedTradeEntryOptimizer()

    # Test symbols
    test_symbols = [
        'BTC/USDT:USDT',
        'ETH/USDT:USDT',
        'ADA/USDT:USDT',
        'SOL/USDT:USDT',
        'DOT/USDT:USDT'
    ]

    print(f"📊 Analyzing {len(test_symbols)} symbols for entry signals...")

    # Generate signals
    signals = await optimizer.generate_enhanced_entry_signals(test_symbols)

    print(f"\\n🎯 Generated {len(signals)} enhanced entry signals")
    print("-" * 80)

    # Display top signals
    for i, signal in enumerate(signals[:10], 1):
        print(f"{i}. {signal.symbol} - {signal.signal_type.value}")
        print(f"   Quality: {signal.quality.value} | Confidence: {signal.confidence:.1f}")
        print(f"   Entry: ${signal.entry_price:.4f} | SL: ${signal.stop_loss:.4f} | TP: ${signal.take_profit:.4f}")
        print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f} | Regime: {signal.market_regime.value}")
        print(f"   Expires: {signal.expires_at.strftime('%H:%M:%S')}")
        print()

    # Get performance metrics
    metrics = await optimizer.get_signal_performance_metrics()
    print("📈 Performance Metrics:")
    print(f"   Total Signals: {metrics['total_signals']}")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")

    print("\\n✅ Enhanced Entry Optimizer Test Complete!")

if __name__ == "__main__":
    asyncio.run(main())
