#!/usr/bin/env python3
"""
🔍 MULTI-TIMEFRAME ANALYZER
Advanced multi-timeframe analysis for trend alignment and signal confirmation

This system implements:
✅ Multi-timeframe trend analysis and alignment
✅ Timeframe hierarchy validation
✅ Cross-timeframe signal confirmation
✅ Higher timeframe context for lower timeframe signals
✅ Timeframe-specific entry timing optimization
✅ Trend strength measurement across timeframes
✅ Timeframe convergence and divergence detection
"""

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeframeHierarchy(Enum):
    """Timeframe hierarchy levels"""
    MICRO = "1m"      # Micro timing
    SHORT = "5m"      # Short-term entry
    MEDIUM = "15m"    # Medium-term confirmation
    PRIMARY = "1h"    # Primary trend
    MAJOR = "4h"      # Major trend
    DAILY = "1d"      # Daily context

class TrendDirection(Enum):
    """Trend direction classifications"""
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    WEAK_BULL = "WEAK_BULL"
    SIDEWAYS = "SIDEWAYS"
    WEAK_BEAR = "WEAK_BEAR"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"

class TimeframeAlignment(Enum):
    """Timeframe alignment status"""
    PERFECT = "PERFECT"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    CONFLICTING = "CONFLICTING"

@dataclass
class TimeframeSignal:
    """Signal analysis for a specific timeframe"""
    timeframe: str
    hierarchy_level: TimeframeHierarchy
    trend_direction: TrendDirection
    trend_strength: float
    momentum: float
    volatility: float
    support_resistance: Dict[str, float]
    key_levels: List[float]
    signal_quality: float
    last_updated: datetime

@dataclass
class MultiTimeframeAnalysis:
    """Comprehensive multi-timeframe analysis"""
    symbol: str
    primary_trend: TrendDirection
    trend_strength: float
    alignment_score: float
    alignment_status: TimeframeAlignment
    timeframe_signals: Dict[str, TimeframeSignal]
    convergence_zones: List[Dict[str, Any]]
    optimal_entry_timeframe: str
    higher_timeframe_context: Dict[str, Any]
    risk_adjustment_factor: float
    confidence_score: float
    analysis_timestamp: datetime

@dataclass
class TimeframeConvergence:
    """Timeframe convergence zone"""
    price_level: float
    timeframe_count: int
    supporting_timeframes: List[str]
    convergence_type: str  # 'support', 'resistance', 'pivot'
    strength: float
    last_touched: datetime

class MultiTimeframeAnalyzer:
    """Advanced multi-timeframe analysis system"""

    def __init__(self):
        # Timeframe hierarchy and weights
        self.timeframe_config = {
            '1m': {'hierarchy': TimeframeHierarchy.MICRO, 'weight': 0.1, 'lookback': 120},
            '5m': {'hierarchy': TimeframeHierarchy.SHORT, 'weight': 0.2, 'lookback': 96},
            '15m': {'hierarchy': TimeframeHierarchy.MEDIUM, 'weight': 0.3, 'lookback': 64},
            '1h': {'hierarchy': TimeframeHierarchy.PRIMARY, 'weight': 0.4, 'lookback': 48},
            '4h': {'hierarchy': TimeframeHierarchy.MAJOR, 'weight': 0.5, 'lookback': 42},
            '1d': {'hierarchy': TimeframeHierarchy.DAILY, 'weight': 0.6, 'lookback': 30},
        }

        # Trend strength thresholds
        self.trend_thresholds = {
            'weak': 0.1,
            'moderate': 0.25,
            'strong': 0.4,
            'very_strong': 0.6,
        }

        # Alignment score thresholds
        self.alignment_thresholds = {
            TimeframeAlignment.PERFECT: 0.9,
            TimeframeAlignment.STRONG: 0.7,
            TimeframeAlignment.MODERATE: 0.5,
            TimeframeAlignment.WEAK: 0.3,
            TimeframeAlignment.CONFLICTING: 0.0,
        }

    async def analyze_multi_timeframe(self, symbol: str, ohlcv_data: Dict[str, pd.DataFrame],
                                    current_price: float) -> MultiTimeframeAnalysis:
        """
        Perform comprehensive multi-timeframe analysis

        Args:
            symbol: Trading symbol
            ohlcv_data: OHLCV data organized by timeframe
            current_price: Current market price

        Returns:
            Comprehensive multi-timeframe analysis
        """

        try:
            # Analyze each timeframe individually
            timeframe_signals = {}
            for tf, data in ohlcv_data.items():
                if data is not None and len(data) >= 10:
                    signal = await self._analyze_single_timeframe(symbol, tf, data)
                    timeframe_signals[tf] = signal

            if not timeframe_signals:
                return self._create_empty_analysis(symbol)

            # Calculate overall trend and alignment
            primary_trend = self._calculate_primary_trend(timeframe_signals)
            trend_strength = self._calculate_trend_strength(timeframe_signals)
            alignment_score = self._calculate_alignment_score(timeframe_signals)
            alignment_status = self._assess_alignment_status(alignment_score)

            # Find convergence zones
            convergence_zones = self._identify_convergence_zones(timeframe_signals, current_price)

            # Determine optimal entry timeframe
            optimal_entry_timeframe = self._find_optimal_entry_timeframe(timeframe_signals, alignment_status)

            # Get higher timeframe context
            higher_timeframe_context = self._get_higher_timeframe_context(timeframe_signals)

            # Calculate risk adjustment
            risk_adjustment_factor = self._calculate_risk_adjustment(alignment_score, trend_strength)

            # Overall confidence
            confidence_score = self._calculate_mtf_confidence(
                alignment_score, trend_strength, len(timeframe_signals)
            )

            analysis = MultiTimeframeAnalysis(
                symbol=symbol,
                primary_trend=primary_trend,
                trend_strength=trend_strength,
                alignment_score=alignment_score,
                alignment_status=alignment_status,
                timeframe_signals=timeframe_signals,
                convergence_zones=convergence_zones,
                optimal_entry_timeframe=optimal_entry_timeframe,
                higher_timeframe_context=higher_timeframe_context,
                risk_adjustment_factor=risk_adjustment_factor,
                confidence_score=confidence_score,
                analysis_timestamp=datetime.now()
            )

            logger.info(f"🔍 Multi-Timeframe Analysis for {symbol}:")
            logger.info(f"   Primary Trend: {primary_trend.value}")
            logger.info(f"   Trend Strength: {trend_strength:.2f}")
            logger.info(f"   Alignment: {alignment_status.value} ({alignment_score:.2f})")
            logger.info(f"   Optimal Entry TF: {optimal_entry_timeframe}")
            logger.info(f"   Confidence: {confidence_score:.2f}")

            return analysis

        except Exception as e:
            logger.error(f"Multi-timeframe analysis failed for {symbol}: {e}")
            return self._create_empty_analysis(symbol)

    async def _analyze_single_timeframe(self, symbol: str, timeframe: str, data: pd.DataFrame) -> TimeframeSignal:
        """Analyze a single timeframe for trend and signals"""

        try:
            config = self.timeframe_config.get(timeframe, {'hierarchy': TimeframeHierarchy.MEDIUM, 'weight': 0.3})

            # Calculate trend direction and strength
            trend_direction, trend_strength = self._calculate_trend_direction(data)

            # Calculate momentum
            momentum = self._calculate_momentum(data)

            # Calculate volatility
            volatility = self._calculate_volatility(data)

            # Identify support and resistance levels
            support_resistance = self._identify_support_resistance(data)

            # Find key levels
            key_levels = self._find_key_levels(data, support_resistance)

            # Calculate signal quality
            signal_quality = self._calculate_signal_quality(trend_strength, momentum, volatility)

            signal = TimeframeSignal(
                timeframe=timeframe,
                hierarchy_level=config['hierarchy'],
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                momentum=momentum,
                volatility=volatility,
                support_resistance=support_resistance,
                key_levels=key_levels,
                signal_quality=signal_quality,
                last_updated=datetime.now()
            )

            return signal

        except Exception as e:
            logger.warning(f"Timeframe analysis failed for {symbol} {timeframe}: {e}")
            # Return default signal
            return TimeframeSignal(
                timeframe=timeframe,
                hierarchy_level=self.timeframe_config.get(timeframe, {}).get('hierarchy', TimeframeHierarchy.MEDIUM),
                trend_direction=TrendDirection.SIDEWAYS,
                trend_strength=0.0,
                momentum=0.0,
                volatility=0.1,
                support_resistance={},
                key_levels=[],
                signal_quality=0.0,
                last_updated=datetime.now()
            )

    def _calculate_trend_direction(self, data: pd.DataFrame) -> Tuple[TrendDirection, float]:
        """Calculate trend direction and strength for a timeframe"""

        if len(data) < 20 or 'close' not in data.columns:
            return TrendDirection.SIDEWAYS, 0.0

        try:
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values

            # Calculate various trend indicators
            lookback = min(20, len(closes))

            # Price slope (trend direction)
            x = np.arange(lookback)
            slope, _ = np.polyfit(x, closes[-lookback:], 1)
            slope_pct = slope / closes[-lookback] * lookback  # Annualized slope

            # Moving averages
            ma_short = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
            ma_long = np.mean(closes[-lookback:])

            # Trend strength based on MA separation
            ma_separation = abs(ma_short - ma_long) / ma_long

            # Volatility adjusted trend strength
            volatility = np.std(closes[-lookback:]) / np.mean(closes[-lookback:])
            adjusted_strength = ma_separation / (1 + volatility)

            # Classify trend direction
            if slope_pct > 0.02 and adjusted_strength > self.trend_thresholds['strong']:
                trend_direction = TrendDirection.STRONG_BULL
            elif slope_pct > 0.01 and adjusted_strength > self.trend_thresholds['moderate']:
                trend_direction = TrendDirection.BULL
            elif slope_pct > 0.005:
                trend_direction = TrendDirection.WEAK_BULL
            elif slope_pct < -0.02 and adjusted_strength > self.trend_thresholds['strong']:
                trend_direction = TrendDirection.STRONG_BEAR
            elif slope_pct < -0.01 and adjusted_strength > self.trend_thresholds['moderate']:
                trend_direction = TrendDirection.BEAR
            elif slope_pct < -0.005:
                trend_direction = TrendDirection.WEAK_BEAR
            else:
                trend_direction = TrendDirection.SIDEWAYS

            return trend_direction, min(1.0, adjusted_strength)

        except Exception as e:
            logger.warning(f"Trend calculation error: {e}")
            return TrendDirection.SIDEWAYS, 0.0

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum for a timeframe"""

        if len(data) < 10 or 'close' not in data.columns:
            return 0.0

        try:
            closes = data['close'].values

            # RSI-style momentum calculation
            lookback = min(14, len(closes) - 1)
            gains = np.diff(closes[-lookback:])
            avg_gain = np.mean(gains[gains > 0]) if np.any(gains > 0) else 0
            avg_loss = -np.mean(gains[gains < 0]) if np.any(gains < 0) else 0

            if avg_loss == 0:
                momentum = 1.0
            else:
                rs = avg_gain / avg_loss
                momentum = 1.0 - (1.0 / (1.0 + rs))

            return momentum

        except Exception:
            return 0.5

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility for a timeframe"""

        if len(data) < 5 or 'close' not in data.columns:
            return 0.0

        try:
            closes = data['close'].values
            lookback = min(20, len(closes))

            # Calculate returns
            returns = np.diff(closes[-lookback:]) / closes[-lookback:-1]
            volatility = np.std(returns)

            return min(1.0, volatility * 10)  # Scale for 0-1 range

        except Exception:
            return 0.1

    def _identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identify support and resistance levels"""

        if len(data) < 20:
            return {}

        try:
            highs = data['high'].values
            lows = data['low'].values

            # Find peaks and valleys
            peak_indices, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
            valley_indices, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)

            support_levels = lows[valley_indices]
            resistance_levels = highs[peak_indices]

            # Get most recent levels
            recent_support = np.mean(support_levels[-3:]) if len(support_levels) >= 3 else lows[-1]
            recent_resistance = np.mean(resistance_levels[-3:]) if len(resistance_levels) >= 3 else highs[-1]

            return {
                'support': recent_support,
                'resistance': recent_resistance,
                'pivot': (recent_support + recent_resistance) / 2
            }

        except Exception as e:
            logger.warning(f"Support/resistance identification error: {e}")
            return {}

    def _find_key_levels(self, data: pd.DataFrame, support_resistance: Dict[str, float]) -> List[float]:
        """Find key price levels for the timeframe"""

        key_levels = []

        # Add support and resistance
        if 'support' in support_resistance:
            key_levels.append(support_resistance['support'])
        if 'resistance' in support_resistance:
            key_levels.append(support_resistance['resistance'])
        if 'pivot' in support_resistance:
            key_levels.append(support_resistance['pivot'])

        # Add recent high/low
        if len(data) > 0:
            key_levels.extend([data['high'].max(), data['low'].min()])

        return list(set(key_levels))  # Remove duplicates

    def _calculate_signal_quality(self, trend_strength: float, momentum: float, volatility: float) -> float:
        """Calculate overall signal quality for the timeframe"""

        # Weight the components
        quality = (
            trend_strength * 0.4 +      # Trend strength
            momentum * 0.3 +           # Momentum
            (1 - volatility) * 0.3     # Inverse volatility (lower volatility = higher quality)
        )

        return min(1.0, max(0.0, quality))

    def _calculate_primary_trend(self, timeframe_signals: Dict[str, TimeframeSignal]) -> TrendDirection:
        """Calculate the primary trend across all timeframes"""

        if not timeframe_signals:
            return TrendDirection.SIDEWAYS

        # Weight votes by timeframe hierarchy
        trend_votes = {}

        for tf, signal in timeframe_signals.items():
            weight = self.timeframe_config.get(tf, {}).get('weight', 0.3)
            trend = signal.trend_direction

            if trend not in trend_votes:
                trend_votes[trend] = 0
            trend_votes[trend] += weight * signal.signal_quality

        # Find trend with highest weighted vote
        if trend_votes:
            primary_trend = max(trend_votes.items(), key=lambda x: x[1])[0]
            return primary_trend

        return TrendDirection.SIDEWAYS

    def _calculate_trend_strength(self, timeframe_signals: Dict[str, TimeframeSignal]) -> float:
        """Calculate overall trend strength across timeframes"""

        if not timeframe_signals:
            return 0.0

        # Weighted average of trend strengths
        total_weight = 0
        weighted_strength = 0

        for tf, signal in timeframe_signals.items():
            weight = self.timeframe_config.get(tf, {}).get('weight', 0.3)
            strength = signal.trend_strength * signal.signal_quality

            weighted_strength += strength * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_strength / total_weight

        return 0.0

    def _calculate_alignment_score(self, timeframe_signals: Dict[str, TimeframeSignal]) -> float:
        """Calculate alignment score across timeframes"""

        if len(timeframe_signals) < 2:
            return 0.5

        try:
            # Group trends by bullish/bearish
            bullish_signals = []
            bearish_signals = []

            for tf, signal in timeframe_signals.items():
                weight = self.timeframe_config.get(tf, {}).get('weight', 0.3)
                quality = signal.signal_quality

                trend_value = self._trend_to_numeric(signal.trend_direction)

                if trend_value > 0:
                    bullish_signals.append((weight, quality, trend_value))
                elif trend_value < 0:
                    bearish_signals.append((weight, quality, abs(trend_value)))

            # Calculate alignment
            if bullish_signals and bearish_signals:
                # Conflicting trends - low alignment
                bull_strength = sum(w * q * v for w, q, v in bullish_signals)
                bear_strength = sum(w * q * v for w, q, v in bearish_signals)

                # Alignment is inverse of the conflict
                conflict_ratio = min(bull_strength, bear_strength) / max(bull_strength, bear_strength)
                alignment_score = 1.0 - conflict_ratio
            elif bullish_signals:
                # All bullish - perfect alignment
                alignment_score = sum(w * q * v for w, q, v in bullish_signals) / len(bullish_signals)
            elif bearish_signals:
                # All bearish - perfect alignment
                alignment_score = sum(w * q * v for w, q, v in bearish_signals) / len(bearish_signals)
            else:
                # No clear trends
                alignment_score = 0.0

            return min(1.0, alignment_score)

        except Exception as e:
            logger.warning(f"Alignment calculation error: {e}")
            return 0.0

    def _trend_to_numeric(self, trend: TrendDirection) -> float:
        """Convert trend direction to numeric value"""

        trend_values = {
            TrendDirection.STRONG_BULL: 1.0,
            TrendDirection.BULL: 0.7,
            TrendDirection.WEAK_BULL: 0.4,
            TrendDirection.SIDEWAYS: 0.0,
            TrendDirection.WEAK_BEAR: -0.4,
            TrendDirection.BEAR: -0.7,
            TrendDirection.STRONG_BEAR: -1.0,
        }

        return trend_values.get(trend, 0.0)

    def _assess_alignment_status(self, alignment_score: float) -> TimeframeAlignment:
        """Assess alignment status based on score"""

        for status, threshold in self.alignment_thresholds.items():
            if alignment_score >= threshold:
                return status

        return TimeframeAlignment.CONFLICTING

    def _identify_convergence_zones(self, timeframe_signals: Dict[str, TimeframeSignal],
                                   current_price: float) -> List[Dict[str, Any]]:
        """Identify price zones where multiple timeframes converge"""

        convergence_zones = []

        try:
            # Collect all key levels from all timeframes
            all_levels = []
            for tf, signal in timeframe_signals.items():
                for level in signal.key_levels:
                    all_levels.append({
                        'price': level,
                        'timeframe': tf,
                        'strength': signal.signal_quality
                    })

            # Group levels by price proximity (within 1% of each other)
            price_tolerance = current_price * 0.01
            grouped_levels = self._group_nearby_levels(all_levels, price_tolerance)

            # Create convergence zones
            for group in grouped_levels:
                if len(group) >= 2:  # Need at least 2 timeframes
                    avg_price = np.mean([level['price'] for level in group])
                    timeframe_count = len(group)
                    supporting_timeframes = [level['timeframe'] for level in group]
                    avg_strength = np.mean([level['strength'] for level in group])

                    # Determine convergence type
                    convergence_type = self._classify_convergence_type(avg_price, current_price)

                    zone = {
                        'price_level': avg_price,
                        'timeframe_count': timeframe_count,
                        'supporting_timeframes': supporting_timeframes,
                        'convergence_type': convergence_type,
                        'strength': avg_strength,
                        'last_touched': datetime.now()
                    }

                    convergence_zones.append(zone)

            # Sort by strength and proximity to current price
            convergence_zones.sort(key=lambda x: (
                x['strength'],
                -abs(x['price_level'] - current_price)  # Closer levels first
            ), reverse=True)

        except Exception as e:
            logger.warning(f"Convergence zone identification error: {e}")

        return convergence_zones

    def _group_nearby_levels(self, levels: List[Dict[str, Any]], tolerance: float) -> List[List[Dict[str, Any]]]:
        """Group price levels that are within tolerance of each other"""

        if not levels:
            return []

        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x['price'])

        groups = []
        current_group = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if level['price'] - current_group[-1]['price'] <= tolerance:
                current_group.append(level)
            else:
                if len(current_group) > 0:
                    groups.append(current_group)
                current_group = [level]

        if current_group:
            groups.append(current_group)

        return groups

    def _classify_convergence_type(self, level_price: float, current_price: float) -> str:
        """Classify the type of convergence"""

        if abs(level_price - current_price) / current_price < 0.005:  # Within 0.5%
            return 'pivot'
        elif level_price > current_price:
            return 'resistance'
        else:
            return 'support'

    def _find_optimal_entry_timeframe(self, timeframe_signals: Dict[str, TimeframeSignal],
                                    alignment_status: TimeframeAlignment) -> str:
        """Find the optimal timeframe for entry timing"""

        # Default to 15m if alignment is good
        if alignment_status in [TimeframeAlignment.PERFECT, TimeframeAlignment.STRONG]:
            return '15m'

        # For weaker alignment, use shorter timeframe for precision
        elif alignment_status == TimeframeAlignment.MODERATE:
            return '5m'

        # For weak/conflicting alignment, use micro timeframe
        else:
            return '1m'

    def _get_higher_timeframe_context(self, timeframe_signals: Dict[str, TimeframeSignal]) -> Dict[str, Any]:
        """Get context from higher timeframes"""

        context = {
            'major_trend': TrendDirection.SIDEWAYS,
            'key_levels': [],
            'volatility': 0.0,
            'momentum': 0.0
        }

        try:
            # Focus on higher timeframes (1h, 4h, 1d)
            higher_tf_signals = {}
            for tf in ['1h', '4h', '1d']:
                if tf in timeframe_signals:
                    higher_tf_signals[tf] = timeframe_signals[tf]

            if higher_tf_signals:
                # Get primary trend from higher timeframes
                context['major_trend'] = self._calculate_primary_trend(higher_tf_signals)

                # Collect key levels
                all_levels = []
                for signal in higher_tf_signals.values():
                    all_levels.extend(signal.key_levels)

                context['key_levels'] = list(set(all_levels))

                # Average volatility and momentum
                context['volatility'] = np.mean([s.volatility for s in higher_tf_signals.values()])
                context['momentum'] = np.mean([s.momentum for s in higher_tf_signals.values()])

        except Exception as e:
            logger.warning(f"Higher timeframe context error: {e}")

        return context

    def _calculate_risk_adjustment(self, alignment_score: float, trend_strength: float) -> float:
        """Calculate risk adjustment factor based on alignment and trend strength"""

        # Lower alignment = higher risk = higher adjustment factor
        alignment_factor = 2.0 - alignment_score

        # Weaker trend = higher risk
        trend_factor = 2.0 - trend_strength

        # Combine factors (weighted average)
        risk_factor = (alignment_factor * 0.6 + trend_factor * 0.4)

        # Cap between 1.0 (normal risk) and 2.0 (double risk)
        return min(2.0, max(1.0, risk_factor))

    def _calculate_mtf_confidence(self, alignment_score: float, trend_strength: float,
                                timeframe_count: int) -> float:
        """Calculate overall multi-timeframe confidence"""

        # Base confidence from alignment and strength
        confidence = (alignment_score + trend_strength) / 2

        # Bonus for more timeframes analyzed
        timeframe_bonus = min(0.2, timeframe_count * 0.05)

        return min(1.0, confidence + timeframe_bonus)

    def _create_empty_analysis(self, symbol: str) -> MultiTimeframeAnalysis:
        """Create empty analysis when data is insufficient"""

        return MultiTimeframeAnalysis(
            symbol=symbol,
            primary_trend=TrendDirection.SIDEWAYS,
            trend_strength=0.0,
            alignment_score=0.0,
            alignment_status=TimeframeAlignment.CONFLICTING,
            timeframe_signals={},
            convergence_zones=[],
            optimal_entry_timeframe='15m',
            higher_timeframe_context={},
            risk_adjustment_factor=1.5,
            confidence_score=0.0,
            analysis_timestamp=datetime.now()
        )

    async def validate_entry_with_mtf(self, symbol: str, side: str, entry_price: float,
                                    mtf_analysis: MultiTimeframeAnalysis) -> Dict[str, Any]:
        """
        Validate entry signal using multi-timeframe analysis

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            entry_price: Proposed entry price
            mtf_analysis: Multi-timeframe analysis result

        Returns:
            Validation result with recommendations
        """

        validation = {
            'validated': False,
            'confidence': 0.0,
            'recommendations': [],
            'risk_adjustments': [],
            'optimal_timing': None
        }

        try:
            # Check trend alignment
            trend_alignment = self._check_trend_alignment(side, mtf_analysis.primary_trend)

            # Check timeframe alignment
            tf_alignment_score = mtf_analysis.alignment_score

            # Check convergence zones
            convergence_validation = self._validate_with_convergence_zones(
                entry_price, mtf_analysis.convergence_zones
            )

            # Check higher timeframe context
            context_validation = self._validate_higher_timeframe_context(
                entry_price, mtf_analysis.higher_timeframe_context
            )

            # Calculate overall validation score
            validation_score = (
                trend_alignment * 0.3 +
                tf_alignment_score * 0.3 +
                convergence_validation['score'] * 0.2 +
                context_validation['score'] * 0.2
            )

            validation['validated'] = validation_score >= 0.6
            validation['confidence'] = validation_score

            # Generate recommendations
            validation['recommendations'] = (
                trend_alignment['reasons'] +
                convergence_validation['reasons'] +
                context_validation['reasons']
            )

            # Risk adjustments
            validation['risk_adjustments'] = self._generate_risk_adjustments(
                mtf_analysis.risk_adjustment_factor, tf_alignment_score
            )

            # Optimal timing
            validation['optimal_timing'] = mtf_analysis.optimal_entry_timeframe

        except Exception as e:
            logger.error(f"MTF entry validation failed for {symbol}: {e}")
            validation['recommendations'] = [f"Validation error: {e}"]

        return validation

    def _check_trend_alignment(self, side: str, primary_trend: TrendDirection) -> Dict[str, Any]:
        """Check if entry side aligns with primary trend"""

        # Define favorable trends for each side
        favorable_trends = {
            'buy': [TrendDirection.STRONG_BULL, TrendDirection.BULL, TrendDirection.WEAK_BULL],
            'sell': [TrendDirection.STRONG_BEAR, TrendDirection.BEAR, TrendDirection.WEAK_BEAR]
        }

        favorable = primary_trend in favorable_trends.get(side, [])

        if favorable:
            score = 0.9
            reasons = [f"Entry aligns with {primary_trend.value.replace('_', ' ').lower()} trend"]
        else:
            score = 0.3
            reasons = [f"Entry conflicts with {primary_trend.value.replace('_', ' ').lower()} trend"]

        return {'score': score, 'reasons': reasons}

    def _validate_with_convergence_zones(self, entry_price: float,
                                       convergence_zones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate entry against convergence zones"""

        if not convergence_zones:
            return {'score': 0.5, 'reasons': ['No convergence zones identified']}

        # Find nearest convergence zone
        nearest_zone = min(convergence_zones,
                          key=lambda x: abs(x['price_level'] - entry_price))

        distance_pct = abs(nearest_zone['price_level'] - entry_price) / entry_price

        if distance_pct < 0.005:  # Within 0.5%
            if nearest_zone['convergence_type'] == 'support':
                score = 0.8
                reasons = [f"Entry near strong support convergence ({nearest_zone['timeframe_count']} TFs)"]
            elif nearest_zone['convergence_type'] == 'resistance':
                score = 0.3
                reasons = [f"Entry near resistance convergence - use caution"]
            else:
                score = 0.7
                reasons = [f"Entry at pivot convergence point"]
        else:
            score = 0.5
            reasons = [f"Entry {distance_pct*100:.1f}% from nearest convergence zone"]

        return {'score': score, 'reasons': reasons}

    def _validate_higher_timeframe_context(self, entry_price: float,
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entry against higher timeframe context"""

        score = 0.5
        reasons = []

        try:
            # Check proximity to key levels
            key_levels = context.get('key_levels', [])
            if key_levels:
                nearest_level = min(key_levels, key=lambda x: abs(x - entry_price))
                distance_pct = abs(nearest_level - entry_price) / entry_price

                if distance_pct < 0.01:  # Within 1%
                    score = 0.8
                    reasons.append(f"Entry near higher TF key level ({distance_pct*100:.1f}% distance)")
                else:
                    reasons.append(f"Entry {distance_pct*100:.1f}% from higher TF key levels")

            # Check volatility
            volatility = context.get('volatility', 0.1)
            if volatility > 0.2:
                reasons.append("Higher TF volatility - consider wider stops")
            elif volatility < 0.05:
                reasons.append("Higher TF low volatility - favorable conditions")

            # Check momentum
            momentum = context.get('momentum', 0.5)
            if momentum > 0.7:
                reasons.append("Strong higher TF momentum supports entry")
            elif momentum < 0.3:
                reasons.append("Weak higher TF momentum - monitor closely")

        except Exception as e:
            reasons.append(f"Context validation error: {e}")

        return {'score': score, 'reasons': reasons}

    def _generate_risk_adjustments(self, risk_factor: float, alignment_score: float) -> List[str]:
        """Generate risk adjustment recommendations"""

        adjustments = []

        if risk_factor > 1.5:
            adjustments.append(".1f")
            adjustments.append("Use wider stop loss")
            adjustments.append("Reduce position size")

        if alignment_score < 0.5:
            adjustments.append("Consider reducing leverage")
            adjustments.append("Monitor closely for trend changes")

        if risk_factor > 1.2:
            adjustments.append("Consider partial exits earlier")

        return adjustments

# Global multi-timeframe analyzer instance
mtf_analyzer = MultiTimeframeAnalyzer()
