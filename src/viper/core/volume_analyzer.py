#!/usr/bin/env python3
"""
📊 ADVANCED VOLUME ANALYZER
Comprehensive volume analysis for signal validation and market strength assessment

This system implements:
✅ Volume spike detection and analysis
✅ Volume profile analysis for support/resistance
✅ On-Balance Volume (OBV) divergence detection
✅ Volume confirmation for breakouts and reversals
✅ Volume-based trend strength measurement
✅ Accumulation/Distribution analysis
✅ Volume-weighted price calculations
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
from scipy.signal import find_peaks
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolumeSignalType(Enum):
    """Types of volume signals"""
    SPIKE = "SPIKE"
    DIVERGENCE = "DIVERGENCE"
    CONFIRMATION = "CONFIRMATION"
    DISTRIBUTION = "DISTRIBUTION"
    ACCUMULATION = "ACCUMULATION"
    BREAKOUT_VOLUME = "BREAKOUT_VOLUME"
    REVERSAL_VOLUME = "REVERSAL_VOLUME"

class VolumeStrength(Enum):
    """Volume strength levels"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    EXTREME = 5

@dataclass
class VolumeSpike:
    """Represents a volume spike event"""
    timestamp: datetime
    price: float
    volume: float
    spike_ratio: float
    strength: VolumeStrength
    duration: int  # candles
    context: str
    significance: float

@dataclass
class VolumeAnalysis:
    """Comprehensive volume analysis result"""
    symbol: str
    timeframe: str
    volume_trend: str
    average_volume: float
    current_volume_ratio: float
    volume_strength: VolumeStrength
    obv_signal: str
    volume_divergence: bool
    accumulation_distribution: float
    volume_profile_zones: Dict[str, float]
    key_signals: List[str]
    confidence_score: float
    analyzed_at: datetime

@dataclass
class VolumeConfirmation:
    """Volume confirmation for price movements"""
    confirmed: bool
    strength: float
    signal_type: VolumeSignalType
    reasons: List[str]
    confidence: float

class AdvancedVolumeAnalyzer:
    """Advanced volume analysis system for trading signals"""

    def __init__(self):
        self.volume_lookback_periods = {
            '1m': 60,   # 1 hour
            '5m': 48,   # 4 hours
            '15m': 32,  # 8 hours
            '1h': 24,   # 24 hours
            '4h': 21,   # 3.5 days
        }

        self.spike_thresholds = {
            VolumeStrength.VERY_LOW: 1.2,
            VolumeStrength.LOW: 1.5,
            VolumeStrength.MODERATE: 2.0,
            VolumeStrength.HIGH: 3.0,
            VolumeStrength.EXTREME: 5.0,
        }

        # Moving averages for volume trend analysis
        self.volume_ma_periods = [10, 20, 50]

        # Volume profile analysis parameters
        self.price_bins = 20  # Number of price bins for volume profile

    async def analyze_volume_comprehensive(self, symbol: str, ohlcv_data: Dict[str, pd.DataFrame],
                                        current_price: float, current_volume: float) -> VolumeAnalysis:
        """
        Perform comprehensive volume analysis

        Args:
            symbol: Trading symbol
            ohlcv_data: OHLCV data by timeframe
            current_price: Current price
            current_volume: Current volume

        Returns:
            Comprehensive volume analysis
        """

        try:
            # Use primary timeframe for main analysis (1h if available, else first available)
            primary_tf = '1h' if '1h' in ohlcv_data else list(ohlcv_data.keys())[0]
            data = ohlcv_data[primary_tf]

            if data is None or len(data) < 10:
                return self._create_empty_analysis(symbol, primary_tf)

            # Calculate volume metrics
            volume_trend = self._analyze_volume_trend(data)
            average_volume = self._calculate_average_volume(data)
            current_volume_ratio = current_volume / average_volume if average_volume > 0 else 1.0
            volume_strength = self._assess_volume_strength(current_volume_ratio)

            # OBV analysis
            obv_signal = self._analyze_obv(data)

            # Volume divergence
            volume_divergence = self._detect_volume_divergence(data)

            # Accumulation/Distribution
            accumulation_distribution = self._calculate_accumulation_distribution(data)

            # Volume profile zones
            volume_profile_zones = self._calculate_volume_profile(data)

            # Key signals
            key_signals = self._identify_key_volume_signals(
                data, current_price, current_volume, volume_trend
            )

            # Overall confidence
            confidence_score = self._calculate_volume_confidence(
                volume_strength, obv_signal, volume_divergence, current_volume_ratio
            )

            analysis = VolumeAnalysis(
                symbol=symbol,
                timeframe=primary_tf,
                volume_trend=volume_trend,
                average_volume=average_volume,
                current_volume_ratio=current_volume_ratio,
                volume_strength=volume_strength,
                obv_signal=obv_signal,
                volume_divergence=volume_divergence,
                accumulation_distribution=accumulation_distribution,
                volume_profile_zones=volume_profile_zones,
                key_signals=key_signals,
                confidence_score=confidence_score,
                analyzed_at=datetime.now()
            )

            logger.info(f"📊 Volume Analysis for {symbol} ({primary_tf}):")
            logger.info(f"   Trend: {volume_trend}")
            logger.info(f"   Strength: {volume_strength.name}")
            logger.info(f"   Current/Avg: {current_volume_ratio:.2f}")
            logger.info(f"   OBV Signal: {obv_signal}")

            return analysis

        except Exception as e:
            logger.error(f"Volume analysis failed for {symbol}: {e}")
            return self._create_empty_analysis(symbol, '1h')

    def _analyze_volume_trend(self, data: pd.DataFrame) -> str:
        """Analyze volume trend using moving averages"""

        if 'volume' not in data.columns or len(data) < 50:
            return "insufficient_data"

        volumes = data['volume'].values

        # Calculate moving averages
        ma10 = np.convolve(volumes, np.ones(10)/10, mode='valid')
        ma20 = np.convolve(volumes, np.ones(20)/20, mode='valid')

        if len(ma10) < 2 or len(ma20) < 2:
            return "insufficient_data"

        # Compare recent MAs
        recent_ma10 = np.mean(ma10[-5:])  # Last 5 values of 10-period MA
        recent_ma20 = np.mean(ma20[-5:])  # Last 5 values of 20-period MA

        if recent_ma10 > recent_ma20 * 1.05:
            return "increasing"
        elif recent_ma10 < recent_ma20 * 0.95:
            return "decreasing"
        else:
            return "stable"

    def _calculate_average_volume(self, data: pd.DataFrame) -> float:
        """Calculate average volume over lookback period"""

        if 'volume' not in data.columns:
            return 0.0

        lookback = min(20, len(data))
        recent_volumes = data['volume'].tail(lookback)

        return recent_volumes.mean()

    def _assess_volume_strength(self, volume_ratio: float) -> VolumeStrength:
        """Assess volume strength based on ratio to average"""

        for strength in reversed(VolumeStrength):
            if volume_ratio >= self.spike_thresholds[strength]:
                return strength

        return VolumeStrength.VERY_LOW

    def _analyze_obv(self, data: pd.DataFrame) -> str:
        """Analyze On-Balance Volume for trend confirmation"""

        if len(data) < 20 or 'close' not in data.columns or 'volume' not in data.columns:
            return "insufficient_data"

        try:
            # Calculate OBV
            obv = self._calculate_obv(data)

            if len(obv) < 10:
                return "insufficient_data"

            # Simple trend analysis on OBV
            recent_obv = obv[-10:]
            obv_trend = np.polyfit(range(len(recent_obv)), recent_obv, 1)[0]

            prices = data['close'].values[-10:]
            price_trend = np.polyfit(range(len(prices)), prices, 1)[0]

            # Check for divergence
            if obv_trend > 0 and price_trend > 0:
                return "bullish_confirmation"
            elif obv_trend < 0 and price_trend < 0:
                return "bearish_confirmation"
            elif obv_trend > 0 and price_trend < 0:
                return "bullish_divergence"
            elif obv_trend < 0 and price_trend > 0:
                return "bearish_divergence"
            else:
                return "neutral"

        except Exception as e:
            logger.warning(f"OBV analysis error: {e}")
            return "calculation_error"

    def _calculate_obv(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate On-Balance Volume"""

        closes = data['close'].values
        volumes = data['volume'].values

        obv = np.zeros(len(closes))
        obv[0] = volumes[0]

        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]

        return obv

    def _detect_volume_divergence(self, data: pd.DataFrame) -> bool:
        """Detect volume divergence patterns"""

        if len(data) < 20:
            return False

        try:
            # Simple divergence detection
            prices = data['close'].values
            volumes = data['volume'].values

            # Find recent peaks in price and volume
            price_peaks, _ = find_peaks(prices, distance=5)
            volume_peaks, _ = find_peaks(volumes, distance=5)

            if len(price_peaks) >= 2 and len(volume_peaks) >= 2:
                # Check if recent price peak has lower volume than previous
                recent_price_peak_idx = price_peaks[-1]
                recent_volume_at_peak = volumes[recent_price_peak_idx]

                # Find previous price peak
                prev_price_peaks = price_peaks[price_peaks < recent_price_peak_idx]
                if len(prev_price_peaks) > 0:
                    prev_price_peak_idx = prev_price_peaks[-1]
                    prev_volume_at_peak = volumes[prev_price_peak_idx]

                    # Divergence if recent volume is significantly lower
                    if recent_volume_at_peak < prev_volume_at_peak * 0.7:
                        return True

            return False

        except Exception as e:
            logger.warning(f"Volume divergence detection error: {e}")
            return False

    def _calculate_accumulation_distribution(self, data: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution Line"""

        if len(data) < 5 or not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
            return 0.0

        try:
            # Calculate Money Flow Multiplier
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values

            # Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = np.nan_to_num(mfm, nan=0)  # Handle division by zero

            # Money Flow Volume
            mfv = mfm * volume

            # Accumulation/Distribution Line
            adl = np.cumsum(mfv)

            # Return recent trend (slope of last 10 periods)
            if len(adl) >= 10:
                recent_adl = adl[-10:]
                slope = np.polyfit(range(len(recent_adl)), recent_adl, 1)[0]
                return slope
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Accumulation/Distribution calculation error: {e}")
            return 0.0

    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile to identify key price levels"""

        if len(data) < 20 or not all(col in data.columns for col in ['high', 'low', 'volume']):
            return {}

        try:
            # Create price bins
            price_min = data['low'].min()
            price_max = data['high'].max()
            bins = np.linspace(price_min, price_max, self.price_bins + 1)

            # Calculate volume per price bin
            volume_profile = np.zeros(self.price_bins)

            for i in range(len(data)):
                high = data['high'].iloc[i]
                low = data['low'].iloc[i]
                volume = data['volume'].iloc[i]

                # Find bins that this candle spans
                bin_indices = np.where((bins[:-1] <= high) & (bins[1:] >= low))[0]

                if len(bin_indices) > 0:
                    # Distribute volume across spanned bins
                    volume_per_bin = volume / len(bin_indices)
                    for bin_idx in bin_indices:
                        if bin_idx < len(volume_profile):
                            volume_profile[bin_idx] += volume_per_bin

            # Find high volume zones
            avg_volume_per_bin = np.mean(volume_profile)
            high_volume_threshold = avg_volume_per_bin * 1.5

            high_volume_bins = np.where(volume_profile > high_volume_threshold)[0]

            zones = {}
            for bin_idx in high_volume_bins:
                bin_price = (bins[bin_idx] + bins[bin_idx + 1]) / 2
                zones[f"zone_{bin_idx}"] = bin_price

            return zones

        except Exception as e:
            logger.warning(f"Volume profile calculation error: {e}")
            return {}

    def _identify_key_volume_signals(self, data: pd.DataFrame, current_price: float,
                                   current_volume: float, volume_trend: str) -> List[str]:
        """Identify key volume signals"""

        signals = []

        try:
            # Volume spike signal
            avg_volume = self._calculate_average_volume(data)
            if current_volume > avg_volume * 2.0:
                signals.append("Volume Spike Detected")

            # Volume trend signals
            if volume_trend == "increasing":
                signals.append("Volume Trend Increasing")
            elif volume_trend == "decreasing":
                signals.append("Volume Trend Decreasing")

            # Price-volume relationship
            recent_prices = data['close'].values[-5:]
            recent_volumes = data['volume'].values[-5:]

            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            volume_trend_val = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]

            if price_trend > 0 and volume_trend_val > 0:
                signals.append("Bullish Volume Confirmation")
            elif price_trend < 0 and volume_trend_val > 0:
                signals.append("Bearish Volume Divergence")

        except Exception as e:
            logger.warning(f"Key volume signals identification error: {e}")

        return signals

    def _calculate_volume_confidence(self, volume_strength: VolumeStrength, obv_signal: str,
                                   volume_divergence: bool, volume_ratio: float) -> float:
        """Calculate overall volume analysis confidence"""

        confidence = 0.5  # Base confidence

        # Volume strength contribution
        strength_scores = {
            VolumeStrength.VERY_LOW: 0.0,
            VolumeStrength.LOW: 0.2,
            VolumeStrength.MODERATE: 0.4,
            VolumeStrength.HIGH: 0.7,
            VolumeStrength.EXTREME: 0.9,
        }
        confidence += strength_scores.get(volume_strength, 0.0)

        # OBV signal contribution
        if "confirmation" in obv_signal:
            confidence += 0.2
        elif "divergence" in obv_signal:
            confidence += 0.1

        # Volume divergence penalty
        if volume_divergence:
            confidence -= 0.1

        # Volume ratio bonus
        if volume_ratio > 1.5:
            confidence += 0.1

        return min(1.0, max(0.0, confidence))

    def _create_empty_analysis(self, symbol: str, timeframe: str) -> VolumeAnalysis:
        """Create empty volume analysis when data is insufficient"""

        return VolumeAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            volume_trend="insufficient_data",
            average_volume=0.0,
            current_volume_ratio=1.0,
            volume_strength=VolumeStrength.VERY_LOW,
            obv_signal="insufficient_data",
            volume_divergence=False,
            accumulation_distribution=0.0,
            volume_profile_zones={},
            key_signals=["Insufficient data for analysis"],
            confidence_score=0.0,
            analyzed_at=datetime.now()
        )

    async def confirm_signal_with_volume(self, symbol: str, signal_type: str,
                                       ohlcv_data: Dict[str, pd.DataFrame],
                                       current_price: float, current_volume: float) -> VolumeConfirmation:
        """
        Confirm trading signals using volume analysis

        Args:
            symbol: Trading symbol
            signal_type: Type of signal to confirm ('breakout', 'reversal', 'continuation')
            ohlcv_data: OHLCV data by timeframe
            current_price: Current price
            current_volume: Current volume

        Returns:
            Volume confirmation result
        """

        try:
            # Get comprehensive volume analysis
            volume_analysis = await self.analyze_volume_comprehensive(
                symbol, ohlcv_data, current_price, current_volume
            )

            confirmed = False
            strength = 0.0
            reasons = []

            # Signal-specific confirmation logic
            if signal_type == 'breakout':
                confirmed, strength, reasons = self._confirm_breakout_signal(volume_analysis)
            elif signal_type == 'reversal':
                confirmed, strength, reasons = self._confirm_reversal_signal(volume_analysis)
            elif signal_type == 'continuation':
                confirmed, strength, reasons = self._confirm_continuation_signal(volume_analysis)
            else:
                confirmed, strength, reasons = self._confirm_general_signal(volume_analysis)

            # Determine signal type
            signal_type_enum = self._determine_volume_signal_type(
                signal_type, volume_analysis, confirmed
            )

            confidence = volume_analysis.confidence_score * strength

            return VolumeConfirmation(
                confirmed=confirmed,
                strength=strength,
                signal_type=signal_type_enum,
                reasons=reasons,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Volume signal confirmation failed for {symbol}: {e}")
            return VolumeConfirmation(
                confirmed=False,
                strength=0.0,
                signal_type=VolumeSignalType.CONFIRMATION,
                reasons=[f"Confirmation error: {e}"],
                confidence=0.0
            )

    def _confirm_breakout_signal(self, volume_analysis: VolumeAnalysis) -> Tuple[bool, float, List[str]]:
        """Confirm breakout signals with volume"""

        reasons = []

        # High volume is critical for breakouts
        if volume_analysis.volume_strength in [VolumeStrength.HIGH, VolumeStrength.EXTREME]:
            reasons.append("High breakout volume confirmed")
            strength = 0.9
        elif volume_analysis.volume_strength == VolumeStrength.MODERATE:
            reasons.append("Moderate breakout volume")
            strength = 0.6
        else:
            reasons.append("Insufficient volume for breakout")
            return False, 0.3, reasons

        # Volume trend should be increasing
        if volume_analysis.volume_trend == "increasing":
            reasons.append("Volume trend supports breakout")
            strength += 0.1

        # OBV confirmation
        if "bullish" in volume_analysis.obv_signal:
            reasons.append("OBV confirms breakout")
            strength += 0.1

        return True, min(1.0, strength), reasons

    def _confirm_reversal_signal(self, volume_analysis: VolumeAnalysis) -> Tuple[bool, float, List[str]]:
        """Confirm reversal signals with volume"""

        reasons = []

        # Volume spike often accompanies reversals
        if volume_analysis.volume_strength in [VolumeStrength.MODERATE, VolumeStrength.HIGH, VolumeStrength.EXTREME]:
            reasons.append("Volume spike supports reversal")
            strength = 0.8
        else:
            reasons.append("Low volume weakens reversal signal")
            strength = 0.4

        # Look for volume divergence (higher volume on reversal)
        if volume_analysis.volume_divergence:
            reasons.append("Volume divergence pattern detected")
            strength += 0.1

        # OBV divergence is a strong reversal signal
        if "divergence" in volume_analysis.obv_signal:
            reasons.append("OBV divergence confirms reversal")
            strength += 0.2

        confirmed = strength >= 0.6
        return confirmed, strength, reasons

    def _confirm_continuation_signal(self, volume_analysis: VolumeAnalysis) -> Tuple[bool, float, List[str]]:
        """Confirm continuation signals with volume"""

        reasons = []

        # Moderate to high volume supports continuation
        if volume_analysis.volume_strength in [VolumeStrength.MODERATE, VolumeStrength.HIGH]:
            reasons.append("Volume supports trend continuation")
            strength = 0.7
        elif volume_analysis.volume_strength == VolumeStrength.EXTREME:
            reasons.append("Very high volume may indicate exhaustion")
            strength = 0.5
        else:
            reasons.append("Low volume weakens continuation signal")
            strength = 0.4

        # Consistent volume trend
        if volume_analysis.volume_trend == "stable":
            reasons.append("Stable volume supports continuation")
            strength += 0.1

        # OBV confirmation
        if "confirmation" in volume_analysis.obv_signal:
            reasons.append("OBV confirms continuation")
            strength += 0.1

        confirmed = strength >= 0.5
        return confirmed, strength, reasons

    def _confirm_general_signal(self, volume_analysis: VolumeAnalysis) -> Tuple[bool, float, List[str]]:
        """General signal confirmation"""

        reasons = []

        # Basic volume confirmation
        if volume_analysis.volume_strength.value >= VolumeStrength.MODERATE.value:
            reasons.append("Volume supports signal")
            strength = 0.6
        else:
            reasons.append("Volume is weak")
            strength = 0.3

        # Overall confidence
        strength *= volume_analysis.confidence_score

        confirmed = strength >= 0.4
        return confirmed, strength, reasons

    def _determine_volume_signal_type(self, original_signal_type: str,
                                    volume_analysis: VolumeAnalysis, confirmed: bool) -> VolumeSignalType:
        """Determine the specific volume signal type"""

        if not confirmed:
            return VolumeSignalType.CONFIRMATION

        if original_signal_type == 'breakout':
            return VolumeSignalType.BREAKOUT_VOLUME
        elif original_signal_type == 'reversal':
            return VolumeSignalType.REVERSAL_VOLUME
        elif volume_analysis.volume_divergence:
            return VolumeSignalType.DIVERGENCE
        elif volume_analysis.volume_strength.value >= VolumeStrength.HIGH.value:
            return VolumeSignalType.SPIKE
        else:
            return VolumeSignalType.CONFIRMATION

    async def detect_volume_spikes(self, symbol: str, ohlcv_data: Dict[str, pd.DataFrame]) -> List[VolumeSpike]:
        """
        Detect significant volume spikes across timeframes

        Args:
            symbol: Trading symbol
            ohlcv_data: OHLCV data by timeframe

        Returns:
            List of detected volume spikes
        """

        spikes = []

        try:
            for timeframe, data in ohlcv_data.items():
                if data is None or len(data) < 20:
                    continue

                timeframe_spikes = self._detect_spikes_in_timeframe(symbol, timeframe, data)
                spikes.extend(timeframe_spikes)

            # Sort by significance and recency
            spikes.sort(key=lambda x: (x.significance, x.timestamp), reverse=True)

        except Exception as e:
            logger.error(f"Volume spike detection failed for {symbol}: {e}")

        return spikes

    def _detect_spikes_in_timeframe(self, symbol: str, timeframe: str, data: pd.DataFrame) -> List[VolumeSpike]:
        """Detect volume spikes within a single timeframe"""

        spikes = []

        try:
            if 'volume' not in data.columns:
                return spikes

            volumes = data['volume'].values
            prices = data['close'].values

            # Calculate dynamic threshold based on recent volatility
            lookback = min(20, len(volumes) - 1)
            avg_volume = np.mean(volumes[-lookback:])
            std_volume = np.std(volumes[-lookback:])

            # Spike threshold: mean + 2*std
            spike_threshold = avg_volume + 2 * std_volume

            # Find spikes
            for i in range(len(volumes)):
                if volumes[i] > spike_threshold:
                    spike_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
                    strength = self._assess_volume_strength(spike_ratio)

                    # Calculate duration (consecutive candles above threshold)
                    duration = 1
                    for j in range(i + 1, min(i + 5, len(volumes))):
                        if volumes[j] > spike_threshold:
                            duration += 1
                        else:
                            break

                    # Calculate significance
                    significance = (
                        spike_ratio * 0.4 +
                        strength.value / 5 * 0.3 +
                        duration / 5 * 0.3  # Max 5 candles
                    )

                    spike = VolumeSpike(
                        timestamp=data.index[i] if hasattr(data, 'index') else datetime.now(),
                        price=prices[i],
                        volume=volumes[i],
                        spike_ratio=spike_ratio,
                        strength=strength,
                        duration=duration,
                        context=f"{timeframe}_spike",
                        significance=min(1.0, significance)
                    )

                    spikes.append(spike)

        except Exception as e:
            logger.warning(f"Spike detection error for {symbol} {timeframe}: {e}")

        return spikes

# Global volume analyzer instance
volume_analyzer = AdvancedVolumeAnalyzer()
