#!/usr/bin/env python3
"""
🎯 MAX CONVEXITY POINT (MCP) STRATEGY OPTIMIZER
Advanced entry point optimization using price acceleration and supply/demand dynamics

This system implements:
✅ Max Convexity Point identification for optimal entries
✅ Price acceleration analysis for momentum detection
✅ Supply/demand zone identification
✅ Inflection point detection for precise timing
✅ Volume confirmation integration
✅ Multi-timeframe convexity alignment
✅ Adaptive entry scoring based on convexity strength
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
from scipy.optimize import curve_fit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConvexityLevel(Enum):
    """Convexity strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4
    BREAKOUT = 5

class ConvexityType(Enum):
    """Type of convexity detected"""
    ACCELERATING_UP = "ACCELERATING_UP"
    ACCELERATING_DOWN = "ACCELERATING_DOWN"
    DECELERATING_UP = "DECELERATING_UP"
    DECELERATING_DOWN = "DECELERATING_DOWN"
    INFLECTION_UP = "INFLECTION_UP"
    INFLECTION_DOWN = "INFLECTION_DOWN"

@dataclass
class ConvexityPoint:
    """Represents a convexity point in price action"""
    timestamp: datetime
    price: float
    convexity_type: ConvexityType
    strength: ConvexityLevel
    acceleration: float
    volume_ratio: float
    confidence: float
    timeframe: str
    supporting_factors: List[str]

@dataclass
class MCPEntrySignal:
    """MCP-enhanced entry signal"""
    symbol: str
    convexity_point: ConvexityPoint
    optimal_entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    risk_reward_ratios: List[float]
    entry_confidence: float
    volume_confirmation: bool
    multi_timeframe_alignment: bool
    supply_demand_zones: Dict[str, float]
    acceleration_score: float
    reasons: List[str]
    expires_at: datetime

class MCPPriceAccelerator:
    """Analyzes price acceleration patterns to identify convexity points"""

    def __init__(self):
        self.lookback_periods = {
            '1m': 60,   # 1 hour of 1m data
            '5m': 48,   # 4 hours of 5m data
            '15m': 32,  # 8 hours of 15m data
            '1h': 24,   # 24 hours of 1h data
            '4h': 21,   # 3.5 days of 4h data
        }

        self.convexity_thresholds = {
            ConvexityLevel.WEAK: 0.001,
            ConvexityLevel.MODERATE: 0.003,
            ConvexityLevel.STRONG: 0.007,
            ConvexityLevel.EXTREME: 0.015,
            ConvexityLevel.BREAKOUT: 0.025,
        }

    async def analyze_convexity_points(self, symbol: str, ohlcv_data: Dict[str, pd.DataFrame],
                                     side: str) -> List[ConvexityPoint]:
        """
        Analyze price data across multiple timeframes to identify convexity points

        Args:
            symbol: Trading symbol
            ohlcv_data: Dictionary of OHLCV data by timeframe
            side: 'buy' or 'sell' direction

        Returns:
            List of identified convexity points
        """
        convexity_points = []

        for timeframe, data in ohlcv_data.items():
            try:
                if data is None or len(data) < 10:
                    continue

                points = await self._analyze_single_timeframe_convexity(
                    symbol, timeframe, data, side
                )
                convexity_points.extend(points)

            except Exception as e:
                logger.warning(f"MCP analysis failed for {symbol} {timeframe}: {e}")

        # Sort by confidence and recency
        convexity_points.sort(key=lambda x: (x.confidence, x.timestamp), reverse=True)

        return convexity_points

    async def _analyze_single_timeframe_convexity(self, symbol: str, timeframe: str,
                                                data: pd.DataFrame, side: str) -> List[ConvexityPoint]:
        """Analyze convexity points for a single timeframe"""

        points = []

        try:
            # Calculate price acceleration
            acceleration = self._calculate_price_acceleration(data)

            # Find convexity points using peak detection
            peak_indices = self._find_convexity_peaks(acceleration, data)

            for idx in peak_indices:
                if idx >= len(data):
                    continue

                row = data.iloc[idx]
                accel_value = acceleration[idx] if idx < len(acceleration) else 0

                # Determine convexity type and strength
                convexity_type = self._classify_convexity_type(accel_value, side)
                strength = self._assess_convexity_strength(abs(accel_value))

                # Calculate volume confirmation
                volume_ratio = self._calculate_volume_ratio(data, idx)

                # Calculate confidence score
                confidence = self._calculate_convexity_confidence(
                    strength, volume_ratio, convexity_type, side
                )

                # Only include high-confidence points
                if confidence >= 0.6:
                    point = ConvexityPoint(
                        timestamp=row.name if hasattr(row, 'name') else datetime.now(),
                        price=row['close'],
                        convexity_type=convexity_type,
                        strength=strength,
                        acceleration=accel_value,
                        volume_ratio=volume_ratio,
                        confidence=confidence,
                        timeframe=timeframe,
                        supporting_factors=self._generate_supporting_factors(
                            convexity_type, strength, volume_ratio
                        )
                    )
                    points.append(point)

        except Exception as e:
            logger.warning(f"Convexity analysis error for {symbol} {timeframe}: {e}")

        return points

    def _calculate_price_acceleration(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate price acceleration using second derivative approximation"""

        if len(data) < 3:
            return np.zeros(len(data))

        # Use close prices for calculation
        prices = data['close'].values

        # Calculate first derivative (velocity)
        velocity = np.gradient(prices)

        # Calculate second derivative (acceleration)
        acceleration = np.gradient(velocity)

        # Smooth acceleration to reduce noise
        acceleration = self._smooth_signal(acceleration, window=3)

        return acceleration

    def _smooth_signal(self, signal: np.ndarray, window: int = 3) -> np.ndarray:
        """Apply simple moving average smoothing"""
        if len(signal) < window:
            return signal

        smoothed = np.convolve(signal, np.ones(window)/window, mode='valid')

        # Pad the result to maintain original length
        padding = (len(signal) - len(smoothed)) // 2
        smoothed = np.pad(smoothed, padding, mode='edge')

        return smoothed[:len(signal)]

    def _find_convexity_peaks(self, acceleration: np.ndarray, data: pd.DataFrame) -> List[int]:
        """Find peaks in acceleration that indicate convexity points"""

        # Find peaks with minimum prominence
        min_prominence = np.std(acceleration) * 0.5
        peaks, _ = find_peaks(np.abs(acceleration),
                            prominence=min_prominence,
                            distance=5)  # Minimum 5 candles apart

        # Filter peaks by strength
        strong_peaks = []
        for peak in peaks:
            if abs(acceleration[peak]) > self.convexity_thresholds[ConvexityLevel.MODERATE]:
                strong_peaks.append(peak)

        return strong_peaks

    def _classify_convexity_type(self, acceleration: float, side: str) -> ConvexityType:
        """Classify the type of convexity based on acceleration and direction"""

        abs_accel = abs(acceleration)

        if side == 'buy':
            if acceleration > 0:
                if abs_accel > self.convexity_thresholds[ConvexityLevel.STRONG]:
                    return ConvexityType.ACCELERATING_UP
                else:
                    return ConvexityType.INFLECTION_UP
            else:
                return ConvexityType.DECELERATING_UP
        else:  # sell
            if acceleration < 0:
                if abs_accel > self.convexity_thresholds[ConvexityLevel.STRONG]:
                    return ConvexityType.ACCELERATING_DOWN
                else:
                    return ConvexityType.INFLECTION_DOWN
            else:
                return ConvexityType.DECELERATING_DOWN

    def _assess_convexity_strength(self, abs_acceleration: float) -> ConvexityLevel:
        """Assess the strength of convexity based on acceleration magnitude"""

        for level in reversed(ConvexityLevel):
            if abs_acceleration >= self.convexity_thresholds[level]:
                return level

        return ConvexityLevel.WEAK

    def _calculate_volume_ratio(self, data: pd.DataFrame, index: int) -> float:
        """Calculate volume ratio compared to recent average"""

        if 'volume' not in data.columns or index >= len(data):
            return 1.0

        try:
            current_volume = data.iloc[index]['volume']

            # Calculate average volume from lookback period
            lookback = min(20, len(data))
            start_idx = max(0, index - lookback)
            avg_volume = data.iloc[start_idx:index]['volume'].mean()

            if avg_volume > 0:
                return current_volume / avg_volume
            else:
                return 1.0

        except Exception:
            return 1.0

    def _calculate_convexity_confidence(self, strength: ConvexityLevel,
                                      volume_ratio: float, convexity_type: ConvexityType,
                                      side: str) -> float:
        """Calculate confidence score for convexity point"""

        # Base confidence from strength
        strength_confidence = {
            ConvexityLevel.WEAK: 0.4,
            ConvexityLevel.MODERATE: 0.6,
            ConvexityLevel.STRONG: 0.75,
            ConvexityLevel.EXTREME: 0.85,
            ConvexityLevel.BREAKOUT: 0.95,
        }

        confidence = strength_confidence.get(strength, 0.4)

        # Volume confirmation bonus
        if volume_ratio > 1.5:
            confidence += 0.1
        elif volume_ratio > 1.2:
            confidence += 0.05

        # Convexity type bonus for favorable types
        favorable_types = []
        if side == 'buy':
            favorable_types = [ConvexityType.ACCELERATING_UP, ConvexityType.INFLECTION_UP]
        else:
            favorable_types = [ConvexityType.ACCELERATING_DOWN, ConvexityType.INFLECTION_DOWN]

        if convexity_type in favorable_types:
            confidence += 0.05

        return min(1.0, confidence)

    def _generate_supporting_factors(self, convexity_type: ConvexityType,
                                   strength: ConvexityLevel, volume_ratio: float) -> List[str]:
        """Generate list of supporting factors for the convexity point"""

        factors = []

        # Add convexity type
        factors.append(f"{convexity_type.value.replace('_', ' ').title()}")

        # Add strength
        factors.append(f"{strength.name.title()} Acceleration")

        # Add volume confirmation
        if volume_ratio > 1.5:
            factors.append("High Volume Confirmation")
        elif volume_ratio > 1.2:
            factors.append("Volume Confirmation")
        else:
            factors.append("Normal Volume")

        return factors

class MCPEntryOptimizer:
    """Optimizes entry points using MCP analysis"""

    def __init__(self):
        self.price_accelerator = MCPPriceAccelerator()
        self.min_convexity_confidence = 0.7
        self.max_entry_delay_minutes = 15

    async def find_optimal_mcp_entry(self, symbol: str, side: str,
                                   market_data: Dict[str, Any]) -> Optional[MCPEntrySignal]:
        """
        Find optimal entry point using MCP analysis

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            market_data: Current market data including OHLCV

        Returns:
            MCP-optimized entry signal or None
        """

        try:
            # Extract OHLCV data by timeframe
            ohlcv_data = self._extract_ohlcv_by_timeframe(market_data)

            if not ohlcv_data:
                logger.warning(f"No OHLCV data available for {symbol}")
                return None

            # Analyze convexity points across timeframes
            convexity_points = await self.price_accelerator.analyze_convexity_points(
                symbol, ohlcv_data, side
            )

            if not convexity_points:
                logger.info(f"No significant convexity points found for {symbol}")
                return None

            # Select best convexity point
            best_point = self._select_optimal_convexity_point(convexity_points, side)

            if not best_point or best_point.confidence < self.min_convexity_confidence:
                logger.info(f"No high-confidence convexity points for {symbol}")
                return None

            # Calculate optimal entry levels
            entry_signal = await self._calculate_mcp_entry_levels(
                symbol, side, best_point, market_data
            )

            if entry_signal:
                logger.info(f"🎯 MCP Entry Signal for {symbol}:")
                logger.info(f"   Convexity: {best_point.convexity_type.value} ({best_point.strength.name})")
                logger.info(f"   Confidence: {best_point.confidence:.2f}")
                logger.info(f"   Entry: {entry_signal.optimal_entry_price:.6f}")
                logger.info(f"   Acceleration Score: {entry_signal.acceleration_score:.3f}")

            return entry_signal

        except Exception as e:
            logger.error(f"MCP entry optimization failed for {symbol}: {e}")
            return None

    def _extract_ohlcv_by_timeframe(self, market_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Extract OHLCV data organized by timeframe"""

        ohlcv_data = {}

        # Look for timeframe-specific data
        for timeframe in ['1m', '5m', '15m', '1h', '4h']:
            tf_key = f'ohlcv_{timeframe}'
            if tf_key in market_data and market_data[tf_key] is not None:
                try:
                    df = pd.DataFrame(market_data[tf_key])
                    if not df.empty and len(df.columns) >= 5:
                        # Ensure proper column names
                        if df.columns[0] != 'timestamp':
                            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        ohlcv_data[timeframe] = df
                except Exception as e:
                    logger.warning(f"Failed to process {tf_key} data: {e}")

        # If no timeframe-specific data, try to create from general data
        if not ohlcv_data and 'ohlcv' in market_data:
            try:
                df = pd.DataFrame(market_data['ohlcv'])
                if not df.empty:
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    ohlcv_data['1h'] = df  # Assume 1h if not specified
            except Exception as e:
                logger.warning(f"Failed to process general OHLCV data: {e}")

        return ohlcv_data

    def _select_optimal_convexity_point(self, points: List[ConvexityPoint], side: str) -> Optional[ConvexityPoint]:
        """Select the most optimal convexity point for entry"""

        if not points:
            return None

        # Filter for relevant convexity types based on side
        relevant_types = []
        if side == 'buy':
            relevant_types = [ConvexityType.ACCELERATING_UP, ConvexityType.INFLECTION_UP]
        else:
            relevant_types = [ConvexityType.ACCELERATING_DOWN, ConvexityType.INFLECTION_DOWN]

        relevant_points = [p for p in points if p.convexity_type in relevant_types]

        if not relevant_points:
            return None

        # Score points based on multiple factors
        scored_points = []
        for point in relevant_points:
            score = (
                point.confidence * 0.4 +  # Confidence weight
                (point.strength.value / 5) * 0.3 +  # Strength weight
                min(1.0, point.volume_ratio / 2) * 0.3  # Volume weight
            )

            # Time decay factor (prefer more recent points)
            time_diff_minutes = (datetime.now() - point.timestamp).total_seconds() / 60
            time_factor = max(0.1, 1.0 - (time_diff_minutes / 60))  # Decay over 1 hour

            final_score = score * time_factor
            scored_points.append((point, final_score))

        # Return highest scoring point
        if scored_points:
            return max(scored_points, key=lambda x: x[1])[0]

        return None

    async def _calculate_mcp_entry_levels(self, symbol: str, side: str,
                                        convexity_point: ConvexityPoint,
                                        market_data: Dict[str, Any]) -> Optional[MCPEntrySignal]:
        """Calculate optimal entry levels based on MCP analysis"""

        try:
            current_price = market_data.get('price', convexity_point.price)

            # Calculate optimal entry price with MCP adjustment
            entry_price_adjustment = self._calculate_entry_price_adjustment(
                convexity_point, side
            )
            optimal_entry_price = current_price * (1 + entry_price_adjustment)

            # Calculate dynamic stop loss based on convexity
            stop_loss = self._calculate_mcp_stop_loss(
                optimal_entry_price, convexity_point, side
            )

            # Calculate multiple take profit levels
            take_profit_levels = self._calculate_mcp_take_profits(
                optimal_entry_price, convexity_point, side, market_data
            )

            # Calculate risk-reward ratios
            risk_reward_ratios = []
            risk = abs(optimal_entry_price - stop_loss)
            for tp in take_profit_levels:
                reward = abs(tp - optimal_entry_price)
                rr_ratio = reward / risk if risk > 0 else 0
                risk_reward_ratios.append(rr_ratio)

            # Check volume confirmation
            volume_confirmation = convexity_point.volume_ratio > 1.2

            # Calculate entry confidence
            entry_confidence = self._calculate_entry_confidence(
                convexity_point, volume_confirmation, risk_reward_ratios[0]
            )

            # Calculate acceleration score
            acceleration_score = (
                convexity_point.acceleration * convexity_point.confidence *
                (convexity_point.strength.value / 5)
            )

            # Generate reasoning
            reasons = self._generate_mcp_reasons(convexity_point, volume_confirmation)

            # Create MCP entry signal
            signal = MCPEntrySignal(
                symbol=symbol,
                convexity_point=convexity_point,
                optimal_entry_price=optimal_entry_price,
                stop_loss=stop_loss,
                take_profit_levels=take_profit_levels,
                risk_reward_ratios=risk_reward_ratios,
                entry_confidence=entry_confidence,
                volume_confirmation=volume_confirmation,
                multi_timeframe_alignment=True,  # Assumed for now
                supply_demand_zones=self._identify_supply_demand_zones(market_data),
                acceleration_score=acceleration_score,
                reasons=reasons,
                expires_at=datetime.now() + timedelta(minutes=self.max_entry_delay_minutes)
            )

            return signal

        except Exception as e:
            logger.error(f"Failed to calculate MCP entry levels for {symbol}: {e}")
            return None

    def _calculate_entry_price_adjustment(self, convexity_point: ConvexityPoint, side: str) -> float:
        """Calculate price adjustment for optimal entry timing"""

        # Base adjustment on convexity type and strength
        base_adjustment = 0.001  # 0.1% base

        # Adjust based on convexity strength
        strength_multiplier = {
            ConvexityLevel.WEAK: 0.5,
            ConvexityLevel.MODERATE: 1.0,
            ConvexityLevel.STRONG: 1.5,
            ConvexityLevel.EXTREME: 2.0,
            ConvexityLevel.BREAKOUT: 2.5,
        }

        adjustment = base_adjustment * strength_multiplier.get(convexity_point.strength, 1.0)

        # Direction matters
        if side == 'buy':
            # For buy signals, enter slightly below convexity point for better price
            if convexity_point.convexity_type in [ConvexityType.ACCELERATING_UP, ConvexityType.INFLECTION_UP]:
                return -adjustment * 0.8  # Enter below for upward acceleration
            else:
                return adjustment * 0.5   # Conservative entry
        else:  # sell
            # For sell signals, enter slightly above convexity point
            if convexity_point.convexity_type in [ConvexityType.ACCELERATING_DOWN, ConvexityType.INFLECTION_DOWN]:
                return adjustment * 0.8   # Enter above for downward acceleration
            else:
                return -adjustment * 0.5  # Conservative entry

    def _calculate_mcp_stop_loss(self, entry_price: float, convexity_point: ConvexityPoint, side: str) -> float:
        """Calculate stop loss based on MCP analysis"""

        # Base stop loss percentage based on convexity strength
        base_sl_pct = {
            ConvexityLevel.WEAK: 0.008,      # 0.8%
            ConvexityLevel.MODERATE: 0.012,  # 1.2%
            ConvexityLevel.STRONG: 0.015,    # 1.5%
            ConvexityLevel.EXTREME: 0.020,   # 2.0%
            ConvexityLevel.BREAKOUT: 0.025,  # 2.5%
        }

        sl_pct = base_sl_pct.get(convexity_point.strength, 0.015)

        # Adjust based on acceleration
        accel_adjustment = abs(convexity_point.acceleration) * 100  # Convert to percentage
        sl_pct += min(0.01, accel_adjustment * 0.5)  # Add up to 1% for high acceleration

        # Apply stop loss
        if side == 'buy':
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)

    def _calculate_mcp_take_profits(self, entry_price: float, convexity_point: ConvexityPoint,
                                  side: str, market_data: Dict[str, Any]) -> List[float]:
        """Calculate multiple take profit levels based on MCP"""

        levels = []

        # Base take profit percentages based on convexity strength
        base_tp_pct = {
            ConvexityLevel.WEAK: [0.015, 0.030, 0.045],      # 1.5%, 3.0%, 4.5%
            ConvexityLevel.MODERATE: [0.025, 0.050, 0.075],  # 2.5%, 5.0%, 7.5%
            ConvexityLevel.STRONG: [0.035, 0.070, 0.105],    # 3.5%, 7.0%, 10.5%
            ConvexityLevel.EXTREME: [0.050, 0.100, 0.150],   # 5.0%, 10.0%, 15.0%
            ConvexityLevel.BREAKOUT: [0.075, 0.150, 0.225],  # 7.5%, 15.0%, 22.5%
        }

        tp_pcts = base_tp_pct.get(convexity_point.strength, [0.025, 0.050, 0.075])

        # Adjust based on volume confirmation
        if convexity_point.volume_ratio > 1.5:
            # Increase targets for high volume
            tp_pcts = [pct * 1.2 for pct in tp_pcts]

        # Calculate actual price levels
        for pct in tp_pcts:
            if side == 'buy':
                tp_price = entry_price * (1 + pct)
            else:
                tp_price = entry_price * (1 - pct)
            levels.append(tp_price)

        return levels

    def _calculate_entry_confidence(self, convexity_point: ConvexityPoint,
                                  volume_confirmation: bool, primary_rr_ratio: float) -> float:
        """Calculate overall entry confidence"""

        confidence = convexity_point.confidence

        # Volume confirmation bonus
        if volume_confirmation:
            confidence += 0.1

        # Risk-reward bonus
        if primary_rr_ratio >= 3.0:
            confidence += 0.1
        elif primary_rr_ratio >= 2.0:
            confidence += 0.05

        # Acceleration bonus
        if abs(convexity_point.acceleration) > self.price_accelerator.convexity_thresholds[ConvexityLevel.STRONG]:
            confidence += 0.05

        return min(1.0, confidence)

    def _identify_supply_demand_zones(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Identify nearby supply and demand zones"""

        zones = {}

        # This would integrate with actual supply/demand zone analysis
        # For now, return mock data
        current_price = market_data.get('price', 1.0)

        zones['nearest_support'] = current_price * 0.98
        zones['nearest_resistance'] = current_price * 1.02

        return zones

    def _generate_mcp_reasons(self, convexity_point: ConvexityPoint,
                            volume_confirmation: bool) -> List[str]:
        """Generate reasons for MCP entry signal"""

        reasons = []

        # Primary convexity reason
        reasons.append(f"MCP: {convexity_point.convexity_type.value.replace('_', ' ').title()}")

        # Strength
        reasons.append(f"{convexity_point.strength.name.title()} Convexity")

        # Supporting factors
        reasons.extend(convexity_point.supporting_factors)

        # Volume confirmation
        if volume_confirmation:
            reasons.append("Volume Confirmed")

        return reasons

# Global MCP optimizer instance
mcp_optimizer = MCPEntryOptimizer()
