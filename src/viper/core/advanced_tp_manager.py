#!/usr/bin/env python3
"""
🎯 ADVANCED TAKE PROFIT MANAGER
Intelligent take profit optimization using MCP inflection points and market dynamics

This system implements:
✅ Dynamic take profit levels based on MCP analysis
✅ Partial exit strategies with scaling
✅ Trailing stop optimization using convexity
✅ Market condition adaptation
✅ Volume-based profit taking
✅ Multi-timeframe exit signals
✅ Risk-adjusted profit targets
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TPExitType(Enum):
    """Types of take profit exits"""
    PARTIAL = "PARTIAL"
    FULL = "FULL"
    TRAILING = "TRAILING"
    TIME_BASED = "TIME_BASED"
    VOLUME_BASED = "VOLUME_BASED"

class TPLevel(Enum):
    """Take profit level classifications"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    BREAKOUT = "BREAKOUT"

@dataclass
class TPExitPoint:
    """Represents a take profit exit point"""
    price: float
    percentage: float
    exit_type: TPExitType
    confidence: float
    volume_requirement: float
    time_window: timedelta
    conditions: List[str]

@dataclass
class AdvancedTPPlan:
    """Complete take profit management plan"""
    symbol: str
    entry_price: float
    exit_points: List[TPExitPoint]
    trailing_stop_config: Dict[str, Any]
    total_target_profit: float
    risk_reward_ratio: float
    max_holding_time: timedelta
    volume_based_exits: bool
    dynamic_adjustment: bool
    created_at: datetime
    last_updated: datetime

class MCPTakeProfitOptimizer:
    """Advanced take profit optimization using MCP analysis"""

    def __init__(self):
        self.min_profit_threshold = 0.005  # 0.5% minimum profit
        self.max_holding_period = timedelta(hours=24)
        self.trailing_stop_activation_pct = 0.02  # 2% profit before trailing

        # Default exit configurations
        self.exit_configurations = {
            TPLevel.CONSERVATIVE: {
                'levels': [0.015, 0.035, 0.065],  # 1.5%, 3.5%, 6.5%
                'allocations': [0.3, 0.3, 0.4],   # 30%, 30%, 40%
                'trailing_pct': 0.008
            },
            TPLevel.MODERATE: {
                'levels': [0.025, 0.055, 0.095],  # 2.5%, 5.5%, 9.5%
                'allocations': [0.25, 0.35, 0.4],  # 25%, 35%, 40%
                'trailing_pct': 0.012
            },
            TPLevel.AGGRESSIVE: {
                'levels': [0.04, 0.085, 0.145],   # 4%, 8.5%, 14.5%
                'allocations': [0.2, 0.3, 0.5],    # 20%, 30%, 50%
                'trailing_pct': 0.018
            },
            TPLevel.BREAKOUT: {
                'levels': [0.06, 0.125, 0.21],    # 6%, 12.5%, 21%
                'allocations': [0.15, 0.35, 0.5],  # 15%, 35%, 50%
                'trailing_pct': 0.025
            }
        }

    async def create_advanced_tp_plan(self, symbol: str, side: str, entry_price: float,
                                    convexity_point: Any, market_data: Dict[str, Any]) -> AdvancedTPPlan:
        """
        Create an advanced take profit plan using MCP analysis

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            entry_price: Entry price
            convexity_point: MCP convexity point analysis
            market_data: Current market data

        Returns:
            Advanced TP plan with multiple exit points
        """

        try:
            # Determine TP level based on convexity strength and market conditions
            tp_level = self._determine_tp_level(convexity_point, market_data)

            # Get base configuration for the level
            config = self.exit_configurations[tp_level]

            # Calculate exit points
            exit_points = self._calculate_exit_points(
                entry_price, side, config, convexity_point, market_data
            )

            # Configure trailing stop
            trailing_config = self._configure_trailing_stop(
                entry_price, side, config, convexity_point
            )

            # Calculate total target profit
            total_target = self._calculate_total_target_profit(
                entry_price, exit_points, config
            )

            # Calculate risk-reward ratio (would need stop loss for full calculation)
            risk_reward_ratio = self._estimate_risk_reward(exit_points, convexity_point)

            # Create advanced TP plan
            plan = AdvancedTPPlan(
                symbol=symbol,
                entry_price=entry_price,
                exit_points=exit_points,
                trailing_stop_config=trailing_config,
                total_target_profit=total_target,
                risk_reward_ratio=risk_reward_ratio,
                max_holding_time=self._calculate_max_holding_time(convexity_point),
                volume_based_exits=self._should_use_volume_exits(market_data),
                dynamic_adjustment=True,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )

            logger.info(f"🎯 Advanced TP Plan for {symbol}:")
            logger.info(f"   Level: {tp_level.value}")
            logger.info(f"   Exit Points: {len(exit_points)}")
            logger.info(f"   Total Target: {total_target:.2f}%")
            logger.info(f"   Trailing: {trailing_config.get('activation_pct', 0)*100:.1f}%")

            return plan

        except Exception as e:
            logger.error(f"Failed to create advanced TP plan for {symbol}: {e}")
            return self._create_fallback_tp_plan(symbol, side, entry_price)

    def _determine_tp_level(self, convexity_point: Any, market_data: Dict[str, Any]) -> TPLevel:
        """Determine appropriate take profit level based on convexity and market conditions"""

        # Get convexity strength
        if hasattr(convexity_point, 'strength'):
            strength = convexity_point.strength
        else:
            # Fallback if no convexity data
            volume = market_data.get('volume', 0)
            change_24h = abs(market_data.get('change_24h', 0))

            if volume > 1000000 and change_24h > 3:
                return TPLevel.BREAKOUT
            elif change_24h > 2:
                return TPLevel.AGGRESSIVE
            elif change_24h > 1:
                return TPLevel.MODERATE
            else:
                return TPLevel.CONSERVATIVE

        # Map convexity strength to TP level
        strength_to_level = {
            'WEAK': TPLevel.CONSERVATIVE,
            'MODERATE': TPLevel.MODERATE,
            'STRONG': TPLevel.AGGRESSIVE,
            'EXTREME': TPLevel.AGGRESSIVE,
            'BREAKOUT': TPLevel.BREAKOUT
        }

        # Get strength name
        if hasattr(strength, 'name'):
            strength_name = strength.name
        else:
            strength_name = str(strength).upper()

        return strength_to_level.get(strength_name, TPLevel.MODERATE)

    def _calculate_exit_points(self, entry_price: float, side: str, config: Dict[str, Any],
                             convexity_point: Any, market_data: Dict[str, Any]) -> List[TPExitPoint]:
        """Calculate optimal exit points based on configuration and market conditions"""

        exit_points = []
        levels = config['levels']
        allocations = config['allocations']

        for i, (level_pct, allocation) in enumerate(zip(levels, allocations)):
            # Calculate exit price
            if side == 'buy':
                exit_price = entry_price * (1 + level_pct)
            else:
                exit_price = entry_price * (1 - level_pct)

            # Adjust percentage based on convexity
            adjusted_pct = self._adjust_tp_percentage(level_pct, convexity_point, i)

            # Determine exit type
            exit_type = self._determine_exit_type(i, convexity_point, market_data)

            # Calculate confidence
            confidence = self._calculate_exit_confidence(i, convexity_point, market_data)

            # Volume requirement for exit
            volume_requirement = self._calculate_volume_requirement(exit_price, market_data)

            # Time window for exit
            time_window = self._calculate_exit_time_window(i, convexity_point)

            # Generate conditions
            conditions = self._generate_exit_conditions(i, convexity_point, market_data)

            exit_point = TPExitPoint(
                price=exit_price,
                percentage=adjusted_pct,
                exit_type=exit_type,
                confidence=confidence,
                volume_requirement=volume_requirement,
                time_window=time_window,
                conditions=conditions
            )

            exit_points.append(exit_point)

        return exit_points

    def _adjust_tp_percentage(self, base_pct: float, convexity_point: Any, level_index: int) -> float:
        """Adjust take profit percentage based on convexity analysis"""

        if not hasattr(convexity_point, 'acceleration'):
            return base_pct

        acceleration = convexity_point.acceleration

        # Higher acceleration suggests stronger moves, adjust targets accordingly
        accel_multiplier = 1.0 + (abs(acceleration) * 50)  # Convert acceleration to multiplier

        # Scale adjustment based on level (more aggressive for later levels)
        level_multiplier = 1.0 + (level_index * 0.1)

        adjusted_pct = base_pct * accel_multiplier * level_multiplier

        # Cap maximum adjustment
        return min(adjusted_pct, base_pct * 2.0)

    def _determine_exit_type(self, level_index: int, convexity_point: Any,
                           market_data: Dict[str, Any]) -> TPExitType:
        """Determine the type of exit for this level"""

        if level_index == 0:
            return TPExitType.PARTIAL  # First exit is always partial
        elif level_index == 1:
            # Check for volume spike potential
            volume = market_data.get('volume', 0)
            if volume > market_data.get('avg_volume', 0) * 1.5:
                return TPExitType.VOLUME_BASED
            else:
                return TPExitType.PARTIAL
        else:
            # Final exit - check if we should use trailing
            if hasattr(convexity_point, 'strength'):
                strength_name = convexity_point.strength.name if hasattr(convexity_point.strength, 'name') else str(convexity_point.strength)
                if strength_name in ['BREAKOUT', 'EXTREME']:
                    return TPExitType.TRAILING
            return TPExitType.FULL

    def _calculate_exit_confidence(self, level_index: int, convexity_point: Any,
                                 market_data: Dict[str, Any]) -> float:
        """Calculate confidence level for this exit point"""

        base_confidence = 0.7 - (level_index * 0.1)  # Decreasing confidence for later exits

        # Boost confidence based on convexity strength
        if hasattr(convexity_point, 'strength'):
            strength_name = convexity_point.strength.name if hasattr(convexity_point.strength, 'name') else str(convexity_point.strength)
            strength_bonus = {
                'WEAK': 0.0,
                'MODERATE': 0.05,
                'STRONG': 0.1,
                'EXTREME': 0.15,
                'BREAKOUT': 0.2
            }
            base_confidence += strength_bonus.get(strength_name, 0.0)

        # Volume confirmation bonus
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume', volume * 0.8)
        if volume > avg_volume * 1.2:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _calculate_volume_requirement(self, exit_price: float, market_data: Dict[str, Any]) -> float:
        """Calculate volume requirement for exit execution"""

        avg_volume = market_data.get('avg_volume', market_data.get('volume', 100000))
        base_requirement = avg_volume * 0.8  # 80% of average volume

        # Scale requirement based on price level (higher prices need more volume)
        price_factor = exit_price / market_data.get('price', exit_price)
        volume_requirement = base_requirement * price_factor

        return volume_requirement

    def _calculate_exit_time_window(self, level_index: int, convexity_point: Any) -> timedelta:
        """Calculate time window allowed for this exit"""

        base_hours = 4 + (level_index * 2)  # 4h, 6h, 8h for levels 0, 1, 2

        # Extend time for weaker convexity
        if hasattr(convexity_point, 'strength'):
            strength_name = convexity_point.strength.name if hasattr(convexity_point.strength, 'name') else str(convexity_point.strength)
            if strength_name in ['WEAK', 'MODERATE']:
                base_hours += 2

        return timedelta(hours=base_hours)

    def _generate_exit_conditions(self, level_index: int, convexity_point: Any,
                                market_data: Dict[str, Any]) -> List[str]:
        """Generate conditions that must be met for exit"""

        conditions = []

        # Basic conditions
        conditions.append(f"Price reaches TP{level_index + 1}")

        # Volume condition
        if level_index > 0:
            conditions.append("Volume confirmation required")

        # Time condition
        conditions.append("Within time window")

        # Convexity-specific conditions
        if hasattr(convexity_point, 'volume_ratio') and convexity_point.volume_ratio > 1.5:
            conditions.append("Maintain volume momentum")

        return conditions

    def _configure_trailing_stop(self, entry_price: float, side: str, config: Dict[str, Any],
                               convexity_point: Any) -> Dict[str, Any]:
        """Configure trailing stop parameters"""

        trailing_pct = config.get('trailing_pct', 0.015)

        # Adjust trailing percentage based on convexity
        if hasattr(convexity_point, 'strength'):
            strength_name = convexity_point.strength.name if hasattr(convexity_point.strength, 'name') else str(convexity_point.strength)
            if strength_name in ['BREAKOUT', 'EXTREME']:
                trailing_pct *= 1.5  # Wider trailing for strong moves

        return {
            'activation_pct': self.trailing_stop_activation_pct,
            'trailing_pct': trailing_pct,
            'side': side,
            'entry_price': entry_price,
            'current_trail_price': entry_price,
            'activated': False
        }

    def _calculate_total_target_profit(self, entry_price: float, exit_points: List[TPExitPoint],
                                    config: Dict[str, Any]) -> float:
        """Calculate total target profit percentage"""

        total_profit = 0
        allocations = config['allocations']

        for exit_point, allocation in zip(exit_points, allocations):
            profit_pct = abs(exit_point.price - entry_price) / entry_price
            total_profit += profit_pct * allocation

        return total_profit * 100  # Convert to percentage

    def _estimate_risk_reward(self, exit_points: List[TPExitPoint], convexity_point: Any) -> float:
        """Estimate risk-reward ratio"""

        if not exit_points:
            return 1.0

        # Use first exit point as primary target
        primary_target = exit_points[0]

        # Estimate stop loss based on convexity (rough approximation)
        if hasattr(convexity_point, 'strength'):
            strength_name = convexity_point.strength.name if hasattr(convexity_point.strength, 'name') else str(convexity_point.strength)
            sl_pct_estimates = {
                'WEAK': 0.01,
                'MODERATE': 0.015,
                'STRONG': 0.02,
                'EXTREME': 0.025,
                'BREAKOUT': 0.03
            }
            risk_pct = sl_pct_estimates.get(strength_name, 0.015)
        else:
            risk_pct = 0.015  # Default 1.5%

        reward_pct = primary_target.percentage

        return reward_pct / risk_pct if risk_pct > 0 else 1.0

    def _calculate_max_holding_time(self, convexity_point: Any) -> timedelta:
        """Calculate maximum holding time based on convexity"""

        base_hours = 12  # Base 12 hours

        if hasattr(convexity_point, 'strength'):
            strength_name = convexity_point.strength.name if hasattr(convexity_point.strength, 'name') else str(convexity_point.strength)
            if strength_name in ['BREAKOUT', 'EXTREME']:
                base_hours = 24  # Allow longer holding for strong moves
            elif strength_name == 'WEAK':
                base_hours = 6   # Shorter holding for weak signals

        return timedelta(hours=base_hours)

    def _should_use_volume_exits(self, market_data: Dict[str, Any]) -> bool:
        """Determine if volume-based exits should be used"""

        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume', volume * 0.7)

        # Use volume exits if current volume is significantly above average
        return volume > avg_volume * 1.3

    def _create_fallback_tp_plan(self, symbol: str, side: str, entry_price: float) -> AdvancedTPPlan:
        """Create a fallback TP plan when MCP analysis fails"""

        logger.warning(f"Using fallback TP plan for {symbol}")

        # Simple three-level exit plan
        exit_points = []

        for i, pct in enumerate([0.02, 0.045, 0.08]):
            if side == 'buy':
                exit_price = entry_price * (1 + pct)
            else:
                exit_price = entry_price * (1 - pct)

            exit_point = TPExitPoint(
                price=exit_price,
                percentage=pct,
                exit_type=TPExitType.PARTIAL if i < 2 else TPExitType.FULL,
                confidence=0.6 - (i * 0.1),
                volume_requirement=100000,
                time_window=timedelta(hours=8 + i * 2),
                conditions=["Price target reached", "Basic confirmation"]
            )
            exit_points.append(exit_point)

        return AdvancedTPPlan(
            symbol=symbol,
            entry_price=entry_price,
            exit_points=exit_points,
            trailing_stop_config={
                'activation_pct': 0.02,
                'trailing_pct': 0.012,
                'activated': False
            },
            total_target_profit=4.5,
            risk_reward_ratio=2.0,
            max_holding_time=timedelta(hours=12),
            volume_based_exits=False,
            dynamic_adjustment=False,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

    async def evaluate_tp_execution(self, plan: AdvancedTPPlan, current_price: float,
                                  current_volume: float, time_held: timedelta) -> Dict[str, Any]:
        """
        Evaluate whether to execute take profit based on current conditions

        Args:
            plan: Advanced TP plan
            current_price: Current market price
            current_volume: Current volume
            time_held: Time position has been held

        Returns:
            Dictionary with execution recommendations
        """

        try:
            recommendations = {
                'should_exit': False,
                'exit_points': [],
                'partial_exits': [],
                'trailing_updates': [],
                'confidence': 0.0,
                'reasoning': []
            }

            # Check each exit point
            for i, exit_point in enumerate(plan.exit_points):
                should_exit_here = self._evaluate_exit_point(
                    exit_point, current_price, current_volume, time_held, plan.side
                )

                if should_exit_here:
                    recommendations['exit_points'].append({
                        'level': i + 1,
                        'price': exit_point.price,
                        'type': exit_point.exit_type.value,
                        'confidence': exit_point.confidence
                    })

                    # For partial exits, add to separate list
                    if exit_point.exit_type == TPExitType.PARTIAL:
                        recommendations['partial_exits'].append(i + 1)

            # Check trailing stop
            trailing_update = self._evaluate_trailing_stop(
                plan.trailing_stop_config, current_price, plan.side
            )

            if trailing_update:
                recommendations['trailing_updates'].append(trailing_update)

            # Overall recommendation
            if recommendations['exit_points'] or recommendations['trailing_updates']:
                recommendations['should_exit'] = True
                recommendations['confidence'] = self._calculate_overall_exit_confidence(
                    recommendations['exit_points']
                )

            recommendations['reasoning'] = self._generate_exit_reasoning(recommendations)

            return recommendations

        except Exception as e:
            logger.error(f"TP execution evaluation failed: {e}")
            return {
                'should_exit': False,
                'exit_points': [],
                'confidence': 0.0,
                'reasoning': [f"Evaluation error: {e}"]
            }

    def _evaluate_exit_point(self, exit_point: TPExitPoint, current_price: float,
                           current_volume: float, time_held: timedelta, side: str) -> bool:
        """Evaluate if an exit point should be triggered"""

        # Check price condition
        if side == 'buy':
            price_reached = current_price >= exit_point.price
        else:
            price_reached = current_price <= exit_point.price

        if not price_reached:
            return False

        # Check volume condition
        volume_sufficient = current_volume >= exit_point.volume_requirement

        # Check time condition
        time_valid = time_held <= exit_point.time_window

        # For volume-based exits, volume is critical
        if exit_point.exit_type == TPExitType.VOLUME_BASED and not volume_sufficient:
            return False

        # For time-based exits, time is critical
        if exit_point.exit_type == TPExitType.TIME_BASED and not time_valid:
            return False

        # General rule: need price + at least one other condition
        return price_reached and (volume_sufficient or time_valid)

    def _evaluate_trailing_stop(self, trailing_config: Dict[str, Any],
                              current_price: float, side: str) -> Optional[Dict[str, Any]]:
        """Evaluate trailing stop status"""

        if not trailing_config.get('activated', False):
            # Check if trailing should be activated
            entry_price = trailing_config['entry_price']
            activation_pct = trailing_config['activation_pct']

            if side == 'buy':
                profit_pct = (current_price - entry_price) / entry_price
            else:
                profit_pct = (entry_price - current_price) / entry_price

            if profit_pct >= activation_pct:
                return {
                    'action': 'activate',
                    'trail_price': current_price,
                    'reason': f"Profit reached {profit_pct*100:.1f}%, activating trailing stop"
                }
        else:
            # Update trailing stop
            current_trail = trailing_config['current_trail_price']
            trailing_pct = trailing_config['trailing_pct']

            if side == 'buy':
                new_trail = current_price * (1 - trailing_pct)
                if new_trail > current_trail:
                    return {
                        'action': 'update',
                        'trail_price': new_trail,
                        'reason': "Updating trailing stop higher"
                    }
                elif current_price <= current_trail:
                    return {
                        'action': 'trigger',
                        'exit_price': current_price,
                        'reason': "Trailing stop triggered"
                    }
            else:
                new_trail = current_price * (1 + trailing_pct)
                if new_trail < current_trail:
                    return {
                        'action': 'update',
                        'trail_price': new_trail,
                        'reason': "Updating trailing stop lower"
                    }
                elif current_price >= current_trail:
                    return {
                        'action': 'trigger',
                        'exit_price': current_price,
                        'reason': "Trailing stop triggered"
                    }

        return None

    def _calculate_overall_exit_confidence(self, exit_points: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence for exit recommendations"""

        if not exit_points:
            return 0.0

        # Average confidence of all exit points
        total_confidence = sum(point['confidence'] for point in exit_points)
        return total_confidence / len(exit_points)

    def _generate_exit_reasoning(self, recommendations: Dict[str, Any]) -> List[str]:
        """Generate reasoning for exit recommendations"""

        reasoning = []

        if recommendations['exit_points']:
            reasoning.append(f"{len(recommendations['exit_points'])} exit point(s) triggered")

        if recommendations['partial_exits']:
            reasoning.append(f"Partial exits recommended for levels: {recommendations['partial_exits']}")

        if recommendations['trailing_updates']:
            for update in recommendations['trailing_updates']:
                reasoning.append(update['reason'])

        if not reasoning:
            reasoning.append("No exit conditions met")

        return reasoning

# Global TP optimizer instance
advanced_tp_optimizer = MCPTakeProfitOptimizer()
