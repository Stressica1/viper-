#!/usr/bin/env python3
"""
🛡️ ADAPTIVE RISK MANAGER
Dynamic risk management system with ATR-based position sizing and adaptive stops

This system implements:
✅ ATR-based dynamic stop loss calculation
✅ Volatility-adjusted position sizing
✅ Market condition adaptation (trending vs ranging)
✅ Adaptive take profit based on volatility
✅ Risk-adjusted leverage scaling
✅ Portfolio-level risk management
✅ Emergency risk controls
✅ Performance-based risk adjustment
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

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"

class AdaptiveAction(Enum):
    """Adaptive risk management actions"""
    INCREASE_POSITION = "INCREASE_POSITION"
    DECREASE_POSITION = "DECREASE_POSITION"
    WIDEN_STOPS = "WIDEN_STOPS"
    TIGHTEN_STOPS = "TIGHTEN_STOPS"
    REDUCE_LEVERAGE = "REDUCE_LEVERAGE"
    INCREASE_LEVERAGE = "INCREASE_LEVERAGE"
    CLOSE_POSITION = "CLOSE_POSITION"
    HOLD_POSITION = "HOLD_POSITION"

@dataclass
class ATRCalculation:
    """ATR calculation result"""
    value: float
    period: int
    smoothing: str
    last_updated: datetime

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    symbol: str
    volatility: float
    atr_value: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    risk_reward_ratio: float
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    market_regime: MarketRegime
    risk_level: RiskLevel
    last_updated: datetime

@dataclass
class AdaptiveRiskPlan:
    """Adaptive risk management plan"""
    symbol: str
    entry_price: float
    position_size: float
    leverage: float
    stop_loss_price: float
    take_profit_levels: List[float]
    trailing_stop_config: Dict[str, Any]
    max_loss_pct: float
    max_holding_time: timedelta
    risk_adjustment_factor: float
    market_regime: MarketRegime
    volatility_adjustment: float
    dynamic_tp_enabled: bool
    emergency_stop_enabled: bool
    created_at: datetime
    last_adjusted: datetime

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_exposure: float
    total_positions: int
    max_exposure_limit: float
    daily_loss_limit: float
    current_daily_loss: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    last_updated: datetime

class AdaptiveRiskManager:
    """Advanced adaptive risk management system"""

    def __init__(self):
        self.atr_periods = [7, 14, 21]  # Multiple ATR periods for robustness
        self.base_risk_per_trade = 0.02  # 2% base risk per trade
        self.max_risk_per_trade = 0.05   # 5% maximum risk per trade
        self.portfolio_risk_limit = 0.10  # 10% maximum portfolio risk
        self.daily_loss_limit = 0.03     # 3% daily loss limit

        # Volatility adjustment factors
        self.volatility_multipliers = {
            RiskLevel.VERY_LOW: 0.8,
            RiskLevel.LOW: 0.9,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.HIGH: 1.2,
            RiskLevel.VERY_HIGH: 1.5,
        }

        # Market regime adjustments
        self.regime_adjustments = {
            MarketRegime.TRENDING_UP: {'risk_multiplier': 1.2, 'leverage_cap': 25},
            MarketRegime.TRENDING_DOWN: {'risk_multiplier': 1.2, 'leverage_cap': 25},
            MarketRegime.RANGING: {'risk_multiplier': 0.8, 'leverage_cap': 10},
            MarketRegime.HIGH_VOLATILITY: {'risk_multiplier': 0.7, 'leverage_cap': 5},
            MarketRegime.LOW_VOLATILITY: {'risk_multiplier': 1.3, 'leverage_cap': 50},
            MarketRegime.BREAKOUT: {'risk_multiplier': 1.5, 'leverage_cap': 25},
        }

    async def create_adaptive_risk_plan(self, symbol: str, side: str, entry_price: float,
                                      account_balance: float, ohlcv_data: Dict[str, pd.DataFrame],
                                      mtf_analysis: Any = None, volume_analysis: Any = None) -> AdaptiveRiskPlan:
        """
        Create an adaptive risk management plan

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            entry_price: Entry price
            account_balance: Account balance
            ohlcv_data: OHLCV data by timeframe
            mtf_analysis: Multi-timeframe analysis (optional)
            volume_analysis: Volume analysis (optional)

        Returns:
            Adaptive risk management plan
        """

        try:
            # Calculate ATR for volatility assessment
            atr_calc = self._calculate_atr(ohlcv_data)

            # Assess market regime
            market_regime = self._assess_market_regime(ohlcv_data, mtf_analysis)

            # Calculate volatility-adjusted risk metrics
            risk_metrics = await self._calculate_risk_metrics(
                symbol, entry_price, account_balance, atr_calc, market_regime,
                mtf_analysis, volume_analysis
            )

            # Calculate position size
            position_size, leverage = self._calculate_adaptive_position_size(
                risk_metrics, account_balance, entry_price
            )

            # Calculate dynamic stop loss
            stop_loss_price = self._calculate_dynamic_stop_loss(
                entry_price, side, atr_calc, market_regime, risk_metrics.volatility
            )

            # Calculate adaptive take profit levels
            take_profit_levels = self._calculate_adaptive_take_profits(
                entry_price, side, atr_calc, market_regime, risk_metrics
            )

            # Configure trailing stop
            trailing_config = self._configure_adaptive_trailing_stop(
                entry_price, side, atr_calc, market_regime
            )

            # Calculate maximum holding time
            max_holding_time = self._calculate_max_holding_time(market_regime, mtf_analysis)

            # Risk adjustment factor
            risk_adjustment_factor = self._calculate_risk_adjustment_factor(
                risk_metrics, market_regime
            )

            # Volatility adjustment
            volatility_adjustment = self.volatility_multipliers.get(risk_metrics.risk_level, 1.0)

            plan = AdaptiveRiskPlan(
                symbol=symbol,
                entry_price=entry_price,
                position_size=position_size,
                leverage=leverage,
                stop_loss_price=stop_loss_price,
                take_profit_levels=take_profit_levels,
                trailing_stop_config=trailing_config,
                max_loss_pct=self.base_risk_per_trade * risk_adjustment_factor,
                max_holding_time=max_holding_time,
                risk_adjustment_factor=risk_adjustment_factor,
                market_regime=market_regime,
                volatility_adjustment=volatility_adjustment,
                dynamic_tp_enabled=True,
                emergency_stop_enabled=True,
                created_at=datetime.now(),
                last_adjusted=datetime.now()
            )

            logger.info(f"🛡️ Adaptive Risk Plan for {symbol}:")
            logger.info(f"   Position Size: {position_size:.6f} ({leverage:.1f}x leverage)")
            logger.info(f"   Stop Loss: {stop_loss_price:.6f} ({risk_metrics.stop_loss_pct*100:.2f}%)")
            logger.info(f"   Take Profits: {[f'{tp:.6f}' for tp in take_profit_levels]}")
            logger.info(f"   Market Regime: {market_regime.value}")
            logger.info(f"   Risk Level: {risk_metrics.risk_level.name}")

            return plan

        except Exception as e:
            logger.error(f"Failed to create adaptive risk plan for {symbol}: {e}")
            return self._create_fallback_risk_plan(symbol, side, entry_price, account_balance)

    def _calculate_atr(self, ohlcv_data: Dict[str, pd.DataFrame]) -> ATRCalculation:
        """Calculate Average True Range for volatility assessment"""

        # Use primary timeframe (1h if available, else first available)
        primary_tf = '1h' if '1h' in ohlcv_data else list(ohlcv_data.keys())[0]
        data = ohlcv_data[primary_tf]

        if data is None or len(data) < 20:
            return ATRCalculation(value=0.01, period=14, smoothing='sma', last_updated=datetime.now())

        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values

            # Calculate True Range
            tr = np.zeros(len(close))
            for i in range(1, len(close)):
                tr[i] = max(
                    high[i] - low[i],  # Current range
                    abs(high[i] - close[i-1]),  # High to previous close
                    abs(low[i] - close[i-1])    # Low to previous close
                )

            # Calculate ATR using multiple periods and average them
            atr_values = []
            for period in self.atr_periods:
                if len(tr) >= period:
                    atr = np.mean(tr[-period:])
                    atr_values.append(atr)

            final_atr = np.mean(atr_values) if atr_values else np.mean(tr[-14:])

            return ATRCalculation(
                value=final_atr,
                period=14,  # Average period
                smoothing='multi_period_average',
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.warning(f"ATR calculation error: {e}")
            return ATRCalculation(value=0.01, period=14, smoothing='fallback', last_updated=datetime.now())

    def _assess_market_regime(self, ohlcv_data: Dict[str, pd.DataFrame],
                            mtf_analysis: Any = None) -> MarketRegime:
        """Assess current market regime"""

        # Use multi-timeframe analysis if available
        if mtf_analysis and hasattr(mtf_analysis, 'primary_trend'):
            trend = mtf_analysis.primary_trend
            if hasattr(trend, 'name'):
                trend_name = trend.name
            else:
                trend_name = str(trend).upper()

            if 'BULL' in trend_name:
                return MarketRegime.TRENDING_UP
            elif 'BEAR' in trend_name:
                return MarketRegime.TRENDING_DOWN

        # Fallback to price action analysis
        primary_tf = '1h' if '1h' in ohlcv_data else list(ohlcv_data.keys())[0]
        data = ohlcv_data[primary_tf]

        if data is None or len(data) < 20:
            return MarketRegime.RANGING

        try:
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values

            # Calculate trend strength
            lookback = min(20, len(closes))
            slope = np.polyfit(range(lookback), closes[-lookback:], 1)[0]
            slope_pct = slope / closes[-lookback] * lookback

            # Calculate volatility
            returns = np.diff(closes[-lookback:]) / closes[-lookback:-1]
            volatility = np.std(returns)

            # Calculate range
            price_range = (highs[-lookback:].max() - lows[-lookback:].min()) / closes[-1]

            # Classify regime
            if abs(slope_pct) > 0.03 and volatility < 0.02:
                return MarketRegime.TRENDING_UP if slope_pct > 0 else MarketRegime.TRENDING_DOWN
            elif volatility > 0.05:
                return MarketRegime.HIGH_VOLATILITY
            elif price_range > 0.05:
                return MarketRegime.BREAKOUT
            elif volatility < 0.01:
                return MarketRegime.LOW_VOLATILITY
            else:
                return MarketRegime.RANGING

        except Exception as e:
            logger.warning(f"Market regime assessment error: {e}")
            return MarketRegime.RANGING

    async def _calculate_risk_metrics(self, symbol: str, entry_price: float, account_balance: float,
                                    atr_calc: ATRCalculation, market_regime: MarketRegime,
                                    mtf_analysis: Any = None, volume_analysis: Any = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""

        # Base ATR percentage
        atr_pct = atr_calc.value / entry_price

        # Adjust ATR based on market regime
        regime_multiplier = self.regime_adjustments[market_regime]['risk_multiplier']
        adjusted_atr_pct = atr_pct * regime_multiplier

        # Calculate volatility (simplified)
        volatility = min(1.0, adjusted_atr_pct * 10)  # Scale to 0-1

        # Determine risk level
        risk_level = self._assess_risk_level(volatility, market_regime)

        # Calculate position sizing metrics
        risk_per_trade = self.base_risk_per_trade * self.volatility_multipliers[risk_level]
        position_size_pct = min(self.max_risk_per_trade, risk_per_trade)

        # Stop loss percentage (1.5-3x ATR based on risk level)
        sl_multipliers = {
            RiskLevel.VERY_LOW: 1.5,
            RiskLevel.LOW: 2.0,
            RiskLevel.MODERATE: 2.5,
            RiskLevel.HIGH: 3.0,
            RiskLevel.VERY_HIGH: 3.5,
        }
        stop_loss_pct = adjusted_atr_pct * sl_multipliers[risk_level]

        # Take profit percentage (2-4x stop loss based on regime)
        tp_multipliers = {
            MarketRegime.TRENDING_UP: 4.0,
            MarketRegime.TRENDING_DOWN: 4.0,
            MarketRegime.RANGING: 2.5,
            MarketRegime.HIGH_VOLATILITY: 2.0,
            MarketRegime.LOW_VOLATILITY: 3.5,
            MarketRegime.BREAKOUT: 5.0,
        }
        take_profit_pct = stop_loss_pct * tp_multipliers[market_regime]

        # Calculate Sharpe ratio (simplified placeholder)
        sharpe_ratio = 1.5  # Placeholder

        # Max drawdown (simplified)
        max_drawdown = stop_loss_pct * 2

        # Win rate estimate based on regime
        win_rate_estimates = {
            MarketRegime.TRENDING_UP: 0.65,
            MarketRegime.TRENDING_DOWN: 0.65,
            MarketRegime.RANGING: 0.55,
            MarketRegime.HIGH_VOLATILITY: 0.50,
            MarketRegime.LOW_VOLATILITY: 0.60,
            MarketRegime.BREAKOUT: 0.70,
        }
        win_rate = win_rate_estimates[market_regime]

        # Risk-reward ratio
        risk_reward_ratio = take_profit_pct / stop_loss_pct

        return RiskMetrics(
            symbol=symbol,
            volatility=volatility,
            atr_value=atr_calc.value,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            risk_reward_ratio=risk_reward_ratio,
            position_size_pct=position_size_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            market_regime=market_regime,
            risk_level=risk_level,
            last_updated=datetime.now()
        )

    def _assess_risk_level(self, volatility: float, market_regime: MarketRegime) -> RiskLevel:
        """Assess overall risk level"""

        # Base risk from volatility
        if volatility < 0.1:
            base_risk = RiskLevel.VERY_LOW
        elif volatility < 0.2:
            base_risk = RiskLevel.LOW
        elif volatility < 0.3:
            base_risk = RiskLevel.MODERATE
        elif volatility < 0.5:
            base_risk = RiskLevel.HIGH
        else:
            base_risk = RiskLevel.VERY_HIGH

        # Adjust for market regime
        regime_adjustments = {
            MarketRegime.HIGH_VOLATILITY: 1,  # Increase risk level
            MarketRegime.LOW_VOLATILITY: -1,  # Decrease risk level
            MarketRegime.BREAKOUT: 1,        # Increase risk level
        }

        adjustment = regime_adjustments.get(market_regime, 0)
        final_risk_level = max(1, min(5, base_risk.value + adjustment))

        return RiskLevel(final_risk_level)

    def _calculate_adaptive_position_size(self, risk_metrics: RiskMetrics,
                                       account_balance: float, entry_price: float) -> Tuple[float, float]:
        """Calculate adaptive position size and leverage"""

        # Risk-based position size
        risk_amount = account_balance * risk_metrics.position_size_pct
        stop_loss_distance = entry_price * risk_metrics.stop_loss_pct

        if stop_loss_distance > 0:
            position_value = risk_amount / stop_loss_distance
        else:
            position_value = risk_amount / (entry_price * 0.02)  # Fallback 2% stop

        # Apply leverage cap based on market regime
        regime_config = self.regime_adjustments[risk_metrics.market_regime]
        leverage_cap = regime_config['leverage_cap']

        # Calculate leverage needed
        required_leverage = min(leverage_cap, (position_value / account_balance) * 100)

        # Ensure leverage is reasonable
        leverage = max(1, min(leverage_cap, required_leverage))

        # Adjust position size based on leverage
        final_position_size = (account_balance * leverage * risk_metrics.position_size_pct) / entry_price

        return final_position_size, leverage

    def _calculate_dynamic_stop_loss(self, entry_price: float, side: str,
                                   atr_calc: ATRCalculation, market_regime: MarketRegime,
                                   volatility: float) -> float:
        """Calculate dynamic stop loss based on ATR and market conditions"""

        # Base ATR percentage
        atr_pct = atr_calc.value / entry_price

        # Adjust multiplier based on market regime
        regime_multipliers = {
            MarketRegime.TRENDING_UP: 2.0,
            MarketRegime.TRENDING_DOWN: 2.0,
            MarketRegime.RANGING: 1.5,
            MarketRegime.HIGH_VOLATILITY: 2.5,
            MarketRegime.LOW_VOLATILITY: 1.2,
            MarketRegime.BREAKOUT: 3.0,
        }

        multiplier = regime_multipliers[market_regime]
        stop_loss_pct = atr_pct * multiplier

        # Ensure minimum stop loss
        min_sl_pct = 0.005  # 0.5% minimum
        stop_loss_pct = max(min_sl_pct, stop_loss_pct)

        # Apply stop loss
        if side == 'buy':
            return entry_price * (1 - stop_loss_pct)
        else:
            return entry_price * (1 + stop_loss_pct)

    def _calculate_adaptive_take_profits(self, entry_price: float, side: str,
                                       atr_calc: ATRCalculation, market_regime: MarketRegime,
                                       risk_metrics: RiskMetrics) -> List[float]:
        """Calculate adaptive take profit levels"""

        levels = []

        # Base ATR percentage
        atr_pct = atr_calc.value / entry_price

        # Different multipliers for different target levels
        if market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # Aggressive targets for trending markets
            tp_multipliers = [2.5, 4.0, 6.0]
        elif market_regime == MarketRegime.BREAKOUT:
            # Very aggressive for breakouts
            tp_multipliers = [3.0, 5.0, 8.0]
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            # Conservative for low volatility
            tp_multipliers = [2.0, 3.0, 4.5]
        else:
            # Moderate for ranging/high volatility
            tp_multipliers = [2.0, 3.5, 5.0]

        for multiplier in tp_multipliers:
            tp_pct = atr_pct * multiplier
            if side == 'buy':
                tp_price = entry_price * (1 + tp_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)
            levels.append(tp_price)

        return levels

    def _configure_adaptive_trailing_stop(self, entry_price: float, side: str,
                                        atr_calc: ATRCalculation, market_regime: MarketRegime) -> Dict[str, Any]:
        """Configure adaptive trailing stop"""

        # Base trailing percentage on ATR
        atr_pct = atr_calc.value / entry_price

        # Adjust based on market regime
        regime_trailing_multipliers = {
            MarketRegime.TRENDING_UP: 1.5,
            MarketRegime.TRENDING_DOWN: 1.5,
            MarketRegime.RANGING: 1.0,
            MarketRegime.HIGH_VOLATILITY: 2.0,
            MarketRegime.LOW_VOLATILITY: 1.2,
            MarketRegime.BREAKOUT: 2.5,
        }

        trailing_pct = atr_pct * regime_trailing_multipliers[market_regime]

        # Minimum trailing stop
        trailing_pct = max(0.008, trailing_pct)  # 0.8% minimum

        return {
            'activation_pct': trailing_pct * 1.5,  # Activate after 1.5x trailing distance profit
            'trailing_pct': trailing_pct,
            'side': side,
            'entry_price': entry_price,
            'current_trail_price': entry_price,
            'activated': False,
            'adaptive': True
        }

    def _calculate_max_holding_time(self, market_regime: MarketRegime, mtf_analysis: Any = None) -> timedelta:
        """Calculate maximum holding time based on market regime"""

        base_hours = {
            MarketRegime.TRENDING_UP: 48,
            MarketRegime.TRENDING_DOWN: 48,
            MarketRegime.RANGING: 12,
            MarketRegime.HIGH_VOLATILITY: 6,
            MarketRegime.LOW_VOLATILITY: 72,
            MarketRegime.BREAKOUT: 24,
        }

        hours = base_hours[market_regime]

        # Adjust based on multi-timeframe analysis if available
        if mtf_analysis and hasattr(mtf_analysis, 'risk_adjustment_factor'):
            risk_factor = mtf_analysis.risk_adjustment_factor
            hours = int(hours / risk_factor)  # Shorter holding for higher risk

        return timedelta(hours=hours)

    def _calculate_risk_adjustment_factor(self, risk_metrics: RiskMetrics,
                                        market_regime: MarketRegime) -> float:
        """Calculate overall risk adjustment factor"""

        # Base factor from risk level
        base_factors = {
            RiskLevel.VERY_LOW: 0.8,
            RiskLevel.LOW: 0.9,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.HIGH: 1.2,
            RiskLevel.VERY_HIGH: 1.5,
        }

        factor = base_factors[risk_metrics.risk_level]

        # Adjust for market regime
        regime_risk_multiplier = self.regime_adjustments[market_regime]['risk_multiplier']
        factor *= regime_risk_multiplier

        return min(2.0, max(0.5, factor))

    def _create_fallback_risk_plan(self, symbol: str, side: str, entry_price: float,
                                 account_balance: float) -> AdaptiveRiskPlan:
        """Create fallback risk plan when analysis fails"""

        logger.warning(f"Using fallback risk plan for {symbol}")

        # Conservative fallback parameters
        position_size = (account_balance * 0.01) / entry_price  # 1% risk
        leverage = 5  # Conservative leverage

        # Simple stop loss and take profit
        stop_loss_pct = 0.02  # 2%
        take_profit_levels = [
            entry_price * (1 + stop_loss_pct * 2),   # 2:1 reward
            entry_price * (1 + stop_loss_pct * 3),   # 3:1 reward
        ]

        if side == 'buy':
            stop_loss_price = entry_price * (1 - stop_loss_pct)
        else:
            stop_loss_price = entry_price * (1 + stop_loss_pct)

        return AdaptiveRiskPlan(
            symbol=symbol,
            entry_price=entry_price,
            position_size=position_size,
            leverage=leverage,
            stop_loss_price=stop_loss_price,
            take_profit_levels=take_profit_levels,
            trailing_stop_config={
                'activation_pct': 0.03,
                'trailing_pct': 0.015,
                'activated': False
            },
            max_loss_pct=0.02,
            max_holding_time=timedelta(hours=24),
            risk_adjustment_factor=1.2,
            market_regime=MarketRegime.RANGING,
            volatility_adjustment=1.0,
            dynamic_tp_enabled=False,
            emergency_stop_enabled=True,
            created_at=datetime.now(),
            last_adjusted=datetime.now()
        )

    async def evaluate_portfolio_risk(self, positions: List[Dict[str, Any]],
                                    account_balance: float) -> PortfolioRisk:
        """
        Evaluate portfolio-level risk

        Args:
            positions: List of current positions
            account_balance: Current account balance

        Returns:
            Portfolio risk assessment
        """

        try:
            total_positions = len(positions)
            total_exposure = sum(pos.get('size', 0) * pos.get('entry_price', 1) for pos in positions)

            # Calculate exposure as percentage of account
            exposure_pct = total_exposure / account_balance if account_balance > 0 else 0

            # Simple concentration risk (largest position / total exposure)
            if positions:
                position_sizes = [pos.get('size', 0) * pos.get('entry_price', 1) for pos in positions]
                concentration_risk = max(position_sizes) / total_exposure if total_exposure > 0 else 0
            else:
                concentration_risk = 0

            # Placeholder values for other risks (would need more data)
            correlation_risk = 0.5  # Medium correlation risk
            liquidity_risk = 0.3    # Low liquidity risk

            return PortfolioRisk(
                total_exposure=exposure_pct,
                total_positions=total_positions,
                max_exposure_limit=self.portfolio_risk_limit,
                daily_loss_limit=self.daily_loss_limit,
                current_daily_loss=0.0,  # Would need P&L tracking
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.error(f"Portfolio risk evaluation failed: {e}")
            return PortfolioRisk(
                total_exposure=0.0,
                total_positions=0,
                max_exposure_limit=self.portfolio_risk_limit,
                daily_loss_limit=self.daily_loss_limit,
                current_daily_loss=0.0,
                correlation_risk=0.0,
                concentration_risk=0.0,
                liquidity_risk=0.0,
                last_updated=datetime.now()
            )

    async def get_adaptive_action(self, plan: AdaptiveRiskPlan, current_price: float,
                                time_held: timedelta, unrealized_pnl: float) -> AdaptiveAction:
        """
        Get adaptive risk management action based on current conditions

        Args:
            plan: Current risk plan
            current_price: Current market price
            time_held: Time position has been held
            unrealized_pnl: Unrealized profit/loss

        Returns:
            Recommended adaptive action
        """

        try:
            # Calculate current P&L percentage
            pnl_pct = unrealized_pnl / (plan.entry_price * plan.position_size)

            # Check for emergency conditions
            if pnl_pct <= -plan.max_loss_pct:
                return AdaptiveAction.CLOSE_POSITION

            # Check holding time limit
            if time_held >= plan.max_holding_time:
                return AdaptiveAction.CLOSE_POSITION

            # Check for trailing stop activation
            trailing_config = plan.trailing_stop_config
            if not trailing_config.get('activated', False):
                activation_threshold = plan.entry_price * (1 + trailing_config.get('activation_pct', 0))
                if plan.entry_price < current_price <= activation_threshold:
                    return AdaptiveAction.TIGHTEN_STOPS

            # Profit-based adjustments
            if pnl_pct > 0.05:  # 5% profit
                return AdaptiveAction.INCREASE_LEVERAGE
            elif pnl_pct < -0.02:  # 2% loss
                return AdaptiveAction.DECREASE_POSITION

            # Default action
            return AdaptiveAction.HOLD_POSITION

        except Exception as e:
            logger.error(f"Adaptive action evaluation failed: {e}")
            return AdaptiveAction.HOLD_POSITION

# Global adaptive risk manager instance
adaptive_risk_manager = AdaptiveRiskManager()
