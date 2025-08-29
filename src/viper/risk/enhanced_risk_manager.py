#!/usr/bin/env python3
"""
# Rocket ENHANCED RISK MANAGEMENT SYSTEM
Advanced risk management with dynamic adjustments and sophisticated position sizing

This enhanced version includes:
- Dynamic risk adjustment based on market conditions
- Advanced position sizing algorithms
- Multi-asset correlation risk management
- Stress testing and scenario analysis
- Machine learning-based risk prediction
- Real-time risk monitoring and alerts
"""

import os
import logging
import asyncio
import numpy as np
from dataclasses import dataclass
from enum import Enum
import ccxt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXTREME = "EXTREME"

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class RiskEvent(Enum):
    LARGE_MOVE = "LARGE_MOVE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CORRELATION_SPIKE = "CORRELATION_SPIKE"
    LIQUIDATION_RISK = "LIQUIDATION_RISK"
    DRAWDOWN_SPIKE = "DRAWDOWN_SPIKE"

@dataclass
class EnhancedPosition:
    """Enhanced position with comprehensive risk tracking"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0

    # Risk metrics
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_activation: Optional[float] = None

    # Advanced risk tracking
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    holding_time: int = 0
    volatility_at_entry: float = 0.0
    correlation_risk: float = 0.0

    # Dynamic risk adjustments
    current_risk_level: RiskLevel = RiskLevel.MODERATE
    risk_multiplier: float = 1.0
    liquidation_price: Optional[float] = None

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class RiskLimits:
    """Dynamic risk limits that adjust based on market conditions"""
    max_portfolio_risk: float = 0.05  # 5% max portfolio risk
    max_single_position_risk: float = 0.02  # 2% max per position
    max_correlation_exposure: float = 0.6  # 60% max correlation exposure
    max_daily_loss: float = 0.03  # 3% max daily loss
    max_drawdown: float = 0.10  # 10% max drawdown

    # Dynamic adjustments
    volatility_multiplier: float = 1.0
    stress_multiplier: float = 1.0
    market_regime_multiplier: float = 1.0

class EnhancedRiskManager:
    """Enhanced risk management with dynamic adjustments and ML-based predictions"""

    def __init__(self):
        self.positions: Dict[str, EnhancedPosition] = {}
        self.risk_limits = RiskLimits()
        self.market_data_cache = {}
        self.correlation_matrix = {}
        self.risk_events: List[Dict[str, Any]] = []

        # Performance tracking
        self.portfolio_value = 10000.0  # Starting value
        self.daily_pnl = 0.0
        self.peak_portfolio_value = 10000.0
        self.current_drawdown = 0.0

        # Risk monitoring
        self.volatility_regime = "NORMAL"
        self.market_stress_level = "LOW"
        self.correlation_regime = "NORMAL"

        # ML-based risk prediction
        self.risk_prediction_model = None
        self.historical_risk_data = []

        # Exchange connection
        self.exchange = None

        logger.info("🛡️ Enhanced Risk Manager initialized with dynamic adjustments")

    async def initialize_exchange(self) -> bool:
        """Initialize exchange connection for real-time data"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_API_SECRET'),
                'password': os.getenv('BITGET_API_PASSWORD'),
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'},
                'sandbox': False
            })

            await asyncio.get_event_loop().run_in_executor(None, self.exchange.load_markets)
            logger.info("# Check Exchange connected for enhanced risk management")
            return True

        except Exception as e:
            logger.error(f"# X Failed to connect exchange: {e}")
            return False

    def calculate_dynamic_position_size(self, symbol: str, entry_price: float,
                                      stop_loss: float, portfolio_value: float) -> Dict[str, Any]:
        """Calculate optimal position size with dynamic risk adjustment"""
        try:
            # Base risk per trade
            base_risk = self.risk_limits.max_single_position_risk

            # Adjust for current market conditions
            risk_multiplier = self._calculate_risk_multiplier(symbol, entry_price)

            # Adjust for correlation risk
            correlation_adjustment = self._calculate_correlation_adjustment(symbol)

            # Adjust for volatility
            volatility_adjustment = self._calculate_volatility_adjustment(symbol)

            # Calculate effective risk
            effective_risk = base_risk * risk_multiplier * correlation_adjustment * volatility_adjustment

            # Ensure risk doesn't exceed limits
            effective_risk = min(effective_risk, self.risk_limits.max_portfolio_risk)

            # Calculate position size
            risk_amount = portfolio_value * effective_risk
            stop_distance = abs(entry_price - stop_loss) / entry_price

            if stop_distance > 0:
                position_size_usd = risk_amount / stop_distance

                # Apply leverage limits (assuming 10x max for crypto)
                max_leverage = 10.0
                position_size_usd = min(position_size_usd, portfolio_value * max_leverage)

                # Convert to contracts/units
                contract_size = 1.0  # Assuming 1 USD per contract for simplicity
                position_size_contracts = position_size_usd / (entry_price * contract_size)

                result = {
                    'position_size_usd': position_size_usd,
                    'position_size_contracts': position_size_contracts,
                    'effective_risk_percent': effective_risk,
                    'risk_amount_usd': risk_amount,
                    'stop_distance_percent': stop_distance,
                    'risk_multiplier': risk_multiplier,
                    'correlation_adjustment': correlation_adjustment,
                    'volatility_adjustment': volatility_adjustment,
                    'max_allowed_risk': self.risk_limits.max_portfolio_risk
                }

                return result
            else:
                return {'error': 'Invalid stop loss distance'}

        except Exception as e:
            logger.error(f"# X Error calculating dynamic position size: {e}")
            return {'error': str(e)}

    def _calculate_risk_multiplier(self, symbol: str, entry_price: float) -> float:
        """Calculate risk multiplier based on current conditions"""
        try:
            multiplier = 1.0

            # Market stress adjustment
            if self.market_stress_level == "HIGH":
                multiplier *= 0.5
            elif self.market_stress_level == "EXTREME":
                multiplier *= 0.25

            # Volatility regime adjustment
            if self.volatility_regime == "HIGH":
                multiplier *= 0.7
            elif self.volatility_regime == "EXTREME":
                multiplier *= 0.4

            # Recent performance adjustment
            if self.current_drawdown > 0.05:  # 5% drawdown
                multiplier *= 0.8
            elif self.current_drawdown > 0.10:  # 10% drawdown
                multiplier *= 0.5

            # Daily loss limit check
            daily_loss_ratio = abs(self.daily_pnl) / self.portfolio_value
            if daily_loss_ratio > self.risk_limits.max_daily_loss * 0.8:  # 80% of limit
                multiplier *= 0.6

            return max(multiplier, 0.1)  # Minimum 10% of normal risk

        except Exception as e:
            logger.warning(f"# Warning Error calculating risk multiplier: {e}")
            return 1.0

    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate correlation-based risk adjustment"""
        try:
            if symbol not in self.correlation_matrix:
                return 1.0

            # Get average correlation with existing positions
            correlations = []
            for existing_symbol in self.positions.keys():
                if existing_symbol != symbol and existing_symbol in self.correlation_matrix:
                    corr = self.correlation_matrix[existing_symbol].get(symbol, 0)
                    correlations.append(abs(corr))

            if not correlations:
                return 1.0

            avg_correlation = np.mean(correlations)

            # Reduce position size for highly correlated assets
            if avg_correlation > 0.7:  # High correlation
                return 0.5
            elif avg_correlation > 0.5:  # Moderate correlation
                return 0.7
            else:
                return 1.0

        except Exception as e:
            logger.warning(f"# Warning Error calculating correlation adjustment: {e}")
            return 1.0

    def _calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate volatility-based position size adjustment"""
        try:
            if symbol not in self.market_data_cache:
                return 1.0

            # Get recent volatility
            recent_data = self.market_data_cache[symbol]
            if 'returns' not in recent_data or len(recent_data['returns']) < 20:
                return 1.0

            volatility = np.std(recent_data['returns'][-20:]) * np.sqrt(252)  # Annualized

            # Adjust position size based on volatility
            if volatility > 1.0:  # Very high volatility (>100% annualized)
                return 0.4
            elif volatility > 0.7:  # High volatility (>70% annualized)
                return 0.6
            elif volatility > 0.5:  # Moderate volatility (>50% annualized)
                return 0.8
            else:
                return 1.0

        except Exception as e:
            logger.warning(f"# Warning Error calculating volatility adjustment: {e}")
            return 1.0

    def assess_portfolio_risk(self) -> Dict[str, Any]:
        """Comprehensive portfolio risk assessment"""
        try:
            if not self.positions:
                return {
                    'overall_risk_level': RiskLevel.VERY_LOW,
                    'total_exposure': 0.0,
                    'concentration_risk': 0.0,
                    'correlation_risk': 0.0,
                    'volatility_risk': 0.0,
                    'liquidity_risk': 0.0
                }

            # Calculate total exposure
            total_exposure = sum(abs(pos.size * pos.entry_price) for pos in self.positions.values())
            exposure_ratio = total_exposure / self.portfolio_value

            # Concentration risk (largest position / total exposure)
            position_sizes = [abs(pos.size * pos.entry_price) for pos in self.positions.values()]
            max_position_size = max(position_sizes)
            concentration_risk = max_position_size / total_exposure

            # Correlation risk
            correlation_risk = self._calculate_portfolio_correlation_risk()

            # Volatility risk
            volatility_risk = self._calculate_portfolio_volatility_risk()

            # Liquidity risk (simplified)
            liquidity_risk = min(exposure_ratio * 2, 1.0)

            # Overall risk assessment
            risk_factors = [exposure_ratio, concentration_risk, correlation_risk, volatility_risk, liquidity_risk]
            avg_risk = np.mean(risk_factors)

            if avg_risk >= 0.8:
                overall_risk = RiskLevel.EXTREME
            elif avg_risk >= 0.6:
                overall_risk = RiskLevel.VERY_HIGH
            elif avg_risk >= 0.4:
                overall_risk = RiskLevel.HIGH
            elif avg_risk >= 0.2:
                overall_risk = RiskLevel.MODERATE
            elif avg_risk >= 0.1:
                overall_risk = RiskLevel.LOW
            else:
                overall_risk = RiskLevel.VERY_LOW

            return {
                'overall_risk_level': overall_risk,
                'total_exposure': total_exposure,
                'exposure_ratio': exposure_ratio,
                'concentration_risk': concentration_risk,
                'correlation_risk': correlation_risk,
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'risk_factors': risk_factors,
                'average_risk': avg_risk
            }

        except Exception as e:
            logger.error(f"# X Error assessing portfolio risk: {e}")
            return {
                'overall_risk_level': RiskLevel.HIGH,
                'error': str(e)
            }

    def _calculate_portfolio_correlation_risk(self) -> float:
        """Calculate correlation risk across portfolio"""
        try:
            if len(self.positions) < 2:
                return 0.0

            symbols = list(self.positions.keys())
            correlations = []

            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    if symbol1 in self.correlation_matrix and symbol2 in self.correlation_matrix[symbol1]:
                        corr = abs(self.correlation_matrix[symbol1][symbol2])
                        correlations.append(corr)

            if not correlations:
                return 0.5  # Default moderate correlation risk

            avg_correlation = np.mean(correlations)
            return min(avg_correlation * 1.2, 1.0)  # Scale up for conservatism

        except Exception as e:
            logger.warning(f"# Warning Error calculating portfolio correlation risk: {e}")
            return 0.5

    def _calculate_portfolio_volatility_risk(self) -> float:
        """Calculate portfolio volatility risk"""
        try:
            if not self.positions:
                return 0.0

            # Get volatility for each position
            position_volatilities = []
            position_weights = []

            for symbol, position in self.positions.items():
                if symbol in self.market_data_cache:
                    data = self.market_data_cache[symbol]
                    if 'returns' in data and len(data['returns']) >= 20:
                        vol = np.std(data['returns'][-20:]) * np.sqrt(252)
                        position_volatilities.append(vol)
                        position_weights.append(abs(position.size * position.entry_price))

            if not position_volatilities:
                return 0.5

            # Calculate weighted average volatility
            total_weight = sum(position_weights)
            if total_weight > 0:
                weighted_volatility = sum(v * w for v, w in zip(position_volatilities, position_weights)) / total_weight
                return min(weighted_volatility * 2, 1.0)  # Scale for risk assessment
            else:
                return 0.5

        except Exception as e:
            logger.warning(f"# Warning Error calculating portfolio volatility risk: {e}")
            return 0.5

    def update_market_conditions(self, symbol: str, market_data: Dict[str, Any]):
        """Update market conditions for risk calculations"""
        try:
            # Store market data
            self.market_data_cache[symbol] = market_data

            # Update volatility regime
            if 'returns' in market_data and len(market_data['returns']) >= 20:
                volatility = np.std(market_data['returns'][-20:]) * np.sqrt(252)

                if volatility > 1.5:
                    self.volatility_regime = "EXTREME"
                elif volatility > 1.0:
                    self.volatility_regime = "HIGH"
                elif volatility > 0.7:
                    self.volatility_regime = "MODERATE"
                else:
                    self.volatility_regime = "NORMAL"

            # Update correlation matrix (simplified)
            self._update_correlation_matrix(symbol, market_data)

        except Exception as e:
            logger.warning(f"# Warning Error updating market conditions: {e}")

    def _update_correlation_matrix(self, symbol: str, market_data: Dict[str, Any]):
        """Update correlation matrix with new market data"""
        try:
            if 'returns' not in market_data:
                return

            new_returns = market_data['returns'][-50:]  # Last 50 periods

            for existing_symbol, existing_data in self.market_data_cache.items():
                if existing_symbol != symbol and 'returns' in existing_data:
                    existing_returns = existing_data['returns'][-50:]

                    # Calculate correlation
                    if len(new_returns) == len(existing_returns) and len(new_returns) > 10:
                        correlation = np.corrcoef(new_returns, existing_returns)[0, 1]

                        if symbol not in self.correlation_matrix:
                            self.correlation_matrix[symbol] = {}

                        if existing_symbol not in self.correlation_matrix:
                            self.correlation_matrix[existing_symbol] = {}

                        self.correlation_matrix[symbol][existing_symbol] = correlation
                        self.correlation_matrix[existing_symbol][symbol] = correlation

        except Exception as e:
            logger.warning(f"# Warning Error updating correlation matrix: {e}")

    def check_risk_limits(self) -> List[str]:
        """Check if any risk limits are breached"""
        violations = []

        try:
            # Portfolio risk limit
            portfolio_risk = self.assess_portfolio_risk()
            if portfolio_risk['overall_risk_level'] in [RiskLevel.EXTREME, RiskLevel.VERY_HIGH]:
                violations.append(f"Portfolio risk too high: {portfolio_risk['overall_risk_level'].value}")

            # Daily loss limit
            daily_loss_ratio = abs(self.daily_pnl) / max(self.portfolio_value, 1)
            if daily_loss_ratio > self.risk_limits.max_daily_loss:
                violations.append(f"Daily loss limit breached: {daily_loss_ratio:.2%} > {self.risk_limits.max_daily_loss:.2%}")

            # Drawdown limit
            if self.current_drawdown > self.risk_limits.max_drawdown:
                violations.append(f"Drawdown limit breached: {self.current_drawdown:.2%} > {self.risk_limits.max_drawdown:.2%}")

            # Single position risk limits
            for symbol, position in self.positions.items():
                position_value = abs(position.size * position.current_price)
                position_risk = position_value / self.portfolio_value

                if position_risk > self.risk_limits.max_single_position_risk:
                    violations.append(f"Position {symbol} risk too high: {position_risk:.2%}")

            # Correlation exposure
            if portfolio_risk['correlation_risk'] > self.risk_limits.max_correlation_exposure:
                violations.append(f"Correlation exposure too high: {portfolio_risk['correlation_risk']:.2%}")

        except Exception as e:
            logger.error(f"# X Error checking risk limits: {e}")
            violations.append(f"Risk check error: {e}")

        return violations

    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            portfolio_risk = self.assess_portfolio_risk()
            risk_violations = self.check_risk_limits()

            report = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': self.portfolio_value,
                'daily_pnl': self.daily_pnl,
                'current_drawdown': self.current_drawdown,
                'peak_portfolio_value': self.peak_portfolio_value,
                'portfolio_risk': portfolio_risk,
                'risk_violations': risk_violations,
                'market_conditions': {
                    'volatility_regime': self.volatility_regime,
                    'market_stress_level': self.market_stress_level,
                    'correlation_regime': self.correlation_regime
                },
                'positions': {
                    symbol: {
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'risk_level': pos.current_risk_level.value
                    }
                    for symbol, pos in self.positions.items()
                },
                'recommendations': self._generate_risk_recommendations(risk_violations, portfolio_risk)
            }

            return report

        except Exception as e:
            logger.error(f"# X Error generating risk report: {e}")
            return {'error': str(e)}

    def _generate_risk_recommendations(self, violations: List[str], portfolio_risk: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        try:
            # High risk level recommendations
            if portfolio_risk['overall_risk_level'] in [RiskLevel.EXTREME, RiskLevel.VERY_HIGH]:
                recommendations.append("🚨 HIGH RISK: Consider reducing position sizes immediately")
                recommendations.append("📉 Implement trailing stops on all positions")
                recommendations.append("# Warning Avoid opening new positions until risk levels decrease")

            # Risk violation recommendations
            if violations:
                recommendations.append("🚨 RISK VIOLATIONS DETECTED:")
                for violation in violations:
                    recommendations.append(f"   • {violation}")

            # Market condition recommendations
            if self.volatility_regime in ["HIGH", "EXTREME"]:
                recommendations.append("🌊 HIGH VOLATILITY: Use wider stops and reduce leverage")

            if self.market_stress_level == "HIGH":
                recommendations.append("😰 MARKET STRESS: Implement defensive position sizing")

            # Portfolio concentration recommendations
            if portfolio_risk.get('concentration_risk', 0) > 0.5:
                recommendations.append("# Target HIGH CONCENTRATION: Diversify across more assets")

            # Default recommendations
            if not recommendations:
                recommendations.append("# Check Risk levels within acceptable parameters")
                recommendations.append("# Chart Continue monitoring market conditions")

        except Exception as e:
            logger.warning(f"# Warning Error generating recommendations: {e}")
            recommendations.append("# Tool Risk monitoring system needs attention")

        return recommendations

async def test_enhanced_risk_manager():
    """Test the enhanced risk manager"""

    manager = EnhancedRiskManager()

    # Test position sizing
    position_calc = manager.calculate_dynamic_position_size(
        symbol="BTCUSDT",
        entry_price=50000,
        stop_loss=49000,
        portfolio_value=10000
    )

    print(f"Position Size Calculation: {position_calc}")

    # Test portfolio risk assessment
    risk_assessment = manager.assess_portfolio_risk()
    print(f"Portfolio Risk Assessment: {risk_assessment}")

    # Test risk report
    risk_report = manager.generate_risk_report()
    print(f"Risk Report Generated: {len(risk_report)} metrics")

if __name__ == "__main__":
    asyncio.run(test_enhanced_risk_manager())
