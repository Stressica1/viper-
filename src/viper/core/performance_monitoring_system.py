#!/usr/bin/env python3
"""
# Rocket PERFORMANCE MONITORING AND OPTIMIZATION SYSTEM
Comprehensive monitoring, analytics, and automated optimization for trading systems

This system includes:
- Real-time performance tracking
- Automated strategy optimization
- Risk-adjusted performance metrics
- Performance benchmarking
- Predictive performance analytics
- Automated parameter tuning
- Performance alerting and notifications
"""

import os
import json
import time
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import psutil
import gc
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    EXPECTANCY = "expectancy"
    RECOVERY_FACTOR = "recovery_factor"

class OptimizationTarget(Enum):
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_WIN_RATE = "maximize_win_rate"
    MAXIMIZE_PROFIT_FACTOR = "maximize_profit_factor"
    BALANCED_OPTIMIZATION = "balanced_optimization"

@dataclass
class PerformanceSnapshot:
    """Performance snapshot with comprehensive metrics"""
    timestamp: datetime
    portfolio_value: float
    daily_pnl: float
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    trades_executed: int
    active_positions: int
    system_metrics: Dict[str, Any]

@dataclass
class OptimizationResult:
    """Result of optimization run"""
    timestamp: datetime
    target: OptimizationTarget
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    improvement: float
    confidence: float

class PerformanceMonitoringSystem:
    """Comprehensive performance monitoring and optimization system"""

    def __init__(self, history_size: int = 10000, optimization_interval: int = 3600):
        self.history_size = history_size
        self.optimization_interval = optimization_interval

        # Performance data storage
        self.performance_history = deque(maxlen=history_size)
        self.trade_history = deque(maxlen=history_size)
        self.system_metrics_history = deque(maxlen=history_size)

        # Optimization tracking
        self.optimization_results = []
        self.current_parameters = {}
        self.baseline_performance = {}

        # Real-time monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alerts = []

        # Performance targets and thresholds
        self.performance_targets = {
            'min_sharpe_ratio': 1.0,
            'max_drawdown_limit': 0.15,
            'min_win_rate': 0.55,
            'min_profit_factor': 1.2,
            'max_volatility': 0.25
        }

        # Optimization settings
        self.optimization_settings = {
            'parameter_ranges': {
                'risk_per_trade': [0.005, 0.02],
                'take_profit_pct': [1.0, 5.0],
                'stop_loss_pct': [1.0, 5.0],
                'max_positions': [5, 20],
                'min_viper_score': [60.0, 90.0]
            },
            'optimization_algorithm': 'bayesian',
            'max_iterations': 50,
            'early_stopping_patience': 10
        }

        # Predictive analytics
        self.performance_predictor = None
        self.risk_predictor = None

        logger.info("# Chart Performance Monitoring System initialized")

    def start_monitoring(self):
        """Start real-time performance monitoring"""
        try:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()

            logger.info("# Check Performance monitoring started")

        except Exception as e:
            logger.error(f"# X Error starting monitoring: {e}")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        try:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)

            logger.info("🛑 Performance monitoring stopped")

        except Exception as e:
            logger.error(f"# X Error stopping monitoring: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect current performance snapshot
                snapshot = self._collect_performance_snapshot()

                if snapshot:
                    self.performance_history.append(snapshot)

                    # Check for alerts
                    self._check_performance_alerts(snapshot)

                    # Update predictive models
                    self._update_predictive_models()

                # Sleep for monitoring interval
                time.sleep(60)  # 1 minute intervals

            except Exception as e:
                logger.error(f"# X Error in monitoring loop: {e}")
                time.sleep(60)

    def _collect_performance_snapshot(self) -> Optional[PerformanceSnapshot]:
        """Collect current performance snapshot"""
        try:
            # This would integrate with your trading system
            # For now, we'll create a mock snapshot
            current_time = datetime.now()

            snapshot = PerformanceSnapshot(
                timestamp=current_time,
                portfolio_value=10000.0,  # Would get from trading system
                daily_pnl=0.0,
                total_pnl=0.0,
                win_rate=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                trades_executed=0,
                active_positions=0,
                system_metrics=self._collect_system_metrics()
            )

            return snapshot

        except Exception as e:
            logger.error(f"# X Error collecting performance snapshot: {e}")
            return None

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_connections': len(psutil.net_connections()),
                'active_threads': threading.active_count(),
                'gc_collections': gc.get_stats(),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"# Warning Error collecting system metrics: {e}")
            return {}

    def _check_performance_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance alerts"""
        try:
            alerts = []

            # Sharpe ratio alert
            if snapshot.sharpe_ratio < self.performance_targets['min_sharpe_ratio']:
                alerts.append({
                    'type': 'SHARPE_RATIO_LOW',
                    'message': f'Sharpe ratio {snapshot.sharpe_ratio:.2f} below target {self.performance_targets["min_sharpe_ratio"]:.2f}',
                    'severity': 'HIGH',
                    'timestamp': snapshot.timestamp
                })

            # Drawdown alert
            if snapshot.max_drawdown > self.performance_targets['max_drawdown_limit']:
                alerts.append({
                    'type': 'DRAWDOWN_HIGH',
                    'message': f'Max drawdown {snapshot.max_drawdown:.2%} exceeds limit {self.performance_targets["max_drawdown_limit"]:.2%}',
                    'severity': 'CRITICAL',
                    'timestamp': snapshot.timestamp
                })

            # Win rate alert
            if snapshot.win_rate < self.performance_targets['min_win_rate']:
                alerts.append({
                    'type': 'WIN_RATE_LOW',
                    'message': f'Win rate {snapshot.win_rate:.2%} below target {self.performance_targets["min_win_rate"]:.2%}',
                    'severity': 'MEDIUM',
                    'timestamp': snapshot.timestamp
                })

            # System resource alerts
            system_metrics = snapshot.system_metrics
            if system_metrics.get('memory_percent', 0) > 90:
                alerts.append({
                    'type': 'HIGH_MEMORY_USAGE',
                    'message': f'Memory usage at {system_metrics["memory_percent"]:.1f}%',
                    'severity': 'HIGH',
                    'timestamp': snapshot.timestamp
                })

            # Add alerts to queue
            self.alerts.extend(alerts)

            # Keep only recent alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]

        except Exception as e:
            logger.error(f"# X Error checking performance alerts: {e}")

    def calculate_advanced_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        try:
            if len(returns) < 10:
                return {}

            # Basic metrics
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            sharpe_ratio = (returns.mean() * 252 - risk_free_rate) / volatility if volatility > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (returns.mean() * 252 - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calmar ratio
            calmar_ratio = -total_return / max_drawdown if max_drawdown < 0 else 0

            # Win rate and profit factor
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]

            win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0

            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if avg_loss > 0 and len(losing_trades) > 0 else 0

            # Expectancy
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

            # Recovery factor
            recovery_factor = -total_return / max_drawdown if max_drawdown < 0 else 0

            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'recovery_factor': recovery_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }

        except Exception as e:
            logger.error(f"# X Error calculating advanced metrics: {e}")
            return {}

    def optimize_strategy_parameters(self, target: OptimizationTarget = OptimizationTarget.BALANCED_OPTIMIZATION) -> Optional[OptimizationResult]:
        """Optimize strategy parameters using historical performance"""
        try:
            logger.info(f"# Target Starting parameter optimization for target: {target.value}")

            if len(self.performance_history) < 50:
                logger.warning("# Warning Insufficient performance history for optimization")
                return None

            # Prepare optimization data
            optimization_data = self._prepare_optimization_data()

            if not optimization_data:
                return None

            # Run optimization based on target
            if target == OptimizationTarget.MAXIMIZE_SHARPE:
                result = self._optimize_for_sharpe(optimization_data)
            elif target == OptimizationTarget.MINIMIZE_DRAWDOWN:
                result = self._optimize_for_drawdown(optimization_data)
            elif target == OptimizationTarget.MAXIMIZE_WIN_RATE:
                result = self._optimize_for_win_rate(optimization_data)
            elif target == OptimizationTarget.MAXIMIZE_PROFIT_FACTOR:
                result = self._optimize_for_profit_factor(optimization_data)
            else:
                result = self._optimize_balanced(optimization_data)

            if result:
                self.optimization_results.append(result)

                # Keep only recent results
                if len(self.optimization_results) > 20:
                    self.optimization_results = self.optimization_results[-20:]

                logger.info(f"# Check Optimization completed: {result.improvement:.2%} improvement")
                return result

            return None

        except Exception as e:
            logger.error(f"# X Error in parameter optimization: {e}")
            return None

    def _prepare_optimization_data(self) -> Optional[Dict[str, Any]]:
        """Prepare data for optimization"""
        try:
            if len(self.performance_history) < 20:
                return None

            # Extract performance metrics
            performance_data = []
            for snapshot in list(self.performance_history)[-100:]:  # Last 100 snapshots
                performance_data.append({
                    'sharpe_ratio': snapshot.sharpe_ratio,
                    'max_drawdown': snapshot.max_drawdown,
                    'win_rate': snapshot.win_rate,
                    'profit_factor': getattr(snapshot, 'profit_factor', 1.0),
                    'volatility': snapshot.volatility,
                    'total_return': getattr(snapshot, 'total_return', 0.0)
                })

            return {
                'performance_data': performance_data,
                'current_parameters': self.current_parameters.copy(),
                'baseline_metrics': self.baseline_performance.copy()
            }

        except Exception as e:
            logger.error(f"# X Error preparing optimization data: {e}")
            return None

    def _optimize_for_sharpe(self, data: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize parameters to maximize Sharpe ratio"""
        try:
            # Simple parameter grid search (in production, use more sophisticated optimization)
            best_params = {}
            best_score = -float('inf')

            # Test different parameter combinations
            for risk_per_trade in np.linspace(0.005, 0.02, 5):
                for take_profit_pct in np.linspace(2.0, 4.0, 3):
                    for stop_loss_pct in np.linspace(1.0, 3.0, 3):

                        # Simulate performance with these parameters
                        simulated_score = self._simulate_performance_with_params(
                            data, risk_per_trade, take_profit_pct, stop_loss_pct
                        )

                        if simulated_score > best_score:
                            best_score = simulated_score
                            best_params = {
                                'risk_per_trade': risk_per_trade,
                                'take_profit_pct': take_profit_pct,
                                'stop_loss_pct': stop_loss_pct
                            }

            improvement = best_score - data['baseline_metrics'].get('sharpe_ratio', 0)

            return OptimizationResult(
                timestamp=datetime.now(),
                target=OptimizationTarget.MAXIMIZE_SHARPE,
                parameters=best_params,
                performance_metrics={'sharpe_ratio': best_score},
                improvement=improvement,
                confidence=0.8
            )

        except Exception as e:
            logger.error(f"# X Error optimizing for Sharpe ratio: {e}")
            return None

    def _optimize_for_drawdown(self, data: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize parameters to minimize drawdown"""
        try:
            # Focus on reducing volatility and position sizes
            best_params = {}
            best_score = float('inf')  # Minimize drawdown

            for risk_per_trade in np.linspace(0.005, 0.015, 5):  # Lower risk range
                for max_positions in [5, 10, 15]:

                    # Simulate performance
                    simulated_drawdown = self._simulate_drawdown_with_params(
                        data, risk_per_trade, max_positions
                    )

                    if simulated_drawdown < best_score:
                        best_score = simulated_drawdown
                        best_params = {
                            'risk_per_trade': risk_per_trade,
                            'max_positions': max_positions
                        }

            improvement = data['baseline_metrics'].get('max_drawdown', 0) - best_score

            return OptimizationResult(
                timestamp=datetime.now(),
                target=OptimizationTarget.MINIMIZE_DRAWDOWN,
                parameters=best_params,
                performance_metrics={'max_drawdown': best_score},
                improvement=improvement,
                confidence=0.75
            )

        except Exception as e:
            logger.error(f"# X Error optimizing for drawdown: {e}")
            return None

    def _optimize_balanced(self, data: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Balanced optimization across multiple metrics"""
        try:
            # Create composite score
            best_params = {}
            best_score = -float('inf')

            for risk_per_trade in np.linspace(0.008, 0.015, 4):
                for take_profit_pct in np.linspace(2.5, 3.5, 3):
                    for min_score in np.linspace(65, 80, 4):

                        # Calculate composite score
                        composite_score = self._calculate_composite_score(
                            data, risk_per_trade, take_profit_pct, min_score
                        )

                        if composite_score > best_score:
                            best_score = composite_score
                            best_params = {
                                'risk_per_trade': risk_per_trade,
                                'take_profit_pct': take_profit_pct,
                                'min_viper_score': min_score
                            }

            improvement = best_score - self._calculate_current_composite_score(data)

            return OptimizationResult(
                timestamp=datetime.now(),
                target=OptimizationTarget.BALANCED_OPTIMIZATION,
                parameters=best_params,
                performance_metrics={'composite_score': best_score},
                improvement=improvement,
                confidence=0.7
            )

        except Exception as e:
            logger.error(f"# X Error in balanced optimization: {e}")
            return None

    def _simulate_performance_with_params(self, data: Dict[str, Any], risk_per_trade: float,
                                        take_profit_pct: float, stop_loss_pct: float) -> float:
        """Simulate performance with given parameters"""
        # This is a simplified simulation - in production, use backtesting
        try:
            # Use historical data to estimate performance
            performance_data = data['performance_data']

            # Simple model: Sharpe ratio scales with risk management quality
            risk_management_quality = 1.0 - (risk_per_trade - 0.01)**2 / 0.0001  # Optimal around 1%
            profit_taking_quality = min(take_profit_pct / 3.0, 1.0)  # Optimal around 3%
            stop_loss_quality = min(stop_loss_pct / 2.0, 1.0)  # Optimal around 2%

            composite_quality = (risk_management_quality + profit_taking_quality + stop_loss_quality) / 3

            # Estimate Sharpe ratio
            base_sharpe = np.mean([p['sharpe_ratio'] for p in performance_data[-10:]])
            estimated_sharpe = base_sharpe * (0.8 + 0.4 * composite_quality)

            return max(estimated_sharpe, 0.1)

        except Exception as e:
            logger.warning(f"# Warning Error simulating performance: {e}")
            return 0.5

    def _simulate_drawdown_with_params(self, data: Dict[str, Any], risk_per_trade: float,
                                     max_positions: int) -> float:
        """Simulate drawdown with given parameters"""
        try:
            # Lower risk and fewer positions should reduce drawdown
            base_drawdown = np.mean([p['max_drawdown'] for p in data['performance_data'][-10:]])

            # Risk reduction factor
            risk_reduction = (0.01 - risk_per_trade) / 0.01  # More reduction with lower risk

            # Position concentration factor
            position_factor = max_positions / 15.0  # Normalize to 15 positions

            # Estimate new drawdown
            estimated_drawdown = base_drawdown * (1 - 0.3 * risk_reduction) * position_factor

            return max(estimated_drawdown, 0.01)

        except Exception as e:
            logger.warning(f"# Warning Error simulating drawdown: {e}")
            return 0.1

    def _calculate_composite_score(self, data: Dict[str, Any], risk_per_trade: float,
                                 take_profit_pct: float, min_score: float) -> float:
        """Calculate composite performance score"""
        try:
            # Normalize parameters to 0-1 scale
            risk_score = 1.0 - abs(risk_per_trade - 0.012) / 0.01
            tp_score = 1.0 - abs(take_profit_pct - 3.0) / 2.0
            score_threshold = min_score / 75.0

            # Get recent performance
            recent_perf = data['performance_data'][-5:]
            avg_sharpe = np.mean([p['sharpe_ratio'] for p in recent_perf])
            avg_win_rate = np.mean([p['win_rate'] for p in recent_perf])

            # Composite score
            composite = (risk_score * 0.3 + tp_score * 0.2 + score_threshold * 0.2 +
                        avg_sharpe * 0.15 + avg_win_rate * 0.15)

            return composite

        except Exception as e:
            logger.warning(f"# Warning Error calculating composite score: {e}")
            return 0.5

    def _calculate_current_composite_score(self, data: Dict[str, Any]) -> float:
        """Calculate current composite score"""
        try:
            recent_perf = data['performance_data'][-5:]
            current_params = data.get('current_parameters', {})

            risk_per_trade = current_params.get('risk_per_trade', 0.015)
            take_profit_pct = current_params.get('take_profit_pct', 3.0)
            min_score = current_params.get('min_viper_score', 70.0)

            return self._calculate_composite_score(data, risk_per_trade, take_profit_pct, min_score)

        except Exception as e:
            logger.warning(f"# Warning Error calculating current composite score: {e}")
            return 0.5

    def _update_predictive_models(self):
        """Update predictive models for performance forecasting"""
        try:
            if len(self.performance_history) < 20:
                return

            # Prepare data for prediction
            performance_data = []
            for snapshot in list(self.performance_history)[-50:]:
                performance_data.append({
                    'sharpe_ratio': snapshot.sharpe_ratio,
                    'max_drawdown': snapshot.max_drawdown,
                    'win_rate': snapshot.win_rate,
                    'volatility': snapshot.volatility,
                    'timestamp': snapshot.timestamp
                })

            # Simple linear trend prediction
            if len(performance_data) >= 10:
                df = pd.DataFrame(performance_data)
                df['time_index'] = range(len(df))

                # Predict Sharpe ratio trend
                X = df[['time_index']]
                y = df['sharpe_ratio']

                model = LinearRegression()
                model.fit(X, y)

                # Store prediction slope
                self.sharpe_trend_slope = model.coef_[0]

        except Exception as e:
            logger.warning(f"# Warning Error updating predictive models: {e}")

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            if not self.performance_history:
                return {'error': 'No performance data available'}

            # Calculate comprehensive metrics
            recent_snapshots = list(self.performance_history)[-100:]

            # Extract returns for analysis
            returns = pd.Series([s.total_pnl for s in recent_snapshots])

            # Calculate advanced metrics
            advanced_metrics = self.calculate_advanced_metrics(returns.pct_change().dropna())

            # System health metrics
            system_health = self._analyze_system_health()

            # Optimization summary
            optimization_summary = self._summarize_optimizations()

            report = {
                'timestamp': datetime.now().isoformat(),
                'period': {
                    'start': recent_snapshots[0].timestamp.isoformat(),
                    'end': recent_snapshots[-1].timestamp.isoformat(),
                    'total_snapshots': len(recent_snapshots)
                },
                'performance_metrics': advanced_metrics,
                'system_health': system_health,
                'optimization_summary': optimization_summary,
                'alerts': self.alerts[-10:],  # Last 10 alerts
                'recommendations': self._generate_recommendations(advanced_metrics, system_health)
            }

            return report

        except Exception as e:
            logger.error(f"# X Error generating performance report: {e}")
            return {'error': str(e)}

    def _analyze_system_health(self) -> Dict[str, Any]:
        """Analyze system health metrics"""
        try:
            if not self.performance_history:
                return {}

            recent_system_metrics = [s.system_metrics for s in list(self.performance_history)[-20:]]

            # Calculate averages
            avg_cpu = np.mean([m.get('cpu_percent', 0) for m in recent_system_metrics])
            avg_memory = np.mean([m.get('memory_percent', 0) for m in recent_system_metrics])
            avg_disk = np.mean([m.get('disk_usage', 0) for m in recent_system_metrics])

            # Determine health status
            health_score = 100 - (avg_cpu * 0.3 + avg_memory * 0.4 + avg_disk * 0.3)

            if health_score >= 80:
                health_status = "EXCELLENT"
            elif health_score >= 60:
                health_status = "GOOD"
            elif health_score >= 40:
                health_status = "FAIR"
            else:
                health_status = "POOR"

            return {
                'health_score': health_score,
                'health_status': health_status,
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'avg_disk_usage': avg_disk,
                'active_alerts': len([a for a in self.alerts if a.get('severity') in ['HIGH', 'CRITICAL']])
            }

        except Exception as e:
            logger.error(f"# X Error analyzing system health: {e}")
            return {}

    def _summarize_optimizations(self) -> Dict[str, Any]:
        """Summarize optimization results"""
        try:
            if not self.optimization_results:
                return {'total_optimizations': 0, 'avg_improvement': 0.0}

            recent_results = self.optimization_results[-10:]

            total_improvements = [r.improvement for r in recent_results]
            avg_improvement = np.mean(total_improvements)

            successful_optimizations = len([r for r in recent_results if r.improvement > 0])

            return {
                'total_optimizations': len(self.optimization_results),
                'recent_optimizations': len(recent_results),
                'successful_optimizations': successful_optimizations,
                'avg_improvement': avg_improvement,
                'best_improvement': max(total_improvements) if total_improvements else 0.0,
                'last_optimization': recent_results[-1].timestamp.isoformat() if recent_results else None
            }

        except Exception as e:
            logger.warning(f"# Warning Error summarizing optimizations: {e}")
            return {}

    def _generate_recommendations(self, metrics: Dict[str, Any], system_health: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        try:
            # Performance-based recommendations
            if metrics.get('sharpe_ratio', 0) < 1.0:
                recommendations.append("# Chart Sharpe ratio below 1.0 - consider reducing risk or improving strategy")

            if metrics.get('max_drawdown', 0) > 0.15:
                recommendations.append("# Warning High drawdown detected - implement stricter risk controls")

            if metrics.get('win_rate', 0) < 0.55:
                recommendations.append("# Target Win rate below 55% - review entry/exit criteria")

            # System health recommendations
            if system_health.get('avg_memory_percent', 0) > 85:
                recommendations.append("💾 High memory usage - consider memory optimization")

            if system_health.get('avg_cpu_percent', 0) > 80:
                recommendations.append("⚡ High CPU usage - optimize processing algorithms")

            # Optimization recommendations
            if len(self.optimization_results) > 0:
                last_optimization = self.optimization_results[-1]
                if last_optimization.improvement < 0.01:
                    recommendations.append("# Tool Recent optimization had minimal impact - try different parameters")

            # Default recommendations
            if not recommendations:
                recommendations.append("# Check System performance within acceptable parameters")
                recommendations.append("📈 Continue monitoring and periodic optimization")

        except Exception as e:
            logger.warning(f"# Warning Error generating recommendations: {e}")
            recommendations.append("# Tool System monitoring active - review logs for details")

        return recommendations

async def test_performance_system():
    """Test the performance monitoring system"""

    system = PerformanceMonitoringSystem()


    # Simulate some performance data
    for i in range(20):
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now() - timedelta(minutes=i),
            portfolio_value=10000 + np.random.normal(0, 100),
            daily_pnl=np.random.normal(0, 50),
            total_pnl=np.random.normal(0, 200),
            win_rate=0.55 + np.random.normal(0, 0.05),
            sharpe_ratio=1.2 + np.random.normal(0, 0.2),
            max_drawdown=0.1 + abs(np.random.normal(0, 0.05)),
            volatility=0.15 + np.random.normal(0, 0.03),
            trades_executed=np.secrets.randbelow(max_val - min_val + 1) + min_val  # Was: random.randint(5, 15),
            active_positions=np.secrets.randbelow(max_val - min_val + 1) + min_val  # Was: random.randint(1, 5),
            system_metrics={
                'cpu_percent': 50 + np.random.normal(0, 10),
                'memory_percent': 60 + np.random.normal(0, 15),
                'disk_usage': 70 + np.random.normal(0, 5)
            }
        )
        system.performance_history.append(snapshot)

    # Test metrics calculation
    returns = pd.Series([s.total_pnl for s in list(system.performance_history)])
    metrics = system.calculate_advanced_metrics(returns.pct_change().dropna())

    for key, value in metrics.items():

    # Test optimization
    optimization_result = system.optimize_strategy_parameters(OptimizationTarget.BALANCED_OPTIMIZATION)

    if optimization_result:
        print(f"   Target: {optimization_result.target.value}")
        print(f"   Improvement: {optimization_result.improvement:.2%}")
        print(f"   Best Parameters: {optimization_result.parameters}")

    # Test performance report
    report = system.generate_performance_report()

    print(f"# Check Report Generated with {len(report.get('recommendations', []))} recommendations")

if __name__ == "__main__":
    asyncio.run(test_performance_system())
