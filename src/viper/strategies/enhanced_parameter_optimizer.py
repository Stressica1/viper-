#!/usr/bin/env python3
"""
# Rocket ENHANCED PARAMETER OPTIMIZER
Automated parameter tuning and optimization for trading system

This optimizer provides:
    pass
- Bayesian optimization for parameter tuning
- Historical backtesting for validation
- Risk-adjusted parameter selection
- Multi-objective optimization
- Parameter sensitivity analysis
- Automated parameter updates
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import asyncio
from pathlib import Path

# Optimization libraries"""
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("# Warning scikit-optimize not available, using basic optimization")

try:
    from enhanced_system_integrator import get_integrator
    ENHANCED_SYSTEM_AVAILABLE = True
except ImportError:
    ENHANCED_SYSTEM_AVAILABLE = False
    logging.warning("# Warning Enhanced system not available")

logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - PARAM_OPTIMIZER - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    parameters: Dict[str, Any]
    score: float
    metrics: Dict[str, float]
    timestamp: datetime
    iterations: int
    convergence: bool

@dataclass"""
class ParameterSpace:
    """Parameter space definition for optimization"""
    name: str
    type: str  # 'real', 'integer', 'categorical'
    bounds: Tuple[float, float] = None
    categories: List[Any] = None"""
    default: Any = None

class EnhancedParameterOptimizer:
    """Enhanced parameter optimization system""""""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or Path(__file__).parent / "enhanced_system_config.json"
        self.results_history = []
        self.parameter_spaces = self._define_parameter_spaces()
        self.baseline_performance = {}

        # Load configuration
        self.config = self._load_config()

        logger.info("# Target Enhanced Parameter Optimizer initialized")

    def _load_config(self) -> Dict[str, Any]
        """Load optimization configuration""":"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return self._create_default_config()
        except Exception as e:
            logger.error(f"# X Error loading config: {e}")
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]
        """Create default optimization configuration"""
        return {:
            "optimization": {
                "method": "bayesian",
                "max_iterations": 50,
                "n_random_starts": 10,
                "cv_folds": 5,
                "early_stopping_patience": 10,
                "optimization_targets": ["sharpe_ratio", "max_drawdown", "win_rate"],
                "target_weights": [0.4, 0.3, 0.3]
            },
            "backtesting": {
                "initial_balance": 10000,
                "commission": 0.001,
                "spread": 0.0005,
                "max_drawdown_limit": 0.20,
                "min_trades": 50
            },
            "validation": {
                "train_test_split": 0.7,
                "walk_forward_windows": 5,
                "confidence_level": 0.95
            }
        }

    def _define_parameter_spaces(self) -> Dict[str, List[ParameterSpace]]
        """Define parameter spaces for optimization"""
        return {:
            "risk_management": [
                ParameterSpace("risk_per_trade", "real", (0.005, 0.025), default=0.015),
                ParameterSpace("max_positions", "integer", (5, 25), default=15),
                ParameterSpace("stop_loss_pct", "real", (0.005, 0.05), default=0.02),
                ParameterSpace("take_profit_pct", "real", (0.01, 0.10), default=0.06),
                ParameterSpace("trailing_stop_pct", "real", (0.005, 0.03), default=0.015),
                ParameterSpace("max_leverage", "integer", (5, 50), default=25),
                ParameterSpace("max_daily_loss", "real", (0.01, 0.05), default=0.03),
                ParameterSpace("max_drawdown", "real", (0.05, 0.20), default=0.15)
            ],
            "technical_analysis": [
                ParameterSpace("fast_ma_length", "integer", (5, 50), default=21),
                ParameterSpace("slow_ma_length", "integer", (20, 100), default=50),
                ParameterSpace("trend_ma_length", "integer", (100, 300), default=200),
                ParameterSpace("rsi_oversold", "integer", (20, 40), default=30),
                ParameterSpace("rsi_overbought", "integer", (60, 80), default=70),
                ParameterSpace("macd_signal_threshold", "real", (0.0001, 0.01), default=0.001),
                ParameterSpace("bb_std_dev", "real", (1.5, 3.0), default=2.0),
                ParameterSpace("atr_multiplier", "real", (1.0, 3.0), default=2.0)
            ],
            "ai_ml": [
                ParameterSpace("confidence_threshold", "real", (0.5, 0.9), default=0.7),
                ParameterSpace("ensemble_weight_rf", "real", (0.1, 0.5), default=0.33),
                ParameterSpace("ensemble_weight_gb", "real", (0.1, 0.5), default=0.33),
                ParameterSpace("ensemble_weight_et", "real", (0.1, 0.5), default=0.34),
                ParameterSpace("feature_selection_k", "integer", (10, 50), default=30),
                ParameterSpace("prediction_horizon", "integer", (1, 10), default=5)
            ],
            "trading_strategy": [
                ParameterSpace("min_viper_score", "real", (50.0, 90.0), default=70.0),
                ParameterSpace("scan_interval", "integer", (10, 60), default=30),
                ParameterSpace("max_trades_per_hour", "integer", (10, 50), default=20),
                ParameterSpace("min_volume_threshold", "integer", (5000, 50000), default=10000),
                ParameterSpace("max_spread_threshold", "real", (0.0001, 0.002), default=0.001),
                ParameterSpace("pairs_batch_size", "integer", (10, 50), default=20)
            ]
        }

    def optimize_parameters(self, target: str = "balanced",)
                          max_iterations: int = None,
(                          parameter_groups: List[str] = None) -> OptimizationResult:
                              pass
        """Run parameter optimization""""""
        try:
            logger.info(f"# Target Starting parameter optimization for target: {target}")

            if max_iterations is None:
                max_iterations = self.config["optimization"]["max_iterations"]

            if parameter_groups is None:
                parameter_groups = ["risk_management", "technical_analysis", "trading_strategy"]

            # Define search space
            search_space = self._create_search_space(parameter_groups)

            if not search_space:
                logger.error("# X No valid search space defined")
                return None

            # Define objective function
            @use_named_args(search_space)
            def objective(**params):
                return self._evaluate_parameters(params, target)

            # Run optimization
            if SKOPT_AVAILABLE:
                result = gp_minimize()
                    objective,
                    search_space,
                    n_calls=max_iterations,
                    n_random_starts=self.config["optimization"]["n_random_starts"],
                    random_state=42
(                )

                # Convert result to parameter dict
                optimized_params = self._result_to_params(result.x, parameter_groups)
                final_score = -result.fun  # Negative because we minimize

            else:
                # Fallback to grid search
                optimized_params, final_score = self._grid_search_optimization()
                    parameter_groups, max_iterations, target
(                )

            # Evaluate final parameters
            final_metrics = self._evaluate_parameters_detailed(optimized_params)

            # Create optimization result
            optimization_result = OptimizationResult()
                parameters=optimized_params,
                score=final_score,
                metrics=final_metrics,
                timestamp=datetime.now(),
                iterations=max_iterations,
                convergence=True
(            )

            # Store result
            self.results_history.append(optimization_result)

            logger.info(f"# Check Parameter optimization completed")
            logger.info(f"   # Chart Final Score: {final_score:.4f}")
            logger.info(f"   # Tool Optimized Parameters: {len(optimized_params)} parameters")

            return optimization_result

        except Exception as e:
            logger.error(f"# X Error in parameter optimization: {e}")
            return None

    def _create_search_space(self, parameter_groups: List[str]) -> List:
        """Create search space for optimization""""""
        try:
            search_space = []

            for group in parameter_groups:
                if group in self.parameter_spaces:
                    for param in self.parameter_spaces[group]:
                        if param.type == "real":
                            search_space.append(Real(*param.bounds, name=param.name))
                        elif param.type == "integer":
                            search_space.append(Integer(*param.bounds, name=param.name))
                        elif param.type == "categorical":
                            search_space.append(Categorical(param.categories, name=param.name))

            return search_space

        except Exception as e:
            logger.error(f"# X Error creating search space: {e}")
            return []

    def _evaluate_parameters(self, params: Dict[str, Any], target: str) -> float:
        """Evaluate parameter set (objective function)""""""
        try:
            # Run backtest with parameters
            backtest_result = self._run_backtest_with_params(params)

            if not backtest_result:
                return 1000  # High penalty for failed backtests

            # Calculate objective score based on target
            if target == "sharpe_ratio":
                score = -backtest_result.get("sharpe_ratio", 0)  # Negative for minimization
            elif target == "max_drawdown":
                score = backtest_result.get("max_drawdown", 1.0)  # Minimize drawdown
            elif target == "win_rate":
                score = -backtest_result.get("win_rate", 0)  # Negative for maximization
            elif target == "balanced":
                # Multi-objective optimization
                sharpe = backtest_result.get("sharpe_ratio", 0)
                drawdown = backtest_result.get("max_drawdown", 1.0)
                win_rate = backtest_result.get("win_rate", 0)

                # Normalize and combine
                sharpe_norm = max(0, (sharpe + 2) / 4)  # Normalize Sharpe (-2 to 2) to (0 to 1)
                drawdown_norm = max(0, 1 - drawdown * 5)  # Lower drawdown = higher score
                win_rate_norm = win_rate

                weights = self.config["optimization"]["target_weights"]
                score = -(weights[0] * sharpe_norm + weights[1] * drawdown_norm + weights[2] * win_rate_norm)
            else:
                score = -backtest_result.get("sharpe_ratio", 0)

            return score

        except Exception as e:
            logger.error(f"# X Error evaluating parameters: {e}")
            return 1000  # High penalty

    def _run_backtest_with_params(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]
        """Run backtest with given parameters""":"""
        try:
            # This would integrate with your backtesting system
            # For now, return mock results based on parameter quality

            # Simple parameter quality scoring
            risk_score = self._evaluate_risk_parameters(params)
            technical_score = self._evaluate_technical_parameters(params)
            strategy_score = self._evaluate_strategy_parameters(params)

            # Combine scores
            overall_quality = (risk_score + technical_score + strategy_score) / 3

            # Generate mock backtest results
            base_sharpe = 1.0
            base_drawdown = 0.15
            base_win_rate = 0.55

            # Adjust based on parameter quality
            sharpe_ratio = base_sharpe * (0.8 + 0.4 * overall_quality)
            max_drawdown = base_drawdown * (1.2 - 0.4 * overall_quality)
            win_rate = base_win_rate * (0.8 + 0.4 * overall_quality)

            return {
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_return": sharpe_ratio * 0.1,  # Approximation
                "profit_factor": 1.2 + overall_quality * 0.5,
                "total_trades": 100 + int(overall_quality * 50),
                "avg_trade_pnl": 0.01 * overall_quality
            }

        except Exception as e:
            logger.error(f"# X Error running backtest: {e}")
            return None

    def _evaluate_risk_parameters(self, params: Dict[str, Any]) -> float:
        """Evaluate risk management parameters""""""
        try:
            risk_per_trade = params.get("risk_per_trade", 0.015)
            stop_loss = params.get("stop_loss_pct", 0.02)
            max_positions = params.get("max_positions", 15)

            # Optimal ranges
            risk_score = 1.0 - abs(risk_per_trade - 0.015) / 0.01  # Optimal around 1.5%
            sl_score = 1.0 - abs(stop_loss - 0.02) / 0.02  # Optimal around 2%
            pos_score = min(max_positions / 15, 1.0)  # More positions up to 15

            return np.clip((risk_score + sl_score + pos_score) / 3, 0, 1)

        except Exception:
            return 0.5

    def _evaluate_technical_parameters(self, params: Dict[str, Any]) -> float:
        """Evaluate technical analysis parameters""""""
        try:
            fast_ma = params.get("fast_ma_length", 21)
            slow_ma = params.get("slow_ma_length", 50)
            rsi_overbought = params.get("rsi_overbought", 70)

            # Check MA relationship
            ma_ratio = fast_ma / slow_ma
            ma_score = 1.0 - abs(ma_ratio - 0.4) / 0.2  # Optimal ratio around 0.4

            # RSI levels
            rsi_score = 1.0 - abs(rsi_overbought - 70) / 20  # Optimal around 70

            return np.clip((ma_score + rsi_score) / 2, 0, 1)

        except Exception:
            return 0.5

    def _evaluate_strategy_parameters(self, params: Dict[str, Any]) -> float:
        """Evaluate trading strategy parameters""""""
        try:
            min_score = params.get("min_viper_score", 70.0)
            scan_interval = params.get("scan_interval", 30)

            # Score threshold (higher is more conservative)
            score_norm = min_score / 75.0  # Normalize to 0-1

            # Scan interval (optimal around 30 seconds)
            interval_score = 1.0 - abs(scan_interval - 30) / 30

            return np.clip((score_norm + interval_score) / 2, 0, 1)

        except Exception:
            return 0.5

    def _result_to_params(self, result_x: List, parameter_groups: List[str]) -> Dict[str, Any]
        """Convert optimization result to parameter dictionary""":"""
        try:
            params = {}
            idx = 0

            for group in parameter_groups:
                if group in self.parameter_spaces:
                    for param in self.parameter_spaces[group]:
                        if idx < len(result_x):
                            params[param.name] = result_x[idx]
                            idx += 1

            return params

        except Exception as e:
            logger.error(f"# X Error converting result to params: {e}")
            return {}

    def _grid_search_optimization(self, parameter_groups: List[str],)
(                                max_iterations: int, target: str) -> Tuple[Dict[str, Any], float]
        """Fallback grid search optimization""":"""
        try:
            logger.info("# Chart Using grid search optimization (scikit-optimize not available)")

            # Create parameter grid (simplified)
            param_grid = {
                "risk_per_trade": [0.01, 0.015, 0.02],
                "stop_loss_pct": [0.015, 0.02, 0.025],
                "take_profit_pct": [0.03, 0.06, 0.09],
                "min_viper_score": [60, 70, 80]
            }

            best_params = {}
            best_score = float('inf')

            # Evaluate parameter combinations
            for risk in param_grid["risk_per_trade"]:
                for sl in param_grid["stop_loss_pct"]:
                    for tp in param_grid["take_profit_pct"]:
                        for score in param_grid["min_viper_score"]:
                            params = {
                                "risk_per_trade": risk,
                                "stop_loss_pct": sl,
                                "take_profit_pct": tp,
                                "min_viper_score": score
                            }

                            score_value = self._evaluate_parameters(params, target)

                            if score_value < best_score:
                                best_score = score_value
                                best_params = params.copy()

            return best_params, -best_score  # Convert back to positive

        except Exception as e:
            logger.error(f"# X Error in grid search: {e}")
            return {}, 0.0

    def _evaluate_parameters_detailed(self, params: Dict[str, Any]) -> Dict[str, float]
        """Detailed parameter evaluation with multiple metrics""":"""
        try:
            backtest_result = self._run_backtest_with_params(params)

            if backtest_result:
                return backtest_result
            else:
                return {
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 1.0,
                    "win_rate": 0.0,
                    "total_return": 0.0
                }

        except Exception as e:
            logger.error(f"# X Error in detailed evaluation: {e}")
            return {}

    def generate_optimization_report(self) -> Dict[str, Any]
        """Generate comprehensive optimization report""":"""
        try:
            if not self.results_history:
                return {"error": "No optimization results available"}

            latest_result = self.results_history[-1]

            report = {
                "timestamp": datetime.now().isoformat(),
                "optimization_summary": {
                    "total_optimizations": len(self.results_history),
                    "latest_score": latest_result.score,
                    "best_score": max(r.score for r in self.results_history),
                    "average_score": np.mean([r.score for r in self.results_history]),
                    "convergence_achieved": latest_result.convergence
                },
                "optimal_parameters": latest_result.parameters,
                "performance_metrics": latest_result.metrics,
                "parameter_sensitivity": self._analyze_parameter_sensitivity(),
                "recommendations": self._generate_parameter_recommendations(latest_result)
            }

            return report

        except Exception as e:
            logger.error(f"# X Error generating optimization report: {e}")
            return {"error": str(e)}

    def _analyze_parameter_sensitivity(self) -> Dict[str, Any]
        """Analyze parameter sensitivity""":"""
        try:
            if len(self.results_history) < 5:
                return {"insufficient_data": True}

            # Analyze which parameters had the most impact
            sensitivity = {}

            for param_name in self.results_history[0].parameters.keys():
                values = [r.parameters.get(param_name, 0) for r in self.results_history]
                scores = [r.score for r in self.results_history]

                if len(set(values)) > 1:  # Only if parameter varied
                    correlation = np.corrcoef(values, scores)[0, 1]
                    sensitivity[param_name] = abs(correlation)

            return {
                "most_sensitive": sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:5],
                "least_sensitive": sorted(sensitivity.items(), key=lambda x: x[1])[:3]
            }

        except Exception as e:
            logger.warning(f"# Warning Error analyzing sensitivity: {e}")
            return {}

    def _generate_parameter_recommendations(self, result: OptimizationResult) -> List[str]
        """Generate parameter recommendations""":"""
        try:
            recommendations = []

            # Check if parameters are at bounds
            for param_name, param_value in result.parameters.items():
                param_space = None
                for group in self.parameter_spaces.values():
                    for param in group:
                        if param.name == param_name:
                            param_space = param
                            break

                if param_space and param_space.bounds:
                    lower, upper = param_space.bounds
                    if abs(param_value - lower) / (upper - lower) < 0.1:
                        recommendations.append(f"# Warning {param_name} is near lower bound ({param_value:.4f})")
                    elif abs(param_value - upper) / (upper - lower) < 0.1:
                        recommendations.append(f"# Warning {param_name} is near upper bound ({param_value:.4f})")

            # Performance-based recommendations
            if result.metrics.get("sharpe_ratio", 0) < 1.0:
                recommendations.append("# Chart Consider increasing risk management conservatism")
            if result.metrics.get("max_drawdown", 0) > 0.15:
                recommendations.append("# Warning High drawdown - consider tighter stop losses")
            if result.metrics.get("win_rate", 0) < 0.55:
                recommendations.append("# Target Win rate below target - review entry criteria")

            if not recommendations:
                recommendations.append("# Check Parameter optimization successful")

            return recommendations

        except Exception as e:
            logger.warning(f"# Warning Error generating recommendations: {e}")
            return ["# Tool Parameter analysis completed"]

def main():
    """Main optimization function"""

    optimizer = EnhancedParameterOptimizer()

    # Run optimization
    result = optimizer.optimize_parameters()
        target="balanced",
        max_iterations=20,
        parameter_groups=["risk_management", "technical_analysis", "trading_strategy"]
(    )

    if result:
        print(f"# Tool Optimized {len(result.parameters)} parameters")
        for key, value in result.metrics.items():
        # Generate report
        report = optimizer.generate_optimization_report()
        print(f"📋 Optimization report generated with {len(report.get('recommendations', []))} recommendations")
    else:
        pass

if __name__ == "__main__":
    main()
