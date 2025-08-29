#!/usr/bin/env python3
"""
🎯 STRATEGY OPTIMIZER - Dynamic Performance-Based Allocation
===========================================================

Advanced strategy optimization for $30 portfolio with performance-based weighting.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class StrategyScore:
    strategy_name: str
    performance_score: float
    risk_score: float
    recommended_weight: float

@dataclass
class OptimizationResult:
    timestamp: str
    original_weights: Dict[str, float]
    optimized_weights: Dict[str, float]
    performance_improvement: float
    strategy_scores: Dict[str, StrategyScore]

class StrategyOptimizer:
    def __init__(self):
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            from .strategy_metrics_dashboard import StrategyMetricsDashboard
            self.dashboard = StrategyMetricsDashboard()
        except ImportError:
            from strategy_metrics_dashboard import StrategyMetricsDashboard
            self.dashboard = StrategyMetricsDashboard()

        self.params = {
            'min_weight': 0.02,  # 2%
            'max_weight': 0.40,  # 40%
        }

    def optimize_strategy_weights(self) -> OptimizationResult:
        print("🎯 VIPER STRATEGY OPTIMIZATION FOR $30 PORTFOLIO")
        print("=" * 50)

        strategies = self.dashboard.strategies
        original_weights = {name: s.weight for name, s in strategies.items()}

        # Calculate strategy scores
        strategy_scores = {}
        for name, strategy in strategies.items():
            if strategy.status == 'active':
                score = self._calculate_strategy_score(strategy)
                strategy_scores[name] = score

        # Optimize weights using performance-based allocation
        optimized_weights = self._performance_based_optimization(strategy_scores)

        # Calculate improvement
        performance_improvement = self._calculate_improvement(original_weights, optimized_weights, strategy_scores)

        result = OptimizationResult(
            timestamp=datetime.now().isoformat(),
            original_weights=original_weights,
            optimized_weights=optimized_weights,
            performance_improvement=performance_improvement,
            strategy_scores=strategy_scores
        )

        return result

    def _calculate_strategy_score(self, strategy) -> StrategyScore:
        """Calculate strategy performance score"""
        sharpe_score = min(100, max(0, (strategy.sharpe_ratio + 2) * 25))
        win_rate_score = strategy.win_rate
        return_score = min(100, max(0, strategy.total_return + 50))

        performance_score = (sharpe_score * 0.4 + win_rate_score * 0.35 + return_score * 0.25)
        risk_score = max(0, 100 - strategy.volatility - abs(strategy.max_drawdown) * 2)

        return StrategyScore(
            strategy_name=strategy.strategy_name,
            performance_score=performance_score,
            risk_score=risk_score,
            recommended_weight=0.0
        )

    def _performance_based_optimization(self, strategy_scores: Dict[str, StrategyScore]) -> Dict[str, float]:
        """Performance-based weight optimization for $30 portfolio"""
        sorted_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1].performance_score,
            reverse=True
        )

        weights = {}
        remaining_weight = 100.0

        # Allocate weights based on performance ranking
        for i, (name, score) in enumerate(sorted_strategies):
            if i == 0:  # Best performing strategy
                base_weight = 35.0
            elif i == 1:  # Second best
                base_weight = 25.0
            elif i == 2:  # Third best
                base_weight = 20.0
            else:  # Others get minimum weight
                base_weight = self.params['min_weight'] * 100

            # Adjust based on risk score
            risk_adjustment = score.risk_score / 100
            adjusted_weight = base_weight * (0.7 + 0.3 * risk_adjustment)

            adjusted_weight = max(self.params['min_weight'] * 100,
                                min(self.params['max_weight'] * 100, adjusted_weight))

            weights[name] = adjusted_weight
            remaining_weight -= adjusted_weight

        # Distribute remaining weight to top performers
        if remaining_weight > 0:
            top_performers = [name for name, _ in sorted_strategies[:3]]
            extra_per_strategy = remaining_weight / len(top_performers)

            for name in top_performers:
                new_weight = weights[name] + extra_per_strategy
                weights[name] = min(self.params['max_weight'] * 100, new_weight)

        # Normalize to 100%
        total_weight = sum(weights.values())
        for name in weights:
            weights[name] = (weights[name] / total_weight) * 100

        return weights

    def _calculate_improvement(self, original_weights: Dict[str, float],
                             optimized_weights: Dict[str, float],
                             strategy_scores: Dict[str, StrategyScore]) -> float:
        """Calculate performance improvement"""
        original_score = sum(
            score.performance_score * original_weights.get(name, 0) / 100
            for name, score in strategy_scores.items()
        )
        optimized_score = sum(
            score.performance_score * optimized_weights.get(name, 0) / 100
            for name, score in strategy_scores.items()
        )

        if original_score > 0:
            return ((optimized_score - original_score) / original_score) * 100
        return 0

    def display_optimization_results(self, result: OptimizationResult):
        """Display optimization results"""
        print(f"\n🎯 STRATEGY OPTIMIZATION RESULTS - $30 PORTFOLIO")
        print("=" * 60)
        print(f"📈 Performance Improvement: {result.performance_improvement:.1f}%")
        print(f"💰 Portfolio Value: $30.00")

        print(f"\n📋 CAPITAL ALLOCATION OPTIMIZATION")
        print("-" * 40)
        print(f"{'Strategy':<25} {'Original':>10} {'Optimized':>10} {'Change':>8} {'$ Amount':>9}")
        print("-" * 74)

        for strategy_name in result.strategy_scores.keys():
            original = result.original_weights.get(strategy_name, 0)
            optimized = result.optimized_weights.get(strategy_name, 0)
            change = optimized - original
            dollar_amount = (optimized / 100) * 30

            print(f"{strategy_name[:24]:<25} {original:>9.1f}% {optimized:>9.1f}% {change:>+7.1f}% ${dollar_amount:>7.2f}")

        print(f"\n🏆 TOP PERFORMING STRATEGIES")
        print("-" * 30)

        sorted_strategies = sorted(
            result.strategy_scores.items(),
            key=lambda x: x[1].performance_score,
            reverse=True
        )

        for i, (name, score) in enumerate(sorted_strategies[:3], 1):
            optimized_weight = result.optimized_weights.get(name, 0)
            dollar_allocation = (optimized_weight / 100) * 30

            print(f"{i}. {name}")
            print(f"   Performance Score: {score.performance_score:.1f}/100")
            print(f"   Optimized Allocation: {optimized_weight:.1f}% (${dollar_allocation:.2f})")

    def apply_optimized_weights(self, result: OptimizationResult):
        """Apply optimized weights"""
        print("🔄 Applying optimized strategy weights to $30 portfolio...")

        for strategy_name, new_weight in result.optimized_weights.items():
            if strategy_name in self.dashboard.strategies:
                old_weight = self.dashboard.strategies[strategy_name].weight
                dollar_allocation = (new_weight / 100) * 30

                self.dashboard.strategies[strategy_name].weight = new_weight
                print(f"📊 {strategy_name}: {old_weight:.1f}% → {new_weight:.1f}% (${dollar_allocation:.2f})")

        self.dashboard._calculate_portfolio_metrics()
        print("✅ Optimized weights applied to $30 portfolio successfully!")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='VIPER Strategy Optimizer - $30 Portfolio')
    parser.add_argument('--optimize', '-o', action='store_true', help='Run strategy optimization')
    parser.add_argument('--apply', '-a', action='store_true', help='Apply optimized weights')

    args = parser.parse_args()

    optimizer = StrategyOptimizer()

    if args.optimize:
        result = optimizer.optimize_strategy_weights()
        optimizer.display_optimization_results(result)

        if args.apply:
            optimizer.apply_optimized_weights(result)
            print("\n✅ Optimization complete and weights applied to $30 portfolio!")

    else:
        print("🎯 VIPER Strategy Optimizer - $30 Portfolio")
        print("=" * 50)
        print("Optimizes strategy weights based on performance metrics")
        print("for maximum returns on your $30 portfolio")
        print("\nUse --optimize to run optimization")
        print("Use --apply to apply optimized weights")

if __name__ == '__main__':
    main()
