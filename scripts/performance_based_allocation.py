#!/usr/bin/env python3
"""
🎯 PERFORMANCE-BASED ALLOCATION - Dynamic Capital Distribution
===========================================================

Advanced system for allocating capital based on strategy performance metrics.

Features:
- Performance-weighted capital allocation
- Sharpe ratio prioritization
- Win rate optimization
- Risk-adjusted return maximization
- Dynamic rebalancing based on performance

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import our components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .strategy_metrics_dashboard import StrategyMetricsDashboard
except ImportError:
    from strategy_metrics_dashboard import StrategyMetricsDashboard

@dataclass
class AllocationResult:
    """Results of performance-based allocation"""
    timestamp: str
    original_allocation: Dict[str, float]
    optimized_allocation: Dict[str, float]
    performance_scores: Dict[str, float]
    capital_allocation: Dict[str, float]  # Dollar amounts for $30 portfolio

class PerformanceBasedAllocator:
    """Dynamic capital allocation based on strategy performance"""

    def __init__(self, dashboard: Optional[StrategyMetricsDashboard] = None):
        self.dashboard = dashboard or StrategyMetricsDashboard()
        self.portfolio_value = 30.0  # $30 portfolio

        # Allocation weights (can be adjusted)
        self.weights = {
            'sharpe_ratio': 0.35,    # Sharpe ratio importance
            'win_rate': 0.25,        # Win rate importance
            'total_return': 0.20,    # Return importance
            'profit_factor': 0.15,   # Profit factor importance
            'consistency': 0.05      # Consistency bonus
        }

        # Minimum and maximum allocations
        self.min_allocation = 0.5   # $0.50 minimum per strategy
        self.max_allocation = 12.0  # $12.00 maximum per strategy (40% of portfolio)

    def optimize_allocation(self) -> AllocationResult:
        """Optimize capital allocation based on performance"""
        print("🎯 VIPER PERFORMANCE-BASED ALLOCATION OPTIMIZATION")
        print("=" * 60)

        strategies = self.dashboard.strategies
        original_allocation = {}

        # Calculate performance scores and current allocations
        performance_scores = {}
        for name, strategy in strategies.items():
            if strategy.status == 'active':
                # Current dollar allocation
                original_allocation[name] = (strategy.weight / 100) * self.portfolio_value

                # Calculate composite performance score
                score = self._calculate_performance_score(strategy)
                performance_scores[name] = score

        # Optimize allocations based on performance scores
        optimized_allocation = self._optimize_allocations(performance_scores)

        # Convert to dollar amounts
        capital_allocation = {}
        for name, weight in optimized_allocation.items():
            capital_allocation[name] = (weight / 100) * self.portfolio_value

        # Create result
        result = AllocationResult(
            timestamp=datetime.now().isoformat(),
            original_allocation=original_allocation,
            optimized_allocation=optimized_allocation,
            performance_scores=performance_scores,
            capital_allocation=capital_allocation
        )

        # Update strategy weights in dashboard
        self._apply_allocations(optimized_allocation)

        return result

    def _calculate_performance_score(self, strategy) -> float:
        """Calculate composite performance score for a strategy"""
        # Normalize each metric to 0-100 scale

        # Sharpe ratio (higher is better, typically -3 to +3)
        sharpe_score = min(100, max(0, (strategy.sharpe_ratio + 3) * 25))

        # Win rate (already 0-100)
        win_rate_score = strategy.win_rate

        # Total return (can be negative, normalize to 0-100)
        return_score = min(100, max(0, strategy.total_return + 50))

        # Profit factor (higher is better, typically 0.5 to 3.0)
        profit_factor_score = min(100, strategy.profit_factor * 33.33)

        # Consistency bonus based on volatility and drawdown
        consistency_score = max(0, 100 - strategy.volatility - abs(strategy.max_drawdown))

        # Calculate weighted score
        total_score = (
            sharpe_score * self.weights['sharpe_ratio'] +
            win_rate_score * self.weights['win_rate'] +
            return_score * self.weights['total_return'] +
            profit_factor_score * self.weights['profit_factor'] +
            consistency_score * self.weights['consistency']
        )

        return total_score

    def _optimize_allocations(self, performance_scores: Dict[str, float]) -> Dict[str, float]:
        """Optimize allocations based on performance scores"""
        # Sort strategies by performance score (highest first)
        sorted_strategies = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)

        optimized_weights = {}
        remaining_weight = 100.0

        # Allocate to top performers first
        for i, (name, score) in enumerate(sorted_strategies):
            if i == 0:  # Best performing strategy gets maximum allocation
                allocation = min(self.max_allocation / self.portfolio_value * 100, 35.0)
            elif i == 1:  # Second best gets medium allocation
                allocation = min(self.max_allocation / self.portfolio_value * 100 * 0.8, 25.0)
            elif i == 2:  # Third best gets smaller allocation
                allocation = min(self.max_allocation / self.portfolio_value * 100 * 0.6, 20.0)
            else:  # Remaining strategies get minimum allocation
                allocation = max(self.min_allocation / self.portfolio_value * 100, 5.0)

            # Ensure we don't exceed remaining weight
            allocation = min(allocation, remaining_weight)
            optimized_weights[name] = allocation
            remaining_weight -= allocation

            # Stop if we've allocated all weight or reached minimum threshold
            if remaining_weight <= 5.0:
                break

        # If we have remaining weight, distribute to lower performers
        if remaining_weight > 0:
            remaining_strategies = [name for name in performance_scores.keys() if name not in optimized_weights]
            if remaining_strategies:
                weight_per_strategy = remaining_weight / len(remaining_strategies)
                for name in remaining_strategies:
                    optimized_weights[name] = weight_per_strategy

        return optimized_weights

    def _apply_allocations(self, optimized_weights: Dict[str, float]):
        """Apply optimized weights to strategies"""
        for name, weight in optimized_weights.items():
            if name in self.dashboard.strategies:
                self.dashboard.strategies[name].weight = weight

        # Recalculate portfolio metrics
        self.dashboard._calculate_portfolio_metrics()

    def display_allocation_results(self, result: AllocationResult):
        """Display allocation optimization results"""
        print(f"\n🎯 ALLOCATION OPTIMIZATION RESULTS - ${self.portfolio_value:.0f} PORTFOLIO")
        print("=" * 80)
        print(f"📅 Timestamp: {result.timestamp}")

        print(f"\n📊 CAPITAL ALLOCATION COMPARISON")
        print("-" * 50)
        print(f"{'Strategy':<25} {'Original':>10} {'Optimized':>10} {'Change':>8}")
        print("-" * 53)

        total_original = 0
        total_optimized = 0

        for strategy_name in result.performance_scores.keys():
            original = result.original_allocation.get(strategy_name, 0)
            optimized = result.capital_allocation.get(strategy_name, 0)
            change = optimized - original

            total_original += original
            total_optimized += optimized

            print(f"{strategy_name[:24]:<25} ${original:>8.2f} ${optimized:>8.2f} ${change:>+7.2f}")

        print("-" * 53)
        print(f"{'TOTAL':<25} ${total_original:>8.2f} ${total_optimized:>8.2f} ${total_optimized-total_original:>+7.2f}")

        print(f"\n🏆 PERFORMANCE SCORES & ALLOCATIONS")
        print("-" * 45)

        # Sort by performance score
        sorted_by_score = sorted(result.performance_scores.items(), key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(sorted_by_score, 1):
            allocation = result.capital_allocation.get(name, 0)
            weight = result.optimized_allocation.get(name, 0)

            print(f"{i}. {name}")
            print(f"   Performance Score: {score:.1f}/100")
            print(f"   Allocation: ${allocation:.2f} ({weight:.1f}%)")

            # Show key metrics
            strategy = self.dashboard.strategies.get(name)
            if strategy:
                print(f"   Sharpe: {strategy.sharpe_ratio:.2f} | Win Rate: {strategy.win_rate:.1f}% | Return: {strategy.total_return:.1f}%")

        print(f"\n💡 ALLOCATION STRATEGY")
        print("-" * 25)
        print("• Top performer gets maximum allocation (up to $12.00)")
        print("• Second best gets medium allocation (up to $9.00)")
        print("• Third best gets smaller allocation (up to $6.00)")
        print("• Remaining strategies get minimum allocation ($0.50+)"        print("• Focus on Sharpe ratio, win rate, and risk-adjusted returns")

    def export_allocation_report(self, result: AllocationResult) -> str:
        """Export allocation results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"allocation_optimization_{timestamp}.json"

        # Create reports directory
        reports_dir = Path("reports") / "allocation"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_data = {
            'timestamp': result.timestamp,
            'portfolio_value': self.portfolio_value,
            'allocation_weights': self.weights,
            'original_allocation': result.original_allocation,
            'optimized_allocation': result.optimized_allocation,
            'capital_allocation': result.capital_allocation,
            'performance_scores': result.performance_scores,
            'strategy_details': {}
        }

        # Add strategy details
        for name in result.performance_scores.keys():
            strategy = self.dashboard.strategies.get(name)
            if strategy:
                report_data['strategy_details'][name] = {
                    'sharpe_ratio': strategy.sharpe_ratio,
                    'win_rate': strategy.win_rate,
                    'total_return': strategy.total_return,
                    'max_drawdown': strategy.max_drawdown,
                    'profit_factor': strategy.profit_factor,
                    'volatility': strategy.volatility,
                    'total_trades': strategy.total_trades
                }

        filepath = reports_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='VIPER Performance-Based Allocator')
    parser.add_argument('--optimize', '-o', action='store_true', help='Run allocation optimization')
    parser.add_argument('--apply', '-a', action='store_true', help='Apply optimized allocations')
    parser.add_argument('--export', '-e', action='store_true', help='Export allocation report')
    parser.add_argument('--portfolio', '-p', type=float, default=30.0, help='Portfolio value (default: $30.00)')

    args = parser.parse_args()

    # Create allocator
    allocator = PerformanceBasedAllocator()
    allocator.portfolio_value = args.portfolio

    if args.optimize:
        # Run optimization
        result = allocator.optimize_allocation()

        # Display results
        allocator.display_allocation_results(result)

        if args.export:
            report_path = allocator.export_allocation_report(result)
            print(f"\n📄 Allocation report saved: {report_path}")

        print("
✅ Performance-based allocation optimization completed!")
        print(f"💰 Portfolio: ${allocator.portfolio_value:.2f}")
        print("🎯 Capital allocated based on strategy performance metrics"
    else:
        print("🎯 VIPER Performance-Based Allocation System")
        print("=" * 50)
        print(f"💰 Portfolio Value: ${allocator.portfolio_value:.2f}")
        print("\nUse --optimize to run performance-based allocation")
        print("Use --export to save detailed allocation report")
        print("\nExample:")
        print("  python scripts/performance_based_allocation.py --optimize --export")

if __name__ == '__main__':
    main()
