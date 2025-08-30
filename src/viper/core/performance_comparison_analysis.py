#!/usr/bin/env python3
"""
# Rocket PERFORMANCE COMPARISON ANALYSIS
Comprehensive analysis and visualization of system performance improvements

This analysis provides:
    pass
- Detailed performance metric comparisons
- Statistical significance testing
- Visual performance dashboards
- Risk-adjusted return analysis
- Market condition performance breakdown
- Deployment readiness assessment
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - PERFORMANCE_ANALYSIS - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)"""

class PerformanceComparisonAnalysis:
    """Comprehensive performance comparison and analysis system""""""

    def __init__(self):
        self.baseline_results = []
        self.enhanced_results = []
        self.comparison_metrics = {}
        self.visualization_path = project_root / "performance_analysis"

        # Create visualization directory
        self.visualization_path.mkdir(exist_ok=True)

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        logger.info("# Chart Performance Comparison Analysis initialized")

    def load_performance_data(self, baseline_path: Optional[str] = None,)
(                            enhanced_path: Optional[str] = None) -> bool:
                                pass
        """Load performance data from backtesting results""""""
        try:
            logger.info("📂 Loading performance data...")

            # Load baseline results
            if baseline_path:
                baseline_file = Path(baseline_path)
            else:
                baseline_file = project_root / "baseline_backtest_results.json"

            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    self.baseline_results = json.load(f)
                logger.info(f"# Check Baseline results loaded: {len(self.baseline_results)} records")
            else:
                logger.warning(f"# Warning Baseline results file not found: {baseline_file}")

            # Load enhanced results
            if enhanced_path:
                enhanced_file = Path(enhanced_path)
            else:
                enhanced_file = project_root / "enhanced_backtest_results.json"

            if enhanced_file.exists():
                with open(enhanced_file, 'r') as f:
                    self.enhanced_results = json.load(f)
                logger.info(f"# Check Enhanced results loaded: {len(self.enhanced_results)} records")
            else:
                logger.warning(f"# Warning Enhanced results file not found: {enhanced_file}")

            return len(self.baseline_results) > 0 and len(self.enhanced_results) > 0

        except Exception as e:
            logger.error(f"# X Error loading performance data: {e}")
            return False

    def generate_comprehensive_analysis(self) -> Dict[str, Any]
        """Generate comprehensive performance analysis"""
        logger.info("🔬 Generating comprehensive performance analysis...")

        analysis_results = {:
            "analysis_timestamp": datetime.now().isoformat(),
            "baseline_summary": self._calculate_system_summary(self.baseline_results, "baseline"),
            "enhanced_summary": self._calculate_system_summary(self.enhanced_results, "enhanced"),
            "performance_comparison": self._compare_performance_metrics(),
            "statistical_analysis": self._perform_statistical_analysis(),
            "market_condition_analysis": self._analyze_market_conditions(),
            "risk_adjusted_analysis": self._analyze_risk_adjusted_performance(),
            "visualization_files": [],
            "recommendations": []
        }

        # Generate visualizations
        logger.info("# Chart Generating performance visualizations...")
        viz_files = self._generate_performance_visualizations()
        analysis_results["visualization_files"] = viz_files

        # Generate recommendations
        analysis_results["recommendations"] = self._generate_performance_recommendations()
            analysis_results
(        )

        # Save analysis report
        self._save_analysis_report(analysis_results)

        logger.info("# Check Comprehensive analysis completed")
        return analysis_results

    def _calculate_system_summary(self, results: List[Dict], system_name: str) -> Dict[str, Any]
        """Calculate summary statistics for a system""":"""
        try:
            if not results:
                return {"error": "No results available"}

            # Extract key metrics
            metrics = {
                "total_return": [],
                "sharpe_ratio": [],
                "max_drawdown": [],
                "win_rate": [],
                "profit_factor": [],
                "total_trades": [],
                "avg_trade_pnl": []
            }

            for result in results:
                for metric in metrics:
                    if metric in result:
                        metrics[metric].append(result[metric])

            # Calculate summary statistics
            summary = {}
            for metric, values in metrics.items():
                if values:
                    summary[f"{metric}_mean"] = np.mean(values)
                    summary[f"{metric}_median"] = np.median(values)
                    summary[f"{metric}_std"] = np.std(values)
                    summary[f"{metric}_min"] = np.min(values)
                    summary[f"{metric}_max"] = np.max(values)
                else:
                    summary[f"{metric}_mean"] = 0

            summary["total_tests"] = len(results)
            summary["successful_tests"] = len([r for r in results if r.get("total_return", 0) > 0])

            return summary

        except Exception as e:
            logger.error(f"# X Error calculating system summary: {e}")
            return {"error": str(e)}

    def _compare_performance_metrics(self) -> Dict[str, Any]
        """Compare performance metrics between systems""":"""
        try:
            logger.info("   📈 Comparing performance metrics...")

            baseline_summary = self._calculate_system_summary(self.baseline_results, "baseline")
            enhanced_summary = self._calculate_system_summary(self.enhanced_results, "enhanced")

            comparison = {}

            # Compare key metrics
            key_metrics = ["total_return", "sharpe_ratio", "win_rate", "profit_factor"]

            for metric in key_metrics:
                baseline_value = baseline_summary.get(f"{metric}_mean", 0)
                enhanced_value = enhanced_summary.get(f"{metric}_mean", 0)

                if baseline_value != 0:
                    improvement_pct = ((enhanced_value - baseline_value) / abs(baseline_value)) * 100
                else:
                    improvement_pct = 0

                comparison[metric] = {
                    "baseline": baseline_value,
                    "enhanced": enhanced_value,
                    "improvement_pct": improvement_pct,
                    "improvement_absolute": enhanced_value - baseline_value
                }

            # Risk metrics (lower is better)
            risk_metrics = ["max_drawdown"]
            for metric in risk_metrics:
                baseline_value = baseline_summary.get(f"{metric}_mean", 0)
                enhanced_value = enhanced_summary.get(f"{metric}_mean", 0)

                if baseline_value != 0:
                    improvement_pct = ((baseline_value - enhanced_value) / abs(baseline_value)) * 100
                else:
                    improvement_pct = 0

                comparison[metric] = {
                    "baseline": baseline_value,
                    "enhanced": enhanced_value,
                    "improvement_pct": improvement_pct,
                    "improvement_absolute": baseline_value - enhanced_value
                }

            return comparison

        except Exception as e:
            logger.error(f"# X Error comparing performance metrics: {e}")
            return {}

    def _perform_statistical_analysis(self) -> Dict[str, Any]
        """Perform statistical analysis of performance differences""":"""
        try:
            logger.info("   # Chart Performing statistical analysis...")

            statistical_results = {}

            # Extract metric arrays
            metrics_to_test = ["total_return", "sharpe_ratio", "win_rate", "max_drawdown"]

            for metric in metrics_to_test:
                baseline_values = [r.get(metric, 0) for r in self.baseline_results]
                enhanced_values = [r.get(metric, 0) for r in self.enhanced_results]

                if len(baseline_values) >= 3 and len(enhanced_values) >= 3:
                    # Perform t-test
                    try:
                        t_stat, p_value = stats.ttest_ind(baseline_values, enhanced_values)

                        statistical_results[metric] = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "baseline_mean": np.mean(baseline_values),
                            "enhanced_mean": np.mean(enhanced_values),
                            "effect_size": abs(np.mean(enhanced_values) - np.mean(baseline_values)) / np.std(baseline_values)
                        }
                    except Exception as e:
                        logger.warning(f"# Warning Statistical test failed for {metric}: {e}")

            return statistical_results

        except Exception as e:
            logger.error(f"# X Error performing statistical analysis: {e}")
            return {}

    def _analyze_market_conditions(self) -> Dict[str, Any]
        """Analyze performance across different market conditions""":"""
        try:
            logger.info("   # Chart Analyzing market condition performance...")

            market_analysis = {}

            # Group results by market condition
            conditions = ["bull_market", "volatile_market", "bear_market", "recovery_market"]

            for condition in conditions:
                baseline_condition_results = [
                    r for r in self.baseline_results
                    if r.get("market_condition") == condition:
                ]
                enhanced_condition_results = [
                    r for r in self.enhanced_results
                    if r.get("market_condition") == condition:
                ]

                if baseline_condition_results and enhanced_condition_results:
                    baseline_avg_return = np.mean([r.get("total_return", 0) for r in baseline_condition_results])
                    enhanced_avg_return = np.mean([r.get("total_return", 0) for r in enhanced_condition_results])

                    improvement = ((enhanced_avg_return - baseline_avg_return) / abs(baseline_avg_return)) * 100 if baseline_avg_return != 0 else 0

                    market_analysis[condition] = {
                        "baseline_return": baseline_avg_return,
                        "enhanced_return": enhanced_avg_return,
                        "improvement_pct": improvement,
                        "baseline_tests": len(baseline_condition_results),
                        "enhanced_tests": len(enhanced_condition_results)
                    }

            return market_analysis

        except Exception as e:
            logger.error(f"# X Error analyzing market conditions: {e}")
            return {}

    def _analyze_risk_adjusted_performance(self) -> Dict[str, Any]
        """Analyze risk-adjusted performance metrics""":"""
        try:
            logger.info("   🛡️ Analyzing risk-adjusted performance...")

            risk_adjusted_metrics = {}

            # Calculate risk-adjusted returns
            for system_name, results in [("baseline", self.baseline_results), ("enhanced", self.enhanced_results)]:
                returns = [r.get("total_return", 0) for r in results]
                volatilities = [r.get("volatility", 0.01) for r in results]
                drawdowns = [r.get("max_drawdown", 0) for r in results]

                if returns and volatilities:
                    # Sharpe ratio (already calculated)
                    sharpe_ratios = [r.get("sharpe_ratio", 0) for r in results]

                    # Sortino ratio (downside deviation)
                    downside_returns = [r for r in returns if r < 0]
                    if downside_returns:
                        downside_volatility = np.std(downside_returns)
                        sortino_ratios = [r / downside_volatility if downside_volatility > 0 else 0 for r in returns]
                    else:
                        sortino_ratios = [0] * len(returns)

                    # Calmar ratio
                    calmar_ratios = [r / d if d > 0 else 0 for r, d in zip(returns, drawdowns)]

                    risk_adjusted_metrics[system_name] = {
                        "avg_sharpe": np.mean(sharpe_ratios),
                        "avg_sortino": np.mean(sortino_ratios),
                        "avg_calmar": np.mean(calmar_ratios),
                        "sharpe_volatility": np.std(sharpe_ratios),
                        "best_sharpe": max(sharpe_ratios),
                        "worst_sharpe": min(sharpe_ratios)
                    }

            return risk_adjusted_metrics

        except Exception as e:
            logger.error(f"# X Error analyzing risk-adjusted performance: {e}")
            return {}

    def _generate_performance_visualizations(self) -> List[str]
        """Generate comprehensive performance visualizations""":"""
        try:
            logger.info("   # Chart Generating performance visualizations...")

            generated_files = []

            # 1. Performance comparison bar chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Enhanced VIPER Trading System - Performance Comparison', fontsize=16, fontweight='bold')

            # Returns comparison
            baseline_returns = [r.get("total_return", 0) for r in self.baseline_results]
            enhanced_returns = [r.get("total_return", 0) for r in self.enhanced_results]

            axes[0, 0].bar(['Baseline', 'Enhanced'], [np.mean(baseline_returns), np.mean(enhanced_returns)],)
(                          color=['skyblue', 'lightgreen'], alpha=0.7)
            axes[0, 0].set_title('Average Total Returns')
            axes[0, 0].set_ylabel('Return (%)')
            axes[0, 0].grid(True, alpha=0.3)

            # Sharpe ratio comparison
            baseline_sharpe = [r.get("sharpe_ratio", 0) for r in self.baseline_results]
            enhanced_sharpe = [r.get("sharpe_ratio", 0) for r in self.enhanced_results]

            axes[0, 1].bar(['Baseline', 'Enhanced'], [np.mean(baseline_sharpe), np.mean(enhanced_sharpe)],)
(                          color=['skyblue', 'lightgreen'], alpha=0.7)
            axes[0, 1].set_title('Average Sharpe Ratio')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].grid(True, alpha=0.3)

            # Win rate comparison
            baseline_win_rate = [r.get("win_rate", 0) for r in self.baseline_results]
            enhanced_win_rate = [r.get("win_rate", 0) for r in self.enhanced_results]

            axes[1, 0].bar(['Baseline', 'Enhanced'], [np.mean(baseline_win_rate), np.mean(enhanced_win_rate)],)
(                          color=['skyblue', 'lightgreen'], alpha=0.7)
            axes[1, 0].set_title('Average Win Rate')
            axes[1, 0].set_ylabel('Win Rate (%)')
            axes[1, 0].grid(True, alpha=0.3)

            # Max drawdown comparison (lower is better)
            baseline_drawdown = [r.get("max_drawdown", 0) for r in self.baseline_results]
            enhanced_drawdown = [r.get("max_drawdown", 0) for r in self.enhanced_results]

            axes[1, 1].bar(['Baseline', 'Enhanced'], [np.mean(baseline_drawdown), np.mean(enhanced_drawdown)],)
(                          color=['lightcoral', 'lightgreen'], alpha=0.7)
            axes[1, 1].set_title('Average Max Drawdown (Lower is Better)')
            axes[1, 1].set_ylabel('Max Drawdown (%)')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            performance_chart_path = self.visualization_path / "performance_comparison.png"
            plt.savefig(performance_chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(str(performance_chart_path))

            # 2. Returns distribution histogram
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.hist(baseline_returns, bins=20, alpha=0.7, label='Baseline', color='skyblue', edgecolor='black')
            plt.hist(enhanced_returns, bins=20, alpha=0.7, label='Enhanced', color='lightgreen', edgecolor='black')
            plt.title('Returns Distribution Comparison')
            plt.xlabel('Total Return (%)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            plt.boxplot([baseline_returns, enhanced_returns], labels=['Baseline', 'Enhanced'])
            plt.title('Returns Distribution Box Plot')
            plt.ylabel('Total Return (%)')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            returns_dist_path = self.visualization_path / "returns_distribution.png"
            plt.savefig(returns_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(str(returns_dist_path))

            # 3. Risk-return scatter plot
            plt.figure(figsize=(10, 8))

            baseline_volatility = [r.get("volatility", 0) for r in self.baseline_results]
            enhanced_volatility = [r.get("volatility", 0) for r in self.enhanced_results]

            plt.scatter(baseline_volatility, baseline_returns, alpha=0.7, label='Baseline',)
(                       color='skyblue', s=100, edgecolors='black')
            plt.scatter(enhanced_volatility, enhanced_returns, alpha=0.7, label='Enhanced',)
(                       color='lightgreen', s=100, edgecolors='black')

            plt.title('Risk-Return Profile Comparison')
            plt.xlabel('Volatility (Risk)')
            plt.ylabel('Total Return')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add reference lines
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
            plt.axvline(x=np.mean(baseline_volatility), color='blue', linestyle=':', alpha=0.5, label='Avg Baseline Risk')

            risk_return_path = self.visualization_path / "risk_return_profile.png"
            plt.savefig(risk_return_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(str(risk_return_path))

            # 4. Market condition performance heatmap
            market_conditions = ["bull_market", "volatile_market", "bear_market", "recovery_market"]

            baseline_market_returns = []
            enhanced_market_returns = []

            for condition in market_conditions:
                baseline_condition_returns = [
                    r.get("total_return", 0) for r in self.baseline_results
                    if r.get("market_condition") == condition:
                ]
                enhanced_condition_returns = [
                    r.get("total_return", 0) for r in self.enhanced_results
                    if r.get("market_condition") == condition:
                ]

                baseline_avg = np.mean(baseline_condition_returns) if baseline_condition_returns else 0
                enhanced_avg = np.mean(enhanced_condition_returns) if enhanced_condition_returns else 0

                baseline_market_returns.append(baseline_avg)
                enhanced_market_returns.append(enhanced_avg)

            # Create heatmap data
            heatmap_data = np.array([baseline_market_returns, enhanced_market_returns])

            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',)
(                       xticklabels=market_conditions, yticklabels=['Baseline', 'Enhanced'])
            plt.title('Market Condition Performance Heatmap')
            plt.xlabel('Market Conditions')
            plt.ylabel('System')

            market_heatmap_path = self.visualization_path / "market_condition_heatmap.png"
            plt.savefig(market_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(str(market_heatmap_path))

            logger.info(f"   # Check Generated {len(generated_files)} visualization files")
            return generated_files

        except Exception as e:
            logger.error(f"# X Error generating visualizations: {e}")
            return []

    def _generate_performance_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]
        """Generate performance-based recommendations""":"""
        try:
            recommendations = []

            # Performance improvement analysis
            comparison = analysis_results.get("performance_comparison", {})

            # Return improvement
            return_improvement = comparison.get("total_return", {}).get("improvement_pct", 0)
            if return_improvement > 20:
                recommendations.append("# Party EXCELLENT: Strong return improvement (+20%+) - Full confidence in enhanced system")
            elif return_improvement > 10:
                recommendations.append("# Check GOOD: Solid return improvement (10-20%) - Ready for production deployment")
            elif return_improvement > 5:
                recommendations.append("# Warning MODERATE: Some improvement (5-10%) - Monitor performance closely")
            else:
                recommendations.append("# X CONCERN: Limited improvement (<5%) - Consider further optimization")

            # Risk improvement
            drawdown_improvement = comparison.get("max_drawdown", {}).get("improvement_pct", 0)
            if drawdown_improvement > 15:
                recommendations.append("🛡️ EXCELLENT: Significant risk reduction - Enhanced risk management working well")
            elif drawdown_improvement > 5:
                recommendations.append("🛡️ GOOD: Moderate risk improvement - Acceptable risk reduction")
            else:
                recommendations.append("# Warning LIMITED: Minimal risk improvement - Focus on risk management enhancements")

            # Statistical significance
            statistical_analysis = analysis_results.get("statistical_analysis", {})
            significant_improvements = sum(1 for metric in statistical_analysis.values())
(                                         if metric.get("significant", False))
            if significant_improvements >= 3:
                recommendations.append("# Chart STRONG: Multiple statistically significant improvements - High confidence")
            elif significant_improvements >= 2:
                recommendations.append("# Chart MODERATE: Some statistically significant improvements - Good confidence")
            else:
                recommendations.append("# Chart WEAK: Limited statistical significance - More testing recommended")

            # Market condition analysis
            market_analysis = analysis_results.get("market_condition_analysis", {})
            strong_conditions = sum(1 for condition in market_analysis.values())
(                                  if condition.get("improvement_pct", 0) > 15)
            if strong_conditions >= 3:
                recommendations.append("🌍 ROBUST: Strong performance across multiple market conditions")
            elif strong_conditions >= 2:
                recommendations.append("🌍 ADAPTABLE: Good performance in various conditions")
            else:
                recommendations.append("🌍 CONDITION DEPENDENT: Performance varies by market condition")

            # Deployment recommendations
            if return_improvement > 10 and drawdown_improvement > 5:
                recommendations.append("# Rocket DEPLOYMENT READY: Meets performance and risk criteria")
                recommendations.append("📈 Implement gradual rollout with 25-50% capital initially")
                recommendations.append("# Chart Enable comprehensive monitoring and alerting")
            elif return_improvement > 5 and drawdown_improvement > 2:
                recommendations.append("# Warning CONDITIONAL DEPLOYMENT: Monitor closely with rollback plan")
                recommendations.append("# Chart Start with small position sizes and scale gradually")
            else:
                recommendations.append("# X NOT READY: Further optimization required before deployment")
                recommendations.append("# Tool Focus on improving return and risk metrics")

            return recommendations

        except Exception as e:
            logger.warning(f"# Warning Error generating recommendations: {e}")
            return ["# Search Analysis completed - review detailed report for recommendations"]

    def _save_analysis_report(self, analysis_results: Dict[str, Any],)
(                            report_path: Optional[str] = None):
        """Save comprehensive analysis report""""""
        try:
            if report_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = project_root / f"performance_analysis_report_{timestamp}.json"

            with open(report_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)

            logger.info(f"📋 Analysis report saved to: {report_path}")

        except Exception as e:
            logger.error(f"# X Error saving analysis report: {e}")

    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary of analysis results""""""
        try:
            summary_lines = [
                "=" * 80,
                "ENHANCED VIPER TRADING SYSTEM - PERFORMANCE ANALYSIS EXECUTIVE SUMMARY",
                "=" * 80,
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "PERFORMANCE OVERVIEW:",
                "-" * 30
            ]

            # Performance comparison
            comparison = analysis_results.get("performance_comparison", {})
            for metric, data in comparison.items():
                improvement = data.get("improvement_pct", 0)
                baseline = data.get("baseline", 0)
                enhanced = data.get("enhanced", 0)

                summary_lines.append(".1f")
(                                   ".4f")

            # Statistical significance
            statistical = analysis_results.get("statistical_analysis", {})
            significant_count = sum(1 for metric in statistical.values())
(                                  if metric.get("significant", False))
            summary_lines.append(f"Statistically Significant Improvements: {significant_count}/{len(statistical)}")

            # Market conditions
            market_analysis = analysis_results.get("market_condition_analysis", {})
            strong_conditions = sum(1 for condition in market_analysis.values())
(                                  if condition.get("improvement_pct", 0) > 10)
            summary_lines.append(f"Strong Performance in Market Conditions: {strong_conditions}/{len(market_analysis)}")

            # Recommendations
            recommendations = analysis_results.get("recommendations", [])
            summary_lines.extend([)
                "",
                "KEY RECOMMENDATIONS:",
                "-" * 25
(            ])
            summary_lines.extend(recommendations[:5])  # Top 5 recommendations

            summary_lines.extend([)
                "",
                "VISUALIZATIONS GENERATED:",
                "-" * 30
(            ])

            viz_files = analysis_results.get("visualization_files", [])
            for viz_file in viz_files:
                summary_lines.append(f"• {Path(viz_file).name}")

            summary_lines.extend([)
                "",
                "=" * 80,
                "END OF EXECUTIVE SUMMARY",
                "=" * 80
(            ])

            return "\n".join(summary_lines)

        except Exception as e:
            logger.error(f"# X Error generating executive summary: {e}")
            return f"Error generating summary: {e}"

def run_performance_analysis():
    """Run comprehensive performance analysis"""

    analyzer = PerformanceComparisonAnalysis()"""

    try:
        # Load performance data
        data_loaded = analyzer.load_performance_data()
        if not data_loaded:
            print("# X No performance data available for analysis")
            return False

        # Generate comprehensive analysis
        analysis_results = analyzer.generate_comprehensive_analysis()

        # Generate executive summary
        executive_summary = analyzer.generate_executive_summary(analysis_results)

        # Print executive summary

        # Print key metrics
        comparison = analysis_results.get("performance_comparison", {})

        for metric, data in comparison.items():
            improvement = data.get("improvement_pct", 0)
            status = "📈" if improvement > 0 else "📉"


        return True

    except Exception as e:
        return False

if __name__ == "__main__":
    success = run_performance_analysis()
    if success:
        print("\n# Party Performance analysis completed successfully!")
        print("📋 Check the analysis report and visualizations for detailed insights")
    else:
        print("\n# Warning Performance analysis encountered issues")
        print("# Tool Please check the logs and ensure performance data is available")
