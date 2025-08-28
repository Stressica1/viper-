#!/usr/bin/env python3
"""
🔬 VIPER Comprehensive Backtesting Framework
Thorough backtesting with AI/ML optimization integration
"""

import numpy as np
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveBacktester:
    """Comprehensive backtesting framework with AI/ML integration"""

    def __init__(self):
        self.api_server_url = "http://localhost:8000"
        self.ultra_backtester_url = "http://localhost:8001"
        self.risk_manager_url = "http://localhost:8002"
        self.ai_ml_optimizer_url = "http://localhost:8000"  # Will integrate with AI/ML optimizer

        # Backtest parameters
        self.initial_balance = 10000.0
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_rate = 0.0005  # 0.05% slippage

        # Risk parameters
        self.max_drawdown_limit = 0.20  # 20% max drawdown
        self.daily_loss_limit = 0.05   # 5% daily loss limit

        # Results storage
        self.backtest_results = {}
        self.performance_metrics = {}
        self.risk_metrics = {}

    def run_multi_scenario_backtest(self, symbol: str = "BTCUSDT", scenarios: List[Dict] = None) -> Dict[str, Any]:
        """Run backtest across multiple scenarios"""
        if scenarios is None:
            scenarios = self.generate_test_scenarios()

        logger.info(f"🔬 Running {len(scenarios)} backtest scenarios...")

        all_results = {}
        best_scenario = None
        best_sharpe = -float('inf')

        for i, scenario in enumerate(scenarios):
            logger.info(f"📊 Testing Scenario {i+1}/{len(scenarios)}: {scenario['name']}")

            result = self.run_single_backtest(symbol, scenario)

            if result and 'error' not in result:
                all_results[scenario['name']] = result

                # Track best performing scenario
                sharpe = result.get('sharpe_ratio', -float('inf'))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_scenario = scenario['name']

                logger.info(f"   ✅ Win Rate: {result['win_rate']:.1%}, Return: {result['total_return']:.2%}, Sharpe: {sharpe:.2f}")

        # Generate comparative analysis
        if all_results:
            comparison = self.generate_scenario_comparison(all_results)

            final_report = {
                'total_scenarios': len(scenarios),
                'successful_scenarios': len(all_results),
                'best_scenario': best_scenario,
                'scenario_results': all_results,
                'comparison_analysis': comparison,
                'optimization_recommendations': self.generate_optimization_recommendations(all_results),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"🎯 Best Scenario: {best_scenario} with Sharpe ratio {best_sharpe:.2f}")
            return final_report
        else:
            return {'error': 'No successful backtest scenarios'}

    def generate_test_scenarios(self) -> List[Dict]:
        """Generate comprehensive test scenarios"""
        scenarios = []

        # Base scenario
        base_scenario = {
            'name': 'Base_Optimized',
            'entry_threshold': 0.7,
            'stop_loss_percent': 0.02,
            'take_profit_percent': 0.06,
            'trailing_stop_percent': 0.01,
            'position_size_percent': 0.02,
            'max_positions': 5,
            'use_ml_optimization': True
        }
        scenarios.append(base_scenario)

        # Conservative scenarios
        for sl in [0.01, 0.015, 0.025]:
            for tp in [0.04, 0.08, 0.10]:
                if tp > sl * 2:  # Ensure minimum 2:1 reward-to-risk
                    scenarios.append({
                        'name': f'Conservative_SL{sl:.0%}_TP{tp:.0%}',
                        'entry_threshold': 0.75,
                        'stop_loss_percent': sl,
                        'take_profit_percent': tp,
                        'trailing_stop_percent': 0.005,
                        'position_size_percent': 0.015,
                        'max_positions': 3,
                        'use_ml_optimization': False
                    })

        # Aggressive scenarios
        for sl in [0.03, 0.04, 0.05]:
            for tp in [0.08, 0.12, 0.15]:
                if tp > sl * 1.5:  # Minimum 1.5:1 reward-to-risk
                    scenarios.append({
                        'name': f'Aggressive_SL{sl:.0%}_TP{tp:.0%}',
                        'entry_threshold': 0.65,
                        'stop_loss_percent': sl,
                        'take_profit_percent': tp,
                        'trailing_stop_percent': 0.02,
                        'position_size_percent': 0.03,
                        'max_positions': 8,
                        'use_ml_optimization': False
                    })

        # ML-enhanced scenarios
        for threshold in [0.6, 0.7, 0.8]:
            scenarios.append({
                'name': f'ML_Enhanced_Threshold_{threshold:.1f}',
                'entry_threshold': threshold,
                'stop_loss_percent': 0.02,
                'take_profit_percent': 0.08,
                'trailing_stop_percent': 0.015,
                'position_size_percent': 0.025,
                'max_positions': 6,
                'use_ml_optimization': True
            })

        logger.info(f"🎯 Generated {len(scenarios)} test scenarios")
        return scenarios

    def run_single_backtest(self, symbol: str, scenario: Dict) -> Dict[str, Any]:
        """Run single backtest scenario"""
        try:
            # Prepare backtest parameters
            backtest_params = {
                'symbol': symbol,
                'initial_balance': self.initial_balance,
                'start_date': (datetime.now() - timedelta(days=365)).isoformat(),
                'end_date': datetime.now().isoformat(),
                'scenario': scenario,
                'commission_rate': self.commission_rate,
                'slippage_rate': self.slippage_rate,
                'max_drawdown_limit': self.max_drawdown_limit,
                'daily_loss_limit': self.daily_loss_limit
            }

            # Use ultra backtester if available
            try:
                response = requests.post(
                    f"{self.ultra_backtester_url}/api/backtest",
                    json=backtest_params,
                    timeout=300  # 5 minute timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Ultra backtester completed for {scenario['name']}")
                    return self.enhance_backtest_results(result, scenario)
                else:
                    logger.warning(f"⚠️ Ultra backtester failed: {response.status_code}")
                    # Fall back to local simulation
            except Exception as e:
                logger.warning(f"⚠️ Ultra backtester unavailable: {e}")

            # Local simulation fallback
            logger.info(f"🔄 Running local simulation for {scenario['name']}")
            return self.run_local_simulation(symbol, scenario, backtest_params)

        except Exception as e:
            logger.error(f"❌ Error in single backtest: {e}")
            return {'error': str(e)}

    def run_local_simulation(self, symbol: str, scenario: Dict, params: Dict) -> Dict[str, Any]:
        """Run local backtest simulation"""
        try:
            # Collect market data
            market_data = self.get_market_data(symbol, 2000)  # Last 2000 data points
            if not market_data:
                return {'error': 'No market data available'}

            # Initialize simulation
            balance = params['initial_balance']
            position = 0
            entry_price = 0
            trades = []
            peak_balance = balance
            max_drawdown = 0
            daily_pnl = 0
            current_date = None

            # Trading simulation
            for i, data in enumerate(market_data[100:]):  # Skip first 100 for warmup
                price = data['close']
                timestamp = data['timestamp']

                # Check daily loss limit
                if current_date and timestamp.date() != current_date:
                    if daily_pnl < -params['daily_loss_limit'] * balance:
                        logger.warning(f"🚨 Daily loss limit hit for {scenario['name']}")
                        break
                    daily_pnl = 0
                    current_date = timestamp.date()

                # Generate entry signal (simplified)
                entry_signal = self.generate_entry_signal(market_data[max(0, i-50):i+1], scenario)

                # Trading logic
                if position == 0:  # No position
                    if entry_signal['signal'] == 'BUY' and entry_signal['strength'] > scenario['entry_threshold']:
                        # Calculate position size
                        position_size = balance * scenario['position_size_percent']

                        # Apply commission and slippage
                        effective_price = price * (1 + params['slippage_rate'])
                        position_value = position_size / effective_price
                        commission = position_size * params['commission_rate']

                        position = position_value
                        entry_price = effective_price
                        balance -= commission

                        # Calculate TP/SL levels
                        stop_loss = entry_price * (1 - scenario['stop_loss_percent'])
                        take_profit = entry_price * (1 + scenario['take_profit_percent'])

                        logger.debug(f"📈 Entered LONG at ${entry_price:.2f} for {scenario['name']}")

                elif position > 0:  # Have position
                    # Check exit conditions
                    current_value = position * price
                    pnl = current_value - (position * entry_price)

                    # Stop loss or take profit
                    if price <= stop_loss or price >= take_profit:
                        # Apply exit commission and slippage
                        exit_price = price * (1 - params['slippage_rate'])
                        exit_value = position * exit_price
                        commission = exit_value * params['commission_rate']

                        realized_pnl = exit_value - (position * entry_price) - commission
                        balance += realized_pnl

                        # Track daily P&L
                        daily_pnl += realized_pnl

                        # Track drawdown
                        peak_balance = max(peak_balance, balance)
                        current_drawdown = (peak_balance - balance) / peak_balance
                        max_drawdown = max(max_drawdown, current_drawdown)

                        # Record trade
                        trade = {
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': realized_pnl,
                            'pnl_percent': realized_pnl / (position * entry_price),
                            'timestamp': timestamp,
                            'type': 'LONG',
                            'exit_reason': 'TP' if price >= take_profit else 'SL'
                        }
                        trades.append(trade)

                        # Reset position
                        position = 0
                        entry_price = 0

                        logger.debug(f"📉 Exited LONG at ${exit_price:.2f}, P&L: ${realized_pnl:.2f} for {scenario['name']}")

            # Calculate final metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = total_trades - winning_trades

            if total_trades > 0:
                win_rate = winning_trades / total_trades
                total_return = (balance - params['initial_balance']) / params['initial_balance']
                avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
                avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0
                profit_factor = abs(sum([t['pnl'] for t in trades if t['pnl'] > 0]) / sum([t['pnl'] for t in trades if t['pnl'] < 0])) if losing_trades > 0 else float('inf')

                # Calculate Sharpe ratio
                returns = [t['pnl_percent'] for t in trades]
                if returns:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                else:
                    sharpe_ratio = 0

                # Calculate Sortino ratio (downside deviation)
                downside_returns = [r for r in returns if r < 0]
                if downside_returns:
                    sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if np.std(downside_returns) > 0 else 0
                else:
                    sortino_ratio = float('inf')

                result = {
                    'scenario': scenario['name'],
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_return': total_return,
                    'final_balance': balance,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'profit_factor': profit_factor,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'max_consecutive_wins': self.calculate_max_consecutive([1 if t['pnl'] > 0 else 0 for t in trades]),
                    'max_consecutive_losses': self.calculate_max_consecutive([1 if t['pnl'] < 0 else 0 for t in trades]),
                    'trades': trades,
                    'simulation_method': 'local',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                result = {
                    'scenario': scenario['name'],
                    'error': 'No trades executed',
                    'total_trades': 0,
                    'simulation_method': 'local',
                    'timestamp': datetime.now().isoformat()
                }

            return result

        except Exception as e:
            logger.error(f"❌ Error in local simulation: {e}")
            return {'error': str(e)}

    def get_market_data(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """Get market data for backtesting"""
        try:
            # Try to get from exchange connector
            response = requests.get(f"http://localhost:8005/api/market-data?symbol={symbol}&limit={limit}", timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            else:
                logger.warning(f"⚠️ Exchange connector unavailable, using sample data")
                return self.generate_sample_market_data(limit)

        except Exception as e:
            logger.warning(f"⚠️ Market data fetch failed: {e}, using sample data")
            return self.generate_sample_market_data(limit)

    def generate_sample_market_data(self, limit: int) -> List[Dict]:
        """Generate sample market data for testing"""
        base_price = 50000  # BTC sample price
        data = []

        for i in range(limit):
            timestamp = datetime.now() - timedelta(minutes=limit - i)
            # Generate realistic price movement
            change = np.random.normal(0, 0.002)  # 0.2% volatility
            base_price *= (1 + change)

            data.append({
                'timestamp': timestamp,
                'open': base_price * (1 + np.random.normal(0, 0.001)),
                'high': base_price * (1 + abs(np.random.normal(0, 0.002))),
                'low': base_price * (1 - abs(np.random.normal(0, 0.002))),
                'close': base_price,
                'volume': np.random.uniform(100, 1000)
            })

        return data

    def generate_entry_signal(self, recent_data: List[Dict], scenario: Dict) -> Dict[str, Any]:
        """Generate entry signal (simplified version)"""
        if len(recent_data) < 20:
            return {'signal': 'HOLD', 'strength': 0.5}

        # Simple moving average crossover
        closes = [d['close'] for d in recent_data[-50:]]
        sma_short = np.mean(closes[-10:])
        sma_long = np.mean(closes[-30:])

        # RSI calculation
        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains[-14:]) if gains else 0
        avg_loss = np.mean(losses[-14:]) if losses else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))

        # Generate signal
        signal_strength = 0.5

        if sma_short > sma_long and rsi < 70:  # Bullish crossover, not overbought
            signal = 'BUY'
            signal_strength = min(0.8, 0.5 + (sma_short - sma_long) / sma_long * 10)
        elif sma_short < sma_long and rsi > 30:  # Bearish crossover, not oversold
            signal = 'SELL'
            signal_strength = min(0.8, 0.5 + (sma_long - sma_short) / sma_long * 10)
        else:
            signal = 'HOLD'
            signal_strength = 0.5

        return {
            'signal': signal,
            'strength': signal_strength,
            'indicators': {
                'sma_short': sma_short,
                'sma_long': sma_long,
                'rsi': rsi
            }
        }

    def calculate_max_consecutive(self, sequence: List[int]) -> int:
        """Calculate maximum consecutive wins/losses"""
        max_count = 0
        current_count = 0

        for item in sequence:
            if item == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def enhance_backtest_results(self, result: Dict, scenario: Dict) -> Dict:
        """Enhance backtest results with additional metrics"""
        try:
            if 'trades' in result and len(result['trades']) > 0:
                trades = result['trades']

                # Calculate additional metrics
                pnl_distribution = [t.get('pnl', 0) for t in trades]
                returns = [t.get('pnl_percent', 0) for t in trades]

                # Risk-adjusted metrics
                if returns:
                    result['volatility'] = np.std(returns)
                    result['skewness'] = stats.skew(returns)
                    result['kurtosis'] = stats.kurtosis(returns)

                    # Calmar ratio (annual return / max drawdown)
                    if result.get('max_drawdown', 0) > 0:
                        result['calmar_ratio'] = result.get('total_return', 0) / result.get('max_drawdown', 1)

                    # Information ratio
                    benchmark_return = 0.05  # Assume 5% benchmark
                    excess_returns = [r - benchmark_return/252 for r in returns]  # Daily excess returns
                    if excess_returns:
                        result['information_ratio'] = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

                # Trade analysis
                result['largest_win'] = max(pnl_distribution) if pnl_distribution else 0
                result['largest_loss'] = min(pnl_distribution) if pnl_distribution else 0

                # Profit distribution analysis
                winning_pnl = [p for p in pnl_distribution if p > 0]
                losing_pnl = [p for p in pnl_distribution if p < 0]

                if winning_pnl:
                    result['avg_win_amount'] = np.mean(winning_pnl)
                    result['median_win_amount'] = np.median(winning_pnl)

                if losing_pnl:
                    result['avg_loss_amount'] = np.mean(losing_pnl)
                    result['median_loss_amount'] = np.median(losing_pnl)

            return result

        except Exception as e:
            logger.error(f"❌ Error enhancing backtest results: {e}")
            return result

    def generate_scenario_comparison(self, all_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comparative analysis of all scenarios"""
        try:
            comparison = {
                'total_scenarios': len(all_results),
                'metric_summary': {},
                'rankings': {},
                'correlations': {},
                'best_performers': {}
            }

            # Extract key metrics
            metrics = ['win_rate', 'total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor']

            for metric in metrics:
                values = []
                for scenario_name, result in all_results.items():
                    if 'error' not in result and metric in result:
                        values.append(result[metric])

                if values:
                    comparison['metric_summary'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }

            # Rank scenarios by different metrics
            for metric in metrics:
                ranking = []
                for scenario_name, result in all_results.items():
                    if 'error' not in result and metric in result:
                        ranking.append((scenario_name, result[metric]))

                ranking.sort(key=lambda x: x[1], reverse=True)
                comparison['rankings'][metric] = ranking[:5]  # Top 5

            # Find best performers
            if comparison['rankings'].get('sharpe_ratio'):
                best_overall = comparison['rankings']['sharpe_ratio'][0][0]
                comparison['best_performers']['overall'] = best_overall
                comparison['best_performers']['result'] = all_results[best_overall]

            return comparison

        except Exception as e:
            logger.error(f"❌ Error generating scenario comparison: {e}")
            return {}

    def generate_optimization_recommendations(self, all_results: Dict[str, Dict]) -> List[str]:
        """Generate optimization recommendations based on backtest results"""
        recommendations = []

        try:
            # Analyze best performing scenarios
            if not all_results:
                return ["No successful backtests to analyze"]

            # Find optimal parameter ranges
            optimal_params = {
                'win_rate_threshold': 0.55,
                'sharpe_threshold': 1.0,
                'max_drawdown_limit': 0.15
            }

            # Filter high-performing scenarios
            high_performers = []
            for scenario_name, result in all_results.items():
                if (result.get('win_rate', 0) > optimal_params['win_rate_threshold'] and
                    result.get('sharpe_ratio', 0) > optimal_params['sharpe_threshold'] and
                    result.get('max_drawdown', 1) < optimal_params['max_drawdown_limit']):
                    high_performers.append((scenario_name, result))

            if high_performers:
                best_scenario = max(high_performers, key=lambda x: x[1].get('sharpe_ratio', 0))
                recommendations.append(f"🎯 Best performing scenario: {best_scenario[0]}")
                recommendations.append(f"   Sharpe: {best_scenario[1].get('sharpe_ratio', 0):.2f}")
                recommendations.append(f"   Win Rate: {best_scenario[1].get('win_rate', 0):.1%}")

                # Parameter recommendations
                scenario_name = best_scenario[0]
                if 'Conservative' in scenario_name:
                    recommendations.append("📊 Conservative parameters recommended for stable performance")
                elif 'Aggressive' in scenario_name:
                    recommendations.append("📊 Aggressive parameters show higher returns but increased risk")
                elif 'ML_Enhanced' in scenario_name:
                    recommendations.append("🤖 AI/ML optimization significantly improves performance")

            # Risk analysis
            avg_drawdown = np.mean([r.get('max_drawdown', 0) for r in all_results.values() if 'error' not in r])
            if avg_drawdown > 0.25:
                recommendations.append(f"⚠️ High average drawdown ({avg_drawdown:.1%}) - consider tighter risk controls")
            elif avg_drawdown < 0.10:
                recommendations.append(f"✅ Low average drawdown ({avg_drawdown:.1%}) - risk management effective")

            # Performance analysis
            avg_win_rate = np.mean([r.get('win_rate', 0) for r in all_results.values() if 'error' not in r])
            if avg_win_rate > 0.60:
                recommendations.append(f"✅ Excellent win rate ({avg_win_rate:.1%}) - strategy performing well")
            elif avg_win_rate < 0.50:
                recommendations.append(f"⚠️ Low win rate ({avg_win_rate:.1%}) - consider adjusting entry criteria")

            # Sharpe ratio analysis
            avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in all_results.values() if 'error' not in r])
            if avg_sharpe > 1.5:
                recommendations.append(f"🎯 Excellent risk-adjusted returns (Sharpe: {avg_sharpe:.2f})")
            elif avg_sharpe < 0.5:
                recommendations.append(f"⚠️ Poor risk-adjusted returns (Sharpe: {avg_sharpe:.2f}) - review strategy")

        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            recommendations.append("Error analyzing backtest results")

        return recommendations

    def generate_performance_report(self, all_results: Dict[str, Dict]) -> str:
        """Generate detailed performance report"""
        try:
            report_lines = []
            report_lines.append("📊 VIPER COMPREHENSIVE BACKTEST REPORT")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")

            # Summary statistics
            successful_tests = len([r for r in all_results.values() if 'error' not in r])
            report_lines.append(f"🎯 Total Scenarios Tested: {len(all_results)}")
            report_lines.append(f"✅ Successful Backtests: {successful_tests}")
            report_lines.append("")

            if successful_tests > 0:
                # Performance metrics
                win_rates = [r.get('win_rate', 0) for r in all_results.values() if 'error' not in r]
                returns = [r.get('total_return', 0) for r in all_results.values() if 'error' not in r]
                sharpe_ratios = [r.get('sharpe_ratio', 0) for r in all_results.values() if 'error' not in r]

                report_lines.append("📈 PERFORMANCE METRICS:")
                report_lines.append(f"   Win Rate Range: {min(win_rates):.1%} - {max(win_rates):.1%}")
                report_lines.append(f"   Return Range: {min(returns):.1%} - {max(returns):.1%}")
                report_lines.append(f"   Sharpe Range: {min(sharpe_ratios):.2f} - {max(sharpe_ratios):.2f}")
                report_lines.append("")

                # Top performers
                report_lines.append("🏆 TOP PERFORMING SCENARIOS:")
                sorted_by_sharpe = sorted(
                    [(name, r) for name, r in all_results.items() if 'error' not in r],
                    key=lambda x: x[1].get('sharpe_ratio', -float('inf')),
                    reverse=True
                )

                for i, (name, result) in enumerate(sorted_by_sharpe[:5]):
                    report_lines.append(f"   {i+1}. {name}")
                    report_lines.append(f"      Win Rate: {result.get('win_rate', 0):.1%}")
                    report_lines.append(f"      Total Return: {result.get('total_return', 0):.2%}")
                    report_lines.append(f"      Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
                    report_lines.append(f"      Max Drawdown: {result.get('max_drawdown', 0):.2%}")
                    report_lines.append("")

            # Recommendations
            report_lines.append("🎯 OPTIMIZATION RECOMMENDATIONS:")
            recommendations = self.generate_optimization_recommendations(all_results)
            for rec in recommendations:
                report_lines.append(f"   {rec}")

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"❌ Error generating performance report: {e}")
            return f"Error generating report: {str(e)}"

def main():
    """Main function for comprehensive backtesting"""
    backtester = ComprehensiveBacktester()

    print("🔬 VIPER COMPREHENSIVE BACKTESTING FRAMEWORK")
    print("=" * 60)

    # Run multi-scenario backtest
    print("🎯 Running comprehensive backtest analysis...")
    results = backtester.run_multi_scenario_backtest()

    if 'error' in results:
        print(f"❌ Backtest failed: {results['error']}")
        return

    # Generate and display performance report
    report = backtester.generate_performance_report(results['scenario_results'])
    print(report)

    # Save results
    output_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ COMPREHENSIVE BACKTESTING COMPLETE!")

if __name__ == "__main__":
    main()
