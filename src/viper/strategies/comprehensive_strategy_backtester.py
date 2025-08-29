#!/usr/bin/env python3
"""
# Rocket COMPREHENSIVE STRATEGY BACKTESTER FOR LOWER TIMEFRAMES
Advanced backtesting engine specifically designed for 30min and under timeframes

Features:
# Check Multiple strategy comparison for lower timeframes (5m, 15m, 30m)
# Check Enhanced visual display with interactive charts and tables
# Check Monte Carlo simulation for robust results
# Check Walk-forward analysis for out-of-sample validation
# Check Performance ranking and strategy selection
# Check Real-time progress monitoring with beautiful displays
# Check Automated best strategy selection for live trading
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box
from rich.text import Text

from dataclasses import dataclass, asdict

# Import existing VIPER components
sys.path.append(str(Path(__file__).parent))

try:
    pass
except ImportError as e:
    logging.warning(f"Some imports failed: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - STRATEGY_BACKTESTER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    timeframes: List[str]
    parameters: Dict[str, Any]
    description: str
    risk_level: str  # 'low', 'medium', 'high'

@dataclass
class BacktestResult:
    """Comprehensive backtest result"""
    strategy_name: str
    timeframe: str
    symbol: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    recovery_factor: float
    confidence_interval_95: Tuple[float, float]
    monte_carlo_var_95: float
    walk_forward_consistency: float
    execution_time: float
    final_balance: float

@dataclass
class StrategyRanking:
    """Strategy ranking with scoring metrics"""
    strategy_name: str
    timeframe: str
    composite_score: float
    rank: int
    risk_adjusted_return: float
    consistency_score: float
    robustness_score: float
    recommendation: str

class ComprehensiveStrategyBacktester:
    """
    Comprehensive backtesting system for lower timeframe strategies
    """
    
    def __init__(self):
        self.console = Console()
        self.results_path = Path("backtest_results")
        self.results_path.mkdir(exist_ok=True)
        
        # Strategy configurations
        self.strategies = self._initialize_strategies()
        self.timeframes = ['5m', '15m', '30m']  # Lower timeframes focus
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
        
        # Backtesting parameters
        self.initial_balance = 10000.0
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.monte_carlo_simulations = 1000
        self.walk_forward_periods = 5
        
        # Results storage
        self.all_results: List[BacktestResult] = []
        self.strategy_rankings: List[StrategyRanking] = []
        
        logger.info("# Rocket Comprehensive Strategy Backtester initialized")
        
    def _initialize_strategies(self) -> List[StrategyConfig]:
        """Initialize available trading strategies"""
        return [
            StrategyConfig(
                name="VIPER_Momentum",
                timeframes=['5m', '15m', '30m'],
                parameters={
                    'momentum_period': 14,
                    'volatility_filter': 0.02,
                    'trend_strength_threshold': 0.7,
                    'entry_confirmation': True
                },
                description="VIPER momentum-based strategy with volatility filtering",
                risk_level="medium"
            ),
            StrategyConfig(
                name="Predictive_Ranges",
                timeframes=['5m', '15m', '30m'],
                parameters={
                    'lookback_periods': [20, 50],
                    'projection_periods': [5, 10],
                    'confidence_threshold': 0.7,
                    'confluence_threshold': 0.8
                },
                description="Predictive support/resistance range strategy",
                risk_level="low"
            ),
            StrategyConfig(
                name="Enhanced_Scalper",
                timeframes=['5m', '15m'],
                parameters={
                    'fast_ma': 5,
                    'slow_ma': 20,
                    'atr_period': 14,
                    'atr_multiplier': 2.0,
                    'volume_threshold': 1.5
                },
                description="Enhanced scalping strategy for quick entries/exits",
                risk_level="high"
            ),
            StrategyConfig(
                name="Trend_Following_Optimized",
                timeframes=['15m', '30m'],
                parameters={
                    'trend_period': 50,
                    'signal_period': 14,
                    'exit_threshold': 0.5,
                    'risk_multiplier': 1.5
                },
                description="Optimized trend following for lower timeframes",
                risk_level="medium"
            ),
            StrategyConfig(
                name="Mean_Reversion_Pro",
                timeframes=['5m', '15m'],
                parameters={
                    'bollinger_period': 20,
                    'bollinger_std': 2.0,
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70
                },
                description="Professional mean reversion strategy",
                risk_level="medium"
            )
        ]
    
    async def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """Run comprehensive backtesting for all strategies and timeframes"""
        console.print("\n# Rocket [bold blue]COMPREHENSIVE STRATEGY BACKTESTING STARTED[/bold blue]")
        console.print(f"# Chart Testing {len(self.strategies)} strategies across {len(self.timeframes)} timeframes")
        console.print(f"# Target Focus: Lower timeframes (30min and under)")
        
        start_time = datetime.now()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            
            # Calculate total tasks
            total_tasks = len(self.strategies) * len(self.timeframes) * len(self.symbols)
            main_task = progress.add_task(f"🔄 Running {total_tasks} backtests...", total=total_tasks)
            
            # Run backtests for all combinations
            tasks = []
            for strategy in self.strategies:
                for timeframe in strategy.timeframes:
                    if timeframe in self.timeframes:  # Only lower timeframes
                        for symbol in self.symbols:
                            tasks.append((strategy, timeframe, symbol))
            
            # Execute backtests in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self._run_single_backtest, strategy, timeframe, symbol)
                    for strategy, timeframe, symbol in tasks
                ]
                
                for future in as_completed(futures):
                    try:
                        result = await asyncio.wrap_future(future)
                        if result:
                            self.all_results.append(result)
                        progress.advance(main_task)
                    except Exception as e:
                        logger.error(f"Backtest failed: {e}")
                        progress.advance(main_task)
        
        # Analyze results and create rankings
        await self._analyze_and_rank_results()
        
        # Generate comprehensive report
        report = await self._generate_comprehensive_report()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        console.print(f"\n# Check [bold green]Backtesting completed in {execution_time:.2f} seconds[/bold green]")
        
        return report
    
    def _run_single_backtest(self, strategy: StrategyConfig, timeframe: str, symbol: str) -> Optional[BacktestResult]:
        """Run a single backtest for strategy-timeframe-symbol combination"""
        try:
            # Generate sample historical data (in production, use real data)
            historical_data = self._generate_sample_data(symbol, timeframe, days=90)
            
            # Run strategy simulation
            signals = self._simulate_strategy(strategy, historical_data, timeframe)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(signals, historical_data, strategy.name, timeframe, symbol)
            
            # Run Monte Carlo simulation
            monte_carlo_results = self._run_monte_carlo_simulation(signals, 100)  # Reduced for speed
            
            # Walk-forward analysis
            walk_forward_consistency = self._calculate_walk_forward_consistency(strategy, historical_data, timeframe)
            
            return BacktestResult(
                strategy_name=strategy.name,
                timeframe=timeframe,
                symbol=symbol,
                total_return=performance['total_return'],
                sharpe_ratio=performance['sharpe_ratio'],
                max_drawdown=performance['max_drawdown'],
                win_rate=performance['win_rate'],
                profit_factor=performance['profit_factor'],
                total_trades=performance['total_trades'],
                avg_trade_return=performance['avg_trade_return'],
                volatility=performance['volatility'],
                calmar_ratio=performance['calmar_ratio'],
                sortino_ratio=performance['sortino_ratio'],
                recovery_factor=performance['recovery_factor'],
                confidence_interval_95=monte_carlo_results['ci_95'],
                monte_carlo_var_95=monte_carlo_results['var_95'],
                walk_forward_consistency=walk_forward_consistency,
                execution_time=performance['execution_time'],
                final_balance=performance['final_balance']
            )
            
        except Exception as e:
            logger.error(f"Error in backtest for {strategy.name} {timeframe} {symbol}: {e}")
            return None
    
    def _generate_sample_data(self, symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame:
        """Generate realistic sample OHLCV data for backtesting"""
        # Convert timeframe to frequency
        freq_map = {'5m': 5, '15m': 15, '30m': 30, '1h': 60}
        freq_minutes = freq_map.get(timeframe, 5)
        
        # Calculate number of candles
        candles_per_day = (24 * 60) // freq_minutes
        total_candles = days * candles_per_day
        
        # Generate realistic price data using random walk with drift
        np.random.seed(42)  # For reproducible results
        base_price = {'BTCUSDT': 50000, 'ETHUSDT': 3000, 'ADAUSDT': 0.5, 'SOLUSDT': 100, 'DOTUSDT': 10}.get(symbol, 1000)
        
        returns = np.random.normal(0.0001, 0.02, total_candles)  # Small positive drift with realistic volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i in range(total_candles):
            price = prices[i]
            noise = np.random.uniform(-0.01, 0.01)
            
            high = price * (1 + abs(noise))
            low = price * (1 - abs(noise))
            open_price = price * (1 + np.random.uniform(-0.005, 0.005))
            close_price = price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': datetime.now() - timedelta(minutes=freq_minutes * (total_candles - i)),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _simulate_strategy(self, strategy: StrategyConfig, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Simulate strategy signals on historical data"""
        signals = data.copy()
        signals['signal'] = 0
        signals['position'] = 0
        signals['returns'] = 0
        
        # Simple strategy simulation based on strategy type
        if 'momentum' in strategy.name.lower():
            # Momentum strategy
            momentum_period = strategy.parameters.get('momentum_period', 14)
            signals['momentum'] = signals['close'].pct_change(momentum_period)
            signals['signal'] = np.where(signals['momentum'] > 0.01, 1, 
                                       np.where(signals['momentum'] < -0.01, -1, 0))
        
        elif 'predictive' in strategy.name.lower():
            # Predictive ranges strategy
            lookback = strategy.parameters.get('lookback_periods', [20])[0]
            signals['sma'] = signals['close'].rolling(lookback).mean()
            signals['signal'] = np.where(signals['close'] > signals['sma'] * 1.01, 1,
                                       np.where(signals['close'] < signals['sma'] * 0.99, -1, 0))
        
        elif 'scalper' in strategy.name.lower():
            # Scalping strategy
            fast_ma = strategy.parameters.get('fast_ma', 5)
            slow_ma = strategy.parameters.get('slow_ma', 20)
            signals['fast_ma'] = signals['close'].rolling(fast_ma).mean()
            signals['slow_ma'] = signals['close'].rolling(slow_ma).mean()
            signals['signal'] = np.where(signals['fast_ma'] > signals['slow_ma'], 1,
                                       np.where(signals['fast_ma'] < signals['slow_ma'], -1, 0))
        
        elif 'trend' in strategy.name.lower():
            # Trend following
            trend_period = strategy.parameters.get('trend_period', 50)
            signals['trend'] = signals['close'].rolling(trend_period).mean()
            signals['signal'] = np.where(signals['close'] > signals['trend'], 1,
                                       np.where(signals['close'] < signals['trend'], -1, 0))
        
        elif 'mean_reversion' in strategy.name.lower():
            # Mean reversion
            period = strategy.parameters.get('bollinger_period', 20)
            std_dev = strategy.parameters.get('bollinger_std', 2.0)
            signals['sma'] = signals['close'].rolling(period).mean()
            signals['std'] = signals['close'].rolling(period).std()
            signals['upper_band'] = signals['sma'] + (signals['std'] * std_dev)
            signals['lower_band'] = signals['sma'] - (signals['std'] * std_dev)
            
            signals['signal'] = np.where(signals['close'] < signals['lower_band'], 1,
                                       np.where(signals['close'] > signals['upper_band'], -1, 0))
        
        # Calculate position and returns
        signals['position'] = signals['signal'].shift(1).fillna(0)
        signals['returns'] = signals['position'] * signals['close'].pct_change()
        
        return signals
    
    def _calculate_performance_metrics(self, signals: pd.DataFrame, data: pd.DataFrame, 
                                     strategy_name: str, timeframe: str, symbol: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        start_time = datetime.now()
        
        # Basic returns calculation
        returns = signals['returns'].dropna()
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252 * 24 * 60 / {'5m': 5, '15m': 15, '30m': 30}[timeframe])
        sharpe_ratio = (returns.mean() * 252 * 24 * 60 / {'5m': 5, '15m': 15, '30m': 30}[timeframe]) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Trade statistics
        trades = signals[signals['signal'] != 0]
        total_trades = len(trades)
        winning_trades = len(trades[trades['returns'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades[trades['returns'] > 0]['returns'].sum()
        gross_loss = abs(trades[trades['returns'] < 0]['returns'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Advanced metrics
        negative_returns = returns[returns < 0]
        sortino_ratio = (returns.mean() * 252) / (negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 else 0
        calmar_ratio = (total_return * 252) / max_drawdown if max_drawdown > 0 else 0
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'avg_trade_return': returns.mean(),
            'volatility': volatility,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'recovery_factor': recovery_factor,
            'execution_time': execution_time,
            'final_balance': self.initial_balance * (1 + total_return)
        }
    
    def _run_monte_carlo_simulation(self, signals: pd.DataFrame, simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation on backtest results"""
        returns = signals['returns'].dropna()
        
        if len(returns) == 0:
            return {'ci_95': (0, 0), 'var_95': 0}
        
        # Bootstrap simulation
        simulation_results = []
        for _ in range(simulations):
            bootstrap_returns = np.secrets.choice(returns, size=len(returns), replace=True)
            total_return = (1 + bootstrap_returns).prod() - 1
            simulation_results.append(total_return)
        
        simulation_results = np.array(simulation_results)
        
        # Calculate confidence interval and VaR
        ci_95_lower = np.percentile(simulation_results, 2.5)
        ci_95_upper = np.percentile(simulation_results, 97.5)
        var_95 = np.percentile(simulation_results, 5)
        
        return {
            'ci_95': (ci_95_lower, ci_95_upper),
            'var_95': var_95,
            'mean': np.mean(simulation_results),
            'std': np.std(simulation_results)
        }
    
    def _calculate_walk_forward_consistency(self, strategy: StrategyConfig, data: pd.DataFrame, timeframe: str) -> float:
        """Calculate walk-forward analysis consistency score"""
        try:
            # Split data into periods for walk-forward analysis
            split_size = len(data) // self.walk_forward_periods
            period_returns = []
            
            for i in range(self.walk_forward_periods):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < self.walk_forward_periods - 1 else len(data)
                
                if end_idx <= start_idx:
                    break
                
                period_data = data.iloc[start_idx:end_idx]
                if len(period_data) < 20:  # Minimum data points
                    continue
                
                signals = self._simulate_strategy(strategy, period_data, timeframe)
                returns = signals['returns'].dropna()
                
                if len(returns) > 0:
                    period_return = (1 + returns).prod() - 1
                    period_returns.append(period_return)
            
            if len(period_returns) < 2:
                return 0.5
            
            # Calculate consistency as inverse of coefficient of variation
            mean_return = np.mean(period_returns)
            std_return = np.std(period_returns)
            
            if std_return == 0:
                return 1.0
            
            cv = abs(std_return / mean_return) if mean_return != 0 else float('inf')
            consistency = 1 / (1 + cv)  # Higher consistency for lower CV
            
            return min(consistency, 1.0)
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {e}")
            return 0.5
    
    async def _analyze_and_rank_results(self):
        """Analyze results and create strategy rankings"""
        if not self.all_results:
            return
        
        console.print("\n# Chart [bold blue]ANALYZING AND RANKING STRATEGIES[/bold blue]")
        
        # Group results by strategy and timeframe
        strategy_timeframe_results = {}
        
        for result in self.all_results:
            key = f"{result.strategy_name}_{result.timeframe}"
            if key not in strategy_timeframe_results:
                strategy_timeframe_results[key] = []
            strategy_timeframe_results[key].append(result)
        
        # Calculate aggregate metrics for each strategy-timeframe combination
        rankings = []
        
        for key, results in strategy_timeframe_results.items():
            strategy_name, timeframe = key.split('_', 1)
            
            # Calculate aggregate metrics across all symbols
            avg_total_return = np.mean([r.total_return for r in results])
            avg_sharpe_ratio = np.mean([r.sharpe_ratio for r in results])
            avg_max_drawdown = np.mean([r.max_drawdown for r in results])
            avg_win_rate = np.mean([r.win_rate for r in results])
            avg_consistency = np.mean([r.walk_forward_consistency for r in results])
            
            # Calculate composite scores
            risk_adjusted_return = avg_total_return / (avg_max_drawdown + 0.01)  # Avoid division by zero
            consistency_score = avg_consistency
            robustness_score = min(avg_win_rate * 2, 1.0)  # Win rate contribution
            
            # Composite score (weighted combination)
            composite_score = (
                risk_adjusted_return * 0.4 +
                avg_sharpe_ratio * 0.3 +
                consistency_score * 0.2 +
                robustness_score * 0.1
            )
            
            # Generate recommendation
            if composite_score > 0.8:
                recommendation = "🟢 HIGHLY RECOMMENDED"
            elif composite_score > 0.6:
                recommendation = "🟡 RECOMMENDED"
            elif composite_score > 0.4:
                recommendation = "🟠 MODERATE"
            else:
                recommendation = "🔴 NOT RECOMMENDED"
            
            rankings.append(StrategyRanking(
                strategy_name=strategy_name,
                timeframe=timeframe,
                composite_score=composite_score,
                rank=0,  # Will be set after sorting
                risk_adjusted_return=risk_adjusted_return,
                consistency_score=consistency_score,
                robustness_score=robustness_score,
                recommendation=recommendation
            ))
        
        # Sort by composite score and assign ranks
        rankings.sort(key=lambda x: x.composite_score, reverse=True)
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1
        
        self.strategy_rankings = rankings
        
        console.print(f"# Check Analyzed {len(rankings)} strategy-timeframe combinations")
    
    async def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtesting report with enhanced display"""
        console.print("\n📈 [bold blue]GENERATING COMPREHENSIVE REPORT[/bold blue]")
        
        # Display top strategies table
        self._display_top_strategies_table()
        
        # Display detailed performance metrics
        await self._display_performance_charts()
        
        # Generate best strategy recommendation
        best_strategy = self._get_best_strategy_recommendation()
        
        # Save detailed results
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_backtests': len(self.all_results),
            'strategies_tested': len(self.strategies),
            'timeframes': self.timeframes,
            'symbols': self.symbols,
            'best_strategy': best_strategy,
            'all_rankings': [asdict(r) for r in self.strategy_rankings],
            'detailed_results': [asdict(r) for r in self.all_results]
        }
        
        # Save report to file
        report_file = self.results_path / f"comprehensive_backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        console.print(f"\n📄 [bold green]Report saved to: {report_file}[/bold green]")
        
        return report_data
    
    def _display_top_strategies_table(self):
        """Display top strategies in a beautiful table"""
        console.print("\n🏆 [bold yellow]TOP STRATEGY RANKINGS[/bold yellow]")
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Strategy", style="cyan", width=20)
        table.add_column("Timeframe", style="green", width=10)
        table.add_column("Score", justify="right", style="yellow", width=8)
        table.add_column("Risk Adj Return", justify="right", style="blue", width=12)
        table.add_column("Consistency", justify="right", style="magenta", width=12)
        table.add_column("Recommendation", width=20)
        
        for ranking in self.strategy_rankings[:10]:  # Top 10
            table.add_row(
                str(ranking.rank),
                ranking.strategy_name,
                ranking.timeframe,
                f"{ranking.composite_score:.3f}",
                f"{ranking.risk_adjusted_return:.3f}",
                f"{ranking.consistency_score:.3f}",
                ranking.recommendation
            )
        
        console.print(table)
    
    async def _display_performance_charts(self):
        """Generate and display performance visualization charts"""
        console.print("\n# Chart [bold blue]GENERATING PERFORMANCE VISUALIZATIONS[/bold blue]")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📈 STRATEGY PERFORMANCE ANALYSIS - LOWER TIMEFRAMES', fontsize=16, fontweight='bold')
        
        # 1. Strategy Performance Comparison
        ax1 = axes[0, 0]
        strategies = [r.strategy_name for r in self.strategy_rankings[:8]]
        scores = [r.composite_score for r in self.strategy_rankings[:8]]
        colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
        
        bars = ax1.barh(strategies, scores, color=colors)
        ax1.set_xlabel('Composite Score')
        ax1.set_title('🏆 Strategy Performance Ranking')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        # 2. Risk-Return Scatter Plot
        ax2 = axes[0, 1]
        risk_adj_returns = [r.risk_adjusted_return for r in self.strategy_rankings]
        consistency_scores = [r.consistency_score for r in self.strategy_rankings]
        
        # Ensure colors array matches data size
        colors_for_scatter = [r.composite_score for r in self.strategy_rankings]
        
        scatter = ax2.scatter(risk_adj_returns, consistency_scores, 
                            c=colors_for_scatter, cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Risk-Adjusted Return')
        ax2.set_ylabel('Consistency Score')
        ax2.set_title('# Target Risk-Return vs Consistency')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label='Composite Score')
        
        # 3. Timeframe Performance Distribution
        ax3 = axes[1, 0]
        timeframe_performance = {}
        for result in self.all_results:
            if result.timeframe not in timeframe_performance:
                timeframe_performance[result.timeframe] = []
            timeframe_performance[result.timeframe].append(result.total_return)
        
        timeframes = list(timeframe_performance.keys())
        performance_data = [timeframe_performance[tf] for tf in timeframes]
        
        bp = ax3.boxplot(performance_data, labels=timeframes, patch_artist=True)
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax3.set_xlabel('Timeframe')
        ax3.set_ylabel('Total Return')
        ax3.set_title('# Chart Performance Distribution by Timeframe')
        ax3.grid(True, alpha=0.3)
        
        # 4. Win Rate vs Profit Factor
        ax4 = axes[1, 1]
        win_rates = [r.win_rate for r in self.all_results]
        profit_factors = [min(r.profit_factor, 5) for r in self.all_results]  # Cap at 5 for visualization
        
        ax4.scatter(win_rates, profit_factors, alpha=0.6, c='purple', s=50)
        ax4.set_xlabel('Win Rate')
        ax4.set_ylabel('Profit Factor (capped at 5)')
        ax4.set_title('💰 Win Rate vs Profit Factor')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the chart
        chart_file = self.results_path / f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        console.print(f"# Chart Performance charts saved to: {chart_file}")
        
        # Display chart path
        console.print(f"🖼️  [bold blue]Charts generated and saved[/bold blue]")
        
        plt.close()
    
    def _get_best_strategy_recommendation(self) -> Dict[str, Any]:
        """Get the best strategy recommendation for each timeframe"""
        console.print("\n# Target [bold green]BEST STRATEGY RECOMMENDATIONS[/bold green]")
        
        best_strategies = {}
        
        # Find best strategy for each timeframe
        for timeframe in self.timeframes:
            timeframe_rankings = [r for r in self.strategy_rankings if r.timeframe == timeframe]
            if timeframe_rankings:
                best = timeframe_rankings[0]  # Already sorted by composite score
                best_strategies[timeframe] = {
                    'strategy_name': best.strategy_name,
                    'composite_score': best.composite_score,
                    'risk_adjusted_return': best.risk_adjusted_return,
                    'consistency_score': best.consistency_score,
                    'recommendation': best.recommendation
                }
        
        # Display recommendations in a panel
        recommendation_text = Text()
        recommendation_text.append("# Rocket OPTIMAL STRATEGIES FOR LOWER TIMEFRAMES\n\n", style="bold blue")
        
        for timeframe, strategy in best_strategies.items():
            recommendation_text.append(f"⏰ {timeframe}: ", style="bold yellow")
            recommendation_text.append(f"{strategy['strategy_name']}\n", style="cyan")
            recommendation_text.append(f"   # Chart Score: {strategy['composite_score']:.3f} | ", style="dim")
            recommendation_text.append(f"# Target Risk-Adj Return: {strategy['risk_adjusted_return']:.3f}\n", style="dim")
            recommendation_text.append(f"   {strategy['recommendation']}\n\n", style="green")
        
        # Overall best strategy
        if self.strategy_rankings:
            overall_best = self.strategy_rankings[0]
            recommendation_text.append("🏆 OVERALL BEST PERFORMER:\n", style="bold magenta")
            recommendation_text.append(f"   {overall_best.strategy_name} ({overall_best.timeframe})\n", style="bold cyan")
            recommendation_text.append(f"   Composite Score: {overall_best.composite_score:.3f}", style="bold green")
        
        console.print(Panel(recommendation_text, title="# Target STRATEGY RECOMMENDATIONS", border_style="green"))
        
        return {
            'by_timeframe': best_strategies,
            'overall_best': {
                'strategy_name': self.strategy_rankings[0].strategy_name,
                'timeframe': self.strategy_rankings[0].timeframe,
                'composite_score': self.strategy_rankings[0].composite_score
            } if self.strategy_rankings else None
        }

async def main():
    """Main execution function"""
    try:
        console.print("\n# Rocket [bold blue]COMPREHENSIVE STRATEGY BACKTESTER[/bold blue]")
        console.print("# Chart [yellow]Focusing on Lower Timeframes (30min and under)[/yellow]\n")
        
        backtester = ComprehensiveStrategyBacktester()
        
        # Run comprehensive backtesting
        report = await backtester.run_comprehensive_backtest()
        
        console.print("\n# Check [bold green]BACKTESTING COMPLETED SUCCESSFULLY![/bold green]")
        console.print(f"📄 [cyan]Total backtests: {report['total_backtests']}[/cyan]")
        console.print(f"# Target [cyan]Best overall strategy: {report['best_strategy']['overall_best']['strategy_name']} ({report['best_strategy']['overall_best']['timeframe']})[/cyan]")
        
        return report
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        console.print(f"# X [bold red]Error: {e}[/bold red]")
        return None

if __name__ == "__main__":
    asyncio.run(main())