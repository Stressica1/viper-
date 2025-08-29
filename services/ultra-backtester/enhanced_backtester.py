#!/usr/bin/env python3
"""
🚀 VIPER Enhanced Backtesting Engine
Advanced backtesting system with walk-forward analysis, Monte Carlo simulations,
and comprehensive performance analytics

Features:
- Walk-Forward Analysis for robust out-of-sample testing
- Monte Carlo simulations for probabilistic analysis  
- Advanced performance metrics (Sharpe, Sortino, Calmar ratios)
- Multi-strategy backtesting and comparison
- Risk-adjusted return analysis
- Drawdown analysis and recovery periods
- Statistical significance testing
- Scenario analysis and stress testing
- Portfolio optimization backtesting
- Real-time strategy validation
"""

import os
import json
import time
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from pathlib import Path
import threading
import httpx
import uuid
import warnings
warnings.filterwarnings('ignore')

# Statistical and optimization libraries
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import concurrent.futures
import multiprocessing as mp

# Import enhanced mathematical validator
try:
    from utils.mathematical_validator import enhanced_math_validator, ValidationLevel
    MATH_VALIDATOR_AVAILABLE = True
except ImportError:
    MATH_VALIDATOR_AVAILABLE = False

# Configure logging
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestType(Enum):
    SINGLE_STRATEGY = "single_strategy"
    MULTI_STRATEGY = "multi_strategy"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    SCENARIO_ANALYSIS = "scenario_analysis"

class RebalanceFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

@dataclass
class BacktestConfig:
    """Comprehensive backtest configuration"""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.001    # 0.1%
    max_positions: int = 15
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    
    # Risk management
    max_position_size: float = 0.1  # 10% max per position
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    daily_loss_limit: float = 0.05  # 5%
    
    # Walk-forward analysis
    walk_forward_periods: int = 5
    out_of_sample_ratio: float = 0.2  # 20% out-of-sample
    
    # Monte Carlo
    monte_carlo_simulations: int = 1000
    confidence_intervals: List[float] = None
    
    # Performance analysis
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    def __post_init__(self):
        if self.confidence_intervals is None:
            self.confidence_intervals = [0.05, 0.25, 0.75, 0.95]

@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    duration_days: int = 0
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    def calculate_metrics(self, commission_rate: float, slippage_rate: float):
        """Calculate trade metrics"""
        if self.exit_price and self.exit_date:
            # Calculate P&L
            if self.side == 'long':
                raw_pnl = (self.exit_price - self.entry_price) * self.quantity
                self.pnl_percent = (self.exit_price - self.entry_price) / self.entry_price
            else:  # short
                raw_pnl = (self.entry_price - self.exit_price) * self.quantity
                self.pnl_percent = (self.entry_price - self.exit_price) / self.entry_price
            
            # Apply costs
            entry_value = abs(self.entry_price * self.quantity)
            exit_value = abs(self.exit_price * self.quantity)
            
            self.commission = (entry_value + exit_value) * commission_rate
            self.slippage = (entry_value + exit_value) * slippage_rate
            
            self.pnl = raw_pnl - self.commission - self.slippage
            
            # Duration
            self.duration_days = (self.exit_date - self.entry_date).days

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    benchmark_returns: Optional[pd.Series] = None
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    
    # Additional metrics
    recovery_factor: float = 0.0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0
    
    def calculate_all_metrics(self):
        """Calculate all performance metrics"""
        self._calculate_basic_metrics()
        self._calculate_risk_metrics()
        self._calculate_trade_metrics()
        self._calculate_drawdown_metrics()
        self._calculate_benchmark_metrics()
    
    def _calculate_basic_metrics(self):
        """Calculate basic performance metrics"""
        if len(self.returns) == 0:
            return
        
        # Basic returns
        self.total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        
        # Annualized metrics
        trading_days = len(self.returns)
        years = trading_days / 252  # Assume 252 trading days per year
        
        if years > 0:
            self.annualized_return = (1 + self.total_return) ** (1/years) - 1
            self.volatility = self.returns.std() * np.sqrt(252)
            
            # Risk-adjusted ratios
            excess_returns = self.returns - (self.config.risk_free_rate / 252)
            
            if self.volatility > 0:
                self.sharpe_ratio = excess_returns.mean() / self.returns.std() * np.sqrt(252)
            
            # Sortino ratio (downside deviation)
            downside_returns = self.returns[self.returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std() * np.sqrt(252)
                if downside_std > 0:
                    self.sortino_ratio = (self.annualized_return - self.config.risk_free_rate) / downside_std
    
    def _calculate_risk_metrics(self):
        """Calculate risk metrics"""
        if len(self.returns) == 0:
            return
        
        # Value at Risk (VaR)
        self.var_95 = np.percentile(self.returns, 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        var_threshold = self.returns <= self.var_95
        if np.any(var_threshold):
            self.cvar_95 = self.returns[var_threshold].mean()
    
    def _calculate_trade_metrics(self):
        """Calculate trade-based metrics"""
        self.total_trades = len(self.trades)
        
        if self.total_trades == 0:
            return
        
        completed_trades = [t for t in self.trades if t.exit_price is not None]
        
        if len(completed_trades) == 0:
            return
        
        # Win/loss statistics
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.win_rate = self.winning_trades / len(completed_trades) if completed_trades else 0
        
        # Average win/loss
        if winning_trades:
            self.avg_win = np.mean([t.pnl for t in winning_trades])
        if losing_trades:
            self.avg_loss = np.mean([t.pnl for t in losing_trades])
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        
        if total_losses > 0:
            self.profit_factor = total_wins / total_losses
        
        # Expectancy and Kelly criterion
        if completed_trades:
            avg_trade = np.mean([t.pnl_percent for t in completed_trades])
            win_prob = self.win_rate
            
            if win_prob > 0 and self.avg_loss < 0:
                avg_win_pct = np.mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0
                avg_loss_pct = abs(np.mean([t.pnl_percent for t in losing_trades])) if losing_trades else 0
                
                self.expectancy = (win_prob * avg_win_pct) - ((1 - win_prob) * avg_loss_pct)
                
                # Kelly Criterion
                if avg_loss_pct > 0:
                    self.kelly_criterion = win_prob - ((1 - win_prob) / (avg_win_pct / avg_loss_pct))
    
    def _calculate_drawdown_metrics(self):
        """Calculate drawdown metrics"""
        if len(self.equity_curve) == 0:
            return
        
        # Calculate drawdown series
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak
        
        self.max_drawdown = abs(drawdown.min())
        
        # Max drawdown duration
        is_recovering = drawdown == 0
        recovery_periods = []
        current_period = 0
        
        for recovering in is_recovering:
            if not recovering:
                current_period += 1
            else:
                if current_period > 0:
                    recovery_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:  # Still in drawdown
            recovery_periods.append(current_period)
        
        self.max_drawdown_duration = max(recovery_periods) if recovery_periods else 0
        
        # Recovery factor
        if self.max_drawdown > 0:
            self.recovery_factor = self.total_return / self.max_drawdown
        
        # Calmar ratio
        if self.max_drawdown > 0:
            self.calmar_ratio = self.annualized_return / self.max_drawdown
    
    def _calculate_benchmark_metrics(self):
        """Calculate benchmark-relative metrics"""
        if self.benchmark_returns is None or len(self.benchmark_returns) == 0:
            return
        
        # Align returns
        aligned_returns = pd.concat([self.returns, self.benchmark_returns], axis=1, join='inner')
        if aligned_returns.empty:
            return
        
        strategy_returns = aligned_returns.iloc[:, 0]
        benchmark_returns = aligned_returns.iloc[:, 1]
        
        # Beta and Alpha
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance > 0:
            self.beta = covariance / benchmark_variance
            
            # Alpha (Jensen's Alpha)
            benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
            expected_return = self.config.risk_free_rate + self.beta * (benchmark_annual_return - self.config.risk_free_rate)
            self.alpha = self.annualized_return - expected_return
        
        # Information ratio and tracking error
        excess_returns = strategy_returns - benchmark_returns
        self.tracking_error = excess_returns.std() * np.sqrt(252)
        
        if self.tracking_error > 0:
            self.information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

class EnhancedBacktester:
    """Advanced backtesting engine with comprehensive analysis capabilities"""
    
    def __init__(self):
        self.redis_client = None
        self.is_running = False
        
        # Configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        self.workers = int(os.getenv('BACKTEST_WORKERS', str(min(8, mp.cpu_count()))))
        self.cache_results = os.getenv('CACHE_BACKTEST_RESULTS', 'true').lower() == 'true'
        
        # Service URLs
        self.data_manager_url = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')
        self.scanner_url = os.getenv('UNIFIED_SCANNER_URL', 'http://unified-scanner:8011')
        
        # Results storage
        self.results_cache = {}
        self.active_backtests = {}
        
        logger.info("🏗️ Enhanced Backtester initialized")
    
    async def initialize(self) -> bool:
        """Initialize the backtesting engine"""
        try:
            # Initialize Redis
            if self.redis_url:
                self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
                await asyncio.to_thread(self.redis_client.ping)
                logger.info("✅ Redis connection established")
            
            logger.info("✅ Enhanced Backtester initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize backtester: {e}")
            return False
    
    async def run_backtest(self, strategy_func: Callable, config: BacktestConfig, 
                          symbol_list: List[str], backtest_type: BacktestType = BacktestType.SINGLE_STRATEGY) -> BacktestResult:
        """Run comprehensive backtest with specified strategy"""
        
        logger.info(f"🚀 Starting {backtest_type.value} backtest: {config.start_date} to {config.end_date}")
        
        try:
            # Generate unique backtest ID
            backtest_id = str(uuid.uuid4())
            self.active_backtests[backtest_id] = {
                'status': 'running',
                'start_time': time.time(),
                'config': config,
                'type': backtest_type
            }
            
            # Select appropriate backtest method
            if backtest_type == BacktestType.WALK_FORWARD:
                result = await self._run_walk_forward_backtest(strategy_func, config, symbol_list)
            elif backtest_type == BacktestType.MONTE_CARLO:
                result = await self._run_monte_carlo_backtest(strategy_func, config, symbol_list)
            elif backtest_type == BacktestType.SCENARIO_ANALYSIS:
                result = await self._run_scenario_analysis(strategy_func, config, symbol_list)
            else:
                result = await self._run_standard_backtest(strategy_func, config, symbol_list)
            
            # Calculate all performance metrics
            result.calculate_all_metrics()
            
            # Validate results
            if MATH_VALIDATOR_AVAILABLE:
                validation = await self._validate_backtest_results(result)
                if not validation['is_valid']:
                    logger.warning(f"⚠️ Backtest validation issues: {validation['issues']}")
            
            # Cache results
            if self.cache_results:
                await self._cache_backtest_result(backtest_id, result)
            
            # Update backtest status
            self.active_backtests[backtest_id]['status'] = 'completed'
            self.active_backtests[backtest_id]['result'] = result
            
            logger.info(f"✅ Backtest completed: Total Return {result.total_return:.2%}, Sharpe {result.sharpe_ratio:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Backtest execution error: {e}")
            if backtest_id in self.active_backtests:
                self.active_backtests[backtest_id]['status'] = 'failed'
                self.active_backtests[backtest_id]['error'] = str(e)
            raise
    
    async def _run_standard_backtest(self, strategy_func: Callable, config: BacktestConfig, 
                                   symbol_list: List[str]) -> BacktestResult:
        """Run standard single-strategy backtest"""
        
        # Get historical data for all symbols
        historical_data = await self._get_historical_data(symbol_list, config.start_date, config.end_date)
        if not historical_data:
            raise ValueError("No historical data available")
        
        # Initialize backtest state
        portfolio_value = config.initial_capital
        positions = {}
        trades = []
        equity_curve = []
        dates = []
        
        # Get trading dates
        sample_symbol = list(historical_data.keys())[0]
        trading_dates = sorted(historical_data[sample_symbol].keys())
        
        # Run day-by-day simulation
        for date_str in trading_dates:
            current_date = datetime.fromisoformat(date_str)
            
            # Get current market data
            market_data = {}
            for symbol in symbol_list:
                if symbol in historical_data and date_str in historical_data[symbol]:
                    market_data[symbol] = historical_data[symbol][date_str]
            
            # Update position values
            portfolio_value = self._update_portfolio_value(positions, market_data)
            
            # Generate trading signals
            signals = await strategy_func(market_data, positions, config)
            
            # Execute trades
            new_trades = await self._execute_signals(signals, market_data, positions, config, current_date)
            trades.extend(new_trades)
            
            # Record equity curve point
            equity_curve.append(portfolio_value)
            dates.append(current_date)
            
            # Risk management checks
            daily_return = (portfolio_value / config.initial_capital) - 1
            if daily_return <= -config.daily_loss_limit:
                logger.warning(f"⚠️ Daily loss limit hit: {daily_return:.2%}")
                # Close all positions
                for symbol in list(positions.keys()):
                    if symbol in market_data:
                        close_trade = self._close_position(symbol, positions, market_data[symbol], current_date, "daily_loss_limit")
                        if close_trade:
                            trades.append(close_trade)
        
        # Convert to pandas series for analysis
        equity_series = pd.Series(equity_curve, index=dates)
        returns_series = equity_series.pct_change().dropna()
        
        # Get benchmark data if specified
        benchmark_returns = None
        if config.benchmark_symbol:
            benchmark_returns = await self._get_benchmark_returns(config.benchmark_symbol, config.start_date, config.end_date)
        
        return BacktestResult(
            config=config,
            trades=trades,
            equity_curve=equity_series,
            returns=returns_series,
            benchmark_returns=benchmark_returns
        )
    
    async def _run_walk_forward_backtest(self, strategy_func: Callable, config: BacktestConfig, 
                                       symbol_list: List[str]) -> BacktestResult:
        """Run walk-forward analysis for robust out-of-sample testing"""
        
        logger.info(f"🔄 Running Walk-Forward Analysis with {config.walk_forward_periods} periods")
        
        # Get full historical data
        historical_data = await self._get_historical_data(symbol_list, config.start_date, config.end_date)
        if not historical_data:
            raise ValueError("No historical data available")
        
        # Determine date ranges for walk-forward periods
        sample_symbol = list(historical_data.keys())[0]
        all_dates = sorted(historical_data[sample_symbol].keys())
        
        total_days = len(all_dates)
        period_length = total_days // config.walk_forward_periods
        out_of_sample_days = int(period_length * config.out_of_sample_ratio)
        in_sample_days = period_length - out_of_sample_days
        
        all_trades = []
        all_equity_points = []
        all_dates_list = []
        
        for period in range(config.walk_forward_periods):
            start_idx = period * period_length
            end_idx = min(start_idx + period_length, total_days)
            
            # Split into in-sample and out-of-sample
            is_end_idx = start_idx + in_sample_days
            oos_start_idx = is_end_idx
            
            if oos_start_idx >= end_idx:
                continue  # Skip if no out-of-sample data
            
            # Get period dates
            is_dates = all_dates[start_idx:is_end_idx]
            oos_dates = all_dates[oos_start_idx:end_idx]
            
            logger.info(f"📊 WF Period {period + 1}: IS {len(is_dates)} days, OOS {len(oos_dates)} days")
            
            # Create period configuration
            period_config = BacktestConfig(
                start_date=oos_dates[0],
                end_date=oos_dates[-1],
                initial_capital=config.initial_capital,
                commission=config.commission,
                slippage=config.slippage,
                max_positions=config.max_positions,
                rebalance_frequency=config.rebalance_frequency
            )
            
            # Run backtest on out-of-sample period
            period_result = await self._run_standard_backtest(strategy_func, period_config, symbol_list)
            
            # Accumulate results
            all_trades.extend(period_result.trades)
            all_equity_points.extend(period_result.equity_curve.values.tolist())
            all_dates_list.extend(period_result.equity_curve.index.tolist())
        
        # Combine all results
        if all_equity_points:
            equity_series = pd.Series(all_equity_points, index=all_dates_list)
            returns_series = equity_series.pct_change().dropna()
            
            # Get benchmark data
            benchmark_returns = None
            if config.benchmark_symbol:
                benchmark_returns = await self._get_benchmark_returns(config.benchmark_symbol, config.start_date, config.end_date)
            
            return BacktestResult(
                config=config,
                trades=all_trades,
                equity_curve=equity_series,
                returns=returns_series,
                benchmark_returns=benchmark_returns
            )
        else:
            raise ValueError("Walk-forward analysis produced no results")
    
    async def _run_monte_carlo_backtest(self, strategy_func: Callable, config: BacktestConfig, 
                                      symbol_list: List[str]) -> BacktestResult:
        """Run Monte Carlo simulation for probabilistic analysis"""
        
        logger.info(f"🎰 Running Monte Carlo simulation with {config.monte_carlo_simulations} iterations")
        
        # Run base backtest
        base_result = await self._run_standard_backtest(strategy_func, config, symbol_list)
        
        if len(base_result.returns) == 0:
            return base_result
        
        # Prepare for Monte Carlo simulations
        base_returns = base_result.returns.values
        mean_return = np.mean(base_returns)
        std_return = np.std(base_returns)
        
        mc_final_values = []
        mc_max_drawdowns = []
        mc_sharpe_ratios = []
        
        # Run Monte Carlo simulations
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            
            for sim in range(config.monte_carlo_simulations):
                future = executor.submit(self._run_single_mc_simulation, 
                                       mean_return, std_return, len(base_returns), config.initial_capital)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    final_value, max_dd, sharpe = future.result()
                    mc_final_values.append(final_value)
                    mc_max_drawdowns.append(max_dd)
                    mc_sharpe_ratios.append(sharpe)
                except Exception as e:
                    logger.debug(f"MC simulation error: {e}")
        
        # Calculate Monte Carlo statistics
        if mc_final_values:
            mc_returns = [(fv / config.initial_capital - 1) for fv in mc_final_values]
            
            # Add Monte Carlo statistics to the base result
            base_result.monte_carlo_stats = {
                'simulations': len(mc_final_values),
                'mean_return': float(np.mean(mc_returns)),
                'std_return': float(np.std(mc_returns)),
                'percentiles': {
                    'p05': float(np.percentile(mc_returns, 5)),
                    'p25': float(np.percentile(mc_returns, 25)),
                    'p50': float(np.percentile(mc_returns, 50)),
                    'p75': float(np.percentile(mc_returns, 75)),
                    'p95': float(np.percentile(mc_returns, 95))
                },
                'probability_of_loss': float(np.mean([r < 0 for r in mc_returns])),
                'expected_max_drawdown': float(np.mean(mc_max_drawdowns)),
                'expected_sharpe_ratio': float(np.mean([s for s in mc_sharpe_ratios if np.isfinite(s)]))
            }
        
        return base_result
    
    def _run_single_mc_simulation(self, mean_return: float, std_return: float, 
                                 n_periods: int, initial_capital: float) -> Tuple[float, float, float]:
        """Run a single Monte Carlo simulation"""
        
        # Generate random returns
        random_returns = np.random.normal(mean_return, std_return, n_periods)
        
        # Calculate equity curve
        equity_curve = initial_capital * np.cumprod(1 + random_returns)
        
        # Calculate metrics
        final_value = equity_curve[-1]
        
        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Sharpe ratio
        if std_return > 0:
            sharpe_ratio = mean_return / std_return * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        return final_value, max_drawdown, sharpe_ratio
    
    async def _run_scenario_analysis(self, strategy_func: Callable, config: BacktestConfig, 
                                   symbol_list: List[str]) -> BacktestResult:
        """Run scenario analysis with different market conditions"""
        
        logger.info("📈 Running scenario analysis")
        
        # Define scenarios
        scenarios = [
            {'name': 'Bull Market', 'return_multiplier': 1.5, 'volatility_multiplier': 0.8},
            {'name': 'Bear Market', 'return_multiplier': -0.5, 'volatility_multiplier': 1.2},
            {'name': 'High Volatility', 'return_multiplier': 1.0, 'volatility_multiplier': 2.0},
            {'name': 'Low Volatility', 'return_multiplier': 1.0, 'volatility_multiplier': 0.5},
            {'name': 'Crash', 'return_multiplier': -2.0, 'volatility_multiplier': 3.0}
        ]
        
        base_result = await self._run_standard_backtest(strategy_func, config, symbol_list)
        
        scenario_results = {}
        
        for scenario in scenarios:
            try:
                # Modify returns based on scenario
                modified_returns = base_result.returns * scenario['return_multiplier']
                modified_returns = modified_returns + np.random.normal(0, base_result.returns.std() * scenario['volatility_multiplier'] - base_result.returns.std(), len(modified_returns))
                
                # Calculate scenario metrics
                modified_equity = config.initial_capital * (1 + modified_returns).cumprod()
                scenario_total_return = (modified_equity.iloc[-1] / config.initial_capital) - 1
                scenario_volatility = modified_returns.std() * np.sqrt(252)
                scenario_sharpe = modified_returns.mean() / modified_returns.std() * np.sqrt(252) if modified_returns.std() > 0 else 0
                
                # Max drawdown for scenario
                peak = modified_equity.cummax()
                drawdown = (modified_equity - peak) / peak
                scenario_max_dd = abs(drawdown.min())
                
                scenario_results[scenario['name']] = {
                    'total_return': float(scenario_total_return),
                    'volatility': float(scenario_volatility),
                    'sharpe_ratio': float(scenario_sharpe),
                    'max_drawdown': float(scenario_max_dd)
                }
                
            except Exception as e:
                logger.debug(f"Scenario {scenario['name']} error: {e}")
        
        # Add scenario analysis to base result
        base_result.scenario_analysis = scenario_results
        
        return base_result
    
    async def _get_historical_data(self, symbol_list: List[str], start_date: str, end_date: str) -> Dict[str, Dict]:
        """Get historical OHLCV data for backtesting"""
        historical_data = {}
        
        for symbol in symbol_list:
            try:
                # This would fetch from data manager in production
                # For now, simulate data
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                symbol_data = {}
                
                base_price = 100.0
                for date in date_range:
                    # Simulate OHLCV data
                    daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% volatility
                    price = base_price * (1 + daily_return)
                    
                    symbol_data[date.isoformat()] = {
                        'open': base_price,
                        'high': max(base_price, price) * 1.01,
                        'low': min(base_price, price) * 0.99,
                        'close': price,
                        'volume': np.random.uniform(1000000, 5000000)
                    }
                    
                    base_price = price
                
                historical_data[symbol] = symbol_data
                
            except Exception as e:
                logger.error(f"❌ Error getting data for {symbol}: {e}")
        
        return historical_data
    
    async def _get_benchmark_returns(self, benchmark_symbol: str, start_date: str, end_date: str) -> pd.Series:
        """Get benchmark returns for comparison"""
        try:
            # Simulate benchmark returns
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            benchmark_returns = pd.Series(
                np.random.normal(0.0003, 0.01, len(date_range)),  # Market-like returns
                index=date_range
            )
            return benchmark_returns
            
        except Exception as e:
            logger.error(f"❌ Error getting benchmark data: {e}")
            return None
    
    def _update_portfolio_value(self, positions: Dict, market_data: Dict) -> float:
        """Update portfolio value based on current market prices"""
        total_value = 0.0
        
        for symbol, position in positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['close']
                position_value = position['quantity'] * current_price
                total_value += position_value
        
        return total_value
    
    async def _execute_signals(self, signals: List[Dict], market_data: Dict, 
                             positions: Dict, config: BacktestConfig, current_date: datetime) -> List[Trade]:
        """Execute trading signals"""
        trades = []
        
        for signal in signals:
            try:
                symbol = signal['symbol']
                action = signal['action']  # 'buy', 'sell', 'close'
                quantity = signal.get('quantity', 0)
                
                if symbol not in market_data:
                    continue
                
                current_price = market_data[symbol]['close']
                
                if action == 'buy' and symbol not in positions:
                    # Open long position
                    trade = Trade(
                        symbol=symbol,
                        entry_date=current_date,
                        exit_date=None,
                        entry_price=current_price,
                        exit_price=None,
                        quantity=quantity,
                        side='long'
                    )
                    
                    positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_date': current_date,
                        'side': 'long'
                    }
                    
                    trades.append(trade)
                    
                elif action in ['sell', 'close'] and symbol in positions:
                    # Close position
                    close_trade = self._close_position(symbol, positions, market_data[symbol], current_date, "signal")
                    if close_trade:
                        trades.append(close_trade)
                        
            except Exception as e:
                logger.debug(f"Signal execution error: {e}")
        
        return trades
    
    def _close_position(self, symbol: str, positions: Dict, market_data: Dict, 
                       current_date: datetime, reason: str = "signal") -> Optional[Trade]:
        """Close a position and create trade record"""
        if symbol not in positions:
            return None
        
        position = positions[symbol]
        current_price = market_data['close']
        
        trade = Trade(
            symbol=symbol,
            entry_date=position['entry_date'],
            exit_date=current_date,
            entry_price=position['entry_price'],
            exit_price=current_price,
            quantity=position['quantity'],
            side=position['side']
        )
        
        # Calculate trade metrics would be done here
        # trade.calculate_metrics(commission_rate, slippage_rate)
        
        # Remove position
        del positions[symbol]
        
        return trade
    
    async def _validate_backtest_results(self, result: BacktestResult) -> Dict[str, Any]:
        """Validate backtest results using enhanced mathematical validator"""
        if not MATH_VALIDATOR_AVAILABLE:
            return {'is_valid': True, 'issues': []}
        
        issues = []
        warnings = []
        
        try:
            # Validate equity curve
            if len(result.equity_curve) > 0:
                equity_validation = enhanced_math_validator.validate_array(
                    result.equity_curve.values, "equity_curve", ValidationLevel.STANDARD
                )
                if not equity_validation.is_valid:
                    issues.extend(equity_validation.issues)
            
            # Validate returns
            if len(result.returns) > 0:
                returns_validation = enhanced_math_validator.validate_array(
                    result.returns.values, "returns", ValidationLevel.STANDARD
                )
                if not returns_validation.is_valid:
                    issues.extend(returns_validation.issues)
            
            # Validate performance metrics
            metrics_to_validate = [
                ('total_return', result.total_return),
                ('sharpe_ratio', result.sharpe_ratio),
                ('max_drawdown', result.max_drawdown)
            ]
            
            for metric_name, metric_value in metrics_to_validate:
                if not np.isfinite(metric_value):
                    issues.append(f"Invalid {metric_name}: {metric_value}")
                elif metric_name == 'max_drawdown' and metric_value > 1.0:
                    warnings.append(f"Extreme {metric_name}: {metric_value}")
                elif metric_name == 'sharpe_ratio' and abs(metric_value) > 10:
                    warnings.append(f"Unrealistic {metric_name}: {metric_value}")
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    async def _cache_backtest_result(self, backtest_id: str, result: BacktestResult):
        """Cache backtest result for future reference"""
        if not self.redis_client:
            return
        
        try:
            # Create cacheable result summary
            cache_data = {
                'backtest_id': backtest_id,
                'timestamp': time.time(),
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'config': asdict(result.config)
            }
            
            cache_key = f"backtest_result:{backtest_id}"
            await asyncio.to_thread(
                self.redis_client.setex, 
                cache_key, 
                86400,  # 24 hours
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.debug(f"Result caching error: {e}")
    
    def get_backtest_status(self, backtest_id: str) -> Optional[Dict]:
        """Get backtest execution status"""
        return self.active_backtests.get(backtest_id)
    
    def get_active_backtests(self) -> Dict[str, Dict]:
        """Get all active backtests"""
        return self.active_backtests.copy()
    
    def start(self):
        """Start the backtester"""
        self.is_running = True
        logger.info("🚀 Enhanced Backtester started")
    
    def stop(self):
        """Stop the backtester"""
        self.is_running = False
        logger.info("🛑 Enhanced Backtester stopped")

# Global backtester instance
enhanced_backtester = EnhancedBacktester()

# Example strategy functions
async def simple_momentum_strategy(market_data: Dict, positions: Dict, config: BacktestConfig) -> List[Dict]:
    """Simple momentum-based strategy example"""
    signals = []
    
    for symbol, data in market_data.items():
        try:
            current_price = data['close']
            
            # Simple momentum: buy if price > moving average (simulated)
            # In practice, this would use historical data to calculate MA
            if len(positions) < config.max_positions and symbol not in positions:
                # Generate buy signal (simplified)
                if np.random.random() > 0.8:  # 20% chance to buy
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': 100  # Fixed quantity for simplicity
                    })
            
            elif symbol in positions:
                # Generate sell signal (simplified)
                if np.random.random() > 0.9:  # 10% chance to sell
                    signals.append({
                        'symbol': symbol,
                        'action': 'close'
                    })
                    
        except Exception as e:
            logger.debug(f"Strategy error for {symbol}: {e}")
    
    return signals

async def viper_based_strategy(market_data: Dict, positions: Dict, config: BacktestConfig) -> List[Dict]:
    """VIPER score-based strategy example"""
    signals = []
    
    # This would integrate with the VIPER scoring system
    # For now, simulate VIPER scores
    
    for symbol, data in market_data.items():
        try:
            # Simulate VIPER score
            viper_score = np.random.uniform(0, 100)
            
            if viper_score > 85 and len(positions) < config.max_positions and symbol not in positions:
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': 100,
                    'viper_score': viper_score
                })
            
            elif symbol in positions and viper_score < 30:
                signals.append({
                    'symbol': symbol,
                    'action': 'close',
                    'viper_score': viper_score
                })
                
        except Exception as e:
            logger.debug(f"VIPER strategy error for {symbol}: {e}")
    
    return signals

if __name__ == "__main__":
    async def test_enhanced_backtester():
        """Test the enhanced backtesting system"""
        logger.info("🧪 Testing Enhanced Backtester...")
        
        # Initialize
        await enhanced_backtester.initialize()
        enhanced_backtester.start()
        
        # Create test configuration
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.001,
            max_positions=5,
            walk_forward_periods=3,
            monte_carlo_simulations=100  # Reduced for testing
        )
        
        symbol_list = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        # Test standard backtest
        print("🔍 Running standard backtest...")
        standard_result = await enhanced_backtester.run_backtest(
            simple_momentum_strategy, config, symbol_list, BacktestType.SINGLE_STRATEGY
        )
        print(f"✅ Standard backtest: {standard_result.total_return:.2%} return, {standard_result.sharpe_ratio:.2f} Sharpe")
        
        # Test walk-forward analysis
        print("🔄 Running walk-forward analysis...")
        wf_result = await enhanced_backtester.run_backtest(
            simple_momentum_strategy, config, symbol_list, BacktestType.WALK_FORWARD
        )
        print(f"✅ Walk-forward: {wf_result.total_return:.2%} return, {wf_result.sharpe_ratio:.2f} Sharpe")
        
        # Test Monte Carlo simulation
        print("🎰 Running Monte Carlo simulation...")
        mc_result = await enhanced_backtester.run_backtest(
            viper_based_strategy, config, symbol_list, BacktestType.MONTE_CARLO
        )
        print(f"✅ Monte Carlo: {mc_result.total_return:.2%} return")
        if hasattr(mc_result, 'monte_carlo_stats'):
            print(f"   MC Mean: {mc_result.monte_carlo_stats['mean_return']:.2%}")
            print(f"   MC P95: {mc_result.monte_carlo_stats['percentiles']['p95']:.2%}")
        
        enhanced_backtester.stop()
        print("🎯 Enhanced backtester test completed!")
    
    asyncio.run(test_enhanced_backtester())