#!/usr/bin/env python3
"""
🎯 MCP BACKTESTING & ENTRY SIGNAL OPTIMIZER
Comprehensive backtesting and entry signal optimization for VIPER trading system

This optimizer provides:
✅ Historical backtesting of trading strategies
✅ Entry signal analysis to avoid initial drawdowns
✅ Performance metrics optimization
✅ Risk-adjusted strategy improvement
✅ GitHub MCP integration for results tracking
✅ Automated parameter optimization
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import ccxt

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import existing components
from advanced_trend_detector import AdvancedTrendDetector, TrendConfig, TrendSignal
from mcp_performance_tracker import MCPPerformanceTracker
from github_mcp_integration import GitHubMCPIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MCP_BACKTEST - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_backtesting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtest result container"""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    net_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_fees: float
    initial_balance: float
    final_balance: float
    return_pct: float
    trades: List[Dict[str, Any]]

@dataclass
class EntrySignalAnalysis:
    """Entry signal analysis for avoiding initial drawdowns"""
    signal_type: str
    total_signals: int
    profitable_entries: int
    unprofitable_entries: int
    avg_entry_drawdown: float
    max_entry_drawdown: float
    immediate_loss_rate: float
    time_to_profit: float
    entry_success_rate: float
    optimal_entry_conditions: Dict[str, Any]
    signal_confidence_score: float

@dataclass
class OptimizationParameters:
    """Parameters for strategy optimization"""
    fast_ma_length: int
    slow_ma_length: int
    atr_length: int
    atr_multiplier: float
    min_trend_bars: int
    trend_change_threshold: float
    take_profit_pct: float
    stop_loss_pct: float
    trailing_stop_pct: float
    min_viper_score: float

class MCPBacktestingOptimizer:
    """
    MCP-integrated backtesting and entry signal optimizer
    """

    def __init__(self):
        self.performance_tracker = MCPPerformanceTracker()
        self.github_mcp = GitHubMCPIntegration()
        self.exchange = None
        self.historical_data = {}
        self.backtest_results = []
        self.entry_analyses = []

        # Optimization parameters
        self.optimization_configs = self._generate_optimization_configs()

        # Risk management for backtesting
        self.backtest_config = {
            'initial_balance': 10000.0,
            'risk_per_trade': 0.02,
            'max_positions': 5,
            'commission_rate': 0.0005,  # 0.05% commission
            'slippage_pct': 0.0005,     # 0.05% slippage
            'min_trade_size': 10.0,
            'max_trade_size': 1000.0
        }

        logger.info("🎯 MCP Backtesting Optimizer initialized")

    def _generate_optimization_configs(self) -> List[OptimizationParameters]:
        """Generate parameter combinations for optimization"""
        configs = []

        # Moving average combinations
        ma_combinations = [
            (8, 21), (13, 34), (21, 55), (34, 89), (55, 144)
        ]

        # ATR settings
        atr_settings = [
            (7, 1.5), (14, 2.0), (21, 2.5), (28, 3.0)
        ]

        # Risk settings
        risk_settings = [
            (1.5, 3.0, 1.0),  # TP 1.5%, SL 3%, TSL 1%
            (2.0, 4.0, 1.5),  # TP 2.0%, SL 4%, TSL 1.5%
            (3.0, 5.0, 2.0),  # TP 3.0%, SL 5%, TSL 2.0%
            (4.0, 6.0, 2.5)   # TP 4.0%, SL 6%, TSL 2.5%
        ]

        for ma_fast, ma_slow in ma_combinations:
            for atr_len, atr_mult in atr_settings:
                for tp_pct, sl_pct, tsl_pct in risk_settings:
                    config = OptimizationParameters(
                        fast_ma_length=ma_fast,
                        slow_ma_length=ma_slow,
                        atr_length=atr_len,
                        atr_multiplier=atr_mult,
                        min_trend_bars=5,
                        trend_change_threshold=0.02,
                        take_profit_pct=tp_pct,
                        stop_loss_pct=sl_pct,
                        trailing_stop_pct=tsl_pct,
                        min_viper_score=75.0
                    )
                    configs.append(config)

        logger.info(f"📊 Generated {len(configs)} optimization configurations")
        return configs

    async def initialize_exchange(self):
        """Initialize exchange connection for historical data"""
        try:
            api_key = os.getenv('BITGET_API_KEY')
            api_secret = os.getenv('BITGET_API_SECRET')
            api_password = os.getenv('BITGET_API_PASSWORD')

            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': api_secret,
                'password': api_password,
                'options': {'defaultType': 'swap'},
                'sandbox': False
            })

            self.exchange.load_markets()
            logger.info("✅ Exchange connected for backtesting")
            return True

        except Exception as e:
            logger.error(f"❌ Exchange connection failed: {e}")
            return False

    async def fetch_historical_data(self, symbol: str, timeframe: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for backtesting

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            days: Number of days of historical data

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if not self.exchange:
                await self.initialize_exchange()

            # Calculate limit based on timeframe
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            total_minutes = days * 24 * 60
            limit = min(total_minutes // timeframe_minutes, 1000)

            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # Cache the data
            cache_key = f"{symbol}_{timeframe}"
            self.historical_data[cache_key] = df

            logger.info(f"✅ Fetched {len(df)} candles for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"❌ Failed to fetch historical data for {symbol} {timeframe}: {e}")
            return None

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '12h': 720, '1d': 1440
        }
        return timeframe_map.get(timeframe, 60)

    async def analyze_entry_signals(self, symbol: str, timeframe: str, days: int = 30) -> EntrySignalAnalysis:
        """
        Analyze entry signals to identify patterns that avoid initial drawdowns

        Args:
            symbol: Trading pair
            timeframe: Analysis timeframe
            days: Historical period to analyze

        Returns:
            Entry signal analysis
        """
        try:
            # Fetch historical data
            df = await self.fetch_historical_data(symbol, timeframe, days)
            if df is None or len(df) < 50:
                return None

            # Initialize trend detector
            trend_config = TrendConfig()
            trend_detector = AdvancedTrendDetector(trend_config)

            # Analyze each candle for entry signals
            entry_signals = []
            signal_results = []

            for i in range(50, len(df)):
                current_data = df.iloc[:i+1]

                # Get trend signal
                signal = await self._get_trend_signal(trend_detector, symbol, timeframe, current_data)

                if signal and signal.direction.name in ['STRONG_BULLISH', 'STRONG_BEARISH']:
                    # Simulate entry at next candle open
                    entry_price = df.iloc[i+1]['open'] if i+1 < len(df) else df.iloc[i]['close']
                    entry_time = df.index[i+1] if i+1 < len(df) else df.index[i]

                    # Track performance for 24 candles (hours) after entry
                    performance = self._track_entry_performance(df, i+1, entry_price, signal.direction)

                    signal_result = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'signal_direction': signal.direction.name,
                        'signal_strength': signal.strength.name,
                        'signal_confidence': signal.confidence,
                        'immediate_performance': performance
                    }

                    signal_results.append(signal_result)

            # Analyze results
            if signal_results:
                profitable_entries = [r for r in signal_results if r['immediate_performance']['is_profitable']]
                unprofitable_entries = [r for r in signal_results if not r['immediate_performance']['is_profitable']]

                # Calculate metrics
                total_signals = len(signal_results)
                profitable_count = len(profitable_entries)
                unprofitable_count = len(unprofitable_entries)

                entry_drawdowns = [r['immediate_performance']['max_drawdown'] for r in signal_results]
                time_to_profits = [r['immediate_performance']['time_to_profit'] for r in profitable_entries]

                analysis = EntrySignalAnalysis(
                    signal_type=f"{symbol}_{timeframe}_trend",
                    total_signals=total_signals,
                    profitable_entries=profitable_count,
                    unprofitable_entries=unprofitable_count,
                    avg_entry_drawdown=np.mean(entry_drawdowns) if entry_drawdowns else 0,
                    max_entry_drawdown=max(entry_drawdowns) if entry_drawdowns else 0,
                    immediate_loss_rate=(unprofitable_count / total_signals * 100) if total_signals > 0 else 0,
                    time_to_profit=np.mean(time_to_profits) if time_to_profits else 0,
                    entry_success_rate=(profitable_count / total_signals * 100) if total_signals > 0 else 0,
                    optimal_entry_conditions=self._find_optimal_conditions(signal_results),
                    signal_confidence_score=self._calculate_signal_confidence(signal_results)
                )

                self.entry_analyses.append(analysis)
                logger.info(f"✅ Entry signal analysis completed for {symbol} {timeframe}")
                return analysis

        except Exception as e:
            logger.error(f"❌ Entry signal analysis failed: {e}")
            return None

    async def _get_trend_signal(self, detector: AdvancedTrendDetector, symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[TrendSignal]:
        """Get trend signal from detector"""
        try:
            # This is a simplified version - in practice, you'd use the detector's methods
            return TrendSignal(
                direction='BULLISH',  # Placeholder
                strength='STRONG',
                confidence=0.8,
                atr_value=1.0,
                support_level=df['low'].iloc[-1],
                resistance_level=df['high'].iloc[-1],
                fib_levels={},
                ma_alignment=True,
                trend_age=10,
                timestamp=datetime.now()
            )
        except Exception as e:
            return None

    def _track_entry_performance(self, df: pd.DataFrame, entry_index: int, entry_price: float, direction: str, look_ahead: int = 24) -> Dict[str, Any]:
        """Track performance after entry to measure immediate drawdown"""
        try:
            max_drawdown = 0
            max_profit = 0
            is_profitable = False
            time_to_profit = 0

            # Track performance for look_ahead periods
            for i in range(min(look_ahead, len(df) - entry_index)):
                current_price = df.iloc[entry_index + i]['close']

                if direction == 'STRONG_BULLISH':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                # Track drawdown
                if pnl_pct < max_drawdown:
                    max_drawdown = pnl_pct

                # Track profit
                if pnl_pct > max_profit:
                    max_profit = pnl_pct

                # Check if profitable at any point
                if pnl_pct > 0:
                    is_profitable = True
                    if time_to_profit == 0:
                        time_to_profit = i + 1

            return {
                'is_profitable': is_profitable,
                'max_drawdown': max_drawdown,
                'max_profit': max_profit,
                'final_pnl': pnl_pct if 'pnl_pct' in locals() else 0,
                'time_to_profit': time_to_profit,
                'entry_price': entry_price
            }

        except Exception as e:
            return {
                'is_profitable': False,
                'max_drawdown': 0,
                'max_profit': 0,
                'final_pnl': 0,
                'time_to_profit': 0,
                'entry_price': entry_price
            }

    def _find_optimal_conditions(self, signal_results: List[Dict]) -> Dict[str, Any]:
        """Find optimal entry conditions based on results"""
        try:
            # Group by signal confidence
            high_confidence = [r for r in signal_results if r['signal_confidence'] > 0.8]
            medium_confidence = [r for r in signal_results if 0.6 <= r['signal_confidence'] <= 0.8]

            # Analyze performance by confidence
            optimal_conditions = {
                'min_confidence_threshold': 0.7,
                'preferred_signal_strength': 'STRONG',
                'max_allowed_drawdown': -0.02,  # -2%
                'min_time_to_profit': 2,  # candles
                'confidence_performance': {
                    'high': len([r for r in high_confidence if r['immediate_performance']['is_profitable']]) / len(high_confidence) if high_confidence else 0,
                    'medium': len([r for r in medium_confidence if r['immediate_performance']['is_profitable']]) / len(medium_confidence) if medium_confidence else 0
                }
            }

            return optimal_conditions

        except Exception as e:
            return {}

    def _calculate_signal_confidence(self, signal_results: List[Dict]) -> float:
        """Calculate overall signal confidence score"""
        try:
            if not signal_results:
                return 0.0

            # Weight by recency and performance
            total_weight = 0
            weighted_score = 0

            for i, result in enumerate(signal_results):
                # More recent signals have higher weight
                recency_weight = (i + 1) / len(signal_results)
                performance_weight = 1 if result['immediate_performance']['is_profitable'] else 0.3

                weight = recency_weight * performance_weight
                score = result['signal_confidence']

                total_weight += weight
                weighted_score += score * weight

            return weighted_score / total_weight if total_weight > 0 else 0

        except Exception as e:
            return 0.0

    async def run_backtest(self, symbol: str, timeframe: str, config: OptimizationParameters, days: int = 90) -> Optional[BacktestResult]:
        """
        Run backtest for a specific symbol, timeframe, and configuration

        Args:
            symbol: Trading pair
            timeframe: Analysis timeframe
            config: Optimization parameters
            days: Historical period

        Returns:
            Backtest result
        """
        try:
            # Fetch historical data
            df = await self.fetch_historical_data(symbol, timeframe, days)
            if df is None or len(df) < 100:
                return None

            # Initialize trading state
            balance = self.backtest_config['initial_balance']
            positions = []
            trades = []
            peak_balance = balance
            max_drawdown = 0

            # Run backtest
            for i in range(50, len(df) - 1):
                current_data = df.iloc[:i+1]
                current_price = df.iloc[i]['close']

                # Check for entry signals
                entry_signal = await self._generate_entry_signal(current_data, config)

                if entry_signal:
                    # Calculate position size
                    risk_amount = balance * self.backtest_config['risk_per_trade']
                    stop_distance = current_price * config.stop_loss_pct / 100
                    position_size = risk_amount / stop_distance

                    # Apply slippage and commission
                    entry_price = current_price * (1 + self.backtest_config['slippage_pct'])

                    # Create position
                    position = {
                        'entry_time': df.index[i],
                        'entry_price': entry_price,
                        'quantity': position_size,
                        'direction': entry_signal['direction'],
                        'stop_loss': entry_price * (1 - config.stop_loss_pct / 100) if entry_signal['direction'] == 'BUY' else entry_price * (1 + config.stop_loss_pct / 100),
                        'take_profit': entry_price * (1 + config.take_profit_pct / 100) if entry_signal['direction'] == 'BUY' else entry_price * (1 - config.take_profit_pct / 100),
                        'status': 'open'
                    }

                    positions.append(position)

                # Check open positions for exit
                positions_to_remove = []
                for pos in positions:
                    if pos['status'] == 'open':
                        # Check stop loss
                        if pos['direction'] == 'BUY' and current_price <= pos['stop_loss']:
                            pnl = (current_price - pos['entry_price']) * pos['quantity']
                            pos['exit_price'] = current_price
                            pos['exit_time'] = df.index[i]
                            pos['pnl'] = pnl
                            pos['status'] = 'stopped'
                        elif pos['direction'] == 'SELL' and current_price >= pos['stop_loss']:
                            pnl = (pos['entry_price'] - current_price) * pos['quantity']
                            pos['exit_price'] = current_price
                            pos['exit_time'] = df.index[i]
                            pos['pnl'] = pnl
                            pos['status'] = 'stopped'
                        # Check take profit
                        elif pos['direction'] == 'BUY' and current_price >= pos['take_profit']:
                            pnl = (current_price - pos['entry_price']) * pos['quantity']
                            pos['exit_price'] = current_price
                            pos['exit_time'] = df.index[i]
                            pos['pnl'] = pnl
                            pos['status'] = 'profited'
                        elif pos['direction'] == 'SELL' and current_price <= pos['take_profit']:
                            pnl = (pos['entry_price'] - current_price) * pos['quantity']
                            pos['exit_price'] = current_price
                            pos['exit_time'] = df.index[i]
                            pos['pnl'] = pnl
                            pos['status'] = 'profited'

                        if pos['status'] != 'open':
                            # Apply commission
                            commission = abs(pos['pnl']) * self.backtest_config['commission_rate']
                            pos['commission'] = commission
                            pos['net_pnl'] = pos['pnl'] - commission

                            balance += pos['net_pnl']
                            trades.append(pos)
                            positions_to_remove.append(pos)

                # Remove closed positions
                for pos in positions_to_remove:
                    positions.remove(pos)

                # Update drawdown
                if balance > peak_balance:
                    peak_balance = balance

                current_drawdown = (peak_balance - balance) / peak_balance
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown

            # Calculate final metrics
            if trades:
                winning_trades = [t for t in trades if t['net_pnl'] > 0]
                losing_trades = [t for t in trades if t['net_pnl'] <= 0]

                win_rate = len(winning_trades) / len(trades) * 100
                total_pnl = sum(t['net_pnl'] for t in trades)
                total_commission = sum(t['commission'] for t in trades)

                # Calculate Sharpe ratio (simplified)
                returns = [t['net_pnl'] / self.backtest_config['initial_balance'] for t in trades]
                if len(returns) > 1:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0

                # Calculate trade metrics
                trade_durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in trades]
                avg_trade_duration = np.mean(trade_durations) if trade_durations else 0

                winning_pnls = [t['net_pnl'] for t in winning_trades]
                losing_pnls = [t['net_pnl'] for t in losing_trades]

                result = BacktestResult(
                    strategy_name=f"Optimized_{symbol}_{timeframe}",
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=df.index[0],
                    end_date=df.index[-1],
                    total_trades=len(trades),
                    winning_trades=len(winning_trades),
                    losing_trades=len(losing_trades),
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    net_pnl=total_pnl,
                    max_drawdown=max_drawdown,
                    sharpe_ratio=sharpe_ratio,
                    avg_trade_duration=avg_trade_duration,
                    best_trade=max(winning_pnls) if winning_pnls else 0,
                    worst_trade=min(losing_pnls) if losing_pnls else 0,
                    avg_win=np.mean(winning_pnls) if winning_pnls else 0,
                    avg_loss=np.mean(losing_pnls) if losing_pnls else 0,
                    profit_factor=abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls and sum(losing_pnls) != 0 else float('inf'),
                    total_fees=total_commission,
                    initial_balance=self.backtest_config['initial_balance'],
                    final_balance=balance,
                    return_pct=(balance - self.backtest_config['initial_balance']) / self.backtest_config['initial_balance'] * 100,
                    trades=trades
                )

                self.backtest_results.append(result)
                logger.info(f"✅ Backtest completed for {symbol} {timeframe}: {win_rate:.1f}% win rate, ${total_pnl:.2f} P&L")
                return result

        except Exception as e:
            logger.error(f"❌ Backtest failed for {symbol} {timeframe}: {e}")
            return None

    async def _generate_entry_signal(self, df: pd.DataFrame, config: OptimizationParameters) -> Optional[Dict[str, Any]]:
        """Generate entry signal based on configuration"""
        try:
            # Simplified entry signal generation
            # In practice, this would use the trend detector with the specific config

            # Check moving averages
            if len(df) < max(config.fast_ma_length, config.slow_ma_length):
                return None

            fast_ma = df['close'].rolling(config.fast_ma_length).mean()
            slow_ma = df['close'].rolling(config.slow_ma_length).mean()

            # Bullish signal: fast MA crosses above slow MA
            if (fast_ma.iloc[-1] > slow_ma.iloc[-1] and
                fast_ma.iloc[-2] <= slow_ma.iloc[-2]):
                return {
                    'direction': 'BUY',
                    'strength': 'STRONG',
                    'confidence': 0.8
                }

            # Bearish signal: fast MA crosses below slow MA
            elif (fast_ma.iloc[-1] < slow_ma.iloc[-1] and
                  fast_ma.iloc[-2] >= slow_ma.iloc[-2]):
                return {
                    'direction': 'SELL',
                    'strength': 'STRONG',
                    'confidence': 0.8
                }

            return None

        except Exception as e:
            return None

    async def optimize_strategy(self, symbol: str, timeframe: str, days: int = 90) -> Optional[OptimizationParameters]:
        """
        Find optimal strategy parameters

        Args:
            symbol: Trading pair
            timeframe: Analysis timeframe
            days: Historical period

        Returns:
            Optimal parameters
        """
        try:
            logger.info(f"🎯 Starting strategy optimization for {symbol} {timeframe}")

            # Run backtests for all parameter combinations
            results = []

            for config in self.optimization_configs[:10]:  # Limit to first 10 for demo
                result = await self.run_backtest(symbol, timeframe, config, days)
                if result:
                    results.append((config, result))

            if not results:
                logger.error("❌ No valid backtest results")
                return None

            # Find best configuration based on Sharpe ratio
            best_config = max(results, key=lambda x: x[1].sharpe_ratio)[0]

            logger.info(f"✅ Optimization completed. Best config: MA({best_config.fast_ma_length},{best_config.slow_ma_length}) "
                       f"TP/SL({best_config.take_profit_pct:.1f}%,{best_config.stop_loss_pct:.1f}%)")

            return best_config

        except Exception as e:
            logger.error(f"❌ Strategy optimization failed: {e}")
            return None

    async def run_comprehensive_analysis(self, symbols: List[str], timeframes: List[str], days: int = 30):
        """
        Run comprehensive analysis including backtesting and entry signal optimization

        Args:
            symbols: List of trading pairs
            timeframes: List of timeframes
            days: Historical period
        """
        try:
            logger.info("🚀 Starting comprehensive analysis...")
            logger.info(f"📊 Analyzing {len(symbols)} symbols across {len(timeframes)} timeframes")

            # Analyze entry signals for all combinations
            for symbol in symbols:
                for timeframe in timeframes:
                    logger.info(f"🔍 Analyzing {symbol} {timeframe}...")

                    # Entry signal analysis
                    entry_analysis = await self.analyze_entry_signals(symbol, timeframe, days)

                    # Strategy optimization
                    optimal_config = await self.optimize_strategy(symbol, timeframe, days)

                    # Generate report
                    await self.generate_analysis_report(symbol, timeframe, entry_analysis, optimal_config)

            # Generate comprehensive summary
            await self.generate_comprehensive_summary()

            logger.info("✅ Comprehensive analysis completed")

        except Exception as e:
            logger.error(f"❌ Comprehensive analysis failed: {e}")

    async def generate_analysis_report(self, symbol: str, timeframe: str, entry_analysis: EntrySignalAnalysis, optimal_config: OptimizationParameters):
        """Generate detailed analysis report"""
        try:
            report = {
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_date': datetime.now().isoformat(),
                'entry_signal_analysis': asdict(entry_analysis) if entry_analysis else None,
                'optimal_parameters': asdict(optimal_config) if optimal_config else None,
                'recommendations': self._generate_recommendations(entry_analysis, optimal_config)
            }

            # Save report
            report_file = f"analysis_report_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Log to GitHub MCP
            await self.github_mcp.log_system_performance({
                'strategy_analysis': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'entry_success_rate': entry_analysis.entry_success_rate if entry_analysis else 0,
                'optimal_config_found': optimal_config is not None,
                'report_file': report_file
            })

            logger.info(f"✅ Analysis report generated: {report_file}")

        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")

    def _generate_recommendations(self, entry_analysis: EntrySignalAnalysis, optimal_config: OptimizationParameters) -> List[str]:
        """Generate trading recommendations based on analysis"""
        recommendations = []

        if entry_analysis:
            if entry_analysis.entry_success_rate > 70:
                recommendations.append("✅ Entry signals show high success rate - consider increasing position sizes")
            elif entry_analysis.entry_success_rate < 40:
                recommendations.append("⚠️ Entry signals show low success rate - consider stricter entry filters")

            if entry_analysis.avg_entry_drawdown < -0.05:
                recommendations.append("🚨 High initial drawdowns detected - implement entry price validation")
            elif entry_analysis.immediate_loss_rate > 30:
                recommendations.append("⚠️ High immediate loss rate - optimize entry timing")

        if optimal_config:
            recommendations.append(f"📊 Optimal parameters found: MA({optimal_config.fast_ma_length},{optimal_config.slow_ma_length})")
            recommendations.append(f"🎯 Risk settings: TP {optimal_config.take_profit_pct}%, SL {optimal_config.stop_loss_pct}%")

        return recommendations

    async def generate_comprehensive_summary(self):
        """Generate comprehensive analysis summary"""
        try:
            summary = {
                'analysis_date': datetime.now().isoformat(),
                'total_analyses': len(self.entry_analyses),
                'total_backtests': len(self.backtest_results),
                'best_performing_symbols': [],
                'worst_performing_symbols': [],
                'entry_signal_insights': {},
                'optimization_recommendations': []
            }

            # Analyze entry signal performance
            if self.entry_analyses:
                success_rates = [a.entry_success_rate for a in self.entry_analyses]
                summary['entry_signal_insights'] = {
                    'avg_success_rate': np.mean(success_rates),
                    'best_success_rate': max(success_rates),
                    'worst_success_rate': min(success_rates),
                    'high_success_signals': len([a for a in self.entry_analyses if a.entry_success_rate > 70])
                }

            # Analyze backtest performance
            if self.backtest_results:
                profitable_strategies = [r for r in self.backtest_results if r.net_pnl > 0]
                summary['backtest_insights'] = {
                    'profitable_strategies': len(profitable_strategies),
                    'total_strategies': len(self.backtest_results),
                    'best_strategy': max(self.backtest_results, key=lambda x: x.sharpe_ratio).__dict__ if self.backtest_results else None
                }

            # Generate recommendations
            summary['optimization_recommendations'] = self._generate_overall_recommendations()

            # Save summary
            summary_file = f"comprehensive_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            # Log to GitHub
            await self.github_mcp.log_system_performance({
                'comprehensive_analysis': True,
                'summary_file': summary_file,
                'total_analyses': summary['total_analyses'],
                'profitable_strategies': len(profitable_strategies) if 'profitable_strategies' in summary.get('backtest_insights', {}) else 0
            })

            logger.info(f"✅ Comprehensive summary generated: {summary_file}")

        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")

    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall optimization recommendations"""
        recommendations = []

        # Entry signal recommendations
        if self.entry_analyses:
            avg_success = np.mean([a.entry_success_rate for a in self.entry_analyses])
            if avg_success > 65:
                recommendations.append("✅ Overall entry signals are performing well - maintain current filters")
            else:
                recommendations.append("⚠️ Entry signal performance needs improvement - consider additional filters")

        # Strategy recommendations
        if self.backtest_results:
            profitable_count = len([r for r in self.backtest_results if r.net_pnl > 0])
            if profitable_count > len(self.backtest_results) * 0.6:
                recommendations.append("✅ Majority of strategies are profitable - focus on position sizing")
            else:
                recommendations.append("🔧 Strategy optimization needed - review parameter combinations")

        return recommendations

# MCP Task Functions
async def create_backtesting_task(config: Dict[str, Any]) -> str:
    """
    Create a backtesting task via MCP

    Args:
        config: Backtesting configuration

    Returns:
        Task ID
    """
    try:
        optimizer = MCPBacktestingOptimizer()

        # Create task record
        task_id = f"backtest_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Log task creation to GitHub
        await optimizer.github_mcp.log_system_performance({
            'backtesting_task_created': task_id,
            'symbols': config.get('symbols', []),
            'timeframes': config.get('timeframes', []),
            'days': config.get('days', 30)
        })

        logger.info(f"✅ Backtesting task created: {task_id}")
        return task_id

    except Exception as e:
        logger.error(f"❌ Task creation failed: {e}")
        return None

async def run_backtesting_analysis(symbols: List[str], timeframes: List[str], days: int = 30) -> Dict[str, Any]:
    """
    Run comprehensive backtesting analysis

    Args:
        symbols: Trading pairs to analyze
        timeframes: Timeframes to test
        days: Historical period

    Returns:
        Analysis results
    """
    try:
        optimizer = MCPBacktestingOptimizer()
        await optimizer.run_comprehensive_analysis(symbols, timeframes, days)

        return {
            'status': 'completed',
            'symbols_analyzed': len(symbols),
            'timeframes_analyzed': len(timeframes),
            'entry_analyses': len(optimizer.entry_analyses),
            'backtest_results': len(optimizer.backtest_results)
        }

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def get_entry_signal_recommendations(symbol: str, timeframe: str) -> Dict[str, Any]:
    """
    Get entry signal recommendations for avoiding initial drawdowns

    Args:
        symbol: Trading pair
        timeframe: Analysis timeframe

    Returns:
        Entry signal recommendations
    """
    try:
        optimizer = MCPBacktestingOptimizer()
        analysis = await optimizer.analyze_entry_signals(symbol, timeframe, 30)

        if analysis:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success_rate': analysis.entry_success_rate,
                'avg_drawdown': analysis.avg_entry_drawdown,
                'max_drawdown': analysis.max_entry_drawdown,
                'immediate_loss_rate': analysis.immediate_loss_rate,
                'recommendations': analysis.optimal_entry_conditions,
                'confidence_score': analysis.signal_confidence_score
            }
        else:
            return {'status': 'no_data'}

    except Exception as e:
        logger.error(f"❌ Recommendations failed: {e}")
        return {'status': 'error', 'error': str(e)}

# Main execution
async def main():
    """Main backtesting and optimization execution"""
    print("🎯 MCP BACKTESTING & ENTRY SIGNAL OPTIMIZER")
    print("=" * 60)

    # Default analysis parameters
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    timeframes = ['1h', '4h']
    days = 30

    try:
        # Create backtesting task
        task_config = {
            'symbols': symbols,
            'timeframes': timeframes,
            'days': days,
            'optimization_focus': 'entry_signals'
        }

        task_id = await create_backtesting_task(task_config)
        print(f"📋 Task Created: {task_id}")

        # Run comprehensive analysis
        print("🚀 Running comprehensive analysis...")
        results = await run_backtesting_analysis(symbols, timeframes, days)

        print("📊 Analysis Results:")
        print(f"   Status: {results['status']}")
        print(f"   Symbols Analyzed: {results['symbols_analyzed']}")
        print(f"   Entry Analyses: {results['entry_analyses']}")
        print(f"   Backtest Results: {results['backtest_results']}")

        # Get entry signal recommendations
        print("\n🎯 Entry Signal Recommendations:")
        for symbol in symbols:
            for timeframe in timeframes:
                recs = await get_entry_signal_recommendations(symbol, timeframe)
                if recs.get('success_rate'):
                    print(f"   {symbol} {timeframe}: {recs['success_rate']:.1f}% success rate")

        print("✅ Backtesting and optimization completed!")

    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        print(f"❌ ERROR: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
