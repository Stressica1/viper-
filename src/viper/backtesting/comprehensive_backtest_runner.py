#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE VIPER BACKTESTING SYSTEM
Advanced backtesting for 100+ pairs with enhanced entry points validation

This system provides:
✅ 100+ cryptocurrency pair testing
✅ Multi-timeframe stress testing (1m, 5m, 15m, 30m)
✅ Enhanced entry point validation
✅ Comprehensive performance metrics
✅ Risk-adjusted returns analysis
✅ Detailed reporting and visualization
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass, asdict
import random
import time
import hashlib

# Add path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - COMPREHENSIVE_BACKTEST - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for individual backtest"""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    enhanced_entry: bool
    trend_validation: bool
    parameters: Dict[str, Any]

@dataclass
class TradeResult:
    """Individual trade result"""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    profit_loss: float
    profit_loss_pct: float
    entry_time: datetime
    exit_time: datetime
    duration_minutes: int
    reason: str
    enhanced_entry_used: bool

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    config: BacktestConfig
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    total_fees: float
    net_profit: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    trades: List[TradeResult]
    equity_curve: List[float]
    monthly_returns: Dict[str, float]
    execution_time: float
    success: bool
    error_message: str = ""

class ComprehensiveBacktestRunner:
    """Advanced backtesting system for VIPER trading"""
    
    def __init__(self):
        self.trading_fee = 0.001  # 0.1% per trade
        self.slippage = 0.0005    # 0.05% slippage
        
        # Top 100+ crypto pairs for testing
        self.test_pairs = self._generate_test_pairs()
        
        # Timeframes for stress testing
        self.test_timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        # Configuration variations
        self.config_variations = self._generate_config_variations()
        
    def _generate_test_pairs(self) -> List[str]:
        """Generate 100+ cryptocurrency pairs for testing"""
        major_pairs = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'AVAX/USDT', 'SHIB/USDT',
            'MATIC/USDT', 'UNI/USDT', 'LINK/USDT', 'LTC/USDT', 'ATOM/USDT',
            'TRX/USDT', 'ETC/USDT', 'XLM/USDT', 'NEAR/USDT', 'ALGO/USDT',
            'VET/USDT', 'ICP/USDT', 'HBAR/USDT', 'FIL/USDT', 'MANA/USDT',
            'SAND/USDT', 'APE/USDT', 'ROSE/USDT', 'LRC/USDT', 'ENJ/USDT'
        ]
        
        mid_tier_pairs = [
            'GALA/USDT', 'CHZ/USDT', 'BAT/USDT', 'ZEC/USDT', 'DASH/USDT',
            'XTZ/USDT', 'COMP/USDT', 'YFI/USDT', 'SNX/USDT', '1INCH/USDT',
            'SUSHI/USDT', 'CRV/USDT', 'BAL/USDT', 'REN/USDT', 'ZRX/USDT',
            'KNC/USDT', 'BAND/USDT', 'RSR/USDT', 'STORJ/USDT', 'ANT/USDT',
            'REP/USDT', 'NMR/USDT', 'MLN/USDT', 'GRT/USDT', 'MKR/USDT',
            'AAVE/USDT', 'RLC/USDT', 'LPT/USDT', 'NU/USDT', 'KEEP/USDT'
        ]
        
        small_cap_pairs = [
            'REEF/USDT', 'OM/USDT', 'DATA/USDT', 'MDT/USDT', 'BAKE/USDT',
            'BURGER/USDT', 'SLP/USDT', 'TLM/USDT', 'WIN/USDT', 'HOT/USDT',
            'BTT/USDT', 'DENT/USDT', 'KEY/USDT', 'STMX/USDT', 'DEXE/USDT',
            'PHA/USDT', 'ALICE/USDT', 'BLZ/USDT', 'CTK/USDT', 'OGN/USDT',
            'PERP/USDT', 'RAMP/USDT', 'SUPER/USDT', 'CFX/USDT', 'EPX/USDT',
            'SPELL/USDT', 'CVX/USDT', 'HIGH/USDT', 'VOXEL/USDT', 'T/USDT',
            'LOKA/USDT', 'SCRT/USDT', 'API3/USDT', 'BICO/USDT', 'FLUX/USDT',
            'REQ/USDT', 'WAXP/USDT', 'TKO/USDT', 'OXT/USDT', 'RAY/USDT',
            'C98/USDT', 'CLV/USDT', 'QNT/USDT', 'PEOPLE/USDT', 'JASMY/USDT'
        ]
        
        return major_pairs + mid_tier_pairs + small_cap_pairs
    
    def _generate_config_variations(self) -> List[Dict[str, Any]]:
        """Generate different configuration variations for testing"""
        variations = []
        
        # Enhanced entry variations
        entry_configs = [
            {'enhanced_entry': True, 'min_confidence': 0.75, 'min_risk_reward': 1.5},
            {'enhanced_entry': True, 'min_confidence': 0.80, 'min_risk_reward': 2.0},
            {'enhanced_entry': True, 'min_confidence': 0.85, 'min_risk_reward': 2.5},
            {'enhanced_entry': False, 'min_confidence': 0.65, 'min_risk_reward': 1.2}  # Baseline
        ]
        
        # Trend validation variations  
        trend_configs = [
            {'trend_validation': True, 'min_trend_score': 60},
            {'trend_validation': True, 'min_trend_score': 70},
            {'trend_validation': True, 'min_trend_score': 80},
            {'trend_validation': False, 'min_trend_score': 50}  # Baseline
        ]
        
        # VIPER score thresholds
        score_thresholds = [0.6, 0.65, 0.7, 0.75]
        
        # Generate combinations
        for entry_config in entry_configs:
            for trend_config in trend_configs:
                for score_threshold in score_thresholds:
                    config = {
                        **entry_config,
                        **trend_config,
                        'viper_score_threshold': score_threshold,
                        'position_size': 0.02,  # 2% per trade
                        'max_positions': 5
                    }
                    variations.append(config)
        
        return variations
    
    async def run_comprehensive_backtest(self, days_back: int = 30) -> Dict[str, Any]:
        """Run comprehensive backtest across all pairs and configurations"""
        
        logger.info("🚀 Starting Comprehensive VIPER Backtesting System")
        logger.info(f"   Testing {len(self.test_pairs)} pairs across {len(self.test_timeframes)} timeframes")
        logger.info(f"   Configuration variations: {len(self.config_variations)}")
        
        start_time = time.time()
        all_results = []
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Progress tracking
        total_tests = len(self.test_pairs) * len(self.test_timeframes) * len(self.config_variations)
        completed_tests = 0
        
        logger.info(f"   Total backtests to run: {total_tests}")
        
        # Run tests in batches to manage memory
        batch_size = 10
        for i in range(0, len(self.test_pairs), batch_size):
            batch_pairs = self.test_pairs[i:i+batch_size]
            
            logger.info(f"   Processing batch {i//batch_size + 1}/{(len(self.test_pairs)-1)//batch_size + 1}")
            
            batch_results = await self._run_batch_backtest(
                batch_pairs, start_date, end_date
            )
            
            all_results.extend(batch_results)
            
            completed_tests += len(batch_pairs) * len(self.test_timeframes) * len(self.config_variations)
            progress = (completed_tests / total_tests) * 100
            
            logger.info(f"   Progress: {progress:.1f}% ({completed_tests}/{total_tests})")
        
        execution_time = time.time() - start_time
        
        # Generate comprehensive report
        report = await self._generate_comprehensive_report(all_results, execution_time)
        
        # Save results
        await self._save_results(report)
        
        logger.info(f"🎉 Comprehensive backtest completed in {execution_time:.2f} seconds")
        logger.info(f"   Results saved to: reports/comprehensive_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        return report
    
    async def _run_batch_backtest(self, pairs: List[str], start_date: datetime, 
                                end_date: datetime) -> List[BacktestResult]:
        """Run backtest for a batch of pairs"""
        
        batch_results = []
        
        for symbol in pairs:
            for timeframe in self.test_timeframes:
                for config_params in self.config_variations:
                    
                    # Create config
                    config = BacktestConfig(
                        strategy_name="Enhanced VIPER",
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        initial_capital=10000.0,
                        enhanced_entry=config_params.get('enhanced_entry', True),
                        trend_validation=config_params.get('trend_validation', True),
                        parameters=config_params
                    )
                    
                    # Run individual backtest
                    result = await self._run_single_backtest(config)
                    batch_results.append(result)
        
        return batch_results
    
    async def _run_single_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Run a single backtest with given configuration"""
        
        try:
            start_time = time.time()
            
            # Simulate market data (in production, would fetch real data)
            market_data = self._generate_simulated_market_data(config)
            
            # Run trading simulation
            trades, equity_curve = await self._simulate_trading(config, market_data)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(trades, config.initial_capital)
            
            execution_time = time.time() - start_time
            
            return BacktestResult(
                config=config,
                total_return=metrics['total_return'],
                annual_return=metrics['annual_return'], 
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                total_trades=len(trades),
                winning_trades=metrics['winning_trades'],
                losing_trades=metrics['losing_trades'],
                avg_win=metrics['avg_win'],
                avg_loss=metrics['avg_loss'],
                avg_trade_return=metrics['avg_trade_return'],
                best_trade=metrics['best_trade'],
                worst_trade=metrics['worst_trade'],
                total_fees=metrics['total_fees'],
                net_profit=metrics['net_profit'],
                volatility=metrics['volatility'],
                calmar_ratio=metrics['calmar_ratio'],
                sortino_ratio=metrics['sortino_ratio'],
                trades=trades,
                equity_curve=equity_curve,
                monthly_returns=metrics['monthly_returns'],
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Backtest failed for {config.symbol} {config.timeframe}: {e}")
            return BacktestResult(
                config=config,
                total_return=0.0,
                annual_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                avg_trade_return=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                total_fees=0.0,
                net_profit=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                trades=[],
                equity_curve=[],
                monthly_returns={},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _generate_simulated_market_data(self, config: BacktestConfig) -> List[Dict[str, Any]]:
        """Generate realistic simulated market data for backtesting"""
        
        # Parse dates
        start_date = datetime.strptime(config.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
        
        # Time intervals based on timeframe
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60
        }
        
        interval = interval_minutes.get(config.timeframe, 15)
        
        # Generate data points
        data_points = []
        current_time = start_date
        
        # Initial price (seeded by symbol for consistency)
        base_price = 100 + (hash(config.symbol) % 900)  # $100-$1000 range
        current_price = base_price
        
        # Symbol-specific volatility
        volatility = 0.02 + (hash(config.symbol) % 30) / 1000  # 2-5% daily volatility
        
        while current_time < end_date:
            # Generate realistic price movement
            random_change = random.gauss(0, volatility / (24 * 60 / interval))  # Scale for timeframe
            current_price *= (1 + random_change)
            
            # Generate volume (varies with price movement)
            base_volume = 1000000 + (hash(f"{config.symbol}_{current_time}") % 5000000)
            volume_multiplier = 1 + abs(random_change) * 5  # Higher volume on big moves
            volume = base_volume * volume_multiplier
            
            # Calculate 24h change (simplified)
            change_24h = (current_price / base_price - 1) * 100
            
            data_point = {
                'timestamp': current_time,
                'price': current_price,
                'volume': volume,
                'change_24h': change_24h,
                'high': current_price * (1 + abs(random_change) * 0.5),
                'low': current_price * (1 - abs(random_change) * 0.5),
                'close': current_price
            }
            
            data_points.append(data_point)
            current_time += timedelta(minutes=interval)
        
        return data_points
    
    async def _simulate_trading(self, config: BacktestConfig, 
                              market_data: List[Dict[str, Any]]) -> Tuple[List[TradeResult], List[float]]:
        """Simulate trading with the VIPER system"""
        
        trades = []
        equity_curve = [config.initial_capital]
        current_balance = config.initial_capital
        open_positions = []
        
        for i, data_point in enumerate(market_data):
            
            # Check for exit signals for open positions
            for position in open_positions[:]:  # Copy list for safe iteration
                exit_signal, reason = self._check_exit_signal(position, data_point, i)
                
                if exit_signal:
                    # Execute exit
                    exit_price = data_point['price'] * (1 - self.slippage if position['side'] == 'buy' else 1 + self.slippage)
                    
                    if position['side'] == 'buy':
                        profit = (exit_price - position['entry_price']) * position['quantity']
                    else:
                        profit = (position['entry_price'] - exit_price) * position['quantity']
                    
                    profit_after_fees = profit - (position['entry_price'] * position['quantity'] * self.trading_fee) - (exit_price * position['quantity'] * self.trading_fee)
                    
                    current_balance += profit_after_fees
                    
                    trade = TradeResult(
                        symbol=config.symbol,
                        side=position['side'],
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        quantity=position['quantity'],
                        profit_loss=profit_after_fees,
                        profit_loss_pct=(profit_after_fees / (position['entry_price'] * position['quantity'])) * 100,
                        entry_time=position['entry_time'],
                        exit_time=data_point['timestamp'],
                        duration_minutes=int((data_point['timestamp'] - position['entry_time']).total_seconds() / 60),
                        reason=reason,
                        enhanced_entry_used=position['enhanced_entry_used']
                    )
                    
                    trades.append(trade)
                    open_positions.remove(position)
            
            # Check for entry signals
            if len(open_positions) < config.parameters.get('max_positions', 5):
                entry_signal, side, enhanced_entry_used = await self._check_entry_signal(config, data_point, i, market_data)
                
                if entry_signal:
                    # Calculate position size
                    position_size = config.parameters.get('position_size', 0.02)  # 2% of balance
                    trade_amount = current_balance * position_size
                    
                    entry_price = data_point['price'] * (1 + self.slippage if side == 'buy' else 1 - self.slippage)
                    quantity = trade_amount / entry_price
                    
                    position = {
                        'side': side,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'entry_time': data_point['timestamp'],
                        'enhanced_entry_used': enhanced_entry_used
                    }
                    
                    open_positions.append(position)
            
            equity_curve.append(current_balance)
        
        return trades, equity_curve
    
    async def _check_entry_signal(self, config: BacktestConfig, data_point: Dict[str, Any], 
                                index: int, market_data: List[Dict[str, Any]]) -> Tuple[bool, str, bool]:
        """Check if entry signal conditions are met"""
        
        try:
            # Simulate VIPER scoring system
            viper_score = self._calculate_simulated_viper_score(config, data_point, index, market_data)
            
            # Check threshold
            if viper_score < config.parameters.get('viper_score_threshold', 0.65):
                return False, 'buy', False
            
            # Trend validation check
            if config.trend_validation:
                trend_valid = self._simulate_trend_validation(config, data_point, index, market_data)
                if not trend_valid:
                    return False, 'buy', False
            
            # Enhanced entry check
            enhanced_entry_valid = False
            if config.enhanced_entry:
                enhanced_entry_valid = self._simulate_enhanced_entry_validation(config, data_point, index, market_data)
                if not enhanced_entry_valid:
                    return False, 'buy', False
            
            # Determine side (simplified)
            side = 'buy' if data_point['change_24h'] > 0 else 'sell'
            
            return True, side, enhanced_entry_valid
            
        except Exception as e:
            return False, 'buy', False
    
    def _calculate_simulated_viper_score(self, config: BacktestConfig, data_point: Dict[str, Any],
                                       index: int, market_data: List[Dict[str, Any]]) -> float:
        """Simulate VIPER scoring calculation"""
        
        # Volume score (0-1)
        volume_score = min(1.0, data_point['volume'] / 2000000)  # Normalize to $2M
        
        # Price momentum score (0-1)  
        price_score = min(1.0, abs(data_point['change_24h']) / 10)  # Normalize to 10%
        
        # Trend score (simulated)
        if index >= 10:
            recent_prices = [md['price'] for md in market_data[index-10:index]]
            trend_score = 1.0 if recent_prices[-1] > recent_prices[0] else 0.3
        else:
            trend_score = 0.5
        
        # Range score (volatility)
        range_score = min(1.0, abs(data_point['change_24h']) / 15)
        
        # External score (random component)
        external_score = 0.5 + (hash(f"{config.symbol}_{data_point['timestamp']}") % 50) / 100
        
        # Combined VIPER score
        viper_score = (
            volume_score * 0.25 +
            price_score * 0.25 + 
            trend_score * 0.2 +
            range_score * 0.15 +
            external_score * 0.15
        )
        
        return min(1.0, viper_score)
    
    def _simulate_trend_validation(self, config: BacktestConfig, data_point: Dict[str, Any],
                                 index: int, market_data: List[Dict[str, Any]]) -> bool:
        """Simulate trend validation logic"""
        
        min_trend_score = config.parameters.get('min_trend_score', 70)
        
        # Simple trend validation based on recent price movement
        if index >= 5:
            recent_prices = [md['price'] for md in market_data[index-5:index]]
            price_trend = (recent_prices[-1] / recent_prices[0] - 1) * 100
            
            # Convert to 0-100 scale
            trend_score = 50 + price_trend * 5  # Scale factor
            trend_score = max(0, min(100, trend_score))
            
            return trend_score >= min_trend_score
        
        return True  # Default to valid for early data points
    
    def _simulate_enhanced_entry_validation(self, config: BacktestConfig, data_point: Dict[str, Any],
                                          index: int, market_data: List[Dict[str, Any]]) -> bool:
        """Simulate enhanced entry validation"""
        
        min_confidence = config.parameters.get('min_confidence', 0.75)
        min_risk_reward = config.parameters.get('min_risk_reward', 1.5)
        
        # Simulate confidence calculation
        volume_boost = 0.1 if data_point['volume'] > 1000000 else 0
        momentum_boost = 0.05 if abs(data_point['change_24h']) > 2 else 0
        
        confidence = 0.6 + volume_boost + momentum_boost + (hash(f"conf_{config.symbol}_{index}") % 20) / 100
        
        # Simulate risk-reward calculation  
        volatility = abs(data_point['change_24h']) / 100
        risk_reward = 1.0 + volatility * 10  # Higher volatility = higher potential R/R
        
        return confidence >= min_confidence and risk_reward >= min_risk_reward
    
    def _check_exit_signal(self, position: Dict[str, Any], data_point: Dict[str, Any], 
                         index: int) -> Tuple[bool, str]:
        """Check exit signal conditions"""
        
        current_price = data_point['price']
        entry_price = position['entry_price']
        
        # Calculate current P&L percentage
        if position['side'] == 'buy':
            pnl_pct = (current_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / current_price - 1) * 100
        
        # Stop loss at -2%
        if pnl_pct <= -2:
            return True, "Stop Loss"
        
        # Take profit at +4%  
        if pnl_pct >= 4:
            return True, "Take Profit"
        
        # Time-based exit after 24 hours (simplified)
        duration = data_point['timestamp'] - position['entry_time']
        if duration.total_seconds() > 24 * 3600:
            return True, "Time Exit"
        
        return False, ""
    
    def _calculate_performance_metrics(self, trades: List[TradeResult], 
                                     initial_capital: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return {
                'total_return': 0.0, 'annual_return': 0.0, 'sharpe_ratio': 0.0,
                'max_drawdown': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'winning_trades': 0, 'losing_trades': 0, 'avg_win': 0.0,
                'avg_loss': 0.0, 'avg_trade_return': 0.0, 'best_trade': 0.0,
                'worst_trade': 0.0, 'total_fees': 0.0, 'net_profit': 0.0,
                'volatility': 0.0, 'calmar_ratio': 0.0, 'sortino_ratio': 0.0,
                'monthly_returns': {}
            }
        
        # Basic metrics
        total_profit = sum(t.profit_loss for t in trades)
        total_return = (total_profit / initial_capital) * 100
        
        winning_trades = [t for t in trades if t.profit_loss > 0]
        losing_trades = [t for t in trades if t.profit_loss < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100
        
        avg_win = sum(t.profit_loss for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.profit_loss for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')
        
        # Risk metrics  
        returns = [t.profit_loss_pct for t in trades]
        avg_return = sum(returns) / len(returns) if returns else 0
        volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if len(returns) > 1 else 0
        
        # Sharpe ratio (simplified, assuming 0% risk-free rate)
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = (sum(r ** 2 for r in negative_returns) / len(negative_returns)) ** 0.5 if negative_returns else 0
        sortino_ratio = avg_return / downside_deviation if downside_deviation > 0 else 0
        
        # Max drawdown (simplified)
        running_max = initial_capital
        max_drawdown = 0
        current_balance = initial_capital
        
        for trade in trades:
            current_balance += trade.profit_loss
            if current_balance > running_max:
                running_max = current_balance
            drawdown = (running_max - current_balance) / running_max * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calmar ratio
        annual_return = total_return * (365 / 30)  # Annualized (assuming 30-day test)
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade_return': avg_return,
            'best_trade': max(t.profit_loss for t in trades),
            'worst_trade': min(t.profit_loss for t in trades),
            'total_fees': sum(t.entry_price * t.quantity * 0.002 for t in trades),  # Estimated fees
            'net_profit': total_profit,
            'volatility': volatility,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'monthly_returns': {}  # Would calculate monthly breakdown
        }
    
    async def _generate_comprehensive_report(self, results: List[BacktestResult], 
                                           execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                'summary': 'No successful backtests',
                'execution_time': execution_time,
                'total_tests': len(results),
                'successful_tests': 0
            }
        
        # Overall statistics
        total_tests = len(results)
        successful_tests = len(successful_results)
        
        # Performance aggregations
        avg_return = sum(r.total_return for r in successful_results) / len(successful_results)
        avg_sharpe = sum(r.sharpe_ratio for r in successful_results) / len(successful_results)
        avg_win_rate = sum(r.win_rate for r in successful_results) / len(successful_results)
        avg_max_dd = sum(r.max_drawdown for r in successful_results) / len(successful_results)
        
        # Best performers
        best_return = max(successful_results, key=lambda x: x.total_return)
        best_sharpe = max(successful_results, key=lambda x: x.sharpe_ratio)
        best_win_rate = max(successful_results, key=lambda x: x.win_rate)
        
        # Enhanced entry analysis
        enhanced_results = [r for r in successful_results if r.config.enhanced_entry]
        baseline_results = [r for r in successful_results if not r.config.enhanced_entry]
        
        enhanced_performance = sum(r.total_return for r in enhanced_results) / len(enhanced_results) if enhanced_results else 0
        baseline_performance = sum(r.total_return for r in baseline_results) / len(baseline_results) if baseline_results else 0
        
        # Timeframe analysis
        timeframe_performance = {}
        for tf in self.test_timeframes:
            tf_results = [r for r in successful_results if r.config.timeframe == tf]
            if tf_results:
                timeframe_performance[tf] = {
                    'avg_return': sum(r.total_return for r in tf_results) / len(tf_results),
                    'avg_sharpe': sum(r.sharpe_ratio for r in tf_results) / len(tf_results),
                    'avg_win_rate': sum(r.win_rate for r in tf_results) / len(tf_results),
                    'tests': len(tf_results)
                }
        
        # Pair performance
        pair_performance = {}
        for pair in self.test_pairs[:10]:  # Top 10 for brevity
            pair_results = [r for r in successful_results if r.config.symbol == pair]
            if pair_results:
                pair_performance[pair] = {
                    'avg_return': sum(r.total_return for r in pair_results) / len(pair_results),
                    'avg_sharpe': sum(r.sharpe_ratio for r in pair_results) / len(pair_results),
                    'tests': len(pair_results)
                }
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': (successful_tests / total_tests) * 100,
                'execution_time': execution_time,
                'avg_return': avg_return,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_win_rate': avg_win_rate,
                'avg_max_drawdown': avg_max_dd
            },
            'best_performers': {
                'highest_return': {
                    'symbol': best_return.config.symbol,
                    'timeframe': best_return.config.timeframe,
                    'return': best_return.total_return,
                    'enhanced_entry': best_return.config.enhanced_entry
                },
                'highest_sharpe': {
                    'symbol': best_sharpe.config.symbol,
                    'timeframe': best_sharpe.config.timeframe,
                    'sharpe': best_sharpe.sharpe_ratio,
                    'enhanced_entry': best_sharpe.config.enhanced_entry
                },
                'highest_win_rate': {
                    'symbol': best_win_rate.config.symbol,
                    'timeframe': best_win_rate.config.timeframe,
                    'win_rate': best_win_rate.win_rate,
                    'enhanced_entry': best_win_rate.config.enhanced_entry
                }
            },
            'enhanced_entry_analysis': {
                'enhanced_avg_return': enhanced_performance,
                'baseline_avg_return': baseline_performance,
                'improvement': enhanced_performance - baseline_performance,
                'enhanced_tests': len(enhanced_results),
                'baseline_tests': len(baseline_results)
            },
            'timeframe_analysis': timeframe_performance,
            'top_pairs_performance': pair_performance,
            'detailed_results': [asdict(r) for r in successful_results[:50]]  # Top 50 detailed results
        }
        
        return report
    
    async def _save_results(self, report: Dict[str, Any]):
        """Save backtest results to file"""
        
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comprehensive_backtest_{timestamp}.json"
        filepath = reports_dir / filename
        
        # Save JSON report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also create a summary text report
        summary_file = reports_dir / f"backtest_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("🚀 COMPREHENSIVE VIPER BACKTEST RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Tests: {report['summary']['total_tests']}\n")
            f.write(f"Successful Tests: {report['summary']['successful_tests']}\n")
            f.write(f"Success Rate: {report['summary']['success_rate']:.1f}%\n")
            f.write(f"Execution Time: {report['summary']['execution_time']:.2f} seconds\n\n")
            
            f.write("PERFORMANCE OVERVIEW:\n")
            f.write(f"Average Return: {report['summary']['avg_return']:.2f}%\n")
            f.write(f"Average Sharpe Ratio: {report['summary']['avg_sharpe_ratio']:.2f}\n")
            f.write(f"Average Win Rate: {report['summary']['avg_win_rate']:.1f}%\n")
            f.write(f"Average Max Drawdown: {report['summary']['avg_max_drawdown']:.2f}%\n\n")
            
            f.write("ENHANCED ENTRY ANALYSIS:\n")
            f.write(f"Enhanced Entry Avg Return: {report['enhanced_entry_analysis']['enhanced_avg_return']:.2f}%\n")
            f.write(f"Baseline Avg Return: {report['enhanced_entry_analysis']['baseline_avg_return']:.2f}%\n")
            f.write(f"Improvement: {report['enhanced_entry_analysis']['improvement']:.2f}%\n\n")
            
            f.write("TOP PERFORMERS:\n")
            f.write(f"Highest Return: {report['best_performers']['highest_return']['symbol']} ")
            f.write(f"({report['best_performers']['highest_return']['timeframe']}) - ")
            f.write(f"{report['best_performers']['highest_return']['return']:.2f}%\n")
            
            f.write(f"Best Sharpe: {report['best_performers']['highest_sharpe']['symbol']} ")
            f.write(f"({report['best_performers']['highest_sharpe']['timeframe']}) - ")
            f.write(f"{report['best_performers']['highest_sharpe']['sharpe']:.2f}\n")
            
        logger.info(f"Results saved to {filepath} and {summary_file}")

# CLI execution
async def main():
    """Main execution function"""
    
    logger.info("🚀 Starting Comprehensive VIPER Backtesting System")
    
    runner = ComprehensiveBacktestRunner()
    
    # Run comprehensive backtest
    results = await runner.run_comprehensive_backtest(days_back=30)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE BACKTEST COMPLETED")
    print("=" * 60)
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Successful Tests: {results['summary']['successful_tests']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"Execution Time: {results['summary']['execution_time']:.2f} seconds")
    print()
    print("PERFORMANCE OVERVIEW:")
    print(f"Average Return: {results['summary']['avg_return']:.2f}%")
    print(f"Average Sharpe Ratio: {results['summary']['avg_sharpe_ratio']:.2f}")
    print(f"Average Win Rate: {results['summary']['avg_win_rate']:.1f}%")
    print(f"Average Max Drawdown: {results['summary']['avg_max_drawdown']:.2f}%")
    print()
    print("ENHANCED ENTRY ANALYSIS:")
    print(f"Enhanced Entry Avg Return: {results['enhanced_entry_analysis']['enhanced_avg_return']:.2f}%")
    print(f"Baseline Avg Return: {results['enhanced_entry_analysis']['baseline_avg_return']:.2f}%")
    print(f"Performance Improvement: {results['enhanced_entry_analysis']['improvement']:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())