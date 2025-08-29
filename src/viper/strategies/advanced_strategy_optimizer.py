#!/usr/bin/env python3
"""
🚀 ADVANCED STRATEGY OPTIMIZER - COMPREHENSIVE TP/SL & PARAMETER OPTIMIZATION
Addresses user feedback: "45% WIN RATE IS A JOKE" 

This optimizer provides:
✅ Comprehensive TP/SL ratio testing (1:1 to 1:5 risk-reward)
✅ Technical indicator parameter sweeps (lengths, multipliers)
✅ Multi-objective optimization targeting 55%+ win rates
✅ Advanced parameter combinations testing
✅ Real-time optimization progress with results
✅ Automated best configuration selection
✅ Detailed performance improvement comparisons

Target: Achieve 55%+ win rates with optimized TP/SL and technical parameters
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.text import Text
from rich import box
from rich.layout import Layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - STRATEGY_OPTIMIZER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()

@dataclass
class OptimizedStrategyResult:
    """Results from strategy optimization"""
    strategy_name: str
    timeframe: str
    symbol: str
    
    # Original performance (before optimization)
    original_win_rate: float
    original_total_return: float
    original_sharpe: float
    original_max_drawdown: float
    
    # Optimized performance
    optimized_win_rate: float
    optimized_total_return: float
    optimized_sharpe: float
    optimized_max_drawdown: float
    optimized_profit_factor: float
    optimized_total_trades: int
    
    # Optimal parameters found
    optimal_stop_loss_pct: float
    optimal_take_profit_pct: float
    optimal_risk_reward_ratio: float
    optimal_technical_params: Dict[str, Any]
    
    # Performance improvements
    win_rate_improvement: float
    return_improvement: float
    sharpe_improvement: float
    
    # Optimization metadata
    optimization_iterations: int
    optimization_time: float
    convergence_achieved: bool

class AdvancedStrategyOptimizer:
    """
    Advanced strategy optimizer targeting high win rates (55%+) with comprehensive TP/SL optimization
    """
    
    def __init__(self):
        self.console = Console()
        self.results_path = Path("optimization_results")
        self.results_path.mkdir(exist_ok=True)
        
        # Target performance metrics - ADDRESSING USER FEEDBACK!
        self.target_win_rate = 0.55  # 55% minimum win rate target (user wants better than 45%)
        self.target_sharpe = 1.5     # Target Sharpe ratio
        self.max_acceptable_drawdown = 0.15  # 15% maximum drawdown
        
        # Optimization parameters
        self.max_iterations_per_strategy = 500
        self.early_stopping_patience = 50
        self.min_trades_for_validation = 30
        
        # Results storage
        self.optimization_results: List[OptimizedStrategyResult] = []
        self.best_configurations: Dict[str, Dict[str, Any]] = {}
        
        # Strategy templates
        self.strategy_templates = self._initialize_strategy_templates()
        
        # Data generation settings
        self.timeframes = ['5m', '15m', '30m']
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
        self.backtest_days = 90
        
        logger.info("🚀 Advanced Strategy Optimizer initialized - targeting 55%+ win rates")
    
    def _initialize_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize strategy templates for optimization"""
        return {
            'VIPER_Momentum': {
                'type': 'momentum',
                'base_params': {
                    'momentum_period': 14,
                    'volatility_lookback': 20,
                    'trend_confirmation': True,
                    'volume_filter': True
                },
                'optimizable_params': [
                    'momentum_period', 'volatility_lookback', 'fast_ma', 'slow_ma'
                ]
            },
            'Enhanced_Scalper': {
                'type': 'scalping',
                'base_params': {
                    'fast_ma': 5,
                    'slow_ma': 20,
                    'volume_multiplier': 1.5,
                    'quick_exit': True
                },
                'optimizable_params': [
                    'fast_ma', 'slow_ma', 'rsi_period', 'atr_period'
                ]
            },
            'Mean_Reversion_Pro': {
                'type': 'mean_reversion',
                'base_params': {
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'rsi_period': 14,
                    'mean_revert_threshold': 0.02
                },
                'optimizable_params': [
                    'bb_period', 'bb_std', 'rsi_period', 'rsi_overbought', 'rsi_oversold'
                ]
            },
            'Trend_Following_Optimized': {
                'type': 'trend_following',
                'base_params': {
                    'trend_ma': 50,
                    'signal_ma': 12,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9
                },
                'optimizable_params': [
                    'trend_ma', 'signal_ma', 'macd_fast', 'macd_slow', 'atr_multiplier'
                ]
            },
            'Predictive_Ranges': {
                'type': 'range_trading',
                'base_params': {
                    'support_resistance_period': 20,
                    'breakout_threshold': 0.5,
                    'range_filter': True
                },
                'optimizable_params': [
                    'support_resistance_period', 'atr_period', 'bb_period'
                ]
            }
        }

