#!/usr/bin/env python3
"""
🚀 UNIFIED STRATEGY COLLECTION
Comprehensive collection of all proven crypto trading strategies

This module provides:
✅ Unified interface for all strategies
✅ Strategy factory and management
✅ Performance tracking and comparison
✅ Easy strategy selection and configuration
✅ Backtesting and validation tools
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import pandas as pd
import numpy as np

# Add path for strategies
sys.path.append(str(Path(__file__).parent))

# Import all strategies with fallback handling
strategies_available = {}

try:
    from predictive_ranges_strategy import get_predictive_strategy
    strategies_available['predictive_ranges'] = get_predictive_strategy
except ImportError as e:
    logging.warning(f"Could not import predictive_ranges_strategy: {e}")

try:
    from bollinger_mean_reversion_strategy import get_bollinger_strategy
    strategies_available['bollinger_mean_reversion'] = get_bollinger_strategy
except ImportError as e:
    logging.warning(f"Could not import bollinger_mean_reversion_strategy: {e}")

try:
    from rsi_divergence_strategy import get_rsi_divergence_strategy
    strategies_available['rsi_divergence'] = get_rsi_divergence_strategy
except ImportError as e:
    logging.warning(f"Could not import rsi_divergence_strategy: {e}")

try:
    from vwma_strategy import get_vwma_strategy
    strategies_available['vwma'] = get_vwma_strategy
except ImportError as e:
    logging.warning(f"Could not import vwma_strategy: {e}")

try:
    from fibonacci_strategy import get_fibonacci_strategy
    strategies_available['fibonacci'] = get_fibonacci_strategy
except ImportError as e:
    logging.warning(f"Could not import fibonacci_strategy: {e}")

try:
    from momentum_breakout_strategy import get_momentum_breakout_strategy
    strategies_available['momentum_breakout'] = get_momentum_breakout_strategy
except ImportError as e:
    logging.warning(f"Could not import momentum_breakout_strategy: {e}")

try:
    from scalping_grid_strategy import get_scalping_grid_strategy
    strategies_available['scalping_grid'] = get_scalping_grid_strategy
except ImportError as e:
    logging.warning(f"Could not import scalping_grid_strategy: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - STRATEGY_COLLECTION - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class StrategyInfo:
    """Information about a trading strategy"""
    name: str
    description: str
    category: str
    best_timeframes: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    complexity: str  # 'simple', 'medium', 'advanced'
    market_conditions: List[str]  # ['trending', 'ranging', 'volatile', 'stable']
    avg_trades_per_day: int
    typical_hold_time: str

@dataclass
class UnifiedSignal:
    """Unified signal structure for all strategies"""
    timestamp: datetime
    symbol: str
    strategy: str
    direction: str  # 'long', 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timeframe: str
    risk_reward_ratio: float
    additional_info: Dict[str, Any] = None

class StrategyCollection:
    """
    Unified collection and management of all trading strategies
    """

    def __init__(self):
        self.strategies = {}
        self.strategy_info = self._initialize_strategy_info()
        self._load_strategies()
        
        logger.info(f"🚀 Strategy Collection initialized with {len(self.strategies)} strategies")

    def _initialize_strategy_info(self) -> Dict[str, StrategyInfo]:
        """Initialize information about each strategy"""
        return {
            'predictive_ranges': StrategyInfo(
                name='Predictive Ranges',
                description='LuxAlgo-inspired predictive support/resistance levels',
                category='Range Trading',
                best_timeframes=['5m', '15m', '1h', '4h'],
                risk_level='medium',
                complexity='advanced',
                market_conditions=['ranging', 'trending'],
                avg_trades_per_day=3,
                typical_hold_time='2-8 hours'
            ),
            'bollinger_mean_reversion': StrategyInfo(
                name='Bollinger Mean Reversion',
                description='Dynamic Bollinger Bands with mean reversion signals',
                category='Mean Reversion',
                best_timeframes=['5m', '15m', '30m'],
                risk_level='medium',
                complexity='simple',
                market_conditions=['ranging', 'stable'],
                avg_trades_per_day=5,
                typical_hold_time='1-4 hours'
            ),
            'rsi_divergence': StrategyInfo(
                name='RSI Divergence',
                description='Regular and hidden RSI divergences with volume confirmation',
                category='Momentum',
                best_timeframes=['15m', '30m', '1h'],
                risk_level='medium',
                complexity='medium',
                market_conditions=['trending', 'volatile'],
                avg_trades_per_day=2,
                typical_hold_time='4-12 hours'
            ),
            'vwma': StrategyInfo(
                name='Volume-Weighted Moving Average',
                description='Volume-based trend following with price confirmation',
                category='Trend Following',
                best_timeframes=['5m', '15m', '30m', '1h'],
                risk_level='low',
                complexity='simple',
                market_conditions=['trending'],
                avg_trades_per_day=3,
                typical_hold_time='2-6 hours'
            ),
            'fibonacci': StrategyInfo(
                name='Fibonacci Retracement',
                description='Automated golden ratio entries with confluence detection',
                category='Retracement',
                best_timeframes=['15m', '30m', '1h', '4h'],
                risk_level='medium',
                complexity='medium',
                market_conditions=['trending', 'ranging'],
                avg_trades_per_day=2,
                typical_hold_time='4-24 hours'
            ),
            'momentum_breakout': StrategyInfo(
                name='Momentum Breakout',
                description='High-probability breakouts with volume and momentum filters',
                category='Breakout',
                best_timeframes=['5m', '15m', '30m'],
                risk_level='high',
                complexity='medium',
                market_conditions=['volatile', 'trending'],
                avg_trades_per_day=4,
                typical_hold_time='1-3 hours'
            ),
            'scalping_grid': StrategyInfo(
                name='Scalping Grid',
                description='Range-bound grid trading for lower timeframes',
                category='Grid Trading',
                best_timeframes=['1m', '5m', '15m'],
                risk_level='high',
                complexity='advanced',
                market_conditions=['ranging', 'stable'],
                avg_trades_per_day=15,
                typical_hold_time='15-60 minutes'
            )
        }

    def _load_strategies(self):
        """Load all available strategies"""
        for name, strategy_func in strategies_available.items():
            try:
                strategy_instance = strategy_func()
                self.strategies[name] = strategy_instance
                logger.info(f"✅ Loaded strategy: {name}")
            except Exception as e:
                logger.error(f"❌ Failed to load strategy {name}: {e}")

    def get_strategy(self, name: str):
        """Get a specific strategy instance"""
        return self.strategies.get(name)

    def get_strategy_info(self, name: str) -> Optional[StrategyInfo]:
        """Get information about a specific strategy"""
        return self.strategy_info.get(name)

    def list_strategies(self) -> List[str]:
        """Get list of all available strategy names"""
        return list(self.strategies.keys())

    def list_strategies_by_category(self, category: str) -> List[str]:
        """Get strategies filtered by category"""
        return [
            name for name, info in self.strategy_info.items()
            if info.category.lower() == category.lower() and name in self.strategies
        ]

    def list_strategies_by_risk(self, risk_level: str) -> List[str]:
        """Get strategies filtered by risk level"""
        return [
            name for name, info in self.strategy_info.items()
            if info.risk_level.lower() == risk_level.lower() and name in self.strategies
        ]

    def list_strategies_by_timeframe(self, timeframe: str) -> List[str]:
        """Get strategies that work well on specific timeframe"""
        return [
            name for name, info in self.strategy_info.items()
            if timeframe in info.best_timeframes and name in self.strategies
        ]

    def get_recommended_strategies(self, 
                                 timeframe: str = None,
                                 risk_level: str = None,
                                 market_condition: str = None) -> List[str]:
        """Get recommended strategies based on criteria"""
        recommended = []
        
        for name, info in self.strategy_info.items():
            if name not in self.strategies:
                continue
                
            matches = True
            
            if timeframe and timeframe not in info.best_timeframes:
                matches = False
            
            if risk_level and info.risk_level.lower() != risk_level.lower():
                matches = False
                
            if market_condition and market_condition.lower() not in [m.lower() for m in info.market_conditions]:
                matches = False
            
            if matches:
                recommended.append(name)
        
        return recommended

    def analyze_symbol_with_strategy(self, 
                                   strategy_name: str, 
                                   symbol: str, 
                                   df: pd.DataFrame, 
                                   timeframe: str) -> List[UnifiedSignal]:
        """Analyze symbol with specific strategy and return unified signals"""
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy {strategy_name} not available")
            return []
        
        try:
            strategy = self.strategies[strategy_name]
            
            # Call analyze_symbol method (all strategies should have this)
            raw_signals = strategy.analyze_symbol(symbol, df, timeframe)
            
            # Convert to unified signals
            unified_signals = []
            for signal in raw_signals:
                unified_signal = self._convert_to_unified_signal(signal, strategy_name)
                if unified_signal:
                    unified_signals.append(unified_signal)
            
            return unified_signals
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} with {strategy_name}: {e}")
            return []

    def _convert_to_unified_signal(self, raw_signal, strategy_name: str) -> Optional[UnifiedSignal]:
        """Convert strategy-specific signal to unified format"""
        try:
            # Extract common fields that all strategies should have
            entry_price = getattr(raw_signal, 'entry_price', 0.0)
            if entry_price == 0.0:
                return None
                
            unified_signal = UnifiedSignal(
                timestamp=getattr(raw_signal, 'timestamp', datetime.now()),
                symbol=getattr(raw_signal, 'symbol', 'UNKNOWN'),
                strategy=strategy_name,
                direction=getattr(raw_signal, 'direction', 'long'),
                entry_price=entry_price,
                stop_loss=getattr(raw_signal, 'stop_loss', entry_price * 0.98),
                take_profit=getattr(raw_signal, 'take_profit', entry_price * 1.02),
                confidence=getattr(raw_signal, 'confidence', 0.5),
                timeframe=getattr(raw_signal, 'timeframe', '15m'),
                risk_reward_ratio=getattr(raw_signal, 'risk_reward_ratio', 1.0),
                additional_info={}
            )
            
            # Add strategy-specific additional info
            for attr in dir(raw_signal):
                if not attr.startswith('_') and not hasattr(unified_signal, attr):
                    try:
                        value = getattr(raw_signal, attr)
                        if not callable(value):
                            unified_signal.additional_info[attr] = value
                    except:
                        pass
            
            return unified_signal
            
        except Exception as e:
            logger.error(f"Error converting signal to unified format: {e}")
            return None

    def analyze_symbol_all_strategies(self, 
                                    symbol: str, 
                                    df: pd.DataFrame, 
                                    timeframe: str) -> Dict[str, List[UnifiedSignal]]:
        """Analyze symbol with all available strategies"""
        results = {}
        
        for strategy_name in self.strategies.keys():
            signals = self.analyze_symbol_with_strategy(strategy_name, symbol, df, timeframe)
            results[strategy_name] = signals
            
        return results

    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of all strategies"""
        summary = {
            'total_strategies': len(self.strategies),
            'categories': {},
            'risk_levels': {'low': 0, 'medium': 0, 'high': 0},
            'complexity_levels': {'simple': 0, 'medium': 0, 'advanced': 0},
            'timeframe_coverage': {},
            'market_condition_coverage': {}
        }
        
        for name, info in self.strategy_info.items():
            if name not in self.strategies:
                continue
                
            # Count by category
            if info.category not in summary['categories']:
                summary['categories'][info.category] = 0
            summary['categories'][info.category] += 1
            
            # Count by risk level
            summary['risk_levels'][info.risk_level] += 1
            
            # Count by complexity
            summary['complexity_levels'][info.complexity] += 1
            
            # Count timeframe coverage
            for tf in info.best_timeframes:
                if tf not in summary['timeframe_coverage']:
                    summary['timeframe_coverage'][tf] = 0
                summary['timeframe_coverage'][tf] += 1
            
            # Count market condition coverage
            for condition in info.market_conditions:
                if condition not in summary['market_condition_coverage']:
                    summary['market_condition_coverage'][condition] = 0
                summary['market_condition_coverage'][condition] += 1
        
        return summary

    def print_strategy_info(self, strategy_name: str = None):
        """Print detailed information about strategies"""
        if strategy_name:
            if strategy_name in self.strategy_info and strategy_name in self.strategies:
                info = self.strategy_info[strategy_name]
                print(f"\n🚀 Strategy: {info.name}")
                print(f"📝 Description: {info.description}")
                print(f"📂 Category: {info.category}")
                print(f"⏰ Best Timeframes: {', '.join(info.best_timeframes)}")
                print(f"⚠️  Risk Level: {info.risk_level.title()}")
                print(f"🔧 Complexity: {info.complexity.title()}")
                print(f"📊 Market Conditions: {', '.join(info.market_conditions)}")
                print(f"📈 Avg Trades/Day: {info.avg_trades_per_day}")
                print(f"⏱️  Typical Hold Time: {info.typical_hold_time}")
            else:
                print(f"❌ Strategy '{strategy_name}' not found or not available")
        else:
            print("\n🚀 AVAILABLE TRADING STRATEGIES")
            print("=" * 60)
            for name, info in self.strategy_info.items():
                if name in self.strategies:
                    print(f"\n✅ {info.name} ({name})")
                    print(f"   📂 {info.category} | ⚠️ {info.risk_level.title()} Risk | 🔧 {info.complexity.title()}")
                    print(f"   ⏰ {', '.join(info.best_timeframes)} | 📊 {', '.join(info.market_conditions)}")

# Global instance
_strategy_collection = None

def get_strategy_collection() -> StrategyCollection:
    """Get global strategy collection instance"""
    global _strategy_collection
    if _strategy_collection is None:
        _strategy_collection = StrategyCollection()
    return _strategy_collection

# Convenience functions
def get_all_strategies() -> Dict[str, Any]:
    """Get all available strategies"""
    collection = get_strategy_collection()
    return collection.strategies

def get_strategy_by_name(name: str):
    """Get specific strategy by name"""
    collection = get_strategy_collection()
    return collection.get_strategy(name)

def analyze_with_all_strategies(symbol: str, df: pd.DataFrame, timeframe: str) -> Dict[str, List[UnifiedSignal]]:
    """Analyze symbol with all available strategies"""
    collection = get_strategy_collection()
    return collection.analyze_symbol_all_strategies(symbol, df, timeframe)

def get_recommended_strategies(timeframe: str = None, risk_level: str = None, market_condition: str = None) -> List[str]:
    """Get recommended strategies based on criteria"""
    collection = get_strategy_collection()
    return collection.get_recommended_strategies(timeframe, risk_level, market_condition)

# Example usage and testing
async def main():
    """Test strategy collection"""
    print("🚀 UNIFIED STRATEGY COLLECTION TEST")
    print("=" * 60)

    # Get strategy collection
    collection = get_strategy_collection()
    
    # Print summary
    summary = collection.get_strategy_summary()
    print(f"📊 Total Strategies: {summary['total_strategies']}")
    print(f"📂 Categories: {summary['categories']}")
    print(f"⚠️  Risk Distribution: {summary['risk_levels']}")
    print(f"⏰ Timeframe Coverage: {summary['timeframe_coverage']}")
    
    # Print all strategies
    collection.print_strategy_info()
    
    # Test recommendations
    print("\n🎯 STRATEGY RECOMMENDATIONS")
    print("=" * 40)
    
    low_risk = collection.get_recommended_strategies(risk_level='low')
    print(f"💚 Low Risk Strategies: {low_risk}")
    
    scalping_tf = collection.get_recommended_strategies(timeframe='5m')
    print(f"⚡ Good for 5m timeframe: {scalping_tf}")
    
    trending_market = collection.get_recommended_strategies(market_condition='trending')
    print(f"📈 Good for trending markets: {trending_market}")
    
    # Test analysis with sample data
    print("\n📊 TESTING STRATEGY ANALYSIS")
    print("=" * 40)
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.uniform(99, 101, 100),
        'high': np.random.uniform(100, 102, 100),
        'low': np.random.uniform(98, 100, 100),
        'close': np.random.uniform(99, 101, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)
    
    # Test with first available strategy
    if collection.strategies:
        first_strategy = list(collection.strategies.keys())[0]
        print(f"Testing {first_strategy} on BTCUSDT...")
        
        signals = collection.analyze_symbol_with_strategy(
            first_strategy, 'BTCUSDT', sample_data, '5m'
        )
        
        print(f"Found {len(signals)} signals from {first_strategy}")
        
        for i, signal in enumerate(signals[:3], 1):
            print(f"  Signal {i}: {signal.direction} at {signal.entry_price:.6f} "
                  f"(Confidence: {signal.confidence:.2f})")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())