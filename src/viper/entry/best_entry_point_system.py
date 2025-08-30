#!/usr/bin/env python3
"""
🎯 BEST ENTRY POINT SYSTEM
Comprehensive system to ensure we get the absolute best entry points for trades

This system:
- Aggregates entry signals from all optimization systems
- Validates and scores each entry point
- Filters for only the highest quality entries
- Provides real-time entry point monitoring
- Ensures optimal entry timing and pricing
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BEST_ENTRY_SYSTEM - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EntryQuality(Enum):
    PREMIUM = 5      # Only the absolute best entries
    EXCELLENT = 4    # High quality entries
    GOOD = 3         # Acceptable entries
    FAIR = 2         # Below average entries  
    POOR = 1         # Low quality entries

@dataclass
class BestEntryPoint:
    """Represents the best possible entry point with comprehensive analysis"""
    symbol: str
    direction: str  # 'buy' or 'sell'
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Quality metrics
    overall_quality: EntryQuality
    confidence_score: float
    risk_reward_ratio: float
    
    # Entry analysis
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    timing_score: float
    
    # System consensus
    system_consensus: Dict[str, float]  # Scores from each entry system
    consensus_strength: float
    
    # Risk metrics
    position_size: float
    expected_profit: float
    max_drawdown: float
    win_probability: float
    
    # Metadata
    timeframe: str
    created_at: datetime
    valid_until: datetime
    reasons: List[str]

class BestEntryPointSystem:
    """
    System to identify and validate the absolute best entry points
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize all entry systems
        self._initialize_entry_systems()
        
        # Entry point tracking
        self.current_best_entries: Dict[str, BestEntryPoint] = {}
        self.entry_history: List[BestEntryPoint] = []
        
        logger.info("🎯 Best Entry Point System initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for best entry point detection"""
        return {
            # Quality thresholds - Adjusted for better functionality
            'min_overall_confidence': 0.6,  # Reduced from 0.8
            'min_consensus_strength': 0.5,  # Reduced from 0.7 
            'min_risk_reward_ratio': 2.0,   # Reduced from 2.5
            'min_win_probability': 0.5,     # Reduced from 0.6
            
            # System weights
            'optimized_entry_weight': 0.3,
            'enhanced_optimizer_weight': 0.25,
            'signal_generator_weight': 0.25,
            'entry_point_manager_weight': 0.2,
            
            # Entry validation
            'max_entries_per_symbol': 1,  # Only the best entry per symbol
            'entry_validity_hours': 2,
            'revalidation_interval_minutes': 15,
            
            # Quality requirements - Adjusted thresholds
            'premium_threshold': 0.85,   # Reduced from 0.9
            'excellent_threshold': 0.75, # Reduced from 0.8
            'good_threshold': 0.65,      # Reduced from 0.7
            'fair_threshold': 0.55       # Reduced from 0.6
        }
    
    def _initialize_entry_systems(self):
        """Initialize all entry optimization systems"""
        self.systems = {}
        
        try:
            from ..execution.optimized_trade_entry_system import OptimizedTradeEntrySystem
            self.systems['optimized_entry'] = OptimizedTradeEntrySystem()
            logger.info("✅ Optimized Entry System loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Optimized Entry System: {e}")
        
        try:
            from ..strategies.enhanced_trade_entry_optimizer import EnhancedTradeEntryOptimizer
            self.systems['enhanced_optimizer'] = EnhancedTradeEntryOptimizer()
            logger.info("✅ Enhanced Trade Entry Optimizer loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Enhanced Optimizer: {e}")
        
        try:
            from ..core.enhanced_entry_signals import EnhancedEntrySignalGenerator
            self.systems['signal_generator'] = EnhancedEntrySignalGenerator()
            logger.info("✅ Enhanced Entry Signal Generator loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Signal Generator: {e}")
        
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent.parent.parent / "scripts"))
            from optimal_entry_point_manager import OptimalEntryPointManager
            self.systems['entry_point_manager'] = OptimalEntryPointManager()
            logger.info("✅ Optimal Entry Point Manager loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Entry Point Manager: {e}")
    
    async def find_best_entry_points(self, symbols: List[str], 
                                   market_data: Dict[str, Dict[str, pd.DataFrame]],
                                   account_balance: float) -> Dict[str, BestEntryPoint]:
        """
        Find the absolute best entry points for given symbols
        """
        best_entries = {}
        
        for symbol in symbols:
            try:
                symbol_market_data = market_data.get(symbol, {})
                if not symbol_market_data:
                    logger.warning(f"⚠️ No market data for {symbol}")
                    continue
                
                # Get current price
                current_price = self._get_current_price(symbol_market_data)
                if not current_price:
                    continue
                
                # Collect entry signals from all systems
                all_signals = await self._collect_all_entry_signals(
                    symbol, symbol_market_data, current_price, account_balance
                )
                
                # Find the best entry point
                best_entry = await self._analyze_and_select_best_entry(
                    symbol, all_signals, current_price, account_balance
                )
                
                if best_entry and self._validate_entry_quality(best_entry):
                    best_entries[symbol] = best_entry
                    self.current_best_entries[symbol] = best_entry
                    logger.info(f"🎯 Found BEST entry for {symbol}: "
                              f"{best_entry.direction} @ {best_entry.entry_price:.6f} "
                              f"(Quality: {best_entry.overall_quality.name}, "
                              f"Confidence: {best_entry.confidence_score:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Error finding best entry for {symbol}: {e}")
        
        return best_entries
    
    async def _collect_all_entry_signals(self, symbol: str, market_data: Dict[str, pd.DataFrame],
                                       current_price: float, account_balance: float) -> Dict[str, Any]:
        """Collect entry signals from all systems"""
        signals = {}
        
        # Optimized Entry System
        if 'optimized_entry' in self.systems:
            try:
                optimized_signals = await self.systems['optimized_entry'].analyze_optimal_entries(
                    symbol, market_data, current_price, account_balance
                )
                signals['optimized_entry'] = optimized_signals
            except Exception as e:
                logger.warning(f"⚠️ Optimized entry error for {symbol}: {e}")
        
        # Enhanced Trade Entry Optimizer
        if 'enhanced_optimizer' in self.systems:
            try:
                enhanced_signals = await self.systems['enhanced_optimizer'].analyze_entry_signals(
                    symbol, market_data
                )
                signals['enhanced_optimizer'] = enhanced_signals
            except Exception as e:
                logger.warning(f"⚠️ Enhanced optimizer error for {symbol}: {e}")
        
        # Entry Signal Generator
        if 'signal_generator' in self.systems:
            try:
                generated_signals = await self.systems['signal_generator'].generate_entry_signals(
                    symbol, market_data, current_price
                )
                signals['signal_generator'] = generated_signals
            except Exception as e:
                logger.warning(f"⚠️ Signal generator error for {symbol}: {e}")
        
        # Entry Point Manager
        if 'entry_point_manager' in self.systems:
            try:
                manager_analysis = self.systems['entry_point_manager'].analyze_entry_point(symbol)
                signals['entry_point_manager'] = manager_analysis
            except Exception as e:
                logger.warning(f"⚠️ Entry point manager error for {symbol}: {e}")
        
        return signals
    
    async def _analyze_and_select_best_entry(self, symbol: str, all_signals: Dict[str, Any],
                                           current_price: float, account_balance: float) -> Optional[BestEntryPoint]:
        """Analyze all signals and select the absolute best entry point"""
        
        if not all_signals:
            return None
        
        # Calculate system consensus
        system_scores = self._calculate_system_consensus(all_signals)
        if not system_scores:
            return None
        
        # Find the direction with highest consensus
        best_direction = max(system_scores.keys(), key=lambda d: system_scores[d]['consensus'])
        best_consensus = system_scores[best_direction]
        
        # Calculate comprehensive scores
        confidence_score = best_consensus['consensus']
        technical_score = best_consensus.get('technical_score', 0.7)
        timing_score = best_consensus.get('timing_score', 0.7)
        
        # Calculate optimal levels
        entry_price, stop_loss, take_profit = self._calculate_optimal_levels(
            symbol, best_direction, current_price, all_signals
        )
        
        # Calculate risk metrics
        risk_reward_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        
        if risk_reward_ratio < self.config['min_risk_reward_ratio']:
            logger.debug(f"🔍 Entry for {symbol} rejected: R/R {risk_reward_ratio:.2f} < {self.config['min_risk_reward_ratio']}")
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(
            symbol, entry_price, stop_loss, account_balance
        )
        
        # Determine entry quality
        overall_quality = self._determine_entry_quality(confidence_score)
        
        # Create best entry point
        best_entry = BestEntryPoint(
            symbol=symbol,
            direction=best_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            overall_quality=overall_quality,
            confidence_score=confidence_score,
            risk_reward_ratio=risk_reward_ratio,
            technical_score=technical_score,
            fundamental_score=0.7,  # Placeholder
            sentiment_score=0.7,    # Placeholder
            timing_score=timing_score,
            system_consensus=best_consensus['system_scores'],
            consensus_strength=best_consensus['consensus'],
            position_size=position_size,
            expected_profit=position_size * abs(take_profit - entry_price) / entry_price,
            max_drawdown=position_size * abs(entry_price - stop_loss) / entry_price,
            win_probability=min(confidence_score * 1.2, 0.95),
            timeframe='1h',
            created_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=self.config['entry_validity_hours']),
            reasons=best_consensus.get('reasons', ['Multi-system consensus'])
        )
        
        return best_entry
    
    def _calculate_system_consensus(self, all_signals: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate consensus across all entry systems"""
        direction_scores = {'buy': {'scores': [], 'system_scores': {}}, 
                          'sell': {'scores': [], 'system_scores': {}}}
        
        for system_name, signals in all_signals.items():
            if not signals:
                continue
            
            weight = self._get_system_weight(system_name)
            
            # Process different signal formats
            if isinstance(signals, list):
                for signal in signals:
                    if hasattr(signal, 'direction') and hasattr(signal, 'confidence_score'):
                        direction = signal.direction
                        score = signal.confidence_score * weight
                        direction_scores[direction]['scores'].append(score)
                        direction_scores[direction]['system_scores'][system_name] = score
            
            elif isinstance(signals, dict):
                # Handle dict format from entry point manager
                if 'recommendation' in signals:
                    rec = signals['recommendation']
                    confidence = signals.get('confidence', 0.5)
                    
                    if rec in ['BUY', 'STRONG_BUY']:
                        score = confidence * weight
                        direction_scores['buy']['scores'].append(score)
                        direction_scores['buy']['system_scores'][system_name] = score
                    elif rec in ['SELL', 'STRONG_SELL']:
                        score = confidence * weight
                        direction_scores['sell']['scores'].append(score)
                        direction_scores['sell']['system_scores'][system_name] = score
        
        # Calculate final consensus scores
        result = {}
        for direction, data in direction_scores.items():
            if data['scores']:
                consensus = np.mean(data['scores'])
                if consensus >= 0.5:  # Only include directions with positive consensus
                    result[direction] = {
                        'consensus': consensus,
                        'system_scores': data['system_scores'],
                        'technical_score': consensus,  # Simplified
                        'timing_score': consensus,     # Simplified
                        'reasons': [f'{len(data["scores"])} systems agree on {direction}']
                    }
        
        return result
    
    def _get_system_weight(self, system_name: str) -> float:
        """Get weight for each entry system"""
        weights = {
            'optimized_entry': self.config['optimized_entry_weight'],
            'enhanced_optimizer': self.config['enhanced_optimizer_weight'],
            'signal_generator': self.config['signal_generator_weight'],
            'entry_point_manager': self.config['entry_point_manager_weight']
        }
        return weights.get(system_name, 0.1)
    
    def _calculate_optimal_levels(self, symbol: str, direction: str, current_price: float,
                                all_signals: Dict[str, Any]) -> Tuple[float, float, float]:
        """Calculate optimal entry, stop loss, and take profit levels"""
        
        # Collect all suggested levels
        entry_prices = []
        stop_losses = []
        take_profits = []
        
        for system_name, signals in all_signals.items():
            if isinstance(signals, list):
                for signal in signals:
                    if (hasattr(signal, 'direction') and signal.direction == direction and
                        hasattr(signal, 'entry_price')):
                        entry_prices.append(signal.entry_price)
                        if hasattr(signal, 'stop_loss'):
                            stop_losses.append(signal.stop_loss)
                        if hasattr(signal, 'take_profit'):
                            take_profits.append(signal.take_profit)
        
        # Calculate optimal levels (weighted average)
        if entry_prices:
            entry_price = np.mean(entry_prices)
        else:
            entry_price = current_price
        
        if stop_losses:
            stop_loss = np.mean(stop_losses)
        else:
            # Default stop loss (2% from entry)
            if direction == 'buy':
                stop_loss = entry_price * 0.98
            else:
                stop_loss = entry_price * 1.02
        
        if take_profits:
            take_profit = np.mean(take_profits)
        else:
            # Default take profit (3:1 R/R)
            risk = abs(entry_price - stop_loss)
            if direction == 'buy':
                take_profit = entry_price + (risk * 3)
            else:
                take_profit = entry_price - (risk * 3)
        
        return entry_price, stop_loss, take_profit
    
    def _calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float, account_balance: float) -> float:
        """Calculate optimal position size"""
        risk_per_trade = 0.01  # 1% risk per trade
        max_loss = account_balance * risk_per_trade
        
        price_risk = abs(entry_price - stop_loss)
        if price_risk > 0:
            position_size = max_loss / price_risk
        else:
            position_size = account_balance * 0.1  # 10% fallback
        
        return min(position_size, account_balance * 0.2)  # Max 20% of balance
    
    def _determine_entry_quality(self, confidence_score: float) -> EntryQuality:
        """Determine entry quality based on confidence score"""
        if confidence_score >= self.config['premium_threshold']:
            return EntryQuality.PREMIUM
        elif confidence_score >= self.config['excellent_threshold']:
            return EntryQuality.EXCELLENT
        elif confidence_score >= self.config['good_threshold']:
            return EntryQuality.GOOD
        elif confidence_score >= self.config['fair_threshold']:
            return EntryQuality.FAIR
        else:
            return EntryQuality.POOR
    
    def _validate_entry_quality(self, entry: BestEntryPoint) -> bool:
        """Validate that entry meets quality requirements"""
        
        # Check minimum confidence
        if entry.confidence_score < self.config['min_overall_confidence']:
            logger.debug(f"🔍 Entry rejected: Low confidence {entry.confidence_score:.3f}")
            return False
        
        # Check minimum consensus
        if entry.consensus_strength < self.config['min_consensus_strength']:
            logger.debug(f"🔍 Entry rejected: Low consensus {entry.consensus_strength:.3f}")
            return False
        
        # Check risk/reward ratio
        if entry.risk_reward_ratio < self.config['min_risk_reward_ratio']:
            logger.debug(f"🔍 Entry rejected: Low R/R {entry.risk_reward_ratio:.2f}")
            return False
        
        # Check win probability
        if entry.win_probability < self.config['min_win_probability']:
            logger.debug(f"🔍 Entry rejected: Low win probability {entry.win_probability:.3f}")
            return False
        
        # Check entry quality
        if entry.overall_quality == EntryQuality.POOR:
            logger.debug(f"🔍 Entry rejected: Poor quality")
            return False
        
        return True
    
    def _get_current_price(self, market_data: Dict[str, pd.DataFrame]) -> Optional[float]:
        """Get current price from market data"""
        for timeframe, df in market_data.items():
            if not df.empty and 'close' in df.columns:
                return float(df['close'].iloc[-1])
        return None
    
    def get_entry_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive report on entry point quality"""
        total_entries = len(self.current_best_entries)
        
        if total_entries == 0:
            return {
                'total_entries': 0,
                'quality_distribution': {},
                'average_confidence': 0,
                'average_rr_ratio': 0,
                'systems_active': len(self.systems)
            }
        
        quality_counts = {}
        confidences = []
        rr_ratios = []
        
        for entry in self.current_best_entries.values():
            quality_counts[entry.overall_quality.name] = quality_counts.get(entry.overall_quality.name, 0) + 1
            confidences.append(entry.confidence_score)
            rr_ratios.append(entry.risk_reward_ratio)
        
        return {
            'total_entries': total_entries,
            'quality_distribution': quality_counts,
            'average_confidence': np.mean(confidences) if confidences else 0,
            'average_rr_ratio': np.mean(rr_ratios) if rr_ratios else 0,
            'systems_active': len(self.systems),
            'premium_entries': quality_counts.get('PREMIUM', 0),
            'excellent_entries': quality_counts.get('EXCELLENT', 0)
        }

def get_best_entry_point_system() -> BestEntryPointSystem:
    """Factory function to get the best entry point system"""
    return BestEntryPointSystem()