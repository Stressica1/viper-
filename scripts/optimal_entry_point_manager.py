#!/usr/bin/env python3
"""
# Target OPTIMAL ENTRY POINT MANAGER
Enhanced entry point optimization with mathematical validation and MCP integration

This script provides:
    pass
- Optimal entry point configurations
- Mathematical validation of entry signals
- MCP server integration for real-time optimization
- Performance monitoring and analytics
- Risk-adjusted entry point calculation
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our utilities"""
try:
    from utils.mathematical_validator import validate_array, safe_divide
    from config.optimal_mcp_config import OPTIMAL_MCP_CONFIG
except ImportError as e:
    # Create minimal fallback
    def validate_array(arr, name="array"):
        return {'is_valid': True, 'issues': [], 'statistics': {}}
    def safe_divide(num, den, default=0.0):
        return num / den if den != 0 else default
    OPTIMAL_MCP_CONFIG = {}

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - ENTRY_OPTIMIZER - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)

class OptimalEntryPointManager:
    """Enhanced entry point optimization manager""""""
    
    def __init__(self):
        self.project_root = project_root
        self.mcp_config = OPTIMAL_MCP_CONFIG
        
        # Optimal entry point configurations
        self.entry_configs = {
            # Technical Analysis Thresholds
            'rsi_oversold': float(os.getenv('RSI_OVERSOLD_THRESHOLD', '30')),
            'rsi_overbought': float(os.getenv('RSI_OVERBOUGHT_THRESHOLD', '70')),
            'rsi_neutral_low': float(os.getenv('RSI_NEUTRAL_LOW', '40')),
            'rsi_neutral_high': float(os.getenv('RSI_NEUTRAL_HIGH', '60')),
            
            # MACD Configuration
            'macd_signal_threshold': float(os.getenv('MACD_SIGNAL_THRESHOLD', '0.001')),
            'macd_histogram_threshold': float(os.getenv('MACD_HISTOGRAM_THRESHOLD', '0.0005')),
            
            # Bollinger Bands
            'bb_lower_threshold': float(os.getenv('BB_LOWER_THRESHOLD', '0.1')),  # 10% from lower band
            'bb_upper_threshold': float(os.getenv('BB_UPPER_THRESHOLD', '0.9')),  # 90% to upper band
            'bb_squeeze_threshold': float(os.getenv('BB_SQUEEZE_THRESHOLD', '0.02')),  # 2% width
            
            # Volume Analysis
            'volume_multiplier_min': float(os.getenv('VOLUME_MULTIPLIER_MIN', '1.5')),
            'volume_multiplier_strong': float(os.getenv('VOLUME_MULTIPLIER_STRONG', '2.0')),
            'volume_sma_period': int(os.getenv('VOLUME_SMA_PERIOD', '20')),
            
            # Trend Analysis
            'trend_strength_min': float(os.getenv('TREND_STRENGTH_MIN', '0.6')),
            'trend_strength_strong': float(os.getenv('TREND_STRENGTH_STRONG', '0.8')),
            'ema_crossover_threshold': float(os.getenv('EMA_CROSSOVER_THRESHOLD', '0.005')),
            
            # Risk Management - Enhanced with execution cost awareness
            'volatility_max': float(os.getenv('VOLATILITY_MAX_THRESHOLD', '0.05')),  # 5% daily volatility
            'volatility_optimal': float(os.getenv('VOLATILITY_OPTIMAL', '0.02')),  # 2% daily volatility
            'drawdown_max': float(os.getenv('DRAWDOWN_MAX_THRESHOLD', '0.15')),  # 15% max drawdown
            'max_execution_cost': float(os.getenv('MAX_EXECUTION_COST', '3.0')),  # $3 max execution cost
            'preferred_spread_bps': float(os.getenv('PREFERRED_SPREAD_BPS', '5')),  # 5bps preferred spread
            
            # Confidence Levels
            'confidence_min': float(os.getenv('CONFIDENCE_MIN_THRESHOLD', '0.6')),
            'confidence_high': float(os.getenv('CONFIDENCE_HIGH_THRESHOLD', '0.8')),
            'confidence_excellent': float(os.getenv('CONFIDENCE_EXCELLENT_THRESHOLD', '0.9')),
        }
        
        # Entry point scoring weights - optimized for execution cost awareness
        self.scoring_weights = {
            'technical_analysis': 0.25,    # Reduced from 0.35
            'trend_strength': 0.20,        # Reduced from 0.25
            'volume_confirmation': 0.20,   # Same
            'execution_cost': 0.25,        # NEW - major component for cost control
            'market_regime': 0.10          # Increased from 0.05
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals_generated': 0,
            'successful_entries': 0,
            'failed_entries': 0,
            'average_confidence': 0.0,
            'last_optimization_time': datetime.now(),
            'optimization_count': 0
        }
    
    def calculate_optimal_entry_score(self, market_data: Dict[str, Any]) -> Dict[str, Any]
        """Calculate optimal entry score with mathematical validation"""
        
        entry_result = {:
            'timestamp': datetime.now().isoformat(),
            'symbol': market_data.get('symbol', 'UNKNOWN'),
            'entry_score': 0.0,
            'confidence': 0.0,
            'recommendation': 'HOLD',
            'component_scores': {},
            'risk_metrics': {},
            'validation_results': {},
            'optimization_suggestions': []
        }"""
        
        try:
            # Validate input data
            validation_results = self.validate_market_data(market_data)
            entry_result['validation_results'] = validation_results
            
            if not validation_results['is_valid']:
                entry_result['recommendation'] = 'HOLD'
                entry_result['optimization_suggestions'] = validation_results.get('recommendations', [])
                return entry_result
            
            # Calculate component scores
            entry_result['component_scores'] = {
                'technical_analysis': self.calculate_technical_analysis_score(market_data),
                'trend_strength': self.calculate_trend_strength_score(market_data),
                'volume_confirmation': self.calculate_volume_confirmation_score(market_data),
                'execution_cost': self.calculate_execution_cost_score(market_data),  # NEW component
                'market_regime': self.calculate_market_regime_score(market_data)
            }
            
            # Calculate weighted entry score
            entry_score = sum()
                entry_result['component_scores'][component] * self.scoring_weights[component]
                for component in self.scoring_weights:
(            )
            
            # Validate entry score
            entry_score = np.clip(entry_score, 0.0, 1.0)
            entry_result['entry_score'] = float(entry_score)
            
            # Calculate confidence based on score consistency and data quality
            confidence = self.calculate_entry_confidence(entry_result['component_scores'], validation_results)
            entry_result['confidence'] = float(confidence)
            
            # Generate recommendation
            entry_result['recommendation'] = self.generate_entry_recommendation(entry_score, confidence)
            
            # Calculate risk metrics
            entry_result['risk_metrics'] = self.calculate_risk_metrics(market_data, entry_score)
            
            # Generate optimization suggestions
            entry_result['optimization_suggestions'] = self.generate_optimization_suggestions(entry_result)
            
            # Update performance tracking
            self.update_performance_metrics(entry_result)
            
            logger.info(f"# Target Entry Score: {entry_score:.3f}, Confidence: {confidence:.3f}, Recommendation: {entry_result['recommendation']}")
            
        except Exception as e:
            logger.error(f"# X Error calculating entry score: {e}")
            entry_result['error'] = str(e)
            entry_result['recommendation'] = 'HOLD'
        
        return entry_result
    
    def validate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]
        """Validate market data for entry point calculation"""
        
        validation = {:
            'is_valid': True,
            'issues': [],
            'recommendations': [],
            'data_quality_score': 0.0
        }"""
        
        try:
            # Required fields
            required_fields = [
                'close', 'volume', 'rsi', 'macd', 'macd_signal', 
                'bb_upper', 'bb_lower', 'bb_middle', 'sma_20', 'ema_12', 'ema_26'
            ]
            
            missing_fields = [field for field in required_fields if field not in market_data]
            if missing_fields:
                validation['issues'].extend([f"Missing field: {field}" for field in missing_fields])
                validation['is_valid'] = False
            
            # Validate numeric fields
            for field in required_fields:
                if field in market_data:
                    value = market_data[field]
                    if not isinstance(value, (int, float)) or not np.isfinite(value):
                        validation['issues'].append(f"Invalid {field} value: {value}")
                        validation['is_valid'] = False
            
            # Validate price relationships
            if 'bb_lower' in market_data and 'bb_upper' in market_data and 'close' in market_data:
                bb_lower, bb_upper, close = market_data['bb_lower'], market_data['bb_upper'], market_data['close']
                if not (bb_lower <= close <= bb_upper * 1.1):  # Allow 10% overshoot
                    validation['issues'].append("Price outside reasonable Bollinger Band range")
            
            # Validate RSI range
            if 'rsi' in market_data:
                rsi = market_data['rsi']
                if not (0 <= rsi <= 100):
                    validation['issues'].append(f"RSI out of range: {rsi}")
                    validation['is_valid'] = False
            
            # Calculate data quality score
            quality_factors = []
            
            # Completeness
            completeness = (len(required_fields) - len(missing_fields)) / len(required_fields)
            quality_factors.append(completeness)
            
            # Validity
            invalid_count = len([issue for issue in validation['issues'] if 'Invalid' in issue])
            validity = max(0, 1 - invalid_count / max(len(required_fields), 1))
            quality_factors.append(validity)
            
            # Consistency (check for reasonable relationships)
            consistency = 1.0  # Start with perfect consistency
            if 'close' in market_data and 'sma_20' in market_data:
                price_sma_ratio = abs(market_data['close'] / market_data['sma_20'] - 1)
                if price_sma_ratio > 0.5:  # Price more than 50% from SMA
                    consistency *= 0.8
            
            quality_factors.append(consistency)
            
            validation['data_quality_score'] = np.mean(quality_factors)
            
            # Generate recommendations based on issues
            if validation['issues']:
                validation['recommendations'].append("Improve data quality before entry calculation")
            if validation['data_quality_score'] < 0.8:
                validation['recommendations'].append("Consider using alternative data sources")
        
        except Exception as e:
            validation['is_valid'] = False
            validation['issues'].append(f"Validation error: {e}")
        
        return validation
    
    def calculate_technical_analysis_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate technical analysis component score""""""
        try:
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            macd_signal = market_data.get('macd_signal', 0)
            bb_position = self.calculate_bb_position(market_data)
            
            scores = []
            
            # RSI Score (higher score for oversold/overbought conditions with mean reversion bias)
            if rsi <= self.entry_configs['rsi_oversold']:
                rsi_score = 0.9  # Strong buy signal when oversold
            elif rsi >= self.entry_configs['rsi_overbought']:
                rsi_score = 0.9  # Strong sell signal when overbought
            elif self.entry_configs['rsi_neutral_low'] <= rsi <= self.entry_configs['rsi_neutral_high']:
                rsi_score = 0.5  # Neutral
            else:
                rsi_score = 0.3  # Weak signal
            
            scores.append(rsi_score)
            
            # MACD Score (momentum confirmation)
            macd_diff = macd - macd_signal
            if abs(macd_diff) > self.entry_configs['macd_signal_threshold']:
                macd_score = min(0.9, abs(macd_diff) * 1000)  # Scale MACD difference
            else:
                macd_score = 0.4  # Weak momentum
            
            scores.append(macd_score)
            
            # Bollinger Band Score (mean reversion)
            if bb_position <= self.entry_configs['bb_lower_threshold']:
                bb_score = 0.9  # Near lower band - potential bounce
            elif bb_position >= self.entry_configs['bb_upper_threshold']:
                bb_score = 0.9  # Near upper band - potential reversal
            else:
                bb_score = max(0.2, 1 - abs(bb_position - 0.5) * 2)  # Distance from middle
            
            scores.append(bb_score)
            
            # Return average score
            return np.mean(scores)
            
        except Exception as e:
            logger.warning(f"# Warning Error in technical analysis score: {e}")
            return 0.5  # Neutral score on error
    
    def calculate_bb_position(self, market_data: Dict[str, Any]) -> float:
        """Calculate position within Bollinger Bands (0 = lower band, 1 = upper band)""""""
        try:
            close = market_data.get('close', 0)
            bb_lower = market_data.get('bb_lower', close)
            bb_upper = market_data.get('bb_upper', close)
            
            if bb_upper == bb_lower:
                return 0.5  # Middle position if no band spread
            
            position = (close - bb_lower) / (bb_upper - bb_lower)
            return np.clip(position, 0, 1)
            
        except Exception:
            return 0.5
    
    def calculate_trend_strength_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate trend strength component score""""""
        try:
            close = market_data.get('close', 0)
            sma_20 = market_data.get('sma_20', close)
            ema_12 = market_data.get('ema_12', close)
            ema_26 = market_data.get('ema_26', close)
            
            scores = []
            
            # Price vs SMA trend
            price_sma_ratio = safe_divide(close, sma_20, 1.0)
            if price_sma_ratio > 1.02:  # 2% above SMA
                trend_score = min(0.9, (price_sma_ratio - 1) * 10)
            elif price_sma_ratio < 0.98:  # 2% below SMA
                trend_score = min(0.9, (1 - price_sma_ratio) * 10)
            else:
                trend_score = 0.4  # Sideways trend
            
            scores.append(trend_score)
            
            # EMA crossover
            ema_ratio = safe_divide(ema_12, ema_26, 1.0)
            if abs(ema_ratio - 1) > self.entry_configs['ema_crossover_threshold']:
                crossover_score = min(0.9, abs(ema_ratio - 1) * 100)
            else:
                crossover_score = 0.3
            
            scores.append(crossover_score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.warning(f"# Warning Error in trend strength score: {e}")
            return 0.5
    
    def calculate_volume_confirmation_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume confirmation component score""""""
        try:
            volume = market_data.get('volume', 0)
            volume_sma = market_data.get('volume_sma', volume)
            
            if volume_sma == 0:
                return 0.3  # Low score for zero volume
            
            volume_ratio = volume / volume_sma
            
            if volume_ratio >= self.entry_configs['volume_multiplier_strong']:
                return 0.9  # Strong volume confirmation
            elif volume_ratio >= self.entry_configs['volume_multiplier_min']:
                return 0.7  # Good volume confirmation
            elif volume_ratio >= 0.8:
                return 0.5  # Adequate volume
            else:
                return 0.2  # Weak volume
                
        except Exception as e:
            logger.warning(f"# Warning Error in volume confirmation score: {e}")
            return 0.5
    
    def calculate_execution_cost_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate execution cost component score with position-size awareness""""""
        try:
            spread = market_data.get('spread', 0.001)  # Default 10bps if not provided
            volume = market_data.get('volume', 0)
            
            # Estimate execution cost for a typical position
            typical_position_usd = 5000  # $5k position for calculation
            
            # Spread cost (half spread for market orders)
            spread_cost = typical_position_usd * spread / 2
            
            # Market impact using square-root model
            market_impact_rate = 0.0001 * (typical_position_usd / max(volume, 100_000)) ** 0.5 if volume > 0 else 0.005
            market_impact_cost = typical_position_usd * market_impact_rate
            
            total_cost = spread_cost + market_impact_cost
            
            # Score based on execution cost thresholds
            if total_cost >= 5.0:
                return 0.1  # Very poor execution conditions
            elif total_cost >= 3.0:
                return 0.3  # Poor execution conditions (above $3 threshold)
            elif total_cost >= 2.0:
                return 0.5  # Moderate execution conditions
            elif total_cost >= 1.0:
                return 0.7  # Good execution conditions
            else:
                return 0.9  # Excellent execution conditions
            
        except Exception as e:
            logger.warning(f"# Warning Error in execution cost score: {e}")
            return 0.5  # Neutral score on error
    
    def calculate_market_regime_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate market regime component score""""""
        try:
            # This is a simplified market regime detection
            # In practice, this would use more sophisticated regime detection
            
            rsi = market_data.get('rsi', 50)
            volatility = market_data.get('volatility', 0.02)
            
            # Trending market (good for momentum strategies)
            if rsi < 30 or rsi > 70:
                if volatility < 0.03:  # Low volatility trending
                    return 0.8
                else:  # High volatility trending
                    return 0.6
            else:  # Range-bound market
                if volatility < 0.02:  # Low volatility range
                    return 0.7
                else:  # High volatility range
                    return 0.4
                    
        except Exception as e:
            logger.warning(f"# Warning Error in market regime score: {e}")
            return 0.5
    
    def calculate_entry_confidence(self, component_scores: Dict[str, float], )
(                                 validation_results: Dict[str, Any]) -> float:
                                     pass
        """Calculate overall confidence in entry signal""""""
        try:
            # Base confidence from component score consistency
            scores = list(component_scores.values())
            if len(scores) == 0:
                return 0.5
            
            mean_score = np.mean(scores)
            score_std = np.std(scores)
            
            # Higher consistency (lower std) means higher confidence
            consistency_factor = max(0.3, 1 - score_std * 2)
            
            # Data quality factor
            data_quality_factor = validation_results.get('data_quality_score', 0.8)
            
            # Signal strength factor (distance from neutral 0.5)
            signal_strength_factor = abs(mean_score - 0.5) * 2
            
            # Combine factors
            confidence = (consistency_factor * 0.4 + )
                         data_quality_factor * 0.3 + 
(                         signal_strength_factor * 0.3)
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"# Warning Error calculating confidence: {e}")
            return 0.5
    
    def generate_entry_recommendation(self, entry_score: float, confidence: float) -> str:
        """Generate trading recommendation based on entry score and confidence"""
        
        # Confidence-adjusted thresholds
        buy_threshold = 0.7 - (confidence - 0.5) * 0.1
        strong_buy_threshold = 0.85 - (confidence - 0.5) * 0.1
        sell_threshold = 0.3 + (confidence - 0.5) * 0.1
        strong_sell_threshold = 0.15 + (confidence - 0.5) * 0.1"""
        
        if confidence < self.entry_configs['confidence_min']:
            return 'HOLD'  # Low confidence, hold position
        elif entry_score >= strong_buy_threshold and confidence >= self.entry_configs['confidence_high']:
            return 'STRONG_BUY'
        elif entry_score >= buy_threshold:
            return 'BUY'
        elif entry_score <= strong_sell_threshold and confidence >= self.entry_configs['confidence_high']:
            return 'STRONG_SELL'
        elif entry_score <= sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def calculate_risk_metrics(self, market_data: Dict[str, Any], entry_score: float) -> Dict[str, Any]
        """Calculate risk metrics for the entry point"""
        
        risk_metrics = {:
            'volatility_risk': 'UNKNOWN',
            'liquidity_risk': 'UNKNOWN',
            'position_sizing_suggestion': 1.0,
            'stop_loss_suggestion': 0.02,  # 2% default
            'take_profit_suggestion': 0.06  # 6% default (3:1 ratio)
        }"""
        
        try:
            volatility = market_data.get('volatility', 0.02)
            volume = market_data.get('volume', 0)
            volume_sma = market_data.get('volume_sma', volume)
            
            # Volatility risk assessment
            if volatility <= self.entry_configs['volatility_optimal']:
                risk_metrics['volatility_risk'] = 'LOW'
                risk_metrics['position_sizing_suggestion'] = 1.0
            elif volatility <= self.entry_configs['volatility_max']:
                risk_metrics['volatility_risk'] = 'MEDIUM'
                risk_metrics['position_sizing_suggestion'] = 0.7
            else:
                risk_metrics['volatility_risk'] = 'HIGH'
                risk_metrics['position_sizing_suggestion'] = 0.4
            
            # Liquidity risk assessment
            volume_ratio = safe_divide(volume, volume_sma, 1.0)
            if volume_ratio >= 2.0:
                risk_metrics['liquidity_risk'] = 'LOW'
            elif volume_ratio >= 1.0:
                risk_metrics['liquidity_risk'] = 'MEDIUM'
            else:
                risk_metrics['liquidity_risk'] = 'HIGH'
                risk_metrics['position_sizing_suggestion'] *= 0.8  # Reduce position size
            
            # Adjust stop loss and take profit based on volatility
            risk_metrics['stop_loss_suggestion'] = max(0.01, volatility * 2)  # 2x daily volatility
            risk_metrics['take_profit_suggestion'] = risk_metrics['stop_loss_suggestion'] * 3  # 3:1 ratio
            
        except Exception as e:
            logger.warning(f"# Warning Error calculating risk metrics: {e}")
        
        return risk_metrics
    
    def generate_optimization_suggestions(self, entry_result: Dict[str, Any]) -> List[str]
        """Generate optimization suggestions based on entry analysis"""
        
        suggestions = []
        :"""
        try:
            component_scores = entry_result.get('component_scores', {})
            confidence = entry_result.get('confidence', 0.5)
            validation_results = entry_result.get('validation_results', {})
            
            # Low confidence suggestions
            if confidence < self.entry_configs['confidence_min']:
                suggestions.append("# Search Wait for higher confidence signals before entering")
                suggestions.append("# Chart Consider additional technical indicators for confirmation")
            
            # Component-specific suggestions
            if component_scores.get('technical_analysis', 0.5) < 0.4:
                suggestions.append("📈 Technical indicators not aligned - wait for better setup")
            
            if component_scores.get('volume_confirmation', 0.5) < 0.4:
                suggestions.append("# Chart Low volume confirmation - consider waiting for higher volume")
            
            if component_scores.get('risk_assessment', 0.5) < 0.4:
                suggestions.append("# Warning High volatility detected - reduce position size or wait")
            
            # Data quality suggestions
            data_quality = validation_results.get('data_quality_score', 1.0)
            if data_quality < 0.8:
                suggestions.append("# Tool Improve data quality for better entry point analysis")
            
            # Performance suggestions
            if self.performance_metrics['total_signals_generated'] > 10:
                success_rate = safe_divide()
                    self.performance_metrics['successful_entries'],
                    self.performance_metrics['total_signals_generated'],
                    0.0
(                )
                if success_rate < 0.6:
                    suggestions.append("# Chart Consider adjusting entry thresholds - success rate below 60%")
            
        except Exception as e:
            logger.warning(f"# Warning Error generating suggestions: {e}")
            suggestions.append("# Tool System optimization recommended")
        
        return suggestions
    
    def update_performance_metrics(self, entry_result: Dict[str, Any]):
        """Update performance tracking metrics""""""
        try:
            self.performance_metrics['total_signals_generated'] += 1
            self.performance_metrics['last_optimization_time'] = datetime.now()
            self.performance_metrics['optimization_count'] += 1
            
            # Update average confidence
            old_avg = self.performance_metrics['average_confidence']
            new_confidence = entry_result.get('confidence', 0.5)
            count = self.performance_metrics['total_signals_generated']
            
            self.performance_metrics['average_confidence'] = \
                (old_avg * (count - 1) + new_confidence) / count
        
        except Exception as e:
            logger.warning(f"# Warning Error updating performance metrics: {e}")
    
    def save_performance_report(self):
        """Save performance metrics to file""""""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.project_root / "reports" / f"entry_point_performance_{timestamp}.json"
            
            # Ensure reports directory exists
            report_file.parent.mkdir(exist_ok=True)
            
            performance_report = {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': self.performance_metrics,
                'entry_configs': self.entry_configs,
                'scoring_weights': self.scoring_weights
            }
            
            with open(report_file, 'w') as f:
                json.dump(performance_report, f, indent=2, default=str)
            
            logger.info(f"# Chart Performance report saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"# X Error saving performance report: {e}")

def main():
    """Run entry point optimization example"""
    print("# Target OPTIMAL ENTRY POINT MANAGER - EXAMPLE RUN")
    
    # Initialize manager
    manager = OptimalEntryPointManager()
    
    # Example market data
    example_data = {
        'symbol': 'BTC/USDT',
        'close': 50000.0,
        'volume': 1500000.0,
        'volume_sma': 1000000.0,
        'rsi': 35.0,  # Oversold
        'macd': 0.002,
        'macd_signal': 0.001,
        'bb_upper': 52000.0,
        'bb_lower': 48000.0,
        'bb_middle': 50000.0,
        'sma_20': 49500.0,
        'ema_12': 49800.0,
        'ema_26': 49600.0,
        'volatility': 0.025
    }
    
    # Calculate entry point
    result = manager.calculate_optimal_entry_score(example_data)
    
    # Display results
    print(f"# Target Entry Score: {result['entry_score']:.3f}")
    print(f"# Search Confidence: {result['confidence']:.3f}")
    print(f"# Chart Recommendation: {result['recommendation']}")
    for component, score in result['component_scores'].items():
    for metric, value in result['risk_metrics'].items():
    if result['optimization_suggestions']:
        for suggestion in result['optimization_suggestions']:
    # Save performance report
    manager.save_performance_report()
    

if __name__ == "__main__":
    main()