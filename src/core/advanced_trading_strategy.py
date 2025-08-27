#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Advanced Trading Strategy Engine
Coin-specific configurations and advanced signal generation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import asyncio
import redis
import aiohttp

logger = logging.getLogger(__name__)

class CoinCategory(Enum):
    """Categorization of coins for optimized strategies"""
    MAJOR_CRYPTO = "major"      # BTC, ETH
    ALTCOINS = "altcoins"       # ADA, LINK, DOT, etc.
    MEME_COINS = "meme"         # DOGE, SHIB, etc.
    DEFI_TOKENS = "defi"        # UNI, SUSHI, COMP, etc.
    LAYER1 = "layer1"           # SOL, AVAX, NEAR, etc.
    STABLECOINS = "stable"      # USDT, USDC (rarely traded)

class TradingSignal(Enum):
    """Enhanced trading signals with confidence levels"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class AdvancedTradingStrategy:
    """
    Advanced trading strategy with coin-specific optimizations
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or self._create_redis_client()
        
        # Coin-specific configurations
        self.coin_configs = self._load_coin_configurations()
        
        # Strategy parameters by coin category
        self.strategy_params = {
            CoinCategory.MAJOR_CRYPTO: {
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'sma_fast': 12,
                'sma_slow': 26,
                'volatility_threshold': 0.02,
                'volume_multiplier': 1.5,
                'leverage_multiplier': 1.0,
                'risk_per_trade': 0.015  # 1.5% for major cryptos
            },
            CoinCategory.ALTCOINS: {
                'rsi_overbought': 75,
                'rsi_oversold': 25,
                'sma_fast': 10,
                'sma_slow': 20,
                'volatility_threshold': 0.03,
                'volume_multiplier': 2.0,
                'leverage_multiplier': 0.8,
                'risk_per_trade': 0.01  # 1% for altcoins (more volatile)
            },
            CoinCategory.MEME_COINS: {
                'rsi_overbought': 80,
                'rsi_oversold': 20,
                'sma_fast': 8,
                'sma_slow': 16,
                'volatility_threshold': 0.05,
                'volume_multiplier': 3.0,
                'leverage_multiplier': 0.6,
                'risk_per_trade': 0.005  # 0.5% for meme coins (extremely volatile)
            },
            CoinCategory.DEFI_TOKENS: {
                'rsi_overbought': 72,
                'rsi_oversold': 28,
                'sma_fast': 14,
                'sma_slow': 28,
                'volatility_threshold': 0.025,
                'volume_multiplier': 1.8,
                'leverage_multiplier': 0.9,
                'risk_per_trade': 0.012  # 1.2% for DeFi tokens
            },
            CoinCategory.LAYER1: {
                'rsi_overbought': 73,
                'rsi_oversold': 27,
                'sma_fast': 13,
                'sma_slow': 24,
                'volatility_threshold': 0.028,
                'volume_multiplier': 1.7,
                'leverage_multiplier': 0.85,
                'risk_per_trade': 0.013  # 1.3% for Layer 1 tokens
            }
        }
        
        logger.info("🎯 Advanced Trading Strategy initialized with coin-specific configurations")
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client"""
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        return redis.Redis.from_url(redis_url, decode_responses=True)
    
    def _load_coin_configurations(self) -> Dict[str, Dict]:
        """Load coin-specific configurations"""
        return {
            # Major Cryptocurrencies - Most stable, lower risk
            'BTC/USDT:USDT': {
                'category': CoinCategory.MAJOR_CRYPTO,
                'min_volume': 50000000,  # $50M daily volume minimum
                'max_spread': 0.001,     # 0.1% max spread
                'optimal_timeframes': ['1h', '4h', '1d'],
                'leverage_preference': 20,
                'volatility_adjustment': 1.0
            },
            'ETH/USDT:USDT': {
                'category': CoinCategory.MAJOR_CRYPTO,
                'min_volume': 30000000,  # $30M daily volume minimum
                'max_spread': 0.0012,    # 0.12% max spread
                'optimal_timeframes': ['1h', '4h'],
                'leverage_preference': 25,
                'volatility_adjustment': 1.1
            },
            
            # Altcoins - Moderate risk
            'ADA/USDT:USDT': {
                'category': CoinCategory.ALTCOINS,
                'min_volume': 5000000,   # $5M daily volume
                'max_spread': 0.002,     # 0.2% max spread
                'optimal_timeframes': ['30m', '1h'],
                'leverage_preference': 15,
                'volatility_adjustment': 1.3
            },
            'LINK/USDT:USDT': {
                'category': CoinCategory.ALTCOINS,
                'min_volume': 3000000,
                'max_spread': 0.0025,
                'optimal_timeframes': ['15m', '30m', '1h'],
                'leverage_preference': 18,
                'volatility_adjustment': 1.25
            },
            'DOT/USDT:USDT': {
                'category': CoinCategory.ALTCOINS,
                'min_volume': 2000000,
                'max_spread': 0.003,
                'optimal_timeframes': ['30m', '1h'],
                'leverage_preference': 16,
                'volatility_adjustment': 1.4
            },
            
            # DeFi Tokens - Higher risk but good opportunities
            'UNI/USDT:USDT': {
                'category': CoinCategory.DEFI_TOKENS,
                'min_volume': 4000000,
                'max_spread': 0.0025,
                'optimal_timeframes': ['15m', '30m'],
                'leverage_preference': 12,
                'volatility_adjustment': 1.5
            },
            'SUSHI/USDT:USDT': {
                'category': CoinCategory.DEFI_TOKENS,
                'min_volume': 1000000,
                'max_spread': 0.004,
                'optimal_timeframes': ['15m', '30m'],
                'leverage_preference': 10,
                'volatility_adjustment': 1.6
            },
            
            # Layer 1 tokens - Growing sector
            'SOL/USDT:USDT': {
                'category': CoinCategory.LAYER1,
                'min_volume': 8000000,
                'max_spread': 0.002,
                'optimal_timeframes': ['15m', '30m', '1h'],
                'leverage_preference': 20,
                'volatility_adjustment': 1.35
            },
            'AVAX/USDT:USDT': {
                'category': CoinCategory.LAYER1,
                'min_volume': 3000000,
                'max_spread': 0.003,
                'optimal_timeframes': ['30m', '1h'],
                'leverage_preference': 15,
                'volatility_adjustment': 1.45
            },
            
            # Meme Coins - Highest risk, highest reward potential
            'DOGE/USDT:USDT': {
                'category': CoinCategory.MEME_COINS,
                'min_volume': 2000000,
                'max_spread': 0.005,
                'optimal_timeframes': ['5m', '15m'],
                'leverage_preference': 8,
                'volatility_adjustment': 2.0
            },
            'SHIB/USDT:USDT': {
                'category': CoinCategory.MEME_COINS,
                'min_volume': 1000000,
                'max_spread': 0.006,
                'optimal_timeframes': ['5m', '15m'],
                'leverage_preference': 6,
                'volatility_adjustment': 2.2
            }
        }
    
    def categorize_coin(self, symbol: str) -> CoinCategory:
        """Categorize a coin based on its symbol"""
        if symbol in self.coin_configs:
            return self.coin_configs[symbol]['category']
        
        # Auto-categorization for unknown coins
        base = symbol.split('/')[0] if '/' in symbol else symbol
        
        if base in ['BTC', 'ETH']:
            return CoinCategory.MAJOR_CRYPTO
        elif base in ['DOGE', 'SHIB', 'PEPE', 'FLOKI']:
            return CoinCategory.MEME_COINS
        elif base in ['UNI', 'SUSHI', 'COMP', 'AAVE', 'MKR']:
            return CoinCategory.DEFI_TOKENS
        elif base in ['SOL', 'AVAX', 'NEAR', 'ALGO', 'ATOM']:
            return CoinCategory.LAYER1
        else:
            return CoinCategory.ALTCOINS
    
    def get_strategy_params(self, symbol: str) -> Dict:
        """Get strategy parameters for a specific coin"""
        category = self.categorize_coin(symbol)
        return self.strategy_params.get(category, self.strategy_params[CoinCategory.ALTCOINS])
    
    def calculate_technical_indicators(self, ohlcv_data: List[List], params: Dict) -> Dict:
        """Calculate technical indicators with coin-specific parameters"""
        try:
            if len(ohlcv_data) < max(params['sma_slow'], params.get('rsi_period', 14)):
                return {}
            
            # Convert to DataFrame for easier calculation
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Simple Moving Averages
            sma_fast = df['close'].rolling(window=params['sma_fast']).mean().iloc[-1]
            sma_slow = df['close'].rolling(window=params['sma_slow']).mean().iloc[-1]
            
            # RSI calculation
            rsi = self._calculate_rsi(df['close'], period=14)
            
            # Volume analysis
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility (ATR approximation)
            high_low = df['high'] - df['low']
            volatility = high_low.rolling(window=14).mean().iloc[-1] / df['close'].iloc[-1]
            
            # Price momentum
            price_change_1h = (df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4]  # 4 periods back
            price_change_4h = (df['close'].iloc[-1] - df['close'].iloc[-16]) / df['close'].iloc[-16] if len(df) >= 16 else 0
            
            return {
                'sma_fast': float(sma_fast),
                'sma_slow': float(sma_slow),
                'rsi': float(rsi),
                'volume_ratio': float(volume_ratio),
                'volatility': float(volatility),
                'price_change_1h': float(price_change_1h),
                'price_change_4h': float(price_change_4h),
                'current_price': float(df['close'].iloc[-1])
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0  # Neutral RSI if calculation fails
    
    def generate_advanced_signal(self, symbol: str, market_data: Dict, 
                               historical_data: List[List]) -> Dict:
        """Generate advanced trading signal with multi-factor analysis"""
        try:
            # Get coin-specific parameters
            params = self.get_strategy_params(symbol)
            category = self.categorize_coin(symbol)
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(historical_data, params)
            if not indicators:
                return self._create_signal(symbol, TradingSignal.HOLD, 0, "Insufficient data")
            
            # Extract current market data
            ticker = market_data.get('ticker', {})
            current_price = indicators['current_price']
            
            # Multi-factor scoring system
            scores = {
                'trend_score': self._calculate_trend_score(indicators, params),
                'momentum_score': self._calculate_momentum_score(indicators, params),
                'volume_score': self._calculate_volume_score(indicators, params),
                'volatility_score': self._calculate_volatility_score(indicators, params, category),
                'rsi_score': self._calculate_rsi_score(indicators, params)
            }
            
            # Calculate composite score
            composite_score = (
                scores['trend_score'] * 0.25 +      # Trend direction
                scores['momentum_score'] * 0.20 +   # Price momentum
                scores['volume_score'] * 0.20 +     # Volume confirmation
                scores['volatility_score'] * 0.20 + # Volatility assessment
                scores['rsi_score'] * 0.15          # RSI confirmation
            )
            
            # Generate signal based on composite score and category-specific thresholds
            signal_info = self._determine_signal_from_score(composite_score, category, scores)
            
            # Add coin-specific adjustments
            confidence = self._adjust_confidence_for_coin(
                signal_info['confidence'], symbol, category, indicators
            )
            
            return self._create_signal(
                symbol=symbol,
                signal_type=signal_info['signal'],
                confidence=confidence,
                reason=signal_info['reason'],
                scores=scores,
                indicators=indicators,
                category=category.value
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating advanced signal for {symbol}: {e}")
            return self._create_signal(symbol, TradingSignal.HOLD, 0, f"Error: {e}")
    
    def _calculate_trend_score(self, indicators: Dict, params: Dict) -> float:
        """Calculate trend strength score (0-100)"""
        try:
            sma_fast = indicators.get('sma_fast', 0)
            sma_slow = indicators.get('sma_slow', 0)
            current_price = indicators.get('current_price', 0)
            
            if sma_fast == 0 or sma_slow == 0:
                return 50  # Neutral
            
            # Price vs SMA analysis
            price_vs_fast = (current_price - sma_fast) / sma_fast
            sma_trend = (sma_fast - sma_slow) / sma_slow
            
            # Combine signals
            if price_vs_fast > 0 and sma_trend > 0:
                # Bullish trend
                trend_score = 50 + min(50, (price_vs_fast + sma_trend) * 500)
            elif price_vs_fast < 0 and sma_trend < 0:
                # Bearish trend
                trend_score = 50 - min(50, abs(price_vs_fast + sma_trend) * 500)
            else:
                # Mixed signals
                trend_score = 50
            
            return max(0, min(100, trend_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating trend score: {e}")
            return 50
    
    def _calculate_momentum_score(self, indicators: Dict, params: Dict) -> float:
        """Calculate momentum score (0-100)"""
        try:
            price_change_1h = indicators.get('price_change_1h', 0)
            price_change_4h = indicators.get('price_change_4h', 0)
            
            # Weight recent momentum more heavily
            momentum = price_change_1h * 0.7 + price_change_4h * 0.3
            
            # Convert to 0-100 score
            momentum_score = 50 + momentum * 1000  # Scale momentum
            
            return max(0, min(100, momentum_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating momentum score: {e}")
            return 50
    
    def _calculate_volume_score(self, indicators: Dict, params: Dict) -> float:
        """Calculate volume confirmation score (0-100)"""
        try:
            volume_ratio = indicators.get('volume_ratio', 1)
            volume_multiplier = params.get('volume_multiplier', 1.5)
            
            if volume_ratio >= volume_multiplier:
                # High volume confirmation
                volume_score = 70 + min(30, (volume_ratio - volume_multiplier) * 15)
            elif volume_ratio >= 1.0:
                # Normal volume
                volume_score = 40 + (volume_ratio * 30)
            else:
                # Low volume warning
                volume_score = volume_ratio * 40
            
            return max(0, min(100, volume_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating volume score: {e}")
            return 50
    
    def _calculate_volatility_score(self, indicators: Dict, params: Dict, category: CoinCategory) -> float:
        """Calculate volatility appropriateness score (0-100)"""
        try:
            volatility = indicators.get('volatility', 0)
            threshold = params.get('volatility_threshold', 0.02)
            
            # Different volatility preferences by category
            if category == CoinCategory.MEME_COINS:
                # Meme coins: high volatility is good for trading
                if volatility > threshold:
                    volatility_score = 70 + min(30, (volatility - threshold) * 600)
                else:
                    volatility_score = volatility / threshold * 70
            else:
                # Other categories: moderate volatility preferred
                if volatility < threshold * 0.5:
                    # Too low volatility
                    volatility_score = volatility / (threshold * 0.5) * 30
                elif volatility > threshold * 2:
                    # Too high volatility
                    volatility_score = 100 - min(50, (volatility - threshold * 2) * 1000)
                else:
                    # Good volatility range
                    volatility_score = 60 + 40 * (1 - abs(volatility - threshold) / threshold)
            
            return max(0, min(100, volatility_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility score: {e}")
            return 50
    
    def _calculate_rsi_score(self, indicators: Dict, params: Dict) -> float:
        """Calculate RSI-based score (0-100)"""
        try:
            rsi = indicators.get('rsi', 50)
            overbought = params.get('rsi_overbought', 70)
            oversold = params.get('rsi_oversold', 30)
            
            if rsi <= oversold:
                # Strong buy signal from RSI
                rsi_score = 80 + (oversold - rsi) * 2
            elif rsi >= overbought:
                # Strong sell signal from RSI
                rsi_score = 20 - (rsi - overbought) * 2
            elif 40 <= rsi <= 60:
                # Neutral zone
                rsi_score = 50
            elif rsi < 50:
                # Moderate buy zone
                rsi_score = 50 + (50 - rsi) * 0.6
            else:
                # Moderate sell zone
                rsi_score = 50 - (rsi - 50) * 0.6
            
            return max(0, min(100, rsi_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating RSI score: {e}")
            return 50
    
    def _determine_signal_from_score(self, composite_score: float, 
                                   category: CoinCategory, scores: Dict) -> Dict:
        """Determine trading signal from composite score"""
        try:
            # Category-specific thresholds
            thresholds = {
                CoinCategory.MAJOR_CRYPTO: {'strong': 75, 'weak': 60, 'hold_min': 40, 'hold_max': 60},
                CoinCategory.ALTCOINS: {'strong': 70, 'weak': 55, 'hold_min': 35, 'hold_max': 65},
                CoinCategory.MEME_COINS: {'strong': 80, 'weak': 65, 'hold_min': 30, 'hold_max': 70},
                CoinCategory.DEFI_TOKENS: {'strong': 72, 'weak': 58, 'hold_min': 38, 'hold_max': 62},
                CoinCategory.LAYER1: {'strong': 73, 'weak': 57, 'hold_min': 37, 'hold_max': 63}
            }
            
            thresh = thresholds.get(category, thresholds[CoinCategory.ALTCOINS])
            
            if composite_score >= thresh['strong']:
                return {
                    'signal': TradingSignal.STRONG_BUY if composite_score > 50 else TradingSignal.STRONG_SELL,
                    'confidence': min(95, composite_score),
                    'reason': f"Strong signal: composite score {composite_score:.1f} (threshold: {thresh['strong']})"
                }
            elif composite_score >= thresh['weak'] or composite_score <= (100 - thresh['weak']):
                if composite_score > 50:
                    signal_type = TradingSignal.BUY if composite_score > 55 else TradingSignal.WEAK_BUY
                else:
                    signal_type = TradingSignal.SELL if composite_score < 45 else TradingSignal.WEAK_SELL
                
                return {
                    'signal': signal_type,
                    'confidence': min(85, abs(composite_score - 50) * 2),
                    'reason': f"Moderate signal: composite score {composite_score:.1f}"
                }
            else:
                return {
                    'signal': TradingSignal.HOLD,
                    'confidence': 100 - abs(composite_score - 50) * 2,
                    'reason': f"Hold signal: composite score {composite_score:.1f} in neutral range"
                }
                
        except Exception as e:
            logger.error(f"❌ Error determining signal: {e}")
            return {
                'signal': TradingSignal.HOLD,
                'confidence': 0,
                'reason': f"Error: {e}"
            }
    
    def _adjust_confidence_for_coin(self, base_confidence: float, symbol: str, 
                                   category: CoinCategory, indicators: Dict) -> float:
        """Adjust confidence based on coin-specific factors"""
        try:
            config = self.coin_configs.get(symbol, {})
            
            # Volume adjustment
            volume_ratio = indicators.get('volume_ratio', 1)
            min_volume_met = volume_ratio >= 1.0  # At least average volume
            
            # Spread adjustment (if available in market data)
            spread_factor = 1.0  # Default if no spread data
            
            # Category-specific confidence adjustments
            category_multiplier = {
                CoinCategory.MAJOR_CRYPTO: 1.1,     # Boost confidence for major cryptos
                CoinCategory.ALTCOINS: 1.0,         # Standard confidence
                CoinCategory.MEME_COINS: 0.8,       # Reduce confidence due to higher risk
                CoinCategory.DEFI_TOKENS: 0.95,     # Slightly reduce due to DeFi risks
                CoinCategory.LAYER1: 1.05           # Slight boost for Layer 1 tokens
            }.get(category, 1.0)
            
            # Apply adjustments
            adjusted_confidence = base_confidence * category_multiplier
            
            if not min_volume_met:
                adjusted_confidence *= 0.7  # Reduce confidence for low volume
            
            return max(0, min(100, adjusted_confidence))
            
        except Exception as e:
            logger.error(f"❌ Error adjusting confidence: {e}")
            return base_confidence
    
    def _create_signal(self, symbol: str, signal_type: TradingSignal, 
                      confidence: float, reason: str, scores: Dict = None,
                      indicators: Dict = None, category: str = None) -> Dict:
        """Create standardized signal object"""
        return {
            'symbol': symbol,
            'signal': signal_type.value,
            'confidence': round(confidence, 2),
            'reason': reason,
            'category': category,
            'scores': scores or {},
            'indicators': indicators or {},
            'timestamp': datetime.now().isoformat(),
            'strategy': 'VIPER_Advanced_v2.0'
        }
    
    def get_optimal_position_size(self, symbol: str, balance: float, 
                                 signal_confidence: float) -> Dict:
        """Calculate optimal position size for a symbol"""
        try:
            params = self.get_strategy_params(symbol)
            category = self.categorize_coin(symbol)
            config = self.coin_configs.get(symbol, {})
            
            # Base risk from strategy parameters
            base_risk = params.get('risk_per_trade', 0.02)
            
            # Adjust risk based on confidence
            confidence_multiplier = signal_confidence / 100
            adjusted_risk = base_risk * confidence_multiplier
            
            # Apply leverage preference
            leverage = config.get('leverage_preference', 10)
            
            # Calculate position size
            risk_amount = balance * adjusted_risk
            position_value = risk_amount * leverage
            
            return {
                'risk_per_trade': round(adjusted_risk, 4),
                'position_value': round(position_value, 2),
                'recommended_leverage': leverage,
                'risk_amount': round(risk_amount, 2),
                'category': category.value,
                'confidence_factor': round(confidence_multiplier, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size for {symbol}: {e}")
            return {
                'risk_per_trade': 0.01,
                'position_value': balance * 0.01,
                'recommended_leverage': 5,
                'error': str(e)
            }
    
    def evaluate_market_conditions(self) -> Dict:
        """Evaluate overall market conditions for risk management"""
        try:
            # Get recent signals from Redis
            signal_keys = self.redis_client.keys("viper:signal:*")
            
            if not signal_keys:
                return {
                    'market_sentiment': 'neutral',
                    'signal_count': 0,
                    'risk_level': 'medium',
                    'recommendation': 'No recent signals available'
                }
            
            # Analyze recent signals
            buy_signals = 0
            sell_signals = 0
            total_confidence = 0
            
            for key in signal_keys[-50:]:  # Last 50 signals
                try:
                    signal_data = self.redis_client.get(key)
                    if signal_data:
                        signal = json.loads(signal_data)
                        signal_type = signal.get('signal', 'HOLD')
                        confidence = signal.get('confidence', 0)
                        
                        if 'BUY' in signal_type:
                            buy_signals += 1
                        elif 'SELL' in signal_type:
                            sell_signals += 1
                        
                        total_confidence += confidence
                except:
                    continue
            
            # Calculate market sentiment
            total_signals = buy_signals + sell_signals
            if total_signals > 0:
                buy_ratio = buy_signals / total_signals
                avg_confidence = total_confidence / len(signal_keys)
                
                if buy_ratio > 0.7:
                    sentiment = 'bullish'
                    risk_level = 'low' if avg_confidence > 70 else 'medium'
                elif buy_ratio < 0.3:
                    sentiment = 'bearish'
                    risk_level = 'low' if avg_confidence > 70 else 'medium'
                else:
                    sentiment = 'neutral'
                    risk_level = 'medium'
            else:
                sentiment = 'neutral'
                risk_level = 'medium'
            
            return {
                'market_sentiment': sentiment,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'total_signals': total_signals,
                'buy_ratio': round(buy_ratio if total_signals > 0 else 0.5, 2),
                'average_confidence': round(avg_confidence if total_signals > 0 else 0, 1),
                'risk_level': risk_level,
                'recommendation': self._get_market_recommendation(sentiment, risk_level, avg_confidence if total_signals > 0 else 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluating market conditions: {e}")
            return {
                'market_sentiment': 'unknown',
                'signal_count': 0,
                'risk_level': 'high',
                'error': str(e)
            }
    
    def _get_market_recommendation(self, sentiment: str, risk_level: str, avg_confidence: float) -> str:
        """Generate market recommendation based on conditions"""
        if sentiment == 'bullish' and risk_level == 'low' and avg_confidence > 75:
            return "🚀 Aggressive long positions recommended with high confidence"
        elif sentiment == 'bearish' and risk_level == 'low' and avg_confidence > 75:
            return "📉 Short positions recommended with high confidence"
        elif sentiment in ['bullish', 'bearish'] and risk_level == 'medium':
            return f"⚡ Moderate {sentiment} positions with standard risk management"
        elif risk_level == 'high':
            return "🛑 High risk detected - reduce position sizes or wait for better conditions"
        else:
            return "📊 Neutral market - focus on high-confidence individual signals"
    
    async def store_signal(self, signal: Dict) -> bool:
        """Store signal in Redis with expiration"""
        try:
            signal_key = f"viper:signal:{signal['symbol']}:{int(time.time())}"
            
            # Store with 1 hour expiration
            self.redis_client.setex(
                signal_key,
                3600,
                json.dumps(signal)
            )
            
            # Also store latest signal for this symbol
            self.redis_client.setex(
                f"viper:latest_signal:{signal['symbol']}",
                1800,  # 30 minutes
                json.dumps(signal)
            )
            
            logger.info(f"✅ Signal stored for {signal['symbol']}: {signal['signal']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing signal: {e}")
            return False
    
    def get_coin_configuration(self, symbol: str) -> Dict:
        """Get comprehensive configuration for a specific coin"""
        category = self.categorize_coin(symbol)
        params = self.get_strategy_params(symbol)
        config = self.coin_configs.get(symbol, {})
        
        return {
            'symbol': symbol,
            'category': category.value,
            'strategy_parameters': params,
            'coin_configuration': config,
            'risk_profile': self._get_risk_profile(category),
            'recommended_timeframes': config.get('optimal_timeframes', ['1h']),
            'leverage_recommendation': config.get('leverage_preference', 10)
        }
    
    def _get_risk_profile(self, category: CoinCategory) -> Dict:
        """Get risk profile for a coin category"""
        profiles = {
            CoinCategory.MAJOR_CRYPTO: {
                'risk_level': 'Low',
                'volatility': 'Moderate',
                'liquidity': 'Very High',
                'recommended_allocation': '40-60%'
            },
            CoinCategory.ALTCOINS: {
                'risk_level': 'Medium',
                'volatility': 'High',
                'liquidity': 'High',
                'recommended_allocation': '20-30%'
            },
            CoinCategory.MEME_COINS: {
                'risk_level': 'Very High',
                'volatility': 'Extreme',
                'liquidity': 'Variable',
                'recommended_allocation': '5-10%'
            },
            CoinCategory.DEFI_TOKENS: {
                'risk_level': 'High',
                'volatility': 'High',
                'liquidity': 'Medium',
                'recommended_allocation': '10-20%'
            },
            CoinCategory.LAYER1: {
                'risk_level': 'Medium-High',
                'volatility': 'High',
                'liquidity': 'Medium-High',
                'recommended_allocation': '15-25%'
            }
        }
        
        return profiles.get(category, profiles[CoinCategory.ALTCOINS])

# Global strategy instance for easy access
advanced_strategy = AdvancedTradingStrategy()

def get_strategy_instance() -> AdvancedTradingStrategy:
    """Get the global strategy instance"""
    return advanced_strategy