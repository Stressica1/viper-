#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Advanced Scoring Engine
Multi-factor scoring system for trading opportunities
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import redis
import asyncio

from .advanced_trading_strategy import CoinCategory, TradingSignal

logger = logging.getLogger(__name__)

class ScoreComponent(Enum):
    """Individual scoring components"""
    TECHNICAL = "technical"      # Technical indicators
    FUNDAMENTAL = "fundamental"  # Market fundamentals
    SENTIMENT = "sentiment"      # Market sentiment
    VOLUME = "volume"           # Volume analysis
    VOLATILITY = "volatility"   # Volatility assessment
    LIQUIDITY = "liquidity"     # Liquidity analysis
    MOMENTUM = "momentum"       # Price momentum
    CORRELATION = "correlation" # Cross-asset correlation

@dataclass
class ScoreBreakdown:
    """Detailed score breakdown"""
    symbol: str
    overall_score: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    volume_score: float
    volatility_score: float
    liquidity_score: float
    momentum_score: float
    correlation_score: float
    category: str
    timestamp: str

class VIPERScoringEngine:
    """
    Advanced multi-factor scoring engine for trading opportunities
    V.I.P.E.R = Volume, Indicators, Price, External, Risk
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or self._create_redis_client()
        
        # Scoring weights by coin category
        self.category_weights = {
            CoinCategory.MAJOR_CRYPTO: {
                ScoreComponent.TECHNICAL: 0.25,
                ScoreComponent.FUNDAMENTAL: 0.20,
                ScoreComponent.SENTIMENT: 0.15,
                ScoreComponent.VOLUME: 0.15,
                ScoreComponent.VOLATILITY: 0.10,
                ScoreComponent.LIQUIDITY: 0.10,
                ScoreComponent.MOMENTUM: 0.05
            },
            CoinCategory.ALTCOINS: {
                ScoreComponent.TECHNICAL: 0.30,
                ScoreComponent.FUNDAMENTAL: 0.15,
                ScoreComponent.SENTIMENT: 0.20,
                ScoreComponent.VOLUME: 0.15,
                ScoreComponent.VOLATILITY: 0.10,
                ScoreComponent.LIQUIDITY: 0.05,
                ScoreComponent.MOMENTUM: 0.05
            },
            CoinCategory.MEME_COINS: {
                ScoreComponent.TECHNICAL: 0.20,
                ScoreComponent.FUNDAMENTAL: 0.05,
                ScoreComponent.SENTIMENT: 0.35,  # Sentiment very important for meme coins
                ScoreComponent.VOLUME: 0.20,
                ScoreComponent.VOLATILITY: 0.15,
                ScoreComponent.LIQUIDITY: 0.03,
                ScoreComponent.MOMENTUM: 0.02
            },
            CoinCategory.DEFI_TOKENS: {
                ScoreComponent.TECHNICAL: 0.25,
                ScoreComponent.FUNDAMENTAL: 0.25,  # DeFi fundamentals matter
                ScoreComponent.SENTIMENT: 0.15,
                ScoreComponent.VOLUME: 0.15,
                ScoreComponent.VOLATILITY: 0.10,
                ScoreComponent.LIQUIDITY: 0.05,
                ScoreComponent.MOMENTUM: 0.05
            },
            CoinCategory.LAYER1: {
                ScoreComponent.TECHNICAL: 0.25,
                ScoreComponent.FUNDAMENTAL: 0.20,
                ScoreComponent.SENTIMENT: 0.18,
                ScoreComponent.VOLUME: 0.15,
                ScoreComponent.VOLATILITY: 0.12,
                ScoreComponent.LIQUIDITY: 0.05,
                ScoreComponent.MOMENTUM: 0.05
            }
        }
        
        # Historical data for correlation analysis
        self.market_history = {}
        
        logger.info("🎯 VIPER Scoring Engine initialized with multi-factor analysis")
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client"""
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        return redis.Redis.from_url(redis_url, decode_responses=True)
    
    def calculate_technical_score(self, indicators: Dict, symbol: str) -> Tuple[float, Dict]:
        """Calculate technical analysis score (0-100)"""
        try:
            scores = {}
            
            # RSI scoring (0-25 points)
            rsi = indicators.get('rsi', 50)
            if rsi <= 30:
                rsi_score = 20 + (30 - rsi) * 0.5  # Oversold bonus
            elif rsi >= 70:
                rsi_score = 20 + (rsi - 70) * 0.5  # Overbought bonus
            elif 40 <= rsi <= 60:
                rsi_score = 15  # Neutral zone
            else:
                rsi_score = 10
            scores['rsi'] = min(25, rsi_score)
            
            # Moving Average scoring (0-25 points)
            sma_fast = indicators.get('sma_fast', 0)
            sma_slow = indicators.get('sma_slow', 0)
            current_price = indicators.get('current_price', 0)
            
            if sma_fast > 0 and sma_slow > 0 and current_price > 0:
                if current_price > sma_fast > sma_slow:
                    ma_score = 25  # Strong bullish
                elif current_price < sma_fast < sma_slow:
                    ma_score = 25  # Strong bearish (tradeable)
                elif current_price > sma_fast:
                    ma_score = 15  # Moderate bullish
                elif current_price < sma_fast:
                    ma_score = 15  # Moderate bearish
                else:
                    ma_score = 5   # Choppy
            else:
                ma_score = 0
            scores['moving_averages'] = ma_score
            
            # Price momentum scoring (0-25 points)
            momentum_1h = indicators.get('price_change_1h', 0)
            momentum_4h = indicators.get('price_change_4h', 0)
            
            # Strong momentum in either direction is good for trading
            combined_momentum = abs(momentum_1h) * 0.6 + abs(momentum_4h) * 0.4
            momentum_score = min(25, combined_momentum * 500)  # Scale appropriately
            scores['momentum'] = momentum_score
            
            # Volatility scoring (0-25 points)
            volatility = indicators.get('volatility', 0)
            if 0.01 <= volatility <= 0.05:  # Sweet spot for volatility
                vol_score = 25
            elif volatility < 0.01:
                vol_score = volatility * 1000  # Scale up low volatility
            elif volatility > 0.1:
                vol_score = max(5, 25 - (volatility - 0.05) * 200)  # Penalize extreme volatility
            else:
                vol_score = 20
            scores['volatility'] = vol_score
            
            # Total technical score
            total_score = sum(scores.values())
            
            return min(100, total_score), scores
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical score for {symbol}: {e}")
            return 0, {}
    
    def calculate_fundamental_score(self, symbol: str, market_data: Dict) -> Tuple[float, Dict]:
        """Calculate fundamental analysis score (0-100)"""
        try:
            scores = {}
            ticker = market_data.get('ticker', {})
            
            # Market cap proxy (0-30 points) - based on symbol and volume
            base_symbol = symbol.split('/')[0]
            volume_24h = ticker.get('baseVolume', 0)
            
            # Major cryptos get higher fundamental scores
            if base_symbol in ['BTC', 'ETH']:
                mcap_score = 30
            elif base_symbol in ['SOL', 'BNB', 'ADA']:
                mcap_score = 25
            elif volume_24h > 10000000:  # High volume = likely good fundamentals
                mcap_score = 20
            elif volume_24h > 1000000:
                mcap_score = 15
            else:
                mcap_score = 10
            scores['market_cap_proxy'] = mcap_score
            
            # Ecosystem strength (0-25 points)
            ecosystem_scores = {
                'BTC': 25, 'ETH': 25, 'SOL': 20, 'BNB': 20,
                'ADA': 18, 'DOT': 18, 'AVAX': 18, 'MATIC': 17,
                'LINK': 16, 'UNI': 15, 'SUSHI': 12, 'COMP': 12,
                'DOGE': 8, 'SHIB': 6  # Lower for meme coins
            }
            ecosystem_score = ecosystem_scores.get(base_symbol, 10)
            scores['ecosystem'] = ecosystem_score
            
            # Trading activity (0-25 points)
            if volume_24h > 50000000:
                activity_score = 25
            elif volume_24h > 10000000:
                activity_score = 20
            elif volume_24h > 1000000:
                activity_score = 15
            elif volume_24h > 100000:
                activity_score = 10
            else:
                activity_score = 5
            scores['activity'] = activity_score
            
            # Price stability over time (0-20 points)
            price_change = abs(ticker.get('percentage', 0))
            if 2 <= price_change <= 8:  # Good volatility for trading
                stability_score = 20
            elif 1 <= price_change <= 2:
                stability_score = 15
            elif 8 <= price_change <= 15:
                stability_score = 15
            else:
                stability_score = 8
            scores['stability'] = stability_score
            
            total_score = sum(scores.values())
            return min(100, total_score), scores
            
        except Exception as e:
            logger.error(f"❌ Error calculating fundamental score for {symbol}: {e}")
            return 0, {}
    
    def calculate_sentiment_score(self, symbol: str, market_data: Dict) -> Tuple[float, Dict]:
        """Calculate market sentiment score (0-100)"""
        try:
            scores = {}
            ticker = market_data.get('ticker', {})
            
            # Price momentum sentiment (0-35 points)
            price_change = ticker.get('percentage', 0)
            if price_change > 5:
                momentum_sentiment = 35  # Very bullish
            elif price_change > 2:
                momentum_sentiment = 25  # Bullish
            elif price_change > 0:
                momentum_sentiment = 15  # Slightly bullish
            elif price_change > -2:
                momentum_sentiment = 15  # Slightly bearish (tradeable)
            elif price_change > -5:
                momentum_sentiment = 25  # Bearish (tradeable)
            else:
                momentum_sentiment = 35  # Very bearish (high opportunity)
            scores['momentum_sentiment'] = momentum_sentiment
            
            # Volume sentiment (0-25 points)
            volume_24h = ticker.get('baseVolume', 0)
            # High volume indicates strong sentiment
            if volume_24h > 50000000:
                volume_sentiment = 25
            elif volume_24h > 10000000:
                volume_sentiment = 20
            elif volume_24h > 1000000:
                volume_sentiment = 15
            else:
                volume_sentiment = volume_24h / 100000  # Scale smaller volumes
            scores['volume_sentiment'] = min(25, volume_sentiment)
            
            # Market structure sentiment (0-20 points)
            # Based on bid-ask dynamics
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            if bid > 0 and ask > 0:
                spread = (ask - bid) / ask
                if spread < 0.001:  # Tight spread = good sentiment
                    structure_sentiment = 20
                elif spread < 0.005:
                    structure_sentiment = 15
                elif spread < 0.01:
                    structure_sentiment = 10
                else:
                    structure_sentiment = 5
            else:
                structure_sentiment = 10
            scores['structure_sentiment'] = structure_sentiment
            
            # Time-based sentiment (0-20 points)
            # Different times have different sentiment patterns
            current_hour = datetime.now().hour
            if 8 <= current_hour <= 10:  # European market open
                time_sentiment = 20
            elif 13 <= current_hour <= 15:  # US market open
                time_sentiment = 20
            elif 21 <= current_hour <= 23:  # Asian market active
                time_sentiment = 18
            else:
                time_sentiment = 12
            scores['time_sentiment'] = time_sentiment
            
            total_score = sum(scores.values())
            return min(100, total_score), scores
            
        except Exception as e:
            logger.error(f"❌ Error calculating sentiment score for {symbol}: {e}")
            return 50, {}
    
    def calculate_liquidity_score(self, symbol: str, market_data: Dict) -> Tuple[float, Dict]:
        """Calculate liquidity score (0-100)"""
        try:
            scores = {}
            ticker = market_data.get('ticker', {})
            orderbook = market_data.get('orderbook', {})
            
            # Spread analysis (0-40 points)
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            if bid > 0 and ask > 0:
                spread = (ask - bid) / ask
                if spread < 0.001:
                    spread_score = 40
                elif spread < 0.003:
                    spread_score = 30
                elif spread < 0.006:
                    spread_score = 20
                elif spread < 0.01:
                    spread_score = 10
                else:
                    spread_score = 5
            else:
                spread_score = 0
            scores['spread'] = spread_score
            
            # Volume consistency (0-30 points)
            volume_24h = ticker.get('baseVolume', 0)
            if volume_24h > 10000000:
                volume_consistency = 30
            elif volume_24h > 1000000:
                volume_consistency = 25
            elif volume_24h > 100000:
                volume_consistency = 20
            elif volume_24h > 10000:
                volume_consistency = 15
            else:
                volume_consistency = volume_24h / 1000  # Scale smaller volumes
            scores['volume_consistency'] = min(30, volume_consistency)
            
            # Order book depth (0-30 points)
            if orderbook:
                bids = len(orderbook.get('bids', []))
                asks = len(orderbook.get('asks', []))
                depth = bids + asks
                depth_score = min(30, depth * 3)  # 3 points per order level
            else:
                depth_score = 15  # Default if no orderbook data
            scores['order_book_depth'] = depth_score
            
            total_score = sum(scores.values())
            return min(100, total_score), scores
            
        except Exception as e:
            logger.error(f"❌ Error calculating liquidity score for {symbol}: {e}")
            return 50, {}
    
    def calculate_risk_adjusted_score(self, base_score: float, symbol: str, 
                                    category: CoinCategory, market_data: Dict) -> Tuple[float, Dict]:
        """Calculate risk-adjusted score with category-specific adjustments"""
        try:
            adjustments = {}
            ticker = market_data.get('ticker', {})
            
            # Category risk adjustment
            risk_multipliers = {
                CoinCategory.MAJOR_CRYPTO: 1.0,    # No adjustment (baseline)
                CoinCategory.ALTCOINS: 0.95,       # Slight penalty for higher risk
                CoinCategory.MEME_COINS: 0.8,      # Higher penalty for extreme risk
                CoinCategory.DEFI_TOKENS: 0.9,     # Moderate penalty for smart contract risk
                CoinCategory.LAYER1: 0.93          # Small penalty for protocol risk
            }
            
            risk_multiplier = risk_multipliers.get(category, 0.95)
            adjustments['category_risk'] = risk_multiplier
            
            # Volatility risk adjustment
            price_change = abs(ticker.get('percentage', 0))
            if price_change > 15:
                volatility_adj = 0.8  # High volatility penalty
            elif price_change > 10:
                volatility_adj = 0.9
            elif price_change > 5:
                volatility_adj = 1.0  # Ideal volatility
            elif price_change > 2:
                volatility_adj = 0.95
            else:
                volatility_adj = 0.85  # Too stable for good trading
            adjustments['volatility_risk'] = volatility_adj
            
            # Volume risk adjustment
            volume_24h = ticker.get('baseVolume', 0)
            if volume_24h > 50000000:
                volume_adj = 1.05  # Bonus for high liquidity
            elif volume_24h > 10000000:
                volume_adj = 1.0
            elif volume_24h > 1000000:
                volume_adj = 0.95
            else:
                volume_adj = 0.85  # Penalty for low liquidity
            adjustments['volume_risk'] = volume_adj
            
            # Calculate final adjusted score
            final_multiplier = (
                risk_multiplier * 
                volatility_adj * 
                volume_adj
            )
            
            adjusted_score = base_score * final_multiplier
            adjustments['final_multiplier'] = final_multiplier
            
            return min(100, max(0, adjusted_score)), adjustments
            
        except Exception as e:
            logger.error(f"❌ Error calculating risk-adjusted score for {symbol}: {e}")
            return base_score, {}
    
    def calculate_comprehensive_score(self, symbol: str, market_data: Dict, 
                                    historical_data: List, indicators: Dict) -> ScoreBreakdown:
        """Calculate comprehensive VIPER score with full breakdown"""
        try:
            # Determine coin category
            base_symbol = symbol.split('/')[0]
            if base_symbol in ['BTC', 'ETH']:
                category = CoinCategory.MAJOR_CRYPTO
            elif base_symbol in ['DOGE', 'SHIB', 'PEPE']:
                category = CoinCategory.MEME_COINS
            elif base_symbol in ['UNI', 'SUSHI', 'COMP', 'AAVE']:
                category = CoinCategory.DEFI_TOKENS
            elif base_symbol in ['SOL', 'AVAX', 'NEAR', 'ALGO']:
                category = CoinCategory.LAYER1
            else:
                category = CoinCategory.ALTCOINS
            
            # Calculate individual scores
            technical_score, tech_breakdown = self.calculate_technical_score(indicators, symbol)
            fundamental_score, fund_breakdown = self.calculate_fundamental_score(symbol, market_data)
            sentiment_score, sent_breakdown = self.calculate_sentiment_score(symbol, market_data)
            liquidity_score, liq_breakdown = self.calculate_liquidity_score(symbol, market_data)
            
            # Volume score (simplified from liquidity)
            volume_score = liq_breakdown.get('volume_consistency', 50)
            
            # Volatility score (from technical)
            volatility_score = tech_breakdown.get('volatility', 50)
            
            # Momentum score (from technical)
            momentum_score = tech_breakdown.get('momentum', 50)
            
            # Correlation score (placeholder for now)
            correlation_score = 50  # Would require cross-asset analysis
            
            # Get category-specific weights
            weights = self.category_weights.get(category, self.category_weights[CoinCategory.ALTCOINS])
            
            # Calculate weighted overall score
            overall_score = (
                technical_score * weights.get(ScoreComponent.TECHNICAL, 0.25) +
                fundamental_score * weights.get(ScoreComponent.FUNDAMENTAL, 0.20) +
                sentiment_score * weights.get(ScoreComponent.SENTIMENT, 0.15) +
                volume_score * weights.get(ScoreComponent.VOLUME, 0.15) +
                volatility_score * weights.get(ScoreComponent.VOLATILITY, 0.10) +
                liquidity_score * weights.get(ScoreComponent.LIQUIDITY, 0.10) +
                momentum_score * weights.get(ScoreComponent.MOMENTUM, 0.05)
            )
            
            # Apply risk adjustments
            risk_adjusted_score, risk_adjustments = self.calculate_risk_adjusted_score(
                overall_score, symbol, category, market_data
            )
            
            return ScoreBreakdown(
                symbol=symbol,
                overall_score=round(risk_adjusted_score, 2),
                technical_score=round(technical_score, 2),
                fundamental_score=round(fundamental_score, 2),
                sentiment_score=round(sentiment_score, 2),
                volume_score=round(volume_score, 2),
                volatility_score=round(volatility_score, 2),
                liquidity_score=round(liquidity_score, 2),
                momentum_score=round(momentum_score, 2),
                correlation_score=round(correlation_score, 2),
                category=category.value,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating comprehensive score for {symbol}: {e}")
            return ScoreBreakdown(
                symbol=symbol, overall_score=0, technical_score=0,
                fundamental_score=0, sentiment_score=0, volume_score=0,
                volatility_score=0, liquidity_score=0, momentum_score=0,
                correlation_score=0, category='error', 
                timestamp=datetime.now().isoformat()
            )
    
    async def score_opportunity(self, symbol: str, market_data: Dict, 
                              historical_data: List) -> Dict:
        """Score a trading opportunity and return actionable analysis"""
        try:
            # Calculate technical indicators
            indicators = self._calculate_indicators(historical_data)
            
            # Get comprehensive score breakdown
            score_breakdown = self.calculate_comprehensive_score(
                symbol, market_data, historical_data, indicators
            )
            
            # Determine trading recommendation
            recommendation = self._generate_trading_recommendation(score_breakdown, indicators)
            
            # Calculate position sizing recommendation
            position_sizing = self._calculate_optimal_position(score_breakdown, market_data)
            
            # Store score in Redis
            await self._store_score(score_breakdown)
            
            return {
                'symbol': symbol,
                'score_breakdown': score_breakdown.__dict__,
                'recommendation': recommendation,
                'position_sizing': position_sizing,
                'indicators': indicators,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error scoring opportunity for {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_indicators(self, ohlcv_data: List) -> Dict:
        """Calculate technical indicators from OHLCV data"""
        try:
            if len(ohlcv_data) < 50:
                return {}
            
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype({'close': float, 'high': float, 'low': float, 'volume': float})
            
            # RSI
            rsi = self._calculate_rsi(df['close'])
            
            # Moving averages
            sma_10 = df['close'].rolling(10).mean().iloc[-1]
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            
            # MACD
            macd_line, macd_signal = self._calculate_macd(df['close'])
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(df['close'])
            
            # Volume analysis
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # ATR (Average True Range)
            atr = self._calculate_atr(df)
            
            return {
                'rsi': float(rsi),
                'sma_10': float(sma_10),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'macd': float(macd_line),
                'macd_signal': float(macd_signal),
                'bb_upper': float(bb_upper),
                'bb_lower': float(bb_lower),
                'bb_middle': float(bb_middle),
                'volume_ratio': float(volume_ratio),
                'atr': float(atr),
                'current_price': float(df['close'].iloc[-1]),
                'price_change_1h': float((df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4]),
                'price_change_4h': float((df['close'].iloc[-1] - df['close'].iloc[-16]) / df['close'].iloc[-16]) if len(df) >= 16 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            return float(macd_line.iloc[-1]), float(macd_signal.iloc[-1])
        except:
            return 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            upper = sma + (rolling_std * std)
            lower = sma - (rolling_std * std)
            return float(upper.iloc[-1]), float(lower.iloc[-1]), float(sma.iloc[-1])
        except:
            current = float(prices.iloc[-1])
            return current * 1.02, current * 0.98, current
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(period).mean()
            return float(atr.iloc[-1])
        except:
            return 0.0
    
    def _generate_trading_recommendation(self, score_breakdown: ScoreBreakdown, 
                                       indicators: Dict) -> Dict:
        """Generate comprehensive trading recommendation"""
        try:
            # Score-based signal strength
            if score_breakdown.overall_score >= 80:
                strength = "VERY_STRONG"
                action_confidence = "HIGH"
            elif score_breakdown.overall_score >= 65:
                strength = "STRONG"  
                action_confidence = "MEDIUM_HIGH"
            elif score_breakdown.overall_score >= 50:
                strength = "MODERATE"
                action_confidence = "MEDIUM"
            elif score_breakdown.overall_score >= 35:
                strength = "WEAK"
                action_confidence = "LOW"
            else:
                strength = "VERY_WEAK"
                action_confidence = "VERY_LOW"
            
            # Technical signal direction
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            current_price = indicators.get('current_price', 0)
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            
            # Determine direction
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI signals
            if rsi < 30:
                bullish_signals += 2  # Strong oversold
            elif rsi < 40:
                bullish_signals += 1  # Moderate oversold
            elif rsi > 70:
                bearish_signals += 2  # Strong overbought
            elif rsi > 60:
                bearish_signals += 1  # Moderate overbought
            
            # MACD signals
            if macd > macd_signal and macd > 0:
                bullish_signals += 2
            elif macd > macd_signal:
                bullish_signals += 1
            elif macd < macd_signal and macd < 0:
                bearish_signals += 2
            elif macd < macd_signal:
                bearish_signals += 1
            
            # Bollinger Bands signals
            if current_price < bb_lower:
                bullish_signals += 1  # Oversold
            elif current_price > bb_upper:
                bearish_signals += 1  # Overbought
            
            # Determine final recommendation
            if bullish_signals > bearish_signals + 1:
                direction = "BUY"
                signal_confidence = min(95, 50 + (bullish_signals - bearish_signals) * 10)
            elif bearish_signals > bullish_signals + 1:
                direction = "SELL"
                signal_confidence = min(95, 50 + (bearish_signals - bullish_signals) * 10)
            else:
                direction = "HOLD"
                signal_confidence = 50
            
            return {
                'action': direction,
                'strength': strength,
                'confidence': action_confidence,
                'signal_confidence': signal_confidence,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'score': score_breakdown.overall_score,
                'category': score_breakdown.category,
                'reasoning': self._generate_reasoning(direction, strength, bullish_signals, bearish_signals, indicators)
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendation: {e}")
            return {
                'action': 'HOLD',
                'strength': 'UNKNOWN',
                'confidence': 'LOW',
                'error': str(e)
            }
    
    def _generate_reasoning(self, direction: str, strength: str, 
                          bullish: int, bearish: int, indicators: Dict) -> str:
        """Generate human-readable reasoning for the recommendation"""
        try:
            reasons = []
            
            # Direction reasoning
            if direction == 'BUY':
                reasons.append(f"🚀 BUY signal with {bullish} bullish indicators vs {bearish} bearish")
            elif direction == 'SELL':
                reasons.append(f"📉 SELL signal with {bearish} bearish indicators vs {bullish} bullish")
            else:
                reasons.append(f"📊 HOLD - Mixed signals ({bullish} bullish, {bearish} bearish)")
            
            # Technical reasons
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                reasons.append(f"RSI overbought ({rsi:.1f})")
            
            # Volume reasons
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio > 2:
                reasons.append(f"High volume ({volume_ratio:.1f}x avg)")
            elif volume_ratio < 0.5:
                reasons.append(f"Low volume ({volume_ratio:.1f}x avg)")
            
            # Price action reasons
            price_change_1h = indicators.get('price_change_1h', 0)
            if abs(price_change_1h) > 0.02:
                reasons.append(f"Strong 1h momentum ({price_change_1h*100:.1f}%)")
            
            return " | ".join(reasons)
            
        except Exception as e:
            return f"Analysis available (details error: {e})"
    
    def _calculate_optimal_position(self, score_breakdown: ScoreBreakdown, 
                                  market_data: Dict) -> Dict:
        """Calculate optimal position sizing based on score"""
        try:
            # Base position size based on score and category
            base_sizes = {
                'major': 0.02,      # 2% for major cryptos
                'altcoins': 0.015,  # 1.5% for altcoins
                'meme': 0.005,      # 0.5% for meme coins
                'defi': 0.01,       # 1% for DeFi tokens
                'layer1': 0.012     # 1.2% for Layer 1
            }
            
            base_size = base_sizes.get(score_breakdown.category, 0.01)
            
            # Adjust based on score
            score_multiplier = score_breakdown.overall_score / 100
            
            # Adjust based on confidence
            confidence_multiplier = min(1.5, score_breakdown.overall_score / 70)
            
            # Final position size
            position_size_percent = base_size * score_multiplier * confidence_multiplier
            
            # Leverage recommendation based on category and score
            leverage_recommendations = {
                'major': 20 if score_breakdown.overall_score > 75 else 15,
                'altcoins': 15 if score_breakdown.overall_score > 70 else 10,
                'meme': 8 if score_breakdown.overall_score > 80 else 5,
                'defi': 12 if score_breakdown.overall_score > 75 else 8,
                'layer1': 16 if score_breakdown.overall_score > 75 else 12
            }
            
            recommended_leverage = leverage_recommendations.get(score_breakdown.category, 10)
            
            return {
                'position_size_percent': round(position_size_percent, 4),
                'recommended_leverage': recommended_leverage,
                'max_position_value_percent': round(position_size_percent * recommended_leverage, 2),
                'risk_level': 'LOW' if score_breakdown.overall_score > 75 else 'MEDIUM' if score_breakdown.overall_score > 50 else 'HIGH',
                'category': score_breakdown.category,
                'score_factor': round(score_multiplier, 2),
                'confidence_factor': round(confidence_multiplier, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating optimal position: {e}")
            return {
                'position_size_percent': 0.01,
                'recommended_leverage': 5,
                'risk_level': 'HIGH',
                'error': str(e)
            }
    
    async def _store_score(self, score_breakdown: ScoreBreakdown):
        """Store score breakdown in Redis"""
        try:
            score_key = f"viper:score:{score_breakdown.symbol}:{int(time.time())}"
            
            self.redis_client.setex(
                score_key,
                3600,  # 1 hour expiry
                json.dumps(score_breakdown.__dict__)
            )
            
            # Store latest score for quick access
            self.redis_client.setex(
                f"viper:latest_score:{score_breakdown.symbol}",
                1800,  # 30 minutes
                json.dumps(score_breakdown.__dict__)
            )
            
        except Exception as e:
            logger.error(f"❌ Error storing score: {e}")
    
    def get_market_ranking(self, limit: int = 20) -> List[Dict]:
        """Get market ranking by overall scores"""
        try:
            # Get all recent scores from Redis
            score_keys = self.redis_client.keys("viper:latest_score:*")
            
            rankings = []
            
            for key in score_keys:
                try:
                    score_data = self.redis_client.get(key)
                    if score_data:
                        score = json.loads(score_data)
                        rankings.append(score)
                except:
                    continue
            
            # Sort by overall score
            rankings.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
            
            return rankings[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error getting market ranking: {e}")
            return []
    
    def analyze_portfolio_diversification(self, active_positions: List[str]) -> Dict:
        """Analyze portfolio diversification across categories"""
        try:
            if not active_positions:
                return {
                    'diversification_score': 100,
                    'recommendation': 'No active positions - can diversify freely'
                }
            
            # Count positions by category
            category_counts = {}
            for symbol in active_positions:
                base = symbol.split('/')[0]
                if base in ['BTC', 'ETH']:
                    category = 'major'
                elif base in ['DOGE', 'SHIB']:
                    category = 'meme'
                elif base in ['UNI', 'SUSHI', 'COMP']:
                    category = 'defi'
                elif base in ['SOL', 'AVAX', 'NEAR']:
                    category = 'layer1'
                else:
                    category = 'altcoins'
                
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Calculate diversification score
            total_positions = len(active_positions)
            num_categories = len(category_counts)
            
            # Ideal diversification: positions spread across 3-4 categories
            if num_categories >= 4:
                diversification_score = 100
            elif num_categories == 3:
                diversification_score = 85
            elif num_categories == 2:
                diversification_score = 60
            else:
                diversification_score = 30
            
            # Check for over-concentration
            max_concentration = max(category_counts.values()) / total_positions
            if max_concentration > 0.6:  # More than 60% in one category
                diversification_score *= 0.7
            
            # Generate recommendations
            recommendations = []
            if num_categories < 3:
                missing_categories = [cat for cat in ['major', 'altcoins', 'defi', 'layer1'] 
                                    if cat not in category_counts]
                recommendations.append(f"Consider adding positions in: {', '.join(missing_categories)}")
            
            if max_concentration > 0.5:
                dominant_category = max(category_counts, key=category_counts.get)
                recommendations.append(f"Over-concentrated in {dominant_category} ({max_concentration*100:.1f}%)")
            
            return {
                'diversification_score': round(diversification_score, 1),
                'category_distribution': category_counts,
                'total_positions': total_positions,
                'num_categories': num_categories,
                'max_concentration': round(max_concentration, 2),
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing diversification: {e}")
            return {'error': str(e)}

# Global scoring engine instance
scoring_engine = VIPERScoringEngine()

def get_scoring_engine() -> VIPERScoringEngine:
    """Get the global scoring engine instance"""
    return scoring_engine