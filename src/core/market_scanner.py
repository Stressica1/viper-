#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Advanced Market Scanner
Real-time market scanning with intelligent pair selection
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import ccxt
import redis
import aiohttp
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .advanced_trading_strategy import AdvancedTradingStrategy, CoinCategory

logger = logging.getLogger(__name__)

@dataclass
class ScanResult:
    """Data class for scan results"""
    symbol: str
    category: str
    score: float
    signal: str
    confidence: float
    volume_24h: float
    price: float
    spread: float
    leverage_available: bool
    error: Optional[str] = None

class MarketScanner:
    """
    Advanced market scanner with real-time monitoring and intelligent filtering
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or self._create_redis_client()
        self.strategy = AdvancedTradingStrategy(self.redis_client)
        
        # Exchange configuration (may be None in offline mode)
        self.exchange = self._initialize_exchange()
        
        # Scanning configuration
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', '300'))  # 5 minutes
        self.min_volume_threshold = float(os.getenv('MIN_VOLUME_THRESHOLD', '100000'))
        self.max_spread_threshold = float(os.getenv('MAX_SPREAD_THRESHOLD', '0.01'))
        self.max_concurrent_scans = int(os.getenv('MAX_CONCURRENT_SCANS', '20'))
        
        # Tracking variables
        self.is_scanning = False
        self.scan_results: List[ScanResult] = []
        self.error_count = 0
        self.last_scan_time = None
        
        # Priority symbols for scanning (most profitable typically)
        self.priority_symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'BNB/USDT:USDT', 'ADA/USDT:USDT', 'LINK/USDT:USDT',
            'DOT/USDT:USDT', 'AVAX/USDT:USDT', 'UNI/USDT:USDT',
            'MATIC/USDT:USDT', 'DOGE/USDT:USDT', 'SHIB/USDT:USDT'
        ]
        
        logger.info("🔍 Advanced Market Scanner initialized")
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client"""
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        return redis.Redis.from_url(redis_url, decode_responses=True)
    
    def _initialize_exchange(self) -> Optional[ccxt.Exchange]:
        """Initialize exchange connection (graceful fallback for network issues)"""
        try:
            exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY', ''),
                'secret': os.getenv('BITGET_API_SECRET', ''),
                'password': os.getenv('BITGET_API_PASSWORD', ''),
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                },
                'sandbox': False,
            })
            
            # Load markets with timeout and error handling
            exchange.load_markets()
            logger.info(f"✅ Exchange initialized with {len(exchange.symbols)} markets")
            return exchange
            
        except Exception as e:
            logger.warning(f"⚠️ Exchange initialization failed, running in offline mode: {e}")
            # Return None to indicate offline mode - scanner can still work with cached data
            return None
    
    async def scan_single_pair(self, symbol: str) -> ScanResult:
        """Scan a single trading pair for opportunities"""
        try:
            # Get market data
            ticker = await self._fetch_ticker_async(symbol)
            if not ticker:
                return ScanResult(
                    symbol=symbol, category='unknown', score=0, signal='ERROR',
                    confidence=0, volume_24h=0, price=0, spread=0,
                    leverage_available=False, error="Failed to fetch ticker"
                )
            
            # Get historical data for technical analysis
            historical_data = await self._fetch_ohlcv_async(symbol, '1h', 100)
            if not historical_data or len(historical_data) < 50:
                return ScanResult(
                    symbol=symbol, category='unknown', score=0, signal='HOLD',
                    confidence=0, volume_24h=ticker.get('baseVolume', 0),
                    price=ticker.get('last', 0), spread=0,
                    leverage_available=False, error="Insufficient historical data"
                )
            
            # Generate advanced signal
            market_data = {'ticker': ticker}
            signal_data = self.strategy.generate_advanced_signal(symbol, market_data, historical_data)
            
            # Calculate market quality score
            quality_score = self._calculate_market_quality(ticker, symbol)
            
            # Check leverage availability
            leverage_available = await self._check_leverage_async(symbol)
            
            # Calculate spread
            spread = 0
            if ticker.get('bid') and ticker.get('ask'):
                spread = (ticker['ask'] - ticker['bid']) / ticker['ask']
            
            return ScanResult(
                symbol=symbol,
                category=signal_data.get('category', 'unknown'),
                score=quality_score,
                signal=signal_data.get('signal', 'HOLD'),
                confidence=signal_data.get('confidence', 0),
                volume_24h=ticker.get('baseVolume', 0),
                price=ticker.get('last', 0),
                spread=spread,
                leverage_available=leverage_available
            )
            
        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            return ScanResult(
                symbol=symbol, category='error', score=0, signal='ERROR',
                confidence=0, volume_24h=0, price=0, spread=0,
                leverage_available=False, error=str(e)
            )
    
    async def _fetch_ticker_async(self, symbol: str) -> Optional[Dict]:
        """Async wrapper for fetching ticker data"""
        if self.exchange is None:
            logger.warning(f"⚠️ Exchange not available, cannot fetch ticker for {symbol}")
            return None
            
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.exchange.fetch_ticker, symbol)
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch ticker for {symbol}: {e}")
            return None
    
    async def _fetch_ohlcv_async(self, symbol: str, timeframe: str, limit: int) -> Optional[List]:
        """Async wrapper for fetching OHLCV data"""
        if self.exchange is None:
            logger.warning(f"⚠️ Exchange not available, cannot fetch OHLCV for {symbol}")
            return None
            
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.exchange.fetch_ohlcv, symbol, timeframe, limit)
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch OHLCV for {symbol}: {e}")
            return None
    
    async def _check_leverage_async(self, symbol: str) -> bool:
        """Async check for leverage availability"""
        if self.exchange is None:
            return False
            
        try:
            loop = asyncio.get_event_loop()
            market = await loop.run_in_executor(None, self.exchange.market, symbol)
            
            # For Bitget swaps, leverage is typically available
            if market.get('type') == 'swap' and market.get('active'):
                return True
            
            return False
        except:
            return False
    
    def _calculate_market_quality(self, ticker: Dict, symbol: str) -> float:
        """Calculate market quality score (0-100)"""
        try:
            # Volume score (0-40 points)
            volume = ticker.get('baseVolume', 0)
            volume_score = min(40, (volume / 1000000) * 10)  # Scale by millions
            
            # Spread score (0-20 points)
            if ticker.get('bid') and ticker.get('ask'):
                spread = (ticker['ask'] - ticker['bid']) / ticker['ask']
                spread_score = max(0, 20 - (spread * 10000))  # Lower spread = higher score
            else:
                spread_score = 0
            
            # Price stability score (0-20 points)
            price_change = abs(ticker.get('percentage', 0))
            if 1 <= price_change <= 5:  # Sweet spot for volatility
                stability_score = 20
            elif price_change < 1:
                stability_score = 10  # Too stable
            elif price_change > 10:
                stability_score = 5   # Too volatile
            else:
                stability_score = 15
            
            # Market cap proxy (0-20 points) - based on symbol popularity
            popularity_score = 20 if symbol in self.priority_symbols else 10
            
            total_score = volume_score + spread_score + stability_score + popularity_score
            return min(100, max(0, total_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating market quality for {symbol}: {e}")
            return 0
    
    async def scan_all_markets(self) -> List[ScanResult]:
        """Scan all available markets with concurrent processing"""
        try:
            logger.info("🔍 Starting comprehensive market scan...")
            
            # Handle offline mode
            if self.exchange is None:
                logger.warning("⚠️ Exchange not available, using priority symbols only")
                symbols = self.priority_symbols
            else:
                # Get all USDT perpetual swap pairs
                symbols = [s for s in self.exchange.symbols if ':USDT' in s and s.endswith('T')]
            
            # Prioritize scanning - priority symbols first
            priority_symbols = [s for s in symbols if s in self.priority_symbols]
            other_symbols = [s for s in symbols if s not in self.priority_symbols]
            ordered_symbols = priority_symbols + other_symbols
            
            logger.info(f"📊 Scanning {len(ordered_symbols)} markets ({len(priority_symbols)} priority)")
            
            # Scan in batches to avoid overwhelming the API
            batch_size = self.max_concurrent_scans
            results = []
            
            for i in range(0, len(ordered_symbols), batch_size):
                batch = ordered_symbols[i:i + batch_size]
                logger.info(f"🔄 Processing batch {i//batch_size + 1} ({len(batch)} symbols)")
                
                # Concurrent scanning for current batch
                batch_tasks = [self.scan_single_pair(symbol) for symbol in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, ScanResult):
                        results.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"❌ Batch scan exception: {result}")
                
                # Small delay between batches
                await asyncio.sleep(1)
            
            # Store results
            self.scan_results = results
            await self._store_scan_results(results)
            
            logger.info(f"✅ Market scan completed: {len(results)} pairs analyzed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in market scan: {e}")
            return []
    
    async def _store_scan_results(self, results: List[ScanResult]) -> bool:
        """Store scan results in Redis"""
        try:
            # Store individual results
            for result in results:
                if result.error:
                    continue
                
                result_data = {
                    'symbol': result.symbol,
                    'category': result.category,
                    'score': result.score,
                    'signal': result.signal,
                    'confidence': result.confidence,
                    'volume_24h': result.volume_24h,
                    'price': result.price,
                    'spread': result.spread,
                    'leverage_available': result.leverage_available,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store with 1 hour expiration
                self.redis_client.setex(
                    f"viper:scan:{result.symbol}",
                    3600,
                    json.dumps(result_data)
                )
            
            # Store summary
            summary = {
                'total_scanned': len(results),
                'successful_scans': len([r for r in results if not r.error]),
                'error_count': len([r for r in results if r.error]),
                'buy_signals': len([r for r in results if 'BUY' in r.signal]),
                'sell_signals': len([r for r in results if 'SELL' in r.signal]),
                'high_confidence': len([r for r in results if r.confidence > 70]),
                'high_quality': len([r for r in results if r.score > 70]),
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                'viper:scan_summary',
                3600,
                json.dumps(summary)
            )
            
            logger.info("✅ Scan results stored in Redis")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing scan results: {e}")
            return False
    
    def get_top_opportunities(self, limit: int = 10) -> List[Dict]:
        """Get top trading opportunities from recent scan"""
        try:
            if not self.scan_results:
                return []
            
            # Filter for valid signals
            valid_results = [
                r for r in self.scan_results 
                if not r.error and r.signal != 'HOLD' and r.confidence > 60
            ]
            
            # Sort by combined score (quality + confidence)
            scored_results = []
            for result in valid_results:
                combined_score = (result.score * 0.4) + (result.confidence * 0.6)
                scored_results.append((result, combined_score))
            
            # Sort by combined score
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top opportunities
            top_opportunities = []
            for result, combined_score in scored_results[:limit]:
                opportunity = {
                    'symbol': result.symbol,
                    'signal': result.signal,
                    'confidence': result.confidence,
                    'quality_score': result.score,
                    'combined_score': round(combined_score, 1),
                    'category': result.category,
                    'volume_24h': result.volume_24h,
                    'price': result.price,
                    'spread': result.spread,
                    'leverage_available': result.leverage_available,
                    'coin_config': self.strategy.get_coin_configuration(result.symbol)
                }
                top_opportunities.append(opportunity)
            
            return top_opportunities
            
        except Exception as e:
            logger.error(f"❌ Error getting top opportunities: {e}")
            return []
    
    def filter_tradeable_pairs(self, min_score: float = 50, min_confidence: float = 60) -> List[str]:
        """Filter pairs that meet trading criteria"""
        try:
            tradeable = []
            
            for result in self.scan_results:
                if (not result.error and 
                    result.score >= min_score and 
                    result.confidence >= min_confidence and
                    result.volume_24h >= self.min_volume_threshold and
                    result.spread <= self.max_spread_threshold and
                    result.signal != 'HOLD'):
                    
                    tradeable.append(result.symbol)
            
            logger.info(f"🎯 Found {len(tradeable)} tradeable pairs meeting criteria")
            return tradeable
            
        except Exception as e:
            logger.error(f"❌ Error filtering tradeable pairs: {e}")
            return []
    
    async def real_time_monitoring(self):
        """Continuous real-time monitoring of priority symbols"""
        logger.info("🔄 Starting real-time market monitoring...")
        
        while True:
            try:
                # Monitor priority symbols more frequently
                monitor_tasks = [
                    self.scan_single_pair(symbol) 
                    for symbol in self.priority_symbols[:10]  # Top 10 priority
                ]
                
                results = await asyncio.gather(*monitor_tasks, return_exceptions=True)
                
                # Process results and send alerts for high-confidence signals
                for result in results:
                    if (isinstance(result, ScanResult) and 
                        not result.error and 
                        result.confidence > 80 and
                        result.signal in ['STRONG_BUY', 'STRONG_SELL', 'BUY', 'SELL']):
                        
                        await self._send_trading_alert(result)
                
                # Wait before next monitoring cycle
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"❌ Error in real-time monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _send_trading_alert(self, result: ScanResult):
        """Send trading alert for high-confidence signals"""
        try:
            alert = {
                'type': 'high_confidence_signal',
                'symbol': result.symbol,
                'signal': result.signal,
                'confidence': result.confidence,
                'price': result.price,
                'category': result.category,
                'timestamp': datetime.now().isoformat(),
                'message': f"🚀 {result.signal} signal for {result.symbol} with {result.confidence:.1f}% confidence"
            }
            
            # Store in Redis for alert system
            self.redis_client.lpush('viper:trading_alerts', json.dumps(alert))
            self.redis_client.expire('viper:trading_alerts', 3600)  # 1 hour expiry
            
            # Publish to alert channel
            self.redis_client.publish('viper:alerts', json.dumps(alert))
            
            logger.info(f"🚨 Trading alert sent: {alert['message']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending trading alert: {e}")
    
    async def comprehensive_market_scan(self) -> Dict:
        """Perform comprehensive market analysis"""
        try:
            start_time = time.time()
            
            # Perform full market scan
            results = await self.scan_all_markets()
            
            # Analyze results by category
            category_analysis = {}
            for category in CoinCategory:
                category_results = [r for r in results if r.category == category.value]
                if category_results:
                    avg_score = sum(r.score for r in category_results) / len(category_results)
                    avg_confidence = sum(r.confidence for r in category_results) / len(category_results)
                    category_analysis[category.value] = {
                        'total_pairs': len(category_results),
                        'avg_score': avg_score,
                        'avg_confidence': avg_confidence,
                        'buy_signals': len([r for r in category_results if 'BUY' in r.signal]),
                        'sell_signals': len([r for r in category_results if 'SELL' in r.signal]),
                        'top_pair': max(category_results, key=lambda x: x.score).symbol
                    }
            
            # Get top opportunities
            opportunities = self.get_top_opportunities(20)
            
            # Calculate performance metrics
            scan_time = time.time() - start_time
            success_rate = len([r for r in results if not r.error]) / len(results) if results else 0
            
            analysis = {
                'scan_summary': {
                    'total_pairs_scanned': len(results),
                    'successful_scans': len([r for r in results if not r.error]),
                    'error_count': len([r for r in results if r.error]),
                    'scan_time_seconds': round(scan_time, 2),
                    'success_rate': round(success_rate * 100, 1),
                    'timestamp': datetime.now().isoformat()
                },
                'category_analysis': category_analysis,
                'top_opportunities': opportunities,
                'tradeable_pairs': self.filter_tradeable_pairs(),
                'market_conditions': self.strategy.evaluate_market_conditions(),
                'recommendations': self._generate_trading_recommendations(opportunities)
            }
            
            # Store comprehensive analysis
            await self._store_market_analysis(analysis)
            
            logger.info(f"✅ Comprehensive market scan completed in {scan_time:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive market scan: {e}")
            return {'error': str(e)}
    
    async def _store_market_analysis(self, analysis: Dict):
        """Store market analysis results"""
        try:
            # Store full analysis
            self.redis_client.setex(
                'viper:market_analysis',
                1800,  # 30 minutes
                json.dumps(analysis)
            )
            
            # Store top opportunities separately for quick access
            self.redis_client.setex(
                'viper:top_opportunities',
                1800,
                json.dumps(analysis.get('top_opportunities', []))
            )
            
            logger.info("✅ Market analysis stored successfully")
            
        except Exception as e:
            logger.error(f"❌ Error storing market analysis: {e}")
    
    def _generate_trading_recommendations(self, opportunities: List[Dict]) -> List[str]:
        """Generate actionable trading recommendations"""
        recommendations = []
        
        try:
            # Categorize opportunities
            buy_ops = [o for o in opportunities if 'BUY' in o['signal']]
            sell_ops = [o for o in opportunities if 'SELL' in o['signal']]
            
            # High confidence recommendations
            high_conf_buy = [o for o in buy_ops if o['confidence'] > 80]
            high_conf_sell = [o for o in sell_ops if o['confidence'] > 80]
            
            if high_conf_buy:
                top_buy = high_conf_buy[0]
                recommendations.append(
                    f"🚀 PRIORITY BUY: {top_buy['symbol']} - {top_buy['confidence']:.1f}% confidence, Quality: {top_buy['quality_score']:.1f}"
                )
            
            if high_conf_sell:
                top_sell = high_conf_sell[0]
                recommendations.append(
                    f"📉 PRIORITY SELL: {top_sell['symbol']} - {top_sell['confidence']:.1f}% confidence, Quality: {top_sell['quality_score']:.1f}"
                )
            
            # Category recommendations
            categories_with_opportunities = {}
            for opp in opportunities[:10]:  # Top 10
                category = opp.get('category', 'unknown')
                if category not in categories_with_opportunities:
                    categories_with_opportunities[category] = []
                categories_with_opportunities[category].append(opp)
            
            for category, ops in categories_with_opportunities.items():
                if len(ops) >= 2:
                    recommendations.append(
                        f"💎 {category.upper()} SECTOR: {len(ops)} opportunities - Top: {ops[0]['symbol']} ({ops[0]['confidence']:.1f}% conf)"
                    )
            
            # Risk warnings
            high_risk_ops = [o for o in opportunities if o.get('category') == 'meme']
            if high_risk_ops:
                recommendations.append(
                    f"⚠️ MEME COIN ALERT: {len(high_risk_ops)} high-risk opportunities - Use smaller position sizes"
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return ["❌ Error generating recommendations"]
    
    def start_continuous_scanning(self, interval: int = 300):
        """Start continuous market scanning"""
        async def scan_loop():
            while self.is_scanning:
                try:
                    logger.info("🔍 Starting scheduled market scan...")
                    await self.comprehensive_market_scan()
                    
                    # Wait for next scan
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"❌ Error in scan loop: {e}")
                    await asyncio.sleep(60)  # Shorter delay on error
        
        self.is_scanning = True
        
        # Start scanning and monitoring tasks
        loop = asyncio.get_event_loop()
        loop.create_task(scan_loop())
        loop.create_task(self.real_time_monitoring())
        
        logger.info(f"✅ Continuous scanning started (interval: {interval}s)")
    
    def stop_scanning(self):
        """Stop continuous scanning"""
        self.is_scanning = False
        logger.info("🛑 Market scanning stopped")
    
    def get_scan_status(self) -> Dict:
        """Get current scanning status"""
        return {
            'is_scanning': self.is_scanning,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'total_results': len(self.scan_results),
            'error_count': self.error_count,
            'priority_symbols': self.priority_symbols,
            'scan_interval': self.scan_interval,
            'min_volume_threshold': self.min_volume_threshold,
            'max_spread_threshold': self.max_spread_threshold
        }
    
    def get_market_overview(self) -> Dict:
        """Get market overview from latest scan"""
        try:
            if not self.scan_results:
                return {'error': 'No scan results available'}
            
            # Calculate overview metrics
            valid_results = [r for r in self.scan_results if not r.error]
            
            if not valid_results:
                return {'error': 'No valid scan results'}
            
            overview = {
                'total_markets': len(valid_results),
                'average_score': round(sum(r.score for r in valid_results) / len(valid_results), 1),
                'average_confidence': round(sum(r.confidence for r in valid_results) / len(valid_results), 1),
                'signals': {
                    'BUY': len([r for r in valid_results if 'BUY' in r.signal]),
                    'SELL': len([r for r in valid_results if 'SELL' in r.signal]),
                    'HOLD': len([r for r in valid_results if r.signal == 'HOLD'])
                },
                'high_quality_pairs': len([r for r in valid_results if r.score > 70]),
                'high_confidence_signals': len([r for r in valid_results if r.confidence > 75]),
                'categories': {}
            }
            
            # Category breakdown
            for category in CoinCategory:
                cat_results = [r for r in valid_results if r.category == category.value]
                if cat_results:
                    avg_score = sum(r.score for r in cat_results) / len(cat_results)
                    overview['categories'][category.value] = {
                        'count': len(cat_results),
                        'avg_score': round(avg_score, 1),
                        'opportunities': len([r for r in cat_results if r.signal != 'HOLD'])
                    }
            
            return overview
            
        except Exception as e:
            logger.error(f"❌ Error generating market overview: {e}")
            return {'error': str(e)}

# Global scanner instance (lazy initialization)
_market_scanner = None

def get_scanner_instance() -> MarketScanner:
    """Get the global scanner instance (lazy initialization)"""
    global _market_scanner
    if _market_scanner is None:
        _market_scanner = MarketScanner()
    return _market_scanner