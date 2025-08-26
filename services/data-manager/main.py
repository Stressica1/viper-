#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Data Manager Service
Market data synchronization, caching, and persistence layer

Features:
- Real-time market data fetching from Bitget
- Redis caching with TTL management
- Historical data storage and retrieval
- Data validation and error handling
- RESTful API for data access
"""

import os
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import redis
import ccxt
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import threading
import schedule

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'data-manager')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataManager:
    """Data Manager for market data synchronization and caching"""

    def __init__(self):
        self.redis_client = None
        self.exchange = None
        self.is_running = False

        # Configuration - Intelligent Cache TTL
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')

        # Intelligent TTL configuration based on data volatility
        self.cache_ttl = {
            'ticker': int(os.getenv('CACHE_TTL_TICKER_SECONDS', '30')),
            'orderbook': int(os.getenv('CACHE_TTL_ORDERBOOK_SECONDS', '10')),
            'trades': int(os.getenv('CACHE_TTL_TRADES_SECONDS', '60')),
            'ohlcv_1m': int(os.getenv('CACHE_TTL_OHLCV_1M_SECONDS', '300')),
            'ohlcv_5m': int(os.getenv('CACHE_TTL_OHLCV_5M_SECONDS', '900')),
            'ohlcv_1h': int(os.getenv('CACHE_TTL_OHLCV_1H_SECONDS', '3600')),
            'ohlcv_4h': int(os.getenv('CACHE_TTL_OHLCV_4H_SECONDS', '14400')),
            'market_info': int(os.getenv('CACHE_TTL_MARKET_INFO_SECONDS', '86400')),
            'analytics': int(os.getenv('CACHE_TTL_ANALYTICS_SECONDS', '1800'))
        }

        # Cache management settings
        self.update_interval = int(os.getenv('UPDATE_INTERVAL_SECONDS', '30'))
        self.cache_cleanup_interval = int(os.getenv('CACHE_CLEANUP_INTERVAL_SECONDS', '300'))
        self.max_cache_memory_percent = int(os.getenv('MAX_CACHE_MEMORY_PERCENT', '80'))

        # Fallback TTL for backward compatibility
        self.default_cache_ttl = int(os.getenv('CACHE_TTL_SECONDS', '300'))

        # Fetch ALL supported symbols and timeframes
        self.symbols = self.fetch_all_trading_pairs()
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

    def fetch_all_trading_pairs(self):
        """Fetch ALL available trading pairs from Bitget"""
        try:
            spot_instruments_url = f"{self.redis_url.replace('redis://', 'http://').replace(':6379', '')}/api/v2/spot/public/symbols"
            # Actually use the exchange API directly
            import ccxt
            exchange = ccxt.bitget({
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                },
            })

            # Get all markets
            markets = exchange.load_markets()
            usdt_pairs = []

            for symbol, market in markets.items():
                if market['quote'] == 'USDT' and market['active']:
                    usdt_pairs.append(symbol)

            logger.info(f"📊 Data Manager monitoring {len(usdt_pairs)} USDT trading pairs")

            # Limit to top 200 pairs for performance (can be adjusted)
            if len(usdt_pairs) > 200:
                logger.info(f"📊 Limiting to top 200 pairs (found {len(usdt_pairs)} total)")
                usdt_pairs = usdt_pairs[:200]

            return sorted(usdt_pairs)

        except Exception as e:
            logger.warning(f"❌ Error fetching trading pairs: {e}, using fallback")
            return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT']  # Fallback

        logger.info("🏗️ Initializing Data Manager...")

    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False

    def initialize_exchange(self) -> bool:
        """Initialize exchange connection for data fetching"""
        try:
            self.exchange = ccxt.bitget({
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                },
            })
            self.exchange.load_markets()
            logger.info("✅ Exchange connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange: {e}")
            return False

    def cache_key(self, symbol: str, timeframe: str = None, data_type: str = "ticker") -> str:
        """Generate Redis cache key"""
        if timeframe:
            return f"viper:{data_type}:{symbol.replace('/', '_').replace(':', '_')}:{timeframe}"
        return f"viper:{data_type}:{symbol.replace('/', '_').replace(':', '_')}"

    def fetch_ticker_data(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker data"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            if ticker:
                data = {
                    'symbol': symbol,
                    'price': ticker.get('last', 0),
                    'bid': ticker.get('bid', 0),
                    'ask': ticker.get('ask', 0),
                    'volume': ticker.get('baseVolume', 0),
                    'quote_volume': ticker.get('quoteVolume', 0),
                    'high': ticker.get('high', 0),
                    'low': ticker.get('low', 0),
                    'open': ticker.get('open', 0),
                    'timestamp': datetime.now().isoformat(),
                    'exchange_timestamp': ticker.get('timestamp')
                }
                return data
            return None
        except Exception as e:
            logger.error(f"❌ Failed to fetch ticker for {symbol}: {e}")
            return None

    def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[List]:
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if ohlcv:
                formatted_data = []
                for candle in ohlcv:
                    formatted_data.append({
                        'timestamp': candle[0],
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
                return formatted_data
            return None
        except Exception as e:
            logger.error(f"❌ Failed to fetch OHLCV for {symbol} {timeframe}: {e}")
            return None

    def update_ticker_cache(self, symbol: str) -> bool:
        """Update ticker data in cache with intelligent TTL"""
        try:
            data = self.fetch_ticker_data(symbol)
            if data:
                cache_key = self.cache_key(symbol, data_type="ticker")
                ttl = self.cache_ttl.get('ticker', self.default_cache_ttl)
                self.redis_client.setex(cache_key, ttl, json.dumps(data))

                # Cache metadata for monitoring
                self._update_cache_metadata(cache_key, 'ticker', ttl)

                logger.debug(f"📊 Updated ticker cache for {symbol} (TTL: {ttl}s)")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update ticker cache for {symbol}: {e}")
            return False

    def update_ohlcv_cache(self, symbol: str, timeframe: str) -> bool:
        """Update OHLCV data in cache with intelligent TTL"""
        try:
            data = self.fetch_ohlcv_data(symbol, timeframe)
            if data:
                cache_key = self.cache_key(symbol, timeframe, "ohlcv")

                # Select appropriate TTL based on timeframe
                ttl_key = f'ohlcv_{timeframe}'
                ttl = self.cache_ttl.get(ttl_key, self.default_cache_ttl)

                self.redis_client.setex(cache_key, ttl, json.dumps(data))

                # Cache metadata for monitoring
                self._update_cache_metadata(cache_key, f'ohlcv_{timeframe}', ttl)

                logger.debug(f"📊 Updated OHLCV cache for {symbol} {timeframe} (TTL: {ttl}s)")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update OHLCV cache for {symbol} {timeframe}: {e}")
            return False

    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache with hit/miss tracking"""
        try:
            data = self.redis_client.get(cache_key)
            if data:
                # Track cache hit
                self._increment_cache_metric('hits')
                return json.loads(data)
            else:
                # Track cache miss
                self._increment_cache_metric('misses')
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get cached data for key {cache_key}: {e}")
            self._increment_cache_metric('errors')
            return None

    def _update_cache_metadata(self, cache_key: str, data_type: str, ttl: int):
        """Update cache metadata for monitoring"""
        try:
            metadata_key = f"viper:metadata:{cache_key}"
            metadata = {
                'data_type': data_type,
                'ttl': ttl,
                'created_at': datetime.now().isoformat(),
                'size_bytes': len(self.redis_client.get(cache_key) or ''),
                'access_count': 0,
                'last_accessed': datetime.now().isoformat()
            }
            self.redis_client.setex(metadata_key, ttl, json.dumps(metadata))
        except Exception as e:
            logger.debug(f"Failed to update cache metadata for {cache_key}: {e}")

    def _increment_cache_metric(self, metric_type: str):
        """Increment cache performance metrics"""
        try:
            metric_key = f"viper:metrics:cache:{metric_type}"
            self.redis_client.incr(metric_key)
            # Set expiry on metrics to prevent unbounded growth
            self.redis_client.expire(metric_key, 86400)  # 24 hours
        except Exception as e:
            logger.debug(f"Failed to increment cache metric {metric_type}: {e}")

    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        try:
            metrics = {}
            metric_types = ['hits', 'misses', 'errors']

            for metric_type in metric_types:
                key = f"viper:metrics:cache:{metric_type}"
                value = self.redis_client.get(key)
                metrics[metric_type] = int(value) if value else 0

            # Calculate hit rate
            total_requests = metrics['hits'] + metrics['misses']
            if total_requests > 0:
                metrics['hit_rate'] = round((metrics['hits'] / total_requests) * 100, 2)
            else:
                metrics['hit_rate'] = 0.0

            # Memory usage
            info = self.redis_client.info('memory')
            metrics['memory_used_bytes'] = info.get('used_memory', 0)
            metrics['memory_peak_bytes'] = info.get('used_memory_peak', 0)

            return metrics
        except Exception as e:
            logger.error(f"Failed to get cache metrics: {e}")
            return {}

    def warmup_cache(self):
        """Warm up cache with frequently accessed data"""
        logger.info("🔥 Warming up cache with frequently accessed data...")

        try:
            # Warm up ticker data for all symbols
            for symbol in self.symbols:
                self.update_ticker_cache(symbol)
                logger.debug(f"🔥 Warmed up ticker cache for {symbol}")

            # Warm up recent OHLCV data (1m and 5m timeframes)
            for symbol in self.symbols:
                for timeframe in ['1m', '5m']:
                    self.update_ohlcv_cache(symbol, timeframe)
                    logger.debug(f"🔥 Warmed up OHLCV cache for {symbol} {timeframe}")

            logger.info("✅ Cache warmup completed successfully")

        except Exception as e:
            logger.error(f"❌ Cache warmup failed: {e}")

    def cleanup_expired_cache(self):
        """Clean up expired cache entries and optimize memory"""
        try:
            logger.debug("🧹 Starting cache cleanup...")

            # Redis automatically expires keys, but we can optimize by removing old metadata
            metadata_keys = self.redis_client.keys("viper:metadata:*")
            cleaned_count = 0

            for key in metadata_keys:
                # Check if the main key still exists
                main_key = key.replace("viper:metadata:", "")
                if not self.redis_client.exists(main_key):
                    self.redis_client.delete(key)
                    cleaned_count += 1

            if cleaned_count > 0:
                logger.debug(f"🧹 Cleaned up {cleaned_count} expired metadata entries")

            # Log memory usage
            info = self.redis_client.info('memory')
            used_memory = info.get('used_memory_human', 'unknown')
            logger.debug(f"📊 Redis memory usage: {used_memory}")

        except Exception as e:
            logger.error(f"❌ Cache cleanup failed: {e}")

    def start_data_collection(self):
        """Start the data collection loop with cache optimization"""
        logger.info("🚀 Starting optimized data collection loop...")
        self.is_running = True

        # Initial cache warmup
        self.warmup_cache()

        # Schedule data updates
        schedule.every(self.update_interval).seconds.do(self.update_all_data)
        schedule.every(30).seconds.do(self.update_all_tickers)

        # Schedule cache maintenance
        schedule.every(self.cache_cleanup_interval).seconds.do(self.cleanup_expired_cache)

        # Schedule periodic cache warmup (every hour)
        schedule.every(3600).seconds.do(self.warmup_cache)

        logger.info("📊 Cache optimization enabled:")
        logger.info(f"   • Cache warmup: Every 1 hour")
        logger.info(f"   • Cache cleanup: Every {self.cache_cleanup_interval}s")
        logger.info(f"   • Update interval: {self.update_interval}s")

        while self.is_running:
            schedule.run_pending()
            time.sleep(1)

    def update_all_tickers(self):
        """Update all ticker data"""
        logger.info("📊 Updating all ticker data...")
        for symbol in self.symbols:
            self.update_ticker_cache(symbol)

    def update_all_data(self):
        """Update all market data"""
        logger.info("📊 Updating all market data...")
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                self.update_ohlcv_cache(symbol, timeframe)

    def stop(self):
        """Stop the data manager"""
        logger.info("🛑 Stopping Data Manager...")
        self.is_running = False

# FastAPI application
app = FastAPI(
    title="VIPER Data Manager",
    version="1.0.0",
    description="Market data synchronization and caching service"
)

data_manager = DataManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if not data_manager.initialize_redis():
        logger.error("❌ Failed to initialize Redis. Exiting...")
        return

    if not data_manager.initialize_exchange():
        logger.error("❌ Failed to initialize exchange. Exiting...")
        return

    # Start data collection in background thread
    thread = threading.Thread(target=data_manager.start_data_collection, daemon=True)
    thread.start()
    logger.info("✅ Data Manager started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    data_manager.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        data_manager.redis_client.ping()

        return {
            "status": "healthy",
            "service": "data-manager",
            "redis_connected": True,
            "exchange_connected": data_manager.exchange is not None,
            "data_collection_running": data_manager.is_running
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "data-manager",
                "error": str(e)
            }
        )

@app.get("/api/cache/metrics")
async def get_cache_metrics():
    """Get cache performance metrics"""
    try:
        metrics = data_manager.get_cache_metrics()
        return {
            "service": "data-manager",
            "cache_performance": metrics,
            "cache_configuration": {
                "update_interval_seconds": data_manager.update_interval,
                "cache_cleanup_interval_seconds": data_manager.cache_cleanup_interval,
                "max_memory_percent": data_manager.max_cache_memory_percent,
                "ttl_config": data_manager.cache_ttl
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to get cache metrics: {str(e)}"
            }
        )

@app.post("/api/cache/warmup")
async def warmup_cache():
    """Manually trigger cache warmup"""
    try:
        data_manager.warmup_cache()
        return {
            "status": "success",
            "message": "Cache warmup initiated",
            "service": "data-manager"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Cache warmup failed: {str(e)}"
            }
        )

@app.post("/api/cache/cleanup")
async def cleanup_cache():
    """Manually trigger cache cleanup"""
    try:
        data_manager.cleanup_expired_cache()
        return {
            "status": "success",
            "message": "Cache cleanup completed",
            "service": "data-manager"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Cache cleanup failed: {str(e)}"
            }
        )

@app.get("/api/ticker/{symbol}")
async def get_ticker(symbol: str):
    """Get ticker data for a symbol"""
    if symbol not in data_manager.symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not supported")

    cache_key = data_manager.cache_key(symbol, data_type="ticker")
    data = data_manager.get_cached_data(cache_key)

    if not data:
        # Try to fetch fresh data
        data = data_manager.fetch_ticker_data(symbol)
        if data:
            data_manager.redis_client.setex(cache_key, data_manager.cache_ttl, json.dumps(data))

    if not data:
        raise HTTPException(status_code=503, detail=f"Unable to fetch ticker data for {symbol}")

    return data

@app.get("/api/ohlcv/{symbol}")
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)"),
    limit: int = Query(100, description="Number of candles", ge=1, le=1000)
):
    """Get OHLCV data for a symbol"""
    if symbol not in data_manager.symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not supported")

    if timeframe not in data_manager.timeframes:
        raise HTTPException(status_code=400, detail=f"Timeframe {timeframe} not supported")

    cache_key = data_manager.cache_key(symbol, timeframe, "ohlcv")
    data = data_manager.get_cached_data(cache_key)

    if not data:
        # Try to fetch fresh data
        data = data_manager.fetch_ohlcv_data(symbol, timeframe, limit)
        if data:
            data_manager.redis_client.setex(cache_key, data_manager.cache_ttl, json.dumps(data))

    if not data:
        raise HTTPException(status_code=503, detail=f"Unable to fetch OHLCV data for {symbol}")

    return data[:limit] if data else []

@app.get("/api/symbols")
async def get_supported_symbols():
    """Get list of supported symbols"""
    return {
        "symbols": data_manager.symbols,
        "timeframes": data_manager.timeframes
    }

@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        info = data_manager.redis_client.info()
        keys = data_manager.redis_client.keys("viper:*")

        return {
            "total_keys": len(keys),
            "viper_keys": len([k for k in keys if k.startswith("viper:")]),
            "redis_info": {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "uptime_days": info.get("uptime_in_days", 0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get cache stats: {e}")

@app.delete("/api/cache/{pattern}")
async def clear_cache(pattern: str):
    """Clear cache entries matching pattern"""
    try:
        keys = data_manager.redis_client.keys(f"viper:{pattern}")
        if keys:
            data_manager.redis_client.delete(*keys)
            return {"cleared_keys": len(keys)}
        return {"cleared_keys": 0}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to clear cache: {e}")

if __name__ == "__main__":
    port = int(os.getenv("DATA_MANAGER_PORT", 8000))
    logger.info(f"Starting VIPER Data Manager on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
