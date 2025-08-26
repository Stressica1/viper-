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

        # Configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        self.cache_ttl = int(os.getenv('CACHE_TTL_SECONDS', '300'))
        self.update_interval = int(os.getenv('UPDATE_INTERVAL_SECONDS', '60'))

        # Supported symbols and timeframes
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT']
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

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
        """Update ticker data in cache"""
        try:
            data = self.fetch_ticker_data(symbol)
            if data:
                cache_key = self.cache_key(symbol, data_type="ticker")
                self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(data))
                logger.debug(f"📊 Updated ticker cache for {symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update ticker cache for {symbol}: {e}")
            return False

    def update_ohlcv_cache(self, symbol: str, timeframe: str) -> bool:
        """Update OHLCV data in cache"""
        try:
            data = self.fetch_ohlcv_data(symbol, timeframe)
            if data:
                cache_key = self.cache_key(symbol, timeframe, "ohlcv")
                self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(data))
                logger.debug(f"📊 Updated OHLCV cache for {symbol} {timeframe}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update OHLCV cache for {symbol} {timeframe}: {e}")
            return False

    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache"""
        try:
            data = self.redis_client.get(cache_key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get cached data for key {cache_key}: {e}")
            return None

    def start_data_collection(self):
        """Start the data collection loop"""
        logger.info("🚀 Starting data collection loop...")
        self.is_running = True

        # Schedule data updates
        schedule.every(self.update_interval).seconds.do(self.update_all_data)
        schedule.every(30).seconds.do(self.update_all_tickers)

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
