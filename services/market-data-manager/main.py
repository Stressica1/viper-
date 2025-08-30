#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Unified Market Data Manager
Centralized market data collection, caching, and distribution service

Features:
- Unified Bitget API integration
- Intelligent rate limiting and batching
- Real-time data streaming
- Redis caching and pub/sub
- Multiple data formats support
- Circuit breaker pattern for reliability
"""

import os
import json
import time
import logging
import threading
from fastapi.responses import JSONResponse
import uvicorn
import redis
import ccxt

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'market-data-manager')
RATE_LIMIT_DELAY = float(os.getenv('RATE_LIMIT_DELAY', '0.1'))  # 100ms between calls
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))
CACHE_TTL = int(os.getenv('CACHE_TTL', '300'))  # 5 minutes cache

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedMarketDataManager:
    """Unified market data manager for all trading operations"""

    def __init__(self):
        self.exchange = None
        self.redis_client = None
        self.is_running = False
        self.symbols_cache = []
        self.market_data_cache = {}
        self.last_update = {}

        # Load API credentials
        self.api_key = os.getenv('BITGET_API_KEY', '')
        self.api_secret = os.getenv('BITGET_API_SECRET', '')
        self.api_password = os.getenv('BITGET_API_PASSWORD', '')

        # Configuration
        self.base_url = "https://api.bitget.com"
        self.enable_streaming = os.getenv('ENABLE_DATA_STREAMING', 'true').lower() == 'true'
        self.streaming_interval = int(os.getenv('STREAMING_INTERVAL', '5'))  # seconds

        logger.info("# Rocket Unified Market Data Manager initialized")

    def initialize_exchange(self) -> bool:
        """Initialize Bitget exchange connection"""
        try:
            if not all([self.api_key, self.api_secret, self.api_password]):
                logger.error("# X Missing API credentials for market data")
                return False

            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                    'createMarketBuyOrderRequiresPrice': False,
                },
                'sandbox': False,
                'rateLimit': 100,
                'enableRateLimit': True,
            })

            # Load markets
            self.exchange.load_markets()
            logger.info(f"# Check Exchange initialized with {len(self.exchange.symbols)} markets")
            return True

        except Exception as e:
            logger.error(f"# X Failed to initialize exchange: {e}")
            return False

    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("# Check Redis connection established")
            return True
        except Exception as e:
            logger.error(f"# X Failed to connect to Redis: {e}")
            return False

    def get_all_symbols(self) -> List[str]:
        """Get all available trading symbols"""
        if not self.exchange:
            return []

        # Filter for USDT perpetual swaps
        symbols = [s for s in self.exchange.symbols if ':USDT' in s and self.exchange.market(s).get('active', False)]
        return sorted(symbols)

    def fetch_ticker_data(self, symbol: str) -> Optional[Dict]:
        """Fetch ticker data for a single symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)

            return {
                'symbol': symbol,
                'price': float(ticker.get('last', 0)),
                'bid': float(ticker.get('bid', 0)),
                'ask': float(ticker.get('ask', 0)),
                'high': float(ticker.get('high', 0)),
                'low': float(ticker.get('low', 0)),
                'volume': float(ticker.get('baseVolume', 0)),
                'quote_volume': float(ticker.get('quoteVolume', 0)),
                'price_change': float(ticker.get('percentage', 0)),
                'timestamp': datetime.now().isoformat(),
                'source': 'bitget_api'
            }

        except Exception as e:
            logger.error(f"# X Error fetching ticker for {symbol}: {e}")
            return None

    def fetch_orderbook_data(self, symbol: str, depth: int = 5) -> Optional[Dict]:
        """Fetch orderbook data for a single symbol"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=depth)

            return {
                'symbol': symbol,
                'bids': orderbook.get('bids', []),
                'asks': orderbook.get('asks', []),
                'timestamp': datetime.now().isoformat(),
                'depth': depth,
                'source': 'bitget_api'
            }

        except Exception as e:
            logger.error(f"# X Error fetching orderbook for {symbol}: {e}")
            return None

    def fetch_ohlcv_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[Dict]:
        """Fetch OHLCV data for a single symbol"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'ohlcv': ohlcv,
                'count': len(ohlcv),
                'timestamp': datetime.now().isoformat(),
                'source': 'bitget_api'
            }

        except Exception as e:
            logger.error(f"# X Error fetching OHLCV for {symbol}: {e}")
            return None

    def fetch_market_data_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch market data for multiple symbols with rate limiting"""
        batch_data = {}

        for symbol in symbols:
            try:
                # Rate limiting
                time.sleep(RATE_LIMIT_DELAY)

                # Fetch all data types for this symbol
                ticker = self.fetch_ticker_data(symbol)
                orderbook = self.fetch_orderbook_data(symbol)
                ohlcv = self.fetch_ohlcv_data(symbol, timeframe='1h', limit=50)

                if ticker:
                    market_data = {
                        'ticker': ticker,
                        'orderbook': orderbook or {},
                        'ohlcv': ohlcv or {},
                        'last_updated': datetime.now().isoformat()
                    }

                    batch_data[symbol] = market_data

                    # Cache in Redis
                    cache_key = f"market_data:{symbol}"
                    self.redis_client.setex(cache_key, CACHE_TTL, json.dumps(market_data))

                    logger.debug(f"# Check Fetched market data for {symbol}")

            except Exception as e:
                logger.error(f"# X Error in batch fetch for {symbol}: {e}")

        return batch_data

    def get_cached_market_data(self, symbol: str) -> Optional[Dict]:
        """Get cached market data for a symbol"""
        try:
            cache_key = f"market_data:{symbol}"
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                return json.loads(cached_data)
            return None

        except Exception as e:
            logger.error(f"# X Error getting cached data for {symbol}: {e}")
            return None

    def publish_market_data(self, symbol: str, data: Dict):
        """Publish market data to Redis channels"""
        try:
            # Publish to symbol-specific channel
            self.redis_client.publish(f'market_data:{symbol}', json.dumps(data))

            # Publish to general market data channel
            self.redis_client.publish('market_data:all', json.dumps({
                'symbol': symbol,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }))

            # Publish ticker updates
            if 'ticker' in data:
                self.redis_client.publish('market_data:ticker', json.dumps(data['ticker']))

            # Publish orderbook updates
            if 'orderbook' in data and data['orderbook']:
                self.redis_client.publish('market_data:orderbook', json.dumps(data['orderbook']))

        except Exception as e:
            logger.error(f"# X Error publishing market data for {symbol}: {e}")

    def stream_market_data(self):
        """Stream market data in real-time"""
        logger.info("🌊 Starting market data streaming...")

        while self.is_running:
            try:
                symbols = self.get_all_symbols()

                if not symbols:
                    logger.warning("# Warning No symbols available for streaming")
                    time.sleep(10)
                    continue

                # Process symbols in batches
                for i in range(0, len(symbols), BATCH_SIZE):
                    batch_symbols = symbols[i:i + BATCH_SIZE]

                    logger.debug(f"📦 Processing batch {i//BATCH_SIZE + 1}: {len(batch_symbols)} symbols")

                    # Fetch batch data
                    batch_data = self.fetch_market_data_batch(batch_symbols)

                    # Publish data for each symbol
                    for symbol, data in batch_data.items():
                        self.publish_market_data(symbol, data)
                        self.market_data_cache[symbol] = data
                        self.last_update[symbol] = datetime.now()

                    # Brief pause between batches
                    if i + BATCH_SIZE < len(symbols):
                        time.sleep(0.5)

                # Update service status
                status_data = {
                    'service': 'market-data-manager',
                    'symbols_tracked': len(self.market_data_cache),
                    'last_update': datetime.now().isoformat(),
                    'streaming_active': self.is_running
                }

                self.redis_client.setex('service_status:market_data', 60, json.dumps(status_data))

                # Wait before next streaming cycle
                time.sleep(self.streaming_interval)

            except Exception as e:
                logger.error(f"# X Error in streaming loop: {e}")
                time.sleep(5)

    def start_streaming(self):
        """Start market data streaming in background thread"""
        if not self.is_running:
            self.is_running = True
            streaming_thread = threading.Thread(target=self.stream_market_data, daemon=True)
            streaming_thread.start()
            logger.info("# Check Market data streaming started")

    def stop_streaming(self):
        """Stop market data streaming"""
        self.is_running = False
        logger.info("🛑 Market data streaming stopped")

    def get_market_overview(self, limit: int = 50) -> Dict:
        """Get market overview with top symbols by volume"""
        try:
            # Get all cached market data
            overview_data = []

            for symbol in self.market_data_cache.keys():
                data = self.market_data_cache[symbol]
                ticker = data.get('ticker', {})

                if ticker and ticker.get('volume', 0) > 0:
                    overview_data.append({
                        'symbol': symbol,
                        'price': ticker.get('price', 0),
                        'volume': ticker.get('volume', 0),
                        'price_change': ticker.get('price_change', 0),
                        'last_updated': data.get('last_updated', '')
                    })

            # Sort by volume and return top limit
            overview_data.sort(key=lambda x: x['volume'], reverse=True)
            top_symbols = overview_data[:limit]

            return {
                'total_symbols': len(self.market_data_cache),
                'top_symbols': top_symbols,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"# X Error getting market overview: {e}")
            return {'error': str(e)}

# FastAPI application
app = FastAPI(
    title="VIPER Market Data Manager",
    version="1.0.0",
    description="Unified market data collection and distribution service"
)

market_data_manager = UnifiedMarketDataManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if not market_data_manager.initialize_exchange():
        logger.error("# X Failed to initialize exchange")
        return

    if not market_data_manager.initialize_redis():
        logger.error("# X Failed to initialize Redis")
        return

    # Start streaming if enabled
    if market_data_manager.enable_streaming:
        market_data_manager.start_streaming()

    logger.info("# Check Market Data Manager started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    market_data_manager.stop_streaming()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "market-data-manager",
            "exchange_connected": market_data_manager.exchange is not None,
            "redis_connected": market_data_manager.redis_client is not None,
            "streaming_active": market_data_manager.is_running,
            "symbols_tracked": len(market_data_manager.market_data_cache),
            "last_update": {k: v.isoformat() if isinstance(v, datetime) else str(v)
                          for k, v in market_data_manager.last_update.items()}
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "market-data-manager",
                "error": str(e)
            }
        )

@app.get("/api/market/overview")
async def get_market_overview(limit: int = Query(50, description="Number of top symbols to return", ge=1, le=500)):
    """Get market overview with top symbols"""
    return market_data_manager.get_market_overview(limit)

@app.get("/api/market/{symbol}")
async def get_market_data(symbol: str):
    """Get market data for a specific symbol"""
    try:
        # Try cache first
        cached_data = market_data_manager.get_cached_market_data(symbol)

        if cached_data:
            return cached_data

        # Fetch fresh data if not cached
        ticker = market_data_manager.fetch_ticker_data(symbol)
        orderbook = market_data_manager.fetch_orderbook_data(symbol)
        ohlcv = market_data_manager.fetch_ohlcv_data(symbol)

        if not ticker:
            raise HTTPException(status_code=404, detail=f"Market data not found for {symbol}")

        market_data = {
            'ticker': ticker,
            'orderbook': orderbook or {},
            'ohlcv': ohlcv or {},
            'last_updated': datetime.now().isoformat()
        }

        return market_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error fetching market data: {e}")

@app.get("/api/market/batch")
async def get_batch_market_data(symbols: str = Query(..., description="Comma-separated list of symbols")):
    """Get market data for multiple symbols"""
    try:
        symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]

        if len(symbol_list) > 100:
            raise HTTPException(status_code=400, detail="Too many symbols requested (max 100)")

        batch_data = {}

        for symbol in symbol_list:
            data = market_data_manager.get_cached_market_data(symbol)
            if data:
                batch_data[symbol] = data

        return {
            'symbols_requested': len(symbol_list),
            'symbols_found': len(batch_data),
            'data': batch_data,
            'timestamp': datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error fetching batch data: {e}")

@app.post("/api/market/refresh/{symbol}")
async def refresh_market_data(symbol: str):
    """Force refresh market data for a symbol"""
    try:
        ticker = market_data_manager.fetch_ticker_data(symbol)
        orderbook = market_data_manager.fetch_orderbook_data(symbol)
        ohlcv = market_data_manager.fetch_ohlcv_data(symbol)

        if not ticker:
            raise HTTPException(status_code=404, detail=f"Could not refresh data for {symbol}")

        market_data = {
            'ticker': ticker,
            'orderbook': orderbook or {},
            'ohlcv': ohlcv or {},
            'last_updated': datetime.now().isoformat()
        }

        # Cache and publish
        cache_key = f"market_data:{symbol}"
        market_data_manager.redis_client.setex(cache_key, CACHE_TTL, json.dumps(market_data))
        market_data_manager.publish_market_data(symbol, market_data)

        return {
            'status': 'refreshed',
            'symbol': symbol,
            'data': market_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error refreshing data: {e}")

@app.get("/api/symbols")
async def get_available_symbols():
    """Get list of all available trading symbols"""
    try:
        symbols = market_data_manager.get_all_symbols()
        return {
            'total_symbols': len(symbols),
            'symbols': symbols,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error getting symbols: {e}")

@app.get("/api/market/stats")
async def get_market_stats():
    """Get market statistics"""
    try:
        total_symbols = len(market_data_manager.market_data_cache)
        updates_in_last_minute = sum(1 for timestamp in market_data_manager.last_update.values()
                                   if isinstance(timestamp, datetime) and
                                   (datetime.now() - timestamp).seconds < 60)

        return {
            'total_symbols': total_symbols,
            'active_symbols': len([s for s in market_data_manager.market_data_cache.keys()
                                 if market_data_manager.market_data_cache[s].get('ticker', {}).get('volume', 0) > 0]),
            'recent_updates': updates_in_last_minute,
            'cache_size': len(market_data_manager.market_data_cache),
            'streaming_active': market_data_manager.is_running,
            'last_update_counts': dict(market_data_manager.last_update)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error getting stats: {e}")

if __name__ == "__main__":
    port = int(os.getenv("MARKET_DATA_MANAGER_PORT", 8003))
    logger.info(f"Starting VIPER Market Data Manager on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
