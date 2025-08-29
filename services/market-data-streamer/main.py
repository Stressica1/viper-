#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Market Data Streamer
Real-time market data streaming and processing service

Features:
- WebSocket connections to Bitget
- Real-time price feeds
- Order book data
- Trade tickers
- Market statistics
- Redis pub/sub distribution
"""

import os
import time
import json
import logging
import asyncio
import threading
from datetime import datetime
import redis
import requests
import ccxt

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'market-data-streamer')
VAULT_URL = os.getenv('VAULT_URL', 'http://credential-vault:8008')
VAULT_ACCESS_TOKEN = os.getenv('VAULT_ACCESS_TOKEN', '')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataStreamer:
    """Real-time market data streaming service"""

    def __init__(self):
        self.redis_client = None
        self.exchange = None
        self.is_running = False
        self.active_streams = {}
        self.subscribed_symbols = set()
        self.last_update = {}

        # Configuration
        self.update_interval = float(os.getenv('UPDATE_INTERVAL', '1.0'))
        self.max_reconnect_attempts = int(os.getenv('MAX_RECONNECT_ATTEMPTS', '5'))
        self.reconnect_delay = float(os.getenv('RECONNECT_DELAY', '5.0'))

    def connect_services(self):
        """Connect to Redis and exchange"""
        try:
            # Connect to Redis
            self.redis_client = redis.Redis.from_url(REDIS_URL)
            self.redis_client.ping()
            logger.info("# Check Connected to Redis")

            # Load exchange credentials
            self.load_exchange_credentials()

            # Initialize exchange for swap trading
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                    'watchOrderBook': True,
                    'watchTicker': True,
                    'watchTrades': True,
                }
            })
            logger.info("# Check Connected to Bitget exchange")

        except Exception as e:
            logger.error(f"# X Failed to connect services: {e}")
            raise

    def load_exchange_credentials(self):
        """Load API credentials from vault"""
        try:
            response = requests.get(
                f"{VAULT_URL}/credentials/retrieve/bitget/api_key",
                headers={'Authorization': f'Bearer {VAULT_ACCESS_TOKEN}'}
            )
            self.api_key = response.json().get('value')

            response = requests.get(
                f"{VAULT_URL}/credentials/retrieve/bitget/api_secret",
                headers={'Authorization': f'Bearer {VAULT_ACCESS_TOKEN}'}
            )
            self.api_secret = response.json().get('value')

            response = requests.get(
                f"{VAULT_URL}/credentials/retrieve/bitget/api_password",
                headers={'Authorization': f'Bearer {VAULT_ACCESS_TOKEN}'}
            )
            self.api_password = response.json().get('value')

            logger.info("# Check Loaded exchange credentials from vault")

        except Exception as e:
            logger.error(f"# X Failed to load credentials: {e}")
            raise

    async def stream_market_data(self, symbol: str):
        """Stream real-time market data for a symbol"""
        reconnect_count = 0

        while self.is_running and reconnect_count < self.max_reconnect_attempts:
            try:
                # Watch order book
                orderbook_stream = await self.exchange.watch_order_book(symbol)

                # Watch ticker
                ticker_stream = await self.exchange.watch_ticker(symbol)

                # Watch trades
                trades_stream = await self.exchange.watch_trades(symbol)

                logger.info(f"📡 Streaming data for {symbol}")

                while self.is_running:
                    # Process order book updates
                    if orderbook_stream:
                        orderbook_data = {
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'orderbook',
                            'data': orderbook_stream
                        }
                        self.redis_client.publish('market_data:orderbook', json.dumps(orderbook_data))
                        self.cache_data(f'orderbook:{symbol}', orderbook_data)

                    # Process ticker updates
                    if ticker_stream:
                        ticker_data = {
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'ticker',
                            'data': ticker_stream
                        }
                        self.redis_client.publish('market_data:ticker', json.dumps(ticker_data))
                        self.cache_data(f'ticker:{symbol}', ticker_data)

                    # Process trade updates
                    if trades_stream:
                        for trade in trades_stream:
                            trade_data = {
                                'symbol': symbol,
                                'timestamp': datetime.now().isoformat(),
                                'type': 'trade',
                                'data': trade
                            }
                            self.redis_client.publish('market_data:trades', json.dumps(trade_data))

                    await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"# X Error streaming {symbol}: {e}")
                reconnect_count += 1
                if reconnect_count < self.max_reconnect_attempts:
                    logger.info(f"🔄 Reconnecting in {self.reconnect_delay}s... ({reconnect_count}/{self.max_reconnect_attempts})")
                    await asyncio.sleep(self.reconnect_delay)

    def cache_data(self, key: str, data: Dict):
        """Cache market data in Redis"""
        try:
            self.redis_client.setex(key, 300, json.dumps(data))  # 5 minute cache
        except Exception as e:
            logger.error(f"# X Failed to cache data: {e}")

    async def start_streaming(self, symbols: List[str]):
        """Start streaming for multiple symbols"""
        logger.info(f"# Rocket Starting market data streaming for {len(symbols)} symbols")

        tasks = []
        for symbol in symbols:
            if symbol not in self.active_streams:
                task = asyncio.create_task(self.stream_market_data(symbol))
                self.active_streams[symbol] = task
                tasks.append(task)
                self.subscribed_symbols.add(symbol)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def start_background_streaming(self, symbols: List[str]):
        """Start streaming in background thread"""
        def run_streaming():
            asyncio.run(self.start_streaming(symbols))

        thread = threading.Thread(target=run_streaming, daemon=True)
        thread.start()
        logger.info("# Target Market data streaming started in background")

    def get_cached_data(self, symbol: str, data_type: str) -> Optional[Dict]:
        """Get cached market data"""
        try:
            key = f'{data_type}:{symbol}'
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"# X Failed to get cached data: {e}")
            return None

    def start(self):
        """Start the market data streamer"""
        try:
            logger.info("# Rocket Starting Market Data Streamer...")

            # Connect to services
            self.connect_services()

            # Get trading pairs from exchange
            markets = self.exchange.load_markets()
            symbols = [symbol for symbol in markets.keys() if 'USDT' in symbol and markets[symbol]['active']]

            # Filter for 25x leverage pairs only
            leverage_pairs = self.filter_leverage_pairs(symbols[:100])  # Limit to first 100 for now

            logger.info(f"# Chart Found {len(leverage_pairs)} leverage pairs to stream")

            # Start streaming
            self.is_running = True
            self.start_background_streaming(leverage_pairs)

            # Keep main thread alive
            while self.is_running:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("⏹️ Stopping Market Data Streamer...")
            self.stop()
        except Exception as e:
            logger.error(f"# X Market Data Streamer error: {e}")
            self.stop()

    def filter_leverage_pairs(self, symbols: List[str]) -> List[str]:
        """Filter symbols for 25x leverage support"""
        leverage_pairs = []
        for symbol in symbols:
            try:
                # Check if pair supports leverage
                leverage_info = self.exchange.fetch_leverage_tiers(symbol)
                if leverage_info and any(tier.get('maxLeverage', 0) >= 25 for tier in leverage_info):
                    leverage_pairs.append(symbol)
            except Exception as e:
                logger.debug(f"Could not check leverage for {symbol}: {e}")
                continue

        return leverage_pairs

    def stop(self):
        """Stop the market data streamer"""
        self.is_running = False
        logger.info("# Check Market Data Streamer stopped")

def create_app():
    """Create FastAPI application for health checks and metrics"""
    from fastapi import FastAPI

    app = FastAPI(title="Market Data Streamer", version="1.0.0")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "market-data-streamer"}

    @app.get("/metrics")
    async def get_metrics():
        return {
            "active_streams": len(streamer.active_streams),
            "subscribed_symbols": list(streamer.subscribed_symbols),
            "last_update": streamer.last_update
        }

    return app

if __name__ == "__main__":
    # Check if running as API server or streamer
    if os.getenv('API_MODE', 'false').lower() == 'true':
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run as streaming service
        streamer = MarketDataStreamer()
        streamer.start()
