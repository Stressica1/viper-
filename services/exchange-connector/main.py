#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Exchange Connector Service
Unified Bitget API client with rate limiting, order management, and error handling

Features:
- Bitget API integration with automatic retry logic
- Rate limiting and request throttling
- Order management and position tracking
- Account balance and position monitoring
- RESTful API for trading operations
"""

import os
import json
import time
import logging
import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import ccxt
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import uvicorn
import redis
from pathlib import Path
import threading
import hashlib
import hmac

# Add shared directory to path for credential client
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
try:
    from credential_client import get_credential_client, get_bitget_credentials
    CREDENTIAL_CLIENT_AVAILABLE = True
except ImportError:
    CREDENTIAL_CLIENT_AVAILABLE = False

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'exchange-connector')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API requests"""

    def __init__(self, requests_per_second: float = 2.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.request_count = 0
        self.window_start = time.time()

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()

        # Reset counter every second
        if current_time - self.window_start >= 1.0:
            self.request_count = 0
            self.window_start = current_time

        # Check if we need to wait
        if self.request_count >= self.requests_per_second:
            sleep_time = 1.0 - (current_time - self.window_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.request_count = 0
            self.window_start = time.time()

        # Ensure minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)

        self.last_request_time = time.time()
        self.request_count += 1

class ExchangeConnector:
    """Exchange connector for Bitget API interactions"""

    def __init__(self):
        self.exchange = None
        self.redis_client = None
        self.rate_limiter = RateLimiter(requests_per_second=2.0)
        self.is_running = False

        # Load configuration
        self.api_key = ''
        self.api_secret = ''
        self.api_password = ''
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        self.rate_limit_buffer = float(os.getenv('RATE_LIMIT_BUFFER', '0.1'))

        # Initialize credential client
        self.credential_client = None
        if CREDENTIAL_CLIENT_AVAILABLE:
            self.credential_client = get_credential_client()

        # Trading parameters
        self.supported_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT']
        self.active_orders = {}

        logger.info("🏗️ Initializing Exchange Connector...")

    async def load_credentials(self) -> bool:
        """Load API credentials from the credential vault"""
        if not CREDENTIAL_CLIENT_AVAILABLE or not self.credential_client:
            logger.warning("⚠️ Credential client not available, using environment variables")
            self.api_key = os.getenv('BITGET_API_KEY', '')
            self.api_secret = os.getenv('BITGET_API_SECRET', '')
            self.api_password = os.getenv('BITGET_API_PASSWORD', '')
            return True

        try:
            credentials = await self.credential_client.get_bitget_credentials()

            if credentials:
                self.api_key = credentials.get('api_key', '')
                self.api_secret = credentials.get('api_secret', '')
                self.api_password = credentials.get('api_password', '')
                logger.info("✅ Successfully loaded credentials from vault")
                return True
            else:
                logger.warning("⚠️ No credentials found in vault, using environment variables")
                self.api_key = os.getenv('BITGET_API_KEY', '')
                self.api_secret = os.getenv('BITGET_API_SECRET', '')
                self.api_password = os.getenv('BITGET_API_PASSWORD', '')
                return True

        except Exception as e:
            logger.error(f"❌ Failed to load credentials from vault: {e}")
            # Fallback to environment variables
            self.api_key = os.getenv('BITGET_API_KEY', '')
            self.api_secret = os.getenv('BITGET_API_SECRET', '')
            self.api_password = os.getenv('BITGET_API_PASSWORD', '')
            return True

    def load_credentials_sync(self) -> bool:
        """Synchronous version for non-async contexts"""
        if not CREDENTIAL_CLIENT_AVAILABLE or not self.credential_client:
            logger.warning("⚠️ Credential client not available, using environment variables")
            self.api_key = os.getenv('BITGET_API_KEY', '')
            self.api_secret = os.getenv('BITGET_API_SECRET', '')
            self.api_password = os.getenv('BITGET_API_PASSWORD', '')
            return True

        try:
            credentials = self.credential_client.get_bitget_credentials_sync()

            if credentials:
                self.api_key = credentials.get('api_key', '')
                self.api_secret = credentials.get('api_secret', '')
                self.api_password = credentials.get('api_password', '')
                logger.info("✅ Successfully loaded credentials from vault")
                return True
            else:
                logger.warning("⚠️ No credentials found in vault, using environment variables")
                self.api_key = os.getenv('BITGET_API_KEY', '')
                self.api_secret = os.getenv('BITGET_API_SECRET', '')
                self.api_password = os.getenv('BITGET_API_PASSWORD', '')
                return True

        except Exception as e:
            logger.error(f"❌ Failed to load credentials from vault: {e}")
            # Fallback to environment variables
            self.api_key = os.getenv('BITGET_API_KEY', '')
            self.api_secret = os.getenv('BITGET_API_SECRET', '')
            self.api_password = os.getenv('BITGET_API_PASSWORD', '')
            return True

    def initialize_exchange(self) -> bool:
        """Initialize Bitget exchange connection"""
        try:
            if not all([self.api_key, self.api_secret, self.api_password]):
                logger.warning("⚠️ API credentials not provided - running in read-only mode")
                self.exchange = ccxt.bitget({
                    'options': {
                        'defaultType': 'swap',
                        'adjustForTimeDifference': True,
                    },
                    'sandbox': False,
                })
            else:
                self.exchange = ccxt.bitget({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'password': self.api_password,
                    'options': {
                        'defaultType': 'swap',
                        'adjustForTimeDifference': True,
                    },
                    'sandbox': False,
                })

            self.exchange.load_markets()
            logger.info("✅ Exchange connection established")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange: {e}")
            return False

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

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported"""
        return symbol in self.supported_symbols

    def get_account_balance(self) -> Optional[Dict]:
        """Get account balance"""
        try:
            self.rate_limiter.wait_if_needed()
            balance = self.exchange.fetch_balance()

            # Extract USDT balance
            usdt_balance = balance.get('USDT', {})
            return {
                'total': float(usdt_balance.get('total', 0)),
                'free': float(usdt_balance.get('free', 0)),
                'used': float(usdt_balance.get('used', 0)),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
            return None

    def get_positions(self) -> Optional[List]:
        """Get current positions"""
        try:
            self.rate_limiter.wait_if_needed()
            positions = self.exchange.fetch_positions()

            formatted_positions = []
            for position in positions:
                if position['contracts'] > 0:  # Only include open positions
                    formatted_positions.append({
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'size': position['contracts'],
                        'entry_price': position['entryPrice'],
                        'mark_price': position['markPrice'],
                        'unrealized_pnl': position['unrealizedPnl'],
                        'leverage': position['leverage'],
                        'timestamp': datetime.now().isoformat()
                    })

            return formatted_positions
        except Exception as e:
            logger.error(f"❌ Failed to fetch positions: {e}")
            return None

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker data for symbol"""
        if not self.validate_symbol(symbol):
            return None

        try:
            self.rate_limiter.wait_if_needed()
            ticker = self.exchange.fetch_ticker(symbol)

            return {
                'symbol': symbol,
                'price': ticker.get('last', 0),
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'volume': ticker.get('baseVolume', 0),
                'high': ticker.get('high', 0),
                'low': ticker.get('low', 0),
                'open': ticker.get('open', 0),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Failed to fetch ticker for {symbol}: {e}")
            return None

    def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Get order book for symbol"""
        if not self.validate_symbol(symbol):
            return None

        try:
            self.rate_limiter.wait_if_needed()
            order_book = self.exchange.fetch_order_book(symbol, limit)

            return {
                'symbol': symbol,
                'bids': order_book.get('bids', []),
                'asks': order_book.get('asks', []),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Failed to fetch order book for {symbol}: {e}")
            return None

    def create_order(self, symbol: str, side: str, order_type: str,
                    amount: float, price: Optional[float] = None) -> Optional[Dict]:
        """Create a new order"""
        if not self.validate_symbol(symbol):
            return None

        if not all([self.api_key, self.api_secret, self.api_password]):
            logger.error("❌ Cannot create orders without API credentials")
            return None

        try:
            self.rate_limiter.wait_if_needed()

            # Create order
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side.lower(),
                amount=amount,
                price=price
            )

            # Store order in active orders
            order_id = order['id']
            self.active_orders[order_id] = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'amount': amount,
                'price': price,
                'status': order.get('status', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"✅ Order created: {order_id} - {side.upper()} {amount} {symbol}")

            return {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'amount': amount,
                'price': price,
                'status': order.get('status', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Failed to create {side} order for {symbol}: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict]:
        """Cancel an existing order"""
        if not self.validate_symbol(symbol):
            return None

        try:
            self.rate_limiter.wait_if_needed()
            result = self.exchange.cancel_order(order_id, symbol)

            # Remove from active orders if present
            if order_id in self.active_orders:
                del self.active_orders[order_id]

            logger.info(f"✅ Order cancelled: {order_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to cancel order {order_id}: {e}")
            return None

    def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict]:
        """Get order status"""
        if not self.validate_symbol(symbol):
            return None

        try:
            self.rate_limiter.wait_if_needed()
            order = self.exchange.fetch_order(order_id, symbol)

            return {
                'order_id': order['id'],
                'symbol': symbol,
                'side': order.get('side', ''),
                'type': order.get('type', ''),
                'amount': order.get('amount', 0),
                'filled': order.get('filled', 0),
                'remaining': order.get('remaining', 0),
                'price': order.get('price', 0),
                'average': order.get('average', 0),
                'status': order.get('status', ''),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Failed to get order status for {order_id}: {e}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> Optional[List]:
        """Get open orders"""
        try:
            self.rate_limiter.wait_if_needed()
            orders = self.exchange.fetch_open_orders(symbol)

            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'order_id': order['id'],
                    'symbol': order['symbol'],
                    'side': order.get('side', ''),
                    'type': order.get('type', ''),
                    'amount': order.get('amount', 0),
                    'price': order.get('price', 0),
                    'status': order.get('status', ''),
                    'timestamp': datetime.now().isoformat()
                })

            return formatted_orders

        except Exception as e:
            logger.error(f"❌ Failed to fetch open orders: {e}")
            return None

    def start_monitoring(self):
        """Start order monitoring loop"""
        logger.info("🚀 Starting order monitoring...")
        self.is_running = True

        while self.is_running:
            try:
                # Update active orders status
                for order_id, order_info in list(self.active_orders.items()):
                    status = self.get_order_status(order_id, order_info['symbol'])
                    if status and status['status'] in ['closed', 'canceled', 'expired']:
                        logger.info(f"📊 Order {order_id} completed with status: {status['status']}")
                        del self.active_orders[order_id]

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(10)

    def stop(self):
        """Stop the connector"""
        logger.info("🛑 Stopping Exchange Connector...")
        self.is_running = False

# FastAPI application
app = FastAPI(
    title="VIPER Exchange Connector",
    version="1.0.0",
    description="Bitget API client and order management service"
)

connector = ExchangeConnector()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Load credentials from vault first
    if not await connector.load_credentials():
        logger.error("❌ Failed to load credentials. Exiting...")
        return

    if not connector.initialize_exchange():
        logger.error("❌ Failed to initialize exchange. Exiting...")
        return

    if not connector.initialize_redis():
        logger.warning("⚠️ Failed to initialize Redis. Continuing without caching...")

    # Start monitoring in background thread
    thread = threading.Thread(target=connector.start_monitoring, daemon=True)
    thread.start()
    logger.info("✅ Exchange Connector started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    connector.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "exchange-connector",
            "exchange_connected": connector.exchange is not None,
            "redis_connected": connector.redis_client is not None,
            "api_credentials": bool(all([connector.api_key, connector.api_secret, connector.api_password])),
            "active_orders": len(connector.active_orders),
            "monitoring_running": connector.is_running
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "exchange-connector",
                "error": str(e)
            }
        )

@app.get("/api/balance")
async def get_balance():
    """Get account balance"""
    balance = connector.get_account_balance()
    if balance is None:
        raise HTTPException(status_code=503, detail="Unable to fetch balance")
    return balance

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    positions = connector.get_positions()
    if positions is None:
        raise HTTPException(status_code=503, detail="Unable to fetch positions")
    return {"positions": positions}

@app.get("/api/ticker/{symbol}")
async def get_ticker(symbol: str):
    """Get ticker data"""
    if not connector.validate_symbol(symbol):
        raise HTTPException(status_code=400, detail=f"Symbol {symbol} not supported")

    ticker = connector.get_ticker(symbol)
    if ticker is None:
        raise HTTPException(status_code=503, detail=f"Unable to fetch ticker for {symbol}")
    return ticker

@app.get("/api/orderbook/{symbol}")
async def get_order_book(
    symbol: str,
    limit: int = Query(20, description="Number of levels", ge=1, le=100)
):
    """Get order book"""
    if not connector.validate_symbol(symbol):
        raise HTTPException(status_code=400, detail=f"Symbol {symbol} not supported")

    order_book = connector.get_order_book(symbol, limit)
    if order_book is None:
        raise HTTPException(status_code=503, detail=f"Unable to fetch order book for {symbol}")
    return order_book

@app.post("/api/orders")
async def create_order(request: Request):
    """Create a new order"""
    try:
        data = await request.json()

        required_fields = ['symbol', 'side', 'type', 'amount']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        symbol = data['symbol']
        side = data['side']
        order_type = data['type']
        amount = float(data['amount'])
        price = data.get('price')

        if side.lower() not in ['buy', 'sell']:
            raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'")

        if order_type.lower() not in ['market', 'limit']:
            raise HTTPException(status_code=400, detail="Type must be 'market' or 'limit'")

        if order_type.lower() == 'limit' and price is None:
            raise HTTPException(status_code=400, detail="Price required for limit orders")

        result = connector.create_order(symbol, side, order_type, amount, price)
        if result is None:
            raise HTTPException(status_code=503, detail="Failed to create order")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating order: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: str, symbol: str):
    """Cancel an order"""
    result = connector.cancel_order(order_id, symbol)
    if result is None:
        raise HTTPException(status_code=503, detail=f"Failed to cancel order {order_id}")
    return {"status": "cancelled", "order_id": order_id}

@app.get("/api/orders/{order_id}")
async def get_order_status(order_id: str, symbol: str):
    """Get order status"""
    status = connector.get_order_status(order_id, symbol)
    if status is None:
        raise HTTPException(status_code=503, detail=f"Unable to get status for order {order_id}")
    return status

@app.get("/api/orders")
async def get_open_orders(symbol: Optional[str] = None):
    """Get open orders"""
    orders = connector.get_open_orders(symbol)
    if orders is None:
        raise HTTPException(status_code=503, detail="Unable to fetch open orders")
    return {"orders": orders}

@app.get("/api/symbols")
async def get_supported_symbols():
    """Get supported symbols"""
    return {"symbols": connector.supported_symbols}

if __name__ == "__main__":
    port = int(os.getenv("EXCHANGE_CONNECTOR_PORT", 8000))
    logger.info(f"Starting VIPER Exchange Connector on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
