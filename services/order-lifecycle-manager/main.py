#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Order Lifecycle Manager
Complete order management from signal to execution and monitoring

Features:
- Signal processing and validation
- Order creation and submission
- Execution monitoring and tracking
- Position management and synchronization
- Risk validation at each step
- Event-driven order lifecycle
"""

import os
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import redis
import requests
import ccxt

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'order-lifecycle-manager')
VAULT_URL = os.getenv('VAULT_URL', 'http://credential-vault:8008')
VAULT_ACCESS_TOKEN = os.getenv('VAULT_ACCESS_TOKEN', '')

# Service URLs
RISK_MANAGER_URL = os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8000')
EXCHANGE_CONNECTOR_URL = os.getenv('EXCHANGE_CONNECTOR_URL', 'http://exchange-connector:8000')
DATA_MANAGER_URL = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')

# Order configuration
MAX_SLIPPAGE = float(os.getenv('MAX_SLIPPAGE', '0.001'))  # 0.1%
ORDER_TIMEOUT = int(os.getenv('ORDER_TIMEOUT', '300'))  # 5 minutes
RETRY_ATTEMPTS = int(os.getenv('RETRY_ATTEMPTS', '3'))

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderLifecycleManager:
    """Complete order lifecycle management service"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False
        self.active_orders = {}  # Track active orders
        self.order_history = []  # Store order history
        self.position_cache = {}  # Cache current positions

        # Statistics
        self.stats = {
            'orders_processed': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'total_volume': 0.0,
            'success_rate': 0.0
        }

    def connect_services(self):
        """Connect to Redis and external services"""
        try:
            # Connect to Redis
            self.redis_client = redis.Redis.from_url(REDIS_URL)
            self.redis_client.ping()
            logger.info("✅ Connected to Redis")

            # Load exchange credentials
            self.load_exchange_credentials()

            logger.info("✅ Connected to all services")

        except Exception as e:
            logger.error(f"❌ Failed to connect services: {e}")
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

            logger.info("✅ Loaded exchange credentials from vault")

        except Exception as e:
            logger.error(f"❌ Failed to load credentials: {e}")
            raise

    def validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal before processing"""
        try:
            required_fields = ['symbol', 'type', 'price', 'viper_score', 'confidence', 'timestamp']
            for field in required_fields:
                if field not in signal:
                    logger.error(f"❌ Missing required field: {field}")
                    return False

            # Check VIPER score threshold
            viper_threshold = float(os.getenv('VIPER_THRESHOLD', '85'))
            if signal.get('viper_score', 0) < viper_threshold:
                logger.info(f"📊 Signal rejected: VIPER score {signal['viper_score']} < {viper_threshold}")
                return False

            # Check confidence threshold
            confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '80'))
            if signal.get('confidence', 0) < confidence_threshold:
                logger.info(f"📊 Signal rejected: Confidence {signal['confidence']} < {confidence_threshold}")
                return False

            # Check symbol is supported
            supported_symbols = self.get_supported_symbols()
            if signal['symbol'] not in supported_symbols:
                logger.error(f"❌ Symbol not supported: {signal['symbol']}")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Error validating signal: {e}")
            return False

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        try:
            # Get from data manager or cache
            cache_key = 'supported_symbols'
            cached = self.redis_client.get(cache_key)

            if cached:
                return json.loads(cached)

            # Fetch from data manager
            response = requests.get(f"{DATA_MANAGER_URL}/symbols", timeout=10)
            if response.status_code == 200:
                symbols = response.json().get('symbols', [])
                # Cache for 1 hour
                self.redis_client.setex(cache_key, 3600, json.dumps(symbols))
                return symbols

            # Fallback to common symbols
            return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'ADA/USDT:USDT']

        except Exception as e:
            logger.error(f"❌ Error getting supported symbols: {e}")
            return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'ADA/USDT:USDT']

    def calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal and risk management"""
        try:
            # Request position size from risk manager
            risk_request = {
                'symbol': signal['symbol'],
                'price': signal['price'],
                'signal_type': signal['type'],
                'viper_score': signal['viper_score'],
                'confidence': signal['confidence']
            }

            response = requests.post(
                f"{RISK_MANAGER_URL}/api/position/size",
                json=risk_request,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                position_size = result.get('position_size', 0)

                # Apply slippage protection
                max_position = result.get('max_position_size', 0)
                position_size = min(position_size, max_position * (1 - MAX_SLIPPAGE))

                logger.info(f"📊 Calculated position size: {position_size} for {signal['symbol']}")
                return position_size
            else:
                logger.error(f"❌ Risk manager error: {response.text}")
                return 0

        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0

    def create_order(self, signal: Dict, position_size: float) -> Dict:
        """Create order object from signal"""
        try:
            order_id = f"viper_{signal['symbol'].replace('/', '_')}_{int(datetime.now().timestamp())}"

            order = {
                'order_id': order_id,
                'symbol': signal['symbol'],
                'type': signal['type'],
                'side': 'buy' if signal['type'] == 'LONG' else 'sell',
                'amount': position_size,
                'price': signal['price'],
                'status': OrderStatus.PENDING.value,
                'created_at': datetime.now().isoformat(),
                'signal_data': signal,
                'retry_count': 0,
                'exchange_order_id': None
            }

            logger.info(f"📋 Created order: {order_id} for {signal['symbol']}")
            return order

        except Exception as e:
            logger.error(f"❌ Error creating order: {e}")
            return None

    def submit_order(self, order: Dict) -> bool:
        """Submit order to exchange"""
        try:
            # Update order status
            order['status'] = OrderStatus.VALIDATING.value
            self.active_orders[order['order_id']] = order

            # Submit to exchange connector
            exchange_request = {
                'order_id': order['order_id'],
                'symbol': order['symbol'],
                'type': 'market',  # Use market orders for speed
                'side': order['side'],
                'amount': order['amount'],
                'price': order['price']
            }

            response = requests.post(
                f"{EXCHANGE_CONNECTOR_URL}/orders",
                json=exchange_request,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                exchange_order_id = result.get('exchange_order_id')

                if exchange_order_id:
                    order['exchange_order_id'] = exchange_order_id
                    order['status'] = OrderStatus.SUBMITTED.value
                    order['submitted_at'] = datetime.now().isoformat()

                    # Publish order update
                    self.redis_client.publish('order_updates', json.dumps({
                        'order_id': order['order_id'],
                        'status': order['status'],
                        'exchange_order_id': exchange_order_id,
                        'timestamp': datetime.now().isoformat()
                    }))

                    logger.info(f"📤 Order submitted: {order['order_id']} -> {exchange_order_id}")
                    return True
                else:
                    logger.error(f"❌ No exchange order ID received: {result}")
                    order['status'] = OrderStatus.REJECTED.value
                    return False
            else:
                logger.error(f"❌ Exchange submission failed: {response.text}")
                order['status'] = OrderStatus.REJECTED.value
                return False

        except Exception as e:
            logger.error(f"❌ Error submitting order: {e}")
            order['status'] = OrderStatus.REJECTED.value
            return False

    def monitor_order(self, order: Dict):
        """Monitor order execution"""
        try:
            if not order.get('exchange_order_id'):
                return

            # Check order status with exchange
            response = requests.get(
                f"{EXCHANGE_CONNECTOR_URL}/orders/{order['exchange_order_id']}",
                timeout=10
            )

            if response.status_code == 200:
                order_status = response.json()

                current_status = order_status.get('status', '').lower()
                filled_amount = order_status.get('filled', 0)
                remaining_amount = order_status.get('remaining', order['amount'])

                # Update order based on exchange status
                if current_status in ['filled', 'closed']:
                    order['status'] = OrderStatus.FILLED.value
                    order['filled_amount'] = filled_amount
                    order['remaining_amount'] = remaining_amount
                    order['completed_at'] = datetime.now().isoformat()

                    # Update statistics
                    self.stats['orders_filled'] += 1
                    self.stats['total_volume'] += filled_amount * order['price']

                    # Publish position update
                    self.redis_client.publish('position_updates', json.dumps({
                        'symbol': order['symbol'],
                        'type': 'position_opened',
                        'amount': filled_amount,
                        'price': order['price'],
                        'order_id': order['order_id'],
                        'timestamp': datetime.now().isoformat()
                    }))

                    logger.info(f"✅ Order filled: {order['order_id']} - {filled_amount}/{order['amount']}")

                elif current_status in ['canceled', 'cancelled']:
                    order['status'] = OrderStatus.CANCELLED.value
                    order['cancelled_at'] = datetime.now().isoformat()
                    logger.info(f"❌ Order cancelled: {order['order_id']}")

                elif current_status == 'partial':
                    order['status'] = OrderStatus.PARTIAL.value
                    order['filled_amount'] = filled_amount
                    order['remaining_amount'] = remaining_amount
                    logger.info(f"📊 Order partial: {order['order_id']} - {filled_amount}/{order['amount']}")

                # Publish order update
                self.redis_client.publish('order_updates', json.dumps({
                    'order_id': order['order_id'],
                    'status': order['status'],
                    'filled_amount': filled_amount,
                    'remaining_amount': remaining_amount,
                    'timestamp': datetime.now().isoformat()
                }))

        except Exception as e:
            logger.error(f"❌ Error monitoring order {order.get('order_id')}: {e}")

    def process_signal(self, signal_data: Dict):
        """Process incoming trading signal through complete lifecycle"""
        try:
            logger.info(f"🎯 Processing signal: {signal_data.get('symbol')} {signal_data.get('type')}")

            # Step 1: Validate signal
            if not self.validate_signal(signal_data):
                logger.info("❌ Signal validation failed")
                return

            # Step 2: Calculate position size
            position_size = self.calculate_position_size(signal_data)
            if position_size <= 0:
                logger.info("❌ Invalid position size calculated")
                return

            # Step 3: Create order
            order = self.create_order(signal_data, position_size)
            if not order:
                logger.error("❌ Failed to create order")
                return

            # Step 4: Submit order
            if not self.submit_order(order):
                logger.error("❌ Failed to submit order")
                return

            # Step 5: Monitor order execution
            # Start monitoring in background
            monitor_thread = threading.Thread(
                target=self.monitor_order_execution,
                args=(order,),
                daemon=True
            )
            monitor_thread.start()

            # Update statistics
            self.stats['orders_processed'] += 1

            logger.info(f"🚀 Order lifecycle started: {order['order_id']}")

        except Exception as e:
            logger.error(f"❌ Error processing signal: {e}")

    def monitor_order_execution(self, order: Dict):
        """Monitor order until completion or timeout"""
        start_time = datetime.now()
        timeout = timedelta(seconds=ORDER_TIMEOUT)

        while (datetime.now() - start_time) < timeout:
            try:
                self.monitor_order(order)

                # Check if order is complete
                if order['status'] in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value, OrderStatus.REJECTED.value]:
                    break

                # Wait before next check
                asyncio.run(asyncio.sleep(5))

            except Exception as e:
                logger.error(f"❌ Error in order monitoring: {e}")
                break

        # Handle timeout
        if order['status'] not in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value, OrderStatus.REJECTED.value]:
            order['status'] = OrderStatus.EXPIRED.value
            order['expired_at'] = datetime.now().isoformat()
            logger.warning(f"⏰ Order expired: {order['order_id']}")

            # Publish expiration event
            self.redis_client.publish('order_updates', json.dumps({
                'order_id': order['order_id'],
                'status': order['status'],
                'timestamp': datetime.now().isoformat()
            }))

    def subscribe_to_signals(self):
        """Subscribe to trading signals"""
        try:
            pubsub = self.redis_client.pubsub()

            # Subscribe to trading signals
            channels = ['trading_signals', 'signals']
            pubsub.subscribe(*channels)

            logger.info(f"📡 Subscribed to signal channels: {channels}")

            # Process signals
            for message in pubsub.listen():
                if not self.is_running:
                    break

                if message['type'] == 'message':
                    try:
                        signal_data = json.loads(message['data'])
                        self.process_signal(signal_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to decode signal: {e}")

        except Exception as e:
            logger.error(f"❌ Error in signal subscription: {e}")

    def start_background_processing(self):
        """Start background order processing"""
        def run_signal_processor():
            self.subscribe_to_signals()

        thread = threading.Thread(target=run_signal_processor, daemon=True)
        thread.start()
        logger.info("🎯 Order lifecycle processing started")

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of specific order"""
        return self.active_orders.get(order_id)

    def get_active_orders(self) -> Dict[str, Dict]:
        """Get all active orders"""
        return self.active_orders.copy()

    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order processing statistics"""
        total_orders = self.stats['orders_processed']
        if total_orders > 0:
            self.stats['success_rate'] = (self.stats['orders_filled'] / total_orders) * 100

        return self.stats.copy()

    def start(self):
        """Start the order lifecycle manager"""
        try:
            logger.info("🚀 Starting Order Lifecycle Manager...")

            # Connect to services
            self.connect_services()

            # Start processing
            self.is_running = True
            self.start_background_processing()

            # Keep main thread alive
            while self.is_running:
                # Publish periodic status updates
                status = {
                    'service': 'order-lifecycle-manager',
                    'active_orders': len(self.active_orders),
                    'orders_processed': self.stats['orders_processed'],
                    'orders_filled': self.stats['orders_filled'],
                    'success_rate': self.stats['success_rate'],
                    'timestamp': datetime.now().isoformat()
                }

                self.redis_client.publish('service_status', json.dumps(status))
                asyncio.run(asyncio.sleep(60))  # Update every minute

        except KeyboardInterrupt:
            logger.info("⏹️ Stopping Order Lifecycle Manager...")
            self.stop()
        except Exception as e:
            logger.error(f"❌ Order Lifecycle Manager error: {e}")
            self.stop()

    def stop(self):
        """Stop the order lifecycle manager"""
        self.is_running = False
        logger.info("✅ Order Lifecycle Manager stopped")

def create_app():
    """Create FastAPI application for health checks and API"""
    from fastapi import FastAPI

    app = FastAPI(title="Order Lifecycle Manager", version="1.0.0")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "order-lifecycle-manager"}

    @app.get("/orders")
    async def get_orders():
        orders = processor.get_active_orders()
        return {"orders": orders, "count": len(orders)}

    @app.get("/orders/{order_id}")
    async def get_order_status(order_id: str):
        order = processor.get_order_status(order_id)
        if order:
            return {"order": order}
        return {"error": "Order not found"}, 404

    @app.get("/stats")
    async def get_statistics():
        stats = processor.get_order_statistics()
        return {"statistics": stats}

    @app.post("/test-order")
    async def test_order(signal: Dict):
        processor.process_signal(signal)
        return {"message": "Signal queued for processing"}

    return app

if __name__ == "__main__":
    # Check if running as API server or processor
    if os.getenv('API_MODE', 'false').lower() == 'true':
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run as order lifecycle manager
        processor = OrderLifecycleManager()
        processor.start()
