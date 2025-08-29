#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Order Lifecycle Manager
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
from dataclasses import dataclass
import redis
import requests

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

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class OrderWithTPSL:
    """Order structure with TP/SL/TSL support"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    order_type: OrderType
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_activation: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class OrderLifecycleManager:
    """Complete order lifecycle management service"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False
        self.active_orders = {}  # Track active orders
        self.order_history = []  # Store order history
        self.position_cache = {}  # Cache current positions
        self.tp_sl_orders = {}  # Track TP/SL orders by symbol

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
            logger.info("# Check Connected to Redis")

            # Load exchange credentials
            self.load_exchange_credentials()

            logger.info("# Check Connected to all services")

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

    def validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal before processing"""
        try:
            required_fields = ['symbol', 'type', 'price', 'viper_score', 'confidence', 'timestamp']
            for field in required_fields:
                if field not in signal:
                    logger.error(f"# X Missing required field: {field}")
                    return False

            # Check VIPER score threshold
            viper_threshold = float(os.getenv('VIPER_THRESHOLD', '85'))
            if signal.get('viper_score', 0) < viper_threshold:
                logger.info(f"# Chart Signal rejected: VIPER score {signal['viper_score']} < {viper_threshold}")
                return False

            # Check confidence threshold
            confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '80'))
            if signal.get('confidence', 0) < confidence_threshold:
                logger.info(f"# Chart Signal rejected: Confidence {signal['confidence']} < {confidence_threshold}")
                return False

            # Check symbol is supported
            supported_symbols = self.get_supported_symbols()
            if signal['symbol'] not in supported_symbols:
                logger.error(f"# X Symbol not supported: {signal['symbol']}")
                return False

            return True

        except Exception as e:
            logger.error(f"# X Error validating signal: {e}")
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
            logger.error(f"# X Error getting supported symbols: {e}")
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

                logger.info(f"# Chart Calculated position size: {position_size} for {signal['symbol']}")
                return position_size
            else:
                logger.error(f"# X Risk manager error: {response.text}")
                return 0

        except Exception as e:
            logger.error(f"# X Error calculating position size: {e}")
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
            logger.error(f"# X Error creating order: {e}")
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
                    logger.error(f"# X No exchange order ID received: {result}")
                    order['status'] = OrderStatus.REJECTED.value
                    return False
            else:
                logger.error(f"# X Exchange submission failed: {response.text}")
                order['status'] = OrderStatus.REJECTED.value
                return False

        except Exception as e:
            logger.error(f"# X Error submitting order: {e}")
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

                    logger.info(f"# Check Order filled: {order['order_id']} - {filled_amount}/{order['amount']}")

                elif current_status in ['canceled', 'cancelled']:
                    order['status'] = OrderStatus.CANCELLED.value
                    order['cancelled_at'] = datetime.now().isoformat()
                    logger.info(f"# X Order cancelled: {order['order_id']}")

                elif current_status == 'partial':
                    order['status'] = OrderStatus.PARTIAL.value
                    order['filled_amount'] = filled_amount
                    order['remaining_amount'] = remaining_amount
                    logger.info(f"# Chart Order partial: {order['order_id']} - {filled_amount}/{order['amount']}")

                # Publish order update
                self.redis_client.publish('order_updates', json.dumps({
                    'order_id': order['order_id'],
                    'status': order['status'],
                    'filled_amount': filled_amount,
                    'remaining_amount': remaining_amount,
                    'timestamp': datetime.now().isoformat()
                }))

        except Exception as e:
            logger.error(f"# X Error monitoring order {order.get('order_id')}: {e}")

    def process_signal(self, signal_data: Dict):
        """Process incoming trading signal through complete lifecycle"""
        try:
            logger.info(f"# Target Processing signal: {signal_data.get('symbol')} {signal_data.get('type')}")

            # Step 1: Validate signal
            if not self.validate_signal(signal_data):
                logger.info("# X Signal validation failed")
                return

            # Step 2: Calculate position size
            position_size = self.calculate_position_size(signal_data)
            if position_size <= 0:
                logger.info("# X Invalid position size calculated")
                return

            # Step 3: Create order
            order = self.create_order(signal_data, position_size)
            if not order:
                logger.error("# X Failed to create order")
                return

            # Step 4: Submit order
            if not self.submit_order(order):
                logger.error("# X Failed to submit order")
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

            logger.info(f"# Rocket Order lifecycle started: {order['order_id']}")

        except Exception as e:
            logger.error(f"# X Error processing signal: {e}")

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
                logger.error(f"# X Error in order monitoring: {e}")
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
                        logger.error(f"# X Failed to decode signal: {e}")

        except Exception as e:
            logger.error(f"# X Error in signal subscription: {e}")

    def start_background_processing(self):
        """Start background order processing"""
        def run_signal_processor():
            self.subscribe_to_signals()

        thread = threading.Thread(target=run_signal_processor, daemon=True)
        thread.start()
        logger.info("# Target Order lifecycle processing started")

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

    def create_order_with_tp_sl_tsl(self, symbol: str, side: str, size: float, entry_price: float,
                                   stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                                   trailing_stop: Optional[float] = None, trailing_activation: Optional[float] = None) -> OrderWithTPSL:
        """Create an order with TP/SL/TSL parameters"""
        side_enum = PositionSide.LONG if side.upper() == 'LONG' else PositionSide.SHORT

        order = OrderWithTPSL(
            symbol=symbol,
            side=side_enum,
            size=size,
            entry_price=entry_price,
            order_type=OrderType.MARKET,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            trailing_activation=trailing_activation
        )

        return order

    def place_tp_sl_tsl_orders(self, order: OrderWithTPSL) -> Dict[str, Any]:
        """Place the main order and associated TP/SL/TSL orders"""
        try:
            results = {
                'main_order': None,
                'stop_loss_order': None,
                'take_profit_order': None,
                'trailing_stop_order': None,
                'success': False,
                'errors': []
            }

            # Place main market order
            main_order_data = {
                'symbol': order.symbol,
                'side': order.side.value,
                'size': order.size,
                'price': order.entry_price,
                'type': 'market'
            }

            main_order_id = self.submit_market_order(main_order_data)
            if main_order_id:
                order.order_id = main_order_id
                order.status = OrderStatus.SUBMITTED
                results['main_order'] = main_order_id
                self.active_orders[main_order_id] = order
                results['success'] = True
                logger.info(f"# Check Placed main order for {order.symbol}: {main_order_id}")
            else:
                results['errors'].append("Failed to place main order")
                return results

            # Place Stop Loss order if specified
            if order.stop_loss:
                sl_order_data = {
                    'symbol': order.symbol,
                    'side': 'sell' if order.side == PositionSide.LONG else 'buy',
                    'size': order.size,
                    'price': order.stop_loss,
                    'type': 'stop',
                    'stop_price': order.stop_loss
                }

                sl_order_id = self.submit_stop_order(sl_order_data)
                if sl_order_id:
                    results['stop_loss_order'] = sl_order_id
                    logger.info(f"# Check Placed SL order for {order.symbol}: {sl_order_id}")
                else:
                    results['errors'].append("Failed to place stop loss order")

            # Place Take Profit order if specified
            if order.take_profit:
                tp_order_data = {
                    'symbol': order.symbol,
                    'side': 'sell' if order.side == PositionSide.LONG else 'buy',
                    'size': order.size,
                    'price': order.take_profit,
                    'type': 'limit'
                }

                tp_order_id = self.submit_limit_order(tp_order_data)
                if tp_order_id:
                    results['take_profit_order'] = tp_order_id
                    logger.info(f"# Check Placed TP order for {order.symbol}: {tp_order_id}")
                else:
                    results['errors'].append("Failed to place take profit order")

            # Store TP/SL orders for tracking
            self.tp_sl_orders[order.symbol] = {
                'main_order': main_order_id,
                'stop_loss': results.get('stop_loss_order'),
                'take_profit': results.get('take_profit_order'),
                'trailing_stop': None,  # Will be managed dynamically
                'order_details': order
            }

            return results

        except Exception as e:
            logger.error(f"# X Error placing TP/SL/TSL orders: {e}")
            return {
                'success': False,
                'errors': [str(e)]
            }

    def submit_market_order(self, order_data: Dict) -> Optional[str]:
        """Submit a market order to the exchange"""
        try:
            response = requests.post(
                f"{EXCHANGE_CONNECTOR_URL}/api/orders/market",
                json=order_data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('order_id')
            else:
                logger.error(f"# X Failed to submit market order: {response.text}")
                return None

        except Exception as e:
            logger.error(f"# X Error submitting market order: {e}")
            return None

    def submit_limit_order(self, order_data: Dict) -> Optional[str]:
        """Submit a limit order to the exchange"""
        try:
            response = requests.post(
                f"{EXCHANGE_CONNECTOR_URL}/api/orders/limit",
                json=order_data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('order_id')
            else:
                logger.error(f"# X Failed to submit limit order: {response.text}")
                return None

        except Exception as e:
            logger.error(f"# X Error submitting limit order: {e}")
            return None

    def submit_stop_order(self, order_data: Dict) -> Optional[str]:
        """Submit a stop order to the exchange"""
        try:
            response = requests.post(
                f"{EXCHANGE_CONNECTOR_URL}/api/orders/stop",
                json=order_data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('order_id')
            else:
                logger.error(f"# X Failed to submit stop order: {response.text}")
                return None

        except Exception as e:
            logger.error(f"# X Error submitting stop order: {e}")
            return None

    def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[str]:
        """Update trailing stop order for a symbol"""
        if symbol not in self.tp_sl_orders:
            return None

        order_info = self.tp_sl_orders[symbol]
        order_details = order_info['order_details']

        if not order_details.trailing_stop or not order_details.trailing_activation:
            return None

        # Calculate new trailing stop price
        if order_details.side == PositionSide.LONG:
            if current_price >= order_details.trailing_activation:
                new_stop = current_price * (1 - 0.01)  # 1% trailing stop
                if new_stop > order_details.trailing_stop:
                    order_details.trailing_stop = new_stop
                    # Update the trailing stop order on exchange
                    return self.update_stop_order(order_info.get('trailing_stop'), new_stop)
        else:
            if current_price <= order_details.trailing_activation:
                new_stop = current_price * (1 + 0.01)  # 1% trailing stop for shorts
                if new_stop < order_details.trailing_stop:
                    order_details.trailing_stop = new_stop
                    # Update the trailing stop order on exchange
                    return self.update_stop_order(order_info.get('trailing_stop'), new_stop)

        return None

    def update_stop_order(self, order_id: str, new_price: float) -> Optional[str]:
        """Update an existing stop order with new price"""
        try:
            update_data = {
                'order_id': order_id,
                'new_price': new_price
            }

            response = requests.put(
                f"{EXCHANGE_CONNECTOR_URL}/api/orders/stop",
                json=update_data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('order_id')
            else:
                logger.error(f"# X Failed to update stop order: {response.text}")
                return None

        except Exception as e:
            logger.error(f"# X Error updating stop order: {e}")
            return None

    def cancel_tp_sl_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all TP/SL orders for a symbol"""
        results = {
            'cancelled': [],
            'failed': [],
            'success': True
        }

        if symbol not in self.tp_sl_orders:
            return results

        order_info = self.tp_sl_orders[symbol]

        # Cancel stop loss order
        if order_info.get('stop_loss'):
            if self.cancel_order(order_info['stop_loss']):
                results['cancelled'].append('stop_loss')
            else:
                results['failed'].append('stop_loss')

        # Cancel take profit order
        if order_info.get('take_profit'):
            if self.cancel_order(order_info['take_profit']):
                results['cancelled'].append('take_profit')
            else:
                results['failed'].append('take_profit')

        # Cancel trailing stop order
        if order_info.get('trailing_stop'):
            if self.cancel_order(order_info['trailing_stop']):
                results['cancelled'].append('trailing_stop')
            else:
                results['failed'].append('trailing_stop')

        if results['failed']:
            results['success'] = False

        # Remove from tracking
        del self.tp_sl_orders[symbol]

        return results

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order"""
        try:
            response = requests.delete(
                f"{EXCHANGE_CONNECTOR_URL}/api/orders/{order_id}",
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"# Check Cancelled order: {order_id}")
                return True
            else:
                logger.error(f"# X Failed to cancel order {order_id}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"# X Error cancelling order {order_id}: {e}")
            return False

    def get_tp_sl_status(self, symbol: str) -> Optional[Dict]:
        """Get TP/SL status for a symbol"""
        if symbol not in self.tp_sl_orders:
            return None

        order_info = self.tp_sl_orders[symbol]
        return {
            'symbol': symbol,
            'main_order': order_info.get('main_order'),
            'stop_loss': order_info.get('stop_loss'),
            'take_profit': order_info.get('take_profit'),
            'trailing_stop': order_info.get('trailing_stop'),
            'order_details': {
                'side': order_info['order_details'].side.value,
                'size': order_info['order_details'].size,
                'entry_price': order_info['order_details'].entry_price,
                'stop_loss': order_info['order_details'].stop_loss,
                'take_profit': order_info['order_details'].take_profit,
                'trailing_stop': order_info['order_details'].trailing_stop,
                'trailing_activation': order_info['order_details'].trailing_activation
            }
        }

    def start(self):
        """Start the order lifecycle manager"""
        try:
            logger.info("# Rocket Starting Order Lifecycle Manager...")

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
            logger.error(f"# X Order Lifecycle Manager error: {e}")
            self.stop()

    def stop(self):
        """Stop the order lifecycle manager"""
        self.is_running = False
        logger.info("# Check Order Lifecycle Manager stopped")

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

    # TP/SL/TSL Management Endpoints
    @app.post("/api/tp-sl-tsl/create-order")
    async def create_tp_sl_tsl_order(request: Dict):
        """Create and place an order with TP/SL/TSL"""
        try:
            data = request
            symbol = data['symbol']
            side = data['side']
            size = float(data['size'])
            entry_price = float(data['entry_price'])
            stop_loss = data.get('stop_loss')
            take_profit = data.get('take_profit')
            trailing_stop = data.get('trailing_stop')
            trailing_activation = data.get('trailing_activation')

            # Create order with TP/SL/TSL
            order = processor.create_order_with_tp_sl_tsl(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                stop_loss=float(stop_loss) if stop_loss else None,
                take_profit=float(take_profit) if take_profit else None,
                trailing_stop=float(trailing_stop) if trailing_stop else None,
                trailing_activation=float(trailing_activation) if trailing_activation else None
            )

            # Place the orders
            results = processor.place_tp_sl_tsl_orders(order)

            return {
                'order': {
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'size': order.size,
                    'entry_price': order.entry_price,
                    'stop_loss': order.stop_loss,
                    'take_profit': order.take_profit,
                    'trailing_stop': order.trailing_stop,
                    'trailing_activation': order.trailing_activation,
                    'order_id': order.order_id,
                    'status': order.status.value
                },
                'placement_results': results
            }

        except Exception as e:
            logger.error(f"# X Error creating TP/SL/TSL order: {e}")
            return {"error": str(e)}, 500

    @app.post("/api/tp-sl-tsl/update-price")
    async def update_trailing_stop(request: Dict):
        """Update trailing stop based on current price"""
        try:
            data = request
            symbol = data['symbol']
            current_price = float(data['current_price'])

            update_result = processor.update_trailing_stop(symbol, current_price)

            return {
                'symbol': symbol,
                'current_price': current_price,
                'trailing_stop_updated': update_result is not None,
                'new_order_id': update_result
            }

        except Exception as e:
            logger.error(f"# X Error updating trailing stop: {e}")
            return {"error": str(e)}, 500

    @app.get("/api/tp-sl-tsl/status/{symbol}")
    async def get_tp_sl_status(symbol: str):
        """Get TP/SL status for a symbol"""
        status = processor.get_tp_sl_status(symbol)
        if status:
            return status
        return {"error": f"No TP/SL orders found for {symbol}"}, 404

    @app.delete("/api/tp-sl-tsl/orders/{symbol}")
    async def cancel_tp_sl_orders(symbol: str):
        """Cancel all TP/SL orders for a symbol"""
        results = processor.cancel_tp_sl_orders(symbol)
        return {
            'symbol': symbol,
            'results': results
        }

    @app.get("/api/tp-sl-tsl/orders")
    async def get_all_tp_sl_orders():
        """Get all active TP/SL orders"""
        all_orders = {}
        for symbol in processor.tp_sl_orders.keys():
            status = processor.get_tp_sl_status(symbol)
            if status:
                all_orders[symbol] = status

        return {
            'orders': all_orders,
            'total_symbols': len(all_orders)
        }

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
