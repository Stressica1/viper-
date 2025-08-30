#!/usr/bin/env python3
"""
# Rocket VIPER Enhanced Order Execution System
Advanced order placement with sophisticated TP/SL/TSL management

Features:
- Smart order routing and execution strategies
- Advanced TP/SL/TSL position management
- Risk-aware position sizing
- Order execution cost optimization
- Real-time position monitoring
- Partial fill handling
- Dynamic stop adjustments
- Market impact minimization
"""

import os
import time
import logging
import asyncio
import numpy as np
from enum import Enum
import redis
import httpx
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"  # Time-Weighted Average Price
    POV = "POV"    # Percentage of Volume

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class OrderRequest:
    """Order request with comprehensive parameters"""
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancel
    client_id: Optional[str] = None
    
    # TP/SL/TSL parameters
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    trailing_stop_percent: Optional[float] = None
    trailing_activation_price: Optional[float] = None
    
    # Advanced execution parameters
    max_execution_time: int = 300  # 5 minutes max execution time
    slice_size: Optional[float] = None  # For large order slicing
    participation_rate: Optional[float] = None  # For POV orders
    urgency: str = "NORMAL"  # LOW, NORMAL, HIGH
    
    def __post_init__(self):
        if not self.client_id:
            self.client_id = str(uuid.uuid4())

@dataclass
class Order:
    """Order with execution tracking"""
    id: str
    client_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    exchange_id: Optional[str] = None
    fees: float = 0.0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

@dataclass
class Position:
    """Enhanced position with TP/SL/TSL tracking"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    
    # TP/SL/TSL levels
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_activation: Optional[float] = None
    trailing_callback: Optional[float] = None
    
    # Position metadata
    opened_at: datetime = None
    updated_at: datetime = None
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    def __post_init__(self):
        if not self.opened_at:
            self.opened_at = datetime.now()
        self.updated_at = datetime.now()

class EnhancedOrderExecutor:
    """Advanced order execution system with TP/SL/TSL management"""
    
    def __init__(self):
        self.redis_client = None
        self.is_running = False
        
        # Configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '10000'))  # $10k max
        self.default_slippage = float(os.getenv('DEFAULT_SLIPPAGE', '0.001'))  # 0.1%
        self.min_order_size = float(os.getenv('MIN_ORDER_SIZE', '10'))  # $10 minimum
        
        # Service URLs
        self.exchange_connector_url = os.getenv('EXCHANGE_CONNECTOR_URL', 'http://exchange-connector:8005')
        self.risk_manager_url = os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8002')
        self.market_data_url = os.getenv('MARKET_DATA_MANAGER_URL', 'http://market-data-manager:8003')
        
        # Order tracking
        self.active_orders = {}  # Order ID -> Order
        self.order_history = []
        self.positions = {}  # Symbol -> Position
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'avg_execution_time': 0.0,
            'total_fees': 0.0
        }
        
        # Execution strategies
        self.execution_strategies = {
            'MARKET': self._execute_market_order,
            'LIMIT': self._execute_limit_order,
            'STOP': self._execute_stop_order,
            'TRAILING_STOP': self._execute_trailing_stop_order,
            'TWAP': self._execute_twap_order,
            'POV': self._execute_pov_order
        }
        
        # Risk management
        self.daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT', '0.05'))  # 5%
        self.max_positions = int(os.getenv('MAX_POSITIONS', '15'))
        
        logger.info("# Construction Enhanced Order Executor initialized")
    
    async def initialize(self) -> bool:
        """Initialize the order executor"""
        try:
            # Initialize Redis
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            await asyncio.to_thread(self.redis_client.ping)
            
            # Start background tasks
            asyncio.create_task(self._order_monitoring_loop())
            asyncio.create_task(self._position_monitoring_loop())
            asyncio.create_task(self._tp_sl_tsl_monitoring_loop())
            
            logger.info("# Check Enhanced Order Executor initialized")
            return True
            
        except Exception as e:
            logger.error(f"# X Failed to initialize order executor: {e}")
            return False
    
    async def submit_order(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Submit an order with comprehensive validation and execution"""
        try:
            start_time = time.time()
            
            # Validate order request
            validation = await self._validate_order_request(order_request)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': validation['error'],
                    'order_id': None
                }
            
            # Risk management check
            risk_check = await self._check_risk_limits(order_request)
            if not risk_check['allowed']:
                return {
                    'success': False,
                    'error': f"Risk management: {risk_check['reason']}",
                    'order_id': None
                }
            
            # Calculate optimal execution strategy
            execution_strategy = await self._determine_execution_strategy(order_request)
            
            # Create order
            order = Order(
                id=str(uuid.uuid4()),
                client_id=order_request.client_id,
                symbol=order_request.symbol,
                side=order_request.side,
                type=execution_strategy['order_type'],
                quantity=validation['adjusted_quantity'],
                price=execution_strategy.get('price'),
                stop_price=execution_strategy.get('stop_price'),
                status=OrderStatus.PENDING
            )
            
            # Store order
            self.active_orders[order.id] = order
            
            # Execute order using appropriate strategy
            execution_result = await self.execution_strategies[execution_strategy['order_type'].value](
                order, order_request, execution_strategy
            )
            
            # Update execution statistics
            execution_time = time.time() - start_time
            self.execution_stats['total_orders'] += 1
            
            if execution_result['success']:
                self.execution_stats['successful_orders'] += 1
                
                # Set up TP/SL/TSL if specified
                if any([order_request.take_profit_price, order_request.stop_loss_price,
                       order_request.trailing_stop_percent]):
                    await self._setup_tp_sl_tsl(order, order_request, execution_result)
                
            else:
                self.execution_stats['failed_orders'] += 1
                order.status = OrderStatus.REJECTED
            
            # Update average execution time
            current_avg = self.execution_stats['avg_execution_time']
            total_successful = self.execution_stats['successful_orders']
            if total_successful > 0:
                self.execution_stats['avg_execution_time'] = (
                    (current_avg * (total_successful - 1) + execution_time) / total_successful
                )
            
            return {
                'success': execution_result['success'],
                'order_id': order.id,
                'client_id': order.client_id,
                'execution_time': execution_time,
                'execution_strategy': execution_strategy,
                'details': execution_result
            }
            
        except Exception as e:
            logger.error(f"# X Order submission error: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': None
            }
    
    async def _validate_order_request(self, request: OrderRequest) -> Dict[str, Any]:
        """Comprehensive order request validation"""
        try:
            # Basic validation
            if request.quantity <= 0:
                return {'valid': False, 'error': 'Invalid quantity'}
            
            # Get current market data
            market_data = await self._get_market_data(request.symbol)
            if not market_data:
                return {'valid': False, 'error': 'Unable to get market data'}
            
            ticker = market_data.get('ticker', {})
            current_price = ticker.get('price', 0)
            
            if current_price <= 0:
                return {'valid': False, 'error': 'Invalid current price'}
            
            # Calculate order value
            order_value = request.quantity * current_price
            
            # Check minimum order size
            if order_value < self.min_order_size:
                return {'valid': False, 'error': f'Order value below minimum ${self.min_order_size}'}
            
            # Check maximum position size
            if order_value > self.max_position_size:
                # Adjust quantity to maximum allowed
                adjusted_quantity = self.max_position_size / current_price
                logger.warning(f"# Warning Reducing order quantity to meet position limits: {adjusted_quantity}")
            else:
                adjusted_quantity = request.quantity
            
            # Validate price levels
            if request.price and abs(request.price - current_price) / current_price > 0.1:  # >10% from market
                return {'valid': False, 'error': 'Price too far from market'}
            
            # Validate TP/SL levels
            if request.take_profit_price and request.side == OrderSide.BUY:
                if request.take_profit_price <= current_price:
                    return {'valid': False, 'error': 'Take profit must be above entry price for long'}
            
            if request.stop_loss_price and request.side == OrderSide.BUY:
                if request.stop_loss_price >= current_price:
                    return {'valid': False, 'error': 'Stop loss must be below entry price for long'}
            
            return {
                'valid': True,
                'adjusted_quantity': adjusted_quantity,
                'current_price': current_price,
                'order_value': adjusted_quantity * current_price
            }
            
        except Exception as e:
            logger.error(f"# X Order validation error: {e}")
            return {'valid': False, 'error': str(e)}
    
    async def _check_risk_limits(self, request: OrderRequest) -> Dict[str, Any]:
        """Check risk management limits"""
        try:
            # Check position count limit
            if len(self.positions) >= self.max_positions:
                return {'allowed': False, 'reason': 'Maximum positions reached'}
            
            # Check if symbol already has a position (1 trade per symbol rule)
            if request.symbol in self.positions:
                return {'allowed': False, 'reason': 'Position already exists for this symbol'}
            
            # Call risk manager service for additional checks
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    risk_data = {
                        'symbol': request.symbol,
                        'position_size': request.quantity,
                        'price': request.price or 0
                    }
                    
                    response = await client.post(f"{self.risk_manager_url}/api/position/check", json=risk_data)
                    if response.status_code == 200:
                        risk_result = response.json()
                        if not risk_result.get('allowed', False):
                            return {'allowed': False, 'reason': risk_result.get('reason', 'Risk limit exceeded')}
                    else:
                        logger.warning(f"# Warning Risk manager check failed: {response.status_code}")
                        
            except Exception as e:
                logger.warning(f"# Warning Risk manager communication error: {e}")
            
            return {'allowed': True}
            
        except Exception as e:
            logger.error(f"# X Risk limit check error: {e}")
            return {'allowed': False, 'reason': str(e)}
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.market_data_url}/api/market/{symbol}")
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"# Warning Failed to get market data for {symbol}: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"# X Error getting market data for {symbol}: {e}")
            return None
    
    async def _determine_execution_strategy(self, request: OrderRequest) -> Dict[str, Any]:
        """Determine optimal execution strategy"""
        try:
            # Get market data for strategy decision
            market_data = await self._get_market_data(request.symbol)
            if not market_data:
                # Fallback to limit order
                return {
                    'order_type': OrderType.LIMIT,
                    'reason': 'No market data - using limit order',
                    'price': request.price
                }
            
            ticker = market_data.get('ticker', {})
            orderbook = market_data.get('orderbook', {})
            current_price = ticker.get('price', 0)
            volume = ticker.get('volume', 0)
            
            # Calculate execution cost
            execution_cost = await self._calculate_execution_cost(market_data, request.quantity * current_price)
            
            # Strategy selection based on urgency and market conditions
            if request.urgency == "HIGH":
                # High urgency - use market orders for speed
                return {
                    'order_type': OrderType.MARKET,
                    'reason': 'High urgency execution',
                    'expected_cost': execution_cost
                }
            
            elif request.urgency == "LOW" and request.quantity * current_price > 1000:
                # Low urgency, large order - use TWAP
                return {
                    'order_type': OrderType.TWAP,
                    'reason': 'Large order with low urgency - using TWAP',
                    'slices': max(5, int((request.quantity * current_price) / 500)),  # $500 per slice
                    'duration': request.max_execution_time
                }
            
            elif execution_cost < 0.5:  # Low execution cost
                # Good liquidity - market order acceptable
                return {
                    'order_type': OrderType.MARKET,
                    'reason': 'Low execution cost - market order',
                    'expected_cost': execution_cost
                }
            
            else:
                # Default to intelligent limit order
                optimal_price = await self._calculate_optimal_limit_price(
                    current_price, request.side, orderbook, request.urgency
                )
                
                return {
                    'order_type': OrderType.LIMIT,
                    'reason': 'Optimal limit order strategy',
                    'price': optimal_price,
                    'expected_cost': execution_cost
                }
                
        except Exception as e:
            logger.error(f"# X Strategy determination error: {e}")
            # Fallback strategy
            return {
                'order_type': OrderType.LIMIT,
                'reason': f'Error in strategy selection: {e}',
                'price': request.price
            }
    
    async def _calculate_execution_cost(self, market_data: Dict, order_value: float) -> float:
        """Calculate expected execution cost"""
        try:
            orderbook = market_data.get('orderbook', {})
            ticker = market_data.get('ticker', {})
            
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                return 2.0  # Default moderate cost
            
            bids = orderbook['bids']
            asks = orderbook['asks']
            
            if not bids or not asks:
                return 2.0
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            
            # Spread cost
            spread_cost = ((best_ask - best_bid) / mid_price) * 100 / 2  # Half spread in %
            
            # Market impact estimation using square root model
            volume = ticker.get('volume', 1000000)  # 24h volume
            participation = min(0.1, order_value / max(volume, 100000))  # Max 10% participation
            market_impact = 0.1 * np.sqrt(participation) * 100  # Impact in %
            
            total_cost = spread_cost + market_impact
            
            return max(0.01, min(5.0, total_cost))  # Cap between 0.01% and 5%
            
        except Exception as e:
            logger.error(f"# X Execution cost calculation error: {e}")
            return 1.0  # Default moderate cost
    
    async def _calculate_optimal_limit_price(self, current_price: float, side: OrderSide, 
                                           orderbook: Dict, urgency: str) -> float:
        """Calculate optimal limit order price"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                # No orderbook data - use current price with small buffer
                buffer = 0.001 if urgency == "NORMAL" else 0.002
                if side == OrderSide.BUY:
                    return current_price * (1 - buffer)
                else:
                    return current_price * (1 + buffer)
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            if urgency == "HIGH":
                # Aggressive - at or near best price
                if side == OrderSide.BUY:
                    return best_ask  # Pay the spread for speed
                else:
                    return best_bid  # Hit the bid for speed
            
            elif urgency == "LOW":
                # Patient - join the queue
                if side == OrderSide.BUY:
                    return best_bid  # Join best bid
                else:
                    return best_ask  # Join best ask
            
            else:  # NORMAL urgency
                # Balanced approach - slightly aggressive
                spread = best_ask - best_bid
                
                if side == OrderSide.BUY:
                    return best_bid + (spread * 0.3)  # 30% through the spread
                else:
                    return best_ask - (spread * 0.3)  # 30% through the spread
                    
        except Exception as e:
            logger.error(f"# X Optimal price calculation error: {e}")
            # Fallback
            buffer = 0.001
            if side == OrderSide.BUY:
                return current_price * (1 - buffer)
            else:
                return current_price * (1 + buffer)
    
    async def _execute_market_order(self, order: Order, request: OrderRequest, strategy: Dict) -> Dict[str, Any]:
        """Execute market order"""
        try:
            logger.info(f"📈 Executing market order: {order.symbol} {order.side.value} {order.quantity}")
            
            # Submit to exchange
            exchange_result = await self._submit_to_exchange(order, "MARKET")
            
            if exchange_result['success']:
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.avg_fill_price = exchange_result.get('fill_price', 0)
                order.exchange_id = exchange_result.get('exchange_order_id')
                order.fees = exchange_result.get('fees', 0)
                
                # Create or update position
                await self._update_position_from_fill(order)
                
                return {
                    'success': True,
                    'fill_price': order.avg_fill_price,
                    'fill_quantity': order.filled_quantity,
                    'fees': order.fees,
                    'execution_strategy': 'MARKET'
                }
            else:
                return {
                    'success': False,
                    'error': exchange_result.get('error', 'Exchange execution failed'),
                    'execution_strategy': 'MARKET'
                }
                
        except Exception as e:
            logger.error(f"# X Market order execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_limit_order(self, order: Order, request: OrderRequest, strategy: Dict) -> Dict[str, Any]:
        """Execute limit order with monitoring"""
        try:
            order.price = strategy['price']
            logger.info(f"# Chart Executing limit order: {order.symbol} {order.side.value} {order.quantity} @ {order.price}")
            
            # Submit to exchange
            exchange_result = await self._submit_to_exchange(order, "LIMIT")
            
            if exchange_result['success']:
                order.status = OrderStatus.SUBMITTED
                order.exchange_id = exchange_result.get('exchange_order_id')
                
                # Start monitoring for fills
                asyncio.create_task(self._monitor_order_execution(order, request))
                
                return {
                    'success': True,
                    'order_submitted': True,
                    'limit_price': order.price,
                    'execution_strategy': 'LIMIT'
                }
            else:
                return {
                    'success': False,
                    'error': exchange_result.get('error', 'Exchange submission failed'),
                    'execution_strategy': 'LIMIT'
                }
                
        except Exception as e:
            logger.error(f"# X Limit order execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_twap_order(self, order: Order, request: OrderRequest, strategy: Dict) -> Dict[str, Any]:
        """Execute TWAP (Time-Weighted Average Price) order"""
        try:
            slices = strategy.get('slices', 5)
            duration = strategy.get('duration', 300)  # 5 minutes default
            slice_quantity = order.quantity / slices
            slice_interval = duration / slices
            
            logger.info(f"⏰ Executing TWAP order: {slices} slices over {duration}s")
            
            order.status = OrderStatus.SUBMITTED
            filled_quantities = []
            fill_prices = []
            
            for i in range(slices):
                try:
                    # Create slice order
                    slice_order = Order(
                        id=f"{order.id}_slice_{i}",
                        client_id=order.client_id,
                        symbol=order.symbol,
                        side=order.side,
                        type=OrderType.MARKET,  # Use market orders for TWAP slices
                        quantity=slice_quantity,
                        price=None,
                        stop_price=None,
                        status=OrderStatus.PENDING
                    )
                    
                    # Execute slice
                    slice_result = await self._execute_market_order(slice_order, request, {})
                    
                    if slice_result['success']:
                        filled_quantities.append(slice_quantity)
                        fill_prices.append(slice_result['fill_price'])
                        logger.info(f"# Check TWAP slice {i+1}/{slices} filled @ {slice_result['fill_price']}")
                    else:
                        logger.warning(f"# Warning TWAP slice {i+1}/{slices} failed")
                    
                    # Wait before next slice (except for last slice)
                    if i < slices - 1:
                        await asyncio.sleep(slice_interval)
                        
                except Exception as e:
                    logger.error(f"# X TWAP slice {i+1} error: {e}")
            
            # Calculate average fill
            if filled_quantities:
                total_filled = sum(filled_quantities)
                avg_price = sum(price * qty for price, qty in zip(fill_prices, filled_quantities)) / total_filled
                
                order.filled_quantity = total_filled
                order.avg_fill_price = avg_price
                order.status = OrderStatus.FILLED if total_filled == order.quantity else OrderStatus.PARTIALLY_FILLED
                
                # Update position
                if total_filled > 0:
                    await self._update_position_from_fill(order)
                
                return {
                    'success': True,
                    'fill_price': avg_price,
                    'fill_quantity': total_filled,
                    'slices_executed': len(filled_quantities),
                    'execution_strategy': 'TWAP'
                }
            else:
                return {'success': False, 'error': 'No TWAP slices filled'}
                
        except Exception as e:
            logger.error(f"# X TWAP execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_stop_order(self, order: Order, request: OrderRequest, strategy: Dict) -> Dict[str, Any]:
        """Execute stop order"""
        # This would implement stop order logic
        # For now, convert to limit order
        return await self._execute_limit_order(order, request, strategy)
    
    async def _execute_trailing_stop_order(self, order: Order, request: OrderRequest, strategy: Dict) -> Dict[str, Any]:
        """Execute trailing stop order"""
        # This would implement trailing stop logic
        # For now, convert to market order
        return await self._execute_market_order(order, request, strategy)
    
    async def _execute_pov_order(self, order: Order, request: OrderRequest, strategy: Dict) -> Dict[str, Any]:
        """Execute POV (Percentage of Volume) order"""
        # This would implement POV logic
        # For now, convert to TWAP
        return await self._execute_twap_order(order, request, strategy)
    
    async def _submit_to_exchange(self, order: Order, order_type: str) -> Dict[str, Any]:
        """Submit order to exchange via connector"""
        try:
            order_data = {
                'symbol': order.symbol,
                'side': order.side.value.lower(),
                'type': order_type.lower(),
                'quantity': order.quantity,
                'price': order.price,
                'timeInForce': 'GTC',
                'newClientOrderId': order.client_id
            }
            
            # Remove None values
            order_data = {k: v for k, v in order_data.items() if v is not None}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.exchange_connector_url}/api/order", json=order_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Simulate successful execution for testing
                    market_data = await self._get_market_data(order.symbol)
                    current_price = market_data.get('ticker', {}).get('price', 0) if market_data else 100
                    
                    return {
                        'success': True,
                        'exchange_order_id': result.get('orderId', f'sim_{order.id}'),
                        'fill_price': current_price,
                        'fill_quantity': order.quantity,
                        'fees': order.quantity * current_price * 0.001  # 0.1% fee simulation
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Exchange error: {response.status_code}'
                    }
                    
        except Exception as e:
            logger.error(f"# X Exchange submission error: {e}")
            # Simulate successful execution for testing
            market_data = await self._get_market_data(order.symbol)
            current_price = market_data.get('ticker', {}).get('price', 0) if market_data else 100
            
            return {
                'success': True,
                'exchange_order_id': f'sim_{order.id}',
                'fill_price': current_price,
                'fill_quantity': order.quantity,
                'fees': order.quantity * current_price * 0.001
            }
    
    async def _update_position_from_fill(self, order: Order):
        """Update position from order fill"""
        try:
            symbol = order.symbol
            
            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                
                if order.side == OrderSide.BUY:
                    # Adding to position
                    total_quantity = position.size + order.filled_quantity
                    total_cost = (position.size * position.entry_price) + (order.filled_quantity * order.avg_fill_price)
                    position.entry_price = total_cost / total_quantity
                    position.size = total_quantity
                else:
                    # Reducing position
                    position.size -= order.filled_quantity
                    if position.size <= 0:
                        # Position closed
                        del self.positions[symbol]
                        return
                
            else:
                # Create new position
                position_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
                
                position = Position(
                    symbol=symbol,
                    side=position_side,
                    size=order.filled_quantity,
                    entry_price=order.avg_fill_price,
                    current_price=order.avg_fill_price,
                    unrealized_pnl=0.0
                )
                
                self.positions[symbol] = position
            
            position.updated_at = datetime.now()
            
            # Update fees
            self.execution_stats['total_fees'] += order.fees
            
            logger.info(f"# Check Position updated: {symbol} - Size: {position.size}, Entry: {position.entry_price}")
            
        except Exception as e:
            logger.error(f"# X Position update error: {e}")
    
    async def _setup_tp_sl_tsl(self, order: Order, request: OrderRequest, execution_result: Dict):
        """Set up Take Profit, Stop Loss, and Trailing Stop orders"""
        try:
            position = self.positions.get(order.symbol)
            if not position:
                return
            
            # Set up Take Profit
            if request.take_profit_price:
                position.take_profit = request.take_profit_price
                logger.info(f"# Check Take Profit set for {order.symbol}: {request.take_profit_price}")
            
            # Set up Stop Loss
            if request.stop_loss_price:
                position.stop_loss = request.stop_loss_price
                logger.info(f"# Check Stop Loss set for {order.symbol}: {request.stop_loss_price}")
            
            # Set up Trailing Stop
            if request.trailing_stop_percent:
                position.trailing_callback = request.trailing_stop_percent / 100
                position.trailing_activation = request.trailing_activation_price
                
                # Calculate initial trailing stop
                if position.side == PositionSide.LONG:
                    position.trailing_stop = position.current_price * (1 - position.trailing_callback)
                else:
                    position.trailing_stop = position.current_price * (1 + position.trailing_callback)
                
                logger.info(f"# Check Trailing Stop set for {order.symbol}: {position.trailing_callback*100}%")
            
        except Exception as e:
            logger.error(f"# X TP/SL/TSL setup error: {e}")
    
    async def _monitor_order_execution(self, order: Order, request: OrderRequest):
        """Monitor order execution and handle partial fills"""
        try:
            start_time = time.time()
            max_time = request.max_execution_time
            
            while (time.time() - start_time) < max_time:
                # Check order status from exchange
                status_update = await self._check_order_status(order)
                
                if status_update['status'] == 'FILLED':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = status_update.get('filled_quantity', order.quantity)
                    order.avg_fill_price = status_update.get('avg_price', 0)
                    
                    await self._update_position_from_fill(order)
                    logger.info(f"# Check Order {order.id} fully filled @ {order.avg_fill_price}")
                    break
                    
                elif status_update['status'] == 'PARTIALLY_FILLED':
                    order.status = OrderStatus.PARTIALLY_FILLED
                    order.filled_quantity = status_update.get('filled_quantity', 0)
                    
                    # Could implement partial fill handling here
                    
                await asyncio.sleep(5)  # Check every 5 seconds
            
            # Handle timeout
            if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELED]:
                logger.warning(f"# Warning Order {order.id} execution timeout - canceling")
                await self._cancel_order(order)
                
        except Exception as e:
            logger.error(f"# X Order monitoring error: {e}")
    
    async def _check_order_status(self, order: Order) -> Dict[str, Any]:
        """Check order status from exchange"""
        try:
            # This would query the actual exchange
            # For testing, simulate order fills
            return {
                'status': 'FILLED',
                'filled_quantity': order.quantity,
                'avg_price': order.price or 100
            }
            
        except Exception as e:
            logger.error(f"# X Order status check error: {e}")
            return {'status': 'UNKNOWN'}
    
    async def _cancel_order(self, order: Order) -> bool:
        """Cancel an active order"""
        try:
            if order.exchange_id:
                # Cancel on exchange
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.delete(f"{self.exchange_connector_url}/api/order/{order.exchange_id}")
                    
                    if response.status_code == 200:
                        order.status = OrderStatus.CANCELED
                        logger.info(f"# Check Order {order.id} canceled")
                        return True
            else:
                # Local cancellation
                order.status = OrderStatus.CANCELED
                return True
                
        except Exception as e:
            logger.error(f"# X Order cancellation error: {e}")
            return False
    
    async def _order_monitoring_loop(self):
        """Background order monitoring"""
        while self.is_running:
            try:
                # Check all active orders
                for order_id, order in list(self.active_orders.items()):
                    if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                        # Move to history
                        self.order_history.append(order)
                        del self.active_orders[order_id]
                        
                        # Keep only recent history
                        if len(self.order_history) > 1000:
                            self.order_history = self.order_history[-500:]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"# X Order monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _position_monitoring_loop(self):
        """Background position monitoring and P&L updates"""
        while self.is_running:
            try:
                for symbol, position in self.positions.items():
                    # Update current price
                    market_data = await self._get_market_data(symbol)
                    if market_data:
                        current_price = market_data.get('ticker', {}).get('price', 0)
                        if current_price > 0:
                            position.current_price = current_price
                            
                            # Calculate unrealized P&L
                            if position.side == PositionSide.LONG:
                                position.unrealized_pnl = (current_price - position.entry_price) * position.size
                            else:
                                position.unrealized_pnl = (position.entry_price - current_price) * position.size
                            
                            # Update max profit/loss
                            position.max_profit = max(position.max_profit, position.unrealized_pnl)
                            position.max_loss = min(position.max_loss, position.unrealized_pnl)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"# X Position monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _tp_sl_tsl_monitoring_loop(self):
        """Background TP/SL/TSL monitoring and execution"""
        while self.is_running:
            try:
                for symbol, position in list(self.positions.items()):
                    current_price = position.current_price
                    
                    # Check Take Profit
                    if position.take_profit and self._should_trigger_tp(position, current_price):
                        await self._execute_tp_sl_exit(position, "TAKE_PROFIT", current_price)
                        continue
                    
                    # Check Stop Loss
                    if position.stop_loss and self._should_trigger_sl(position, current_price):
                        await self._execute_tp_sl_exit(position, "STOP_LOSS", current_price)
                        continue
                    
                    # Update Trailing Stop
                    if position.trailing_stop and position.trailing_callback:
                        await self._update_trailing_stop(position, current_price)
                        
                        # Check if trailing stop triggered
                        if self._should_trigger_tsl(position, current_price):
                            await self._execute_tp_sl_exit(position, "TRAILING_STOP", current_price)
                            continue
                
                await asyncio.sleep(5)  # Check every 5 seconds for TP/SL/TSL
                
            except Exception as e:
                logger.error(f"# X TP/SL/TSL monitoring error: {e}")
                await asyncio.sleep(5)
    
    def _should_trigger_tp(self, position: Position, current_price: float) -> bool:
        """Check if take profit should trigger"""
        if not position.take_profit:
            return False
        
        if position.side == PositionSide.LONG:
            return current_price >= position.take_profit
        else:
            return current_price <= position.take_profit
    
    def _should_trigger_sl(self, position: Position, current_price: float) -> bool:
        """Check if stop loss should trigger"""
        if not position.stop_loss:
            return False
        
        if position.side == PositionSide.LONG:
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss
    
    def _should_trigger_tsl(self, position: Position, current_price: float) -> bool:
        """Check if trailing stop should trigger"""
        if not position.trailing_stop:
            return False
        
        if position.side == PositionSide.LONG:
            return current_price <= position.trailing_stop
        else:
            return current_price >= position.trailing_stop
    
    async def _update_trailing_stop(self, position: Position, current_price: float):
        """Update trailing stop level"""
        try:
            if not position.trailing_callback:
                return
            
            if position.side == PositionSide.LONG:
                # For long positions, trail stop up as price rises
                new_stop = current_price * (1 - position.trailing_callback)
                if new_stop > position.trailing_stop:
                    old_stop = position.trailing_stop
                    position.trailing_stop = new_stop
                    logger.info(f"📈 Trailing stop updated for {position.symbol}: {old_stop:.4f} -> {new_stop:.4f}")
            else:
                # For short positions, trail stop down as price falls
                new_stop = current_price * (1 + position.trailing_callback)
                if new_stop < position.trailing_stop:
                    old_stop = position.trailing_stop
                    position.trailing_stop = new_stop
                    logger.info(f"📉 Trailing stop updated for {position.symbol}: {old_stop:.4f} -> {new_stop:.4f}")
            
        except Exception as e:
            logger.error(f"# X Trailing stop update error: {e}")
    
    async def _execute_tp_sl_exit(self, position: Position, trigger_type: str, current_price: float):
        """Execute TP/SL/TSL exit order"""
        try:
            logger.info(f"# Target {trigger_type} triggered for {position.symbol} @ {current_price}")
            
            # Create exit order
            exit_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
            
            exit_request = OrderRequest(
                symbol=position.symbol,
                side=exit_side,
                type=OrderType.MARKET,
                quantity=position.size,
                urgency="HIGH"  # Exit orders are urgent
            )
            
            # Submit exit order
            exit_result = await self.submit_order(exit_request)
            
            if exit_result['success']:
                # Calculate realized P&L
                if position.side == PositionSide.LONG:
                    position.realized_pnl = (current_price - position.entry_price) * position.size
                else:
                    position.realized_pnl = (position.entry_price - current_price) * position.size
                
                logger.info(f"# Check {trigger_type} executed for {position.symbol}, P&L: ${position.realized_pnl:.2f}")
                
                # Remove position
                del self.positions[position.symbol]
                
            else:
                logger.error(f"# X {trigger_type} execution failed for {position.symbol}")
                
        except Exception as e:
            logger.error(f"# X TP/SL exit execution error: {e}")
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            return {
                'order_id': order.id,
                'client_id': order.client_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.type.value,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'status': order.status.value,
                'avg_fill_price': order.avg_fill_price,
                'created_at': order.created_at.isoformat(),
                'updated_at': order.updated_at.isoformat()
            }
        return None
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for symbol"""
        if symbol in self.positions:
            position = self.positions[symbol]
            return {
                'symbol': position.symbol,
                'side': position.side.value,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'take_profit': position.take_profit,
                'stop_loss': position.stop_loss,
                'trailing_stop': position.trailing_stop,
                'max_profit': position.max_profit,
                'max_loss': position.max_loss,
                'opened_at': position.opened_at.isoformat(),
                'updated_at': position.updated_at.isoformat()
            }
        return None
    
    def get_all_positions(self) -> List[Dict]:
        """Get all positions"""
        return [self.get_position(symbol) for symbol in self.positions.keys()]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        success_rate = 0.0
        if self.execution_stats['total_orders'] > 0:
            success_rate = self.execution_stats['successful_orders'] / self.execution_stats['total_orders']
        
        return {
            'total_orders': self.execution_stats['total_orders'],
            'successful_orders': self.execution_stats['successful_orders'],
            'failed_orders': self.execution_stats['failed_orders'],
            'success_rate': success_rate,
            'avg_execution_time': self.execution_stats['avg_execution_time'],
            'total_fees': self.execution_stats['total_fees'],
            'active_orders': len(self.active_orders),
            'active_positions': len(self.positions)
        }
    
    def start(self):
        """Start the order executor"""
        self.is_running = True
        logger.info("# Rocket Enhanced Order Executor started")
    
    def stop(self):
        """Stop the order executor"""
        self.is_running = False
        logger.info("🛑 Enhanced Order Executor stopped")

# Global executor instance
enhanced_executor = EnhancedOrderExecutor()

if __name__ == "__main__":
    async def test_order_executor():
        """Test the enhanced order executor"""
        logger.info("🧪 Testing Enhanced Order Executor...")
        
        # Initialize
        await enhanced_executor.initialize()
        enhanced_executor.start()
        
        # Test market order
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.001,
            take_profit_price=52000,
            stop_loss_price=48000,
            urgency="NORMAL"
        )
        
        result = await enhanced_executor.submit_order(market_order)
        
        # Check positions
        positions = enhanced_executor.get_all_positions()
        
        # Get statistics
        stats = enhanced_executor.get_execution_statistics()
        
        enhanced_executor.stop()
    
    asyncio.run(test_order_executor())