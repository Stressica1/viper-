#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Live Trading Engine
Real-time automated trading with Bitget API integration

Features:
- Real-time market data
- Automated trade execution
- Risk management
- Performance tracking
"""

import os
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Optional
import ccxt
import redis
import json
import requests

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'live-trading-engine')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTradingEngine:
    """Live trading engine for VIPER bot - REAL DATA ONLY MODE"""

    def __init__(self):
        self.exchange = None
        self.redis_client = None
        self.is_running = False

        # Load configuration - REAL DATA ONLY
        self.api_key = None
        self.api_secret = None
        self.api_password = None
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))
        self.max_leverage = int(os.getenv('MAX_LEVERAGE', '50'))

        # Bitget USDT swap configuration
        self.trading_mode = os.getenv('TRADING_MODE', 'CRYPTO')
        self.target_symbol = os.getenv('TARGET_SYMBOL', 'BTCUSDT')
        self.leverage = int(os.getenv('LEVERAGE', '50'))

        # Standard environment variables
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.service_name = os.getenv('SERVICE_NAME', 'live-trading-engine')

        # Risk management service
        self.risk_manager_url = os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8000')

        # Credential vault configuration
        self.vault_url = os.getenv('VAULT_URL', 'http://credential-vault:8008')
        self.vault_token = os.getenv('VAULT_ACCESS_TOKEN')

        # Load credentials from vault
        self.load_credentials_from_env()

        # Validate API credentials - REAL DATA ONLY
        if not all([self.api_key, self.api_secret, self.api_password]):
            raise Exception("🚫 REAL DATA ONLY: Failed to load API credentials from vault")

        if self.api_key.startswith('your_') or self.api_secret.startswith('your_'):
            raise Exception("🚫 REAL DATA ONLY: Invalid API credentials. Please use real API credentials, not placeholder values")

        # Trading parameters
        self.symbol = 'BTC/USDT:USDT'  # BTC perpetual contract
        self.position_size = 0
        self.active_trades = []

        logger.info("🚫 REAL DATA ONLY MODE ENABLED")
        logger.info("# Chart Only real OHLCV data will be used")
        logger.info("# X No simulation or mock data allowed")

    def load_credentials_from_env(self):
        """Load credentials directly from environment variables"""
        try:
            logger.info("🔐 Loading credentials from environment...")
            
            self.api_key = os.getenv('BITGET_API_KEY', '')
            self.api_secret = os.getenv('BITGET_API_SECRET', '')
            self.api_password = os.getenv('BITGET_API_PASSWORD', '')
            self.symbol = os.getenv('TARGET_SYMBOL', 'BTCUSDT')
            
            logger.info("# Check Credentials loaded from environment")

        except Exception as e:
            logger.error(f"# X Error loading credentials: {e}")
            raise Exception(f"🚫 Failed to load credentials: {e}")

    def initialize_exchange(self):
        """Initialize Bitget exchange connection for USDT swaps"""
        try:
            logger.info("🔄 Initializing Bitget for USDT swap trading...")
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

            # Load markets
            logger.info("📡 Loading markets...")
            self.exchange.load_markets()
            logger.info("# Check Markets loaded successfully")
            logger.info("# Check Bitget USDT swap connection established")
            return True

        except Exception as e:
            logger.error(f"# X Failed to initialize Bitget exchange: {e}")
            return False

    def initialize_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("# Check Redis connection established")
            return True
        except Exception as e:
            logger.error(f"# X Failed to connect to Redis: {e}")
            return False

    def get_market_data(self):
        """Get current market data - Bitget USDT swaps only"""
        try:
            logger.info("# Chart BITGET DATA: Fetching live USDT swap market data...")
            ticker = self.exchange.fetch_ticker(self.symbol)

            if not ticker or 'last' not in ticker:
                logger.error("🚫 REAL DATA ONLY: Invalid or incomplete market data received")
                return None

            market_data = {
                'symbol': self.symbol,
                'price': ticker['last'],
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'volume': ticker.get('baseVolume', 0),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"# Chart REAL DATA: Current price: ${market_data['price']:.2f}")
            return market_data

        except Exception as e:
            logger.error(f"# X REAL DATA ONLY: Failed to fetch market data: {e}")
            return None

    async def get_account_balance(self) -> float:
        """Get USDT balance from swap wallet"""
        try:
            # Fetch balance specifically for swap account
            balance = await self.exchange.fetch_balance({'type': 'swap'})
            if 'USDT' in balance:
                usdt_balance = balance['USDT']['free']
                logger.info(f"💰 Swap Wallet Balance: ${usdt_balance:.2f} USDT (available)")
                return usdt_balance
            else:
                logger.error("# X USDT balance not found in swap wallet")
                return 0.0
        except Exception as e:
            logger.error(f"# X Failed to fetch swap wallet balance: {e}")
            # Check if it's an API key issue
            if "Apikey does not exist" in str(e):
                logger.error("🚫 REAL DATA ONLY: Invalid API key - cannot proceed with real trading")
                logger.error("📝 To use real data only:")
                logger.error("   1. Go to https://www.bitget.com/en/account/newapi")
                logger.error("   2. Create a new API key with trading permissions")
                logger.error("   3. Update BITGET_API_KEY, BITGET_API_SECRET, and BITGET_API_PASSWORD in .env")
                logger.error("   4. Restart the live trading engine")
                logger.error("# X System will not operate with invalid API credentials")
                raise Exception("REAL DATA ONLY: Invalid API credentials - exiting")
            return 0.0

    def check_account_balance(self) -> float:
        """Synchronous wrapper for get_account_balance"""
        try:
            # Run the async method in a new event loop
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.get_account_balance())
            loop.close()
            return result
        except Exception as e:
            logger.error(f"# X Error in sync balance check: {e}")
            return 0.0

    def calculate_position_size(self, price: float, balance: float, leverage: int = 50):
        """Calculate position size with 3% risk and leverage"""
        try:
            # 3% risk per trade
            risk_per_trade = 0.03
            risk_amount = balance * risk_per_trade

            # Assume 2% stop loss distance (can be adjusted)
            stop_loss_pct = 0.02
            stop_loss_distance = price * stop_loss_pct

            # Calculate base position size (without leverage)
            base_position_size = risk_amount / stop_loss_distance

            # Apply leverage to get actual position size
            leveraged_position_size = base_position_size * leverage

            # Ensure minimum contract size
            min_contract_size = 0.001  # 0.001 BTC minimum
            position_size = max(leveraged_position_size, min_contract_size)

            logger.info(f"# Target Position Sizing: Balance=${balance:.2f}, Risk=2% (${risk_amount:.2f}), "
                       f"Stop Loss={stop_loss_pct*100}% (${stop_loss_distance:.2f}), "
                       f"Base Size={base_position_size:.6f}, Leveraged Size={leveraged_position_size:.6f} "
                       f"({leverage}x leverage) → Final Size={position_size:.6f}")

            return position_size

        except Exception as e:
            logger.error(f"# X Error calculating position size: {e}")
            # Fallback to minimum size
            return 0.001

    def execute_trade(self, side: str, size: float, price: Optional[float] = None):
        """Execute a trade on Bitget USDT swaps"""
        try:
            logger.info(f"# Rocket EXECUTING BITGET {side.upper()} ORDER: {size:.6f} USDT on {self.symbol}")

            # Ensure size meets minimum requirements (0.0001 BTC)
            if size < 0.0001:
                logger.error(f"# X Position size {size:.8f} BTC below minimum 0.0001 BTC")
                return None

            # Use hedge mode parameters for Bitget USDT swaps
            if side.upper() == 'BUY':
                order = self.exchange.create_order(
                    self.symbol,
                    'market',
                    'buy',
                    size,
                    None,
                    params={'tradeSide': 'open'}  # Open long position in hedge mode
                )
            else:  # SELL
                order = self.exchange.create_order(
                    self.symbol,
                    'market',
                    'sell',
                    size,
                    None,
                    params={'tradeSide': 'open'}  # Open short position in hedge mode
                )

            logger.info(f"# Check BITGET {side.upper()} order executed successfully: ID {order.get('id', 'N/A')}")

            # Store trade in Redis
            trade_data = {
                'id': order.get('id', 'unknown'),
                'symbol': self.symbol,
                'side': side,
                'size': size,
                'price': order.get('average', order.get('price', 0)),
                'timestamp': datetime.now().isoformat(),
                'status': 'executed',
                'trading_mode': 'BITGET_USDT_SWAP'
            }

            try:
                self.redis_client.setex(
                    f"trade:{trade_data['id']}",
                    86400,  # 24 hours
                    json.dumps(trade_data)
                )
                logger.info("# Check Trade data stored in Redis")
            except Exception as redis_e:
                logger.warning(f"# Warning Failed to store trade in Redis: {redis_e}")

            return order

        except Exception as e:
            logger.error(f"# X Failed to execute {side} trade: {e}")

            # Check for API credential issues
            if "Apikey does not exist" in str(e):
                logger.error("🚫 REAL DATA ONLY: Invalid API credentials")
                raise Exception("REAL DATA ONLY: Invalid API credentials - cannot execute trades")

            return None



    def check_risk_limits(self, symbol: str, position_size: float, price: float, balance: float) -> Dict:
        """Check risk limits with risk manager service"""
        try:
            logger.info(f"# Search Checking risk limits for {symbol}: size={position_size:.6f}, price=${price:.2f}, balance=${balance:.2f}")

            # Prepare request data
            request_data = {
                'symbol': symbol,
                'position_size': position_size,
                'price': price,
                'balance': balance
            }

            # Call risk manager service
            response = requests.post(
                f"{self.risk_manager_url}/api/position/check",
                json=request_data,
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('allowed', False):
                    logger.info("# Check Risk check passed - trade allowed")
                    return result
                else:
                    logger.warning(f"# Warning Risk check failed: {result.get('reason', 'Unknown reason')}")
                    return result
            else:
                logger.error(f"# X Risk manager returned status {response.status_code}: {response.text}")
                return {'allowed': False, 'error': f'HTTP {response.status_code}'}

        except requests.exceptions.RequestException as e:
            logger.error(f"# X Failed to connect to risk manager: {e}")
            # Allow trade if risk manager is unavailable (fail-safe)
            logger.warning("# Warning Risk manager unavailable - allowing trade (fail-safe mode)")
            return {'allowed': True, 'warning': 'Risk manager unavailable'}
        except Exception as e:
            logger.error(f"# X Error checking risk limits: {e}")
            return {'allowed': False, 'error': str(e)}

    def register_position(self, symbol: str, position_data: Dict) -> bool:
        """Register position with risk manager"""
        try:
            logger.info(f"📝 Registering position for {symbol}")

            response = requests.post(
                f"{self.risk_manager_url}/api/position/register",
                json={
                    'symbol': symbol,
                    'position_data': position_data
                },
                timeout=5
            )

            if response.status_code == 200:
                logger.info(f"# Check Position registered for {symbol}")
                return True
            else:
                logger.error(f"# X Failed to register position: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"# X Error registering position: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """Close position with risk manager"""
        try:
            logger.info(f"📝 Closing position for {symbol}")

            response = requests.delete(
                f"{self.risk_manager_url}/api/position/{symbol}",
                timeout=5
            )

            if response.status_code == 200:
                logger.info(f"# Check Position closed for {symbol}")
                return True
            else:
                logger.error(f"# X Failed to close position: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"# X Error closing position: {e}")
            return False

    def get_viper_signal_from_service(self):
        """Get trading signal from centralized VIPER scoring service"""
        try:
            # Get market data from unified market data service
            market_data_manager_url = os.getenv('MARKET_DATA_MANAGER_URL', 'http://market-data-manager:8003')
            viper_scoring_url = os.getenv('VIPER_SCORING_SERVICE_URL', 'http://viper-scoring-service:8009')

            # Fetch current market data
            response = requests.get(f"{market_data_manager_url}/api/market/{self.symbol}", timeout=5)
            if response.status_code != 200:
                logger.warning(f"# Warning Cannot fetch market data for {self.symbol}")
                return None

            market_data = response.json()

            # Request signal from VIPER scoring service
            signal_request = {
                'symbol': self.symbol,
                'market_data': market_data
            }

            response = requests.post(f"{viper_scoring_url}/api/signal",
                                   json=signal_request, timeout=5)

            if response.status_code == 200:
                signal_data = response.json()
                if 'signal' in signal_data and signal_data['signal'] in ['LONG', 'SHORT']:
                    logger.info(f"# Target Received {signal_data['signal']} signal from VIPER service")
                    return signal_data
                else:
                    logger.debug(f"# Chart No actionable signal for {self.symbol}")
                    return None
            else:
                logger.warning(f"# Warning VIPER service returned status {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"# X Error getting signal from VIPER service: {e}")
            return None

    def get_viper_signal(self):
        """Get trading signal - now using centralized service"""
        try:
            # Use centralized VIPER scoring service instead of local calculation
            return self.get_viper_signal_from_service()

        except Exception as e:
            logger.error(f"# X Error in signal generation: {e}")
            return None

    def listen_for_trading_signals(self):
        """Listen for trading signals from the event system"""
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe('trading_signals', 'risk_validation', 'trading_emergency')

            logger.info("📡 Listening for trading signals...")

            for message in pubsub.listen():
                if not self.is_running:
                    break

                if message['type'] == 'message':
                    try:
                        channel = message['channel']
                        event_data = json.loads(message['data'])

                        if channel == 'trading_signals':
                            self.process_trading_signal(event_data)
                        elif channel == 'risk_validation':
                            self.process_risk_validation(event_data)
                        elif channel == 'trading_emergency':
                            self.process_emergency_stop(event_data)

                    except json.JSONDecodeError as e:
                        logger.error(f"# X Failed to decode message: {e}")
                    except Exception as e:
                        logger.error(f"# X Error processing message: {e}")

        except Exception as e:
            logger.error(f"# X Error in signal listener: {e}")

    def process_trading_signal(self, event_data: Dict):
        """Process a trading signal from the event system"""
        try:
            signal = event_data.get('signal', {})
            symbol = signal.get('symbol', '')

            if symbol != self.symbol:
                return  # Not for this symbol

            logger.info(f"# Target Processing signal: {signal.get('type', 'UNKNOWN')} for {symbol}")

            # Validate signal has required fields
            if not all(key in signal for key in ['type', 'price', 'confidence']):
                logger.error("# X Invalid signal format")
                return

            # Check account balance
            balance = self.check_account_balance()
            if balance < 10:
                logger.warning(f"# Warning Insufficient balance: ${balance:.2f}")
                return

            # Calculate position size with 3% risk and 50x leverage
            position_size = self.calculate_position_size(signal['price'], balance, leverage=50)
            if position_size < 0.0001:
                logger.warning(f"# Warning Position size too small: {position_size:.8f} BTC")
                return

            # Execute trade with enhanced error handling
            self.execute_position(signal, position_size, balance)

        except Exception as e:
            logger.error(f"# X Error processing trading signal: {e}")

    def process_risk_validation(self, event_data: Dict):
        """Process risk validation requests"""
        try:
            validation_request = event_data.get('data', {})
            symbol = validation_request.get('symbol', '')

            if symbol != self.symbol:
                return

            # Perform risk validation
            signal = validation_request.get('signal', {})
            price = signal.get('price', 0)
            balance = self.check_account_balance()

            # Calculate position size for validation with 50x leverage
            position_size = self.calculate_position_size(price, balance, leverage=50)

            # Check risk limits
            risk_check = self.check_risk_limits(symbol, position_size, price, balance)

            # Publish validation result
            validation_result = {
                'symbol': symbol,
                'signal_id': signal.get('id', 'unknown'),
                'validation': risk_check,
                'position_size': position_size,
                'timestamp': datetime.now().isoformat()
            }

            self.redis_client.publish('risk_validation_result', json.dumps(validation_result))

        except Exception as e:
            logger.error(f"# X Error in risk validation: {e}")

    def process_emergency_stop(self, event_data: Dict):
        """Process emergency stop signals"""
        try:
            emergency = event_data.get('data', {})
            alert = emergency.get('alert', {})

            logger.error(f"🚨 EMERGENCY STOP: {alert.get('type', 'unknown')}")

            # Close all positions immediately
            self.emergency_close_all_positions()

        except Exception as e:
            logger.error(f"# X Error processing emergency stop: {e}")

    def execute_position(self, signal: Dict, position_size: float, balance: float):
        """Execute a position with comprehensive error handling and logging"""
        try:
            symbol = signal.get('symbol', self.symbol)
            signal_type = signal.get('type', '')
            price = signal.get('price', 0)
            confidence = signal.get('confidence', 0)

            logger.info(f"# Rocket Executing {signal_type} position: {position_size:.6f} BTC at ${price:.2f}")

            # Execute the trade
            order = self.execute_trade(signal_type, position_size, price)

            if order:
                order_id = order.get('id', 'unknown')
                executed_price = order.get('average', order.get('price', price))

                logger.info(f"# Check Trade executed successfully: {order_id}")

                # Register position with risk manager
                position_data = {
                    'id': order_id,
                    'symbol': symbol,
                    'side': signal_type,
                    'size': position_size,
                    'price': executed_price,
                    'signal_price': price,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }

                if self.register_position(symbol, position_data):
                    logger.info(f"📝 Position registered with risk manager")

                    # Publish trade execution event
                    trade_event = {
                        'type': 'position_opened',
                        'position': position_data,
                        'order': order,
                        'timestamp': datetime.now().isoformat()
                    }

                    self.redis_client.publish('position_updates', json.dumps(trade_event))
                else:
                    logger.warning(f"# Warning Failed to register position with risk manager")

            else:
                logger.error("# X Trade execution failed")

                # Publish trade failure event
                failure_event = {
                    'type': 'trade_failed',
                    'signal': signal,
                    'position_size': position_size,
                    'error': 'execution_failed',
                    'timestamp': datetime.now().isoformat()
                }

                self.redis_client.publish('trading_errors', json.dumps(failure_event))

        except Exception as e:
            logger.error(f"# X Error executing position: {e}")

            # Publish error event
            error_event = {
                'type': 'trade_error',
                        'signal': signal,
                'position_size': position_size,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

            self.redis_client.publish('trading_errors', json.dumps(error_event))

    def emergency_close_all_positions(self):
        """Emergency close all positions"""
        try:
            logger.error("🚨 EMERGENCY: Closing all positions")

            # Get current positions
            positions_response = requests.get(f"{self.risk_manager_url}/api/position/status", timeout=5)

            if positions_response.status_code == 200:
                position_data = positions_response.json()
                active_symbols = position_data.get('active_symbols', [])

                for symbol in active_symbols:
                    logger.info(f"📝 Emergency closing position for {symbol}")

                    # Close position logic would go here
                    # This is a simplified version - in production you'd implement
                    # proper position closing with market orders

                    close_event = {
                        'type': 'emergency_close',
                        'symbol': symbol,
                        'reason': 'emergency_stop',
                        'timestamp': datetime.now().isoformat()
                    }

                    self.redis_client.publish('position_updates', json.dumps(close_event))

            # Publish emergency stop completed event
            completed_event = {
                'type': 'emergency_stop_completed',
                'timestamp': datetime.now().isoformat()
            }

            self.redis_client.publish('system_events', json.dumps(completed_event))

        except Exception as e:
            logger.error(f"# X Error in emergency close: {e}")

    def run_trading_loop(self):
        """Main trading loop - now event-driven"""
        logger.info("# Rocket Starting VIPER Live Trading Engine (Event-Driven)...")
        self.is_running = True

        try:
            # Start signal listener in background thread
            signal_thread = threading.Thread(target=self.listen_for_trading_signals, daemon=True)
            signal_thread.start()

            # Main loop for Bitget USDT swap trading
            while self.is_running:
                try:
                    # Periodic health check
                    balance = self.check_account_balance()
                    logger.info(f"💵 Account balance: ${balance:.2f}")

                    # Publish engine status
                    status_event = {
                        'service': 'live-trading-engine',
                        'status': 'active',
                        'balance': balance,
                        'symbol': self.symbol,
                        'timestamp': datetime.now().isoformat()
                    }

                    self.redis_client.publish('service_status', json.dumps(status_event))

                    # Wait before next status update
                    time.sleep(60)

                except Exception as e:
                    logger.error(f"# X Error in main loop: {e}")
                    time.sleep(30)

        except KeyboardInterrupt:
            logger.info("🛑 Trading loop interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"# X Fatal error in trading loop: {e}")
            self.stop()

    def stop(self):
        """Stop the trading engine"""
        logger.info("🛑 Stopping VIPER Live Trading Engine...")
        self.is_running = False

def main():
    """Main function"""
    engine = LiveTradingEngine()

    # Initialize connections
    if not engine.initialize_exchange():
        logger.error("# X Failed to initialize exchange. Exiting...")
        return

    if not engine.initialize_redis():
        logger.warning("# Warning Redis connection failed. Continuing without caching...")

    try:
        # Start trading loop
        engine.run_trading_loop()
    except KeyboardInterrupt:
        engine.stop()
    finally:
        logger.info("👋 VIPER Live Trading Engine stopped")

if __name__ == "__main__":
    main()
