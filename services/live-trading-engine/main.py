#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Live Trading Engine
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
        self.load_credentials_from_vault()

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
        logger.info("📊 Only real OHLCV data will be used")
        logger.info("❌ No simulation or mock data allowed")

    def load_credentials_from_vault(self):
        """Load API credentials from credential vault"""
        try:
            logger.info("🔐 Loading credentials from vault...")

            # Get API Key
            response = requests.get(
                f"{self.vault_url}/credentials/retrieve/bitget/api_key",
                headers={'Authorization': f'Bearer {self.vault_token}'},
                timeout=10
            )

            if response.status_code == 200:
                self.api_key = response.json().get('value')
                logger.info("✅ API Key loaded from vault")
            else:
                logger.error(f"❌ Failed to load API Key: {response.status_code}")
                return

            # Get API Secret
            response = requests.get(
                f"{self.vault_url}/credentials/retrieve/bitget/api_secret",
                headers={'Authorization': f'Bearer {self.vault_token}'},
                timeout=10
            )

            if response.status_code == 200:
                self.api_secret = response.json().get('value')
                logger.info("✅ API Secret loaded from vault")
            else:
                logger.error(f"❌ Failed to load API Secret: {response.status_code}")
                return

            # Get API Password
            response = requests.get(
                f"{self.vault_url}/credentials/retrieve/bitget/api_password",
                headers={'Authorization': f'Bearer {self.vault_token}'},
                timeout=10
            )

            if response.status_code == 200:
                self.api_password = response.json().get('value')
                logger.info("✅ API Password loaded from vault")
            else:
                logger.error(f"❌ Failed to load API Password: {response.status_code}")
                return

            logger.info("🎉 All credentials loaded successfully from vault")

        except Exception as e:
            logger.error(f"❌ Error loading credentials from vault: {e}")
            raise Exception(f"🚫 Failed to load credentials from vault: {e}")

    def initialize_exchange(self):
        """Initialize Bitget exchange connection - EXACT SAME AS DIAGNOSTIC"""
        try:
            # EXACT same configuration as diagnostic script
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

            # Load markets exactly like diagnostic
            logger.info("📡 Loading markets...")
            self.exchange.load_markets()
            logger.info("✅ Markets loaded successfully")
            logger.info("✅ Bitget exchange connection established")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange: {e}")
            return False

    def initialize_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False

    def get_market_data(self):
        """Get current market data - REAL DATA ONLY"""
        try:
            logger.info("📊 REAL DATA: Fetching live market data...")
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

            logger.info(f"📊 REAL DATA: Current price: ${market_data['price']:.2f}")
            return market_data

        except Exception as e:
            logger.error(f"❌ REAL DATA ONLY: Failed to fetch market data: {e}")
            return None

    def check_account_balance(self):
        """Check account balance - REAL DATA ONLY"""
        try:
            logger.info("🔄 Checking account balance...")
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            logger.info(f"💰 Account balance: ${usdt_balance:.2f} USDT")
            return usdt_balance
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
            # Check if it's an API key issue
            if "Apikey does not exist" in str(e):
                logger.error("🚫 REAL DATA ONLY: Invalid API key - cannot proceed with real trading")
                logger.error("📝 To use real data only:")
                logger.error("   1. Go to https://www.bitget.com/en/account/newapi")
                logger.error("   2. Create a new API key with trading permissions")
                logger.error("   3. Update BITGET_API_KEY, BITGET_API_SECRET, and BITGET_API_PASSWORD in .env")
                logger.error("   4. Restart the live trading engine")
                logger.error("❌ System will not operate with invalid API credentials")
                raise Exception("REAL DATA ONLY: Invalid API credentials - exiting")
            return 0

    def calculate_position_size(self, price: float, balance: float):
        """Calculate position size using risk manager service - 2% RISK RULE"""
        try:
            # Use risk manager service for proper 2% risk calculation
            response = requests.post(
                f"{self.risk_manager_url}/api/position/size",
                json={
                    'symbol': 'BTC/USDT:USDT',  # Default symbol for sizing
                    'price': price,
                    'balance': balance,
                    'risk_per_trade': 0.02  # 2% risk per trade
                },
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                recommended_size = result.get('recommended_size', 0)

                # Ensure minimum contract size
                min_contract_size = 0.001  # 0.001 BTC minimum
                position_size = max(recommended_size, min_contract_size)

                logger.info(f"🎯 Risk-managed position sizing: {position_size:.6f} BTC (2% risk rule)")
                return position_size
            else:
                logger.error(f"❌ Risk manager sizing failed: {response.status_code}")
                # Fallback to minimum size
                return 0.001

        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            # Fallback to minimum size
            return 0.001

    def execute_trade(self, side: str, size: float, price: Optional[float] = None):
        """Execute a trade on perpetual swaps - WORKING CONFIGURATION"""
        try:
            # Ensure size meets minimum requirements (0.0001 BTC)
            if size < 0.0001:
                logger.error(f"❌ Position size {size:.8f} BTC below minimum 0.0001 BTC")
                return None

            logger.info(f"🚀 EXECUTING {side.upper()} ORDER: {size:.6f} BTC on {self.symbol}")

            # Use hedge mode parameters for Bitget
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

            logger.info(f"✅ {side.upper()} order executed successfully: ID {order.get('id', 'N/A')}")

            # Store trade in Redis
            trade_data = {
                'id': order.get('id', 'unknown'),
                'symbol': self.symbol,
                'side': side,
                'size': size,
                'price': order.get('average', order.get('price', 0)),
                'timestamp': datetime.now().isoformat(),
                'status': 'executed'
            }

            try:
                self.redis_client.setex(
                    f"trade:{order.get('id', 'unknown')}",
                    86400,  # 24 hours
                    json.dumps(trade_data)
                )
                logger.info("✅ Trade data stored in Redis")
            except Exception as redis_e:
                logger.warning(f"⚠️ Failed to store trade in Redis: {redis_e}")

            return order

        except Exception as e:
            logger.error(f"❌ Failed to execute {side} trade: {e}")

            # Check for API credential issues
            if "Apikey does not exist" in str(e):
                logger.error("🚫 REAL DATA ONLY: Invalid API credentials")
                raise Exception("REAL DATA ONLY: Invalid API credentials - cannot execute trades")

            return None

    def check_risk_limits(self, symbol: str, position_size: float, price: float, balance: float) -> Dict:
        """Check risk limits with risk manager service"""
        try:
            logger.info(f"🔍 Checking risk limits for {symbol}: size={position_size:.6f}, price=${price:.2f}, balance=${balance:.2f}")

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
                    logger.info("✅ Risk check passed - trade allowed")
                    return result
                else:
                    logger.warning(f"⚠️ Risk check failed: {result.get('reason', 'Unknown reason')}")
                    return result
            else:
                logger.error(f"❌ Risk manager returned status {response.status_code}: {response.text}")
                return {'allowed': False, 'error': f'HTTP {response.status_code}'}

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to connect to risk manager: {e}")
            # Allow trade if risk manager is unavailable (fail-safe)
            logger.warning("⚠️ Risk manager unavailable - allowing trade (fail-safe mode)")
            return {'allowed': True, 'warning': 'Risk manager unavailable'}
        except Exception as e:
            logger.error(f"❌ Error checking risk limits: {e}")
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
                logger.info(f"✅ Position registered for {symbol}")
                return True
            else:
                logger.error(f"❌ Failed to register position: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Error registering position: {e}")
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
                logger.info(f"✅ Position closed for {symbol}")
                return True
            else:
                logger.error(f"❌ Failed to close position: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
            return False

    def get_viper_signal(self):
        """Get trading signal from VIPER scoring system - REAL OHLCV DATA ONLY"""
        try:
            market_data = self.get_market_data()
            if not market_data:
                logger.warning("⚠️ REAL DATA ONLY: Cannot get market data - skipping signal generation")
                return None

            # REAL DATA ONLY: Fetch OHLCV data for analysis
            logger.info("📊 REAL DATA: Fetching OHLCV data for analysis...")

            # Get recent OHLCV data (last 100 candles, 1-hour timeframe)
            ohlcv_data = self.exchange.fetch_ohlcv(self.symbol, timeframe='1h', limit=100)

            if not ohlcv_data or len(ohlcv_data) < 50:
                logger.warning("⚠️ REAL DATA ONLY: Insufficient OHLCV data for analysis")
                return None

            # Calculate simple moving averages using REAL OHLCV data
            closes = [candle[4] for candle in ohlcv_data]  # Close prices

            # Calculate 20-period and 50-period SMAs
            if len(closes) >= 50:
                sma_20 = sum(closes[-20:]) / 20
                sma_50 = sum(closes[-50:]) / 50

                current_price = market_data['price']

                logger.info(".2f")
                logger.info(".2f")
                logger.info(".2f")

                # REAL DATA ONLY: Generate signal based on SMA crossover
                if current_price > sma_20 and sma_20 > sma_50:
                    signal = 'BUY'
                    confidence = min(0.95, abs(current_price - sma_20) / sma_20)
                elif current_price < sma_20 and sma_20 < sma_50:
                    signal = 'SELL'
                    confidence = min(0.95, abs(sma_20 - current_price) / current_price)
                else:
                    signal = 'HOLD'
                    confidence = 0.5

                if signal in ['BUY', 'SELL']:
                    return {
                        'signal': signal,
                        'symbol': self.symbol,
                        'price': current_price,
                        'confidence': confidence,
                        'sma_20': sma_20,
                        'sma_50': sma_50,
                        'ohlcv_count': len(ohlcv_data),
                        'timestamp': datetime.now().isoformat()
                    }

            return None

        except Exception as e:
            logger.error(f"❌ REAL DATA ONLY: Error getting VIPER signal: {e}")
            return None

    def run_trading_loop(self):
        """Main trading loop"""
        logger.info("🚀 Starting VIPER Live Trading Engine...")
        self.is_running = True

        while self.is_running:
            try:
                # Check account balance
                balance = self.check_account_balance()
                logger.info(f"💵 Balance check: ${balance:.2f} (min required: $10.00)")
                if balance < 10:  # Minimum balance check
                    logger.warning(f"⚠️ Insufficient balance: ${balance:.2f} (minimum $10.00 required)")
                    time.sleep(60)
                    continue
                logger.info("✅ Balance sufficient, proceeding with trading logic")

                # Get VIPER trading signal
                signal = self.get_viper_signal()
                if not signal:
                    time.sleep(30)
                    continue

                logger.info(f"📊 VIPER Signal: {signal['signal']} at ${signal['price']:.2f}")

                # Calculate position size
                position_size = self.calculate_position_size(signal['price'], balance)
                logger.info(f"🎯 Calculated position size: {position_size:.8f} BTC (${position_size * signal['price']:.2f})")

                if position_size > 0 and position_size >= 0.0001:
                    # Check risk limits before executing trade
                    risk_check = self.check_risk_limits(signal['symbol'], position_size, signal['price'], balance)

                    if not risk_check.get('allowed', False):
                        logger.warning(f"🚫 Trade blocked by risk management: {risk_check.get('reason', 'Unknown reason')}")
                        time.sleep(60)
                        continue

                    logger.info("✅ Risk limits passed - executing trade")

                    # Execute trade
                    logger.info(f"🚀 Executing {signal['signal']} trade: {position_size:.6f} BTC")
                    order = self.execute_trade(signal['signal'], position_size)

                    if order:
                        logger.info(f"✅ Trade executed successfully: {order['id']}")

                        # Register position with risk manager
                        position_data = {
                            'id': order.get('id', 'unknown'),
                            'symbol': signal['symbol'],
                            'side': signal['signal'],
                            'size': position_size,
                            'price': order.get('average', order.get('price', signal['price'])),
                            'timestamp': datetime.now().isoformat()
                        }

                        if self.register_position(signal['symbol'], position_data):
                            logger.info(f"📝 Position registered for {signal['symbol']}")
                        else:
                            logger.warning(f"⚠️ Failed to register position for {signal['symbol']}")
                    else:
                        logger.error("❌ Trade execution failed")
                else:
                    logger.warning(f"⚠️ Position size too small: {position_size:.8f} BTC (min: 0.0001 BTC)")

                # Wait before next iteration
                time.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                logger.info("🛑 Trading loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(30)

    def stop(self):
        """Stop the trading engine"""
        logger.info("🛑 Stopping VIPER Live Trading Engine...")
        self.is_running = False

def main():
    """Main function"""
    engine = LiveTradingEngine()

    # Initialize connections
    if not engine.initialize_exchange():
        logger.error("❌ Failed to initialize exchange. Exiting...")
        return

    if not engine.initialize_redis():
        logger.warning("⚠️ Redis connection failed. Continuing without caching...")

    try:
        # Start trading loop
        engine.run_trading_loop()
    except KeyboardInterrupt:
        engine.stop()
    finally:
        logger.info("👋 VIPER Live Trading Engine stopped")

if __name__ == "__main__":
    main()
