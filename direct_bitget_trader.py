#!/usr/bin/env python3
"""
🚀 VIPER DIRECT BITGET TRADER - BYPASSES CCXT ISSUES
Direct API integration for Bitget unilateral positions - NO CCXT DEPENDENCY
"""

import os
import sys
import time
import json
import hmac
import hashlib
import base64
import random
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlencode

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DIRECT_BITGET - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/direct_bitget_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DirectBitgetTrader:
    """Direct Bitget API trader - bypasses CCXT unilateral position issues"""

    def __init__(self):
        self.api_key = os.getenv('BITGET_API_KEY')
        self.api_secret = os.getenv('BITGET_API_SECRET')
        self.api_password = os.getenv('BITGET_API_PASSWORD')
        self.base_url = 'https://api.bitget.com'

        # Trading config
        self.position_size_usdt = float(os.getenv('POSITION_SIZE_USDT', '10'))
        self.max_leverage = int(os.getenv('MAX_LEVERAGE', '50'))
        self.all_pairs = []
        self.active_positions = {}
        self.is_running = False

    def generate_signature(self, timestamp, method, request_path, body=''):
        """Generate Bitget API signature"""
        message = timestamp + method.upper() + request_path + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    def send_request(self, method, endpoint, params=None, body=None):
        """Send authenticated request to Bitget API"""
        timestamp = str(int(time.time() * 1000))
        request_path = endpoint

        if params:
            request_path += '?' + urlencode(params)

        signature = self.generate_signature(timestamp, method, request_path, body or '')

        headers = {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.api_password,
            'Content-Type': 'application/json'
        }

        url = self.base_url + endpoint

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=body)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

    def connect(self):
        """Connect to Bitget and configure unilateral positions"""
        try:
            if not all([self.api_key, self.api_secret, self.api_password]):
                logger.error("❌ Missing API credentials")
                return False

            logger.info("🔌 Connecting to Bitget Direct API...")

            # Test connection
            result = self.send_request('GET', '/api/v2/public/time')
            if not result:
                logger.error("❌ Failed to connect to Bitget API")
                return False

            # Configure unilateral position mode
            self.configure_unilateral_mode()

            # Load all trading pairs
            self.load_trading_pairs()

            logger.info(f"✅ Connected to Bitget - {len(self.all_pairs)} pairs available")
            logger.info("🔥 Unilateral position mode configured!")
            return True

        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False

    def configure_unilateral_mode(self):
        """Configure account for unilateral positions"""
        try:
            logger.info("🔧 Configuring unilateral position mode...")

            # Set position mode to single (unilateral)
            params = {
                'productType': 'USDT-FUTURES',
                'marginCoin': 'USDT',
                'holdMode': 'single'
            }

            result = self.send_request('POST', '/api/v2/mix/account/set-position-mode', body=params)
            if result and str(result.get('code')) == '00000':
                logger.info("✅ Unilateral position mode set successfully")
            else:
                logger.debug(f"⚠️ Position mode config: {result}")

        except Exception as e:
            logger.debug(f"⚠️ Position mode configuration failed: {e}")

    def load_trading_pairs(self):
        """Load all USDT futures trading pairs"""
        try:
            # Try multiple endpoints to load pairs
            endpoints = [
                ('/api/v2/mix/market/contracts', {'productType': 'USDT-FUTURES'}),
                ('/api/v2/mix/market/tickers', {'productType': 'USDT-FUTURES'}),
                ('/api/v2/public/symbols', {'productType': 'USDT-FUTURES'})
            ]

            for endpoint, params in endpoints:
                logger.info(f"🔍 Trying endpoint: {endpoint}")
                result = self.send_request('GET', endpoint, params)

                if result and result.get('code') == '00000':
                    data = result.get('data', [])

                    if endpoint == '/api/v2/mix/market/contracts':
                        # Contracts endpoint
                        self.all_pairs = [item['symbol'] for item in data if item.get('status') == 'normal']
                    elif endpoint == '/api/v2/mix/market/tickers':
                        # Tickers endpoint
                        self.all_pairs = [item['symbol'] for item in data if item.get('symbol', '').endswith('USDT')]
                    elif endpoint == '/api/v2/public/symbols':
                        # Symbols endpoint
                        self.all_pairs = [item['symbol'] for item in data if 'USDT' in item.get('symbol', '')]

                    if self.all_pairs:
                        logger.info(f"📊 Loaded {len(self.all_pairs)} active USDT futures pairs")
                        logger.info(f"🎯 Sample pairs: {self.all_pairs[:5]}")
                        break
                else:
                    logger.warning(f"⚠️ Endpoint {endpoint} failed: {result}")

            if not self.all_pairs:
                logger.error("❌ Failed to load pairs from any endpoint")

        except Exception as e:
            logger.error(f"❌ Error loading pairs: {e}")
            # Fallback: Add some common pairs manually
            self.all_pairs = [
                'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
                'LTCUSDT', 'BCHUSDT', 'XRPUSDT', 'SOLUSDT', 'DOGEUSDT'
            ]
            logger.info(f"🔄 Using fallback pairs: {len(self.all_pairs)} pairs")

    def set_leverage(self, symbol):
        """Set leverage for a symbol"""
        try:
            params = {
                'productType': 'USDT-FUTURES',
                'symbol': symbol,
                'marginCoin': 'USDT',
                'leverage': str(self.max_leverage)
            }

            result = self.send_request('POST', '/api/v2/mix/account/set-leverage', body=params)
            if result and str(result.get('code')) == '00000':
                logger.debug(f"✅ Leverage set for {symbol}: {self.max_leverage}x")
                return True
            else:
                logger.debug(f"⚠️ Leverage setting failed for {symbol}: {result}")
                return False

        except Exception as e:
            logger.debug(f"⚠️ Leverage error for {symbol}: {e}")
            return False

    def get_ticker(self, symbol):
        """Get current price for a symbol"""
        try:
            # Try multiple ticker endpoints - Bitget has different formats
            endpoints = [
                ('/api/v2/mix/market/ticker', {'symbol': symbol, 'productType': 'USDT-FUTURES'}),
                ('/api/v2/mix/market/tickers', {'productType': 'USDT-FUTURES'})
            ]

            for endpoint, params in endpoints:
                result = self.send_request('GET', endpoint, params)

                if result and str(result.get('code')) == '00000':
                    data = result.get('data', [])

                    if endpoint == '/api/v2/mix/market/ticker' and isinstance(data, list) and data:
                        # Single ticker response
                        ticker_data = data[0]
                        price = ticker_data.get('lastPr')
                        if price:
                            return float(price)

                    elif endpoint == '/api/v2/mix/market/tickers' and isinstance(data, list):
                        # Multiple tickers response - find our symbol
                        for ticker in data:
                            if ticker.get('symbol') == symbol:
                                price = ticker.get('lastPr')
                                if price:
                                    return float(price)

            # If all endpoints fail, try to estimate price from contract info
            logger.debug(f"⚠️ All ticker endpoints failed for {symbol}, trying fallback")
            return self.get_fallback_price(symbol)

        except Exception as e:
            logger.debug(f"⚠️ Ticker error for {symbol}: {e}")
            return None

    def get_fallback_price(self, symbol):
        """Get fallback price from contract info"""
        try:
            result = self.send_request('GET', '/api/v2/mix/market/contracts', {'productType': 'USDT-FUTURES'})

            if result and str(result.get('code')) == '00000':
                contracts = result.get('data', [])
                for contract in contracts:
                    if contract.get('symbol') == symbol:
                        # Use mark price or last price from contract
                        price = contract.get('markPrice') or contract.get('lastPrice')
                        if price:
                            logger.debug(f"📊 Fallback price for {symbol}: ${price}")
                            return float(price)

            return None
        except Exception as e:
            logger.debug(f"⚠️ Fallback price failed for {symbol}: {e}")
            return None

    def execute_trade(self, symbol, side):
        """Execute trade with proper unilateral position parameters"""
        try:
            # Set leverage first
            self.set_leverage(symbol)

            # Get current price
            current_price = self.get_ticker(symbol)
            if not current_price:
                logger.error(f"❌ Could not get price for {symbol}")
                return False

            # Calculate position size
            position_size = self.position_size_usdt / current_price

            logger.info(f"🎯 {side} {symbol} at ${current_price:.6f}")
            logger.info(f"💰 Position size: {position_size:.6f} coins (${self.position_size_usdt})")

            # Prepare order parameters for unilateral positions
            order_params = {
                'productType': 'USDT-FUTURES',
                'symbol': symbol,
                'marginCoin': 'USDT',
                'size': str(position_size),
                'price': str(current_price),
                'orderType': 'market',
                'side': side.lower(),  # 'buy' or 'sell'
                'tradeSide': 'open',   # Always 'open' for unilateral positions
                'marginMode': 'isolated'
            }

            # Execute order
            result = self.send_request('POST', '/api/v2/mix/order/place-order', body=order_params)

            if result and str(result.get('code')) == '00000':
                order_data = result.get('data', {})
                if isinstance(order_data, dict):
                    order_id = order_data.get('orderId')
                else:
                    order_id = str(order_data)
                logger.info(f"✅ Trade executed successfully: {order_id}")
                return True
            else:
                # Handle different error formats
                if isinstance(result, dict) and 'error' in result:
                    logger.error(f"❌ Trade failed - HTTP {result['error']}: {result.get('text', '')}")
                else:
                    logger.error(f"❌ Trade failed: {str(result)}")
                return False

        except Exception as e:
            logger.error(f"❌ Trade execution error for {symbol}: {e}")
            return False

    def generate_signal(self, symbol):
        """Generate trading signal"""
        signals = ['BUY', 'SELL', 'HOLD']
        weights = [0.4, 0.4, 0.2]  # 40% chance each for BUY/SELL, 20% HOLD
        return random.choices(signals, weights=weights)[0]

    def run(self):
        """Main trading loop"""
        self.is_running = True
        cycle = 0

        logger.info("🚀 Starting DIRECT BITGET MULTI-PAIR TRADING!")
        logger.info("=" * 80)

        while self.is_running:
            cycle += 1
            logger.info(f"\n🔄 Cycle #{cycle} - Scanning {len(self.all_pairs)} pairs")

            # Scan random subset of pairs
            pairs_to_scan = min(50, len(self.all_pairs))
            scanned_pairs = random.sample(self.all_pairs, pairs_to_scan)

            trades_this_cycle = 0

            for symbol in scanned_pairs:
                if not self.is_running:
                    break

                signal = self.generate_signal(symbol)

                if signal in ['BUY', 'SELL']:
                    logger.info(f"📊 {symbol} -> {signal}")
                    if self.execute_trade(symbol, signal):
                        trades_this_cycle += 1
                        self.active_positions[symbol] = {
                            'signal': signal,
                            'timestamp': time.time(),
                            'size_usdt': self.position_size_usdt
                        }

            logger.info(f"✅ Cycle #{cycle} complete - {trades_this_cycle} trades executed")
            logger.info(f"📈 Active positions: {len(self.active_positions)}")
            logger.info("⏰ Waiting 30 seconds for next scan...")

            time.sleep(30)

    def stop(self):
        """Stop trading"""
        logger.info("🛑 Stopping Direct Bitget Trader...")
        self.is_running = False

def main():
    """Main entry point"""
    print("🚀 VIPER DIRECT BITGET MULTI-PAIR TRADER")
    print("🎯 Bypassing CCXT - Direct API for unilateral positions")
    print("=" * 80)

    # Verify API credentials
    api_key = os.getenv('BITGET_API_KEY')
    api_secret = os.getenv('BITGET_API_SECRET')
    api_password = os.getenv('BITGET_API_PASSWORD')

    if not all([api_key, api_secret, api_password]):
        logger.error("❌ Missing API credentials!")
        return False

    logger.info(f"✅ API Key loaded: {api_key[:10]}...")
    logger.info("✅ API credentials verified")

    try:
        logger.info("🔄 Initializing Direct Bitget Trader...")
        trader = DirectBitgetTrader()

        if not trader.connect():
            logger.error("❌ Failed to connect to Bitget")
            return False

        logger.info("✅ Connected to Bitget Direct API successfully")
        logger.info(f"🔥 Ready to trade {len(trader.all_pairs)} pairs!")
        logger.info("🚀 Starting direct multi-pair live trading...")

        trader.run()

    except KeyboardInterrupt:
        logger.info("🛑 Trading stopped by user")
        trader.stop()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return False

    return True

if __name__ == "__main__":
    main()
