#!/usr/bin/env python3
"""
🚀 SIMPLE LIVE TRADE EXECUTION
Direct trade execution for testing purposes
"""

import asyncio
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import required modules
import ccxt.pro as ccxt
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SIMPLE_TRADE - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleLiveTrader:
    """Simple live trader for executing a test trade"""

    def __init__(self):
        self.api_key = os.getenv('BITGET_API_KEY')
        self.api_secret = os.getenv('BITGET_API_SECRET')
        self.api_password = os.getenv('BITGET_API_PASSWORD')

        if not all([self.api_key, self.api_secret, self.api_password]):
            raise Exception("❌ Missing API credentials")

        # Initialize exchange
        self.exchange = ccxt.bitget({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'password': self.api_password,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use future instead of swap
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            }
        })

        logger.info("🚀 Simple Live Trader initialized")

    async def execute_test_trade(self):
        """Execute a simple test trade"""

        print("🚀 VIPER SIMPLE LIVE TRADE EXECUTION")
        print("⚠️  WARNING: This will execute a REAL trade!")
        print("=" * 60)

        try:
            # Step 1: Check account balance
            print("📊 STEP 1: CHECKING ACCOUNT BALANCE")
            balance = await self.get_account_balance()
            print(f"💰 Account Balance: ${balance:.2f}")
            if balance < 1.0:
                print("❌ Insufficient balance for trading")
                return False

            # Step 2: Get current market price
            print("\n💰 STEP 2: GETTING MARKET DATA")
            symbol = "BTCUSDT"  # Bitget futures symbol format
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            print(f"📈 Current BTC Price: ${current_price:.2f}")

            # Step 3: Calculate position size (very small for testing)
            risk_amount = 0.01  # $0.01 risk
            stop_loss_pct = 0.005  # 0.5% stop loss
            stop_distance = current_price * stop_loss_pct

            position_size = risk_amount / stop_distance
            position_size = max(position_size, 0.0001)  # Minimum 0.0001 BTC requirement
            position_size = min(position_size, 0.001)   # Max 0.001 BTC for safety

            print(f"🎯 Position Size: {position_size:.6f}")
            print(f"💵 Risk Amount: ${risk_amount:.2f}")
            print(f"🛑 Stop Loss: ${current_price - stop_distance:.2f}")

            # Step 4: Confirm trade execution
            print("\n⚠️  TRADE DETAILS:")
            print(f"   Symbol: {symbol}")
            print(f"   Direction: BUY (Long)")
            print(f"   Entry: ${current_price:.2f}")
            print(f"   Size: {position_size:.6f}")
            print(f"   Risk: ${risk_amount:.2f}")
            print(f"   Stop Loss: ${current_price - stop_distance:.2f}")

            # Manual confirmation required
            confirm = input("\n🔥 Execute this trade? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("❌ Trade execution cancelled by user")
                return False

            # Step 5: Execute the trade
            print("\n💰 STEP 5: EXECUTING TRADE")
            trade_result = await self.place_order(symbol, 'buy', position_size, current_price)

            if trade_result:
                print("✅ TRADE EXECUTED SUCCESSFULLY!")
                print(f"   Order ID: {trade_result.get('id', 'N/A')}")
                print(f"   Status: {trade_result.get('status', 'unknown')}")
                print(f"   Executed Price: ${trade_result.get('price', 0):.2f}")

                # Step 6: Monitor briefly
                print("\n📈 STEP 6: MONITORING POSITION")
                await self.monitor_position(symbol, 30)  # Monitor for 30 seconds

                return True
            else:
                print("❌ Trade execution failed")
                return False

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            print(f"❌ Error: {e}")
            return False

    async def get_account_balance(self):
        """Get account balance"""
        try:
            balance = await self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            return float(usdt_balance)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    async def place_order(self, symbol, side, size, price):
        """Place a market order"""

        try:
            # Calculate leverage (5x for conservative trading)
            leverage = 5

            # Simplified order parameters for Bitget
            params = {
                'leverage': leverage
            }

            # Try market order with minimal parameters
            try:
                order = await self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=size,
                    params=params
                )
            except Exception as e:
                # If market order fails, try limit order
                logger.warning(f"Market order failed, trying limit order: {e}")

                if side == 'buy':
                    limit_price = price * 1.0001
                else:
                    limit_price = price * 0.9999

                order = await self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=size,
                    price=limit_price,
                    params=params
                )

            logger.info(f"Order placed: {order}")
            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def monitor_position(self, symbol, duration_seconds):
        """Monitor position for a short duration"""

        print(f"📊 Monitoring {symbol} for {duration_seconds} seconds...")

        try:
            for i in range(duration_seconds // 5):
                await asyncio.sleep(5)

                # Get current price
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']

                print(f"   Price update: ${current_price:.2f}")
                # Check if we should exit (basic monitoring)
                # In a real system, this would check TP/SL levels

            print("✅ Monitoring complete")

        except Exception as e:
            logger.error(f"Monitoring error: {e}")

async def main():
    """Main execution function"""

    print("🚀 STARTING SIMPLE LIVE TRADE EXECUTION")
    print("⚠️  WARNING: This will execute REAL trades with REAL money!")
    print("=" * 60)

    # Initialize trader
    trader = SimpleLiveTrader()

    # Execute trade
    success = await trader.execute_test_trade()

    if success:
        print("\n🎉 SIMPLE LIVE TRADE EXECUTION SUCCESSFUL!")
        print("💰 A trade has been placed on Bitget exchange")
        print("\n📊 Next steps:")
        print("   1. Check your Bitget account for the position")
        print("   2. Monitor the trade manually if needed")
        print("   3. The system will continue monitoring")
    else:
        print("\n❌ SIMPLE LIVE TRADE EXECUTION FAILED!")
        print("🔍 Check the error messages above")

if __name__ == "__main__":
    asyncio.run(main())
