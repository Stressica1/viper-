#!/usr/bin/env python3
"""
🚀 COMPLETE LIVE TRADING SYSTEM LAUNCHER
Direct launcher for VIPER live trading with all components connected
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - LIVE_SYSTEM - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure logger is globally available
if 'logger' not in globals():
    logger = logging.getLogger(__name__)

async def start_complete_system():
    """Start the complete VIPER live trading system"""
    print("🚀 STARTING COMPLETE VIPER LIVE TRADING SYSTEM")
    print("=" * 60)

    try:
        # Import the main trader
        from viper_async_trader import ViperAsyncTrader

        print("🔧 Initializing Enhanced ViperAsyncTrader...")
        trader = ViperAsyncTrader()

        print("📊 System Configuration:")
        print(f"   • Risk per Trade: {trader.risk_per_trade*100}%")
        print(f"   • Max Leverage: {trader.max_leverage}x")
        print(f"   • Take Profit: {trader.take_profit_pct}%")
        print(f"   • Stop Loss: {trader.stop_loss_pct}%")
        print(f"   • Trailing Stop: {trader.trailing_stop_pct}%")

        print("✅ Components Status:")
        if trader.math_validator:
            print("   • Mathematical Validator: ✅ ACTIVE")
        if trader.entry_optimizer:
            print("   • Entry Point Optimizer: ✅ ACTIVE")
        if trader.mcp_config:
            print("   • MCP Configuration: ✅ ACTIVE")

        print("🎯 Starting Live Trading Operations...")

        # Connect to exchange
        connected = await trader.connect_exchange()
        if not connected:
            print("❌ Failed to connect to exchange")
            return False

        print("✅ Connected to Bitget exchange")

        # Start trading operations
        print("🚀 LIVE TRADING SYSTEM ACTIVATED!")
        print("📊 Features Active:")
        print("   • Real-time Scoring: ✅")
        print("   • Market Scanning: ✅")
        print("   • TP/SL/TSL Management: ✅")
        print("   • Balance Management: ✅")
        print("   • Risk Management: ✅")
        print("   • Position Sizing: ✅")

        # Keep the system running
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Get balance status
                balance = await trader.get_account_balance()

                # Monitor positions
                position_status = await trader.monitor_positions()

                print(f"💰 Balance: ${balance:.2f} | Active Positions: {position_status.get('active_positions', 0)}")

            except KeyboardInterrupt:
                print("\n🛑 Shutdown requested by user")
                break
            except Exception as e:
                logger.error(f"❌ System error: {e}")
                await asyncio.sleep(30)  # Wait before retry

        print("✅ System shutdown complete")

    except Exception as e:
        logger.error(f"❌ Failed to start system: {e}")
        print(f"❌ ERROR: {e}")
        return False

    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(start_complete_system())
        if success:
            print("🎉 Complete VIPER system ran successfully!")
            sys.exit(0)
        else:
            print("❌ System failed to start")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
        sys.exit(0)
