#!/usr/bin/env python3
"""
# Rocket COMPLETE LIVE TRADING SYSTEM LAUNCHER
Direct launcher for VIPER live trading with all components connected
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - LIVE_SYSTEM - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)

# Ensure logger is globally available"""
if 'logger' not in globals():
    logger = logging.getLogger(__name__)

async def start_complete_system():
    """Start the complete VIPER live trading system with mandatory enforcement"""
    print("# Rocket STARTING COMPLETE VIPER LIVE TRADING SYSTEM")
    
    # Enforce Docker and MCP requirements first
    try:
        from docker_mcp_enforcer import enforce_docker_mcp_requirements
        
        if not enforce_docker_mcp_requirements():
            sys.exit(1)
        
    except ImportError as e:
        sys.exit(1)
    
    # Validate live trading environment
    from dotenv import load_dotenv
    load_dotenv()
    
    if os.getenv('USE_MOCK_DATA', '').lower() == 'true':
        sys.exit(1)

    try:
        # Import the main trader
        from src.viper.execution.viper_async_trader import ViperAsyncTrader

        print("# Tool Initializing Complete Trading System...")
        trader = ViperAsyncTrader()

        print(f"   • Risk per Trade: {trader.risk_per_trade*100}%")
        print(f"   • Max Leverage: {trader.max_leverage}x")
        print(f"   • Take Profit: {trader.take_profit_pct}%")
        print(f"   • Trailing Stop: {trader.trailing_stop_pct}%")

        if trader.math_validator:
        if trader.entry_optimizer:
            pass
        if trader.mcp_config:
            pass


        # Connect to exchange
        connected = await trader.connect_exchange()
        if not connected:
            return False


        # Start trading operations

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
                break
            except Exception as e:
                logger.error(f"# X System error: {e}")
                await asyncio.sleep(30)  # Wait before retry


    except Exception as e:
        logger.error(f"# X Failed to start system: {e}")
        return False

    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(start_complete_system())
        if success:
            print("# Party Complete VIPER system ran successfully!")
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
