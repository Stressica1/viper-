#!/usr/bin/env python3
"""
# Target FINAL VIPER LIVE TRADING LAUNCHER
Complete system with all components connected for live trading
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - FINAL_SYSTEM - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)

async def run_complete_live_system():
    """Run the complete VIPER live trading system with mandatory enforcement"""
    print("🔒 MANDATORY DOCKER & MCP ENFORCEMENT ACTIVE")
    
    # Enforce Docker and MCP requirements
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
        # Import the enhanced trader
from .viper_async_trader import ViperAsyncTrader


        print("# Tool Initializing Complete Trading System...")
        trader = ViperAsyncTrader()

        print(f"   • Risk per Trade: {trader.risk_per_trade*100}%")
        print(f"   • Max Leverage: {trader.max_leverage}x")
        print(f"   • Take Profit: {trader.take_profit_pct}%")
        print(f"   • Trailing Stop: {trader.trailing_stop_pct}%")
        print(f"   • Max Positions: {trader.max_positions}")
        print(f"   • Max Concurrent Jobs: {trader.max_concurrent_jobs}")

        connected = await trader.connect_exchange()

        if not connected:
            return False


        # Get initial balance
        try:
            balance = await trader.get_account_balance()
            print(f"💰 Swap Wallet Balance: ${balance:.2f} USDT")
        except Exception as e:
            pass


        print("# Chart Monitoring market for trading opportunities...")

        # Start the main trading loop
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                print(f"\n🔄 Cycle #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")

                # Get current balance
                try:
                    balance = await trader.get_account_balance()
                except Exception as e:
                    pass

                # Monitor positions
                try:
                    position_status = await trader.monitor_positions()
                    active_positions = position_status.get('active_positions', 0)
                except Exception as e:
                    pass

                # Run system diagnostics occasionally
                if cycle_count % 10 == 0:  # Every 10 cycles
                    try:
                        diagnostic_results = await trader.run_system_diagnostic()
                        if diagnostic_results:
                            score = diagnostic_results.get('overall_score', 'N/A')
                    except Exception as e:
                        pass

                # Wait before next cycle (30 seconds)
                await asyncio.sleep(30)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"# X System error in cycle {cycle_count}: {e}")
                await asyncio.sleep(10)  # Shorter wait on error

        print("# Party Thank you for using VIPER Live Trading System!")

    except Exception as e:
        logger.error(f"# X Failed to start complete system: {e}")
        return False

    return True

if __name__ == "__main__":
    try:
        pass

        success = asyncio.run(run_complete_live_system())

        if success:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        sys.exit(1)
