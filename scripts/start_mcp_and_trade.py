#!/usr/bin/env python3
"""
# Rocket VIPER Trading System - Start MCP Server and Execute Swap Trades
Automated startup of MCP server followed by comprehensive swap trading
"""

import os
import sys
import time
import subprocess
import requests
import threading
from pathlib import Path
from datetime import datetime

class MCPTradingOrchestrator:
    """Orchestrator for MCP server startup and automated trading"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.mcp_server_path = self.project_root / 'mcp-trading-server'
        self.env_file = self.project_root / '.env'
        self.mcp_process = None

        # Trading configuration
        self.mcp_server_url = "http://localhost:8015"
        self.max_retries = 10
        self.retry_delay = 5

    def print_header(self):
        """Print orchestrator header"""
#==============================================================================#
# # Rocket VIPER MCP TRADING ORCHESTRATOR                                          #
# 🔥 Automated MCP Server Startup | # Chart Swap Trading Execution                 #
# ⚡ All Pairs Trading | 🧠 AI Integration | 📈 50x Leverage                   #
#==============================================================================#
        """)

    def check_environment(self) -> bool:
        """Check if environment is properly configured"""

        if not self.env_file.exists():
            return False

        # Check critical environment variables
        required_vars = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD']
        missing_vars = []

        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            print(f"# Warning  Missing API credentials: {', '.join(missing_vars)}")
            print("   Trading will use demo mode (no real trades)")

        return True

    def start_mcp_server(self) -> bool:
        """Start the MCP trading server"""

        try:
            # Change to MCP server directory
            os.chdir(self.mcp_server_path)

            # Start MCP server in background
            self.mcp_process = subprocess.Popen(
                ['node', 'index.js'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to start
            print("⏳ Waiting for MCP server to initialize...")
            time.sleep(10)

            # Check if server is running
            if self.mcp_process.poll() is None:
                return self.wait_for_mcp_health()
            else:
                stdout, stderr = self.mcp_process.communicate()
                return False

        except Exception as e:
            return False

    def wait_for_mcp_health(self) -> bool:
        """Wait for MCP server to become healthy"""

        for attempt in range(self.max_retries):
            try:
                response = requests.get(f"{self.mcp_server_url}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        return False

    def start_swap_trading(self) -> None:
        """Start comprehensive swap trading for all pairs"""
        print("\n# Rocket STARTING COMPREHENSIVE SWAP TRADING...")
        print("🔥 Executing trades for all available pairs via MCP")
        print("⚡ Using 50x leverage with risk management")

        # Import and run the MCP swap trader
        try:
            sys.path.append(str(self.project_root / 'scripts'))
            from mcp_swap_trader import MCPSwapTrader

            trader = MCPSwapTrader(self.mcp_server_url)

            # Start trading in a separate thread to allow monitoring
            trading_thread = threading.Thread(target=trader.start_mcp_swap_trading)
            trading_thread.daemon = True
            trading_thread.start()

            print("# Chart Trading all available pairs with AI-powered signals")
            print("🛑 Press Ctrl+C to stop trading and close all positions")

            # Keep main thread alive for monitoring
            try:
                while True:
                    time.sleep(60)
                    print(f"# Chart Status: {len(trader.active_positions)} active positions, {trader.trades_executed} trades executed")
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping trading and closing all positions...")
                trader.stop()
                trading_thread.join(timeout=30)

        except ImportError as e:
        except Exception as e:

    def cleanup(self):
        """Clean up resources"""
        if self.mcp_process:
            try:
                self.mcp_process.terminate()
                self.mcp_process.wait(timeout=10)
            except Exception:
                self.mcp_process.kill()

    def run_orchestrator(self):
        """Run the complete MCP trading orchestrator"""
        self.print_header()

        if not self.check_environment():
            print("# X Environment check failed. Please configure your .env file.")
            return

        try:
            # Start MCP server
            if not self.start_mcp_server():
                print("# X Failed to start MCP server. Cannot proceed with trading.")
                return

            # Start swap trading
            self.start_swap_trading()

        except KeyboardInterrupt:
            print("\n\n👋 MCP Trading Orchestrator terminated by user")
        except Exception as e:
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    orchestrator = MCPTradingOrchestrator()
    orchestrator.run_orchestrator()

if __name__ == "__main__":
    main()
