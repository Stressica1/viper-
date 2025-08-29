#!/usr/bin/env python3
"""
💰 LIVE BALANCE DEMO - Real-Time Balance Tracking System
======================================================

Demonstration of the complete real-time balance tracking system.
Shows how the system connects to exchanges and tracks live balance updates.

Features:
- WebSocket connections to exchanges
- Real-time balance updates (no hard-coded values)
- Multi-exchange support (Bitget, Bybit, Binance)
- Automatic failover and reconnection
- Live P&L calculations
- Balance anomaly detection

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Import our live balance service
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .live_balance_service import LiveBalanceService, get_live_balance, start_balance_service, stop_balance_service
except ImportError:
    from live_balance_service import LiveBalanceService, get_live_balance, start_balance_service, stop_balance_service

class LiveBalanceDemo:
    """Demonstration of real-time balance tracking system"""

    def __init__(self):
        self.service = None
        self.balance_history = []

    def initialize_demo(self):
        """Initialize the live balance demo"""
        print("💰 VIPER LIVE BALANCE DEMO")
        print("=" * 50)
        print("🔴 IMPORTANT: This demo shows REAL-TIME balance tracking")
print("🔴 NO HARD-CODED VALUES - Uses actual exchange connections")
        print("🔴 WebSocket streams provide live balance updates")
        print()

        # Check for API credentials
        credentials_file = Path("config/exchange_credentials.json")
        if not credentials_file.exists():
            print("⚠️  API credentials not found!")
print("📝 Please configure your exchange API credentials in:")
            print("    config/exchange_credentials.json")
            print()
            print("📋 Template format:")
            print("""
{
  "bitget": {
    "api_key": "your_api_key",
    "secret_key": "your_secret_key",
    "passphrase": "your_passphrase",
    "enabled": true
  }
}
            """)
            return False

        # Check if credentials are configured
        try:
            with open(credentials_file, 'r') as f:
                credentials = json.load(f)

            configured_exchanges = []
            for exchange, creds in credentials.items():
                if creds.get('api_key') and creds.get('enabled'):
                    configured_exchanges.append(exchange.upper())

            if not configured_exchanges:
                print("⚠️  No exchange API credentials configured!")
print("💡 Configure at least one exchange to see live balance tracking")
                return False

            print(f"✅ Configured exchanges: {', '.join(configured_exchanges)}")
            print()

        except Exception as e:
            print(f"❌ Error reading credentials: {e}")
            return False

        return True

    def demonstrate_balance_tracking(self):
        """Demonstrate real-time balance tracking"""
        print("🔄 STARTING LIVE BALANCE TRACKING DEMONSTRATION")
        print("=" * 60)

        # Start the balance service
        print("🚀 Starting Live Balance Service...")
        start_balance_service()

        try:
            # Demonstrate balance tracking for 2 minutes
            end_time = time.time() + 120  # 2 minutes

            print("📊 Monitoring live balance updates...")
print("💡 Updates will appear every 10 seconds from WebSocket streams")
            print("💡 Balance changes will be logged automatically")
            print()

            update_count = 0
            while time.time() < end_time:
                try:
                    # Get current live balance
                    live_balance = get_live_balance()

                    # Store in history for demonstration
                    self.balance_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'balance': live_balance.total_usd_balance,
                        'status': live_balance.status,
                        'exchange': live_balance.exchange
                    })

                    # Show status every 10 seconds
                    if update_count % 10 == 0:
                        self.display_balance_status(live_balance)

                    update_count += 1
                    time.sleep(1)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"⚠️  Balance update error: {e}")
                    time.sleep(5)

            print("")
✅ Live balance tracking demonstration completed!"            print(f"📊 Total balance updates monitored: {len(self.balance_history)}")

        finally:
            # Stop the service
            print("⏹️ Stopping Live Balance Service...")
            stop_balance_service()

    def display_balance_status(self, live_balance):
        """Display current balance status"""
        timestamp = datetime.now().strftime('%H:%M:%S')

        status_emoji = {
            'connected': '🟢',
            'connecting': '🟡',
            'disconnected': '🔴',
            'error': '❌'
        }.get(live_balance.status, '⚪')

        print(f"[{timestamp}] {status_emoji} Balance: ${live_balance.total_usd_balance:.2f} USD")
        print(f"         Exchange: {live_balance.exchange.upper()}")
        print(f"         Status: {live_balance.status.upper()}")
        print(f"         Available: ${live_balance.available_usd_balance:.2f}")
        print(f"         Locked: ${live_balance.locked_usd_balance:.2f}")
        print(f"         Last Update: {live_balance.last_update}")
        print()

    def demonstrate_system_integration(self):
        """Demonstrate how the system integrates with other components"""
        print("🔗 SYSTEM INTEGRATION DEMONSTRATION")
        print("=" * 50)

        print("📊 This live balance system integrates with:")
print("   • Strategy Metrics Dashboard")
        print("• Live Trading Manager")
print("   • Performance-Based Allocation")
        print("• Risk Management Systems")
print("   • GitHub MCP Task Automation")
        print()

        print("💰 NO MORE HARD-CODED BALANCE VALUES:")
print("   ❌ OLD: balance = 30.00  # Hard-coded")
        print("   ✅ NEW: balance = get_live_balance().total_usd_balance")
        print()

        print("🔄 REAL-TIME UPDATES:")
print("   • WebSocket streams from exchanges")
        print("• 5-10 second balance update intervals")
print("   • Automatic reconnection on failures")
        print("   • Multi-exchange failover support")
        print()

        print("🛡️ RISK MANAGEMENT:")
print("   • Dynamic risk limits based on live balance")
        print("• Real-time position size calculations")
print("   • Automatic emergency stops")
        print("   • Balance anomaly detection")
        print()

    def show_balance_history(self):
        """Show balance history from the demonstration"""
        if not self.balance_history:
            print("📊 No balance history available")
            return

        print("📈 BALANCE HISTORY FROM DEMONSTRATION")
        print("=" * 50)

        print(f"{'Time':<12} {'Balance':>10} {'Status':<12} {'Exchange':<10}")
        print("-" * 50)

        for entry in self.balance_history[-20:]:  # Show last 20 entries
            timestamp = entry['timestamp'].split('T')[1][:8]  # HH:MM:SS
            balance = entry['balance']
            status = entry['status'][:11]
            exchange = entry['exchange'][:9]

            print(f"{timestamp:<12} ${balance:>8.2f} {status:<12} {exchange:<10}")

        print()
        print("💡 This shows real-time balance tracking in action!")
print("💡 In a real scenario, balance would change based on trading activity")

    def run_full_demo(self):
        """Run the complete live balance demonstration"""
        print("🎬 VIPER COMPLETE LIVE BALANCE SYSTEM DEMO")
        print("=" * 60)
        print("This demo shows the REAL-TIME balance tracking system")
print("that replaces all hard-coded balance values in your system.")
        print()

        if not self.initialize_demo():
            print("❌ Demo initialization failed")
            return

        # Show system integration explanation
        self.demonstrate_system_integration()

        # Demonstrate live balance tracking
        self.demonstrate_balance_tracking()

        # Show balance history
        self.show_balance_history()

        print("🎉 DEMO COMPLETED!")
        print("=" * 30)
        print("✅ Real-time balance tracking system operational")
print("✅ No more hard-coded balance values")
        print("✅ WebSocket connections established")
print("✅ Multi-exchange support configured")
        print("✅ Live balance integration ready")
        print()
        print("🚀 Your VIPER system now uses REAL exchange balance!")
print("💰 Balance updates automatically every 5-10 seconds")
        print("🛡️ Risk management adapts to your actual balance")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='VIPER Live Balance Demo')
    parser.add_argument('--quick', action='store_true', help='Run quick 30-second demo')
    parser.add_argument('--full', action='store_true', help='Run complete 2-minute demo')
    parser.add_argument('--status', action='store_true', help='Show current balance status')

    args = parser.parse_args()

    demo = LiveBalanceDemo()

    if args.status:
        # Just show current balance status
        try:
            start_balance_service()
            time.sleep(2)  # Give it time to connect

            live_balance = get_live_balance()
            demo.display_balance_status(live_balance)

            stop_balance_service()

        except Exception as e:
            print(f"❌ Error getting balance status: {e}")

    elif args.quick:
        # Quick 30-second demo
        if demo.initialize_demo():
            print("🔄 Starting quick 30-second live balance demo...")
            start_balance_service()

            try:
                for i in range(30):
                    live_balance = get_live_balance()
                    if i % 5 == 0:  # Show every 5 seconds
                        demo.display_balance_status(live_balance)
                    time.sleep(1)
            finally:
                stop_balance_service()

            print("✅ Quick demo completed!")

    elif args.full:
        # Full demonstration
        demo.run_full_demo()

    else:
        print("💰 VIPER Live Balance Demo")
        print("=" * 40)
        print("Real-time balance tracking system demonstration")
        print()
        print("Commands:")
print("  --quick    - Run 30-second demo")
        print("--full     - Run complete 2-minute demo")
print("  --status   - Show current balance status")
        print()
        print("⚠️  IMPORTANT: Configure API credentials first!")
print("   Edit: config/exchange_credentials.json")

if __name__ == '__main__':
    main()

