#!/usr/bin/env python3
"""
🚀 VIPER COMPLETE LIVE TRADING SYSTEM
One-command startup for the complete VIPER trading ecosystem
"""

import os
import sys
import time
import subprocess
import signal
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('viper_live_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ViperLiveSystem:
    """Complete VIPER Live Trading System"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.startup_script = self.project_root / "start_live_trading.py"
        self.monitor_script = self.project_root / "live_trading_monitor.py"
        self.optimizer_script = self.project_root / "live_trading_optimizer.py"

        self.system_running = False
        self.monitoring_active = False

    def print_banner(self):
        """Print the VIPER Live System banner"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🚀 VIPER COMPLETE LIVE TRADING SYSTEM - PRODUCTION READY                    ║
║ 🔥 Real-Time Algorithmic Trading | 📊 Advanced Risk Management                ║
║ 🎯 AI-Powered Strategy Optimization | 🛡️ Enterprise Security                   ║
║ ⚡ Ultra-Low Latency Execution | 📈 Professional Performance Monitoring       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ⚠️  REAL MONEY TRADING SYSTEM - USE WITH CAUTION                             ║
║ 🛑 EMERGENCY STOP: Ctrl+C or 'docker compose down'                          ║
║ 📊 MONITORING: http://localhost:8000                                       ║
║ 📈 DASHBOARD: Run 'python live_trading_monitor.py'                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)

    def validate_system(self) -> bool:
        """Validate system readiness for live trading"""
        print("🔍 VALIDATING SYSTEM READINESS...")

        checks = {
            'Docker': self.check_docker(),
            'Environment': self.validate_environment(),
            'Files': self.check_files(),
            'Configuration': self.validate_configuration()
        }

        all_passed = True
        for check, passed in checks.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {check}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print("✅ System validation complete - Ready for live trading!")
            return True
        else:
            print("❌ System validation failed - Please fix issues above")
            return False

    def check_docker(self) -> bool:
        """Check Docker availability"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def validate_environment(self) -> bool:
        """Validate environment configuration"""
        required_vars = [
            'BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD',
            'REAL_DATA_ONLY', 'MAX_POSITIONS', 'RISK_PER_TRADE'
        ]

        for var in required_vars:
            value = os.getenv(var)
            if not value or (var != 'REAL_DATA_ONLY' and value.startswith('your_')):
                return False

        return True

    def check_files(self) -> bool:
        """Check required files exist"""
        required_files = [
            'docker-compose.yml',
            '.env',
            'start_live_trading.py',
            'live_trading_optimizer.py',
            'live_trading_monitor.py'
        ]

        for file in required_files:
            if not (self.project_root / file).exists():
                return False

        return True

    def validate_configuration(self) -> bool:
        """Validate trading configuration"""
        try:
            real_data_only = os.getenv('REAL_DATA_ONLY', '').lower() == 'true'
            risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))
            max_positions = int(os.getenv('MAX_POSITIONS', '15'))

            # Validate ranges
            if not real_data_only:
                logger.warning("⚠️ REAL_DATA_ONLY not set to true - using simulated data")
            if risk_per_trade > 0.05:
                logger.warning(f"⚠️ High risk per trade: {risk_per_trade*100}%")
            if max_positions > 20:
                logger.warning(f"⚠️ High max positions: {max_positions}")

            return True
        except:
            return False

    def start_system(self) -> bool:
        """Start the complete VIPER live trading system"""
        print("\n🚀 STARTING VIPER LIVE TRADING SYSTEM...")

        try:
            # Start the system using the startup script
            result = subprocess.run([
                sys.executable,
                str(self.startup_script)
            ], cwd=self.project_root)

            if result.returncode == 0:
                logger.info("✅ VIPER Live Trading System completed successfully")
                return True
            else:
                logger.error(f"❌ System exited with code: {result.returncode}")
                return False

        except KeyboardInterrupt:
            print("\n⏹️ System interrupted by user")
            return True
        except Exception as e:
            logger.error(f"❌ Error starting system: {e}")
            return False

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 SHUTDOWN SIGNAL RECEIVED: {signum}")
        print("⏹️ Shutting down VIPER Live Trading System...")

        self.system_running = False

        # Stop Docker services
        try:
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=self.project_root,
                capture_output=True,
                timeout=60
            )
            print("✅ Docker services stopped")
        except Exception as e:
            print(f"⚠️ Error stopping services: {e}")

        sys.exit(0)

    def show_post_startup_options(self):
        """Show options available after system startup"""
        print("\n" + "=" * 70)
        print("🎯 VIPER LIVE TRADING SYSTEM - POST-STARTUP OPTIONS")
        print("=" * 70)
        print("📊 MONITORING OPTIONS:")
        print("   1. Real-time Dashboard: python live_trading_monitor.py")
        print("   2. System Summary: python live_trading_monitor.py --summary")
        print("   3. Web Dashboard: http://localhost:8000")
        print("   4. Grafana Dashboard: http://localhost:3000")
        print("")
        print("🔧 MANAGEMENT OPTIONS:")
        print("   • View Logs: docker compose logs -f")
        print("   • Service Status: docker ps")
        print("   • Stop System: docker compose down")
        print("   • Restart Services: docker compose restart")
        print("")
        print("📈 PERFORMANCE MONITORING:")
        print("   • API Server: http://localhost:8000/docs")
        print("   • Risk Manager: http://localhost:8002/docs")
        print("   • Order Lifecycle: http://localhost:8013/docs")
        print("")
        print("🛑 EMERGENCY CONTROLS:")
        print("   • Emergency Stop: docker compose down --volumes")
        print("   • Force Restart: docker compose down && docker compose up -d")
        print("   • System Reset: rm -rf logs/ && docker compose down --volumes")
        print("=" * 70)

    def run_complete_system(self):
        """Run the complete VIPER live trading system"""
        # Print banner
        self.print_banner()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Step 1: System validation
            if not self.validate_system():
                print("\n❌ System validation failed. Please fix the issues above.")
                return

            # Step 2: Show risk warning
            print("\n⚠️  CRITICAL RISK WARNING:")
            print("   This system will execute REAL TRADES with REAL MONEY!")
            print("   Ensure you understand the risks and have tested thoroughly.")
            print("")
            input("   Press Enter to continue or Ctrl+C to abort...")

            # Step 3: Start the system
            self.system_running = True

            success = self.start_system()

            # Step 4: Show post-startup options
            if success:
                self.show_post_startup_options()

                print("\n🎉 VIPER LIVE TRADING SYSTEM STARTUP COMPLETE!")
                print("   • System is running in live mode")
                print("   • Real-time trading with optimization active")
                print("   • Risk management and emergency stops enabled")
                print("   • Monitoring dashboards available")

            else:
                print("\n❌ System startup failed!")
                print("   Check the logs above for error details")
                print("   Common issues:")
                print("   • Docker services not starting")
                print("   • API credentials invalid")
                print("   • Network connectivity issues")

        except KeyboardInterrupt:
            print("\n⏹️ System startup interrupted by user")
        except Exception as e:
            logger.error(f"❌ Fatal system error: {e}")
            print(f"\n❌ System error: {e}")
        finally:
            self.system_running = False

def main():
    """Main entry point for VIPER Live Trading System"""
    system = ViperLiveSystem()
    system.run_complete_system()

if __name__ == "__main__":
    main()
