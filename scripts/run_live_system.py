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

# Add project paths for new structure
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Enhanced terminal display
try:
    from src.viper.utils.terminal_display import (
        terminal, display_error, display_success, display_warning, 
        print_banner, print_status
    )
    ENHANCED_DISPLAY = True
except ImportError:
    ENHANCED_DISPLAY = False
    def display_error(msg, details=None): print(f"❌ {msg}")
    def display_success(msg, details=None): print(f"✅ {msg}")
    def display_warning(msg, details=None): print(f"⚠️ {msg}")
    def print_banner(): print("🚀 VIPER COMPLETE LIVE TRADING SYSTEM")
    def print_status(status): print("Status:", status)

# Load environment variables
load_dotenv()

# Configure logging with enhanced formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/viper_live_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ViperLiveSystem:
    """Complete VIPER Live Trading System"""

    def __init__(self):
        # Validate live trading mode first
        if os.getenv('USE_MOCK_DATA', '').lower() == 'true':
            logger.error("❌ Mock data mode not allowed in live system")
            sys.exit(1)
        
        # Enforce Docker and MCP requirements
        try:
            from docker_mcp_enforcer import enforce_docker_mcp_requirements
            
            logger.info("🔒 Enforcing Docker & MCP requirements...")
            if not enforce_docker_mcp_requirements():
                logger.error("❌ Docker/MCP requirements not met")
                sys.exit(1)
            logger.info("✅ Docker & MCP enforcement passed")
            
        except ImportError as e:
            logger.error(f"❌ Cannot import enforcement system: {e}")
            sys.exit(1)
        
        self.project_root = Path(__file__).parent
        self.startup_script = self.project_root / "start_live_trading_mandatory.py"
        self.monitor_script = self.project_root / "live_trading_monitor.py"
        self.optimizer_script = self.project_root / "live_trading_optimizer.py"

        self.system_running = False
        self.monitoring_active = False
        
        logger.info("✅ Live system initialized with mandatory enforcement")

    def print_banner(self):
        """Print the VIPER Live System banner"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🚀 VIPER LIVE TRADING SYSTEM - LIVE MODE ONLY - DOCKER & MCP ENFORCED       ║
║ 🔥 Real-Time Live Trading | 📊 Mandatory Risk Management                      ║
║ 🎯 MCP-Powered Automation | 🛡️ Docker Infrastructure Required                 ║
║ ⚡ Live Market Execution | 📈 Real-Time Performance Monitoring                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 🚨 LIVE MONEY TRADING SYSTEM - NO SIMULATION MODE                            ║
║ 🔒 DOCKER & MCP ENFORCEMENT ACTIVE                                           ║
║ 🛑 EMERGENCY STOP: Ctrl+C or 'docker compose down'                          ║
║ 📊 MONITORING: http://localhost:8000                                       ║
║ 📈 MCP SERVER: http://localhost:8015                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)

    def validate_system(self) -> bool:
        """Validate system readiness for live trading"""

        checks = {
            'Docker': self.check_docker(),
            'Environment': self.validate_environment(),
            'Files': self.check_files(),
            'Configuration': self.validate_configuration()
        }

        all_passed = True
        for check, passed in checks.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
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
        except Exception:
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
        except Exception:
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
            return True
        except Exception as e:
            logger.error(f"❌ Error starting system: {e}")
            return False

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
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
        except Exception as e:

        sys.exit(0)

    def show_post_startup_options(self):
        """Show options available after system startup"""
        print("🎯 VIPER LIVE TRADING SYSTEM - POST-STARTUP OPTIONS")
        print("   1. Real-time Dashboard: python live_trading_monitor.py")
        print("   2. System Summary: python live_trading_monitor.py --summary")
        print("   3. Web Dashboard: http://localhost:8000")
        print("   4. Grafana Dashboard: http://localhost:3000")
        print("   • Restart Services: docker compose restart")
        print("   • API Server: http://localhost:8000/docs")
        print("   • Risk Manager: http://localhost:8002/docs")
        print("   • Order Lifecycle: http://localhost:8013/docs")
        print("   • Emergency Stop: docker compose down --volumes")
        print("   • Force Restart: docker compose down && docker compose up -d")
        print("   • System Reset: rm -rf logs/ && docker compose down --volumes")

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
            print("   This system will execute REAL TRADES with REAL MONEY!")
            input("   Press Enter to continue or Ctrl+C to abort...")

            # Step 3: Start the system
            self.system_running = True

            success = self.start_system()

            # Step 4: Show post-startup options
            if success:
                self.show_post_startup_options()

                print("\n🎉 VIPER LIVE TRADING SYSTEM STARTUP COMPLETE!")
                print("   • Real-time trading with optimization active")
                print("   • Risk management and emergency stops enabled")

            else:
                print("   Check the logs above for error details")

        except KeyboardInterrupt:
        except Exception as e:
            logger.error(f"❌ Fatal system error: {e}")
        finally:
            self.system_running = False

def main():
    """Main entry point for VIPER Live Trading System"""
    system = ViperLiveSystem()
    system.run_complete_system()

if __name__ == "__main__":
    main()
