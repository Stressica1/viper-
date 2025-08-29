#!/usr/bin/env python3
"""
# Rocket VIPER Live Trading System Launcher - LIVE MODE ONLY
Complete system startup with mandatory Docker & MCP enforcement
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTradingLauncher:
    """Complete live trading system launcher with mandatory enforcement"""

    def __init__(self):
        # Validate live trading mode first
        if os.getenv('USE_MOCK_DATA', '').lower() == 'true':
            logger.error("# X Mock data mode not allowed in live trading launcher")
            sys.exit(1)
        
        # Enforce live trading requirements
        from docker_mcp_enforcer import enforce_docker_mcp_requirements
        
        logger.info("🔒 Enforcing Docker & MCP requirements...")
        if not enforce_docker_mcp_requirements():
            logger.error("# X Docker/MCP requirements not met")
            sys.exit(1)
        
        self.project_root = Path(__file__).parent
        self.docker_compose_file = self.project_root / "docker-compose.yml"
        self.live_optimizer_script = self.project_root / "live_trading_optimizer.py"

        # Check if Docker is available
        self.docker_available = self.check_docker()

        # Check if all required files exist
        self.files_ready = self.check_required_files()
        
        logger.info("# Check Live trading launcher initialized with enforcement")

    def check_docker(self) -> bool:
        """Check if Docker is available and running"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"# Check Docker available: {result.stdout.strip()}")
                return True
            else:
                logger.error("# X Docker not available")
                return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("# X Docker not installed or not accessible")
            return False

    def check_required_files(self) -> bool:
        """Check if all required files exist"""
        required_files = [
            self.docker_compose_file,
            self.live_optimizer_script,
            self.project_root / ".env",
            self.project_root / "requirements.txt"
        ]

        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            logger.error(f"# X Missing required files: {missing_files}")
            return False

        logger.info("# Check All required files present")
        return True

    def validate_environment(self) -> bool:
        """Validate environment configuration"""
        logger.info("# Search Validating environment configuration...")

        # Check critical environment variables
        required_vars = [
            'BITGET_API_KEY',
            'BITGET_API_SECRET',
            'BITGET_API_PASSWORD',
            'REAL_DATA_ONLY',
            'MAX_POSITIONS',
            'RISK_PER_TRADE',
            'DAILY_LOSS_LIMIT'
        ]

        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if not value or (var != 'REAL_DATA_ONLY' and value.startswith('your_')):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"# X Missing or invalid environment variables: {missing_vars}")
            return False

        # Validate specific values
        real_data_only = os.getenv('REAL_DATA_ONLY', '').lower() == 'true'
        if not real_data_only:
            logger.warning("# Warning REAL_DATA_ONLY is not set to 'true' - system will use simulated data")

        risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))
        if risk_per_trade > 0.05:  # More than 5%
            logger.warning(f"# Warning High risk per trade: {risk_per_trade*100}%")

        max_positions = int(os.getenv('MAX_POSITIONS', '15'))
        if max_positions > 20:
            logger.warning(f"# Warning High maximum positions: {max_positions}")

        logger.info("# Check Environment configuration validated")
        return True

    def start_docker_services(self) -> bool:
        """Start all Docker services"""
        logger.info("🐳 Starting Docker services...")

        try:
            # Clean up any existing containers
            logger.info("🧹 Cleaning up existing containers...")
            subprocess.run(
                ["docker", "compose", "down", "--volumes", "--remove-orphans"],
                cwd=self.project_root,
                capture_output=True,
                timeout=60
            )

            # Start services
            logger.info("# Rocket Starting all VIPER services...")
            result = subprocess.run(
                ["docker", "compose", "up", "-d", "--build"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                logger.info("# Check Docker services started successfully")
                # Wait for services to be healthy
                logger.info("⏳ Waiting for services to be ready...")
                time.sleep(30)
                return True
            else:
                logger.error(f"# X Failed to start Docker services: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("# X Timeout starting Docker services")
            return False
        except Exception as e:
            logger.error(f"# X Error starting Docker services: {e}")
            return False

    def check_service_health(self) -> bool:
        """Check if all services are healthy"""
        logger.info("🏥 Checking service health...")

        services_to_check = [
            ("API Server", "http://localhost:8000/health"),
            ("Risk Manager", "http://localhost:8002/health"),
            ("Order Lifecycle Manager", "http://localhost:8013/health"),
            ("Exchange Connector", "http://localhost:8005/health"),
            ("Signal Processor", "http://localhost:8011/health"),
            ("Live Trading Engine", "http://localhost:8007/health")
        ]

        unhealthy_services = []

        for service_name, health_url in services_to_check:
            try:
                import requests
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"   # Check {service_name}: Healthy")
                else:
                    logger.error(f"   # X {service_name}: Status {response.status_code}")
                    unhealthy_services.append(service_name)
            except Exception as e:
                logger.error(f"   # X {service_name}: {e}")
                unhealthy_services.append(service_name)

        if unhealthy_services:
            logger.error(f"# X Unhealthy services: {unhealthy_services}")
            return False

        logger.info("# Check All services are healthy")
        return True

    def start_live_trading_optimizer(self):
        """Start the live trading optimizer"""
        logger.info("# Rocket Starting Live Trading Optimizer...")

        try:
            # Run the live trading optimizer
            result = subprocess.run([
                sys.executable,
                str(self.live_optimizer_script)
            ], cwd=self.project_root)

            if result.returncode == 0:
                logger.info("# Check Live trading optimizer completed successfully")
            else:
                logger.warning(f"# Warning Live trading optimizer exited with code: {result.returncode}")

        except KeyboardInterrupt:
            logger.info("⏹️ Live trading interrupted by user")
        except Exception as e:
            logger.error(f"# X Error running live trading optimizer: {e}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Shutdown signal received: {signum}")
        logger.info("⏹️ Shutting down VIPER Live Trading System...")

        # Stop Docker services
        try:
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=self.project_root,
                capture_output=True,
                timeout=60
            )
            logger.info("# Check Docker services stopped")
        except Exception as e:
            logger.error(f"# X Error stopping Docker services: {e}")

        sys.exit(0)

    def run_system_startup(self):
        """Run complete system startup sequence"""
        print("   • Monitor closely during initial operation")
        print("   • Emergency stop: Ctrl+C or 'docker compose down'")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Step 1: Validate prerequisites
            if not self.docker_available:
                print("# X Docker is not available. Please install Docker Desktop.")
                return

            if not self.files_ready:
                return

            if not self.validate_environment():
                print("   Please check your .env file and ensure all required variables are set.")
                return


            # Step 2: Start Docker services
            if not self.start_docker_services():
                return

            # Step 3: Check service health
            if not self.check_service_health():
                print("   Check Docker logs: docker compose logs")
                return

            # Step 4: Start live trading
            print("# Target System will begin live trading with optimization")
            print("# Chart Monitor performance at: http://localhost:8000")

            self.start_live_trading_optimizer()

        except KeyboardInterrupt:
        except Exception as e:
            logger.error(f"# X System startup error: {e}")
        finally:

            # Cleanup
            try:
                subprocess.run(
                    ["docker", "compose", "down"],
                    cwd=self.project_root,
                    capture_output=True,
                    timeout=60
                )
            except Exception as e:

            print("# Check VIPER Live Trading System shutdown complete")

def main():
    """Main entry point"""
    launcher = LiveTradingLauncher()
    launcher.run_system_startup()

if __name__ == "__main__":
    main()
