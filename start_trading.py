#!/usr/bin/env python3
"""
🚀 VIPER TRADING BOT - ONE-COMMAND START
START TRADING IN SECONDS!

This script starts the VIPER trading bot with all required services.
Just run: python start_trading.py

Prerequisites:
1. Run setup.py first
2. Add your API keys to .env file
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configuration from environment variables
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
THREAD_POOL_SIZE = int(os.getenv('THREAD_POOL_SIZE', '10'))
CONNECTION_POOL_SIZE = int(os.getenv('CONNECTION_POOL_SIZE', '20'))
REQUEST_TIMEOUT_SECONDS = int(os.getenv('REQUEST_TIMEOUT_SECONDS', '30'))
HEALTH_CHECK_TIMEOUT_SECONDS = int(os.getenv('HEALTH_CHECK_TIMEOUT_SECONDS', '10'))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
TEST_MODE = os.getenv('TEST_MODE', 'false').lower() == 'true'
COMPOSE_PROJECT_NAME = os.getenv('COMPOSE_PROJECT_NAME', 'viper-trading')
DOCKER_MODE = os.getenv('DOCKER_MODE', 'true').lower() == 'true'
PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', '9090'))
GRAFANA_PORT = int(os.getenv('GRAFANA_PORT', '3000'))

# Colors for better output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Print startup header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 VIPER TRADING BOT - STARTING SYSTEM{Colors.END}")
    print(f"{Colors.BOLD}READY TO MAKE MONEY! 💰{Colors.END}")
    print("="*60)

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_step(message):
    """Print step message"""
    print(f"{Colors.BLUE}🔄 {message}{Colors.END}")

def run_command(cmd, description, check=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def load_environment():
    """Load environment variables"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return True
    except ImportError:
        print_error("python-dotenv not installed. Run: pip install python-dotenv")
        return False

def validate_system_configuration():
    """Validate system configuration from environment variables"""
    print_step("Validating system configuration...")
    
    config_issues = []
    
    # Validate worker configuration
    if MAX_WORKERS < 1 or MAX_WORKERS > 32:
        config_issues.append(f"MAX_WORKERS should be between 1-32, got {MAX_WORKERS}")
    
    if THREAD_POOL_SIZE < 1:
        config_issues.append(f"THREAD_POOL_SIZE should be >= 1, got {THREAD_POOL_SIZE}")
    
    if CONNECTION_POOL_SIZE < 1:
        config_issues.append(f"CONNECTION_POOL_SIZE should be >= 1, got {CONNECTION_POOL_SIZE}")
    
    # Validate timeout configuration
    if REQUEST_TIMEOUT_SECONDS < 5:
        config_issues.append(f"REQUEST_TIMEOUT_SECONDS should be >= 5, got {REQUEST_TIMEOUT_SECONDS}")
    
    if HEALTH_CHECK_TIMEOUT_SECONDS < 1:
        config_issues.append(f"HEALTH_CHECK_TIMEOUT_SECONDS should be >= 1, got {HEALTH_CHECK_TIMEOUT_SECONDS}")
    
    # Check mode settings
    if TEST_MODE and not DEBUG_MODE:
        print_warning("TEST_MODE is enabled but DEBUG_MODE is not - this may hide important debugging info")
    
    # Check Docker mode
    if not DOCKER_MODE:
        print_warning("DOCKER_MODE is disabled - services may not start correctly")
    
    # Check external service ports
    external_ports = {
        'Prometheus': PROMETHEUS_PORT,
        'Grafana': GRAFANA_PORT
    }
    
    for service, port in external_ports.items():
        if port < 1024 or port > 65535:
            config_issues.append(f"{service} port should be between 1024-65535, got {port}")
    
    if config_issues:
        print_error("Configuration validation failed:")
        for issue in config_issues:
            print(f"  - {issue}")
        return False
    
    print_success("System configuration validated")
    print(f"  • Max Workers: {MAX_WORKERS}")
    print(f"  • Thread Pool Size: {THREAD_POOL_SIZE}")
    print(f"  • Connection Pool Size: {CONNECTION_POOL_SIZE}")
    print(f"  • Request Timeout: {REQUEST_TIMEOUT_SECONDS}s")
    print(f"  • Health Check Timeout: {HEALTH_CHECK_TIMEOUT_SECONDS}s")
    print(f"  • Debug Mode: {DEBUG_MODE}")
    print(f"  • Test Mode: {TEST_MODE}")
    print(f"  • Docker Mode: {DOCKER_MODE}")
    print(f"  • Prometheus Port: {PROMETHEUS_PORT}")
    print(f"  • Grafana Port: {GRAFANA_PORT}")
    return True

def validate_api_keys():
    """Validate API keys are configured"""
    print_step("Checking API configuration...")
    
    required_keys = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD']
    missing_keys = []
    
    for key in required_keys:
        value = os.getenv(key, '')
        if not value or value.startswith('your_'):
            missing_keys.append(key)
    
    if missing_keys:
        print_error("API keys not configured!")
        print(f"{Colors.YELLOW}Please edit .env file and add your REAL Bitget API credentials:{Colors.END}")
        for key in missing_keys:
            print(f"  {key}=your_real_{key.lower()}_here")
        print(f"\n{Colors.YELLOW}Get API keys from: https://www.bitget.com/en/account/newapi{Colors.END}")
        return False
    
    print_success("API keys configured")
    return True

def start_docker_services():
    """Start Docker services"""
    print_step("Starting Docker services...")
    
    # Check if Docker is running
    success, _, _ = run_command("docker ps", "Checking Docker", check=False)
    if not success:
        print_error("Docker is not running!")
        print(f"{Colors.YELLOW}Please start Docker Desktop and try again{Colors.END}")
        return False
    
    print_success("Docker is running")
    
    # Start essential services
    print("   Starting Redis...")
    success, _, _ = run_command("docker compose up -d redis", "Starting Redis", check=False)
    if success:
        print_success("Redis started")
    else:
        print_warning("Redis may already be running")
    
    # Wait for Redis to be ready
    print("   Waiting for Redis to be ready...")
    time.sleep(5)
    
    # Start other essential services
    essential_services = [
        "data-manager",
        "risk-manager", 
        "exchange-connector",
        "monitoring-service"
    ]
    
    print("   Starting core trading services...")
    for service in essential_services:
        print(f"     Starting {service}...")
        success, _, _ = run_command(f"docker compose up -d {service}", f"Starting {service}", check=False)
        if success:
            print(f"     ✓ {service}")
        else:
            print(f"     ! {service} (may already be running)")
    
    # Wait for services to be ready
    print("   Waiting for services to initialize...")
    time.sleep(10)
    
    return True

def start_trading_system():
    """Start the main trading system"""
    print_step("Starting VIPER Trading System...")
    
    # Try to find the main trading script
    possible_scripts = [
        "src/viper/core/main.py",
        "scripts/start_live_trading.py",
        "scripts/live_trading_manager.py",
        "run_live_trader.py"
    ]
    
    main_script = None
    for script in possible_scripts:
        if Path(script).exists():
            main_script = script
            break
    
    if not main_script:
        print_error("Could not find main trading script!")
        print(f"{Colors.YELLOW}Available options:{Colors.END}")
        for script in possible_scripts:
            if Path(script).exists():
                print(f"  python {script}")
        return False
    
    print_success(f"Found main script: {main_script}")
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}🚀 LAUNCHING LIVE TRADING SYSTEM...{Colors.END}")
    print(f"{Colors.RED}{Colors.BOLD}⚠️  WARNING: This will trade with REAL MONEY! ⚠️{Colors.END}")
    
    # Final confirmation
    print(f"\n{Colors.YELLOW}Press Enter to continue or Ctrl+C to cancel...{Colors.END}")
    try:
        input()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Trading cancelled by user{Colors.END}")
        return False
    
    # Start the trading system
    print(f"\n{Colors.GREEN}Starting trading system...{Colors.END}")
    
    try:
        # Run the main trading script
        result = subprocess.run([sys.executable, main_script], check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Trading stopped by user{Colors.END}")
        return True
    except Exception as e:
        print_error(f"Failed to start trading system: {e}")
        return False

def check_system_status():
    """Check system status"""
    print_step("Checking system status...")
    
    # Check Docker services
    success, output, _ = run_command("docker compose ps", "Checking services", check=False)
    if success:
        running_services = [line for line in output.split('\n') if 'Up' in line]
        print_success(f"Docker services running: {len(running_services)}")
    else:
        print_warning("Could not check Docker services")
    
    # Check Redis connection
    success, _, _ = run_command("docker exec -it $(docker compose ps -q redis) redis-cli ping", "Testing Redis", check=False)
    if success:
        print_success("Redis is responding")
    else:
        print_warning("Redis connection test failed")
    
    return True

def main():
    """Main startup function"""
    print_header()
    
    # Pre-flight checks
    print_step("Running pre-flight checks...")
    
    # Check if setup was run
    if not Path(".env").exists():
        print_error("Setup not completed!")
        print(f"{Colors.YELLOW}Please run: python setup.py{Colors.END}")
        return 1
    
    # Load environment
    if not load_environment():
        return 1
    
    # Validate system configuration
    if not validate_system_configuration():
        return 1
    
    # Validate API keys
    if not validate_api_keys():
        return 1
    
    # System status check
    if not check_system_status():
        print_warning("Some system checks failed - continuing anyway")
    
    # Start services
    if not start_docker_services():
        print_error("Failed to start Docker services")
        return 1
    
    # Start trading
    print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 ALL SYSTEMS READY!{Colors.END}")
    
    if not start_trading_system():
        print_error("Failed to start trading system")
        return 1
    
    print_success("Trading system started successfully!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Startup cancelled by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Startup failed with error: {e}")
        sys.exit(1)