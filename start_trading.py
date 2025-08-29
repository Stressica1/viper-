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