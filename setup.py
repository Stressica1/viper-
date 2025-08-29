#!/usr/bin/env python3
"""
🚀 VIPER TRADING BOT - ONE-COMMAND SETUP
THE EASIEST SETUP EVER!

This script does EVERYTHING needed to get the VIPER trading bot ready to trade.
Just run: python setup.py
Then add your API keys to .env file and you're ready to trade!
"""

import os
import sys
import subprocess
import shutil
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

def print_step(step, description):
    """Print a setup step with nice formatting"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}🚀 Step {step}: {description}{Colors.END}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def run_command(cmd, description, check=True):
    """Run a command and return success status"""
    try:
        print(f"   Running: {description}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        if result.returncode == 0:
            print_success(f"{description} completed")
            return True, result.stdout
        else:
            print_error(f"{description} failed: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print_error(f"{description} failed: {str(e)}")
        return False, str(e)

def install_python_dependencies():
    """Install Python dependencies"""
    print_step(1, "Installing Python Dependencies")
    
    # Check Python version
    success, output = run_command("python --version", "Checking Python version", check=False)
    if success:
        print_success(f"Python found: {output.strip()}")
    else:
        print_error("Python not found - please install Python 3.7+")
        return False

    # Install pip if needed
    success, _ = run_command("pip --version", "Checking pip", check=False)
    if not success:
        print_error("pip not found - please install pip")
        return False

    # Install requirements
    print("   Installing required packages...")
    
    # Essential packages for the setup
    essential_packages = [
        "python-dotenv",
        "requests", 
        "fastapi",
        "uvicorn",
        "redis",
        "docker",
        "pydantic",
        "pandas",
        "numpy"
    ]
    
    for package in essential_packages:
        success, _ = run_command(f"pip install {package}", f"Installing {package}", check=False)
        if success:
            print(f"   ✓ {package}")
        else:
            print(f"   ! Failed to install {package} (will try later)")

    # Try to install from requirements.txt if it exists
    if Path("requirements.txt").exists():
        success, _ = run_command("pip install -r requirements.txt", "Installing from requirements.txt", check=False)
        if success:
            print_success("All dependencies installed from requirements.txt")
        else:
            print_warning("Some dependencies from requirements.txt failed - continuing anyway")

    return True

def setup_configuration():
    """Setup configuration files"""
    print_step(2, "Setting Up Configuration")
    
    # Create .env from example
    if not Path(".env").exists():
        if Path(".env.example").exists():
            shutil.copy(".env.example", ".env")
            print_success("Created .env file from template")
        else:
            # Create basic .env file
            env_content = """# 🚀 VIPER Trading Bot Configuration
# Add your real Bitget API credentials here

# CRITICAL: Replace with your REAL Bitget API credentials
BITGET_API_KEY=your_bitget_api_key_here
BITGET_API_SECRET=your_bitget_api_secret_here
BITGET_API_PASSWORD=your_bitget_api_password_here

# Trading Configuration
USE_MOCK_DATA=false
FORCE_LIVE_TRADING=true
RISK_PER_TRADE=0.02
MAX_POSITIONS=15
MAX_LEVERAGE=50

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379

# Service Ports
API_SERVER_PORT=8000
MCP_PORT=8015
"""
            with open(".env", "w") as f:
                f.write(env_content)
            print_success("Created basic .env file")
    else:
        print_success(".env file already exists")

    # Create necessary directories
    directories = ["logs", "reports", "data", "config", "backtest_results"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✓ Created {directory}/ directory")

    return True

def setup_docker():
    """Setup Docker services"""
    print_step(3, "Setting Up Docker Services")
    
    # Check if Docker is installed
    success, output = run_command("docker --version", "Checking Docker installation", check=False)
    if not success:
        print_error("Docker not found!")
        print(f"{Colors.YELLOW}Please install Docker Desktop from: https://www.docker.com/products/docker-desktop{Colors.END}")
        return False
    
    print_success(f"Docker found: {output.strip()}")
    
    # Check Docker Compose
    success, output = run_command("docker compose version", "Checking Docker Compose", check=False)
    if not success:
        print_warning("Docker Compose not found, trying docker-compose...")
        success, output = run_command("docker-compose --version", "Checking docker-compose", check=False)
        if not success:
            print_error("Docker Compose not available")
            return False
    
    print_success("Docker Compose available")
    
    # Start essential services
    print("   Starting essential Docker services...")
    success, _ = run_command("docker compose up -d redis", "Starting Redis", check=False)
    if success:
        print_success("Redis started")
    else:
        print_warning("Could not start Redis (you can start it later)")
    
    # Wait a bit for Redis to start
    time.sleep(3)
    
    return True

def validate_setup():
    """Validate the setup"""
    print_step(4, "Validating Setup")
    
    validation_passed = True
    
    # Check Python imports
    try:
        import dotenv
        print("   ✓ python-dotenv available")
    except ImportError:
        print("   ❌ python-dotenv not available")
        validation_passed = False
    
    try:
        import requests
        print("   ✓ requests available")
    except ImportError:
        print("   ❌ requests not available")
        validation_passed = False
    
    # Check configuration files
    if Path(".env").exists():
        print("   ✓ .env file exists")
    else:
        print("   ❌ .env file missing")
        validation_passed = False
    
    # Check Docker
    success, _ = run_command("docker ps", "Checking Docker daemon", check=False)
    if success:
        print("   ✓ Docker daemon running")
    else:
        print("   ⚠️  Docker daemon not running (start it before trading)")
    
    # Check Redis
    success, _ = run_command("docker compose ps redis", "Checking Redis", check=False)
    if success:
        print("   ✓ Redis service available")
    else:
        print("   ⚠️  Redis not running (will be started when needed)")
    
    if validation_passed:
        print_success("Setup validation passed!")
    else:
        print_warning("Setup validation found issues - check above")
    
    return validation_passed

def main():
    """Main setup function"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 VIPER TRADING BOT - ONE-COMMAND SETUP{Colors.END}")
    print(f"{Colors.BOLD}THE EASIEST SETUP EVER!{Colors.END}")
    print("="*60)
    
    print(f"\n{Colors.YELLOW}This will set up everything needed to run the VIPER trading bot.{Colors.END}")
    print(f"{Colors.YELLOW}After setup, you just need to add your API keys to .env and start trading!{Colors.END}")
    
    # Run setup steps
    steps = [
        ("Install Python Dependencies", install_python_dependencies),
        ("Setup Configuration", setup_configuration),
        ("Setup Docker Services", setup_docker),
        ("Validate Setup", validate_setup)
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        try:
            if step_func():
                success_count += 1
        except Exception as e:
            print_error(f"Error in {step_name}: {e}")
    
    # Final report
    print(f"\n{'='*60}")
    print(f"{Colors.BOLD}SETUP COMPLETE: {success_count}/{len(steps)} steps successful{Colors.END}")
    
    if success_count >= 3:  # Allow for some optional failures
        print_success("🎉 VIPER Trading Bot is ready!")
        print(f"\n{Colors.BOLD}{Colors.GREEN}NEXT STEPS:{Colors.END}")
        print(f"{Colors.YELLOW}1. Edit .env file with your REAL Bitget API credentials{Colors.END}")
        print(f"{Colors.YELLOW}2. Run: python start_trading.py{Colors.END}")
        print(f"{Colors.YELLOW}3. Start trading! 💰{Colors.END}")
        
        print(f"\n{Colors.BOLD}IMPORTANT:{Colors.END}")
        print(f"{Colors.RED}• This system trades with REAL MONEY{Colors.END}")
        print(f"{Colors.RED}• Make sure you understand the risks{Colors.END}")
        print(f"{Colors.RED}• Start with small amounts{Colors.END}")
        
    else:
        print_error("Setup failed - please check the errors above and try again")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup cancelled by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed with error: {e}")
        sys.exit(1)