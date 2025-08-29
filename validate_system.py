#!/usr/bin/env python3
"""
🔍 VIPER TRADING BOT - SYSTEM VALIDATOR
Comprehensive validation of the trading system setup

This script checks everything is properly configured and ready to trade.
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
    """Print validation header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}🔍 VIPER TRADING BOT - SYSTEM VALIDATOR{Colors.END}")
    print(f"{Colors.BOLD}Checking if everything is ready to trade{Colors.END}")
    print("="*60)

def print_section(title):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}📋 {title}{Colors.END}")
    print("-" * (len(title) + 4))

def check_passed(message):
    """Print passed check"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")
    return True

def check_failed(message, details=None):
    """Print failed check"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")
    if details:
        print(f"   {Colors.YELLOW}💡 {details}{Colors.END}")
    return False

def check_warning(message, details=None):
    """Print warning check"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")
    if details:
        print(f"   {details}")
    return True

def run_command(cmd, timeout=10):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_python_environment():
    """Check Python environment"""
    print_section("Python Environment")
    
    score = 0
    total = 0
    
    # Python version
    total += 1
    success, output, _ = run_command("python --version")
    if success:
        version = output.strip()
        if "3.7" in version or "3.8" in version or "3.9" in version or "3.10" in version or "3.11" in version or "3.12" in version:
            check_passed(f"Python version: {version}")
            score += 1
        else:
            check_warning(f"Python version: {version} (may not be fully compatible)")
    else:
        check_failed("Python not found", "Install Python 3.7+")
    
    # Essential packages
    packages = ['dotenv', 'requests', 'fastapi', 'redis', 'docker', 'pandas', 'numpy']
    for package in packages:
        total += 1
        try:
            __import__(package)
            check_passed(f"Package {package} available")
            score += 1
        except ImportError:
            check_failed(f"Package {package} missing", f"Run: pip install {package}")
    
    return score, total

def check_configuration():
    """Check configuration files"""
    print_section("Configuration")
    
    score = 0
    total = 0
    
    # .env file
    total += 1
    if Path(".env").exists():
        check_passed(".env file exists")
        score += 1
        
        # Load and check environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check API keys
            total += 3
            api_keys = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD']
            for key in api_keys:
                value = os.getenv(key, '')
                if value and not value.startswith('your_'):
                    check_passed(f"{key} configured")
                    score += 1
                elif value.startswith('your_'):
                    check_warning(f"{key} needs real value", "Edit .env with actual API credentials")
                else:
                    check_failed(f"{key} missing", "Add to .env file")
            
        except ImportError:
            check_failed("Cannot load .env file", "Install python-dotenv")
            
    else:
        check_failed(".env file missing", "Run: python setup.py")
    
    # Required directories
    dirs = ['logs', 'reports', 'data', 'config']
    for directory in dirs:
        total += 1
        if Path(directory).exists():
            check_passed(f"Directory {directory}/ exists")
            score += 1
        else:
            check_failed(f"Directory {directory}/ missing", "Run: python setup.py")
    
    return score, total

def check_docker():
    """Check Docker setup"""
    print_section("Docker Services")
    
    score = 0
    total = 0
    
    # Docker installation
    total += 1
    success, output, _ = run_command("docker --version")
    if success:
        check_passed(f"Docker: {output.strip()}")
        score += 1
    else:
        check_failed("Docker not installed", "Install Docker Desktop")
        return score, total
    
    # Docker daemon
    total += 1
    success, _, _ = run_command("docker ps")
    if success:
        check_passed("Docker daemon running")
        score += 1
    else:
        check_failed("Docker daemon not running", "Start Docker Desktop")
        return score, total
    
    # Docker Compose
    total += 1
    success, _, _ = run_command("docker compose version")
    if success:
        check_passed("Docker Compose available")
        score += 1
    else:
        check_failed("Docker Compose not available")
    
    # Check if docker-compose.yml exists
    total += 1
    if Path("docker-compose.yml").exists():
        check_passed("docker-compose.yml found")
        score += 1
    else:
        check_failed("docker-compose.yml missing")
    
    # Check running services
    total += 1
    success, output, _ = run_command("docker compose ps --services --filter status=running")
    if success:
        running_services = [s for s in output.strip().split('\n') if s]
        if running_services:
            check_passed(f"Docker services running: {len(running_services)}")
            score += 1
        else:
            check_warning("No Docker services running", "They will start automatically")
    else:
        check_warning("Could not check Docker services")
    
    return score, total

def check_system_health():
    """Check system health"""
    print_section("System Health")
    
    score = 0
    total = 0
    
    # Redis connection
    total += 1
    success, _, _ = run_command("docker compose up -d redis", timeout=30)
    if success:
        time.sleep(3)  # Wait for Redis to start
        success, _, _ = run_command("docker compose exec -T redis redis-cli ping", timeout=10)
        if success:
            check_passed("Redis service healthy")
            score += 1
        else:
            check_warning("Redis not responding", "Will start automatically when needed")
    else:
        check_warning("Could not start Redis", "May start automatically")
    
    # Check main scripts
    scripts = [
        ("setup.py", "Main setup script"),
        ("start_trading.py", "Trading launcher"),
        ("src/viper/core/main.py", "Core trading system")
    ]
    
    for script_path, description in scripts:
        total += 1
        if Path(script_path).exists():
            check_passed(f"{description} found")
            score += 1
        else:
            check_failed(f"{description} missing: {script_path}")
    
    return score, total

def main():
    """Main validation function"""
    print_header()
    
    total_score = 0
    total_checks = 0
    
    # Run all checks
    checks = [
        ("Python Environment", check_python_environment),
        ("Configuration", check_configuration),
        ("Docker Services", check_docker),
        ("System Health", check_system_health)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            score, total = check_func()
            total_score += score
            total_checks += total
            results.append((check_name, score, total))
        except Exception as e:
            print(f"{Colors.RED}Error in {check_name}: {e}{Colors.END}")
            results.append((check_name, 0, 1))
            total_checks += 1
    
    # Final report
    print(f"\n{'='*60}")
    print(f"{Colors.BOLD}VALIDATION RESULTS{Colors.END}")
    print("-" * 18)
    
    for check_name, score, total in results:
        percentage = (score / total * 100) if total > 0 else 0
        if percentage >= 80:
            status = f"{Colors.GREEN}✅ GOOD"
        elif percentage >= 60:
            status = f"{Colors.YELLOW}⚠️  WARNING"
        else:
            status = f"{Colors.RED}❌ FAILED"
        
        print(f"{check_name:20} {score:2}/{total:2} {status}{Colors.END}")
    
    overall_percentage = (total_score / total_checks * 100) if total_checks > 0 else 0
    
    print("-" * 30)
    print(f"{'OVERALL':20} {total_score:2}/{total_checks:2}", end=" ")
    
    if overall_percentage >= 80:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ READY TO TRADE!{Colors.END}")
        print(f"\n{Colors.GREEN}🎉 System validation passed!{Colors.END}")
        print(f"{Colors.YELLOW}Next steps:{Colors.END}")
        print(f"  1. Edit .env with real API keys")
        print(f"  2. Run: python start_trading.py")
    elif overall_percentage >= 60:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  MOSTLY READY{Colors.END}")
        print(f"\n{Colors.YELLOW}System mostly ready, but check warnings above{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ NOT READY{Colors.END}")
        print(f"\n{Colors.RED}System not ready. Please fix errors above.{Colors.END}")
        print(f"{Colors.YELLOW}Try running: python setup.py{Colors.END}")
    
    return 0 if overall_percentage >= 60 else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation cancelled by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Validation failed with error: {e}{Colors.END}")
        sys.exit(1)