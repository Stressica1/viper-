#!/usr/bin/env python3
"""
🎉 VIPER Trading Bot - Setup Completion Validator
Simple validation to confirm all setup requirements are met
"""

import os
import sys
from pathlib import Path

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment loading: SUCCESS")
except Exception as e:
    print(f"❌ Environment loading: FAILED - {e}")
    sys.exit(1)

def check_python_packages():
    """Check if all required packages are available"""
    required_packages = [
        'fastapi', 'uvicorn', 'redis', 'ccxt', 'pandas', 
        'numpy', 'requests', 'aiohttp', 'dotenv'
    ]
    
    print("\n📦 Python Dependencies Check:")
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            all_good = False
    
    return all_good

def check_environment_variables():
    """Check if all required environment variables are set"""
    required_env_vars = [
        'REDIS_URL', 'LOG_LEVEL', 'VAULT_MASTER_KEY', 'VAULT_ACCESS_TOKENS',
        'BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD',
        'RISK_PER_TRADE', 'MAX_LEVERAGE', 'DAILY_LOSS_LIMIT'
    ]
    
    print("\n🔧 Environment Variables Check:")
    all_good = True
    for var in required_env_vars:
        value = os.getenv(var)
        if value:
            # Hide sensitive values
            if any(sensitive in var.lower() for sensitive in ['key', 'secret', 'password', 'token']):
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"  ✅ {var} = {display_value}")
        else:
            print(f"  ❌ {var} - NOT SET")
            all_good = False
    
    return all_good

def check_configuration_files():
    """Check if all required configuration files exist"""
    required_files = [
        '.env', 'docker-compose.yml', 'requirements.txt', 
        'pyproject.toml', 'scripts/start_microservices.py'
    ]
    
    print("\n📁 Configuration Files Check:")
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_good = False
    
    return all_good

def check_docker_setup():
    """Check if Docker and Docker Compose are available"""
    import subprocess
    
    print("\n🐳 Docker Environment Check:")
    all_good = True
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Docker: {result.stdout.strip()}")
        else:
            print("  ❌ Docker - Not available")
            all_good = False
    except Exception:
        print("  ❌ Docker - Not available")
        all_good = False
    
    # Check Docker Compose
    try:
        result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Docker Compose: {result.stdout.strip()}")
        else:
            print("  ❌ Docker Compose - Not available")
            all_good = False
    except Exception:
        print("  ❌ Docker Compose - Not available")
        all_good = False
    
    return all_good

def main():
    """Main validation function"""
    print("🚀 VIPER Trading Bot - Setup Completion Validator")
    print("=" * 60)
    print("Checking if all setup requirements are satisfied...\n")
    
    # Run all checks
    checks = [
        ("Python Dependencies", check_python_packages),
        ("Environment Variables", check_environment_variables),
        ("Configuration Files", check_configuration_files),
        ("Docker Environment", check_docker_setup),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 SETUP COMPLETION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 VIPER TRADING BOT SETUP IS COMPLETE!")
        print("📋 All requirements satisfied - system is ready to run")
        print("🚀 Next steps:")
        print("   1. Configure real API keys (optional): python scripts/configure_api.py")  
        print("   2. Start services: python scripts/start_microservices.py start")
        print("   3. Access dashboard: http://localhost:8000")
        return True
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("📋 Some requirements are not satisfied")
        print("🔧 Please fix the failed checks above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)