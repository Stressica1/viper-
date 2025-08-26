#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - System Validation Script

This script validates that your VIPER installation is working correctly.
Run this after installation or if you're experiencing issues.

Usage: python scripts/quick_validation.py
"""

import os
import sys
import subprocess
import json
import urllib.request
import socket
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class VIPERValidator:
    """VIPER system validation and diagnostics"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.issues = []
        self.warnings = []
        self.info = []
        
    def print_header(self):
        """Print validation header"""
        print(f"""
{Colors.HEADER}{Colors.BOLD}
🔍 VIPER System Validation
==========================
Comprehensive system health check and diagnostics
{Colors.ENDC}

{Colors.OKBLUE}Validating your VIPER Trading Bot installation...{Colors.ENDC}
""")
    
    def check_python_environment(self) -> bool:
        """Check Python version and virtual environment"""
        print(f"{Colors.BOLD}🐍 Python Environment{Colors.ENDC}")
        
        # Check Python version
        version = sys.version_info
        if version >= (3, 11):
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        else:
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.11+")
            self.issues.append("Python version too old")
            return False
        
        # Check if in virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        if in_venv:
            print("✅ Virtual environment detected")
            self.info.append("Running in virtual environment")
        else:
            print("⚠️ Not in virtual environment (recommended)")
            self.warnings.append("Consider using virtual environment")
        
        return True
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        print(f"\n{Colors.BOLD}📦 Python Dependencies{Colors.ENDC}")
        
        required_packages = {
            'fastapi': 'Web framework',
            'uvicorn': 'ASGI server',
            'redis': 'Redis client',
            'ccxt': 'Exchange connector',
            'pandas': 'Data processing',
            'numpy': 'Numerical computing',
            'requests': 'HTTP client',
            'aiohttp': 'Async HTTP client',
            'dotenv': 'Environment loader'
        }
        
        optional_packages = {
            'docker': 'Docker integration',
            'cryptography': 'Encryption support',
            'websockets': 'WebSocket support',
            'click': 'CLI framework',
            'rich': 'Rich terminal output',
            'psutil': 'System monitoring'
        }
        
        missing_required = []
        missing_optional = []
        
        # Check required packages
        for package, description in required_packages.items():
            try:
                __import__(package)
                print(f"✅ {package} - {description}")
            except ImportError:
                print(f"❌ {package} - {description} (MISSING)")
                missing_required.append(package)
        
        # Check optional packages
        for package, description in optional_packages.items():
            try:
                __import__(package)
                print(f"✅ {package} - {description} (Optional)")
            except ImportError:
                print(f"⚠️ {package} - {description} (Optional - not installed)")
                missing_optional.append(package)
        
        if missing_required:
            self.issues.extend([f"Missing required package: {pkg}" for pkg in missing_required])
            return False
        
        if missing_optional:
            self.warnings.extend([f"Missing optional package: {pkg}" for pkg in missing_optional])
        
        return True
    
    def check_configuration_files(self) -> bool:
        """Check if configuration files exist and are valid"""
        print(f"\n{Colors.BOLD}⚙️ Configuration Files{Colors.ENDC}")
        
        config_files = [
            ('.env.template', 'Environment template', True),
            ('.env', 'Environment configuration', False),
            ('pyproject.toml', 'Project configuration', True),
            ('scripts/start_microservices.py', 'Service manager', True),
            ('scripts/configure_api.py', 'API configurator', True),
        ]
        
        all_good = True
        
        for file_path, description, required in config_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                print(f"✅ {file_path} - {description}")
            else:
                if required:
                    print(f"❌ {file_path} - {description} (MISSING)")
                    self.issues.append(f"Missing required file: {file_path}")
                    all_good = False
                else:
                    print(f"⚠️ {file_path} - {description} (Not configured)")
                    self.warnings.append(f"File not found: {file_path}")
        
        return all_good
    
    def check_environment_configuration(self) -> bool:
        """Check environment configuration"""
        print(f"\n{Colors.BOLD}🔧 Environment Configuration{Colors.ENDC}")
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Environment file loaded")
        except Exception as e:
            print(f"❌ Failed to load environment: {e}")
            self.issues.append("Environment loading failed")
            return False
        
        # Check key environment variables
        env_vars = {
            'BITGET_API_KEY': 'Bitget API Key',
            'BITGET_API_SECRET': 'Bitget API Secret', 
            'BITGET_API_PASSWORD': 'Bitget API Password',
            'REDIS_URL': 'Redis connection URL',
            'API_SERVER_PORT': 'API server port'
        }
        
        for var, description in env_vars.items():
            value = os.getenv(var)
            if value:
                if var.startswith('BITGET_') and value.startswith('your_'):
                    print(f"⚠️ {var} - {description} (Placeholder value)")
                    self.warnings.append(f"{var} is not configured with real credentials")
                else:
                    print(f"✅ {var} - {description}")
            else:
                print(f"⚠️ {var} - {description} (Not set)")
                self.warnings.append(f"{var} not set")
        
        return True
    
    def check_docker(self) -> bool:
        """Check Docker installation and status"""
        print(f"\n{Colors.BOLD}🐳 Docker Environment{Colors.ENDC}")
        
        # Check Docker installation
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Docker installed - {result.stdout.strip()}")
            else:
                print("❌ Docker not working properly")
                self.issues.append("Docker installation issue")
                return False
        except FileNotFoundError:
            print("❌ Docker not installed")
            self.issues.append("Docker not installed")
            return False
        
        # Check Docker daemon
        try:
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker daemon running")
            else:
                print("❌ Docker daemon not running")
                self.issues.append("Docker daemon not running")
                return False
        except:
            print("❌ Docker daemon not accessible")
            self.issues.append("Docker daemon not accessible")
            return False
        
        # Check Docker Compose
        try:
            result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Docker Compose - {result.stdout.strip()}")
            else:
                print("❌ Docker Compose not working")
                self.issues.append("Docker Compose issue")
                return False
        except FileNotFoundError:
            print("❌ Docker Compose not installed")
            self.issues.append("Docker Compose not installed")
            return False
        
        # Check compose file
        compose_file = self.root_dir / 'infrastructure' / 'docker-compose.yml'
        if compose_file.exists():
            print("✅ Docker compose configuration found")
        else:
            print("❌ Docker compose configuration missing")
            self.issues.append("Docker compose file missing")
            return False
        
        return True
    
    def check_network_connectivity(self) -> bool:
        """Check network connectivity and external services"""
        print(f"\n{Colors.BOLD}🌐 Network Connectivity{Colors.ENDC}")
        
        # Check internet connectivity
        try:
            urllib.request.urlopen('https://google.com', timeout=5)
            print("✅ Internet connectivity")
        except:
            print("❌ No internet connectivity")
            self.issues.append("No internet connection")
            return False
        
        # Check PyPI access (for package installation)
        try:
            urllib.request.urlopen('https://pypi.org', timeout=5)
            print("✅ PyPI accessible")
        except:
            print("⚠️ PyPI not accessible")
            self.warnings.append("PyPI not accessible - package installation may fail")
        
        # Check Bitget API accessibility (if configured)
        try:
            urllib.request.urlopen('https://api.bitget.com', timeout=5)
            print("✅ Bitget API accessible")
        except:
            print("⚠️ Bitget API not accessible")
            self.warnings.append("Bitget API not accessible - live trading may not work")
        
        return True
    
    def check_ports(self) -> bool:
        """Check if required ports are available"""
        print(f"\n{Colors.BOLD}🔌 Port Availability{Colors.ENDC}")
        
        required_ports = {
            8000: 'API Server',
            8001: 'Ultra Backtester',
            8002: 'Risk Manager',
            8003: 'Data Manager',
            8004: 'Strategy Optimizer',
            8005: 'Exchange Connector',
            8006: 'Monitoring Service',
            8007: 'Live Trading Engine',
            6379: 'Redis',
            9090: 'Prometheus'
        }
        
        all_available = True
        
        for port, service in required_ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print(f"⚠️ Port {port} ({service}) - Already in use")
                self.warnings.append(f"Port {port} already in use")
            else:
                print(f"✅ Port {port} ({service}) - Available")
        
        return all_available
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        print(f"\n{Colors.BOLD}💾 Disk Space{Colors.ENDC}")
        
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.root_dir)
            
            # Convert to GB
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            
            print(f"📊 Disk usage: {used_gb:.1f}GB used, {free_gb:.1f}GB free of {total_gb:.1f}GB total")
            
            if free_gb < 2:
                print(f"❌ Low disk space: {free_gb:.1f}GB free (need at least 2GB)")
                self.issues.append("Insufficient disk space")
                return False
            elif free_gb < 5:
                print(f"⚠️ Limited disk space: {free_gb:.1f}GB free (5GB+ recommended)")
                self.warnings.append("Limited disk space")
            else:
                print(f"✅ Sufficient disk space: {free_gb:.1f}GB free")
            
            return True
            
        except Exception as e:
            print(f"❌ Could not check disk space: {e}")
            self.warnings.append("Could not check disk space")
            return True  # Don't fail validation for this
    
    def test_api_connection(self) -> bool:
        """Test API connection if credentials are configured"""
        print(f"\n{Colors.BOLD}🔗 API Connection Test{Colors.ENDC}")
        
        # Load environment
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except:
            print("⚠️ Could not load environment for API test")
            return True
        
        # Check if real API keys are configured
        api_key = os.getenv('BITGET_API_KEY', '')
        api_secret = os.getenv('BITGET_API_SECRET', '')
        api_password = os.getenv('BITGET_API_PASSWORD', '')
        
        if not api_key or api_key.startswith('your_'):
            print("⚠️ No real API keys configured - skipping API test")
            self.info.append("API test skipped - no real credentials")
            return True
        
        try:
            import ccxt
            
            exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': api_secret,
                'password': api_password,
                'sandbox': True,  # Use sandbox for testing
                'timeout': 10000
            })
            
            # Test connection
            exchange.fetch_markets()
            print("✅ Bitget API connection successful (sandbox)")
            return True
            
        except ImportError:
            print("⚠️ ccxt not installed - skipping API test")
            self.warnings.append("ccxt not available for API testing")
            return True
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            self.issues.append(f"API connection failed: {e}")
            return False
    
    def print_summary(self):
        """Print validation summary"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}📊 Validation Summary{Colors.ENDC}")
        print("=" * 50)
        
        if not self.issues:
            print(f"{Colors.OKGREEN}✅ All critical checks passed!{Colors.ENDC}")
            print(f"{Colors.OKGREEN}Your VIPER system appears to be properly configured.{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}❌ {len(self.issues)} critical issue(s) found:{Colors.ENDC}")
            for issue in self.issues:
                print(f"   • {issue}")
        
        if self.warnings:
            print(f"\n{Colors.WARNING}⚠️ {len(self.warnings)} warning(s):{Colors.ENDC}")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.info:
            print(f"\n{Colors.OKCYAN}ℹ️ Additional information:{Colors.ENDC}")
            for info in self.info:
                print(f"   • {info}")
        
        # Recommendations
        print(f"\n{Colors.BOLD}💡 Recommendations:{Colors.ENDC}")
        
        if self.issues:
            print(f"{Colors.FAIL}1. Fix the critical issues listed above before using VIPER{Colors.ENDC}")
            print(f"{Colors.FAIL}2. Re-run this validation script after fixing issues{Colors.ENDC}")
        else:
            print(f"{Colors.OKGREEN}1. Your system is ready! Try starting VIPER:{Colors.ENDC}")
            print(f"   python scripts/start_microservices.py start")
            print(f"{Colors.OKGREEN}2. Access the dashboard at: http://localhost:8000{Colors.ENDC}")
        
        if 'API test skipped' in ' '.join(self.info):
            print(f"{Colors.OKCYAN}3. Configure API keys when ready for live trading:{Colors.ENDC}")
            print(f"   python scripts/configure_api.py")
        
        return len(self.issues) == 0
    
    def run_validation(self) -> bool:
        """Run complete system validation"""
        self.print_header()
        
        validation_steps = [
            ("Python Environment", self.check_python_environment),
            ("Dependencies", self.check_dependencies),
            ("Configuration Files", self.check_configuration_files),
            ("Environment Settings", self.check_environment_configuration),
            ("Docker", self.check_docker),
            ("Network Connectivity", self.check_network_connectivity),
            ("Port Availability", self.check_ports),
            ("Disk Space", self.check_disk_space),
            ("API Connection", self.test_api_connection)
        ]
        
        success = True
        
        for step_name, step_func in validation_steps:
            try:
                step_result = step_func()
                if not step_result:
                    success = False
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                self.issues.append(f"{step_name} check failed: {e}")
                success = False
        
        self.print_summary()
        return success

def main():
    """Main validation function"""
    try:
        validator = VIPERValidator()
        success = validator.run_validation()
        
        if success:
            print(f"\n{Colors.OKGREEN}🎉 Validation completed successfully!{Colors.ENDC}")
            sys.exit(0)
        else:
            print(f"\n{Colors.FAIL}❌ Validation found critical issues.{Colors.ENDC}")
            print(f"{Colors.FAIL}Please fix the issues above and run validation again.{Colors.ENDC}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Validation interrupted by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Fatal validation error: {e}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()