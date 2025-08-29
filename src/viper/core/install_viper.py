#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Automated Setup Script

This script automates the complete installation and configuration of VIPER.
Just run: python setup.py

Features:
- ✅ Automatic dependency installation
- ✅ Environment configuration
- ✅ Docker setup and validation
- ✅ API key configuration wizard
- ✅ System validation and testing
- ✅ Step-by-step progress tracking
"""

import os
import sys
import subprocess
import platform
import json
import urllib.request
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import getpass

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
    UNDERLINE = '\033[4m'

class VIPERSetup:
    """VIPER Trading Bot automated setup manager"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.python_exe = sys.executable
        self.setup_steps = []
        self.errors = []
        
    def print_header(self):
        """Print welcome header"""
        print(f"""
{Colors.HEADER}{Colors.BOLD}
🚀 VIPER Trading Bot - Automated Setup
=====================================
Ultra High-Performance Algorithmic Trading Platform
{Colors.ENDC}

{Colors.OKBLUE}This script will automatically install and configure VIPER for you.
The process typically takes 3-5 minutes.{Colors.ENDC}

{Colors.WARNING}Prerequisites:
- Docker Desktop installed and running
- Python 3.11+ installed
- Stable internet connection{Colors.ENDC}
""")

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        print(f"\n{Colors.BOLD}📋 Checking Prerequisites...{Colors.ENDC}")
        
        checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 11):
            print(f"✅ Python {python_version.major}.{python_version.minor} - OK")
            checks.append(True)
        else:
            print(f"❌ Python {python_version.major}.{python_version.minor} - Need 3.11+")
            checks.append(False)
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Docker - {result.stdout.strip()}")
                checks.append(True)
            else:
                print("❌ Docker - Not found or not working")
                checks.append(False)
        except FileNotFoundError:
            print("❌ Docker - Not installed")
            checks.append(False)
        
        # Check Docker daemon
        try:
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker daemon - Running")
                checks.append(True)
            else:
                print("❌ Docker daemon - Not running (Start Docker Desktop)")
                checks.append(False)
        except:
            print("❌ Docker daemon - Not accessible")
            checks.append(False)
        
        # Check Git
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Git - {result.stdout.strip()}")
                checks.append(True)
            else:
                print("❌ Git - Not found")
                checks.append(False)
        except FileNotFoundError:
            print("❌ Git - Not installed")
            checks.append(False)
        
        # Check internet connection
        try:
            urllib.request.urlopen('https://pypi.org', timeout=10)
            print("✅ Internet connection - OK")
            checks.append(True)
        except:
            print("❌ Internet connection - Failed")
            checks.append(False)
        
        success = all(checks)
        if success:
            print(f"\n{Colors.OKGREEN}✅ All prerequisites met!{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}❌ Please fix the issues above before continuing.{Colors.ENDC}")
            
        return success
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        print(f"\n{Colors.BOLD}📦 Installing Python Dependencies...{Colors.ENDC}")
        
        try:
            # Upgrade pip first
            print("Upgrading pip...")
            result = subprocess.run([self.python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Failed to upgrade pip: {result.stderr}")
            
            # Install core dependencies
            print("Installing core dependencies...")
            result = subprocess.run([self.python_exe, '-m', 'pip', 'install', '-e', '.'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to install core dependencies: {result.stderr}")
                return False
            
            # Install MCP dependencies
            print("Installing MCP dependencies...")
            result = subprocess.run([self.python_exe, '-m', 'pip', 'install', '-e', '.[mcp]'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Failed to install MCP dependencies: {result.stderr}")
            
            print(f"{Colors.OKGREEN}✅ Dependencies installed successfully!{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Set up environment configuration"""
        print(f"\n{Colors.BOLD}⚙️ Setting Up Environment...{Colors.ENDC}")
        
        try:
            env_template = self.root_dir / '.env.template'
            env_file = self.root_dir / '.env'
            
            if not env_template.exists():
                print("❌ .env.template not found")
                return False
            
            if not env_file.exists():
                print("Copying .env.template to .env...")
                shutil.copy2(env_template, env_file)
                print("✅ Environment file created")
            else:
                print("✅ Environment file already exists")
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up environment: {e}")
            return False
    
    def setup_docker(self) -> bool:
        """Set up Docker environment"""
        print(f"\n{Colors.BOLD}🐳 Setting Up Docker Environment...{Colors.ENDC}")
        
        try:
            # Check if docker-compose.yml exists
            docker_compose = self.root_dir / 'infrastructure' / 'docker-compose.yml'
            if not docker_compose.exists():
                print("❌ Docker compose file not found")
                return False
            
            # Pull base images
            print("Pulling Docker images (this may take a few minutes)...")
            result = subprocess.run(['docker', 'compose', '-f', str(docker_compose), 'pull'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Some images may not have been pulled: {result.stderr}")
            
            print(f"{Colors.OKGREEN}✅ Docker environment ready!{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up Docker: {e}")
            return False
    
    def configure_api_keys(self) -> bool:
        """Configure API keys interactively"""
        print(f"\n{Colors.BOLD}🔐 API Key Configuration{Colors.ENDC}")
        
        print(f"""
{Colors.OKCYAN}To use VIPER for live trading, you need Bitget API credentials.
You can skip this step and configure later if you want to test with demo data first.{Colors.ENDC}

{Colors.WARNING}To get Bitget API keys:
1. Go to https://www.bitget.com/en/account/newapi
2. Create a new API key
3. Enable: Read Info, Spot Trading, Futures Trading
4. Set IP restrictions for security
5. Save your API key, secret, and password{Colors.ENDC}
""")
        
        configure = input(f"\n{Colors.BOLD}Do you want to configure API keys now? (y/N): {Colors.ENDC}").strip().lower()
        
        if configure in ['y', 'yes']:
            try:
                # Run the existing API configuration script
                script_path = self.root_dir / 'scripts' / 'configure_api.py'
                if script_path.exists():
                    result = subprocess.run([self.python_exe, str(script_path)], 
                                          cwd=self.root_dir)
                    if result.returncode == 0:
                        print(f"{Colors.OKGREEN}✅ API keys configured successfully!{Colors.ENDC}")
                        return True
                    else:
                        print(f"{Colors.FAIL}❌ API configuration failed{Colors.ENDC}")
                        return False
                else:
                    print(f"{Colors.FAIL}❌ API configuration script not found{Colors.ENDC}")
                    return False
            except Exception as e:
                print(f"❌ Error configuring API keys: {e}")
                return False
        else:
            print(f"{Colors.WARNING}⚠️ Skipping API configuration - demo mode only{Colors.ENDC}")
            return True
    
    def validate_installation(self) -> bool:
        """Validate the installation"""
        print(f"\n{Colors.BOLD}✅ Validating Installation...{Colors.ENDC}")
        
        try:
            # Test Python imports
            print("Testing Python imports...")
            test_imports = [
                'fastapi', 'uvicorn', 'redis', 'ccxt', 'pandas', 
                'numpy', 'requests', 'dotenv', 'aiohttp'
            ]
            
            for module in test_imports:
                try:
                    __import__(module)
                    print(f"✅ {module}")
                except ImportError:
                    print(f"❌ {module} - Not installed")
                    return False
            
            # Test environment loading
            print("Testing environment configuration...")
            try:
                from dotenv import load_dotenv
                load_dotenv()
                print("✅ Environment configuration loaded")
            except Exception as e:
                print(f"❌ Environment configuration failed: {e}")
                return False
            
            print(f"{Colors.OKGREEN}✅ Installation validation passed!{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print(f"""
{Colors.HEADER}{Colors.BOLD}
🎉 VIPER Setup Complete!
========================{Colors.ENDC}

{Colors.OKGREEN}Your VIPER Trading Bot is now installed and ready to use!{Colors.ENDC}

{Colors.BOLD}Next Steps:{Colors.ENDC}

{Colors.OKCYAN}1. Start the system:{Colors.ENDC}
   python scripts/start_microservices.py start

{Colors.OKCYAN}2. Open the dashboard:{Colors.ENDC}
   http://localhost:8000

{Colors.OKCYAN}3. Check system status:{Colors.ENDC}
   python scripts/start_microservices.py status

{Colors.OKCYAN}4. Run your first backtest:{Colors.ENDC}
   Use the web dashboard or run backtests directly

{Colors.OKCYAN}5. Configure live trading:{Colors.ENDC}
   Add your Bitget API keys when ready for live trading

{Colors.BOLD}Documentation:{Colors.ENDC}
- 📖 Complete Guide: INSTALLATION.md
- 👤 User Manual: docs/USER_GUIDE.md
- 🛠️ Technical Docs: docs/TECHNICAL_DOC.md
- 🤖 MCP Guide: docs/MCP_INTEGRATION_GUIDE.md

{Colors.BOLD}Troubleshooting:{Colors.ENDC}
- Check logs in the 'logs/' directory
- Run: python scripts/quick_validation.py
- View Docker logs: docker-compose -f infrastructure/docker-compose.yml logs

{Colors.WARNING}⚠️ Safety Reminder:
- Start with paper trading
- Test thoroughly before live trading  
- Never risk more than you can afford to lose
- Keep your API keys secure{Colors.ENDC}

{Colors.HEADER}Happy Trading with VIPER! 🚀{Colors.ENDC}
""")
    
    def run_setup(self):
        """Run the complete setup process"""
        self.print_header()
        
        # Track setup progress
        steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Dependencies", self.install_dependencies),
            ("Environment", self.setup_environment),
            ("Docker Setup", self.setup_docker),
            ("API Configuration", self.configure_api_keys),
            ("Validation", self.validate_installation)
        ]
        
        print(f"\n{Colors.BOLD}Setup Progress:{Colors.ENDC}")
        for i, (name, _) in enumerate(steps, 1):
            print(f"{i}. {name}")
        
        # Execute setup steps
        for i, (step_name, step_func) in enumerate(steps, 1):
            print(f"\n{Colors.BOLD}Step {i}/{len(steps)}: {step_name}{Colors.ENDC}")
            print("=" * 50)
            
            try:
                success = step_func()
                if not success:
                    print(f"\n{Colors.FAIL}❌ Setup failed at step: {step_name}{Colors.ENDC}")
                    print(f"{Colors.FAIL}Please check the error messages above and try again.{Colors.ENDC}")
                    return False
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Setup interrupted by user.{Colors.ENDC}")
                return False
            except Exception as e:
                print(f"\n{Colors.FAIL}❌ Unexpected error in {step_name}: {e}{Colors.ENDC}")
                return False
        
        # Success!
        self.print_next_steps()
        return True

def main():
    """Main setup function"""
    try:
        setup = VIPERSetup()
        success = setup.run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup interrupted by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Fatal error: {e}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()