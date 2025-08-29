#!/usr/bin/env python3
"""
🎯 VIPER TRADING BOT - MASTER LAUNCHER
The easiest way to use VIPER trading bot!

This script provides a simple menu to:
- Setup the system
- Run validation 
- Test with demo mode
- Start live trading
- Get help
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for better output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_banner():
    """Print the main banner"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 VIPER TRADING BOT - MASTER LAUNCHER{Colors.END}")
    print(f"{Colors.BOLD}The Easiest Way to Trade Cryptocurrency!{Colors.END}")
    print("="*60)

def print_menu():
    """Print the main menu"""
    print(f"\n{Colors.BOLD}What would you like to do?{Colors.END}")
    print(f"\n{Colors.GREEN}1. 🔧 Setup System{Colors.END} - Install and configure everything")
    print(f"{Colors.BLUE}2. 🔍 Validate System{Colors.END} - Check if everything is ready")
    print(f"{Colors.YELLOW}3. 🎮 Demo Mode{Colors.END} - Test safely with simulated data")
    print(f"{Colors.BOLD}4. 💰 Start Live Trading{Colors.END} - Trade with real money!")
    print(f"{Colors.BLUE}5. 📚 Help & Documentation{Colors.END} - Get help and guides")
    print(f"{Colors.RED}6. ❌ Exit{Colors.END}")

def run_script(script_name, description):
    """Run a Python script"""
    if not Path(script_name).exists():
        print(f"{Colors.RED}❌ {script_name} not found!{Colors.END}")
        return False
    
    print(f"\n{Colors.BLUE}🚀 Starting {description}...{Colors.END}")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, script_name], check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}{description} cancelled by user{Colors.END}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ Error running {description}: {e}{Colors.END}")
        return False

def show_help():
    """Show help and documentation"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}📚 VIPER TRADING BOT - HELP{Colors.END}")
    print("-" * 30)
    
    print(f"\n{Colors.GREEN}📋 Quick Start Guide:{Colors.END}")
    print("1. Run Setup System (option 1)")
    print("2. Edit .env file with your Bitget API keys")
    print("3. Run Validate System (option 2)")
    print("4. Start Live Trading (option 4)")
    
    print(f"\n{Colors.YELLOW}🎮 Want to test first?{Colors.END}")
    print("- Use Demo Mode (option 3) to see how it works safely")
    print("- No real money or API keys needed for demo")
    
    print(f"\n{Colors.BLUE}📖 Documentation:{Colors.END}")
    print("- EASY_SETUP.md - Simple setup guide") 
    print("- README.md - Full documentation")
    print("- docs/ - Detailed technical docs")
    
    print(f"\n{Colors.RED}⚠️  Important Notes:{Colors.END}")
    print("- Live trading uses REAL MONEY")
    print("- Always test with small amounts first")
    print("- Understand the risks before trading")
    print("- Keep your API keys secure and private")
    
    print(f"\n{Colors.YELLOW}🆘 Getting Help:{Colors.END}")
    print("- Run validate_system.py to check for issues")
    print("- Check logs/ directory for error messages")
    print("- Review .env file for configuration problems")

def get_system_status():
    """Get a quick system status"""
    status = []
    
    # Check if setup was run
    if Path(".env").exists():
        status.append(f"{Colors.GREEN}✅ Configuration file exists{Colors.END}")
    else:
        status.append(f"{Colors.RED}❌ Configuration missing - run setup{Colors.END}")
    
    # Check Docker
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, check=True)
        status.append(f"{Colors.GREEN}✅ Docker is running{Colors.END}")
    except:
        status.append(f"{Colors.RED}❌ Docker not running{Colors.END}")
    
    # Check API keys
    if Path(".env").exists():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('BITGET_API_KEY', '')
            if api_key and not api_key.startswith('your_'):
                status.append(f"{Colors.GREEN}✅ API keys configured{Colors.END}")
            else:
                status.append(f"{Colors.YELLOW}⚠️  API keys need real values{Colors.END}")
        except:
            status.append(f"{Colors.YELLOW}⚠️  Cannot check API keys{Colors.END}")
    
    return status

def main():
    """Main launcher function"""
    print_banner()
    
    # Show system status
    print(f"\n{Colors.BOLD}📊 System Status:{Colors.END}")
    status = get_system_status()
    for item in status:
        print(f"  {item}")
    
    while True:
        print_menu()
        
        try:
            choice = input(f"\n{Colors.BOLD}Enter your choice (1-6): {Colors.END}").strip()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Goodbye!{Colors.END}")
            break
        
        if choice == "1":
            # Setup system
            success = run_script("setup.py", "System Setup")
            if success:
                print(f"\n{Colors.GREEN}✅ Setup completed! Now edit .env with your API keys.{Colors.END}")
            input(f"\n{Colors.BLUE}Press Enter to continue...{Colors.END}")
            
        elif choice == "2":
            # Validate system
            success = run_script("validate_system.py", "System Validation")
            input(f"\n{Colors.BLUE}Press Enter to continue...{Colors.END}")
            
        elif choice == "3":
            # Demo mode
            success = run_script("demo.py", "Demo Mode")
            input(f"\n{Colors.BLUE}Press Enter to continue...{Colors.END}")
            
        elif choice == "4":
            # Live trading
            print(f"\n{Colors.RED}{Colors.BOLD}⚠️  WARNING: LIVE TRADING MODE ⚠️{Colors.END}")
            print(f"{Colors.RED}This will trade with REAL MONEY!{Colors.END}")
            confirm = input(f"\n{Colors.YELLOW}Type 'YES' to continue: {Colors.END}")
            
            if confirm.upper() == "YES":
                success = run_script("start_trading.py", "Live Trading")
            else:
                print(f"{Colors.YELLOW}Live trading cancelled{Colors.END}")
            
            input(f"\n{Colors.BLUE}Press Enter to continue...{Colors.END}")
            
        elif choice == "5":
            # Help
            show_help()
            input(f"\n{Colors.BLUE}Press Enter to continue...{Colors.END}")
            
        elif choice == "6":
            # Exit
            print(f"\n{Colors.GREEN}Thanks for using VIPER Trading Bot! 🚀{Colors.END}")
            break
            
        else:
            print(f"\n{Colors.RED}Invalid choice. Please enter 1-6.{Colors.END}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        sys.exit(1)