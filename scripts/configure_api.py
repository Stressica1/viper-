#!/usr/bin/env python3
"""
🔐 VIPER API Key Configuration Script
Interactive configuration of Bitget API credentials
"""

import os
import sys
import re
from pathlib import Path
from getpass import getpass

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Terminal colors for better user experience  
class Colors:
    BOLD = '\033[1m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    OKCYAN = '\033[96m'

def print_banner():
    """Print configuration banner"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}🔐 VIPER API Key Configuration{Colors.ENDC}")
    print("=" * 50)

def validate_api_key_format(api_key):
    """Basic validation of API key format"""
    if not api_key:
        return False, "API key cannot be empty"
    if len(api_key) < 10:
        return False, "API key appears too short"
    if api_key.startswith('your_'):
        return False, "Please enter your actual API key, not the placeholder"
    return True, "Valid format"

def validate_api_secret_format(api_secret):
    """Basic validation of API secret format"""
    if not api_secret:
        return False, "API secret cannot be empty"
    if len(api_secret) < 20:
        return False, "API secret appears too short"
    if api_secret.startswith('your_'):
        return False, "Please enter your actual API secret, not the placeholder"
    return True, "Valid format"

def validate_api_password_format(api_password):
    """Basic validation of API password format"""
    if not api_password:
        return False, "API password cannot be empty"
    if len(api_password) < 4:
        return False, "API password appears too short"
    if api_password.startswith('your_'):
        return False, "Please enter your actual API password, not the placeholder"
    return True, "Valid format"

def get_api_credentials():
    """Get API credentials from user input"""
    print(f"\n{Colors.WARNING}Get your Bitget API credentials from:")
    print(f"https://www.bitget.com/en/account/newapi{Colors.ENDC}")
    print(f"\n{Colors.WARNING}Required permissions:")
    print("• Read Info")
    print("• Spot Trading") 
    print("• Futures Trading")
    print("• Enable IP restrictions for security{Colors.ENDC}\n")
    
    # Get API Key
    while True:
        api_key = input(f"{Colors.BOLD}Enter your Bitget API Key: {Colors.ENDC}").strip()
        is_valid, message = validate_api_key_format(api_key)
        if is_valid:
            break
        print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")
    
    # Get API Secret  
    while True:
        api_secret = getpass(f"{Colors.BOLD}Enter your Bitget API Secret: {Colors.ENDC}")
        is_valid, message = validate_api_secret_format(api_secret)
        if is_valid:
            break
        print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")
    
    # Get API Password
    while True:
        api_password = getpass(f"{Colors.BOLD}Enter your Bitget API Password: {Colors.ENDC}")
        is_valid, message = validate_api_password_format(api_password)
        if is_valid:
            break
        print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")
    
    return api_key, api_secret, api_password

def update_env_file(api_key, api_secret, api_password):
    """Update .env file with API credentials"""
    env_path = project_root / '.env'
    
    # Read existing .env file if it exists
    env_content = ""
    if env_path.exists():
        with open(env_path, 'r') as f:
            env_content = f.read()
    
    # Update or add API credentials
    updated_content = env_content
    
    # Define the credentials to update
    credentials = {
        'BITGET_API_KEY': api_key,
        'BITGET_API_SECRET': api_secret, 
        'BITGET_API_PASSWORD': api_password
    }
    
    for key, value in credentials.items():
        # Check if key already exists
        pattern = rf'^{key}=.*$'
        if re.search(pattern, updated_content, re.MULTILINE):
            # Replace existing value
            updated_content = re.sub(pattern, f'{key}={value}', updated_content, flags=re.MULTILINE)
        else:
            # Add new key at the end
            if updated_content and not updated_content.endswith('\n'):
                updated_content += '\n'
            updated_content += f'{key}={value}\n'
    
    # Ensure essential trading configuration exists
    essential_config = {
        'LIVE_TRADING_ENABLED': 'true',
        'TEST_MODE': 'false',
        'LOG_LEVEL': 'INFO'
    }
    
    for key, default_value in essential_config.items():
        pattern = rf'^{key}=.*$'
        if not re.search(pattern, updated_content, re.MULTILINE):
            updated_content += f'{key}={default_value}\n'
    
    # Write updated content back to .env
    with open(env_path, 'w') as f:
        f.write(updated_content)
    
    print(f"{Colors.OKGREEN}✅ API credentials saved to .env file{Colors.ENDC}")

def test_api_configuration():
    """Test if API configuration is working"""
    try:
        # Import and test the validation function from start_trading.py
        sys.path.insert(0, str(project_root))
        
        # Load environment to test
        from dotenv import load_dotenv
        load_dotenv(project_root / '.env')
        
        # Test the same validation logic as start_trading.py
        required_keys = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD']
        missing_keys = []
        
        for key in required_keys:
            value = os.getenv(key, '')
            if not value or value.startswith('your_'):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"{Colors.FAIL}❌ Configuration test failed - missing keys: {', '.join(missing_keys)}{Colors.ENDC}")
            return False
        else:
            print(f"{Colors.OKGREEN}✅ API configuration test passed{Colors.ENDC}")
            return True
            
    except Exception as e:
        print(f"{Colors.FAIL}❌ Configuration test failed: {e}{Colors.ENDC}")
        return False

def main():
    """Main configuration function"""
    print_banner()
    
    try:
        # Get API credentials from user
        api_key, api_secret, api_password = get_api_credentials()
        
        # Update .env file
        update_env_file(api_key, api_secret, api_password)
        
        # Test configuration
        if test_api_configuration():
            print(f"\n{Colors.OKGREEN}🎉 API configuration completed successfully!{Colors.ENDC}")
            print(f"\n{Colors.BOLD}Next steps:{Colors.ENDC}")
            print(f"1. Run: {Colors.OKCYAN}python start_trading.py{Colors.ENDC}")
            print(f"2. Your trading bot is ready to go! 🚀")
            return True
        else:
            print(f"\n{Colors.FAIL}❌ Configuration failed validation{Colors.ENDC}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Configuration cancelled by user{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Configuration failed: {e}{Colors.ENDC}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)