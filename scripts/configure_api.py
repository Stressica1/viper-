#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - API Configuration Wizard
Interactive setup for Bitget API credentials and system configuration

Usage:
    python configure_api.py
"""

import os
import json
import getpass
from pathlib import Path
from datetime import datetime
import subprocess
import sys

def print_header():
    """Print welcome header"""
    print("""
🚀 VIPER Trading Bot - API Configuration Wizard
===============================================
🔐 Secure API credential setup for live trading
    """)

def validate_api_key_format(api_key: str) -> bool:
    """Validate Bitget API key format"""
    if not api_key or api_key.startswith(('your_', 'YOUR_', 'test', 'demo')):
        return False

    # Bitget API keys typically start with 'bg_'
    if not api_key.startswith('bg_'):
        print("⚠️  Warning: API key doesn't start with 'bg_' - please verify this is correct")

    return len(api_key) > 10

def validate_api_secret_format(api_secret: str) -> bool:
    """Validate Bitget API secret format"""
    if not api_secret or api_secret.startswith(('your_', 'YOUR_', 'test', 'demo')):
        return False

    return len(api_secret) > 20

def validate_api_password_format(api_password: str) -> bool:
    """Validate Bitget API password format"""
    if not api_password or api_password.startswith(('your_', 'YOUR_', 'test', 'demo')):
        return False

    return len(api_password) > 5

def get_credentials_interactive():
    """Get credentials from user interactively"""
    print("\n🔑 Bitget API Configuration")
    print("-" * 30)
    print("⚠️  Please provide your Bitget API credentials:")
    print("   📍 Get them from: https://www.bitget.com/en/account/newapi")
    print("   🔒 Enable: Read Info, Spot Trading, Futures Trading")
    print()

    credentials = {}

    # API Key
    while True:
        print("1️⃣  API Key:")
        api_key = getpass.getpass("   Enter your Bitget API Key: ").strip()

        if validate_api_key_format(api_key):
            credentials['BITGET_API_KEY'] = api_key
            print("   ✅ API Key accepted")
            break
        else:
            print("   ❌ Invalid API Key format. Please try again.")
            print("   💡 API keys typically start with 'bg_' and are 30+ characters")

    # API Secret
    while True:
        print("\n2️⃣  API Secret:")
        api_secret = getpass.getpass("   Enter your Bitget API Secret: ").strip()

        if validate_api_secret_format(api_secret):
            credentials['BITGET_API_SECRET'] = api_secret
            print("   ✅ API Secret accepted")
            break
        else:
            print("   ❌ Invalid API Secret format. Please try again.")
            print("   💡 API secrets are typically 50+ characters")

    # API Password
    while True:
        print("\n3️⃣  API Password:")
        api_password = getpass.getpass("   Enter your Bitget API Password: ").strip()

        if validate_api_password_format(api_password):
            credentials['BITGET_API_PASSWORD'] = api_password
            print("   ✅ API Password accepted")
            break
        else:
            print("   ❌ Invalid API Password format. Please try again.")
            print("   💡 API passwords are typically 6+ characters")

    return credentials

def update_env_file(credentials):
    """Update the .env file with real credentials"""
    env_file = Path(".env")

    if not env_file.exists():
        print("❌ .env file not found!")
        return False

    print(f"\n📝 Updating {env_file}...")

    # Create backup
    backup_file = env_file.with_suffix('.backup')
    import shutil
    shutil.copy2(str(env_file), str(backup_file))
    print(f"📋 Created backup: {backup_file}")

    # Read current content
    with open(env_file, 'r') as f:
        content = f.read()

    # Replace placeholders
    content = content.replace(
        'BITGET_API_KEY=YOUR_ACTUAL_API_KEY_HERE',
        f'BITGET_API_KEY={credentials["BITGET_API_KEY"]}'
    )
    content = content.replace(
        'BITGET_API_SECRET=YOUR_ACTUAL_API_SECRET_HERE',
        f'BITGET_API_SECRET={credentials["BITGET_API_SECRET"]}'
    )
    content = content.replace(
        'BITGET_API_PASSWORD=YOUR_ACTUAL_API_PASSWORD_HERE',
        f'BITGET_API_PASSWORD={credentials["BITGET_API_PASSWORD"]}'
    )

    # Write updated content
    with open(env_file, 'w') as f:
        f.write(content)

    print("✅ .env file updated successfully!")
    return True

def update_docker_compose_files(credentials):
    """Update docker-compose files with credentials"""
    compose_files = ['docker-compose.yml', 'docker-compose.live.yml']

    for filename in compose_files:
        file_path = Path(filename)
        if not file_path.exists():
            continue

        print(f"📝 Updating {filename}...")

        with open(file_path, 'r') as f:
            content = f.read()

        # Replace environment variable placeholders
        content = content.replace('${BITGET_API_KEY}', credentials['BITGET_API_KEY'])
        content = content.replace('${BITGET_API_SECRET}', credentials['BITGET_API_SECRET'])
        content = content.replace('${BITGET_API_PASSWORD}', credentials['BITGET_API_PASSWORD'])

        with open(file_path, 'w') as f:
            f.write(content)

        print(f"✅ {filename} updated!")

def create_service_credentials(credentials):
    """Create service-specific credential files"""
    services_dir = Path("services")

    if not services_dir.exists():
        return

    print("\n📁 Creating service-specific configurations...")

    for service_dir in services_dir.iterdir():
        if service_dir.is_dir():
            cred_file = service_dir / "credentials.env"

            with open(cred_file, 'w') as f:
                f.write("# VIPER Service Credentials\n")
                f.write(f"BITGET_API_KEY={credentials['BITGET_API_KEY']}\n")
                f.write(f"BITGET_API_SECRET={credentials['BITGET_API_SECRET']}\n")
                f.write(f"BITGET_API_PASSWORD={credentials['BITGET_API_PASSWORD']}\n")

            print(f"✅ Created credentials for {service_dir.name}")

def save_encrypted_credentials(credentials):
    """Save encrypted credentials for future use"""
    import hashlib
    import platform

    # Simple encryption using system info
    salt = platform.node() + platform.machine()
    key = hashlib.sha256(salt.encode()).digest()

    data = json.dumps(credentials)
    encrypted = bytearray()

    for i, byte in enumerate(data.encode()):
        encrypted.append(byte ^ key[i % len(key)])

    cred_data = {
        'encrypted': encrypted.hex(),
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }

    with open('.credentials.json', 'w') as f:
        json.dump(cred_data, f, indent=2)

    print("🔒 Encrypted credentials saved to .credentials.json")

def test_docker_connection():
    """Test if Docker is running"""
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def main():
    """Main configuration wizard"""
    print_header()

    # Check if Docker is running
    if not test_docker_connection():
        print("❌ Docker is not running!")
        print("💡 Please start Docker Desktop first, then run this script again.")
        print("   🔗 Download: https://www.docker.com/products/docker-desktop")
        return

    print("✅ Docker is running!")

    # Check if .env file exists
    if not Path(".env").exists():
        print("❌ .env file not found!")
        print("💡 Please run: python start_microservices.py start")
        print("   This will create the necessary configuration files.")
        return

    # Get credentials from user
    credentials = get_credentials_interactive()

    print("\n🎉 Credentials collected successfully!")
    print("🚀 Applying credentials system-wide...")

    # Update files
    if not update_env_file(credentials):
        return

    update_docker_compose_files(credentials)
    create_service_credentials(credentials)
    save_encrypted_credentials(credentials)

    print("\n🎊 Configuration Complete!")
    print("=" * 40)
    print("✅ All API credentials have been configured system-wide")
    print("✅ Service-specific credential files created")
    print("✅ Encrypted backup saved for security")

    print("\n🚀 Ready to start live trading!")
    print("💡 Run: python start_microservices.py start")

    # Offer to start the system
    print("\n❓ Would you like to start the VIPER trading system now?")
    if input("   (y/N): ").lower().startswith('y'):
        print("🚀 Starting VIPER trading system...")
        os.system("python start_microservices.py start")
    else:
        print("💡 You can start the system later with:")
        print("   python start_microservices.py start")

if __name__ == "__main__":
    main()
