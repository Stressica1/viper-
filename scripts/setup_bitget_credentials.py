#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Quick Bitget API Setup
Securely stores your Bitget API credentials for live trading
"""

import os
import json
import requests
from getpass import getpass

def setup_bitget_credentials():
    """Setup Bitget API credentials in the secure vault"""

    print("🔐 VIPER Bitget API Setup")
    print("=========================")
    print()
    print("You need to create API credentials at: https://www.bitget.com/en/account/newapi")
    print("Make sure to:")
    print("  ✅ Enable 'Read Info' permissions")
    print("  ✅ Enable 'Trade' permissions")
    print("  ✅ Set IP restrictions (recommended)")
    print()

    # Get API credentials from user
    api_key = input("Enter your Bitget API Key: ").strip()
    api_secret = getpass("Enter your Bitget API Secret: ").strip()
    api_password = getpass("Enter your Bitget API Password: ").strip()

    if not all([api_key, api_secret, api_password]):
        print("❌ All fields are required!")
        return False

    # Store in vault
    vault_url = "http://localhost:8008"
    token = "c0b25118b40ac34959df9ea2ffc04089eb8138932c1a44e8c920f01547b1e4a6"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Store each credential separately
    credentials = [
        ('service', 'bitget'),
        ('key', 'api_key'),
        ('value', api_key)
    ]

    try:
        # Store API Key
        response = requests.post(
            f"{vault_url}/credentials/store",
            data={'service': 'bitget', 'key': 'api_key', 'value': api_key},
            headers=headers
        )

        if response.status_code != 200:
            print(f"❌ Failed to store API key: {response.status_code}")
            return False

        # Store API Secret
        response = requests.post(
            f"{vault_url}/credentials/store",
            data={'service': 'bitget', 'key': 'api_secret', 'value': api_secret},
            headers=headers
        )

        if response.status_code != 200:
            print(f"❌ Failed to store API secret: {response.status_code}")
            return False

        # Store API Password
        response = requests.post(
            f"{vault_url}/credentials/store",
            data={'service': 'bitget', 'key': 'api_password', 'value': api_password},
            headers=headers
        )

        if response.status_code == 200:
            print("✅ Bitget API credentials stored successfully!")
            print("🔥 Your live trading system is now ready!")
            return True
        else:
            print(f"❌ Failed to store credentials: {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"❌ Error connecting to vault: {e}")
        return False

if __name__ == "__main__":
    success = setup_bitget_credentials()
    if success:
        print("\n🚀 Next steps:")
        print("1. Run: docker-compose -f docker-compose.live.yml up -d")
        print("2. Check: docker logs viper-trading-live-trading-engine-1")
        print("3. Monitor: http://localhost:8000")
    else:
        print("\n❌ Setup failed. Please try again.")
