#!/usr/bin/env python3
"""
Store Bitget API credentials in the credential vault
"""

import requests
import json

# Your Bitget API credentials
api_key = "bg_d20a392139710bc38b8ab39e970114eb"
api_secret = "23ed4a7fe10b9c947d41a15223647f1b263f0d932b7d5e9e7bdfac01d3b84b36"
api_password = "22672267"

# Vault configuration
vault_url = "http://localhost:8008"
vault_token = "c0b25118b40ac34959df9ea2ffc04089eb8138932c1a44e8c920f01547b1e4a6"

headers = {
    'Authorization': f'Bearer {vault_token}',
    'Content-Type': 'application/json'
}

print("🔐 Storing Bitget API credentials in vault...")

# Store API Key
try:
    response = requests.post(
        f"{vault_url}/credentials/store",
        params={
            'service': 'bitget',
            'key': 'api_key',
            'value': api_key
        },
        headers=headers,
        timeout=10
    )

    if response.status_code == 200:
        print("✅ API Key stored successfully")
    else:
        print(f"❌ Failed to store API Key: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error storing API Key: {e}")

# Store API Secret
try:
    response = requests.post(
        f"{vault_url}/credentials/store",
        params={
            'service': 'bitget',
            'key': 'api_secret',
            'value': api_secret
        },
        headers=headers,
        timeout=10
    )

    if response.status_code == 200:
        print("✅ API Secret stored successfully")
    else:
        print(f"❌ Failed to store API Secret: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error storing API Secret: {e}")

# Store API Password
try:
    response = requests.post(
        f"{vault_url}/credentials/store",
        params={
            'service': 'bitget',
            'key': 'api_password',
            'value': api_password
        },
        headers=headers,
        timeout=10
    )

    if response.status_code == 200:
        print("✅ API Password stored successfully")
    else:
        print(f"❌ Failed to store API Password: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error storing API Password: {e}")

print("🎉 Credential storage complete!")
