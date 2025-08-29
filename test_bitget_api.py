#!/usr/bin/env python3
"""
Test Bitget API endpoints to debug the issues
"""

import os
import sys
import time
import json
import hmac
import hashlib
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

class BitgetAPITester:
    def __init__(self):
        self.api_key = os.getenv('BITGET_API_KEY')
        self.api_secret = os.getenv('BITGET_API_SECRET')
        self.api_password = os.getenv('BITGET_API_PASSWORD')
        self.base_url = 'https://api.bitget.com'

    def generate_signature(self, timestamp, method, request_path, body=''):
        message = timestamp + method.upper() + request_path + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    def send_request(self, method, endpoint, params=None, body=None):
        timestamp = str(int(time.time() * 1000))
        request_path = endpoint

        if params:
            request_path += '?' + '&'.join([f"{k}={v}" for k, v in params.items()])

        signature = self.generate_signature(timestamp, method, request_path, body or '')

        headers = {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.api_password,
            'Content-Type': 'application/json'
        }

        url = self.base_url + endpoint
        print(f"Testing: {method} {url}")
        if params:
            print(f"Params: {params}")
        if body:
            print(f"Body: {body}")

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=body, timeout=10)
            else:
                print(f"❌ Unsupported method: {method}")
                return None

            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result}")
                return result
            else:
                print(f"❌ Error: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Exception: {e}")
            return None

def main():
    print("🔧 Testing Bitget API endpoints...")
    tester = BitgetAPITester()

    # Test 1: Time endpoint (no auth required)
    print("\n1. Testing public time endpoint:")
    result = tester.send_request('GET', '/api/v2/public/time')
    if result:
        print("✅ Public endpoint works")
    else:
        print("❌ Public endpoint failed")

    # Test 2: Account info
    print("\n2. Testing account info:")
    result = tester.send_request('GET', '/api/v2/mix/account/accounts', {'productType': 'USDT-FUTURES'})
    if result:
        print("✅ Account endpoint works")
    else:
        print("❌ Account endpoint failed")

    # Test 3: Contracts
    print("\n3. Testing contracts endpoint:")
    result = tester.send_request('GET', '/api/v2/mix/market/contracts', {'productType': 'USDT-FUTURES'})
    if result and result.get('code') == '00000':
        data = result.get('data', [])
        print(f"✅ Found {len(data)} contracts")
        if data:
            print(f"Sample: {data[0]}")
    else:
        print("❌ Contracts endpoint failed")

    # Test 4: Ticker
    print("\n4. Testing ticker endpoint:")
    result = tester.send_request('GET', '/api/v2/mix/market/ticker', {'symbol': 'BTCUSDT'})
    if result:
        print("✅ Ticker endpoint works")
    else:
        print("❌ Ticker endpoint failed")

    # Test 5: Tickers (plural)
    print("\n5. Testing tickers endpoint:")
    result = tester.send_request('GET', '/api/v2/mix/market/tickers', {'productType': 'USDT-FUTURES'})
    if result and result.get('code') == '00000':
        data = result.get('data', [])
        print(f"✅ Found {len(data)} tickers")
        if data:
            print(f"Sample: {data[0]}")
    else:
        print("❌ Tickers endpoint failed")

if __name__ == "__main__":
    main()
