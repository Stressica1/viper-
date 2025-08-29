#!/usr/bin/env python3
"""
Test specific working symbols on Bitget
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

class SymbolTester:
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

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=body, timeout=10)
            else:
                return None

            if response.status_code == 200:
                return response.json()
            else:
                return {'error': response.status_code, 'text': response.text}

        except Exception as e:
            return {'error': str(e)}

def main():
    print("🔧 Testing specific working symbols on Bitget...")
    tester = SymbolTester()

    # Test some common symbols that should work
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LTCUSDT']

    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")

        # Test ticker
        result = tester.send_request('GET', '/api/v2/mix/market/ticker', {'symbol': symbol})
        if result and 'error' not in result and result.get('code') == '00000':
            data = result.get('data', [{}])[0]
            price = data.get('lastPr', 'N/A')
            print(f"✅ Ticker: ${price}")
        else:
            print(f"❌ Ticker failed: {result}")

        # Test if we can get contract info
        result = tester.send_request('GET', '/api/v2/mix/market/contracts', {'productType': 'USDT-FUTURES'})
        if result and 'error' not in result and result.get('code') == '00000':
            contracts = result.get('data', [])
            matching_contracts = [c for c in contracts if c.get('symbol') == symbol]
            if matching_contracts:
                print(f"✅ Contract found: {matching_contracts[0]['symbol']}")
            else:
                print(f"❌ Contract not found in list")

        time.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    main()
