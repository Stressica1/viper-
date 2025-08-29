#!/usr/bin/env python3
"""Test script to check environment variable loading"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Environment Variables Check:")
print(f"MIN_VOLUME_THRESHOLD: {os.getenv('MIN_VOLUME_THRESHOLD')}")
print(f"MIN_LEVERAGE_REQUIRED: {os.getenv('MIN_LEVERAGE_REQUIRED')}")
print(f"MAX_SPREAD_THRESHOLD: {os.getenv('MAX_SPREAD_THRESHOLD')}")

# Check if .env file exists and show its path
print(f"\nCurrent working directory: {os.getcwd()}")
print(f".env file exists: {os.path.exists('.env')}")

if os.path.exists('.env'):
    print("Contents of .env file (first 10 lines):")
    with open('.env', 'r') as f:
        lines = f.readlines()[:10]
        for i, line in enumerate(lines, 1):
            print(f"{i}: {line.strip()}")
