#!/usr/bin/env python3
"""
VIPER Trading System - Store Credentials Securely
Interactive script to encrypt and store your API credentials in the secure vault
"""

import os
import sys
import json
import base64
import secrets
import hashlib
from typing import Dict, Optional
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Store the provided credentials securely"""
    logger.info("🔐 VIPER Secure Credential Storage")
    logger.info("=" * 50)

    # Use the provided credentials
    bitget_api_key = os.getenv('BITGET_API_KEY', '')
    bitget_api_secret = os.getenv('BITGET_API_SECRET', '')
    bitget_api_password = os.getenv('BITGET_API_PASSWORD', '')
    github_token = os.getenv('GITHUB_PAT', '')

    logger.info("📝 Using provided API credentials...")

    if not all([bitget_api_key, bitget_api_secret, bitget_api_password]):
        logger.error("❌ All Bitget credentials are required!")
        return

    # Connect to Redis
    logger.info("🔌 Connecting to Redis...")
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

    try:
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("✅ Connected to Redis")
    except Exception as e:
        logger.error(f"❌ Cannot connect to Redis: {e}")
        logger.info("💡 Make sure Redis is running: docker-compose up redis -d")
        return

    # Setup encryption
    logger.info("🔐 Setting up encryption...")

    # Load or generate master key
    master_key_file = '.vault_master_key'
    if os.path.exists(master_key_file):
        with open(master_key_file, 'r') as f:
            master_key = f.read().strip()
        logger.info("✅ Loaded existing master key")
    else:
        master_key = secrets.token_hex(32)
        with open(master_key_file, 'w') as f:
            f.write(master_key)
        os.chmod(master_key_file, 0o400)
        logger.info("✅ Generated new master key")

    # Setup encryption
    salt = b'viper_vault_salt_2024'
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
    fernet = Fernet(key)

    # Store credentials
    logger.info("💾 Encrypting and storing credentials...")

    # Store Bitget credentials for exchange-connector
    credentials = [
        ('exchange-connector', 'BITGET_API_KEY', bitget_api_key),
        ('exchange-connector', 'BITGET_API_SECRET', bitget_api_secret),
        ('exchange-connector', 'BITGET_API_PASSWORD', bitget_api_password),
        ('live-trading-engine', 'BITGET_API_KEY', bitget_api_key),
        ('live-trading-engine', 'BITGET_API_SECRET', bitget_api_secret),
        ('live-trading-engine', 'BITGET_API_PASSWORD', bitget_api_password),
        ('api-server', 'GITHUB_PAT', github_token),
        ('monitoring-service', 'GITHUB_PAT', github_token),
        ('strategy-optimizer', 'GITHUB_PAT', github_token),
    ]

    for service, key_name, value in credentials:
        try:
            encrypted_value = fernet.encrypt(value.encode()).decode()

            credential_data = {
                'service': service,
                'key': key_name,
                'encrypted_value': encrypted_value,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }

            redis_key = f"credential:{service}:{key_name}"
            redis_client.set(redis_key, json.dumps(credential_data))

            logger.info(f"✅ Stored {service}:{key_name}")

        except Exception as e:
            logger.error(f"❌ Failed to store {service}:{key_name}: {e}")

    # Generate access tokens
    logger.info("🔑 Generating access tokens...")

    services = [
        'api-server', 'ultra-backtester', 'strategy-optimizer',
        'live-trading-engine', 'data-manager', 'exchange-connector',
        'risk-manager', 'monitoring-service'
    ]

    tokens = []
    for service in services:
        token = secrets.token_hex(32)
        tokens.append(token)
        logger.info(f"🔑 Token for {service}: {token[:16]}...")

    # Save tokens
    with open('.vault_tokens', 'w') as f:
        f.write(','.join(tokens))
    os.chmod('.vault_tokens', 0o400)

    # Update .env file
    logger.info("📝 Updating .env file...")

    env_content = f"""# VIPER Trading System Configuration
# Using secure credential vault for sensitive data

# System Configuration
DOCKER_MODE=true
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379

# Credential Vault Configuration
VAULT_MASTER_KEY={master_key}
VAULT_ACCESS_TOKENS={','.join(tokens)}
VAULT_URL=http://credential-vault:8008

# Individual Service Tokens
VAULT_ACCESS_TOKEN_API_SERVER={tokens[0]}
VAULT_ACCESS_TOKEN_ULTRA_BACKTESTER={tokens[1]}
VAULT_ACCESS_TOKEN_STRATEGY_OPTIMIZER={tokens[2]}
VAULT_ACCESS_TOKEN_LIVE_TRADING_ENGINE={tokens[3]}
VAULT_ACCESS_TOKEN_DATA_MANAGER={tokens[4]}
VAULT_ACCESS_TOKEN_EXCHANGE_CONNECTOR={tokens[5]}
VAULT_ACCESS_TOKEN_RISK_MANAGER={tokens[6]}
VAULT_ACCESS_TOKEN_MONITORING_SERVICE={tokens[7]}

# Service Ports
API_SERVER_PORT=8000
ULTRA_BACKTESTER_PORT=8001
RISK_MANAGER_PORT=8002
DATA_MANAGER_PORT=8003
STRATEGY_OPTIMIZER_PORT=8004
EXCHANGE_CONNECTOR_PORT=8005
MONITORING_SERVICE_PORT=8006
LIVE_TRADING_ENGINE_PORT=8007
CREDENTIAL_VAULT_PORT=8008

# Trading Configuration
RISK_PER_TRADE=0.02
MAX_LEVERAGE=50
VIPER_THRESHOLD=85
ATR_LENGTH=200

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Service Discovery
API_SERVER_URL=http://api-server:8000
ULTRA_BACKTESTER_URL=http://ultra-backtester:8000
RISK_MANAGER_URL=http://risk-manager:8000
DATA_MANAGER_URL=http://data-manager:8000
STRATEGY_OPTIMIZER_URL=http://strategy-optimizer:8000
EXCHANGE_CONNECTOR_URL=http://exchange-connector:8000
MONITORING_SERVICE_URL=http://monitoring-service:8000
LIVE_TRADING_ENGINE_URL=http://live-trading-engine:8000
CREDENTIAL_VAULT_URL=http://credential-vault:8008

# Git Configuration
GIT_CONFIGURED=true
"""

    with open('.env', 'w') as f:
        f.write(env_content)

    logger.info("\n🎉 SECURE CREDENTIAL STORAGE COMPLETE!")
    logger.info("=" * 50)
    logger.info("✅ All credentials encrypted and stored in Redis")
    logger.info("✅ Access tokens generated for all services")
    logger.info("✅ .env file updated with vault configuration")
    logger.info("✅ Master key saved to .vault_master_key")
    logger.info("✅ Access tokens saved to .vault_tokens")
    logger.info("\n🔒 SECURITY NOTES:")
    logger.info("- Keep .vault_master_key secure and backup it")
    logger.info("- Never share the master key or access tokens")
    logger.info("- Credentials are encrypted in Redis")
    logger.info("- Use this script again to update credentials")
    logger.info("\n🚀 Next: Start your services with:")
    logger.info("python start_microservices.py start")

if __name__ == "__main__":
    main()
