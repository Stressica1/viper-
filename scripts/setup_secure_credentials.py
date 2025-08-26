#!/usr/bin/env python3
"""
VIPER Trading System - Secure Credential Setup
Encrypts and stores API credentials securely in the credential vault
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

class SecureCredentialSetup:
    """Setup secure credentials for VIPER system"""

    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.master_key_file = '.vault_master_key'
        self.vault_tokens_file = '.vault_tokens'

        # Initialize Redis
        self.redis_client = None

    def connect_redis(self) -> bool:
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Connected to Redis")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False

    def generate_master_key(self) -> str:
        """Generate a secure master key"""
        master_key = secrets.token_hex(32)
        logger.info(f"🔐 Generated master key: {master_key}")

        # Save master key securely
        with open(self.master_key_file, 'w') as f:
            f.write(master_key)

        # Make file read-only for security
        os.chmod(self.master_key_file, 0o400)
        logger.info("✅ Master key saved to .vault_master_key")

        return master_key

    def load_master_key(self) -> Optional[str]:
        """Load existing master key"""
        if os.path.exists(self.master_key_file):
            with open(self.master_key_file, 'r') as f:
                return f.read().strip()
        return None

    def setup_encryption(self, master_key: str) -> Fernet:
        """Setup encryption using master key"""
        salt = b'viper_vault_salt_2024'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return Fernet(key)

    def generate_access_tokens(self) -> list:
        """Generate access tokens for services"""
        tokens = []
        services = ['api-server', 'live-trading-engine', 'exchange-connector',
                   'risk-manager', 'strategy-optimizer', 'ultra-backtester',
                   'data-manager', 'monitoring-service']

        for service in services:
            token = secrets.token_hex(32)
            tokens.append(token)
            logger.info(f"🔑 Generated token for {service}: {token[:16]}...")

        # Save tokens
        with open(self.vault_tokens_file, 'w') as f:
            f.write(','.join(tokens))

        os.chmod(self.vault_tokens_file, 0o400)
        logger.info("✅ Access tokens saved to .vault_tokens")

        return tokens

    def encrypt_and_store_credential(self, fernet: Fernet, service: str, key: str, value: str):
        """Encrypt and store a credential"""
        try:
            # Encrypt the credential
            encrypted_value = fernet.encrypt(value.encode()).decode()

            # Store in Redis with metadata
            credential_data = {
                'service': service,
                'key': key,
                'encrypted_value': encrypted_value,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }

            redis_key = f"credential:{service}:{key}"
            self.redis_client.set(redis_key, json.dumps(credential_data))

            logger.info(f"✅ Stored encrypted credential for {service}:{key}")

        except Exception as e:
            logger.error(f"❌ Failed to store credential {service}:{key}: {e}")

    def store_bitget_credentials(self, fernet: Fernet, api_key: str, api_secret: str, api_password: str):
        """Store Bitget API credentials"""
        logger.info("🔐 Encrypting and storing Bitget API credentials...")

        # Store credentials for exchange-connector service
        self.encrypt_and_store_credential(fernet, 'exchange-connector', 'BITGET_API_KEY', api_key)
        self.encrypt_and_store_credential(fernet, 'exchange-connector', 'BITGET_API_SECRET', api_secret)
        self.encrypt_and_store_credential(fernet, 'exchange-connector', 'BITGET_API_PASSWORD', api_password)

        # Also store for live-trading-engine (can access exchange-connector credentials)
        self.encrypt_and_store_credential(fernet, 'live-trading-engine', 'BITGET_API_KEY', api_key)
        self.encrypt_and_store_credential(fernet, 'live-trading-engine', 'BITGET_API_SECRET', api_secret)
        self.encrypt_and_store_credential(fernet, 'live-trading-engine', 'BITGET_API_PASSWORD', api_password)

        logger.info("✅ Bitget credentials encrypted and stored")

    def store_github_token(self, fernet: Fernet, github_token: str):
        """Store GitHub PAT"""
        logger.info("🔐 Encrypting and storing GitHub token...")

        # Store for all services that might need it
        services = ['api-server', 'monitoring-service', 'strategy-optimizer']
        for service in services:
            self.encrypt_and_store_credential(fernet, service, 'GITHUB_PAT', github_token)

        logger.info("✅ GitHub token encrypted and stored")

    def update_env_file(self, master_key: str, tokens: list):
        """Update .env file to use vault references"""
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

# Trading Configuration (use vault for sensitive data)
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

# Git Configuration (use vault)
GIT_CONFIGURED=true
"""

        with open('.env.secure', 'w') as f:
            f.write(env_content)

        logger.info("✅ Created .env.secure with vault configuration")

        # Backup original .env
        if os.path.exists('.env'):
            import shutil
            shutil.copy('.env', '.env.backup')
            logger.info("✅ Original .env backed up to .env.backup")

        # Replace .env with secure version
        os.rename('.env.secure', '.env')
        logger.info("✅ .env updated to use credential vault")

    def main(self):
        """Main setup process"""
        logger.info("🚀 Starting secure credential setup...")

        # Get credentials from user
        print("\n🔐 SECURE CREDENTIAL SETUP")
        print("=" * 50)

        # Bitget credentials
        bitget_api_key = input("Enter Bitget API Key: ").strip()
        bitget_api_secret = input("Enter Bitget API Secret: ").strip()
        bitget_api_password = input("Enter Bitget API Password: ").strip()

        # GitHub token
        github_token = input("Enter GitHub Personal Access Token: ").strip()

        # Connect to Redis
        if not self.connect_redis():
            logger.error("❌ Cannot proceed without Redis connection")
            return

        # Setup encryption
        master_key = self.load_master_key()
        if not master_key:
            master_key = self.generate_master_key()

        fernet = self.setup_encryption(master_key)

        # Generate access tokens
        tokens = self.generate_access_tokens()

        # Store credentials
        self.store_bitget_credentials(fernet, bitget_api_key, bitget_api_secret, bitget_api_password)
        self.store_github_token(fernet, github_token)

        # Update .env file
        self.update_env_file(master_key, tokens)

        logger.info("\n🎉 SECURE CREDENTIAL SETUP COMPLETE!")
        logger.info("=" * 50)
        logger.info("✅ All credentials encrypted and stored in vault")
        logger.info("✅ .env file updated to use credential vault")
        logger.info("✅ Access tokens generated for all services")
        logger.info("✅ Master key saved to .vault_master_key")
        logger.info("\n🔒 IMPORTANT:")
        logger.info("- Keep .vault_master_key secure and backup it")
        logger.info("- Never share the master key or access tokens")
        logger.info("- Use 'python setup_secure_credentials.py' to update credentials")

if __name__ == "__main__":
    setup = SecureCredentialSetup()
    setup.main()
