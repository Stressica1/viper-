#!/usr/bin/env python3
"""
VIPER Trading System - Secure Credential Vault Service
Manages encrypted API credentials and secrets for the trading system
"""

import os
import sys
import json
import base64
import hashlib
import secrets
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

class CredentialVault:
    """Secure credential vault with encryption and access control"""

    def __init__(self):
        self.app = FastAPI(
            title="VIPER Credential Vault",
            version="1.0.0",
            description="Secure credential management for VIPER Trading System"
        )

        # Load configuration
        self.vault_key = os.getenv('VAULT_MASTER_KEY', self._generate_master_key())
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        self.service_name = 'credential-vault'

        # Initialize encryption
        self.fernet = self._setup_encryption()

        # Setup security
        self.security = HTTPBearer()
        self.valid_tokens = self._load_valid_tokens()

        # Initialize Redis
        self.redis_client = None

        # Setup routes
        self._setup_routes()

    def _generate_master_key(self) -> str:
        """Generate a new master key if none exists"""
        key = secrets.token_hex(32)
        logger.warning(f"Generated new master key: {key}")
        return key

    def _setup_encryption(self) -> Fernet:
        """Setup encryption using master key"""
        # Derive key from master key using PBKDF2
        salt = b'viper_vault_salt_2024'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.vault_key.encode()))
        return Fernet(key)

    def _load_valid_tokens(self) -> set:
        """Load valid access tokens"""
        tokens = os.getenv('VAULT_ACCESS_TOKENS', '').split(',')
        return set(t.strip() for t in tokens if t.strip())

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "service": self.service_name,
                "timestamp": datetime.now().isoformat()
            }

        @self.app.post("/credentials/store")
        async def store_credential(
            service: str,
            key: str,
            value: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Store an encrypted credential"""
            if not self._verify_token(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid access token"
                )

            try:
                # Encrypt the credential
                encrypted_value = self.fernet.encrypt(value.encode()).decode()

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

                logger.info(f"✅ Stored credential for {service}:{key}")
                return {"status": "stored", "service": service, "key": key}

            except Exception as e:
                logger.error(f"❌ Failed to store credential: {e}")
                raise HTTPException(status_code=500, detail="Failed to store credential")

        @self.app.get("/credentials/retrieve/{service}/{key}")
        async def retrieve_credential(
            service: str,
            key: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Retrieve and decrypt a credential"""
            if not self._verify_token(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid access token"
                )

            try:
                redis_key = f"credential:{service}:{key}"
                data = self.redis_client.get(redis_key)

                if not data:
                    raise HTTPException(status_code=404, detail="Credential not found")

                credential_data = json.loads(data)

                # Decrypt the value
                decrypted_value = self.fernet.decrypt(
                    credential_data['encrypted_value'].encode()
                ).decode()

                logger.info(f"✅ Retrieved credential for {service}:{key}")
                return {
                    "service": service,
                    "key": key,
                    "value": decrypted_value,
                    "updated_at": credential_data['updated_at']
                }

            except Exception as e:
                logger.error(f"❌ Failed to retrieve credential: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve credential")

        @self.app.get("/credentials/list/{service}")
        async def list_credentials(
            service: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """List all credentials for a service"""
            if not self._verify_token(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid access token"
                )

            try:
                pattern = f"credential:{service}:*"
                keys = self.redis_client.keys(pattern)

                credentials_list = []
                for key in keys:
                    data = self.redis_client.get(key)
                    if data:
                        cred_data = json.loads(data)
                        credentials_list.append({
                            "key": cred_data['key'],
                            "updated_at": cred_data['updated_at']
                        })

                return {"service": service, "credentials": credentials_list}

            except Exception as e:
                logger.error(f"❌ Failed to list credentials: {e}")
                raise HTTPException(status_code=500, detail="Failed to list credentials")

        @self.app.delete("/credentials/delete/{service}/{key}")
        async def delete_credential(
            service: str,
            key: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Delete a credential"""
            if not self._verify_token(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid access token"
                )

            try:
                redis_key = f"credential:{service}:{key}"
                result = self.redis_client.delete(redis_key)

                if result == 0:
                    raise HTTPException(status_code=404, detail="Credential not found")

                logger.info(f"✅ Deleted credential for {service}:{key}")
                return {"status": "deleted", "service": service, "key": key}

            except Exception as e:
                logger.error(f"❌ Failed to delete credential: {e}")
                raise HTTPException(status_code=500, detail="Failed to delete credential")

    def _verify_token(self, token: str) -> bool:
        """Verify access token"""
        return token in self.valid_tokens

    async def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False

    def initialize_redis_sync(self) -> bool:
        """Initialize Redis connection synchronously"""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False

    def start_server(self, host: str = "0.0.0.0", port: int = 8008):
        """Start the credential vault server"""
        # Initialize Redis synchronously first
        if not self.initialize_redis_sync():
            logger.error("❌ Cannot start server without Redis")
            return

        logger.info(f"🚀 Starting {self.service_name} on {host}:{port}")
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

def main():
    """Main entry point"""
    vault = CredentialVault()

    # Run the server
    vault.start_server()

if __name__ == "__main__":
    main()
