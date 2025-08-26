#!/usr/bin/env python3
"""
VIPER Trading System - Secure Credential Client
Client utility for retrieving encrypted credentials from the credential vault
"""

import os
import json
import logging
import httpx
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import redis

logger = logging.getLogger(__name__)

class CredentialClient:
    """Client for securely retrieving credentials from the vault"""

    def __init__(self, vault_url: str = None, access_token: str = None, redis_url: str = None):
        self.vault_url = vault_url or os.getenv('VAULT_URL', 'http://credential-vault:8008')
        self.access_token = access_token or os.getenv('VAULT_ACCESS_TOKEN', '')
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://redis:6379')
        self.service_name = os.getenv('SERVICE_NAME', 'unknown-service')
        self.redis_client = None

        # Setup Redis connection
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")

    async def get_credential(self, service: str, key: str) -> Optional[str]:
        """Retrieve and decrypt a credential from the vault"""
        try:
            # First try to get from Redis cache
            if self.redis_client:
                cache_key = f"cached_credential:{service}:{key}"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    cached_cred = json.loads(cached_data)
                    logger.info(f"✅ Retrieved cached credential for {service}:{key}")
                    return cached_cred['value']

            # Retrieve from vault
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.vault_url}/credentials/retrieve/{service}/{key}",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    value = data['value']

                    # Cache the credential in Redis for 1 hour
                    if self.redis_client:
                        cache_data = {
                            'value': value,
                            'cached_at': data.get('updated_at', '')
                        }
                        self.redis_client.setex(cache_key, 3600, json.dumps(cache_data))

                    logger.info(f"✅ Retrieved credential for {service}:{key}")
                    return value
                else:
                    logger.error(f"❌ Failed to retrieve credential {service}:{key}: {response.text}")
                    return None

        except Exception as e:
            logger.error(f"❌ Error retrieving credential {service}:{key}: {e}")
            return None

    async def get_bitget_credentials(self) -> Dict[str, str]:
        """Get Bitget API credentials for the current service"""
        credentials = {}

        # Try to get credentials for this specific service first
        api_key = await self.get_credential(self.service_name, 'BITGET_API_KEY')
        api_secret = await self.get_credential(self.service_name, 'BITGET_API_SECRET')
        api_password = await self.get_credential(self.service_name, 'BITGET_API_PASSWORD')

        # If not found for this service, try exchange-connector
        if not api_key:
            api_key = await self.get_credential('exchange-connector', 'BITGET_API_KEY')
        if not api_secret:
            api_secret = await self.get_credential('exchange-connector', 'BITGET_API_SECRET')
        if not api_password:
            api_password = await self.get_credential('exchange-connector', 'BITGET_API_PASSWORD')

        if api_key and api_secret and api_password:
            credentials = {
                'api_key': api_key,
                'api_secret': api_secret,
                'api_password': api_password
            }
            logger.info("✅ Retrieved Bitget API credentials")
        else:
            logger.warning("❌ Could not retrieve complete Bitget credentials")

        return credentials

    async def get_github_token(self) -> Optional[str]:
        """Get GitHub Personal Access Token"""
        # Try to get from this service first, then from api-server
        token = await self.get_credential(self.service_name, 'GITHUB_PAT')
        if not token:
            token = await self.get_credential('api-server', 'GITHUB_PAT')

        if token:
            logger.info("✅ Retrieved GitHub token")
        else:
            logger.warning("❌ Could not retrieve GitHub token")

        return token

    def get_credential_sync(self, service: str, key: str) -> Optional[str]:
        """Synchronous version for non-async contexts"""
        try:
            import asyncio
            return asyncio.run(self.get_credential(service, key))
        except Exception as e:
            logger.error(f"❌ Sync credential retrieval failed: {e}")
            return None

    def get_bitget_credentials_sync(self) -> Dict[str, str]:
        """Synchronous version of get_bitget_credentials"""
        try:
            import asyncio
            return asyncio.run(self.get_bitget_credentials())
        except Exception as e:
            logger.error(f"❌ Sync Bitget credentials retrieval failed: {e}")
            return {}

    def get_github_token_sync(self) -> Optional[str]:
        """Synchronous version of get_github_token"""
        try:
            import asyncio
            return asyncio.run(self.get_github_token())
        except Exception as e:
            logger.error(f"❌ Sync GitHub token retrieval failed: {e}")
            return None

# Global instance for easy access
_credential_client = None

def get_credential_client() -> CredentialClient:
    """Get or create the global credential client instance"""
    global _credential_client
    if _credential_client is None:
        _credential_client = CredentialClient()
    return _credential_client

# Convenience functions for common use cases
async def get_bitget_credentials_async() -> Dict[str, str]:
    """Async convenience function to get Bitget credentials"""
    client = get_credential_client()
    return await client.get_bitget_credentials()

def get_bitget_credentials() -> Dict[str, str]:
    """Sync convenience function to get Bitget credentials"""
    client = get_credential_client()
    return client.get_bitget_credentials_sync()

async def get_github_token_async() -> Optional[str]:
    """Async convenience function to get GitHub token"""
    client = get_credential_client()
    return await client.get_github_token()

def get_github_token() -> Optional[str]:
    """Sync convenience function to get GitHub token"""
    client = get_credential_client()
    return client.get_github_token_sync()

# Auto-initialize access token from environment
def initialize_access_token(service_name: str = None):
    """Initialize the credential client with the correct access token for the service"""
    global _credential_client

    service = service_name or os.getenv('SERVICE_NAME', 'unknown-service')

    # Load access tokens and find the one for this service
    vault_tokens = os.getenv('VAULT_ACCESS_TOKENS', '')
    if vault_tokens:
        tokens = vault_tokens.split(',')
        # Use service index to determine which token to use
        services = ['api-server', 'ultra-backtester', 'strategy-optimizer',
                   'live-trading-engine', 'data-manager', 'exchange-connector',
                   'risk-manager', 'monitoring-service']

        try:
            service_index = services.index(service)
            if service_index < len(tokens):
                access_token = tokens[service_index].strip()
                _credential_client = CredentialClient(access_token=access_token)
                logger.info(f"✅ Initialized credential client for {service}")
                return True
        except ValueError:
            pass

    # Fallback: create client without specific token
    _credential_client = CredentialClient()
    logger.warning(f"⚠️ Could not find access token for {service}, using fallback")
    return False

# Initialize on import if SERVICE_NAME is set
if os.getenv('SERVICE_NAME'):
    initialize_access_token()
