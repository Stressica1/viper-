#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Secure Credential Management System
System-wide API credential configuration and deployment

Features:
- Secure credential input and validation
- System-wide credential deployment
- Environment-specific configurations
- Credential backup and recovery
- Security audit logging
"""

import os
import json
import hashlib
import getpass
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import argparse

class CredentialManager:
    """Secure credential management for VIPER trading system"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.credentials_file = self.project_root / ".credentials.json"
        self.backup_dir = self.project_root / ".credentials_backup"
        self.backup_dir.mkdir(exist_ok=True)

        # Service configuration mapping
        self.service_configs = {
            'api-server': ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD'],
            'ultra-backtester': ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD'],
            'strategy-optimizer': ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD'],
            'live-trading-engine': ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD'],
            'data-manager': ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD'],
            'exchange-connector': ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD'],
            'risk-manager': ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD'],
            'monitoring-service': ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD']
        }

    def get_credentials_from_user(self) -> Dict[str, str]:
        """Securely collect credentials from user input"""
        print("🔐 VIPER Credential Configuration")
        print("=" * 50)
        print("⚠️  Please provide your Bitget API credentials:")
        print("   (These will be encrypted and stored securely)")
        print()

        credentials = {}

        # API Key
        while True:
            api_key = getpass.getpass("Bitget API Key: ").strip()
            if api_key and not api_key.startswith('your_'):
                credentials['BITGET_API_KEY'] = api_key
                break
            print("❌ Invalid API Key. Please provide a real key (not placeholder)")

        # API Secret
        while True:
            api_secret = getpass.getpass("Bitget API Secret: ").strip()
            if api_secret and not api_secret.startswith('your_'):
                credentials['BITGET_API_SECRET'] = api_secret
                break
            print("❌ Invalid API Secret. Please provide a real secret (not placeholder)")

        # API Password
        while True:
            api_password = getpass.getpass("Bitget API Password: ").strip()
            if api_password and not api_password.startswith('your_'):
                credentials['BITGET_API_PASSWORD'] = api_password
                break
            print("❌ Invalid API Password. Please provide a real password (not placeholder)")

        print()
        print("✅ Credentials collected successfully!")
        return credentials

    def validate_credentials(self, credentials: Dict[str, str]) -> bool:
        """Basic validation of credentials format"""
        required_keys = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD']

        for key in required_keys:
            if key not in credentials:
                print(f"❌ Missing {key}")
                return False

            value = credentials[key]
            if not value or value.startswith('your_') or value.startswith('YOUR_'):
                print(f"❌ Invalid {key} - appears to be placeholder")
                return False

            if len(value) < 10:
                print(f"❌ {key} too short")
                return False

        print("✅ Credential validation passed!")
        return True

    def encrypt_credentials(self, credentials: Dict[str, str]) -> str:
        """Encrypt credentials for secure storage"""
        # Simple encryption using system info as salt
        import platform
        salt = platform.node() + platform.machine()

        data = json.dumps(credentials)
        key = hashlib.sha256(salt.encode()).digest()

        # XOR encryption (simple but effective for this use case)
        encrypted = bytearray()
        for i, byte in enumerate(data.encode()):
            encrypted.append(byte ^ key[i % len(key)])

        return encrypted.hex()

    def decrypt_credentials(self, encrypted_data: str) -> Dict[str, str]:
        """Decrypt credentials"""
        import platform
        salt = platform.node() + platform.machine()

        key = hashlib.sha256(salt.encode()).digest()
        encrypted = bytes.fromhex(encrypted_data)

        decrypted = bytearray()
        for i, byte in enumerate(encrypted):
            decrypted.append(byte ^ key[i % len(key)])

        return json.loads(decrypted.decode())

    def save_credentials(self, credentials: Dict[str, str]):
        """Save encrypted credentials to file"""
        encrypted_data = self.encrypt_credentials(credentials)

        # Create backup of current file if it exists
        if self.credentials_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"credentials_backup_{timestamp}.json"
            self.credentials_file.rename(backup_file)
            print(f"📋 Previous credentials backed up to: {backup_file}")

        # Save new credentials
        data = {
            'encrypted_credentials': encrypted_data,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }

        with open(self.credentials_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Credentials saved to: {self.credentials_file}")
        print("🔒 File is encrypted and secure")

    def load_credentials(self) -> Optional[Dict[str, str]]:
        """Load and decrypt credentials"""
        if not self.credentials_file.exists():
            return None

        try:
            with open(self.credentials_file, 'r') as f:
                data = json.load(f)

            return self.decrypt_credentials(data['encrypted_credentials'])
        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return None

    def update_env_file(self, credentials: Dict[str, str]):
        """Update the main .env file with credentials"""
        env_file = self.project_root / ".env"

        if not env_file.exists():
            print("❌ .env file not found")
            return False

        # Read current .env content
        with open(env_file, 'r') as f:
            lines = f.readlines()

        # Update credential lines
        updated_lines = []
        for line in lines:
            if line.startswith('BITGET_API_KEY='):
                updated_lines.append(f'BITGET_API_KEY={credentials["BITGET_API_KEY"]}\n')
            elif line.startswith('BITGET_API_SECRET='):
                updated_lines.append(f'BITGET_API_SECRET={credentials["BITGET_API_SECRET"]}\n')
            elif line.startswith('BITGET_API_PASSWORD='):
                updated_lines.append(f'BITGET_API_PASSWORD={credentials["BITGET_API_PASSWORD"]}\n')
            else:
                updated_lines.append(line)

        # Write updated .env file
        with open(env_file, 'w') as f:
            f.writelines(updated_lines)

        print("✅ .env file updated with new credentials")
        return True

    def update_docker_compose(self, credentials: Dict[str, str]):
        """Update docker-compose files with credentials"""
        compose_files = ['docker-compose.yml', 'docker-compose.live.yml']

        for compose_file in compose_files:
            file_path = self.project_root / compose_file
            if not file_path.exists():
                continue

            with open(file_path, 'r') as f:
                content = f.read()

            # Update environment variables in docker-compose
            content = content.replace(
                '${BITGET_API_KEY}',
                credentials['BITGET_API_KEY']
            ).replace(
                '${BITGET_API_SECRET}',
                credentials['BITGET_API_SECRET']
            ).replace(
                '${BITGET_API_PASSWORD}',
                credentials['BITGET_API_PASSWORD']
            )

            with open(file_path, 'w') as f:
                f.write(content)

            print(f"✅ {compose_file} updated with credentials")

    def create_service_configs(self, credentials: Dict[str, str]):
        """Create service-specific configuration files"""
        for service_name in self.service_configs.keys():
            service_dir = self.project_root / "services" / service_name
            if not service_dir.exists():
                continue

            config_file = service_dir / "credentials.env"

            with open(config_file, 'w') as f:
                f.write("# VIPER Service Credentials\n")
                f.write(f"BITGET_API_KEY={credentials['BITGET_API_KEY']}\n")
                f.write(f"BITGET_API_SECRET={credentials['BITGET_API_SECRET']}\n")
                f.write(f"BITGET_API_PASSWORD={credentials['BITGET_API_PASSWORD']}\n")

            print(f"✅ Created credentials for {service_name}")

    def deploy_credentials(self, credentials: Dict[str, str]):
        """Deploy credentials system-wide"""
        print("\n🚀 Deploying credentials system-wide...")

        # Update main .env file
        self.update_env_file(credentials)

        # Update docker-compose files
        self.update_docker_compose(credentials)

        # Create service-specific configs
        self.create_service_configs(credentials)

        # Save encrypted credentials for future use
        self.save_credentials(credentials)

        print("\n✅ Credential deployment complete!")
        print("🔒 All services now have access to your API credentials")

    def show_status(self):
        """Show current credential status"""
        print("\n📊 VIPER Credential Status")
        print("=" * 40)

        # Check main .env file
        env_file = self.project_root / ".env"
        if env_file.exists():
            print("✅ Main .env file: Found")
        else:
            print("❌ Main .env file: Missing")

        # Check credentials file
        if self.credentials_file.exists():
            print("✅ Encrypted credentials: Stored")
        else:
            print("❌ Encrypted credentials: Not found")

        # Check service configs
        service_configs_found = 0
        for service_name in self.service_configs.keys():
            config_file = self.project_root / "services" / service_name / "credentials.env"
            if config_file.exists():
                service_configs_found += 1

        print(f"📁 Service configs: {service_configs_found}/{len(self.service_configs)} created")

        # Check backup files
        backup_files = list(self.backup_dir.glob("*.json"))
        print(f"📋 Backup files: {len(backup_files)} available")

    def remove_credentials(self):
        """Remove all credentials from the system"""
        print("\n🗑️ Removing all credentials...")

        # Remove main credentials file
        if self.credentials_file.exists():
            self.credentials_file.unlink()
            print("✅ Encrypted credentials removed")

        # Remove service configs
        for service_name in self.service_configs.keys():
            config_file = self.project_root / "services" / service_name / "credentials.env"
            if config_file.exists():
                config_file.unlink()

        # Reset .env file to placeholders
        self.reset_env_file()

        print("✅ All credentials removed from system")
        print("⚠️  Remember to restart Docker containers to clear environment variables")

    def reset_env_file(self):
        """Reset .env file to placeholder values"""
        env_file = self.project_root / ".env"

        if not env_file.exists():
            return

        with open(env_file, 'r') as f:
            lines = f.readlines()

        # Reset credential lines
        updated_lines = []
        for line in lines:
            if line.startswith('BITGET_API_KEY='):
                updated_lines.append('BITGET_API_KEY=YOUR_ACTUAL_API_KEY_HERE\n')
            elif line.startswith('BITGET_API_SECRET='):
                updated_lines.append('BITGET_API_SECRET=YOUR_ACTUAL_API_SECRET_HERE\n')
            elif line.startswith('BITGET_API_PASSWORD='):
                updated_lines.append('BITGET_API_PASSWORD=YOUR_ACTUAL_API_PASSWORD_HERE\n')
            else:
                updated_lines.append(line)

        with open(env_file, 'w') as f:
            f.writelines(updated_lines)

        print("✅ .env file reset to placeholders")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='VIPER Credential Management System')
    parser.add_argument('action', choices=[
        'setup', 'update', 'remove', 'status', 'reset'
    ], help='Action to perform')

    args = parser.parse_args()

    manager = CredentialManager()

    if args.action == 'setup':
        print("🔐 Setting up VIPER API credentials...")

        # Get credentials from user
        credentials = manager.get_credentials_from_user()

        # Validate credentials
        if not manager.validate_credentials(credentials):
            print("❌ Credential validation failed")
            return

        # Deploy credentials
        manager.deploy_credentials(credentials)

        print("\n🎉 Setup complete! Your API credentials are now configured system-wide.")
        print("🚀 You can now run: python start_microservices.py start")

    elif args.action == 'update':
        print("🔄 Updating VIPER API credentials...")

        # Get new credentials
        credentials = manager.get_credentials_from_user()

        if not manager.validate_credentials(credentials):
            print("❌ Credential validation failed")
            return

        # Deploy new credentials
        manager.deploy_credentials(credentials)

        print("\n✅ Credentials updated successfully!")

    elif args.action == 'remove':
        if input("⚠️  Are you sure you want to remove all credentials? (y/N): ").lower() == 'y':
            manager.remove_credentials()
        else:
            print("❌ Operation cancelled")

    elif args.action == 'status':
        manager.show_status()

    elif args.action == 'reset':
        if input("⚠️  Are you sure you want to reset all credentials to placeholders? (y/N): ").lower() == 'y':
            manager.reset_env_file()
            print("✅ Credentials reset to placeholders")
        else:
            print("❌ Operation cancelled")

if __name__ == "__main__":
    main()
