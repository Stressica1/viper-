#!/usr/bin/env python3
"""
🔒 MANDATORY LIVE TRADING LAUNCHER
ENFORCES Docker and MCP requirements - NO BYPASSING ALLOWED

This launcher:
✅ FORCES Docker services to be running
✅ FORCES MCP server to be operational  
✅ ENFORCES live trading mode only
✅ BLOCKS execution if requirements not met
✅ NO MOCK DATA OR DEMO MODE ALLOWED

⚠️ LIVE TRADING ONLY - REAL MONEY WILL BE USED
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MANDATORY_LAUNCHER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate live trading environment configuration"""
    logger.info("🔍 Validating live trading environment...")
    
    # Check for .env file
    env_file = Path('.env')
    if not env_file.exists():
        logger.error("❌ .env file not found")
        logger.error("Create .env file with live trading configuration")
        return False
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Validate critical settings
    if os.getenv('USE_MOCK_DATA', '').lower() == 'true':
        logger.error("❌ Mock data mode detected in environment")
        logger.error("Set USE_MOCK_DATA=false in .env file")
        return False
    
    if not os.getenv('FORCE_LIVE_TRADING', '').lower() == 'true':
        logger.error("❌ Live trading mode not forced")
        logger.error("Set FORCE_LIVE_TRADING=true in .env file")
        return False
    
    # Check API credentials
    required_creds = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD']
    for cred in required_creds:
        value = os.getenv(cred, '')
        if not value or value.startswith('your_') or value.startswith('test_'):
            logger.error(f"❌ Invalid {cred}: {value}")
            logger.error("Real Bitget API credentials required for live trading")
            return False
    
    logger.info("✅ Live trading environment validated")
    return True

def check_docker_services():
    """Check that Docker services are running"""
    logger.info("🐳 Checking Docker services...")
    
    try:
        # Check if Docker is available
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("❌ Docker not available")
            return False
        
        # Check if docker-compose.yml exists
        compose_file = Path('docker-compose.yml')
        if not compose_file.exists():
            logger.error("❌ docker-compose.yml not found")
            return False
        
        # Check if services are running
        result = subprocess.run(['docker', 'compose', 'ps', '--services', '--filter', 'status=running'], 
                               capture_output=True, text=True)
        
        running_services = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        required_services = ['redis', 'mcp-server']
        missing_services = [svc for svc in required_services if svc not in running_services]
        
        if missing_services:
            logger.error(f"❌ Required Docker services not running: {missing_services}")
            logger.error("Start services with: docker compose up -d")
            return False
        
        logger.info("✅ Docker services validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Docker validation error: {e}")
        return False

def check_mcp_server():
    """Check that MCP server is responding"""
    logger.info("🤖 Checking MCP server...")
    
    try:
        import requests
        
        mcp_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8015')
        response = requests.get(f"{mcp_url}/health", timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ MCP server responding")
            return True
        else:
            logger.error(f"❌ MCP server returned status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ MCP server not responding: {e}")
        logger.error("Ensure MCP server is running: docker compose up mcp-server -d")
        return False

def start_docker_services():
    """Start Docker services if not running"""
    logger.info("🚀 Starting Docker services...")
    
    try:
        result = subprocess.run(['docker', 'compose', 'up', '-d'], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"❌ Failed to start Docker services: {result.stderr}")
            return False
        
        logger.info("✅ Docker services started")
        
        # Wait for services to be healthy
        logger.info("⏳ Waiting for services to be healthy...")
        time.sleep(30)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error starting Docker services: {e}")
        return False

def main():
    """Main launcher with mandatory enforcement"""
    print("🔒 VIPER MANDATORY LIVE TRADING LAUNCHER")
    print("=" * 70)
    print("🚨 LIVE TRADING MODE ONLY - NO MOCK DATA OR DEMO")
    print("🔒 DOCKER AND MCP ENFORCEMENT ACTIVE")
    print("=" * 70)
    
    # Step 1: Validate environment
    if not validate_environment():
        logger.error("💀 Environment validation failed")
        sys.exit(1)
    
    # Step 2: Check Docker services
    if not check_docker_services():
        logger.info("🚀 Attempting to start Docker services...")
        if not start_docker_services():
            logger.error("💀 Cannot start required Docker services")
            sys.exit(1)
        
        # Re-check after starting
        if not check_docker_services():
            logger.error("💀 Docker services still not available")
            sys.exit(1)
    
    # Step 3: Check MCP server
    if not check_mcp_server():
        logger.error("💀 MCP server not available")
        logger.error("Ensure MCP server is configured and running")
        sys.exit(1)
    
    print("✅ ALL MANDATORY REQUIREMENTS MET")
    print("🚀 Starting live trading system...")
    print("=" * 70)
    print("⚠️ WARNING: This will execute real trades with real money!")
    print("⚠️ Press Ctrl+C within 10 seconds to cancel")
    print("=" * 70)
    
    try:
        for i in range(10, 0, -1):
            print(f"Starting in {i} seconds...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Launch cancelled by user")
        sys.exit(0)
    
    print("\n🚀 LAUNCHING LIVE TRADING SYSTEM...")
    
    # Import and start the main system
    try:
        from main import main as start_main_system
        start_main_system()
        
    except Exception as e:
        logger.error(f"❌ Failed to start live trading system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()