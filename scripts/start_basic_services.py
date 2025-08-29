#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Simple Service Startup
Minimal startup script for completion testing
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def start_basic_services():
    """Start basic services needed for completion testing"""
    logger.info("🚀 Starting basic VIPER services for completion testing...")
    
    # Change to the directory containing docker-compose.yml
    os.chdir(Path(__file__).parent)
    
    # Start only essential services for testing
    essential_services = [
        'api-server',
        'ultra-backtester', 
        'redis'
    ]
    
    try:
        # Check if docker compose is available
        result = subprocess.run(['docker', 'compose', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            # Try old docker-compose format
            cmd = ['docker-compose']
        else:
            cmd = ['docker', 'compose']
            
        # Start the services
        for service in essential_services:
            logger.info(f"Starting {service}...")
            result = subprocess.run(cmd + ['up', '-d', service], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"✅ {service} started successfully")
            else:
                logger.warning(f"⚠️ {service} may have issues: {result.stderr}")
        
        # Wait for services to be ready
        logger.info("⏳ Waiting for services to be ready...")
        time.sleep(15)
        
        logger.info("✅ Basic services startup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error starting services: {e}")
        return False

def check_service_health():
    """Basic health check for started services"""
    import requests
    
    services = {
        'API Server': 'http://localhost:8000/health',
        'Ultra Backtester': 'http://localhost:8001/health'
    }
    
    logger.info("🔍 Checking service health...")
    healthy_count = 0
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {name}: Healthy")
                healthy_count += 1
            else:
                logger.warning(f"⚠️ {name}: Status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ {name}: Not responding ({e})")
    
    return healthy_count

if __name__ == "__main__":
    
    if start_basic_services():
        healthy = check_service_health()
        print(f"\n📊 Services Health: {healthy} services healthy")
        
        if healthy > 0:
            print("Run: python src/utils/complete_viper_system.py")
        else:
            print("\n⚠️ Services may need more time to start. Wait a moment and try again.")
    else:
        print("\n❌ Failed to start services. Check Docker installation and permissions.")
        sys.exit(1)