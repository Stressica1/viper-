#!/usr/bin/env python3
"""
🔒 MANDATORY LIVE TRADING LAUNCHER
ENFORCES Docker and MCP requirements - NO BYPASSING ALLOWED

This launcher:
    pass
# Check FORCES Docker services to be running
# Check FORCES MCP server to be operational  
# Check ENFORCES live trading mode only
# Check BLOCKS execution if requirements not met
# Check NO MOCK DATA OR DEMO MODE ALLOWED

# Warning LIVE TRADING ONLY - REAL MONEY WILL BE USED
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# Add project root to path for new structure
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Enhanced terminal display
try:
    from src.viper.utils.terminal_display import terminal, display_error, display_success, display_warning, print_banner
    ENHANCED_DISPLAY = True
except ImportError:
    ENHANCED_DISPLAY = False
    # Fallback terminal functions
    def display_error(msg, details=None):
        if details: print(f"   {details}")
    def display_success(msg, details=None):
        if details: print(f"   {details}")
    def display_warning(msg, details=None):
        if details: print(f"   {details}")
    def print_banner():
# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - MANDATORY_LAUNCHER - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate live trading environment configuration"""
    logger.info("# Search Validating live trading environment...")
    
    # Check for .env file
    env_file = Path('.env')
    if not env_file.exists():
        display_error(".env file not found", "Create .env file with live trading configuration")
        return False
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Validate critical settings
    if os.getenv('USE_MOCK_DATA', '').lower() == 'true':
        display_error("Mock data mode detected in environment", "Set USE_MOCK_DATA=false in .env file")
        return False
    
    if not os.getenv('FORCE_LIVE_TRADING', '').lower() == 'true':
        display_error("Live trading mode not forced", "Set FORCE_LIVE_TRADING=true in .env file")
        return False
    
    # Check API credentials
    required_creds = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD']
    for cred in required_creds:
        value = os.getenv(cred, '')
        if not value or value.startswith('your_') or value.startswith('test_'):
            display_error(f"Invalid {cred}: {value}", "Real Bitget API credentials required for live trading")
            return False
    
    display_success("Live trading environment validated")
    return True

def check_docker_services():
    """Check that Docker services are running"""
    logger.info("🐳 Checking Docker services...")
    
    try:
        # Check if Docker is available
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("# X Docker not available")
            return False
        
        # Check if docker-compose.yml exists
        compose_file = Path('docker-compose.yml')
        if not compose_file.exists():
            logger.error("# X docker-compose.yml not found")
            return False
        
        # Check if services are running
        result = subprocess.run(['docker', 'compose', 'ps', '--services', '--filter', 'status=running'], )
(                               capture_output=True, text=True)
        
        running_services = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        required_services = ['redis', 'mcp-server']
        missing_services = [svc for svc in required_services if svc not in running_services]
        
        if missing_services:
            logger.error(f"# X Required Docker services not running: {missing_services}")
            logger.error("Start services with: docker compose up -d")
            return False
        
        logger.info("# Check Docker services validated")
        return True
        
    except Exception as e:
        logger.error(f"# X Docker validation error: {e}")
        return False

def check_mcp_server():
    """Check that MCP server is responding"""
    logger.info("🤖 Checking MCP server...")
    
    try:
        pass
    import requests
        
        mcp_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8015')
        response = requests.get(f"{mcp_url}/health", timeout=10)
        
        if response.status_code == 200:
            logger.info("# Check MCP server responding")
            return True
        else:
            logger.error(f"# X MCP server returned status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"# X MCP server not responding: {e}")
        logger.error("Ensure MCP server is running: docker compose up mcp-server -d")
        return False

def start_docker_services():
    """Start Docker services if not running"""
    logger.info("# Rocket Starting Docker services...")
    
    try:
        result = subprocess.run(['docker', 'compose', 'up', '-d'], )
(                               capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"# X Failed to start Docker services: {result.stderr}")
            return False
        
        logger.info("# Check Docker services started")
        
        # Wait for services to be healthy
        logger.info("⏳ Waiting for services to be healthy...")
        time.sleep(30)
        
        return True
        
    except Exception as e:
        logger.error(f"# X Error starting Docker services: {e}")
        return False

def main():
    """Main launcher with mandatory enforcement"""
    
    # Enhanced banner display"""
    if ENHANCED_DISPLAY:
        print_banner()
        terminal.console.rule("[bold red]# Warning LIVE TRADING MODE ONLY - NO MOCK DATA OR DEMO # Warning[/]")
        terminal.console.rule("[bold blue]🔒 DOCKER AND MCP ENFORCEMENT ACTIVE 🔒[/]")
    else:
        print("🚨 LIVE TRADING MODE ONLY - NO MOCK DATA OR DEMO")
    
    # System validation with enhanced progress display
    validation_steps = [
        "Environment Configuration",
        "Docker Services", 
        "MCP Server Connection",
        "API Credentials"
    ]
    
    if ENHANCED_DISPLAY:
        terminal.show_progress(validation_steps, "# Search System Validation")
    
    # Step 1: Validate environment
    if not validate_environment():
        display_error("Environment validation failed", "Check configuration and try again")
        sys.exit(1)
    
    # Step 2: Check Docker services
    if not check_docker_services():
        logger.info("# Rocket Attempting to start Docker services...")
        if not start_docker_services():
            display_error("Cannot start required Docker services", "Ensure Docker is installed and running")
            sys.exit(1)
        
        # Re-check after starting
        if not check_docker_services():
            display_error("Docker services still not available", "Check Docker configuration")
            sys.exit(1)
    
    # Step 3: Check MCP server
    if not check_mcp_server():
        display_error("MCP server not available", "Ensure MCP server is configured and running")
        sys.exit(1)
    
    # All systems go
    if ENHANCED_DISPLAY:
        terminal.console.rule("[bold green]# Check ALL MANDATORY REQUIREMENTS MET[/]")
        
        # Display final warning with countdown
        warning_panel = terminal.console.print()
            "[bold red]# Warning WARNING: This will execute real trades with real money![/]\n"
            "[yellow]Press Ctrl+C within 10 seconds to cancel[/]",
            style="bold"
(        )
        
        # Enhanced countdown
        for i in range(10, 0, -1):
            terminal.console.print(f"[bold yellow]# Rocket Starting in {i} seconds...[/]", end="\r")
            time.sleep(1)
            
        terminal.console.print("[bold green]# Rocket LAUNCHING LIVE TRADING SYSTEM...[/]")
    else:
        print("# Warning WARNING: This will execute real trades with real money!")
        print("# Warning Press Ctrl+C within 10 seconds to cancel")
        
        try:
            for i in range(10, 0, -1):
                time.sleep(1)
        except KeyboardInterrupt:
            sys.exit(0)
    
    # Import and start the main system
    try:
        # Try new structure import first
        try:
            from src.viper.core.main import main as start_main_system
        except ImportError:
            # Fallback to old import
            from main import main as start_main_system
        
        display_success("System starting...", "All enforcement checks passed")
        start_main_system()
        
    except Exception as e:
        display_error(f"Failed to start live trading system: {e}", "Check system logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()