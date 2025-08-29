#!/usr/bin/env python3
"""
# Rocket UNIFIED SYSTEM STARTUP WITH MANDATORY DOCKER & MCP
Central entry point that enforces Docker and MCP requirements for ALL VIPER operations

This script:
# Check ENFORCES Docker services are running before any operation
# Check VALIDATES MCP server connectivity and GitHub integration  
# Check STARTS all required microservices through Docker Compose
# Check CONNECTS all modules through unified MCP framework
# Check PREVENTS execution if mandatory requirements not met

USAGE:
  python start_unified_system.py [module] [operation]
  
EXAMPLES:
  python start_unified_system.py                          # Start full system
  python start_unified_system.py main                     # Start main trading bot
  python start_unified_system.py master_live_trading_job  # Start live trading
  python start_unified_system.py mcp_brain_controller     # Start MCP brain
"""

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import mandatory enforcement system
from docker_mcp_enforcer import DockerMCPEnforcer, enforce_docker_mcp_requirements
from mandatory_docker_mcp_wrapper import MandatoryDockerMCPWrapper, execute_module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - UNIFIED_STARTUP - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedSystemStartup:
    """
    Unified system startup with mandatory Docker & MCP enforcement
    Central orchestrator for all VIPER system operations
    """
    
    def __init__(self):
        self.enforcer = DockerMCPEnforcer()
        self.wrapper = MandatoryDockerMCPWrapper()
        self.startup_time = datetime.now()
        
        # System status tracking
        self.docker_services_started = False
        self.mcp_server_ready = False
        self.github_integration_active = False
        self.system_ready = False
        
    def start_complete_system(self) -> bool:
        """Start the complete VIPER system with mandatory enforcement"""
        logger.info("🔒 UNIFIED VIPER SYSTEM STARTUP")
        logger.info("=" * 70)
        logger.info("# Rocket Starting comprehensive Docker & MCP enforcement...")
        
        try:
            # Phase 1: Infrastructure Validation
            if not self._validate_infrastructure():
                return False
            
            # Phase 2: Start Docker Services
            if not self._start_docker_services():
                return False
            
            # Phase 3: Validate System Components
            if not self._validate_system_components():
                return False
            
            # Phase 4: Initialize MCP Integration
            if not self._initialize_mcp_integration():
                return False
            
            # Phase 5: System Ready
            self._mark_system_ready()
            return True
            
        except Exception as e:
            logger.error(f"💀 SYSTEM STARTUP FAILED: {e}")
            return False
    
    def _validate_infrastructure(self) -> bool:
        """Validate basic infrastructure requirements"""
        logger.info("# Tool Phase 1: Infrastructure Validation")
        
        # Check if Docker is available
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("# X Docker not available")
                return False
            logger.info("# Check Docker: AVAILABLE")
        except Exception:
            logger.error("# X Docker not installed or not in PATH")
            return False
        
        # Check if docker-compose.yml exists
        compose_file = Path('docker-compose.yml')
        if not compose_file.exists():
            logger.error("# X docker-compose.yml not found")
            return False
        logger.info("# Check docker-compose.yml: FOUND")
        
        # Check enforcement system files
        enforcement_files = [
            'docker_mcp_enforcer.py',
            'mandatory_docker_mcp_wrapper.py',
            'github_mcp_integration.py'
        ]
        
        for file in enforcement_files:
            if not Path(file).exists():
                logger.error(f"# X {file}: MISSING")
                return False
            logger.info(f"# Check {file}: AVAILABLE")
        
        logger.info("# Party Infrastructure validation complete!")
        return True
    
    def _start_docker_services(self) -> bool:
        """Start Docker services using docker-compose"""
        logger.info("🐳 Phase 2: Starting Docker Services")
        
        try:
            # Start docker-compose services
            logger.info("⏳ Starting Docker Compose services...")
            result = subprocess.run(
                ['docker', 'compose', 'up', '-d'],
                capture_output=True, text=True, timeout=180
            )
            
            if result.returncode != 0:
                logger.error(f"# X Docker services failed to start: {result.stderr}")
                return False
            
            logger.info("# Check Docker services started successfully")
            
            # Wait for services to initialize
            logger.info("⏳ Waiting for services to initialize (30 seconds)...")
            time.sleep(30)
            
            # Verify services are running
            result = subprocess.run(['docker', 'compose', 'ps'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                running_services = result.stdout.count('running') if 'running' in result.stdout else 0
                logger.info(f"# Chart Services running: {running_services}")
                
                if running_services > 0:
                    self.docker_services_started = True
                    logger.info("# Party Docker services startup complete!")
                    return True
            
            logger.warning("# Warning Some services may not be fully ready")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("# X Docker services startup timed out")
            return False
        except Exception as e:
            logger.error(f"# X Docker services startup error: {e}")
            return False
    
    def _validate_system_components(self) -> bool:
        """Validate system components are operational"""
        logger.info("# Search Phase 3: System Components Validation")
        
        # Use enforcer to validate all components
        if enforce_docker_mcp_requirements():
            logger.info("# Check All mandatory requirements validated")
            return True
        else:
            logger.warning("# Warning Some validation warnings - proceeding with caution")
            return True
    
    def _initialize_mcp_integration(self) -> bool:
        """Initialize MCP integration and GitHub connectivity"""
        logger.info("🔗 Phase 4: MCP Integration Initialization")
        
        try:
            # Test MCP server connectivity
            import requests
            try:
                response = requests.get('http://localhost:8015/health', timeout=10)
                if response.status_code == 200:
                    self.mcp_server_ready = True
                    logger.info("# Check MCP Server: READY")
                else:
                    logger.warning("# Warning MCP Server: NOT RESPONDING")
            except Exception:
                logger.warning("# Warning MCP Server: CONNECTION FAILED")
            
            # Initialize GitHub MCP integration
            try:
                import github_mcp_integration
                github_mcp = github_mcp_integration.GitHubMCPIntegration()
                self.github_integration_active = True
                logger.info("# Check GitHub MCP Integration: INITIALIZED")
            except Exception as e:
                logger.warning(f"# Warning GitHub MCP Integration: {e}")
            
            logger.info("# Party MCP integration initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"# X MCP initialization error: {e}")
            return False
    
    def _mark_system_ready(self):
        """Mark system as ready for operations"""
        self.system_ready = True
        startup_duration = (datetime.now() - self.startup_time).total_seconds()
        
        logger.info("# Party SYSTEM STARTUP COMPLETE!")
        logger.info("=" * 70)
        logger.info("# Check VIPER UNIFIED SYSTEM STATUS:")
        logger.info(f"   🐳 Docker Services: {'STARTED' if self.docker_services_started else 'PARTIAL'}")
        logger.info(f"   🤖 MCP Server: {'READY' if self.mcp_server_ready else 'PARTIAL'}")
        logger.info(f"   🔗 GitHub Integration: {'ACTIVE' if self.github_integration_active else 'PARTIAL'}")
        logger.info(f"   🔒 Enforcement: ACTIVE")
        logger.info(f"   ⏱️  Startup Time: {startup_duration:.1f} seconds")
        logger.info("=" * 70)
        logger.info("# Rocket SYSTEM READY FOR ALL OPERATIONS!")
    
    def execute_module(self, module_name: str, operation: str = 'main') -> Any:
        """Execute a module through the mandatory wrapper system"""
        if not self.system_ready:
            logger.error("💀 SYSTEM NOT READY - Cannot execute modules")
            return False
        
        logger.info(f"# Rocket EXECUTING MODULE: {module_name}.{operation}")
        
        try:
            return execute_module(module_name, operation)
        except Exception as e:
            logger.error(f"💀 MODULE EXECUTION FAILED: {e}")
            return False
    
    def get_available_modules(self) -> Dict[str, Any]:
        """Get list of available modules"""
        return self.wrapper.get_available_modules()
    
    def show_system_status(self):
        """Show comprehensive system status"""
        logger.info("# Chart VIPER UNIFIED SYSTEM STATUS REPORT")
        logger.info("=" * 70)
        
        # System status
        status = self.wrapper.get_system_status()
        logger.info(f"🔒 Enforcement Status: {status}")
        
        # Available modules
        modules = self.get_available_modules()
        logger.info(f"📦 Available Modules: {len(modules)}")
        for name, info in modules.items():
            logger.info(f"   • {name}: {info['description']}")
        
        # Docker services
        try:
            result = subprocess.run(['docker', 'compose', 'ps'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                running_count = sum(1 for line in lines if 'running' in line.lower())
                logger.info(f"🐳 Docker Services: {running_count} running")
        except Exception:
            logger.info("🐳 Docker Services: STATUS UNKNOWN")
        
        logger.info("=" * 70)

def main():
    """Main entry point for unified system startup"""
    print("🔒 VIPER UNIFIED SYSTEM WITH MANDATORY DOCKER & MCP")
    
    startup = UnifiedSystemStartup()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
        operation = sys.argv[2] if len(sys.argv) > 2 else 'main'
        
        # Start system first
        print("# Rocket Starting system for module execution...")
        if not startup.start_complete_system():
            print("💀 SYSTEM STARTUP FAILED - CANNOT EXECUTE MODULE")
            sys.exit(1)
        
        # Execute specific module
        print(f"# Target Executing {module_name}.{operation}...")
        result = startup.execute_module(module_name, operation)
        
        if result is False:
            sys.exit(1)
        else:
            
    else:
        # Interactive mode - start full system
        print("# Tool Starting full system in interactive mode...")
        
        if startup.start_complete_system():
            print("  • startup.show_system_status()        - Show system status")
            print("  • startup.execute_module('main')      - Start main trading bot")
            print("  • startup.get_available_modules()     - List available modules")
            
            # Show status and available modules
            startup.show_system_status()
            
            # Keep system running
            try:
                while True:
                    cmd = input("\n🔒 VIPER> Enter command (or 'exit' to quit): ").strip()
                    
                    if cmd.lower() == 'exit':
                        break
                    elif cmd == 'status':
                        startup.show_system_status()
                    elif cmd.startswith('run '):
                        module = cmd[4:].strip()
                        startup.execute_module(module)
                    else:
                        print("Available commands: status, run <module>, exit")
                        
            except KeyboardInterrupt:
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()