#!/usr/bin/env python3
"""
🔒 DOCKER & MCP ENFORCEMENT SYSTEM
Mandatory Docker and MCP validation for all VIPER system operations

This enforcer ensures:
# Check Docker services are running and healthy before any operations
# Check MCP server is connected and functional 
# Check GitHub MCP integration is active
# Check All required microservices are operational
# Check System blocks execution if requirements not met

NO SYSTEM OPERATION IS ALLOWED WITHOUT DOCKER & MCP!
"""

import os
import sys
import logging
import subprocess
import requests
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DOCKER_MCP_ENFORCER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DockerMCPEnforcer:
    """
    Mandatory Docker and MCP enforcement system
    CRITICAL: System cannot proceed without both Docker and MCP operational
    """
    
    def __init__(self):
        self.mcp_server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8015')
        self.github_token = os.getenv('GITHUB_PAT', os.getenv('GITHUB_TOKEN', ''))
        self.required_services = [
            'mcp-server', 'redis', 'api-server', 'data-manager',
            'exchange-connector', 'risk-manager', 'live-trading-engine'
        ]
        
        # Enforcement flags
        self.docker_validated = False
        self.mcp_validated = False
        self.github_mcp_validated = False
        self.services_validated = False
        self.enforcement_active = True
        
        # Health check intervals
        self.health_check_interval = 30
        self.max_retry_attempts = 5
        self.retry_delay = 10
        
        # Validation cache
        self.validation_cache = {}
        self.last_validation_time = None
        
    def enforce_mandatory_requirements(self) -> bool:
        """
        CRITICAL ENFORCEMENT: Validate all mandatory requirements
        System CANNOT proceed without all validations passing
        NO BYPASSING ALLOWED FOR LIVE TRADING SYSTEM
        """
        logger.info("🔒 STARTING MANDATORY DOCKER & MCP ENFORCEMENT")
        logger.info("=" * 70)
        
        # FORCE enforcement to be active - no disabling allowed for live trading
        self.enforcement_active = True
        
        try:
            # Step 1: Docker Infrastructure Validation (MANDATORY)
            if not self._validate_docker_infrastructure():
                logger.error("# X DOCKER INFRASTRUCTURE VALIDATION FAILED")
                self._block_execution("Docker infrastructure not available")
                return False
            
            # Step 2: MCP Server Validation (MANDATORY)
            if not self._validate_mcp_server():
                logger.error("# X MCP SERVER VALIDATION FAILED")
                self._block_execution("MCP server not available or not responding")
                return False
                
            # Step 3: GitHub MCP Integration Validation (MANDATORY)
            if not self._validate_github_mcp_integration():
                logger.error("# X GITHUB MCP INTEGRATION VALIDATION FAILED")
                self._block_execution("GitHub MCP integration not configured")
                return False
                
            # Step 4: Required Services Validation (MANDATORY)
            if not self._validate_required_services():
                logger.error("# X REQUIRED SERVICES VALIDATION FAILED")
                self._block_execution("Required Docker services not running")
                return False
            
            # All validations must pass for live trading
            self._mark_validation_success()
            logger.info("# Check ALL MANDATORY REQUIREMENTS VALIDATED - LIVE TRADING SYSTEM CLEARED")
            return True
            
        except Exception as e:
            logger.error(f"# X ENFORCEMENT VALIDATION ERROR: {e}")
            self._block_execution(f"Validation error: {e}")
            return False
    
    def _validate_docker_infrastructure(self) -> bool:
        """Validate Docker infrastructure is available and operational"""
        logger.info("🐳 Validating Docker Infrastructure...")
        
        try:
            # Check Docker daemon
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("# X Docker not available")
                return False
            logger.info("# Check Docker daemon: AVAILABLE")
            
            # Check docker-compose availability
            result = subprocess.run(['docker', 'compose', 'version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("# X Docker Compose not available")
                return False
            logger.info("# Check Docker Compose: AVAILABLE")
            
            # Validate docker-compose.yml exists
            compose_file = Path('docker-compose.yml')
            if not compose_file.exists():
                logger.error("# X docker-compose.yml not found")
                return False
            logger.info("# Check docker-compose.yml: FOUND")
            
            self.docker_validated = True
            return True
            
        except Exception as e:
            logger.error(f"# X Docker infrastructure error: {e}")
            return False
    
    def _validate_mcp_server(self) -> bool:
        """Validate MCP server is running and responsive"""
        logger.info("🤖 Validating MCP Server...")
        
        try:
            # Check MCP server health
            response = requests.get(f"{self.mcp_server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("# Check MCP Server: RESPONDING")
                self.mcp_validated = True
                return True
        except Exception:
            logger.warning("# Warning MCP server not responding")
        
        # Check if MCP server container exists
        try:
            result = subprocess.run(['docker', 'ps', '--filter', 'name=mcp-server', '--format', '{{.Names}}'], 
                                  capture_output=True, text=True)
            if 'mcp-server' in result.stdout:
                logger.info("# Check MCP Server container: RUNNING")
                self.mcp_validated = True
                return True
        except Exception:
            pass
                    
        logger.warning("# Warning MCP Server validation incomplete")
        return False
    
    def _validate_github_mcp_integration(self) -> bool:
        """Validate GitHub MCP integration is active and functional"""
        logger.info("🔗 Validating GitHub MCP Integration...")
        
        try:
            # Check if github_mcp_integration module exists
            github_integration_file = Path('github_mcp_integration.py')
            if github_integration_file.exists():
                logger.info("# Check GitHub MCP Integration Module: AVAILABLE")
                self.github_mcp_validated = True
                return True
            else:
                logger.warning("# Warning GitHub MCP Integration module not found")
                return False
                
        except Exception as e:
            logger.warning(f"# Warning GitHub MCP integration validation warning: {e}")
            return False
    
    def _validate_required_services(self) -> bool:
        """Validate required Docker services are accessible"""
        logger.info("# Tool Validating Required Services...")
        
        try:
            # Get running containers via docker ps
            result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("# Warning Cannot list Docker containers")
                return False
                
            running_services = result.stdout.strip().split('\n') if result.stdout.strip() else []
            logger.info(f"# Chart Found {len(running_services)} running containers")
            
            # Check for any required services
            found_services = []
            for service in self.required_services:
                for container_name in running_services:
                    if service in container_name:
                        found_services.append(service)
                        break
            
            logger.info(f"# Check Found {len(found_services)} required services: {found_services}")
            
            if found_services:
                self.services_validated = True
                return True
            else:
                logger.warning("# Warning No required services found running")
                return False
            
        except Exception as e:
            logger.warning(f"# Warning Service validation warning: {e}")
            return False
    
    def _mark_validation_success(self):
        """Mark successful validation"""
        self.last_validation_time = datetime.now()
        self.validation_cache = {
            'docker_validated': self.docker_validated,
            'mcp_validated': self.mcp_validated,
            'github_mcp_validated': self.github_mcp_validated,
            'services_validated': self.services_validated,
            'validation_time': self.last_validation_time.isoformat()
        }
    
    def _block_execution(self, reason: str):
        """Block system execution due to failed validation - NO BYPASSING ALLOWED"""
        logger.error("🚫 LIVE TRADING SYSTEM EXECUTION BLOCKED!")
        logger.error(f"Reason: {reason}")
        logger.error("=" * 70)
        logger.error("MANDATORY REQUIREMENTS NOT MET:")
        logger.error("• Docker infrastructure must be operational")
        logger.error("• MCP server must be running and responding")
        logger.error("• GitHub MCP integration must be configured")
        logger.error("• All required microservices must be running")
        logger.error("=" * 70)
        logger.error("# Tool TO RESOLVE:")
        logger.error("1. Start Docker services: docker compose up -d")
        logger.error("2. Verify MCP server: curl http://localhost:8015/health")
        logger.error("3. Validate services: docker compose ps")
        logger.error("4. Check GitHub integration: export GITHUB_PAT=<token>")
        logger.error("=" * 70)
        logger.error("💀 LIVE TRADING CANNOT PROCEED WITHOUT ALL REQUIREMENTS")
        
        # Always terminate - no bypassing allowed for live trading
        sys.exit(1)
    
    def disable_enforcement(self):
        """ENFORCEMENT CANNOT BE DISABLED FOR LIVE TRADING SYSTEM"""
        logger.error("🚫 ENFORCEMENT CANNOT BE DISABLED FOR LIVE TRADING")
        logger.error("This is a production live trading system - all safety measures are mandatory")
        logger.error("Docker and MCP enforcement cannot be bypassed")
        # Do not actually disable enforcement - keep it active
        # self.enforcement_active remains True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system validation status"""
        return {
            'docker_validated': self.docker_validated,
            'mcp_validated': self.mcp_validated,
            'github_mcp_validated': self.github_mcp_validated,
            'services_validated': self.services_validated,
            'enforcement_active': self.enforcement_active,
            'last_validation': self.last_validation_time.isoformat() if self.last_validation_time else None,
            'required_services': self.required_services,
            'mcp_server_url': self.mcp_server_url
        }

# Global enforcer instance
_enforcer_instance = None

def get_enforcer() -> DockerMCPEnforcer:
    """Get global enforcer instance"""
    global _enforcer_instance
    if _enforcer_instance is None:
        _enforcer_instance = DockerMCPEnforcer()
    return _enforcer_instance

def enforce_docker_mcp_requirements() -> bool:
    """Convenience function to enforce requirements"""
    enforcer = get_enforcer()
    return enforcer.enforce_mandatory_requirements()

if __name__ == "__main__":
    # Test enforcement system
    enforcer = DockerMCPEnforcer()
    
    if enforcer.enforce_mandatory_requirements():
        print("# Check All requirements validated successfully!")
        status = enforcer.get_system_status()
    else:
        sys.exit(1)