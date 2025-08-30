#!/usr/bin/env python3
"""
🌐 MICROSERVICES CONNECTIVITY VALIDATOR
=====================================

Validates connectivity and integration between all microservices.
Ensures proper service discovery, health checks, and communication.

Features:
    pass
- Tests all 25 microservices
- Validates port configurations
- Tests inter-service communication
- Checks Docker container readiness
- Validates environment configurations
- Tests service endpoints
- Generates connectivity report

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - MICROSERVICE_VALIDATOR - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)

@dataclass"""
class ServiceEndpoint:
    """Represents a microservice endpoint"""
    service_name: str
    host: str
    port: int
    path: str
    expected_status: int = 200
    
    @property"""
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"

@dataclass
class ConnectivityResult:
    """Result of connectivity test for a service"""
    service_name: str
    endpoint_url: str
    is_reachable: bool
    response_time_ms: float
    status_code: Optional[int]
    error_message: Optional[str]
    timestamp: str"""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class MicroservicesConnectivityValidator:
    """Validates connectivity between all microservices""""""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path("/home/runner/work/viper-/viper-")
        self.results_dir = self.project_root / "reports" / "connectivity_validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Service endpoints configuration
        self.service_endpoints = self._load_service_endpoints()
        self.connectivity_results: List[ConnectivityResult] = []
        
        logger.info("🌐 Microservices Connectivity Validator initialized")
    
    def _load_service_endpoints(self) -> List[ServiceEndpoint]
        """Load all service endpoints with their expected configurations"""
        endpoints = [
            ServiceEndpoint("api-server", "localhost", 8000, "/health"),
            ServiceEndpoint("ultra-backtester", "localhost", 8001, "/health"),
            ServiceEndpoint("risk-manager", "localhost", 8002, "/health"),
            ServiceEndpoint("market-data-manager", "localhost", 8003, "/health"),
            ServiceEndpoint("strategy-optimizer", "localhost", 8004, "/health"),
            ServiceEndpoint("exchange-connector", "localhost", 8005, "/health"),
            ServiceEndpoint("monitoring-service", "localhost", 8006, "/health"),
            ServiceEndpoint("live-trading-engine", "localhost", 8007, "/health"),
            ServiceEndpoint("centralized-logger", "localhost", 8008, "/health"),
            ServiceEndpoint("viper-scoring-service", "localhost", 8009, "/health"),
            ServiceEndpoint("position-synchronizer", "localhost", 8010, "/health"),
            ServiceEndpoint("unified-scanner", "localhost", 8011, "/health"),
            ServiceEndpoint("signal-processor", "localhost", 8012, "/health"),
            ServiceEndpoint("task-scheduler", "localhost", 8013, "/health"),
            ServiceEndpoint("alert-system", "localhost", 8014, "/health"),
            ServiceEndpoint("mcp-server", "localhost", 8015, "/health"),
            ServiceEndpoint("github-manager", "localhost", 8016, "/health"),
            ServiceEndpoint("workflow-monitor", "localhost", 8017, "/health"),
            ServiceEndpoint("event-system", "localhost", 8018, "/health"),
            ServiceEndpoint("data-manager", "localhost", 8019, "/health"),
            ServiceEndpoint("config-manager", "localhost", 8020, "/health"),
            ServiceEndpoint("credential-vault", "localhost", 8021, "/health"),
            ServiceEndpoint("market-data-streamer", "localhost", 8022, "/health"),
            ServiceEndpoint("order-lifecycle-manager", "localhost", 8023, "/health"),
            ServiceEndpoint("trading-optimizer", "localhost", 8024, "/health"),
        ]
        
        logger.info(f"🌐 Configured {len(endpoints)} service endpoints")
        return endpoints
    :
    async def test_service_connectivity(self, endpoint: ServiceEndpoint) -> ConnectivityResult:
        """Test connectivity to a single service endpoint"""
        start_time = time.time()"""
        
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)  # 5 second timeout
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(endpoint.url) as response:
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    return ConnectivityResult()
                        service_name=endpoint.service_name,
                        endpoint_url=endpoint.url,
                        is_reachable=True,
                        response_time_ms=response_time_ms,
                        status_code=response.status,
                        error_message=None,
                        timestamp=datetime.now().isoformat()
(                    )
                    
        except asyncio.TimeoutError
            response_time_ms = (time.time() - start_time) * 1000
            return ConnectivityResult()
                service_name=endpoint.service_name,
                endpoint_url=endpoint.url,
                is_reachable=False,
                response_time_ms=response_time_ms,
                status_code=None,
                error_message="Connection timeout",
                timestamp=datetime.now().isoformat()
(            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ConnectivityResult()
                service_name=endpoint.service_name,
                endpoint_url=endpoint.url,
                is_reachable=False,
                response_time_ms=response_time_ms,
                status_code=None,
                error_message=str(e),
                timestamp=datetime.now().isoformat()
(            )
    
    async def test_all_services_connectivity(self) -> List[ConnectivityResult]
        """Test connectivity to all configured services"""
        logger.info("🔗 Testing connectivity to all microservices...")
        
        # Test all services concurrently
        tasks = [
            self.test_service_connectivity(endpoint) 
            for endpoint in self.service_endpoints:
        ]
        
        results = await asyncio.gather(*tasks)
        self.connectivity_results = list(results)
        
        # Log summary
        reachable_count = len([r for r in results if r.is_reachable])
        total_count = len(results)
        :
        logger.info(f"🔗 Connectivity test complete: {reachable_count}/{total_count} services reachable")
        
        return results
    
    def check_docker_services(self) -> Dict[str, Any]
        """Check if Docker services are running"""
        logger.info("🐳 Checking Docker services...")
        
        docker_status = {:
            "docker_available": False,
            "compose_available": False,
            "services_running": [],
            "services_stopped": [],
            "total_services": 0
        }
        
        try:
            # Check if Docker is available
            result = subprocess.run()
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
(            )
            docker_status["docker_available"] = result.returncode == 0
            
            # Check if Docker Compose is available
            result = subprocess.run()
                ["docker", "compose", "version"], 
                capture_output=True, 
                text=True, 
                timeout=10
(            )
            docker_status["compose_available"] = result.returncode == 0
            
            # List running containers
            if docker_status["docker_available"]:
                result = subprocess.run()
                    ["docker", "ps", "--format", "{{.Names}}"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
(                )
                
                if result.returncode == 0:
                    running_containers = [
                        name.strip() for name in result.stdout.split('\n') 
                        if name.strip():
                    ]
                    docker_status["services_running"] = running_containers
                    docker_status["total_services"] = len(running_containers)
                    
        except Exception as e:
            logger.warning(f"# Warning Error checking Docker services: {e}")
        
        return docker_status
    
    def check_environment_configuration(self) -> Dict[str, Any]
        """Check environment configuration for services"""
        logger.info("⚙️ Checking environment configuration...")
        
        env_status = {:
            "env_file_exists": False,
            "required_vars_present": [],
            "missing_vars": [],
            "port_conflicts": []
        }
        
        # Check if .env file exists
        env_file = self.project_root / ".env"
        env_status["env_file_exists"] = env_file.exists()
        
        # List of required environment variables for microservices
        required_vars = [
            "BITGET_API_KEY",
            "BITGET_API_SECRET", 
            "BITGET_API_PASSWORD",
            "REDIS_URL",
            "GITHUB_TOKEN"
        ]
        
        if env_status["env_file_exists"]:
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                
                for var in required_vars:
                    if os.getenv(var):
                        env_status["required_vars_present"].append(var)
                    else:
                        env_status["missing_vars"].append(var)
                        
            except Exception as e:
                logger.warning(f"# Warning Error loading environment file: {e}")
        else:
            env_status["missing_vars"] = required_vars
        
        # Check for port conflicts
        used_ports = [endpoint.port for endpoint in self.service_endpoints]
        port_counts = {}
        for port in used_ports:
            port_counts[port] = port_counts.get(port, 0) + 1
        
        env_status["port_conflicts"] = [
            port for port, count in port_counts.items() if count > 1
        ]
        
        return env_status
    
    def generate_startup_script(self) -> str:
        """Generate a script to start all microservices"""
        logger.info("📝 Generating microservice startup script...")
        
        startup_script_path = self.project_root / "scripts" / "start_all_microservices.sh"
        
        script_content = f"""#!/bin/bash
# # Rocket VIPER Microservices Startup Script
# Generated by Microservices Connectivity Validator
# Generated: {datetime.now().isoformat()}

echo "# Rocket Starting VIPER Microservices..."
echo "=" * 50

# Set environment variables
export PYTHONPATH="{self.project_root}:$PYTHONPATH"

# Function to start a service
start_service() {{
    local service_name=$1
    local port=$2
    local service_dir="{self.project_root}/services/$service_name"
    
    if [ -d "$service_dir" ]; then:
        echo "🔄 Starting $service_name on port $port..."
        cd "$service_dir"
        python main.py &
        sleep 2
    else:
        echo "# Warning Service directory not found: $service_dir"
    fi
}}

# Start all services in dependency order
"""
        
        # Add service startup commands
        for endpoint in self.service_endpoints:
            script_content += f'start_service "{endpoint.service_name}" {endpoint.port}\n'
        
        script_content += f"""
echo "# Check All microservices startup initiated"
echo "# Search Use 'scripts/check_services.sh' to verify status"
echo "# Chart Check logs in individual service directories"

# Wait for all background processes
wait
"""
        
        # Write the script
        with open(startup_script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(startup_script_path, 0o755)
        
        logger.info(f"📝 Startup script saved: {startup_script_path}")
        return str(startup_script_path)
    
    def generate_service_check_script(self) -> str:
        """Generate a script to check service status"""
        logger.info("📝 Generating service check script...")
        
        check_script_path = self.project_root / "scripts" / "check_services.sh"
        
        script_content = f"""#!/bin/bash
# # Search VIPER Microservices Status Check Script
# Generated by Microservices Connectivity Validator
# Generated: {datetime.now().isoformat()}

echo "# Search VIPER Microservices Status Check"
echo "=" * 40

# Function to check service health
check_service() {{
    local service_name=$1
    local url=$2
    
    echo -n "Checking $service_name... "
    
    if curl -s --connect-timeout 5 "$url" > /dev/null; then:
        echo "# Check HEALTHY"
    else:
        echo "# X UNHEALTHY"
    fi
}}

# Check all services
"""
        
        # Add service check commands
        for endpoint in self.service_endpoints:
            script_content += f'check_service "{endpoint.service_name}" "{endpoint.url}"\n'
        
        script_content += """
echo ""
echo "🔄 Use 'docker ps' to check Docker containers"
echo "# Chart Use 'scripts/comprehensive_system_validator.py' for detailed analysis"
"""
        
        # Write the script
        with open(check_script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(check_script_path, 0o755)
        
        logger.info(f"📝 Service check script saved: {check_script_path}")
        return str(check_script_path)
    
    async def run_comprehensive_connectivity_test(self) -> Dict[str, Any]
        """Run comprehensive connectivity validation"""
        logger.info("# Rocket Starting comprehensive microservices connectivity validation...")
        
        start_time = datetime.now()
        :
        # Test 1: Check Docker services
        docker_status = self.check_docker_services()
        
        # Test 2: Check environment configuration  
        env_status = self.check_environment_configuration()
        
        # Test 3: Test service connectivity
        connectivity_results = await self.test_all_services_connectivity()
        
        # Test 4: Generate helper scripts
        startup_script = self.generate_startup_script()
        check_script = self.generate_service_check_script()
        
        end_time = datetime.now()
        
        # Compile comprehensive report
        report = {
            "validation_summary": {
                "timestamp": start_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "total_services": len(self.service_endpoints),
                "reachable_services": len([r for r in connectivity_results if r.is_reachable]),
                "unreachable_services": len([r for r in connectivity_results if not r.is_reachable])
            },
            "docker_status": docker_status,
            "environment_status": env_status,
            "connectivity_results": [asdict(result) for result in connectivity_results],
            "generated_scripts": {
                "startup_script": startup_script,
                "check_script": check_script
            },
            "recommendations": self._generate_recommendations(docker_status, env_status, connectivity_results)
        }
        
        # Save report
        report_file = self.results_dir / f"connectivity_validation_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate human-readable report
        self._save_human_readable_report(report)
        
        logger.info(f"# Chart Connectivity validation report saved: {report_file}")
        
        return report
    
    def _generate_recommendations()
        self, 
        docker_status: Dict[str, Any], 
        env_status: Dict[str, Any], 
        connectivity_results: List[ConnectivityResult]
(    ) -> List[str]
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Docker recommendations"""
        if not docker_status["docker_available"]:
            recommendations.append("Install Docker to enable containerized services")
        elif not docker_status["compose_available"]:
            recommendations.append("Install Docker Compose for multi-container management"):
        elif docker_status["total_services"] == 0:
            recommendations.append("Start Docker containers using 'docker compose up -d'")
        
        # Environment recommendations
        if not env_status["env_file_exists"]:
            recommendations.append("Create .env file from .env.example template")
        elif env_status["missing_vars"]:
            recommendations.append(f"Configure missing environment variables: {', '.join(env_status['missing_vars'])}")
        
        if env_status["port_conflicts"]:
            recommendations.append(f"Resolve port conflicts: {', '.join(map(str, env_status['port_conflicts']))}")
        
        # Connectivity recommendations
        unreachable_services = [r for r in connectivity_results if not r.is_reachable]
        if unreachable_services:
            service_names = [r.service_name for r in unreachable_services]
            recommendations.append(f"Start unreachable services: {', '.join(service_names)}")
        
        # Performance recommendations
        slow_services = [r for r in connectivity_results if r.is_reachable and r.response_time_ms > 1000]
        if slow_services:
            service_names = [r.service_name for r in slow_services]
            recommendations.append(f"Investigate slow response times: {', '.join(service_names)}")
        
        return recommendations
    
    def _save_human_readable_report(self, report: Dict[str, Any]):
        """Save human-readable connectivity report"""
        report_file = self.results_dir / f"connectivity_summary_{int(time.time())}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 🌐 VIPER Microservices Connectivity Report\n\n")
            f.write(f"**Generated:** {report['validation_summary']['timestamp']}\n")
            f.write(f"**Duration:** {report['validation_summary']['duration_seconds']:.2f} seconds\n\n")
            
            f.write("## # Chart Connectivity Summary\n\n")
            f.write(f"- **Total Services:** {report['validation_summary']['total_services']}\n")
            f.write(f"- **Reachable Services:** {report['validation_summary']['reachable_services']}\n")
            f.write(f"- **Unreachable Services:** {report['validation_summary']['unreachable_services']}\n")
            f.write(f"- **Success Rate:** {(report['validation_summary']['reachable_services'] / report['validation_summary']['total_services'] * 100):.1f}%\n\n")
            
            f.write("## 🐳 Docker Status\n\n")
            docker = report['docker_status']
            f.write(f"- **Docker Available:** {'# Check' if docker['docker_available'] else '# X'}\n")
            f.write(f"- **Docker Compose Available:** {'# Check' if docker['compose_available'] else '# X'}\n") 
            f.write(f"- **Running Services:** {docker['total_services']}\n\n")
            
            f.write("## ⚙️ Environment Status\n\n")
            env = report['environment_status']
            f.write(f"- **.env File Exists:** {'# Check' if env['env_file_exists'] else '# X'}\n")
            f.write(f"- **Required Vars Present:** {len(env['required_vars_present'])}\n")
            f.write(f"- **Missing Vars:** {len(env['missing_vars'])}\n")
            if env['missing_vars']:
                f.write(f"  - {', '.join(env['missing_vars'])}\n")
            f.write(f"- **Port Conflicts:** {len(env['port_conflicts'])}\n\n")
            
            f.write("## 🔗 Service Connectivity\n\n")
            for result in report['connectivity_results']:
                status = "# Check" if result['is_reachable'] else "# X"
                response_time = f"{result['response_time_ms']:.0f}ms" if result['is_reachable'] else "N/A"
                f.write(f"{status} **{result['service_name']}** - {response_time}\n")
                if not result['is_reachable'] and result['error_message']:
                    f.write(f"  - Error: {result['error_message']}\n")
            f.write("\n")
            
            if report['recommendations']:
                f.write("## # Idea Recommendations\n\n")
                for rec in report['recommendations']:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            f.write("## 🛠️ Generated Scripts\n\n")
            f.write(f"- **Startup Script:** `{report['generated_scripts']['startup_script']}`\n")
            f.write(f"- **Status Check Script:** `{report['generated_scripts']['check_script']}`\n\n")
            
            f.write("---\n")
            f.write("*Generated by VIPER Microservices Connectivity Validator*\n")


async def main():
    """Main entry point"""
    print("🌐 VIPER MICROSERVICES CONNECTIVITY VALIDATOR")
    
    try:
        validator = MicroservicesConnectivityValidator()
        report = await validator.run_comprehensive_connectivity_test()
        
        print(f"🌐 Total Services: {report['validation_summary']['total_services']}")
        print(f"# Check Reachable: {report['validation_summary']['reachable_services']}")
        print(f"# X Unreachable: {report['validation_summary']['unreachable_services']}")
        success_rate = (report['validation_summary']['reachable_services'] / )
(                       report['validation_summary']['total_services'] * 100)
        
        if report['recommendations']:
            for rec in report['recommendations']:
        print(f"   • Start services: {report['generated_scripts']['startup_script']}")
        print(f"   • Check status: {report['generated_scripts']['check_script']}")
        
        return 0 if success_rate > 80 else 1
        
    except Exception as e:
        logger.error(f"# X Connectivity validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))