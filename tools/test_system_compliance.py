#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - System Compliance Test Suite
Tests that all microservices architecture rules are being followed

Usage:
    python test_system_compliance.py
"""

import os
import json
import yaml
import requests
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

class SystemComplianceTester:
    """Test suite for microservices architecture compliance"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.services_dir = self.project_root / "services"
        self.test_results = {}

    def run_all_tests(self):
        """Run all compliance tests"""
        print("🚀 VIPER System Compliance Test Suite")
        print("=" * 50)

        tests = [
            self.test_docker_configuration,
            self.test_service_structure,
            self.test_health_endpoints,
            self.test_environment_variables,
            self.test_service_discovery,
            self.test_logging_configuration,
            self.test_circuit_breaker_implementation,
            self.test_dependency_management
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            try:
                result = test()
                if result:
                    passed += 1
                    print(f"✅ {test.__name__}")
                else:
                    print(f"❌ {test.__name__}")
            except Exception as e:
                print(f"❌ {test.__name__}: {e}")

        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All microservices architecture rules are being followed!")
            return True
        else:
            print("⚠️  Some rules are not being followed. See above for details.")
            return False

    def test_docker_configuration(self) -> bool:
        """Test Docker configuration compliance"""
        docker_compose_path = self.project_root / "docker-compose.yml"

        if not docker_compose_path.exists():
            print("  ❌ docker-compose.yml not found")
            return False

        try:
            with open(docker_compose_path, 'r') as f:
                compose_config = yaml.safe_load(f)

            # Check for deprecated version
            if 'version' in compose_config:
                print("  ❌ Deprecated 'version' field found in docker-compose.yml")
                return False

            # Check for services
            if 'services' not in compose_config:
                print("  ❌ No services defined in docker-compose.yml")
                return False

            services = compose_config['services']
            required_services = ['api-server', 'ultra-backtester', 'strategy-optimizer',
                               'live-trading-engine', 'data-manager', 'exchange-connector',
                               'risk-manager', 'monitoring-service', 'redis', 'prometheus', 'grafana']

            missing_services = []
            for service in required_services:
                if service not in services:
                    missing_services.append(service)

            if missing_services:
                print(f"  ❌ Missing services: {missing_services}")
                return False

            # Check for health checks
            services_without_health = []
            for service_name, service_config in services.items():
                if service_name in ['redis', 'prometheus', 'grafana']:
                    continue  # Infrastructure services
                if 'healthcheck' not in service_config:
                    services_without_health.append(service_name)

            if services_without_health:
                print(f"  ❌ Services without health checks: {services_without_health}")
                return False

            return True

        except Exception as e:
            print(f"  ❌ Error reading docker-compose.yml: {e}")
            return False

    def test_service_structure(self) -> bool:
        """Test service structure compliance"""
        if not self.services_dir.exists():
            print("  ❌ services directory not found")
            return False

        required_services = ['api-server', 'ultra-backtester', 'strategy-optimizer',
                           'live-trading-engine', 'data-manager', 'exchange-connector',
                           'risk-manager', 'monitoring-service']

        missing_services = []
        for service in required_services:
            service_dir = self.services_dir / service
            if not service_dir.exists():
                missing_services.append(service)

        if missing_services:
            print(f"  ❌ Missing service directories: {missing_services}")
            return False

        # Check for required files in each service
        required_files = ['main.py', 'Dockerfile', 'requirements.txt']
        services_missing_files = []

        for service in required_services:
            service_dir = self.services_dir / service
            missing_files = []
            for req_file in required_files:
                if not (service_dir / req_file).exists():
                    missing_files.append(req_file)

            if missing_files:
                services_missing_files.append(f"{service}: {missing_files}")

        if services_missing_files:
            print(f"  ❌ Services missing required files: {services_missing_files}")
            return False

        return True

    def test_health_endpoints(self) -> bool:
        """Test health endpoint compliance"""
        services_with_health = 0
        total_services = 0

        for service_dir in self.services_dir.iterdir():
            if not service_dir.is_dir():
                continue

            main_py = service_dir / "main.py"
            if not main_py.exists():
                continue

            total_services += 1

            try:
                with open(main_py, 'r') as f:
                    content = f.read()
                    if '@app.get("/health")' in content or 'def health_check' in content:
                        services_with_health += 1
            except Exception as e:
                print(f"  ❌ Error reading {main_py}: {e}")
                return False

        if total_services == 0:
            print("  ❌ No services found")
            return False

        if services_with_health != total_services:
            print(f"  ❌ Only {services_with_health}/{total_services} services have health endpoints")
            return False

        return True

    def test_environment_variables(self) -> bool:
        """Test environment variable standardization"""
        docker_compose_path = self.project_root / "docker-compose.yml"

        try:
            with open(docker_compose_path, 'r') as f:
                compose_config = yaml.safe_load(f)

            services = compose_config['services']

            # Check standard environment variables
            required_env_vars = ['REDIS_URL', 'LOG_LEVEL', 'SERVICE_NAME']
            services_missing_env = []

            for service_name, service_config in services.items():
                if service_name in ['redis', 'prometheus', 'grafana']:
                    continue  # Infrastructure services

                env_vars = service_config.get('environment', [])
                missing_vars = []

                for req_var in required_env_vars:
                    if not any(req_var in str(env_var) for env_var in env_vars):
                        missing_vars.append(req_var)

                if missing_vars:
                    services_missing_env.append(f"{service_name}: {missing_vars}")

            if services_missing_env:
                print(f"  ❌ Services missing standard env vars: {services_missing_env}")
                return False

            return True

        except Exception as e:
            print(f"  ❌ Error checking environment variables: {e}")
            return False

    def test_service_discovery(self) -> bool:
        """Test service discovery compliance"""
        # Check if services are using proper DNS names instead of localhost
        issues_found = []

        for service_dir in self.services_dir.iterdir():
            if not service_dir.is_dir():
                continue

            main_py = service_dir / "main.py"
            if not main_py.exists():
                continue

            try:
                with open(main_py, 'r') as f:
                    content = f.read()

                    # Check for localhost usage (which is bad in Docker)
                    if 'localhost' in content and 'http://localhost' in content:
                        issues_found.append(f"{service_dir.name}: uses localhost")

                    # Check for hardcoded IP addresses
                    import re
                    ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
                    if re.search(ip_pattern, content):
                        issues_found.append(f"{service_dir.name}: uses hardcoded IPs")

            except Exception as e:
                print(f"  ❌ Error reading {main_py}: {e}")
                return False

        if issues_found:
            print(f"  ❌ Service discovery issues: {issues_found}")
            return False

        return True

    def test_logging_configuration(self) -> bool:
        """Test logging configuration standardization"""
        issues_found = []

        for service_dir in self.services_dir.iterdir():
            if not service_dir.is_dir():
                continue

            main_py = service_dir / "main.py"
            if not main_py.exists():
                continue

            try:
                with open(main_py, 'r') as f:
                    content = f.read()

                    # Check for LOG_LEVEL environment variable usage
                    if 'LOG_LEVEL' not in content:
                        issues_found.append(f"{service_dir.name}: not using LOG_LEVEL env var")

                    # Check for consistent logging format
                    if 'getattr(logging' not in content:
                        issues_found.append(f"{service_dir.name}: not using dynamic log level")

            except Exception as e:
                print(f"  ❌ Error reading {main_py}: {e}")
                return False

        if issues_found:
            print(f"  ❌ Logging configuration issues: {issues_found}")
            return False

        return True

    def test_circuit_breaker_implementation(self) -> bool:
        """Test circuit breaker implementation"""
        shared_dir = self.services_dir / "shared"
        circuit_breaker_file = shared_dir / "circuit_breaker.py"

        if not circuit_breaker_file.exists():
            print("  ❌ Circuit breaker implementation not found")
            return False

        # Check if services are importing circuit breaker
        services_using_circuit_breaker = 0
        total_services = 0

        for service_dir in self.services_dir.iterdir():
            if not service_dir.is_dir() or service_dir.name == 'shared':
                continue

            main_py = service_dir / "main.py"
            if not main_py.exists():
                continue

            total_services += 1

            try:
                with open(main_py, 'r') as f:
                    content = f.read()
                    if 'circuit_breaker' in content or 'call_service' in content:
                        services_using_circuit_breaker += 1
            except Exception as e:
                print(f"  ❌ Error reading {main_py}: {e}")
                return False

        if services_using_circuit_breaker == 0:
            print("  ❌ No services are using circuit breaker pattern")
            return False

        print(f"  ✅ {services_using_circuit_breaker}/{total_services} services using circuit breaker")
        return True

    def test_dependency_management(self) -> bool:
        """Test Docker dependency management"""
        docker_compose_path = self.project_root / "docker-compose.yml"

        try:
            with open(docker_compose_path, 'r') as f:
                compose_config = yaml.safe_load(f)

            services = compose_config['services']

            # Check for proper dependency chains
            services_without_depends_on = []
            services_with_condition = 0

            for service_name, service_config in services.items():
                if service_name in ['redis', 'prometheus', 'grafana']:
                    continue  # Infrastructure services don't depend on others

                depends_on = service_config.get('depends_on', {})

                if not depends_on:
                    services_without_depends_on.append(service_name)
                else:
                    # Check for health check conditions
                    if isinstance(depends_on, dict):
                        for dep_name, dep_config in depends_on.items():
                            if isinstance(dep_config, dict) and 'condition' in dep_config:
                                services_with_condition += 1

            if services_without_depends_on:
                print(f"  ❌ Services without dependencies: {services_without_depends_on}")
                return False

            if services_with_condition == 0:
                print("  ❌ No services using health check conditions")
                return False

            return True

        except Exception as e:
            print(f"  ❌ Error checking dependency management: {e}")
            return False

def main():
    """Main test runner"""
    tester = SystemComplianceTester()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 SYSTEM COMPLIANCE VERIFICATION PASSED!")
        print("✅ All microservices architecture rules are being followed")
        print("🚀 System is ready for live trading deployment")
    else:
        print("\n⚠️  SYSTEM COMPLIANCE VERIFICATION FAILED!")
        print("❌ Some microservices architecture rules are not being followed")
        print("🔧 Please fix the issues above before deploying to production")

    return success

if __name__ == "__main__":
    main()
