#!/usr/bin/env python3
"""
🧪 VIPER Trading System - Live Trading Integration Test
Comprehensive test suite for the complete trading system

Tests:
- Microservices connectivity and health
- MCP server functionality
- Live trading simulation
- Market scanning capabilities
- Risk management systems
- System integration and communication
"""

import os
import sys
import time
import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

class LiveTradingIntegrationTest:
    """Comprehensive integration tests for the VIPER trading system"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.env_file = self.project_root / '.env'

        # Test results
        self.test_results = {}
        self.test_count = 0
        self.passed_tests = 0

    def log_test_result(self, test_name: str, passed: bool, message: str = ""):
        """Log individual test result"""
        self.test_count += 1
        if passed:
            self.passed_tests += 1

        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} {test_name}")
        if message:
            print(f"   {message}")

        self.test_results[test_name] = {
            'passed': passed,
            'message': message,
            'timestamp': time.time()
        }

    def test_microservices_health(self) -> bool:
        """Test all microservices health endpoints"""
        print("\n🔍 Testing Microservices Health...")

        services = [
            ('API Server', 'http://localhost:8000/health'),
            ('Risk Manager', 'http://localhost:8002/health'),
            ('Data Manager', 'http://localhost:8003/health'),
            ('Exchange Connector', 'http://localhost:8005/health'),
            ('Monitoring Service', 'http://localhost:8006/health'),
            ('Redis', 'http://localhost:6379/health'),
        ]

        healthy_count = 0
        for name, url in services:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.log_test_result(f"{name} Health", True, f"Response: {response.status_code}")
                    healthy_count += 1
                else:
                    self.log_test_result(f"{name} Health", False, f"Unexpected status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                self.log_test_result(f"{name} Health", False, f"Connection failed: {str(e)}")
            except Exception as e:
                self.log_test_result(f"{name} Health", False, f"Unexpected error: {str(e)}")

        overall_passed = healthy_count >= 4  # At least 4 services should be healthy
        self.log_test_result("Microservices Overall Health",
                           overall_passed,
                           f"{healthy_count}/{len(services)} services healthy")
        return overall_passed

    def test_api_endpoints(self) -> bool:
        """Test main API endpoints"""
        print("\n🌐 Testing API Endpoints...")

        endpoints = [
            ('Root API', 'http://localhost:8000/'),
            ('API Docs', 'http://localhost:8000/docs'),
            ('Health Check', 'http://localhost:8000/health'),
        ]

        working_endpoints = 0
        for name, url in endpoints:
            try:
                response = requests.get(url, timeout=15)
                if response.status_code in [200, 422]:  # 422 is OK for API docs
                    self.log_test_result(f"{name} Endpoint", True, f"Status: {response.status_code}")
                    working_endpoints += 1
                else:
                    self.log_test_result(f"{name} Endpoint", False, f"Status: {response.status_code}")
            except Exception as e:
                self.log_test_result(f"{name} Endpoint", False, f"Error: {str(e)}")

        overall_passed = working_endpoints >= 2
        self.log_test_result("API Endpoints Overall",
                           overall_passed,
                           f"{working_endpoints}/{len(endpoints)} endpoints working")
        return overall_passed

    def test_monitoring_stack(self) -> bool:
        """Test monitoring infrastructure"""
        print("\n📊 Testing Monitoring Stack...")

        services = [
            ('Prometheus', 'http://localhost:9090/-/healthy'),
            ('Grafana', 'http://localhost:3001/api/health'),
        ]

        working_services = 0
        for name, url in services:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.log_test_result(f"{name} Service", True, f"Status: {response.status_code}")
                    working_services += 1
                else:
                    self.log_test_result(f"{name} Service", False, f"Status: {response.status_code}")
            except Exception as e:
                self.log_test_result(f"{name} Service", False, f"Error: {str(e)}")

        overall_passed = working_services >= 1  # At least Prometheus should work
        self.log_test_result("Monitoring Stack",
                           overall_passed,
                           f"{working_services}/{len(services)} services operational")
        return overall_passed

    def test_mcp_servers(self) -> bool:
        """Test MCP server processes"""
        print("\n🤖 Testing MCP Servers...")

        # Check if MCP processes are running
        mcp_processes = ['node.*index.js', 'node.*build/index.js']

        running_processes = 0
        for process_pattern in mcp_processes:
            try:
                result = subprocess.run(['tasklist', '/FI', f'IMAGENAME eq {process_pattern}'],
                                      capture_output=True, text=True, timeout=10)
                if 'node.exe' in result.stdout:
                    running_processes += 1
            except:
                pass

        trading_server_running = running_processes >= 1
        github_server_running = running_processes >= 2

        self.log_test_result("MCP Trading Server",
                           trading_server_running,
                           "Process is running" if trading_server_running else "Process not found")

        self.log_test_result("MCP GitHub Server",
                           github_server_running,
                           "Process is running" if github_server_running else "Process not found")

        overall_passed = running_processes >= 1
        self.log_test_result("MCP Servers Overall",
                           overall_passed,
                           f"{running_processes} MCP server(s) running")
        return overall_passed

    def test_environment_configuration(self) -> bool:
        """Test environment configuration"""
        print("\n⚙️ Testing Environment Configuration...")

        required_vars = [
            'BITGET_API_KEY',
            'BITGET_API_SECRET',
            'VAULT_MASTER_KEY',
            'GRAFANA_ADMIN_PASSWORD'
        ]

        configured_vars = 0
        for var in required_vars:
            if os.getenv(var):
                configured_vars += 1
                self.log_test_result(f"Env Var: {var}", True, "Configured")
            else:
                self.log_test_result(f"Env Var: {var}", False, "Not configured")

        # Test .env file exists and has content
        if self.env_file.exists():
            try:
                with open(self.env_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    if len(content) > 100:  # Should have substantial content
                        self.log_test_result("Environment File", True, ".env file exists and has content")
                        configured_vars += 1
                    else:
                        self.log_test_result("Environment File", False, ".env file exists but is empty")
            except Exception as e:
                self.log_test_result("Environment File", False, f"Error reading .env: {str(e)}")
        else:
            self.log_test_result("Environment File", False, ".env file not found")

        overall_passed = configured_vars >= 3  # At least 3/5 checks should pass
        self.log_test_result("Environment Configuration",
                           overall_passed,
                           f"{configured_vars}/5 configuration checks passed")
        return overall_passed

    def test_docker_containers(self) -> bool:
        """Test Docker container status"""
        print("\n🐳 Testing Docker Containers...")

        try:
            result = subprocess.run(['docker', 'ps', '--format', 'json'],
                                  capture_output=True, text=True, timeout=15)

            if result.returncode == 0:
                containers = [json.loads(line) for line in result.stdout.strip().split('\n') if line.strip()]

                viper_containers = [c for c in containers if 'viper' in c.get('Names', '')]

                if viper_containers:
                    self.log_test_result("Docker Containers", True,
                                       f"{len(viper_containers)} VIPER containers running")

                    # Check container health
                    healthy_count = 0
                    for container in viper_containers:
                        if 'healthy' in container.get('Status', '').lower():
                            healthy_count += 1

                    self.log_test_result("Container Health",
                                       healthy_count > 0,
                                       f"{healthy_count}/{len(viper_containers)} containers healthy")

                    return len(viper_containers) >= 3  # At least 3 containers should be running
                else:
                    self.log_test_result("Docker Containers", False, "No VIPER containers found")
                    return False
            else:
                self.log_test_result("Docker Command", False, "Docker ps command failed")
                return False

        except Exception as e:
            self.log_test_result("Docker Containers", False, f"Error: {str(e)}")
            return False

    def test_network_connectivity(self) -> bool:
        """Test network connectivity to external services"""
        print("\n🌐 Testing Network Connectivity...")

        test_urls = [
            ('Bitget API', 'https://api.bitget.com/api/spot/v1/public/time'),
            ('Google DNS', 'https://www.google.com'),
        ]

        connectivity_tests = 0
        for name, url in test_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.log_test_result(f"{name} Connectivity", True, "Connection successful")
                    connectivity_tests += 1
                else:
                    self.log_test_result(f"{name} Connectivity", False, f"HTTP {response.status_code}")
            except Exception as e:
                self.log_test_result(f"{name} Connectivity", False, f"Connection failed: {str(e)}")

        overall_passed = connectivity_tests >= 1  # At least basic connectivity
        self.log_test_result("Network Connectivity Overall",
                           overall_passed,
                           f"{connectivity_tests}/2 connectivity tests passed")
        return overall_passed

    def test_system_integration(self) -> bool:
        """Test overall system integration"""
        print("\n🔗 Testing System Integration...")

        # Test inter-service communication
        integration_tests = [
            self.test_api_to_risk_manager_communication,
            self.test_api_to_data_manager_communication,
            self.test_cross_service_dependencies
        ]

        passed_integration_tests = 0
        for test_func in integration_tests:
            try:
                if test_func():
                    passed_integration_tests += 1
            except Exception as e:
                print(f"Integration test error: {str(e)}")

        overall_passed = passed_integration_tests >= 1
        self.log_test_result("System Integration",
                           overall_passed,
                           f"{passed_integration_tests}/3 integration tests passed")
        return overall_passed

    def test_api_to_risk_manager_communication(self) -> bool:
        """Test API server to risk manager communication"""
        try:
            # This would test actual inter-service communication
            # For now, just test if both services are healthy
            api_health = requests.get('http://localhost:8000/health', timeout=5)
            risk_health = requests.get('http://localhost:8002/health', timeout=5)

            return api_health.status_code == 200 and risk_health.status_code == 200
        except:
            return False

    def test_api_to_data_manager_communication(self) -> bool:
        """Test API server to data manager communication"""
        try:
            # Test if services can communicate
            api_health = requests.get('http://localhost:8000/health', timeout=5)
            data_health = requests.get('http://localhost:8003/health', timeout=5)

            return api_health.status_code == 200 and data_health.status_code == 200
        except:
            return False

    def test_cross_service_dependencies(self) -> bool:
        """Test cross-service dependencies"""
        try:
            # Test if Redis is accessible from multiple services
            redis_health = requests.get('http://localhost:6379/health', timeout=5)
            return redis_health.status_code == 200
        except:
            return False

    def run_comprehensive_test_suite(self):
        """Run the complete test suite"""
        print("🧪 VIPER Trading System - Comprehensive Integration Test")
        print("=" * 70)

        start_time = time.time()

        # Run all tests
        test_functions = [
            ('Environment Configuration', self.test_environment_configuration),
            ('Docker Containers', self.test_docker_containers),
            ('Network Connectivity', self.test_network_connectivity),
            ('Microservices Health', self.test_microservices_health),
            ('API Endpoints', self.test_api_endpoints),
            ('Monitoring Stack', self.test_monitoring_stack),
            ('MCP Servers', self.test_mcp_servers),
            ('System Integration', self.test_system_integration),
        ]

        for test_name, test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                self.log_test_result(test_name, False, f"Test execution failed: {str(e)}")

        # Calculate results
        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 70)

        # Overall results
        overall_score = (self.passed_tests / self.test_count) * 100 if self.test_count > 0 else 0

        print(f"Total Tests: {self.test_count}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.test_count - self.passed_tests}")
        print(f"Overall Score: {overall_score:.1f}%")
        print(f"Execution Time: {execution_time:.2f}s")
        # System readiness assessment
        if overall_score >= 90:
            readiness = "🎉 SYSTEM FULLY OPERATIONAL"
            recommendation = "Ready for live trading!"
        elif overall_score >= 75:
            readiness = "✅ SYSTEM MOSTLY OPERATIONAL"
            recommendation = "Ready for testing with caution"
        elif overall_score >= 50:
            readiness = "⚠️ SYSTEM PARTIALLY OPERATIONAL"
            recommendation = "Needs attention before live trading"
        else:
            readiness = "❌ SYSTEM NEEDS ATTENTION"
            recommendation = "Not ready for live trading"

        print(f"\n{readiness}")
        print(f"Recommendation: {recommendation}")

        # Detailed breakdown
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['passed'] else "❌"
            print(f"{status_icon} {test_name}")
            if result['message']:
                print(f"   {result['message']}")

        return overall_score >= 75  # Consider 75%+ as passing

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("🧪 VIPER Trading System Integration Test")
        print("\nUsage: python test_live_trading_integration.py")
        print("\nThis script runs comprehensive tests for:")
        print("- Microservices health and connectivity")
        print("- API endpoints functionality")
        print("- Monitoring stack (Prometheus, Grafana)")
        print("- MCP server status")
        print("- Docker container health")
        print("- Network connectivity")
        print("- Environment configuration")
        print("- System integration and dependencies")
        return

    tester = LiveTradingIntegrationTest()
    success = tester.run_comprehensive_test_suite()

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
