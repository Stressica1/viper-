#!/usr/bin/env python3
"""
🚀 VIPER Trading System - MCP System Diagnostic & Connection Tool
Comprehensive system scan, diagnosis, and component connection via MCP

Features:
- MCP server integration for unified control
- Full system health assessment
- Component connectivity testing
- Pipeline diagnostics and optimization
- Real-time monitoring setup
- Automated issue detection and resolution
"""

import os
import sys
import time
import json
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import ccxt
from enum import Enum

# Load environment variables
BITGET_API_KEY = os.getenv('BITGET_API_KEY', '')
BITGET_API_SECRET = os.getenv('BITGET_API_SECRET', '')
BITGET_API_PASSWORD = os.getenv('BITGET_API_PASSWORD', '')

class ComponentStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    CONNECTING = "CONNECTING"
    UNKNOWN = "UNKNOWN"

class MCPSystemDiagnostic:
    """
    MCP-powered system diagnostic and connection tool
    """

    def __init__(self):
        """Initialize MCP system diagnostic"""
        self.mcp_server_url = "http://localhost:8015"  # Docker MCP server
        self.is_running = False
        self.diagnostic_results = {}
        self.component_status = {}

        # Service endpoints
        self.services = {
            'api-server': 'http://localhost:8000',
            'data-manager': 'http://localhost:8003',
            'exchange-connector': 'http://localhost:8005',
            'risk-manager': 'http://localhost:8002',
            'monitoring-service': 'http://localhost:8006',
            'credential-vault': 'http://localhost:8008',
            'mcp-server': 'http://localhost:8015',
            'redis': 'http://localhost:6379'
        }

        print("🔧 VIPER MCP System Diagnostic Initialized")
        print(f"📊 Services to diagnose: {len(self.services)}")

    def check_mcp_server(self) -> bool:
        """Check if MCP server is running and accessible"""
        try:
            response = requests.get(f"{self.mcp_server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def diagnose_service(self, service_name: str, endpoint: str) -> Dict[str, Any]:
        """Diagnose individual service health and connectivity"""

        result = {
            'service': service_name,
            'endpoint': endpoint,
            'status': ComponentStatus.UNKNOWN.value,
            'response_time': None,
            'error': None,
            'details': {}
        }

        try:
            start_time = time.time()
            response = requests.get(f"{endpoint}/health", timeout=10)
            response_time = time.time() - start_time

            result['response_time'] = round(response_time * 1000, 2)  # ms

            if response.status_code == 200:
                result['status'] = ComponentStatus.HEALTHY.value
                try:
                    result['details'] = response.json()
                except Exception:
                    result['details'] = {'message': 'Service responding but no JSON data'}
            elif response.status_code >= 500:
                result['status'] = ComponentStatus.DOWN.value
                result['error'] = f"Server error: {response.status_code}"
            else:
                result['status'] = ComponentStatus.DEGRADED.value
                result['error'] = f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            result['status'] = ComponentStatus.DEGRADED.value
            result['error'] = "Connection timeout"
        except requests.exceptions.ConnectionError:
            result['status'] = ComponentStatus.DOWN.value
            result['error'] = "Connection refused"
        except Exception as e:
            result['status'] = ComponentStatus.DOWN.value
            result['error'] = str(e)

        return result

    def test_service_connectivity(self, service_name: str, endpoint: str) -> Dict[str, Any]:
        """Test connectivity between services"""

        result = {
            'service': service_name,
            'connectivity_tests': {},
            'pipeline_status': 'UNKNOWN'
        }

        # Test basic connectivity
        basic_test = self.diagnose_service(service_name, endpoint)
        result['connectivity_tests']['basic'] = basic_test

        # Test inter-service communication if MCP server is available
        if self.check_mcp_server():
            try:
                # Test MCP connection to service
                mcp_payload = {
                    'service': service_name,
                    'action': 'diagnose',
                    'endpoint': endpoint
                }

                response = requests.post(
                    f"{self.mcp_server_url}/diagnose_service",
                    json=mcp_payload,
                    timeout=15
                )

                if response.status_code == 200:
                    result['connectivity_tests']['mcp_integration'] = response.json()
                    result['pipeline_status'] = 'CONNECTED'
                else:
                    result['connectivity_tests']['mcp_integration'] = {'error': f'MCP error: {response.status_code}'}
                    result['pipeline_status'] = 'MCP_ERROR'

            except Exception as e:
                result['connectivity_tests']['mcp_integration'] = {'error': str(e)}
                result['pipeline_status'] = 'MCP_ERROR'
        else:
            result['pipeline_status'] = 'MCP_UNAVAILABLE'
            result['connectivity_tests']['mcp_integration'] = {'error': 'MCP server not available'}

        return result

    def perform_full_system_scan(self) -> Dict[str, Any]:
        """Perform comprehensive system scan"""
        print("\n🚀 STARTING COMPREHENSIVE SYSTEM SCAN...")

        scan_results = {
            'timestamp': datetime.now().isoformat(),
            'scan_duration': None,
            'services_scanned': len(self.services),
            'services_healthy': 0,
            'services_degraded': 0,
            'services_down': 0,
            'overall_status': 'UNKNOWN',
            'service_details': {},
            'connectivity_matrix': {},
            'recommendations': []
        }

        start_time = time.time()

        # Scan each service
        for service_name, endpoint in self.services.items():
            service_result = self.test_service_connectivity(service_name, endpoint)
            scan_results['service_details'][service_name] = service_result

            # Count status
            basic_status = service_result['connectivity_tests']['basic']['status']
            if basic_status == ComponentStatus.HEALTHY.value:
                scan_results['services_healthy'] += 1
            elif basic_status == ComponentStatus.DEGRADED.value:
                scan_results['services_degraded'] += 1
            else:
                scan_results['services_down'] += 1

        scan_results['scan_duration'] = round(time.time() - start_time, 2)

        # Determine overall status
        if scan_results['services_down'] > 0:
            scan_results['overall_status'] = 'CRITICAL'
        elif scan_results['services_degraded'] > 0:
            scan_results['overall_status'] = 'DEGRADED'
        elif scan_results['services_healthy'] == len(self.services):
            scan_results['overall_status'] = 'HEALTHY'
        else:
            scan_results['overall_status'] = 'PARTIAL'

        # Generate recommendations
        scan_results['recommendations'] = self.generate_recommendations(scan_results)

        return scan_results

    def generate_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on scan results"""
        recommendations = []

        # Check for down services
        down_services = [
            service for service, result in scan_results['service_details'].items()
            if result['connectivity_tests']['basic']['status'] == ComponentStatus.DOWN.value
        ]

        if down_services:
            recommendations.append(f"🚨 CRITICAL: Restart these services: {', '.join(down_services)}")

        # Check for degraded services
        degraded_services = [
            service for service, result in scan_results['service_details'].items()
            if result['connectivity_tests']['basic']['status'] == ComponentStatus.DEGRADED.value
        ]

        if degraded_services:
            recommendations.append(f"⚠️ WARNING: Investigate these degraded services: {', '.join(degraded_services)}")

        # Check MCP connectivity
        mcp_issues = []
        for service, result in scan_results['service_details'].items():
            if result['pipeline_status'] in ['MCP_ERROR', 'MCP_UNAVAILABLE']:
                mcp_issues.append(service)

        if mcp_issues:
            recommendations.append(f"🔧 MCP: Fix connectivity issues for: {', '.join(mcp_issues)}")

        # Check response times
        slow_services = []
        for service, result in scan_results['service_details'].items():
            response_time = result['connectivity_tests']['basic'].get('response_time')
            if response_time and response_time > 1000:  # > 1 second
                slow_services.append(f"{service} ({response_time}ms)")

        if slow_services:
            recommendations.append(f"⚡ PERFORMANCE: Optimize these slow services: {', '.join(slow_services)}")

        # Overall recommendations
        if scan_results['overall_status'] == 'HEALTHY':
            recommendations.append("✅ SYSTEM HEALTHY: All services operational and connected")
        elif scan_results['overall_status'] == 'PARTIAL':
            recommendations.append("🔄 PARTIAL: Most services working, minor issues detected")
        else:
            recommendations.append("🚨 MAINTENANCE REQUIRED: System needs immediate attention")

        return recommendations

    def connect_remaining_components(self) -> Dict[str, Any]:
        """Connect any remaining unconnected components"""
        print("🚀 Using MCP to establish full system connectivity")

        connection_results = {
            'timestamp': datetime.now().isoformat(),
            'components_connected': 0,
            'components_failed': 0,
            'connection_details': {},
            'mcp_integration_status': 'UNKNOWN'
        }

        if not self.check_mcp_server():
            print("❌ MCP server not available for component connection")
            connection_results['mcp_integration_status'] = 'MCP_UNAVAILABLE'
            return connection_results

        print("✅ MCP server available - proceeding with component connections")

        # Connect each service via MCP
        for service_name, endpoint in self.services.items():

            try:
                # Register service with MCP server
                registration_payload = {
                    'service_name': service_name,
                    'endpoint': endpoint,
                    'service_type': 'viper_component',
                    'capabilities': ['health_check', 'diagnostics', 'monitoring']
                }

                response = requests.post(
                    f"{self.mcp_server_url}/register_service",
                    json=registration_payload,
                    timeout=15
                )

                if response.status_code == 200:
                    result = response.json()
                    connection_results['connection_details'][service_name] = {
                        'status': 'CONNECTED',
                        'details': result
                    }
                    connection_results['components_connected'] += 1
                    print(f"  ✅ {service_name} connected successfully")
                else:
                    connection_results['connection_details'][service_name] = {
                        'status': 'FAILED',
                        'error': f'HTTP {response.status_code}'
                    }
                    connection_results['components_failed'] += 1
                    print(f"  ❌ {service_name} connection failed: HTTP {response.status_code}")

            except Exception as e:
                connection_results['connection_details'][service_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                connection_results['components_failed'] += 1
                print(f"  ❌ {service_name} connection error: {e}")

        # Set overall MCP integration status
        if connection_results['components_failed'] == 0:
            connection_results['mcp_integration_status'] = 'FULLY_CONNECTED'
        elif connection_results['components_connected'] > 0:
            connection_results['mcp_integration_status'] = 'PARTIALLY_CONNECTED'
        else:
            connection_results['mcp_integration_status'] = 'CONNECTION_FAILED'

        return connection_results

    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete system diagnostic"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🚀 VIPER MCP SYSTEM DIAGNOSTIC - FULL SCAN & CONNECTION                      ║
║ 🔍 Comprehensive System Analysis | 🔗 Component Connection | 📊 Health Report ║
║ ⚡ Real-time Diagnostics | 🧠 MCP Integration | 📈 Performance Monitoring     ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)

        diagnostic_report = {
            'diagnostic_start': datetime.now().isoformat(),
            'system_scan': {},
            'component_connections': {},
            'mcp_status': 'UNKNOWN',
            'overall_health': 'UNKNOWN',
            'execution_time': None
        }

        start_time = time.time()

        try:
            # Step 1: System Scan
            diagnostic_report['system_scan'] = self.perform_full_system_scan()

            # Step 2: Component Connection
            diagnostic_report['component_connections'] = self.connect_remaining_components()

            # Step 3: MCP Status Check
            print("\n🤖 PHASE 3: MCP INTEGRATION VERIFICATION")
            diagnostic_report['mcp_status'] = 'AVAILABLE' if self.check_mcp_server() else 'UNAVAILABLE'

            # Step 4: Overall Health Assessment
            scan_health = diagnostic_report['system_scan']['overall_status']
            connection_health = diagnostic_report['component_connections']['mcp_integration_status']

            if scan_health == 'HEALTHY' and connection_health in ['FULLY_CONNECTED', 'PARTIALLY_CONNECTED']:
                diagnostic_report['overall_health'] = 'EXCELLENT'
            elif scan_health in ['HEALTHY', 'PARTIAL'] and connection_health == 'PARTIALLY_CONNECTED':
                diagnostic_report['overall_health'] = 'GOOD'
            elif scan_health == 'DEGRADED' or connection_health == 'MCP_ERROR':
                diagnostic_report['overall_health'] = 'FAIR'
            else:
                diagnostic_report['overall_health'] = 'CRITICAL'

        except Exception as e:
            diagnostic_report['overall_health'] = 'ERROR'
            diagnostic_report['error'] = str(e)

        diagnostic_report['execution_time'] = round(time.time() - start_time, 2)

        # Print summary
        self.print_diagnostic_summary(diagnostic_report)

        return diagnostic_report

    def print_diagnostic_summary(self, report: Dict[str, Any]) -> None:
        """Print diagnostic summary"""

        # Overall health
        health = report.get('overall_health', 'UNKNOWN')
        health_icon = {
            'EXCELLENT': '🌟',
            'GOOD': '✅',
            'FAIR': '⚠️',
            'CRITICAL': '🚨',
            'ERROR': '❌'
        }.get(health, '❓')

        print(f"🏥 Overall Health: {health_icon} {health}")

        # System scan results
        scan = report.get('system_scan', {})
        print(f"📊 Services Scanned: {scan.get('services_scanned', 0)}")
        print(f"   ✅ Healthy: {scan.get('services_healthy', 0)}")
        print(f"   ⚠️ Degraded: {scan.get('services_degraded', 0)}")
        print(f"   ❌ Down: {scan.get('services_down', 0)}")

        # MCP status
        mcp_status = report.get('mcp_status', 'UNKNOWN')
        mcp_icon = '✅' if mcp_status == 'AVAILABLE' else '❌'

        # Component connections
        connections = report.get('component_connections', {})
        print(f"🔗 Components Connected: {connections.get('components_connected', 0)}")
        print(f"   ❌ Failed Connections: {connections.get('components_failed', 0)}")

        # Execution time
        exec_time = report.get('execution_time', 0)
        print(f"⚡ Execution Time: {exec_time:.2f} seconds")

        # Recommendations
        recommendations = scan.get('recommendations', [])
        if recommendations:
            for rec in recommendations:


def main():
    """Main entry point"""
    diagnostic = MCPSystemDiagnostic()

    try:
        report = diagnostic.run_full_diagnostic()

        # Save report to file
        report_file = f"mcp_diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Detailed report saved to: {report_file}")

    except KeyboardInterrupt:
        print("\n\n👋 MCP System Diagnostic terminated by user")
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()
