#!/usr/bin/env python3
"""
🚀 VIPER Trading System - Comprehensive Health Check & Diagnostic
Complete system analysis focusing on operational services
"""

import os
import sys
import time
import json
import requests
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

class ServiceStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    CONNECTING = "CONNECTING"
    UNKNOWN = "UNKNOWN"

class SystemHealthChecker:
    """
    Comprehensive system health checker for VIPER trading system
    """

    def __init__(self):
        """Initialize health checker"""
        self.services = {
            'api-server': {'port': 8000, 'name': 'API Server', 'critical': True},
            'data-manager': {'port': 8003, 'name': 'Data Manager', 'critical': True},
            'exchange-connector': {'port': 8005, 'name': 'Exchange Connector', 'critical': True},
            'risk-manager': {'port': 8002, 'name': 'Risk Manager', 'critical': True},
            'monitoring-service': {'port': 8006, 'name': 'Monitoring Service', 'critical': False},
            'credential-vault': {'port': 8008, 'name': 'Credential Vault', 'critical': False},
            'redis': {'port': 6379, 'name': 'Redis Cache', 'critical': True},
            'mcp-server': {'port': 8015, 'name': 'MCP Server', 'critical': False}
        }

        print("🔍 VIPER System Health Checker Initialized")
        print(f"📊 Services to check: {len(self.services)}")
        print("=" * 80)

    def check_service_health(self, service_name: str, port: int, name: str) -> Dict[str, Any]:
        """Check individual service health"""
        result = {
            'service': service_name,
            'name': name,
            'port': port,
            'status': ServiceStatus.UNKNOWN.value,
            'response_time': None,
            'error': None,
            'details': {}
        }

        try:
            start_time = time.time()

            # Special handling for Redis (different protocol)
            if service_name == 'redis':
                # Use redis-cli to check Redis health
                try:
                    result_redis = subprocess.run(
                        ['docker', 'exec', f'viper-trading-{service_name}-1', 'redis-cli', 'ping'],
                        capture_output=True, text=True, timeout=5
                    )
                    response_time = time.time() - start_time

                    if result_redis.returncode == 0 and 'PONG' in result_redis.stdout:
                        result['status'] = ServiceStatus.HEALTHY.value
                        result['response_time'] = round(response_time * 1000, 2)
                        result['details'] = {'ping_response': result_redis.stdout.strip()}
                    else:
                        result['status'] = ServiceStatus.DOWN.value
                        result['error'] = 'Redis ping failed'
                except subprocess.TimeoutExpired:
                    result['status'] = ServiceStatus.DEGRADED.value
                    result['error'] = 'Redis connection timeout'
                except Exception as e:
                    result['status'] = ServiceStatus.DOWN.value
                    result['error'] = f'Redis check error: {e}'
            else:
                # HTTP health check for other services
                response = requests.get(f"http://localhost:{port}/health", timeout=10)
                response_time = time.time() - start_time

                result['response_time'] = round(response_time * 1000, 2)

                if response.status_code == 200:
                    result['status'] = ServiceStatus.HEALTHY.value
                    try:
                        result['details'] = response.json()
                    except:
                        result['details'] = {'message': 'Service responding but no JSON data'}
                elif response.status_code >= 500:
                    result['status'] = ServiceStatus.DOWN.value
                    result['error'] = f"Server error: {response.status_code}"
                else:
                    result['status'] = ServiceStatus.DEGRADED.value
                    result['error'] = f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            result['status'] = ServiceStatus.DEGRADED.value
            result['error'] = "Connection timeout"
        except requests.exceptions.ConnectionError:
            result['status'] = ServiceStatus.DOWN.value
            result['error'] = "Connection refused"
        except Exception as e:
            result['status'] = ServiceStatus.DOWN.value
            result['error'] = str(e)

        return result

    def perform_comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all services"""
        print("\n🚀 STARTING COMPREHENSIVE HEALTH CHECK...")
        print("🔍 Checking all services and their connectivity")
        print("=" * 80)

        health_report = {
            'timestamp': datetime.now().isoformat(),
            'check_duration': None,
            'services_checked': len(self.services),
            'services_healthy': 0,
            'services_degraded': 0,
            'services_down': 0,
            'critical_services_healthy': 0,
            'critical_services_down': 0,
            'overall_status': 'UNKNOWN',
            'service_details': {},
            'recommendations': []
        }

        start_time = time.time()

        # Check each service
        for service_name, config in self.services.items():
            print(f"🔍 Checking {config['name']}...")
            service_result = self.check_service_health(
                service_name, config['port'], config['name']
            )
            health_report['service_details'][service_name] = service_result

            # Count results
            if service_result['status'] == ServiceStatus.HEALTHY.value:
                health_report['services_healthy'] += 1
                if config['critical']:
                    health_report['critical_services_healthy'] += 1
            elif service_result['status'] == ServiceStatus.DEGRADED.value:
                health_report['services_degraded'] += 1
            else:
                health_report['services_down'] += 1
                if config['critical']:
                    health_report['critical_services_down'] += 1

        health_report['check_duration'] = round(time.time() - start_time, 2)

        # Determine overall status
        if health_report['critical_services_down'] > 0:
            health_report['overall_status'] = 'CRITICAL'
        elif health_report['services_down'] > 0:
            health_report['overall_status'] = 'DEGRADED'
        elif health_report['services_healthy'] == len(self.services):
            health_report['overall_status'] = 'EXCELLENT'
        else:
            health_report['overall_status'] = 'GOOD'

        # Generate recommendations
        health_report['recommendations'] = self.generate_recommendations(health_report)

        return health_report

    def generate_recommendations(self, health_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health check results"""
        recommendations = []

        # Check for critical service issues
        if health_report['critical_services_down'] > 0:
            down_critical = [
                config['name'] for service, config in self.services.items()
                if config['critical'] and
                health_report['service_details'][service]['status'] == ServiceStatus.DOWN.value
            ]
            recommendations.append(f"🚨 CRITICAL: Restart these essential services: {', '.join(down_critical)}")

        # Check for non-critical service issues
        if health_report['services_down'] > health_report['critical_services_down']:
            down_non_critical = [
                config['name'] for service, config in self.services.items()
                if not config['critical'] and
                health_report['service_details'][service]['status'] == ServiceStatus.DOWN.value
            ]
            if down_non_critical:
                recommendations.append(f"⚠️ WARNING: These non-critical services are down: {', '.join(down_non_critical)}")

        # Check for degraded services
        degraded_services = [
            config['name'] for service, config in self.services.items()
            if health_report['service_details'][service]['status'] == ServiceStatus.DEGRADED.value
        ]

        if degraded_services:
            recommendations.append(f"🔧 PERFORMANCE: Investigate these slow/degraded services: {', '.join(degraded_services)}")

        # Check response times
        slow_services = []
        for service, result in health_report['service_details'].items():
            response_time = result.get('response_time')
            if response_time and response_time > 1000:  # > 1 second
                slow_services.append(f"{self.services[service]['name']} ({response_time}ms)")

        if slow_services:
            recommendations.append(f"⚡ OPTIMIZE: These services have slow response times: {', '.join(slow_services)}")

        # Overall recommendations
        if health_report['overall_status'] == 'EXCELLENT':
            recommendations.append("✅ SYSTEM EXCELLENT: All services operational and performing well")
        elif health_report['overall_status'] == 'GOOD':
            recommendations.append("✅ SYSTEM GOOD: Core services operational with minor issues")
        elif health_report['overall_status'] == 'DEGRADED':
            recommendations.append("⚠️ SYSTEM DEGRADED: Some services need attention but system is functional")
        else:
            recommendations.append("🚨 SYSTEM CRITICAL: Immediate maintenance required")

        return recommendations

    def test_system_connectivity(self) -> Dict[str, Any]:
        """Test connectivity between services"""
        print("\n🔗 TESTING SYSTEM CONNECTIVITY...")
        print("🚀 Checking inter-service communication")
        print("=" * 80)

        connectivity_report = {
            'timestamp': datetime.now().isoformat(),
            'connectivity_tests': {},
            'pipeline_status': {},
            'overall_connectivity': 'UNKNOWN'
        }

        # Test API Server connectivity to other services
        api_server_status = self.services['api-server']
        if api_server_status:
            print("🔗 Testing API Server connectivity to other services...")

            # Test connection to data-manager
            try:
                response = requests.get("http://localhost:8000/api/data/status", timeout=5)
                connectivity_report['connectivity_tests']['api_to_data'] = {
                    'status': 'CONNECTED' if response.status_code == 200 else 'FAILED',
                    'response_code': response.status_code
                }
            except:
                connectivity_report['connectivity_tests']['api_to_data'] = {'status': 'FAILED'}

            # Test connection to risk-manager
            try:
                response = requests.get("http://localhost:8000/api/risk/status", timeout=5)
                connectivity_report['connectivity_tests']['api_to_risk'] = {
                    'status': 'CONNECTED' if response.status_code == 200 else 'FAILED',
                    'response_code': response.status_code
                }
            except:
                connectivity_report['connectivity_tests']['api_to_risk'] = {'status': 'FAILED'}

        # Determine overall connectivity
        connected_count = sum(1 for test in connectivity_report['connectivity_tests'].values()
                            if test['status'] == 'CONNECTED')
        total_tests = len(connectivity_report['connectivity_tests'])

        if connected_count == total_tests:
            connectivity_report['overall_connectivity'] = 'FULLY_CONNECTED'
        elif connected_count > 0:
            connectivity_report['overall_connectivity'] = 'PARTIALLY_CONNECTED'
        else:
            connectivity_report['overall_connectivity'] = 'NOT_CONNECTED'

        return connectivity_report

    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete system diagnostic"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🚀 VIPER SYSTEM HEALTH DIAGNOSTIC - COMPLETE ANALYSIS                       ║
║ 🔍 Comprehensive Service Health | 🔗 Connectivity Testing | 📊 Performance   ║
║ ⚡ Real-time Monitoring | 🧠 System Analysis | 📈 Health Reporting           ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)

        diagnostic_report = {
            'diagnostic_start': datetime.now().isoformat(),
            'health_check': {},
            'connectivity_test': {},
            'system_analysis': {},
            'overall_health': 'UNKNOWN',
            'execution_time': None
        }

        start_time = time.time()

        try:
            # Phase 1: Health Check
            print("\n📊 PHASE 1: SYSTEM HEALTH CHECK")
            diagnostic_report['health_check'] = self.perform_comprehensive_health_check()

            # Phase 2: Connectivity Test
            print("\n🔗 PHASE 2: CONNECTIVITY ANALYSIS")
            diagnostic_report['connectivity_test'] = self.test_system_connectivity()

            # Phase 3: System Analysis
            print("\n🧠 PHASE 3: SYSTEM ANALYSIS")
            diagnostic_report['system_analysis'] = self.perform_system_analysis(diagnostic_report)

            # Determine overall health
            health_status = diagnostic_report['health_check']['overall_status']
            connectivity_status = diagnostic_report['connectivity_test']['overall_connectivity']

            if health_status in ['EXCELLENT', 'GOOD'] and connectivity_status in ['FULLY_CONNECTED', 'PARTIALLY_CONNECTED']:
                diagnostic_report['overall_health'] = 'SYSTEM_OPERATIONAL'
            elif health_status == 'DEGRADED' or connectivity_status == 'PARTIALLY_CONNECTED':
                diagnostic_report['overall_health'] = 'SYSTEM_DEGRADED'
            else:
                diagnostic_report['overall_health'] = 'SYSTEM_CRITICAL'

        except Exception as e:
            print(f"❌ Diagnostic error: {e}")
            diagnostic_report['overall_health'] = 'ERROR'
            diagnostic_report['error'] = str(e)

        diagnostic_report['execution_time'] = round(time.time() - start_time, 2)

        # Print comprehensive summary
        self.print_comprehensive_summary(diagnostic_report)

        return diagnostic_report

    def perform_system_analysis(self, diagnostic_report: Dict[str, Any]) -> Dict[str, Any]:
        """Perform system-wide analysis"""
        analysis = {
            'trading_readiness': 'UNKNOWN',
            'infrastructure_status': 'UNKNOWN',
            'monitoring_coverage': 'UNKNOWN',
            'risk_management': 'UNKNOWN'
        }

        # Analyze trading readiness
        critical_services = ['api-server', 'data-manager', 'exchange-connector', 'risk-manager', 'redis']
        critical_healthy = sum(1 for service in critical_services
                             if diagnostic_report['health_check']['service_details'].get(service, {}).get('status') == 'HEALTHY')

        if critical_healthy == len(critical_services):
            analysis['trading_readiness'] = 'READY_FOR_TRADING'
        elif critical_healthy >= len(critical_services) - 1:
            analysis['trading_readiness'] = 'READY_WITH_LIMITATIONS'
        else:
            analysis['trading_readiness'] = 'NOT_READY_FOR_TRADING'

        # Analyze infrastructure status
        infrastructure_services = ['redis', 'monitoring-service']
        infra_healthy = sum(1 for service in infrastructure_services
                          if diagnostic_report['health_check']['service_details'].get(service, {}).get('status') == 'HEALTHY')

        if infra_healthy == len(infrastructure_services):
            analysis['infrastructure_status'] = 'INFRASTRUCTURE_SOLID'
        else:
            analysis['infrastructure_status'] = 'INFRASTRUCTURE_DEGRADED'

        # Analyze monitoring coverage
        monitoring_services = ['monitoring-service']
        monitoring_healthy = sum(1 for service in monitoring_services
                               if diagnostic_report['health_check']['service_details'].get(service, {}).get('status') == 'HEALTHY')

        if monitoring_healthy == len(monitoring_services):
            analysis['monitoring_coverage'] = 'FULL_MONITORING'
        else:
            analysis['monitoring_coverage'] = 'LIMITED_MONITORING'

        # Analyze risk management
        risk_services = ['risk-manager', 'exchange-connector']
        risk_healthy = sum(1 for service in risk_services
                         if diagnostic_report['health_check']['service_details'].get(service, {}).get('status') == 'HEALTHY')

        if risk_healthy == len(risk_services):
            analysis['risk_management'] = 'RISK_MANAGEMENT_ACTIVE'
        else:
            analysis['risk_management'] = 'RISK_MANAGEMENT_DEGRADED'

        return analysis

    def print_comprehensive_summary(self, report: Dict[str, Any]) -> None:
        """Print comprehensive diagnostic summary"""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE SYSTEM DIAGNOSTIC SUMMARY")
        print("=" * 80)

        # Overall health
        health = report.get('overall_health', 'UNKNOWN')
        health_icon = {
            'SYSTEM_OPERATIONAL': '🚀',
            'SYSTEM_DEGRADED': '⚠️',
            'SYSTEM_CRITICAL': '🚨',
            'ERROR': '❌'
        }.get(health, '❓')

        print(f"🏥 Overall System Health: {health_icon} {health}")

        # Health check results
        health_check = report.get('health_check', {})
        print(f"📊 Services Checked: {health_check.get('services_checked', 0)}")
        print(f"   ✅ Healthy: {health_check.get('services_healthy', 0)}")
        print(f"   ⚠️ Degraded: {health_check.get('services_degraded', 0)}")
        print(f"   ❌ Down: {health_check.get('services_down', 0)}")
        print(f"   🔴 Critical Down: {health_check.get('critical_services_down', 0)}")

        # Connectivity results
        connectivity = report.get('connectivity_test', {})
        print(f"🔗 Connectivity Status: {connectivity.get('overall_connectivity', 'UNKNOWN')}")

        # System analysis
        analysis = report.get('system_analysis', {})
        print(f"🎯 Trading Readiness: {analysis.get('trading_readiness', 'UNKNOWN')}")
        print(f"🏗️ Infrastructure: {analysis.get('infrastructure_status', 'UNKNOWN')}")
        print(f"📊 Monitoring: {analysis.get('monitoring_coverage', 'UNKNOWN')}")
        print(f"🛡️ Risk Management: {analysis.get('risk_management', 'UNKNOWN')}")

        # Execution time
        exec_time = report.get('execution_time', 0)
        print(f"⚡ Analysis Time: {exec_time:.2f} seconds")

        # Recommendations
        recommendations = health_check.get('recommendations', [])
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        print("=" * 80)

        # Action items based on health
        if health == 'SYSTEM_OPERATIONAL':
            print("🎉 SYSTEM STATUS: FULLY OPERATIONAL - Ready for trading!")
        elif health == 'SYSTEM_DEGRADED':
            print("⚠️ SYSTEM STATUS: DEGRADED - Limited functionality available")
        else:
            print("🚨 SYSTEM STATUS: CRITICAL - Immediate maintenance required")

def main():
    """Main entry point"""
    checker = SystemHealthChecker()

    try:
        report = checker.run_full_diagnostic()

        # Save detailed report
        report_file = f"system_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Detailed report saved to: {report_file}")

    except KeyboardInterrupt:
        print("\n\n👋 System Health Check terminated by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
