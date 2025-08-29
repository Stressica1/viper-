#!/usr/bin/env python3
"""
# Rocket VIPER Trading System - Comprehensive Diagnostic & Status Monitor
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import subprocess
import os

class ViperDiagnostic:
    """Comprehensive VIPER system diagnostic"""
    
    def __init__(self):
        self.services = {
            'API Server': ('http://localhost:8000', '/health'),
            'Risk Manager': ('http://localhost:8002', '/health'),
            'Order Lifecycle': ('http://localhost:8013', '/health'),
            'Exchange Connector': ('http://localhost:8005', '/health'),
            'Signal Processor': ('http://localhost:8011', '/health'),
            'Live Trading Engine': ('http://localhost:8007', '/health'),
            'Ultra Backtester': ('http://localhost:8001', '/health'),
            'Strategy Optimizer': ('http://localhost:8004', '/health'),
            'Data Manager': ('http://localhost:8003', '/health'),
            'Monitoring Service': ('http://localhost:8006', '/health'),
            'Credential Vault': ('http://localhost:8008', '/health'),
            'GitHub Manager': ('http://localhost:8001', '/health'),
            'Trading Optimizer': ('http://localhost:8009', '/health'),
        }
        
        self.trading_endpoints = {
            'Account Balance': ('http://localhost:8005', '/api/balance'),
            'Open Positions': ('http://localhost:8002', '/api/tp-sl-tsl/positions'),
            'Trading Signals': ('http://localhost:8011', '/api/signals/current'),
            'Order Status': ('http://localhost:8013', '/api/tp-sl-tsl/status/'),
            'Risk Metrics': ('http://localhost:8002', '/api/tp-sl-tsl/config'),
        }
    
    def check_service_health(self, service_name: str, base_url: str, health_endpoint: str) -> Dict[str, Any]:
        """Check individual service health"""
        try:
            response = requests.get(f"{base_url}{health_endpoint}", timeout=5)
            if response.status_code == 200:
                return {'status': 'healthy', 'response_time': response.elapsed.total_seconds()}
            else:
                return {'status': 'unhealthy', 'error': f'HTTP {response.status_code}'}
        except requests.exceptions.RequestException as e:
            return {'status': 'unreachable', 'error': str(e)}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def check_trading_functionality(self) -> Dict[str, Any]:
        """Check trading functionality"""
        results = {}
        
        for name, (base_url, endpoint) in self.trading_endpoints.items():
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    results[name] = {'status': 'operational', 'data': response.json()}
                else:
                    results[name] = {'status': 'error', 'error': f'HTTP {response.status_code}'}
            except Exception as e:
                results[name] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'trading': {},
            'performance': {},
            'system': {}
        }
        
        # Check all services
        for service_name, (base_url, health_endpoint) in self.services.items():
            metrics['services'][service_name] = self.check_service_health(
                service_name, base_url, health_endpoint
            )
        
        # Check trading functionality
        metrics['trading'] = self.check_trading_functionality()
        
        # System information
        try:
            import psutil
            metrics['system'] = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except ImportError:
            metrics['system'] = {'note': 'psutil not available'}
        
        return metrics
    
    def run_comprehensive_diagnostic(self) -> Dict[str, Any]:
        """Run complete system diagnostic"""
        
        # Get system metrics
        metrics = self.get_system_metrics()
        
        # Analyze results
        analysis = self.analyze_diagnostic_results(metrics)
        
        # Generate report
        report = {
            'diagnostic_time': datetime.now().isoformat(),
            'system_metrics': metrics,
            'analysis': analysis,
            'recommendations': self.generate_recommendations(analysis)
        }
        
        return report
    
    def analyze_diagnostic_results(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze diagnostic results"""
        analysis = {
            'service_health': {'healthy': 0, 'unhealthy': 0, 'total': 0},
            'trading_status': {'operational': 0, 'error': 0, 'total': 0},
            'critical_issues': [],
            'warnings': []
        }
        
        # Analyze service health
        for service, status in metrics['services'].items():
            analysis['service_health']['total'] += 1
            if status['status'] == 'healthy':
                analysis['service_health']['healthy'] += 1
            else:
                analysis['service_health']['unhealthy'] += 1
                analysis['critical_issues'].append(f"{service}: {status.get('error', 'Unknown error')}")
        
        # Analyze trading functionality
        for endpoint, status in metrics['trading'].items():
            analysis['trading_status']['total'] += 1
            if status['status'] == 'operational':
                analysis['trading_status']['operational'] += 1
            else:
                analysis['trading_status']['error'] += 1
                analysis['warnings'].append(f"{endpoint}: {status.get('error', 'Not operational')}")
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Service health recommendations
        healthy_percent = (analysis['service_health']['healthy'] / max(1, analysis['service_health']['total'])) * 100
        
        if healthy_percent < 80:
            recommendations.append("🚨 CRITICAL: Many services are unhealthy. Check Docker containers and logs.")
        elif healthy_percent < 100:
            recommendations.append("# Warning  WARNING: Some services are unhealthy. Check individual service logs.")
        
        # Trading functionality recommendations
        operational_percent = (analysis['trading_status']['operational'] / max(1, analysis['trading_status']['total'])) * 100
        
        if operational_percent < 50:
            recommendations.append("🚨 CRITICAL: Trading functionality is severely impaired.")
        elif operational_percent < 100:
            recommendations.append("# Warning  WARNING: Some trading endpoints are not responding.")
        
        if not recommendations:
            recommendations.append("# Check SYSTEM STATUS: All services operational and trading ready!")
        
        return recommendations
    
    def print_diagnostic_report(self, report: Dict[str, Any]):
        """Print formatted diagnostic report"""
        print(f"\n# Chart DIAGNOSTIC REPORT - {report['diagnostic_time']}")
        
        analysis = report['analysis']
        
        # Service Health
        healthy = analysis['service_health']['healthy']
        total = analysis['service_health']['total']
        print(f"   Healthy: {healthy}/{total} ({healthy/max(1,total)*100:.1f}%)")
        
        if analysis['critical_issues']:
            print("🚨 CRITICAL ISSUES:")
            for issue in analysis['critical_issues'][:5]:  # Show first 5
                print(f"   • {issue}")
        
        # Trading Status
        operational = analysis['trading_status']['operational']
        total_trading = analysis['trading_status']['total']
        print(f"   Operational: {operational}/{total_trading} ({operational/max(1,total_trading)*100:.1f}%)")
        
        if analysis['warnings']:
            print("# Warning WARNINGS:")
            for warning in analysis['warnings'][:3]:  # Show first 3
                print(f"   • {warning}")
        
        # Recommendations
        print("# Idea RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        

if __name__ == "__main__":
    diagnostic = ViperDiagnostic()
    report = diagnostic.run_comprehensive_diagnostic()
    diagnostic.print_diagnostic_report(report)
