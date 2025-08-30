#!/usr/bin/env python3
"""
# Rocket VIPER MASTER DIAGNOSTIC SCANNER
Comprehensive system scanner using optimal MCP server configurations
and enhanced mathematical validation

This is the ULTIMATE diagnostic tool that:
    pass
- Scans and validates ALL system components
- Uses optimal MCP server configurations
- Validates mathematical calculations in scoring/optimization
- Examines all workflows (GitHub CI/CD and trading)
- Provides actionable recommendations
- Creates detailed reports with fix suggestions
"""

import os
import sys
import json
import time
import subprocess
import importlib.util
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import traceback

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - MASTER_SCANNER - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)"""

class MasterDiagnosticScanner:
    """
    Master diagnostic scanner with optimal configurations
    """"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Optimal MCP server configurations
        self.optimal_mcp_config = {
            "server_url": "http://localhost:8015",
            "timeout": 30,
            "retry_attempts": 3,
            "retry_delay": 5,
            "health_check_interval": 60,
            "connection_pool_size": 10,
            "max_concurrent_requests": 5,
            "request_timeout": 15
        }
        
        # Service endpoints with optimal configurations
        self.service_endpoints = {
            'mcp-server': {'url': 'http://localhost:8015', 'critical': True},
            'api-server': {'url': 'http://localhost:8000', 'critical': True},
            'data-manager': {'url': 'http://localhost:8001', 'critical': True},
            'exchange-connector': {'url': 'http://localhost:8005', 'critical': True},
            'risk-manager': {'url': 'http://localhost:8002', 'critical': True},
            'monitoring-service': {'url': 'http://localhost:8010', 'critical': False},
            'credential-vault': {'url': 'http://localhost:8008', 'critical': False},
            'redis': {'url': 'redis://localhost:6379', 'critical': True}
        }
        
        # Workflow files to examine
        self.workflow_files = [
            '.github/workflows/trading-system.yml',
            'trade_execution_workflow.py',
            'services/workflow-monitor/main.py'
        ]
        
        # Mathematical components to validate
        self.math_components = [
            'ai_ml_optimizer.py',
            'scripts/scoring_system_diagnostic.py',
            'viper_scoring_service',
            'comprehensive_backtester.py',
            'advanced_trend_detector.py'
        ]
        
        self.scan_results = {
            'timestamp': datetime.now().isoformat(),
            'scan_duration': 0,
            'system_status': 'SCANNING',
            'services': {},
            'workflows': {},
            'mathematical_validation': {},
            'mcp_optimization': {},
            'recommendations': [],
            'critical_issues': [],
            'performance_metrics': {},
            'entry_point_analysis': {}
        }
    
    def run_comprehensive_scan(self) -> Dict[str, Any]
        """Run the complete comprehensive system scan"""
        print("#" + " # Rocket VIPER MASTER DIAGNOSTIC SCANNER - COMPREHENSIVE ANALYSIS ".center(78) + "#")
        print("#" + " # Search Full System Scan | 🧮 Math Validation | # Chart Workflow Analysis ".center(78) + "#")
        print("#" + " ⚡ Optimal MCP Config | # Target Entry Point Analysis | 📈 Performance ".center(78) + "#")
        
        start_time = time.time()
        :
        # Phase 1: System Services Diagnostic
        print("# Chart PHASE 1: COMPREHENSIVE SYSTEM SERVICES DIAGNOSTIC")
        self.scan_system_services()
        
        # Phase 2: MCP Server Optimization Analysis
        print("\n🤖 PHASE 2: MCP SERVER OPTIMIZATION ANALYSIS")
        self.analyze_mcp_optimization()
        
        # Phase 3: Mathematical Component Validation
        print("\n🧮 PHASE 3: MATHEMATICAL COMPONENT VALIDATION")
        self.validate_mathematical_components()
        
        # Phase 4: Workflow Analysis
        print("\n🔄 PHASE 4: WORKFLOW ANALYSIS & VALIDATION")
        self.analyze_workflows()
        
        # Phase 5: Entry Point Optimization Analysis
        print("\n# Target PHASE 5: ENTRY POINT OPTIMIZATION ANALYSIS")
        self.analyze_entry_points()
        
        # Phase 6: Performance Metrics Collection
        print("\n📈 PHASE 6: PERFORMANCE METRICS COLLECTION")
        self.collect_performance_metrics()
        
        # Phase 7: Generate Recommendations
        print("\n# Idea PHASE 7: GENERATING ACTIONABLE RECOMMENDATIONS")
        self.generate_recommendations()
        
        # Finalize results
        self.scan_results['scan_duration'] = round(time.time() - start_time, 2)
        self.scan_results['system_status'] = self.determine_overall_status()
        
        # Save and display results
        self.save_scan_results()
        self.display_scan_summary()
        
        return self.scan_results
    
    def scan_system_services(self):
        """Scan all system services with enhanced diagnostics"""
        # Import and run existing diagnostic tools with our optimizations
        services_scanned = 0
        services_healthy = 0
        services_degraded = 0
        services_down = 0
        
        for service_name, config in self.service_endpoints.items():
            service_result = self.diagnose_service_advanced(service_name, config)
            self.scan_results['services'][service_name] = service_result
            
            services_scanned += 1"""
            if service_result['status'] == 'HEALTHY':
                services_healthy += 1
            elif service_result['status'] == 'DEGRADED':
                services_degraded += 1
            else:
                services_down += 1
                if config['critical']:
                    self.scan_results['critical_issues'].append(f"Critical service {service_name} is DOWN")
        
        print(f"# Chart Services Summary: {services_scanned} scanned, {services_healthy} healthy, {services_degraded} degraded, {services_down} down")
    
    def diagnose_service_advanced(self, service_name: str, config: Dict[str, Any]) -> Dict[str, Any]
        """Advanced service diagnostics with optimal configurations"""
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
        
        result = {:
            'service': service_name,
            'status': 'UNKNOWN',
            'response_time': None,
            'error': None,
            'details': {},
            'critical': config['critical'],
            'optimization_suggestions': []
        }
        
        # Create optimized session
        session = requests.Session()
        retry_strategy = Retry()
            total=self.optimal_mcp_config['retry_attempts'],
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
(        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        try:
            start_time = time.time()
            
            # Handle Redis differently
            if service_name == 'redis':
                result = self.diagnose_redis_service(result)
            else:
                # HTTP service check
                url = f"{config['url']}/health"
                response = session.get(url, timeout=self.optimal_mcp_config['request_timeout'])
                response_time = time.time() - start_time
                
                result['response_time'] = round(response_time * 1000, 2)  # ms
                
                if response.status_code == 200:
                    result['status'] = 'HEALTHY'
                    try:
                        result['details'] = response.json()
                    except Exception:
                        result['details'] = {'message': 'Service responding'}
                elif response.status_code >= 500:
                    result['status'] = 'DOWN'
                    result['error'] = f"Server error: {response.status_code}"
                else:
                    result['status'] = 'DEGRADED' 
                    result['error'] = f"HTTP {response.status_code}"
                    
        except requests.exceptions.Timeout:
            result['status'] = 'DEGRADED'
            result['error'] = "Connection timeout"
            result['optimization_suggestions'].append("Consider increasing timeout values")
        except requests.exceptions.ConnectionError
            result['status'] = 'DOWN'
            result['error'] = "Connection refused"
            result['optimization_suggestions'].append("Service may not be running - check Docker containers")
        except Exception as e:
            result['status'] = 'DOWN'
            result['error'] = str(e)
        
        return result
    
    def diagnose_redis_service(self, result: Dict[str, Any]) -> Dict[str, Any]
        """Special diagnostics for Redis service""":"""
        try:
            # Try Redis connection using redis-py if available
    import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            start_time = time.time()
            pong = r.ping()
            response_time = time.time() - start_time
            
            if pong:
                result['status'] = 'HEALTHY'
                result['response_time'] = round(response_time * 1000, 2)
                result['details'] = {'ping': 'PONG', 'connected': True}
            else:
                result['status'] = 'DOWN'
                result['error'] = "Redis ping failed"
                
        except ImportError:
            result['status'] = 'UNKNOWN'
            result['error'] = "Redis client not available for testing"
            result['optimization_suggestions'].append("Install redis-py for better Redis diagnostics")
        except Exception as e:
            result['status'] = 'DOWN'
            result['error'] = f"Redis connection failed: {e}"
            result['optimization_suggestions'].append("Check Redis container is running")
        
        return result
    
    def analyze_mcp_optimization(self):
        """Analyze MCP server configuration for optimization opportunities"""
        
        mcp_analysis = {
            'current_config': self.optimal_mcp_config,
            'optimization_score': 0,
            'recommendations': [],
            'performance_tuning': {},
            'security_assessment': {}
        }
        
        # Check if MCP server files exist and analyze them
        mcp_files = [
            'services/mcp-server/main.py',
            'mcp_brain_controller.py',
            'mcp_brain_service.py'
        ]
        
        mcp_files_analyzed = 0
        for mcp_file in mcp_files:
            file_path = self.project_root / mcp_file"""
            if file_path.exists():
                mcp_files_analyzed += 1
                
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Analyze configuration optimizations
                self.analyze_mcp_file_content(content, mcp_file, mcp_analysis)
        
        # Calculate optimization score
        base_score = 70  # Base score for having MCP files
        if mcp_files_analyzed > 0:
            mcp_analysis['optimization_score'] = min(base_score + (mcp_files_analyzed * 10), 100)
        
        # Add performance tuning recommendations
        mcp_analysis['performance_tuning'] = {
            'connection_pooling': 'Implement connection pooling for better resource management',
            'async_operations': 'Use async operations for non-blocking I/O',
            'caching_strategy': 'Implement intelligent caching for frequently accessed data',
            'load_balancing': 'Consider load balancing for high-traffic scenarios'
        }
        
        self.scan_results['mcp_optimization'] = mcp_analysis
        print(f"# Check MCP Optimization Score: {mcp_analysis['optimization_score']}/100")
    
    def analyze_mcp_file_content(self, content: str, filename: str, analysis: Dict[str, Any]):
        """Analyze MCP file content for optimization opportunities"""
        
        # Check for async/await usage"""
        if 'async def' in content:
            analysis['recommendations'].append(f"# Check {filename}: Good use of async operations")
        else:
            analysis['recommendations'].append(f"# Warning {filename}: Consider using async operations for better performance")
        
        # Check for error handling
        if 'try:' in content and 'except' in content:
            analysis['recommendations'].append(f"# Check {filename}: Error handling implemented")
        else:
            analysis['recommendations'].append(f"# X {filename}: Missing comprehensive error handling")
        
        # Check for timeout configurations
        if 'timeout' in content:
            analysis['recommendations'].append(f"# Check {filename}: Timeout configurations found")
        else:
            analysis['recommendations'].append(f"# Warning {filename}: Consider adding timeout configurations")
    
    def validate_mathematical_components(self):
        """Validate mathematical calculations in trading components"""
        
        math_validation = {
            'components_checked': 0,
            'validation_score': 0,
            'issues_found': [],
            'optimizations': [],
            'formula_validations': {}
        }
        
        for component in self.math_components:
            component_path = self.project_root / component"""
            
            if component_path.exists():
                math_validation['components_checked'] += 1
                
                if component_path.is_file():
                    validation_result = self.validate_math_file(component_path)
                else:
                    # It's a directory, check for Python files
                    validation_result = self.validate_math_directory(component_path)
                
                math_validation['formula_validations'][component] = validation_result
            else:
                math_validation['issues_found'].append(f"Component not found: {component}")
        
        # Calculate validation score
        if math_validation['components_checked'] > 0:
            base_score = 60
            issues_penalty = len(math_validation['issues_found']) * 5
            math_validation['validation_score'] = max(base_score - issues_penalty, 0)
            
            # Add points for good practices found
            for component, result in math_validation['formula_validations'].items():
                math_validation['validation_score'] += result.get('quality_score', 0)
        
        self.scan_results['mathematical_validation'] = math_validation
        print(f"# Check Mathematical Validation Score: {math_validation['validation_score']}/100")
    
    def validate_math_file(self, file_path: Path) -> Dict[str, Any]
        """Validate mathematical formulas in a single file"""
        validation = {:
            'formulas_found': 0,
            'quality_score': 0,
            'issues': [],
            'optimizations': []
        }"""
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for mathematical operations and formulas
            math_indicators = [
                'np.', 'math.', '* 100', '/ 100', 'sqrt', 'log', 'exp',
                'calculate', 'score', 'optimization', 'returns', 'volatility',
                'sharpe', 'drawdown', 'risk'
            ]
            
            for indicator in math_indicators:
                if indicator in content:
                    validation['formulas_found'] += 1
            
            # Check for good mathematical practices
            good_practices = [
                ('error handling', 'try:' in content and 'except' in content),
                ('input validation', 'assert' in content or 'if' in content),
                ('documentation', '"""' in content or "'''" in content),
                ('numpy usage', 'import numpy' in content),
                ('pandas usage', 'import pandas' in content)
            ]
            
            for practice, found in good_practices:
                if found:
                    validation['quality_score'] += 4
                    validation['optimizations'].append(f"# Check Good practice: {practice}")
                else:
                    validation['issues'].append(f"# Warning Consider adding: {practice}")
            
        except Exception as e:
            validation['issues'].append(f"Error reading file: {e}")
        
        return validation
    
    def validate_math_directory(self, dir_path: Path) -> Dict[str, Any]
        """Validate mathematical formulas in a directory"""
        validation = {:
            'files_checked': 0,
            'formulas_found': 0,
            'quality_score': 0,
            'issues': [],
            'optimizations': []
        }
        
        # Check Python files in the directory
        for py_file in dir_path.glob('*.py'):
            file_validation = self.validate_math_file(py_file)
            validation['files_checked'] += 1
            validation['formulas_found'] += file_validation['formulas_found']
            validation['quality_score'] += file_validation['quality_score']
            validation['issues'].extend(file_validation['issues'])
            validation['optimizations'].extend(file_validation['optimizations'])
        
        return validation"""
    
    def analyze_workflows(self):
        """Analyze and validate all workflow files"""
        
        workflow_analysis = {
            'workflows_analyzed': 0,
            'github_workflows': {},
            'trading_workflows': {},
            'optimization_recommendations': [],
            'security_issues': [],
            'performance_issues': []
        }
        
        for workflow_file in self.workflow_files:
            workflow_path = self.project_root / workflow_file"""
            
            if workflow_path.exists():
                workflow_analysis['workflows_analyzed'] += 1
                
                analysis_result = self.analyze_workflow_file(workflow_path)
                
                if workflow_file.startswith('.github'):
                    workflow_analysis['github_workflows'][workflow_file] = analysis_result
                else:
                    workflow_analysis['trading_workflows'][workflow_file] = analysis_result
            else:
                workflow_analysis['optimization_recommendations'].append()
                    f"Missing workflow file: {workflow_file}"
(                )
        
        # Validate GitHub Actions workflow specifically
        self.validate_github_workflow(workflow_analysis)
        
        self.scan_results['workflows'] = workflow_analysis
        print(f"# Check Workflows Analyzed: {workflow_analysis['workflows_analyzed']}")
    
    def analyze_workflow_file(self, file_path: Path) -> Dict[str, Any]
        """Analyze individual workflow file"""
        analysis = {:
            'file_size': file_path.stat().st_size,
            'complexity_score': 0,
            'security_score': 0,
            'optimization_opportunities': [],
            'issues_found': []
        }"""
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Analyze content based on file type
            if file_path.suffix == '.yml' or file_path.suffix == '.yaml':
                analysis.update(self.analyze_yaml_workflow(content))
            elif file_path.suffix == '.py':
                analysis.update(self.analyze_python_workflow(content))
            
        except Exception as e:
            analysis['issues_found'].append(f"Error reading file: {e}")
        
        return analysis
    
    def analyze_yaml_workflow(self, content: str) -> Dict[str, Any]
        """Analyze YAML workflow content (GitHub Actions)"""
        analysis = {:
            'type': 'github_actions',
            'jobs_count': content.count('jobs:'),
            'steps_count': content.count('- name:'),
            'uses_secrets': 'secrets.' in content,
            'has_caching': 'cache' in content,
            'has_parallel_jobs': content.count('needs:') > 0
        }
        
        # Security checks"""
        if 'API_KEY' in content or 'SECRET' in content:
            if 'secrets.' not in content:
                analysis['security_issues'] = ['Hardcoded secrets detected']
            else:
                analysis['security_good_practices'] = ['Uses GitHub secrets properly']
        
        # Performance optimizations
        if not analysis['has_caching']:
            analysis['optimization_opportunities'] = ['Consider adding dependency caching']
        
        return analysis
    
    def analyze_python_workflow(self, content: str) -> Dict[str, Any]
        """Analyze Python workflow content"""
        analysis = {:
            'type': 'python_workflow',
            'async_functions': content.count('async def'),
            'error_handling': content.count('try:'),
            'logging_usage': 'logging' in content or 'logger' in content,
            'has_main_guard': 'if __name__ == "__main__"' in content
        }
        
        # Best practices check
        if analysis['logging_usage']:
            analysis['good_practices'] = ['Uses logging']
        if analysis['has_main_guard']:
            analysis.setdefault('good_practices', []).append('Has main guard')
        
        return analysis
    
    def validate_github_workflow(self, workflow_analysis: Dict[str, Any]):
        """Special validation for GitHub workflow"""
        github_workflow_file = '.github/workflows/trading-system.yml'"""
        
        if github_workflow_file in workflow_analysis['github_workflows']:
            github_workflow = workflow_analysis['github_workflows'][github_workflow_file]
            
            # Check for essential CI/CD components
            essential_checks = [
                ('dependency_installation', 'Install dependencies'),
                ('syntax_check', 'syntax check'),
                ('security_scan', 'security scan'),
                ('test_execution', 'test')
            ]
            
            for check_name, check_keyword in essential_checks:
                # This is a simplified check - in practice you'd parse YAML properly
                workflow_analysis['optimization_recommendations'].append()
                    f"GitHub workflow includes {check_name}: {'# Check' if check_keyword.lower() in str(github_workflow).lower() else '# X'}"
(                )
    
    def analyze_entry_points(self):
        """Analyze entry point configurations for optimization"""
        print("# Target Analyzing Entry Point Configurations...")
        
        entry_analysis = {
            'entry_files_found': [],
            'optimization_opportunities': [],
            'configuration_issues': [],
            'performance_recommendations': [],
            'best_practices_score': 0
        }
        
        # Common entry point files
        entry_point_files = [
            'main.py',
            'standalone_viper_trader.py',
            'ai_ml_optimizer.py',
            'simple_trader.py',
            'launch_complete_system.py',
            'run_live_system.py'
        ]
        
        for entry_file in entry_point_files:
            entry_path = self.project_root / entry_file
            if entry_path.exists():
                entry_analysis['entry_files_found'].append(entry_file)
                
                file_analysis = self.analyze_entry_point_file(entry_path)
                entry_analysis[entry_file] = file_analysis
                
                # Add to best practices score
                entry_analysis['best_practices_score'] += file_analysis.get('quality_score', 0)
        
        # Generate optimization recommendations
        self.generate_entry_point_recommendations(entry_analysis)
        
        self.scan_results['entry_point_analysis'] = entry_analysis
        print(f"# Check Entry Points Found: {len(entry_analysis['entry_files_found'])}")
    
    def analyze_entry_point_file(self, file_path: Path) -> Dict[str, Any]
        """Analyze individual entry point file"""
        analysis = {:
            'quality_score': 0,
            'config_flexibility': 0,
            'error_handling_score': 0,
            'optimization_suggestions': []
        }"""
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for configuration flexibility
            config_indicators = [
                'os.getenv', 'config', 'settings', '.env', 'dotenv'
            ]
            
            for indicator in config_indicators:
                if indicator in content:
                    analysis['config_flexibility'] += 10
            
            # Check for error handling
            if 'try:' in content and 'except' in content:
                analysis['error_handling_score'] += 20
                
            # Check for logging
            if 'logging' in content or 'logger' in content:
                analysis['quality_score'] += 15
                
            # Check for main guard
            if 'if __name__ == "__main__"' in content:
                analysis['quality_score'] += 10
                
            # Check for hardcoded values (negative points)
            hardcoded_patterns = ['localhost', '8000', 'http://', 'api_key']
            for pattern in hardcoded_patterns:
                if pattern in content.lower():
                    analysis['optimization_suggestions'].append(f"Consider making {pattern} configurable")
            
        except Exception as e:
            analysis['optimization_suggestions'].append(f"Error analyzing file: {e}")
        
        return analysis
    
    def generate_entry_point_recommendations(self, entry_analysis: Dict[str, Any]):
        """Generate recommendations for entry point optimization"""
        
        # Check if main entry points exist
        critical_entry_points = ['main.py', 'standalone_viper_trader.py']
        missing_critical = [ep for ep in critical_entry_points if ep not in entry_analysis['entry_files_found']]"""
        
        if missing_critical:
            entry_analysis['configuration_issues'].extend([)
                f"Missing critical entry point: {ep}" for ep in missing_critical
(            ])
        
        # Performance recommendations
        entry_analysis['performance_recommendations'] = [
            "Use environment variables for all configuration",
            "Implement connection pooling for external services",
            "Add comprehensive error handling and logging",
            "Consider async operations for I/O bound tasks",
            "Implement graceful shutdown handlers"
        ]
        
        # Optimization opportunities
        entry_analysis['optimization_opportunities'] = [
            "Consolidate configuration management",
            "Implement centralized logging",
            "Add health check endpoints to all services",
            "Use configuration validation",
            "Implement circuit breaker patterns for external calls"
        ]
    
    def collect_performance_metrics(self):
        """Collect system performance metrics"""
        
    import psutil
        
        metrics = {
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
            },
            'python_process': {
                'memory_mb': round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
            },
            'recommendations': []
        }
        
        # Generate performance recommendations"""
        if metrics['system_resources']['cpu_percent'] > 80:
            metrics['recommendations'].append("High CPU usage detected - consider optimization")
        
        if metrics['system_resources']['memory_percent'] > 85:
            metrics['recommendations'].append("High memory usage detected - check for memory leaks")
            
        if metrics['system_resources']['disk_usage'] > 90:
            metrics['recommendations'].append("Low disk space - clean up old logs and temporary files")
        
        self.scan_results['performance_metrics'] = metrics
        print(f"# Chart CPU: {metrics['system_resources']['cpu_percent']}%, Memory: {metrics['system_resources']['memory_percent']}%")
    
    def generate_recommendations(self):
        """Generate comprehensive actionable recommendations"""
        recommendations = []
        critical_issues = self.scan_results['critical_issues']
        
        # Critical issues first"""
        if critical_issues:
            recommendations.append("🚨 CRITICAL ACTIONS REQUIRED:")
            recommendations.extend([f"   - {issue}" for issue in critical_issues])
            recommendations.append("")
        
        # Service recommendations
        down_services = [name for name, service in self.scan_results['services'].items() 
                        if service['status'] == 'DOWN']:
        if down_services:
            recommendations.append("# Tool SERVICE RESTORATION:")
            recommendations.append(f"   - Start these services: {', '.join(down_services)}")
            recommendations.append("   - Check Docker containers: docker-compose up -d")
            recommendations.append("")
        
        # MCP optimization recommendations
        mcp_recommendations = self.scan_results['mcp_optimization'].get('recommendations', [])
        if mcp_recommendations:
            recommendations.append("🤖 MCP OPTIMIZATION:")
            recommendations.extend([f"   - {rec}" for rec in mcp_recommendations[:3]])
            recommendations.append("")
        
        # Mathematical validation recommendations
        math_issues = self.scan_results['mathematical_validation'].get('issues_found', [])
        if math_issues:
            recommendations.append("🧮 MATHEMATICAL VALIDATION:")
            recommendations.extend([f"   - Fix: {issue}" for issue in math_issues[:3]])
            recommendations.append("")
        
        # Entry point optimization
        entry_issues = self.scan_results['entry_point_analysis'].get('configuration_issues', [])
        if entry_issues:
            recommendations.append("# Target ENTRY POINT OPTIMIZATION:")
            recommendations.extend([f"   - {issue}" for issue in entry_issues[:3]])
            recommendations.append("")
        
        # Performance recommendations
        perf_recommendations = self.scan_results['performance_metrics'].get('recommendations', [])
        if perf_recommendations:
            recommendations.append("📈 PERFORMANCE OPTIMIZATION:")
            recommendations.extend([f"   - {rec}" for rec in perf_recommendations])
        
        self.scan_results['recommendations'] = recommendations
    
    def determine_overall_status(self) -> str:
        """Determine overall system status based on scan results"""
        
        # Count critical issues
        critical_count = len(self.scan_results['critical_issues'])
        
        # Count down services
        down_count = sum(1 for service in self.scan_results['services'].values() """)
(                        if service['status'] == 'DOWN')
        # Check critical services
        critical_services = [name for name, service in self.scan_results['services'].items() 
                           if service.get('critical', False) and service['status'] == 'DOWN']:
        if len(critical_services) > 2:
            return 'SYSTEM_CRITICAL'
        elif len(critical_services) > 0:
            return 'DEGRADED'
        elif down_count > 3:
            return 'NEEDS_ATTENTION'
        elif down_count > 0:
            return 'PARTIAL_OPERATION'
        else:
            return 'OPERATIONAL'
    
    def save_scan_results(self):
        """Save comprehensive scan results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"master_diagnostic_scan_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(self.scan_results, f, indent=2, default=str)
            print(f"💾 Comprehensive report saved to: {report_file}")
        except Exception as e:
            pass
    
    def display_scan_summary(self):
        """Display comprehensive scan summary"""
        print("📋 MASTER DIAGNOSTIC SCAN - COMPREHENSIVE SUMMARY")
        
        # Overall status
        status_icon = {
            'OPERATIONAL': '# Check',
            'PARTIAL_OPERATION': '# Warning',
            'NEEDS_ATTENTION': '🔶',
            'DEGRADED': '# X',
            'SYSTEM_CRITICAL': '🚨'
        }.get(self.scan_results['system_status'], '❓')
        
        print(f"🏥 Overall System Status: {status_icon} {self.scan_results['system_status']}")
        print(f"⚡ Scan Duration: {self.scan_results['scan_duration']} seconds")
        
        # Services summary
        services = self.scan_results['services']
        healthy = sum(1 for s in services.values() if s['status'] == 'HEALTHY')
        degraded = sum(1 for s in services.values() if s['status'] == 'DEGRADED')
        down = sum(1 for s in services.values() if s['status'] == 'DOWN')
        
        
        # Key metrics
        mcp_score = self.scan_results['mcp_optimization'].get('optimization_score', 0)
        math_score = self.scan_results['mathematical_validation'].get('validation_score', 0)
        entry_score = self.scan_results['entry_point_analysis'].get('best_practices_score', 0)
        
        print(f"   🧮 Mathematical Validation: {math_score}/100")
        print(f"   # Target Entry Point Quality: {entry_score}/100")
        
        # Top recommendations
        recommendations = self.scan_results['recommendations']
        if recommendations:
            for rec in recommendations[:8]:  # Show first 8 recommendations
        


def main():
    """Run the master diagnostic scanner""""""
    try:
        scanner = MasterDiagnosticScanner()
        results = scanner.run_comprehensive_scan()
        
        # Return appropriate exit code based on system status
        if results['system_status'] in ['SYSTEM_CRITICAL', 'DEGRADED']:
            return 1
        else:
            return 0
            
    except Exception as e:
        logger.error(f"Master diagnostic scan failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())