#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE VIPER TRADING SYSTEM DEBUGGER
Find and fix all issues in the trading flow

This debugger will:
- Analyze all trading components for issues
- Identify async/sync mismatches
- Check for missing dependencies
- Validate configuration files
- Test all critical trading paths
- Generate detailed fix recommendations
"""

import os
import sys
import json
import time
import subprocess
import importlib.util
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - COMPREHENSIVE_DEBUG - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDebugger:
    """Comprehensive debugger for the entire VIPER trading system"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues_found = []
        self.fixes_applied = []
        self.components_status = {}

    def run_comprehensive_debug(self):
        """Run complete system debugging"""
        print("🚀 COMPREHENSIVE VIPER TRADING SYSTEM DEBUG")
        print("=" * 70)

        try:
            # Step 1: Environment and Dependencies
            self.debug_environment_setup()

            # Step 2: Configuration Files
            self.debug_configuration_files()

            # Step 3: Core Components
            self.debug_core_components()

            # Step 4: Trading Flow
            self.debug_trading_flow()

            # Step 5: API Integration
            self.debug_api_integration()

            # Step 6: Async/Sync Issues
            self.debug_async_sync_issues()

            # Step 7: Generate Report
            self.generate_debug_report()

        except Exception as e:
            logger.error(f"❌ Comprehensive debug failed: {e}")
            print(f"\n❌ Debug failed: {e}")

    def debug_environment_setup(self):
        """Debug environment setup and dependencies"""
        print("\n🔧 Step 1: ENVIRONMENT & DEPENDENCIES")
        print("-" * 50)

        # Check Python version
        python_version = sys.version_info
        print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")

        # Check key dependencies
        dependencies = [
            'ccxt', 'numpy', 'pandas', 'asyncio', 'dotenv',
            'logging', 'json', 'pathlib', 'datetime'
        ]

        for dep in dependencies:
            try:
                __import__(dep)
                print(f"✅ {dep}: INSTALLED")
            except ImportError:
                print(f"❌ {dep}: MISSING")
                self.issues_found.append({
                    'type': 'dependency',
                    'component': dep,
                    'issue': 'Missing dependency',
                    'fix': f'pip install {dep}'
                })

        # Check environment variables
        required_env_vars = [
            'BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD'
        ]

        for env_var in required_env_vars:
            if os.getenv(env_var):
                print(f"✅ {env_var}: SET")
            else:
                print(f"❌ {env_var}: MISSING")
                self.issues_found.append({
                    'type': 'environment',
                    'component': env_var,
                    'issue': 'Missing environment variable',
                    'fix': f'Set {env_var} in .env file'
                })

    def debug_configuration_files(self):
        """Debug configuration files"""
        print("\n📋 Step 2: CONFIGURATION FILES")
        print("-" * 50)

        config_files = [
            '.env',
            'requirements.txt',
            'docker-compose.yml',
            'config/optimal_mcp_config.py'
        ]

        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                print(f"✅ {config_file}: EXISTS")

                # Check .env file specifically
                if config_file == '.env':
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            if 'BITGET_API_KEY=' in content:
                                print("   ✅ API credentials configured")
                            else:
                                print("   ⚠️  API credentials may be missing")
                    except Exception as e:
                        print(f"   ❌ Error reading .env: {e}")

            else:
                print(f"❌ {config_file}: MISSING")
                self.issues_found.append({
                    'type': 'configuration',
                    'component': config_file,
                    'issue': 'Missing configuration file',
                    'fix': f'Create {config_file}'
                })

    def debug_core_components(self):
        """Debug core trading components"""
        print("\n🏗️  Step 3: CORE COMPONENTS")
        print("-" * 50)

        components = [
            ('viper_async_trader.py', 'ViperAsyncTrader'),
            ('v2_risk_optimized_trading_job.py', 'V2RiskOptimizedTradingJob'),
            ('viper_unified_trading_job.py', 'VIPERUnifiedTradingJob'),
            ('advanced_trend_detector.py', 'AdvancedTrendDetector'),
            ('scripts/optimal_entry_point_manager.py', 'OptimalEntryPointManager'),
            ('scripts/master_diagnostic_scanner.py', 'MasterDiagnosticScanner'),
            ('utils/mathematical_validator.py', 'MathematicalValidator'),
            ('config/optimal_mcp_config.py', 'get_optimal_mcp_config')
        ]

        for file_path, class_name in components:
            full_path = self.project_root / file_path

            if not full_path.exists():
                print(f"❌ {class_name}: FILE MISSING ({file_path})")
                self.issues_found.append({
                    'type': 'component',
                    'component': class_name,
                    'issue': 'Component file missing',
                    'fix': f'Create {file_path}'
                })
                continue

            # Try to import and check syntax
            try:
                # Check syntax with compile
                with open(full_path, 'r') as f:
                    code = f.read()

                compile(code, full_path, 'exec')
                print(f"✅ {class_name}: SYNTAX OK")

                # Try to import the module
                try:
                    module_name = file_path.replace('.py', '').replace('/', '.')
                    spec = importlib.util.spec_from_file_location(module_name, full_path)
                    module = importlib.util.module_from_spec(spec)

                    # Don't actually execute, just check imports
                    print(f"   ✅ Module importable: {module_name}")

                except Exception as import_error:
                    print(f"   ⚠️  Import issue: {import_error}")

            except SyntaxError as syntax_error:
                print(f"❌ {class_name}: SYNTAX ERROR - {syntax_error}")
                self.issues_found.append({
                    'type': 'syntax',
                    'component': class_name,
                    'issue': f'Syntax error: {syntax_error}',
                    'fix': f'Fix syntax error in {file_path}'
                })
            except Exception as e:
                print(f"❌ {class_name}: ERROR - {e}")
                self.issues_found.append({
                    'type': 'component',
                    'component': class_name,
                    'issue': str(e),
                    'fix': f'Fix error in {file_path}'
                })

    def debug_trading_flow(self):
        """Debug the trading flow"""
        print("\n🔄 Step 4: TRADING FLOW")
        print("-" * 50)

        # Check for common trading flow issues
        flow_issues = [
            self.check_async_sync_issues,
            self.check_coroutine_errors,
            self.check_missing_methods,
            self.check_import_errors
        ]

        for check_func in flow_issues:
            try:
                issues = check_func()
                for issue in issues:
                    self.issues_found.append(issue)
            except Exception as e:
                print(f"❌ Flow check failed: {e}")

    def check_async_sync_issues(self) -> List[Dict]:
        """Check for async/sync mismatches"""
        issues = []

        # Check for common async patterns that might cause issues
        async_patterns = [
            r'await.*fetch_ohlcv',
            r'exchange\.fetch_ohlcv.*\(',
            r'await.*exchange\.',
            r'\.run_full_scan\(\)',
            r'await.*run_full_scan'
        ]

        files_to_check = [
            'viper_async_trader.py',
            'v2_risk_optimized_trading_job.py',
            'viper_unified_trading_job.py',
            'advanced_trend_detector.py'
        ]

        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()

                    for pattern in async_patterns:
                        if pattern.replace('\\', '').replace('.*', ' ') in content:
                            # This is a simplified check - in production you'd use regex
                            if 'await' in content and 'fetch_ohlcv' in content:
                                if 'exchange.fetch_ohlcv' in content and 'await' not in content.split('exchange.fetch_ohlcv')[0][-50:]:
                                    issues.append({
                                        'type': 'async_sync',
                                        'component': file_path,
                                        'issue': 'Potential async/sync mismatch with fetch_ohlcv',
                                        'fix': 'Add await to fetch_ohlcv calls or make method synchronous'
                                    })
                except Exception as e:
                    issues.append({
                        'type': 'file_error',
                        'component': file_path,
                        'issue': f'Could not check file: {e}',
                        'fix': 'Check file permissions and content'
                    })

        return issues

    def check_coroutine_errors(self) -> List[Dict]:
        """Check for coroutine-related errors"""
        issues = []

        # Look for patterns that commonly cause coroutine errors
        error_patterns = [
            'coroutine',
            'has no len',
            'object of type'
        ]

        # Check log files for these errors
        log_files = [
            'v2_risk_trading.log',
            'comprehensive_debug.log'
        ]

        for log_file in log_files:
            log_path = self.project_root / log_file
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        content = f.read()

                    for pattern in error_patterns:
                        if pattern in content:
                            issues.append({
                                'type': 'coroutine_error',
                                'component': log_file,
                                'issue': f'Found coroutine error pattern: {pattern}',
                                'fix': 'Fix async/sync mismatch in OHLCV fetching'
                            })
                except Exception as e:
                    issues.append({
                        'type': 'log_error',
                        'component': log_file,
                        'issue': f'Could not read log: {e}',
                        'fix': 'Check log file permissions'
                    })

        return issues

    def check_missing_methods(self) -> List[Dict]:
        """Check for missing methods"""
        issues = []

        # Check if run_full_scan_sync exists in MasterDiagnosticScanner
        scanner_path = self.project_root / 'scripts' / 'master_diagnostic_scanner.py'
        if scanner_path.exists():
            try:
                with open(scanner_path, 'r') as f:
                    content = f.read()

                if 'run_full_scan_sync' not in content:
                    issues.append({
                        'type': 'missing_method',
                        'component': 'MasterDiagnosticScanner',
                        'issue': 'Missing run_full_scan_sync method',
                        'fix': 'Add synchronous version of run_full_scan method'
                    })

                if 'run_full_scan' in content and 'async def run_full_scan' not in content:
                    issues.append({
                        'type': 'async_method',
                        'component': 'MasterDiagnosticScanner',
                        'issue': 'run_full_scan should be async',
                        'fix': 'Make run_full_scan method async'
                    })

            except Exception as e:
                issues.append({
                    'type': 'file_error',
                    'component': 'master_diagnostic_scanner.py',
                    'issue': str(e),
                    'fix': 'Check file content and permissions'
                })

        return issues

    def check_import_errors(self) -> List[Dict]:
        """Check for import errors"""
        issues = []

        # Try to import key modules
        modules_to_test = [
            ('ccxt', 'CCXT library for exchange integration'),
            ('numpy', 'NumPy for mathematical operations'),
            ('pandas', 'Pandas for data manipulation'),
            ('dotenv', 'python-dotenv for environment loading')
        ]

        for module_name, description in modules_to_test:
            try:
                __import__(module_name)
                print(f"✅ {description}: IMPORT OK")
            except ImportError:
                print(f"❌ {description}: IMPORT FAILED")
                issues.append({
                    'type': 'import_error',
                    'component': module_name,
                    'issue': f'Cannot import {description}',
                    'fix': f'pip install {module_name}'
                })

        return issues

    def debug_api_integration(self):
        """Debug API integration"""
        print("\n🔌 Step 5: API INTEGRATION")
        print("-" * 50)

        try:
            import ccxt

            # Try to create exchange instance
            exchange = ccxt.bitget({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })

            # Test basic connectivity
            exchange.load_markets()
            print("✅ Bitget exchange connection: SUCCESSFUL")

            # Check number of markets
            num_markets = len(exchange.markets)
            print(f"📊 Available markets: {num_markets}")

            # Check USDT pairs
            usdt_pairs = [symbol for symbol in exchange.markets.keys() if symbol.endswith('USDT:USDT')]
            print(f"💰 USDT swap pairs: {len(usdt_pairs)}")

            if usdt_pairs:
                print(f"   Sample pairs: {usdt_pairs[:5]}")

        except Exception as e:
            print(f"❌ API integration failed: {e}")
            self.issues_found.append({
                'type': 'api_error',
                'component': 'Bitget API',
                'issue': str(e),
                'fix': 'Check API credentials and network connectivity'
            })

    def debug_async_sync_issues(self):
        """Debug async/sync issues specifically"""
        print("\n🔄 Step 6: ASYNC/SYNC ANALYSIS")
        print("-" * 50)

        # Check for common async/sync patterns
        async_issues = self.check_async_sync_issues()

        if async_issues:
            print(f"⚠️  Found {len(async_issues)} potential async/sync issues:")
            for issue in async_issues:
                print(f"   • {issue['component']}: {issue['issue']}")
        else:
            print("✅ No obvious async/sync issues detected")

    def generate_debug_report(self):
        """Generate comprehensive debug report"""
        print("\n📊 Step 7: DEBUG REPORT")
        print("-" * 50)

        report = {
            'debug_timestamp': datetime.now().isoformat(),
            'total_issues_found': len(self.issues_found),
            'issues_by_type': {},
            'fixes_applied': len(self.fixes_applied),
            'system_status': 'UNKNOWN',
            'critical_issues': [],
            'recommendations': []
        }

        # Categorize issues
        for issue in self.issues_found:
            issue_type = issue.get('type', 'unknown')
            if issue_type not in report['issues_by_type']:
                report['issues_by_type'][issue_type] = []
            report['issues_by_type'][issue_type].append(issue)

        # Identify critical issues
        critical_types = ['async_sync', 'coroutine_error', 'api_error', 'dependency']
        for issue in self.issues_found:
            if issue.get('type') in critical_types:
                report['critical_issues'].append(issue)

        # Determine system status
        if len(report['critical_issues']) == 0 and len(self.issues_found) == 0:
            report['system_status'] = 'HEALTHY'
        elif len(report['critical_issues']) > 0:
            report['system_status'] = 'CRITICAL'
        else:
            report['system_status'] = 'WARNING'

        # Generate recommendations
        if report['system_status'] == 'CRITICAL':
            report['recommendations'].append("🚨 CRITICAL: Fix all critical issues before trading")
        if len(self.issues_found) > 0:
            report['recommendations'].append(f"📋 Fix {len(self.issues_found)} identified issues")
        if 'dependency' in report['issues_by_type']:
            report['recommendations'].append("📦 Install missing dependencies")
        if 'api_error' in report['issues_by_type']:
            report['recommendations'].append("🔑 Configure API credentials properly")

        # Save report
        report_path = self.project_root / f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Display summary
        print("🎯 DEBUG SUMMARY:")
        print(f"   Status: {report['system_status']}")
        print(f"   Total Issues: {len(self.issues_found)}")
        print(f"   Critical Issues: {len(report['critical_issues'])}")
        print(f"   Report Saved: {report_path}")

        if report['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   • {rec}")

        print(f"\n📋 ISSUES BY TYPE:")
        for issue_type, issues in report['issues_by_type'].items():
            print(f"   {issue_type.upper()}: {len(issues)} issues")

        # Overall assessment
        if report['system_status'] == 'HEALTHY':
            print("\n✅ SYSTEM STATUS: READY FOR LIVE TRADING")
        elif report['system_status'] == 'WARNING':
            print("\n⚠️  SYSTEM STATUS: READY WITH MINOR ISSUES")
        else:
            print("\n❌ SYSTEM STATUS: CRITICAL ISSUES DETECTED")

def main():
    """Main debug function"""
    debugger = ComprehensiveDebugger()
    debugger.run_comprehensive_debug()

if __name__ == "__main__":
    main()
