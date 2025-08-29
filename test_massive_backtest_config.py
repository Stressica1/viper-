#!/usr/bin/env python3
"""
🧪 TEST MASSIVE BACKTEST CONFIGURATION
Validate the 50-pair 200-config setup before full execution

This test provides:
✅ Configuration validation
✅ Resource requirement assessment
✅ Execution time estimation
✅ Sample backtest execution
✅ System compatibility check
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import components
from massive_backtest_orchestrator import MassiveBacktestOrchestrator
from mcp_backtesting_optimizer import MCPBacktestingOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CONFIG_TEST - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MassiveBacktestConfigTester:
    """Test and validate massive backtest configuration"""

    def __init__(self):
        self.config_path = project_root / "massive_backtest_config.json"
        self.config_data = None
        self.orchestrator = None

    async def load_and_validate_config(self) -> bool:
        """Load and validate configuration file"""
        try:
            print("🔍 LOADING CONFIGURATION")
            print("=" * 40)

            if not self.config_path.exists():
                print("❌ Configuration file not found")
                return False

            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)

            print("✅ Configuration file loaded")

            # Validate structure
            required_keys = [
                'massive_backtest_configuration',
                'trading_pairs',
                'configuration_variations',
                'backtest_parameters',
                'processing_strategy'
            ]

            massive_config = self.config_data.get('massive_backtest_configuration', {})

            for key in required_keys:
                if key not in massive_config:
                    print(f"❌ Missing required key: {key}")
                    return False

            print("✅ Configuration structure validated")

            # Initialize orchestrator for further validation
            self.orchestrator = MassiveBacktestOrchestrator(self.config_path)

            return True

        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False

    def analyze_configuration_scale(self) -> Dict[str, Any]:
        """Analyze the scale of the backtesting operation"""
        try:
            massive_config = self.config_data['massive_backtest_configuration']

            # Extract dimensions
            trading_pairs = len(massive_config['trading_pairs'])
            timeframes = len(massive_config['timeframes'])

            # Calculate configuration variations
            variations = massive_config['configuration_variations']
            ma_variations = len(variations['moving_average_configs'])
            atr_variations = len(variations['atr_configs'])
            risk_variations = len(variations['risk_configs'])
            trend_variations = len(variations['trend_configs'])
            entry_variations = len(variations['entry_filters'])

            total_configs = (ma_variations * atr_variations * risk_variations *
                           trend_variations * entry_variations)

            total_combinations = trading_pairs * timeframes * total_configs

            # Calculate memory requirements (rough estimate)
            memory_per_backtest_mb = 50  # MB per backtest
            total_memory_required_gb = (total_combinations * memory_per_backtest_mb) / 1024

            # Calculate execution time (rough estimate)
            seconds_per_backtest = 30  # Conservative estimate
            total_seconds = total_combinations * seconds_per_backtest
            total_hours = total_seconds / 3600
            total_days = total_hours / 24

            # Parallel processing adjustment
            max_concurrent = massive_config['processing_strategy']['max_concurrent_pairs']
            effective_hours = total_hours / max_concurrent
            effective_days = effective_hours / 24

            return {
                'trading_pairs': trading_pairs,
                'timeframes': timeframes,
                'ma_variations': ma_variations,
                'atr_variations': atr_variations,
                'risk_variations': risk_variations,
                'trend_variations': trend_variations,
                'entry_variations': entry_variations,
                'total_configs': total_configs,
                'total_combinations': total_combinations,
                'estimated_memory_gb': total_memory_required_gb,
                'estimated_total_hours': total_hours,
                'estimated_total_days': total_days,
                'estimated_effective_hours': effective_hours,
                'estimated_effective_days': effective_days,
                'max_concurrent_pairs': max_concurrent
            }

        except Exception as e:
            logger.error(f"❌ Scale analysis failed: {e}")
            return {}

    def validate_system_requirements(self, scale_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system requirements against scale analysis"""
        try:
            requirements_check = {
                'memory_sufficient': False,
                'time_feasible': False,
                'warnings': [],
                'recommendations': []
            }

            # Memory check (rough - we need about 8GB free for processing)
            estimated_memory = scale_analysis.get('estimated_memory_gb', 0)
            if estimated_memory > 8:
                requirements_check['warnings'].append(f"High memory requirement: {estimated_memory:.1f}GB estimated")
                requirements_check['recommendations'].append("Consider reducing concurrent pairs or historical days")
            else:
                requirements_check['memory_sufficient'] = True

            # Time feasibility (should complete within 30 days)
            effective_days = scale_analysis.get('estimated_effective_days', 0)
            if effective_days > 30:
                requirements_check['warnings'].append(f"Long execution time: {effective_days:.1f} days estimated")
                requirements_check['recommendations'].append("Consider distributed processing or reduce scope")
            elif effective_days > 7:
                requirements_check['warnings'].append(f"Extended execution: {effective_days:.1f} days estimated")
                requirements_check['recommendations'].append("Plan for multi-day execution")
            else:
                requirements_check['time_feasible'] = True

            # Scale warnings
            total_combinations = scale_analysis.get('total_combinations', 0)
            if total_combinations > 100000:
                requirements_check['warnings'].append("Very large scale operation")
                requirements_check['recommendations'].append("Consider phased execution")
            elif total_combinations > 50000:
                requirements_check['warnings'].append("Large scale operation")
                requirements_check['recommendations'].append("Monitor system resources closely")

            return requirements_check

        except Exception as e:
            logger.error(f"❌ Requirements validation failed: {e}")
            return {}

    async def run_sample_backtest(self) -> Dict[str, Any]:
        """Run a sample backtest to validate the system"""
        try:
            print("🧪 RUNNING SAMPLE BACKTEST")
            print("=" * 30)

            # Initialize optimizer
            optimizer = MCPBacktestingOptimizer()
            await optimizer.initialize_exchange()

            # Run a single backtest with sample parameters
            sample_symbol = 'BTCUSDT'
            sample_timeframe = '1h'
            sample_days = 30

            print(f"📊 Testing: {sample_symbol} {sample_timeframe} ({sample_days} days)")

            # Create sample configuration
            from mcp_backtesting_optimizer import OptimizationParameters
            sample_config = OptimizationParameters(
                fast_ma_length=21,
                slow_ma_length=50,
                atr_length=14,
                atr_multiplier=2.0,
                take_profit_pct=3.0,
                stop_loss_pct=5.0,
                trailing_stop_pct=2.0
            )

            start_time = datetime.now()
            result = await optimizer.run_backtest(
                symbol=sample_symbol,
                timeframe=sample_timeframe,
                config=sample_config,
                days=sample_days
            )
            end_time = datetime.now()

            execution_time = (end_time - start_time).total_seconds()

            if result:
                print("✅ Sample backtest successful")
                print(f"   Win Rate: {result.win_rate:.1f}%")
                print(f"   Total P&L: ${result.total_pnl:.2f}")
                print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
                print(f"   Execution Time: {execution_time:.2f} seconds")

                return {
                    'success': True,
                    'result': result.__dict__,
                    'execution_time_seconds': execution_time
                }
            else:
                print("❌ Sample backtest failed")
                return {'success': False, 'error': 'Backtest returned no results'}

        except Exception as e:
            logger.error(f"❌ Sample backtest failed: {e}")
            return {'success': False, 'error': str(e)}

    async def generate_test_report(self, scale_analysis: Dict[str, Any],
                                 requirements_check: Dict[str, Any],
                                 sample_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        try:
            report = {
                'test_timestamp': datetime.now().isoformat(),
                'configuration_validation': 'PASSED' if self.config_data else 'FAILED',
                'scale_analysis': scale_analysis,
                'system_requirements': requirements_check,
                'sample_backtest': sample_result,
                'recommendations': [],
                'readiness_assessment': 'READY'
            }

            # Generate recommendations based on analysis
            if requirements_check.get('warnings'):
                report['recommendations'].extend([
                    "Address system requirement warnings before full execution",
                    "Consider phased approach for large-scale testing"
                ])

            total_combinations = scale_analysis.get('total_combinations', 0)
            if total_combinations > 50000:
                report['recommendations'].append(
                    "Very large operation - consider distributed processing"
                )

            effective_days = scale_analysis.get('estimated_effective_days', 0)
            if effective_days > 14:
                report['recommendations'].append(
                    "Long execution time - plan for multi-week operation"
                )

            # Assess overall readiness
            if not requirements_check.get('memory_sufficient', False):
                report['readiness_assessment'] = 'NOT_READY_MEMORY'
            elif not requirements_check.get('time_feasible', False):
                report['readiness_assessment'] = 'NOT_READY_TIME'
            elif not sample_result.get('success', False):
                report['readiness_assessment'] = 'NOT_READY_SYSTEM'

            # Save test report
            report_filename = f"massive_backtest_config_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            print(f"💾 Test report saved: {report_filename}")
            return report

        except Exception as e:
            logger.error(f"❌ Test report generation failed: {e}")
            return {}

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive configuration test"""
        try:
            print("🧪 COMPREHENSIVE CONFIGURATION TEST")
            print("=" * 50)
            print("Testing massive backtest configuration and system readiness")
            print("=" * 50)

            # Step 1: Load and validate configuration
            print("\n1️⃣ CONFIGURATION VALIDATION")
            config_valid = await self.load_and_validate_config()
            if not config_valid:
                return {'status': 'FAILED', 'step': 'configuration'}

            # Step 2: Analyze scale
            print("\n2️⃣ SCALE ANALYSIS")
            scale_analysis = self.analyze_configuration_scale()
            if not scale_analysis:
                return {'status': 'FAILED', 'step': 'scale_analysis'}

            print(f"   Trading Pairs: {scale_analysis['trading_pairs']}")
            print(f"   Total Configurations: {scale_analysis['total_configs']}")
            print(f"   Total Combinations: {scale_analysis['total_combinations']:,}")
            print(f"   Estimated Memory: {scale_analysis['estimated_memory_gb']:.1f}GB")
            print(f"   Estimated Time: {scale_analysis['estimated_effective_days']:.1f} days")

            # Step 3: Validate system requirements
            print("\n3️⃣ SYSTEM REQUIREMENTS CHECK")
            requirements_check = self.validate_system_requirements(scale_analysis)

            if requirements_check.get('warnings'):
                print("⚠️ WARNINGS:")
                for warning in requirements_check['warnings']:
                    print(f"   - {warning}")

            if requirements_check.get('recommendations'):
                print("💡 RECOMMENDATIONS:")
                for rec in requirements_check['recommendations']:
                    print(f"   - {rec}")

            # Step 4: Run sample backtest
            print("\n4️⃣ SAMPLE BACKTEST VALIDATION")
            sample_result = await self.run_sample_backtest()

            # Step 5: Generate comprehensive report
            print("\n5️⃣ GENERATING TEST REPORT")
            test_report = await self.generate_test_report(
                scale_analysis, requirements_check, sample_result
            )

            # Final assessment
            print("\n" + "=" * 50)
            print("📊 TEST RESULTS SUMMARY")
            print("=" * 50)

            readiness = test_report.get('readiness_assessment', 'UNKNOWN')

            if readiness == 'READY':
                print("✅ SYSTEM READY FOR MASSIVE BACKTEST")
                print("   All validations passed")
                print("   Sample backtest successful")
                print("   System requirements met")
            elif readiness.startswith('NOT_READY'):
                print(f"❌ SYSTEM NOT READY: {readiness}")
                print("   Address issues before proceeding")
            else:
                print("⚠️ SYSTEM STATUS UNCLEAR")
                print("   Review test report for details")

            print("
📋 RECOMMENDATIONS:"            recommendations = test_report.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    print(f"   • {rec}")
            else:
                print("   • No specific recommendations")

            print("
📄 Detailed results saved to test report file"            return test_report

        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}

def main():
    """Main test function"""
    print("🧪 MASSIVE BACKTEST CONFIGURATION TESTER")
    print("=" * 50)

    tester = MassiveBacktestConfigTester()

    async def run_test():
        result = await tester.run_comprehensive_test()

        if result.get('status') == 'FAILED':
            print(f"\n❌ TEST FAILED at step: {result.get('step', 'unknown')}")
            if result.get('error'):
                print(f"Error: {result['error']}")
            return 1
        else:
            print("\n✅ CONFIGURATION TEST COMPLETED")
            return 0

    try:
        exit_code = asyncio.run(run_test())
        return exit_code
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
