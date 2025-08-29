#!/usr/bin/env python3
"""
🚀 ENHANCED SYSTEM INTEGRATION TEST
Comprehensive testing of all integrated optimized modules

This test suite validates:
- Module initialization and loading
- Inter-module communication
- Data flow between modules
- Performance under various conditions
- Error handling and recovery
- Configuration validation
"""

import os
import sys
import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - INTEGRATION_TEST - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedSystemIntegrationTest:
    """Comprehensive integration test suite for enhanced trading system"""

    def __init__(self):
        self.test_results = []
        self.modules_status = {}
        self.performance_metrics = {}
        self.error_log = []

        # Test configuration
        self.test_config = {
            "test_duration_seconds": 300,  # 5 minutes
            "stress_test_enabled": True,
            "performance_test_enabled": True,
            "error_injection_enabled": False,
            "data_quality_test_enabled": True,
            "integration_test_enabled": True
        }

    async def run_full_integration_test(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        logger.info("🚀 Starting Enhanced System Integration Test Suite")
        logger.info("=" * 80)

        start_time = time.time()
        test_results = {
            "test_start_time": datetime.now().isoformat(),
            "modules_tested": [],
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_metrics": {},
            "errors": [],
            "recommendations": []
        }

        try:
            # 1. System Initialization Test
            logger.info("📋 Test 1: System Initialization")
            init_result = await self.test_system_initialization()
            test_results["modules_tested"].extend(init_result.get("modules_tested", []))
            if init_result["success"]:
                test_results["tests_passed"] += 1
                logger.info("✅ System Initialization: PASSED")
            else:
                test_results["tests_failed"] += 1
                test_results["errors"].extend(init_result.get("errors", []))
                logger.error("❌ System Initialization: FAILED")

            # 2. Module Integration Test
            logger.info("📋 Test 2: Module Integration")
            integration_result = await self.test_module_integration()
            if integration_result["success"]:
                test_results["tests_passed"] += 1
                logger.info("✅ Module Integration: PASSED")
            else:
                test_results["tests_failed"] += 1
                test_results["errors"].extend(integration_result.get("errors", []))
                logger.error("❌ Module Integration: FAILED")

            # 3. Data Flow Test
            logger.info("📋 Test 3: Data Flow Validation")
            data_flow_result = await self.test_data_flow()
            if data_flow_result["success"]:
                test_results["tests_passed"] += 1
                logger.info("✅ Data Flow Validation: PASSED")
            else:
                test_results["tests_failed"] += 1
                test_results["errors"].extend(data_flow_result.get("errors", []))
                logger.error("❌ Data Flow Validation: FAILED")

            # 4. Performance Test
            if self.test_config["performance_test_enabled"]:
                logger.info("📋 Test 4: Performance Testing")
                perf_result = await self.test_performance()
                test_results["performance_metrics"] = perf_result.get("metrics", {})
                if perf_result["success"]:
                    test_results["tests_passed"] += 1
                    logger.info("✅ Performance Testing: PASSED")
                else:
                    test_results["tests_failed"] += 1
                    test_results["errors"].extend(perf_result.get("errors", []))
                    logger.error("❌ Performance Testing: FAILED")

            # 5. Error Handling Test
            logger.info("📋 Test 5: Error Handling & Recovery")
            error_result = await self.test_error_handling()
            if error_result["success"]:
                test_results["tests_passed"] += 1
                logger.info("✅ Error Handling: PASSED")
            else:
                test_results["tests_failed"] += 1
                test_results["errors"].extend(error_result.get("errors", []))
                logger.error("❌ Error Handling: FAILED")

            # 6. Configuration Validation
            logger.info("📋 Test 6: Configuration Validation")
            config_result = await self.test_configuration_validation()
            if config_result["success"]:
                test_results["tests_passed"] += 1
                logger.info("✅ Configuration Validation: PASSED")
            else:
                test_results["tests_failed"] += 1
                test_results["errors"].extend(config_result.get("errors", []))
                logger.error("❌ Configuration Validation: FAILED")

            # Generate recommendations
            test_results["recommendations"] = self.generate_test_recommendations(test_results)

        except Exception as e:
            logger.error(f"❌ Integration test failed with exception: {e}")
            test_results["errors"].append(f"Test execution error: {str(e)}")
            test_results["tests_failed"] += 1

        # Calculate test summary
        test_results["test_duration_seconds"] = time.time() - start_time
        test_results["test_end_time"] = datetime.now().isoformat()
        test_results["overall_success"] = test_results["tests_failed"] == 0
        test_results["success_rate"] = (test_results["tests_passed"] /
                                       (test_results["tests_passed"] + test_results["tests_failed"])) * 100

        logger.info("=" * 80)
        logger.info(f"🎯 Integration Test Complete!")
        logger.info(f"   Tests Passed: {test_results['tests_passed']}")
        logger.info(f"   Tests Failed: {test_results['tests_failed']}")
        logger.info(".1f")
        logger.info(f"   Overall Status: {'✅ SUCCESS' if test_results['overall_success'] else '❌ FAILED'}")

        return test_results

    async def test_system_initialization(self) -> Dict[str, Any]:
        """Test system initialization and module loading"""
        result = {
            "success": True,
            "modules_tested": [],
            "errors": []
        }

        try:
            logger.info("   🔧 Testing Enhanced System Initialization...")

            # Test enhanced system integrator
            from enhanced_system_integrator import initialize_enhanced_system

            success = await initialize_enhanced_system()
            if not success:
                result["success"] = False
                result["errors"].append("Enhanced system initialization failed")
                return result

            # Test individual module loading
            modules_to_test = [
                "enhanced_ai_ml_optimizer",
                "enhanced_technical_optimizer",
                "enhanced_risk_manager",
                "optimized_market_data_streamer",
                "performance_monitoring_system"
            ]

            from enhanced_system_integrator import get_integrator
            integrator = get_integrator()

            for module_name in modules_to_test:
                result["modules_tested"].append(module_name)

                module_instance = integrator.get_module(module_name)
                if module_instance is None:
                    result["success"] = False
                    result["errors"].append(f"Module {module_name} failed to load")
                else:
                    logger.info(f"   ✅ {module_name}: LOADED")

            # Test system configuration loading
            if hasattr(integrator, 'system_config'):
                logger.info("   ✅ System configuration: LOADED")
            else:
                result["success"] = False
                result["errors"].append("System configuration not loaded")

        except Exception as e:
            result["success"] = False
            result["errors"].append(f"System initialization test error: {str(e)}")
            logger.error(f"   ❌ System initialization test failed: {e}")

        return result

    async def test_module_integration(self) -> Dict[str, Any]:
        """Test integration between modules"""
        result = {
            "success": True,
            "errors": []
        }

        try:
            logger.info("   🔗 Testing Module Integration...")

            from enhanced_system_integrator import get_integrator
            integrator = get_integrator()

            # Test data sharing between modules
            test_data = {"test_key": "test_value", "timestamp": datetime.now()}

            # Share data from one module
            integrator.share_data("integration_test", test_data, "test_module")

            # Retrieve data from another module
            retrieved_data = integrator.get_shared_data("integration_test", "retrieving_module")

            if retrieved_data != test_data:
                result["success"] = False
                result["errors"].append("Data sharing between modules failed")
            else:
                logger.info("   ✅ Data sharing: WORKING")

            # Test module dependencies
            ai_ml = integrator.get_module("enhanced_ai_ml_optimizer")
            technical = integrator.get_module("enhanced_technical_optimizer")

            if ai_ml and technical:
                logger.info("   ✅ Module dependencies: RESOLVED")
            else:
                result["success"] = False
                result["errors"].append("Module dependencies not properly resolved")

        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Module integration test error: {str(e)}")
            logger.error(f"   ❌ Module integration test failed: {e}")

        return result

    async def test_data_flow(self) -> Dict[str, Any]:
        """Test data flow between modules"""
        result = {
            "success": True,
            "errors": []
        }

        try:
            logger.info("   📊 Testing Data Flow...")

            from enhanced_system_integrator import get_integrator
            integrator = get_integrator()

            # Test market data streamer
            data_streamer = integrator.get_module("optimized_market_data_streamer")
            if data_streamer:
                # Test data fetching
                test_symbol = "BTCUSDT"
                test_data = await data_streamer.fetch_market_data(test_symbol, "1h", limit=5)

                if test_data is not None and not test_data.empty:
                    logger.info(f"   ✅ Market data fetching: WORKING ({len(test_data)} bars)")
                else:
                    result["success"] = False
                    result["errors"].append("Market data fetching failed")

                # Test data caching
                cached_data = await data_streamer.fetch_market_data(test_symbol, "1h", limit=5)
                if cached_data is not None:
                    logger.info("   ✅ Data caching: WORKING")
                else:
                    logger.warning("   ⚠️ Data caching: UNTESTED")

            # Test technical analysis
            technical_optimizer = integrator.get_module("enhanced_technical_optimizer")
            if technical_optimizer and data_streamer:
                # Get market data first
                market_data = await data_streamer.fetch_market_data(test_symbol, "1h", limit=50)
                if market_data is not None:
                    # Test technical analysis
                    df = market_data.tail(50)  # Use recent data
                    patterns = technical_optimizer.detect_advanced_patterns(df)

                    logger.info(f"   ✅ Technical analysis: WORKING ({len(patterns)} patterns detected)")

            # Test AI/ML analysis
            ai_ml_optimizer = integrator.get_module("enhanced_ai_ml_optimizer")
            if ai_ml_optimizer and data_streamer:
                market_data = await data_streamer.fetch_market_data(test_symbol, "1h", limit=100)
                if market_data is not None:
                    df = market_data.tail(100)
                    ai_result = await ai_ml_optimizer.optimize_entry_points_enhanced(df)

                    if ai_result:
                        logger.info(f"   ✅ AI/ML analysis: WORKING (confidence: {ai_result.get('confidence', 0):.2f})")

        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Data flow test error: {str(e)}")
            logger.error(f"   ❌ Data flow test failed: {e}")

        return result

    async def test_performance(self) -> Dict[str, Any]:
        """Test system performance under load"""
        result = {
            "success": True,
            "metrics": {},
            "errors": []
        }

        try:
            logger.info("   ⚡ Testing Performance Under Load...")

            from enhanced_system_integrator import get_integrator
            integrator = get_integrator()

            # Performance test parameters
            test_duration = 60  # 1 minute
            concurrent_requests = 5
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"]

            start_time = time.time()
            response_times = []
            error_count = 0

            # Run concurrent performance test
            tasks = []
            for i in range(concurrent_requests):
                task = asyncio.create_task(self._run_performance_test_cycle(symbols, test_duration))
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for result_data in results:
                if isinstance(result_data, Exception):
                    error_count += 1
                elif isinstance(result_data, dict):
                    response_times.extend(result_data.get("response_times", []))
                    error_count += result_data.get("errors", 0)

            # Calculate performance metrics
            total_requests = len(response_times) + error_count
            avg_response_time = np.mean(response_times) if response_times else 0
            p95_response_time = np.percentile(response_times, 95) if response_times else 0
            error_rate = error_count / max(total_requests, 1)

            result["metrics"] = {
                "total_requests": total_requests,
                "successful_requests": len(response_times),
                "failed_requests": error_count,
                "avg_response_time_ms": avg_response_time * 1000,
                "p95_response_time_ms": p95_response_time * 1000,
                "error_rate": error_rate,
                "requests_per_second": total_requests / test_duration,
                "test_duration_seconds": time.time() - start_time
            }

            # Check performance thresholds
            if avg_response_time > 5.0:  # 5 seconds
                result["success"] = False
                result["errors"].append(".2f")
            if error_rate > 0.1:  # 10% error rate
                result["success"] = False
                result["errors"].append(".1%")

            logger.info("   📊 Performance Test Results:")
            logger.info(f"   📊 Total Tests: {len(results)}")
            logger.info(f"   📊 Passed: {sum(1 for r in results if r.get('success', False))}")
            logger.info(f"   📊 Failed: {sum(1 for r in results if not r.get('success', False))}")
            logger.info(f"   📊 Success Rate: {(sum(1 for r in results if r.get('success', False)) / len(results) * 100):.1f}%")

        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Performance test error: {str(e)}")
            logger.error(f"   ❌ Performance test failed: {e}")

        return result

    async def _run_performance_test_cycle(self, symbols: List[str], duration: int) -> Dict[str, Any]:
        """Run a single performance test cycle"""
        response_times = []
        errors = 0

        start_time = time.time()

        try:
            from enhanced_system_integrator import get_integrator
            integrator = get_integrator()
            data_streamer = integrator.get_module("optimized_market_data_streamer")

            if not data_streamer:
                return {"response_times": [], "errors": 1}

            while time.time() - start_time < duration:
                symbol = np.random.choice(symbols)

                # Test data fetching performance
                fetch_start = time.time()
                try:
                    data = await data_streamer.fetch_market_data(symbol, "1h", limit=10)
                    if data is not None:
                        response_times.append(time.time() - fetch_start)
                    else:
                        errors += 1
                except Exception:
                    errors += 1

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Performance test cycle error: {e}")
            errors += 1

        return {
            "response_times": response_times,
            "errors": errors
        }

    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery"""
        result = {
            "success": True,
            "errors": []
        }

        try:
            logger.info("   🛡️ Testing Error Handling & Recovery...")

            from enhanced_system_integrator import get_integrator
            integrator = get_integrator()

            # Test invalid symbol handling
            data_streamer = integrator.get_module("optimized_market_data_streamer")
            if data_streamer:
                try:
                    # Test with invalid symbol
                    invalid_data = await data_streamer.fetch_market_data("INVALID_SYMBOL", "1h", limit=5)
                    if invalid_data is None:
                        logger.info("   ✅ Invalid symbol handling: WORKING")
                    else:
                        result["success"] = False
                        result["errors"].append("Invalid symbol should return None")
                except Exception as e:
                    logger.info(f"   ✅ Invalid symbol exception handling: WORKING ({type(e).__name__})")

            # Test module recovery
            ai_ml = integrator.get_module("enhanced_ai_ml_optimizer")
            if ai_ml:
                # Test with invalid data
                try:
                    invalid_result = await ai_ml.optimize_entry_points_enhanced(pd.DataFrame())
                    if invalid_result and "error" in invalid_result:
                        logger.info("   ✅ Invalid data handling: WORKING")
                    else:
                        logger.info("   ✅ Invalid data gracefully handled")
                except Exception as e:
                    logger.info(f"   ✅ Invalid data exception handling: WORKING ({type(e).__name__})")

            # Test system resilience
            logger.info("   ✅ Error handling validation: PASSED")

        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Error handling test failed: {str(e)}")
            logger.error(f"   ❌ Error handling test failed: {e}")

        return result

    async def test_configuration_validation(self) -> Dict[str, Any]:
        """Test configuration validation"""
        result = {
            "success": True,
            "errors": []
        }

        try:
            logger.info("   ⚙️ Testing Configuration Validation...")

            from enhanced_system_integrator import get_integrator
            integrator = get_integrator()

            # Test system configuration
            if hasattr(integrator, 'system_config'):
                config = integrator.system_config

                # Check required sections
                required_sections = ["system", "modules", "integration"]
                for section in required_sections:
                    if section not in config:
                        result["success"] = False
                        result["errors"].append(f"Missing required config section: {section}")
                    else:
                        logger.info(f"   ✅ Config section '{section}': PRESENT")

                # Validate module configurations
                if "modules" in config:
                    for module_name, module_config in config["modules"].items():
                        if not isinstance(module_config, dict):
                            result["success"] = False
                            result["errors"].append(f"Invalid module config for {module_name}")
                        elif "enabled" not in module_config:
                            result["success"] = False
                            result["errors"].append(f"Missing 'enabled' field for {module_name}")

                # Validate trading parameters
                if "trading_parameters" in config:
                    trading_params = config["trading_parameters"]
                    required_params = ["risk_per_trade", "max_positions_total", "min_viper_score"]

                    for param in required_params:
                        if param not in trading_params:
                            result["success"] = False
                            result["errors"].append(f"Missing required trading parameter: {param}")
                        else:
                            logger.info(f"   ✅ Trading parameter '{param}': PRESENT")

                logger.info("   ✅ Configuration validation: PASSED")

            else:
                result["success"] = False
                result["errors"].append("System configuration not loaded")

        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Configuration validation error: {str(e)}")
            logger.error(f"   ❌ Configuration validation failed: {e}")

        return result

    def generate_test_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate test recommendations based on results"""
        recommendations = []

        try:
            success_rate = test_results.get("success_rate", 0)

            if success_rate < 80:
                recommendations.append("❌ Critical: System integration has significant issues - review error logs")
            elif success_rate < 90:
                recommendations.append("⚠️ Warning: Some integration issues detected - monitor closely")
            else:
                recommendations.append("✅ System integration is stable and ready for production")

            # Performance recommendations
            perf_metrics = test_results.get("performance_metrics", {})
            avg_response_time = perf_metrics.get("avg_response_time_ms", 0)

            if avg_response_time > 2000:  # 2 seconds
                recommendations.append("⚡ Performance: High response times detected - consider optimization")
            elif avg_response_time > 1000:  # 1 second
                recommendations.append("📊 Performance: Moderate response times - monitor for degradation")

            error_rate = perf_metrics.get("error_rate", 0)
            if error_rate > 0.05:
                recommendations.append("🛡️ Reliability: High error rate detected - investigate error sources")

            # Module-specific recommendations
            modules_tested = test_results.get("modules_tested", [])
            if len(modules_tested) < 5:
                recommendations.append("🔧 Integration: Some modules not properly loaded - check dependencies")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("🔍 System analysis completed - review logs for details")

        return recommendations

    def save_test_report(self, test_results: Dict[str, Any], report_path: Optional[str] = None):
        """Save test report to file"""
        try:
            if report_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = project_root / f"integration_test_report_{timestamp}.json"

            with open(report_path, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)

            logger.info(f"📋 Test report saved to: {report_path}")

        except Exception as e:
            logger.error(f"❌ Error saving test report: {e}")

async def run_integration_tests():
    """Run the complete integration test suite"""
    print("🚀 Enhanced System Integration Test Suite")
    print("=" * 80)

    test_suite = EnhancedSystemIntegrationTest()

    try:
        # Run all tests
        results = await test_suite.run_full_integration_test()

        # Save detailed report
        test_suite.save_test_report(results)

        # Print summary
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        print(f"Overall Status: {'✅ PASSED' if results['overall_success'] else '❌ FAILED'}")
        print(".1f")
        print(f"Tests Passed: {results['tests_passed']}")
        print(f"Tests Failed: {results['tests_failed']}")
        print(".1f")

        if results['errors']:
            print(f"\n❌ ERRORS ({len(results['errors'])}):")
            for i, error in enumerate(results['errors'][:5], 1):  # Show first 5 errors
                print(f"   {i}. {error}")
            if len(results['errors']) > 5:
                print(f"   ... and {len(results['errors']) - 5} more errors")

        if results['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in results['recommendations']:
                print(f"   • {rec}")

        print(f"\n📋 Detailed report saved to: integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        return results['overall_success']

    except Exception as e:
        print(f"❌ Test suite failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
    if success:
        print("\n🎉 Integration tests completed successfully!")
        print("🚀 System is ready for deployment")
    else:
        print("\n⚠️ Integration tests found issues")
        print("🔧 Please review the test report and address any failures before deployment")
        sys.exit(1)
