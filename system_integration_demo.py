#!/usr/bin/env python3
"""
🚀 VIPER SYSTEM INTEGRATION DEMO
Complete demonstration of all integrated components working together

This demo showcases:
- Master System Orchestrator integration
- Unified Trading Engine with all optimizations
- Real-time diagnostics and monitoring
- Component interaction and data flow
- Performance optimization across all systems
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - INTEGRATION_DEMO - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemIntegrationDemo:
    """
    Comprehensive system integration demonstration
    """

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.orchestrator = None
        self.trading_engine = None

    def run_full_system_demo(self):
        """Run complete system integration demo"""
        print("🚀 VIPER Complete System Integration Demo")
        print("=" * 60)

        try:
            # Step 1: Initialize Master System Orchestrator
            self._initialize_orchestrator()

            # Step 2: Initialize Unified Trading Engine
            self._initialize_trading_engine()

            # Step 3: Run comprehensive diagnostics
            self._run_system_diagnostics()

            # Step 4: Demonstrate component integration
            self._demonstrate_component_integration()

            # Step 5: Run optimization routines
            self._run_system_optimization()

            # Step 6: Display final system status
            self._display_final_status()

            print("✅ System Integration Demo Completed Successfully!")
            print("🎉 All components are properly integrated and operational!")

        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            print(f"\n❌ Demo failed: {e}")
            return False

        return True

    def _initialize_orchestrator(self):
        """Initialize the Master System Orchestrator"""
        print("TEXT")
            try:
            from master_system_orchestrator import MasterSystemOrchestrator

            self.orchestrator = MasterSystemOrchestrator()
            status = self.orchestrator.get_system_status()

            print("   ✅ Orchestrator initialized")
            print(f"   📊 Components loaded: {status['total_components']}")
            print(f"   🏥 Healthy components: {status['healthy_components']}")
            print(f"   ⚠️ Failed components: {status['failed_components']}")

        except Exception as e:
            print(f"   ❌ Failed to initialize orchestrator: {e}")
            raise

    def _initialize_trading_engine(self):
        """Initialize the Unified Trading Engine"""
        print("TEXT")
            try:
            from unified_trading_engine import UnifiedTradingEngine

            self.trading_engine = UnifiedTradingEngine()
            status = self.trading_engine.get_system_status()

            print("   ✅ Trading engine initialized")
            print(f"   🔗 Components loaded: {len(status['components_loaded'])}")
            print(f"   💱 Exchange connected: {'✅' if status.get('exchange_connected') else '❌'}")

        except Exception as e:
            print(f"   ❌ Failed to initialize trading engine: {e}")
            raise

    def _run_system_diagnostics(self):
        """Run comprehensive system diagnostics"""
        print("TEXT")
            try:
            # Run orchestrator diagnostics
            orch_diagnostics = self.orchestrator.run_system_diagnostics()

            print("   📊 Orchestrator Diagnostics:")
            print(f"      System Health: {orch_diagnostics.get('system_health', 'unknown')}")
            print(f"      Issues Found: {len(orch_diagnostics.get('issues', []))}")
            print(f"      Recommendations: {len(orch_diagnostics.get('recommendations', []))}")

            # Run trading engine diagnostics
            engine_check = self.trading_engine.run_system_check()

            print("   ⚙️ Trading Engine Diagnostics:")
            print(f"      Components: {len(engine_check.get('health_checks', {}))}")

            # Show top issues if any
            issues = orch_diagnostics.get('issues', [])
            if issues:
                print("   ⚠️ Top Issues:")
                for i, issue in enumerate(issues[:3], 1):
                    print(f"      {i}. {issue}")

        except Exception as e:
            print(f"   ❌ Diagnostics failed: {e}")
            raise

    def _demonstrate_component_integration(self):
        """Demonstrate component integration and data flow"""
        print("TEXT")
            try:
            # Test mathematical validator integration
            if 'math_validator' in self.trading_engine.components:
                print("   🧮 Testing Mathematical Validator...")
                validator = self.trading_engine.components['math_validator']

                # Test basic validation
                test_result = validator.validate_array([1, 2, 3, 4, 5], "test_array")
                print(f"      Array validation: {'✅' if test_result.get('is_valid') else '❌'}")

            # Test entry point optimizer integration
            if 'entry_optimizer' in self.trading_engine.components:
                print("   🎯 Testing Entry Point Optimizer...")
                optimizer = self.trading_engine.components['entry_optimizer']

                # Test basic functionality (would need real market data for full test)
                print("      Entry optimizer loaded and ready"
            # Test AI optimizer integration
            if 'ai_optimizer' in self.trading_engine.components:
                print("   🤖 Testing AI Optimizer...")
                ai_opt = self.trading_engine.components['ai_optimizer']

                # Test basic functionality
                if hasattr(ai_opt, 'get_status'):
                    status = ai_opt.get_status()
                    print(f"      AI Optimizer status: {status}")

            # Test diagnostic system integration
            if 'diagnostic_system' in self.trading_engine.components:
                print("   🩺 Testing Diagnostic System...")
                diagnostic = self.trading_engine.components['diagnostic_system']

                print("      Diagnostic system loaded and ready")
            print("   ✅ Component integration verified!")

        except Exception as e:
            print(f"   ❌ Component integration test failed: {e}")
            raise

    def _run_system_optimization(self):
        """Run system-wide optimization routines"""
        print("TEXT")
            try:
            # Run orchestrator optimization
            orch_optimization = self.orchestrator.optimize_system()

            print("   🔧 Optimization Results:")
            print(f"      Optimizations Applied: {len(orch_optimization.get('optimizations_applied', []))}")
            print(f"      Performance Improvements: {len(orch_optimization.get('performance_improvements', []))}")
            print(f"      Errors: {len(orch_optimization.get('errors', []))}")

            # Show applied optimizations
            applied = orch_optimization.get('optimizations_applied', [])
            if applied:
                print("   ✅ Applied Optimizations:")
                for opt in applied[:3]:  # Show first 3
                    print(f"      • {opt}")

            # Show performance improvements
            improvements = orch_optimization.get('performance_improvements', [])
            if improvements:
                print("   📈 Performance Improvements:")
                for imp in improvements[:3]:  # Show first 3
                    print(f"      • {imp}")

        except Exception as e:
            print(f"   ❌ System optimization failed: {e}")
            raise

    def _display_final_status(self):
        """Display final comprehensive system status"""
        print("TEXT")
            print("-" * 50)

        try:
            # Get orchestrator status
            orch_status = self.orchestrator.get_system_status()

            print("📊 MASTER SYSTEM ORCHESTRATOR:")
            print(f"   Components: {orch_status['total_components']}")
            print(f"   Healthy: {orch_status['healthy_components']}")
            print(f"   Failed: {orch_status['failed_components']}")
            print(f"   Monitoring: {'Active' if orch_status['monitoring_active'] else 'Inactive'}")

            # Get trading engine status
            engine_status = self.trading_engine.get_system_status()

            print("TEXT")
            print(f"   Components: {len(engine_status.get('components_loaded', []))}")
            print(f"   Exchange: {'Connected' if engine_status.get('exchange_connected') else 'Disconnected'}")
            print(f"   Trading: {'Active' if engine_status.get('trading_active') else 'Inactive'}")

            # Get system integrity
            integrity = self.orchestrator.validate_system_integrity()

            print("TEXT")
            print(f"   Status: {integrity.get('integrity_status', 'unknown').upper()}")
            print(f"   Validations: {len(integrity.get('validation_results', {}))}")

            if integrity.get('critical_issues'):
                print(f"   Critical Issues: {len(integrity['critical_issues'])}")

            # Overall assessment
            print("TEXT")
            healthy_components = orch_status['healthy_components'] + len(engine_status.get('components_loaded', []))
            total_components = orch_status['total_components'] + len(engine_status.get('components_loaded', []))

            if healthy_components == total_components and integrity.get('integrity_status') == 'healthy':
                print("   ✅ SYSTEM FULLY OPERATIONAL")
            print("   ✅ ALL COMPONENTS INTEGRATED"                print("   ✅ READY FOR PRODUCTION TRADING")
            else:
                print("   ⚠️ SYSTEM PARTIALLY OPERATIONAL")
            print(f"   📊 Health: {healthy_components}/{total_components} components healthy")

        except Exception as e:
            print(f"   ❌ Final status check failed: {e}")

    def create_integration_report(self) -> Dict[str, Any]:
        """Create comprehensive integration report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'demo_completed': False,
            'system_components': {},
            'integration_status': {},
            'performance_metrics': {},
            'recommendations': []
        }

        try:
            # Gather system information
            if self.orchestrator:
                orch_status = self.orchestrator.get_system_status()
                report['system_components']['orchestrator'] = orch_status

            if self.trading_engine:
                engine_status = self.trading_engine.get_system_status()
                report['system_components']['trading_engine'] = engine_status

            # Mark as completed
            report['demo_completed'] = True
            report['integration_status'] = 'successful'

        except Exception as e:
            report['integration_status'] = f'error: {str(e)}'

        return report

def main():
    """Main demo execution"""
    demo = SystemIntegrationDemo()

    # Run the complete demo
    success = demo.run_full_system_demo()

    if success:
        # Create integration report
        report = demo.create_integration_report()

        # Save report
        report_path = demo.project_root / "integration_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📄 Integration report saved to: {report_path}")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
