#!/usr/bin/env python3
"""
🚀 VIPER INTEGRATED SYSTEM LAUNCHER
One-click launcher for the complete integrated VIPER trading system

This launcher provides:
- Quick system initialization and validation
- Choice between different operational modes
- Real-time monitoring and diagnostics
- Easy access to all integrated components
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional

class IntegratedSystemLauncher:
    """
    Launcher for the complete integrated VIPER system
    """

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.available_modes = {
            'demo': 'Run system integration demo',
            'diagnostics': 'Run comprehensive system diagnostics',
            'monitor': 'Start real-time system monitoring',
            'trading': 'Start live trading with all optimizations',
            'optimize': 'Run system optimization routines',
            'status': 'Display current system status'
        }

    def launch(self, mode: str, **kwargs):
        """Launch the system in the specified mode"""
        print("🚀 VIPER Integrated System Launcher")
        print("=" * 50)

        if mode not in self.available_modes:
            print(f"❌ Invalid mode: {mode}")
            print(f"Available modes: {', '.join(self.available_modes.keys())}")
            return False

        print(f"🎯 Launching mode: {mode}")
        print(f"📝 Description: {self.available_modes[mode]}")
        print()

        try:
            if mode == 'demo':
                return self._launch_demo()
            elif mode == 'diagnostics':
                return self._launch_diagnostics()
            elif mode == 'monitor':
                return self._launch_monitor()
            elif mode == 'trading':
                return self._launch_trading()
            elif mode == 'optimize':
                return self._launch_optimize()
            elif mode == 'status':
                return self._launch_status()

        except Exception as e:
            print(f"❌ Launch failed: {e}")
            return False

    def _launch_demo(self) -> bool:
        """Launch system integration demo"""
        print("🎭 Starting System Integration Demo...")
print("This will test all components and their interactions")
        print()

        try:
            # Import and run demo
            from system_integration_demo import SystemIntegrationDemo

            demo = SystemIntegrationDemo()
            success = demo.run_full_system_demo()

            if success:
                print("✅ Demo completed successfully!")
print("🎉 All system components are properly integrated!")
            else:
                print("❌ Demo failed - check system logs for details")
            return success

        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Make sure all required modules are installed")
            return False
        except Exception as e:
            print(f"❌ Demo execution failed: {e}")
            return False

    def _launch_diagnostics(self) -> bool:
        """Launch comprehensive diagnostics"""
        print("🔍 Running Comprehensive System Diagnostics...")
print("This will scan all components and provide detailed health reports")
        print()

        try:
            # Import master diagnostic scanner
            from scripts.master_diagnostic_scanner import MasterDiagnosticScanner

            scanner = MasterDiagnosticScanner()
            results = scanner.run_full_diagnostic()

            print("📊 Diagnostic Results:")
            print(f"   System Health: {results.get('overall_health', 'unknown')}")
            print(f"   Components Scanned: {len(results.get('component_results', {}))}")
            print(f"   Issues Found: {len(results.get('issues', []))}")

            if results.get('issues'):
                print("⚠️ Issues Detected:")
            for i, issue in enumerate(results['issues'][:5], 1):
                    print(f"   {i}. {issue}")

            if results.get('recommendations'):
                print("💡 Recommendations:")
            for i, rec in enumerate(results['recommendations'][:5], 1):
                    print(f"   {i}. {rec}")

            return True

        except Exception as e:
            print(f"❌ Diagnostics failed: {e}")
            return False

    def _launch_monitor(self) -> bool:
        """Launch real-time monitoring"""
        print("📊 Starting Real-Time System Monitoring...")
print("This will continuously monitor system health")
        print("Press Ctrl+C to stop monitoring")
        print()

        try:
            # Import orchestrator
            from master_system_orchestrator import MasterSystemOrchestrator

            orchestrator = MasterSystemOrchestrator()

            # Start monitoring
            print("🔄 Monitoring active... (Ctrl+C to stop)")
            orchestrator.start_monitoring()

            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("🛑 Stopping monitoring...")
            orchestrator.stop_monitoring()
                print("✅ Monitoring stopped")
                return True

        except Exception as e:
            print(f"❌ Monitoring failed: {e}")
            return False

    def _launch_trading(self) -> bool:
        """Launch live trading system"""
        print("💰 Starting Live Trading System...")
print("⚠️  WARNING: This will execute real trades!")
        print("Make sure you have sufficient funds and understand the risks")
        print()

        # Safety confirmation
        confirm = input("Are you sure you want to start live trading? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y']:
            print("❌ Trading launch cancelled")
            return False

        try:
            # Import trading engine
            import asyncio
            from unified_trading_engine import UnifiedTradingEngine

            async def run_trading():
                engine = UnifiedTradingEngine()
                await engine.start_trading_engine()

            print("🚀 Launching trading engine...")
            asyncio.run(run_trading())

        except KeyboardInterrupt:
            print("🛑 Trading stopped by user")
            return True
        except Exception as e:
            print(f"❌ Trading system failed: {e}")
            return False

    def _launch_optimize(self) -> bool:
        """Launch system optimization"""
        print("⚡ Running System Optimization...")
print("This will optimize all system components for maximum performance")
        print()

        try:
            # Import orchestrator
            from master_system_orchestrator import MasterSystemOrchestrator

            orchestrator = MasterSystemOrchestrator()

            # Run optimization
            print("🔧 Optimizing system components...")
            results = orchestrator.optimize_system()

            print("📊 Optimization Results:")
            print(f"   Optimizations Applied: {len(results.get('optimizations_applied', []))}")
            print(f"   Performance Improvements: {len(results.get('performance_improvements', []))}")
            print(f"   Errors: {len(results.get('errors', []))}")

            # Show details
            if results.get('optimizations_applied'):
                print("✅ Applied Optimizations:")
            for opt in results['optimizations_applied'][:5]:
                    print(f"   • {opt}")

            if results.get('performance_improvements'):
                print("📈 Performance Improvements:")
            for imp in results['performance_improvements'][:5]:
                    print(f"   • {imp}")

            return True

        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return False

    def _launch_status(self) -> bool:
        """Display current system status"""
        print("📊 Current System Status")
        print("=" * 30)

        try:
            # Import orchestrator
            from master_system_orchestrator import MasterSystemOrchestrator

            orchestrator = MasterSystemOrchestrator()
            status = orchestrator.get_system_status()

            print(f"📊 Master Orchestrator:")
            print(f"   Components: {status['total_components']}")
            print(f"   Healthy: {status['healthy_components']}")
            print(f"   Failed: {status['failed_components']}")
            print(f"   Monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")

            # Try to get trading engine status
            try:
                from unified_trading_engine import UnifiedTradingEngine
                engine = UnifiedTradingEngine()
                engine_status = engine.get_system_status()

                print(f"\n⚡ Trading Engine:")
                print(f"   Components: {len(engine_status.get('components_loaded', []))}")
                print(f"   Exchange: {'Connected' if engine_status.get('exchange_connected') else 'Disconnected'}")
                print(f"   Trading: {'Active' if engine_status.get('trading_active') else 'Inactive'}")

            except Exception as e:
                print(f"\n⚡ Trading Engine: Status unavailable ({e})")

            # Overall assessment
            total_healthy = status['healthy_components']
            total_components = status['total_components']

            print(f"\n🎯 Overall Status:")
            if total_healthy == total_components:
                print("   ✅ SYSTEM FULLY OPERATIONAL")
            else:
                print(f"   ⚠️ SYSTEM PARTIALLY OPERATIONAL ({total_healthy}/{total_components} healthy)")

            return True

        except Exception as e:
            print(f"❌ Status check failed: {e}")
            return False

    def show_help(self):
        """Show available launch modes"""
        print("🚀 VIPER Integrated System Launcher")
        print("=" * 50)
        print("Available modes:")
        print()

        for mode, description in self.available_modes.items():
            print(f"  {mode:12} - {description}")

        print()
        print("Usage:")
print("  python launch_integrated_system.py <mode>")
        print()
        print("Examples:")
print("  python launch_integrated_system.py demo")
        print("python launch_integrated_system.py diagnostics")
print("  python launch_integrated_system.py monitor")
        print("  python launch_integrated_system.py status")

def main():
    """Main launcher function"""
    launcher = IntegratedSystemLauncher()

    if len(sys.argv) < 2:
        launcher.show_help()
        return 0

    mode = sys.argv[1].lower()

    # Launch in specified mode
    success = launcher.launch(mode)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
