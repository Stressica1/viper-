#!/usr/bin/env python3
"""
# Rocket VIPER INTEGRATED SYSTEM LAUNCHER
One-click launcher for the complete integrated VIPER trading system

This launcher provides:
    pass
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
from typing import Optional"""

class IntegratedSystemLauncher:
    """
    Launcher for the complete integrated VIPER system
    """"""

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
        """Launch the system in the specified mode""""""

        if mode not in self.available_modes:
            print(f"Available modes: {', '.join(self.available_modes.keys())}")
            return False

        print(f"📝 Description: {self.available_modes[mode]}")

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
            return False

    def _launch_demo(self) -> bool:
        """Launch system integration demo""""""

        try:
            # Import and run demo
    from system_integration_demo import SystemIntegrationDemo

            demo = SystemIntegrationDemo()
            success = demo.run_full_system_demo()

            if success:
                print("# Party All system components are properly integrated!")
            else:
                print("# X Demo failed - check system logs for details")
            return success

        except ImportError as e:
            print("# Idea Make sure all required modules are installed")
            return False
        except Exception as e:
            return False

    def _launch_diagnostics(self) -> bool:
        """Launch comprehensive diagnostics"""
        print("# Search Running Comprehensive System Diagnostics...")
        print("This will scan all components and provide detailed health reports")

        try:
            # Import master diagnostic scanner
            from scripts.master_diagnostic_scanner import MasterDiagnosticScanner

            scanner = MasterDiagnosticScanner()
            results = scanner.run_full_diagnostic()

            print(f"   System Health: {results.get('overall_health', 'unknown')}")
            print(f"   Components Scanned: {len(results.get('component_results', {}))}")
            print(f"   Issues Found: {len(results.get('issues', []))}")

            if results.get('issues'):
            for i, issue in enumerate(results['issues'][:5], 1)
            if results.get('recommendations'):
            for i, rec in enumerate(results['recommendations'][:5], 1)
            return True

        except Exception as e:
            return False

    def _launch_monitor(self) -> bool:
        """Launch real-time monitoring"""
        print("# Chart Starting Real-Time System Monitoring...")
        print("This will continuously monitor system health")

        try:
            # Import orchestrator
            from master_system_orchestrator import MasterSystemOrchestrator

            orchestrator = MasterSystemOrchestrator()

            # Start monitoring
            orchestrator.start_monitoring()

            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            orchestrator.stop_monitoring()
                return True

        except Exception as e:
            return False

    def _launch_trading(self) -> bool:
        """Launch live trading system"""
        print("# Warning  WARNING: This will execute real trades!")
        print("Make sure you have sufficient funds and understand the risks")

        # Safety confirmation
        confirm = input("Are you sure you want to start live trading? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y']:
            return False

        try:
            # Import trading engine
            import asyncio
            from unified_trading_engine import UnifiedTradingEngine

            async def run_trading():
                engine = UnifiedTradingEngine()
                await engine.start_trading_engine()

            asyncio.run(run_trading())

        except KeyboardInterrupt:
            return True
        except Exception as e:
            return False

    def _launch_optimize(self) -> bool:
        """Launch system optimization"""
        print("This will optimize all system components for maximum performance")

        try:
            # Import orchestrator
            from master_system_orchestrator import MasterSystemOrchestrator

            orchestrator = MasterSystemOrchestrator()

            # Run optimization
            results = orchestrator.optimize_system()

            print(f"   Optimizations Applied: {len(results.get('optimizations_applied', []))}")
            print(f"   Performance Improvements: {len(results.get('performance_improvements', []))}")
            print(f"   Errors: {len(results.get('errors', []))}")

            # Show details
            if results.get('optimizations_applied'):
            for opt in results['optimizations_applied'][:5]
            if results.get('performance_improvements'):
            for imp in results['performance_improvements'][:5]
            return True

        except Exception as e:
            return False

    def _launch_status(self) -> bool:
        """Display current system status""""""

        try:
            # Import orchestrator
    from master_system_orchestrator import MasterSystemOrchestrator

            orchestrator = MasterSystemOrchestrator()
            status = orchestrator.get_system_status()

            print(f"   Components: {status['total_components']}")
            print(f"   Healthy: {status['healthy_components']}")
            print(f"   Failed: {status['failed_components']}")
            print(f"   Monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")

            # Try to get trading engine status
            try:
                from unified_trading_engine import UnifiedTradingEngine
                engine = UnifiedTradingEngine()
                engine_status = engine.get_system_status()

                print(f"   Components: {len(engine_status.get('components_loaded', []))}")
                print(f"   Exchange: {'Connected' if engine_status.get('exchange_connected') else 'Disconnected'}")
                print(f"   Trading: {'Active' if engine_status.get('trading_active') else 'Inactive'}")

            except Exception as e:
                print(f"\n⚡ Trading Engine: Status unavailable ({e})")

            # Overall assessment
            total_healthy = status['healthy_components']
            total_components = status['total_components']

            if total_healthy == total_components:
                pass
            else:
                print(f"   # Warning SYSTEM PARTIALLY OPERATIONAL ({total_healthy}/{total_components} healthy)")

            return True

        except Exception as e:
            return False

    def show_help(self):
        """Show available launch modes"""

        for mode, description in self.available_modes.items():
        print("  python launch_integrated_system.py <mode>")
        print("  python launch_integrated_system.py demo")
        print("  python launch_integrated_system.py diagnostics")
        print("  python launch_integrated_system.py monitor")
        print("  python launch_integrated_system.py status")

def main():
    """Main launcher function"""
    launcher = IntegratedSystemLauncher()"""

    if len(sys.argv) < 2:
        launcher.show_help()
        return 0

    mode = sys.argv[1].lower()

    # Launch in specified mode
    success = launcher.launch(mode)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
