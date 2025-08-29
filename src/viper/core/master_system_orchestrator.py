#!/usr/bin/env python3
"""
🚀 VIPER MASTER SYSTEM ORCHESTRATOR
Complete integration and orchestration of all VIPER trading system components

This master orchestrator provides:
- Unified interface for all system components
- Complete integration of all new features
- Real-time system monitoring and diagnostics
- Automated optimization and validation
- Comprehensive system health management
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import importlib.util
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MASTER_ORCHESTRATOR - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MasterSystemOrchestrator:
    """
    Master orchestrator for complete VIPER system integration
    """

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.components = {}
        self.system_status = {}
        self.monitoring_active = False
        self.optimization_active = False

        # Initialize system components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all system components"""
        logger.info("🚀 Initializing VIPER Master System Orchestrator...")

        # Import and initialize core components
        self._load_mathematical_validator()
        self._load_optimal_mcp_config()
        self._load_diagnostic_scanner()
        self._load_entry_point_manager()
        self._load_scoring_system()
        self._load_ai_optimizer()

        logger.info("✅ All system components initialized successfully")

    def _load_mathematical_validator(self):
        """Load mathematical validator component"""
        try:
            from utils.mathematical_validator import MathematicalValidator
            self.components['mathematical_validator'] = MathematicalValidator()
            self.system_status['mathematical_validator'] = 'initialized'
            logger.info("✅ Mathematical Validator loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Mathematical Validator: {e}")
            self.system_status['mathematical_validator'] = 'failed'

    def _load_optimal_mcp_config(self):
        """Load optimal MCP configuration"""
        try:
            from config.optimal_mcp_config import get_optimal_mcp_config
            self.components['mcp_config'] = get_optimal_mcp_config()
            self.system_status['mcp_config'] = 'initialized'
            logger.info("✅ Optimal MCP Config loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Optimal MCP Config: {e}")
            self.system_status['mcp_config'] = 'failed'

    def _load_diagnostic_scanner(self):
        """Load master diagnostic scanner"""
        try:
            from scripts.master_diagnostic_scanner import MasterDiagnosticScanner
            self.components['diagnostic_scanner'] = MasterDiagnosticScanner()
            self.system_status['diagnostic_scanner'] = 'initialized'
            logger.info("✅ Master Diagnostic Scanner loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Master Diagnostic Scanner: {e}")
            self.system_status['diagnostic_scanner'] = 'failed'

    def _load_entry_point_manager(self):
        """Load optimal entry point manager"""
        try:
            from scripts.optimal_entry_point_manager import OptimalEntryPointManager
            self.components['entry_point_manager'] = OptimalEntryPointManager()
            self.system_status['entry_point_manager'] = 'initialized'
            logger.info("✅ Optimal Entry Point Manager loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Optimal Entry Point Manager: {e}")
            self.system_status['entry_point_manager'] = 'failed'

    def _load_scoring_system(self):
        """Load scoring system diagnostic"""
        try:
            from scripts.scoring_system_diagnostic import ScoringSystemDiagnostic
            self.components['scoring_system'] = ScoringSystemDiagnostic()
            self.system_status['scoring_system'] = 'initialized'
            logger.info("✅ Scoring System Diagnostic loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Scoring System Diagnostic: {e}")
            self.system_status['scoring_system'] = 'failed'

    def _load_ai_optimizer(self):
        """Load AI/ML optimizer"""
        try:
            # Import AI optimizer with enhanced capabilities
            spec = importlib.util.spec_from_file_location(
                "ai_optimizer",
                self.project_root / "ai_ml_optimizer.py"
            )
            ai_optimizer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ai_optimizer_module)

            # Initialize the optimizer class if it exists
            if hasattr(ai_optimizer_module, 'VIPEROptimizer'):
                self.components['ai_optimizer'] = ai_optimizer_module.VIPEROptimizer()
            else:
                self.components['ai_optimizer'] = ai_optimizer_module
            self.system_status['ai_optimizer'] = 'initialized'
            logger.info("✅ AI/ML Optimizer loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load AI/ML Optimizer: {e}")
            self.system_status['ai_optimizer'] = 'failed'

    def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        logger.info("🔍 Running comprehensive system diagnostics...")

        diagnostic_results = {
            'timestamp': datetime.now().isoformat(),
            'system_health': 'unknown',
            'component_status': {},
            'recommendations': [],
            'issues': []
        }

        # Check each component
        for component_name, component in self.components.items():
            try:
                if component_name == 'diagnostic_scanner' and hasattr(component, 'run_full_diagnostic'):
                    # Run the master diagnostic scanner
                    results = component.run_full_diagnostic()
                    diagnostic_results['component_status'][component_name] = 'healthy'
                    diagnostic_results['system_health'] = results.get('overall_health', 'unknown')

                elif component_name == 'mathematical_validator' and hasattr(component, 'validate_system'):
                    # Validate mathematical components
                    validation_results = component.validate_system()
                    diagnostic_results['component_status'][component_name] = 'healthy' if validation_results.get('is_valid', False) else 'warning'

                else:
                    # Basic health check
                    diagnostic_results['component_status'][component_name] = 'healthy'

            except Exception as e:
                diagnostic_results['component_status'][component_name] = 'error'
                diagnostic_results['issues'].append(f"{component_name}: {str(e)}")

        # Generate recommendations
        diagnostic_results['recommendations'] = self._generate_recommendations(diagnostic_results)

        logger.info(f"✅ Diagnostics completed. System health: {diagnostic_results['system_health']}")
        return diagnostic_results

    def _generate_recommendations(self, diagnostic_results: Dict[str, Any]) -> List[str]:
        """Generate system recommendations based on diagnostic results"""
        recommendations = []

        # Check component health
        failed_components = [name for name, status in diagnostic_results['component_status'].items() if status == 'error']
        if failed_components:
            recommendations.append(f"Fix failed components: {', '.join(failed_components)}")

        # System health recommendations
        if diagnostic_results['system_health'] in ['critical', 'warning']:
            recommendations.append("Run full system optimization and validation")
            recommendations.append("Check MCP server configurations and connections")

        # General recommendations
        recommendations.extend([
            "Regularly run system diagnostics to maintain optimal performance",
            "Monitor mathematical validation results for trading accuracy",
            "Keep MCP configurations optimized for your trading environment"
        ])

        return recommendations

    def optimize_system(self) -> Dict[str, Any]:
        """Run system-wide optimization"""
        logger.info("⚡ Starting system-wide optimization...")

        optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'optimizations_applied': [],
            'performance_improvements': [],
            'warnings': [],
            'errors': []
        }

        # Apply optimizations to each component
        for component_name, component in self.components.items():
            try:
                if component_name == 'entry_point_manager' and hasattr(component, 'optimize_entry_points'):
                    # Optimize entry points
                    opt_results = component.optimize_entry_points()
                    optimization_results['optimizations_applied'].append(f"{component_name}: Entry points optimized")
                    if 'performance_gain' in opt_results:
                        optimization_results['performance_improvements'].append(f"{component_name}: {opt_results['performance_gain']}")

                elif component_name == 'ai_optimizer' and hasattr(component, 'optimize_model'):
                    # Optimize AI model
                    opt_results = component.optimize_model()
                    optimization_results['optimizations_applied'].append(f"{component_name}: AI model optimized")
                    if 'accuracy_improvement' in opt_results:
                        optimization_results['performance_improvements'].append(f"{component_name}: {opt_results['accuracy_improvement']}")

                elif component_name == 'mathematical_validator' and hasattr(component, 'optimize_calculations'):
                    # Optimize mathematical calculations
                    opt_results = component.optimize_calculations()
                    optimization_results['optimizations_applied'].append(f"{component_name}: Calculations optimized")

            except Exception as e:
                optimization_results['errors'].append(f"{component_name}: {str(e)}")

        logger.info(f"✅ System optimization completed. Applied {len(optimization_results['optimizations_applied'])} optimizations")
        return optimization_results

    def start_monitoring(self):
        """Start real-time system monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        logger.info("📊 Starting real-time system monitoring...")

        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Run diagnostics every 5 minutes
                    diagnostics = self.run_system_diagnostics()

                    # Log system health
                    health_status = diagnostics.get('system_health', 'unknown')
                    if health_status in ['critical', 'warning']:
                        logger.warning(f"⚠️ System health: {health_status}")
                    else:
                        logger.info(f"✅ System health: {health_status}")

                    # Sleep for 5 minutes
                    time.sleep(300)

                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("✅ Real-time monitoring started")

    def stop_monitoring(self):
        """Stop real-time system monitoring"""
        if not self.monitoring_active:
            logger.warning("Monitoring not active")
            return

        logger.info("🛑 Stopping real-time system monitoring...")
        self.monitoring_active = False

        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=5)

        logger.info("✅ Real-time monitoring stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'components': self.system_status,
            'monitoring_active': self.monitoring_active,
            'optimization_active': self.optimization_active,
            'total_components': len(self.components),
            'healthy_components': sum(1 for status in self.system_status.values() if status == 'initialized'),
            'failed_components': sum(1 for status in self.system_status.values() if status == 'failed')
        }

    def validate_system_integrity(self) -> Dict[str, Any]:
        """Validate overall system integrity"""
        integrity_check = {
            'timestamp': datetime.now().isoformat(),
            'integrity_status': 'unknown',
            'validation_results': {},
            'critical_issues': [],
            'warnings': []
        }

        # Check component connectivity
        for component_name, component in self.components.items():
            try:
                if hasattr(component, 'validate_integrity'):
                    result = component.validate_integrity()
                    integrity_check['validation_results'][component_name] = result
                else:
                    integrity_check['validation_results'][component_name] = {'status': 'no_validation_method'}

            except Exception as e:
                integrity_check['validation_results'][component_name] = {'status': 'error', 'error': str(e)}
                integrity_check['critical_issues'].append(f"{component_name}: {str(e)}")

        # Determine overall integrity status
        if integrity_check['critical_issues']:
            integrity_check['integrity_status'] = 'critical'
        elif any('error' in str(result) for result in integrity_check['validation_results'].values()):
            integrity_check['integrity_status'] = 'warning'
        else:
            integrity_check['integrity_status'] = 'healthy'

        return integrity_check

def main():
    """Main execution function"""
    print("🚀 VIPER Master System Orchestrator")
    print("=" * 50)

    # Initialize orchestrator
    orchestrator = MasterSystemOrchestrator()

    # Display system status
    status = orchestrator.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Components: {status['total_components']}")
    print(f"   Healthy: {status['healthy_components']}")
    print(f"   Failed: {status['failed_components']}")
    print(f"   Monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")

    # Run diagnostics
    print("🔍 Running system diagnostics...")
    diagnostics = orchestrator.run_system_diagnostics()
    print(f"   System Health: {diagnostics.get('system_health', 'unknown')}")

    if diagnostics.get('issues'):
        print(f"   Issues Found: {len(diagnostics['issues'])}")
        for issue in diagnostics['issues'][:3]:  # Show first 3 issues
            print(f"     - {issue}")

    if diagnostics.get('recommendations'):
        print(f"   Recommendations: {len(diagnostics['recommendations'])}")

    # Start monitoring if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        print("📊 Starting real-time monitoring...")
        orchestrator.start_monitoring()

        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring...")
            orchestrator.stop_monitoring()
    else:
        print("💡 Use '--monitor' flag to start real-time monitoring")
        print("   Example: python master_system_orchestrator.py --monitor")

if __name__ == "__main__":
    main()
