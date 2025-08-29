#!/usr/bin/env python3
"""
# Rocket DEPLOYMENT CHECKLIST & ROLLBACK PROCEDURES
Comprehensive deployment guide for Enhanced VIPER Trading System

This checklist ensures:
- Safe and gradual system deployment
- Comprehensive pre-deployment validation
- Emergency rollback procedures
- Production monitoring setup
- Performance benchmarking
- Risk management validation
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DEPLOYMENT_CHECKLIST - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentChecklist:
    """Comprehensive deployment checklist and rollback system"""

    def __init__(self):
        self.checklist_items = []
        self.rollback_procedures = []
        self.deployment_log = []
        self.backup_paths = {}

        # Create deployment directory
        self.deployment_dir = project_root / "deployment_backup"
        self.deployment_dir.mkdir(exist_ok=True)

        logger.info("📋 Deployment Checklist initialized")

    def run_full_deployment_checklist(self) -> Dict[str, Any]:
        """Run complete deployment checklist"""
        logger.info("# Rocket Starting Enhanced VIPER Deployment Checklist")
        logger.info("=" * 80)

        checklist_results = {
            "checklist_start_time": datetime.now().isoformat(),
            "pre_deployment_checks": {},
            "integration_validation": {},
            "performance_validation": {},
            "security_validation": {},
            "deployment_readiness": {},
            "rollback_plan": {},
            "overall_status": "UNKNOWN"
        }

        try:
            # 1. Pre-deployment system checks
            logger.info("📋 Step 1: Pre-deployment System Checks")
            checklist_results["pre_deployment_checks"] = self._run_pre_deployment_checks()

            # 2. Integration validation
            logger.info("📋 Step 2: Integration Validation")
            checklist_results["integration_validation"] = self._run_integration_validation()

            # 3. Performance validation
            logger.info("📋 Step 3: Performance Validation")
            checklist_results["performance_validation"] = self._run_performance_validation()

            # 4. Security validation
            logger.info("📋 Step 4: Security Validation")
            checklist_results["security_validation"] = self._run_security_validation()

            # 5. Deployment readiness assessment
            logger.info("📋 Step 5: Deployment Readiness Assessment")
            checklist_results["deployment_readiness"] = self._assess_deployment_readiness(
                checklist_results
            )

            # 6. Rollback plan creation
            logger.info("📋 Step 6: Rollback Plan Creation")
            checklist_results["rollback_plan"] = self._create_rollback_plan()

            # Overall assessment
            checklist_results["overall_status"] = self._calculate_overall_status(checklist_results)

        except Exception as e:
            logger.error(f"# X Deployment checklist failed: {e}")
            checklist_results["error"] = str(e)
            checklist_results["overall_status"] = "FAILED"

        # Save checklist results
        self._save_checklist_results(checklist_results)

        logger.info("=" * 80)
        logger.info(f"# Target Deployment Checklist Complete: {checklist_results['overall_status']}")

        return checklist_results

    def _run_pre_deployment_checks(self) -> Dict[str, Any]:
        """Run pre-deployment system checks"""
        checks = {}

        try:
            # System requirements check
            checks["system_requirements"] = self._check_system_requirements()

            # Dependency validation
            checks["dependencies"] = self._check_dependencies()

            # Configuration validation
            checks["configuration"] = self._check_configuration_files()

            # Database/API connectivity
            checks["connectivity"] = self._check_system_connectivity()

            # Resource availability
            checks["resources"] = self._check_resource_availability()

            # Backup creation
            checks["backup"] = self._create_system_backup()

        except Exception as e:
            logger.error(f"# X Pre-deployment checks failed: {e}")
            checks["error"] = str(e)

        return checks

    def _check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements"""
        requirements = {
            "python_version": {"required": "3.8+", "current": f"{sys.version_info.major}.{sys.version_info.minor}"},
            "memory_gb": {"required": 8, "current": self._get_system_memory_gb()},
            "disk_space_gb": {"required": 10, "current": self._get_available_disk_space_gb()},
            "cpu_cores": {"required": 4, "current": os.cpu_count() or 0}
        }

        # Validate requirements
        all_passed = True
        for req_name, req_data in requirements.items():
            if req_name == "python_version":
                required_version = tuple(map(int, req_data["required"].replace("+", "").split(".")))
                current_version = (sys.version_info.major, sys.version_info.minor)
                requirements[req_name]["passed"] = current_version >= required_version
            else:
                requirements[req_name]["passed"] = req_data["current"] >= req_data["required"]

            if not requirements[req_name]["passed"]:
                all_passed = False

        return {
            "requirements": requirements,
            "all_passed": all_passed,
            "summary": f"{sum(1 for r in requirements.values() if r.get('passed', False))}/{len(requirements)} requirements met"
        }

    def _get_system_memory_gb(self) -> float:
        """Get system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            return 0.0

    def _get_available_disk_space_gb(self) -> float:
        """Get available disk space in GB"""
        try:
            import shutil
            total, used, free = shutil.disk_usage('/')
            return free / (1024**3)
        except Exception:
            return 0.0

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check Python package dependencies"""
        required_packages = [
            "numpy", "pandas", "ccxt", "asyncio", "json", "pathlib",
            "sklearn", "matplotlib", "seaborn", "scipy"
        ]

        dependency_status = {}
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
                dependency_status[package] = "AVAILABLE"
            except ImportError:
                dependency_status[package] = "MISSING"
                missing_packages.append(package)

        return {
            "dependency_status": dependency_status,
            "missing_packages": missing_packages,
            "all_available": len(missing_packages) == 0,
            "summary": f"{len(required_packages) - len(missing_packages)}/{len(required_packages)} dependencies available"
        }

    def _check_configuration_files(self) -> Dict[str, Any]:
        """Check configuration files"""
        config_files = [
            "enhanced_system_config.json",
            "config/enhanced_ai_ml_config.json",
            "config/enhanced_technical_config.json",
            "config/enhanced_risk_config.json",
            "config/optimized_data_streamer_config.json",
            "config/performance_monitoring_config.json"
        ]

        config_status = {}

        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        json.load(f)
                    config_status[config_file] = "VALID"
                except Exception as e:
                    config_status[config_file] = f"INVALID: {e}"
            else:
                config_status[config_file] = "MISSING"

        valid_configs = sum(1 for status in config_status.values() if status == "VALID")
        total_configs = len(config_files)

        return {
            "config_status": config_status,
            "all_valid": valid_configs == total_configs,
            "summary": f"{valid_configs}/{total_configs} configuration files valid"
        }

    def _check_system_connectivity(self) -> Dict[str, Any]:
        """Check system connectivity"""
        connectivity_checks = {}

        # Check internet connectivity
        try:
            import requests
            response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
            connectivity_checks["internet"] = "CONNECTED" if response.status_code == 200 else "FAILED"
        except Exception:
            connectivity_checks["internet"] = "FAILED"

        # Check Bitget API connectivity
        try:
            import ccxt
            exchange = ccxt.bitget()
            exchange.load_markets()
            connectivity_checks["bitget_api"] = "CONNECTED"
        except Exception as e:
            connectivity_checks["bitget_api"] = f"FAILED: {e}"

        # Check Redis connectivity (if configured)
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            try:
                import redis
                r = redis.from_url(redis_url)
                r.ping()
                connectivity_checks["redis"] = "CONNECTED"
            except Exception as e:
                connectivity_checks["redis"] = f"FAILED: {e}"
        else:
            connectivity_checks["redis"] = "NOT_CONFIGURED"

        all_connected = all(
            status in ["CONNECTED", "NOT_CONFIGURED"]
            for status in connectivity_checks.values()
        )

        return {
            "connectivity_checks": connectivity_checks,
            "all_connected": all_connected,
            "summary": f"{sum(1 for s in connectivity_checks.values() if s == 'CONNECTED')}/{len(connectivity_checks)} connections successful"
        }

    def _check_resource_availability(self) -> Dict[str, Any]:
        """Check resource availability"""
        try:
            import psutil

            resources = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3)
            }

            # Check thresholds
            cpu_ok = resources["cpu_percent"] < 80
            memory_ok = resources["memory_percent"] < 85
            disk_ok = resources["disk_usage_percent"] < 90
            memory_available_ok = resources["available_memory_gb"] > 4

            all_ok = cpu_ok and memory_ok and disk_ok and memory_available_ok

            return {
                "resources": resources,
                "thresholds_met": {
                    "cpu": cpu_ok,
                    "memory": memory_ok,
                    "disk": disk_ok,
                    "memory_available": memory_available_ok
                },
                "all_ok": all_ok,
                "summary": f"{sum([cpu_ok, memory_ok, disk_ok, memory_available_ok])}/4 resource thresholds met"
            }

        except Exception as e:
            return {
                "error": str(e),
                "all_ok": False,
                "summary": "Resource check failed"
            }

    def _create_system_backup(self) -> Dict[str, Any]:
        """Create system backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.deployment_dir / f"backup_{timestamp}"

            # Create backup directory
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Backup configuration files
            config_files = [
                "enhanced_system_config.json",
                ".env",
                "viper_unified_trading_job.py",
                "viper_async_trader.py",
                "v2_risk_optimized_trading_job.py"
            ]

            backed_up_files = []
            for config_file in config_files:
                source_path = project_root / config_file
                if source_path.exists():
                    shutil.copy2(source_path, backup_dir / config_file)
                    backed_up_files.append(config_file)

            # Store backup path
            self.backup_paths["pre_deployment"] = str(backup_dir)

            return {
                "backup_created": True,
                "backup_path": str(backup_dir),
                "files_backed_up": backed_up_files,
                "summary": f"Backup created with {len(backed_up_files)} files"
            }

        except Exception as e:
            logger.error(f"# X Backup creation failed: {e}")
            return {
                "backup_created": False,
                "error": str(e),
                "summary": "Backup creation failed"
            }

    def _run_integration_validation(self) -> Dict[str, Any]:
        """Run integration validation"""
        validation_results = {}

        try:
            # Import and run integration test
            from enhanced_system_integration_test import EnhancedSystemIntegrationTest

            integration_test = EnhancedSystemIntegrationTest()
            test_results = asyncio.run(integration_test.run_full_integration_test())

            validation_results = {
                "integration_test_run": True,
                "test_results": test_results,
                "success_rate": (test_results.get("tests_passed", 0) /
                               (test_results.get("tests_passed", 0) + test_results.get("tests_failed", 0))),
                "integration_healthy": test_results.get("overall_success", False)
            }

        except Exception as e:
            logger.error(f"# X Integration validation failed: {e}")
            validation_results = {
                "integration_test_run": False,
                "error": str(e),
                "integration_healthy": False
            }

        return validation_results

    def _run_performance_validation(self) -> Dict[str, Any]:
        """Run performance validation"""
        validation_results = {}

        try:
            # Import and run performance validation
            from enhanced_backtesting_validation import EnhancedBacktestingValidation

            validator = EnhancedBacktestingValidation()
            validation_results = asyncio.run(validator.run_comprehensive_validation())

            validation_results["performance_validation_run"] = True

        except Exception as e:
            logger.error(f"# X Performance validation failed: {e}")
            validation_results = {
                "performance_validation_run": False,
                "error": str(e),
                "deployment_ready": False
            }

        return validation_results

    def _run_security_validation(self) -> Dict[str, Any]:
        """Run security validation"""
        security_checks = {}

        try:
            # Check for sensitive data in configuration files
            config_files = [
                "enhanced_system_config.json",
                ".env",
                "config/enhanced_ai_ml_config.json"
            ]

            sensitive_patterns = [
                "API_KEY", "API_SECRET", "PASSWORD", "SECRET_KEY", "TOKEN"
            ]

            security_issues = []
            for config_file in config_files:
                config_path = project_root / config_file
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            content = f.read()

                        for pattern in sensitive_patterns:
                            if pattern.lower() in content.lower():
                                # Check if it's properly masked (contains *** or similar)
                                if "***" not in content and "xxx" not in content.lower():
                                    security_issues.append(f"Potential sensitive data in {config_file}: {pattern}")

                    except Exception as e:
                        security_issues.append(f"Could not check {config_file}: {e}")

            # File permissions check
            sensitive_files = [".env", "config/enhanced_risk_config.json"]
            permission_issues = []

            for sensitive_file in sensitive_files:
                file_path = project_root / sensitive_file
                if file_path.exists():
                    try:
                        import stat
                        file_stat = os.stat(file_path)
                        # Check if file is readable by others
                        if file_stat.st_mode & stat.S_IRGRP or file_stat.st_mode & stat.S_IROTH:
                            permission_issues.append(f"{sensitive_file} has overly permissive permissions")
                    except Exception as e:
                        permission_issues.append(f"Could not check permissions for {sensitive_file}: {e}")

            security_checks = {
                "sensitive_data_check": {
                    "issues_found": len(security_issues),
                    "issues": security_issues,
                    "passed": len(security_issues) == 0
                },
                "file_permissions_check": {
                    "issues_found": len(permission_issues),
                    "issues": permission_issues,
                    "passed": len(permission_issues) == 0
                },
                "overall_security": len(security_issues) == 0 and len(permission_issues) == 0
            }

        except Exception as e:
            logger.error(f"# X Security validation failed: {e}")
            security_checks = {
                "error": str(e),
                "overall_security": False
            }

        return security_checks

    def _assess_deployment_readiness(self, checklist_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall deployment readiness"""
        try:
            readiness_score = 0
            max_score = 5
            readiness_factors = {}

            # Pre-deployment checks (1 point)
            pre_checks = checklist_results.get("pre_deployment_checks", {})
            pre_score = sum(1 for check in pre_checks.values()
                          if isinstance(check, dict) and check.get("all_passed", check.get("all_ok", False))):
            readiness_factors["pre_deployment"] = pre_score / len(pre_checks) if pre_checks else 0

            # Integration validation (1 point)
            integration = checklist_results.get("integration_validation", {})
            readiness_factors["integration"] = 1.0 if integration.get("integration_healthy", False) else 0.0

            # Performance validation (1 point)
            performance = checklist_results.get("performance_validation", {})
            readiness_factors["performance"] = 1.0 if performance.get("deployment_ready", False) else 0.0

            # Security validation (1 point)
            security = checklist_results.get("security_validation", {})
            readiness_factors["security"] = 1.0 if security.get("overall_security", False) else 0.0

            # Configuration validation (1 point)
            config_check = checklist_results.get("pre_deployment_checks", {}).get("configuration", {})
            readiness_factors["configuration"] = 1.0 if config_check.get("all_valid", False) else 0.0

            # Calculate overall readiness
            readiness_score = sum(readiness_factors.values()) / max_score

            if readiness_score >= 0.9:
                readiness_level = "EXCELLENT"
                recommendation = "FULL DEPLOYMENT READY"
            elif readiness_score >= 0.8:
                readiness_level = "GOOD"
                recommendation = "STAGED DEPLOYMENT READY"
            elif readiness_score >= 0.6:
                readiness_level = "FAIR"
                recommendation = "MONITORED DEPLOYMENT"
            else:
                readiness_level = "POOR"
                recommendation = "NOT READY - REQUIRES FIXES"

            return {
                "readiness_score": readiness_score,
                "readiness_level": readiness_level,
                "recommendation": recommendation,
                "readiness_factors": readiness_factors,
                "deployment_blockers": self._identify_deployment_blockers(checklist_results)
            }

        except Exception as e:
            logger.error(f"# X Deployment readiness assessment failed: {e}")
            return {
                "error": str(e),
                "readiness_score": 0,
                "readiness_level": "UNKNOWN",
                "recommendation": "ASSESSMENT FAILED"
            }

    def _identify_deployment_blockers(self, checklist_results: Dict[str, Any]) -> List[str]:
        """Identify deployment blockers"""
        blockers = []

        try:
            # Check pre-deployment issues
            pre_checks = checklist_results.get("pre_deployment_checks", {})
            for check_name, check_result in pre_checks.items():
                if isinstance(check_result, dict):
                    if not check_result.get("all_passed", check_result.get("all_ok", True)):
                        blockers.append(f"Pre-deployment check failed: {check_name}")

            # Check integration issues
            integration = checklist_results.get("integration_validation", {})
            if not integration.get("integration_healthy", True):
                blockers.append("Integration validation failed")

            # Check performance issues
            performance = checklist_results.get("performance_validation", {})
            if not performance.get("deployment_ready", True):
                blockers.append("Performance validation not passed")

            # Check security issues
            security = checklist_results.get("security_validation", {})
            if not security.get("overall_security", True):
                blockers.append("Security validation failed")

        except Exception as e:
            blockers.append(f"Blocker identification failed: {e}")

        return blockers

    def _create_rollback_plan(self) -> Dict[str, Any]:
        """Create comprehensive rollback plan"""
        rollback_plan = {
            "rollback_triggers": [
                "System performance degradation > 20%",
                "Error rate > 5% for 30 minutes",
                "Memory usage > 90% sustained",
                "API connectivity loss > 10 minutes",
                "Manual emergency stop triggered"
            ],
            "rollback_procedures": [
                {
                    "step": 1,
                    "action": "Stop enhanced system processes",
                    "command": "pkill -f 'enhanced_'",
                    "timeout": "30 seconds"
                },
                {
                    "step": 2,
                    "action": "Restore configuration files from backup",
                    "backup_location": self.backup_paths.get("pre_deployment", "deployment_backup/"),
                    "files_to_restore": [
                        "enhanced_system_config.json",
                        "viper_unified_trading_job.py",
                        "viper_async_trader.py",
                        "v2_risk_optimized_trading_job.py"
                    ]
                },
                {
                    "step": 3,
                    "action": "Restart baseline trading system",
                    "command": "python viper_unified_trading_job.py",
                    "verification": "Check system logs for successful startup"
                },
                {
                    "step": 4,
                    "action": "Verify baseline system performance",
                    "metrics": ["error_rate < 2%", "response_time < 5s", "positions_update"],
                    "timeout": "5 minutes"
                }
            ],
            "monitoring_during_rollback": [
                "System resource usage",
                "Trading performance metrics",
                "Error rates and logs",
                "Position status and P&L"
            ],
            "post_rollback_actions": [
                "Analyze rollback cause",
                "Update deployment procedures",
                "Schedule enhanced system improvements",
                "Document lessons learned"
            ],
            "emergency_contacts": [
                "System Administrator",
                "Risk Manager",
                "Development Team Lead"
            ],
            "rollback_testing_required": True
        }

        return rollback_plan

    def _calculate_overall_status(self, checklist_results: Dict[str, Any]) -> str:
        """Calculate overall deployment status"""
        try:
            readiness = checklist_results.get("deployment_readiness", {})
            readiness_score = readiness.get("readiness_score", 0)

            if readiness_score >= 0.9:
                return "DEPLOYMENT_READY"
            elif readiness_score >= 0.7:
                return "CONDITIONAL_DEPLOYMENT_READY"
            elif readiness_score >= 0.5:
                return "REQUIRES_IMPROVEMENT"
            else:
                return "NOT_DEPLOYMENT_READY"

        except Exception as e:
            logger.error(f"# X Overall status calculation failed: {e}")
            return "STATUS_CALCULATION_FAILED"

    def _save_checklist_results(self, results: Dict[str, Any],
                              report_path: Optional[str] = None):
        """Save checklist results to file"""
        try:
            if report_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = project_root / f"deployment_checklist_report_{timestamp}.json"

            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"📋 Deployment checklist report saved to: {report_path}")

        except Exception as e:
            logger.error(f"# X Error saving checklist results: {e}")

    def generate_deployment_report(self, checklist_results: Dict[str, Any]) -> str:
        """Generate human-readable deployment report"""
        try:
            report_lines = [
                "=" * 80,
                "ENHANCED VIPER TRADING SYSTEM - DEPLOYMENT READINESS REPORT",
                "=" * 80,
                f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "DEPLOYMENT STATUS OVERVIEW:",
                "-" * 35
            ]

            # Overall status
            overall_status = checklist_results.get("overall_status", "UNKNOWN")
            if overall_status == "DEPLOYMENT_READY":
                report_lines.append("# Party OVERALL STATUS: DEPLOYMENT READY")
                report_lines.append("   # Check All validation criteria met")
                report_lines.append("   # Rocket System ready for production deployment")
            elif overall_status == "CONDITIONAL_DEPLOYMENT_READY":
                report_lines.append("# Warning OVERALL STATUS: CONDITIONAL DEPLOYMENT READY")
                report_lines.append("   # Check Core systems validated")
                report_lines.append("   # Warning Some improvements recommended")
            else:
                report_lines.append("# X OVERALL STATUS: NOT DEPLOYMENT READY")
                report_lines.append("   # X Critical issues need resolution")
                report_lines.append("   # Tool Address blockers before deployment")

            # Readiness breakdown
            readiness = checklist_results.get("deployment_readiness", {})
            report_lines.extend([
                "",
                "DEPLOYMENT READINESS BREAKDOWN:",
                "-" * 40
            ])

            readiness_factors = readiness.get("readiness_factors", {})
            for factor, score in readiness_factors.items():
                status_icon = "# Check" if score >= 0.8 else "# Warning" if score >= 0.6 else "# X"
                report_lines.append(f"   {status_icon} {factor.replace('_', ' ').title()}: {score:.1%}")

            # Deployment blockers
            blockers = readiness.get("deployment_blockers", [])
            if blockers:
                report_lines.extend([
                    "",
                    "DEPLOYMENT BLOCKERS:",
                    "-" * 25
                ])
                for blocker in blockers:
                    report_lines.append(f"   # X {blocker}")

            # Recommendations
            recommendation = readiness.get("recommendation", "UNKNOWN")
            report_lines.extend([
                "",
                "DEPLOYMENT RECOMMENDATIONS:",
                "-" * 35,
                f"   # Idea {recommendation}"
            ])

            # Next steps
            report_lines.extend([
                "",
                "NEXT STEPS:",
                "-" * 15,
                "   1. Review this report with deployment team",
                "   2. Address any identified blockers",
                "   3. Execute rollback plan creation if needed",
                "   4. Schedule deployment window",
                "   5. Prepare monitoring and alerting systems",
                "",
                "=" * 80
            ])

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"# X Error generating deployment report: {e}")
            return f"Error generating report: {e}"

def run_deployment_checklist():
    """Run the complete deployment checklist"""

    checklist = DeploymentChecklist()

    try:
        # Run complete checklist
        results = checklist.run_full_deployment_checklist()

        # Generate and display deployment report
        deployment_report = checklist.generate_deployment_report(results)

        # Save detailed report
        checklist._save_checklist_results(results)

        # Return deployment readiness
        overall_status = results.get("overall_status", "UNKNOWN")

        if overall_status == "DEPLOYMENT_READY":
            print("# Rocket System is ready for production deployment")
            return True
        elif overall_status == "CONDITIONAL_DEPLOYMENT_READY":
            print("# Warning Address recommendations before full deployment")
            return True
        else:
            return False

    except Exception as e:
        print(f"# X Deployment checklist execution failed: {e}")
        return False

if __name__ == "__main__":
    success = run_deployment_checklist()
    if success:
        print("\n# Check Deployment checklist completed successfully!")
        print("📋 Check the detailed report for comprehensive deployment guidance")
    else:
        print("\n# X Deployment checklist found critical issues")
        print("# Tool Review the checklist report and address all blockers before deployment")
        sys.exit(1)
