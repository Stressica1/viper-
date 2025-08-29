#!/usr/bin/env python3
"""
🚨 EMERGENCY ROLLBACK PROCEDURES
Critical system rollback and recovery procedures for Enhanced VIPER Trading System

This script provides:
- Automated rollback to baseline system
- Emergency stop procedures
- System recovery validation
- Performance monitoring during rollback
- Incident documentation and reporting
"""

import os
import sys
import json
import logging
import subprocess
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import shutil
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - EMERGENCY_ROLLBACK - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmergencyRollback:
    """Emergency rollback system for trading platform"""

    def __init__(self):
        self.rollback_log = []
        self.system_state_pre_rollback = {}
        self.rollback_start_time = None
        self.rollback_end_time = None

        # Create rollback directory
        self.rollback_dir = project_root / "emergency_rollback"
        self.rollback_dir.mkdir(exist_ok=True)

        # Load rollback configuration
        self.rollback_config = self._load_rollback_config()

        logger.info("🚨 Emergency Rollback System initialized")

    def _load_rollback_config(self) -> Dict[str, Any]:
        """Load rollback configuration"""
        config_path = project_root / "rollback_config.json"

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load rollback config: {e}")

        # Default configuration
        return {
            "rollback_timeout": 300,  # 5 minutes
            "backup_retention_days": 30,
            "max_rollback_attempts": 3,
            "monitoring_interval": 30,  # seconds
            "critical_services": [
                "viper_unified_trading_job.py",
                "viper_async_trader.py",
                "enhanced_system_integrator.py"
            ],
            "baseline_files": {
                "viper_unified_trading_job.py": "viper_unified_trading_job.baseline",
                "viper_async_trader.py": "viper_async_trader.baseline",
                "enhanced_system_config.json": "enhanced_system_config.baseline"
            }
        }

    def execute_emergency_rollback(self, reason: str = "Manual Emergency",
                                 force: bool = False) -> Dict[str, Any]:
        """Execute emergency rollback procedure"""
        logger.critical("🚨 EMERGENCY ROLLBACK INITIATED")
        logger.critical(f"Reason: {reason}")
        logger.critical("=" * 80)

        rollback_results = {
            "rollback_start_time": datetime.now().isoformat(),
            "reason": reason,
            "force_mode": force,
            "rollback_steps": [],
            "system_state_pre": {},
            "system_state_post": {},
            "rollback_success": False,
            "error_details": None
        }

        try:
            # Capture pre-rollback system state
            rollback_results["system_state_pre"] = self._capture_system_state()

            # Validate rollback readiness
            if not force and not self._validate_rollback_readiness():
                rollback_results["error_details"] = "Rollback readiness validation failed"
                return rollback_results

            # Execute rollback steps
            logger.info("📋 Executing rollback steps...")

            # Step 1: Emergency stop all enhanced processes
            step_result = self._execute_step_1_emergency_stop()
            rollback_results["rollback_steps"].append(step_result)

            # Step 2: Restore configuration files
            step_result = self._execute_step_2_restore_configurations()
            rollback_results["rollback_steps"].append(step_result)

            # Step 3: Restore core trading files
            step_result = self._execute_step_3_restore_core_files()
            rollback_results["rollback_steps"].append(step_result)

            # Step 4: Restart baseline system
            step_result = self._execute_step_4_restart_baseline_system()
            rollback_results["rollback_steps"].append(step_result)

            # Step 5: Validate system recovery
            step_result = self._execute_step_5_validate_recovery()
            rollback_results["rollback_steps"].append(step_result)

            # Step 6: Monitor system stability
            step_result = self._execute_step_6_monitor_stability()
            rollback_results["rollback_steps"].append(step_result)

            # Capture post-rollback system state
            rollback_results["system_state_post"] = self._capture_system_state()

            # Evaluate rollback success
            rollback_results["rollback_success"] = self._evaluate_rollback_success(rollback_results)

        except Exception as e:
            logger.error(f"❌ Emergency rollback failed: {e}")
            rollback_results["error_details"] = str(e)

        # Save rollback results
        self._save_rollback_results(rollback_results)

        # Generate rollback report
        self._generate_rollback_report(rollback_results)

        logger.critical("=" * 80)
        if rollback_results["rollback_success"]:
            logger.critical("✅ EMERGENCY ROLLBACK COMPLETED SUCCESSFULLY")
        else:
            logger.critical("❌ EMERGENCY ROLLBACK FAILED")

        return rollback_results

    def _validate_rollback_readiness(self) -> bool:
        """Validate rollback readiness"""
        logger.info("🔍 Validating rollback readiness...")

        checks = []

        # Check for backup files
        for baseline_file, backup_file in self.rollback_config.get("baseline_files", {}).items():
            backup_path = project_root / backup_file
            if backup_path.exists():
                checks.append(f"✅ {backup_file} backup available")
            else:
                checks.append(f"❌ {backup_file} backup missing")

        # Check system resources
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent < 95:
                checks.append("✅ System memory available")
            else:
                checks.append("❌ System memory critically low")

            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent < 90:
                checks.append("✅ System CPU available")
            else:
                checks.append("❌ System CPU critically high")

        except Exception as e:
            checks.append(f"❌ Resource check failed: {e}")

        # Log validation results
        for check in checks:
            logger.info(f"   {check}")

        # Return overall validation status
        critical_failures = sum(1 for check in checks if "❌" in check)
        readiness_status = critical_failures == 0

        logger.info(f"🔍 Rollback readiness: {'✅ READY' if readiness_status else '❌ NOT READY'}")
        return readiness_status

    def _execute_step_1_emergency_stop(self) -> Dict[str, Any]:
        """Step 1: Emergency stop all enhanced processes"""
        logger.info("🛑 Step 1: Emergency stop all enhanced processes")

        step_result = {
            "step": 1,
            "name": "Emergency Stop",
            "start_time": datetime.now().isoformat(),
            "processes_stopped": [],
            "errors": [],
            "success": False
        }

        try:
            # Define process patterns to stop
            enhanced_patterns = [
                "enhanced_",
                "viper_unified_trading_job.py",
                "viper_async_trader.py",
                "performance_monitoring_system.py",
                "optimized_market_data_streamer.py"
            ]

            processes_stopped = []

            for pattern in enhanced_patterns:
                try:
                    # Find processes matching pattern
                    result = subprocess.run(
                        ["pgrep", "-f", pattern],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        pids = result.stdout.strip().split('\n')
                        for pid in pids:
                            try:
                                # Kill process gracefully first
                                os.kill(int(pid), signal.SIGTERM)
                                time.sleep(2)  # Wait for graceful shutdown

                                # Force kill if still running
                                if psutil.pid_exists(int(pid)):
                                    os.kill(int(pid), signal.SIGKILL)

                                processes_stopped.append(f"PID {pid} ({pattern})")
                                logger.info(f"   🛑 Stopped process PID {pid} ({pattern})")

                            except Exception as e:
                                logger.warning(f"   ⚠️ Could not stop PID {pid}: {e}")
                                step_result["errors"].append(f"Failed to stop PID {pid}: {e}")

                except subprocess.TimeoutExpired:
                    logger.warning(f"   ⚠️ Timeout finding processes for pattern: {pattern}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Error stopping processes for pattern: {pattern}: {e}")
                    step_result["errors"].append(f"Pattern {pattern}: {e}")

            step_result["processes_stopped"] = processes_stopped
            step_result["success"] = True
            logger.info(f"✅ Step 1 complete: Stopped {len(processes_stopped)} processes")

        except Exception as e:
            logger.error(f"❌ Step 1 failed: {e}")
            step_result["errors"].append(str(e))

        step_result["end_time"] = datetime.now().isoformat()
        return step_result

    def _execute_step_2_restore_configurations(self) -> Dict[str, Any]:
        """Step 2: Restore configuration files"""
        logger.info("🔄 Step 2: Restore configuration files")

        step_result = {
            "step": 2,
            "name": "Configuration Restore",
            "start_time": datetime.now().isoformat(),
            "files_restored": [],
            "errors": [],
            "success": False
        }

        try:
            config_files = [
                ("enhanced_system_config.json", "enhanced_system_config.baseline"),
                (".env", ".env.baseline"),
                ("config/enhanced_ai_ml_config.json", "config/enhanced_ai_ml_config.baseline"),
                ("config/enhanced_technical_config.json", "config/enhanced_technical_config.baseline"),
                ("config/enhanced_risk_config.json", "config/enhanced_risk_config.baseline")
            ]

            files_restored = []

            for current_file, backup_file in config_files:
                current_path = project_root / current_file
                backup_path = project_root / backup_file

                if backup_path.exists():
                    try:
                        # Create backup of current file
                        if current_path.exists():
                            current_backup = current_path.with_suffix(current_path.suffix + '.rollback_backup')
                            shutil.copy2(current_path, current_backup)

                        # Restore from backup
                        shutil.copy2(backup_path, current_path)
                        files_restored.append(current_file)
                        logger.info(f"   🔄 Restored {current_file} from {backup_file}")

                    except Exception as e:
                        logger.error(f"   ❌ Failed to restore {current_file}: {e}")
                        step_result["errors"].append(f"Restore {current_file}: {e}")
                else:
                    logger.warning(f"   ⚠️ Backup not found: {backup_file}")

            step_result["files_restored"] = files_restored
            step_result["success"] = len(files_restored) > 0
            logger.info(f"✅ Step 2 complete: Restored {len(files_restored)} configuration files")

        except Exception as e:
            logger.error(f"❌ Step 2 failed: {e}")
            step_result["errors"].append(str(e))

        step_result["end_time"] = datetime.now().isoformat()
        return step_result

    def _execute_step_3_restore_core_files(self) -> Dict[str, Any]:
        """Step 3: Restore core trading files"""
        logger.info("🔄 Step 3: Restore core trading files")

        step_result = {
            "step": 3,
            "name": "Core Files Restore",
            "start_time": datetime.now().isoformat(),
            "files_restored": [],
            "errors": [],
            "success": False
        }

        try:
            core_files = [
                ("viper_unified_trading_job.py", "viper_unified_trading_job.baseline"),
                ("viper_async_trader.py", "viper_async_trader.baseline"),
                ("v2_risk_optimized_trading_job.py", "v2_risk_optimized_trading_job.baseline")
            ]

            files_restored = []

            for current_file, backup_file in core_files:
                current_path = project_root / current_file
                backup_path = project_root / backup_file

                if backup_path.exists():
                    try:
                        # Create backup of current file
                        if current_path.exists():
                            current_backup = current_path.with_suffix(current_path.suffix + '.rollback_backup')
                            shutil.copy2(current_path, current_backup)

                        # Restore from backup
                        shutil.copy2(backup_path, current_path)
                        files_restored.append(current_file)
                        logger.info(f"   🔄 Restored {current_file} from {backup_file}")

                    except Exception as e:
                        logger.error(f"   ❌ Failed to restore {current_file}: {e}")
                        step_result["errors"].append(f"Restore {current_file}: {e}")
                else:
                    logger.warning(f"   ⚠️ Backup not found: {backup_file}")

            step_result["files_restored"] = files_restored
            step_result["success"] = len(files_restored) > 0
            logger.info(f"✅ Step 3 complete: Restored {len(files_restored)} core files")

        except Exception as e:
            logger.error(f"❌ Step 3 failed: {e}")
            step_result["errors"].append(str(e))

        step_result["end_time"] = datetime.now().isoformat()
        return step_result

    def _execute_step_4_restart_baseline_system(self) -> Dict[str, Any]:
        """Step 4: Restart baseline system"""
        logger.info("🚀 Step 4: Restart baseline system")

        step_result = {
            "step": 4,
            "name": "System Restart",
            "start_time": datetime.now().isoformat(),
            "processes_started": [],
            "errors": [],
            "success": False
        }

        try:
            # Start baseline trading system
            baseline_processes = [
                ("python viper_unified_trading_job.py", "viper_unified_trading_job.py")
            ]

            processes_started = []

            for command, process_name in baseline_processes:
                try:
                    # Start process in background
                    process = subprocess.Popen(
                        command.split(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=str(project_root)
                    )

                    # Wait a moment for startup
                    time.sleep(3)

                    if process.poll() is None:  # Process is still running
                        processes_started.append(f"PID {process.pid} ({process_name})")
                        logger.info(f"   🚀 Started {process_name} (PID {process.pid})")
                    else:
                        # Process exited, check for errors
                        stdout, stderr = process.communicate()
                        if stderr:
                            logger.error(f"   ❌ {process_name} failed to start: {stderr.decode()}")
                            step_result["errors"].append(f"{process_name} startup error: {stderr.decode()}")
                        else:
                            logger.warning(f"   ⚠️ {process_name} exited immediately")

                except Exception as e:
                    logger.error(f"   ❌ Failed to start {process_name}: {e}")
                    step_result["errors"].append(f"Start {process_name}: {e}")

            step_result["processes_started"] = processes_started
            step_result["success"] = len(processes_started) > 0
            logger.info(f"✅ Step 4 complete: Started {len(processes_started)} processes")

        except Exception as e:
            logger.error(f"❌ Step 4 failed: {e}")
            step_result["errors"].append(str(e))

        step_result["end_time"] = datetime.now().isoformat()
        return step_result

    def _execute_step_5_validate_recovery(self) -> Dict[str, Any]:
        """Step 5: Validate system recovery"""
        logger.info("🔍 Step 5: Validate system recovery")

        step_result = {
            "step": 5,
            "name": "Recovery Validation",
            "start_time": datetime.now().isoformat(),
            "validation_checks": [],
            "errors": [],
            "success": False
        }

        try:
            validation_checks = []

            # Check 1: System processes running
            try:
                result = subprocess.run(
                    ["pgrep", "-f", "viper_unified_trading_job"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    validation_checks.append("✅ Baseline trading system running")
                else:
                    validation_checks.append("❌ Baseline trading system not running")
                    step_result["errors"].append("Baseline trading system not running")
            except Exception as e:
                validation_checks.append(f"❌ Process check failed: {e}")
                step_result["errors"].append(f"Process check failed: {e}")

            # Check 2: Configuration files valid
            config_files = ["enhanced_system_config.json", ".env"]
            for config_file in config_files:
                config_path = project_root / config_file
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            json.load(f)
                        validation_checks.append(f"✅ {config_file} is valid JSON")
                    except Exception as e:
                        validation_checks.append(f"❌ {config_file} invalid: {e}")
                        step_result["errors"].append(f"{config_file} invalid: {e}")
                else:
                    validation_checks.append(f"❌ {config_file} missing")
                    step_result["errors"].append(f"{config_file} missing")

            # Check 3: System resources stable
            try:
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)

                if memory_percent < 85:
                    validation_checks.append("✅ System memory stable")
                else:
                    validation_checks.append("❌ System memory high")
                    step_result["errors"].append(f"Memory usage: {memory_percent}%")

                if cpu_percent < 80:
                    validation_checks.append("✅ System CPU stable")
                else:
                    validation_checks.append("❌ System CPU high")
                    step_result["errors"].append(f"CPU usage: {cpu_percent}%")

            except Exception as e:
                validation_checks.append(f"❌ Resource check failed: {e}")
                step_result["errors"].append(f"Resource check failed: {e}")

            # Log validation results
            for check in validation_checks:
                logger.info(f"   {check}")

            step_result["validation_checks"] = validation_checks
            step_result["success"] = len(step_result["errors"]) == 0
            logger.info(f"✅ Step 5 complete: {len(validation_checks)} validation checks performed")

        except Exception as e:
            logger.error(f"❌ Step 5 failed: {e}")
            step_result["errors"].append(str(e))

        step_result["end_time"] = datetime.now().isoformat()
        return step_result

    def _execute_step_6_monitor_stability(self) -> Dict[str, Any]:
        """Step 6: Monitor system stability"""
        logger.info("📊 Step 6: Monitor system stability")

        step_result = {
            "step": 6,
            "name": "Stability Monitoring",
            "start_time": datetime.now().isoformat(),
            "monitoring_period": 120,  # 2 minutes
            "stability_metrics": [],
            "errors": [],
            "success": False
        }

        try:
            monitoring_start = time.time()
            stability_metrics = []

            logger.info(f"   📊 Monitoring system for {step_result['monitoring_period']} seconds...")

            while time.time() - monitoring_start < step_result["monitoring_period"]:
                try:
                    # Collect system metrics
                    metrics = {
                        "timestamp": datetime.now().isoformat(),
                        "cpu_percent": psutil.cpu_percent(interval=1),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_usage_percent": psutil.disk_usage('/').percent
                    }

                    # Check critical processes still running
                    critical_processes_running = 0
                    for process_pattern in ["viper_unified_trading_job"]:
                        try:
                            result = subprocess.run(
                                ["pgrep", "-f", process_pattern],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                critical_processes_running += 1
                        except Exception:
                            pass

                    metrics["critical_processes_running"] = critical_processes_running
                    stability_metrics.append(metrics)

                    # Check for critical thresholds
                    if metrics["memory_percent"] > 90:
                        step_result["errors"].append(f"Memory usage critical: {metrics['memory_percent']}%")
                    if metrics["cpu_percent"] > 95:
                        step_result["errors"].append(f"CPU usage critical: {metrics['cpu_percent']}%")
                    if critical_processes_running == 0:
                        step_result["errors"].append("All critical processes stopped")

                    time.sleep(10)  # Check every 10 seconds

                except Exception as e:
                    logger.warning(f"   ⚠️ Monitoring error: {e}")
                    step_result["errors"].append(f"Monitoring error: {e}")

            step_result["stability_metrics"] = stability_metrics
            step_result["success"] = len(step_result["errors"]) == 0

            # Log stability summary
            if stability_metrics:
                avg_cpu = sum(m["cpu_percent"] for m in stability_metrics) / len(stability_metrics)
                avg_memory = sum(m["memory_percent"] for m in stability_metrics) / len(stability_metrics)
                logger.info(f"   📊 Average CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%")

            logger.info(f"✅ Step 6 complete: Stability monitoring finished")

        except Exception as e:
            logger.error(f"❌ Step 6 failed: {e}")
            step_result["errors"].append(str(e))

        step_result["end_time"] = datetime.now().isoformat()
        return step_result

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state"""
        try:
            system_state = {
                "timestamp": datetime.now().isoformat(),
                "processes": {},
                "resources": {},
                "network": {},
                "files": {}
            }

            # Capture running processes
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        if 'viper' in proc.info['name'].lower() or 'enhanced' in proc.info['name'].lower():
                            system_state["processes"][proc.info['name']] = {
                                "pid": proc.info['pid'],
                                "cpu_percent": proc.info['cpu_percent'],
                                "memory_percent": proc.info['memory_percent']
                            }
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                system_state["processes"]["error"] = str(e)

            # Capture system resources
            try:
                system_state["resources"] = {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage_percent": psutil.disk_usage('/').percent,
                    "available_memory_gb": psutil.virtual_memory().available / (1024**3)
                }
            except Exception as e:
                system_state["resources"]["error"] = str(e)

            # Capture network connections
            try:
                connections = psutil.net_connections()
                system_state["network"]["active_connections"] = len(connections)
            except Exception as e:
                system_state["network"]["error"] = str(e)

            # Capture key file modification times
            key_files = [
                "viper_unified_trading_job.py",
                "viper_async_trader.py",
                "enhanced_system_config.json"
            ]

            for file_path in key_files:
                full_path = project_root / file_path
                if full_path.exists():
                    try:
                        system_state["files"][file_path] = {
                            "exists": True,
                            "modified": datetime.fromtimestamp(full_path.stat().st_mtime).isoformat(),
                            "size": full_path.stat().st_size
                        }
                    except Exception as e:
                        system_state["files"][file_path] = {"error": str(e)}
                else:
                    system_state["files"][file_path] = {"exists": False}

            return system_state

        except Exception as e:
            logger.error(f"❌ System state capture failed: {e}")
            return {"error": str(e)}

    def _evaluate_rollback_success(self, rollback_results: Dict[str, Any]) -> bool:
        """Evaluate overall rollback success"""
        try:
            steps = rollback_results.get("rollback_steps", [])

            # Check if all critical steps succeeded
            critical_steps_success = True
            for step in steps[:5]:  # Steps 1-5 are critical
                if not step.get("success", False):
                    critical_steps_success = False
                    logger.warning(f"   ❌ Critical step {step.get('step')} failed")

            # Check for errors in any step
            total_errors = sum(len(step.get("errors", [])) for step in steps)

            # Check system state improvement
            pre_state = rollback_results.get("system_state_pre", {})
            post_state = rollback_results.get("system_state_post", {})

            state_improved = True
            if "resources" in pre_state and "resources" in post_state:
                pre_memory = pre_state["resources"].get("memory_percent", 100)
                post_memory = post_state["resources"].get("memory_percent", 100)

                if post_memory > pre_memory + 10:  # Memory usage increased significantly
                    state_improved = False
                    logger.warning("   ⚠️ System memory usage increased after rollback")

            # Overall success criteria
            success = (
                critical_steps_success and
                total_errors == 0 and
                state_improved
            )

            logger.info(f"🔍 Rollback success evaluation:")
            logger.info(f"   ✅ Critical steps success: {critical_steps_success}")
            logger.info(f"   ✅ Total errors: {total_errors}")
            logger.info(f"   ✅ System state improved: {state_improved}")
            logger.info(f"   🎯 Overall success: {success}")

            return success

        except Exception as e:
            logger.error(f"❌ Rollback evaluation failed: {e}")
            return False

    def _save_rollback_results(self, results: Dict[str, Any]):
        """Save rollback results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = self.rollback_dir / f"rollback_results_{timestamp}.json"

            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"📋 Rollback results saved to: {results_path}")

        except Exception as e:
            logger.error(f"❌ Error saving rollback results: {e}")

    def _generate_rollback_report(self, rollback_results: Dict[str, Any]):
        """Generate human-readable rollback report"""
        try:
            report_lines = [
                "=" * 80,
                "EMERGENCY ROLLBACK REPORT",
                "=" * 80,
                f"Rollback Time: {rollback_results.get('rollback_start_time', 'UNKNOWN')}",
                f"Reason: {rollback_results.get('reason', 'UNKNOWN')}",
                "",
                "ROLLBACK STATUS:",
                "-" * 20
            ]

            success = rollback_results.get("rollback_success", False)
            if success:
                report_lines.append("✅ ROLLBACK COMPLETED SUCCESSFULLY")
            else:
                report_lines.append("❌ ROLLBACK FAILED")
                error_details = rollback_results.get("error_details")
                if error_details:
                    report_lines.append(f"Error: {error_details}")

            # Step summary
            report_lines.extend([
                "",
                "ROLLBACK STEPS SUMMARY:",
                "-" * 30
            ])

            steps = rollback_results.get("rollback_steps", [])
            for step in steps:
                step_num = step.get("step", "?")
                step_name = step.get("name", "Unknown")
                step_success = step.get("success", False)
                errors = len(step.get("errors", []))

                status_icon = "✅" if step_success else "❌"
                report_lines.append(f"{status_icon} Step {step_num}: {step_name}")

                if errors > 0:
                    report_lines.append(f"   ⚠️ {errors} errors occurred")

            # Recommendations
            report_lines.extend([
                "",
                "RECOMMENDATIONS:",
                "-" * 20,
                "1. Review system logs for any anomalies",
                "2. Monitor trading performance for next 24 hours",
                "3. Update deployment procedures based on lessons learned",
                "4. Schedule enhanced system improvements",
                "",
                "=" * 80
            ])

            report_content = "\n".join(report_lines)

            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.rollback_dir / f"rollback_report_{timestamp}.txt"

            with open(report_path, 'w') as f:
                f.write(report_content)

            logger.info(f"📋 Rollback report saved to: {report_path}")

            # Display report

        except Exception as e:
            logger.error(f"❌ Error generating rollback report: {e}")

def execute_emergency_rollback():
    """Execute emergency rollback procedure"""
    print("⚠️  WARNING: This will stop all enhanced processes and restore baseline system")
    print("⚠️  Make sure you have backups and understand the consequences")

    # Get rollback reason
    reason = input("Enter rollback reason (or press Enter for 'Manual Emergency'): ").strip()
    if not reason:
        reason = "Manual Emergency"

    # Confirm execution
    confirm = input(f"Execute emergency rollback for reason: '{reason}'? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        return False

    # Execute rollback
    rollback_system = EmergencyRollback()

    try:
        results = rollback_system.execute_emergency_rollback(reason=reason)

        if results.get("rollback_success", False):
            print("\n✅ EMERGENCY ROLLBACK COMPLETED SUCCESSFULLY!")
            print("📋 Check the rollback report for detailed information")
            return True
        else:
            print("🔧 Review the rollback report and manual intervention may be required")
            return False

    except Exception as e:
        print(f"\n❌ EMERGENCY ROLLBACK EXECUTION FAILED: {e}")
        return False

def monitor_system_health():
    """Monitor system health and trigger rollback if needed"""

    try:
        # Define health thresholds
        thresholds = {
            "cpu_percent": 90,
            "memory_percent": 90,
            "critical_processes_min": 1
        }

        while True:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent

                # Check critical processes
                critical_processes = 0
                for pattern in ["viper_unified_trading_job"]:
                    try:
                        result = subprocess.run(
                            ["pgrep", "-f", pattern],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            critical_processes += len(result.stdout.strip().split('\n'))
                    except Exception:
                        pass

                # Evaluate health
                health_issues = []

                if cpu_percent > thresholds["cpu_percent"]:
                    health_issues.append(f"CPU usage: {cpu_percent:.1f}% (threshold: {thresholds['cpu_percent']}%)")

                if memory_percent > thresholds["memory_percent"]:
                    health_issues.append(f"Memory usage: {memory_percent:.1f}% (threshold: {thresholds['memory_percent']}%)")

                if critical_processes < thresholds["critical_processes_min"]:
                    health_issues.append(f"Critical processes running: {critical_processes} (minimum: {thresholds['critical_processes_min']})")

                # Display current status
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Critical Processes: {critical_processes}")

                if health_issues:
                    for issue in health_issues:

                    # Ask for rollback
                    rollback = input("Execute emergency rollback? (yes/no): ").lower().strip()
                    if rollback in ['yes', 'y']:
                        reason = f"Automated health check: {'; '.join(health_issues)}"
                        execute_emergency_rollback()
                        break
                else:

                time.sleep(30)  # Check every 30 seconds

            except KeyboardInterrupt:
                break
            except Exception as e:
                time.sleep(30)

    except Exception as e:

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Emergency Rollback System")
    parser.add_argument("action", choices=["rollback", "monitor"],
                       help="Action to perform")
    parser.add_argument("--reason", help="Rollback reason")
    parser.add_argument("--force", action="store_true",
                       help="Force rollback without validation")

    args = parser.parse_args()

    if args.action == "rollback":
        if args.reason:
            # Non-interactive rollback
            rollback_system = EmergencyRollback()
            results = rollback_system.execute_emergency_rollback(
                reason=args.reason,
                force=args.force
            )
            success = results.get("rollback_success", False)
            sys.exit(0 if success else 1)
        else:
            # Interactive rollback
            execute_emergency_rollback()
    elif args.action == "monitor":
        monitor_system_health()
