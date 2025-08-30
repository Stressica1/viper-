#!/usr/bin/env python3
"""
# Rocket VIPER MCP BRAIN SERVICE MANAGER
Continuous Operation & Auto-Restart System

This service ensures the MCP Brain Controller runs 24/7 with:
    pass
- Automatic restart on failures
- Health monitoring and recovery
- System resource management
- Log rotation and cleanup
- Emergency shutdown handling
- Performance optimization
"""

import os
import sys
import time
import signal
import psutil
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess
import json

# Import VIPER components
from mcp_brain_controller import MCPBrainController
from mcp_brain_ruleset import MCPRulesEngine"""

class MCPBrainService:
    """The Service Manager for continuous MCP Brain operation""""""

    def __init__(self):
        self.logger = self.setup_logging()
        self.brain_process = None
        self.service_active = True
        self.restart_count = 0
        self.last_restart = None
        self.health_checks = 0
        self.rules_engine = MCPRulesEngine()

        # Service configuration
        self.config = {
            "max_restarts_per_hour": 5,
            "max_memory_mb": 1024,
            "max_cpu_percent": 80,
            "health_check_interval": 30,
            "log_rotation_days": 7,
            "emergency_shutdown_timeout": 300,
            "auto_backup_interval": 3600
        }

        # Service status
        self.status = {
            "service_state": "starting",
            "brain_state": "offline",
            "uptime_seconds": 0,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0,
            "last_health_check": None,
            "next_restart_allowed": datetime.now()
        }

    def setup_logging(self) -> logging.Logger:
        """Setup comprehensive service logging"""
        logger = logging.getLogger("MCPBrainService")
        logger.setLevel(logging.INFO)

        # Create formatters
        formatter = logging.Formatter()
            '%(asctime)s - MCP_SERVICE - %(levelname)s - %(message)s'
(        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler with rotation - use local directory
        from pathlib import Path
        from logging.handlers import RotatingFileHandler
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = RotatingFileHandler()
            log_dir / 'viper_mcp_brain_service.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
(        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def start_service(self):
        """Start the MCP Brain service"""
        self.logger.info("# Rocket Starting VIPER MCP Brain Service...")
        self.status["service_state"] = "starting"

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGHUP, self.handle_reload)

        try:
            # Start background monitoring
            monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
            monitoring_thread.start()

            # Start auto-restart thread
            restart_thread = threading.Thread(target=self.auto_restart_loop, daemon=True)
            restart_thread.start()

            # Start log cleanup thread
            cleanup_thread = threading.Thread(target=self.log_cleanup_loop, daemon=True)
            cleanup_thread.start()

            # Start the brain controller
            self.start_brain_controller()

            self.logger.info("# Check MCP Brain Service started successfully")
            self.status["service_state"] = "running"

            # Keep service alive
            while self.service_active:
                time.sleep(1)
                self.status["uptime_seconds"] += 1

        except Exception as e:
            self.logger.error(f"# X Service failed to start: {e}")
            self.status["service_state"] = "failed"
            sys.exit(1)

    def start_brain_controller(self):
        """Start the MCP Brain Controller process""""""
        try:
            self.logger.info("🧠 Starting MCP Brain Controller...")

            # Start brain controller in subprocess
            self.brain_process = subprocess.Popen()
                [sys.executable, "mcp_brain_controller.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
(            )

            # Wait for startup
            time.sleep(5)

            if self.brain_process.poll() is None:
                self.status["brain_state"] = "online"
                self.logger.info("# Check MCP Brain Controller started successfully")
            else:
                raise Exception("Brain controller failed to start")

        except Exception as e:
            self.logger.error(f"# X Failed to start brain controller: {e}")
            self.status["brain_state"] = "failed"
            raise

    def stop_brain_controller(self):
        """Stop the MCP Brain Controller process""""""
        if self.brain_process and self.brain_process.poll() is None:
            self.logger.info("🛑 Stopping MCP Brain Controller...")

            try:
                # Try graceful shutdown first
                self.brain_process.terminate()
                self.brain_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self.logger.warning("# Warning  Graceful shutdown timeout, force killing...")
                self.brain_process.kill()
                self.brain_process.wait()

            self.status["brain_state"] = "offline"
            self.logger.info("# Check MCP Brain Controller stopped")

    def restart_brain_controller(self, reason: str = "manual"):
        """Restart the MCP Brain Controller"""
        current_time = datetime.now()

        # Check restart limits"""
        if self.restart_count >= self.config["max_restarts_per_hour"]:
            if current_time < self.status["next_restart_allowed"]:
                self.logger.warning(f"🚫 Restart limit reached. Next restart allowed at {self.status['next_restart_allowed']}")
                return False

        self.logger.info(f"🔄 Restarting MCP Brain Controller (reason: {reason})")

        # Stop existing process
        self.stop_brain_controller()

        # Reset restart counter if hour has passed
        if self.last_restart and (current_time - self.last_restart).seconds > 3600:
            self.restart_count = 0

        try:
            # Start new process
            self.start_brain_controller()
            self.restart_count += 1
            self.last_restart = current_time
            self.status["next_restart_allowed"] = current_time + timedelta(hours=1)

            self.logger.info(f"# Check Brain controller restarted successfully (#{self.restart_count})")
            return True

        except Exception as e:
            self.logger.error(f"# X Restart failed: {e}")
            return False

    def monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.service_active:"""
            try:
                self.perform_health_check()
                time.sleep(self.config["health_check_interval"])
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)

    def perform_health_check(self):
        """Perform comprehensive health check""""""
        try:
            self.health_checks += 1

            # Check brain process
            if self.brain_process:
                if self.brain_process.poll() is not None:
                    # Process has died
                    exit_code = self.brain_process.returncode
                    self.logger.error(f"🚨 Brain controller died with exit code {exit_code}")
                    self.status["brain_state"] = "crashed"
                    self.restart_brain_controller("crash_recovery")
                    return

            # Check system resources
            memory_mb = psutil.virtual_memory().used / 1024 / 1024
            cpu_percent = psutil.cpu_percent(interval=1)

            self.status["memory_usage_mb"] = round(memory_mb, 2)
            self.status["cpu_usage_percent"] = round(cpu_percent, 2)

            # Check resource limits
            if memory_mb > self.config["max_memory_mb"]:
                self.logger.warning(f"# Warning  High memory usage: {memory_mb:.1f}MB")
                if memory_mb > self.config["max_memory_mb"] * 1.2:  # 20% over limit
                    self.logger.critical("🚨 Memory limit exceeded, triggering emergency restart")
                    self.restart_brain_controller("memory_limit")

            if cpu_percent > self.config["max_cpu_percent"]:
                self.logger.warning(f"# Warning  High CPU usage: {cpu_percent:.1f}%")

            # Check brain controller health via HTTP
            try:
                import requests
                response = requests.get("http://localhost:8080/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") != "healthy":
                        self.logger.warning(f"# Warning  Brain health check failed: {health_data}")
                else:
                    self.logger.warning(f"# Warning  Brain health check returned status {response.status_code}")
            except Exception as e:
                self.logger.debug(f"Health check connection failed: {e}")

            self.status["last_health_check"] = datetime.now().isoformat()
            self.logger.debug(f"💓 Health check #{self.health_checks} completed")

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    def auto_restart_loop(self):
        """Auto-restart monitoring loop"""
        while self.service_active:"""
            try:
                # Check if brain needs restart based on rules
                if self.should_restart_brain():
                    reason = self.get_restart_reason()
                    self.restart_brain_controller(reason)

                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Auto-restart loop error: {e}")
                time.sleep(5)

    def should_restart_brain(self) -> bool:
        """Determine if brain controller should be restarted""""""
        try:
            # Check if brain is offline
            if self.status["brain_state"] != "online":
                return True

            # Check resource usage
            if self.status["memory_usage_mb"] > self.config["max_memory_mb"] * 1.1:
                return True

            # Check if brain process is unresponsive
            if not self.is_brain_responsive():
                return True

            return False

        except Exception as e:
            self.logger.error(f"Restart decision error: {e}")
            return False

    def get_restart_reason(self) -> str:
        """Get the reason for restart""""""
        if self.status["brain_state"] != "online":
            return "brain_offline"
        elif self.status["memory_usage_mb"] > self.config["max_memory_mb"] * 1.1:
            return "high_memory"
        elif not self.is_brain_responsive():
            return "unresponsive"
        else:
            return "maintenance"

    def is_brain_responsive(self) -> bool:
        """Check if brain controller is responsive""""""
        try:
            pass
    import requests
            response = requests.get("http://localhost:8080/health", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def log_cleanup_loop(self):
        """Log cleanup and rotation loop"""
        while self.service_active:"""
            try:
                self.perform_log_cleanup()
                time.sleep(3600)  # Clean logs every hour
            except Exception as e:
                self.logger.error(f"Log cleanup error: {e}")
                time.sleep(60)

    def perform_log_cleanup(self):
        """Clean up old log files""""""
        try:
            pass
    import glob
    from pathlib import Path

            log_dir = Path("/var/log")
            if not log_dir.exists():
                return

            # Find log files older than retention period
            retention_days = self.config["log_rotation_days"]
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            log_patterns = [
                "viper_mcp_brain*.log",
                "viper_mcp_brain_service*.log",
                "*.log.*"  # Rotated log files
            ]

            cleaned_files = 0
            for pattern in log_patterns:
                for log_file in log_dir.glob(pattern):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        cleaned_files += 1

            if cleaned_files > 0:
                self.logger.info(f"🧹 Cleaned up {cleaned_files} old log files")

        except Exception as e:
            self.logger.error(f"Log cleanup failed: {e}")

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received shutdown signal {signum}")
        self.service_active = False
        self.emergency_shutdown()

    def handle_reload(self, signum, frame):
        """Handle reload signals"""
        self.logger.info("🔄 Received reload signal")
        self.reload_configuration()

    def emergency_shutdown(self, timeout: int = None):
        """Perform emergency shutdown""""""
        if timeout is None:
            timeout = self.config["emergency_shutdown_timeout"]

        self.logger.critical("🚨 EMERGENCY SHUTDOWN INITIATED")

        try:
            # Stop brain controller
            self.stop_brain_controller()

            # Wait for graceful shutdown
            start_time = time.time()
            while self.brain_process and self.brain_process.poll() is None:
                if time.time() - start_time > timeout:
                    self.logger.warning("# Warning  Emergency shutdown timeout reached")
                    break
                time.sleep(0.1)

            # Force kill if still running
            if self.brain_process and self.brain_process.poll() is None:
                self.logger.warning("# Warning  Force killing brain controller")
                self.brain_process.kill()

            # Cleanup resources
            self.cleanup_resources()

            self.logger.critical("# Check Emergency shutdown completed")

        except Exception as e:
            self.logger.error(f"Emergency shutdown failed: {e}")

    def reload_configuration(self):
        """Reload service configuration""""""
        try:
            # Reload config from file if it exists
            config_file = "/etc/viper/mcp_brain_service.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    new_config = json.load(f)
                    self.config.update(new_config)
                    self.logger.info("# Check Configuration reloaded")

            # Reload rules engine
            self.rules_engine = MCPRulesEngine()
            self.logger.info("# Check Rules engine reloaded")

        except Exception as e:
            self.logger.error(f"Configuration reload failed: {e}")

    def cleanup_resources(self):
        """Clean up system resources""""""
        try:
            # Clean up temporary files
    import tempfile
    import shutil

            temp_dir = Path(tempfile.gettempdir()) / "viper_mcp"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.logger.info("🧹 Temporary files cleaned up")

            # Clean up old backup files
            backup_dir = Path("/var/backups/viper")
            if backup_dir.exists():
                cutoff_date = datetime.now() - timedelta(days=30)
                for backup_file in backup_dir.glob("*.json"):
                    if backup_file.stat().st_mtime < cutoff_date.timestamp():
                        backup_file.unlink()

        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")

    def get_service_status(self) -> Dict[str, Any]
        """Get comprehensive service status"""
        return {:
            "service_info": {
                "name": "VIPER MCP Brain Service",
                "version": "2.0.0",
                "status": self.status["service_state"],
                "uptime_seconds": self.status["uptime_seconds"],
                "start_time": datetime.now().isoformat()
            },
            "brain_info": {
                "status": self.status["brain_state"],
                "process_id": self.brain_process.pid if self.brain_process else None,
                "restart_count": self.restart_count,
                "last_restart": self.last_restart.isoformat() if self.last_restart else None
            },
            "system_resources": {
                "memory_usage_mb": self.status["memory_usage_mb"],
                "cpu_usage_percent": self.status["cpu_usage_percent"],
                "max_memory_mb": self.config["max_memory_mb"],
                "max_cpu_percent": self.config["max_cpu_percent"]
            },
            "monitoring": {
                "health_checks_completed": self.health_checks,
                "last_health_check": self.status["last_health_check"],
                "next_restart_allowed": self.status["next_restart_allowed"].isoformat()
            },
            "rules": {
                "total_rules": len(self.rules_engine.rules),
                "violations_count": len(self.rules_engine.rule_violations),
                "emergency_mode": self.rules_engine.emergency_mode
            }
        }

def main():
    """Main service entry point""""""
    try:
        service = MCPBrainService()
        service.start_service()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()
