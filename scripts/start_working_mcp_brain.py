#!/usr/bin/env python3
"""
# Rocket WORKING VIPER MCP BRAIN SYSTEM LAUNCHER
This version actually works and starts the brain controller properly
"""

import os
import sys
import time
import json
import signal
import logging
import subprocess
import threading
from pathlib import Path

class WorkingMCPBrainLauncher:
    """A working launcher that can actually start the MCP Brain System"""

    def __init__(self):
        self.working_dir = Path(__file__).parent
        self.processes = {}
        self.logger = self.setup_logging()

    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - MCP_BRAIN - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def start_brain_controller(self):
        """Start the brain controller"""
        try:
            self.logger.info("🧠 Starting MCP Brain Controller...")

            # Use the simple working version
            brain_script = self.working_dir / "mcp_brain_controller_simple.py"

            if not brain_script.exists():
                self.logger.error(f"# X Brain controller script not found: {brain_script}")
                return False

            # Start the process
            process = subprocess.Popen(
                [sys.executable, str(brain_script)],
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )

            self.processes["brain_controller"] = process
            self.logger.info(f"# Check Brain controller started (PID: {process.pid})")

            # Wait a bit for startup
            time.sleep(3)

            # Test if it's responding
            if self.test_brain_health():
                self.logger.info("# Check Brain controller health check passed")
                return True
            else:
                self.logger.error("# X Brain controller health check failed")
                self.stop_brain_controller()
                return False

        except Exception as e:
            self.logger.error(f"# X Failed to start brain controller: {e}")
            return False

    def test_brain_health(self):
        """Test if brain controller is responding"""
        try:
            import requests
            response = requests.get("http://localhost:8080/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def start_cursor_integration(self):
        """Start cursor integration (optional)"""
        try:
            self.logger.info("🔗 Starting Cursor Integration...")

            cursor_script = self.working_dir / "mcp_cursor_integration.py"

            if not cursor_script.exists():
                self.logger.warning("# Warning  Cursor integration script not found, skipping")
                return True

            process = subprocess.Popen(
                [sys.executable, str(cursor_script)],
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

            self.processes["cursor_integration"] = process
            self.logger.info(f"# Check Cursor integration started (PID: {process.pid})")
            return True

        except Exception as e:
            self.logger.warning(f"# Warning  Cursor integration failed: {e}")
            return True  # Not critical

    def start_service_manager(self):
        """Start service manager (optional)"""
        try:
            self.logger.info("⚙️  Starting Service Manager...")

            service_script = self.working_dir / "mcp_brain_service.py"

            if not service_script.exists():
                self.logger.warning("# Warning  Service manager script not found, skipping")
                return True

            process = subprocess.Popen(
                [sys.executable, str(service_script)],
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

            self.processes["service_manager"] = process
            self.logger.info(f"# Check Service manager started (PID: {process.pid})")
            return True

        except Exception as e:
            self.logger.warning(f"# Warning  Service manager failed: {e}")
            return True  # Not critical

    def monitor_system(self):
        """Monitor the running system"""
        self.logger.info("# Chart System monitoring started...")

        try:
            while True:
                # Check if brain controller is still running
                if "brain_controller" in self.processes:
                    process = self.processes["brain_controller"]
                    if process.poll() is not None:
                        self.logger.error(f"🚨 Brain controller died with code {process.returncode}")
                        self.restart_brain_controller()
                        break

                # Check health
                if not self.test_brain_health():
                    self.logger.warning("# Warning  Brain controller health check failed")

                time.sleep(10)  # Check every 10 seconds

        except KeyboardInterrupt:
            self.logger.info("# Chart Monitoring stopped")

    def restart_brain_controller(self):
        """Restart the brain controller"""
        self.logger.info("🔄 Restarting brain controller...")

        self.stop_brain_controller()
        time.sleep(2)
        self.start_brain_controller()

    def stop_brain_controller(self):
        """Stop the brain controller"""
        if "brain_controller" in self.processes:
            process = self.processes["brain_controller"]
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                self.logger.info("# Check Brain controller stopped")
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    self.logger.warning("# Warning  Force killed brain controller")

            del self.processes["brain_controller"]

    def stop_all(self):
        """Stop all processes"""
        self.logger.info("🛑 Stopping all MCP Brain processes...")

        for name, process in self.processes.items():
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
                self.logger.info(f"# Check {name} stopped")
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=2)
                    self.logger.info(f"# Check {name} force killed")
                except Exception:
                    self.logger.warning(f"# Warning  Could not stop {name}")

        self.processes.clear()

    def get_system_status(self):
        """Get system status"""
        status = {
            "brain_controller": "running" if "brain_controller" in self.processes and self.processes["brain_controller"].poll() is None else "stopped",
            "cursor_integration": "running" if "cursor_integration" in self.processes and self.processes["cursor_integration"].poll() is None else "stopped",
            "service_manager": "running" if "service_manager" in self.processes and self.processes["service_manager"].poll() is None else "stopped",
            "health_check": self.test_brain_health(),
            "processes": {name: process.pid for name, process in self.processes.items() if process.poll() is None}
        }
        return status

    def run(self):
        """Run the MCP Brain System"""
        self.logger.info("# Rocket VIPER MCP Brain System - WORKING VERSION")
        self.logger.info("=" * 60)

        try:
            # Start components
            success = True

            if not self.start_brain_controller():
                success = False

            self.start_cursor_integration()  # Optional
            self.start_service_manager()    # Optional

            if not success:
                self.logger.error("# X Failed to start core components")
                self.stop_all()
                return False

            self.logger.info("# Check MCP Brain System started successfully!")
            self.logger.info("# Chart Dashboard: http://localhost:8080")
            self.logger.info("# Chart Status: http://localhost:8080/health")

            # Start monitoring in background thread
            monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
            monitor_thread.start()

            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)

                    # Print status every minute
                    if int(time.time()) % 60 == 0:
                        status = self.get_system_status()
                        self.logger.info(f"# Chart System Status: {json.dumps(status, indent=2)}")

            except KeyboardInterrupt:
                self.logger.info("🛑 Received shutdown signal")
                self.stop_all()
                return True

        except Exception as e:
            self.logger.error(f"# X System failed: {e}")
            self.stop_all()
            return False

def main():
    """Main entry point"""
    launcher = WorkingMCPBrainLauncher()

    def signal_handler(signum, frame):
        launcher.logger.info(f"🛑 Received signal {signum}")
        launcher.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        success = launcher.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        launcher.logger.error(f"# X Fatal error: {e}")
        launcher.stop_all()
        sys.exit(1)

if __name__ == "__main__":
    main()
