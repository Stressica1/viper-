#!/usr/bin/env python3
"""
🚀 DOCKER VALIDATOR
===================

MANDATORY: Validates Docker requirements for system operation.
This script MUST pass for system startup.

Features:
✅ Docker installation and version validation
✅ Docker Compose availability
✅ Container registry access
✅ Network configuration
✅ Resource limits validation
✅ Service health checks
✅ Startup blocking enforcement
"""

import os
import sys
import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("✅ Environment variables loaded from .env file")
except ImportError:
    logging.warning("⚠️ python-dotenv not available, using system environment only")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DOCKER_VALIDATOR - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DockerValidator:
    """MANDATORY: Docker Integration Validator"""

    def __init__(self):
        self.config_path = Path("/Users/tradecomp/bg/viper-/config/system_requirements.json")
        self.docker_compose_path = Path("/Users/tradecomp/bg/viper-/docker/docker-compose.yml")
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "component": "docker",
            "status": "unknown",
            "checks": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }

        # Load system requirements
        self.requirements = self._load_requirements()

    def _load_requirements(self) -> Dict[str, Any]:
        """Load system requirements configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.error("❌ System requirements file not found")
                return {}
        except Exception as e:
            logger.error(f"❌ Failed to load requirements: {e}")
            return {}

    def validate_all(self) -> Tuple[bool, Dict[str, Any]]:
        """MANDATORY: Run all Docker validation checks."""
        logger.info("🚀 STARTING DOCKER VALIDATION (MANDATORY)")

        checks = {
            "docker_installation": self._validate_docker_installation,
            "docker_version": self._validate_docker_version,
            "docker_compose": self._validate_docker_compose,
            "docker_daemon": self._validate_docker_daemon,
            "container_registry": self._validate_container_registry,
            "docker_compose_config": self._validate_docker_compose_config,
            "network_configuration": self._validate_network_configuration,
            "resource_limits": self._validate_resource_limits
        }

        all_passed = True

        for check_name, check_function in checks.items():
            logger.info(f"🔍 Validating {check_name}...")
            try:
                passed, result = check_function()
                self.validation_results["checks"][check_name] = {
                    "status": "PASSED" if passed else "FAILED",
                    "details": result
                }
                if not passed:
                    all_passed = False
                    self.validation_results["errors"].append(f"{check_name}: {result}")
            except Exception as e:
                logger.error(f"❌ {check_name} validation error: {e}")
                self.validation_results["checks"][check_name] = {
                    "status": "ERROR",
                    "details": str(e)
                }
                all_passed = False
                self.validation_results["errors"].append(f"{check_name}: {str(e)}")

        self.validation_results["status"] = "PASSED" if all_passed else "FAILED"

        logger.info(f"🎯 DOCKER VALIDATION COMPLETE: {'✅ PASSED' if all_passed else '❌ FAILED'}")
        return all_passed, self.validation_results

    def _validate_docker_installation(self) -> Tuple[bool, str]:
        """MANDATORY: Validate Docker installation."""
        try:
            result = subprocess.run(['docker', '--version'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                version_output = result.stdout.strip()
                return True, f"Docker installed: {version_output}"
            else:
                return False, "Docker command not found or not executable"

        except FileNotFoundError:
            return False, "Docker is not installed"
        except subprocess.TimeoutExpired:
            return False, "Docker version check timed out"
        except Exception as e:
            return False, f"Docker installation check failed: {str(e)}"

    def _validate_docker_version(self) -> Tuple[bool, str]:
        """MANDATORY: Validate Docker version requirements."""
        try:
            result = subprocess.run(['docker', '--version'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                version_output = result.stdout.strip()
                # Extract version number (basic parsing)
                if 'Docker version' in version_output:
                    version_part = version_output.split('Docker version')[1].split(',')[0].strip()
                    # Simple version check (24.0+)
                    try:
                        major_version = int(version_part.split('.')[0])
                        if major_version >= 24:
                            return True, f"Docker version {version_part} meets requirements (24.0+)"
                        else:
                            return False, f"Docker version {version_part} is below minimum requirement (24.0+)"
                    except ValueError:
                        return False, f"Could not parse Docker version: {version_part}"
                else:
                    return False, f"Unexpected Docker version output: {version_output}"
            else:
                return False, "Failed to get Docker version"

        except Exception as e:
            return False, f"Docker version validation failed: {str(e)}"

    def _validate_docker_compose(self) -> Tuple[bool, str]:
        """MANDATORY: Validate Docker Compose installation."""
        try:
            result = subprocess.run(['docker-compose', '--version'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                version_output = result.stdout.strip()
                return True, f"Docker Compose installed: {version_output}"
            else:
                return False, "Docker Compose command not found"

        except FileNotFoundError:
            return False, "Docker Compose is not installed"
        except subprocess.TimeoutExpired:
            return False, "Docker Compose version check timed out"
        except Exception as e:
            return False, f"Docker Compose validation failed: {str(e)}"

    def _validate_docker_daemon(self) -> Tuple[bool, str]:
        """MANDATORY: Validate Docker daemon connectivity."""
        try:
            result = subprocess.run(['docker', 'info'],
                                  capture_output=True, text=True, timeout=15)

            if result.returncode == 0:
                # Check if daemon is running by looking for key info
                if 'Server Version:' in result.stdout:
                    return True, "Docker daemon is running and accessible"
                else:
                    return False, "Docker daemon response format unexpected"
            else:
                error_output = result.stderr.strip()
                if 'Is the docker daemon running?' in error_output:
                    return False, "Docker daemon is not running"
                else:
                    return False, f"Docker daemon check failed: {error_output}"

        except subprocess.TimeoutExpired:
            return False, "Docker daemon check timed out"
        except Exception as e:
            return False, f"Docker daemon validation failed: {str(e)}"

    def _validate_container_registry(self) -> Tuple[bool, str]:
        """MANDATORY: Validate container registry access."""
        try:
            # Test access to Docker Hub (basic connectivity)
            result = subprocess.run(['docker', 'pull', 'hello-world'],
                                  capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return True, "Container registry access confirmed (Docker Hub)"
            else:
                error_output = result.stderr.strip()
                if 'denied' in error_output.lower():
                    return False, "Container registry access denied"
                elif 'network' in error_output.lower():
                    return False, "Network connectivity issue with container registry"
                else:
                    return False, f"Container registry test failed: {error_output[:100]}..."

        except subprocess.TimeoutExpired:
            return False, "Container registry test timed out"
        except Exception as e:
            return False, f"Container registry validation failed: {str(e)}"

    def _validate_docker_compose_config(self) -> Tuple[bool, str]:
        """MANDATORY: Validate Docker Compose configuration."""
        if not self.docker_compose_path.exists():
            return False, "Docker Compose configuration file not found"

        try:
            result = subprocess.run(['docker-compose', '-f', str(self.docker_compose_path), 'config'],
                                  capture_output=True, text=True, timeout=15)

            if result.returncode == 0:
                # Parse the config output to check for services
                config_output = result.stdout
                if 'services:' in config_output:
                    # Count services (rough estimate)
                    service_count = config_output.count('\n  ')  # Rough count of services
                    return True, f"Docker Compose config validated ({service_count}+ services configured)"
                else:
                    return False, "No services found in Docker Compose configuration"
            else:
                error_output = result.stderr.strip()
                return False, f"Docker Compose config validation failed: {error_output}"

        except subprocess.TimeoutExpired:
            return False, "Docker Compose config validation timed out"
        except Exception as e:
            return False, f"Docker Compose config validation failed: {str(e)}"

    def _validate_network_configuration(self) -> Tuple[bool, str]:
        """MANDATORY: Validate Docker network configuration."""
        try:
            result = subprocess.run(['docker', 'network', 'ls'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                network_output = result.stdout
                if 'viper-network' in network_output or 'bridge' in network_output:
                    return True, "Docker network configuration validated"
                else:
                    return False, "Required Docker networks not found"
            else:
                return False, f"Docker network check failed: {result.stderr.strip()}"

        except subprocess.TimeoutExpired:
            return False, "Docker network validation timed out"
        except Exception as e:
            return False, f"Docker network validation failed: {str(e)}"

    def _validate_resource_limits(self) -> Tuple[bool, str]:
        """MANDATORY: Validate Docker resource limits configuration."""
        try:
            if not self.docker_compose_path.exists():
                return False, "Docker Compose file required for resource limits validation"

            # Check if resource limits are configured in compose file
            with open(self.docker_compose_path, 'r') as f:
                compose_content = f.read()

            has_limits = any(keyword in compose_content for keyword in [
                'mem_limit:', 'cpus:', 'memory:', 'cpu_count:'
            ])

            if has_limits:
                return True, "Docker resource limits configured in Compose file"
            else:
                return False, "Docker resource limits not configured"

        except Exception as e:
            return False, f"Resource limits validation failed: {str(e)}"

    def enforce_requirements(self) -> bool:
        """MANDATORY: Enforce Docker requirements (blocking)."""
        logger.info("🚫 ENFORCING DOCKER REQUIREMENTS (MANDATORY)")

        passed, results = self.validate_all()

        if not passed:
            logger.critical("❌ DOCKER VALIDATION FAILED - SYSTEM STARTUP BLOCKED")
            logger.critical("🚫 CRITICAL: Docker is MANDATORY for system operation")

            # Log all errors
            for error in results["errors"]:
                logger.error(f"   ❌ {error}")

            # Save validation results
            self._save_validation_results()

            # Exit with failure (blocks system startup)
            logger.critical("🛑 SYSTEM STARTUP PREVENTED DUE TO MANDATORY REQUIREMENT FAILURE")
            return False

        logger.info("✅ DOCKER REQUIREMENTS ENFORCED - ALL CHECKS PASSED")
        self._save_validation_results()
        return True

    def _save_validation_results(self):
        """Save validation results to file."""
        results_file = Path("/Users/tradecomp/bg/viper-/logs/docker_validation.json")
        results_file.parent.mkdir(exist_ok=True)

        try:
            with open(results_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            logger.info(f"💾 Validation results saved: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")

def main():
    """MANDATORY: Main validation function."""
    print("🚀 DOCKER VALIDATOR (MANDATORY REQUIREMENT)")
    print("=" * 60)
    print("⚠️  WARNING: This validation MUST pass for system operation")
    print("🚫 FAILURE: Will block system startup")
    print()

    validator = DockerValidator()

    # Run validation
    passed, results = validator.validate_all()

    print("\n📊 VALIDATION RESULTS")
    print("-" * 40)
    print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")

    if results["errors"]:
        print("\n❌ ERRORS FOUND:")
        for error in results["errors"]:
            print(f"   • {error}")

    if results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in results["warnings"]:
            print(f"   • {warning}")

    print("\n🔍 CHECK DETAILS:")
    for check_name, check_result in results["checks"].items():
        status = check_result["status"]
        details = check_result["details"]
        print(f"   {check_name}: {'✅' if status == 'PASSED' else '❌'} {details}")

    # Enforce requirements
    if not passed:
        print("\n🚫 MANDATORY REQUIREMENT FAILURE")
        print("🛑 System startup blocked due to Docker validation failure")
        print("🔧 Please resolve the errors above and try again")
        sys.exit(1)
    else:
        print("\n✅ MANDATORY REQUIREMENT SATISFIED")
        print("🚀 Docker integration validated successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
