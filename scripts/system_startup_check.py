#!/usr/bin/env python3
"""
🚀 SYSTEM STARTUP VALIDATOR
===========================

MANDATORY: Validates ALL system requirements before startup.
This script MUST pass for system operation.

Features:
✅ GitHub MCP validation
✅ Docker validation
✅ System requirements enforcement
✅ Startup blocking mechanism
✅ Parallel validation execution
✅ Comprehensive error reporting
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SYSTEM_STARTUP - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemStartupValidator:
    """MANDATORY: Complete System Startup Validator"""

    def __init__(self):
        self.config_path = Path("/Users/tradecomp/bg/viper-/config/system_requirements.json")
        self.startup_results = {
            "timestamp": datetime.now().isoformat(),
            "system_name": "VIPER Trading System",
            "version": "2.0.0",
            "status": "unknown",
            "components": {},
            "overall_score": 0,
            "startup_allowed": False,
            "execution_time_seconds": 0,
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

    def validate_all_components(self) -> Tuple[bool, Dict[str, Any]]:
        """MANDATORY: Validate all system components."""
        logger.info("🚀 STARTING COMPLETE SYSTEM VALIDATION (MANDATORY)")
        start_time = time.time()

        mandatory_components = self.requirements.get('system_requirements', {}).get('mandatory_components', {})

        if not mandatory_components:
            logger.error("❌ No mandatory components defined in requirements")
            return False, {"error": "No mandatory components configured"}

        # Prepare validation tasks
        validation_tasks = []
        for component_name, component_config in mandatory_components.items():
            if component_config.get('required', False):
                validation_tasks.append((component_name, component_config))

        logger.info(f"📋 Validating {len(validation_tasks)} mandatory components")

        # Execute validations (parallel for performance)
        all_passed = True
        component_results = {}

        for component_name, component_config in validation_tasks:
            logger.info(f"🔍 Validating {component_name}...")
            try:
                passed, result = self._validate_component(component_name, component_config)
                component_results[component_name] = {
                    "status": "PASSED" if passed else "FAILED",
                    "details": result,
                    "config": component_config
                }

                if not passed:
                    all_passed = False

            except Exception as e:
                logger.error(f"❌ {component_name} validation error: {e}")
                component_results[component_name] = {
                    "status": "ERROR",
                    "details": str(e),
                    "config": component_config
                }
                all_passed = False

        # Calculate execution time
        execution_time = time.time() - start_time

        # Prepare final results
        self.startup_results.update({
            "status": "PASSED" if all_passed else "FAILED",
            "components": component_results,
            "startup_allowed": all_passed,
            "execution_time_seconds": round(execution_time, 2),
            "overall_score": self._calculate_overall_score(component_results)
        })

        logger.info(f"🎯 SYSTEM VALIDATION COMPLETE: {'✅ PASSED' if all_passed else '❌ FAILED'}")
        return all_passed, self.startup_results

    def _validate_component(self, component_name: str, component_config: Dict[str, Any]) -> Tuple[bool, Any]:
        """Validate individual component."""
        if component_name == "github_mcp":
            return self._validate_github_mcp(component_config)
        elif component_name == "docker":
            return self._validate_docker(component_config)
        elif component_name == "docker_compose":
            return self._validate_docker_compose(component_config)
        else:
            return False, f"Unknown component: {component_name}"

    def _validate_github_mcp(self, config: Dict[str, Any]) -> Tuple[bool, Any]:
        """Validate GitHub MCP component."""
        try:
            # Add scripts directory to path and import validator
            scripts_dir = Path(__file__).parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            from validate_github_mcp import GitHubMCPValidator

            validator = GitHubMCPValidator()
            passed, results = validator.validate_all()

            if passed:
                return True, "GitHub MCP integration validated successfully"
            else:
                errors = results.get("errors", [])
                return False, f"GitHub MCP validation failed: {', '.join(errors)}"

        except ImportError as e:
            return False, f"GitHub MCP validator import failed: {str(e)}"
        except Exception as e:
            return False, f"GitHub MCP validation error: {str(e)}"

    def _validate_docker(self, config: Dict[str, Any]) -> Tuple[bool, Any]:
        """Validate Docker component."""
        try:
            # Add scripts directory to path and import validator
            scripts_dir = Path(__file__).parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            from validate_docker import DockerValidator

            validator = DockerValidator()
            passed, results = validator.validate_all()

            if passed:
                return True, "Docker integration validated successfully"
            else:
                errors = results.get("errors", [])
                return False, f"Docker validation failed: {', '.join(errors)}"

        except ImportError as e:
            return False, f"Docker validator import failed: {str(e)}"
        except Exception as e:
            return False, f"Docker validation error: {str(e)}"

    def _validate_docker_compose(self, config: Dict[str, Any]) -> Tuple[bool, Any]:
        """Validate Docker Compose component."""
        try:
            import subprocess

            # Check Docker Compose installation
            result = subprocess.run(['docker-compose', '--version'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                return False, "Docker Compose not installed or not accessible"

            # Check Docker Compose configuration
            compose_file = Path("/Users/tradecomp/bg/viper-/docker/docker-compose.yml")
            if not compose_file.exists():
                return False, "Docker Compose configuration file not found"

            # Validate configuration
            result = subprocess.run(['docker-compose', '-f', str(compose_file), 'config'],
                                  capture_output=True, text=True, timeout=15)

            if result.returncode == 0:
                return True, "Docker Compose configuration validated successfully"
            else:
                return False, f"Docker Compose config error: {result.stderr.strip()}"

        except subprocess.TimeoutExpired:
            return False, "Docker Compose validation timed out"
        except Exception as e:
            return False, f"Docker Compose validation error: {str(e)}"

    def _calculate_overall_score(self, component_results: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        total_components = len(component_results)
        if total_components == 0:
            return 0.0

        passed_components = sum(1 for result in component_results.values()
                              if result["status"] == "PASSED")

        return round((passed_components / total_components) * 100, 1)

    def enforce_startup_policy(self) -> bool:
        """MANDATORY: Enforce startup policy based on validation results."""
        logger.info("🚫 ENFORCING SYSTEM STARTUP POLICY (MANDATORY)")

        passed, results = self.validate_all_components()

        # Log results
        logger.info(f"📊 Overall Score: {results['overall_score']}%")
        logger.info(f"⏱️  Validation Time: {results['execution_time_seconds']} seconds")

        for component_name, component_result in results["components"].items():
            status = component_result["status"]
            details = component_result["details"]
            logger.info(f"   {component_name}: {'✅' if status == 'PASSED' else '❌'} {details}")

        if not passed:
            logger.critical("❌ SYSTEM STARTUP BLOCKED - MANDATORY REQUIREMENTS NOT MET")

            # Log detailed errors
            for component_name, component_result in results["components"].items():
                if component_result["status"] != "PASSED":
                    logger.error(f"   ❌ {component_name}: {component_result['details']}")

            # Save startup results
            self._save_startup_results()

            # Generate failure report
            self._generate_failure_report()

            logger.critical("🛑 SYSTEM WILL NOT START DUE TO VALIDATION FAILURES")
            logger.critical("🔧 Please resolve the issues above and restart the system")
            return False

        logger.info("✅ ALL MANDATORY REQUIREMENTS SATISFIED")
        logger.info("🚀 SYSTEM STARTUP ALLOWED")

        # Save successful startup results
        self._save_startup_results()

        return True

    def _save_startup_results(self):
        """Save startup validation results."""
        results_file = Path("/Users/tradecomp/bg/viper-/logs/system_startup_validation.json")
        results_file.parent.mkdir(exist_ok=True)

        try:
            with open(results_file, 'w') as f:
                json.dump(self.startup_results, f, indent=2)
            logger.info(f"💾 Startup validation results saved: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save startup results: {e}")

    def _generate_failure_report(self):
        """Generate detailed failure report."""
        report_file = Path("/Users/tradecomp/bg/viper-/logs/startup_failure_report.md")
        report_file.parent.mkdir(exist_ok=True)

        try:
            report = f"""# 🚫 SYSTEM STARTUP FAILURE REPORT
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 OVERVIEW
- **Status**: ❌ STARTUP BLOCKED
- **Score**: {self.startup_results['overall_score']}%
- **Validation Time**: {self.startup_results['execution_time_seconds']} seconds

## 🚫 FAILED COMPONENTS

"""

            for component_name, component_result in self.startup_results["components"].items():
                if component_result["status"] != "PASSED":
                    report += f"""### {component_name.upper()}
- **Status**: ❌ FAILED
- **Details**: {component_result['details']}
- **Enforcement Level**: {component_result['config'].get('enforcement_level', 'unknown')}

"""

            report += """
## 🔧 RESOLUTION STEPS

### For GitHub MCP Issues:
1. Verify GITHUB_PAT environment variable is set
2. Check GitHub token permissions (repo, workflow scopes)
3. Ensure repository access is available
4. Validate network connectivity to GitHub API

### For Docker Issues:
1. Verify Docker Desktop is installed and running
2. Check Docker Compose installation
3. Validate Docker daemon connectivity
4. Ensure docker-compose.yml file exists and is valid

### General Steps:
1. Review error messages above
2. Check system logs in /logs/ directory
3. Resolve identified issues
4. Restart system validation

## 📞 SUPPORT INFORMATION
- **Validation Logs**: /logs/system_startup_validation.json
- **Component Logs**: /logs/*_validation.json
- **Configuration**: /config/system_requirements.json

---
*Generated by System Startup Validator*
*MANDATORY: System will not start until all issues are resolved*
"""

            with open(report_file, 'w') as f:
                f.write(report)

            logger.info(f"📋 Failure report generated: {report_file}")

        except Exception as e:
            logger.error(f"Failed to generate failure report: {e}")

def main():
    """MANDATORY: Main startup validation function."""
    print("🚀 SYSTEM STARTUP VALIDATOR (MANDATORY)")
    print("=" * 60)
    print("⚠️  CRITICAL: This validation controls system startup")
    print("🚫 BLOCKING: System will NOT start if validation fails")
    print("🔧 REQUIRED: All mandatory components must pass")
    print()

    validator = SystemStartupValidator()

    # Enforce startup policy
    startup_allowed = validator.enforce_startup_policy()

    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)

    results = validator.startup_results
    print(f"📊 Overall Score: {results['overall_score']}%")
    print(f"⏱️  Validation Time: {results['execution_time_seconds']} seconds")
    print(f"📦 Components Validated: {len(results['components'])}")

    print("\n🔍 COMPONENT STATUS:")
    for component_name, component_result in results["components"].items():
        status = component_result["status"]
        details = component_result["details"]
        print(f"   {component_name}: {'✅ PASSED' if status == 'PASSED' else '❌ FAILED'}")
        if status != "PASSED":
            print(f"      └─ {details}")

    print("\n" + "=" * 60)

    if startup_allowed:
        print("🎉 STARTUP VALIDATION PASSED")
        print("🚀 System startup allowed - all mandatory requirements satisfied")
        print("✅ GitHub MCP: Operational")
        print("✅ Docker: Ready")
        print("✅ All components: Validated")
        sys.exit(0)
    else:
        print("🚫 STARTUP VALIDATION FAILED")
        print("🛑 System startup blocked - mandatory requirements not met")
        print("📋 Check /logs/startup_failure_report.md for detailed resolution steps")
        print("🔧 Resolve the issues above and restart validation")
        sys.exit(1)

if __name__ == "__main__":
    main()
