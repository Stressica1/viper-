#!/usr/bin/env python3
"""
🚀 GITHUB MCP VALIDATOR
=======================

MANDATORY: Validates GitHub MCP integration requirements.
This script MUST pass for system startup.

Features:
✅ GitHub token validation
✅ Repository access verification
✅ API connectivity testing
✅ Permission scope checking
✅ Rate limit monitoring
✅ Error recovery validation
✅ Startup blocking enforcement
"""

import os
import sys
import json
import time
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
    format='%(asctime)s - GITHUB_MCP_VALIDATOR - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubMCPValidator:
    """MANDATORY: GitHub MCP Integration Validator"""

    def __init__(self):
        self.config_path = Path("/Users/tradecomp/bg/viper-/config/system_requirements.json")
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "component": "github_mcp",
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
        """MANDATORY: Run all GitHub MCP validation checks."""
        logger.info("🚀 STARTING GITHUB MCP VALIDATION (MANDATORY)")

        checks = {
            "github_token": self._validate_github_token,
            "repository_access": self._validate_repository_access,
            "api_connectivity": self._validate_api_connectivity,
            "permission_scopes": self._validate_permission_scopes,
            "rate_limits": self._validate_rate_limits,
            "error_recovery": self._validate_error_recovery
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

        logger.info(f"🎯 GITHUB MCP VALIDATION COMPLETE: {'✅ PASSED' if all_passed else '❌ FAILED'}")
        return all_passed, self.validation_results

    def _validate_github_token(self) -> Tuple[bool, str]:
        """MANDATORY: Validate GitHub token presence and format."""
        github_token = os.getenv('GITHUB_PAT') or os.getenv('GITHUB_TOKEN')

        if not github_token:
            return False, "GitHub token not found in environment variables"

        # Check token format
        if not (github_token.startswith('ghp_') or github_token.startswith('github_pat_') or github_token.startswith('gho_')):
            return False, "Invalid GitHub token format"

        if len(github_token) < 20:
            return False, "GitHub token appears to be too short"

        return True, f"GitHub token validated (type: {'Classic' if github_token.startswith('ghp_') else 'Fine-grained' if github_token.startswith('github_pat_') else 'App'})"

    def _validate_repository_access(self) -> Tuple[bool, str]:
        """MANDATORY: Validate repository access permissions."""
        github_token = os.getenv('GITHUB_PAT') or os.getenv('GITHUB_TOKEN')
        github_owner = os.getenv('GITHUB_OWNER', 'tradecomp')
        github_repo = os.getenv('GITHUB_REPO', 'viper')

        if not github_token:
            return False, "GitHub token required for repository access validation"

        try:
            import requests

            headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            # Test repository access
            repo_url = f'https://api.github.com/repos/{github_owner}/{github_repo}'
            response = requests.get(repo_url, headers=headers, timeout=10)

            if response.status_code == 200:
                repo_data = response.json()
                permissions = repo_data.get('permissions', {})

                if permissions.get('push', False):
                    return True, f"Repository access validated: {github_owner}/{github_repo} (push access granted)"
                elif permissions.get('pull', False):
                    return True, f"Repository access validated: {github_owner}/{github_repo} (pull access only)"
                else:
                    return False, f"Insufficient repository permissions: {permissions}"
            elif response.status_code == 404:
                return False, f"Repository not found: {github_owner}/{github_repo}"
            elif response.status_code == 401:
                return False, "GitHub token authentication failed"
            else:
                return False, f"Repository access check failed: HTTP {response.status_code}"

        except ImportError:
            return False, "requests library not available for API testing"
        except Exception as e:
            return False, f"Repository access validation error: {str(e)}"

    def _validate_api_connectivity(self) -> Tuple[bool, str]:
        """MANDATORY: Validate GitHub API connectivity."""
        try:
            import requests

            # Test basic API connectivity
            response = requests.get('https://api.github.com/zen', timeout=5)

            if response.status_code == 200:
                return True, f"GitHub API connectivity confirmed (response: {len(response.text)} chars)"
            else:
                return False, f"GitHub API connectivity failed: HTTP {response.status_code}"

        except ImportError:
            return False, "requests library not available for connectivity testing"
        except requests.exceptions.RequestException as e:
            return False, f"GitHub API connectivity error: {str(e)}"

    def _validate_permission_scopes(self) -> Tuple[bool, str]:
        """MANDATORY: Validate GitHub token permission scopes."""
        github_token = os.getenv('GITHUB_PAT') or os.getenv('GITHUB_TOKEN')

        if not github_token:
            return False, "GitHub token required for scope validation"

        try:
            import requests

            headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            # Test scopes through API
            response = requests.get('https://api.github.com/user', headers=headers, timeout=10)

            if response.status_code == 200:
                # Check X-OAuth-Scopes header for available scopes
                scopes_header = response.headers.get('X-OAuth-Scopes', '')
                scopes = [scope.strip() for scope in scopes_header.split(',') if scope.strip()]

                required_scopes = ['repo', 'workflow']
                missing_scopes = [scope for scope in required_scopes if scope not in scopes]

                if missing_scopes:
                    return False, f"Missing required scopes: {', '.join(missing_scopes)} (available: {', '.join(scopes)})"
                else:
                    return True, f"All required scopes available: {', '.join(scopes)}"
            else:
                return False, f"Scope validation failed: HTTP {response.status_code}"

        except ImportError:
            return False, "requests library not available for scope testing"
        except Exception as e:
            return False, f"Scope validation error: {str(e)}"

    def _validate_rate_limits(self) -> Tuple[bool, str]:
        """MANDATORY: Validate GitHub API rate limits."""
        github_token = os.getenv('GITHUB_PAT') or os.getenv('GITHUB_TOKEN')

        if not github_token:
            return False, "GitHub token required for rate limit validation"

        try:
            import requests

            headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            # Check rate limit status
            response = requests.get('https://api.github.com/rate_limit', headers=headers, timeout=10)

            if response.status_code == 200:
                rate_data = response.json()
                core_limits = rate_data.get('resources', {}).get('core', {})

                remaining = core_limits.get('remaining', 0)
                limit = core_limits.get('limit', 5000)
                reset_time = core_limits.get('reset', 0)

                if remaining > 100:  # Safe buffer
                    reset_datetime = datetime.fromtimestamp(reset_time)
                    return True, f"Rate limit OK: {remaining}/{limit} remaining, resets at {reset_datetime}"
                else:
                    return False, f"Rate limit too low: {remaining}/{limit} remaining"
            else:
                return False, f"Rate limit check failed: HTTP {response.status_code}"

        except ImportError:
            return False, "requests library not available for rate limit testing"
        except Exception as e:
            return False, f"Rate limit validation error: {str(e)}"

    def _validate_error_recovery(self) -> Tuple[bool, str]:
        """MANDATORY: Validate error recovery mechanisms."""
        try:
            # Check if MCP integration scripts exist
            mcp_script = Path("/Users/tradecomp/bg/viper-/scripts/github_mcp_task_tracker.py")
            if not mcp_script.exists():
                return False, "GitHub MCP task tracker script not found"

            # Check if error recovery configuration exists
            recovery_config = Path("/Users/tradecomp/bg/viper-/config/system_requirements.json")
            if not recovery_config.exists():
                return False, "Error recovery configuration not found"

            # Validate error recovery settings
            with open(recovery_config, 'r') as f:
                config = json.load(f)

            mcp_config = config.get('system_requirements', {}).get('mandatory_components', {}).get('github_mcp', {})

            if mcp_config.get('configuration', {}).get('error_recovery', False):
                return True, "Error recovery mechanisms configured and validated"
            else:
                return False, "Error recovery not enabled in configuration"

        except Exception as e:
            return False, f"Error recovery validation failed: {str(e)}"

    def enforce_requirements(self) -> bool:
        """MANDATORY: Enforce GitHub MCP requirements (blocking)."""
        logger.info("🚫 ENFORCING GITHUB MCP REQUIREMENTS (MANDATORY)")

        passed, results = self.validate_all()

        if not passed:
            logger.critical("❌ GITHUB MCP VALIDATION FAILED - SYSTEM STARTUP BLOCKED")
            logger.critical("🚫 CRITICAL: GitHub MCP is MANDATORY for system operation")

            # Log all errors
            for error in results["errors"]:
                logger.error(f"   ❌ {error}")

            # Save validation results
            self._save_validation_results()

            # Exit with failure (blocks system startup)
            logger.critical("🛑 SYSTEM STARTUP PREVENTED DUE TO MANDATORY REQUIREMENT FAILURE")
            return False

        logger.info("✅ GITHUB MCP REQUIREMENTS ENFORCED - ALL CHECKS PASSED")
        self._save_validation_results()
        return True

    def _save_validation_results(self):
        """Save validation results to file."""
        results_file = Path("/Users/tradecomp/bg/viper-/logs/github_mcp_validation.json")
        results_file.parent.mkdir(exist_ok=True)

        try:
            with open(results_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            logger.info(f"💾 Validation results saved: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")

def main():
    """MANDATORY: Main validation function."""
    print("🚀 GITHUB MCP VALIDATOR (MANDATORY REQUIREMENT)")
    print("=" * 60)
    print("⚠️  WARNING: This validation MUST pass for system operation")
    print("🚫 FAILURE: Will block system startup")
    print()

    validator = GitHubMCPValidator()

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
        print("🛑 System startup blocked due to GitHub MCP validation failure")
        print("🔧 Please resolve the errors above and try again")
        sys.exit(1)
    else:
        print("\n✅ MANDATORY REQUIREMENT SATISFIED")
        print("🚀 GitHub MCP integration validated successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
