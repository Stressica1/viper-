#!/usr/bin/env python3
"""
🚨 VIPER POLICY ENFORCEMENT TASK
=================================

GitHub MCP Task for Enforcing ALL VIPER Policies
- Structure enforcement
- Naming conventions
- Version control policies
- Security policies
- Code quality standards
- Development workflow

This task is ALWAYS ACTIVE and monitors all operations for policy compliance.

Author: VIPER Development Team
Version: 2.0.0
Date: 2025-08-29
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import from monitoring scripts directory
from scripts.monitoring.github_mcp_integration import GitHubMCPOrchestration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - POLICY_ENFORCEMENT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'policy_enforcement.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PolicyEnforcementTask:
    """Comprehensive policy enforcement task for VIPER"""

    def __init__(self):
        self.project_root = project_root
        self.mcp_orchestrator = GitHubMCPOrchestration(repo_path=str(self.project_root))
        self.enforcement_stats = {
            'total_operations': 0,
            'violations_detected': 0,
            'blocking_actions': 0,
            'compliance_score': 100
        }

        logger.info("🚨 Policy Enforcement Task initialized - ALL POLICIES ACTIVE")

    async def run_continuous_policy_enforcement(self):
        """Run continuous policy enforcement monitoring"""
        logger.info("🚨 Starting continuous policy enforcement monitoring...")

        while True:
            try:
                # Monitor file system for changes
                await self._monitor_file_system_changes()

                # Check current project state
                await self._check_project_compliance()

                # Generate periodic reports
                await self._generate_periodic_reports()

                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except KeyboardInterrupt:
                logger.info("🛑 Policy enforcement monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Policy enforcement error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def enforce_policy_on_operation(self, operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce policies on a specific operation"""
        logger.info(f"🚨 Enforcing policies on operation: {operation_context.get('operation_type', 'unknown')}")

        self.enforcement_stats['total_operations'] += 1

        # Run comprehensive policy enforcement
        enforcement_results = await self.mcp_orchestrator.run_policy_enforcement_workflow(operation_context)

        # Update statistics
        if enforcement_results.get('policy_violations'):
            self.enforcement_stats['violations_detected'] += len(enforcement_results['policy_violations'])
            self.enforcement_stats['blocking_actions'] += enforcement_results.get('blocking_violations', 0)

        # Update compliance score
        if enforcement_results.get('compliance_score'):
            self.enforcement_stats['compliance_score'] = enforcement_results['compliance_score']

        logger.info(f"🚨 Policy enforcement completed - Score: {self.enforcement_stats['compliance_score']}%")

        return enforcement_results

    async def _monitor_file_system_changes(self):
        """Monitor file system for policy violations"""
        logger.debug("🔍 Monitoring file system for policy violations...")

        # Check for new files that might violate policies
        for file_path in self._get_all_python_files():
            operation_context = {
                'operation_type': 'file_monitoring',
                'file_path': str(file_path),
                'timestamp': datetime.now().isoformat()
            }

            # Read file content for analysis
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                operation_context['file_content'] = content
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")

            # Enforce policies on this file
            await self.enforce_policy_on_operation(operation_context)

    async def _check_project_compliance(self):
        """Check overall project compliance"""
        logger.debug("📊 Checking project compliance...")

        # Check directory structure
        await self._check_directory_structure()

        # Check naming conventions across project
        await self._check_naming_conventions()

        # Check version control compliance
        await self._check_version_control_compliance()

    async def _check_directory_structure(self):
        """Check directory structure compliance"""
        required_dirs = ['config', 'scripts', 'services', 'src', 'docs', 'tests']
        missing_dirs = []

        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            operation_context = {
                'operation_type': 'directory_check',
                'missing_directories': missing_dirs,
                'timestamp': datetime.now().isoformat()
            }
            await self.enforce_policy_on_operation(operation_context)

    async def _check_naming_conventions(self):
        """Check naming conventions across project"""
        for file_path in self._get_all_python_files():
            file_name = file_path.name

            # Check if filename violates naming policy
            if not self._is_valid_filename(file_name):
                operation_context = {
                    'operation_type': 'filename_check',
                    'file_path': str(file_path),
                    'file_name': file_name,
                    'timestamp': datetime.now().isoformat()
                }
                await self.enforce_policy_on_operation(operation_context)

    async def _check_version_control_compliance(self):
        """Check version control compliance"""
        try:
            # Check current branch name
            import subprocess
            result = subprocess.run(['git', 'branch', '--show-current'],
                                  capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                current_branch = result.stdout.strip()
                if current_branch and not self._is_valid_branch_name(current_branch):
                    operation_context = {
                        'operation_type': 'branch_check',
                        'branch_name': current_branch,
                        'timestamp': datetime.now().isoformat()
                    }
                    await self.enforce_policy_on_operation(operation_context)

        except Exception as e:
            logger.warning(f"Could not check git branch: {e}")

    def _get_all_python_files(self):
        """Get all Python files in project"""
        python_files = []
        for file_path in self.project_root.rglob('*.py'):
            # Skip certain directories
            if any(skip_dir in str(file_path) for skip_dir in ['__pycache__', '.git', 'node_modules']):
                continue
            python_files.append(file_path)
        return python_files

    def _is_valid_filename(self, filename: str) -> bool:
        """Check if filename follows naming conventions"""
        if not filename.endswith('.py'):
            return True

        name_without_ext = filename[:-3]

        # Must be snake_case (allow single word files)
        if '_' not in name_without_ext and len(name_without_ext) > 1:
            return False

        # Check for invalid patterns
        invalid_patterns = ['-', 'camelCase', 'PascalCase']
        for pattern in invalid_patterns:
            if pattern in name_without_ext:
                return False

        return True

    def _is_valid_branch_name(self, branch_name: str) -> bool:
        """Check if branch name follows conventions"""
        valid_prefixes = ['feature/', 'bugfix/', 'hotfix/', 'release/']
        return any(branch_name.startswith(prefix) for prefix in valid_prefixes)

    async def _generate_periodic_reports(self):
        """Generate periodic compliance reports"""
        # Generate report every 5 minutes
        if hasattr(self, '_last_report_time'):
            if (datetime.now() - self._last_report_time).seconds < 300:
                return
        else:
            self._last_report_time = datetime.now()

        self._last_report_time = datetime.now()

        # Generate compliance report
        report = await self.mcp_orchestrator.generate_policy_compliance_report()

        # Log report summary
        logger.info(f"📊 Compliance Report - Score: {report['overall_compliance']:.1f}%")
        logger.info(f"📊 Total Operations: {self.enforcement_stats['total_operations']}")
        logger.info(f"📊 Violations Detected: {self.enforcement_stats['violations_detected']}")
        logger.info(f"📊 Blocking Actions: {self.enforcement_stats['blocking_actions']}")

        # Save report to file
        report_path = self.project_root / 'reports' / f'policy_compliance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump({
                'report': report,
                'enforcement_stats': self.enforcement_stats,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

    async def run_manual_policy_check(self, target_path: str = None):
        """Run manual policy check on specific path or entire project"""
        logger.info(f"🔍 Running manual policy check on: {target_path or 'entire project'}")

        if target_path:
            # Check specific file/directory
            path_obj = Path(target_path)
            if path_obj.is_file():
                operation_context = {
                    'operation_type': 'manual_file_check',
                    'file_path': str(path_obj),
                    'timestamp': datetime.now().isoformat()
                }

                if path_obj.suffix == '.py':
                    try:
                        with open(path_obj, 'r', encoding='utf-8') as f:
                            operation_context['file_content'] = f.read()
                    except Exception as e:
                        logger.error(f"Could not read file {path_obj}: {e}")

                result = await self.enforce_policy_on_operation(operation_context)
                return result
            else:
                # Check directory
                violations = []
                for file_path in path_obj.rglob('*.py'):
                    operation_context = {
                        'operation_type': 'manual_directory_check',
                        'file_path': str(file_path),
                        'timestamp': datetime.now().isoformat()
                    }
                    result = await self.enforce_policy_on_operation(operation_context)
                    if result.get('policy_violations'):
                        violations.extend(result['policy_violations'])

                return {'directory': str(path_obj), 'total_violations': len(violations), 'violations': violations}
        else:
            # Check entire project
            await self._check_project_compliance()
            return {'message': 'Project-wide policy check completed'}

    async def demonstrate_policy_enforcement(self):
        """Demonstrate policy enforcement with sample violations"""
        logger.info("🎯 Demonstrating policy enforcement...")

        # Create sample violations for demonstration
        demo_violations = [
            {
                'operation_type': 'demo_violation',
                'file_path': 'scripts/CamelCaseFile.py',  # Invalid naming
                'file_content': 'class camelCaseClass:\n    def CamelCaseMethod(self):\n        api_key = "sk-hardcoded-key"',  # Multiple violations
                'commit_message': 'fix bug',  # Invalid commit message
                'branch_name': 'my-feature-branch'  # Invalid branch name
            }
        ]

        for violation in demo_violations:
            logger.info(f"🚨 Testing policy enforcement on: {violation['file_path']}")
            result = await self.enforce_policy_on_operation(violation)

            if result.get('policy_violations'):
                logger.info(f"✅ Detected {len(result['policy_violations'])} violations")
                for v in result['policy_violations'][:3]:  # Show first 3
                    logger.info(f"   • {v['type'].replace('_', ' ').title()}: {v['description']}")

async def main():
    """Main function for policy enforcement task"""
    print("🚨 VIPER POLICY ENFORCEMENT TASK")
    print("=" * 50)
    print("🔒 Enforcing ALL project policies:")
    print("   • Project structure")
    print("   • Naming conventions")
    print("   • Version control")
    print("   • Security policies")
    print("   • Code quality")
    print("   • Development workflow")
    print("=" * 50)

    # Initialize policy enforcement task
    enforcement_task = PolicyEnforcementTask()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'check':
            # Manual check
            target = sys.argv[2] if len(sys.argv) > 2 else None
            result = await enforcement_task.run_manual_policy_check(target)
            print(f"✅ Manual check completed: {result}")

        elif command == 'demo':
            # Demonstration mode
            await enforcement_task.demonstrate_policy_enforcement()

        elif command == 'continuous':
            # Continuous monitoring
            print("🔄 Starting continuous policy enforcement monitoring...")
            print("Press Ctrl+C to stop")
            await enforcement_task.run_continuous_policy_enforcement()

        else:
            print(f"❌ Unknown command: {command}")
            print("Usage: python policy_enforcement_task.py [check|demo|continuous] [target]")
    else:
        # Default: Run manual check on entire project
        print("🔍 Running project-wide policy check...")
        result = await enforcement_task.run_manual_policy_check()
        print(f"✅ Policy check completed: {result}")

if __name__ == "__main__":
    asyncio.run(main())
