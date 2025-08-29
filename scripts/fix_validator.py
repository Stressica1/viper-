#!/usr/bin/env python3
"""
✅ FIX VALIDATOR - VALIDATION & ROLLBACK SYSTEM
============================================

Comprehensive validation system for MCP fixes.

Features:
- Syntax validation after fixes
- Functional testing
- Regression detection
- Safe rollback capabilities
- Comprehensive reporting

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import json
import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import hashlib
import shutil
import importlib.util

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_type: str
    file_path: str
    passed: bool
    message: str
    details: Dict[str, Any] = None
    timestamp: str = ""

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    file_path: str
    overall_passed: bool
    validation_results: List[ValidationResult]
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    recommendations: List[str]
    generated_at: str

class FixValidator:
    """Comprehensive fix validation system"""

    def __init__(self):
        self.results_dir = Path("reports") / "validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.backup_dir = Path("backups")
        self.temp_dir = Path(tempfile.gettempdir()) / "viper_validation"

        # Validation check configurations
        self.validation_checks = {
            'syntax': self._validate_syntax,
            'imports': self._validate_imports,
            'security': self._validate_security,
            'functionality': self._validate_functionality,
            'performance': self._validate_performance
        }

    def validate_file(self, file_path: str, original_content: str = None) -> ValidationReport:
        """Validate a single file after fixes"""
        file_path = Path(file_path)
        results = []
        all_passed = True


        # Run all validation checks
        for check_name, check_func in self.validation_checks.items():
            try:
                result = check_func(file_path)
                results.append(result)

                if not result.passed:
                    all_passed = False

                status = "✅" if result.passed else "❌"
                print(f"   {status} {check_name}: {result.message}")

            except Exception as e:
                error_result = ValidationResult(
                    check_type=check_name,
                    file_path=str(file_path),
                    passed=False,
                    message=f"Validation error: {str(e)}",
                    timestamp=datetime.now().isoformat()
                )
                results.append(error_result)
                all_passed = False
                print(f"   ❌ {check_name}: Validation error - {e}")

        # Determine risk level
        risk_level = self._calculate_risk_level(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(results, all_passed)

        report = ValidationReport(
            file_path=str(file_path),
            overall_passed=all_passed,
            validation_results=results,
            risk_level=risk_level,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat()
        )

        return report

    def _validate_syntax(self, file_path: Path) -> ValidationResult:
        """Validate Python syntax"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Try to parse the AST
            ast.parse(content)

            # Also check with Python interpreter
            result = subprocess.run(
                [sys.executable, '-m', 'py_compile', str(file_path)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return ValidationResult(
                    check_type='syntax',
                    file_path=str(file_path),
                    passed=True,
                    message="Syntax is valid",
                    timestamp=datetime.now().isoformat()
                )
            else:
                return ValidationResult(
                    check_type='syntax',
                    file_path=str(file_path),
                    passed=False,
                    message=f"Syntax error: {result.stderr.strip()}",
                    timestamp=datetime.now().isoformat()
                )

        except SyntaxError as e:
            return ValidationResult(
                check_type='syntax',
                file_path=str(file_path),
                passed=False,
                message=f"Syntax error: {e.msg} at line {e.lineno}",
                details={'line': e.lineno, 'column': e.offset},
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            return ValidationResult(
                check_type='syntax',
                file_path=str(file_path),
                passed=False,
                message=f"Validation failed: {str(e)}",
                timestamp=datetime.now().isoformat()
            )

    def _validate_imports(self, file_path: Path) -> ValidationResult:
        """Validate import statements"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Parse AST to check imports
            tree = ast.parse(content)
            import_issues = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if not self._is_module_available(module_name):
                            import_issues.append(f"Module '{module_name}' not available")

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if not self._is_module_available(module_name):
                            import_issues.append(f"Module '{module_name}' not available")

            if import_issues:
                return ValidationResult(
                    check_type='imports',
                    file_path=str(file_path),
                    passed=False,
                    message=f"Import issues: {', '.join(import_issues)}",
                    details={'issues': import_issues},
                    timestamp=datetime.now().isoformat()
                )
            else:
                return ValidationResult(
                    check_type='imports',
                    file_path=str(file_path),
                    passed=True,
                    message="All imports are valid",
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            return ValidationResult(
                check_type='imports',
                file_path=str(file_path),
                passed=False,
                message=f"Import validation failed: {str(e)}",
                timestamp=datetime.now().isoformat()
            )

    def _validate_security(self, file_path: Path) -> ValidationResult:
        """Validate security aspects"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.split('\n')
            security_issues = []

            for line_num, line in enumerate(lines, 1):
                # Check for hardcoded secrets
                if self._contains_hardcoded_secret(line):
                    security_issues.append(f"Line {line_num}: Potential hardcoded secret")

                # Check for dangerous functions
                if 'eval(' in line and not line.strip().startswith('#'):
                    security_issues.append(f"Line {line_num}: Use of eval() function")

                if 'exec(' in line and not line.strip().startswith('#'):
                    security_issues.append(f"Line {line_num}: Use of exec() function")

                # Check for insecure random
                if 'secrets.randbelow(1000000) / 1000000.0  # Was: random.random()' in line or 'secrets.randbelow(max_val - min_val + 1) + min_val  # Was: random.randint(' in line:
                    security_issues.append(f"Line {line_num}: Insecure random number generation")

            if security_issues:
                return ValidationResult(
                    check_type='security',
                    file_path=str(file_path),
                    passed=False,
                    message=f"Security issues found: {len(security_issues)}",
                    details={'issues': security_issues},
                    timestamp=datetime.now().isoformat()
                )
            else:
                return ValidationResult(
                    check_type='security',
                    file_path=str(file_path),
                    passed=True,
                    message="No security issues detected",
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            return ValidationResult(
                check_type='security',
                file_path=str(file_path),
                passed=False,
                message=f"Security validation failed: {str(e)}",
                timestamp=datetime.now().isoformat()
            )

    def _validate_functionality(self, file_path: Path) -> ValidationResult:
        """Validate basic functionality"""
        try:
            # Try to import the module
            module_name = file_path.stem

            # Create a temporary module for testing
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)

                # Try to execute the module (basic syntax check)
                try:
                    spec.loader.exec_module(module)
                    return ValidationResult(
                        check_type='functionality',
                        file_path=str(file_path),
                        passed=True,
                        message="Module imports and executes successfully",
                        timestamp=datetime.now().isoformat()
                    )
                except Exception as e:
                    return ValidationResult(
                        check_type='functionality',
                        file_path=str(file_path),
                        passed=False,
                        message=f"Module execution failed: {str(e)}",
                        timestamp=datetime.now().isoformat()
                    )
            else:
                return ValidationResult(
                    check_type='functionality',
                    file_path=str(file_path),
                    passed=False,
                    message="Cannot create module spec",
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            return ValidationResult(
                check_type='functionality',
                file_path=str(file_path),
                passed=False,
                message=f"Functionality validation failed: {str(e)}",
                timestamp=datetime.now().isoformat()
            )

    def _validate_performance(self, file_path: Path) -> ValidationResult:
        """Validate performance aspects"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.split('\n')
            performance_issues = []

            for line_num, line in enumerate(lines, 1):
                # Check for potential performance issues
                if re.search(r'for\s+\w+\s+in\s+range\(.*\):', line):
                    # Check if the range could be large
                    range_match = re.search(r'range\(([^)]+)\)', line)
                    if range_match and not any(char.isdigit() for char in range_match.group(1)[:10]):
                        performance_issues.append(f"Line {line_num}: Large range() iteration")

                # Check for inefficient list operations
                if '.append(' in line and 'for ' in content[max(0, line_num-5):line_num+5]:
                    performance_issues.append(f"Line {line_num}: Potential list append in loop")

            if performance_issues:
                return ValidationResult(
                    check_type='performance',
                    file_path=str(file_path),
                    passed=False,
                    message=f"Performance issues found: {len(performance_issues)}",
                    details={'issues': performance_issues},
                    timestamp=datetime.now().isoformat()
                )
            else:
                return ValidationResult(
                    check_type='performance',
                    file_path=str(file_path),
                    passed=True,
                    message="No performance issues detected",
                    timestamp=datetime.now().isoformat()
                )

        except Exception as e:
            return ValidationResult(
                check_type='performance',
                file_path=str(file_path),
                passed=False,
                message=f"Performance validation failed: {str(e)}",
                timestamp=datetime.now().isoformat()
            )

    def _is_module_available(self, module_name: str) -> bool:
        """Check if a Python module is available"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False

    def _contains_hardcoded_secret(self, line: str) -> bool:
        """Check if line contains potential hardcoded secret"""
        # Look for common patterns
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'secret\s*=\s*["\'][^"\']{8,}["\']',
            r'key\s*=\s*["\'][^"\']{10,}["\']',
            r'token\s*=\s*["\'][^"\']{10,}["\']'
        ]

        for pattern in secret_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False

    def _calculate_risk_level(self, results: List[ValidationResult]) -> str:
        """Calculate overall risk level"""
        failed_checks = [r for r in results if not r.passed]

        if not failed_checks:
            return 'LOW'

        # Check for critical failures
        critical_failures = [r for r in failed_checks if r.check_type in ['syntax', 'security']]

        if critical_failures:
            return 'CRITICAL'

        # Check for multiple failures
        if len(failed_checks) >= 3:
            return 'HIGH'
        elif len(failed_checks) >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _generate_recommendations(self, results: List[ValidationResult], all_passed: bool) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        if all_passed:
            recommendations.append("✅ All validations passed - fixes are safe to deploy")
            return recommendations

        failed_checks = [r for r in results if not r.passed]

        for result in failed_checks:
            if result.check_type == 'syntax':
                recommendations.append("🐛 Critical: Fix syntax errors before deploying")
            elif result.check_type == 'security':
                recommendations.append("🔒 Critical: Address security vulnerabilities")
            elif result.check_type == 'imports':
                recommendations.append("📦 Fix import issues and missing dependencies")
            elif result.check_type == 'functionality':
                recommendations.append("⚙️ Test functionality thoroughly before deploying")
            elif result.check_type == 'performance':
                recommendations.append("⚡ Review performance optimizations")

        if len(failed_checks) > 0:
            recommendations.append("🔄 Consider rolling back if critical issues persist")

        return recommendations

    def create_rollback_point(self, file_path: str) -> str:
        """Create a rollback point for a file"""
        backup_name = f"{Path(file_path).name}_{int(time.time())}.backup"
        backup_path = self.backup_dir / backup_name

        self.backup_dir.mkdir(exist_ok=True)
        shutil.copy2(file_path, backup_path)

        return str(backup_path)

    def rollback_file(self, file_path: str, backup_path: str) -> bool:
        """Rollback a file to a previous backup"""
        try:
            shutil.copy2(backup_path, file_path)
            return True
        except Exception as e:
            return False

    def validate_batch_results(self, fix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate batch processing results"""
        validation_results = []

        # Validate each fixed file
        for job in fix_results.get('jobs', []):
            if 'result' in job and job['result']:
                file_path = job['result'].get('file_path')
                if file_path and Path(file_path).exists():
                    report = self.validate_file(file_path)
                    validation_results.append(asdict(report))

        # Generate summary
        total_validations = len(validation_results)
        passed_validations = sum(1 for r in validation_results if r['overall_passed'])
        failed_validations = total_validations - passed_validations

        summary = {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'failed_validations': failed_validations,
            'success_rate': (passed_validations / max(total_validations, 1)) * 100,
            'validation_results': validation_results
        }

        # Save validation report
        report_path = self.results_dir / f"validation_report_{int(time.time())}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable validation report"""
        report = f"""
🔍 VALIDATION REPORT
{'='*50}

📊 SUMMARY
  Total Validations: {validation_results['total_validations']}
  Passed: {validation_results['passed_validations']}
  Failed: {validation_results['failed_validations']}
  Success Rate: {validation_results['success_rate']:.1f}%

📋 DETAILED RESULTS
"""

        for result in validation_results['validation_results']:
            status = "✅ PASSED" if result['overall_passed'] else "❌ FAILED"
            risk_icon = {
                'LOW': '🟢',
                'MEDIUM': '🟡',
                'HIGH': '🔴',
                'CRITICAL': '🚨'
            }.get(result['risk_level'], '❓')

            report += f"""
{status} {result['file_path']}
  Risk Level: {risk_icon} {result['risk_level']}

  Validation Checks:
"""

            for check in result['validation_results']:
                check_status = "✅" if check['passed'] else "❌"
                report += f"    {check_status} {check['check_type']}: {check['message']}\n"

            if result['recommendations']:
                report += "  Recommendations:\n"
                for rec in result['recommendations']:
                    report += f"    • {rec}\n"

        return report

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Fix Validator - Validation & Rollback System')
    parser.add_argument('file_path', help='Path to file to validate')
    parser.add_argument('--backup', action='store_true', help='Create backup before validation')
    parser.add_argument('--rollback', help='Rollback file to specified backup')
    parser.add_argument('--batch-results', help='Validate batch processing results JSON file')

    args = parser.parse_args()

    validator = FixValidator()

    if args.rollback:
        # Perform rollback
        success = validator.rollback_file(args.file_path, args.rollback)
        if success:
            print(f"✅ Successfully rolled back {args.file_path}")
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.batch_results:
        # Validate batch results
        with open(args.batch_results, 'r') as f:
            batch_data = json.load(f)

        validation_results = validator.validate_batch_results(batch_data)
        report = validator.generate_validation_report(validation_results)


        # Save detailed report
        report_path = validator.results_dir / f"batch_validation_{int(time.time())}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 Detailed report saved: {report_path}")

    else:
        # Validate single file
        if args.backup:
            backup_path = validator.create_rollback_point(args.file_path)

        report = validator.validate_file(args.file_path)

        # Print results
        status = "✅ PASSED" if report.overall_passed else "❌ FAILED"
        risk_icon = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🔴',
            'CRITICAL': '🚨'
        }.get(report.risk_level, '❓')

        print(f"\n{status} Validation Results for {args.file_path}")
        print(f"Risk Level: {risk_icon} {report.risk_level}")

        for result in report.validation_results:
            check_status = "✅" if result.passed else "❌"
            print(f"  {check_status} {result.check_type}: {result.message}")

        if report.recommendations:
            for rec in report.recommendations:

        # Exit with appropriate code
        sys.exit(0 if report.overall_passed else 1)

if __name__ == '__main__':
    main()
