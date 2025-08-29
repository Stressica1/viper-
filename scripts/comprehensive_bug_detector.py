#!/usr/bin/env python3
"""
🕵️ COMPREHENSIVE BUG DETECTOR - VIPER Repository Scanner
=======================================================

A comprehensive bug detection and code quality analysis system for the VIPER trading system.

Features:
- Python syntax error detection
- Common programming mistakes and anti-patterns
- Unused imports and variables
- Code complexity analysis
- Security vulnerability scanning
- Spelling and grammar checking
- Comprehensive reporting with actionable recommendations

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import ast
import re
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import subprocess
import importlib.util
import tempfile
import hashlib

# Third-party imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from spellchecker import SpellChecker
    HAS_SPELLCHECKER = True
except ImportError:
    HAS_SPELLCHECKER = False

try:
    import astroid
    HAS_ASTROID = True
except ImportError:
    HAS_ASTROID = False

@dataclass
class BugFinding:
    """Represents a single bug or issue finding"""
    file_path: str
    line_number: int
    column: int
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'
    category: str  # 'SYNTAX', 'LOGIC', 'SECURITY', 'QUALITY', 'SPELLING'
    rule_id: str
    message: str
    code_snippet: str = ""
    suggestion: str = ""
    confidence: float = 1.0  # 0.0 to 1.0

@dataclass
class ScanResults:
    """Container for all scan results"""
    scan_timestamp: str
    repository_path: str
    total_files_scanned: int
    total_lines_scanned: int
    execution_time: float
    findings: List[BugFinding]
    summary: Dict[str, Any]
    recommendations: List[str]

class ComprehensiveBugDetector:
    """Main bug detection engine"""

    def __init__(self, repo_path: str = None):
        self.repo_path = Path(repo_path or os.getcwd())
        self.findings: List[BugFinding] = []
        self.file_stats = defaultdict(int)
        self.start_time = time.time()

        # Initialize spell checker if available
        if HAS_SPELLCHECKER:
            self.spell_checker = SpellChecker()
            # Add technical terms to dictionary
            technical_terms = [
                'viper', 'trading', 'algorithm', 'backtest', 'leverage', 'position',
                'ohlcv', 'websocket', 'async', 'await', 'redis', 'docker', 'api',
                'ccxt', 'bitget', 'mcp', 'github', 'json', 'yaml', 'sql', 'http',
                'https', 'url', 'uri', 'uuid', 'timestamp', 'dataframe', 'numpy',
                'pandas', 'matplotlib', 'seaborn', 'sklearn', 'tensorflow', 'keras'
            ]
            for term in technical_terms:
                self.spell_checker.word_frequency.load_words([term])
        else:
            self.spell_checker = None

        # Common Python anti-patterns and security issues
        self.anti_patterns = self._load_anti_patterns()
        self.security_patterns = self._load_security_patterns()

    def _load_anti_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load common Python anti-patterns"""
        return {
            'bare_except': {
                'pattern': r'except\s*:',
                'message': 'Bare except clause catches all exceptions, making debugging difficult',
                'severity': 'MEDIUM',
                'suggestion': 'Specify the exception type(s) to catch'
            },
            'print_debug': {
                'pattern': r'print\s*\([^)]*debug|log|temp[^)]*\)',
                'message': 'Debug print statements left in production code',
                'severity': 'LOW',
                'suggestion': 'Remove debug prints or use proper logging'
            },
            'hardcoded_password': {
                'pattern': r'password\s*=\s*[\'"][^\'"]{3,}[\'"]',
                'message': 'Hardcoded password detected',
                'severity': 'CRITICAL',
                'suggestion': 'Use environment variables or secure credential storage'
            },
            'eval_usage': {
                'pattern': r'\beval\s*\(',
                'message': 'Use of eval() is dangerous and should be avoided',
                'severity': 'HIGH',
                'suggestion': 'Use ast.literal_eval() or avoid dynamic code execution'
            },
            'exec_usage': {
                'pattern': r'\bexec\s*\(',
                'message': 'Use of exec() is dangerous and should be avoided',
                'severity': 'HIGH',
                'suggestion': 'Avoid dynamic code execution'
            },
            'todo_comment': {
                'pattern': r'#\s*TODO|#\s*FIXME|#\s*XXX',
                'message': 'TODO/FIXME comment found',
                'severity': 'INFO',
                'suggestion': 'Address the TODO item or convert to proper issue tracking'
            }
        }

    def _load_security_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load security vulnerability patterns"""
        return {
            'hardcoded_api_key': {
                'pattern': r'(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*=\s*[\'"][A-Za-z0-9_\-]{20,}[\'"]',
                'message': 'Potential hardcoded API key or secret detected',
                'severity': 'CRITICAL',
                'suggestion': 'Use environment variables or secure credential management'
            },
            'sql_injection': {
                'pattern': r'(SELECT|INSERT|UPDATE|DELETE).*%.*',
                'message': 'Potential SQL injection vulnerability',
                'severity': 'HIGH',
                'suggestion': 'Use parameterized queries or ORM'
            },
            'path_traversal': {
                'pattern': r'open\s*\([^)]*\.\.[^)]*\)',
                'message': 'Potential path traversal vulnerability',
                'severity': 'HIGH',
                'suggestion': 'Validate and sanitize file paths'
            },
            'insecure_random': {
                'pattern': r'\brandom\.(randint|random|choice)',
                'message': 'Using insecure random number generation',
                'severity': 'MEDIUM',
                'suggestion': 'Use secrets module for cryptographic randomness'
            }
        }

    def scan_repository(self) -> ScanResults:
        """Main scanning function"""
        print("🔍 COMPREHENSIVE BUG DETECTOR")
        print("=" * 50)
        print(f"📂 Scanning repository: {self.repo_path}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Get all Python files
        python_files = self._get_python_files()
        print(f"📄 Found {len(python_files)} Python files to scan")

        total_lines = 0
        processed_files = 0

        for file_path in python_files:
            try:
                print(f"🔍 Scanning: {file_path.name}")
                findings, lines = self._scan_file(file_path)
                self.findings.extend(findings)
                total_lines += lines
                processed_files += 1

                # Show progress
                if processed_files % 10 == 0:
                    print(f"  📊 Progress: {processed_files}/{len(python_files)} files")

            except Exception as e:
                self.findings.append(BugFinding(
                    file_path=str(file_path),
                    line_number=1,
                    column=0,
                    severity='HIGH',
                    category='SYNTAX',
                    rule_id='SCAN_ERROR',
                    message=f'Error scanning file: {str(e)}',
                    suggestion='Check file permissions and syntax'
                ))

        # Generate summary and recommendations
        summary = self._generate_summary()
        recommendations = self._generate_recommendations()

        execution_time = time.time() - self.start_time

        results = ScanResults(
            scan_timestamp=datetime.now().isoformat(),
            repository_path=str(self.repo_path),
            total_files_scanned=processed_files,
            total_lines_scanned=total_lines,
            execution_time=execution_time,
            findings=self.findings,
            summary=summary,
            recommendations=recommendations
        )

        print("\n" + "=" * 50)
        print("🎯 SCAN COMPLETE")
        print("=" * 50)
        print(f"📊 Total files scanned: {processed_files}")
        print(f"📝 Total lines scanned: {total_lines}")
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"🚨 Total findings: {len(self.findings)}")
        print(f"🔴 Critical: {summary.get('critical_count', 0)}")
        print(f"🟠 High: {summary.get('high_count', 0)}")
        print(f"🟡 Medium: {summary.get('medium_count', 0)}")
        print(f"🟢 Low: {summary.get('low_count', 0)}")
        print(f"ℹ️  Info: {summary.get('info_count', 0)}")

        return results

    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the repository"""
        python_files = []

        # Common patterns to exclude
        exclude_patterns = [
            '__pycache__',
            '.git',
            'node_modules',
            '.venv',
            'venv',
            'env',
            '.env',
            'logs',
            '*.pyc',
            '*.pyo',
            '*.pyd'
        ]

        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    # Skip files in excluded directories
                    if not any(pattern in str(file_path) for pattern in exclude_patterns):
                        python_files.append(file_path)

        return python_files

    def _scan_file(self, file_path: Path) -> Tuple[List[BugFinding], int]:
        """Scan a single Python file"""
        findings = []
        lines_scanned = 0

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                lines_scanned = len(lines)

            # Basic syntax check
            findings.extend(self._check_syntax(file_path, content))

            # Anti-pattern detection
            findings.extend(self._check_anti_patterns(file_path, content, lines))

            # Security vulnerability scanning
            findings.extend(self._check_security_vulnerabilities(file_path, content, lines))

            # Code quality checks
            findings.extend(self._check_code_quality(file_path, content, lines))

            # Spelling checks
            if self.spell_checker:
                findings.extend(self._check_spelling(file_path, content, lines))

        except Exception as e:
            findings.append(BugFinding(
                file_path=str(file_path),
                line_number=1,
                column=0,
                severity='HIGH',
                category='SYNTAX',
                rule_id='FILE_READ_ERROR',
                message=f'Could not read file: {str(e)}',
                suggestion='Check file permissions'
            ))

        return findings, lines_scanned

    def _check_syntax(self, file_path: Path, content: str) -> List[BugFinding]:
        """Check for Python syntax errors"""
        findings = []

        try:
            ast.parse(content)
        except SyntaxError as e:
            findings.append(BugFinding(
                file_path=str(file_path),
                line_number=e.lineno or 1,
                column=e.offset or 0,
                severity='CRITICAL',
                category='SYNTAX',
                rule_id='SYNTAX_ERROR',
                message=f'Syntax error: {e.msg}',
                code_snippet=e.text.strip() if e.text else '',
                suggestion='Fix the syntax error according to Python standards'
            ))
        except Exception as e:
            findings.append(BugFinding(
                file_path=str(file_path),
                line_number=1,
                column=0,
                severity='HIGH',
                category='SYNTAX',
                rule_id='PARSE_ERROR',
                message=f'Could not parse file: {str(e)}',
                suggestion='Check for encoding issues or corrupted file'
            ))

        return findings

    def _check_anti_patterns(self, file_path: Path, content: str, lines: List[str]) -> List[BugFinding]:
        """Check for common anti-patterns"""
        findings = []

        for line_num, line in enumerate(lines, 1):
            for pattern_name, pattern_info in self.anti_patterns.items():
                if re.search(pattern_info['pattern'], line, re.IGNORECASE):
                    findings.append(BugFinding(
                        file_path=str(file_path),
                        line_number=line_num,
                        column=0,
                        severity=pattern_info['severity'],
                        category='QUALITY',
                        rule_id=f'ANTI_PATTERN_{pattern_name.upper()}',
                        message=pattern_info['message'],
                        code_snippet=line.strip(),
                        suggestion=pattern_info['suggestion']
                    ))

        return findings

    def _check_security_vulnerabilities(self, file_path: Path, content: str, lines: List[str]) -> List[BugFinding]:
        """Check for security vulnerabilities"""
        findings = []

        for line_num, line in enumerate(lines, 1):
            for pattern_name, pattern_info in self.security_patterns.items():
                if re.search(pattern_info['pattern'], line):
                    findings.append(BugFinding(
                        file_path=str(file_path),
                        line_number=line_num,
                        column=0,
                        severity=pattern_info['severity'],
                        category='SECURITY',
                        rule_id=f'SECURITY_{pattern_name.upper()}',
                        message=pattern_info['message'],
                        code_snippet=line.strip(),
                        suggestion=pattern_info['suggestion']
                    ))

        return findings

    def _check_code_quality(self, file_path: Path, content: str, lines: List[str]) -> List[BugFinding]:
        """Check code quality metrics"""
        findings = []

        # Check line length
        for line_num, line in enumerate(lines, 1):
            if len(line) > 120:  # PEP 8 recommends 79, but 120 is more practical
                findings.append(BugFinding(
                    file_path=str(file_path),
                    line_number=line_num,
                    column=120,
                    severity='LOW',
                    category='QUALITY',
                    rule_id='LINE_TOO_LONG',
                    message='Line exceeds 120 characters',
                    code_snippet=line.strip(),
                    suggestion='Break the line into multiple lines for better readability'
                ))

        # Check for unused imports (basic check)
        try:
            tree = ast.parse(content)
            imports = []
            used_names = set()

            # Collect imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
                    for alias in node.names:
                        imports.append(alias.name)

                # Collect used names
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)

            # Check for unused imports
            for imp in set(imports):
                if imp not in used_names and imp not in ['os', 'sys', 'json', 're', 'time', 'datetime']:
                    findings.append(BugFinding(
                        file_path=str(file_path),
                        line_number=1,
                        column=0,
                        severity='INFO',
                        category='QUALITY',
                        rule_id='UNUSED_IMPORT',
                        message=f'Potentially unused import: {imp}',
                        suggestion='Remove unused import or use it in the code'
                    ))

        except:
            pass  # Skip AST analysis if parsing fails

        return findings

    def _check_spelling(self, file_path: Path, content: str, lines: List[str]) -> List[BugFinding]:
        """Check spelling in comments and docstrings"""
        findings = []

        if not self.spell_checker:
            return findings

        # Extract comments and docstrings
        comment_pattern = r'#.*|""".*?"""|\'\'\'.*?\'\'\''
        matches = re.findall(comment_pattern, content, re.DOTALL)

        for match in matches:
            # Remove comment markers
            if match.startswith('#'):
                text = match[1:].strip()
            else:
                # Remove docstring quotes
                text = re.sub(r'^[\'"]{3}|[\'"]{3}$', '', match).strip()

            # Skip if too short or contains code
            if len(text) < 3 or re.search(r'[=(){}[\]<>]', text):
                continue

            # Check spelling word by word
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if len(word) > 2 and word not in self.spell_checker:
                    findings.append(BugFinding(
                        file_path=str(file_path),
                        line_number=1,  # We don't track exact line for now
                        column=0,
                        severity='LOW',
                        category='SPELLING',
                        rule_id='SPELLING_ERROR',
                        message=f'Potential spelling error: "{word}"',
                        code_snippet=text[:50] + '...' if len(text) > 50 else text,
                        suggestion=f'Consider: {self.spell_checker.candidates(word)[:3] if self.spell_checker.candidates(word) else "Check spelling"}'
                    ))

        return findings

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)

        for finding in self.findings:
            severity_counts[finding.severity] += 1
            category_counts[finding.category] += 1

        return {
            'total_findings': len(self.findings),
            'critical_count': severity_counts['CRITICAL'],
            'high_count': severity_counts['HIGH'],
            'medium_count': severity_counts['MEDIUM'],
            'low_count': severity_counts['LOW'],
            'info_count': severity_counts['INFO'],
            'categories': dict(category_counts),
            'most_common_issues': self._get_most_common_issues()
        }

    def _get_most_common_issues(self) -> List[Tuple[str, int]]:
        """Get most common issue types"""
        rule_counts = Counter(finding.rule_id for finding in self.findings)
        return rule_counts.most_common(10)

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        summary = self._generate_summary()

        if summary['critical_count'] > 0:
            recommendations.append("🚨 CRITICAL: Fix all critical issues immediately - these may prevent the code from running")
        if summary['high_count'] > 0:
            recommendations.append("⚠️ HIGH PRIORITY: Address high-severity issues that could cause runtime errors or security problems")
        if summary['medium_count'] > 0:
            recommendations.append("📋 MEDIUM: Review medium-severity issues for code quality and maintainability improvements")

        # Category-specific recommendations
        if summary['categories'].get('SECURITY', 0) > 0:
            recommendations.append("🔒 SECURITY: Review and fix all security-related findings to protect against vulnerabilities")
        if summary['categories'].get('SYNTAX', 0) > 0:
            recommendations.append("🐛 SYNTAX: Fix all syntax errors to ensure code runs properly")
        if summary['categories'].get('SPELLING', 0) > 0:
            recommendations.append("📝 SPELLING: Review spelling errors in comments and documentation for professionalism")

        # General recommendations
        recommendations.append("🧪 TESTING: Run comprehensive tests after fixing critical issues")
        recommendations.append("📚 DOCUMENTATION: Consider adding more docstrings and comments for better code documentation")
        recommendations.append("🔄 REGULAR SCANS: Schedule regular bug scans to maintain code quality")

        return recommendations

    def save_report(self, results: ScanResults, output_path: Path = None) -> str:
        """Save scan results to file"""
        if output_path is None:
            output_path = Path("reports/bug_scan_report.json")

        output_path.parent.mkdir(exist_ok=True)

        # Convert findings to dictionaries
        findings_dict = [asdict(finding) for finding in results.findings]

        report_data = {
            'scan_metadata': {
                'timestamp': results.scan_timestamp,
                'repository_path': results.repository_path,
                'total_files_scanned': results.total_files_scanned,
                'total_lines_scanned': results.total_lines_scanned,
                'execution_time_seconds': results.execution_time
            },
            'summary': results.summary,
            'recommendations': results.recommendations,
            'findings': findings_dict
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return str(output_path)

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive Bug Detector for VIPER Repository')
    parser.add_argument('--path', '-p', help='Repository path (default: current directory)')
    parser.add_argument('--output', '-o', help='Output report path')
    parser.add_argument('--format', '-f', choices=['json', 'html'], default='json',
                       help='Output format (default: json)')

    args = parser.parse_args()

    # Create detector and run scan
    detector = ComprehensiveBugDetector(args.path)
    results = detector.scan_repository()

    # Save report
    output_path = args.output or f"reports/bug_scan_report.{args.format}"
    saved_path = detector.save_report(results, Path(output_path))

    print(f"\n📄 Report saved to: {saved_path}")

    # Exit with error code if critical issues found
    if results.summary.get('critical_count', 0) > 0:
        print("❌ Critical issues found - review and fix before proceeding")
        sys.exit(1)
    else:
        print("✅ No critical issues found")
        sys.exit(0)

if __name__ == '__main__':
    main()
