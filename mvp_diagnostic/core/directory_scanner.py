#!/usr/bin/env python3
"""
🚀 MVP DIRECTORY SCANNER
Comprehensive file system analysis for the VIPER diagnostic system
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set
import logging

logger = logging.getLogger(__name__)

class DirectoryScanner:
    """Comprehensive directory scanner for MVP diagnostic system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exclude_patterns = set(config.get('exclude_patterns', []))
        self.max_depth = config.get('max_scan_depth', 10)

    def scan_directory(self, root_path: Path) -> Dict[str, Any]:
        """Scan entire directory structure"""
        print(f"🔍 Scanning directory: {root_path}")

        scan_results = {
            'scan_timestamp': datetime.now().isoformat(),
            'root_path': str(root_path),
            'files_analyzed': 0,
            'directories_scanned': 0,
            'file_types': {},
            'python_files': [],
            'config_files': [],
            'log_files': [],
            'large_files': [],
            'issues_found': []
        }

        for file_path in root_path.rglob('*'):
            if file_path.is_file() and not self._should_exclude(file_path):
                self._analyze_file(file_path, scan_results)

        # Sort results
        scan_results['python_files'].sort()
        scan_results['config_files'].sort()
        scan_results['log_files'].sort()

        return scan_results

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded from scanning"""
        file_str = str(file_path)

        for pattern in self.exclude_patterns:
            if pattern in file_str:
                return True

        return False

    def _analyze_file(self, file_path: Path, results: Dict[str, Any]):
        """Analyze individual file"""
        try:
            stat = file_path.stat()
            file_size = stat.st_size
            file_ext = file_path.suffix.lower()

            # Update file type counts
            if file_ext not in results['file_types']:
                results['file_types'][file_ext] = 0
            results['file_types'][file_ext] += 1

            # Categorize files
            if file_ext == '.py':
                results['python_files'].append(str(file_path))
            elif file_ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']:
                results['config_files'].append(str(file_path))
            elif file_ext == '.log':
                results['log_files'].append(str(file_path))

            # Check for large files
            if file_size > 10 * 1024 * 1024:  # 10MB
                results['large_files'].append({
                    'path': str(file_path),
                    'size_mb': file_size / (1024 * 1024)
                })

            results['files_analyzed'] += 1

        except Exception as e:
            results['issues_found'].append({
                'file': str(file_path),
                'error': str(e),
                'type': 'file_analysis_error'
            })

    def generate_scan_report(self, scan_results: Dict[str, Any]) -> str:
        """Generate human-readable scan report"""
        report = f"""# 📊 DIRECTORY SCAN REPORT
**Timestamp:** {scan_results['scan_timestamp']}
**Root Path:** {scan_results['root_path']}

## 📈 Summary
- **Files Analyzed:** {scan_results['files_analyzed']}
- **Python Files:** {len(scan_results['python_files'])}
- **Config Files:** {len(scan_results['config_files'])}
- **Log Files:** {len(scan_results['log_files'])}

## 📁 File Types
"""

        for ext, count in sorted(scan_results['file_types'].items()):
            report += f"- **{ext or 'no extension'}:** {count} files\n"

        if scan_results['large_files']:
            report += f"\n## 📏 Large Files (>10MB)\n"
            for large_file in scan_results['large_files'][:10]:  # Top 10
                report += f"- {large_file['path']}: {large_file['size_mb']:.1f}MB\n"

        if scan_results['issues_found']:
            report += f"\n## ⚠️ Issues Found\n"
            for issue in scan_results['issues_found'][:5]:  # Top 5
                report += f"- **{issue['file']}:** {issue['error']}\n"

        return report
