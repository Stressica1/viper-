#!/usr/bin/env python3
"""
# Rocket Clean Root Enforcer
Automatically enforces a clean root directory by moving misplaced files
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

class CleanRootEnforcer:
    """Automatically maintains a clean root directory"""
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        
        # Files allowed in root
        self.allowed_files = {
            'README.md',
            'LICENSE',
            'LICENSE.txt',
            'LICENSE.md',
            '.gitignore',
            'requirements.txt',
            'docker-compose.yml',
            'docker-compose.yaml',
            'Dockerfile',
            'pyproject.toml',
            'setup.py',
            'setup.cfg',
            '.env.example',
            '.env.template',
            'MANIFEST.in',
            'tox.ini',
            'pytest.ini'
        }
        
        # Auto-move rules
        self.auto_move_rules = {
            'run_*.py': 'scripts/',
            'start_*.py': 'scripts/',
            'launch_*.py': 'scripts/',
            'test_*.py': 'tests/',
            '*_test.py': 'tests/',
            '*.json': 'config/',
            '*diagnostic*.py': 'tools/diagnostics/',
            '*debug*.py': 'tools/diagnostics/',
            'fix_*.py': 'tools/diagnostics/',
            '*validator*.py': 'tools/utilities/',
            'backup_*': 'deployments/backups/',
            '*_backup*': 'deployments/backups/',
            '*.backup': 'deployments/backups/',
            '*.bak': 'deployments/backups/',
            'CHANGELOG*.md': 'docs/',
            'INSTALL*.md': 'docs/',
            'CONTRIBUTING*.md': 'docs/',
            '*.log': 'reports/',
            '*.html': 'reports/',
            '*.pdf': 'reports/',
            '*_report.*': 'reports/'
        }

    def scan_root_violations(self):
        """Scan root directory for violations"""
        violations = []
        
        for item in self.repo_root.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                if item.name not in self.allowed_files:
                    suggested_location = self._get_auto_move_location(item)
                    if suggested_location:
                        violations.append({
                            'file': item,
                            'suggested_location': suggested_location,
                            'auto_movable': True
                        })
                    else:
                        violations.append({
                            'file': item,
                            'suggested_location': 'tools/utilities/',
                            'auto_movable': False
                        })
        
        return violations

    def _get_auto_move_location(self, file_path: Path):
        """Get automatic move location for a file"""
        filename = file_path.name.lower()
        
        for pattern, destination in self.auto_move_rules.items():
            if self._matches_pattern(filename, pattern.lower()):
                return destination
        
        return None

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Simple wildcard pattern matching"""
        if '*' not in pattern:
            return filename == pattern
        
        if pattern.startswith('*') and pattern.endswith('*'):
            middle = pattern[1:-1]
            return middle in filename
        elif pattern.startswith('*'):
            suffix = pattern[1:]
            return filename.endswith(suffix)
        elif pattern.endswith('*'):
            prefix = pattern[:-1]
            return filename.startswith(prefix)
        
        return False

    def clean_root_directory(self, dry_run=True):
        """Clean root directory by moving misplaced files"""
        violations = self.scan_root_violations()
        
        if not violations:
            return
        
        print(f"# Search Found {len(violations)} files in root directory")
        
        moves_executed = 0
        
        for violation in violations:
            source = violation['file']
            dest_dir = self.repo_root / violation['suggested_location']
            destination = dest_dir / source.name
            
            if dry_run:
                print(f"Would move: {source.name} → {violation['suggested_location']}")
            else:
                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(destination))
                    print(f"# Check Moved: {source.name} → {violation['suggested_location']}")
                    moves_executed += 1
                except Exception as e:
        
        if dry_run:
            print(f"\n# Idea This was a dry run. Use --execute to actually move files.")
        else:
            print(f"\n# Check Moved {moves_executed} files to proper locations!")

    def create_root_monitor(self):
        """Create a monitoring script to watch root directory"""
        monitor_script = '''#!/usr/bin/env python3
"""
Root Directory Monitor - Watches for files placed in root
"""

import time
import sys
from pathlib import Path
from clean_root_enforcer import CleanRootEnforcer

def monitor_root():
    repo_root = Path.cwd()
    enforcer = CleanRootEnforcer(repo_root)
    
    
    try:
        while True:
            violations = enforcer.scan_root_violations()
            
            if violations:
                print(f"# Warning  Found {len(violations)} files in root directory:")
                for violation in violations[:3]:  # Show first 3
                
                # Auto-clean if requested
                if '--auto-clean' in sys.argv:
                    enforcer.clean_root_directory(dry_run=False)
            
            time.sleep(10)  # Check every 10 seconds
    
    except KeyboardInterrupt:

if __name__ == "__main__":
    monitor_root()
'''
        
        monitor_path = self.repo_root / 'tools' / 'root_monitor.py'
        with open(monitor_path, 'w') as f:
            f.write(monitor_script)
        
        monitor_path.chmod(0o755)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean Root Enforcer')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually move files (default: dry run)')
    parser.add_argument('--create-monitor', action='store_true', 
                       help='Create root directory monitor')
    
    args = parser.parse_args()
    
    repo_root = Path.cwd()
    enforcer = CleanRootEnforcer(repo_root)
    
    if args.create_monitor:
        enforcer.create_root_monitor()
    
    enforcer.clean_root_directory(dry_run=not args.execute)


if __name__ == "__main__":
    main()