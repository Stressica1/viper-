#!/usr/bin/env python3
"""
# Rocket VIPER Repository Structure Enforcer
Maintains clean repository organization and prevents clutter
"""

from pathlib import Path
import shutil
from datetime import datetime

class RepositoryOrganizer:
    """Enforces clean repository structure and organization rules"""
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.violations = []
        
        # Define allowed files in root directory
        self.allowed_root_files = {
            'README.md',
            'LICENSE',
            '.gitignore',
            'requirements.txt',
            'docker-compose.yml',
            'Dockerfile',
            'pyproject.toml',
            'setup.py',
            'setup.cfg',
            '.env.example',
            '.env.template'
        }
        
        # Define directory structure rules
        self.required_structure = {
            'src/viper': 'Main source code package',
            'src/viper/core': 'Core trading logic',
            'src/viper/strategies': 'Trading strategies',
            'src/viper/optimization': 'Optimization modules',
            'src/viper/execution': 'Trade execution',
            'src/viper/risk': 'Risk management',
            'src/viper/utils': 'Utility modules',
            'scripts': 'Executable scripts',
            'tests': 'Test files',
            'docs': 'Documentation',
            'config': 'Configuration files',
            'services': 'Microservices',
            'infrastructure': 'Infrastructure code',
            'reports': 'Generated reports',
            'deployments': 'Deployment configurations',
            'tools': 'Development tools'
        }
        
        # File categorization rules
        self.file_categories = {
            'python_core': {
                'patterns': ['*trader*.py', '*engine*.py', '*manager*.py', '*system*.py'],
                'destination': 'src/viper/core'
            },
            'python_strategies': {
                'patterns': ['*strategy*.py', '*scalping*.py', '*optimization*.py'],
                'destination': 'src/viper/strategies'
            },
            'python_scripts': {
                'patterns': ['run_*.py', 'start_*.py', 'launch_*.py'],
                'destination': 'scripts'
            },
            'python_utils': {
                'patterns': ['*util*.py', '*helper*.py', '*validator*.py'],
                'destination': 'src/viper/utils'
            },
            'python_diagnostics': {
                'patterns': ['*diagnostic*.py', '*debug*.py', '*fix*.py', '*scan*.py'],
                'destination': 'tools/diagnostics'
            },
            'documentation': {
                'patterns': ['*.md'],
                'destination': 'docs'
            },
            'config_files': {
                'patterns': ['*.yml', '*.yaml', '*.json', '*.conf', '*.cfg'],
                'destination': 'config'
            },
            'backup_files': {
                'patterns': ['*backup*', '*_backup*', 'backup_*'],
                'destination': 'deployments/backups'
            }
        }

    def scan_violations(self) -> List[Dict]:
        """Scan repository for organization violations"""
        print("# Search Scanning repository for organization violations...")
        
        violations = []
        
        # Check root directory clutter
        root_files = [f for f in self.repo_root.iterdir() if f.is_file()]
        for file_path in root_files:
            if file_path.name not in self.allowed_root_files and not file_path.name.startswith('.'):
                violations.append({
                    'type': 'root_clutter',
                    'file': str(file_path.relative_to(self.repo_root)),
                    'message': f"File '{file_path.name}' should not be in root directory",
                    'suggested_location': self._suggest_location(file_path)
                })
        
        # Check for missing required directories
        for dir_path, description in self.required_structure.items():
            full_path = self.repo_root / dir_path
            if not full_path.exists():
                violations.append({
                    'type': 'missing_directory',
                    'directory': dir_path,
                    'message': f"Required directory '{dir_path}' is missing",
                    'description': description
                })
        
        return violations

    def _suggest_location(self, file_path: Path) -> str:
        """Suggest appropriate location for a file"""
        file_name = file_path.name.lower()
        
        # Check each category
        for category, rules in self.file_categories.items():
            for pattern in rules['patterns']:
                if self._matches_pattern(file_name, pattern.lower()):
                    return rules['destination']
        
        # Default suggestions based on extension
        if file_path.suffix == '.py':
            return 'src/viper/core'
        elif file_path.suffix == '.md':
            return 'docs'
        elif file_path.suffix in ['.yml', '.yaml', '.json', '.conf']:
            return 'config'
        else:
            return 'tools/utilities'

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern (simple wildcard matching)"""
        if '*' not in pattern:
            return filename == pattern
        
        parts = pattern.split('*')
        if len(parts) == 2:  # Simple *text* pattern
            prefix, suffix = parts
            return filename.startswith(prefix) and filename.endswith(suffix)
        
        return False

    def organize_files(self, dry_run: bool = True) -> Dict:
        """Organize files according to structure rules"""
        print(f"# Target {'Simulating' if dry_run else 'Executing'} file organization...")
        
        violations = self.scan_violations()
        moves = []
        
        for violation in violations:
            if violation['type'] == 'root_clutter':
                source = self.repo_root / violation['file']
                destination_dir = self.repo_root / violation['suggested_location']
                destination = destination_dir / source.name
                
                moves.append({
                    'source': str(source),
                    'destination': str(destination),
                    'reason': violation['message']
                })
                
                if not dry_run:
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(destination))
                    print(f"# Check Moved {source.name} → {violation['suggested_location']}")
        
        # Create missing directories
        if not dry_run:
            for dir_path in self.required_structure.keys():
                full_path = self.repo_root / dir_path
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
        
        return {
            'violations_found': len(violations),
            'files_moved': len(moves),
            'moves': moves,
            'dry_run': dry_run
        }

    def generate_report(self) -> str:
        """Generate organization report"""
        violations = self.scan_violations()
        
        report = f"""
# # Rocket Repository Organization Report
Generated: {datetime.now().isoformat()}

## Summary
- **Total Violations**: {len(violations)}
- **Root Files**: {len([v for v in violations if v['type'] == 'root_clutter'])}
- **Missing Directories**: {len([v for v in violations if v['type'] == 'missing_directory'])}

## Violations Found

"""
        
        for violation in violations:
            report += f"### {violation['type'].replace('_', ' ').title()}\n"
            report += f"- **File**: {violation.get('file', violation.get('directory', 'N/A'))}\n"
            report += f"- **Issue**: {violation['message']}\n"
            if 'suggested_location' in violation:
                report += f"- **Suggested**: {violation['suggested_location']}\n"
            report += "\n"
        
        return report

    def create_structure_validator(self):
        """Create a structure validation script"""
        validator_script = '''#!/usr/bin/env python3
"""
Repository Structure Validator
Runs pre-commit checks to ensure clean repository organization
"""

from pathlib import Path
from repo_organizer import RepositoryOrganizer

def main():
    repo_root = Path.cwd()
    organizer = RepositoryOrganizer(repo_root)
    
    violations = organizer.scan_violations()
    
    if violations:
        for violation in violations[:5]:  # Show first 5
        
        if len(violations) > 5:
            print(f"  ... and {len(violations) - 5} more violations")
        
        print("\\nRun 'python tools/repo_organizer.py --fix' to resolve")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
'''
        
        validator_path = self.repo_root / 'tools' / 'validate_structure.py'
        validator_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(validator_path, 'w') as f:
            f.write(validator_script)
        
        validator_path.chmod(0o755)
        print(f"# Check Created structure validator: {validator_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VIPER Repository Organizer')
    parser.add_argument('--fix', action='store_true', help='Actually move files (default: dry run)')
    parser.add_argument('--report', action='store_true', help='Generate organization report')
    parser.add_argument('--create-validator', action='store_true', help='Create structure validator')
    
    args = parser.parse_args()
    
    repo_root = Path.cwd()
    organizer = RepositoryOrganizer(repo_root)
    
    if args.report:
        report = organizer.generate_report()
        report_path = repo_root / 'reports' / 'organization_report.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
    
    if args.create_validator:
        organizer.create_structure_validator()
    
    # Always show current state
    result = organizer.organize_files(dry_run=not args.fix)
    
    print(f"  - Violations found: {result['violations_found']}")
    print(f"  - Files to move: {result['files_moved']}")
    print(f"  - Mode: {'EXECUTED' if not result['dry_run'] else 'SIMULATION'}")
    
    if result['files_moved'] > 0 and result['dry_run']:
        print("\n# Idea Run with --fix to actually move files")


if __name__ == "__main__":
    main()