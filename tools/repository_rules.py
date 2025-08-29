#!/usr/bin/env python3
"""
# Rocket VIPER Repository Structure Rules & Enforcement
Complete repository organization rules and enforcement system
"""

import sys
from pathlib import Path
from datetime import datetime

class RepositoryRules:
    """Defines and enforces repository organization rules"""
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        
        # Essential files that MUST be in root
        self.required_root_files = {
            'README.md': 'Main project documentation',
            'requirements.txt': 'Python dependencies',
            '.gitignore': 'Git ignore rules'
        }
        
        # Optional files allowed in root
        self.allowed_root_files = {
            'LICENSE', 'LICENCE', 'LICENSE.txt', 'LICENSE.md',
            'docker-compose.yml', 'docker-compose.yaml',
            'Dockerfile',
            'pyproject.toml', 'setup.py', 'setup.cfg',
            '.env.example', '.env.template',
            'MANIFEST.in',
            'tox.ini',
            'pytest.ini'
        }
        
        # Files that should NEVER be in root
        self.forbidden_root_patterns = {
            '*.py': 'Python files should be in src/ or scripts/',
            '*.json': 'JSON files should be in config/',
            '*.md': 'Documentation should be in docs/ (except README.md)',
            'backup_*': 'Backup files should be in deployments/backups/',
            '*_backup*': 'Backup files should be in deployments/backups/',
            'temp_*': 'Temporary files should not be committed',
            'tmp_*': 'Temporary files should not be committed',
            '*.log': 'Log files should not be committed',
            '*.bak': 'Backup files should not be committed'
        }
        
        # Required directory structure
        self.required_directories = {
            'src/viper': 'Main source code package',
            'src/viper/core': 'Core trading logic',
            'src/viper/strategies': 'Trading strategies',
            'src/viper/execution': 'Trade execution engines',
            'src/viper/risk': 'Risk management',
            'src/viper/utils': 'Utility modules',
            'scripts': 'Executable scripts (run_*, start_*, launch_*)',
            'tests': 'Test files',
            'docs': 'Documentation',
            'config': 'Configuration files',
            'tools': 'Development and diagnostic tools',
            'reports': 'Generated reports'
        }
        
        # File organization rules
        self.file_rules = {
            # Python files
            'run_*.py': 'scripts/',
            'start_*.py': 'scripts/',
            'launch_*.py': 'scripts/',
            'test_*.py': 'tests/',
            '*_test.py': 'tests/',
            '*diagnostic*.py': 'tools/diagnostics/',
            '*debug*.py': 'tools/diagnostics/',
            'fix_*.py': 'tools/diagnostics/',
            '*validator*.py': 'tools/utilities/',
            '*strategy*.py': 'src/viper/strategies/',
            '*optimization*.py': 'src/viper/strategies/',
            '*optimizer*.py': 'src/viper/strategies/',
            '*backtester*.py': 'src/viper/strategies/',
            '*execution*.py': 'src/viper/execution/',
            '*trade*.py': 'src/viper/execution/',
            '*trading*.py': 'src/viper/execution/',
            '*risk*.py': 'src/viper/risk/',
            
            # Configuration files
            '*.json': 'config/',
            '*.yml': 'config/',
            '*.yaml': 'config/',
            '*.conf': 'config/',
            '*.cfg': 'config/',
            '*.ini': 'config/',
            
            # Documentation (except specific files)
            'CHANGELOG*.md': 'docs/',
            'INSTALL*.md': 'docs/',
            'CONTRIBUTING*.md': 'docs/',
            'HISTORY*.md': 'docs/',
            '*.rst': 'docs/',
            
            # Reports
            '*.html': 'reports/',
            '*.pdf': 'reports/',
            '*_report.*': 'reports/',
            
            # Backup files
            'backup_*': 'deployments/backups/',
            '*_backup*': 'deployments/backups/',
            '*.backup': 'deployments/backups/',
            '*.bak': 'deployments/backups/'
        }

    def validate_structure(self) -> Dict[str, List[str]]:
        """Validate complete repository structure"""
        violations = {
            'missing_required_files': [],
            'missing_required_directories': [],
            'forbidden_root_files': [],
            'misplaced_files': [],
            'naming_violations': []
        }
        
        # Check required files
        for file_name, description in self.required_root_files.items():
            if not (self.repo_root / file_name).exists():
                violations['missing_required_files'].append(
                    f"{file_name} - {description}"
                )
        
        # Check required directories
        for dir_path, description in self.required_directories.items():
            if not (self.repo_root / dir_path).exists():
                violations['missing_required_directories'].append(
                    f"{dir_path} - {description}"
                )
        
        # Check root directory for violations
        for item in self.repo_root.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                if (item.name not in self.required_root_files and:
                    item.name not in self.allowed_root_files):
                    
                    # Check against forbidden patterns
                    for pattern in self.forbidden_root_patterns:
                        if self._matches_pattern(item.name, pattern):
                            violations['forbidden_root_files'].append(
                                f"{item.name} - {self.forbidden_root_patterns[pattern]}"
                            )
                            break
        
        # Check file placement across repository
        for file_path in self._get_all_files():
            relative_path = file_path.relative_to(self.repo_root)
            suggested_location = self._get_suggested_location(file_path)
            
            if suggested_location and not str(relative_path).startswith(suggested_location):
                violations['misplaced_files'].append(
                    f"{relative_path} should be in {suggested_location}"
                )
        
        return violations

    def _get_all_files(self) -> List[Path]:
        """Get all files in repository (excluding .git)"""
        files = []
        for item in self.repo_root.rglob('*'):
            if (item.is_file() and:
                '.git' not in item.parts and 
                '__pycache__' not in item.parts):
                files.append(item)
        return files

    def _get_suggested_location(self, file_path: Path) -> str:
        """Get suggested location for a file based on rules"""
        file_name = file_path.name.lower()
        relative_path = str(file_path.relative_to(self.repo_root))
        
        # Don't suggest moves for files that are already in acceptable locations
        if (file_path.name in self.required_root_files or:
            file_path.name in self.allowed_root_files):
            return ""
        
        # Don't suggest moves for service-specific files
        if relative_path.startswith('services/') and file_path.name == 'requirements.txt':
            return ""
        
        # Don't suggest moves for infrastructure config files
        if relative_path.startswith('infrastructure/') and file_path.suffix in ['.yml', '.yaml']:
            return ""
        
        # Don't suggest moves for GitHub workflows
        if relative_path.startswith('.github/workflows/'):
            return ""
        
        # Don't suggest moves for files already in appropriate directories
        current_dir = str(file_path.parent.relative_to(self.repo_root))
        if current_dir.startswith('tests/') and ('test_' in file_name or '_test.' in file_name):
            return ""
        if current_dir.startswith('config/') and file_path.suffix in ['.yml', '.yaml', '.json', '.conf']:
            return ""
        if current_dir.startswith('docs/') and file_path.suffix in ['.md', '.rst', '.txt']:
            return ""
        
        for pattern, destination in self.file_rules.items():
            if self._matches_pattern(file_name, pattern.lower()):
                return destination
        
        return ""

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

    def generate_violations_report(self) -> str:
        """Generate detailed violations report"""
        violations = self.validate_structure()
        
        report = f"""
# # Rocket VIPER Repository Structure Violations Report
Generated: {datetime.now().isoformat()}

## Summary
- **Missing Required Files**: {len(violations['missing_required_files'])}
- **Missing Required Directories**: {len(violations['missing_required_directories'])}
- **Forbidden Root Files**: {len(violations['forbidden_root_files'])}
- **Misplaced Files**: {len(violations['misplaced_files'])}
- **Total Issues**: {sum(len(v) for v in violations.values())}

"""
        
        for category, issues in violations.items():
            if issues:
                report += f"## {category.replace('_', ' ').title()}\n\n"
                for issue in issues:
                    report += f"- {issue}\n"
                report += "\n"
        
        if sum(len(v) for v in violations.values()) == 0:
            report += "## # Check Repository Structure is Clean!\n\nNo violations found.\n"
        
        return report

    def create_enforcement_tools(self):
        """Create enforcement tools and configurations"""
        
        # Create pre-commit hook configuration
        pre_commit_config = """
#!/usr/bin/env python3
import sys
from pathlib import Path

# Import the repository rules
sys.path.append(str(Path(__file__).parent))
from repository_rules import RepositoryRules

def main():
    repo_root = Path.cwd()
    rules = RepositoryRules(repo_root)
    violations = rules.validate_structure()
    
    total_violations = sum(len(v) for v in violations.values())
    
    if total_violations > 0:
        print("# X Repository structure violations detected!")
        print("Run 'python tools/repository_rules.py --report' for details")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
"""
        
        pre_commit_path = self.repo_root / '.git' / 'hooks' / 'pre-commit'
        if pre_commit_path.parent.exists():
            with open(pre_commit_path, 'w') as f:
                f.write(pre_commit_config)
            pre_commit_path.chmod(0o755)
            print(f"# Check Created pre-commit hook: {pre_commit_path}")

    def setup_ci_validation(self):
        """Create GitHub Actions workflow for structure validation"""
        workflow_dir = self.repo_root / '.github' / 'workflows'
        workflow_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_content = """
name: Repository Structure Validation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  validate-structure:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Validate Repository Structure
      run: |
        python tools/repository_rules.py --validate
        if [ $? -ne 0 ]; then:
          echo "Repository structure validation failed!"
          exit 1
        fi
"""
        
        workflow_path = workflow_dir / 'validate_structure.yml'
        with open(workflow_path, 'w') as f:
            f.write(workflow_content.strip())
        


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='VIPER Repository Structure Rules & Enforcement'
    )
    parser.add_argument('--validate', action='store_true', 
                       help='Validate repository structure')
    parser.add_argument('--report', action='store_true', 
                       help='Generate violations report')
    parser.add_argument('--setup-enforcement', action='store_true', 
                       help='Setup enforcement tools (hooks, CI)')
    
    args = parser.parse_args()
    
    repo_root = Path.cwd()
    rules = RepositoryRules(repo_root)
    
    if args.report:
        report = rules.generate_violations_report()
        report_path = repo_root / 'reports' / 'structure_violations.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"# Chart Violations report saved to: {report_path}")
    
    if args.setup_enforcement:
        rules.create_enforcement_tools()
        rules.setup_ci_validation()
    
    if args.validate or not any([args.report, args.setup_enforcement]):
        violations = rules.validate_structure()
        total_violations = sum(len(v) for v in violations.values())
        
        if total_violations > 0:
            print(f"# X Repository structure has {total_violations} violations")
            print("Run with --report for detailed information")
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()