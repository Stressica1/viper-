#!/usr/bin/env python3
"""
# Rocket VIPER Pre-commit Hook - Repository Structure Enforcer
Prevents commits that violate repository organization rules
"""

import sys
import subprocess
from pathlib import Path

def check_root_directory():
    """Check for files that shouldn't be in root directory"""
    violations = []
    
    # Get staged files
    result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        return []
    
    staged_files = result.stdout.strip().split('\n')
    
    # Define allowed root files
    allowed_root_files = {
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
    
    for file_path in staged_files:
        path = Path(file_path)
        
        # Check if file is in root directory
        if len(path.parts) == 1 and path.is_file():
            if path.name not in allowed_root_files and not path.name.startswith('.'):
                violations.append(f"File '{file_path}' should not be in root directory")
    
    return violations

def check_naming_conventions():
    """Check for proper naming conventions"""
    violations = []
    
    # Get staged files
    result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        return []
    
    staged_files = result.stdout.strip().split('\n')
    
    for file_path in staged_files:
        path = Path(file_path)
        
        # Check Python files
        if path.suffix == '.py':
            # Check for test files in wrong location
            if ('test_' in path.name or '_test' in path.name) and 'tests/' not in file_path:
                violations.append(f"Test file '{file_path}' should be in tests/ directory")
            
            # Check for demo files (should not exist)
            if ('demo_' in path.name or '_demo' in path.name):
                violations.append(f"Demo file '{file_path}' should not be committed")
    
    return violations

def main():
    """Main pre-commit check"""
    print("# Search Running VIPER repository structure checks...")
    
    violations = []
    
    # Check root directory
    root_violations = check_root_directory()
    if root_violations:
        violations.extend(root_violations)
    
    # Check naming conventions
    naming_violations = check_naming_conventions()
    if naming_violations:
        violations.extend(naming_violations)
    
    if violations:
        for violation in violations:
        print("   1. Run: python tools/repo_organizer.py --fix")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()