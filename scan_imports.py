#!/usr/bin/env python3
"""
VIPER Trading Bot - Import Dependency Scanner
Scans all Python files for missing imports and dependencies
"""

import os
import sys
import ast
import importlib
import subprocess
from pathlib import Path
from typing import Set, List, Dict, Tuple
from collections import defaultdict

class ImportScanner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.failed_imports = defaultdict(list)
        self.missing_packages = set()
        self.successful_imports = set()
        self.all_imports = set()
        
    def get_python_files(self) -> List[Path]:
        """Get all Python files in the repository"""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Skip .git and __pycache__ directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def extract_imports_from_file(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")
            
        return imports
    
    def test_import(self, module_name: str) -> bool:
        """Test if a module can be imported"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def is_builtin_module(self, module_name: str) -> bool:
        """Check if module is a Python builtin"""
        builtin_modules = {
            'os', 'sys', 'ast', 'json', 'time', 'datetime', 'logging', 
            'collections', 'typing', 'pathlib', 'subprocess', 'urllib',
            'http', 'socket', 'threading', 'asyncio', 'functools',
            'itertools', 'operator', 're', 'math', 'random', 'hashlib',
            'base64', 'uuid', 'warnings', 'traceback', 'inspect',
            'contextlib', 'copy', 'pickle', 'csv', 'io', 'shutil',
            'tempfile', 'glob', 'fnmatch', 'configparser', 'argparse',
            'dataclasses', 'enum', 'abc', 'weakref', 'gc'
        }
        return module_name in builtin_modules
    
    def is_local_module(self, module_name: str) -> bool:
        """Check if module is a local project module"""
        # Check if there's a corresponding Python file or directory
        for py_file in self.get_python_files():
            if py_file.stem == module_name:
                return True
        
        # Check for directories that might be modules
        for item in self.root_dir.rglob('*'):
            if item.is_dir() and item.name == module_name:
                return True
                
        return False
    
    def scan_all_files(self) -> Dict[str, any]:
        """Scan all Python files for imports"""
        print("🔍 Scanning all Python files for imports...")
        
        python_files = self.get_python_files()
        print(f"Found {len(python_files)} Python files")
        
        for file_path in python_files:
            print(f"  Scanning: {file_path.relative_to(self.root_dir)}")
            imports = self.extract_imports_from_file(file_path)
            self.all_imports.update(imports)
            
            for imp in imports:
                if self.is_builtin_module(imp):
                    continue
                    
                if self.is_local_module(imp):
                    continue
                    
                if self.test_import(imp):
                    self.successful_imports.add(imp)
                else:
                    self.failed_imports[str(file_path.relative_to(self.root_dir))].append(imp)
                    self.missing_packages.add(imp)
        
        return {
            'total_files': len(python_files),
            'total_imports': len(self.all_imports),
            'successful_imports': len(self.successful_imports),
            'failed_imports': len(self.missing_packages),
            'failed_details': dict(self.failed_imports),
            'missing_packages': list(self.missing_packages)
        }
    
    def suggest_package_names(self, module_name: str) -> List[str]:
        """Suggest possible package names for a missing module"""
        suggestions = []
        
        # Common package mappings
        package_mappings = {
            'dotenv': 'python-dotenv',
            'yaml': 'PyYAML', 
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'sklearn': 'scikit-learn',
            'bs4': 'beautifulsoup4',
            'jwt': 'PyJWT',
            'psycopg2': 'psycopg2-binary',
            'MySQLdb': 'mysqlclient',
            'lxml': 'lxml',
            'dateutil': 'python-dateutil',
            'magic': 'python-magic',
            'serial': 'pyserial',
            'usb': 'pyusb',
            'win32api': 'pywin32',
            'win32com': 'pywin32',
            'tkinter': 'tk',
            'gi': 'PyGObject',
            'wx': 'wxPython',
        }
        
        if module_name in package_mappings:
            suggestions.append(package_mappings[module_name])
        else:
            # Try common patterns
            suggestions.extend([
                module_name,
                f"py{module_name}",
                f"{module_name}-python",
                f"python-{module_name}",
            ])
        
        return suggestions
    
    def generate_requirements_additions(self) -> List[str]:
        """Generate pip install commands for missing packages"""
        additions = []
        for package in sorted(self.missing_packages):
            suggestions = self.suggest_package_names(package)
            additions.append(f"# For module '{package}': pip install {suggestions[0]}")
            for suggestion in suggestions[1:3]:  # Show top 3 suggestions
                additions.append(f"#   Alternative: pip install {suggestion}")
        return additions

def main():
    print("🚀 VIPER Trading Bot - Import Dependency Scanner")
    print("=" * 60)
    
    scanner = ImportScanner(".")
    results = scanner.scan_all_files()
    
    print(f"\n📊 Scan Results:")
    print(f"  Total Python files scanned: {results['total_files']}")
    print(f"  Total unique imports found: {results['total_imports']}")
    print(f"  Successful imports: {results['successful_imports']}")
    print(f"  Failed imports: {results['failed_imports']}")
    
    if scanner.missing_packages:
        print(f"\n❌ Missing Packages ({len(scanner.missing_packages)}):")
        for package in sorted(scanner.missing_packages):
            print(f"  - {package}")
            
        print(f"\n🔧 Suggested Installation Commands:")
        additions = scanner.generate_requirements_additions()
        for addition in additions:
            print(f"  {addition}")
            
        print(f"\n📝 Files with missing imports:")
        for file_path, missing_imports in scanner.failed_imports.items():
            print(f"  {file_path}:")
            for imp in missing_imports:
                print(f"    - {imp}")
    else:
        print(f"\n✅ All imports successful! No missing packages detected.")
        
    print("\n" + "=" * 60)
    
    # Return error code if missing packages found
    return 1 if scanner.missing_packages else 0

if __name__ == "__main__":
    sys.exit(main())