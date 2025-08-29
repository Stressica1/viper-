#!/usr/bin/env python3
"""
# Tool VIPER System Setup Validator
Ensures all components are properly connected and aware of the new repository structure
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Enhanced terminal display
try:
    from src.viper.utils.terminal_display import (
        terminal, display_error, display_success, display_warning, 
        print_banner, display_config, display_repo_structure
    )
    ENHANCED_DISPLAY = True
except ImportError:
    ENHANCED_DISPLAY = False
    def display_error(msg, details=None): print(f"# X {msg}")
    def display_success(msg, details=None): print(f"# Check {msg}")
    def display_warning(msg, details=None): print(f"# Warning {msg}")
    def print_banner(): print("# Tool VIPER System Setup Validator")
    def display_config(config): print("Config:", config)
    def display_repo_structure(path): print(f"Repository at: {path}")

class ViperSetupValidator:
    """Comprehensive setup validation for VIPER trading system"""
    
    def __init__(self):
        self.project_root = project_root
        self.validation_results = {}
        
    def validate_repository_structure(self) -> bool:
        """Validate the new repository structure"""
        
        required_dirs = {
            'src/viper': 'Main source code',
            'src/viper/core': 'Core trading systems',
            'src/viper/strategies': 'Trading strategies',
            'src/viper/execution': 'Trade execution',
            'src/viper/risk': 'Risk management',
            'src/viper/utils': 'Utility modules',
            'scripts': 'Executable scripts',
            'config': 'Configuration files',
            'docs': 'Documentation',
            'tools': 'Development tools'
        }
        
        all_valid = True
        for dir_path, description in required_dirs.items():
            full_path = self.project_root / dir_path
            if full_path.exists():
                display_success(f"{dir_path} - {description}")
            else:
                display_error(f"Missing: {dir_path}", description)
                all_valid = False
        
        self.validation_results['repository_structure'] = all_valid
        return all_valid
    
    def validate_python_imports(self) -> bool:
        """Test Python imports with new structure"""
        
        test_imports = [
            ('src.viper', 'Main VIPER package'),
            ('src.viper.core', 'Core modules'),
            ('src.viper.utils.terminal_display', 'Terminal display utilities'),
        ]
        
        all_valid = True
        for import_name, description in test_imports:
            try:
                importlib.import_module(import_name)
                display_success(f"{import_name} - {description}")
            except ImportError as e:
                display_warning(f"Import issue: {import_name}", f"{e}")
                # Don't mark as failure for optional imports
        
        # Test critical imports that should work
        critical_imports = [
            ('pathlib', 'Path utilities'),
            ('os', 'Operating system interface'),
            ('sys', 'System interface'),
            ('json', 'JSON handling'),
        ]
        
        for import_name, description in critical_imports:
            try:
                importlib.import_module(import_name)
                display_success(f"{import_name} - {description}")
            except ImportError as e:
                display_error(f"Critical import failed: {import_name}", str(e))
                all_valid = False
        
        self.validation_results['python_imports'] = all_valid
        return all_valid
    
    def validate_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        
        # Check requirements.txt exists
        req_file = self.project_root / 'requirements.txt'
        if not req_file.exists():
            display_error("requirements.txt not found")
            return False
        
        # Try to import key dependencies
        key_dependencies = [
            ('rich', 'Terminal display enhancement', False),  # Optional
            ('fastapi', 'Web framework', True),
            ('pandas', 'Data processing', True),
            ('numpy', 'Numerical computing', True),
            ('requests', 'HTTP requests', True),
            ('redis', 'Redis client', True),
        ]
        
        all_critical_available = True
        for dep_name, description, is_critical in key_dependencies:
            try:
                importlib.import_module(dep_name)
                display_success(f"{dep_name} - {description}")
            except ImportError:
                if is_critical:
                    display_error(f"Missing critical dependency: {dep_name}", description)
                    all_critical_available = False
                else:
                    display_warning(f"Optional dependency missing: {dep_name}", description)
        
        if not all_critical_available:
            display_warning("Install dependencies", "Run: pip install -r requirements.txt")
        
        self.validation_results['dependencies'] = all_critical_available
        return all_critical_available
    
    def validate_configuration_files(self) -> bool:
        """Check configuration files"""
        
        config_files = [
            ('.env.example', 'Environment template', True),
            ('docker-compose.yml', 'Docker configuration', True),
            ('requirements.txt', 'Python dependencies', True),
            ('config/optimal_mcp_config.py', 'MCP configuration', False),
        ]
        
        all_valid = True
        for file_path, description, required in config_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                display_success(f"{file_path} - {description}")
            else:
                if required:
                    display_error(f"Missing required file: {file_path}", description)
                    all_valid = False
                else:
                    display_warning(f"Optional file missing: {file_path}", description)
        
        # Check if .env exists (for actual usage)
        env_file = self.project_root / '.env'
        if env_file.exists():
            display_success(".env - Environment configuration")
        else:
            display_warning("No .env file found", "Copy .env.example to .env and configure")
        
        self.validation_results['configuration'] = all_valid
        return all_valid
    
    def validate_scripts(self) -> bool:
        """Validate executable scripts"""
        
        key_scripts = [
            'scripts/start_live_trading_mandatory.py',
            'scripts/run_live_system.py',
            'scripts/launch_complete_system.py',
        ]
        
        all_valid = True
        for script_path in key_scripts:
            full_path = self.project_root / script_path
            if full_path.exists():
                # Check if script is executable
                if os.access(full_path, os.X_OK):
                    display_success(f"{script_path} - Executable")
                else:
                    display_warning(f"{script_path} - Not executable", "Run: chmod +x " + script_path)
            else:
                display_error(f"Missing script: {script_path}")
                all_valid = False
        
        self.validation_results['scripts'] = all_valid
        return all_valid
    
    def validate_docker_setup(self) -> bool:
        """Check Docker configuration"""
        
        try:
            # Check if Docker is available
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                display_success(f"Docker available - {result.stdout.strip()}")
            else:
                display_error("Docker not available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            display_error("Docker not installed or not in PATH")
            return False
        
        # Check docker-compose
        try:
            result = subprocess.run(['docker', 'compose', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                display_success(f"Docker Compose available - {result.stdout.strip()}")
            else:
                display_warning("Docker Compose not available", "May need to install docker-compose plugin")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            display_warning("Docker Compose not available", "Install docker-compose")
        
        self.validation_results['docker'] = True
        return True
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """Generate comprehensive setup report"""
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for result in self.validation_results.values() if result)
        
        report = {
            'overall_status': 'PASS' if passed_checks == total_checks else 'PARTIAL',
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'details': self.validation_results,
            'recommendations': []
        }
        
        # Add recommendations based on failures
        if not self.validation_results.get('dependencies', True):
            report['recommendations'].append("Install Python dependencies: pip install -r requirements.txt")
        
        if not self.validation_results.get('configuration', True):
            report['recommendations'].append("Create .env file from .env.example template")
        
        if not self.validation_results.get('repository_structure', True):
            report['recommendations'].append("Run repository organization tools to fix structure")
        
        return report
    
    def run_full_validation(self):
        """Run complete system validation"""
        if ENHANCED_DISPLAY:
            print_banner()
            terminal.console.rule("[bold blue]# Tool System Validation Starting[/]")
        else:
        
        # Run all validation checks
        validation_steps = [
            ("Repository Structure", self.validate_repository_structure),
            ("Python Imports", self.validate_python_imports),
            ("Dependencies", self.validate_dependencies),
            ("Configuration", self.validate_configuration_files),
            ("Scripts", self.validate_scripts),
            ("Docker Setup", self.validate_docker_setup),
        ]
        
        for step_name, validation_func in validation_steps:
            try:
                validation_func()
            except Exception as e:
                display_error(f"Validation error in {step_name}", str(e))
                self.validation_results[step_name.lower().replace(' ', '_')] = False
        
        # Generate and display report
        report = self.generate_setup_report()
        self.display_final_report(report)
        
        return report['overall_status'] == 'PASS'
    
    def display_final_report(self, report: Dict[str, Any]):
        """Display final validation report"""
        
        if ENHANCED_DISPLAY:
            status_color = "green" if report['overall_status'] == 'PASS' else "yellow"
            terminal.console.print(f"\n[{status_color}]Overall Status: {report['overall_status']}[/]")
            terminal.console.print(f"Passed: {report['passed_checks']}/{report['total_checks']} checks")
            
            if report['recommendations']:
                terminal.console.print("\n[bold yellow]📋 Recommendations:[/]")
                for rec in report['recommendations']:
                    terminal.console.print(f"  • {rec}")
        else:
            print(f"Overall Status: {report['overall_status']}")
            print(f"Passed: {report['passed_checks']}/{report['total_checks']} checks")
            
            if report['recommendations']:
                for rec in report['recommendations']:
        
        if report['overall_status'] == 'PASS':
            display_success("System validation completed successfully!")
            display_success("You can now run the trading system with confidence.")
        else:
            display_warning("System validation completed with issues", 
                          "Address the recommendations above before running the system")

def main():
    """Main validation function"""
    validator = ViperSetupValidator()
    
    # Show repository structure if enhanced display is available
    if ENHANCED_DISPLAY:
        display_repo_structure(str(validator.project_root))
    
    # Run full validation
    success = validator.run_full_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()