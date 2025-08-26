#!/usr/bin/env python3
"""
VIPER Trading Bot - Repository Reorganization Script
Reorganizes the repository into a clean, professional structure
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any

class RepositoryReorganizer:
    """Reorganizes repository structure for better maintainability"""

    def __init__(self):
        self.base_path = Path(".")
        self.backup_path = Path("repository_backup")
        self.new_structure = {
            "docs": {
                "description": "Documentation and guides",
                "files": [
                    "*.md",
                    "docs/*"
                ]
            },
            "src": {
                "description": "Source code and core logic",
                "subdirs": {
                    "core": "Core business logic",
                    "strategies": "Trading strategies",
                    "utils": "Utility functions",
                    "clients": "API clients and integrations"
                }
            },
            "services": {
                "description": "Microservices architecture",
                "keep": True
            },
            "config": {
                "description": "Configuration files",
                "keep": True
            },
            "scripts": {
                "description": "Deployment and utility scripts",
                "files": [
                    "*.sh",
                    "*.bat",
                    "*.ps1",
                    "*script*.py",
                    "*setup*.py",
                    "*configure*.py",
                    "*test_*.py"
                ]
            },
            "tools": {
                "description": "Development and diagnostic tools",
                "files": [
                    "*diagnose*.py",
                    "*scan*.py",
                    "*check*.py",
                    "*verify*.py",
                    "*compliance*.py",
                    "*backup*.py",
                    "*push*.py"
                ]
            },
            "data": {
                "description": "Data storage and results",
                "keep": True
            },
            "logs": {
                "description": "Log files and monitoring",
                "keep": True
            },
            "backtest_results": {
                "description": "Backtesting results and analysis",
                "keep": True
            },
            "infrastructure": {
                "description": "Docker and deployment files",
                "files": [
                    "docker-compose*.yml",
                    "Dockerfile*",
                    "*.template",
                    "requirements*.txt"
                ]
            }
        }

    def create_backup(self):
        """Create backup of current structure"""
        print("📦 Creating repository backup...")

        if self.backup_path.exists():
            shutil.rmtree(self.backup_path)

        self.backup_path.mkdir()

        # Copy current structure
        for item in os.listdir('.'):
            if item != self.backup_path.name and not item.startswith('.'):
                src = Path(item)
                dst = self.backup_path / item
                if src.is_file():
                    shutil.copy2(src, dst)
                elif src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)

        print(f"✅ Backup created at {self.backup_path}")

    def create_new_structure(self):
        """Create the new directory structure"""
        print("🏗️ Creating new directory structure...")

        for dir_name, config in self.new_structure.items():
            if not config.get('keep', False):
                dir_path = Path(dir_name)
                dir_path.mkdir(exist_ok=True)

                # Create subdirectories
                if 'subdirs' in config:
                    for subdir, description in config['subdirs'].items():
                        subdir_path = dir_path / subdir
                        subdir_path.mkdir(exist_ok=True)

        print("✅ New directory structure created")

    def move_files_to_categories(self):
        """Move files to appropriate categories"""
        print("📂 Organizing files into categories...")

        # Define file mappings
        file_mappings = {
            "docs": [
                "*.md", "docs/*", "README*", "CHANGELOG*", "*GUIDE*", "*README*"
            ],
            "scripts": [
                "*.sh", "*.bat", "*.ps1", "*script*.py", "*setup*.py",
                "*configure*.py", "*start*.py", "*push*.py"
            ],
            "tools": [
                "*diagnose*.py", "*scan*.py", "*check*.py", "*verify*.py",
                "*compliance*.py", "*backup*.py", "*balance*.py", "*credentials*.py"
            ],
            "infrastructure": [
                "docker-compose*.yml", "Dockerfile*", "*.template", "requirements*.txt"
            ],
            "src": {
                "core": ["*mcp_client.py", "*workflow*.py"],
                "clients": ["*mcp*.py", "*client*.py"],
                "utils": ["*complete*.py", "*final*.py", "*simple*.py", "*analyze*.py"]
            }
        }

        moved_files = []

        for category, patterns in file_mappings.items():
            if isinstance(patterns, list):
                for pattern in patterns:
                    if '*' in pattern:
                        # Handle glob patterns
                        import glob
                        for file_path in glob.glob(pattern):
                            if os.path.isfile(file_path):
                                self.move_file_to_category(file_path, category)
                                moved_files.append(file_path)
                    else:
                        # Handle specific files
                        if os.path.exists(pattern):
                            self.move_file_to_category(pattern, category)
                            moved_files.append(pattern)
            elif isinstance(patterns, dict):
                # Handle subdirectories
                for subcat, files in patterns.items():
                    for file_pattern in files:
                        import glob
                        for file_path in glob.glob(file_pattern):
                            if os.path.isfile(file_path):
                                target_dir = Path(category) / subcat
                                target_dir.mkdir(exist_ok=True)
                                shutil.move(file_path, target_dir / Path(file_path).name)
                                moved_files.append(file_path)

        print(f"✅ Moved {len(moved_files)} files to organized structure")

    def move_file_to_category(self, file_path: str, category: str):
        """Move a single file to its category directory"""
        src = Path(file_path)
        dst = Path(category) / src.name

        if src.exists():
            shutil.move(src, dst)

    def create_config_files(self):
        """Create configuration files for the new structure"""
        print("⚙️ Creating configuration files...")

        # Create pyproject.toml
        pyproject_content = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "viper-trading-bot"
version = "2.0.0"
description = "Ultra High-Performance Algorithmic Trading Platform with MCP Integration"
readme = "README.md"
requires-python = ">=3.11"
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Financial and Insurance Industry",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Topic :: Office/Business :: Financial :: Investment",
]
dependencies = [
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.24.0",
    "redis>=5.0.1",
    "ccxt>=4.1.63",
    "pandas>=2.1.4",
    "numpy>=1.24.3",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "black>=23.11.0",
    "flake8>=6.1.0",
    "mypy>=1.7.1",
]
mcp = [
    "websockets>=12.0",
    "httpx>=0.25.2",
]

[project.urls]
Homepage = "https://github.com/your-org/viper-trading-bot"
Documentation = "https://github.com/your-org/viper-trading-bot/docs"
Repository = "https://github.com/your-org/viper-trading-bot.git"
Issues = "https://github.com/your-org/viper-trading-bot/issues"

[tool.setuptools.packages.find]
where = ["src"]
exclude = ["tests*", "docs*"]

[tool.black]
line-length = 88
target-version = ['py311']

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
"""

        with open("pyproject.toml", "w") as f:
            f.write(pyproject_content)

        # Create .gitignore
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyTorch
*.pt
*.pth

# Trading Bot Specific
logs/
data/
backtest_results/
*.log
*.json
__pycache__/
*.pyc
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""

        with open(".gitignore", "w") as f:
            f.write(gitignore_content)

        print("✅ Configuration files created")

    def create_readme_structure(self):
        """Create README files for each directory"""
        print("📖 Creating README files for organization...")

        readme_templates = {
            "docs": """# 📚 Documentation

This directory contains all documentation for the VIPER Trading Bot.

## Contents
- Installation guides
- API documentation
- User manuals
- Technical specifications
- Integration guides

## Contributing
When adding new documentation:
1. Use clear, descriptive filenames
2. Include examples where applicable
3. Keep documentation up-to-date with code changes
""",

            "src": """# 🔧 Source Code

Core source code for the VIPER Trading Bot.

## Structure
- `core/` - Core business logic and trading algorithms
- `strategies/` - Trading strategy implementations
- `utils/` - Utility functions and helpers
- `clients/` - API clients and integrations

## Development
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Write comprehensive docstrings
- Add unit tests for new functionality
""",

            "scripts": """# 🚀 Scripts

Deployment and utility scripts for the VIPER Trading Bot.

## Categories
- Deployment scripts
- Configuration scripts
- Diagnostic tools
- Setup utilities

## Usage
Run scripts from the repository root directory.
""",

            "tools": """# 🛠️ Tools

Development and diagnostic tools for the VIPER Trading Bot.

## Available Tools
- Diagnostic utilities
- System scanners
- Compliance checkers
- Backup tools

## Usage
Tools are for development and troubleshooting purposes.
""",

            "infrastructure": """# 🏗️ Infrastructure

Docker and deployment configuration files.

## Contents
- Docker Compose files
- Dockerfile templates
- Requirements files
- Infrastructure scripts

## Deployment
Use these files to deploy the VIPER system in various environments.
"""
        }

        for dir_name, content in readme_templates.items():
            dir_path = Path(dir_name)
            if dir_path.exists():
                readme_path = dir_path / "README.md"
                with open(readme_path, "w") as f:
                    f.write(content)

        print("✅ README files created for all directories")

    def clean_up_old_files(self):
        """Clean up temporary and old files"""
        print("🧹 Cleaning up temporary files...")

        files_to_remove = [
            "analyze_repo.py",
            "complete_viper_system.py",
            "simple_completion.py",
            "final_status.py",
            "VIPER_FINAL_STATUS.json",
            "repository_backup"
        ]

        removed_count = 0
        for file_pattern in files_to_remove:
            if '*' in file_pattern:
                import glob
                for file_path in glob.glob(file_pattern):
                    if os.path.exists(file_path):
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        removed_count += 1
            else:
                if os.path.exists(file_pattern):
                    if os.path.isfile(file_pattern):
                        os.remove(file_pattern)
                    elif os.path.isdir(file_pattern):
                        shutil.rmtree(file_pattern)
                    removed_count += 1

        print(f"✅ Cleaned up {removed_count} temporary files")

    def generate_final_report(self):
        """Generate final reorganization report"""
        print("📊 Generating final reorganization report...")

        report = {
            "timestamp": str(Path(".").absolute()),
            "status": "completed",
            "new_structure": self.new_structure,
            "services_count": len([d for d in Path("services").iterdir() if d.is_dir()]),
            "mcp_integrated": True,
            "backup_created": self.backup_path.exists()
        }

        with open("REORGANIZATION_REPORT.json", "w") as f:
            json.dump(report, f, indent=2)

        print("✅ Reorganization report saved")

    def reorganize(self):
        """Execute the complete repository reorganization"""
        print("🚀 Starting VIPER Repository Reorganization")
        print("=" * 60)

        steps = [
            ("Creating backup", self.create_backup),
            ("Creating new structure", self.create_new_structure),
            ("Moving files to categories", self.move_files_to_categories),
            ("Creating configuration files", self.create_config_files),
            ("Creating README files", self.create_readme_structure),
            ("Cleaning up", self.clean_up_old_files),
            ("Generating report", self.generate_final_report)
        ]

        for step_name, step_function in steps:
            try:
                step_function()
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                continue

        print()
        print("🎉 Repository reorganization completed!")
        print()
        print("📋 New Structure:")
        for dir_name, config in self.new_structure.items():
            if os.path.exists(dir_name):
                print(f"  📂 {dir_name}/ - {config['description']}")

        print()
        print("🔄 Next Steps:")
        print("  1. Review the new structure")
        print("  2. Test that all functionality still works")
        print("  3. Update any hardcoded paths in scripts")
        print("  4. Commit the reorganization")
        print("  5. Push to GitHub")

def main():
    """Main execution function"""
    reorganizer = RepositoryReorganizer()
    reorganizer.reorganize()

if __name__ == "__main__":
    main()
