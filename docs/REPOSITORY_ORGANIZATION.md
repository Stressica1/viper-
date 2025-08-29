# 🚀 VIPER Repository Organization Guide

## 📁 Repository Structure

The VIPER trading system follows a clean, organized directory structure to maintain code quality and ease of development:

```
viper-/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # License file
├── 📄 .gitignore                  # Git ignore rules with strict organization
├── 📄 requirements.txt            # Python dependencies
├── 📄 docker-compose.yml         # Main docker compose
├── 📁 src/                        # 🐍 Source Code
│   └── viper/                     # Main package
│       ├── core/                  # Core trading engines & systems
│       ├── strategies/            # Trading strategies & optimization
│       ├── execution/             # Trade execution engines
│       ├── risk/                  # Risk management modules
│       └── utils/                 # Utility modules
├── 📁 scripts/                    # 🚀 Executable Scripts
│   ├── run_*.py                   # Runner scripts
│   ├── start_*.py                 # Startup scripts
│   └── launch_*.py                # Launcher scripts
├── 📁 tests/                      # 🧪 Test Files
│   ├── test_*.py                  # Unit tests
│   └── *_test.py                  # Alternative test naming
├── 📁 docs/                       # 📚 Documentation
│   ├── *.md                       # Markdown documentation
│   ├── api/                       # API documentation
│   ├── deployment/                # Deployment guides
│   └── changelog/                 # Change logs
├── 📁 config/                     # ⚙️ Configuration
│   ├── *.json                     # JSON configurations
│   ├── *.yml                      # YAML configurations
│   └── vault/                     # Security configurations
├── 📁 services/                   # 🔧 Microservices
│   ├── mcp-server/                # MCP service
│   ├── trading-engine/            # Trading engine
│   └── ...                        # Other services
├── 📁 infrastructure/             # 🏗️ Infrastructure Code
│   ├── docker/                    # Docker configurations
│   └── shared/                    # Shared infrastructure
├── 📁 reports/                    # 📊 Generated Reports
│   ├── *.html                     # HTML reports
│   ├── *.pdf                      # PDF reports
│   └── *_report.*                 # Analysis reports
├── 📁 deployments/                # 🚀 Deployment Files
│   └── backups/                   # Backup files & deployments
└── 📁 tools/                      # 🛠️ Development Tools
    ├── diagnostics/               # Diagnostic scripts
    ├── utilities/                 # Utility scripts
    └── repo_organizer.py          # Repository organization tools
```

## 🚫 Root Directory Rules

### ✅ Allowed in Root
- `README.md` - Main project documentation
- `LICENSE` - License file
- `.gitignore` - Git ignore configuration
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker compose configuration
- `Dockerfile` - Main dockerfile
- Environment templates (`.env.example`, `.env.template`)
- Build configuration (`pyproject.toml`, `setup.py`)

### ❌ NOT Allowed in Root
- **Python files** (except `setup.py`) → `src/viper/` or `scripts/`
- **JSON files** → `config/`
- **Documentation** (except `README.md`) → `docs/`
- **Backup files** → `deployments/backups/`
- **Log files** → Should not be committed
- **Test files** → `tests/`
- **Diagnostic scripts** → `tools/diagnostics/`
- **Temporary files** → Should not be committed

## 🛠️ Organization Tools

### Repository Organizer
```bash
# Check repository structure
python tools/repository_rules.py --validate

# Generate violations report
python tools/repository_rules.py --report

# Setup enforcement tools (pre-commit hooks, CI)
python tools/repository_rules.py --setup-enforcement
```

### Clean Root Enforcer
```bash
# Check what would be moved (dry run)
python tools/clean_root_enforcer.py

# Actually move misplaced files
python tools/clean_root_enforcer.py --execute

# Create monitoring script
python tools/clean_root_enforcer.py --create-monitor
```

### File Organizer
```bash
# Simulate organization (shows what would be moved)
python tools/repo_organizer.py

# Actually organize files
python tools/repo_organizer.py --fix

# Generate organization report
python tools/repo_organizer.py --report
```

## 🔒 Enforcement Mechanisms

### 1. Git Pre-commit Hooks
- Automatically installed by `repository_rules.py --setup-enforcement`
- Prevents commits that violate structure rules
- Located in `.git/hooks/pre-commit`

### 2. Enhanced .gitignore
- Prevents accidental commits of files in wrong locations
- Blocks demo files, temporary files, and build artifacts
- Enforces clean repository practices

### 3. GitHub Actions CI
- Validates repository structure on every push/PR
- Located in `.github/workflows/validate_structure.yml`
- Fails builds if structure violations are found

### 4. Real-time Monitoring
```bash
# Start root directory monitor
python tools/root_monitor.py

# Start with auto-cleaning
python tools/root_monitor.py --auto-clean
```

## 📋 File Classification Rules

### Python Files
- **Core Systems**: `*trader*.py`, `*engine*.py`, `*system*.py` → `src/viper/core/`
- **Strategies**: `*strategy*.py`, `*optimization*.py` → `src/viper/strategies/`
- **Execution**: `*execution*.py`, `*trading*.py` → `src/viper/execution/`
- **Risk Management**: `*risk*.py` → `src/viper/risk/`
- **Scripts**: `run_*.py`, `start_*.py`, `launch_*.py` → `scripts/`
- **Tests**: `test_*.py`, `*_test.py` → `tests/`
- **Diagnostics**: `*diagnostic*.py`, `*debug*.py`, `fix_*.py` → `tools/diagnostics/`
- **Utilities**: `*util*.py`, `*validator*.py` → `src/viper/utils/`

### Configuration Files
- **JSON/YAML**: `*.json`, `*.yml`, `*.yaml` → `config/`
- **Service Configs**: Stay in respective service directories
- **Infrastructure Configs**: Stay in `infrastructure/`

### Documentation
- **Main README**: `README.md` → Root (stays)
- **Other Docs**: `*.md`, `*.rst`, `*.txt` → `docs/`
- **Reports**: `*.html`, `*.pdf`, `*_report.*` → `reports/`

### Backup Files
- **All Backups**: `backup_*`, `*_backup*`, `*.backup`, `*.bak` → `deployments/backups/`

## 🔧 Maintenance Commands

### Daily Maintenance
```bash
# Check repository health
python tools/repository_rules.py --validate

# Clean up any misplaced files
python tools/clean_root_enforcer.py --execute
```

### Weekly Cleanup
```bash
# Full organization check
python tools/repo_organizer.py --report

# Generate structure report
python tools/repository_rules.py --report
```

### Setup New Environment
```bash
# Setup all enforcement tools
python tools/repository_rules.py --setup-enforcement

# Create monitoring tools
python tools/clean_root_enforcer.py --create-monitor
```

## 🎯 Benefits of Organization

1. **Clean Root Directory**: Only essential files in root
2. **Logical Structure**: Code organized by functionality
3. **Easy Navigation**: Predictable file locations
4. **Automated Enforcement**: Tools prevent disorganization
5. **CI Integration**: Structure validated on every change
6. **Developer Friendly**: Clear rules and automated fixes

## ⚠️ Migration Notes

If you're working on the repository and encounter structure violations:

1. **Run Organizer**: `python tools/repo_organizer.py --fix`
2. **Check Results**: `python tools/repository_rules.py --validate`
3. **Clean Root**: `python tools/clean_root_enforcer.py --execute`
4. **Commit Changes**: Standard git workflow

The tools are designed to be safe and will show you what they're doing before making changes.

## 📞 Support

For questions about repository organization:
1. Check violation reports: `python tools/repository_rules.py --report`
2. Review this guide
3. Use automated fixes: `python tools/repo_organizer.py --fix`