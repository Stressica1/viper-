"""
# Rocket VIPER Trading System
High-Performance Automated Trading Platform
"""

__version__ = "2.5.4"
__author__ = "VIPER Trading Team"

# Ensure proper path setup for imports
import os
import sys
from pathlib import Path

# Add project root to Python path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "src"))

# Export commonly used modules for easier imports
try:
    # Try importing core modules individually
    available_modules = []
    
    try:
        from . import core
        available_modules.append('core')
    except ImportError as e:
    
    try:
        from . import strategies
        available_modules.append('strategies')
    except ImportError:
        pass  # Optional module
    
    try:
        from . import execution
        available_modules.append('execution')
    except ImportError:
        pass  # Optional module
    
    try:
        from . import risk
        available_modules.append('risk')
    except ImportError:
        pass  # Optional module
    
    try:
        from . import utils
        available_modules.append('utils')
    except ImportError:
        pass  # Optional module
    
    __all__ = available_modules
    
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"# Warning Some VIPER modules not available: {e}")
    print("📦 Run: pip install -r requirements.txt to install dependencies")
    
    __all__ = []