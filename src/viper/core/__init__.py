"""
🚀 VIPER Core Trading Systems
Contains the main trading engines and system components
"""

# Core system imports with graceful error handling
try:
    
    # Emergency and system management with individual import handling
    components = []
    
    try:
        components.append('EmergencyStopSystem')
    except ImportError:
        EmergencyStopSystem = None
        
    try:
        components.append('PerformanceMonitor')
    except ImportError:
        PerformanceMonitor = None
    
    __all__ = ['SimpleVIPERTrader', 'ViperLiveJobManager'] + components
    
except ImportError as e:
    print("📦 Install dependencies: pip install -r requirements.txt")
    __all__ = []