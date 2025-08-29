"""
🚀 VIPER Core Trading Systems
Contains the main trading engines and system components
"""

# Core system imports with graceful error handling
try:
    from .main import SimpleVIPERTrader
    from .job_manager import ViperLiveJobManager
    
    # Emergency and system management with individual import handling
    components = []
    
    try:
        from .emergency_stop_system import EmergencyStopSystem
        components.append('EmergencyStopSystem')
    except ImportError:
        EmergencyStopSystem = None
        
    try:
        from .performance_monitoring_system import PerformanceMonitor
        components.append('PerformanceMonitor')
    except ImportError:
        PerformanceMonitor = None
    
    __all__ = ['SimpleVIPERTrader', 'ViperLiveJobManager'] + components
    
except ImportError as e:
    print(f"⚠️ Core module import warning: {e}")
    print("📦 Install dependencies: pip install -r requirements.txt")
    __all__ = []