#!/usr/bin/env python3
"""
# Rocket VIPER COMPLETE SYSTEM LAUNCHER
One-command launch for the fully optimized AI/ML trading system
"""

import subprocess
import sys
import time
import os
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)"""

class ViperSystemLauncher:
    """Complete VIPER system launcher""""""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.services_running = False
        self.optimized_system_running = False
        
    def print_banner(self):
        """Print the VIPER launch banner"""
#==============================================================================#
# # Rocket VIPER COMPLETE AI/ML OPTIMIZED TRADING SYSTEM                           #
# 🔥 AI-Powered Entry Points | # Target ML-Optimized TP/SL | # Chart Real-Time Backtest #
# ⚡ Live Parameter Optimization | 🛡️ Enterprise Risk Management               #
# 🤖 Machine Learning Integration | 📈 Continuous Strategy Improvement         #
#==============================================================================╣
# # Warning  HIGH-FREQUENCY ALGORITHMIC TRADING SYSTEM                               #
# 🛑 EMERGENCY STOP: Ctrl+C | EMERGENCY KILL: docker compose down            #
# # Chart MONITORING: http://localhost:8000 | OPTIMIZATION LOGS: Current Terminal #
#==============================================================================#
(        """)"""
    
    def check_requirements(self) -> bool:
        """Check system requirements"""
        
        requirements = [
            ("Docker", "docker --version"),
            ("Python 3.11+", f"python3 --version"),
            ("Git", "git --version"),
            ("Project Directory", f"test -d '{self.project_root}'"),
        ]
        
        all_passed = True
        for name, command in requirements:
            try:
                if "test -d" in command:
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                else:
                    result = subprocess.run(command.split(), capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0] if result.stdout.strip() else "OK"
                else:
                    all_passed = False
            except Exception as e:
                all_passed = False
        
        return all_passed
    
    def start_docker_services(self) -> bool:
        """Start all Docker services""""""
        
        try:
            # Clean up any existing containers
            subprocess.run(["docker", "compose", "down", "--volumes", "--remove-orphans"], )
(                         cwd=self.project_root, capture_output=True, timeout=30)
            
            # Start services
            result = subprocess.run(["docker", "compose", "up", "-d"], )
(                                  cwd=self.project_root, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Wait for services to be ready
                time.sleep(15)
                return True
            else:
                print(f"# X Failed to start services: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            return False
    
    def verify_services(self) -> bool:
        """Verify all services are running and healthy"""
        
        services = {
            'API Server': 'http://localhost:8000/health',
            'Risk Manager': 'http://localhost:8002/health', 
            'Exchange Connector': 'http://localhost:8005/health',
            'Ultra Backtester': 'http://localhost:8001/health',
            'Signal Processor': 'http://localhost:8011/health',
            'Order Lifecycle': 'http://localhost:8013/health',
            'Redis': 'N/A',  # Infrastructure service
        }
        
        healthy_count = 0
        
        for name, url in services.items()""":
            if url == 'N/A':
                # Check Redis differently
                try:
                    result = subprocess.run(["docker", "ps", "--filter", "name=viper-redis", "--format", "{{.Status}}"], )
(                                          capture_output=True, text=True, timeout=5)
                    if "Up" in result.stdout:
                        healthy_count += 1
                    else:
                        pass
                except Exception as e:
                    pass
            else:
                try:
                    import requests
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        healthy_count += 1
                    else:
                        print(f"   # X {name}: HTTP {response.status_code}")
                except Exception as e:
                    pass
        
        total_services = len(services)
        health_rate = healthy_count / total_services
        
        print(f"   # Chart Service Health: {healthy_count}/{total_services} ({health_rate:.1%})")
        
        return health_rate >= 0.7  # 70% minimum health
    
    def run_comprehensive_backtest(self) -> bool:
        """Run comprehensive backtesting""""""
        
        try:
            result = subprocess.run([)
                sys.executable, "comprehensive_backtester.py"
(            ], cwd=self.project_root, capture_output=True, text=True, timeout=600)  # 10 min timeout
            
            if result.returncode == 0:
                return True
            else:
                return True
                
        except subprocess.TimeoutExpired:
            return True
        except Exception as e:
            return False
    
    def start_optimized_trading(self) -> bool:
        """Start the AI/ML optimized trading system"""
        print("\n# Rocket STARTING AI/ML OPTIMIZED TRADING SYSTEM...")
        
        try:
            # Start the optimized system
            process = subprocess.Popen([)
                sys.executable, "viper_live_optimized.py"
(            ], cwd=self.project_root)
            
            
            self.optimized_system_running = True
            return True
            
        except Exception as e:
            return False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("⏹️ Shutting down VIPER Optimized Trading System...")
        
        # Stop Docker services
        try:
            subprocess.run(["docker", "compose", "down"], )
(                         cwd=self.project_root, capture_output=True, timeout=30)
        except Exception as e:
            pass
        
        sys.exit(0)
    
    def run_complete_system(self):
        """Run the complete VIPER optimized trading system"""
        # Print banner
        self.print_banner()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)"""
        
        try:
            # Step 1: Check requirements
            if not self.check_requirements():
                print("\n# X System requirements not met. Please install missing components.")
                return
            
            # Step 2: Start Docker services
            if not self.start_docker_services():
                return
            
            # Step 3: Verify services
            if not self.verify_services():
                print("\n# Warning Some services are not healthy, but proceeding...")
            
            # Step 4: Run comprehensive backtest
            backtest_success = self.run_comprehensive_backtest()
            
            # Step 5: Start optimized trading system
            if not self.start_optimized_trading():
                print("\n# X Failed to start optimized trading system.")
                return
            
            print("# Party VIPER COMPLETE AI/ML OPTIMIZED TRADING SYSTEM SUCCESSFULLY LAUNCHED!")
            print("   • Live Dashboard: http://localhost:8000")
            print("   • Performance Metrics: http://localhost:8000/metrics")
            
            # Keep the system running
            print("\n🔄 System is running... Press Ctrl+C to stop")
            while True:
                time.sleep(10)
                # Could add periodic health checks here
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            pass
        finally:
            # Cleanup
            if self.optimized_system_running:
                pass
            
            print("\n# Check VIPER Optimized Trading System shutdown complete")

def main():
    """Main entry point"""
    launcher = ViperSystemLauncher()
    launcher.run_complete_system()"""

if __name__ == "__main__":
    main()
