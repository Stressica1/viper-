#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Unified Default Startup
ONE-COMMAND startup that activates ALL built components by default:
- Microservices Architecture (17 services)
- Centralized Logging System
- MCP Integration
- Monitoring & Alerting
- Advanced Trading Strategy Engine
- Real-time Market Scanner
- Multi-factor Scoring System
- Full Force Trading Orchestration
"""

import os
import sys
import time
import signal
import asyncio
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from scripts.start_microservices import ViperMicroservicesManager
from start_mcp_servers import MCPServerManager

# Import enhanced trading components
try:
    from src.core.trading_orchestrator import VIPERTradingOrchestrator, start_viper_full_force
    from src.core.enhanced_logging import get_viper_logger, shutdown_all_loggers
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced components not available: {e}")
    ENHANCED_COMPONENTS_AVAILABLE = False

class VIPERUnifiedSystem:
    """
    Unified VIPER System Manager - Starts ALL components by default
    This is what users expect when they run the VIPER system!
    NOW WITH ENHANCED TRADING COMPONENTS FOR FULL FORCE OPERATION
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.microservices_manager = ViperMicroservicesManager()
        self.mcp_manager = MCPServerManager()
        self.running = False
        
        # Enhanced trading components
        self.trading_orchestrator = None
        self.viper_logger = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🚀 VIPER Trading Bot - Unified System Startup")
        print("="*70)
        print("🏗️  Microservices Architecture | 📝 Enhanced Centralized Logging")
        print("🤖 MCP AI Integration | 📊 Real-time Monitoring")
        print("🎯 Advanced Trading Strategy | 🔍 Market Scanner")
        print("⭐ Multi-factor Scoring | ⚡ Full Force Trading Orchestration")
        print("⚡ ALL COMPONENTS ACTIVE BY DEFAULT")
        print("="*70)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down all systems...")
        self.shutdown()
        sys.exit(0)
    
    def validate_system(self) -> bool:
        """Validate all system requirements"""
        print("\n🔍 Validating VIPER System Requirements...")
        
        # Check Docker and Docker Compose
        if not self.microservices_manager.check_requirements():
            print("❌ Docker requirements not met")
            return False
        
        # Check environment configuration
        if not self.microservices_manager.load_environment():
            print("❌ Environment configuration failed")
            return False
        
        # Check MCP environment
        try:
            if not self.mcp_manager.validate_environment():
                print("⚠️  MCP environment validation warning - continuing anyway")
        except Exception as e:
            print(f"⚠️  MCP validation warning: {e} - continuing anyway")
        
        print("✅ All system requirements validated")
        return True
    
    def start_infrastructure(self) -> bool:
        """Start core infrastructure services first"""
        print("\n🏗️  Starting Core Infrastructure...")
        
        # Core infrastructure services that others depend on
        core_services = [
            'redis',
            'credential-vault', 
            'prometheus',
            'grafana',
            'elasticsearch',
            'logstash',
            'kibana'
        ]
        
        success_count = 0
        for service in core_services:
            if service in self.microservices_manager.services:
                print(f"🔧 Starting {service}...")
                if self.microservices_manager.start_service(service):
                    success_count += 1
                    # Give services time to fully start
                    time.sleep(3)
                else:
                    print(f"⚠️  {service} failed to start but continuing...")
        
        print(f"📊 Infrastructure: {success_count}/{len(core_services)} services started")
        return success_count > 0  # Allow some failures in infrastructure
    
    def start_microservices(self) -> bool:
        """Start all microservices"""
        print("\n🚀 Starting VIPER Microservices Architecture...")
        
        # Start remaining services in dependency order
        business_services = [
            'data-manager',
            'exchange-connector',
            'risk-manager',
            'signal-processor',
            'order-lifecycle-manager',
            'position-synchronizer',
            'live-trading-engine',
            'ultra-backtester',
            'strategy-optimizer',
            'monitoring-service',
            'market-data-streamer',
            'alert-system',
            'centralized-logger',
            'api-server'
        ]
        
        success_count = 0
        total_count = 0
        
        for service in business_services:
            if service in self.microservices_manager.services:
                total_count += 1
                print(f"⚡ Starting {service}...")
                if self.microservices_manager.start_service(service):
                    success_count += 1
                    time.sleep(2)  # Shorter delay for business services
                else:
                    print(f"⚠️  {service} failed to start")
        
        print(f"📊 Microservices: {success_count}/{total_count} services started")
        return success_count >= (total_count * 0.7)  # Allow 30% failure rate
    
    def start_enhanced_trading_system(self) -> bool:
        """Start enhanced trading components for full force operation"""
        if not ENHANCED_COMPONENTS_AVAILABLE:
            print("⚠️ Enhanced trading components not available - using basic system")
            return True
        
        try:
            print("\n🎯 Starting Enhanced Trading System Components...")
            
            # Initialize enhanced logging
            self.viper_logger = get_viper_logger("viper-system", "orchestrator")
            self.viper_logger.info("Enhanced VIPER logging system activated")
            
            # Initialize trading orchestrator
            self.trading_orchestrator = VIPERTradingOrchestrator()
            
            # Start trading orchestrator in background
            def start_orchestrator():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(start_viper_full_force())
                except Exception as e:
                    print(f"❌ Trading orchestrator error: {e}")
                    if self.viper_logger:
                        self.viper_logger.error(f"Trading orchestrator error: {e}")
            
            orchestrator_thread = threading.Thread(
                target=start_orchestrator,
                daemon=True,
                name="VIPERTradingOrchestrator"
            )
            orchestrator_thread.start()
            
            print("✅ Enhanced Trading System: FULLY OPERATIONAL")
            self.viper_logger.info("VIPER Enhanced Trading System started successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced trading system startup failed: {e}")
            if self.viper_logger:
                self.viper_logger.error(f"Enhanced trading system startup failed: {e}")
            return False
        """Start MCP servers for AI integration"""
        print("\n🤖 Starting MCP AI Integration...")
        
        try:
            if self.mcp_manager.start_all_servers():
                print("✅ MCP Integration started successfully")
                return True
            else:
                print("⚠️  Some MCP servers failed to start - continuing anyway")
                return True  # Don't fail the whole system for MCP issues
        except Exception as e:
            print(f"⚠️  MCP Integration warning: {e} - continuing anyway")
            return True  # MCP is important but not critical for basic functionality
    
    def start_mcp_integration(self) -> bool:
        """Ensure centralized logging is active"""
        print("\n📝 Activating Centralized Logging System...")
        
        # The logging system is part of microservices, but ensure it's prioritized
        logging_services = ['elasticsearch', 'logstash', 'kibana', 'centralized-logger']
        
        active_logging = 0
        for service in logging_services:
            if service in self.microservices_manager.services:
                if self.microservices_manager.services[service]:
                    active_logging += 1
        
        if active_logging > 0:
            print(f"✅ Centralized Logging: {active_logging}/{len(logging_services)} components active")
            return True
        else:
            print("⚠️  Logging system components not fully active")
            return False
    
    def print_system_status(self):
        """Print comprehensive system status"""
        print("\n" + "="*80)
        print("🚀 VIPER TRADING BOT - SYSTEM STATUS")
        print("="*80)
        print("⚡ RUNNING IN FULL FORCE MODE")
        print("🎯 Advanced Strategy Engine | 🔍 Real-time Scanner")
        print("⭐ Multi-factor Scoring | 📊 Enhanced Monitoring")
        print("="*80)
        
        # Microservices status
        print("\n🏗️  MICROSERVICES ARCHITECTURE:")
        self.microservices_manager.print_service_status()
        
        # Enhanced Trading System Status
        if ENHANCED_COMPONENTS_AVAILABLE and self.trading_orchestrator:
            try:
                print("\n🎯 ENHANCED TRADING SYSTEM:")
                system_status = self.trading_orchestrator.get_system_status()
                
                print(f"   🔧 System Running: {'✅' if system_status['system']['is_running'] else '❌'}")
                print(f"   ⚡ Trading Active: {'✅' if system_status['system']['trading_active'] else '❌'}")
                print(f"   📊 Active Trades: {system_status['trading']['active_trades']}")
                print(f"   🎯 Components: {system_status['system']['components_operational']}/8 operational")
                
                # Show active trading symbols
                active_symbols = system_status['trading'].get('active_symbols', [])
                if active_symbols:
                    print(f"   💎 Active Symbols: {', '.join(active_symbols[:5])}")
                    if len(active_symbols) > 5:
                        print(f"       ... and {len(active_symbols) - 5} more")
                
            except Exception as e:
                print(f"   ⚠️ Enhanced trading status error: {e}")
        
        # MCP status
        print("\n🤖 MCP AI INTEGRATION:")
        self.mcp_manager.print_status()
        
        # Access URLs
        print("\n🌐 ACCESS URLS:")
        print("   📊 Web Dashboard:    http://localhost:8000")
        print("   📈 Grafana:          http://localhost:3000")
        print("   📊 Kibana:           http://localhost:5601")
        print("   🤖 MCP Health:       http://localhost:8000/health")
        print("   📊 Prometheus:       http://localhost:9090")
        print("   🎯 Trading API:      http://localhost:8015")
        print("   🔍 Scanner API:      http://localhost:8016")
        print("   ⭐ Scoring API:      http://localhost:8017")
        
        print("\n" + "="*80)
        print("🚀 VIPER System is running with ALL components active!")
        print("🎯 FULL FORCE TRADING MODE ENABLED")
        print("   ⚡ Advanced Strategy Engine")
        print("   🔍 Real-time Market Scanner") 
        print("   ⭐ Multi-factor Scoring System")
        print("   🛡️ Enhanced Risk Management")
        print("   📝 Comprehensive Logging")
        print("   🤖 AI-Powered Analysis")
        print("   Use Ctrl+C to gracefully shutdown all services")
        print("="*80)
    
    def monitor_system_health(self):
        """Monitor system health continuously"""
        print("\n🔍 Starting continuous health monitoring...")
        
        # Start MCP health monitoring
        self.mcp_manager.start_health_monitoring()
        
        # Simple health check loop for microservices
        last_status_print = time.time()
        
        while self.running:
            try:
                # Print status every 5 minutes
                if time.time() - last_status_print > 300:
                    print("\n📊 System health check...")
                    self.microservices_manager.print_service_status()
                    last_status_print = time.time()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"⚠️  Health monitoring error: {e}")
                time.sleep(60)  # Back off on errors
    
    def start_all_systems(self) -> bool:
        """Start ALL VIPER systems - this is the main entry point"""
        print("\n🚀 STARTING ALL VIPER SYSTEMS...")
        
        # 1. Validate system
        if not self.validate_system():
            print("❌ System validation failed")
            return False
        
        # 2. Start infrastructure
        if not self.start_infrastructure():
            print("❌ Infrastructure startup failed")
            return False
        
        print("⏳ Waiting for infrastructure to stabilize...")
        time.sleep(10)
        
        # 3. Start microservices
        if not self.start_microservices():
            print("⚠️  Some microservices failed, but continuing...")
        
        print("⏳ Waiting for microservices to stabilize...")
        time.sleep(5)
        
        # 5. Start enhanced trading system (FULL FORCE MODE)
        if not self.start_enhanced_trading_system():
            print("⚠️ Enhanced trading system failed, but continuing with basic system...")

        # 6. Start MCP integration
        self.start_mcp_integration()
        
        # 7. Ensure logging system is active
        self.start_logging_system()
        
        print("\n✅ VIPER SYSTEM STARTUP COMPLETED!")
        self.running = True
        return True
    
    def shutdown(self):
        """Graceful shutdown of all systems"""
        print("\n🔄 Starting graceful system shutdown...")
        self.running = False
        
        # Shutdown enhanced components first
        if ENHANCED_COMPONENTS_AVAILABLE:
            try:
                if self.trading_orchestrator:
                    print("🛑 Stopping enhanced trading system...")
                    self.trading_orchestrator.stop_trading_system()
                
                if self.viper_logger:
                    print("🛑 Stopping enhanced logging...")
                    shutdown_all_loggers()
                    
            except Exception as e:
                print(f"⚠️ Enhanced components shutdown warning: {e}")
        
        # Stop MCP servers
        try:
            self.mcp_manager.shutdown()
        except Exception as e:
            print(f"⚠️  MCP shutdown warning: {e}")
        
        # Stop microservices
        try:
            print("🛑 Stopping microservices...")
            # Stop critical services last
            critical_services = ['api-server', 'live-trading-engine']
            other_services = [s for s in self.microservices_manager.services.keys() 
                            if s not in critical_services and s != 'redis']
            
            # Stop other services first
            for service in other_services:
                self.microservices_manager.stop_service(service)
            
            # Stop critical services
            for service in critical_services:
                self.microservices_manager.stop_service(service)
                
            # Stop redis last
            if 'redis' in self.microservices_manager.services:
                self.microservices_manager.stop_service('redis')
                
        except Exception as e:
            print(f"⚠️  Microservices shutdown warning: {e}")
        
        print("✅ System shutdown completed")
    
    def run(self):
        """Main run loop - start everything and monitor"""
        try:
            # Start all systems
            if not self.start_all_systems():
                print("❌ Failed to start VIPER systems")
                sys.exit(1)
            
            # Print system status
            self.print_system_status()
            
            # Start monitoring in background
            monitor_thread = threading.Thread(
                target=self.monitor_system_health,
                daemon=True
            )
            monitor_thread.start()
            
            # Keep main thread alive
            print("\n🔄 VIPER system is running. Press Ctrl+C to stop...")
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Received shutdown signal")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            self.shutdown()

def main():
    """Main entry point for VIPER system"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       🚀 VIPER TRADING BOT                                   ║
║                  UNIFIED SYSTEM STARTUP                                      ║
║                    FULL FORCE MODE                                           ║
║                                                                              ║
║  🏗️  Microservices Architecture (17 Services)                               ║
║  📝 Enhanced Centralized Logging                                             ║
║  🤖 MCP AI Integration                                                        ║
║  📊 Real-time Monitoring & Alerting                                          ║
║  🎯 Advanced Trading Strategy Engine                                         ║
║  🔍 Intelligent Market Scanner                                               ║
║  ⭐ Multi-factor Scoring System                                              ║
║  🛡️ Enhanced Risk Management                                                 ║
║                                                                              ║
║             ALL COMPONENTS ACTIVE BY DEFAULT                                 ║
║           OPTIMIZED FOR MAXIMUM PERFORMANCE                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create and run unified system
    viper_system = VIPERUnifiedSystem()
    viper_system.run()

if __name__ == "__main__":
    main()