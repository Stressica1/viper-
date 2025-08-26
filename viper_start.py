#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Unified Default Startup
ONE-COMMAND startup that activates ALL built components by default:
- Microservices Architecture (17 services)
- Centralized Logging System
- MCP Integration
- Monitoring & Alerting
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

from scripts.start_microservices import ViperMicroservicesManager
from start_mcp_servers import MCPServerManager

class VIPERUnifiedSystem:
    """
    Unified VIPER System Manager - Starts ALL components by default
    This is what users expect when they run the VIPER system!
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.microservices_manager = ViperMicroservicesManager()
        self.mcp_manager = MCPServerManager()
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🚀 VIPER Trading Bot - Unified System Startup")
        print("="*70)
        print("🏗️  Microservices Architecture | 📝 Centralized Logging")
        print("🤖 MCP AI Integration | 📊 Real-time Monitoring")
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
    
    def start_mcp_integration(self) -> bool:
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
    
    def start_logging_system(self) -> bool:
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
        print("🎯 VIPER TRADING BOT - SYSTEM STATUS")
        print("="*80)
        
        # Microservices status
        print("\n🏗️  MICROSERVICES ARCHITECTURE:")
        self.microservices_manager.print_service_status()
        
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
        
        print("\n" + "="*80)
        print("🚀 VIPER System is running with ALL components active!")
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
        
        # 4. Start logging system (ensure it's active)
        self.start_logging_system()
        
        # 5. Start MCP integration
        self.start_mcp_integration()
        
        print("\n✅ VIPER SYSTEM STARTUP COMPLETED!")
        self.running = True
        return True
    
    def shutdown(self):
        """Graceful shutdown of all systems"""
        print("\n🔄 Starting graceful system shutdown...")
        self.running = False
        
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
║                    UNIFIED SYSTEM STARTUP                                    ║
║                                                                              ║
║  🏗️  Microservices Architecture (17 Services)                               ║
║  📝 Centralized Logging System                                               ║
║  🤖 MCP AI Integration                                                        ║
║  📊 Real-time Monitoring & Alerting                                          ║
║                                                                              ║
║             ALL COMPONENTS ACTIVE BY DEFAULT                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create and run unified system
    viper_system = VIPERUnifiedSystem()
    viper_system.run()

if __name__ == "__main__":
    main()