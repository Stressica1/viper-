#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - MCP Servers Startup Script
Comprehensive startup script for all MCP servers with health monitoring
"""

import os
import sys
import time
import asyncio
import subprocess
import logging
from typing import Dict, List, Optional
from pathlib import Path
import signal
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPServerManager:
    """Manager for all MCP servers"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        
        # MCP Server configurations
        self.mcp_servers = {
            "viper-trading-system": {
                "command": "python",
                "args": ["services/mcp-server/main.py"],
                "port": 8000,
                "description": "VIPER Trading System MCP Server",
                "process": None,
                "status": "stopped"
            },
            "github-project-manager": {
                "command": "python",
                "args": ["services/github-manager/main.py"],
                "port": 8001,
                "description": "GitHub Project Management MCP Server",
                "process": None,
                "status": "stopped"
            },
            "trading-optimizer": {
                "command": "python",
                "args": ["services/trading-optimizer/main.py"],
                "port": 8002,
                "description": "Trading Strategy Optimizer MCP Server",
                "process": None,
                "status": "stopped"
            }
        }
        
        # Health check intervals
        self.health_check_interval = 30  # seconds
        self.health_check_thread = None
        self.running = False
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    def validate_environment(self) -> bool:
        """Validate environment configuration"""
        logger.info("🔍 Validating environment configuration...")
        
        if not self.env_file.exists():
            logger.error(f"❌ .env file not found at {self.env_file}")
            return False
        
        # Check required environment variables
        required_vars = [
            'GITHUB_PAT',
            'GITHUB_OWNER', 
            'GITHUB_REPO',
            'BITGET_API_KEY',
            'BITGET_API_SECRET'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("   Some MCP servers may not function correctly")
        else:
            logger.info("✅ All required environment variables are configured")
        
        # Check Python dependencies
        try:
            import fastapi
            import uvicorn
            import httpx
            logger.info("✅ All required Python packages are available")
        except ImportError as e:
            logger.error(f"❌ Missing Python package: {e}")
            return False
        
        return True
    
    def start_server(self, server_name: str) -> bool:
        """Start a specific MCP server"""
        if server_name not in self.mcp_servers:
            logger.error(f"❌ Unknown MCP server: {server_name}")
            return False
        
        server_config = self.mcp_servers[server_name]
        
        if server_config["status"] == "running":
            logger.info(f"ℹ️  {server_name} is already running")
            return True
        
        try:
            logger.info(f"🚀 Starting {server_name}...")
            
            # Change to project root directory
            os.chdir(self.project_root)
            
            # Start the server process
            cmd = [server_config["command"]] + server_config["args"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )
            
            # Wait a moment to check if it started successfully
            time.sleep(2)
            
            if process.poll() is None:  # Process is still running
                server_config["process"] = process
                server_config["status"] = "running"
                logger.info(f"✅ {server_name} started successfully (PID: {process.pid})")
                return True
            else:
                # Process failed to start
                stdout, stderr = process.communicate()
                logger.error(f"❌ Failed to start {server_name}")
                logger.error(f"   stdout: {stdout}")
                logger.error(f"   stderr: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting {server_name}: {e}")
            return False
    
    def stop_server(self, server_name: str) -> bool:
        """Stop a specific MCP server"""
        if server_name not in self.mcp_servers:
            logger.error(f"❌ Unknown MCP server: {server_name}")
            return False
        
        server_config = self.mcp_servers[server_name]
        
        if server_config["status"] != "running":
            logger.info(f"ℹ️  {server_name} is not running")
            return True
        
        try:
            logger.info(f"🛑 Stopping {server_name}...")
            
            process = server_config["process"]
            if process:
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️  {server_name} didn't stop gracefully, forcing...")
                    process.kill()
                    process.wait()
                
                server_config["process"] = None
                server_config["status"] = "stopped"
                logger.info(f"✅ {server_name} stopped successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error stopping {server_name}: {e}")
            return False
        
        return False
    
    def start_all_servers(self) -> bool:
        """Start all MCP servers"""
        logger.info("🚀 Starting all MCP servers...")
        
        success_count = 0
        total_count = len(self.mcp_servers)
        
        for server_name in self.mcp_servers:
            if self.start_server(server_name):
                success_count += 1
            else:
                logger.error(f"❌ Failed to start {server_name}")
        
        logger.info(f"📊 MCP Servers startup summary: {success_count}/{total_count} successful")
        return success_count == total_count
    
    def stop_all_servers(self) -> bool:
        """Stop all MCP servers"""
        logger.info("🛑 Stopping all MCP servers...")
        
        success_count = 0
        total_count = len(self.mcp_servers)
        
        for server_name in self.mcp_servers:
            if self.stop_server(server_name):
                success_count += 1
            else:
                logger.error(f"❌ Failed to stop {server_name}")
        
        logger.info(f"📊 MCP Servers shutdown summary: {success_count}/{total_count} successful")
        return success_count == total_count
    
    async def check_server_health(self, server_name: str) -> str:
        """Check health of a specific MCP server"""
        server_config = self.mcp_servers[server_name]
        port = server_config["port"]
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    return "healthy"
                else:
                    return "unhealthy"
        except Exception:
            return "unreachable"
    
    async def health_check_loop(self):
        """Continuous health check loop"""
        while self.running:
            try:
                for server_name in self.mcp_servers:
                    health_status = await self.check_server_health(server_name)
                    server_config = self.mcp_servers[server_name]
                    
                    if health_status == "healthy" and server_config["status"] != "healthy":
                        server_config["status"] = "healthy"
                        logger.info(f"✅ {server_name} is healthy")
                    elif health_status != "healthy" and server_config["status"] == "healthy":
                        server_config["status"] = "unhealthy"
                        logger.warning(f"⚠️  {server_name} health check failed: {health_status}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    def start_health_monitoring(self):
        """Start health monitoring in background"""
        if self.health_check_thread and self.health_check_thread.is_alive():
            return
        
        self.running = True
        self.health_check_thread = threading.Thread(
            target=lambda: asyncio.run(self.health_check_loop()),
            daemon=True
        )
        self.health_check_thread.start()
        logger.info("🔍 Health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        logger.info("🔍 Health monitoring stopped")
    
    def print_status(self):
        """Print status of all MCP servers"""
        print("\n" + "="*80)
        print("🚀 VIPER Trading Bot - MCP Servers Status")
        print("="*80)
        
        for server_name, config in self.mcp_servers.items():
            status_icon = {
                "running": "🟢",
                "healthy": "🟢", 
                "stopped": "🔴",
                "unhealthy": "🟡",
                "unreachable": "🟠"
            }.get(config["status"], "❓")
            
            print(f"{status_icon} {server_name:<25} Port: {config['port']:>4} | Status: {config['status']:<12}")
            print(f"   📝 {config['description']}")
            
            if config["process"]:
                print(f"   🔢 PID: {config['process'].pid}")
            
            print()
        
        print("="*80)
    
    def shutdown(self):
        """Graceful shutdown of all servers"""
        logger.info("🔄 Starting graceful shutdown...")
        
        self.stop_health_monitoring()
        self.stop_all_servers()
        
        logger.info("✅ Shutdown completed")

def main():
    """Main entry point"""
    print("🚀 VIPER Trading Bot - MCP Servers Manager")
    print("="*50)
    
    manager = MCPServerManager()
    
    # Validate environment
    if not manager.validate_environment():
        logger.error("❌ Environment validation failed")
        sys.exit(1)
    
    try:
        # Start all servers
        if manager.start_all_servers():
            logger.info("✅ All MCP servers started successfully")
            
            # Start health monitoring
            manager.start_health_monitoring()
            
            # Print initial status
            manager.print_status()
            
            # Keep running until interrupted
            logger.info("🔄 MCP servers are running. Press Ctrl+C to stop...")
            while True:
                time.sleep(10)
                manager.print_status()
                
        else:
            logger.error("❌ Failed to start all MCP servers")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        manager.shutdown()

if __name__ == "__main__":
    main()
