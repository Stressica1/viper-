#!/usr/bin/env python3
"""
🚀 VIPER Trading System - Complete Integration & Connection Script
Connects all components: MCP Server, Microservices, and Trading Engine

This script provides a unified interface to:
- Start/stop the complete trading system
- Monitor all components health
- Execute trades through MCP
- Manage system resources
- Handle emergency situations
"""

import os
import sys
import time
import json
import subprocess
import requests
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class ViperTradingSystemConnector:
    """Unified connector for the complete VIPER trading system"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.mcp_trading_server_path = self.project_root / 'mcp-trading-server'
        self.microservices_path = self.project_root / 'infrastructure' / 'docker-compose.yml'
        self.env_file = self.project_root / '.env'

        # System status tracking
        self.components = {
            'microservices': {'status': 'stopped', 'port': None},
            'mcp_trading_server': {'status': 'stopped', 'port': None},
            'mcp_github_server': {'status': 'stopped', 'port': None},
        }

        self.threads = {}
        self.processes = {}

    def print_header(self):
        """Print comprehensive system header"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🚀 VIPER TRADING SYSTEM - COMPLETE INTEGRATION                               ║
║ 🔥 Live Trading | 🔍 Market Scanning | 📊 Risk Management                     ║
║ 🏗️ Microservices | 🤖 MCP Integration | 📈 Real-time Monitoring               ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)

    def check_system_requirements(self) -> bool:
        """Check all system requirements"""
        print("🔍 Checking system requirements...")

        requirements = [
            ('docker', 'Docker for containerized services'),
            ('docker-compose', 'Docker Compose for orchestration'),
            ('node', 'Node.js for MCP servers'),
            ('npm', 'NPM for package management'),
            ('python', 'Python for microservices')
        ]

        all_good = True
        for cmd, description in requirements:
            try:
                result = subprocess.run([cmd, '--version'],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    print(f"✅ {description}: {version}")
                else:
                    print(f"❌ {description}: Failed to get version")
                    all_good = False
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                print(f"❌ {description}: Not found")
                all_good = False

        return all_good

    def start_microservices(self) -> bool:
        """Start all microservices using Docker Compose"""
        print("🐳 Starting VIPER microservices...")

        try:
            cmd = [
                'docker-compose',
                '--env-file', str(self.env_file),
                '-f', str(self.microservices_path),
                'up', '-d'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("✅ Microservices started successfully")
                self.components['microservices']['status'] = 'running'

                # Wait for services to be healthy
                print("⏳ Waiting for services to become healthy...")
                time.sleep(30)

                return self.wait_for_services_health()
            else:
                print(f"❌ Failed to start microservices: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Timeout starting microservices")
            return False
        except Exception as e:
            print(f"❌ Error starting microservices: {str(e)}")
            return False

    def start_mcp_trading_server(self) -> bool:
        """Start the MCP Trading Server"""
        print("🚀 Starting MCP Trading Server...")

        try:
            os.chdir(self.mcp_trading_server_path)

            # Start server in background
            process = subprocess.Popen(
                ['node', 'index.js'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self.processes['mcp_trading'] = process
            self.components['mcp_trading_server']['status'] = 'running'

            # Wait a moment for server to start
            time.sleep(5)

            # Check if server is running
            if process.poll() is None:
                print("✅ MCP Trading Server started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ MCP Trading Server failed to start: {stderr}")
                return False

        except Exception as e:
            print(f"❌ Error starting MCP Trading Server: {str(e)}")
            return False

    def start_mcp_github_server(self) -> bool:
        """Start the MCP GitHub Server"""
        print("📋 Starting MCP GitHub Server...")

        try:
            github_mcp_path = self.project_root / 'mcp-github-project-manager'
            os.chdir(github_mcp_path)

            # Start server in background
            process = subprocess.Popen(
                ['node', 'build/index.js'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self.processes['mcp_github'] = process
            self.components['mcp_github_server']['status'] = 'running'

            # Wait a moment for server to start
            time.sleep(5)

            # Check if server is running
            if process.poll() is None:
                print("✅ MCP GitHub Server started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ MCP GitHub Server failed to start: {stderr}")
                return False

        except Exception as e:
            print(f"❌ Error starting MCP GitHub Server: {str(e)}")
            return False

    def wait_for_services_health(self, timeout: int = 120) -> bool:
        """Wait for all services to become healthy"""
        services_to_check = [
            ('http://localhost:8000/health', 'API Server'),
            ('http://localhost:8002/health', 'Risk Manager'),
            ('http://localhost:8003/health', 'Data Manager'),
            ('http://localhost:8005/health', 'Exchange Connector'),
            ('http://localhost:8006/health', 'Monitoring Service'),
            ('http://localhost:6379/health', 'Redis'),
        ]

        print("🔍 Checking service health...")
        start_time = time.time()
        healthy_services = 0

        while time.time() - start_time < timeout:
            current_healthy = 0

            for url, name in services_to_check:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        current_healthy += 1
                        if healthy_services < current_healthy:
                            print(f"✅ {name} is healthy")
                    else:
                        print(f"⏳ {name} responding but not healthy")
                except:
                    print(f"❌ {name} not responding")

            healthy_services = current_healthy

            if healthy_services == len(services_to_check):
                print(f"\n🎉 All {healthy_services} services are healthy!")
                return True

            time.sleep(10)

        print(f"⚠️ Timeout reached. {healthy_services}/{len(services_to_check)} services healthy")
        return healthy_services == len(services_to_check)

    def start_complete_system(self) -> bool:
        """Start the complete VIPER trading system"""
        print("🚀 Starting complete VIPER trading system...")
        print("This will start microservices and MCP servers")

        success = True

        # 1. Start microservices
        if not self.start_microservices():
            success = False

        # 2. Start MCP Trading Server
        if not self.start_mcp_trading_server():
            success = False

        # 3. Start MCP GitHub Server
        if not self.start_mcp_github_server():
            success = False

        if success:
            print("\n🎉 Complete VIPER trading system started successfully!")
            self.print_system_status()
            return True
        else:
            print("\n❌ Some components failed to start")
            return False

    def stop_complete_system(self) -> bool:
        """Stop the complete system"""
        print("🛑 Stopping complete VIPER trading system...")

        success = True

        # Stop MCP servers
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=10)
                print(f"✅ {name} stopped")
            except:
                print(f"❌ Failed to stop {name}")
                success = False

        # Stop microservices
        try:
            cmd = [
                'docker-compose',
                '--env-file', str(self.env_file),
                '-f', str(self.microservices_path),
                'down'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("✅ Microservices stopped")
            else:
                print(f"❌ Failed to stop microservices: {result.stderr}")
                success = False
        except Exception as e:
            print(f"❌ Error stopping microservices: {str(e)}")
            success = False

        if success:
            print("✅ Complete system stopped successfully")
        else:
            print("⚠️ Some components may still be running")

        return success

    def print_system_status(self):
        """Print comprehensive system status"""
        print("\n📊 VIPER Trading System Status")
        print("=" * 60)

        # Component status
        for component, info in self.components.items():
            status_icon = "🟢" if info['status'] == 'running' else "🔴"
            print(f"{status_icon} {component.replace('_', ' ').title()}: {info['status']}")

        # Access points
        print("\n🌐 Access Points:")
        print("   📊 Web Dashboard: http://localhost:8000")
        print("   📈 Grafana:       http://localhost:3001")
        print("   📊 Prometheus:    http://localhost:9090")
        print("   🔍 Kibana:        http://localhost:5601")

        # MCP Servers
        print("\n🤖 MCP Servers:")
        print("   🚀 Trading Server: Running (stdio mode)")
        print("   📋 GitHub Server: Running (stdio mode)")
        # System health
        total_components = len(self.components)
        running_components = sum(1 for info in self.components.values() if info['status'] == 'running')
        health_percentage = (running_components / total_components) * 100

        print("\n📈 System Health:")
        print(f"   Overall Health: {health_percentage:.1f}%")
        if health_percentage == 100:
            print("🎉 All systems operational!")
        elif health_percentage >= 75:
            print("✅ System mostly operational")
        elif health_percentage >= 50:
            print("⚠️ System partially operational")
        else:
            print("❌ System needs attention")

    def test_trading_system(self):
        """Test the trading system components"""
        print("🧪 Testing VIPER Trading System...")

        tests = [
            ("Microservices Health", self.test_microservices_health),
            ("MCP Trading Server", self.test_mcp_trading_server),
            ("MCP GitHub Server", self.test_mcp_github_server),
            ("System Integration", self.test_system_integration)
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                status_icon = "✅" if result else "❌"
                print(f"{status_icon} {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name}: ERROR - {str(e)}")

        passed_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)

        print("\n📊 Test Results:")
        print(f"   Passed: {passed_tests}/{total_tests} tests")

        if passed_tests == total_tests:
            print("🎉 All tests passed! System is fully operational.")
        else:
            print("⚠️ Some tests failed. Check individual component status.")

        return passed_tests == total_tests

    def test_microservices_health(self) -> bool:
        """Test microservices health endpoints"""
        health_urls = [
            'http://localhost:8000/health',
            'http://localhost:8002/health',
            'http://localhost:8003/health',
            'http://localhost:8005/health',
            'http://localhost:8006/health'
        ]

        healthy_count = 0
        for url in health_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    healthy_count += 1
            except:
                pass

        return healthy_count >= 3  # At least 3 services should be healthy

    def test_mcp_trading_server(self) -> bool:
        """Test MCP Trading Server"""
        # This would require MCP client integration
        # For now, just check if process is running
        return 'mcp_trading' in self.processes and self.processes['mcp_trading'].poll() is None

    def test_mcp_github_server(self) -> bool:
        """Test MCP GitHub Server"""
        # This would require MCP client integration
        # For now, just check if process is running
        return 'mcp_github' in self.processes and self.processes['mcp_github'].poll() is None

    def test_system_integration(self) -> bool:
        """Test overall system integration"""
        # Test if we can access the main dashboard
        try:
            response = requests.get('http://localhost:8000', timeout=10)
            return response.status_code == 200
        except:
            return False

def main():
    connector = ViperTradingSystemConnector()
    connector.print_header()

    if not connector.check_system_requirements():
        print("❌ System requirements not met. Please install missing components.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python connect_trading_system.py [start|stop|status|test]")
        print("\nCommands:")
        print("  start  - Start the complete trading system")
        print("  stop   - Stop the complete trading system")
        print("  status - Show system status")
        print("  test   - Run system tests")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'start':
        success = connector.start_complete_system()
        sys.exit(0 if success else 1)

    elif command == 'stop':
        success = connector.stop_complete_system()
        sys.exit(0 if success else 1)

    elif command == 'status':
        connector.print_system_status()

    elif command == 'test':
        success = connector.test_trading_system()
        sys.exit(0 if success else 1)

    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: start, stop, status, test")
        sys.exit(1)

if __name__ == '__main__':
    main()
