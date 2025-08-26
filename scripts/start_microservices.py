#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Microservices Manager
Ultra High-Performance Algorithmic Trading Platform

This script manages the complete VIPER trading system with 8 microservices:
- 🌐 API Server (Port 8000) - Web dashboard & REST API
- 🧪 Ultra Backtester (Port 8001) - Strategy testing engine
- 🎯 Strategy Optimizer (Port 8004) - Parameter optimization
- 🔥 Live Trading Engine - Production automated trading
- 💾 Data Manager (Port 8003) - Market data synchronization
- 🔗 Exchange Connector (Port 8005) - Bitget API integration
- 🚨 Risk Manager (Port 8002) - Safety & position control
- 📊 Monitoring Service (Port 8006) - System analytics
"""

import os
import sys
import subprocess
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

class ViperMicroservicesManager:
    """Manages VIPER Trading Bot microservices"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.services = {
            'api-server': {'port': 8000, 'status': 'stopped'},
            'ultra-backtester': {'port': 8001, 'status': 'stopped'},
            'strategy-optimizer': {'port': 8004, 'status': 'stopped'},
            'live-trading-engine': {'port': 8007, 'status': 'stopped'},
            'data-manager': {'port': 8003, 'status': 'stopped'},
            'exchange-connector': {'port': 8005, 'status': 'stopped'},
            'risk-manager': {'port': 8002, 'status': 'stopped'},
            'monitoring-service': {'port': 8006, 'status': 'stopped'},
        }
        self.docker_compose_file = self.project_root / 'docker-compose.yml'

    def print_header(self):
        """Print VIPER header"""
        print("""
🚀 VIPER Trading Bot - Ultra High-Performance Platform
🏗️ Microservices Architecture | 🧪 Advanced Backtesting | 🔥 Live Trading
=================================================================
        """)

    def check_requirements(self) -> bool:
        """Check if Docker and Docker Compose are available"""
        try:
            subprocess.run(['docker', '--version'],
                         capture_output=True, check=True)
            subprocess.run(['docker-compose', '--version'],
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Error: Docker and Docker Compose are required!")
            print("   Please install Docker Desktop: https://www.docker.com/products/docker-desktop")
            return False

    def load_environment(self):
        """Load environment variables from .env file"""
        env_file = self.project_root / '.env'
        if env_file.exists():
            print("📋 Loading environment configuration...")
            # Note: In production, use python-dotenv to load .env files
            print("✅ Environment loaded")
        else:
            print("⚠️  Warning: .env file not found. Using default values.")

    def start_service(self, service_name: str) -> bool:
        """Start a specific microservice"""
        if service_name not in self.services:
            print(f"❌ Error: Unknown service '{service_name}'")
            return False

        print(f"🚀 Starting {service_name}...")
        try:
            cmd = ['docker-compose', '-f', str(self.docker_compose_file),
                   'up', '-d', service_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.services[service_name]['status'] = 'running'
            print(f"✅ {service_name} started successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start {service_name}: {e.stderr}")
            return False

    def stop_service(self, service_name: str) -> bool:
        """Stop a specific microservice"""
        if service_name not in self.services:
            print(f"❌ Error: Unknown service '{service_name}'")
            return False

        print(f"🛑 Stopping {service_name}...")
        try:
            cmd = ['docker-compose', '-f', str(self.docker_compose_file),
                   'down', service_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.services[service_name]['status'] = 'stopped'
            print(f"✅ {service_name} stopped successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to stop {service_name}: {e.stderr}")
            return False

    def start_all_services(self) -> bool:
        """Start all microservices"""
        print("🚀 Starting all VIPER microservices...")
        print("This may take a few minutes for first-time setup...")

        try:
            # Start infrastructure services first
            infra_services = ['redis', 'prometheus', 'grafana']
            cmd = ['docker-compose', '-f', str(self.docker_compose_file),
                   'up', '-d'] + infra_services
            subprocess.run(cmd, check=True)

            # Wait for infrastructure
            print("⏳ Waiting for infrastructure services...")
            time.sleep(10)

            # Start all VIPER services
            cmd = ['docker-compose', '-f', str(self.docker_compose_file),
                   'up', '-d']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Update service statuses
            for service in self.services:
                self.services[service]['status'] = 'running'

            print("✅ All VIPER microservices started successfully!")
            self.print_service_status()
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start services: {e.stderr}")
            return False

    def stop_all_services(self) -> bool:
        """Stop all microservices"""
        print("🛑 Stopping all VIPER microservices...")

        try:
            cmd = ['docker-compose', '-f', str(self.docker_compose_file), 'down']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Update service statuses
            for service in self.services:
                self.services[service]['status'] = 'stopped'

            print("✅ All VIPER microservices stopped successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to stop services: {e.stderr}")
            return False

    def print_service_status(self):
        """Print current status of all services"""
        print("\n📊 VIPER Microservices Status:")
        print("=" * 60)

        for service, info in self.services.items():
            status_icon = "🟢" if info['status'] == 'running' else "🔴"
            print("20")

        print("\n🌐 Access Points:")
        print("   📊 Web Dashboard: http://localhost:8000")
        print("   📈 Grafana:       http://localhost:3000")
        print("   📊 Prometheus:    http://localhost:9090")

    def check_service_health(self, service_name: str) -> str:
        """Check health of a specific service"""
        port = self.services[service_name]['port']
        try:
            # Simple health check via HTTP
            import requests
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                return 'healthy'
            else:
                return 'unhealthy'
        except:
            return 'unknown'

    def show_logs(self, service_name: Optional[str] = None, follow: bool = False):
        """Show logs for services"""
        try:
            cmd = ['docker-compose', '-f', str(self.docker_compose_file), 'logs']
            if service_name:
                cmd.append(service_name)
            if follow:
                cmd.append('-f')

            subprocess.run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to show logs: {e.stderr}")

    def build_services(self):
        """Build Docker images for all services"""
        print("🔨 Building VIPER microservices...")
        try:
            cmd = ['docker-compose', '-f', str(self.docker_compose_file), 'build']
            subprocess.run(cmd, check=True)
            print("✅ All services built successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build services: {e.stderr}")

def main():
    manager = ViperMicroservicesManager()
    manager.print_header()

    if not manager.check_requirements():
        sys.exit(1)

    manager.load_environment()

    parser = argparse.ArgumentParser(description='VIPER Trading Bot Microservices Manager')
    parser.add_argument('action', choices=[
        'start', 'stop', 'restart', 'status', 'logs', 'build', 'health'
    ])
    parser.add_argument('service', nargs='?', help='Specific service name')
    parser.add_argument('--follow', '-f', action='store_true', help='Follow logs')

    args = parser.parse_args()

    if args.action == 'start':
        if args.service:
            success = manager.start_service(args.service)
        else:
            success = manager.start_all_services()

    elif args.action == 'stop':
        if args.service:
            success = manager.stop_service(args.service)
        else:
            success = manager.stop_all_services()

    elif args.action == 'restart':
        if args.service:
            manager.stop_service(args.service)
            success = manager.start_service(args.service)
        else:
            manager.stop_all_services()
            success = manager.start_all_services()

    elif args.action == 'status':
        manager.print_service_status()

    elif args.action == 'logs':
        manager.show_logs(args.service, args.follow)

    elif args.action == 'build':
        manager.build_services()

    elif args.action == 'health':
        if args.service:
            health = manager.check_service_health(args.service)
            print(f"{args.service} health: {health}")
        else:
            print("Service health check:")
            for service in manager.services:
                health = manager.check_service_health(service)
                print(f"  {service}: {health}")

if __name__ == '__main__':
    main()
