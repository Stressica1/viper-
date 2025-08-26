#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Main Entry Point
Ultra High-Performance Algorithmic Trading Platform

🎯 ONE COMMAND STARTS EVERYTHING:
    python main.py

This script orchestrates the complete VIPER trading system startup:
- ✅ 17 Microservices Architecture
- ✅ Centralized Logging System (ELK Stack)
- ✅ MCP AI Integration
- ✅ Real-time Monitoring & Alerting
- ✅ Web Dashboard & API
- ✅ Ultra Backtester & Strategy Optimizer
- ✅ Live Trading Engine (with proper API keys)
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional

class ViperSystemOrchestrator:
    """Main orchestrator for the complete VIPER trading system"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.start_microservices_script = self.project_root / "scripts" / "start_microservices.py"
        self.env_file = self.project_root / ".env"
        self.docker_compose_file = self.project_root / "infrastructure" / "docker-compose.yml"

    def print_banner(self):
        """Print the VIPER startup banner"""
        print("""
🚀 VIPER Trading Bot - Ultra High-Performance Algorithmic Trading Platform
================================================================================
🏆 World-Class Algorithmic Trading System - ONE COMMAND STARTS EVERYTHING!

✅ COMPONENTS STARTING AUTOMATICALLY:
   • 🧪 Ultra Badass Backtester - Strategy testing with predictive ranges
   • 🔥 Live Trading Engine - High-performance automated trading
   • 📊 Professional Analytics - Advanced performance metrics & risk management
   • 🌐 Web Dashboard - Real-time monitoring and control interface
   • 🏗️ 17-Microservices Architecture - Scalable, production-ready system
   • 🤖 MCP Integration - Full Model Context Protocol support for AI agents
   • 📡 Real-time Data Streaming - Live market data with sub-second latency
   • 🚨 Advanced Risk Management - Multi-layered position control & safety
   • 📝 Centralized Logging - ELK stack with comprehensive audit trails
   • 🔐 Secure Credential Management - Vault-based secrets with access tokens

🎯 ACCESS POINTS (after startup):
   • 🌐 Web Dashboard: http://localhost:8000
   • 📊 Grafana Monitoring: http://localhost:3000
   • 📥 Kibana Logs: http://localhost:5601
   • 🤖 MCP Server: http://localhost:8015
================================================================================
        """)

    def check_requirements(self) -> bool:
        """Check if all requirements are met for startup"""
        print("🔍 Checking system requirements...")

        # Check if Docker is available
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✅ Docker: {result.stdout.strip()}")
            else:
                print("❌ Docker: Not available or not running")
                return False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Docker: Not installed or not accessible")
            print("   💡 Install Docker Desktop: https://www.docker.com/products/docker-desktop")
            return False

        # Check if Docker Compose is available
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✅ Docker Compose: {result.stdout.strip()}")
            else:
                print("❌ Docker Compose: Not available")
                return False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Docker Compose: Not installed")
            return False

        # Check if .env file exists
        if not self.env_file.exists():
            print("❌ .env file not found!")
            print(f"   Expected location: {self.env_file}")
            print("   💡 Copy .env.example to .env and configure your API keys")
            return False

        # Check if start_microservices.py exists
        if not self.start_microservices_script.exists():
            print("❌ start_microservices.py script not found!")
            print(f"   Expected location: {self.start_microservices_script}")
            return False

        print("✅ All requirements met!")
        return True

    def validate_environment(self) -> Dict[str, str]:
        """Validate environment configuration and return status"""
        print("📋 Validating environment configuration...")

        validation_results = {}

        # Check critical environment variables
        critical_vars = [
            'BITGET_API_KEY',
            'BITGET_API_SECRET',
            'VAULT_MASTER_KEY',
            'GRAFANA_ADMIN_PASSWORD'
        ]

        try:
            with open(self.env_file, 'r', encoding='utf-8', errors='replace') as f:
                env_content = f.read()
        except UnicodeDecodeError:
            try:
                with open(self.env_file, 'r', encoding='latin-1') as f:
                    env_content = f.read()
            except Exception as e:
                validation_results['env_file'] = f"❌ Error reading .env file: {e}"
                return validation_results

        missing_vars = []
        for var in critical_vars:
            if var not in env_content or f"{var}=" in env_content.split('\n'):
                if not os.getenv(var):
                    missing_vars.append(var)

        if missing_vars:
            validation_results['missing_vars'] = f"⚠️  Missing/empty critical variables: {', '.join(missing_vars)}"
            validation_results['live_trading'] = "❌ Live trading will not work without proper API keys"
        else:
            validation_results['api_keys'] = "✅ API keys configured (live trading ready)"

        # Check service ports
        service_ports = {
            'API_SERVER_PORT': '8000',
            'ULTRA_BACKTESTER_PORT': '8001',
            'RISK_MANAGER_PORT': '8002',
            'DATA_MANAGER_PORT': '8003',
            'STRATEGY_OPTIMIZER_PORT': '8004',
            'EXCHANGE_CONNECTOR_PORT': '8005'
        }

        ports_configured = []
        for port_var, default_port in service_ports.items():
            if port_var in env_content:
                ports_configured.append(f"{port_var} configured")
            else:
                ports_configured.append(f"{port_var} using default ({default_port})")

        validation_results['ports'] = f"✅ Service ports: {', '.join(ports_configured)}"

        return validation_results

    def start_system(self, build_first: bool = False) -> bool:
        """Start the complete VIPER system"""
        print("🚀 Starting VIPER Trading System...")

        try:
            # Change to project root directory
            os.chdir(self.project_root)

            # Build services if requested
            if build_first:
                print("🔨 Building all services first...")
                cmd = [
                    sys.executable, str(self.start_microservices_script),
                    "build"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ Build failed: {result.stderr}")
                    return False
                print("✅ All services built successfully!")

            # Start all microservices
            print("🏗️ Starting all microservices...")
            cmd = [
                sys.executable, str(self.start_microservices_script),
                "start"
            ]

            print("⚡ This will start:")
            print("   • 17 microservices with dependency management")
            print("   • Redis, Prometheus, Grafana monitoring stack")
            print("   • ELK logging stack (Elasticsearch, Logstash, Kibana)")
            print("   • MCP AI integration server")
            print("   • All trading components (backtester, live engine, etc.)")
            print()

            # Run the startup command
            result = subprocess.run(cmd)

            if result.returncode == 0:
                self.print_success_message()
                return True
            else:
                print("❌ Failed to start VIPER system")
                print("💡 Check the logs above for specific error details")
                return False

        except Exception as e:
            print(f"❌ Error starting system: {str(e)}")
            return False

    def print_success_message(self):
        """Print success message with access information"""
        print("""
🎉 VIPER TRADING SYSTEM STARTED SUCCESSFULLY!
================================================

🌐 ACCESS YOUR TRADING PLATFORM:
   • 🏠 Web Dashboard:    http://localhost:8000
   • 📊 Grafana Monitoring: http://localhost:3000
   • 📥 Kibana Logs:      http://localhost:5601
   • 🤖 MCP Server:       http://localhost:8015

📊 SYSTEM STATUS:
   • ✅ 17 Microservices running
   • ✅ Centralized logging active
   • ✅ Real-time monitoring active
   • ✅ MCP AI integration ready
   • ✅ All trading engines ready

🚀 QUICK START GUIDE:
   1. Open http://localhost:8000 in your browser
   2. Check system status and health
   3. Run a backtest to test strategies
   4. Configure live trading (if API keys are set)
   5. Monitor performance in real-time

🛠️ MANAGEMENT COMMANDS:
   • Check status:    python scripts/start_microservices.py status
   • View logs:       python scripts/start_microservices.py logs
   • Stop system:     python scripts/start_microservices.py stop
   • Health check:    python scripts/start_microservices.py health

⚠️  IMPORTANT NOTES:
   • Live trading requires valid Bitget API keys in .env
   • Monitor your positions and risk management closely
   • Backtest thoroughly before live trading
   • This software is for educational/research purposes

🚀 HAPPY TRADING WITH VIPER!
================================================
        """)

    def print_help(self):
        """Print help information"""
        print("""
🚀 VIPER Trading Bot - Help
===========================

USAGE:
    python main.py [OPTIONS]

OPTIONS:
    --build          Build all services before starting
    --status         Show current system status
    --stop           Stop all running services
    --help           Show this help message

EXAMPLES:
    python main.py                    # Start everything
    python main.py --build           # Build and start everything
    python main.py --status          # Show system status
    python main.py --stop            # Stop all services

ACCESS POINTS:
    • Web Dashboard: http://localhost:8000
    • Grafana:       http://localhost:3000
    • Kibana:        http://localhost:5601
    • MCP Server:    http://localhost:8015

For more detailed control:
    python scripts/start_microservices.py [command]
        """)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='🚀 VIPER Trading Bot - One Command Starts Everything',
        add_help=False
    )

    parser.add_argument('--build', action='store_true',
                       help='Build all services before starting')
    parser.add_argument('--status', action='store_true',
                       help='Show current system status')
    parser.add_argument('--stop', action='store_true',
                       help='Stop all running services')
    parser.add_argument('--help', action='store_true',
                       help='Show help message')

    args = parser.parse_args()

    orchestrator = ViperSystemOrchestrator()

    # Show help if requested
    if args.help:
        orchestrator.print_help()
        return

    orchestrator.print_banner()

    # Handle status check
    if args.status:
        print("📊 Checking VIPER system status...")
        if orchestrator.check_requirements():
            validation = orchestrator.validate_environment()
            for key, message in validation.items():
                print(f"   {message}")
            print("\n💡 To start the system: python main.py")
        return

    # Handle stop command
    if args.stop:
        print("🛑 Stopping VIPER system...")
        try:
            os.chdir(orchestrator.project_root)
            cmd = [sys.executable, str(orchestrator.start_microservices_script), "stop"]
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("✅ VIPER system stopped successfully!")
            else:
                print("❌ Failed to stop system properly")
        except Exception as e:
            print(f"❌ Error stopping system: {str(e)}")
        return

    # Check requirements before starting
    if not orchestrator.check_requirements():
        print("\n❌ Cannot start VIPER system due to missing requirements.")
        print("💡 Please fix the issues above and try again.")
        return

    # Validate environment
    validation = orchestrator.validate_environment()
    for key, message in validation.items():
        print(f"   {message}")

    # Confirm startup
    print("\n🚀 Ready to start VIPER Trading System!")
    response = input("Do you want to continue? (y/N): ").strip().lower()

    if response not in ['y', 'yes']:
        print("🛑 Startup cancelled by user.")
        return

    # Start the system
    success = orchestrator.start_system(build_first=args.build)

    if not success:
        print("\n❌ VIPER system failed to start properly.")
        print("💡 Check the logs above for specific error details.")
        print("💡 Try: python scripts/start_microservices.py status")
        sys.exit(1)

if __name__ == "__main__":
    main()
