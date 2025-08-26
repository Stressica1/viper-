#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Optimized Microservices Manager
Ultra High-Performance Algorithmic Trading Platform

This script manages the complete VIPER trading system with optimized performance:
- 🌐 API Server (Port 8000) - Web dashboard & REST API
- 🧪 Ultra Backtester (Port 8001) - Strategy testing engine
- 🎯 Strategy Optimizer (Port 8004) - Parameter optimization
- 🔥 Live Trading Engine - Production automated trading
- 💾 Data Manager (Port 8003) - Market data synchronization
- 🔗 Exchange Connector (Port 8005) - Bitget API integration
- 🚨 Risk Manager (Port 8002) - Safety & position control
- 📊 Monitoring Service (Port 8006) - System analytics

Optimizations:
- ⚡ Parallel service startup with dependency management
- 🔄 Smart health checks with retry logic
- 📊 Real-time progress tracking
- 🛡️ Robust error handling and recovery
- 🚀 Async operations for better performance
"""

import os
import sys
import subprocess
import time
import argparse
import json
import asyncio
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager

@dataclass
class ServiceConfig:
    """Service configuration with health check endpoint"""
    name: str
    port: int
    health_endpoint: str = "/health"
    dependencies: List[str] = None
    startup_timeout: int = 60
    health_check_retries: int = 5

class ViperMicroservicesManager:
    """Optimized VIPER Trading Bot microservices manager"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docker_compose_file = self.project_root / 'infrastructure' / 'docker-compose.yml'
        self.env_file = self.project_root / '.env'

        # Service configurations with dependencies
        self.services: Dict[str, ServiceConfig] = {
            'redis': ServiceConfig('redis', 6379, '/health', [], 30),
            'credential-vault': ServiceConfig('credential-vault', 8008, '/health', ['redis'], 45),
            'exchange-connector': ServiceConfig('exchange-connector', 8005, '/health', ['redis', 'credential-vault'], 60),
            'risk-manager': ServiceConfig('risk-manager', 8002, '/health', ['redis', 'exchange-connector'], 45),
            'data-manager': ServiceConfig('data-manager', 8003, '/health', ['redis'], 45),
            'monitoring-service': ServiceConfig('monitoring-service', 8006, '/health', ['redis'], 45),
            'api-server': ServiceConfig('api-server', 8000, '/health', ['redis', 'data-manager'], 60),
            'ultra-backtester': ServiceConfig('ultra-backtester', 8001, '/health', ['redis', 'data-manager', 'risk-manager'], 90),
            'strategy-optimizer': ServiceConfig('strategy-optimizer', 8004, '/health', ['redis', 'ultra-backtester'], 60),
            'live-trading-engine': ServiceConfig('live-trading-engine', 8007, '/health', ['redis', 'exchange-connector', 'risk-manager', 'credential-vault'], 60),
        }

        # Service status tracking
        self.service_status: Dict[str, str] = {name: 'stopped' for name in self.services}
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="viper-service")
        self._session = None

    def __del__(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self._session and not self._session.closed:
            asyncio.run(self._session.close())

    def print_header(self):
        """Print VIPER header with optimization info"""
        print("""
🚀 VIPER Trading Bot - Optimized Microservices Platform
🏗️ Microservices Architecture | 🧪 Advanced Backtesting | 🔥 Live Trading
⚡ Parallel Processing | 🔄 Smart Health Checks | 📊 Real-time Monitoring
=================================================================
        """)

    def check_requirements(self) -> bool:
        """Check if Docker and Docker Compose are available"""
        print("🔍 Checking system requirements...")

        requirements = [
            ('docker', ['docker', '--version']),
            ('docker-compose', ['docker-compose', '--version']),
        ]

        for name, cmd in requirements:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✅ {name}: {result.stdout.strip()}")
                else:
                    print(f"❌ {name}: Failed to get version")
                    return False
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"❌ {name}: Not available - {e}")
                return False

        return True

    def load_environment(self):
        """Load and validate environment variables"""
        print("📋 Loading environment configuration...")

        if not self.env_file.exists():
            print("❌ Error: .env file not found!")
            print(f"   Expected location: {self.env_file}")
            return False

        # Validate critical environment variables
        critical_vars = [
            'BITGET_API_KEY',
            'BITGET_API_SECRET',
            'VAULT_MASTER_KEY',
            'GRAFANA_ADMIN_PASSWORD'
        ]

        missing_vars = []
        try:
            # Handle encoding issues on Windows
            with open(self.env_file, 'r', encoding='utf-8', errors='replace') as f:
                env_content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding if UTF-8 fails
            with open(self.env_file, 'r', encoding='latin-1') as f:
                env_content = f.read()

        for var in critical_vars:
            if var not in env_content or f"{var}=" in env_content.split('\n'):
                # Check if variable is set (not just present)
                if not os.getenv(var):
                    missing_vars.append(var)

        if missing_vars:
            print("⚠️  Warning: Critical environment variables may be missing:")
            for var in missing_vars:
                print(f"   - {var}")
            print("   Some services may not function correctly.")

        print("✅ Environment configuration loaded")
        return True

    def _run_docker_command(self, cmd: List[str], service_name: str = "service") -> Tuple[bool, str]:
        """Run docker command with proper error handling"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=True
            )
            return True, result.stdout
        except subprocess.TimeoutExpired:
            return False, f"Timeout starting {service_name}"
        except subprocess.CalledProcessError as e:
            return False, f"Command failed: {e.stderr}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def _check_dependencies(self, service_name: str) -> bool:
        """Check if all dependencies for a service are healthy"""
        service_config = self.services[service_name]
        if not service_config.dependencies:
            return True

        unhealthy_deps = []
        for dep in service_config.dependencies:
            if self.service_status.get(dep) != 'running':
                unhealthy_deps.append(dep)

        if unhealthy_deps:
            print(f"⚠️  Dependencies not ready for {service_name}: {', '.join(unhealthy_deps)}")
            return False

        return True

    async def _check_service_health_async(self, service_name: str) -> str:
        """Async health check for a service"""
        service_config = self.services[service_name]
        url = f"http://localhost:{service_config.port}{service_config.health_endpoint}"

        for attempt in range(service_config.health_check_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with self._session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        return 'healthy'
                    else:
                        await asyncio.sleep(1)
            except Exception:
                if attempt < service_config.health_check_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return 'unhealthy'

        return 'unknown'

    @asynccontextmanager
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        try:
            yield self._session
        finally:
            pass  # Don't close here, let __del__ handle it

    def start_service(self, service_name: str) -> bool:
        """Start a specific microservice with dependency checking"""
        if service_name not in self.services:
            print(f"❌ Error: Unknown service '{service_name}'")
            return False

        if not self._check_dependencies(service_name):
            print(f"❌ Cannot start {service_name} - dependencies not satisfied")
            return False

        print(f"🚀 Starting {service_name}...")
        cmd = [
            'docker-compose',
            '--env-file', str(self.env_file),
            '-f', str(self.docker_compose_file),
            'up', '-d', service_name
        ]

        success, output = self._run_docker_command(cmd, service_name)
        if success:
            self.service_status[service_name] = 'starting'
            print(f"✅ {service_name} startup initiated")
            return True
        else:
            print(f"❌ Failed to start {service_name}: {output}")
            return False

    def stop_service(self, service_name: str) -> bool:
        """Stop a specific microservice"""
        if service_name not in self.services:
            print(f"❌ Error: Unknown service '{service_name}'")
            return False

        print(f"🛑 Stopping {service_name}...")
        cmd = [
            'docker-compose',
            '--env-file', str(self.env_file),
            '-f', str(self.docker_compose_file),
            'stop', service_name
        ]

        success, output = self._run_docker_command(cmd, service_name)
        if success:
            self.service_status[service_name] = 'stopped'
            print(f"✅ {service_name} stopped successfully")
            return True
        else:
            print(f"❌ Failed to stop {service_name}: {output}")
            return False

    def _get_service_startup_order(self) -> List[List[str]]:
        """Get services organized by startup priority levels"""
        # Level 0: Infrastructure services
        level0 = ['redis', 'prometheus', 'grafana']

        # Level 1: Core services with no dependencies beyond infrastructure
        level1 = ['credential-vault', 'data-manager', 'monitoring-service']

        # Level 2: Services that depend on level 1
        level2 = ['exchange-connector', 'risk-manager']

        # Level 3: Services that depend on level 2
        level3 = ['api-server', 'live-trading-engine']

        # Level 4: Services that depend on level 3
        level4 = ['ultra-backtester', 'strategy-optimizer']

        return [level0, level1, level2, level3, level4]

    def _start_service_batch(self, services: List[str]) -> List[Tuple[str, bool]]:
        """Start a batch of services in parallel"""
        results = []

        with self.executor as executor:
            futures = {
                executor.submit(self.start_service, service): service
                for service in services
            }

            for future in as_completed(futures):
                service = futures[future]
                try:
                    success = future.result()
                    results.append((service, success))
                except Exception as e:
                    print(f"❌ Exception starting {service}: {e}")
                    results.append((service, False))

        return results

    async def _wait_for_services_health(self, services: List[str], timeout: int = 120) -> bool:
        """Wait for services to become healthy with progress tracking"""
        print(f"⏳ Waiting for {len(services)} services to become healthy...")

        start_time = time.time()
        healthy_count = 0

        while time.time() - start_time < timeout:
            async with self._get_session():
                tasks = [self._check_service_health_async(service) for service in services]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            current_healthy = sum(1 for result in results
                                if isinstance(result, str) and result == 'healthy')

            if current_healthy > healthy_count:
                healthy_count = current_healthy
                print(f"📊 Health check progress: {healthy_count}/{len(services)} services healthy")

            if current_healthy == len(services):
                print("✅ All services are healthy!")
                return True

            await asyncio.sleep(5)

        # Show which services are still not healthy
        unhealthy = []
        for i, service in enumerate(services):
            if not isinstance(results[i], str) or results[i] != 'healthy':
                unhealthy.append(service)

        print(f"⚠️  Timeout reached. Unhealthy services: {', '.join(unhealthy)}")
        return False

    def start_all_services(self) -> bool:
        """Start all microservices with optimized parallel processing"""
        print("🚀 Starting all VIPER microservices with optimized orchestration...")
        print("⚡ Parallel processing | 🔄 Smart dependency management | 📊 Real-time monitoring")

        startup_levels = self._get_service_startup_order()
        total_services = sum(len(level) for level in startup_levels)
        started_count = 0

        for level_num, level_services in enumerate(startup_levels):
            if not level_services:
                continue

            print(f"\n📦 Level {level_num + 1}: Starting {len(level_services)} services...")
            print(f"   Services: {', '.join(level_services)}")

            # Start services in this level in parallel
            results = self._start_service_batch(level_services)

            # Check results
            successful = [service for service, success in results if success]
            failed = [service for service, success in results if not success]

            if failed:
                print(f"❌ Failed to start: {', '.join(failed)}")
                return False

            started_count += len(successful)

            # Wait for services in this level to become healthy
            if level_services:
                asyncio.run(self._wait_for_services_health(level_services))

            print(f"✅ Level {level_num + 1} complete ({started_count}/{total_services} total)")

        print("\n🎉 All VIPER microservices started successfully!")
        self.print_service_status()
        return True

    def stop_all_services(self) -> bool:
        """Stop all microservices with parallel processing"""
        print("🛑 Stopping all VIPER microservices...")

        # Stop services in reverse order
        all_services = [service for level in reversed(self._get_service_startup_order())
                       for service in level if service in self.services]

        results = self._start_service_batch([f"stop_{service}" for service in all_services])

        successful = sum(1 for _, success in results if success)
        total = len(results)

        if successful == total:
            print("✅ All VIPER microservices stopped successfully!")
            return True
        else:
            failed = [service.replace("stop_", "") for service, success in results if not success]
            print(f"❌ Failed to stop: {', '.join(failed)}")
            return False

    def print_service_status(self):
        """Print current status of all services with enhanced formatting"""
        print("\n📊 VIPER Microservices Status:")
        print("=" * 80)

        # Group services by status
        running = []
        starting = []
        stopped = []

        for service_name, config in self.services.items():
            status = self.service_status[service_name]
            service_info = f"{service_name:<25} Port: {config.port:>4}"

            if status == 'running':
                running.append(service_info)
            elif status == 'starting':
                starting.append(service_info)
            else:
                stopped.append(service_info)

        # Print services by status
        if running:
            print("🟢 RUNNING SERVICES:")
            for service in running:
                print(f"   {service} | Status: running")
            print()

        if starting:
            print("🟡 STARTING SERVICES:")
            for service in starting:
                print(f"   {service} | Status: starting")
            print()

        if stopped:
            print("🔴 STOPPED SERVICES:")
            for service in stopped:
                print(f"   {service} | Status: stopped")
            print()

        # Access points
        print("🌐 ACCESS POINTS:")
        print("   📊 Web Dashboard: http://localhost:8000")
        print("   📈 Grafana:       http://localhost:3001")
        print("   📊 Prometheus:    http://localhost:9090")
        print("   🔍 Kibana:        http://localhost:5601")
        # Service health summary
        total = len(self.services)
        running_count = len(running)
        healthy_percent = (running_count / total) * 100 if total > 0 else 0

        print("\n📈 SYSTEM HEALTH:")
        print(f"   Overall Health: {healthy_percent:.1f}%")
    def check_service_health(self, service_name: str) -> str:
        """Check health of a specific service with improved error handling"""
        if service_name not in self.services:
            return 'unknown'

        try:
            # Use asyncio to run the async health check
            return asyncio.run(self._check_service_health_async(service_name))
        except Exception as e:
            print(f"❌ Error checking health for {service_name}: {e}")
            return 'unknown'

    def show_logs(self, service_name: Optional[str] = None, follow: bool = False, tail: int = 100):
        """Show logs for services with enhanced options"""
        try:
            cmd = [
                'docker-compose',
                '--env-file', str(self.env_file),
                '-f', str(self.docker_compose_file),
                'logs'
            ]

            if tail and not follow:
                cmd.extend(['--tail', str(tail)])

            if service_name:
                cmd.append(service_name)

            if follow:
                cmd.append('-f')
                print(f"📋 Showing live logs for {service_name or 'all services'} (Ctrl+C to stop)...")
            else:
                print(f"📋 Showing last {tail} lines of logs for {service_name or 'all services'}...")

            subprocess.run(cmd)

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to show logs: {e.stderr}")
        except KeyboardInterrupt:
            print("\n📋 Log viewing stopped by user")

    def build_services(self, parallel: bool = True):
        """Build Docker images for all services with parallel processing"""
        print("🔨 Building VIPER microservices...")

        if parallel:
            print("⚡ Building services in parallel for faster completion...")

        try:
            cmd = [
                'docker-compose',
                '--env-file', str(self.env_file),
                '-f', str(self.docker_compose_file),
                'build'
            ]

            if parallel:
                cmd.append('--parallel')

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ All services built successfully!")

            # Show build summary
            if result.stdout:
                print("\n📋 Build Summary:")
                # Extract service names from build output
                services_built = []
                for line in result.stdout.split('\n'):
                    if 'Building' in line and 'done' in line:
                        service = line.split()[1] if len(line.split()) > 1 else 'unknown'
                        services_built.append(service)

                if services_built:
                    print(f"   Services built: {', '.join(services_built)}")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build services: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during build: {str(e)}")
            return False

def main():
    """Main entry point with enhanced error handling"""
    try:
        manager = ViperMicroservicesManager()
        manager.print_header()

        if not manager.check_requirements():
            print("💡 Tip: Install Docker Desktop from https://www.docker.com/products/docker-desktop")
            sys.exit(1)

        if not manager.load_environment():
            print("❌ Environment configuration failed. Please check your .env file.")
            sys.exit(1)

        parser = argparse.ArgumentParser(
            description='🚀 VIPER Trading Bot - Optimized Microservices Manager',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python start_microservices.py start              # Start all services
  python start_microservices.py start redis        # Start specific service
  python start_microservices.py status             # Show service status
  python start_microservices.py logs api-server    # Show logs for api-server
  python start_microservices.py logs --follow      # Follow all logs
  python start_microservices.py health             # Check all services health
  python start_microservices.py build              # Build all services
  python start_microservices.py stop               # Stop all services
            """
        )

        parser.add_argument(
            'action',
            choices=['start', 'stop', 'restart', 'status', 'logs', 'build', 'health'],
            help='Action to perform'
        )
        parser.add_argument(
            'service',
            nargs='?',
            help='Specific service name (optional for some actions)'
        )
        parser.add_argument(
            '--follow', '-f',
            action='store_true',
            help='Follow logs in real-time'
        )
        parser.add_argument(
            '--tail', '-t',
            type=int,
            default=100,
            help='Number of log lines to show (default: 100)'
        )
        parser.add_argument(
            '--parallel',
            action='store_true',
            default=True,
            help='Use parallel processing for builds (default: enabled)'
        )

        args = parser.parse_args()

        # Execute the requested action
        action_map = {
            'start': lambda: handle_start_action(manager, args),
            'stop': lambda: handle_stop_action(manager, args),
            'restart': lambda: handle_restart_action(manager, args),
            'status': lambda: manager.print_service_status(),
            'logs': lambda: manager.show_logs(args.service, args.follow, args.tail),
            'build': lambda: manager.build_services(args.parallel),
            'health': lambda: handle_health_action(manager, args)
        }

        success = action_map[args.action]()

        # Exit with appropriate code
        if args.action in ['start', 'stop', 'restart', 'build']:
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("📋 Please check the logs and try again")
        sys.exit(1)

def handle_start_action(manager: ViperMicroservicesManager, args) -> bool:
    """Handle start action with proper validation"""
    if args.service:
        if args.service not in manager.services:
            print(f"❌ Unknown service: {args.service}")
            print(f"   Available services: {', '.join(manager.services.keys())}")
            return False
        return manager.start_service(args.service)
    else:
        return manager.start_all_services()

def handle_stop_action(manager: ViperMicroservicesManager, args) -> bool:
    """Handle stop action with proper validation"""
    if args.service:
        if args.service not in manager.services:
            print(f"❌ Unknown service: {args.service}")
            print(f"   Available services: {', '.join(manager.services.keys())}")
            return False
        return manager.stop_service(args.service)
    else:
        return manager.stop_all_services()

def handle_restart_action(manager: ViperMicroservicesManager, args) -> bool:
    """Handle restart action with proper validation"""
    if args.service:
        if args.service not in manager.services:
            print(f"❌ Unknown service: {args.service}")
            print(f"   Available services: {', '.join(manager.services.keys())}")
            return False
        print(f"🔄 Restarting {args.service}...")
        manager.stop_service(args.service)
        return manager.start_service(args.service)
    else:
        print("🔄 Restarting all VIPER microservices...")
        manager.stop_all_services()
        return manager.start_all_services()

def handle_health_action(manager: ViperMicroservicesManager, args) -> None:
    """Handle health check action with detailed reporting"""
    if args.service:
        if args.service not in manager.services:
            print(f"❌ Unknown service: {args.service}")
            print(f"   Available services: {', '.join(manager.services.keys())}")
            return

        health = manager.check_service_health(args.service)
        status_icon = {
            'healthy': '🟢',
            'unhealthy': '🔴',
            'unknown': '🟡'
        }.get(health, '❓')

        print(f"{status_icon} {args.service}: {health}")
    else:
        print("🔍 Checking health of all services...")
        print("=" * 50)

        health_results = {}
        for service_name in manager.services:
            health = manager.check_service_health(service_name)
            health_results[service_name] = health

        # Group by health status
        healthy = [s for s, h in health_results.items() if h == 'healthy']
        unhealthy = [s for s, h in health_results.items() if h == 'unhealthy']
        unknown = [s for s, h in health_results.items() if h == 'unknown']

        if healthy:
            print("🟢 HEALTHY SERVICES:")
            for service in healthy:
                print(f"   ✅ {service}")
            print()

        if unhealthy:
            print("🔴 UNHEALTHY SERVICES:")
            for service in unhealthy:
                print(f"   ❌ {service}")
            print()

        if unknown:
            print("🟡 UNKNOWN STATUS:")
            for service in unknown:
                print(f"   ❓ {service}")
            print()

        total = len(health_results)
        healthy_count = len(healthy)
        print(f"📊 Health Summary: {healthy_count}/{total} services healthy ({healthy_count/total*100:.1f}%)")

if __name__ == '__main__':
    main()
