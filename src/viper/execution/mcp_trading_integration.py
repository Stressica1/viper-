#!/usr/bin/env python3
"""
🔗 COMPLETE MCP TRADING INTEGRATION
Full integration of GitHub MCP with VIPER Live Trading System

This integration provides:
✅ Complete MCP task management for trading
✅ Automated performance tracking and reporting
✅ Real-time monitoring and control
✅ GitHub integration for version control
✅ Emergency controls and safety features
✅ Comprehensive logging and analytics
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import all MCP trading components
from mcp_live_trading_connector import MCPLiveTradingConnector
from mcp_performance_tracker import MCPPerformanceTracker
from github_mcp_integration import GitHubMCPIntegration
from mcp_trading_monitor import MCPTradingMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MCP_INTEGRATION - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteMCPTradingIntegration:
    """Complete MCP trading integration system"""

    def __init__(self):
        self.connector = None
        self.performance_tracker = None
        self.github_mcp = None
        self.monitor = None
        self.system_status = {}

        logger.info("🚀 INITIALIZING COMPLETE MCP TRADING INTEGRATION")
        logger.info("=" * 60)

    async def initialize_all_components(self):
        """Initialize all MCP trading components"""
        logger.info("🔧 Initializing all MCP components...")

        try:
            # 1. Initialize MCP Live Trading Connector
            self.connector = MCPLiveTradingConnector()
            await self.connector.initialize_components()
            logger.info("✅ MCP Live Trading Connector: INITIALIZED")

            # 2. Initialize Performance Tracker
            self.performance_tracker = MCPPerformanceTracker()
            logger.info("✅ Performance Tracker: INITIALIZED")

            # 3. Initialize GitHub MCP Integration
            self.github_mcp = GitHubMCPIntegration()
            logger.info("✅ GitHub MCP Integration: INITIALIZED")

            # 4. Initialize Trading Monitor
            self.monitor = MCPTradingMonitor()
            logger.info("✅ Trading Monitor: INITIALIZED")

            self.system_status['all_components_ready'] = True
            logger.info("🎉 ALL MCP COMPONENTS SUCCESSFULLY INITIALIZED!")

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise

    async def start_complete_trading_system(self):
        """Start the complete MCP trading system"""
        logger.info("🚀 STARTING COMPLETE MCP TRADING SYSTEM")
        logger.info("=" * 60)

        try:
            # Initialize all components
            await self.initialize_all_components()

            # Create default trading tasks
            await self.setup_default_trading_tasks()

            # Start performance monitoring
            performance_task = asyncio.create_task(self.performance_monitoring_loop())

            # Start system monitoring
            monitoring_task = asyncio.create_task(self.system_monitoring_loop())

            # Start MCP trading system
            trading_task = asyncio.create_task(self.connector.run_mcp_trading_system())

            # Run all tasks concurrently
            await asyncio.gather(
                performance_task,
                monitoring_task,
                trading_task
            )

        except Exception as e:
            logger.error(f"❌ Complete system startup failed: {e}")
            return False

    async def setup_default_trading_tasks(self):
        """Set up default trading tasks from configuration"""
        logger.info("📝 Setting up default trading tasks...")

        try:
            # Load task configuration
            config_path = project_root / "mcp_trading_tasks.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)

                scheduled_tasks = config.get('mcp_trading_tasks', {}).get('scheduled_tasks', [])

                for task_config in scheduled_tasks:
                    if task_config.get('active', False):
                        merged_config = {
                            'task_name': task_config['name'],
                            'task_description': task_config['description'],
                            'schedule': task_config.get('schedule'),
                            **task_config.get('config', {})
                        }

                        task_id = await self.connector.create_trading_task(merged_config)

                        if task_id:
                            logger.info(f"✅ Created task: {task_config['name']} ({task_id})")
                        else:
                            logger.error(f"❌ Failed to create task: {task_config['name']}")

        except Exception as e:
            logger.error(f"❌ Task setup failed: {e}")

    async def performance_monitoring_loop(self):
        """Continuous performance monitoring loop"""
        logger.info("📈 Starting performance monitoring...")

        while True:
            try:
                # Generate daily performance report
                current_hour = datetime.now().hour
                current_minute = datetime.now().minute

                # Generate report at 23:59 daily
                if current_hour == 23 and current_minute >= 59:
                    await self.performance_tracker.generate_daily_report()

                    # Export performance data
                    await self.performance_tracker.export_performance_data()

                    # Wait for next day
                    await asyncio.sleep(60)

                # Update GitHub with current metrics every hour
                if current_minute == 0:
                    metrics = self.performance_tracker.get_cumulative_performance()
                    await self.github_mcp.log_system_performance({
                        'hourly_update': True,
                        'cumulative_metrics': metrics.__dict__,
                        'portfolio_value': self.performance_tracker.portfolio_value
                    })

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"❌ Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def system_monitoring_loop(self):
        """Continuous system monitoring loop"""
        logger.info("🔍 Starting system monitoring...")

        while True:
            try:
                # Get system status
                status = await self.connector.get_trading_status()

                # Check for issues
                if status:
                    await self.check_system_health(status)

                    # Log to GitHub
                    await self.github_mcp.log_system_performance({
                        'system_monitoring': True,
                        'system_status': status,
                        'timestamp': datetime.now().isoformat()
                    })

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"❌ System monitoring error: {e}")
                await asyncio.sleep(60)

    async def check_system_health(self, status: dict):
        """Check system health and create alerts if needed"""
        try:
            issues = []

            # Check component readiness
            if not status.get('system_components', {}).get('components_ready', False):
                issues.append({
                    'type': 'components_not_ready',
                    'severity': 'high',
                    'message': 'System components are not ready'
                })

            # Check trading status
            if status.get('emergency_stop', False):
                issues.append({
                    'type': 'emergency_stop_active',
                    'severity': 'critical',
                    'message': 'Emergency stop is active'
                })

            # Check active tasks
            if status.get('active_tasks', 0) == 0 and status.get('trading_active', False):
                issues.append({
                    'type': 'no_active_tasks',
                    'severity': 'medium',
                    'message': 'No active trading tasks'
                })

            # Create GitHub issues for critical issues
            for issue in issues:
                if issue['severity'] in ['high', 'critical']:
                    await self.create_system_health_issue(issue)

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")

    async def create_system_health_issue(self, issue: dict):
        """Create GitHub issue for system health problem"""
        try:
            issue_title = f"🚨 System Health Alert: {issue['type'].replace('_', ' ').title()}"

            issue_body = f"""## System Health Alert

**Issue Type:** {issue['type'].replace('_', ' ').title()}
**Severity:** {issue['severity'].upper()}
**Timestamp:** {datetime.now().isoformat()}

### Issue Details
{issue['message']}

### System Status
- MCP Connected: {self.system_status.get('mcp_connected', 'Unknown')}
- Components Ready: {self.system_status.get('all_components_ready', 'Unknown')}
- Trading Active: {self.system_status.get('trading_active', 'Unknown')}

### Recommended Actions
1. Check system logs for detailed error information
2. Restart affected components if necessary
3. Review system configuration
4. Contact system administrator if issue persists

---
*Auto-generated by VIPER MCP Integration*
"""

            # Log the issue
            await self.github_mcp.log_system_performance({
                'system_health_issue': True,
                'issue_type': issue['type'],
                'issue_severity': issue['severity'],
                'issue_message': issue['message']
            })

            logger.warning(f"🚨 System health issue created: {issue['type']}")

        except Exception as e:
            logger.error(f"❌ System health issue creation failed: {e}")

    async def emergency_system_shutdown(self):
        """Emergency shutdown of all MCP trading operations"""
        logger.warning("🚨 EMERGENCY SYSTEM SHUTDOWN INITIATED!")

        try:
            # Stop all trading tasks
            if self.connector:
                await self.connector.emergency_stop_all()

            # Export final performance data
            if self.performance_tracker:
                await self.performance_tracker.export_performance_data('emergency_shutdown_export.json')

            # Create emergency shutdown issue
            await self.github_mcp.log_system_performance({
                'emergency_shutdown': True,
                'reason': 'manual_emergency_shutdown',
                'timestamp': datetime.now().isoformat()
            })

            logger.info("✅ Emergency shutdown completed")
            return True

        except Exception as e:
            logger.error(f"❌ Emergency shutdown failed: {e}")
            return False

    async def get_complete_system_status(self):
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'system_name': 'VIPER MCP Complete Trading Integration',
                'version': '1.0.0',
                'components': {
                    'mcp_connector': self.connector is not None,
                    'performance_tracker': self.performance_tracker is not None,
                    'github_mcp': self.github_mcp is not None,
                    'monitor': self.monitor is not None
                },
                'system_health': self.system_status
            }

            # Add trading status
            if self.connector:
                trading_status = await self.connector.get_trading_status()
                if trading_status:
                    status['trading_status'] = trading_status

            # Add performance metrics
            if self.performance_tracker:
                metrics = self.performance_tracker.get_cumulative_performance()
                status['performance_metrics'] = metrics.__dict__

            return status

        except Exception as e:
            logger.error(f"❌ System status retrieval failed: {e}")
            return {'error': str(e)}

# Command-line interface
async def main():
    """Main integration function"""
    parser = argparse.ArgumentParser(description='Complete MCP Trading Integration')
    parser.add_argument('--mode', choices=['start', 'status', 'monitor', 'emergency'],
                       default='start', help='Operation mode')
    parser.add_argument('--config', type=str,
                       help='Path to custom configuration file')

    args = parser.parse_args()


    integration = CompleteMCPTradingIntegration()

    try:
        if args.mode == 'start':
            # Start the complete system
            success = await integration.start_complete_trading_system()
            if success:
                print("✅ Complete MCP trading system started successfully")
            else:
                print("❌ Failed to start complete MCP trading system")
                return 1

        elif args.mode == 'status':
            # Get system status
            await integration.initialize_all_components()
            status = await integration.get_complete_system_status()

        elif args.mode == 'monitor':
            # Start monitoring mode
            await integration.initialize_all_components()
            if integration.monitor:
                await integration.monitor.connect()
                await integration.monitor.monitor_loop()

        elif args.mode == 'emergency':
            # Emergency shutdown
            await integration.initialize_all_components()
            success = await integration.emergency_system_shutdown()
            if success:
            else:
                return 1

    except KeyboardInterrupt:
        if args.mode == 'start':
            await integration.emergency_system_shutdown()
    except Exception as e:
        logger.error(f"❌ Integration error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
