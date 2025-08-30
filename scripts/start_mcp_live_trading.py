#!/usr/bin/env python3
"""
# Rocket START MCP LIVE TRADING
Simple launcher for MCP-powered live trading system

This script provides:
    pass
# Check Easy startup of MCP live trading connector
# Check Task scheduling and management
# Check Real-time monitoring dashboard
# Check Emergency controls and safety features
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

# Import MCP trading components
from mcp_live_trading_connector import MCPLiveTradingConnector, create_live_trading_task, start_live_trading_task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MCP_START - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPLiveTradingLauncher:
    """Launcher for MCP Live Trading System"""

    def __init__(self):
        self.connector = None
        self.tasks_config = {}

        # Load task configuration
        self.load_task_config()

    def load_task_config(self):
        """Load MCP trading tasks configuration"""
        try:
            config_path = project_root / "mcp_trading_tasks.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.tasks_config = config.get('mcp_trading_tasks', {})
                logger.info("# Check Task configuration loaded")
            else:
                logger.warning("# Warning Task configuration file not found")
        except Exception as e:
            logger.error(f"# X Failed to load task config: {e}")

    async def create_default_tasks(self):
        """Create default trading tasks from configuration"""
        logger.info("📝 Creating default trading tasks...")

        try:
            scheduled_tasks = self.tasks_config.get('scheduled_tasks', [])

            for task_config in scheduled_tasks:
                if task_config.get('active', False):
                    # Merge with default config
                    default_config = self.tasks_config.get('default_config', {})
                    merged_config = {**default_config, **task_config.get('config', {})}

                    # Add task metadata
                    merged_config.update({)
                        'task_name': task_config['name'],
                        'task_description': task_config['description'],
                        'schedule': task_config.get('schedule'),
                        'task_id_prefix': task_config['id']
(                    })

                    task_id = await create_live_trading_task(merged_config)

                    if task_id:
                        logger.info(f"# Check Created task: {task_config['name']} ({task_id})")
                    else:
                        logger.error(f"# X Failed to create task: {task_config['name']}")

        except Exception as e:
            logger.error(f"# X Task creation failed: {e}")

    async def start_trading_operations(self):
        """Start all active trading operations"""
        logger.info("# Rocket Starting trading operations...")

        try:
            # Initialize MCP connector
            self.connector = MCPLiveTradingConnector()
            await self.connector.initialize_components()

            # Create and start default tasks
            await self.create_default_tasks()

            # Start the MCP trading system
            await self.connector.run_mcp_trading_system()

        except Exception as e:
            logger.error(f"# X Failed to start trading operations: {e}")
            return False

        return True

    async def run_interactive_mode(self):
        """Run in interactive mode with command interface"""

        while True:"""
            try:
                command = input("MCP Trading> ").strip().lower()

                if command == 'exit':
                    break
                elif command == 'status':
                    if self.connector:
                        status = await self.connector.get_trading_status()
                    else:
                elif command == 'create':
                    pass
                    config = self.tasks_config.get('default_config', {})
                    task_id = await create_live_trading_task(config)
                elif command.startswith('start '):
                    task_id = command.split(' ', 1)[1]
                    success = await start_live_trading_task(task_id)
                elif command.startswith('stop '):
                    task_id = command.split(' ', 1)[1]
                    success = await self.connector.stop_trading_task(task_id)
                elif command == 'emergency':
                    success = await self.connector.emergency_stop_all()
                else:
                    pass

            except KeyboardInterrupt:
                break
            except Exception as e:
                pass

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='MCP Live Trading Launcher')
    parser.add_argument('--mode', choices=['auto', 'interactive'],
                       default='auto', help='Launch mode')
    parser.add_argument('--config', type=str,
                       help='Path to custom config file')

    args = parser.parse_args()


    launcher = MCPLiveTradingLauncher()

    # Load custom config if provided
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
                launcher.tasks_config.update(custom_config.get('mcp_trading_tasks', {}))
        except Exception as e:
            pass

    async def run_launcher():
        if args.mode == 'interactive':
            # Initialize connector for interactive mode
            launcher.connector = MCPLiveTradingConnector()
            await launcher.connector.initialize_components()
            await launcher.run_interactive_mode()
        else:
            # Auto mode - start trading operations
            success = await launcher.start_trading_operations()
            return success

    try:
        success = asyncio.run(run_launcher())
        if success:
            print("# Check MCP Live Trading System completed successfully")
            return 0
        else:
            return 1
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
