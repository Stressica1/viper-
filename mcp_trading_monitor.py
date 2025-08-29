#!/usr/bin/env python3
"""
📊 MCP TRADING MONITOR
Real-time monitoring client for MCP Live Trading System

This monitor provides:
✅ Real-time trading status updates
✅ Task management interface
✅ Performance metrics dashboard
✅ Emergency control panel
✅ WebSocket-based communication
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
import websockets
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MCP_MONITOR - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPTradingMonitor:
    """Real-time monitor for MCP Trading System"""

    def __init__(self, websocket_url: str = "ws://localhost:8765"):
        self.websocket_url = websocket_url
        self.websocket = None
        self.connected = False
        self.last_status = {}

    async def connect(self):
        """Connect to MCP trading WebSocket"""
        try:
            logger.info(f"🔗 Connecting to {self.websocket_url}")
            self.websocket = await websockets.connect(self.websocket_url)
            self.connected = True
            logger.info("✅ Connected to MCP Trading System")
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.connected = False

    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("🔌 Disconnected from MCP Trading System")

    async def send_command(self, command_type: str, **kwargs):
        """Send command to MCP trading system"""
        try:
            if not self.connected:
                logger.error("❌ Not connected to MCP system")
                return None

            message = {
                'type': command_type,
                **kwargs
            }

            await self.websocket.send(json.dumps(message))
            logger.info(f"📤 Sent command: {command_type}")

            # Wait for response
            response = await self.websocket.recv()
            return json.loads(response)

        except Exception as e:
            logger.error(f"❌ Command failed: {e}")
            return None

    async def get_status(self):
        """Get current system status"""
        return await self.send_command('get_status')

    async def create_task(self, config: dict):
        """Create new trading task"""
        return await self.send_command('create_task', config=config)

    async def start_task(self, task_id: str):
        """Start trading task"""
        return await self.send_command('start_task', task_id=task_id)

    async def stop_task(self, task_id: str):
        """Stop trading task"""
        return await self.send_command('stop_task', task_id=task_id)

    async def emergency_stop(self):
        """Emergency stop all operations"""
        return await self.send_command('emergency_stop')

    async def monitor_loop(self):
        """Main monitoring loop"""
        print("📊 MCP TRADING MONITOR")
        print("=" * 40)
        print("Real-time monitoring active...")
        print("Press Ctrl+C to exit")
        print("=" * 40)

        try:
            await self.connect()

            while self.connected:
                try:
                    # Get status update
                    status = await self.get_status()

                    if status:
                        self.last_status = status
                        self.display_status(status)

                    await asyncio.sleep(5)  # Update every 5 seconds

                except websockets.exceptions.ConnectionClosed:
                    logger.warning("⚠️ Connection lost, attempting reconnect...")
                    await self.connect()
                    await asyncio.sleep(2)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        finally:
            await self.disconnect()

    def display_status(self, status: dict):
        """Display formatted status information"""
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen

        print("📊 MCP TRADING SYSTEM STATUS")
        print("=" * 50)
        print(f"Timestamp: {status.get('timestamp', 'N/A')}")
        print(f"MCP Connected: {'✅' if status.get('mcp_connected', False) else '❌'}")
        print(f"Trading Active: {'✅' if status.get('trading_active', False) else '❌'}")
        print(f"Emergency Stop: {'🚨' if status.get('emergency_stop', False) else '✅'}")
        print()

        print("📈 SYSTEM METRICS")
        print("-" * 30)
        print(f"Active Tasks: {status.get('active_tasks', 0)}")
        print(f"Total Tasks: {status.get('total_tasks', 0)}")
        print(f"Components Ready: {'✅' if status.get('system_components', {}).get('components_ready', False) else '❌'}")
        print()

        print("⚙️ TRADING PARAMETERS")
        print("-" * 30)
        params = status.get('trading_parameters', {})
        print(f"Max Positions: {params.get('max_positions', 'N/A')}")
        print(f"Risk per Trade: {params.get('risk_per_trade', 'N/A')}")
        print(f"Leverage: {params.get('leverage', 'N/A')}")
        print(f"Trading Pairs: {', '.join(params.get('trading_pairs', []))}")
        print()

        print("🎯 ACTIVE TASKS")
        print("-" * 30)
        tasks = status.get('tasks', [])
        if tasks:
            for task in tasks:
                status_icon = {
                    'active': '🟢',
                    'running': '🟡',
                    'scheduled': '🔵',
                    'stopped': '🔴',
                    'error': '❌'
                }.get(task.get('status', 'unknown'), '⚪')

                print(f"{status_icon} {task.get('id', 'N/A')} - {task.get('status', 'unknown')}")
                print(f"   Pairs: {', '.join(task.get('trading_pairs', []))}")
        else:
            print("No active tasks")
        print()

        print("💡 CONTROLS")
        print("-" * 30)
        print("Available commands (when in interactive mode):")
        print("  s  - Get detailed status")
        print("  c  - Create new task")
        print("  t  - Start task")
        print("  p  - Stop task")
        print("  e  - Emergency stop")
        print("  q  - Quit")
        print()

async def interactive_monitor():
    """Run interactive monitoring session"""
    monitor = MCPTradingMonitor()

    try:
        await monitor.connect()

        print("🔗 INTERACTIVE MCP TRADING MONITOR")
        print("=" * 50)
        print("Commands:")
        print("  s - Status")
        print("  c - Create task")
        print("  t - Start task")
        print("  p - Stop task")
        print("  e - Emergency stop")
        print("  q - Quit")
        print("=" * 50)

        while True:
            try:
                cmd = input("Monitor> ").strip().lower()

                if cmd == 'q':
                    break
                elif cmd == 's':
                    status = await monitor.get_status()
                    if status:
                        print(json.dumps(status, indent=2))
                    else:
                        print("❌ Failed to get status")
                elif cmd == 'c':
                    # Create a default task
                    config = {
                        'trading_pairs': ['BTCUSDT', 'ETHUSDT'],
                        'max_positions': 3,
                        'risk_per_trade': 0.02,
                        'leverage': 25
                    }
                    result = await monitor.create_task(config)
                    if result:
                        print(f"✅ Task created: {result.get('task_id')}")
                    else:
                        print("❌ Failed to create task")
                elif cmd == 't':
                    task_id = input("Task ID to start: ").strip()
                    if task_id:
                        result = await monitor.start_task(task_id)
                        if result and result.get('success'):
                            print(f"✅ Task started: {task_id}")
                        else:
                            print(f"❌ Failed to start task: {task_id}")
                elif cmd == 'p':
                    task_id = input("Task ID to stop: ").strip()
                    if task_id:
                        result = await monitor.stop_task(task_id)
                        if result and result.get('success'):
                            print(f"✅ Task stopped: {task_id}")
                        else:
                            print(f"❌ Failed to stop task: {task_id}")
                elif cmd == 'e':
                    confirm = input("Confirm emergency stop (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        result = await monitor.emergency_stop()
                        if result and result.get('success'):
                            print("🚨 Emergency stop activated")
                        else:
                            print("❌ Emergency stop failed")
                else:
                    print("❌ Unknown command")

            except KeyboardInterrupt:
                break

    except Exception as e:
        logger.error(f"❌ Interactive monitor error: {e}")
    finally:
        await monitor.disconnect()

def main():
    """Main monitor function"""
    parser = argparse.ArgumentParser(description='MCP Trading Monitor')
    parser.add_argument('--url', type=str, default='ws://localhost:8765',
                       help='WebSocket URL for MCP trading system')
    parser.add_argument('--mode', choices=['monitor', 'interactive'],
                       default='monitor', help='Monitor mode')

    args = parser.parse_args()

    print("📊 MCP TRADING MONITOR")
    print("=" * 30)
    print(f"URL: {args.url}")
    print(f"Mode: {args.mode}")
    print("=" * 30)

    if args.mode == 'interactive':
        asyncio.run(interactive_monitor())
    else:
        monitor = MCPTradingMonitor(args.url)
        asyncio.run(monitor.monitor_loop())

if __name__ == "__main__":
    main()
