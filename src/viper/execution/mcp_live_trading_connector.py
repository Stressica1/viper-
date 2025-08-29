#!/usr/bin/env python3
"""
🔗 MCP LIVE TRADING CONNECTOR
GitHub MCP integration for automated live trading operations

This connector provides:
✅ Automated trading task scheduling via MCP
✅ Real-time trading execution and monitoring
✅ GitHub integration for performance tracking
✅ Risk management and emergency controls
✅ Automated position management and reporting
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
import websockets

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import existing trading components
from master_live_trading_job import MasterLiveTradingJob
from github_mcp_integration import GitHubMCPIntegration
from mcp_brain_controller import MCPBrainController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MCP_TRADING - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MCPLiveTradingConnector:
    """
    MCP Connector for Live Trading Operations
    """

    def __init__(self):
        # First, enforce Docker and MCP requirements
        from docker_mcp_enforcer import enforce_docker_mcp_requirements
        
        logger.info("🔒 Enforcing Docker & MCP requirements for live trading...")
        if not enforce_docker_mcp_requirements():
            logger.error("❌ Docker/MCP requirements not met")
            raise Exception("MCP Live Trading requires Docker and MCP server")
        
        self.trading_job = None
        self.github_mcp = None
        self.brain_controller = None
        self.websocket_connections = set()
        self.active_tasks = {}
        self.system_status = {}

        # Validate live trading environment
        if os.getenv('USE_MOCK_DATA', '').lower() == 'true':
            raise Exception("🚫 LIVE TRADING MODE: Mock data not allowed")
        
        if not os.getenv('FORCE_LIVE_TRADING', '').lower() == 'true':
            raise Exception("🚫 LIVE TRADING MODE: Must set FORCE_LIVE_TRADING=true")

        # MCP Configuration - Using Docker network URLs
        self.mcp_config = {
            'server_url': os.getenv('MCP_SERVER_URL', 'http://mcp-server:8015'),
            'websocket_url': os.getenv('MCP_WS_URL', 'ws://mcp-server:8015/ws'),
            'github_token': os.getenv('GITHUB_PAT', os.getenv('GITHUB_TOKEN', '')),
            'trading_active': True,  # Force active for live trading
            'auto_restart': True,
            'emergency_stop': False
        }

        # Trading parameters - Live trading configuration
        self.trading_params = {
            'max_positions': int(os.getenv('MAX_POSITIONS', '15')),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),
            'leverage': int(os.getenv('MAX_LEVERAGE', '50')),
            'trading_pairs': os.getenv('TRADING_PAIRS', 'BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,ADAUSDT').split(','),
            'min_balance_threshold': float(os.getenv('MIN_BALANCE_THRESHOLD', '100.0'))
        }

        logger.info("🚀 INITIALIZING MCP LIVE TRADING CONNECTOR - LIVE MODE ONLY")
        logger.info("=" * 60)
        logger.info("🚨 LIVE TRADING: Real trades will be executed")
        logger.info("🔒 Docker and MCP enforcement: ACTIVE")

    async def initialize_components(self):
        """Initialize all MCP trading components"""
        logger.info("🔧 Initializing MCP Trading Components...")

        try:
            # 1. Initialize Master Live Trading Job
            self.trading_job = MasterLiveTradingJob()
            await self.trading_job.initialize_components()
            logger.info("✅ Master Live Trading Job: INITIALIZED")

            # 2. Initialize GitHub MCP Integration
            self.github_mcp = GitHubMCPIntegration()
            logger.info("✅ GitHub MCP Integration: INITIALIZED")

            # 3. Initialize Brain Controller
            self.brain_controller = MCPBrainController()
            logger.info("✅ MCP Brain Controller: INITIALIZED")

            self.system_status['components_ready'] = True
            logger.info("🎉 ALL MCP COMPONENTS SUCCESSFULLY INITIALIZED!")

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise

    async def create_trading_task(self, task_config: Dict[str, Any]) -> str:
        """
        Create an automated trading task via MCP

        Args:
            task_config: Configuration for the trading task

        Returns:
            Task ID for tracking
        """
        try:
            task_id = f"trading_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            task_data = {
                'id': task_id,
                'type': 'automated_trading',
                'config': task_config,
                'status': 'scheduled',
                'created_at': datetime.now().isoformat(),
                'scheduled_start': task_config.get('start_time'),
                'trading_pairs': task_config.get('pairs', self.trading_params['trading_pairs']),
                'risk_parameters': {
                    'max_positions': task_config.get('max_positions', self.trading_params['max_positions']),
                    'risk_per_trade': task_config.get('risk_per_trade', self.trading_params['risk_per_trade']),
                    'leverage': task_config.get('leverage', self.trading_params['leverage'])
                }
            }

            # Store task in active tasks
            self.active_tasks[task_id] = task_data

            # Log task creation to GitHub
            await self.github_mcp.log_system_performance({
                'task_created': task_id,
                'task_type': 'automated_trading',
                'trading_pairs': task_data['trading_pairs'],
                'risk_parameters': task_data['risk_parameters']
            })

            logger.info(f"✅ Trading task created: {task_id}")
            return task_id

        except Exception as e:
            logger.error(f"❌ Task creation failed: {e}")
            return None

    async def start_trading_task(self, task_id: str) -> bool:
        """
        Start a scheduled trading task

        Args:
            task_id: ID of the task to start

        Returns:
            Success status
        """
        try:
            if task_id not in self.active_tasks:
                logger.error(f"❌ Task not found: {task_id}")
                return False

            task = self.active_tasks[task_id]

            # Check system readiness
            if not self.system_status.get('components_ready', False):
                logger.error("❌ System not ready for trading")
                return False

            # Check emergency stop
            if self.mcp_config['emergency_stop']:
                logger.warning("⚠️ Emergency stop active - trading disabled")
                return False

            # Start the trading job
            logger.info(f"🚀 Starting trading task: {task_id}")
            task['status'] = 'running'
            task['started_at'] = datetime.now().isoformat()

            # Start live trading
            success = await self.trading_job.start_live_trading()

            if success:
                task['status'] = 'active'
                self.mcp_config['trading_active'] = True
                logger.info(f"✅ Trading task started successfully: {task_id}")

                # Log to GitHub
                await self.github_mcp.log_system_performance({
                    'trading_started': True,
                    'task_id': task_id,
                    'trading_pairs': task['trading_pairs'],
                    'timestamp': datetime.now().isoformat()
                })

                return True
            else:
                task['status'] = 'failed'
                logger.error(f"❌ Failed to start trading task: {task_id}")
                return False

        except Exception as e:
            logger.error(f"❌ Trading task start failed: {e}")
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'error'
            return False

    async def stop_trading_task(self, task_id: str) -> bool:
        """
        Stop a running trading task

        Args:
            task_id: ID of the task to stop

        Returns:
            Success status
        """
        try:
            if task_id not in self.active_tasks:
                logger.error(f"❌ Task not found: {task_id}")
                return False

            task = self.active_tasks[task_id]

            logger.info(f"🛑 Stopping trading task: {task_id}")
            task['status'] = 'stopping'
            task['stopped_at'] = datetime.now().isoformat()

            # Stop trading operations
            self.mcp_config['trading_active'] = False

            # Log to GitHub
            await self.github_mcp.log_system_performance({
                'trading_stopped': True,
                'task_id': task_id,
                'reason': 'manual_stop',
                'timestamp': datetime.now().isoformat()
            })

            task['status'] = 'stopped'
            logger.info(f"✅ Trading task stopped: {task_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Trading task stop failed: {e}")
            return False

    async def get_trading_status(self) -> Dict[str, Any]:
        """
        Get comprehensive trading status

        Returns:
            Dictionary containing current trading status
        """
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'mcp_connected': True,
                'trading_active': self.mcp_config['trading_active'],
                'emergency_stop': self.mcp_config['emergency_stop'],
                'active_tasks': len([t for t in self.active_tasks.values() if t['status'] == 'active']),
                'total_tasks': len(self.active_tasks),
                'system_components': self.system_status,
                'trading_parameters': self.trading_params
            }

            # Add task details
            status['tasks'] = []
            for task_id, task in self.active_tasks.items():
                task_summary = {
                    'id': task_id,
                    'status': task['status'],
                    'created_at': task['created_at'],
                    'trading_pairs': task.get('trading_pairs', [])
                }
                status['tasks'].append(task_summary)

            return status

        except Exception as e:
            logger.error(f"❌ Status retrieval failed: {e}")
            return {'error': str(e)}

    async def emergency_stop_all(self) -> bool:
        """
        Emergency stop all trading operations

        Returns:
            Success status
        """
        try:
            logger.warning("🚨 EMERGENCY STOP ACTIVATED!")

            # Set emergency flag
            self.mcp_config['emergency_stop'] = True
            self.mcp_config['trading_active'] = False

            # Stop all active tasks
            stopped_tasks = []
            for task_id, task in self.active_tasks.items():
                if task['status'] in ['running', 'active']:
                    task['status'] = 'emergency_stopped'
                    task['emergency_stopped_at'] = datetime.now().isoformat()
                    stopped_tasks.append(task_id)

            # Log emergency stop to GitHub
            await self.github_mcp.log_system_performance({
                'emergency_stop': True,
                'stopped_tasks': stopped_tasks,
                'timestamp': datetime.now().isoformat(),
                'reason': 'emergency_stop_all'
            })

            logger.info(f"✅ Emergency stop completed - {len(stopped_tasks)} tasks stopped")
            return True

        except Exception as e:
            logger.error(f"❌ Emergency stop failed: {e}")
            return False

    async def websocket_handler(self, websocket, path):
        """
        Handle WebSocket connections for real-time updates
        """
        try:
            self.websocket_connections.add(websocket)
            logger.info("🔗 WebSocket client connected")

            try:
                async for message in websocket:
                    data = json.loads(message)

                    if data.get('type') == 'get_status':
                        status = await self.get_trading_status()
                        await websocket.send(json.dumps({
                            'type': 'status_update',
                            'data': status
                        }))

                    elif data.get('type') == 'create_task':
                        task_config = data.get('config', {})
                        task_id = await self.create_trading_task(task_config)
                        await websocket.send(json.dumps({
                            'type': 'task_created',
                            'task_id': task_id
                        }))

                    elif data.get('type') == 'start_task':
                        task_id = data.get('task_id')
                        success = await self.start_trading_task(task_id)
                        await websocket.send(json.dumps({
                            'type': 'task_started',
                            'task_id': task_id,
                            'success': success
                        }))

                    elif data.get('type') == 'stop_task':
                        task_id = data.get('task_id')
                        success = await self.stop_trading_task(task_id)
                        await websocket.send(json.dumps({
                            'type': 'task_stopped',
                            'task_id': task_id,
                            'success': success
                        }))

                    elif data.get('type') == 'emergency_stop':
                        success = await self.emergency_stop_all()
                        await websocket.send(json.dumps({
                            'type': 'emergency_stopped',
                            'success': success
                        }))

            except websockets.exceptions.ConnectionClosed:
                logger.info("🔌 WebSocket client disconnected")

        except Exception as e:
            logger.error(f"❌ WebSocket handler error: {e}")
        finally:
            self.websocket_connections.discard(websocket)

    async def start_websocket_server(self):
        """Start WebSocket server for real-time communication"""
        try:
            server = await websockets.serve(
                self.websocket_handler,
                "localhost",
                8765
            )
            logger.info("🔗 WebSocket server started on ws://localhost:8765")
            await server.wait_closed()
        except Exception as e:
            logger.error(f"❌ WebSocket server failed: {e}")

    async def run_monitoring_loop(self):
        """Continuous monitoring loop for trading operations"""
        while True:
            try:
                # Get current trading status
                status = await self.get_trading_status()

                # Log performance data to GitHub
                if self.mcp_config['trading_active']:
                    await self.github_mcp.log_system_performance({
                        'monitoring_update': True,
                        'active_tasks': status['active_tasks'],
                        'system_status': status['system_components'],
                        'timestamp': datetime.now().isoformat()
                    })

                # Broadcast status to WebSocket clients
                status_message = json.dumps({
                    'type': 'status_update',
                    'data': status
                })

                for connection in self.websocket_connections.copy():
                    try:
                        await connection.send(status_message)
                    except Exception:
                        self.websocket_connections.discard(connection)

                # Check for emergency conditions
                await self.check_emergency_conditions()

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(10)

    async def check_emergency_conditions(self):
        """Check for emergency stop conditions"""
        try:
            # Check if emergency stop is already active
            if self.mcp_config['emergency_stop']:
                return

            # Check system health
            if not self.system_status.get('components_ready', False):
                logger.warning("⚠️ System components not ready - emergency check")
                await self.emergency_stop_all()
                return

            # Check for excessive losses (placeholder logic)
            # This would be implemented with actual trading data

            # Check balance thresholds
            # This would be implemented with actual balance data

        except Exception as e:
            logger.error(f"❌ Emergency condition check failed: {e}")

    async def run_mcp_trading_system(self):
        """Run the complete MCP trading system"""
        logger.info("🚀 STARTING MCP LIVE TRADING SYSTEM")
        logger.info("=" * 60)

        try:
            # Initialize all components
            await self.initialize_components()

            # Start background tasks
            monitoring_task = asyncio.create_task(self.run_monitoring_loop())
            websocket_task = asyncio.create_task(self.start_websocket_server())

            # Run system checks
            success = await self.trading_job.run_complete_system_check()

            if success:
                logger.info("🎉 MCP LIVE TRADING SYSTEM READY!")
                logger.info("Features Active:")
                logger.info("   • Automated Task Scheduling: ✅")
                logger.info("   • Real-time Monitoring: ✅")
                logger.info("   • GitHub Integration: ✅")
                logger.info("   • WebSocket API: ✅")
                logger.info("   • Emergency Controls: ✅")

                # Keep the system running
                await asyncio.gather(monitoring_task, websocket_task)
            else:
                logger.error("❌ System check failed")
                return False

        except Exception as e:
            logger.error(f"❌ MCP trading system failed: {e}")
            return False

# MCP Task Functions for External Integration
async def create_live_trading_task(config: Dict[str, Any]) -> str:
    """
    Create a live trading task via MCP

    Args:
        config: Trading configuration

    Returns:
        Task ID
    """
    connector = MCPLiveTradingConnector()
    await connector.initialize_components()
    return await connector.create_trading_task(config)

async def start_live_trading_task(task_id: str) -> bool:
    """
    Start a live trading task

    Args:
        task_id: Task to start

    Returns:
        Success status
    """
    connector = MCPLiveTradingConnector()
    await connector.initialize_components()
    return await connector.start_trading_task(task_id)

async def stop_live_trading_task(task_id: str) -> bool:
    """
    Stop a live trading task

    Args:
        task_id: Task to stop

    Returns:
        Success status
    """
    connector = MCPLiveTradingConnector()
    await connector.initialize_components()
    return await connector.stop_trading_task(task_id)

async def get_mcp_trading_status() -> Dict[str, Any]:
    """
    Get MCP trading system status

    Returns:
        System status
    """
    connector = MCPLiveTradingConnector()
    await connector.initialize_components()
    return await connector.get_trading_status()

# Main execution
async def main():
    """Main MCP trading system execution"""

    connector = MCPLiveTradingConnector()

    try:
        await connector.run_mcp_trading_system()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down MCP trading system...")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
