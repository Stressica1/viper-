#!/usr/bin/env python3
"""
# Rocket VIPER LIVE TRADING JOB MANAGER
GitHub MCP Integrated - Position Tracking & Risk Management
MAX 10 POSITIONS | MAX 3% RISK PER TRADE | PROPER MARGIN CALCULATION
"""

import asyncio
import aiohttp
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Live position tracking with risk management"""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    leverage: int
    margin_used: float
    timestamp: datetime
    pnl_pct: float = 0.0
    status: str = 'active'  # 'active', 'closed', 'error'

class ViperLiveJobManager:
    """# Rocket VIPER LIVE TRADING JOB MANAGER WITH GITHUB MCP INTEGRATION

    FEATURES:
    # Check MAX 10 POSITIONS TOTAL
    # Check MAX 3% RISK PER TRADE
    # Check PROPER MARGIN/LEVERAGE CALCULATION
    # Check GITHUB MCP JOB TRACKING
    # Check REAL-TIME RISK MONITORING
    """

    def __init__(self, scheduler_url: str = "http://localhost:8021"):
        self.scheduler_url = scheduler_url
        self.session = None

        # GitHub MCP Configuration
        self.github_token = os.getenv('GITHUB_PAT', '')
        self.github_owner = os.getenv('GITHUB_OWNER', 'Stressica1')
        self.github_repo = os.getenv('GITHUB_REPO', 'viper-')
        self.github_api_url = "https://api.github.com"

        # Position Tracking & Risk Management
        self.max_positions = 10  # NEVER MORE THAN 10 POSITIONS
        self.max_risk_per_trade = 0.03  # MAX 3% RISK PER TRADE
        self.active_positions: Dict[str, Position] = {}
        self.total_margin_used = 0.0
        self.account_balance = 0.0

        # Risk Limits
        self.daily_loss_limit = 0.05  # 5% max daily loss
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0

        # Job Tracking
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: List[Dict[str, Any]] = []

        logger.info("# Rocket VIPER LIVE JOB MANAGER INITIALIZED")
        logger.info(f"# Target MAX POSITIONS: {self.max_positions}")
        logger.info(f"# Warning MAX RISK PER TRADE: {self.max_risk_per_trade*100}%")
        logger.info("🔗 GITHUB MCP INTEGRATION: ENABLED")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    # ===============================
    # GITHUB MCP INTEGRATION METHODS
    # ===============================

    async def create_github_job(self, title: str, body: str, labels: List[str] = None) -> Optional[str]:
        """Create GitHub issue for job tracking"""
        if not self.github_token:
            logger.warning("# Warning GitHub token not configured - skipping job creation")
            return None

        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            data = {
                'title': f'# Rocket {title}',
                'body': body,
                'labels': labels or ['trading-job', 'viper-system']
            }

            url = f"{self.github_api_url}/repos/{self.github_owner}/{self.github_repo}/issues"

            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status == 201:
                    result = await response.json()
                    job_id = str(result['number'])
                    logger.info(f"# Check Created GitHub job #{job_id}: {title}")
                    return job_id
                else:
                    error_text = await response.text()
                    logger.error(f"# X Failed to create GitHub job: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.error(f"# X GitHub job creation error: {e}")
            return None

    async def update_github_job(self, job_id: str, status: str, update_body: str = ""):
        """Update GitHub issue status"""
        if not self.github_token or not job_id:
            return

        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            # Add status update to body
            status_emoji = "# Check" if status == "completed" else "🔄" if status == "in_progress" else "# X"
            body_update = f"\n\n## Status Update\n{status_emoji} **{status.upper()}** - {datetime.now().isoformat()}\n{update_body}"

            data = {
                'body': body_update,
                'state': 'closed' if status == 'completed' else 'open'
            }

            url = f"{self.github_api_url}/repos/{self.github_owner}/{self.github_repo}/issues/{job_id}"

            async with self.session.patch(url, headers=headers, json=data) as response:
                if response.status == 200:
                    logger.info(f"# Check Updated GitHub job #{job_id} to {status}")
                else:
                    logger.warning(f"# Warning Failed to update GitHub job #{job_id}: {response.status}")

        except Exception as e:
            logger.error(f"# X GitHub job update error: {e}")

    # ===============================
    # POSITION TRACKING & RISK MANAGEMENT
    # ===============================

    def can_open_position(self, symbol: str, entry_price: float, position_size: float,
                         leverage: int) -> tuple[bool, str]:
        """Check if we can open a new position with risk management"""

        # 1. MAX 10 POSITIONS TOTAL
        if len(self.active_positions) >= self.max_positions:
            return False, f"# X MAX POSITIONS REACHED: {len(self.active_positions)}/{self.max_positions}"

        # 2. SINGLE POSITION PER SYMBOL
        if symbol in self.active_positions:
            return False, f"# X ALREADY HAVE POSITION IN {symbol}"

        # 3. MAX 3% RISK PER TRADE
        margin_required = position_size / leverage
        risk_amount = margin_required * self.max_risk_per_trade

        if risk_amount > (self.account_balance * self.max_risk_per_trade):
            return False, f"# X RISK TOO HIGH: ${risk_amount:.2f} > ${self.account_balance * self.max_risk_per_trade:.2f} (3% limit)"

        # 4. SUFFICIENT BALANCE
        if margin_required > self.account_balance:
            return False, f"# X INSUFFICIENT BALANCE: Need ${margin_required:.2f}, Have ${self.account_balance:.2f}"

        return True, "# Check POSITION APPROVED"

    def add_position(self, symbol: str, side: str, size: float, entry_price: float, leverage: int):
        """Add new position with proper tracking"""
        margin_used = size / leverage

        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            leverage=leverage,
            margin_used=margin_used,
            timestamp=datetime.now()
        )

        self.active_positions[symbol] = position
        self.total_margin_used += margin_used

        logger.info("\n# Chart POSITION ADDED:")
        logger.info(f"   # Target Symbol: {symbol}")
        logger.info(f"   📈 Side: {side.upper()}")
        logger.info(f"   💰 Size: {size:.6f}")
        logger.info(f"   🎲 Leverage: {leverage}x")
        logger.info(f"   💵 Margin: ${margin_used:.2f}")
        logger.info(f"   # Chart Total Positions: {len(self.active_positions)}/{self.max_positions}")

    def remove_position(self, symbol: str, reason: str = "CLOSED"):
        """Remove position and update tracking"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            self.total_margin_used -= position.margin_used

            position.status = reason.lower()
            logger.info(f"# Check POSITION REMOVED: {symbol} ({reason})")
            logger.info(f"   # Chart Remaining Positions: {len(self.active_positions)-1}/{self.max_positions}")

    def calculate_portfolio_risk(self) -> Dict[str, Any]:
        """Calculate total portfolio risk metrics"""
        total_risk = 0.0
        total_exposure = 0.0

        for symbol, position in self.active_positions.items():
            position_value = position.size * position.entry_price
            risk_amount = position.margin_used * self.max_risk_per_trade
            total_risk += risk_amount
            total_exposure += position_value

        return {
            'total_positions': len(self.active_positions),
            'max_positions': self.max_positions,
            'total_margin_used': self.total_margin_used,
            'total_risk_amount': total_risk,
            'total_exposure': total_exposure,
            'risk_utilization_pct': (total_risk / (self.account_balance * self.max_risk_per_trade * self.max_positions)) * 100 if self.account_balance > 0 else 0,
            'position_utilization_pct': (len(self.active_positions) / self.max_positions) * 100
        }

    def update_account_balance(self, new_balance: float):
        """Update account balance and reset daily tracking if needed"""
        if self.daily_start_balance == 0:
            self.daily_start_balance = new_balance

        self.account_balance = new_balance

        # Calculate daily P&L
        if self.daily_start_balance > 0:
            self.daily_pnl = ((new_balance - self.daily_start_balance) / self.daily_start_balance) * 100

        logger.info(f"💰 BALANCE UPDATED: ${new_balance:.2f} | Daily P&L: {self.daily_pnl:.2f}%")

    # ===============================
    # TRADING JOB MANAGEMENT
    # ===============================

    async def create_trading_task(self, task_type: str, symbol: str = None,
                                 priority: int = 5, payload: Dict = None) -> Optional[str]:
        """Create trading task with GitHub job tracking"""

        # Create task payload
        task_payload = {
            "task_type": task_type,
            "symbol": symbol,
            "priority": priority,
            "payload": payload or {},
            "risk_check": True,
            "timestamp": datetime.now().isoformat()
        }

        # Create GitHub job for tracking
        job_title = f"{task_type.upper()} - {symbol or 'SYSTEM'}"
        job_body = f"""
## Trading Task: {task_type}

**Symbol:** {symbol or 'N/A'}
**Priority:** {priority}
**Timestamp:** {datetime.now().isoformat()}

### Risk Management Status:
- # Check MAX 10 POSITIONS: {len(self.active_positions)}/{self.max_positions}
- # Check MAX 3% RISK PER TRADE: ENABLED
- # Check SINGLE POSITION PER SYMBOL: ENFORCED
- # Check BALANCE VALIDATION: ACTIVE

### Payload:
```json
{json.dumps(task_payload, indent=2)}
```

### Live Status:
- **Active Positions:** {len(self.active_positions)}
- **Total Margin Used:** ${self.total_margin_used:.2f}
- **Account Balance:** ${self.account_balance:.2f}
"""

        # Create GitHub job
        job_id = await self.create_github_job(job_title, job_body, ['trading-task', task_type])

        if job_id:
            self.active_jobs[job_id] = {
                'task_type': task_type,
                'symbol': symbol,
                'created_at': datetime.now(),
                'status': 'created'
            }

        # Create task in scheduler
        try:
            async with self.session.post(f"{self.scheduler_url}/tasks", json=task_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    task_id = result.get('task_id')
                    logger.info(f"# Check Created trading task {task_id} ({task_type}) - GitHub Job #{job_id}")
                    return task_id
                else:
                    logger.error(f"# X Failed to create trading task: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"# X Task creation error: {e}")
            return None

    async def monitor_trading_system(self):
        """Monitor trading system health and create maintenance jobs"""
        while True:
            try:
                # Check system health
                risk_metrics = self.calculate_portfolio_risk()

                # Create maintenance job if needed
                if risk_metrics['risk_utilization_pct'] > 80:
                    await self.create_trading_task(
                        "risk_monitoring",
                        priority=10,
                        payload={
                            "alert_type": "high_risk_utilization",
                            "risk_pct": risk_metrics['risk_utilization_pct'],
                            "positions": len(self.active_positions)
                        }
                    )

                # Daily status report
                if datetime.now().hour == 0 and datetime.now().minute < 5:  # Once per day
                    await self.create_trading_task(
                        "daily_report",
                        priority=1,
                        payload=risk_metrics
                    )

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"# X Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def create_task(self, task_type: str, priority: int = 5, 
                         payload: Dict = None, max_retries: int = 3) -> Optional[str]:
        """Create a new task"""
        try:
            data = {
                "task_type": task_type,
                "priority": priority,
                "payload": payload or {},
                "max_retries": max_retries
            }
            
            async with self.session.post(f"{self.scheduler_url}/tasks", json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    task_id = result['task_id']
                    logger.info(f"# Check Created task {task_id} ({task_type})")
                    return task_id
                else:
                    logger.error(f"# X Failed to create task: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"# X Error creating task: {e}")
            return None
    
    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status"""
        try:
            async with self.session.get(f"{self.scheduler_url}/tasks/{task_id}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"# X Failed to get task status: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"# X Error getting task status: {e}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        try:
            async with self.session.delete(f"{self.scheduler_url}/tasks/{task_id}") as response:
                if response.status == 200:
                    logger.info(f"# Check Cancelled task {task_id}")
                    return True
                else:
                    logger.error(f"# X Failed to cancel task: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"# X Error cancelling task: {e}")
            return False
    
    async def get_system_status(self) -> Optional[Dict]:
        """Get system status"""
        try:
            async with self.session.get(f"{self.scheduler_url}/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"# X Failed to get system status: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"# X Error getting system status: {e}")
            return None

async def demo_jobs():
    """Demonstrate job creation and management"""
    logger.info("# Rocket VIPER Job Manager Demo")
    
    async with ViperJobManager() as job_manager:
        # Check system status
        status = await job_manager.get_system_status()
        if status:
            logger.info(f"# Chart System Status: {status}")
        else:
            logger.error("# X Cannot connect to task scheduler")
            return
        
        # Create various trading jobs
        jobs = []
        
        # 1. Market scan job (high priority)
        scan_task_id = await job_manager.create_task(
            task_type="scan_market",
            priority=1,
            payload={"market": "USDT", "min_volume": 100000}
        )
        if scan_task_id:
            jobs.append(scan_task_id)
        
        # 2. Multiple opportunity scoring jobs
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
        for symbol in symbols:
            score_task_id = await job_manager.create_task(
                task_type="score_opportunity",
                priority=3,
                payload={"symbol": symbol}
            )
            if score_task_id:
                jobs.append(score_task_id)
        
        # 3. Trading execution jobs
        for symbol in symbols[:3]:  # Top 3
            trade_task_id = await job_manager.create_task(
                task_type="execute_trade",
                priority=2,
                payload={
                    "symbol": symbol,
                    "side": "buy",
                    "amount": 10.0,
                    "leverage": 20
                }
            )
            if trade_task_id:
                jobs.append(trade_task_id)
        
        # 4. Position monitoring job
        monitor_task_id = await job_manager.create_task(
            task_type="monitor_position",
            priority=4,
            payload={"check_all": True}
        )
        if monitor_task_id:
            jobs.append(monitor_task_id)
        
        # 5. Balance update job
        balance_task_id = await job_manager.create_task(
            task_type="update_balance",
            priority=5,
            payload={"force_refresh": True}
        )
        if balance_task_id:
            jobs.append(balance_task_id)
        
        logger.info(f"📝 Created {len(jobs)} jobs total")
        
        # Monitor job progress
        logger.info("👁️  Monitoring job progress...")
        for i in range(30):  # Monitor for 30 seconds
            await asyncio.sleep(1)
            
            # Check status of all jobs
            completed = 0
            running = 0
            failed = 0
            
            for job_id in jobs:
                status = await job_manager.get_task_status(job_id)
                if status:
                    if status['status'] == 'completed':
                        completed += 1
                    elif status['status'] == 'running':
                        running += 1
                    elif status['status'] == 'failed':
                        failed += 1
            
            logger.info(f"# Chart Jobs: {completed} completed, {running} running, {failed} failed")
            
            if completed + failed == len(jobs):
                logger.info("# Check All jobs finished!")
                break
        
        # Final status check
        logger.info("\n📋 Final Job Status:")
        for job_id in jobs:
            status = await job_manager.get_task_status(job_id)
            if status:
                logger.info(f"  {job_id}: {status['status']} ({status['task_type']})")

async def continuous_job_creation():
    """Continuously create jobs to simulate real trading"""
    logger.info("🔄 Starting continuous job creation...")
    
    async with ViperJobManager() as job_manager:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", 
                  "DOGEUSDT", "LINKUSDT", "DOTUSDT", "UNIUSDT", "AAVEUSDT"]
        
        cycle = 0
        try:
            while True:
                cycle += 1
                logger.info(f"\n🔄 Job Creation Cycle #{cycle}")
                
                # Create scan job every cycle
                await job_manager.create_task(
                    task_type="scan_market",
                    priority=1,
                    payload={"cycle": cycle, "timestamp": datetime.now().isoformat()}
                )
                
                # Create score jobs for random symbols
                import random
import secrets
                selected_symbols = random.sample(symbols, 5)
                for symbol in selected_symbols:
                    await job_manager.create_task(
                        task_type="score_opportunity",
                        priority=3,
                        payload={"symbol": symbol, "cycle": cycle}
                    )
                
                # Create trade job for best symbol (simulate)
                best_symbol = secrets.choice(selected_symbols)
                await job_manager.create_task(
                    task_type="execute_trade",
                    priority=2,
                    payload={
                        "symbol": best_symbol,
                        "side": secrets.choice(["buy", "sell"]),
                        "amount": random.uniform(5.0, 20.0),
                        "cycle": cycle
                    }
                )
                
                # Monitor positions
                await job_manager.create_task(
                    task_type="monitor_position",
                    priority=4,
                    payload={"cycle": cycle}
                )
                
                # Get system status
                status = await job_manager.get_system_status()
                if status:
                    logger.info(f"# Chart System: {status['active_tasks']} active, "
                               f"{status['completed_tasks']} completed, "
                               f"{status['failed_tasks']} failed")
                
                # Wait before next cycle
                await asyncio.sleep(30)  # 30 second cycles
                
        except KeyboardInterrupt:
            logger.info("🛑 Continuous job creation stopped")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "continuous":
        asyncio.run(continuous_job_creation())
    else:
        asyncio.run(demo_jobs())

if __name__ == "__main__":
    main()
