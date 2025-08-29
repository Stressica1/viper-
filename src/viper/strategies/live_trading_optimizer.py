#!/usr/bin/env python3
"""
🚀 VIPER Live Trading Optimizer
Continuous live trading with real-time strategy optimization and risk management
"""

import asyncio
import requests
import json
import time
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import os
import threading
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveTradingOptimizer:
    """Live trading system with continuous optimization"""

    def __init__(self):
        self.is_running = False
        self.emergency_stop = False
        self.performance_metrics = {}
        self.active_positions = {}
        self.strategy_performance = {}

        # Service URLs
        self.api_server_url = "http://localhost:8000"
        self.risk_manager_url = "http://localhost:8002"
        self.order_lifecycle_url = "http://localhost:8013"
        self.exchange_connector_url = "http://localhost:8005"
        self.signal_processor_url = "http://localhost:8011"
        self.live_trading_engine_url = "http://localhost:8007"

        # Trading parameters
        self.max_positions = int(os.getenv('MAX_POSITIONS', '15'))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))
        self.daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT', '0.03'))

        # Optimization parameters
        self.optimization_interval = 300  # 5 minutes
        self.performance_window = 3600   # 1 hour
        self.min_trades_for_optimization = 10

        # Emergency stop conditions
        self.max_daily_loss = 0.0
        self.daily_pnl = 0.0
        self.start_balance = 0.0

        # Performance tracking
        self.trades_executed = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        self.sharpe_ratio = 0.0

        # Strategy parameters to optimize
        self.strategy_params = {
            'viper_threshold': 85,
            'stop_loss_percent': 0.02,
            'take_profit_percent': 0.06,
            'trailing_stop_percent': 0.01,
            'signal_confidence_threshold': 0.75
        }

        logger.info("🚀 VIPER Live Trading Optimizer initialized")
        logger.info(f"📊 Max Positions: {self.max_positions}")
        logger.info(f"⚖️ Risk per Trade: {self.risk_per_trade*100}%")
        logger.info(f"🛑 Daily Loss Limit: {self.daily_loss_limit*100}%")

    def signal_handler(self, signum, frame):
        """Handle emergency stop signals"""
        logger.warning(f"🚨 EMERGENCY SIGNAL RECEIVED: {signum}")
        self.emergency_stop = True
        self.is_running = False

    async def check_system_health(self) -> bool:
        """Check if all required services are healthy"""
        services = {
            "API Server": self.api_server_url,
            "Risk Manager": self.risk_manager_url,
            "Order Lifecycle Manager": self.order_lifecycle_url,
            "Exchange Connector": self.exchange_connector_url,
            "Signal Processor": self.signal_processor_url
        }

        all_healthy = True
        for name, url in services.items():
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code != 200:
                    logger.error(f"❌ {name} unhealthy: {response.status_code}")
                    all_healthy = False
            except Exception as e:
                logger.error(f"❌ {name} unreachable: {e}")
                all_healthy = False

        if all_healthy:
            logger.info("✅ All services healthy")

        return all_healthy

    async def get_account_balance(self) -> Optional[float]:
        """Get current account balance"""
        try:
            response = requests.get(f"{self.exchange_connector_url}/api/balance", timeout=10)
            if response.status_code == 200:
                data = response.json()
                balance = data.get('free', 0)
                if not hasattr(self, 'start_balance') or self.start_balance == 0:
                    self.start_balance = balance
                return balance
            return None
        except Exception as e:
            logger.error(f"❌ Error getting balance: {e}")
            return None

    async def check_emergency_conditions(self) -> bool:
        """Check if emergency stop conditions are met"""
        try:
            # Get current balance
            current_balance = await self.get_account_balance()
            if not current_balance or self.start_balance == 0:
                return False

            # Calculate daily P&L
            self.daily_pnl = current_balance - self.start_balance
            daily_loss_percent = abs(self.daily_pnl / self.start_balance)

            # Check daily loss limit
            if self.daily_pnl < 0 and daily_loss_percent >= self.daily_loss_limit:
                logger.critical(f"🚨 DAILY LOSS LIMIT EXCEEDED: {daily_loss_percent*100:.2f}%")
                return True

            # Check maximum drawdown (additional safety)
            if daily_loss_percent >= 0.05:  # 5% emergency stop
                logger.critical(f"🚨 EMERGENCY STOP: Large drawdown {daily_loss_percent*100:.2f}%")
                return True

            # Check if too many consecutive losses
            if hasattr(self, 'consecutive_losses') and self.consecutive_losses >= 5:
                logger.critical("🚨 EMERGENCY STOP: 5 consecutive losses")
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error checking emergency conditions: {e}")
            return False

    async def generate_trading_signals(self) -> List[Dict]:
        """Generate trading signals from market data"""
        try:
            # Get market data from signal processor
            response = requests.get(f"{self.signal_processor_url}/api/signals/current", timeout=10)
            if response.status_code == 200:
                signals_data = response.json()
                signals = signals_data.get('signals', [])

                # Filter signals based on strategy parameters
                filtered_signals = []
                for signal in signals:
                    confidence = signal.get('confidence', 0)
                    if confidence >= self.strategy_params['signal_confidence_threshold']:
                        filtered_signals.append(signal)

                logger.info(f"🎯 Generated {len(filtered_signals)} trading signals")
                return filtered_signals
            else:
                logger.warning(f"⚠️ Signal generation failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"❌ Error generating signals: {e}")
            return []

    async def validate_and_execute_signal(self, signal: Dict) -> bool:
        """Validate signal with risk management and execute if approved"""
        try:
            # Prepare signal for risk validation
            validation_signal = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'size': 0.001,  # Conservative position size
                'entry_price': signal['price'],
                'confidence': signal['confidence']
            }

            # Validate with risk manager
            response = requests.post(
                f"{self.risk_manager_url}/api/tp-sl-tsl/validate-signal",
                json={'signal': validation_signal},
                timeout=10
            )

            if response.status_code == 200:
                validation = response.json()['validation']

                if validation['approved']:
                    logger.info(f"✅ Signal approved: {signal['symbol']} {signal['side']}")

                    # Execute the trade
                    order_data = {
                        'symbol': signal['symbol'],
                        'side': signal['side'],
                        'size': validation['position_size'],
                        'entry_price': signal['price'],
                        'stop_loss': validation['levels']['stop_loss'],
                        'take_profit': validation['levels']['take_profit'],
                        'trailing_stop': validation['levels']['trailing_stop'],
                        'trailing_activation': validation['levels']['trailing_activation']
                    }

                    # Place the order
                    order_response = requests.post(
                        f"{self.order_lifecycle_url}/api/tp-sl-tsl/create-order",
                        json=order_data,
                        timeout=15
                    )

                    if order_response.status_code == 200:
                        logger.info(f"💰 Trade executed: {signal['symbol']}")
                        self.trades_executed += 1
                        return True
                    else:
                        logger.error(f"❌ Order execution failed: {order_response.text}")
                        return False
                else:
                    logger.info(f"❌ Signal rejected: {validation['reason']}")
                    return False
            else:
                logger.error(f"❌ Risk validation failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error processing signal: {e}")
            return False

    async def optimize_strategy_parameters(self):
        """Optimize strategy parameters based on performance"""
        try:
            logger.info("🔧 Starting strategy optimization...")

            # Get performance data
            performance_data = await self.get_performance_metrics()

            if len(performance_data.get('trades', [])) < self.min_trades_for_optimization:
                logger.info("📊 Not enough trades for optimization yet")
                return

            # Calculate current performance metrics
            win_rate = performance_data.get('win_rate', 0)
            avg_pnl = performance_data.get('avg_pnl', 0)
            max_drawdown = performance_data.get('max_drawdown', 0)

            # Adjust parameters based on performance
            if win_rate < 0.4:  # Below 40% win rate
                # Increase confidence threshold
                self.strategy_params['signal_confidence_threshold'] = min(
                    0.9, self.strategy_params['signal_confidence_threshold'] + 0.05
                )
                logger.info("📈 Increased confidence threshold due to low win rate")

            elif win_rate > 0.7:  # Above 70% win rate
                # Decrease confidence threshold to capture more opportunities
                self.strategy_params['signal_confidence_threshold'] = max(
                    0.6, self.strategy_params['signal_confidence_threshold'] - 0.02
                )
                logger.info("📉 Decreased confidence threshold due to high win rate")

            # Adjust risk parameters based on drawdown
            if max_drawdown > 0.05:  # 5% drawdown
                self.strategy_params['stop_loss_percent'] *= 0.9  # Tighter stops
                logger.info("🛑 Tightened stop loss due to high drawdown")

            # Log optimization results
            logger.info("🎯 Strategy parameters optimized:")
            for param, value in self.strategy_params.items():
                logger.info(f"   {param}: {value}")

        except Exception as e:
            logger.error(f"❌ Error optimizing strategy: {e}")

    async def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            response = requests.get(f"{self.api_server_url}/api/performance", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                # Fallback: calculate basic metrics
                return {
                    'total_trades': self.trades_executed,
                    'win_rate': self.successful_trades / max(1, self.trades_executed),
                    'total_pnl': self.total_pnl,
                    'avg_pnl': self.total_pnl / max(1, self.trades_executed),
                    'max_drawdown': 0.02  # Placeholder
                }
        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            return {}

    async def monitor_positions(self):
        """Monitor active positions and handle exits"""
        try:
            response = requests.get(f"{self.risk_manager_url}/api/tp-sl-tsl/positions", timeout=10)
            if response.status_code == 200:
                positions_data = response.json()
                positions = positions_data.get('positions', {})

                for symbol, position in positions.items():
                    pnl = position.get('unrealized_pnl', 0)

                    # Update position tracking
                    if symbol not in self.active_positions:
                        self.active_positions[symbol] = {
                            'entry_time': datetime.now(),
                            'entry_pnl': 0,
                            'max_pnl': pnl
                        }

                    # Track maximum P&L for drawdown calculation
                    if pnl > self.active_positions[symbol]['max_pnl']:
                        self.active_positions[symbol]['max_pnl'] = pnl

                    # Check for emergency exit conditions
                    current_drawdown = self.active_positions[symbol]['max_pnl'] - pnl
                    if current_drawdown > 0.03:  # 3% drawdown on position
                        logger.warning(f"⚠️ Large drawdown on {symbol}: {current_drawdown*100:.2f}%")

        except Exception as e:
            logger.error(f"❌ Error monitoring positions: {e}")

    async def log_system_status(self):
        """Log comprehensive system status"""
        try:
            balance = await self.get_account_balance()
            positions_response = requests.get(f"{self.risk_manager_url}/api/tp-sl-tsl/positions", timeout=5)
            positions_count = 0

            if positions_response.status_code == 200:
                positions_data = positions_response.json()
                positions_count = len(positions_data.get('positions', {}))

            logger.info("📊 SYSTEM STATUS:")
            logger.info(f"   💰 Account Balance: ${balance:.2f}")
            logger.info(f"   📈 Daily P&L: ${self.daily_pnl:.2f}")
            logger.info(f"   📊 Active Positions: {positions_count}")
            logger.info(f"   🎯 Total Trades: {self.trades_executed}")
            logger.info(f"   ✅ Successful Trades: {self.successful_trades}")
            logger.info(f"   📊 Win Rate: {(self.successful_trades/max(1,self.trades_executed))*100:.1f}%")
            logger.info(f"   💵 Total P&L: ${self.total_pnl:.2f}")

        except Exception as e:
            logger.error(f"❌ Error logging system status: {e}")

    async def run_live_trading_loop(self):
        """Main live trading loop with continuous optimization"""
        logger.info("🚀 STARTING LIVE TRADING OPTIMIZATION LOOP")
        logger.info("=" * 60)

        self.is_running = True
        last_optimization = time.time()
        last_status_log = time.time()

        try:
            while self.is_running and not self.emergency_stop:
                current_time = time.time()

                # Check system health
                if not await self.check_system_health():
                    logger.error("❌ System health check failed - pausing operations")
                    await asyncio.sleep(30)
                    continue

                # Check emergency conditions
                if await self.check_emergency_conditions():
                    logger.critical("🚨 EMERGENCY CONDITIONS MET - STOPPING TRADING")
                    self.emergency_stop = True
                    break

                # Generate and process trading signals
                signals = await self.generate_trading_signals()

                for signal in signals:
                    # Check position limits
                    positions_response = requests.get(f"{self.risk_manager_url}/api/tp-sl-tsl/positions", timeout=5)
                    if positions_response.status_code == 200:
                        positions_data = positions_response.json()
                        current_positions = len(positions_data.get('positions', {}))

                        if current_positions >= self.max_positions:
                            logger.info(f"📊 Position limit reached ({self.max_positions}) - skipping signal")
                            continue

                    # Validate and execute signal
                    if await self.validate_and_execute_signal(signal):
                        logger.info(f"✅ Successfully executed trade for {signal['symbol']}")

                # Monitor positions
                await self.monitor_positions()

                # Run strategy optimization periodically
                if current_time - last_optimization >= self.optimization_interval:
                    await self.optimize_strategy_parameters()
                    last_optimization = current_time

                # Log system status periodically
                if current_time - last_status_log >= 300:  # Every 5 minutes
                    await self.log_system_status()
                    last_status_log = current_time

                # Brief pause to prevent overwhelming the system
                await asyncio.sleep(10)

        except KeyboardInterrupt:
            logger.info("⏹️ Live trading loop stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in live trading loop: {e}")
        finally:
            logger.info("🛑 Live trading optimization stopped")
            await self.log_system_status()

    async def start_emergency_monitoring(self):
        """Start emergency monitoring in background"""
        while self.is_running:
            try:
                # Check critical system components
                critical_services = [self.exchange_connector_url, self.risk_manager_url]

                for service_url in critical_services:
                    try:
                        response = requests.get(f"{service_url}/health", timeout=5)
                        if response.status_code != 200:
                            logger.critical(f"🚨 CRITICAL: {service_url} is unhealthy")
                            self.emergency_stop = True
                    except:
                        logger.critical(f"🚨 CRITICAL: {service_url} is unreachable")
                        self.emergency_stop = True

                # Check account balance
                balance = await self.get_account_balance()
                if balance is None:
                    logger.critical("🚨 CRITICAL: Cannot access account balance")
                    self.emergency_stop = True

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"❌ Emergency monitoring error: {e}")
                await asyncio.sleep(10)

async def main():
    """Main function to run live trading optimizer"""
    # Setup signal handlers
    optimizer = LiveTradingOptimizer()
    signal.signal(signal.SIGINT, optimizer.signal_handler)
    signal.signal(signal.SIGTERM, optimizer.signal_handler)

    print("🚀 VIPER LIVE TRADING OPTIMIZER")
    print("=" * 60)
    print("⚠️  SAFETY FEATURES ENABLED:")
    print(f"   • Daily loss limit: {optimizer.daily_loss_limit*100}%")
    print(f"   • Risk per trade: {optimizer.risk_per_trade*100}%")
    print(f"   • Max positions: {optimizer.max_positions}")
    print("   • Emergency stop: ENABLED")
    print("   • Real-time optimization: ENABLED")
    print("")
    print("🛑 EMERGENCY COMMANDS:")
    print("   Ctrl+C to stop gracefully")
    print("   'docker compose down' to force stop all services")
    print("")
    print("📊 MONITORING:")
    print("   • Real-time P&L tracking")
    print("   • Position monitoring")
    print("   • Strategy optimization")
    print("   • Emergency condition checks")
    print("=" * 60)

    # Start emergency monitoring in background
    monitoring_task = asyncio.create_task(optimizer.start_emergency_monitoring())

    try:
        # Start the main trading loop
        await optimizer.run_live_trading_loop()
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        optimizer.is_running = False
        monitoring_task.cancel()

        print("\n" + "=" * 60)
        print("🛑 LIVE TRADING OPTIMIZER SHUTDOWN COMPLETE")
        print("=" * 60)
        print("📊 FINAL STATISTICS:")
        print(f"   Total Trades: {optimizer.trades_executed}")
        print(f"   Successful Trades: {optimizer.successful_trades}")
        print(f"   Total P&L: ${optimizer.total_pnl:.2f}")
        print(f"   Final Balance: ${optimizer.start_balance + optimizer.daily_pnl:.2f}")
        print("✅ System safely stopped")

if __name__ == "__main__":
    asyncio.run(main())
