#!/usr/bin/env python3
"""
🚀 VIPER LIVE OPTIMIZED TRADING BOT
Complete AI/ML-powered trading system with live optimization
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import subprocess
import signal
import sys
import os

# Import our custom modules
from ai_ml_optimizer import AIMLOptimizer
from comprehensive_backtester import ComprehensiveBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('viper_live_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ViperLiveOptimized:
    """Complete AI/ML-powered live trading system"""

    def __init__(self):
        self.ai_optimizer = AIMLOptimizer()
        self.backtester = ComprehensiveBacktester()

        # System state
        self.is_running = False
        self.optimization_active = False
        self.last_optimization = None
        self.optimization_interval = 3600  # 1 hour

        # Performance tracking
        self.performance_history = []
        self.current_parameters = {
            'entry_threshold': 0.7,
            'stop_loss_percent': 0.02,
            'take_profit_percent': 0.06,
            'trailing_stop_percent': 0.01,
            'position_size_percent': 0.02
        }

        # Risk management
        self.emergency_stop = False
        self.daily_loss_limit = 0.03
        self.max_drawdown_limit = 0.15

        logger.info("🚀 VIPER Live Optimized Trading Bot initialized")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 SHUTDOWN SIGNAL RECEIVED: {signum}")
        self.emergency_stop = True
        self.is_running = False

    async def check_system_readiness(self) -> bool:
        """Check if all required services are ready"""
        print("🔍 CHECKING SYSTEM READINESS...")

        services = {
            'API Server': 'http://localhost:8000/health',
            'Risk Manager': 'http://localhost:8002/health',
            'Exchange Connector': 'http://localhost:8005/health',
            'Ultra Backtester': 'http://localhost:8001/health',
            'Signal Processor': 'http://localhost:8011/health',
            'Order Lifecycle': 'http://localhost:8013/health'
        }

        ready_services = 0
        for name, url in services.items():
            try:
                import requests
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"   ✅ {name}: Ready")
                    ready_services += 1
                else:
                    print(f"   ❌ {name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"   ❌ {name}: {e}")

        readiness = ready_services / len(services)
        print(f"   📊 System Readiness: {ready_services}/{len(services)} ({readiness:.1%})")

        return readiness >= 0.8  # 80% readiness threshold

    async def run_initial_backtest(self) -> Dict[str, Any]:
        """Run initial comprehensive backtest to establish baseline"""
        print("🔬 RUNNING INITIAL COMPREHENSIVE BACKTEST...")

        try:
            backtest_results = self.backtester.run_multi_scenario_backtest()

            if 'error' in backtest_results:
                print(f"❌ Initial backtest failed: {backtest_results['error']}")
                return {}

            # Extract best performing scenario
            best_scenario = backtest_results.get('best_scenario')
            if best_scenario and best_scenario in backtest_results.get('scenario_results', {}):
                best_result = backtest_results['scenario_results'][best_scenario]

                print("🎯 BEST BASELINE SCENARIO FOUND:")
                print(f"   Scenario: {best_scenario}")
                print(f"   Win Rate: {best_result.get('win_rate', 0):.1%}")
                print(f"   Total Return: {best_result.get('total_return', 0):.2%}")
                print(f"   Sharpe Ratio: {best_result.get('sharpe_ratio', 0):.2f}")
                print(f"   Max Drawdown: {best_result.get('max_drawdown', 0):.2%}")

                return best_result
            else:
                print("⚠️ No optimal baseline found, using default parameters")
                return {}

        except Exception as e:
            logger.error(f"❌ Error in initial backtest: {e}")
            return {}

    async def optimize_parameters(self) -> Dict[str, Any]:
        """Run AI/ML optimization for trading parameters"""
        print("🤖 RUNNING AI/ML PARAMETER OPTIMIZATION...")

        try:
            # Collect current market data
            market_data = self.ai_optimizer.collect_market_data()
            if market_data.empty:
                print("⚠️ No market data available for optimization")
                return self.current_parameters

            # Run AI/ML optimization
            entry_opt = self.ai_optimizer.optimize_entry_points(market_data)
            tp_sl_opt = self.ai_optimizer.optimize_tp_sl_levels(
                market_data,
                market_data.iloc[-1]['close']
            )

            # Update current parameters based on AI recommendations
            optimized_params = self.current_parameters.copy()

            if entry_opt and 'optimal_threshold' in entry_opt:
                optimized_params['entry_threshold'] = entry_opt['optimal_threshold']

            if tp_sl_opt:
                if 'sl_percent' in tp_sl_opt:
                    optimized_params['stop_loss_percent'] = tp_sl_opt['sl_percent']
                if 'tp_percent' in tp_sl_opt:
                    optimized_params['take_profit_percent'] = tp_sl_opt['tp_percent']
                if 'trailing_stop_percent' in optimized_params and 'trailing_stop_percent' in tp_sl_opt:
                    optimized_params['trailing_stop_percent'] = tp_sl_opt.get('trailing_stop_percent', 0.01)

            print("🎯 OPTIMIZATION RESULTS:")
            print(f"   Entry Threshold: {optimized_params['entry_threshold']:.2f}")
            print(f"   Stop Loss: {optimized_params['stop_loss_percent']:.1%}")
            print(f"   Take Profit: {optimized_params['take_profit_percent']:.1%}")
            print(f"   Trailing Stop: {optimized_params['trailing_stop_percent']:.1%}")

            self.current_parameters = optimized_params
            return optimized_params

        except Exception as e:
            logger.error(f"❌ Error in parameter optimization: {e}")
            return self.current_parameters

    async def validate_optimization(self, optimized_params: Dict[str, Any]) -> bool:
        """Validate optimized parameters through quick backtest"""
        print("🔍 VALIDATING OPTIMIZED PARAMETERS...")

        try:
            # Create validation scenario
            validation_scenario = {
                'name': 'Optimized_Validation',
                'entry_threshold': optimized_params['entry_threshold'],
                'stop_loss_percent': optimized_params['stop_loss_percent'],
                'take_profit_percent': optimized_params['take_profit_percent'],
                'trailing_stop_percent': optimized_params['trailing_stop_percent'],
                'position_size_percent': optimized_params['position_size_percent'],
                'max_positions': 5,
                'use_ml_optimization': True
            }

            # Run quick validation backtest
            validation_result = self.backtester.run_single_backtest("BTCUSDT", validation_scenario)

            if 'error' in validation_result:
                print(f"❌ Validation failed: {validation_result['error']}")
                return False

            # Check if performance meets minimum criteria
            win_rate = validation_result.get('win_rate', 0)
            sharpe_ratio = validation_result.get('sharpe_ratio', 0)
            max_drawdown = validation_result.get('max_drawdown', 1)

            validation_passed = (
                win_rate >= 0.50 and  # Minimum 50% win rate
                sharpe_ratio >= 0.5 and  # Minimum Sharpe ratio
                max_drawdown <= 0.20  # Maximum 20% drawdown
            )

            print("📊 VALIDATION RESULTS:")
            print(f"   Win Rate: {win_rate:.1%} {'✅' if win_rate >= 0.50 else '❌'}")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f} {'✅' if sharpe_ratio >= 0.5 else '❌'}")
            print(f"   Max Drawdown: {max_drawdown:.1%} {'✅' if max_drawdown <= 0.20 else '❌'}")

            if validation_passed:
                print("✅ OPTIMIZED PARAMETERS VALIDATED!")
                return True
            else:
                print("❌ Validation failed - reverting to previous parameters")
                return False

        except Exception as e:
            logger.error(f"❌ Error in validation: {e}")
            return False

    async def apply_trading_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Apply optimized parameters to the trading system"""
        print("⚙️ APPLYING OPTIMIZED TRADING PARAMETERS...")

        try:
            # Update risk manager configuration
            config_update = {
                'default_stop_loss_percent': parameters['stop_loss_percent'],
                'default_take_profit_percent': parameters['take_profit_percent'],
                'trailing_activation_percent': parameters['trailing_stop_percent'],
                'max_positions': 15,
                'risk_per_trade': parameters['position_size_percent'] * 100,  # Convert to percentage
                'max_position_size_percent': parameters['position_size_percent']
            }

            import requests
            response = requests.post(
                'http://localhost:8002/api/tp-sl-tsl/config',
                json=config_update,
                timeout=10
            )

            if response.status_code == 200:
                print("✅ Risk manager parameters updated")
            else:
                print(f"⚠️ Risk manager update failed: {response.status_code}")

            # Update signal processor entry threshold
            signal_config = {
                'entry_threshold': parameters['entry_threshold'],
                'confidence_threshold': parameters['entry_threshold']
            }

            response = requests.post(
                'http://localhost:8011/api/config',
                json=signal_config,
                timeout=10
            )

            if response.status_code == 200:
                print("✅ Signal processor parameters updated")
            else:
                print(f"⚠️ Signal processor update failed: {response.status_code}")

            print("🎯 TRADING PARAMETERS SUCCESSFULLY APPLIED!")
            return True

        except Exception as e:
            logger.error(f"❌ Error applying parameters: {e}")
            return False

    async def monitor_performance(self) -> Dict[str, Any]:
        """Monitor live trading performance"""
        try:
            # Get current performance metrics
            import requests

            # Account balance
            balance_response = requests.get('http://localhost:8005/api/balance', timeout=5)
            balance = 0
            if balance_response.status_code == 200:
                balance_data = balance_response.json()
                balance = balance_data.get('free', 0)

            # Active positions
            positions_response = requests.get('http://localhost:8002/api/tp-sl-tsl/positions', timeout=5)
            active_positions = 0
            if positions_response.status_code == 200:
                positions_data = positions_response.json()
                active_positions = len(positions_data.get('positions', {}))

            # Risk metrics
            risk_response = requests.get('http://localhost:8002/api/tp-sl-tsl/config', timeout=5)
            risk_config = {}
            if risk_response.status_code == 200:
                risk_config = risk_response.json()

            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'account_balance': balance,
                'active_positions': active_positions,
                'risk_config': risk_config
            }

            # Store in performance history
            self.performance_history.append(performance_data)

            # Keep only last 1000 entries
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

            return performance_data

        except Exception as e:
            logger.error(f"❌ Error monitoring performance: {e}")
            return {}

    async def check_risk_limits(self) -> bool:
        """Check if risk limits have been breached"""
        try:
            if len(self.performance_history) < 2:
                return False

            # Calculate daily P&L
            current_balance = self.performance_history[-1].get('account_balance', 0)
            previous_balance = self.performance_history[-2].get('account_balance', current_balance)

            if previous_balance > 0:
                daily_pnl = (current_balance - previous_balance) / previous_balance

                if daily_pnl < -self.daily_loss_limit:
                    logger.critical(f"🚨 DAILY LOSS LIMIT BREACHED: {daily_pnl:.2%}")
                    self.emergency_stop = True
                    return True

            return False

        except Exception as e:
            logger.error(f"❌ Error checking risk limits: {e}")
            return False

    async def continuous_optimization_loop(self):
        """Main continuous optimization loop"""
        print("🔄 STARTING CONTINUOUS OPTIMIZATION LOOP...")

        while self.is_running and not self.emergency_stop:
            try:
                current_time = datetime.now()

                # Check risk limits
                if await self.check_risk_limits():
                    break

                # Run periodic optimization
                if (self.last_optimization is None or
                    (current_time - self.last_optimization).total_seconds() >= self.optimization_interval):

                    print(f"\n🔧 RUNNING PERIODIC OPTIMIZATION ({current_time.strftime('%H:%M:%S')})")

                    # Optimize parameters
                    optimized_params = await self.optimize_parameters()

                    # Validate optimization
                    if await self.validate_optimization(optimized_params):
                        # Apply optimized parameters
                        if await self.apply_trading_parameters(optimized_params):
                            self.last_optimization = current_time
                            print("✅ OPTIMIZATION CYCLE COMPLETED")
                        else:
                            print("⚠️ Failed to apply optimized parameters")
                    else:
                        print("⚠️ Optimization validation failed")

                # Monitor performance
                performance = await self.monitor_performance()

                # Display status update every 5 minutes
                if int(time.time()) % 300 == 0:
                    print("📊 STATUS UPDATE:")
                    print(f"   Balance: ${performance.get('account_balance', 0):.2f}")
                    print(f"   Active Positions: {performance.get('active_positions', 0)}")
                    print(f"   Last Optimization: {self.last_optimization.strftime('%H:%M:%S') if self.last_optimization else 'Never'}")

                # Brief pause to prevent overwhelming the system
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"❌ Error in optimization loop: {e}")
                await asyncio.sleep(60)

    async def run_live_trading_system(self):
        """Run the complete live trading system"""
        print("🚀 VIPER LIVE OPTIMIZED TRADING SYSTEM")
        print("=" * 60)
        print("🤖 AI/ML-Powered | 📊 Real-Time Optimization | 🛡️ Risk Management")
        print("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Step 1: Check system readiness
            if not await self.check_system_readiness():
                print("❌ System not ready. Please ensure all services are running.")
                return

            # Step 2: Run initial comprehensive backtest
            baseline_results = await self.run_initial_backtest()
            if baseline_results:
                print("📈 BASELINE PERFORMANCE ESTABLISHED")
                print(f"   Expected Win Rate: {baseline_results.get('win_rate', 0):.1%}")
                print(f"   Expected Return: {baseline_results.get('total_return', 0):.2%}")
                print(f"   Risk (Max Drawdown): {baseline_results.get('max_drawdown', 0):.2%}")

            # Step 3: Initial AI/ML optimization
            print("🎯 INITIAL AI/ML OPTIMIZATION...")
            initial_params = await self.optimize_parameters()

            if await self.validate_optimization(initial_params):
                await self.apply_trading_parameters(initial_params)
                print("✅ INITIAL OPTIMIZATION COMPLETED")
            else:
                print("⚠️ Using default parameters for initial setup")

            # Step 4: Start continuous optimization
            print("🔄 STARTING CONTINUOUS OPTIMIZATION...")
            print("📊 System will optimize parameters every hour")
            print("🛑 Emergency stop: Ctrl+C")
            print("-" * 60)

            self.is_running = True
            self.last_optimization = datetime.now()

            # Run the continuous optimization loop
            await self.continuous_optimization_loop()

        except KeyboardInterrupt:
            print("⏹️ System shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Fatal error in live trading system: {e}")
            print(f"\n❌ System error: {e}")
        finally:
            self.is_running = False

            print("\n" + "=" * 60)
            print("🛑 VIPER LIVE OPTIMIZED TRADING SYSTEM SHUTDOWN")
            print("=" * 60)

            # Generate final performance report
            if self.performance_history:
                print("📊 FINAL PERFORMANCE SUMMARY:")
                initial_balance = self.performance_history[0].get('account_balance', 0)
                final_balance = self.performance_history[-1].get('account_balance', 0)

                if initial_balance > 0:
                    total_return = (final_balance - initial_balance) / initial_balance
                    print(f"   Initial Balance: ${initial_balance:.2f}")
                    print(f"   Final Balance: ${final_balance:.2f}")
                    print(f"   Total Return: {total_return:.2%}")

                print(f"   Total Monitoring Points: {len(self.performance_history)}")
                print(f"   Optimization Cycles: {1 if self.last_optimization else 0}")

            print("✅ SYSTEM SAFELY SHUTDOWN")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--backtest-only":
            # Run comprehensive backtest only
            backtester = ComprehensiveBacktester()
            backtester.run_multi_scenario_backtest()
            return
        elif sys.argv[1] == "--optimize-only":
            # Run AI/ML optimization only
            optimizer = AIMLOptimizer()
            optimizer.run_comprehensive_backtest()
            return

    # Run complete live optimized system
    system = ViperLiveOptimized()
    asyncio.run(system.run_live_trading_system())

if __name__ == "__main__":
    main()
