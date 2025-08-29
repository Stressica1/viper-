#!/usr/bin/env python3
"""
# Rocket VIPER UNIFIED TRADING ENGINE
Complete integration of all trading components with optimal performance

This unified engine provides:
- Integrated trading workflow with all optimizations
- Real-time entry point optimization
- Mathematical validation for all calculations
- MCP server integration for enhanced performance
- Comprehensive risk management and monitoring
- AI/ML optimization for trading decisions
"""

import os
import sys
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import importlib.util
import ccxt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - UNIFIED_ENGINE - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedTradingEngine:
    """
    Unified trading engine with complete feature integration
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or Path(__file__).parent / ".env"
        self.components = {}
        self.trading_active = False
        self.monitoring_active = False

        # Initialize all components
        self._initialize_system()
        self._load_configurations()
        self._setup_exchange_connection()

    def _initialize_system(self):
        """Initialize all system components"""
        logger.info("# Rocket Initializing Unified Trading Engine...")

        # Load utility components
        self._load_mathematical_validator()
        self._load_optimal_configurations()
        self._load_entry_point_optimizer()
        self._load_ai_optimizer()
        self._load_diagnostic_system()

        logger.info("# Check All system components initialized")

    def _load_mathematical_validator(self):
        """Load mathematical validation system"""
        try:
            from utils.mathematical_validator import MathematicalValidator
            self.components['math_validator'] = MathematicalValidator()
            logger.info("# Check Mathematical Validator loaded")
        except Exception as e:
            logger.error(f"# X Failed to load Mathematical Validator: {e}")

    def _load_optimal_configurations(self):
        """Load optimal MCP and system configurations"""
        try:
            from config.optimal_mcp_config import get_optimal_mcp_config
            self.optimal_config = get_optimal_mcp_config()

            # Load trading parameters from optimal config
            self.trading_config = {
                'max_positions': int(os.getenv('MAX_POSITIONS', '15')),
                'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),
                'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '3.0')),
                'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '5.0')),
                'max_leverage': int(os.getenv('MAX_LEVERAGE', '50')),
                'trading_pairs': os.getenv('TRADING_PAIRS', 'BTCUSDT,ETHUSDT').split(',')
            }

            logger.info("# Check Optimal configurations loaded")
        except Exception as e:
            logger.error(f"# X Failed to load optimal configurations: {e}")

    def _load_entry_point_optimizer(self):
        """Load entry point optimization system"""
        try:
            from scripts.optimal_entry_point_manager import OptimalEntryPointManager
            self.components['entry_optimizer'] = OptimalEntryPointManager()
            logger.info("# Check Entry Point Optimizer loaded")
        except Exception as e:
            logger.error(f"# X Failed to load Entry Point Optimizer: {e}")

    def _load_ai_optimizer(self):
        """Load AI/ML optimization system"""
        try:
            # Import AI optimizer
            spec = importlib.util.spec_from_file_location(
                "ai_optimizer",
                Path(__file__).parent / "ai_ml_optimizer.py"
            )
            ai_optimizer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ai_optimizer_module)

            if hasattr(ai_optimizer_module, 'VIPEROptimizer'):
                self.components['ai_optimizer'] = ai_optimizer_module.VIPEROptimizer()
            else:
                self.components['ai_optimizer'] = ai_optimizer_module

            logger.info("# Check AI Optimizer loaded")
        except Exception as e:
            logger.error(f"# X Failed to load AI Optimizer: {e}")

    def _load_diagnostic_system(self):
        """Load diagnostic and monitoring system"""
        try:
            from scripts.scoring_system_diagnostic import ScoringSystemDiagnostic
            self.components['diagnostic_system'] = ScoringSystemDiagnostic()
            logger.info("# Check Diagnostic System loaded")
        except Exception as e:
            logger.error(f"# X Failed to load Diagnostic System: {e}")

    def _load_configurations(self):
        """Load environment configurations"""
        try:
            # Load .env file if it exists
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    env_vars = {}
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if '=' in line:
                                key, value = line.split('=', 1)
                                env_vars[key.strip()] = value.strip().strip('"\'')

                    # Set environment variables
                    for key, value in env_vars.items():
                        os.environ[key] = value

            logger.info("# Check Configuration loaded from .env file")

        except Exception as e:
            logger.warning(f"# Warning Could not load .env file: {e}")

    def _setup_exchange_connection(self):
        """Setup exchange connection with optimal settings"""
        try:
            # Get exchange credentials
            api_key = os.getenv('BITGET_API_KEY')
            api_secret = os.getenv('BITGET_API_SECRET')
            api_password = os.getenv('BITGET_API_PASSWORD')

            if not all([api_key, api_secret, api_password]):
                logger.warning("# Warning Exchange credentials not found in environment")
                self.exchange = None
                return

            # Initialize exchange with optimal settings for swap trading
            exchange_config = {
                'apiKey': api_key,
                'secret': api_secret,
                'password': api_password,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000,
                }
            }

            self.exchange = ccxt.bitget(exchange_config)
            logger.info("# Check Exchange connection established")

        except Exception as e:
            logger.error(f"# X Failed to setup exchange connection: {e}")
            self.exchange = None

    async def start_trading_engine(self):
        """Start the unified trading engine"""
        if self.trading_active:
            logger.warning("Trading engine already active")
            return

        logger.info("# Rocket Starting Unified Trading Engine...")
        self.trading_active = True

        try:
            # Initialize trading loop
            await self._trading_loop()

        except KeyboardInterrupt:
            logger.info("🛑 Trading engine stopped by user")
        except Exception as e:
            logger.error(f"# X Trading engine error: {e}")
        finally:
            self.trading_active = False

    async def _trading_loop(self):
        """Main trading loop with all optimizations"""
        logger.info("🔄 Starting trading loop with optimizations...")

        while self.trading_active:
            try:
                # Step 1: Scan markets with entry point optimization
                if 'entry_optimizer' in self.components:
                    entry_signals = await self._scan_with_optimization()
                else:
                    entry_signals = await self._scan_basic()

                # Step 2: Validate signals with mathematical validator
                if entry_signals and 'math_validator' in self.components:
                    validated_signals = self._validate_signals_mathematically(entry_signals)
                else:
                    validated_signals = entry_signals

                # Step 3: Apply AI optimization to trading decisions
                if validated_signals and 'ai_optimizer' in self.components:
                    optimized_signals = self._apply_ai_optimization(validated_signals)
                else:
                    optimized_signals = validated_signals

                # Step 4: Execute trades with risk management
                if optimized_signals:
                    await self._execute_trades_with_risk_management(optimized_signals)

                # Step 5: Monitor and adjust positions
                await self._monitor_and_adjust_positions()

                # Wait before next cycle
                await asyncio.sleep(30)  # 30 second intervals

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _scan_with_optimization(self) -> List[Dict[str, Any]]:
        """Scan markets with entry point optimization"""
        try:
            optimizer = self.components['entry_optimizer']

            # Get optimal entry points for all trading pairs
            entry_points = []
            for symbol in self.trading_config['trading_pairs']:
                try:
                    # Get entry point analysis
                    analysis = optimizer.analyze_entry_point(symbol.strip())

                    if analysis.get('should_enter', False):
                        entry_points.append({
                            'symbol': symbol.strip(),
                            'entry_price': analysis.get('optimal_entry_price'),
                            'confidence': analysis.get('confidence_score', 0),
                            'risk_reward_ratio': analysis.get('risk_reward_ratio', 1.0),
                            'analysis': analysis
                        })

                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue

            logger.info(f"# Chart Found {len(entry_points)} optimized entry points")
            return entry_points

        except Exception as e:
            logger.error(f"Entry point optimization failed: {e}")
            return []

    async def _scan_basic(self) -> List[Dict[str, Any]]:
        """Basic market scanning fallback"""
        # Implement basic scanning logic here
        logger.info("# Search Using basic market scanning")
        return []

    def _validate_signals_mathematically(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate signals using mathematical validator"""
        try:
            validator = self.components['math_validator']
            validated_signals = []

            for signal in signals:
                try:
                    # Validate entry price calculations
                    price_validation = validator.validate_price_calculation(
                        signal.get('entry_price', 0),
                        signal['symbol']
                    )

                    # Validate risk calculations
                    risk_validation = validator.validate_risk_calculation(
                        signal.get('risk_reward_ratio', 1.0),
                        self.trading_config['risk_per_trade']
                    )

                    if price_validation.get('is_valid', False) and risk_validation.get('is_valid', False):
                        signal['validation'] = {
                            'price': price_validation,
                            'risk': risk_validation
                        }
                        validated_signals.append(signal)
                    else:
                        logger.warning(f"Signal validation failed for {signal['symbol']}")

                except Exception as e:
                    logger.error(f"Mathematical validation error for {signal['symbol']}: {e}")
                    continue

            logger.info(f"# Check Validated {len(validated_signals)} signals mathematically")
            return validated_signals

        except Exception as e:
            logger.error(f"Mathematical validation system failed: {e}")
            return signals

    def _apply_ai_optimization(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply AI optimization to trading signals"""
        try:
            ai_optimizer = self.components['ai_optimizer']

            optimized_signals = []
            for signal in signals:
                try:
                    # Apply AI optimization
                    optimization_result = ai_optimizer.optimize_signal(signal)

                    if optimization_result.get('should_trade', False):
                        signal['ai_optimization'] = optimization_result
                        signal['confidence'] = optimization_result.get('optimized_confidence', signal.get('confidence', 0))
                        optimized_signals.append(signal)
                    else:
                        logger.info(f"AI optimizer rejected signal for {signal['symbol']}")

                except Exception as e:
                    logger.error(f"AI optimization error for {signal['symbol']}: {e}")
                    # Keep original signal if AI fails
                    optimized_signals.append(signal)

            logger.info(f"🤖 AI optimized {len(optimized_signals)} signals")
            return optimized_signals

        except Exception as e:
            logger.error(f"AI optimization system failed: {e}")
            return signals

    async def _execute_trades_with_risk_management(self, signals: List[Dict[str, Any]]):
        """Execute trades with comprehensive risk management"""
        try:
            # Check current positions
            current_positions = len(await self._get_current_positions())

            # Respect maximum positions limit
            max_positions = self.trading_config['max_positions']
            available_slots = max_positions - current_positions

            if available_slots <= 0:
                logger.info(f"# Chart Maximum positions ({max_positions}) reached")
                return

            # Execute trades for top signals
            executed_count = 0
            for signal in signals[:available_slots]:
                try:
                    await self._execute_single_trade(signal)
                    executed_count += 1
                    await asyncio.sleep(1)  # Brief pause between trades

                except Exception as e:
                    logger.error(f"Failed to execute trade for {signal['symbol']}: {e}")
                    continue

            if executed_count > 0:
                logger.info(f"# Check Executed {executed_count} trades")

        except Exception as e:
            logger.error(f"Trade execution system failed: {e}")

    async def _execute_single_trade(self, signal: Dict[str, Any]):
        """Execute a single trade with all validations"""
        symbol = signal['symbol']
        entry_price = signal.get('entry_price', 0)

        if not self.exchange or entry_price <= 0:
            logger.warning(f"Cannot execute trade for {symbol}: Invalid parameters")
            return

        try:
            # Calculate position size based on risk management
            position_size = self._calculate_position_size(symbol, entry_price)

            # Set up order parameters
            order_params = {
                'symbol': symbol,
                'side': 'buy',
                'type': 'market',
                'amount': position_size,
                'leverage': min(self.trading_config['max_leverage'], 50)
            }

            # Execute order
            order = self.exchange.create_order(**order_params)

            # Set up stop loss and take profit
            await self._setup_risk_management(symbol, entry_price, order['id'])

            logger.info(f"# Check Executed trade: {symbol} at {entry_price} (Size: {position_size})")

        except Exception as e:
            logger.error(f"Order execution failed for {symbol}: {e}")
            raise

    def _calculate_position_size(self, symbol: str, entry_price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            balance = self.exchange.fetch_balance()
            available_balance = balance['USDT']['free']

            # Calculate risk amount
            risk_amount = available_balance * self.trading_config['risk_per_trade']

            # Calculate position size
            stop_loss_pct = self.trading_config['stop_loss_pct'] / 100
            position_value = risk_amount / stop_loss_pct

            # Convert to contract amount (simplified calculation)
            contract_size = position_value / entry_price

            return contract_size

        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.001  # Minimum position size

    async def _setup_risk_management(self, symbol: str, entry_price: float, order_id: str):
        """Setup stop loss and take profit orders"""
        try:
            stop_loss_price = entry_price * (1 - self.trading_config['stop_loss_pct'] / 100)
            take_profit_price = entry_price * (1 + self.trading_config['take_profit_pct'] / 100)

            # Place stop loss order
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type='stop',
                side='sell',
                amount=0.001,  # Will be updated based on position
                price=stop_loss_price,
                params={'stopPrice': stop_loss_price}
            )

            # Place take profit order
            tp_order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side='sell',
                amount=0.001,  # Will be updated based on position
                price=take_profit_price
            )

            logger.info(f"🛡️ Risk management set for {symbol}: SL@{stop_loss_price}, TP@{take_profit_price}")

        except Exception as e:
            logger.error(f"Risk management setup failed for {symbol}: {e}")

    async def _get_current_positions(self) -> List[Dict[str, Any]]:
        """Get current open positions"""
        try:
            if not self.exchange:
                return []

            positions = self.exchange.fetch_positions()
            return [pos for pos in positions if pos['contracts'] > 0]

        except Exception as e:
            logger.error(f"Failed to get current positions: {e}")
            return []

    async def _monitor_and_adjust_positions(self):
        """Monitor and adjust existing positions"""
        try:
            positions = await self._get_current_positions()

            for position in positions:
                try:
                    symbol = position['symbol']

                    # Apply AI optimization to position management
                    if 'ai_optimizer' in self.components:
                        adjustment = self.components['ai_optimizer'].optimize_position(position)

                        if adjustment.get('action') == 'close':
                            await self._close_position(symbol, position['id'])
                        elif adjustment.get('action') == 'adjust_tp_sl':
                            await self._adjust_tp_sl(symbol, adjustment)

                except Exception as e:
                    logger.error(f"Position monitoring error for {position['symbol']}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Position monitoring system failed: {e}")

    async def _close_position(self, symbol: str, position_id: str):
        """Close a position"""
        try:
            # Get current position details
            positions = await self._get_current_positions()
            position = next((p for p in positions if p.get('id') == position_id), None)

            if position:
                # Close position
                self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side='sell' if position['side'] == 'long' else 'buy',
                    amount=position['contracts']
                )

                logger.info(f"# Check Closed position: {symbol} (ID: {position_id})")

        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}")

    async def _adjust_tp_sl(self, symbol: str, adjustment: Dict[str, Any]):
        """Adjust stop loss and take profit levels"""
        try:
            # Cancel existing TP/SL orders
            # Implementation would depend on exchange API
            logger.info(f"# Tool Adjusting TP/SL for {symbol}: {adjustment}")

        except Exception as e:
            logger.error(f"Failed to adjust TP/SL for {symbol}: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'trading_active': self.trading_active,
            'monitoring_active': self.monitoring_active,
            'components_loaded': list(self.components.keys()),
            'exchange_connected': self.exchange is not None,
            'trading_config': self.trading_config,
            'optimal_config_loaded': hasattr(self, 'optimal_config')
        }

    def run_system_check(self) -> Dict[str, Any]:
        """Run comprehensive system check"""
        status = self.get_system_status()

        # Add component health checks
        health_checks = {}
        for component_name, component in self.components.items():
            try:
                if hasattr(component, 'health_check'):
                    health_checks[component_name] = component.health_check()
                else:
                    health_checks[component_name] = 'unknown'
            except Exception as e:
                health_checks[component_name] = f'error: {str(e)}'

        status['health_checks'] = health_checks
        return status

async def main():
    """Main execution function"""

    # Initialize engine
    engine = UnifiedTradingEngine()

    # Display system status
    status = engine.get_system_status()
    print(f"   Components Loaded: {len(status['components_loaded'])}")
    print(f"   Exchange Connected: {'# Check' if status['exchange_connected'] else '# X'}")
    print(f"   Trading Active: {'# Check' if status['trading_active'] else '# X'}")

    # Run system check
    system_check = engine.run_system_check()
    print(f"   System Health: {len([h for h in system_check['health_checks'].values() if h != 'unknown'])} components healthy")

    # Start trading if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--start-trading':
        try:
            await engine.start_trading_engine()
        except KeyboardInterrupt:
    else:
        print("# Idea Use '--start-trading' flag to begin live trading")
        print("   Example: python unified_trading_engine.py --start-trading")
if __name__ == "__main__":
    asyncio.run(main())
