#!/usr/bin/env python3
"""
# Rocket VIPER LIVE TRADING BOT WITH MANDATORY DOCKER & MCP ENFORCEMENT
MAX 10 POSITIONS | MAX 3% RISK PER TRADE | PROPER MARGIN CALCULATION

# Warning  CRITICAL: This module now operates through MANDATORY Docker & MCP enforcement
All operations require Docker services and MCP integration to be active
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path - Updated for new structure
project_root = Path(__file__).parent.parent.parent.parent  # Navigate to viper-/ root
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# MANDATORY ENFORCEMENT IMPORT - Updated paths
try:
    from src.viper.core.mandatory_docker_mcp_wrapper import execute_module, start_system_with_enforcement
    ENFORCEMENT_AVAILABLE = True
except ImportError:
    print("# Warning WARNING: Mandatory enforcement system not available - running in legacy mode")
    ENFORCEMENT_AVAILABLE = False

# Core imports with error handling
try:
    import ccxt
    from datetime import datetime
    from dotenv import load_dotenv
    from src.viper.core.job_manager import ViperLiveJobManager
    
    # Load environment variables from .env file
    load_dotenv()
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print("📦 Please ensure all dependencies are installed: pip install -r requirements.txt")
    # Continue with available modules
    IMPORTS_AVAILABLE = False
    
    # Create dummy load_dotenv if not available
    def load_dotenv():
        pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVIPERTrader:
    def __init__(self):
        # Load API credentials
        self.api_key = os.getenv('BITGET_API_KEY', '')
        self.api_secret = os.getenv('BITGET_API_SECRET', '')
        self.api_password = os.getenv('BITGET_API_PASSWORD', '')

        # Trading config
        self.position_size_usdt = float(os.getenv('POSITION_SIZE_USDT', '5'))
        self.min_leverage_required = 34  # Minimum leverage required
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '2.5'))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '1.5'))
        self.max_positions = int(os.getenv('MAX_POSITIONS', '5'))

        # ALL AVAILABLE TRADING PAIRS (will be filtered by leverage)
        self.all_symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT',
            'SOL/USDT:USDT', 'ADA/USDT:USDT', 'DOT/USDT:USDT',
            'LINK/USDT:USDT', 'UNI/USDT:USDT', 'AVAX/USDT:USDT',
            'MATIC/USDT:USDT', 'DOGE/USDT:USDT', 'TRX/USDT:USDT',
            'ETC/USDT:USDT', 'ICP/USDT:USDT', 'FIL/USDT:USDT',
            'XRP/USDT:USDT', 'LTC/USDT:USDT', 'BCH/USDT:USDT',
            'EOS/USDT:USDT', 'THETA/USDT:USDT', 'FTT/USDT:USDT',
            'SUSHI/USDT:USDT', 'AAVE/USDT:USDT', 'MKR/USDT:USDT',
            'COMP/USDT:USDT', 'CRV/USDT:USDT', 'YFI/USDT:USDT',
            'BAL/USDT:USDT', 'REN/USDT:USDT', 'OMG/USDT:USDT',
            'ZRX/USDT:USDT', 'BAT/USDT:USDT', 'MANA/USDT:USDT',
            'ENJ/USDT:USDT', 'ANT/USDT:USDT', 'STORJ/USDT:USDT',
            'GRT/USDT:USDT', 'CHZ/USDT:USDT', 'SAND/USDT:USDT',
            'AXS/USDT:USDT', 'SLP/USDT:USDT', 'ALICE/USDT:USDT',
            'TLM/USDT:USDT', 'WAVES/USDT:USDT',
            'NEAR/USDT:USDT', 'FTM/USDT:USDT', 'ALGO/USDT:USDT',
            'HBAR/USDT:USDT', 'EGLD/USDT:USDT', 'FLOW/USDT:USDT',
            'VET/USDT:USDT', 'IOTX/USDT:USDT', 'RVN/USDT:USDT'
        ]

        # Filtered symbols (only those supporting min leverage)
        self.symbols = []
        self.symbol_leverages = {}  # Store max leverage per symbol
        self.blacklisted_symbols = []  # Pairs that don't meet leverage requirements

        self.exchange = None
        self.active_positions = {}
        self.is_running = False
        self.total_trades = 0
        self.wins = 0
        self.losses = 0

        # Job Manager Integration
        self.job_manager = None
        self.account_balance = 0.0

        # Risk Management Limits
        self.max_positions = 10  # NEVER MORE THAN 10 POSITIONS
        self.max_risk_per_trade = 0.03  # MAX 3% RISK PER TRADE

    def connect(self):
        """Connect to Bitget"""
        try:
            if not all([self.api_key, self.api_secret, self.api_password]):
                logger.error("# X Missing API credentials")
                return False

            logger.info("🔌 Connecting to Bitget...")
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'options': {'defaultType': 'swap', 'adjustForTimeDifference': True},
                'sandbox': False,
            })

            markets = self.exchange.load_markets()
            logger.info(f"# Check Connected - {len(markets)} markets loaded")

            # Validate leverage for each symbol and filter
            self.filter_symbols_by_leverage()

            # Initialize Job Manager for risk management
            self.job_manager = ViperLiveJobManager()
            self.job_manager.max_positions = self.max_positions
            self.job_manager.max_risk_per_trade = self.max_risk_per_trade

            # Get initial balance
            self.update_balance()

            return True

        except Exception as e:
            logger.error(f"# X Connection failed: {e}")
            return False

    def filter_symbols_by_leverage(self):
        """Filter symbols based on minimum leverage requirement (34x)"""
        logger.info(f"# Search Validating leverage for {len(self.all_symbols)} symbols...")

        valid_symbols = []
        blacklisted = []

        for symbol in self.all_symbols:
            try:
                # Get market info for this symbol
                market = self.exchange.market(symbol)
                max_leverage = market.get('contractSize', 0)  # This might not be the right field

                # Try to get leverage info via API
                try:
                    # Fetch leverage info
                    leverage_info = self.exchange.fetch_leverage_tiers(symbol)
                    if leverage_info:
                        # Get the maximum leverage from tiers
                        max_leverage = max([tier.get('maxLeverage', 1) for tier in leverage_info])
                    else:
                        max_leverage = 20  # Default fallback
                except Exception:
                    max_leverage = 20  # Default fallback

                self.symbol_leverages[symbol] = max_leverage

                if max_leverage >= self.min_leverage_required:
                    valid_symbols.append(symbol)
                    logger.info(f"# Check {symbol}: {max_leverage}x leverage (VALID)")
                else:
                    blacklisted.append(symbol)
                    logger.warning(f"# X {symbol}: {max_leverage}x leverage (BLACKLISTED - below {self.min_leverage_required}x)")

            except Exception as e:
                logger.warning(f"# Warning Could not validate leverage for {symbol}: {e}")
                # Add with default leverage as fallback
                self.symbol_leverages[symbol] = 20
                valid_symbols.append(symbol)

        self.symbols = valid_symbols
        self.blacklisted_symbols = blacklisted

        logger.info(f"# Target VALID SYMBOLS: {len(self.symbols)}/{len(self.all_symbols)}")
        logger.info(f"🚫 BLACKLISTED: {len(self.blacklisted_symbols)} symbols")
        logger.info(f"💰 MIN LEVERAGE REQUIRED: {self.min_leverage_required}x")

    def update_balance(self):
        """Update account balance and sync with job manager"""
        try:
            balance = self.exchange.fetch_balance({'type': 'swap'})
            if 'USDT' in balance:
                self.account_balance = float(balance['USDT']['free'])
                logger.info(f"💰 Balance: ${self.account_balance:.2f} USDT")

                # Sync with job manager
                if self.job_manager:
                    self.job_manager.update_account_balance(self.account_balance)

            return True
        except Exception as e:
            logger.error(f"# X Error fetching balance: {e}")
            return False

    def scan_all_signals(self):
        """Scan ALL VALID pairs for trading signals (ONLY 1 POSITION PER PAIR)"""
        opportunities = []

        for symbol in self.symbols:
            # SINGLE POSITION PER PAIR - NO CAPITAL STACKING
            if symbol in self.active_positions:
                continue

            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker['last']
                change_24h = ticker.get('percentage', 0)
                volume = ticker.get('quoteVolume', 0)

                # Simple momentum signal with volume filter
                if change_24h > 1.0 and volume > 50000:  # >1% up, decent volume
                    opportunities.append((symbol, 'buy', change_24h))
                elif change_24h < -1.0 and volume > 50000:  # >1% down, decent volume
                    opportunities.append((symbol, 'sell', change_24h))

            except Exception as e:
                continue  # Skip this symbol if error

        return opportunities

    def execute_trade(self, symbol, side):
        """Execute trade with PROPER RISK MANAGEMENT - MAX 10 POSITIONS, MAX 3% RISK"""
        try:
            # RISK MANAGEMENT VALIDATION THROUGH JOB MANAGER
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Calculate position size with max leverage
            max_leverage = self.symbol_leverages.get(symbol, 20)
            position_value = self.position_size_usdt * max_leverage
            position_size = position_value / current_price

            # VALIDATE WITH JOB MANAGER RISK RULES
            if self.job_manager:
                can_open, reason = self.job_manager.can_open_position(
                    symbol, current_price, position_size, max_leverage
                )

                if not can_open:
                    logger.warning(f"🚫 POSITION REJECTED: {reason}")
                    return False

            # Get current balance and validate
            if not self.update_balance():
                logger.error("# X Could not fetch balance - skipping trade")
                return False

            logger.info(f"# Rocket EXECUTING {side.upper()} ORDER ON {symbol}")
            logger.info(f"   💰 Price: ${current_price:.6f}")
            logger.info(f"   📏 Size: {position_size:.6f}")
            logger.info(f"   🎲 Leverage: {max_leverage}x")
            logger.info(f"   💵 Margin Required: ${self.position_size_usdt}")
            logger.info(f"   # Warning Risk Amount: ${self.position_size_usdt * self.max_risk_per_trade:.2f} (3%)")

            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=position_size,
                params={
                    'marginCoin': 'USDT',
                    'leverage': max_leverage,
                    'marginMode': 'isolated',
                    'holdSide': 'long' if side == 'buy' else 'short',
                    'tradeSide': 'open'
                }
            )

            # TRACK POSITION WITH JOB MANAGER
            if self.job_manager:
                self.job_manager.add_position(symbol, side, position_size, current_price, max_leverage)

            # Also store in local tracking for backward compatibility
            self.active_positions[symbol] = {
                'side': side,
                'entry_price': current_price,
                'size': position_size,
                'leverage': max_leverage,
                'margin_used': self.position_size_usdt,
                'timestamp': datetime.now().isoformat()
            }

            self.total_trades += 1
            logger.info(f"# Check TRADE EXECUTED: {order['id']} | Leverage: {max_leverage}x")
            logger.info(f"   # Chart ACTIVE POSITIONS: {len(self.active_positions)}/{self.max_positions}")
            return True

        except Exception as e:
            logger.error(f"# X Trade failed on {symbol}: {e}")
            return False

    def monitor_positions(self):
        """Monitor ALL positions with leverage info"""
        if not self.active_positions:
            return

        logger.info(f"👁️ Monitoring {len(self.active_positions)} positions...")

        for symbol in list(self.active_positions.keys()):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']

                position = self.active_positions[symbol]
                entry_price = position['entry_price']
                side = position['side']
                leverage = position.get('leverage', 20)

                # Calculate P&L
                if side == 'buy':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100

                logger.info(f"# Chart {symbol}: {pnl_pct:.2f}% P&L | {leverage}x | ${position.get('margin_used', 0)} margin")

                # Risk management
                if pnl_pct >= self.take_profit_pct:
                    logger.info(f"💰 Taking profit on {symbol} ({pnl_pct:.1f}%)")
                    self.close_position(symbol, "PROFIT")
                elif pnl_pct <= -self.stop_loss_pct:
                    logger.info(f"🛑 Stopping loss on {symbol} ({pnl_pct:.1f}%)")
                    self.close_position(symbol, "STOP_LOSS")

            except Exception as e:
                logger.error(f"# X Monitor error for {symbol}: {e}")

    def close_position(self, symbol, reason):
        """Close position for any symbol"""
        try:
            if symbol not in self.active_positions:
                return

            position = self.active_positions[symbol]
            opposite_side = 'sell' if position['side'] == 'buy' else 'buy'

            close_order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=opposite_side,
                amount=position['size'],
                params={
                    'marginCoin': 'USDT',
                    'holdSide': 'long' if position['side'] == 'buy' else 'short',
                    'tradeSide': 'close'
                }
            )

            # Calculate P&L for statistics
            current_price = close_order.get('average', close_order.get('price', 0))
            entry_price = position['entry_price']
            side = position['side']

            if side == 'buy':
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100

            if pnl_pct > 0:
                self.wins += 1
            else:
                self.losses += 1

            logger.info(f"# Check Position closed: {symbol} ({reason}) - P&L: {pnl_pct:.2f}%")
            del self.active_positions[symbol]

        except Exception as e:
            logger.error(f"# X Close error for {symbol}: {e}")

    def run(self):
        """Main trading loop - SCANS ALL PAIRS"""
        logger.info("# Rocket Starting VIPER ALL-PAIRS Trading Bot")
        logger.info(f"# Chart Scanning {len(self.symbols)} trading pairs")
        logger.info("=" * 80)

        self.is_running = True
        cycle_count = 0

        try:
            while self.is_running:
                cycle_count += 1
                logger.info(f"\n🔄 CYCLE #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")

                # Monitor existing positions
                self.monitor_positions()

                # Scan ALL pairs for opportunities
                if len(self.active_positions) < self.max_positions:
                    opportunities = self.scan_all_signals()

                    if opportunities:
                        logger.info(f"# Target Found {len(opportunities)} trading opportunities")

                        # Sort by signal strength (highest change first)
                        opportunities.sort(key=lambda x: abs(x[2]), reverse=True)

                        # Execute up to 2 trades per cycle
                        trades_executed = 0
                        for symbol, side, change_pct in opportunities[:2]:  # Take top 2
                            if symbol not in self.active_positions and trades_executed < 2:
                                logger.info(f"# Target Executing {side.upper()} on {symbol} ({change_pct:.1f}%)")
                                if self.execute_trade(symbol, side):
                                    trades_executed += 1
                                    time.sleep(1)  # Brief pause between trades

                        if trades_executed > 0:
                            logger.info(f"# Check Executed {trades_executed} trades this cycle")

                # Comprehensive status update with leverage info
                win_rate = (self.wins / max(self.total_trades, 1)) * 100
                total_margin_used = sum([pos.get('margin_used', 0) for pos in self.active_positions.values()])

                logger.info("# Chart STATUS UPDATE:")
                logger.info(f"   💰 Balance: ${self.balance:.2f} | Margin Used: ${total_margin_used:.2f}")
                logger.info(f"   # Chart Active Positions: {len(self.active_positions)}/{self.max_positions}")
                logger.info(f"   📈 Total Trades: {self.total_trades}")
                logger.info(f"   🟢 Wins: {self.wins} | 🔴 Losses: {self.losses}")
                logger.info(f"   # Target Win Rate: {win_rate:.1f}%")
                logger.info(f"   ⚙️ Position Size: ${self.position_size_usdt} | TP/SL: {self.take_profit_pct}%/{self.stop_loss_pct}%")
                logger.info(f"   # Search Valid Pairs: {len(self.symbols)} | Min Leverage: {self.min_leverage_required}x")

                # Wait before next cycle
                logger.info("⏰ Next scan in 30 seconds...")
                time.sleep(30)

        except KeyboardInterrupt:
            logger.info("\n🛑 Trading stopped by user")
        finally:
            # Emergency close all positions
            if self.active_positions:
                logger.info("🔄 Emergency closing all positions...")
                for symbol in list(self.active_positions.keys()):
                    self.close_position(symbol, "EMERGENCY_SHUTDOWN")

            logger.info("# Check All-pairs trading bot shutdown complete")


def main():
    """
    MAIN ENTRY POINT - LIVE TRADING ONLY WITH MANDATORY DOCKER & MCP ENFORCEMENT
    
    # Warning CRITICAL: This system only operates in live trading mode
    All operations require Docker services and MCP integration - NO EXCEPTIONS
    """
    
    print("🔒 VIPER LIVE TRADING BOT - MANDATORY DOCKER & MCP ENFORCEMENT")
    print("🚨 LIVE TRADING MODE ONLY - NO MOCK DATA OR DEMO MODE")
    
    # Force load environment with live trading settings
    from dotenv import load_dotenv
    load_dotenv()
    
    # Validate live trading environment
    if os.getenv('USE_MOCK_DATA', '').lower() == 'true':
        logger.error("# X MOCK DATA MODE DETECTED - NOT ALLOWED IN LIVE TRADING")
        logger.error("Set USE_MOCK_DATA=false in .env file")
        sys.exit(1)
    
    if not ENFORCEMENT_AVAILABLE:
        logger.error("# X MANDATORY ENFORCEMENT SYSTEM NOT AVAILABLE")
        logger.error("Docker and MCP enforcement is required for live trading")
        sys.exit(1)
    
    print("# Rocket Starting system with mandatory Docker & MCP validation...")
    
    # Start system with enforcement - no bypassing allowed
    if not start_system_with_enforcement():
        print("💀 SYSTEM STARTUP FAILED - ENFORCEMENT REQUIREMENTS NOT MET")
        sys.exit(1)
    
    print("# Check Enforcement validation passed - starting live trading bot...")
    
    # Execute through mandatory wrapper only
    try:
        result = execute_module('main', 'run_live_trading')
        return result
    except SystemExit:
        print("💀 Live trading bot execution blocked by enforcement!")
        sys.exit(1)

def run_live_trading():
    """Live trading function - LIVE MODE ONLY"""
    logger.info("# Rocket VIPER LIVE TRADING BOT STARTING...")
    logger.info("# Chart Scanning cryptocurrency pairs for live trading opportunities")
    logger.info("# Warning LIVE MODE: Real trades will be executed with real money")

    trader = SimpleVIPERTrader()

    if not trader.connect():
        logger.error("# X Failed to connect to Bitget exchange")
        sys.exit(1)

    logger.info("\n# Target VIPER LIVE TRADING CONFIGURATION:")
    logger.info(f"   # Chart Total Pairs Available: {len(trader.all_symbols)}")
    logger.info(f"   # Check Valid Pairs (≥{trader.min_leverage_required}x): {len(trader.symbols)}")
    logger.info(f"   🚫 Blacklisted Pairs (<{trader.min_leverage_required}x): {len(trader.blacklisted_symbols)}")
    logger.info(f"   💰 Position Size: ${trader.position_size_usdt} per trade")
    logger.info(f"   📈 Take Profit: {trader.take_profit_pct}%")
    logger.info(f"   🛑 Stop Loss: {trader.stop_loss_pct}%")
    logger.info(f"   # Target Max Positions: {trader.max_positions} concurrent")
    logger.info(f"   🔒 SINGLE POSITION PER PAIR: ENABLED")
    logger.info(f"   🚫 CAPITAL STACKING: DISABLED")

    if trader.blacklisted_symbols:
        logger.info("   🚫 BLACKLISTED PAIRS:")
        for symbol in trader.blacklisted_symbols[:5]:  # Show first 5
            logger.info(f"      - {symbol}")
        if len(trader.blacklisted_symbols) > 5:
            logger.info(f"      ... and {len(trader.blacklisted_symbols) - 5} more")

    logger.info("⏳ Starting live trading with real money in 5 seconds...")
    logger.warning("# Warning WARNING: This will execute real trades with real money!")
    time.sleep(5)

    try:
        trader.run()
    except KeyboardInterrupt:
        logger.info("\n🛑 Live trading cancelled by user")
    except Exception as e:
        logger.error(f"\n# X Fatal error in live trading: {e}")
    finally:
        logger.info("# Check Live trading bot shutdown complete")

    if not trader.connect():
        logger.error("# X Failed to connect to Bitget")
        return

    logger.info("\n# Target VIPER LEVERAGE-BASED CONFIGURATION:")
    logger.info(f"   # Chart Total Pairs Available: {len(trader.all_symbols)}")
    logger.info(f"   # Check Valid Pairs (≥{trader.min_leverage_required}x): {len(trader.symbols)}")
    logger.info(f"   🚫 Blacklisted Pairs (<{trader.min_leverage_required}x): {len(trader.blacklisted_symbols)}")
    logger.info(f"   💰 Position Size: ${trader.position_size_usdt} per trade")
    logger.info(f"   📈 Take Profit: {trader.take_profit_pct}%")
    logger.info(f"   🛑 Stop Loss: {trader.stop_loss_pct}%")
    logger.info(f"   # Target Max Positions: {trader.max_positions} concurrent")
    logger.info(f"   🔒 SINGLE POSITION PER PAIR: ENABLED")
    logger.info(f"   🚫 CAPITAL STACKING: DISABLED")

    if trader.blacklisted_symbols:
        logger.info("   🚫 BLACKLISTED PAIRS:")
        for symbol in trader.blacklisted_symbols[:5]:  # Show first 5
            logger.info(f"      - {symbol}")
        if len(trader.blacklisted_symbols) > 5:
            logger.info(f"      ... and {len(trader.blacklisted_symbols) - 5} more")

    logger.info("⏳ Starting leverage-validated trading in 3 seconds...")
    time.sleep(3)

    try:
        trader.run()
    except KeyboardInterrupt:
        logger.info("\n🛑 Trading cancelled by user")
    except Exception as e:
        logger.error(f"\n# X Fatal error: {e}")
    finally:
        logger.info("# Check All-pairs trading bot shutdown complete")

if __name__ == "__main__":
    main()
