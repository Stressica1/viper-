#!/usr/bin/env python3
"""
🚀 VIPER ALL PAIRS SCANNER - Comprehensive Multi-Pair Trading Job
Scans ALL available Bitget swap pairs with advanced filtering and risk management

This job provides:
- Dynamic pair discovery from Bitget exchange
- Volume and volatility-based pair filtering
- Risk-managed multi-pair trading
- VIPER scoring across all pairs
- Position limits distributed across pairs
- Real-time performance tracking
- Emergency risk controls
"""

import os
import sys
import time
import json
import logging
import asyncio
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import ccxt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VIPER_ALL_PAIRS - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VIPERAllPairsScanner:
    """
    Comprehensive multi-pair trading scanner with ALL Bitget swap pairs
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or Path(__file__).parent / ".env"
        self.is_running = False
        self.positions = {}
        self.all_pairs = []
        self.active_pairs = []
        self.pair_stats = {}
        self.trading_stats = {
            'trades_executed': 0,
            'trades_closed': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'start_time': None,
            'end_time': None,
            'pairs_scanned': 0,
            'opportunities_found': 0
        }

        # Initialize all components
        self._initialize_components()
        self._load_configuration()
        self._setup_exchange()
        self._discover_all_pairs()
        self._setup_signal_handlers()

        logger.info("✅ VIPER All Pairs Scanner initialized successfully")

    def _initialize_components(self):
        """Initialize all trading components"""
        try:
            # Import and initialize core components
            from utils.mathematical_validator import MathematicalValidator
            from config.optimal_mcp_config import get_optimal_mcp_config
            from scripts.optimal_entry_point_manager import OptimalEntryPointManager

            self.math_validator = MathematicalValidator()
            self.mcp_config = get_optimal_mcp_config()
            self.entry_optimizer = OptimalEntryPointManager()

            # Initialize AI optimizer if available
            try:
                spec = importlib.util.spec_from_file_location(
                    "ai_optimizer",
                    Path(__file__).parent / "ai_ml_optimizer.py"
                )
                ai_optimizer_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ai_optimizer_module)

                if hasattr(ai_optimizer_module, 'VIPEROptimizer'):
                    self.ai_optimizer = ai_optimizer_module.VIPEROptimizer()
                else:
                    self.ai_optimizer = None
            except Exception as e:
                logger.warning(f"AI Optimizer not available: {e}")
                self.ai_optimizer = None

            logger.info("✅ All components initialized")

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise

    def _load_configuration(self):
        """Load comprehensive trading configuration for all pairs"""
        try:
            # Trading parameters optimized for multi-pair scanning
            self.trading_config = {
                'max_positions_total': int(os.getenv('MAX_TOTAL_POSITIONS', '10')),  # Total across all pairs
                'max_positions_per_pair': int(os.getenv('MAX_POSITIONS_PER_PAIR', '1')),  # Max per pair
                'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),  # 2% risk per trade
                'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '3.0')),
                'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '5.0')),
                'trailing_stop_pct': float(os.getenv('TRAILING_STOP_PCT', '2.0')),
                'trailing_activation_pct': float(os.getenv('TRAILING_ACTIVATION_PCT', '1.0')),
                'max_leverage': int(os.getenv('MAX_LEVERAGE', '50')),
                'scan_interval': int(os.getenv('SCAN_INTERVAL', '30')),  # 30 seconds for all pairs
                'max_trades_per_hour': int(os.getenv('MAX_TRADES_PER_HOUR', '20')),  # Higher for multi-pair
                'min_viper_score': float(os.getenv('MIN_VIPER_SCORE', '75.0')),
                # Multi-pair specific settings
                'min_volume_threshold': float(os.getenv('MIN_VOLUME_THRESHOLD', '1000000')),  # $1M daily volume
                'min_leverage_required': int(os.getenv('MIN_LEVERAGE_REQUIRED', '34')),  # Min leverage available
                'max_spread_threshold': float(os.getenv('MAX_SPREAD_THRESHOLD', '0.001')),  # 0.1% max spread
                'pairs_batch_size': int(os.getenv('PAIRS_BATCH_SIZE', '20')),  # Process pairs in batches
                'use_real_data_only': os.getenv('USE_REAL_DATA_ONLY', 'true').lower() == 'true',
                'enable_live_risk_management': True,
                'require_api_credentials': True
            }

            # Emergency stop parameters for multi-pair trading
            self.emergency_config = {
                'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '100.0')),  # Higher limit for multi-pair
                'max_total_positions': int(os.getenv('MAX_TOTAL_POSITIONS', '15')),
                'circuit_breaker_enabled': os.getenv('CIRCUIT_BREAKER_ENABLED', 'true').lower() == 'true',
                'max_trades_per_hour': int(os.getenv('MAX_TRADES_PER_HOUR', '30')),
                'max_loss_per_pair': float(os.getenv('MAX_LOSS_PER_PAIR', '10.0'))  # Max loss per pair
            }

            logger.info("✅ Configuration loaded for all pairs scanning")

        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            raise

    def _setup_exchange(self):
        """Setup exchange connection for live multi-pair trading"""
        try:
            api_key = os.getenv('BITGET_API_KEY')
            api_secret = os.getenv('BITGET_API_SECRET')
            api_password = os.getenv('BITGET_API_PASSWORD')

            if not all([api_key, api_secret, api_password]):
                logger.error("❌ LIVE MULTI-PAIR TRADING REQUIRES API CREDENTIALS")
                raise ValueError("Missing required API credentials")

            exchange_config = {
                'apiKey': api_key,
                'secret': api_secret,
                'password': api_password,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000,
                    'defaultType': 'swap',
                }
            }

            self.exchange = ccxt.bitget(exchange_config)

            # Test connection and load markets
            self.exchange.load_markets()
            logger.info("✅ LIVE Exchange connection established for multi-pair scanning")

        except Exception as e:
            logger.error(f"❌ Exchange setup failed: {e}")
            raise

    def _discover_all_pairs(self):
        """Discover ALL available swap pairs on Bitget"""
        try:
            logger.info("🔍 Discovering ALL Bitget swap pairs...")

            # Get all swap markets
            all_markets = self.exchange.markets

            # Filter for active swap pairs only
            swap_pairs = []
            for symbol, market in all_markets.items():
                if (market.get('active', False) and
                    market.get('type') == 'swap' and
                    market.get('quote') == 'USDT'):  # Focus on USDT pairs

                    pair_info = {
                        'symbol': symbol,
                        'base': market.get('base'),
                        'quote': market.get('quote'),
                        'leverage': market.get('leverage', {}).get('max', 1),
                        'precision': market.get('precision', {}),
                        'limits': market.get('limits', {}),
                        'active': market.get('active', False)
                    }
                    swap_pairs.append(pair_info)

            self.all_pairs = swap_pairs
            logger.info(f"📊 Found {len(self.all_pairs)} active USDT swap pairs on Bitget")

            # Filter pairs based on criteria
            self._filter_pairs_by_criteria()

        except Exception as e:
            logger.error(f"❌ Failed to discover pairs: {e}")
            raise

    def _filter_pairs_by_criteria(self):
        """Filter pairs based on volume, leverage, and other criteria"""
        try:
            logger.info("🔍 Filtering pairs by trading criteria...")

            filtered_pairs = []

            for pair in self.all_pairs:
                try:
                    # Get 24h ticker data
                    ticker = self.exchange.fetch_ticker(pair['symbol'])

                    # Apply filters
                    volume_24h = ticker.get('quoteVolume', 0)
                    spread = abs(ticker.get('ask', 0) - ticker.get('bid', 0)) / ticker.get('bid', 1)
                    leverage = pair.get('leverage', 1)

                    # Check criteria
                    if (volume_24h >= self.trading_config['min_volume_threshold'] and
                        spread <= self.trading_config['max_spread_threshold'] and
                        leverage >= self.trading_config['min_leverage_required'] and
                        ticker.get('last', 0) > 0):

                        pair['volume_24h'] = volume_24h
                        pair['spread'] = spread
                        pair['price'] = ticker.get('last', 0)
                        filtered_pairs.append(pair)

                        logger.debug(f"✅ {pair['symbol']}: Vol=${volume_24h:,.0f}, Spread={spread:.4f}, Lev={leverage}x")

                except Exception as e:
                    logger.warning(f"⚠️ Could not filter {pair['symbol']}: {e}")
                    continue

            self.active_pairs = filtered_pairs
            logger.info(f"🎯 Filtered to {len(self.active_pairs)} qualified pairs for trading")

            # Log top pairs by volume
            top_pairs = sorted(self.active_pairs, key=lambda x: x.get('volume_24h', 0), reverse=True)[:10]
            logger.info("🏆 Top 10 pairs by volume:")
            for i, pair in enumerate(top_pairs, 1):
                logger.info(f"   {i}. {pair['symbol']}: ${pair.get('volume_24h', 0):,.0f}")

        except Exception as e:
            logger.error(f"❌ Pair filtering failed: {e}")
            self.active_pairs = self.all_pairs  # Fallback to all pairs

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info("🛑 Received shutdown signal")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start(self):
        """Start the all-pairs scanner"""
        if self.is_running:
            logger.warning("Scanner already running")
            return

        logger.info("🚀 Starting VIPER All Pairs Scanner...")
        logger.info(f"📊 Scanning {len(self.active_pairs)} pairs every {self.trading_config['scan_interval']} seconds")
        self.is_running = True
        self.trading_stats['start_time'] = datetime.now()

        try:
            # Run the main scanning loop
            asyncio.run(self._scanning_loop())

        except KeyboardInterrupt:
            logger.info("🛑 Scanner stopped by user")
        except Exception as e:
            logger.error(f"❌ Scanner error: {e}")
        finally:
            self._cleanup()

    def stop(self):
        """Stop the scanner"""
        logger.info("🛑 Stopping VIPER All Pairs Scanner...")
        self.is_running = False

    def _cleanup(self):
        """Cleanup and generate final report"""
        self.trading_stats['end_time'] = datetime.now()

        # Close all positions
        self._close_all_positions()

        # Generate final report
        self._generate_final_report()

        logger.info("✅ All pairs scanner cleanup completed")

    async def _scanning_loop(self):
        """Main scanning loop for all pairs"""
        logger.info("🔄 Starting comprehensive pair scanning...")

        batch_size = self.trading_config['pairs_batch_size']
        scan_interval = self.trading_config['scan_interval']

        while self.is_running:
            try:
                current_time = datetime.now()

                # Emergency stop check
                if self._should_emergency_stop():
                    logger.warning("🚨 Emergency stop triggered")
                    break

                # Step 1: Scan all pairs in batches
                opportunities = await self._scan_all_pairs_batch(batch_size)

                # Step 2: Score opportunities
                scored_opportunities = self._score_opportunities(opportunities)

                # Step 3: Execute trades with position limits
                if scored_opportunities:
                    executed_trades = await self._execute_trades(scored_opportunities)

                # Step 4: Manage existing positions
                await self._manage_positions()

                # Step 5: Update performance stats
                self._update_performance_stats()

                # Wait before next scan
                await asyncio.sleep(scan_interval)

            except Exception as e:
                logger.error(f"Scanning loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _scan_all_pairs_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Scan all active pairs in batches"""
        opportunities = []
        total_pairs = len(self.active_pairs)

        logger.info(f"📊 Scanning {total_pairs} pairs in batches of {batch_size}...")

        for i in range(0, total_pairs, batch_size):
            batch_pairs = self.active_pairs[i:i + batch_size]
            batch_opportunities = await self._scan_pairs_batch(batch_pairs)
            opportunities.extend(batch_opportunities)

            # Small delay between batches to avoid rate limits
            await asyncio.sleep(0.1)

        logger.info(f"📊 Found {len(opportunities)} opportunities across all pairs")
        self.trading_stats['pairs_scanned'] += total_pairs
        self.trading_stats['opportunities_found'] += len(opportunities)

        return opportunities

    async def _scan_pairs_batch(self, pairs_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scan a batch of pairs concurrently"""
        opportunities = []

        try:
            # Use ThreadPoolExecutor for concurrent scanning
            with ThreadPoolExecutor(max_workers=min(len(pairs_batch), 10)) as executor:
                # Submit all pair scans
                future_to_pair = {
                    executor.submit(self._scan_single_pair, pair): pair
                    for pair in pairs_batch
                }

                # Collect results
                for future in as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    try:
                        opportunity = future.result()
                        if opportunity:
                            opportunities.append(opportunity)
                    except Exception as e:
                        logger.error(f"Error scanning {pair['symbol']}: {e}")

        except Exception as e:
            logger.error(f"Batch scanning failed: {e}")

        return opportunities

    def _scan_single_pair(self, pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Scan a single pair for trading opportunities"""
        try:
            symbol = pair['symbol']

            # Get market data
            ticker = self.exchange.fetch_ticker(symbol)

            # Analyze entry point
            analysis = self.entry_optimizer.analyze_entry_point(symbol)

            if analysis.get('should_enter', False):
                opportunity = {
                    'symbol': symbol,
                    'price': ticker['last'],
                    'volume': ticker['quoteVolume'],
                    'change_24h': ticker['percentage'],
                    'analysis': analysis,
                    'pair_info': pair,
                    'timestamp': datetime.now().isoformat()
                }
                return opportunity

        except Exception as e:
            logger.warning(f"Error scanning {pair['symbol']}: {e}")

        return None

    def _score_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score trading opportunities using VIPER system"""
        scored_opportunities = []

        try:
            for opp in opportunities:
                try:
                    # Calculate VIPER scores
                    viper_scores = self._calculate_viper_scores(opp)

                    # Overall score
                    overall_score = sum(viper_scores.values()) / len(viper_scores)

                    if overall_score >= self.trading_config['min_viper_score']:
                        opp['viper_scores'] = viper_scores
                        opp['overall_score'] = overall_score
                        scored_opportunities.append(opp)

                except Exception as e:
                    logger.error(f"Error scoring opportunity for {opp['symbol']}: {e}")
                    continue

            # Sort by score (highest first)
            scored_opportunities.sort(key=lambda x: x['overall_score'], reverse=True)

            logger.info(f"🎯 Scored {len(scored_opportunities)} opportunities (min score: {self.trading_config['min_viper_score']})")
            return scored_opportunities

        except Exception as e:
            logger.error(f"Opportunity scoring failed: {e}")
            return []

    def _calculate_viper_scores(self, opportunity: Dict[str, Any]) -> Dict[str, float]:
        """Calculate VIPER scores for opportunity"""
        try:
            # Volume Score (30%) - Higher weight for multi-pair
            volume_score = min(opportunity.get('volume', 0) / 5000000, 1.0) * 30  # $5M volume max

            # Price Score (25%) - Volatility analysis
            price_change = abs(opportunity.get('change_24h', 0))
            price_score = min(price_change / 5.0, 1.0) * 25

            # Leverage Score (20%) - Available leverage
            leverage = opportunity.get('pair_info', {}).get('leverage', 1)
            leverage_score = min(leverage / 100, 1.0) * 20

            # Spread Score (15%) - Tighter spreads preferred
            spread = opportunity.get('pair_info', {}).get('spread', 0.001)
            spread_score = max(0, (0.001 - spread) / 0.001) * 15

            # Risk Score (10%) - Position in Bollinger Bands
            risk_score = 10.0  # Placeholder - could be more sophisticated

            return {
                'volume': volume_score,
                'price': price_score,
                'leverage': leverage_score,
                'spread': spread_score,
                'risk': risk_score
            }

        except Exception as e:
            logger.error(f"VIPER scoring error: {e}")
            return {'volume': 0, 'price': 0, 'leverage': 0, 'spread': 0, 'risk': 0}

    async def _execute_trades(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute trades with position limits across all pairs"""
        executed_trades = []
        current_positions = len(self.positions)

        try:
            # Check total position limit
            available_slots = self.trading_config['max_positions_total'] - current_positions

            for opp in opportunities[:available_slots]:
                try:
                    symbol = opp['symbol']

                    # Check if we already have a position in this pair
                    if symbol in self.positions:
                        continue

                    # Calculate position size
                    position_size = self._calculate_position_size(opp)

                    if position_size <= 0:
                        continue

                    # Execute trade
                    trade_result = await self._execute_single_trade(opp, position_size)

                    if trade_result:
                        executed_trades.append(trade_result)
                        self.trading_stats['trades_executed'] += 1

                        # Set up TP/SL
                        await self._setup_tp_sl(opp['symbol'], opp['price'], trade_result['order_id'])

                except Exception as e:
                    logger.error(f"Trade execution error for {opp['symbol']}: {e}")
                    continue

            if executed_trades:
                logger.info(f"✅ Executed {len(executed_trades)} trades across pairs")

            return executed_trades

        except Exception as e:
            logger.error(f"Trade execution system failed: {e}")
            return []

    def _calculate_position_size(self, opportunity: Dict[str, Any]) -> float:
        """Calculate position size with multi-pair risk management"""
        try:
            # Get account balance
            balance = self.exchange.fetch_balance()
            available_balance = balance['USDT']['free']

            if available_balance <= 0:
                return 0

            # Calculate risk amount (2% per trade)
            risk_amount = available_balance * self.trading_config['risk_per_trade']

            # Distribute risk across multiple pairs (reduce per-pair risk)
            pair_risk_factor = min(1.0, 5.0 / len(self.active_pairs))  # Max 5 pairs get full risk
            risk_amount *= pair_risk_factor

            # Calculate position size
            entry_price = opportunity['price']
            stop_loss_pct = self.trading_config['stop_loss_pct'] / 100

            position_value = risk_amount / stop_loss_pct
            position_size = position_value / entry_price

            # Apply leverage limit
            max_position_value = available_balance * self.trading_config['max_leverage']
            position_size = min(position_size, max_position_value / entry_price)

            # Minimum order size
            position_size = max(position_size, 0.001)

            return position_size

        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0

    async def _execute_single_trade(self, opportunity: Dict[str, Any], position_size: float) -> Optional[Dict[str, Any]]:
        """Execute a single trade"""
        try:
            symbol = opportunity['symbol']

            # Execute live trade
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side='buy',
                amount=position_size,
                params={'leverage': min(self.trading_config['max_leverage'], 50)}
            )

            trade_result = {
                'symbol': symbol,
                'side': 'buy',
                'amount': position_size,
                'price': order['price'],
                'timestamp': datetime.now().isoformat(),
                'order_id': order['id']
            }

            logger.info(f"🚀 LIVE TRADE: {symbol} at ${order['price']:.4f} (Size: {position_size:.6f})")

            # Track position
            self.positions[symbol] = {
                'entry_price': trade_result['price'],
                'amount': trade_result['amount'],
                'entry_time': datetime.now(),
                'tp_price': None,
                'sl_price': None,
                'tsl_price': None,
                'highest_price': trade_result['price'],
                'pnl': 0.0
            }

            return trade_result

        except Exception as e:
            logger.error(f"Single trade execution failed: {e}")
            return None

    async def _setup_tp_sl(self, symbol: str, entry_price: float, order_id: str):
        """Setup Take Profit and Stop Loss orders"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return

            # Calculate TP/SL prices
            tp_price = entry_price * (1 + self.trading_config['take_profit_pct'] / 100)
            sl_price = entry_price * (1 - self.trading_config['stop_loss_pct'] / 100)

            # Set TP/SL prices for position tracking
            position['tp_price'] = tp_price
            position['sl_price'] = sl_price

            # Try to place actual TP/SL orders
            try:
                # Place stop loss order
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side='sell',
                    amount=position['amount'],
                    price=sl_price,
                    params={'stopPrice': sl_price}
                )
                position['sl_order_id'] = sl_order['id']

                # Place take profit order
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side='sell',
                    amount=position['amount'],
                    price=tp_price
                )
                position['tp_order_id'] = tp_order['id']

                logger.info(f"🛡️ TP/SL orders placed for {symbol}")

            except Exception as e:
                logger.warning(f"Could not place TP/SL orders: {e}")

        except Exception as e:
            logger.error(f"TP/SL setup failed for {symbol}: {e}")

    async def _manage_positions(self):
        """Manage existing positions across all pairs"""
        try:
            for symbol, position in list(self.positions.items()):
                try:
                    # Get current price
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']

                    # Update highest price for trailing stop
                    if current_price > position['highest_price']:
                        position['highest_price'] = current_price

                        # Update trailing stop if activated
                        if position['tsl_price'] is None and (current_price - position['entry_price']) / position['entry_price'] >= self.trading_config['trailing_activation_pct'] / 100:
                            tsl_price = current_price * (1 - self.trading_config['trailing_stop_pct'] / 100)
                            position['tsl_price'] = tsl_price

                        elif position['tsl_price'] is not None:
                            new_tsl = position['highest_price'] * (1 - self.trading_config['trailing_stop_pct'] / 100)
                            if new_tsl > position['tsl_price']:
                                position['tsl_price'] = new_tsl

                    # Check exit conditions
                    should_close, reason = self._should_close_position(symbol, current_price, position)

                    if should_close:
                        await self._close_position(symbol, reason)
                        self.trading_stats['trades_closed'] += 1

                except Exception as e:
                    logger.error(f"Position management error for {symbol}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Position management system failed: {e}")

    def _should_close_position(self, symbol: str, current_price: float, position: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if position should be closed"""
        try:
            entry_price = position['entry_price']

            # Check Stop Loss
            if position['sl_price'] and current_price <= position['sl_price']:
                return True, "Stop Loss"

            # Check Take Profit
            if position['tp_price'] and current_price >= position['tp_price']:
                return True, "Take Profit"

            # Check Trailing Stop Loss
            if position['tsl_price'] and current_price <= position['tsl_price']:
                return True, "Trailing Stop Loss"

            # Check emergency conditions
            if self._should_emergency_stop():
                return True, "Emergency Stop"

            # Check per-pair loss limit
            pnl = (current_price - entry_price) * position['amount']
            if pnl <= -self.emergency_config['max_loss_per_pair']:
                return True, "Per-Pair Loss Limit"

            return False, ""

        except Exception as e:
            logger.error(f"Position close check failed for {symbol}: {e}")
            return False, ""

    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return

            # Close position
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side='sell',
                amount=position['amount']
            )

            current_price = order['price']
            pnl = (current_price - position['entry_price']) * position['amount']

            logger.info(f"💰 CLOSED {symbol} at ${current_price:.4f} ({reason}) - P&L: ${pnl:.2f}")

            # Cancel TP/SL orders
            for order_type in ['tp_order_id', 'sl_order_id']:
                if order_type in position:
                    try:
                        self.exchange.cancel_order(position[order_type], symbol)
                    except Exception as e:
                        logger.warning(f"Could not cancel {order_type}: {e}")

            # Update stats
            self.trading_stats['total_pnl'] += pnl

            # Remove position
            del self.positions[symbol]

        except Exception as e:
            logger.error(f"Position close failed for {symbol}: {e}")

    def _should_emergency_stop(self) -> bool:
        """Check if emergency stop should be triggered"""
        try:
            # Check daily loss limit
            if self.trading_stats['total_pnl'] <= -self.emergency_config['max_daily_loss']:
                return True

            # Check position limit
            if len(self.positions) >= self.emergency_config['max_total_positions']:
                return True

            # Check hourly trade limit
            trades_this_hour = sum(1 for pos in self.positions.values()
                                 if (datetime.now() - pos['entry_time']).seconds < 3600)
            if trades_this_hour >= self.emergency_config['max_trades_per_hour']:
                return True

            return False

        except Exception as e:
            logger.error(f"Emergency stop check failed: {e}")
            return False

    def _close_all_positions(self):
        """Close all open positions"""
        logger.info("🔒 Emergency closing all positions...")

        for symbol in list(self.positions.keys()):
            try:
                position = self.positions[symbol]

                # Close position synchronously
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side='sell',
                    amount=position['amount']
                )

                pnl = (order['price'] - position['entry_price']) * position['amount']
                self.trading_stats['total_pnl'] += pnl
                logger.info(f"🔒 CLOSED {symbol} - P&L: ${pnl:.2f}")
                del self.positions[symbol]

            except Exception as e:
                logger.error(f"Emergency close failed for {symbol}: {e}")

    def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            total_trades = self.trading_stats['trades_executed']
            closed_trades = self.trading_stats['trades_closed']

            if closed_trades > 0:
                profitable_trades = sum(1 for pnl in [self.trading_stats['total_pnl']] if pnl > 0)
                self.trading_stats['win_rate'] = profitable_trades / closed_trades

        except Exception as e:
            logger.error(f"Performance stats update failed: {e}")

    def _generate_final_report(self):
        """Generate comprehensive final report"""
        try:
            report = {
                'scanning_session': {
                    'start_time': self.trading_stats['start_time'].isoformat() if self.trading_stats['start_time'] else None,
                    'end_time': self.trading_stats['end_time'].isoformat() if self.trading_stats['end_time'] else None,
                    'duration': str(self.trading_stats['end_time'] - self.trading_stats['start_time']) if self.trading_stats['start_time'] and self.trading_stats['end_time'] else None
                },
                'pairs_analysis': {
                    'total_pairs_discovered': len(self.all_pairs),
                    'active_pairs_filtered': len(self.active_pairs),
                    'pairs_scanned': self.trading_stats['pairs_scanned'],
                    'opportunities_found': self.trading_stats['opportunities_found']
                },
                'trading_performance': {
                    'trades_executed': self.trading_stats['trades_executed'],
                    'trades_closed': self.trading_stats['trades_closed'],
                    'total_pnl': self.trading_stats['total_pnl'],
                    'win_rate': self.trading_stats['win_rate']
                },
                'risk_management': {
                    'risk_per_trade': self.trading_config['risk_per_trade'],
                    'max_positions_total': self.trading_config['max_positions_total'],
                    'emergency_stop_triggered': self._should_emergency_stop(),
                    'final_positions': len(self.positions)
                },
                'top_performing_pairs': self._get_top_performing_pairs()
            }

            # Save report
            report_path = Path(__file__).parent / f"all_pairs_scanner_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"📊 Final report saved to: {report_path}")
            logger.info(f"💰 Session P&L: ${self.trading_stats['total_pnl']:.2f}")
            logger.info(f"📈 Win Rate: {self.trading_stats['win_rate']:.1%}")
            logger.info(f"📊 Pairs Scanned: {self.trading_stats['pairs_scanned']}")

        except Exception as e:
            logger.error(f"Final report generation failed: {e}")

    def _get_top_performing_pairs(self) -> List[Dict[str, Any]]:
        """Get top performing pairs from this session"""
        try:
            pair_performance = {}
            for symbol, position in self.positions.items():
                if position.get('pnl', 0) != 0:
                    pair_performance[symbol] = {
                        'pnl': position['pnl'],
                        'entry_price': position['entry_price'],
                        'current_price': position.get('current_price', position['entry_price']),
                        'performance_pct': (position['pnl'] / (position['entry_price'] * position['amount'])) * 100
                    }

            # Sort by P&L
            sorted_pairs = sorted(pair_performance.items(), key=lambda x: x[1]['pnl'], reverse=True)
            return [dict([('symbol', symbol)] + list(data.items())) for symbol, data in sorted_pairs[:10]]

        except Exception as e:
            logger.error(f"Could not get top performing pairs: {e}")
            return []

def main():
    """Main entry point for all pairs scanner"""
    print("🚀 VIPER All Pairs Scanner - Comprehensive Multi-Pair Trading")

    # Initialize scanner
    scanner = VIPERAllPairsScanner()

    # Display configuration
    print(f"   Total Pairs Available: {len(scanner.all_pairs)}")
    print(f"   Active Pairs Filtered: {len(scanner.active_pairs)}")
    print(f"   Risk per Trade: {scanner.trading_config['risk_per_trade']*100:.1f}%")
    print(f"   Max Total Positions: {scanner.trading_config['max_positions_total']}")
    print(f"   Scan Interval: {scanner.trading_config['scan_interval']} seconds")
    print(f"   Min VIPER Score: {scanner.trading_config['min_viper_score']}")

    top_pairs = sorted(scanner.active_pairs, key=lambda x: x.get('volume_24h', 0), reverse=True)[:10]
    for i, pair in enumerate(top_pairs, 1):
        print(f"   {i}. {pair['symbol']}: ${pair.get('volume_24h', 0):,.0f} (Lev: {pair.get('leverage', 1)}x)")

    print("\n🚀 Trading Mode: LIVE MULTI-PAIR TRADING")
    print("⚠️  This will execute REAL trades across ALL qualified pairs")

    # Confirm start
    confirm = input("\n🚨 Execute REAL LIVE TRADES across ALL PAIRS? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        return

    # Start scanner
    print("\n🚀 Starting comprehensive all-pairs scanner...")
    try:
        scanner.start()
    except KeyboardInterrupt:
        print("\n⏹️ Scanner stopped by user")
    except Exception as e:
        print(f"❌ Scanner error: {e}")

if __name__ == "__main__":
    main()
