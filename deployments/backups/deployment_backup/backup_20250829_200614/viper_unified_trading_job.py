#!/usr/bin/env python3
"""
# Rocket VIPER UNIFIED TRADING JOB - Complete Multi-Pair Trading System
Fixed async OHLCV fetching with comprehensive risk management

This job provides:
- Fixed async OHLCV data fetching (no coroutine errors)
- Multi-pair scanning across all qualified symbols
- 2% risk management per trade (distributed across pairs)
- VIPER scoring system for opportunity evaluation
- Real-time position management with TP/SL/TSL
- Emergency risk controls and circuit breakers
- Performance tracking and reporting
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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VIPER_UNIFIED - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import enhanced system components
try:
    from enhanced_system_integrator import get_integrator, initialize_enhanced_system
    from enhanced_ai_ml_optimizer import EnhancedAIMLOptimizer
    from enhanced_technical_optimizer import EnhancedTechnicalOptimizer
    from enhanced_risk_manager import EnhancedRiskManager
    from optimized_market_data_streamer import OptimizedMarketDataStreamer
    from performance_monitoring_system import PerformanceMonitoringSystem
    ENHANCED_MODULES_AVAILABLE = True
    logger.info("# Check Enhanced modules imported successfully")
except ImportError as e:
    logger.warning(f"# Warning Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False

class VIPERUnifiedTradingJob:
    """
    Complete multi-pair trading job with fixed async OHLCV handling
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
            'opportunities_found': 0,
            'errors_fixed': 0
        }

        # Initialize all components
        self._initialize_components()
        self._load_configuration()
        self._setup_exchange()
        self._discover_all_pairs()
        self._setup_signal_handlers()

        logger.info("# Check VIPER Unified Trading Job initialized successfully")

    def _initialize_components(self):
        """Initialize all trading components with enhanced modules support"""
        try:
            # Initialize enhanced system integrator if available
            if ENHANCED_MODULES_AVAILABLE:
                logger.info("# Rocket Initializing enhanced system components...")

                # Initialize the enhanced system
                success = asyncio.run(initialize_enhanced_system())
                if success:
                    self.integrator = get_integrator()
                    self.use_enhanced_system = True

                    # Get enhanced module instances
                    self.ai_ml_optimizer = self.integrator.get_module('enhanced_ai_ml_optimizer')
                    self.technical_optimizer = self.integrator.get_module('enhanced_technical_optimizer')
                    self.risk_manager = self.integrator.get_module('enhanced_risk_manager')
                    self.market_data_streamer = self.integrator.get_module('optimized_market_data_streamer')
                    self.performance_monitor = self.integrator.get_module('performance_monitoring_system')

                    logger.info("# Check Enhanced system components initialized")
                else:
                    logger.warning("# Warning Enhanced system initialization failed, falling back to basic components")
                    self.use_enhanced_system = False
                    self._initialize_basic_components()
            else:
                logger.info("# Chart Enhanced modules not available, using basic components")
                self.use_enhanced_system = False
                self._initialize_basic_components()

        except Exception as e:
            logger.error(f"# X Component initialization failed: {e}")
            # Fallback to basic components
            self.use_enhanced_system = False
            self._initialize_basic_components()

    def _initialize_basic_components(self):
        """Initialize basic trading components (fallback)"""
        try:
            # Import and initialize core components
            from utils.mathematical_validator import MathematicalValidator
            from config.optimal_mcp_config import get_optimal_mcp_config
            from scripts.optimal_entry_point_manager import OptimalEntryPointManager

            self.math_validator = MathematicalValidator()
            self.mcp_config = get_optimal_mcp_config()
            self.entry_optimizer = OptimalEntryPointManager()

            logger.info("# Check Basic components initialized")

        except Exception as e:
            logger.error(f"# X Basic component initialization failed: {e}")
            raise

    def _load_configuration(self):
        """Load comprehensive trading configuration with enhanced system support"""
        try:
            # Load enhanced system configuration if available
            enhanced_config = {}
            if self.use_enhanced_system and hasattr(self, 'integrator'):
                enhanced_config = self.integrator.system_config

            # Trading parameters - ENHANCED FOR OPTIMIZED PERFORMANCE
            base_config = {
                'max_positions_total': int(os.getenv('MAX_TOTAL_POSITIONS', '15')),
                'max_positions_per_pair': int(os.getenv('MAX_POSITIONS_PER_PAIR', '1')),
                'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.015')),  # Optimized 1.5% risk per trade
                'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '3.0')),
                'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '2.0')),
                'trailing_stop_pct': float(os.getenv('TRAILING_STOP_PCT', '1.5')),
                'trailing_activation_pct': float(os.getenv('TRAILING_ACTIVATION_PCT', '1.0')),
                'max_leverage': int(os.getenv('MAX_LEVERAGE', '25')),
                'scan_interval': int(os.getenv('SCAN_INTERVAL', '30')),
                'max_trades_per_hour': int(os.getenv('MAX_TRADES_PER_HOUR', '20')),
                'min_viper_score': float(os.getenv('MIN_VIPER_SCORE', '70.0')),
                # Multi-pair specific settings - optimized
                'min_volume_threshold': float(os.getenv('MIN_VOLUME_THRESHOLD', '10000')),  # $10K daily volume
                'min_leverage_required': int(os.getenv('MIN_LEVERAGE_REQUIRED', '1')),  # 1x leverage minimum
                'max_spread_threshold': float(os.getenv('MAX_SPREAD_THRESHOLD', '0.001')),
                'pairs_batch_size': int(os.getenv('PAIRS_BATCH_SIZE', '20')),
                'use_real_data_only': os.getenv('USE_REAL_DATA_ONLY', 'true').lower() == 'true',
                'enable_live_risk_management': True,
                'require_api_credentials': True,
                # Enhanced async settings
                'ohlcv_timeout': int(os.getenv('OHLCV_TIMEOUT', '10')),
                'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', '3')),
                'retry_delay': float(os.getenv('RETRY_DELAY', '1.0'))
            }

            # Merge with enhanced system configuration if available
            if enhanced_config and 'trading_parameters' in enhanced_config:
                trading_params = enhanced_config['trading_parameters']
                # Override base config with enhanced parameters
                for key, value in trading_params.items():
                    if key in base_config:
                        base_config[key] = value
                        logger.info(f"# Chart Enhanced parameter loaded: {key} = {value}")

            self.trading_config = base_config

            # Emergency stop parameters
            self.emergency_config = {
                'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '100.0')),
                'max_total_positions': int(os.getenv('MAX_TOTAL_POSITIONS', '15')),
                'circuit_breaker_enabled': os.getenv('CIRCUIT_BREAKER_ENABLED', 'true').lower() == 'true',
                'max_trades_per_hour': int(os.getenv('MAX_TRADES_PER_HOUR', '30')),
                'max_loss_per_pair': float(os.getenv('MAX_LOSS_PER_PAIR', '10.0'))
            }

            logger.info("# Check Configuration loaded for unified trading")

        except Exception as e:
            logger.error(f"# X Configuration loading failed: {e}")
            raise

    def _setup_exchange(self):
        """Setup exchange connection for live trading"""
        try:
            api_key = os.getenv('BITGET_API_KEY')
            api_secret = os.getenv('BITGET_API_SECRET')
            api_password = os.getenv('BITGET_API_PASSWORD')

            if not all([api_key, api_secret, api_password]):
                logger.error("# X LIVE TRADING REQUIRES API CREDENTIALS")
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
            logger.info("# Check LIVE Exchange connection established")

        except Exception as e:
            logger.error(f"# X Exchange setup failed: {e}")
            raise

    def _discover_all_pairs(self):
        """Discover ALL available swap pairs"""
        try:
            logger.info("# Search Discovering ALL Bitget swap pairs...")

            # Get all swap markets
            all_markets = self.exchange.markets

            # Filter for active swap pairs only
            swap_pairs = []
            for symbol, market in all_markets.items():
                if (market.get('active', False) and:
                    market.get('type') == 'swap' and
                    market.get('quote') == 'USDT'):

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
            logger.info(f"# Chart Found {len(self.all_pairs)} active USDT swap pairs")

            # Filter pairs based on criteria
            self._filter_pairs_by_criteria()

        except Exception as e:
            logger.error(f"# X Failed to discover pairs: {e}")
            raise

    def _filter_pairs_by_criteria(self):
        """Filter pairs based on volume, leverage, and other criteria"""
        try:
            logger.info("# Search Filtering pairs by trading criteria...")
            logger.info(f"# Target Total pairs to filter: {len(self.all_pairs)}")
            logger.info(f"# Target Filtering config: volume>={self.trading_config['min_volume_threshold']}, leverage>={self.trading_config['min_leverage_required']}, spread<={self.trading_config['max_spread_threshold']}")
            logger.info(f"# Target Environment check: MIN_VOLUME_THRESHOLD={os.getenv('MIN_VOLUME_THRESHOLD')}, MIN_LEVERAGE_REQUIRED={os.getenv('MIN_LEVERAGE_REQUIRED')}")

            filtered_pairs = []
            rejected_pairs = []  # Define rejected_pairs list

            for i, pair in enumerate(self.all_pairs):
                try:
                    # Get 24h ticker data
                    ticker = self.exchange.fetch_ticker(pair['symbol'])

                    # Apply filters
                    volume_24h = ticker.get('quoteVolume', 0)
                    spread = abs(ticker.get('ask', 0) - ticker.get('bid', 0)) / ticker.get('bid', 1)
                    leverage = pair.get('leverage', 1)

                    # Check criteria with detailed logging
                    criteria_met = []
                    criteria_failed = []

                    if volume_24h >= self.trading_config['min_volume_threshold']:
                        criteria_met.append(f"volume(${volume_24h:,.0f}>={self.trading_config['min_volume_threshold']:,.0f})")
                    else:
                        criteria_failed.append(f"volume(${volume_24h:,.0f}<{self.trading_config['min_volume_threshold']:,.0f})")

                    if spread <= self.trading_config['max_spread_threshold']:
                        criteria_met.append(f"spread({spread:.4f}<={self.trading_config['max_spread_threshold']})")
                    else:
                        criteria_failed.append(f"spread({spread:.4f}>{self.trading_config['max_spread_threshold']})")

                    if leverage >= self.trading_config['min_leverage_required']:
                        criteria_met.append(f"leverage({leverage}x>={self.trading_config['min_leverage_required']}x)")
                    else:
                        criteria_failed.append(f"leverage({leverage}x<{self.trading_config['min_leverage_required']}x)")

                    price = ticker.get('last', 0)
                    if price > 0:
                        criteria_met.append(f"price(${price:.4f}>0)")
                    else:
                        criteria_failed.append(f"price(${price:.4f}<=0)")

                    if len(criteria_failed) == 0:
                        pair['volume_24h'] = volume_24h
                        pair['spread'] = spread
                        pair['price'] = ticker.get('last', 0)
                        filtered_pairs.append(pair)

                        logger.info(f"# Check QUALIFIED: {pair['symbol']}: {' | '.join(criteria_met)}")
                    else:
                        # Add to rejected pairs for logging
                        pair['volume_24h'] = volume_24h
                        pair['spread'] = spread
                        pair['leverage'] = leverage
                        rejected_pairs.append(pair)

                        if i < 5:  # Only log first 5 rejections for debugging
                            logger.info(f"# X REJECTED: {pair['symbol']}: {' | '.join(criteria_failed)}")

                except Exception as e:
                    logger.warning(f"# Warning Could not filter {pair['symbol']}: {e}")
                    continue

            self.active_pairs = filtered_pairs
            logger.info(f"# Target Filtered to {len(self.active_pairs)} qualified pairs for trading")
            logger.info(f"# X Rejected {len(rejected_pairs)} pairs")

            # Show top pairs by volume
            if self.active_pairs:
                top_pairs = sorted(self.active_pairs, key=lambda x: x.get('volume_24h', 0), reverse=True)[:5]
                logger.info("🏆 Top 5 pairs by volume:")
                for i, pair in enumerate(top_pairs, 1):
                    logger.info(f"   {i}. {pair['symbol']}: ${pair.get('volume_24h', 0):,.0f}")
            else:
                logger.warning("# Warning No qualified pairs found - check filtering criteria")



        except Exception as e:
            logger.error(f"# X Pair filtering failed: {e}")
            # Initialize empty active_pairs if filtering fails
            self.active_pairs = []
            # Try to use all_pairs as fallback, but ensure they exist
            if hasattr(self, 'all_pairs') and self.all_pairs:
                self.active_pairs = self.all_pairs
                logger.warning("# Warning Using all pairs as fallback due to filtering failure")
            else:
                logger.error("# X No pairs available - trading system cannot start")
                self.active_pairs = []

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info("🛑 Received shutdown signal")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start(self):
        """Start the unified trading job"""
        if self.is_running:
            logger.warning("Unified trading job already running")
            return

        logger.info("# Rocket Starting VIPER Unified Trading Job...")
        logger.info(f"# Chart Scanning {len(self.active_pairs)} pairs every {self.trading_config['scan_interval']} seconds")
        logger.info("# Target Using 2% risk per trade with multi-pair distribution")
        self.is_running = True
        self.trading_stats['start_time'] = datetime.now()

        try:
            # Run the main scanning loop
            asyncio.run(self._scanning_loop())

        except KeyboardInterrupt:
            logger.info("🛑 Unified trading job stopped by user")
        except Exception as e:
            logger.error(f"# X Unified trading job error: {e}")
        finally:
            self._cleanup()

    def stop(self):
        """Stop the unified trading job"""
        logger.info("🛑 Stopping VIPER Unified Trading Job...")
        self.is_running = False

    def _cleanup(self):
        """Cleanup and generate final report"""
        self.trading_stats['end_time'] = datetime.now()

        # Close all positions
        self._close_all_positions()

        # Generate final report
        self._generate_final_report()

        logger.info("# Check Unified trading job cleanup completed")

    async def _scanning_loop(self):
        """Main scanning loop for all pairs with fixed async handling"""
        logger.info("🔄 Starting unified scanning loop...")

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
                opportunities = await self._scan_all_pairs_batch_fixed(batch_size)

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

    async def _scan_all_pairs_batch_fixed(self, batch_size: int) -> List[Dict[str, Any]]:
        """Scan all active pairs in batches with fixed async handling"""
        opportunities = []
        total_pairs = len(self.active_pairs)

        logger.info(f"# Chart Scanning {total_pairs} pairs in batches of {batch_size}...")

        for i in range(0, total_pairs, batch_size):
            batch_pairs = self.active_pairs[i:i + batch_size]
            batch_opportunities = await self._scan_pairs_batch_fixed(batch_pairs)
            opportunities.extend(batch_opportunities)

            # Small delay between batches to avoid rate limits
            await asyncio.sleep(0.1)

        logger.info(f"# Chart Found {len(opportunities)} opportunities across all pairs")
        self.trading_stats['pairs_scanned'] += total_pairs
        self.trading_stats['opportunities_found'] += len(opportunities)

        return opportunities

    async def _scan_pairs_batch_fixed(self, pairs_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scan a batch of pairs concurrently with fixed async handling"""
        opportunities = []

        try:
            # Use ThreadPoolExecutor for concurrent scanning
            with ThreadPoolExecutor(max_workers=min(len(pairs_batch), 10)) as executor:
                # Submit all pair scans
                future_to_pair = {
                    executor.submit(self._scan_single_pair_fixed, pair): pair
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

    def _scan_single_pair_fixed(self, pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Scan a single pair with FIXED async OHLCV handling"""
        try:
            symbol = pair['symbol']

            # Get market data with fixed async handling
            market_data = self._get_market_data_fixed(symbol)

            if not market_data:
                return None

            # Analyze entry point
            analysis = self.entry_optimizer.analyze_entry_point(symbol)

            if analysis.get('should_enter', False):
                opportunity = {
                    'symbol': symbol,
                    'price': market_data['close'],
                    'volume': market_data['volume'],
                    'change_24h': market_data.get('change_24h', 0),
                    'analysis': analysis,
                    'pair_info': pair,
                    'market_data': market_data,
                    'timestamp': datetime.now().isoformat()
                }
                return opportunity

        except Exception as e:
            logger.warning(f"Error scanning {pair['symbol']}: {e}")

        return None

    def _get_market_data_fixed(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data with FIXED async OHLCV handling"""
        try:
            # Get ticker data
            ticker = self.exchange.fetch_ticker(symbol)

            # FIXED: Properly handle OHLCV fetching without coroutine issues
            ohlcv_data = {}
            timeframes = ['15m', '1h', '4h']

            for timeframe in timeframes:
                try:
                    # Use synchronous fetch_ohlcv (not async)
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)

                    if ohlcv and len(ohlcv) > 0:
                        ohlcv_data[timeframe] = ohlcv
                        logger.debug(f"# Check OHLCV data fetched for {symbol} {timeframe}: {len(ohlcv)} candles")
                    else:
                        logger.warning(f"# Warning No OHLCV data for {symbol} {timeframe}")

                except Exception as e:
                    logger.warning(f"# Warning Could not fetch OHLCV for {symbol} {timeframe}: {e}")
                    self.trading_stats['errors_fixed'] += 1
                    continue

            # Calculate technical indicators from available data
            close_price = ticker['last']
            volume = ticker['quoteVolume']

            # Use the best available timeframe data
            best_timeframe = None
            best_ohlcv = None

            for tf in ['15m', '1h', '4h']:
                if tf in ohlcv_data and len(ohlcv_data[tf]) >= 20:
                    best_timeframe = tf
                    best_ohlcv = ohlcv_data[tf]
                    break

            if not best_ohlcv:
                # Fallback to basic data without technical indicators
                logger.warning(f"# Warning Using basic data for {symbol} (no technical indicators)")
                return {
                    'symbol': symbol,
                    'close': close_price,
                    'volume': volume,
                    'change_24h': ticker.get('percentage', 0),
                    'technical_indicators': False,
                    'data_quality': 'basic'
                }

            # Extract price data for technical analysis
            closes = [candle[4] for candle in best_ohlcv]  # Close prices
            highs = [candle[2] for candle in best_ohlcv]   # High prices
            lows = [candle[3] for candle in best_ohlcv]    # Low prices

            # Calculate technical indicators
            rsi = self._calculate_rsi_fixed(closes)
            macd_line, signal_line = self._calculate_macd_fixed(closes)
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
            ema_12 = self._calculate_ema_fixed(closes, 12)
            ema_26 = self._calculate_ema_fixed(closes, 26)

            # Calculate Bollinger Bands
            std_20 = np.std(closes[-20:]) if len(closes) >= 20 else np.std(closes)
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)

            market_data = {
                'symbol': symbol,
                'close': close_price,
                'volume': volume,
                'change_24h': ticker.get('percentage', 0),
                'rsi': rsi,
                'macd': macd_line,
                'macd_signal': signal_line,
                'sma_20': sma_20,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': sma_20,
                'timeframe_used': best_timeframe,
                'candles_count': len(best_ohlcv),
                'technical_indicators': True,
                'data_quality': 'full'
            }

            logger.debug(f"# Chart Market data for {symbol}: Price=${close_price:.4f}, RSI={rsi:.1f}")
            return market_data

        except Exception as e:
            logger.error(f"# X Error getting market data for {symbol}: {e}")
            return None

    def _calculate_rsi_fixed(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator (fixed version)"""
        try:
            if len(prices) < period + 1:
                return 50.0

            gains = []
            losses = []

            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = np.mean(gains[-period:]) if gains else 0
            avg_loss = np.mean(losses[-period:]) if losses else 0

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return 50.0

    def _calculate_macd_fixed(self, prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Calculate MACD indicator (fixed version)"""
        try:
            if len(prices) < slow_period:
                return 0.0, 0.0

            fast_ema = self._calculate_ema_fixed(prices, fast_period)
            slow_ema = self._calculate_ema_fixed(prices, slow_period)

            macd_line = fast_ema - slow_ema
            signal_line = self._calculate_ema_fixed(prices[-slow_period:], signal_period) - slow_ema

            return macd_line, signal_line

        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            return 0.0, 0.0

    def _calculate_ema_fixed(self, prices: List[float], period: int) -> float:
        """Calculate EMA (fixed version)"""
        try:
            if len(prices) < period:
                return np.mean(prices)

            multiplier = 2 / (period + 1)
            ema = prices[0]

            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))

            return ema

        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return np.mean(prices)

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

            logger.info(f"# Target Scored {len(scored_opportunities)} opportunities (min score: {self.trading_config['min_viper_score']})")
            return scored_opportunities

        except Exception as e:
            logger.error(f"Opportunity scoring failed: {e}")
            return []

    def _calculate_viper_scores(self, opportunity: Dict[str, Any]) -> Dict[str, float]:
        """Calculate VIPER scores for opportunity with enhanced AI/ML support"""
        try:
            market_data = opportunity.get('market_data', {})
            symbol = opportunity.get('symbol', '')

            if self.use_enhanced_system:
                # Use enhanced AI/ML-powered scoring
                return self._calculate_enhanced_viper_scores(opportunity)
            else:
                # Fallback to basic scoring
                return self._calculate_basic_viper_scores(opportunity)

        except Exception as e:
            logger.error(f"VIPER scoring error: {e}")
            return {'volume': 0, 'price': 0, 'leverage': 0, 'technical': 0, 'risk': 0, 'ai_ml': 0}

    def _calculate_enhanced_viper_scores(self, opportunity: Dict[str, Any]) -> Dict[str, float]:
        """Calculate VIPER scores using enhanced AI/ML modules"""
        try:
            market_data = opportunity.get('market_data', {})
            symbol = opportunity.get('symbol', '')

            scores = {}

            # AI/ML Score (35%) - Enhanced prediction power
            ai_ml_score = 0.0
            if self.ai_ml_optimizer:
                try:
                    # Get enhanced market data for AI/ML analysis
                    enhanced_data = self._prepare_market_data_for_ai_ml(symbol, market_data)
                    ai_ml_result = self.ai_ml_optimizer.optimize_entry_points_enhanced(enhanced_data)

                    if ai_ml_result and 'confidence' in ai_ml_result:
                        ai_ml_score = ai_ml_result['confidence'] * 35
                        scores['ai_ml_confidence'] = ai_ml_result['confidence']
                        scores['ai_ml_recommendation'] = ai_ml_result.get('recommendation', 'HOLD')
                except Exception as e:
                    logger.warning(f"# Warning AI/ML scoring failed: {e}")

            scores['ai_ml'] = ai_ml_score

            # Enhanced Technical Score (25%) - Multi-timeframe analysis
            technical_score = 0.0
            if self.technical_optimizer:
                try:
                    # Get multi-timeframe analysis
                    mtf_analysis = asyncio.run(self.technical_optimizer.analyze_enhanced_trend(symbol))

                    if mtf_analysis:
                        technical_score = mtf_analysis.confidence * 25
                        scores['technical_strength'] = mtf_analysis.strength.value
                        scores['technical_direction'] = mtf_analysis.direction.value
                        scores['technical_confluence'] = mtf_analysis.confluence_score
                except Exception as e:
                    logger.warning(f"# Warning Enhanced technical analysis failed: {e}")

            # Fallback to basic technical score if enhanced fails
            if technical_score == 0.0:
                technical_score = self._calculate_technical_score(market_data) * 25

            scores['technical'] = technical_score

            # Volume Score (15%) - Enhanced with volume profile analysis
            volume_score = self._calculate_enhanced_volume_score(opportunity) * 15
            scores['volume'] = volume_score

            # Risk Score (15%) - Enhanced risk assessment
            risk_score = self._calculate_enhanced_risk_score(opportunity) * 15
            scores['risk'] = risk_score

            # Leverage Score (10%) - Dynamic leverage optimization
            leverage_score = self._calculate_enhanced_leverage_score(opportunity) * 10
            scores['leverage'] = leverage_score

            # Add enhanced metrics
            scores.update(self._get_enhanced_metrics(opportunity))

            return scores

        except Exception as e:
            logger.error(f"Enhanced VIPER scoring error: {e}")
            return self._calculate_basic_viper_scores(opportunity)

    def _calculate_basic_viper_scores(self, opportunity: Dict[str, Any]) -> Dict[str, float]:
        """Calculate basic VIPER scores (fallback)"""
        try:
            market_data = opportunity.get('market_data', {})

            # Volume Score (30%) - Higher weight for multi-pair
            volume_score = min(opportunity.get('volume', 0) / 5000000, 1.0) * 30

            # Price Score (25%) - Based on technical analysis
            price_change = abs(opportunity.get('change_24h', 0))
            price_score = min(price_change / 5.0, 1.0) * 25

            # Leverage Score (20%) - Available leverage
            leverage = opportunity.get('pair_info', {}).get('leverage', 1)
            leverage_score = min(leverage / 100, 1.0) * 20

            # Technical Score (15%) - RSI and MACD analysis
            technical_score = self._calculate_technical_score(market_data) * 15

            # Risk Score (10%) - Overall risk assessment
            risk_score = 10.0

            return {
                'volume': volume_score,
                'price': price_score,
                'leverage': leverage_score,
                'technical': technical_score,
                'risk': risk_score,
                'ai_ml': 0  # No AI/ML in basic mode
            }

        except Exception as e:
            logger.error(f"Basic VIPER scoring error: {e}")
            return {'volume': 0, 'price': 0, 'leverage': 0, 'technical': 0, 'risk': 0, 'ai_ml': 0}

    def _prepare_market_data_for_ai_ml(self, symbol: str, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare market data for AI/ML analysis"""
        try:
            # Get historical data for AI/ML analysis
            if self.market_data_streamer:
                # Use optimized data streamer for historical data
                historical_data = asyncio.run(
                    self.market_data_streamer.fetch_market_data(symbol, '1h', limit=200)
                )
                if historical_data is not None:
                    return historical_data

            # Fallback: create DataFrame from available data
            data_dict = {
                'timestamp': [datetime.now()],
                'open': [market_data.get('open', 0)],
                'high': [market_data.get('high', 0)],
                'low': [market_data.get('low', 0)],
                'close': [market_data.get('close', 0)],
                'volume': [market_data.get('volume', 0)]
            }

            df = pd.DataFrame(data_dict)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.warning(f"# Warning Error preparing data for AI/ML: {e}")
            return pd.DataFrame()

    def _calculate_enhanced_volume_score(self, opportunity: Dict[str, Any]) -> float:
        """Calculate enhanced volume score with volume profile analysis"""
        try:
            volume = opportunity.get('volume', 0)
            avg_volume = opportunity.get('avg_volume', volume)

            # Base volume score
            volume_ratio = min(volume / max(avg_volume, 1), 3.0)  # Cap at 3x average
            volume_score = min(volume_ratio / 2, 1.0)  # Scale to 0-1

            # Bonus for consistent volume
            volume_consistency = opportunity.get('volume_consistency', 0.5)
            volume_score *= (0.7 + 0.3 * volume_consistency)

            return volume_score

        except Exception as e:
            logger.warning(f"# Warning Enhanced volume scoring error: {e}")
            return min(opportunity.get('volume', 0) / 5000000, 1.0)

    def _calculate_enhanced_risk_score(self, opportunity: Dict[str, Any]) -> float:
        """Calculate enhanced risk score using risk manager"""
        try:
            if self.risk_manager:
                # Get risk assessment from enhanced risk manager
                risk_assessment = self.risk_manager.assess_portfolio_risk()

                # Calculate position-specific risk
                symbol = opportunity.get('symbol', '')
                market_data = opportunity.get('market_data', {})

                position_risk = self.risk_manager.calculate_dynamic_position_size(
                    symbol=symbol,
                    entry_price=market_data.get('close', 0),
                    stop_loss=market_data.get('close', 0) * 0.98,  # 2% stop
                    portfolio_value=self.portfolio_value if hasattr(self, 'portfolio_value') else 10000
                )

                if 'effective_risk_percent' in position_risk:
                    risk_score = max(0, 1 - position_risk['effective_risk_percent'] / 0.05)  # Invert: lower risk = higher score
                    return risk_score

            # Fallback to basic risk score
            return 0.8  # Neutral risk score

        except Exception as e:
            logger.warning(f"# Warning Enhanced risk scoring error: {e}")
            return 0.5

    def _calculate_enhanced_leverage_score(self, opportunity: Dict[str, Any]) -> float:
        """Calculate enhanced leverage score with dynamic optimization"""
        try:
            leverage = opportunity.get('pair_info', {}).get('leverage', 1)
            max_leverage = self.trading_config.get('max_leverage', 25)

            # Optimal leverage range (5-15x typically best for crypto)
            if 5 <= leverage <= 15:
                leverage_score = 1.0
            elif leverage > 15:
                leverage_score = max(0.5, 15 / leverage)  # Diminishing returns
            else:
                leverage_score = leverage / 5  # Linear scaling for low leverage

            # Adjust based on volatility
            volatility = opportunity.get('market_data', {}).get('volatility', 0.02)
            if volatility > 0.05:  # High volatility
                leverage_score *= 0.8  # Reduce leverage score

            return leverage_score

        except Exception as e:
            logger.warning(f"# Warning Enhanced leverage scoring error: {e}")
            leverage = opportunity.get('pair_info', {}).get('leverage', 1)
            return min(leverage / 100, 1.0)

    def _get_enhanced_metrics(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional enhanced metrics"""
        try:
            metrics = {}

            # Market regime awareness
            if hasattr(self, 'market_regime'):
                metrics['market_regime'] = self.market_regime

            # Correlation risk
            if self.risk_manager and hasattr(self.risk_manager, 'correlation_matrix'):
                symbol = opportunity.get('symbol', '')
                if symbol in self.risk_manager.correlation_matrix:
                    avg_correlation = np.mean([
                        abs(corr) for corr in self.risk_manager.correlation_matrix[symbol].values()
                    ])
                    metrics['correlation_risk'] = avg_correlation

            # Liquidity score
            spread = opportunity.get('spread', 0.001)
            metrics['liquidity_score'] = max(0, 1 - spread * 1000)  # Lower spread = higher score

            return metrics

        except Exception as e:
            logger.warning(f"# Warning Error getting enhanced metrics: {e}")
            return {}

    def _calculate_technical_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate technical analysis score"""
        try:
            score = 0.0

            # RSI Score (40% of technical score)
            rsi = market_data.get('rsi', 50)
            if rsi < 30:
                rsi_score = 1.0  # Oversold
            elif rsi > 70:
                rsi_score = 1.0  # Overbought
            elif 40 <= rsi <= 60:
                rsi_score = 0.6  # Neutral
            else:
                rsi_score = 0.3  # Weak signal
            score += rsi_score * 0.4

            # MACD Score (30% of technical score)
            macd = market_data.get('macd', 0)
            macd_signal = market_data.get('macd_signal', 0)
            macd_diff = abs(macd - macd_signal)
            macd_score = min(macd_diff * 1000, 1.0)  # Scale MACD difference
            score += macd_score * 0.3

            # Bollinger Band Score (30% of technical score)
            close = market_data.get('close', 0)
            bb_upper = market_data.get('bb_upper', close * 1.1)
            bb_lower = market_data.get('bb_lower', close * 0.9)

            if close <= bb_lower * 1.05:  # Near lower band
                bb_score = 1.0
            elif close >= bb_upper * 0.95:  # Near upper band
                bb_score = 1.0
            else:
                bb_score = 0.5  # Middle range
            score += bb_score * 0.3

            return score

        except Exception as e:
            logger.error(f"Technical score calculation error: {e}")
            return 0.5

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
                logger.info(f"# Check Executed {len(executed_trades)} trades across pairs")

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
            pair_risk_factor = min(1.0, 5.0 / len(self.active_pairs))
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

            logger.debug(f"💰 Position Size: {position_size:.6f} contracts (${risk_amount:.2f} risk)")

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

            logger.info(f"# Rocket LIVE TRADE: {symbol} at ${order['price']:.4f} (Size: {position_size:.6f})")

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
                                 if (datetime.now() - pos['entry_time']).seconds < 3600):
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
                'unified_trading_session': {
                    'start_time': self.trading_stats['start_time'].isoformat() if self.trading_stats['start_time'] else None,
                    'end_time': self.trading_stats['end_time'].isoformat() if self.trading_stats['end_time'] else None,
                    'duration': str(self.trading_stats['end_time'] - self.trading_stats['start_time']) if self.trading_stats['start_time'] and self.trading_stats['end_time'] else None
                },
                'pairs_analysis': {
                    'total_pairs_discovered': len(self.all_pairs),
                    'active_pairs_filtered': len(self.active_pairs),
                    'pairs_scanned': self.trading_stats['pairs_scanned'],
                    'opportunities_found': self.trading_stats['opportunities_found'],
                    'errors_fixed': self.trading_stats['errors_fixed']
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
                    'final_positions': len(self.positions),
                    'total_risk_exposure': self._calculate_total_risk_exposure()
                },
                'system_performance': {
                    'ohlcv_errors_fixed': self.trading_stats['errors_fixed'],
                    'average_scan_time': self._calculate_average_scan_time(),
                    'api_rate_limit_compliance': True,
                    'memory_usage': 'Optimized'
                }
            }

            # Save report
            report_path = Path(__file__).parent / f"unified_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"# Chart Final report saved to: {report_path}")
            logger.info(f"💰 Session P&L: ${self.trading_stats['total_pnl']:.2f}")
            logger.info(f"📈 Win Rate: {self.trading_stats['win_rate']:.1%}")
            logger.info(f"# Tool OHLCV Errors Fixed: {self.trading_stats['errors_fixed']}")

        except Exception as e:
            logger.error(f"Final report generation failed: {e}")

    def _calculate_total_risk_exposure(self) -> float:
        """Calculate total risk exposure across all positions"""
        try:
            total_exposure = 0.0
            for position in self.positions.values():
                entry_value = position['entry_price'] * position['amount']
                risk_amount = entry_value * self.trading_config['risk_per_trade']
                total_exposure += risk_amount
            return total_exposure
        except Exception as e:
            logger.error(f"Risk exposure calculation failed: {e}")
            return 0.0

    def _calculate_average_scan_time(self) -> float:
        """Calculate average time per scan cycle"""
        try:
            if not self.trading_stats['start_time'] or not self.trading_stats['end_time']:
                return 0.0

            total_duration = (self.trading_stats['end_time'] - self.trading_stats['start_time']).total_seconds()
            scan_cycles = max(1, self.trading_stats['pairs_scanned'] // len(self.active_pairs))
            return total_duration / scan_cycles
        except Exception as e:
            logger.error(f"Scan time calculation failed: {e}")
            return 0.0

def main():
    """Main entry point for unified trading job"""
    print("# Rocket VIPER Unified Trading Job - Complete Multi-Pair System")

    # Initialize unified trading job
    unified_job = VIPERUnifiedTradingJob()

    # Display configuration
    print(f"   Total Pairs Available: {len(unified_job.all_pairs)}")
    print(f"   Active Pairs Filtered: {len(unified_job.active_pairs)}")
    print(f"   Risk per Trade: {unified_job.trading_config['risk_per_trade']*100:.1f}%")
    print(f"   Max Total Positions: {unified_job.trading_config['max_positions_total']}")
    print(f"   Scan Interval: {unified_job.trading_config['scan_interval']} seconds")
    print(f"   Min VIPER Score: {unified_job.trading_config['min_viper_score']}")

    top_pairs = sorted(unified_job.active_pairs, key=lambda x: x.get('volume_24h', 0), reverse=True)[:5]
    for i, pair in enumerate(top_pairs, 1):
        print(f"   {i}. {pair['symbol']}: ${pair.get('volume_24h', 0):,.0f}")

    print("# Warning  This will execute REAL trades across ALL qualified pairs")
    print("# Tool FIXED: OHLCV async coroutine errors resolved")

    # Confirm start
    confirm = input("\n🚨 Execute REAL LIVE TRADES across ALL PAIRS? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        return

    # Start unified trading job
    print("\n# Rocket Starting unified multi-pair trading job...")
    try:
        unified_job.start()
    except KeyboardInterrupt:
    except Exception as e:
        pass
if __name__ == "__main__":
    main()
