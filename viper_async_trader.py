#!/usr/bin/env python3
"""
🚀 VIPER ASYNC TRADING SYSTEM WITH JOBS & TASKS
Advanced concurrent trading with asyncio job management
Features:
- Concurrent scan/score/trade operations
- Background task management
- Real-time position monitoring
- Async API operations for better performance
"""

import os
import asyncio
import aiohttp
import logging
import ccxt.pro as ccxt
import json
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from advanced_trend_detector import AdvancedTrendDetector, TrendConfig, TrendDirection, TrendStrength

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradingJob:
    """Trading job definition"""
    job_id: str
    job_type: str  # 'scan', 'score', 'trade', 'monitor'
    symbol: str = None
    side: str = None
    status: str = 'pending'  # pending, running, completed, failed
    created_at: datetime = None
    completed_at: datetime = None
    result: Dict = None
    error: str = None

@dataclass
class TradingOpportunity:
    """Trading opportunity from scanner"""
    symbol: str
    score: float
    volume: float
    change_24h: float
    price: float
    recommended_side: str
    confidence: float

class ViperAsyncTrader:
    """Advanced async trading system with job management"""
    
    def __init__(self):
        self.exchange = None
        self.session = None
        self.is_running = False
        self.jobs: Dict[str, TradingJob] = {}
        self.active_positions: Dict[str, Dict] = {}
        self.running_tasks: Set[asyncio.Task] = set()
        
        # Configuration
        self.api_key = os.getenv('BITGET_API_KEY', 'bg_d20a392139710bc38b8ab39e970114eb')
        self.api_secret = os.getenv('BITGET_API_SECRET', '23ed4a7fe10b9c947d41a15223647f1b263f0d932b7d5e9e7bdfac01d3b84b36')
        self.api_password = os.getenv('BITGET_API_PASSWORD', '22672267')
        
        # Trading parameters
        self.max_concurrent_jobs = 10
        self.max_positions = 20
        self.risk_per_trade = 0.03
        self.max_leverage = 50

        # TP/SL/TSL Configuration (configurable via environment variables)
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '3.0'))      # % profit target
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '5.0'))          # % stop loss
        self.trailing_stop_pct = float(os.getenv('TRAILING_STOP_PCT', '2.0'))   # % trailing stop
        self.trailing_activation_pct = float(os.getenv('TRAILING_ACTIVATION_PCT', '1.0'))  # % profit to activate trailing
        
        # Job queues
        self.scan_queue = asyncio.Queue()
        self.trade_queue = asyncio.Queue()
        self.monitor_queue = asyncio.Queue()
        
        # Performance metrics
        self.total_jobs = 0
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        
        # Initialize Advanced Trend Detector
        self.trend_config = TrendConfig(
            fast_ma_length=int(os.getenv('FAST_MA_LENGTH', '21')),
            slow_ma_length=int(os.getenv('SLOW_MA_LENGTH', '50')),
            trend_ma_length=int(os.getenv('TREND_MA_LENGTH', '200')),
            atr_length=int(os.getenv('ATR_LENGTH', '14')),
            atr_multiplier=float(os.getenv('ATR_MULTIPLIER', '2.0')),
            min_trend_bars=int(os.getenv('MIN_TREND_BARS', '5')),
            trend_change_threshold=float(os.getenv('TREND_CHANGE_THRESHOLD', '0.02'))
        )
        self.trend_detector = AdvancedTrendDetector(self.trend_config)
        
        logger.info("🚀 VIPER ASYNC TRADER INITIALIZED")
        logger.info(f"💼 Max Concurrent Jobs: {self.max_concurrent_jobs}")
        logger.info(f"📊 Max Positions: {self.max_positions}")
        logger.info(f"🎯 Trend Config: MA({self.trend_config.fast_ma_length},{self.trend_config.slow_ma_length},{self.trend_config.trend_ma_length}) "
                   f"ATR({self.trend_config.atr_length}x{self.trend_config.atr_multiplier})")

    async def get_account_balance(self) -> float:
        """Get USDT balance from swap wallet"""
        try:
            # Fetch balance specifically for swap account
            balance = await self.exchange.fetch_balance({'type': 'swap'})
            if 'USDT' in balance:
                usdt_balance = balance['USDT']['free']
                logger.info(f"💰 Swap Wallet Balance: ${usdt_balance:.2f} USDT (available)")
                return usdt_balance
            else:
                logger.error("❌ USDT balance not found in swap wallet")
                return 0.0
        except Exception as e:
            logger.error(f"❌ Failed to fetch swap wallet balance: {e}")
            # Check if it's an API key issue
            if "Apikey does not exist" in str(e):
                logger.error("🚫 REAL DATA ONLY: Invalid API key - cannot proceed with real trading")
                logger.error("📝 To use real data only:")
                logger.error("   1. Go to https://www.bitget.com/en/account/newapi")
                logger.error("   2. Create a new API key with trading permissions")
                logger.error("   3. Update BITGET_API_KEY, BITGET_API_SECRET, and BITGET_API_PASSWORD in .env")
                logger.error("   4. Restart the live trading engine")
                logger.error("❌ System will not operate with invalid API credentials")
                raise Exception("REAL DATA ONLY: Invalid API credentials - exiting")
            return 0.0

    def calculate_position_size(self, price: float, balance: float, leverage: int = 50):
        """Calculate position size with 3% risk and leverage"""
        try:
            # 3% risk per trade
            risk_per_trade = 0.03
            risk_amount = balance * risk_per_trade

            # Assume 2% stop loss distance (can be adjusted)
            stop_loss_pct = 0.02
            stop_loss_distance = price * stop_loss_pct

            # Calculate base position size (without leverage)
            base_position_size = risk_amount / stop_loss_distance

            # Apply leverage to get actual position size
            leveraged_position_size = base_position_size * leverage

            # Ensure minimum contract size
            min_contract_size = 0.001  # 0.001 BTC minimum
            position_size = max(leveraged_position_size, min_contract_size)

            logger.info(f"🎯 Position Sizing: Balance=${balance:.2f}, Risk=3% (${risk_amount:.2f}), "
                       f"Stop Loss={stop_loss_pct*100}% (${stop_loss_distance:.2f}), "
                       f"Base Size={base_position_size:.6f}, Leveraged Size={leveraged_position_size:.6f} "
                       f"({leverage}x leverage) → Final Size={position_size:.6f}")

            return position_size

        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            # Fallback to minimum size
            return 0.001

    async def connect_exchange(self) -> bool:
        """Connect to Bitget Pro (WebSocket)"""
        try:
            logger.info("🔌 Connecting to Bitget Pro (WebSocket)...")
            
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                    'hedgeMode': True,
                },
                'sandbox': False,
            })
            
            # Create HTTP session for API calls
            self.session = aiohttp.ClientSession()
            
            # Load markets
            await self.exchange.load_markets()
            logger.info(f"✅ Connected to Bitget Pro - {len(self.exchange.markets)} markets")
            
            # Initialize trend detector with same exchange
            self.trend_detector.exchange = self.exchange
            
            # Get available pairs
            swap_pairs = [symbol for symbol, market in self.exchange.markets.items() 
                         if market.get('type') == 'swap' and 'USDT' in symbol]
            logger.info(f"📊 Found {len(swap_pairs)} USDT swap pairs")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to exchange: {e}")
            return False

    def create_job(self, job_type: str, **kwargs) -> TradingJob:
        """Create a new trading job"""
        job_id = f"{job_type}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        job = TradingJob(
            job_id=job_id,
            job_type=job_type,
            symbol=kwargs.get('symbol'),
            side=kwargs.get('side'),
            created_at=datetime.now()
        )
        
        self.jobs[job_id] = job
        self.total_jobs += 1
        
        logger.debug(f"📝 Created job {job_id} ({job_type})")
        return job

    async def execute_job(self, job: TradingJob) -> Dict:
        """Execute a trading job"""
        job.status = 'running'
        
        try:
            if job.job_type == 'scan':
                result = await self.scan_opportunities()
            elif job.job_type == 'score':
                result = await self.score_opportunity(job.symbol)
            elif job.job_type == 'trade':
                result = await self.execute_trade_job(job.symbol, job.side)
            elif job.job_type == 'monitor':
                result = await self.monitor_positions()
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            job.status = 'completed'
            job.completed_at = datetime.now()
            job.result = result
            self.completed_jobs += 1
            
            logger.debug(f"✅ Job {job.job_id} completed")
            return result
            
        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            job.completed_at = datetime.now()
            self.failed_jobs += 1
            
            logger.error(f"❌ Job {job.job_id} failed: {e}")
            return {}

    async def scan_opportunities(self) -> List[TradingOpportunity]:
        """Async market scanning for opportunities"""
        opportunities = []
        
        try:
            # Get random sample of markets for scanning
            all_symbols = [s for s in self.exchange.symbols if 'USDT:USDT' in s]
            scan_symbols = random.sample(all_symbols, min(50, len(all_symbols)))
            
            # Create concurrent tasks for ticker fetching
            tasks = []
            for symbol in scan_symbols:
                if symbol not in self.active_positions:
                    task = asyncio.create_task(self.fetch_ticker_data(symbol))
                    tasks.append(task)
            
            # Wait for all ticker data
            ticker_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(ticker_results):
                if isinstance(result, Exception):
                    continue
                    
                if result:
                    opportunity = await self.score_opportunity_data(result)
                    logger.debug(f"Scored {result['symbol']}: opportunity={opportunity is not None}, score={opportunity.score if opportunity else 'N/A'}")
                    if opportunity and opportunity.score > 0.2:  # Lower threshold for testing
                        opportunities.append(opportunity)
            
            # Sort by score
            opportunities.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"🔍 Scanned {len(scan_symbols)} symbols, found {len(opportunities)} opportunities")
            
        except Exception as e:
            logger.error(f"❌ Error in scan_opportunities: {e}")
        
        return opportunities

    async def fetch_ticker_data(self, symbol: str) -> Optional[Dict]:
        """Fetch ticker data for a single symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)  # Synchronous call
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'volume': ticker.get('quoteVolume', 0),
                'change': ticker.get('percentage', 0),
                'high': ticker.get('high', 0),
                'low': ticker.get('low', 0)
            }
        except Exception as e:
            logger.debug(f"Failed to fetch ticker for {symbol}: {e}")
            return None

    async def score_opportunity(self, symbol: str) -> float:
        """Score a trading opportunity"""
        try:
            ticker_data = await self.fetch_ticker_data(symbol)
            if not ticker_data:
                return 0.0
            
            opportunity = await self.score_opportunity_data(ticker_data)
            return opportunity.score if opportunity else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error scoring {symbol}: {e}")
            return 0.0

    async def score_opportunity_data(self, ticker_data: Dict) -> Optional[TradingOpportunity]:
        """Score opportunity using ENHANCED VIPER algorithm with Advanced Trend Detection"""
        try:
            symbol = ticker_data['symbol']
            price = ticker_data['price']
            volume = ticker_data['volume'] or 0
            change_24h = ticker_data['change'] or 0
            high = ticker_data.get('high', price)
            low = ticker_data.get('low', price)
            
            # ENHANCED VIPER SCORING WEIGHTS with Trend Analysis
            volume_weight = 0.25      # Reduced to make room for trend
            price_weight = 0.25       # Reduced to make room for trend  
            external_weight = 0.15    # Reduced to make room for trend
            range_weight = 0.15       # Reduced to make room for trend
            trend_weight = 0.20       # NEW: Advanced trend analysis
            
            # 1. VOLUME SCORE (0-100) - Historical volume analysis
            volume_score = await self.calculate_volume_score(symbol, volume)
            
            # 2. PRICE SCORE (0-100) - Multi-timeframe momentum
            price_score = await self.calculate_price_score(symbol, price, change_24h)
            
            # 3. EXTERNAL SCORE (0-100) - Market microstructure
            external_score = await self.calculate_external_score(symbol, price)
            
            # 4. RANGE SCORE (0-100) - Volatility analysis
            range_score = await self.calculate_range_score(symbol, price, high, low)
            
            # 5. TREND SCORE (0-100) - Advanced trend detection with ATR & Fibonacci
            trend_score, trend_direction = await self.calculate_advanced_trend_score(symbol)
            
            # Calculate weighted ENHANCED VIPER score
            enhanced_viper_score = (
                volume_score * volume_weight +
                price_score * price_weight +
                external_score * external_weight +
                range_score * range_weight +
                trend_score * trend_weight
            ) / 100.0  # Normalize to 0-1
            
            # Determine signal strength with trend consideration
            if enhanced_viper_score >= 0.9 and trend_score >= 80:
                strength = "VERY_STRONG"
                confidence = 0.95
            elif enhanced_viper_score >= 0.8 and trend_score >= 70:
                strength = "STRONG"  
                confidence = 0.85
            elif enhanced_viper_score >= 0.7 and trend_score >= 60:
                strength = "MODERATE"
                confidence = 0.75
            else:
                strength = "WEAK"
                confidence = 0.5
            
            # ENHANCED side determination using advanced trend analysis
            if trend_direction in [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]:
                recommended_side = 'buy'
                confidence += 0.15  # Boost confidence for trend alignment
            elif trend_direction in [TrendDirection.STRONG_BEARISH, TrendDirection.BEARISH]:
                recommended_side = 'sell'
                confidence += 0.15  # Boost confidence for trend alignment
            elif abs(change_24h) > 3:
                recommended_side = 'buy' if change_24h > 0 else 'sell'
                confidence += 0.05
            elif price_score > 60:  # Bullish momentum
                recommended_side = 'buy'
            elif price_score < 40:  # Bearish momentum
                recommended_side = 'sell'
            else:
                recommended_side = random.choice(['buy', 'sell'])
                confidence *= 0.8  # Reduce confidence for random direction
            
            # Enhanced threshold with trend consideration
            min_score = 0.6 if trend_score >= 70 else 0.65
            
            if enhanced_viper_score > min_score:
                logger.info(f"🎯 Enhanced VIPER Score for {symbol}: {enhanced_viper_score:.3f} ({strength}) - "
                           f"V:{volume_score:.1f} P:{price_score:.1f} E:{external_score:.1f} "
                           f"R:{range_score:.1f} T:{trend_score:.1f} ({trend_direction.value if trend_direction else 'N/A'})")
                
                return TradingOpportunity(
                    symbol=symbol,
                    score=enhanced_viper_score,
                    volume=volume,
                    change_24h=abs(change_24h),
                    price=price,
                    recommended_side=recommended_side,
                    confidence=min(confidence, 1.0)  # Cap at 1.0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error scoring opportunity data: {e}")
            return None

    async def calculate_volume_score(self, symbol: str, current_volume: float) -> float:
        """Calculate VIPER Volume Score - Historical volume analysis"""
        try:
            # Get recent volume data (simulated - in production would use OHLCV history)
            base_score = 50.0
            
            if current_volume <= 0:
                return 0.0
            
            # Volume scoring based on thresholds and relative analysis
            if current_volume > 5000000:  # > $5M volume
                base_score = 95.0
            elif current_volume > 2000000:  # > $2M volume  
                base_score = 85.0
            elif current_volume > 1000000:  # > $1M volume
                base_score = 75.0
            elif current_volume > 500000:   # > $500k volume
                base_score = 65.0
            elif current_volume > 100000:   # > $100k volume
                base_score = 55.0
            else:
                base_score = 30.0
            
            return min(100.0, max(0.0, base_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating volume score for {symbol}: {e}")
            return 50.0

    async def calculate_price_score(self, symbol: str, price: float, change_24h: float) -> float:
        """Calculate VIPER Price Score - Multi-timeframe momentum analysis"""
        try:
            base_score = 50.0  # Neutral
            
            # Short-term momentum (24h change)
            short_momentum = abs(change_24h)
            if change_24h > 0:  # Bullish
                if change_24h > 10:
                    base_score = 95.0
                elif change_24h > 5:
                    base_score = 85.0
                elif change_24h > 3:
                    base_score = 75.0
                elif change_24h > 1:
                    base_score = 65.0
                else:
                    base_score = 55.0
            else:  # Bearish
                if change_24h < -10:
                    base_score = 5.0
                elif change_24h < -5:
                    base_score = 15.0
                elif change_24h < -3:
                    base_score = 25.0
                elif change_24h < -1:
                    base_score = 35.0
                else:
                    base_score = 45.0
            
            return min(100.0, max(0.0, base_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating price score for {symbol}: {e}")
            return 50.0

    async def calculate_external_score(self, symbol: str, price: float) -> float:
        """Calculate VIPER External Score - Market microstructure analysis"""
        try:
            # Simulated market microstructure analysis
            # In production, this would analyze:
            # - Bid/ask spreads
            # - Order book depth
            # - Market maker activity
            # - Cross-exchange arbitrage opportunities
            
            base_score = 50.0
            
            # Simulate spread analysis (tighter spreads = higher scores)
            if 'USDT' in symbol:
                if 'BTC' in symbol or 'ETH' in symbol:
                    base_score = 85.0  # Major pairs have tight spreads
                elif any(coin in symbol for coin in ['BNB', 'SOL', 'ADA']):
                    base_score = 75.0  # Large caps
                else:
                    base_score = 60.0  # Alt coins
            else:
                base_score = 40.0  # Non-USDT pairs typically wider spreads
            
            # Add some randomness to simulate real market microstructure
            import random
            microstructure_factor = random.uniform(0.8, 1.2)
            base_score *= microstructure_factor
            
            return min(100.0, max(0.0, base_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating external score for {symbol}: {e}")
            return 50.0

    async def calculate_range_score(self, symbol: str, price: float, high: float, low: float) -> float:
        """Calculate VIPER Range Score - Volatility and ATR analysis"""
        try:
            if high <= 0 or low <= 0 or price <= 0:
                return 50.0
            
            # Calculate daily range percentage
            daily_range = (high - low) / price * 100
            
            # Current position within range
            range_position = (price - low) / (high - low) if (high - low) > 0 else 0.5
            
            # Range/volatility scoring
            range_score = 50.0
            
            if daily_range > 8:      # Very high volatility
                range_score = 90.0
            elif daily_range > 5:    # High volatility
                range_score = 80.0
            elif daily_range > 3:    # Moderate volatility
                range_score = 70.0
            elif daily_range > 1:    # Low volatility
                range_score = 60.0
            else:                    # Very low volatility
                range_score = 40.0
            
            # Adjust for position in range (mid-range is often better for entries)
            position_factor = 1.0 - abs(range_position - 0.5) * 0.3  # Prefer mid-range
            range_score *= position_factor
            
            return min(100.0, max(0.0, range_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating range score for {symbol}: {e}")
            return 50.0

    async def calculate_advanced_trend_score(self, symbol: str) -> Tuple[float, Optional[TrendDirection]]:
        """Calculate advanced trend score using ATR, MA alignment, and Fibonacci levels"""
        try:
            # Get multi-timeframe trend analysis
            mtf_signals = await self.trend_detector.multi_timeframe_analysis(symbol)
            
            if not mtf_signals:
                return 50.0, TrendDirection.NEUTRAL
            
            # Get consensus trend
            consensus_signal = self.trend_detector.get_consensus_trend(mtf_signals)
            
            if not consensus_signal:
                return 50.0, TrendDirection.NEUTRAL
            
            # Convert trend direction to score
            trend_direction_scores = {
                TrendDirection.STRONG_BULLISH: 95.0,
                TrendDirection.BULLISH: 75.0,
                TrendDirection.NEUTRAL: 50.0,
                TrendDirection.BEARISH: 25.0,
                TrendDirection.STRONG_BEARISH: 5.0
            }
            
            base_score = trend_direction_scores.get(consensus_signal.direction, 50.0)
            
            # Adjust score based on trend strength
            strength_multiplier = {
                TrendStrength.VERY_STRONG: 1.0,
                TrendStrength.STRONG: 0.9,
                TrendStrength.MODERATE: 0.8,
                TrendStrength.WEAK: 0.7,
                TrendStrength.VERY_WEAK: 0.6
            }
            
            multiplier = strength_multiplier.get(consensus_signal.strength, 0.8)
            
            # Adjust score based on confidence
            confidence_boost = consensus_signal.confidence * 10  # 0-1 confidence -> 0-10 boost
            
            # Adjust score based on MA alignment
            ma_boost = 5.0 if consensus_signal.ma_alignment else 0.0
            
            # Calculate final trend score
            final_score = (base_score * multiplier) + confidence_boost + ma_boost
            final_score = min(100.0, max(0.0, final_score))
            
            logger.debug(f"🎯 Trend Analysis for {symbol}: {consensus_signal.direction.value} "
                        f"Strength:{consensus_signal.strength.value} Conf:{consensus_signal.confidence:.2f} "
                        f"Score:{final_score:.1f}")
            
            return final_score, consensus_signal.direction
            
        except Exception as e:
            logger.error(f"❌ Error calculating advanced trend score for {symbol}: {e}")
            return 50.0, TrendDirection.NEUTRAL

    async def execute_trade_job(self, symbol: str, side: str) -> Dict:
        """Execute a trade asynchronously with proper balance and position sizing"""
        try:
            # Get current balance
            balance = await self.get_account_balance()
            if balance <= 0:
                logger.error(f"❌ Insufficient balance for {symbol}: ${balance:.2f}")
                return {}

            # Get current price
            ticker = await self.exchange.fetch_ticker(symbol)
            price = ticker['last']

            # Calculate position size with 3% risk and leverage
            position_size = self.calculate_position_size(price, balance, leverage=self.max_leverage)

            # Enhanced margin validation with exchange-specific checks
            required_margin = (position_size * price) / self.max_leverage
            logger.info(f"   Margin Analysis: Size={position_size:.6f}, Price=${price:.6f}, "
                       f"Leverage={self.max_leverage}x, Required=${required_margin:.2f}, Available=${balance:.2f}")

            # Conservative adjustment: use 90% of available balance to account for exchange fees
            max_safe_margin = balance * 0.9

            if required_margin > max_safe_margin:
                logger.warning(f"⚠️ Required margin (${required_margin:.2f}) exceeds safe limit (${max_safe_margin:.2f}) for {symbol}")
                logger.info(f"   Adjusting position size for safety...")

                # Calculate safe position size
                original_size = position_size
                safe_position_size = (max_safe_margin * self.max_leverage) / price

                # Apply additional safety factor (80% of calculated safe size)
                position_size = safe_position_size * 0.8

                # Recalculate margin with new size
                required_margin = (position_size * price) / self.max_leverage

                logger.info(f"   Position adjusted: {original_size:.6f} → {position_size:.6f}")
                logger.info(f"   Final margin requirement: ${required_margin:.2f} (safe limit: ${max_safe_margin:.2f})")
            else:
                logger.info(f"   ✅ Margin check passed: ${required_margin:.2f} <= ${max_safe_margin:.2f}")

            # Enhanced minimum position size check
            market_info = self.exchange.market(symbol)
            min_contract_size = market_info.get('limits', {}).get('amount', {}).get('min', 0.001)

            if position_size < min_contract_size:
                logger.warning(f"⚠️ Position size ({position_size:.6f}) below exchange minimum ({min_contract_size:.6f}) for {symbol}")
                logger.info("   Skipping this trade opportunity due to size constraints")
                return {}  # Skip this trade opportunity

            # Calculate TP/SL prices
            if side == 'buy':
                take_profit_price = price * (1 + self.take_profit_pct / 100)
                stop_loss_price = price * (1 - self.stop_loss_pct / 100)
                trailing_stop_price = price * (1 - self.trailing_stop_pct / 100)
            else:  # sell/short
                take_profit_price = price * (1 - self.take_profit_pct / 100)
                stop_loss_price = price * (1 + self.stop_loss_pct / 100)
                trailing_stop_price = price * (1 + self.trailing_stop_pct / 100)

            logger.info(f"   TP/SL Setup: TP=${take_profit_price:.6f}, SL=${stop_loss_price:.6f}, "
                       f"TSL=${trailing_stop_price:.6f}")

            # Create main market order
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=position_size,
                params={
                    'leverage': self.max_leverage,
                    'marginMode': 'isolated',
                    'holdSide': 'long' if side == 'buy' else 'short',
                    'tradeSide': 'open'
                }
            )

            # Track position with TP/SL/TSL information
            self.active_positions[symbol] = {
                'side': side,
                'size': position_size,
                'entry_price': price,
                'current_price': price,
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'trailing_stop_price': trailing_stop_price,
                'highest_price': price if side == 'buy' else 0,
                'lowest_price': price if side == 'sell' else float('inf'),
                'trailing_activated': False,
                'leverage': self.max_leverage,
                'timestamp': datetime.now(),
                'order_id': order.get('id', '')
            }
            
            self.total_trades += 1
            
            logger.info(f"✅ Trade executed: {symbol} {side} {position_size:.6f} @ ${price:.6f}")
            
            return {
                'symbol': symbol,
                'side': side,
                'size': position_size,
                'price': price,
                'order_id': order['id']
            }
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed for {symbol}: {e}")
            return {}

    async def monitor_positions(self) -> Dict:
        """Monitor all active positions with enhanced TP/SL/TSL"""
        if not self.active_positions:
            return {'active_positions': 0}

        try:
            # Get current prices for all positions
            monitored = 0

            for symbol, position_data in list(self.active_positions.items()):
                try:
                    # Get current price
                    ticker = await self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']

                    # Update position tracking
                    position_data['current_price'] = current_price
                    entry_price = position_data['entry_price']
                    side = position_data['side']

                    # Calculate P&L percentage
                    if side == 'buy':
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        # Update highest price for trailing stop
                        if current_price > position_data['highest_price']:
                            position_data['highest_price'] = current_price
                    else:  # sell/short
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                        # Update lowest price for trailing stop
                        if current_price < position_data['lowest_price']:
                            position_data['lowest_price'] = current_price

                    logger.debug(f"📊 {symbol}: P&L {pnl_pct:.2f}% (${current_price:.6f})")

                    # Check Take Profit
                    if pnl_pct >= self.take_profit_pct:
                        logger.info(f"🎯 TAKE PROFIT triggered for {symbol}: {pnl_pct:.2f}% profit")
                        await self.close_position(symbol, f"TAKE_PROFIT_{pnl_pct:.2f}%")
                        continue

                    # Check Stop Loss
                    if pnl_pct <= -self.stop_loss_pct:
                        logger.warning(f"🛑 STOP LOSS triggered for {symbol}: {pnl_pct:.2f}% loss")
                        await self.close_position(symbol, f"STOP_LOSS_{pnl_pct:.2f}%")
                        continue

                    # Trailing Stop Loss Logic
                    if side == 'buy':
                        # Activate trailing stop after profit threshold
                        if pnl_pct >= self.trailing_activation_pct and not position_data['trailing_activated']:
                            position_data['trailing_activated'] = True
                            logger.info(f"🚀 Trailing stop activated for {symbol} at {pnl_pct:.2f}% profit")

                        # Update trailing stop if activated
                        if position_data['trailing_activated']:
                            new_trailing_stop = position_data['highest_price'] * (1 - self.trailing_stop_pct / 100)
                            if new_trailing_stop > position_data['trailing_stop_price']:
                                position_data['trailing_stop_price'] = new_trailing_stop
                                logger.debug(f"📈 Updated trailing stop for {symbol}: ${new_trailing_stop:.6f}")

                        # Check trailing stop
                        if current_price <= position_data['trailing_stop_price']:
                            logger.info(f"🎯 TRAILING STOP triggered for {symbol} at ${current_price:.6f}")
                            await self.close_position(symbol, f"TRAILING_STOP_{pnl_pct:.2f}%")
                            continue

                    else:  # sell/short
                        # Activate trailing stop after profit threshold
                        if pnl_pct >= self.trailing_activation_pct and not position_data['trailing_activated']:
                            position_data['trailing_activated'] = True
                            logger.info(f"🚀 Trailing stop activated for {symbol} at {pnl_pct:.2f}% profit")

                        # Update trailing stop if activated
                        if position_data['trailing_activated']:
                            new_trailing_stop = position_data['lowest_price'] * (1 + self.trailing_stop_pct / 100)
                            if new_trailing_stop < position_data['trailing_stop_price']:
                                position_data['trailing_stop_price'] = new_trailing_stop
                                logger.debug(f"📉 Updated trailing stop for {symbol}: ${new_trailing_stop:.6f}")

                        # Check trailing stop
                        if current_price >= position_data['trailing_stop_price']:
                            logger.info(f"🎯 TRAILING STOP triggered for {symbol} at ${current_price:.6f}")
                            await self.close_position(symbol, f"TRAILING_STOP_{pnl_pct:.2f}%")
                            continue

                    monitored += 1

                except Exception as e:
                    logger.error(f"❌ Error monitoring {symbol}: {e}")

            return {
                'active_positions': monitored,
                'total_positions': len(self.active_positions),
                'tp_pct': self.take_profit_pct,
                'sl_pct': self.stop_loss_pct,
                'tsl_pct': self.trailing_stop_pct
            }

        except Exception as e:
            logger.error(f"❌ Error monitoring positions: {e}")
            return {}

    async def close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            if symbol in self.active_positions:
                position_info = self.active_positions[symbol]
                opposite_side = 'sell' if position_info['side'] == 'buy' else 'buy'
                
                await self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=opposite_side,
                    amount=position_info['size'],
                    params={
                        'holdSide': 'long' if position_info['side'] == 'buy' else 'short',
                        'tradeSide': 'close'
                    }
                )
                
                del self.active_positions[symbol]
                logger.info(f"🔄 Closed position {symbol} - {reason}")
                
        except Exception as e:
            logger.error(f"❌ Error closing position {symbol}: {e}")

    async def job_worker(self, worker_id: int):
        """Worker coroutine to process jobs"""
        logger.info(f"👷 Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Try to get jobs from different queues
                job = None
                
                try:
                    # Priority: monitor > trade > scan
                    job = await asyncio.wait_for(self.monitor_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    try:
                        job = await asyncio.wait_for(self.trade_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        try:
                            job = await asyncio.wait_for(self.scan_queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            await asyncio.sleep(1)
                            continue
                
                if job:
                    await self.execute_job(job)
                    
            except Exception as e:
                logger.error(f"❌ Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"👷 Worker {worker_id} stopped")

    async def trading_scheduler(self):
        """Main trading scheduler"""
        logger.info("📅 Trading scheduler started")
        
        while self.is_running:
            try:
                # Schedule scan job every 30 seconds
                scan_job = self.create_job('scan')
                await self.scan_queue.put(scan_job)
                
                # Schedule monitor job every 10 seconds
                monitor_job = self.create_job('monitor')
                await self.monitor_queue.put(monitor_job)
                
                # Process scan results and create trade jobs
                await asyncio.sleep(5)  # Let scan complete
                
                if scan_job.status == 'completed' and scan_job.result:
                    opportunities = scan_job.result
                    
                    # Create trade jobs for top opportunities
                    for i, opp in enumerate(opportunities[:5]):  # Top 5
                        if len(self.active_positions) < self.max_positions:
                            trade_job = self.create_job('trade', symbol=opp.symbol, side=opp.recommended_side)
                            await self.trade_queue.put(trade_job)
                
                # Wait before next cycle
                await asyncio.sleep(25)  # Total 30 second cycle
                
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                await asyncio.sleep(10)

    async def status_reporter(self):
        """Report system status periodically"""
        while self.is_running:
            try:
                pending_jobs = sum(1 for job in self.jobs.values() if job.status == 'pending')
                running_jobs = sum(1 for job in self.jobs.values() if job.status == 'running')
                
                logger.info("=" * 80)
                logger.info("📊 VIPER ASYNC TRADER STATUS")
                logger.info(f"💼 Jobs: {self.total_jobs} total | {self.completed_jobs} completed | {self.failed_jobs} failed")
                logger.info(f"🔄 Active: {running_jobs} running | {pending_jobs} pending")
                logger.info(f"📈 Positions: {len(self.active_positions)} active | {self.total_trades} total trades")
                logger.info(f"💰 Total P&L: ${self.total_pnl:.2f}")
                logger.info("=" * 80)
                
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                logger.error(f"❌ Status reporter error: {e}")
                await asyncio.sleep(30)

    async def run_async_trading(self):
        """Run the async trading system"""
        logger.info("🚀 Starting VIPER Async Trading System")
        self.is_running = True
        
        try:
            # Create worker tasks
            workers = []
            for i in range(self.max_concurrent_jobs):
                worker = asyncio.create_task(self.job_worker(i))
                workers.append(worker)
                self.running_tasks.add(worker)
            
            # Create scheduler and status reporter
            scheduler = asyncio.create_task(self.trading_scheduler())
            reporter = asyncio.create_task(self.status_reporter())
            
            self.running_tasks.add(scheduler)
            self.running_tasks.add(reporter)
            
            logger.info(f"✅ Started {len(workers)} workers")
            logger.info("✅ Started scheduler and status reporter")
            
            # Wait for all tasks
            await asyncio.gather(*self.running_tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested")
        except Exception as e:
            logger.error(f"❌ System error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the trading system"""
        logger.info("🔄 Shutting down async trading system...")
        self.is_running = False
        
        # Cancel all running tasks
        for task in self.running_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks, return_exceptions=True)
        
        # Close exchange connection
        if self.exchange:
            await self.exchange.close()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        logger.info("✅ Async trading system shutdown complete")

async def main():
    """Main async function"""
    logger.info("🚀 VIPER ASYNC TRADER STARTING...")
    
    trader = ViperAsyncTrader()
    
    # Connect to exchange
    if not await trader.connect_exchange():
        logger.error("❌ Failed to connect to exchange")
        return
    
    logger.info("\n" + "=" * 100)
    logger.info("🚨 VIPER ASYNC TRADING SYSTEM")
    logger.info("   - Concurrent job processing")
    logger.info("   - Real-time position monitoring")
    logger.info("   - Advanced opportunity scoring")
    logger.info("   - Async API operations")
    logger.info("   Press Ctrl+C to stop...")
    logger.info("=" * 100)
    
    try:
        await asyncio.sleep(3)
        await trader.run_async_trading()
    except KeyboardInterrupt:
        logger.info("🛑 Trading interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    
    logger.info("✅ VIPER Async Trader stopped")

if __name__ == "__main__":
    asyncio.run(main())
