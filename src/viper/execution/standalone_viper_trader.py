#!/usr/bin/env python3
"""
# Rocket VIPER Trading System - Standalone Trading Component
Complete Scan → Score → Trade → TP/SL Flow in One Script

Features:
- Market scanning for multiple trading pairs
- VIPER score calculation (Volume, Price, External, Range)  
- Automated trade execution with risk management
- Take Profit (TP) and Stop Loss (SL) monitoring
- Real-time position management
- Comprehensive logging and safety controls

Usage:
    python standalone_viper_trader.py

Requirements:
    - Bitget API credentials in .env file
    - Python packages: ccxt, requests, python-dotenv
"""

import os
import sys
import time
import json
import logging
import ccxt
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import threading
import signal

# Load environment variables
load_dotenv()

@dataclass
class TradingPosition:
    """Data class for tracking active positions"""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: str
    order_id: str = None
    unrealized_pnl: float = 0.0

@dataclass 
class VIPERSignal:
    """Data class for VIPER trading signals with execution cost awareness"""
    symbol: str
    signal: str  # 'LONG', 'SHORT', or 'HOLD'
    viper_score: float
    price: float
    price_change: float
    volume: float
    confidence: float
    stop_loss: float
    take_profit: float
    timestamp: str
    execution_cost: float = 0.0  # Expected execution cost in USD
    order_type: str = "MARKET"   # Recommended order type

class StandaloneVIPERTrader:
    """Complete standalone VIPER trading system"""
    
    def __init__(self):
        """Initialize the trading system"""
        self.setup_logging()
        self.logger.info("# Rocket Initializing VIPER Standalone Trader...")
        
        # Configuration
        self.config = self.load_configuration()
        self.exchange = None
        self.active_positions = {}  # {symbol: TradingPosition}
        self.running = True
        
        # Trading parameters - optimized for execution cost awareness
        self.max_positions = self.config.get('max_positions', 5)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)  # 2%
        self.viper_threshold = self.config.get('viper_threshold', 50.0)  # Lowered from 85 due to stricter scoring
        self.scan_interval = self.config.get('scan_interval', 30)  # seconds
        
        # Trading pairs to monitor
        self.trading_pairs = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 
            'BNB/USDT:USDT', 'ADA/USDT:USDT', 'DOT/USDT:USDT',
            'MATIC/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT'
        ]
        
        # Initialize exchange connection
        self.initialize_exchange()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('viper_trader.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_configuration(self) -> Dict:
        """Load trading configuration from environment"""
        return {
            'max_positions': int(os.getenv('MAX_POSITIONS', '15')),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),
            'viper_threshold': float(os.getenv('VIPER_THRESHOLD', '85.0')),
            'scan_interval': int(os.getenv('SCAN_INTERVAL', '30')),
            'stop_loss_percent': float(os.getenv('STOP_LOSS_PERCENT', '0.02')),  # 2%
            'take_profit_percent': float(os.getenv('TAKE_PROFIT_PERCENT', '0.04')),  # 4%
            'daily_loss_limit': float(os.getenv('DAILY_LOSS_LIMIT', '0.05')),  # 5%
        }
    
    def initialize_exchange(self):
        """Initialize Bitget exchange connection"""
        try:
            self.logger.info("🔗 Connecting to Bitget exchange...")
            
            api_key = os.getenv('BITGET_API_KEY')
            api_secret = os.getenv('BITGET_API_SECRET') 
            api_password = os.getenv('BITGET_API_PASSWORD')
            
            if not all([api_key, api_secret, api_password]):
                raise ValueError("Missing Bitget API credentials in environment")
                
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': api_secret,
                'password': api_password,
                'sandbox': False,  # Set to True for testing
                'options': {
                    'defaultType': 'swap',  # Use perpetual futures
                    'adjustForTimeDifference': True,
                }
            })
            
            # Test connection
            self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            self.logger.info(f"# Check Exchange connected. Account balance: {balance.get('USDT', {}).get('free', 0)} USDT")
            
        except Exception as e:
            self.logger.error(f"# X Failed to initialize exchange: {e}")
            raise
    
    def fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch comprehensive market data with advanced metrics for optimization
        Includes volatility, liquidity metrics, and microstructure data
        """
        try:
            # Get ticker data
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Get recent OHLCV data for volatility and trend analysis
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=48)  # 48 hours for better analysis
            
            if not ticker or not ohlcv or len(ohlcv) < 24:
                return None
                
            # Calculate volatility (24-hour rolling)
            recent_prices = [candle[4] for candle in ohlcv[-24:]]  # Closing prices
            if len(recent_prices) > 1:
                price_changes = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                               for i in range(1, len(recent_prices))]
                volatility = (sum(x**2 for x in price_changes) / len(price_changes)) ** 0.5
            else:
                volatility = 0.02  # Default 2% volatility
            
            # Calculate average volume (for liquidity assessment)
            recent_volumes = [candle[5] for candle in ohlcv[-24:]]  # Volume data
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
            
            # Calculate volume trend (current vs average)
            current_volume = ticker['baseVolume']
            volume_ratio = current_volume / max(avg_volume, 1) if avg_volume > 0 else 1.0
            
            # Price momentum over different timeframes
            if len(ohlcv) >= 6:
                price_6h = ohlcv[-6][4]  # 6 hours ago
                short_momentum = (ticker['last'] - price_6h) / price_6h * 100
            else:
                short_momentum = ticker['percentage'] or 0
                
            # Enhanced spread calculation
            bid_ask_spread = 0
            if ticker['bid'] and ticker['ask'] and ticker['last']:
                bid_ask_spread = (ticker['ask'] - ticker['bid']) / ticker['last']
            
            # Time-based liquidity adjustment (simplified - in practice would use market hours)
            current_hour = datetime.now().hour
            if 8 <= current_hour <= 20:  # Assume business hours have better liquidity
                liquidity_adjustment = 1.0
            else:
                liquidity_adjustment = 1.2  # 20% penalty for off-hours
            
            # Order book depth estimation (using volume as proxy)
            depth_score = min(100, current_volume / 1_000_000 * 50)  # Rough depth score 0-100
            
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'price_change': ticker['percentage'],
                'short_momentum': short_momentum,
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'spread': bid_ask_spread,
                'volatility': volatility,
                'liquidity_adjustment': liquidity_adjustment,
                'depth_score': depth_score,
                'timestamp': datetime.now().isoformat(),
                
                # Additional metrics for advanced optimization
                'price_stability': max(0, 100 - volatility * 5000),  # 0-100 stability score
                'market_pressure': min(max(-100, short_momentum * 10), 100),  # -100 to +100
                'liquidity_score': min(100, (current_volume * ticker['last']) / 1_000_000 * 20),  # Dollar liquidity score
            }
            
        except Exception as e:
            self.logger.error(f"# X Error fetching enhanced market data for {symbol}: {e}")
            return None
    
    def calculate_execution_cost(self, market_data: Dict, position_size_usd: float = 5000) -> float:
        """
        Enhanced execution cost calculation with advanced market microstructure modeling
        Includes spread cost, market impact, volatility adjustment, and liquidity premiums
        """
        try:
            spread = market_data.get('spread', 0)
            volume = market_data.get('volume', 0)
            volatility = market_data.get('volatility', 0.02)  # Default 2% daily vol
            price = market_data.get('price', 1.0)
            
            # Base spread cost (half spread for market order)
            spread_cost = position_size_usd * spread / 2
            
            # Enhanced market impact with volatility adjustment
            # Uses advanced square-root law with volatility scaling
            volume_ratio = position_size_usd / max(volume * price, 100_000)  # Position as % of dollar volume
            base_impact_rate = 0.0001 * (volume_ratio ** 0.5)
            
            # Volatility adjustment - higher vol = higher impact
            volatility_multiplier = max(1.0, volatility / 0.02)  # Scale from 2% base volatility
            adjusted_impact_rate = base_impact_rate * volatility_multiplier
            
            # Market impact cost
            market_impact_cost = position_size_usd * adjusted_impact_rate
            
            # Liquidity premium for large positions relative to volume
            if volume_ratio > 0.05:  # Position > 5% of volume
                liquidity_premium = position_size_usd * 0.0002 * (volume_ratio - 0.05)
            else:
                liquidity_premium = 0.0
            
            # Time-of-day adjustment (assuming this could be passed in market_data)
            time_adjustment = market_data.get('liquidity_adjustment', 1.0)
            
            total_execution_cost = (spread_cost + market_impact_cost + liquidity_premium) * time_adjustment
            return total_execution_cost
            
        except Exception as e:
            self.logger.error(f"# X Error calculating execution cost: {e}")
            return 999.0  # High cost to avoid trading

    def optimize_position_size(self, market_data: Dict, base_position_size: float) -> Tuple[float, str]:
        """
        Dynamic position sizing optimization based on market conditions
        Returns: (optimized_position_size, reasoning)
        """
        try:
            spread = market_data.get('spread', 0)
            volume = market_data.get('volume', 0)
            volatility = market_data.get('volatility', 0.02)
            price = market_data.get('price', 1.0)
            
            # Start with base position size
            optimized_size = base_position_size
            reasoning_parts = []
            
            # Volatility adjustment - reduce size in high volatility
            if volatility > 0.04:  # > 4% daily volatility
                vol_reduction = min(0.5, (volatility - 0.04) / 0.04)  # Up to 50% reduction
                optimized_size *= (1 - vol_reduction)
                reasoning_parts.append(f"High volatility ({volatility*100:.1f}%): -{vol_reduction*100:.0f}% size")
            elif volatility < 0.015:  # < 1.5% daily volatility  
                vol_increase = min(0.3, (0.015 - volatility) / 0.015)  # Up to 30% increase
                optimized_size *= (1 + vol_increase)
                reasoning_parts.append(f"Low volatility ({volatility*100:.1f}%): +{vol_increase*100:.0f}% size")
            
            # Liquidity adjustment - reduce size for low liquidity
            dollar_volume = volume * price
            if dollar_volume < 500_000:  # Less than $500k volume
                liquidity_reduction = 0.4  # 40% reduction for low liquidity
                optimized_size *= (1 - liquidity_reduction)
                reasoning_parts.append(f"Low liquidity (${dollar_volume:,.0f}): -{liquidity_reduction*100:.0f}% size")
            elif dollar_volume > 5_000_000:  # More than $5M volume
                liquidity_increase = 0.2  # 20% increase for high liquidity
                optimized_size *= (1 + liquidity_increase)
                reasoning_parts.append(f"High liquidity (${dollar_volume:,.0f}): +{liquidity_increase*100:.0f}% size")
            
            # Spread adjustment - reduce size for wide spreads
            spread_bps = spread * 10000  # Convert to basis points
            if spread_bps > 20:  # > 20 basis points
                spread_reduction = min(0.6, (spread_bps - 20) / 50)  # Up to 60% reduction
                optimized_size *= (1 - spread_reduction)
                reasoning_parts.append(f"Wide spread ({spread_bps:.1f}bps): -{spread_reduction*100:.0f}% size")
            
            # Execution cost efficiency - optimize size to minimize cost per dollar traded
            test_sizes = [optimized_size * mult for mult in [0.5, 0.75, 1.0, 1.25, 1.5]]
            best_efficiency = 0
            best_size = optimized_size
            
            for test_size in test_sizes:
                if test_size > 0:
                    exec_cost = self.calculate_execution_cost(market_data, test_size)
                    efficiency = test_size / max(exec_cost, 0.01)  # Dollars traded per dollar of cost
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_size = test_size
            
            if best_size != optimized_size:
                change_pct = (best_size - optimized_size) / optimized_size * 100
                reasoning_parts.append(f"Cost efficiency: {change_pct:+.0f}% size")
                optimized_size = best_size
            
            # Ensure position size stays within reasonable bounds
            max_size = base_position_size * 2.0  # Never more than 2x base
            min_size = base_position_size * 0.1  # Never less than 10% of base
            optimized_size = max(min_size, min(optimized_size, max_size))
            
            reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No adjustments needed"
            return optimized_size, reasoning
            
        except Exception as e:
            self.logger.error(f"# X Error optimizing position size: {e}")
            return base_position_size, "Error in optimization - using base size"

    def optimize_entry_timing(self, symbol: str, signal, market_data: Dict) -> Dict:
        """
        Advanced entry timing optimization with smart order placement
        Returns: {order_type, price, size, timing_score, reasoning}
        """
        try:
            current_price = market_data.get('price', 0)
            spread = market_data.get('spread', 0)
            volume = market_data.get('volume', 0)
            volatility = market_data.get('volatility', 0.02)
            
            # Base position size
            base_position_size = self.risk_per_trade * 100_000
            
            # Optimize position size first
            optimized_size, size_reasoning = self.optimize_position_size(market_data, base_position_size)
            
            # Calculate bid/ask from spread
            half_spread = spread / 2
            bid_price = current_price - half_spread
            ask_price = current_price + half_spread
            
            # Smart order placement strategy
            order_strategies = []
            
            # Strategy 1: Aggressive Market Order (immediate execution)
            market_cost = self.calculate_execution_cost(market_data, optimized_size)
            market_score = max(0, 100 - market_cost * 20)  # Penalize high costs
            order_strategies.append({
                'type': 'MARKET',
                'price': current_price,
                'size': optimized_size,
                'expected_cost': market_cost,
                'score': market_score,
                'reasoning': f"Immediate execution, cost ${market_cost:.2f}"
            })
            
            # Strategy 2: Patient Limit Order (better price, risk of missing)
            if signal.signal == "LONG":
                # Place limit slightly above bid (better than market, likely to fill)
                limit_price = bid_price + (half_spread * 0.3)  # 30% into spread
                price_improvement = current_price - limit_price
            else:  # SHORT
                # Place limit slightly below ask
                limit_price = ask_price - (half_spread * 0.3)
                price_improvement = limit_price - current_price
                
            # Estimate fill probability based on position in spread
            fill_probability = max(0.7, 0.9 - (abs(limit_price - current_price) / half_spread))
            
            # Adjusted execution cost for limit order (lower due to better price)
            limit_cost_reduction = price_improvement * optimized_size
            limit_cost = market_cost - limit_cost_reduction
            limit_score = (max(0, 100 - limit_cost * 20)) * fill_probability  # Adjust for fill risk
            
            order_strategies.append({
                'type': 'LIMIT',
                'price': limit_price,
                'size': optimized_size,
                'expected_cost': limit_cost,
                'score': limit_score,
                'reasoning': f"Better price (${price_improvement:.4f} improvement), {fill_probability*100:.0f}% fill chance"
            })
            
            # Strategy 3: Iceberg Order for large positions (reduce market impact)
            if optimized_size > volume * current_price * 0.02:  # Position > 2% of volume
                iceberg_chunks = min(5, max(2, int(optimized_size / (volume * current_price * 0.01))))
                chunk_size = optimized_size / iceberg_chunks
                
                # Estimate reduced market impact from smaller chunks
                iceberg_cost = sum([self.calculate_execution_cost(market_data, chunk_size) 
                                  for _ in range(iceberg_chunks)])
                iceberg_score = max(0, 100 - iceberg_cost * 20)
                
                order_strategies.append({
                    'type': 'ICEBERG',
                    'price': current_price,
                    'size': optimized_size,
                    'chunk_size': chunk_size,
                    'chunks': iceberg_chunks,
                    'expected_cost': iceberg_cost,
                    'score': iceberg_score,
                    'reasoning': f"Reduced impact via {iceberg_chunks} chunks of ${chunk_size:,.0f}"
                })
            
            # Select best strategy
            best_strategy = max(order_strategies, key=lambda x: x['score'])
            
            # Add timing score based on market conditions
            timing_factors = []
            timing_score = 50  # Base score
            
            # Volatility timing
            if volatility < 0.015:  # Low volatility - good for entry
                timing_score += 20
                timing_factors.append("Low volatility (+20)")
            elif volatility > 0.05:  # High volatility - wait for calmer conditions
                timing_score -= 15
                timing_factors.append("High volatility (-15)")
            
            # Volume timing  
            if volume > 0:  # Avoid division by zero
                timing_score += 15
                timing_factors.append("Good volume (+15)")
            else:
                timing_score -= 10
                timing_factors.append("Low volume (-10)")
            
            # Spread timing
            spread_bps = spread * 10000
            if spread_bps < 5:  # Tight spread
                timing_score += 10
                timing_factors.append("Tight spread (+10)")
            elif spread_bps > 25:  # Wide spread
                timing_score -= 20
                timing_factors.append("Wide spread (-20)")
            
            timing_reasoning = " | ".join(timing_factors) if timing_factors else "Neutral timing"
            
            return {
                'order_type': best_strategy['type'],
                'price': best_strategy['price'],
                'size': best_strategy['size'],
                'expected_cost': best_strategy['expected_cost'],
                'strategy_score': best_strategy['score'],
                'timing_score': max(0, min(100, timing_score)),
                'reasoning': f"{best_strategy['reasoning']} | {size_reasoning} | {timing_reasoning}",
                **{k: v for k, v in best_strategy.items() if k not in ['type', 'price', 'size', 'expected_cost', 'score', 'reasoning']}
            }
            
        except Exception as e:
            self.logger.error(f"# X Error optimizing entry timing: {e}")
            # Fallback to simple market order
            return {
                'order_type': 'MARKET',
                'price': market_data.get('price', 0),
                'size': self.risk_per_trade * 100_000,
                'expected_cost': 999.0,
                'strategy_score': 0,
                'timing_score': 0,
                'reasoning': 'Error in optimization - using fallback market order'
            }
            
    def calculate_viper_score(self, market_data: Dict) -> float:
        """
        Calculate VIPER score using Volume, Price, External, Range factors
        Enhanced with execution cost awareness to prevent $3+ losses on entry
        Returns score from 0-100 (higher = better opportunity)
        """
        try:
            volume = market_data.get('volume', 0)
            price_change = abs(market_data.get('price_change', 0))
            high = market_data.get('high', 1)
            low = market_data.get('low', 0) 
            current_price = market_data.get('price', 1)
            spread = market_data.get('spread', 0)
            
            # Volume Score (0-100): Higher volume = better liquidity
            volume_score = min((volume / 1_000_000) * 25, 100)  # Scale by 1M volume
            
            # Price Score (0-100): Momentum strength  
            price_score = min(price_change * 20, 100)  # Scale by price change %
            
            # Enhanced External Score (0-100): Execution cost awareness
            # Calculate expected execution cost for typical position size
            position_size = self.risk_per_trade * 100_000  # Assume $100k account for calculation
            execution_cost = self.calculate_execution_cost(market_data, position_size)
            
            # Penalize heavily if execution cost > $3 threshold
            if execution_cost >= 3.0:
                external_score = 0  # Zero score for high execution cost
            elif execution_cost >= 2.0:
                external_score = 30  # Low score for moderate execution cost  
            elif execution_cost >= 1.0:
                external_score = 60  # Medium score
            else:
                # Use improved spread-based scoring for low-cost scenarios
                external_score = max(100 - (spread * 5000), 50)  # More sensitive to spread
            
            # Range Score (0-100): Volatility within reasonable bounds
            if current_price > 0:
                volatility = ((high - low) / current_price) * 100
                range_score = min(volatility * 10, 100) if volatility > 0.5 else volatility * 5
            else:
                range_score = 0
            
            # Weighted VIPER score with increased emphasis on execution cost
            viper_score = (
                volume_score * 0.25 +     # 25% volume weight (reduced)
                price_score * 0.30 +      # 30% momentum weight (reduced) 
                external_score * 0.30 +   # 30% execution cost weight (increased)
                range_score * 0.15        # 15% volatility weight (same)
            )
            
            return min(max(viper_score, 0), 100)  # Clamp to 0-100 range
            
        except Exception as e:
            self.logger.error(f"# X Error calculating VIPER score: {e}")
            return 0.0
    
    def generate_signal(self, symbol: str, market_data: Dict) -> Optional[VIPERSignal]:
        """
        Generate advanced trading signal with comprehensive entry optimization
        Uses dynamic position sizing, timing optimization, and smart order routing
        """
        try:
            # Calculate base VIPER score
            viper_score = self.calculate_viper_score(market_data)
            
            if viper_score < self.viper_threshold:
                return None  # Score too low for trading
                
            # Initial execution cost check
            base_position_size = self.risk_per_trade * 100_000  # Assume $100k account 
            initial_execution_cost = self.calculate_execution_cost(market_data, base_position_size)
            
            if initial_execution_cost >= 3.0:
                self.logger.warning(f"🚫 Signal rejected for {symbol}: execution cost ${initial_execution_cost:.2f} >= $3.00 threshold")
                return None
                
            price_change = market_data.get('price_change', 0)
            current_price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            
            # Determine signal direction with enhanced momentum analysis
            signal = None
            confidence_boost = 0
            
            # Multi-factor momentum analysis for better entry signals
            if price_change > 1.5:  # Strong upward momentum (>1.5%)
                signal = "LONG"
                confidence_boost = min(20, (price_change - 1.5) * 10)  # Bonus for strong momentum
            elif price_change < -1.5:  # Strong downward momentum (<-1.5%)
                signal = "SHORT" 
                confidence_boost = min(20, abs(price_change + 1.5) * 10)
            elif abs(price_change) > 0.8:  # Medium momentum
                # Check volume confirmation for medium momentum
                volume_ratio = volume / max(1_000_000, volume)  # Simplified volume check
                if volume_ratio > 1.2:  # High volume confirmation
                    signal = "LONG" if price_change > 0 else "SHORT"
                    confidence_boost = 10  # Bonus for volume confirmation
                else:
                    return None  # Insufficient volume confirmation
            else:
                return None  # Insufficient momentum
            
            # Create preliminary signal for optimization
            preliminary_signal = type('PreliminarySignal', (), {
                'signal': signal,
                'viper_score': viper_score,
                'price': current_price
            })()
            
            # Apply advanced entry timing optimization
            entry_optimization = self.optimize_entry_timing(symbol, preliminary_signal, market_data)
            
            # Update execution cost based on optimized strategy
            optimized_execution_cost = entry_optimization['expected_cost']
            optimized_size = entry_optimization['size']
            optimized_order_type = entry_optimization['order_type']
            optimized_price = entry_optimization['price']
            
            # Final execution cost check after optimization
            if optimized_execution_cost >= 2.5:  # Slightly lower threshold after optimization
                self.logger.warning(f"🚫 Signal rejected for {symbol} after optimization: cost ${optimized_execution_cost:.2f} still too high")
                return None
            
            # Enhanced confidence calculation
            base_confidence = min(viper_score / 100, 1.0)
            timing_confidence = entry_optimization['timing_score'] / 100
            strategy_confidence = entry_optimization['strategy_score'] / 100
            
            # Weighted confidence score
            enhanced_confidence = (
                base_confidence * 0.4 +      # 40% VIPER score
                timing_confidence * 0.3 +    # 30% timing
                strategy_confidence * 0.3    # 30% strategy
            ) + (confidence_boost / 100)     # Add momentum bonus
            
            enhanced_confidence = min(1.0, enhanced_confidence)  # Cap at 100%
            
            # Calculate stop loss and take profit with advanced risk management
            stop_loss_pct = self.config['stop_loss_percent']
            take_profit_pct = self.config['take_profit_percent']
            
            # More sophisticated risk adjustment based on optimized execution cost
            execution_cost_pct = optimized_execution_cost / optimized_size if optimized_size > 0 else 0.01
            volatility = market_data.get('volatility', 0.02)
            
            # Dynamic stop loss - accounts for execution costs and volatility
            volatility_adjustment = max(1.0, volatility / 0.02)  # Scale from 2% base volatility
            adjusted_stop_loss_pct = max(
                stop_loss_pct * volatility_adjustment,           # Volatility adjustment
                execution_cost_pct + 0.008                      # Execution cost + 0.8% buffer
            )
            
            # Dynamic take profit - ensures good risk/reward after costs
            min_rr_ratio = 2.5  # Minimum 2.5:1 risk/reward after costs
            adjusted_take_profit_pct = max(
                take_profit_pct,
                adjusted_stop_loss_pct * min_rr_ratio,          # Maintain risk/reward
                execution_cost_pct * 4 + 0.015                  # 4x execution cost + 1.5%
            )
            
            # Calculate final price levels
            if signal == "LONG":
                stop_loss = optimized_price * (1 - adjusted_stop_loss_pct)
                take_profit = optimized_price * (1 + adjusted_take_profit_pct)
            else:  # SHORT
                stop_loss = optimized_price * (1 + adjusted_stop_loss_pct)
                take_profit = optimized_price * (1 - adjusted_take_profit_pct)
            
            # Enhanced VIPER signal with optimization data
            optimized_signal = VIPERSignal(
                symbol=symbol,
                signal=signal,
                viper_score=viper_score,
                price=optimized_price,  # Use optimized entry price
                price_change=price_change,
                volume=volume,
                confidence=enhanced_confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now().isoformat(),
                execution_cost=optimized_execution_cost,
                order_type=optimized_order_type
            )
            
            # Add optimization metadata
            optimized_signal.optimization_data = {
                'original_size': base_position_size,
                'optimized_size': optimized_size,
                'size_change_pct': ((optimized_size - base_position_size) / base_position_size * 100),
                'original_cost': initial_execution_cost,
                'cost_savings': initial_execution_cost - optimized_execution_cost,
                'strategy_score': entry_optimization['strategy_score'],
                'timing_score': entry_optimization['timing_score'],
                'reasoning': entry_optimization['reasoning']
            }
            
            # Log optimization results
            self.logger.info(f"# Chart {symbol} Signal Optimized: "
                           f"Size ${base_position_size:,.0f} → ${optimized_size:,.0f} "
                           f"({optimized_signal.optimization_data['size_change_pct']:+.1f}%), "
                           f"Cost ${initial_execution_cost:.2f} → ${optimized_execution_cost:.2f} "
                           f"(${optimized_signal.optimization_data['cost_savings']:+.2f} savings), "
                           f"Confidence {base_confidence:.1%} → {enhanced_confidence:.1%}")
            
            return optimized_signal
            
        except Exception as e:
            self.logger.error(f"# X Error generating optimized signal for {symbol}: {e}")
            return None
    
    def scan_markets(self) -> List[VIPERSignal]:
        """Scan all trading pairs for opportunities"""
        self.logger.info(f"# Search Scanning {len(self.trading_pairs)} trading pairs...")
        
        signals = []
        for symbol in self.trading_pairs:
            try:
                market_data = self.fetch_market_data(symbol)
                if market_data:
                    signal = self.generate_signal(symbol, market_data)
                    if signal:
                        signals.append(signal)
                        self.logger.info(f"# Chart {symbol}: VIPER Score {signal.viper_score:.1f} → {signal.signal} Signal")
                        
            except Exception as e:
                self.logger.error(f"# X Error scanning {symbol}: {e}")
                
        self.logger.info(f"# Target Found {len(signals)} trading opportunities")
        return signals
    
    def calculate_position_size(self, signal: VIPERSignal) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            balance = self.exchange.fetch_balance()
            available_usdt = balance.get('USDT', {}).get('free', 0)
            
            if available_usdt <= 0:
                self.logger.warning("# Warning Insufficient USDT balance")
                return 0
            
            # Risk amount per trade
            risk_amount = available_usdt * self.risk_per_trade
            
            # Calculate position size based on stop loss distance
            price_diff = abs(signal.price - signal.stop_loss)
            if price_diff <= 0:
                return 0
                
            position_value = risk_amount / (price_diff / signal.price)
            position_size = position_value / signal.price
            
            # Apply minimum size constraints
            market = self.exchange.market(signal.symbol)
            min_size = market['limits']['amount']['min'] or 0.001
            position_size = max(position_size, min_size)
            
            return round(position_size, 6)
            
        except Exception as e:
            self.logger.error(f"# X Error calculating position size: {e}")
            return 0
    
    def execute_trade(self, signal: VIPERSignal) -> bool:
        """Execute trade based on VIPER signal"""
        try:
            # Check position limits
            if len(self.active_positions) >= self.max_positions:
                self.logger.warning(f"# Warning Maximum positions ({self.max_positions}) reached")
                return False
            
            # Check if already have position in this symbol
            if signal.symbol in self.active_positions:
                self.logger.info(f"# Warning Already have position in {signal.symbol}")
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(signal)
            if position_size <= 0:
                self.logger.warning(f"# Warning Invalid position size for {signal.symbol}")
                return False
            
            # Determine order side
            side = 'buy' if signal.signal == 'LONG' else 'sell'
            
            self.logger.info(f"# Target Executing {signal.signal} trade for {signal.symbol}")
            self.logger.info(f"   Size: {position_size}, Price: ${signal.price:.2f}")
            self.logger.info(f"   Stop Loss: ${signal.stop_loss:.2f}, Take Profit: ${signal.take_profit:.2f}")
            
            # Execute market order
            order = self.exchange.create_order(
                symbol=signal.symbol,
                type='market',
                side=side,
                amount=position_size,
                price=None  # Market order
            )
            
            if order and order.get('id'):
                # Create position tracking
                position = TradingPosition(
                    symbol=signal.symbol,
                    side=side,
                    size=position_size,
                    entry_price=signal.price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    timestamp=datetime.now().isoformat(),
                    order_id=order['id']
                )
                
                self.active_positions[signal.symbol] = position
                
                self.logger.info(f"# Check Trade executed successfully for {signal.symbol}")
                self.logger.info(f"   Order ID: {order['id']}")
                
                return True
            else:
                self.logger.error(f"# X Trade execution failed for {signal.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"# X Error executing trade for {signal.symbol}: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor active positions for TP/SL conditions"""
        if not self.active_positions:
            return
        
        self.logger.info(f"# Chart Monitoring {len(self.active_positions)} active positions...")
        
        positions_to_close = []
        
        for symbol, position in self.active_positions.items():
            try:
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Calculate P&L
                if position.side == 'buy':
                    pnl = (current_price - position.entry_price) / position.entry_price
                else:  # sell
                    pnl = (position.entry_price - current_price) / position.entry_price
                
                position.unrealized_pnl = pnl
                
                self.logger.info(f"📈 {symbol} ({position.side.upper()}): "
                               f"${current_price:.2f} | P&L: {pnl*100:.2f}%")
                
                # Check stop loss condition
                stop_loss_triggered = False
                if position.side == 'buy' and current_price <= position.stop_loss:
                    stop_loss_triggered = True
                elif position.side == 'sell' and current_price >= position.stop_loss:
                    stop_loss_triggered = True
                
                # Check take profit condition  
                take_profit_triggered = False
                if position.side == 'buy' and current_price >= position.take_profit:
                    take_profit_triggered = True
                elif position.side == 'sell' and current_price <= position.take_profit:
                    take_profit_triggered = True
                
                # Close position if TP or SL triggered
                if stop_loss_triggered:
                    self.logger.info(f"🛑 Stop Loss triggered for {symbol}")
                    positions_to_close.append((symbol, 'stop_loss'))
                elif take_profit_triggered:
                    self.logger.info(f"# Target Take Profit triggered for {symbol}")
                    positions_to_close.append((symbol, 'take_profit'))
                    
            except Exception as e:
                self.logger.error(f"# X Error monitoring position {symbol}: {e}")
        
        # Close positions that hit TP/SL
        for symbol, reason in positions_to_close:
            self.close_position(symbol, reason)
    
    def close_position(self, symbol: str, reason: str = 'manual'):
        """Close an active position"""
        try:
            if symbol not in self.active_positions:
                self.logger.warning(f"# Warning No active position found for {symbol}")
                return False
            
            position = self.active_positions[symbol]
            
            # Determine closing side (opposite of opening)
            close_side = 'sell' if position.side == 'buy' else 'buy'
            
            self.logger.info(f"🔄 Closing {symbol} position ({reason})")
            
            # Execute closing order
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=position.size,
                price=None
            )
            
            if order and order.get('id'):
                # Calculate final P&L
                current_price = self.exchange.fetch_ticker(symbol)['last']
                if position.side == 'buy':
                    final_pnl = (current_price - position.entry_price) / position.entry_price
                else:
                    final_pnl = (position.entry_price - current_price) / position.entry_price
                
                self.logger.info(f"# Check Position closed for {symbol}")
                self.logger.info(f"   Final P&L: {final_pnl*100:.2f}%")
                self.logger.info(f"   Reason: {reason}")
                
                # Remove from active positions
                del self.active_positions[symbol]
                
                return True
            else:
                self.logger.error(f"# X Failed to close position for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"# X Error closing position {symbol}: {e}")
            return False
    
    def print_status(self):
        """Print current trading status"""
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# Chart Active Positions: {len(self.active_positions)}/{self.max_positions}")
        print(f"# Target VIPER Threshold: {self.viper_threshold}")
        print(f"💰 Risk per Trade: {self.risk_per_trade*100:.1f}%")
        
        if self.active_positions:
            for symbol, position in self.active_positions.items():
                pnl_pct = position.unrealized_pnl * 100 if position.unrealized_pnl else 0
                pnl_icon = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                print(f"  {pnl_icon} {symbol} | {position.side.upper()} | "
                      f"P&L: {pnl_pct:.2f}% | Entry: ${position.entry_price:.2f}")
        else:
        
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run(self):
        """Main trading loop"""
        self.logger.info("# Rocket Starting VIPER Standalone Trading System...")
        self.logger.info(f"# Chart Monitoring {len(self.trading_pairs)} trading pairs")
        self.logger.info(f"⏰ Scan interval: {self.scan_interval}s")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Print current status
                self.print_status()
                
                # 1. SCAN: Look for trading opportunities
                signals = self.scan_markets()
                
                # 2. SCORE & TRADE: Execute trades for high-scoring signals
                for signal in signals:
                    if len(self.active_positions) >= self.max_positions:
                        self.logger.info(f"# Warning Position limit reached, skipping new trades")
                        break
                    
                    self.logger.info(f"# Target Processing {signal.signal} signal for {signal.symbol}")
                    self.logger.info(f"   VIPER Score: {signal.viper_score:.1f}")
                    self.logger.info(f"   Confidence: {signal.confidence:.1%}")
                    
                    # Execute trade
                    success = self.execute_trade(signal)
                    if success:
                        self.logger.info(f"# Check Trade executed successfully")
                    else:
                        self.logger.warning(f"# Warning Trade execution failed")
                
                # 3. TP/SL: Monitor existing positions
                self.monitor_positions()
                
                # Sleep until next scan cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, self.scan_interval - elapsed)
                
                if sleep_time > 0:
                    self.logger.info(f"💤 Sleeping for {sleep_time:.1f}s until next scan...")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"# X Unexpected error in main loop: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown procedure"""
        self.logger.info("🛑 Shutting down VIPER Trader...")
        
        # Close all positions if desired (optional safety measure)
        # Uncomment the following lines to close all positions on shutdown
        # for symbol in list(self.active_positions.keys()):
        #     self.close_position(symbol, 'shutdown')
        
        self.logger.info("# Check Shutdown complete")


def main():
    """Main entry point"""
#==============================================================================#
#                # Rocket VIPER STANDALONE TRADING COMPONENT                         #
#                Complete Scan → Score → Trade → TP/SL Flow                    #
#==============================================================================#
    """)
    
    try:
        # Check for required environment variables
        required_vars = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_API_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"# X Missing required environment variables: {', '.join(missing_vars)}")
            print("Please configure your .env file with Bitget API credentials")
            return 1
        
        # Create and run the trader
        trader = StandaloneVIPERTrader()
        trader.run()
        
        return 0
        
    except Exception as e:
        return 1


if __name__ == "__main__":
    sys.exit(main())