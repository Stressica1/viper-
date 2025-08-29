#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Centralized VIPER Scoring Service
Standardized signal generation and scoring algorithm for all trading operations

Features:
- Centralized VIPER (Volume, Price, External, Range) scoring
- Real-time signal generation
- Configurable scoring parameters
- Redis pub/sub integration
- Multiple timeframe support
- Risk-adjusted scoring
"""

import os
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import uvicorn
import redis
import numpy as np
import pandas as pd
from enum import Enum

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'viper-scoring-service')

# VIPER Scoring Configuration
VIPER_THRESHOLD_HIGH = float(os.getenv('VIPER_THRESHOLD_HIGH', '85'))
VIPER_THRESHOLD_MEDIUM = float(os.getenv('VIPER_THRESHOLD_MEDIUM', '70'))
SIGNAL_COOLDOWN = int(os.getenv('SIGNAL_COOLDOWN', '300'))  # 5 minutes
MAX_SIGNALS_PER_SYMBOL = int(os.getenv('MAX_SIGNALS_PER_SYMBOL', '3'))

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"

class SignalStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

class VIPERScoringService:
    """Centralized VIPER scoring and signal generation service"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False
        self.active_signals = {}  # Track active signals per symbol
        self.signal_history = {}  # Track signal history
        self.last_signal_time = {}  # Cooldown tracking
        self.scoring_weights = {
            'volume_score': 0.30,      # Volume importance
            'price_score': 0.30,       # Price momentum
            'external_score': 0.20,    # Market microstructure
            'range_score': 0.20        # Volatility/range
        }

        # Scoring parameters
        self.volume_period = int(os.getenv('VOLUME_PERIOD', '24'))  # hours
        self.price_period = int(os.getenv('PRICE_PERIOD', '4'))     # hours
        self.range_period = int(os.getenv('RANGE_PERIOD', '24'))    # hours

        logger.info("🎯 VIPER Scoring Service initialized")

    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False

    def calculate_volume_score(self, market_data: Dict, symbol: str) -> float:
        """Calculate volume-based score component"""
        try:
            ticker = market_data.get('ticker', {})
            volume = ticker.get('volume', 0)
            quote_volume = ticker.get('quote_volume', 0)

            if volume <= 0:
                return 0.0

            # Get volume history from OHLCV data
            ohlcv_data = market_data.get('ohlcv', {}).get('ohlcv', [])
            if len(ohlcv_data) < 10:
                # Fallback to current volume analysis
                volume_score = min(volume / 10000, 100)  # Normalize to 0-100
            else:
                # Calculate volume relative to recent average
                recent_volumes = [candle[5] for candle in ohlcv_data[-20:]]  # Last 20 candles
                avg_volume = np.mean(recent_volumes)
                current_volume = volume

                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    volume_score = min(volume_ratio * 50, 100)  # Scale and cap at 100
                else:
                    volume_score = 50  # Neutral score

            return max(0, min(100, volume_score))

        except Exception as e:
            logger.error(f"❌ Error calculating volume score for {symbol}: {e}")
            return 0.0

    def calculate_price_score(self, market_data: Dict, symbol: str) -> float:
        """Calculate price momentum score component"""
        try:
            ticker = market_data.get('ticker', {})
            price_change = ticker.get('price_change', 0)
            current_price = ticker.get('price', 0)

            if current_price <= 0:
                return 50.0  # Neutral score

            # Get OHLCV data for trend analysis
            ohlcv_data = market_data.get('ohlcv', {}).get('ohlcv', [])
            price_score = 50.0  # Base neutral score

            if len(ohlcv_data) >= 10:
                # Calculate short-term trend (last 5 candles)
                recent_prices = [candle[4] for candle in ohlcv_data[-5:]]  # Close prices
                if len(recent_prices) >= 2:
                    short_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100

                    # Calculate longer-term trend (last 20 candles)
                    long_prices = [candle[4] for candle in ohlcv_data[-20:]]
                    if len(long_prices) >= 2:
                        long_trend = (long_prices[-1] - long_prices[0]) / long_prices[0] * 100

                        # Combined momentum score
                        momentum_score = (short_trend * 0.6) + (long_trend * 0.4)
                        price_score = 50 + (momentum_score * 0.5)  # Center around 50
                    else:
                        price_score = 50 + (short_trend * 0.3)
            else:
                # Fallback to simple percentage change
                price_score = 50 + (price_change * 0.5)

            return max(0, min(100, price_score))

        except Exception as e:
            logger.error(f"❌ Error calculating price score for {symbol}: {e}")
            return 50.0

    def calculate_external_score(self, market_data: Dict, symbol: str) -> float:
        """Calculate external factors score (market microstructure)"""
        try:
            ticker = market_data.get('ticker', {})
            orderbook = market_data.get('orderbook', {})

            external_score = 50.0  # Base neutral score

            # Spread analysis
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if bids and asks and len(bids) > 0 and len(asks) > 0:
                best_bid = bids[0][0] if isinstance(bids[0], list) else bids[0]
                best_ask = asks[0][0] if isinstance(asks[0], list) else asks[0]
                current_price = ticker.get('price', (best_bid + best_ask) / 2)

                if current_price > 0:
                    spread = (best_ask - best_bid) / current_price * 100  # Spread as percentage
                    spread_score = max(0, 100 - spread * 1000)  # Lower spread = higher score
                    external_score = spread_score * 0.6

            # Order book depth analysis
            total_bid_volume = sum([bid[1] if isinstance(bid, list) else 0 for bid in bids[:5]])
            total_ask_volume = sum([ask[1] if isinstance(ask, list) else 0 for ask in asks[:5]])

            if total_bid_volume + total_ask_volume > 0:
                depth_balance = abs(total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
                depth_score = 100 - (depth_balance * 100)  # Balanced order book = higher score
                external_score += depth_score * 0.4

            return max(0, min(100, external_score))

        except Exception as e:
            logger.error(f"❌ Error calculating external score for {symbol}: {e}")
            return 50.0

    def calculate_range_score(self, market_data: Dict, symbol: str) -> float:
        """Calculate range/volatility score component"""
        try:
            ticker = market_data.get('ticker', {})
            high = ticker.get('high', 0)
            low = ticker.get('low', 0)
            current_price = ticker.get('price', 0)

            if current_price <= 0 or high <= 0 or low <= 0:
                return 50.0

            # Current day range
            daily_range = (high - low) / current_price * 100

            # Get OHLCV data for historical volatility
            ohlcv_data = market_data.get('ohlcv', {}).get('ohlcv', [])
            range_score = 50.0

            if len(ohlcv_data) >= 10:
                # Calculate average true range over last 10 periods
                ranges = []
                for i in range(1, min(11, len(ohlcv_data))):
                    candle_range = ohlcv_data[-(i+1)][2] - ohlcv_data[-(i+1)][3]  # High - Low
                    if candle_range > 0:
                        ranges.append(candle_range)

                if ranges:
                    avg_range = np.mean(ranges)
                    current_range_ratio = daily_range / (avg_range / current_price * 100) if avg_range > 0 else 1
                    range_score = 50 + (current_range_ratio - 1) * 25  # Center around 50

            # Incorporate current daily range
            range_score = (range_score * 0.7) + (min(daily_range * 2, 100) * 0.3)

            return max(0, min(100, range_score))

        except Exception as e:
            logger.error(f"❌ Error calculating range score for {symbol}: {e}")
            return 50.0

    def calculate_viper_score(self, market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Calculate complete VIPER score with all components"""
        try:
            # Calculate individual component scores
            volume_score = self.calculate_volume_score(market_data, symbol)
            price_score = self.calculate_price_score(market_data, symbol)
            external_score = self.calculate_external_score(market_data, symbol)
            range_score = self.calculate_range_score(market_data, symbol)

            # Calculate weighted overall score
            overall_score = (
                volume_score * self.scoring_weights['volume_score'] +
                price_score * self.scoring_weights['price_score'] +
                external_score * self.scoring_weights['external_score'] +
                range_score * self.scoring_weights['range_score']
            )

            # Determine signal strength
            if overall_score >= 90:
                strength = SignalStrength.VERY_STRONG
            elif overall_score >= 80:
                strength = SignalStrength.STRONG
            elif overall_score >= 70:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK

            return {
                'overall_score': round(overall_score, 2),
                'strength': strength.value,
                'components': {
                    'volume_score': round(volume_score, 2),
                    'price_score': round(price_score, 2),
                    'external_score': round(external_score, 2),
                    'range_score': round(range_score, 2)
                },
                'weights': self.scoring_weights,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol
            }

        except Exception as e:
            logger.error(f"❌ Error calculating VIPER score for {symbol}: {e}")
            return {
                'overall_score': 0.0,
                'strength': SignalStrength.WEAK.value,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol
            }

    def generate_signal(self, market_data: Dict, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on VIPER score"""
        try:
            # Check cooldown period
            current_time = datetime.now()
            last_signal = self.last_signal_time.get(symbol)

            if last_signal and (current_time - last_signal).seconds < SIGNAL_COOLDOWN:
                return None

            # Check maximum signals per symbol
            symbol_signals = [s for s in self.active_signals.values()
                            if isinstance(s, dict) and s.get('symbol') == symbol]

            if len(symbol_signals) >= MAX_SIGNALS_PER_SYMBOL:
                return None

            # Calculate VIPER score
            viper_result = self.calculate_viper_score(market_data, symbol)
            overall_score = viper_result['overall_score']

            if overall_score < VIPER_THRESHOLD_MEDIUM:
                return None

            # Determine signal direction
            ticker = market_data.get('ticker', {})
            price_change = ticker.get('price_change', 0)
            current_price = ticker.get('price', 0)

            # Generate signal based on score and price action
            signal_type = None
            confidence = overall_score / 100.0

            if overall_score >= VIPER_THRESHOLD_HIGH:
                # High confidence signal
                if price_change > 0.3:
                    signal_type = SignalType.LONG
                elif price_change < -0.3:
                    signal_type = SignalType.SHORT
                elif overall_score >= 95:
                    # Very high score overrides price change threshold
                    if viper_result['components']['price_score'] > 60:
                        signal_type = SignalType.LONG
                    elif viper_result['components']['price_score'] < 40:
                        signal_type = SignalType.SHORT
            elif overall_score >= VIPER_THRESHOLD_MEDIUM:
                # Medium confidence signal
                if price_change > 0.5:
                    signal_type = SignalType.LONG
                elif price_change < -0.5:
                    signal_type = SignalType.SHORT

            if not signal_type:
                return None

            # Create signal
            signal = {
                'id': f"{symbol}_{int(current_time.timestamp())}",
                'symbol': symbol,
                'type': signal_type.value,
                'viper_score': viper_result,
                'confidence': round(confidence, 3),
                'price': current_price,
                'price_change': price_change,
                'market_data': market_data,
                'timestamp': current_time.isoformat(),
                'expires_at': (current_time + timedelta(minutes=30)).isoformat(),  # 30-minute expiry
                'risk_per_trade': 0.02,  # 2% risk per trade
                'stop_loss': current_price * (0.98 if signal_type == SignalType.LONG else 1.02),
                'take_profit': current_price * (1.03 if signal_type == SignalType.LONG else 0.97),
                'strategy': 'VIPER_CENTRALIZED'
            }

            # Update tracking
            self.last_signal_time[symbol] = current_time
            self.active_signals[signal['id']] = signal

            # Store signal history
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            self.signal_history[symbol].append(signal)

            # Keep only last 100 signals per symbol
            if len(self.signal_history[symbol]) > 100:
                self.signal_history[symbol] = self.signal_history[symbol][-100:]

            logger.info(f"🎯 Generated {signal_type.value} signal for {symbol} (Score: {overall_score:.1f})")

            return signal

        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return None

    def process_market_data(self, market_data: Dict):
        """Process incoming market data and generate signals"""
        try:
            symbol = market_data.get('symbol')
            if not symbol:
                return

            # Generate signal
            signal = self.generate_signal(market_data, symbol)

            if signal:
                # Publish signal to Redis
                self.redis_client.publish('trading_signals', json.dumps(signal))
                self.redis_client.publish(f'signals:{symbol}', json.dumps(signal))

                # Cache signal
                signal_key = f"signal:{signal['id']}"
                self.redis_client.setex(signal_key, 3600, json.dumps(signal))  # 1 hour cache

                logger.info(f"📡 Published {signal['type']} signal for {symbol}")

        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}")

    def subscribe_to_market_data(self):
        """Subscribe to market data streams"""
        try:
            pubsub = self.redis_client.pubsub()

            # Subscribe to market data channels
            channels = ['market_data:all']

            pubsub.subscribe(*channels)
            logger.info(f"📡 Subscribed to market data channels: {channels}")

            # Process messages
            for message in pubsub.listen():
                if not self.is_running:
                    break

                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        self.process_market_data(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to decode message: {e}")

        except Exception as e:
            logger.error(f"❌ Error in market data subscription: {e}")

    def start_background_processing(self):
        """Start signal processing in background thread"""
        def run_processor():
            self.subscribe_to_market_data()

        thread = threading.Thread(target=run_processor, daemon=True)
        thread.start()
        logger.info("🎯 Signal processing started in background")

    def get_active_signals(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get active signals, optionally filtered by symbol"""
        if symbol:
            return {k: v for k, v in self.active_signals.items()
                   if isinstance(v, dict) and v.get('symbol') == symbol}
        return self.active_signals.copy()

    def get_signal_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get signal history for a symbol"""
        history = self.signal_history.get(symbol, [])
        return history[-limit:] if history else []

    def get_viper_scores(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get VIPER scores for multiple symbols"""
        scores = {}
        for symbol in symbols:
            # Get cached market data
            market_data = self.get_cached_market_data(symbol)
            if market_data:
                scores[symbol] = self.calculate_viper_score(market_data, symbol)

        return scores

    def get_cached_market_data(self, symbol: str) -> Optional[Dict]:
        """Get cached market data for scoring"""
        try:
            cache_key = f"market_data:{symbol}"
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                return json.loads(cached_data)
            return None

        except Exception as e:
            logger.error(f"❌ Error getting cached data for {symbol}: {e}")
            return None

    def start(self):
        """Start the VIPER scoring service"""
        try:
            logger.info("🚀 Starting VIPER Scoring Service...")

            # Connect to Redis
            if not self.initialize_redis():
                raise Exception("Failed to connect to Redis")

            # Start processing
            self.is_running = True
            self.start_background_processing()

            # Keep main thread alive
            while self.is_running:
                # Publish periodic status updates
                status = {
                    'service': 'viper-scoring-service',
                    'active_signals': len(self.active_signals),
                    'symbols_tracked': len(self.signal_history),
                    'total_signals_generated': sum(len(history) for history in self.signal_history.values()),
                    'timestamp': datetime.now().isoformat()
                }

                self.redis_client.publish('service_status', json.dumps(status))
                asyncio.run(asyncio.sleep(60))  # Update every minute

        except KeyboardInterrupt:
            logger.info("⏹️ Stopping VIPER Scoring Service...")
            self.stop()
        except Exception as e:
            logger.error(f"❌ VIPER Scoring Service error: {e}")
            self.stop()

    def stop(self):
        """Stop the VIPER scoring service"""
        self.is_running = False
        logger.info("✅ VIPER Scoring Service stopped")

# FastAPI application
app = FastAPI(
    title="VIPER Scoring Service",
    version="1.0.0",
    description="Centralized VIPER scoring and signal generation"
)

viper_service = VIPERScoringService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if not viper_service.initialize_redis():
        logger.error("❌ Failed to initialize Redis")
        return

    # Start background processing
    asyncio.create_task(asyncio.to_thread(viper_service.start_background_processing))
    logger.info("✅ VIPER Scoring Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    viper_service.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "viper-scoring-service",
            "redis_connected": viper_service.redis_client is not None,
            "processing_active": viper_service.is_running,
            "active_signals": len(viper_service.active_signals),
            "symbols_tracked": len(viper_service.signal_history)
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "viper-scoring-service",
                "error": str(e)
            }
        )

@app.post("/api/score")
async def calculate_score(request: Request):
    """Calculate VIPER score for market data"""
    try:
        data = await request.json()
        symbol = data.get('symbol', '')
        market_data = data.get('market_data', {})

        if not symbol or not market_data:
            raise HTTPException(status_code=400, detail="Symbol and market_data are required")

        score_result = viper_service.calculate_viper_score(market_data, symbol)
        return score_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

@app.post("/api/signal")
async def generate_signal(request: Request):
    """Generate trading signal for market data"""
    try:
        data = await request.json()
        symbol = data.get('symbol', '')
        market_data = data.get('market_data', {})

        if not symbol or not market_data:
            raise HTTPException(status_code=400, detail="Symbol and market_data are required")

        signal = viper_service.generate_signal(market_data, symbol)
        if signal:
            return signal
        else:
            return {"message": "No signal generated", "symbol": symbol}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {e}")

@app.get("/api/signals")
async def get_signals(symbol: Optional[str] = None, limit: int = Query(50, ge=1, le=200)):
    """Get active signals"""
    try:
        signals = viper_service.get_active_signals(symbol)
        signal_list = list(signals.values())[-limit:] if signals else []
        return {
            "signals": signal_list,
            "count": len(signal_list),
            "total_active": len(viper_service.active_signals)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get signals: {e}")

@app.get("/api/history/{symbol}")
async def get_signal_history(symbol: str, limit: int = Query(50, ge=1, le=200)):
    """Get signal history for a symbol"""
    try:
        history = viper_service.get_signal_history(symbol, limit)
        return {
            "symbol": symbol,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get history: {e}")

@app.post("/api/batch/score")
async def batch_score(request: Request):
    """Calculate VIPER scores for multiple symbols"""
    try:
        data = await request.json()
        symbols = data.get('symbols', [])

        if not symbols or len(symbols) > 100:
            raise HTTPException(status_code=400, detail="Symbols list required (max 100)")

        scores = viper_service.get_viper_scores(symbols)
        return {
            "scores": scores,
            "symbols_requested": len(symbols),
            "symbols_scored": len(scores)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {e}")

@app.get("/api/config")
async def get_scoring_config():
    """Get current scoring configuration"""
    return {
        "thresholds": {
            "high": VIPER_THRESHOLD_HIGH,
            "medium": VIPER_THRESHOLD_MEDIUM
        },
        "cooldown": SIGNAL_COOLDOWN,
        "max_signals_per_symbol": MAX_SIGNALS_PER_SYMBOL,
        "weights": viper_service.scoring_weights,
        "periods": {
            "volume": viper_service.volume_period,
            "price": viper_service.price_period,
            "range": viper_service.range_period
        }
    }

@app.get("/api/stats")
async def get_scoring_stats():
    """Get scoring service statistics"""
    try:
        total_signals = sum(len(history) for history in viper_service.signal_history.values())
        avg_score = 0.0

        if viper_service.active_signals:
            scores = [s.get('viper_score', {}).get('overall_score', 0)
                     for s in viper_service.active_signals.values()
                     if isinstance(s, dict)]
            if scores:
                avg_score = sum(scores) / len(scores)

        return {
            "active_signals": len(viper_service.active_signals),
            "symbols_tracked": len(viper_service.signal_history),
            "total_signals_generated": total_signals,
            "average_active_score": round(avg_score, 2),
            "signals_by_type": {
                "LONG": len([s for s in viper_service.active_signals.values()
                           if isinstance(s, dict) and s.get('type') == 'LONG']),
                "SHORT": len([s for s in viper_service.active_signals.values()
                            if isinstance(s, dict) and s.get('type') == 'SHORT'])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get stats: {e}")

if __name__ == "__main__":
    port = int(os.getenv("VIPER_SCORING_SERVICE_PORT", 8009))
    logger.info(f"Starting VIPER Scoring Service on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
