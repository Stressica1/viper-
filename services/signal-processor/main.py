#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Signal Processor
Event-driven signal processing and trading signal generation

Features:
- Real-time market data processing
- VIPER strategy implementation
- Signal generation and scoring
- Event-driven architecture
- Redis pub/sub integration
"""

import os
import json
import logging
import asyncio
import threading
import redis
from enum import Enum

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'signal-processor')
VIPER_THRESHOLD = float(os.getenv('VIPER_THRESHOLD', '85'))
SIGNAL_COOLDOWN = int(os.getenv('SIGNAL_COOLDOWN', '300'))  # 5 minutes

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

class SignalProcessor:
    """Event-driven signal processing service"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False
        self.subscribers = []
        self.symbol_data = {}  # Store recent market data per symbol
        self.signals = {}  # Store active signals
        self.last_signal_time = {}  # Track cooldown periods
        self.vip_scores = {}  # Store VIPER scores

        # VIPER Strategy parameters
        self.atr_period = int(os.getenv('ATR_PERIOD', '200'))
        self.sma_period = int(os.getenv('SMA_PERIOD', '20'))
        self.rsi_period = int(os.getenv('RSI_PERIOD', '14'))

    def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL)
            self.redis_client.ping()
            logger.info("✅ Connected to Redis")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise

    def calculate_viper_score(self, symbol: str, market_data: Dict) -> float:
        """Calculate VIPER (Volume, Price, External, Range) score"""
        try:
            # Extract data
            ticker = market_data.get('ticker', {})
            orderbook = market_data.get('orderbook', {})
            trades = market_data.get('trades', [])

            if not ticker or not orderbook:
                return 0.0

            # Volume Analysis (V)
            volume = ticker.get('quoteVolume', 0) or ticker.get('volume', 0)
            volume_score = min(volume / 1000000, 100)  # Normalize to 0-100

            # Price Action (P)
            price_change = ticker.get('percentage', 0) or ticker.get('change', 0)
            price_score = max(0, min(100, 50 + price_change * 10))

            # External Factors (E)
            spread = abs(orderbook.get('asks', [0])[0] - orderbook.get('bids', [0])[0]) if orderbook.get('asks') and orderbook.get('bids') else 0
            spread_score = max(0, 100 - spread * 1000)

            # Range Analysis (R)
            high = ticker.get('high', 0)
            low = ticker.get('low', 0)
            current = ticker.get('last', ticker.get('close', 0))

            if high and low and current:
                range_score = ((current - low) / (high - low)) * 100 if high != low else 50
            else:
                range_score = 50

            # Calculate weighted score
            viper_score = (
                volume_score * 0.3 +
                price_score * 0.3 +
                spread_score * 0.2 +
                range_score * 0.2
            )

            return min(100, max(0, viper_score))

        except Exception as e:
            logger.error(f"❌ Error calculating VIPER score for {symbol}: {e}")
            return 0.0

    def generate_signal(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal based on VIPER strategy"""
        try:
            # Check cooldown period
            current_time = datetime.now()
            last_signal = self.last_signal_time.get(symbol, datetime.min)

            if (current_time - last_signal).seconds < SIGNAL_COOLDOWN:
                return None

            # Calculate VIPER score
            viper_score = self.calculate_viper_score(symbol, market_data)
            self.vip_scores[symbol] = viper_score

            if viper_score < VIPER_THRESHOLD:
                return None

            # Extract market data
            ticker = market_data.get('ticker', {})
            current_price = ticker.get('last', ticker.get('close', 0))

            if not current_price:
                return None

            # Simple trend analysis
            price_change = ticker.get('percentage', 0)

            # Generate signal based on price action and VIPER score
            if price_change > 0.5 and viper_score > 90:
                signal_type = SignalType.LONG
                confidence = min(100, viper_score)
            elif price_change < -0.5 and viper_score > 90:
                signal_type = SignalType.SHORT
                confidence = min(100, viper_score)
            else:
                signal_type = SignalType.HOLD
                confidence = 50

            if signal_type == SignalType.HOLD:
                return None

            # Create signal
            signal = {
                'symbol': symbol,
                'type': signal_type.value,
                'price': current_price,
                'viper_score': viper_score,
                'confidence': confidence,
                'timestamp': current_time.isoformat(),
                'market_data': market_data
            }

            # Update tracking
            self.last_signal_time[symbol] = current_time
            self.signals[symbol] = signal

            logger.info(f"🎯 Generated {signal_type.value} signal for {symbol} (Score: {viper_score:.2f})")

            return signal

        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return None

    def process_market_data(self, message: Dict):
        """Process incoming market data"""
        try:
            data_type = message.get('type')
            symbol = message.get('symbol')

            if not symbol or data_type not in ['ticker', 'orderbook', 'trades']:
                return

            # Update symbol data
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = {}

            self.symbol_data[symbol][data_type] = message.get('data', {})
            self.symbol_data[symbol]['timestamp'] = message.get('timestamp')

            # Check if we have enough data to generate a signal
            symbol_data = self.symbol_data[symbol]
            if 'ticker' in symbol_data and 'orderbook' in symbol_data:
                signal = self.generate_signal(symbol, symbol_data)

                if signal:
                    # Publish signal to Redis
                    self.redis_client.publish('trading_signals', json.dumps(signal))

                    # Publish to specific symbol channel
                    self.redis_client.publish(f'signals:{symbol}', json.dumps(signal))

                    logger.info(f"📡 Published {signal['type']} signal for {symbol}")

        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}")

    def subscribe_to_market_data(self):
        """Subscribe to market data streams"""
        try:
            pubsub = self.redis_client.pubsub()

            # Subscribe to all market data channels
            channels = [
                'market_data:ticker',
                'market_data:orderbook',
                'market_data:trades'
            ]

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

    def get_active_signals(self) -> Dict[str, Dict]:
        """Get all active signals"""
        return self.signals.copy()

    def get_viper_scores(self) -> Dict[str, float]:
        """Get VIPER scores for all symbols"""
        return self.vip_scores.copy()

    def start(self):
        """Start the signal processor"""
        try:
            logger.info("🚀 Starting Signal Processor...")

            # Connect to Redis
            self.connect_redis()

            # Start processing
            self.is_running = True
            self.start_background_processing()

            # Keep main thread alive
            while self.is_running:
                # Publish periodic status updates
                status = {
                    'service': 'signal-processor',
                    'active_signals': len(self.signals),
                    'monitored_symbols': len(self.symbol_data),
                    'viper_scores': len(self.vip_scores),
                    'timestamp': datetime.now().isoformat()
                }

                self.redis_client.publish('service_status', json.dumps(status))
                asyncio.run(asyncio.sleep(60))  # Update every minute

        except KeyboardInterrupt:
            logger.info("⏹️ Stopping Signal Processor...")
            self.stop()
        except Exception as e:
            logger.error(f"❌ Signal Processor error: {e}")
            self.stop()

    def stop(self):
        """Stop the signal processor"""
        self.is_running = False
        logger.info("✅ Signal Processor stopped")

def create_app():
    """Create FastAPI application for health checks and API"""
    from fastapi import FastAPI

    app = FastAPI(title="Signal Processor", version="1.0.0")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "signal-processor"}

    @app.get("/signals")
    async def get_signals():
        signals = processor.get_active_signals()
        return {"signals": signals, "count": len(signals)}

    @app.get("/scores")
    async def get_scores():
        scores = processor.get_viper_scores()
        return {"scores": scores, "count": len(scores)}

    @app.get("/metrics")
    async def get_metrics():
        return {
            "active_signals": len(processor.signals),
            "monitored_symbols": len(processor.symbol_data),
            "viper_scores": len(processor.vip_scores),
            "last_signal_time": {k: v.isoformat() for k, v in processor.last_signal_time.items()}
        }

    return app

if __name__ == "__main__":
    # Check if running as API server or processor
    if os.getenv('API_MODE', 'false').lower() == 'true':
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run as signal processor
        processor = SignalProcessor()
        processor.start()
