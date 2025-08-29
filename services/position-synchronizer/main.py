#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Position Synchronizer
Real-time position synchronization across all services

Features:
- Synchronize positions across all services
- Real-time position tracking and updates
- Position reconciliation with exchange
- Risk exposure monitoring
- Position history and analytics
"""

import os
import json
import logging
import asyncio
import threading
import redis
import requests

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'position-synchronizer')
VAULT_URL = os.getenv('VAULT_URL', 'http://credential-vault:8008')
VAULT_ACCESS_TOKEN = os.getenv('VAULT_ACCESS_TOKEN', '')

# Service URLs
EXCHANGE_CONNECTOR_URL = os.getenv('EXCHANGE_CONNECTOR_URL', 'http://exchange-connector:8000')
RISK_MANAGER_URL = os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8000')
DATA_MANAGER_URL = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')

# Synchronization settings
SYNC_INTERVAL = int(os.getenv('SYNC_INTERVAL', '30'))  # seconds
RECONCILIATION_INTERVAL = int(os.getenv('RECONCILIATION_INTERVAL', '300'))  # 5 minutes
MAX_POSITION_DRIFT = float(os.getenv('MAX_POSITION_DRIFT', '0.001'))  # 0.1%

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionSynchronizer:
    """Position synchronization service"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False
        self.positions = {}  # Current positions
        self.position_history = []  # Historical positions
        self.last_sync = {}
        self.reconciliation_alerts = []

        # Statistics
        self.stats = {
            'total_syncs': 0,
            'reconciliations': 0,
            'discrepancies_found': 0,
            'positions_tracked': 0,
            'total_exposure': 0.0
        }

    def connect_services(self):
        """Connect to Redis and external services"""
        try:
            # Connect to Redis
            self.redis_client = redis.Redis.from_url(REDIS_URL)
            self.redis_client.ping()
            logger.info("✅ Connected to Redis")

            # Load exchange credentials
            self.load_exchange_credentials()

            logger.info("✅ Connected to all services")

        except Exception as e:
            logger.error(f"❌ Failed to connect services: {e}")
            raise

    def load_exchange_credentials(self):
        """Load API credentials from vault"""
        try:
            response = requests.get(
                f"{VAULT_URL}/credentials/retrieve/bitget/api_key",
                headers={'Authorization': f'Bearer {VAULT_ACCESS_TOKEN}'}
            )
            self.api_key = response.json().get('value')

            response = requests.get(
                f"{VAULT_URL}/credentials/retrieve/bitget/api_secret",
                headers={'Authorization': f'Bearer {VAULT_ACCESS_TOKEN}'}
            )
            self.api_secret = response.json().get('value')

            logger.info("✅ Loaded exchange credentials from vault")

        except Exception as e:
            logger.error(f"❌ Failed to load credentials: {e}")
            raise

    def get_exchange_positions(self) -> Dict[str, Dict]:
        """Get current positions from exchange"""
        try:
            response = requests.get(
                f"{EXCHANGE_CONNECTOR_URL}/positions",
                timeout=10
            )

            if response.status_code == 200:
                positions_data = response.json()
                exchange_positions = {}

                for pos in positions_data.get('positions', []):
                    symbol = pos.get('symbol')
                    if symbol:
                        exchange_positions[symbol] = pos

                logger.info(f"📊 Retrieved {len(exchange_positions)} positions from exchange")
                return exchange_positions
            else:
                logger.error(f"❌ Failed to get exchange positions: {response.text}")
                return {}

        except Exception as e:
            logger.error(f"❌ Error getting exchange positions: {e}")
            return {}

    def get_service_positions(self, service_url: str, service_name: str) -> Dict[str, Dict]:
        """Get positions from a specific service"""
        try:
            response = requests.get(
                f"{service_url}/positions",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                positions = data.get('positions', {})

                logger.debug(f"📊 Retrieved {len(positions)} positions from {service_name}")
                return positions
            else:
                logger.warning(f"⚠️ Failed to get positions from {service_name}: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"❌ Error getting positions from {service_name}: {e}")
            return {}

    def synchronize_positions(self):
        """Synchronize positions across all services"""
        try:
            logger.info("🔄 Starting position synchronization...")

            # Get positions from all sources
            exchange_positions = self.get_exchange_positions()
            risk_manager_positions = self.get_service_positions(RISK_MANAGER_URL, "risk-manager")
            data_manager_positions = self.get_service_positions(DATA_MANAGER_URL, "data-manager")

            # Merge all positions
            all_symbols = set(exchange_positions.keys()) | set(risk_manager_positions.keys()) | set(data_manager_positions.keys())

            synced_positions = {}
            discrepancies = []

            for symbol in all_symbols:
                # Get position data from each source
                exchange_pos = exchange_positions.get(symbol, {})
                risk_pos = risk_manager_positions.get(symbol, {})
                data_pos = data_manager_positions.get(symbol, {})

                # Use exchange as source of truth
                if exchange_pos:
                    synced_pos = exchange_pos.copy()
                    synced_pos['last_sync'] = datetime.now().isoformat()
                    synced_pos['sources'] = {
                        'exchange': True,
                        'risk_manager': bool(risk_pos),
                        'data_manager': bool(data_pos)
                    }

                    # Check for discrepancies
                    if risk_pos and abs(float(exchange_pos.get('amount', 0)) - float(risk_pos.get('amount', 0))) > MAX_POSITION_DRIFT:
                        discrepancies.append({
                            'symbol': symbol,
                            'type': 'risk_manager_discrepancy',
                            'exchange_amount': exchange_pos.get('amount', 0),
                            'service_amount': risk_pos.get('amount', 0),
                            'drift': abs(float(exchange_pos.get('amount', 0)) - float(risk_pos.get('amount', 0)))
                        })

                    synced_positions[symbol] = synced_pos
                    logger.debug(f"✅ Synchronized position for {symbol}")

                elif risk_pos:
                    # Use risk manager position if no exchange data
                    synced_pos = risk_pos.copy()
                    synced_pos['last_sync'] = datetime.now().isoformat()
                    synced_pos['sources'] = {
                        'exchange': False,
                        'risk_manager': True,
                        'data_manager': bool(data_pos)
                    }
                    synced_positions[symbol] = synced_pos
                    logger.debug(f"⚠️ Using risk manager position for {symbol} (no exchange data)")

            # Update internal position cache
            self.positions = synced_positions

            # Publish synchronized positions
            self.redis_client.publish('position_sync', json.dumps({
                'positions': synced_positions,
                'timestamp': datetime.now().isoformat(),
                'discrepancies': discrepancies
            }))

            # Update statistics
            self.stats['total_syncs'] += 1
            self.stats['positions_tracked'] = len(synced_positions)
            self.stats['discrepancies_found'] += len(discrepancies)

            # Calculate total exposure
            total_exposure = sum(abs(float(pos.get('amount', 0)) * float(pos.get('price', 0)))
                               for pos in synced_positions.values() if pos.get('amount') and pos.get('price'))
            self.stats['total_exposure'] = total_exposure

            if discrepancies:
                logger.warning(f"⚠️ Found {len(discrepancies)} position discrepancies")

            logger.info(f"✅ Position synchronization completed: {len(synced_positions)} positions, {len(discrepancies)} discrepancies")

        except Exception as e:
            logger.error(f"❌ Error during position synchronization: {e}")

    def reconcile_positions(self):
        """Perform detailed position reconciliation"""
        try:
            logger.info("🔍 Starting position reconciliation...")

            # Get detailed position data from exchange
            exchange_response = requests.get(
                f"{EXCHANGE_CONNECTOR_URL}/positions/detailed",
                timeout=30
            )

            if exchange_response.status_code != 200:
                logger.error("❌ Failed to get detailed exchange positions")
                return

            exchange_data = exchange_response.json()

            # Compare with internal positions
            reconciliation_results = {
                'timestamp': datetime.now().isoformat(),
                'total_exchange_positions': len(exchange_data.get('positions', [])),
                'total_internal_positions': len(self.positions),
                'matches': 0,
                'mismatches': [],
                'missing_positions': [],
                'extra_positions': []
            }

            exchange_positions = {pos['symbol']: pos for pos in exchange_data.get('positions', [])}

            # Check each internal position against exchange
            for symbol, internal_pos in self.positions.items():
                if symbol in exchange_positions:
                    exchange_pos = exchange_positions[symbol]

                    # Compare key metrics
                    internal_amount = float(internal_pos.get('amount', 0))
                    exchange_amount = float(exchange_pos.get('amount', 0))

                    if abs(internal_amount - exchange_amount) > MAX_POSITION_DRIFT:
                        reconciliation_results['mismatches'].append({
                            'symbol': symbol,
                            'internal_amount': internal_amount,
                            'exchange_amount': exchange_amount,
                            'difference': abs(internal_amount - exchange_amount)
                        })
                    else:
                        reconciliation_results['matches'] += 1

                    # Remove from exchange positions to track extras
                    del exchange_positions[symbol]
                else:
                    reconciliation_results['missing_positions'].append({
                        'symbol': symbol,
                        'internal_amount': internal_pos.get('amount', 0)
                    })

            # Any remaining exchange positions are extras
            for symbol, pos in exchange_positions.items():
                reconciliation_results['extra_positions'].append({
                    'symbol': symbol,
                    'exchange_amount': pos.get('amount', 0)
                })

            # Publish reconciliation results
            self.redis_client.publish('position_reconciliation', json.dumps(reconciliation_results))

            # Generate alerts for significant discrepancies
            total_mismatches = len(reconciliation_results['mismatches'])
            if total_mismatches > 0:
                alert_data = {
                    'alert_type': 'position_discrepancy',
                    'mismatches': total_mismatches,
                    'details': reconciliation_results['mismatches'][:5]  # Top 5 mismatches
                }

                self.redis_client.publish('risk_alerts', json.dumps(alert_data))
                logger.warning(f"🚨 Position reconciliation found {total_mismatches} mismatches")

            # Update statistics
            self.stats['reconciliations'] += 1

            logger.info(f"✅ Position reconciliation completed: {reconciliation_results['matches']} matches, {total_mismatches} mismatches")

        except Exception as e:
            logger.error(f"❌ Error during position reconciliation: {e}")

    def process_position_updates(self, update_data: Dict):
        """Process position update events"""
        try:
            update_type = update_data.get('type')
            symbol = update_data.get('symbol')

            if update_type == 'position_opened':
                logger.info(f"📈 Position opened: {symbol} - Amount: {update_data.get('amount')}")

                # Update internal position cache
                if symbol not in self.positions:
                    self.positions[symbol] = {}

                self.positions[symbol].update({
                    'amount': update_data.get('amount'),
                    'price': update_data.get('price'),
                    'last_update': datetime.now().isoformat(),
                    'status': 'open'
                })

                # Add to history
                self.position_history.append({
                    'symbol': symbol,
                    'type': 'opened',
                    'amount': update_data.get('amount'),
                    'price': update_data.get('price'),
                    'timestamp': datetime.now().isoformat()
                })

            elif update_type == 'position_closed':
                logger.info(f"📉 Position closed: {symbol}")

                if symbol in self.positions:
                    self.positions[symbol]['status'] = 'closed'
                    self.positions[symbol]['closed_at'] = datetime.now().isoformat()

                # Add to history
                self.position_history.append({
                    'symbol': symbol,
                    'type': 'closed',
                    'amount': update_data.get('amount', 0),
                    'price': update_data.get('price', 0),
                    'timestamp': datetime.now().isoformat()
                })

            # Publish updated positions
            self.redis_client.publish('position_updates', json.dumps(update_data))

        except Exception as e:
            logger.error(f"❌ Error processing position update: {e}")

    def subscribe_to_position_events(self):
        """Subscribe to position-related events"""
        try:
            pubsub = self.redis_client.pubsub()

            # Subscribe to position events
            channels = [
                'position_updates',
                'order_updates',
                'trade_updates'
            ]

            pubsub.subscribe(*channels)
            logger.info(f"📡 Subscribed to position event channels: {channels}")

            # Process events
            for message in pubsub.listen():
                if not self.is_running:
                    break

                if message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        channel = message['channel'].decode('utf-8')

                        if channel == 'position_updates':
                            self.process_position_updates(event_data)
                        elif channel == 'order_updates':
                            # Handle order updates that affect positions
                            if event_data.get('status') in ['filled', 'partial']:
                                self.process_position_updates({
                                    'type': 'position_opened',
                                    'symbol': event_data.get('symbol'),
                                    'amount': event_data.get('filled_amount', 0),
                                    'price': event_data.get('price', 0)
                                })

                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to decode position event: {e}")

        except Exception as e:
            logger.error(f"❌ Error in position event subscription: {e}")

    def start_background_synchronization(self):
        """Start background synchronization threads"""
        def run_sync_worker():
            while self.is_running:
                try:
                    self.synchronize_positions()
                    asyncio.run(asyncio.sleep(SYNC_INTERVAL))
                except Exception as e:
                    logger.error(f"❌ Sync worker error: {e}")
                    asyncio.run(asyncio.sleep(10))

        def run_reconciliation_worker():
            while self.is_running:
                try:
                    asyncio.run(asyncio.sleep(RECONCILIATION_INTERVAL))
                    self.reconcile_positions()
                except Exception as e:
                    logger.error(f"❌ Reconciliation worker error: {e}")

        def run_event_processor():
            self.subscribe_to_position_events()

        # Start workers
        sync_thread = threading.Thread(target=run_sync_worker, daemon=True)
        reconciliation_thread = threading.Thread(target=run_reconciliation_worker, daemon=True)
        event_thread = threading.Thread(target=run_event_processor, daemon=True)

        sync_thread.start()
        reconciliation_thread.start()
        event_thread.start()

        logger.info("🔄 Position synchronization workers started")

    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions"""
        return self.positions.copy()

    def get_position_history(self, limit: int = 100) -> List[Dict]:
        """Get position history"""
        return self.position_history[-limit:]

    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        return self.stats.copy()

    def start(self):
        """Start the position synchronizer"""
        try:
            logger.info("🚀 Starting Position Synchronizer...")

            # Connect to services
            self.connect_services()

            # Start synchronization
            self.is_running = True
            self.start_background_synchronization()

            # Keep main thread alive
            while self.is_running:
                # Publish periodic status updates
                status = {
                    'service': 'position-synchronizer',
                    'positions_tracked': len(self.positions),
                    'total_syncs': self.stats['total_syncs'],
                    'reconciliations': self.stats['reconciliations'],
                    'discrepancies': self.stats['discrepancies_found'],
                    'total_exposure': self.stats['total_exposure'],
                    'timestamp': datetime.now().isoformat()
                }

                self.redis_client.publish('service_status', json.dumps(status))
                asyncio.run(asyncio.sleep(60))  # Update every minute

        except KeyboardInterrupt:
            logger.info("⏹️ Stopping Position Synchronizer...")
            self.stop()
        except Exception as e:
            logger.error(f"❌ Position Synchronizer error: {e}")
            self.stop()

    def stop(self):
        """Stop the position synchronizer"""
        self.is_running = False
        logger.info("✅ Position Synchronizer stopped")

def create_app():
    """Create FastAPI application for health checks and API"""
    from fastapi import FastAPI

    app = FastAPI(title="Position Synchronizer", version="1.0.0")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "position-synchronizer"}

    @app.get("/positions")
    async def get_positions():
        positions = synchronizer.get_positions()
        return {"positions": positions, "count": len(positions)}

    @app.get("/positions/{symbol}")
    async def get_position(symbol: str):
        positions = synchronizer.get_positions()
        if symbol in positions:
            return {"position": positions[symbol]}
        return {"error": "Position not found"}, 404

    @app.get("/history")
    async def get_history(limit: int = 50):
        history = synchronizer.get_position_history(limit)
        return {"history": history, "count": len(history)}

    @app.get("/stats")
    async def get_statistics():
        stats = synchronizer.get_sync_statistics()
        return {"statistics": stats}

    @app.post("/sync")
    async def trigger_sync():
        synchronizer.synchronize_positions()
        return {"message": "Position synchronization triggered"}

    @app.post("/reconcile")
    async def trigger_reconciliation():
        synchronizer.reconcile_positions()
        return {"message": "Position reconciliation triggered"}

    return app

if __name__ == "__main__":
    # Check if running as API server or synchronizer
    if os.getenv('API_MODE', 'false').lower() == 'true':
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run as position synchronizer
        synchronizer = PositionSynchronizer()
        synchronizer.start()
