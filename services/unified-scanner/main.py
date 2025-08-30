#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Unified Scanner Service
Comprehensive market scanning, pair analysis, and opportunity detection

Features:
- Unified scanning for all trading pairs
- Leverage availability analysis
- VIPER score integration
- Real-time opportunity detection
- Batch processing with rate limiting
- Redis integration for result caching
"""

import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
import redis
import requests
import ccxt

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'unified-scanner')

# Scanner Configuration
SCAN_ALL_PAIRS = os.getenv('SCAN_ALL_PAIRS', 'true').lower() == 'true'
MAX_PAIRS_LIMIT = int(os.getenv('MAX_PAIRS_LIMIT', '500'))
SCAN_INTERVAL_SECONDS = int(os.getenv('SCAN_INTERVAL_SECONDS', '300'))  # 5 minutes
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))
LEVERAGE_SCAN_ENABLED = os.getenv('LEVERAGE_SCAN_ENABLED', 'true').lower() == 'true'

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedScannerService:
    """Unified scanner for all market analysis and opportunity detection"""

    def __init__(self):
        self.redis_client = None
        self.exchange = None
        self.is_running = False
        self.scan_results = {}
        self.market_data_manager_url = os.getenv('MARKET_DATA_MANAGER_URL', 'http://market-data-manager:8003')
        self.viper_scoring_url = os.getenv('VIPER_SCORING_SERVICE_URL', 'http://viper-scoring-service:8009')

        # Initialize exchange for direct API calls when needed
        self.initialize_exchange()

        logger.info("# Search Unified Scanner Service initialized")

    def initialize_exchange(self) -> bool:
        """Initialize exchange connection for direct API access"""
        try:
            api_key = os.getenv('BITGET_API_KEY', '')
            api_secret = os.getenv('BITGET_API_SECRET', '')
            api_password = os.getenv('BITGET_API_PASSWORD', '')

            if not all([api_key, api_secret, api_password]):
                logger.warning("# Warning Missing API credentials for direct exchange access")
                return False

            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': api_secret,
                'password': api_password,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                },
                'sandbox': False,
                'rateLimit': 100,
                'enableRateLimit': True,
            })

            self.exchange.load_markets()
            logger.info("# Check Exchange connection initialized")
            return True

        except Exception as e:
            logger.error(f"# X Failed to initialize exchange: {e}")
            return False

    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("# Check Redis connection established")
            return True
        except Exception as e:
            logger.error(f"# X Failed to connect to Redis: {e}")
            return False

    def fetch_all_trading_pairs(self) -> List[str]:
        """Fetch all available swap trading pairs"""
        try:
            logger.info("# Search Discovering all available swap pairs...")

            # Use market data manager if available
            try:
                response = requests.get(f"{self.market_data_manager_url}/api/symbols", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    all_pairs = data.get('symbols', [])
                    
                    # Filter for swap pairs only
                    swap_pairs = [s for s in all_pairs if 
                                 ('SWAP' in s.upper() or ':' in s)]
                    
                    logger.info(f"# Check Retrieved {len(swap_pairs)} swap pairs from market data manager")
                    return swap_pairs[:MAX_PAIRS_LIMIT] if MAX_PAIRS_LIMIT > 0 else swap_pairs
            except Exception as e:
                logger.warning(f"# Warning Market data manager unavailable: {e}")

            # Fallback to direct API call - filter for swap pairs only
            if self.exchange:
                # Get all markets and filter for active swap pairs
                markets = self.exchange.markets
                swap_pairs = []
                
                for symbol, market in markets.items():
                    if (market.get('active', False) and 
                        market.get('type') == 'swap'):
                        swap_pairs.append(symbol)
                
                logger.info(f"# Check Found {len(swap_pairs)} swap pairs via direct API")
                return swap_pairs[:MAX_PAIRS_LIMIT] if MAX_PAIRS_LIMIT > 0 else swap_pairs

            logger.error("# X No method available to fetch trading pairs")
            return []

        except Exception as e:
            logger.error(f"# X Error fetching trading pairs: {e}")
            return []

    def scan_pair_details(self, symbol: str) -> Dict:
        """Scan detailed information for a specific trading pair"""
        try:
            # Get market data from market data manager
            response = requests.get(f"{self.market_data_manager_url}/api/market/{symbol}", timeout=5)
            if response.status_code != 200:
                logger.warning(f"# Warning Cannot fetch market data for {symbol}")
                return {'symbol': symbol, 'error': 'market_data_unavailable'}

            market_data = response.json()

            pair_info = {
                'symbol': symbol,
                'base': market_data.get('ticker', {}).get('symbol', symbol).split('/')[0] if '/' in symbol else 'UNKNOWN',
                'quote': 'USDT',
                'current_price': market_data.get('ticker', {}).get('price', 0),
                'price_change': market_data.get('ticker', {}).get('price_change', 0),
                'volume': market_data.get('ticker', {}).get('volume', 0),
                'spread': 0,  # Will be calculated from orderbook
                'last_updated': market_data.get('last_updated', ''),
                'source': 'market_data_manager'
            }

            # Add orderbook analysis
            orderbook = market_data.get('orderbook', {})
            if orderbook and 'bids' in orderbook and 'asks' in orderbook:
                bids = orderbook['bids']
                asks = orderbook['asks']

                if bids and asks:
                    best_bid = bids[0][0] if isinstance(bids[0], list) else bids[0]
                    best_ask = asks[0][0] if isinstance(asks[0], list) else asks[0]
                    pair_info['spread'] = ((best_ask - best_bid) / best_bid) * 100 if best_bid > 0 else 0

                    # Orderbook depth analysis
                    total_bid_volume = sum([bid[1] if isinstance(bid, list) else 0 for bid in bids[:5]])
                    total_ask_volume = sum([ask[1] if isinstance(ask, list) else 0 for ask in asks[:5]])
                    pair_info['bid_volume'] = total_bid_volume
                    pair_info['ask_volume'] = total_ask_volume

            # Check leverage availability if enabled
            if LEVERAGE_SCAN_ENABLED and self.exchange:
                pair_info['leverage_available'] = self.check_leverage_availability(symbol)

            # Get VIPER score if scoring service is available
            try:
                score_response = requests.post(f"{self.viper_scoring_url}/api/score",
                                             json={'symbol': symbol, 'market_data': market_data}, timeout=5)
                if score_response.status_code == 200:
                    score_data = score_response.json()
                    pair_info['viper_score'] = score_data
            except Exception as e:
                logger.debug(f"# Warning VIPER scoring unavailable for {symbol}: {e}")

            return pair_info

        except Exception as e:
            logger.error(f"# X Error scanning pair {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def check_leverage_availability(self, symbol: str) -> bool:
        """Check if a symbol supports leverage trading"""
        try:
            if not self.exchange:
                return False

            market = self.exchange.market(symbol)

            # For Bitget perpetual swaps, leverage is typically available
            if market.get('type') == 'swap' and market.get('active'):
                # Check if leverage tiers are available
                try:
                    leverage_tiers = self.exchange.fetch_leverage_tiers(symbol)
                    if leverage_tiers:
                        max_leverage = max([tier.get('maxLeverage', 0) for tier in leverage_tiers.values()])
                        return max_leverage >= 25  # At least 25x leverage
                    else:
                        # Assume leverage is available for active swaps
                        return True
                except Exception:
                    # If we can't get specific leverage info, assume it's available
                    return True

            return False

        except Exception as e:
            logger.error(f"# X Error checking leverage for {symbol}: {e}")
            return False

    def generate_trading_signals(self, pair_info: Dict) -> Optional[Dict]:
        """Generate trading signals based on pair analysis"""
        try:
            symbol = pair_info.get('symbol', '')
            viper_score = pair_info.get('viper_score', {}).get('overall_score', 0)
            price_change = pair_info.get('price_change', 0)
            current_price = pair_info.get('current_price', 0)

            if viper_score < 70 or current_price <= 0:
                return None

            signal = None
            confidence = min(viper_score / 100, 1.0)

            # Generate signal based on VIPER score and price action
            if viper_score >= 85:
                if price_change > 0.5:
                    signal = 'LONG'
                elif price_change < -0.5:
                    signal = 'SHORT'
                elif viper_score >= 90:
                    # Very high score overrides price change threshold
                    if pair_info.get('viper_score', {}).get('components', {}).get('price_score', 50) > 60:
                        signal = 'LONG'
                    elif pair_info.get('viper_score', {}).get('components', {}).get('price_score', 50) < 40:
                        signal = 'SHORT'

            if signal:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'viper_score': viper_score,
                    'price': current_price,
                    'price_change': price_change,
                    'volume': pair_info.get('volume', 0),
                    'timestamp': datetime.now().isoformat(),
                    'strategy': 'unified_scanner'
                }

            return None

        except Exception as e:
            logger.error(f"# X Error generating signal for {pair_info.get('symbol', 'unknown')}: {e}")
            return None

    def scan_pairs_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """Scan a batch of trading pairs"""
        batch_results = {
            'pairs_scanned': [],
            'signals_generated': [],
            'leverage_pairs': [],
            'high_volume_pairs': [],
            'errors': []
        }

        for symbol in symbols:
            try:
                # Scan pair details
                pair_info = self.scan_pair_details(symbol)
                batch_results['pairs_scanned'].append(pair_info)

                # Check for leverage availability
                if pair_info.get('leverage_available', False):
                    batch_results['leverage_pairs'].append(pair_info)

                # Check for high volume
                if pair_info.get('volume', 0) > 10000:  # Arbitrary threshold
                    batch_results['high_volume_pairs'].append(pair_info)

                # Generate trading signal
                signal = self.generate_trading_signals(pair_info)
                if signal:
                    batch_results['signals_generated'].append(signal)

                    # Publish signal to Redis
                    self.redis_client.publish('trading_signals', json.dumps(signal))

            except Exception as e:
                error_info = {'symbol': symbol, 'error': str(e)}
                batch_results['errors'].append(error_info)
                logger.error(f"# X Error scanning {symbol}: {e}")

        return batch_results

    def perform_comprehensive_scan(self) -> Dict[str, Any]:
        """Perform comprehensive scan of all trading pairs"""
        logger.info("# Search Starting comprehensive market scan...")

        # Fetch all available pairs
        all_symbols = self.fetch_all_trading_pairs()

        if not all_symbols:
            return {'error': 'No trading pairs available'}

        logger.info(f"# Chart Scanning {len(all_symbols)} trading pairs")

        # Process in batches
        all_results = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_pairs': len(all_symbols),
            'pairs_scanned': [],
            'signals_generated': [],
            'leverage_pairs': [],
            'high_volume_pairs': [],
            'errors': [],
            'summary': {}
        }

        for i in range(0, len(all_symbols), BATCH_SIZE):
            batch_symbols = all_symbols[i:i + BATCH_SIZE]
            logger.info(f"📦 Processing batch {i//BATCH_SIZE + 1}/{(len(all_symbols) + BATCH_SIZE - 1)//BATCH_SIZE}")

            batch_results = self.scan_pairs_batch(batch_symbols)

            # Aggregate results
            all_results['pairs_scanned'].extend(batch_results['pairs_scanned'])
            all_results['signals_generated'].extend(batch_results['signals_generated'])
            all_results['leverage_pairs'].extend(batch_results['leverage_pairs'])
            all_results['high_volume_pairs'].extend(batch_results['high_volume_pairs'])
            all_results['errors'].extend(batch_results['errors'])

            # Rate limiting between batches
            if i + BATCH_SIZE < len(all_symbols):
                time.sleep(1)

        # Generate summary
        all_results['summary'] = {
            'total_scanned': len(all_results['pairs_scanned']),
            'signals_generated': len(all_results['signals_generated']),
            'leverage_pairs_found': len(all_results['leverage_pairs']),
            'high_volume_pairs': len(all_results['high_volume_pairs']),
            'errors_count': len(all_results['errors']),
            'scan_duration_seconds': time.time() - time.time(),  # Would need to track start time
            'top_signals': sorted(all_results['signals_generated'],
                                key=lambda x: x.get('confidence', 0), reverse=True)[:10]
        }

        # Cache results in Redis
        scan_key = f"scan_results:{int(time.time())}"
        self.redis_client.setex(scan_key, 3600, json.dumps(all_results))  # 1 hour cache

        logger.info(f"# Check Comprehensive scan completed: {len(all_results['signals_generated'])} signals generated")
        return all_results

    def start_periodic_scanning(self):
        """Start periodic scanning in background thread"""
        def scanning_loop():
            while self.is_running:
                try:
                    logger.info("🔄 Starting periodic comprehensive scan...")
                    results = self.perform_comprehensive_scan()

                    # Publish scan completion event
                    completion_event = {
                        'type': 'scan_completed',
                        'results_summary': results.get('summary', {}),
                        'timestamp': datetime.now().isoformat()
                    }

                    self.redis_client.publish('scanner_events', json.dumps(completion_event))

                    # Wait before next scan
                    time.sleep(SCAN_INTERVAL_SECONDS)

                except Exception as e:
                    logger.error(f"# X Error in scanning loop: {e}")
                    time.sleep(60)  # Wait before retrying

        if SCAN_INTERVAL_SECONDS > 0:
            scanning_thread = threading.Thread(target=scanning_loop, daemon=True)
            scanning_thread.start()
            logger.info(f"⏰ Periodic scanning started (interval: {SCAN_INTERVAL_SECONDS}s)")

    def get_scan_history(self, limit: int = 10) -> List[Dict]:
        """Get recent scan results from Redis"""
        try:
            # Get all scan result keys
            scan_keys = self.redis_client.keys("scan_results:*")

            if not scan_keys:
                return []

            # Get the most recent scans
            recent_scans = []
            for key in sorted(scan_keys, reverse=True)[:limit]:
                scan_data = self.redis_client.get(key)
                if scan_data:
                    recent_scans.append(json.loads(scan_data))

            return recent_scans

        except Exception as e:
            logger.error(f"# X Error getting scan history: {e}")
            return []

    def get_opportunities(self, min_score: float = 80, limit: int = 20) -> List[Dict]:
        """Get current trading opportunities"""
        try:
            # Get recent scan results
            recent_scans = self.get_scan_history(limit=1)

            if not recent_scans:
                return []

            latest_scan = recent_scans[0]
            signals = latest_scan.get('signals_generated', [])

            # Filter by minimum score and sort by confidence
            opportunities = [s for s in signals if s.get('viper_score', 0) >= min_score]
            opportunities.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            return opportunities[:limit]

        except Exception as e:
            logger.error(f"# X Error getting opportunities: {e}")
            return []

# FastAPI application
app = FastAPI(
    title="VIPER Unified Scanner",
    version="1.0.0",
    description="Comprehensive market scanning and opportunity detection service"
)

scanner_service = UnifiedScannerService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if not scanner_service.initialize_redis():
        logger.error("# X Failed to initialize Redis")
        return

    if scanner_service.initialize_exchange():
        logger.info("# Check Exchange connection available")
    else:
        logger.warning("# Warning Exchange connection not available - some features limited")

    # Start periodic scanning
    scanner_service.is_running = True
    scanner_service.start_periodic_scanning()

    logger.info("# Check Unified Scanner Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    scanner_service.is_running = False

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "unified-scanner",
            "redis_connected": scanner_service.redis_client is not None,
            "exchange_connected": scanner_service.exchange is not None,
            "scanning_active": scanner_service.is_running,
            "periodic_scanning": SCAN_INTERVAL_SECONDS > 0
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "unified-scanner",
                "error": str(e)
            }
        )

@app.post("/api/scan/start")
async def start_comprehensive_scan():
    """Start a comprehensive market scan"""
    try:
        results = scanner_service.perform_comprehensive_scan()
        return {
            "status": "scan_completed",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")

@app.get("/api/scan/history")
async def get_scan_history(limit: int = Query(10, ge=1, le=50)):
    """Get scan history"""
    try:
        history = scanner_service.get_scan_history(limit)
        return {
            "scan_history": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get history: {e}")

@app.get("/api/opportunities")
async def get_trading_opportunities(
    min_score: float = Query(80, ge=0, le=100),
    limit: int = Query(20, ge=1, le=100)
):
    """Get current trading opportunities"""
    try:
        opportunities = scanner_service.get_opportunities(min_score, limit)
        return {
            "opportunities": opportunities,
            "count": len(opportunities),
            "min_score": min_score
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get opportunities: {e}")

@app.get("/api/pairs/leverage")
async def get_leverage_pairs(limit: int = Query(50, ge=1, le=200)):
    """Get pairs with leverage support"""
    try:
        recent_scans = scanner_service.get_scan_history(limit=1)

        if not recent_scans:
            return {"leverage_pairs": [], "count": 0}

        leverage_pairs = recent_scans[0].get('leverage_pairs', [])
        return {
            "leverage_pairs": leverage_pairs[:limit],
            "count": len(leverage_pairs),
            "total_available": len(leverage_pairs)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get leverage pairs: {e}")

@app.get("/api/pairs/{symbol}")
async def scan_single_pair(symbol: str):
    """Scan a single trading pair"""
    try:
        pair_info = scanner_service.scan_pair_details(symbol)
        return pair_info
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to scan pair: {e}")

@app.get("/api/stats")
async def get_scanner_stats():
    """Get scanner service statistics"""
    try:
        recent_scans = scanner_service.get_scan_history(limit=1)

        stats = {
            "service_status": "active" if scanner_service.is_running else "inactive",
            "redis_connected": scanner_service.redis_client is not None,
            "exchange_connected": scanner_service.exchange is not None,
            "scan_interval": SCAN_INTERVAL_SECONDS,
            "batch_size": BATCH_SIZE,
            "leverage_scan_enabled": LEVERAGE_SCAN_ENABLED
        }

        if recent_scans:
            latest_scan = recent_scans[0]
            stats.update({
                "last_scan_timestamp": latest_scan.get('scan_timestamp'),
                "last_scan_pairs": latest_scan.get('summary', {}).get('total_scanned', 0),
                "last_scan_signals": latest_scan.get('summary', {}).get('signals_generated', 0),
                "last_scan_leverage_pairs": latest_scan.get('summary', {}).get('leverage_pairs_found', 0)
            })

        return stats
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get stats: {e}")

if __name__ == "__main__":
    port = int(os.getenv("UNIFIED_SCANNER_PORT", 8011))
    logger.info(f"Starting VIPER Unified Scanner on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
