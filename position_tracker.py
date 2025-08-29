#!/usr/bin/env python3
"""
🚨 ADVANCED POSITION TRACKER & ADOPTER
Comprehensive position tracking and adoption system for Bitget
"""

import os
import sys
import time
import json
import ccxt
import requests
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment
load_dotenv()

class PositionTracker:
    """Advanced position tracking and adoption system"""

    def __init__(self):
        self.api_key = os.getenv('BITGET_API_KEY')
        self.api_secret = os.getenv('BITGET_API_SECRET')
        self.api_passphrase = os.getenv('BITGET_API_PASSWORD')
        self.base_url = 'https://api.bitget.com'

        # Position tracking state
        self.known_positions = {}  # Positions we know about
        self.last_sync = None
        self.sync_interval = 300  # 5 minutes
        self.position_cache = {}
        self.cache_timeout = 60  # 1 minute

        # Exchange connection
        self.exchange = None
        self._init_exchange()

        # Validation thresholds
        self.min_contract_size = 0.001  # Minimum viable contract size
        self.max_stale_time = 3600  # 1 hour max stale time
        self.max_api_retries = 3

    def _init_exchange(self):
        """Initialize exchange connection"""
        self.exchange = ccxt.bitget({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'password': self.api_passphrase,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
                'hedgeMode': False,
            },
            'sandbox': False,
        })

    def _generate_signature(self, timestamp, method, path, body=''):
        """Generate Bitget API signature"""
        message = str(timestamp) + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_headers(self, method, path, body=''):
        """Get API request headers"""
        timestamp = int(time.time() * 1000)
        signature = self._generate_signature(timestamp, method, path, body)

        return {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': str(timestamp),
            'ACCESS-PASSPHRASE': self.api_passphrase,
            'Content-Type': 'application/json'
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.position_cache:
            return False

        cache_time = self.position_cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_timeout

    def _cache_result(self, cache_key: str, data):
        """Cache API result"""
        self.position_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }

    def get_positions_via_api(self) -> List[Dict]:
        """Get positions via direct API call"""
        cache_key = 'positions_api'
        if self._is_cache_valid(cache_key):
            return self.position_cache[cache_key]['data']

        try:
            path = '/api/v2/mix/position/all-position'
            params = 'productType=USDT-FUTURES&marginCoin=USDT'

            response = requests.get(
                f"{self.base_url}{path}?{params}",
                headers=self._get_headers('GET', path + '?' + params),
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    positions = data.get('data', [])
                    self._cache_result(cache_key, positions)
                    return positions
                else:
                    print(f"❌ API Error: {data.get('msg', 'Unknown error')}")
                    return []
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return []

        except Exception as e:
            print(f"❌ Error fetching positions via API: {e}")
            return []

    def get_positions_via_ccxt(self) -> List[Dict]:
        """Get positions via CCXT"""
        cache_key = 'positions_ccxt'
        if self._is_cache_valid(cache_key):
            return self.position_cache[cache_key]['data']

        try:
            positions = self.exchange.fetch_positions()
            real_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]
            self._cache_result(cache_key, real_positions)
            return real_positions
        except Exception as e:
            print(f"❌ Error fetching positions via CCXT: {e}")
            return []

    def test_position_closability(self, symbol: str, side: str, contracts: float) -> bool:
        """Test if a position can actually be closed"""
        try:
            # Try to close with a very small amount first
            test_contracts = min(contracts * 0.01, 0.001)  # 1% or 0.001 minimum
            opposite_side = 'sell' if side.lower() == 'long' else 'buy'

            order = self.exchange.create_market_order(
                symbol, opposite_side, test_contracts,
                params={'tradeSide': 'close', 'marginMode': 'isolated'}
            )

            # If successful, immediately open it back
            if order:
                reopen_order = self.exchange.create_market_order(
                    symbol, side, test_contracts,
                    params={'tradeSide': 'open', 'marginMode': 'isolated'}
                )
                return True

            return False

        except Exception as e:
            error_msg = str(e)
            if "No position to close" in error_msg:
                return False  # Position doesn't actually exist
            elif "balance" in error_msg.lower():
                return False  # Balance issue
            else:
                return False  # Other error

    def validate_position_status(self, positions: List[Dict]) -> List[Dict]:
        """Validate which positions are actually real and closable"""
        validated_positions = []

        print(f"🔍 Validating {len(positions)} reported positions...")

        for i, pos in enumerate(positions, 1):
            symbol = pos.get('symbol', 'UNKNOWN')
            side = pos.get('side', 'unknown')
            contracts = float(pos.get('contracts', 0) or pos.get('total', 0))
            pnl = float(pos.get('unrealizedPnl', 0))

            print(f"   [{i}/{len(positions)}] Testing {symbol}...")

            # Skip if too small
            if contracts < self.min_contract_size:
                print(f"      ⏭️  SKIPPED: Too small ({contracts} < {self.min_contract_size})")
                continue

            # Test if position can be closed
            is_closable = self.test_position_closability(symbol, side, contracts)

            if is_closable:
                print(f"      ✅ VALID: Position is closable")
                validated_positions.append(pos)
            else:
                print(f"      ❌ INVALID: Position not closable (likely cached/stale)")

            # Rate limiting
            time.sleep(0.5)

        return validated_positions

    def adopt_positions(self, positions: List[Dict]):
        """Adopt and track validated positions"""
        self.known_positions = {}

        for pos in positions:
            symbol = pos.get('symbol', 'UNKNOWN')
            self.known_positions[symbol] = {
                'position': pos,
                'adopted_at': datetime.now(),
                'last_updated': datetime.now(),
                'status': 'active',
                'validation_attempts': 1
            }

        print(f"📊 Adopted {len(self.known_positions)} validated positions")

    def sync_with_exchange(self) -> Dict[str, any]:
        """Comprehensive sync with exchange"""
        print("🔄 SYNCING WITH EXCHANGE...")
        print("=" * 60)

        # Get positions from multiple sources
        api_positions = self.get_positions_via_api()
        ccxt_positions = self.get_positions_via_ccxt()

        print(f"📊 API reports: {len(api_positions)} positions")
        print(f"📊 CCXT reports: {len(ccxt_positions)} positions")

        # Compare sources
        api_symbols = {p.get('symbol') for p in api_positions}
        ccxt_symbols = {p.get('symbol') for p in ccxt_positions}

        common_symbols = api_symbols & ccxt_symbols
        api_only_symbols = api_symbols - ccxt_symbols
        ccxt_only_symbols = ccxt_symbols - api_symbols

        print(f"📊 Common positions: {len(common_symbols)}")
        print(f"📊 API-only positions: {len(api_only_symbols)}")
        print(f"📊 CCXT-only positions: {len(ccxt_only_symbols)}")

        # Use CCXT positions as primary source (more reliable)
        if ccxt_positions:
            print("🔍 Validating CCXT positions...")
            validated_positions = self.validate_position_status(ccxt_positions)

            if validated_positions:
                self.adopt_positions(validated_positions)
                self.last_sync = datetime.now()

                return {
                    'success': True,
                    'validated_positions': len(validated_positions),
                    'total_reported': len(ccxt_positions),
                    'sync_time': datetime.now()
                }
            else:
                print("⚠️  No positions could be validated as closable")
                return {
                    'success': False,
                    'validated_positions': 0,
                    'total_reported': len(ccxt_positions),
                    'error': 'No valid positions found'
                }
        else:
            print("✅ No positions found - clean state")
            self.known_positions = {}
            return {
                'success': True,
                'validated_positions': 0,
                'total_reported': 0,
                'message': 'No positions to track'
            }

    def get_active_positions(self) -> List[Dict]:
        """Get currently active validated positions"""
        active_positions = []

        for symbol, data in self.known_positions.items():
            if data['status'] == 'active':
                position = data['position']
                # Add tracking metadata
                position['_tracked_since'] = data['adopted_at'].isoformat()
                position['_last_updated'] = data['last_updated'].isoformat()
                position['_validation_attempts'] = data['validation_attempts']
                active_positions.append(position)

        return active_positions

    def update_position_status(self, symbol: str, status: str):
        """Update status of a tracked position"""
        if symbol in self.known_positions:
            self.known_positions[symbol]['status'] = status
            self.known_positions[symbol]['last_updated'] = datetime.now()

    def remove_position(self, symbol: str):
        """Remove a position from tracking"""
        if symbol in self.known_positions:
            del self.known_positions[symbol]
            print(f"🗑️  Removed {symbol} from tracking")

    def get_tracking_stats(self) -> Dict[str, any]:
        """Get tracking statistics"""
        active = len([p for p in self.known_positions.values() if p['status'] == 'active'])
        inactive = len([p for p in self.known_positions.values() if p['status'] != 'active'])

        return {
            'total_tracked': len(self.known_positions),
            'active_positions': active,
            'inactive_positions': inactive,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'sync_age_seconds': (datetime.now() - self.last_sync).seconds if self.last_sync else None
        }

def main():
    """Main position tracking demonstration"""
    print("🚨 ADVANCED POSITION TRACKER")
    print("🔧 Comprehensive position tracking and adoption system")
    print("=" * 60)

    tracker = PositionTracker()

    # Perform comprehensive sync
    sync_result = tracker.sync_with_exchange()

    print("\n" + "=" * 60)
    print("📊 SYNC RESULTS:")
    print(f"   ✅ Success: {sync_result.get('success', False)}")
    print(f"   📊 Validated Positions: {sync_result.get('validated_positions', 0)}")
    print(f"   📈 Total Reported: {sync_result.get('total_reported', 0)}")

    if sync_result.get('success'):
        # Show active positions
        active_positions = tracker.get_active_positions()

        if active_positions:
            print(f"\n📋 ACTIVE POSITIONS ({len(active_positions)}):")
            for i, pos in enumerate(active_positions, 1):
                symbol = pos.get('symbol', 'UNKNOWN')
                side = pos.get('side', 'unknown')
                contracts = float(pos.get('contracts', 0))
                pnl = float(pos.get('unrealizedPnl', 0))
                tracked_since = pos.get('_tracked_since', 'unknown')

                print(f"   {i}. {symbol} - {side.upper()} - {contracts} contracts")
                print(f"      P&L: ${pnl:.2f} | Tracked since: {tracked_since}")
        else:
            print("\n✅ NO ACTIVE POSITIONS")
            print("   All positions have been validated as closed or non-existent")

        # Show tracking stats
        stats = tracker.get_tracking_stats()
        print(f"\n📈 TRACKING STATS:")
        print(f"   Total tracked: {stats['total_tracked']}")
        print(f"   Active: {stats['active_positions']}")
        print(f"   Inactive: {stats['inactive_positions']}")

    print("\n✅ Position tracking complete!")

if __name__ == "__main__":
    main()
