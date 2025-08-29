#!/usr/bin/env python3
"""
🚨 POSITION ADOPTION & TRACKING SYSTEM
Advanced system for adopting and tracking positions in VIPER trading bot
"""

import os
import sys
import time
import json
import threading
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dotenv import load_dotenv

# Load environment
load_dotenv()

class PositionAdoptionSystem:
    """Advanced position adoption and tracking system for VIPER"""

    def __init__(self):
        self.api_key = os.getenv('BITGET_API_KEY')
        self.api_secret = os.getenv('BITGET_API_SECRET')
        self.api_passphrase = os.getenv('BITGET_API_PASSWORD')

        # Position tracking
        self.active_positions = {}  # Currently active positions
        self.position_history = {}  # Historical position data
        self.adopted_positions = set()  # Symbols we've adopted
        self.last_adoption_check = None

        # Configuration
        self.adoption_interval = 300  # Check for new positions every 5 minutes
        self.position_timeout = 3600  # Consider position stale after 1 hour
        self.min_contract_size = 0.001  # Minimum viable contract size

        # Callbacks
        self.on_position_adopted = None
        self.on_position_closed = None
        self.on_position_updated = None

        # Background monitoring
        self.monitoring_thread = None
        self.is_monitoring = False

        # Exchange connection
        self.exchange = self._init_exchange()

        print("🚀 Position Adoption System initialized")

    def _init_exchange(self):
        """Initialize exchange connection"""
        return ccxt.bitget({
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

    def adopt_existing_positions(self) -> Dict[str, any]:
        """Adopt any existing positions that the bot didn't create"""
        print("🔍 Checking for existing positions to adopt...")

        try:
            # Get current positions from exchange
            positions = self.exchange.fetch_positions()
            real_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]

            newly_adopted = []
            skipped_positions = []

            for pos in real_positions:
                symbol = pos.get('symbol', 'UNKNOWN')
                contracts = float(pos.get('contracts', 0))

                # Skip if too small
                if contracts < self.min_contract_size:
                    skipped_positions.append(f"{symbol} (too small: {contracts})")
                    continue

                # Skip if already adopted
                if symbol in self.adopted_positions:
                    continue

                # Adopt the position
                self._adopt_position(pos)
                newly_adopted.append(symbol)

            self.last_adoption_check = datetime.now()

            result = {
                'success': True,
                'newly_adopted': len(newly_adopted),
                'skipped': len(skipped_positions),
                'total_positions': len(real_positions),
                'adopted_symbols': newly_adopted,
                'skipped_reasons': skipped_positions
            }

            if newly_adopted:
                print(f"✅ Adopted {len(newly_adopted)} new positions: {', '.join(newly_adopted)}")
            else:
                print("ℹ️  No new positions to adopt")

            if skipped_positions:
                print(f"⏭️  Skipped {len(skipped_positions)} positions: {', '.join(skipped_positions[:3])}...")

            return result

        except Exception as e:
            print(f"❌ Error adopting positions: {e}")
            return {
                'success': False,
                'error': str(e),
                'newly_adopted': 0
            }

    def _adopt_position(self, position_data: Dict):
        """Adopt a single position"""
        symbol = position_data.get('symbol', 'UNKNOWN')

        # Create position record
        position_record = {
            'symbol': symbol,
            'position_data': position_data,
            'adopted_at': datetime.now(),
            'last_updated': datetime.now(),
            'source': 'adopted',  # vs 'created' for bot-created positions
            'status': 'active',
            'contracts': float(position_data.get('contracts', 0)),
            'side': position_data.get('side', 'unknown'),
            'entry_price': float(position_data.get('entryPrice', 0)),
            'current_price': float(position_data.get('markPrice', 0)),
            'unrealized_pnl': float(position_data.get('unrealizedPnl', 0)),
            'leverage': int(position_data.get('leverage', 1))
        }

        # Add to active positions
        self.active_positions[symbol] = position_record
        self.adopted_positions.add(symbol)

        # Add to history
        if symbol not in self.position_history:
            self.position_history[symbol] = []
        self.position_history[symbol].append(position_record)

        # Trigger callback
        if self.on_position_adopted:
            try:
                self.on_position_adopted(position_record)
            except Exception as e:
                print(f"⚠️  Position adoption callback error: {e}")

        print(f"📊 Adopted position: {symbol}")

    def update_position_data(self, symbol: str, new_data: Dict):
        """Update position data"""
        if symbol in self.active_positions:
            old_data = self.active_positions[symbol]

            # Update position record
            self.active_positions[symbol].update({
                'position_data': new_data,
                'last_updated': datetime.now(),
                'contracts': float(new_data.get('contracts', old_data['contracts'])),
                'current_price': float(new_data.get('markPrice', old_data['current_price'])),
                'unrealized_pnl': float(new_data.get('unrealizedPnl', old_data['unrealized_pnl']))
            })

            # Trigger callback
            if self.on_position_updated:
                try:
                    self.on_position_updated(self.active_positions[symbol])
                except Exception as e:
                    print(f"⚠️  Position update callback error: {e}")

    def close_position(self, symbol: str, reason: str = "manual") -> bool:
        """Close a position"""
        if symbol not in self.active_positions:
            print(f"⚠️  No active position found for {symbol}")
            return False

        try:
            position = self.active_positions[symbol]
            side = position['side']
            contracts = position['contracts']

            # Close the position
            opposite_side = 'sell' if side.lower() == 'long' else 'buy'
            order = self.exchange.create_market_order(
                symbol, opposite_side, contracts,
                params={'tradeSide': 'close', 'marginMode': 'isolated'}
            )

            if order:
                # Update position status
                position['status'] = 'closed'
                position['closed_at'] = datetime.now()
                position['close_reason'] = reason
                position['close_order_id'] = order.get('id')

                # Remove from active positions
                del self.active_positions[symbol]

                # Trigger callback
                if self.on_position_closed:
                    try:
                        self.on_position_closed(position)
                    except Exception as e:
                        print(f"⚠️  Position close callback error: {e}")

                print(f"✅ Closed position: {symbol} ({reason})")
                return True
            else:
                print(f"❌ Failed to close position: {symbol}")
                return False

        except Exception as e:
            print(f"❌ Error closing position {symbol}: {e}")
            return False

    def sync_positions(self) -> Dict[str, any]:
        """Sync with current exchange positions"""
        try:
            # Get current positions from exchange
            positions = self.exchange.fetch_positions()
            current_symbols = {p.get('symbol') for p in positions if float(p.get('contracts', 0)) > 0}

            # Find positions that have been closed
            closed_symbols = []
            for symbol in list(self.active_positions.keys()):
                if symbol not in current_symbols:
                    closed_symbols.append(symbol)
                    # Mark as closed
                    if symbol in self.active_positions:
                        self.active_positions[symbol]['status'] = 'closed'
                        self.active_positions[symbol]['closed_at'] = datetime.now()
                        self.active_positions[symbol]['close_reason'] = 'external'

                        # Trigger callback
                        if self.on_position_closed:
                            try:
                                self.on_position_closed(self.active_positions[symbol])
                            except Exception as e:
                                print(f"⚠️  Position close callback error: {e}")

                        del self.active_positions[symbol]

            # Update existing positions
            for pos in positions:
                symbol = pos.get('symbol')
                if symbol in self.active_positions:
                    self.update_position_data(symbol, pos)

            return {
                'success': True,
                'positions_synced': len(current_symbols),
                'positions_closed': len(closed_symbols),
                'active_positions': len(self.active_positions),
                'closed_symbols': closed_symbols
            }

        except Exception as e:
            print(f"❌ Error syncing positions: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        return list(self.active_positions.values())

    def get_position_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get position data by symbol"""
        return self.active_positions.get(symbol)

    def get_position_summary(self) -> Dict[str, any]:
        """Get position summary"""
        active_positions = self.get_active_positions()

        total_pnl = sum(p.get('unrealized_pnl', 0) for p in active_positions)
        total_contracts = sum(p.get('contracts', 0) for p in active_positions)

        winning_positions = [p for p in active_positions if p.get('unrealized_pnl', 0) > 0]
        losing_positions = [p for p in active_positions if p.get('unrealized_pnl', 0) < 0]

        return {
            'total_positions': len(active_positions),
            'winning_positions': len(winning_positions),
            'losing_positions': len(losing_positions),
            'total_pnl': total_pnl,
            'total_contracts': total_contracts,
            'last_sync': self.last_adoption_check.isoformat() if self.last_adoption_check else None
        }

    def start_monitoring(self):
        """Start background position monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("👀 Position monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("🛑 Position monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Sync positions
                sync_result = self.sync_positions()

                if sync_result.get('positions_closed', 0) > 0:
                    print(f"ℹ️  {sync_result['positions_closed']} positions closed externally")

                # Check for new positions to adopt
                if (not self.last_adoption_check or
                    (datetime.now() - self.last_adoption_check).seconds > self.adoption_interval):
                    adoption_result = self.adopt_existing_positions()

            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")

            # Wait before next check
            time.sleep(60)  # Check every minute

def main():
    """Demonstrate position adoption system"""
    print("🚀 POSITION ADOPTION SYSTEM DEMO")
    print("=" * 50)

    system = PositionAdoptionSystem()

    # Initial adoption
    print("1️⃣  Initial position adoption...")
    adoption_result = system.adopt_existing_positions()

    if adoption_result['success']:
        print(f"   ✅ Adopted {adoption_result['newly_adopted']} positions")

        # Show active positions
        active_positions = system.get_active_positions()
        if active_positions:
            print(f"\n📊 ACTIVE POSITIONS ({len(active_positions)}):")
            for i, pos in enumerate(active_positions, 1):
                symbol = pos['symbol']
                side = pos['side']
                contracts = pos['contracts']
                pnl = pos['unrealized_pnl']
                adopted_at = pos['adopted_at'].strftime('%H:%M:%S')

                print(f"   {i}. {symbol} - {side.upper()} - {contracts} contracts")
                print(f"      P&L: ${pnl:.2f} | Adopted: {adopted_at}")

        # Show summary
        summary = system.get_position_summary()
        print(f"\n📈 SUMMARY:")
        print(f"   Total: {summary['total_positions']}")
        print(f"   Winning: {summary['winning_positions']}")
        print(f"   Losing: {summary['losing_positions']}")
        print(f"   Total P&L: ${summary['total_pnl']:.2f}")

    print("\n✅ Position adoption system ready!")

if __name__ == "__main__":
    main()
