#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Complete Trade Execution Workflow
Demonstrates end-to-end trading from signal generation to execution with TP/SL/TSL
"""

import asyncio
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradingWorkflow:
    """Complete trading workflow orchestrator"""

    def __init__(self):
        self.api_server_url = "http://localhost:8000"
        self.risk_manager_url = "http://localhost:8002"
        self.order_lifecycle_url = "http://localhost:8013"
        self.exchange_connector_url = "http://localhost:8005"
        self.signal_processor_url = "http://localhost:8011"
        self.position_synchronizer_url = "http://localhost:8014"

        # Trading parameters
        self.symbol = "BTC/USDT:USDT"
        self.risk_per_trade = 0.02  # 2% per trade
        self.max_position_size_percent = 0.1  # 10% of capital


    async def check_system_status(self) -> bool:
        """Check if all required services are running"""

        services = {
            "API Server": self.api_server_url,
            "Risk Manager": self.risk_manager_url,
            "Order Lifecycle Manager": self.order_lifecycle_url,
            "Exchange Connector": self.exchange_connector_url,
            "Signal Processor": self.signal_processor_url,
            "Position Synchronizer": self.position_synchronizer_url
        }

        all_running = True
        for name, url in services.items():
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                else:
                    print(f"   ❌ {name}: Status {response.status_code}")
                    all_running = False
            except Exception as e:
                print(f"   ❌ {name}: Not accessible ({str(e)[:50]}...)")
                all_running = False

        return all_running

    async def get_market_data(self) -> Optional[Dict]:
        """Get current market data"""

        try:
            response = requests.get(
                f"{self.exchange_connector_url}/api/ticker?symbol={self.symbol}",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                ticker = data.get('ticker', {})
                price = ticker.get('last', ticker.get('close', 0))
                print(f"   📊 24h Change: {ticker.get('percentage', 0):.2f}%")
                print(f"   💰 24h Volume: {ticker.get('quoteVolume', 0):.2f}")
                return ticker
            else:
                print(f"   ❌ Failed to get market data: {response.status_code}")
                return None

        except Exception as e:
            return None

    async def generate_trading_signal(self, market_data: Dict) -> Optional[Dict]:
        """Generate a trading signal using VIPER strategy"""

        try:
            # Send market data to signal processor
            signal_request = {
                'symbol': self.symbol,
                'market_data': market_data,
                'timeframe': '1h',
                'strategy': 'VIPER'
            }

            response = requests.post(
                f"{self.signal_processor_url}/api/signals/generate",
                json=signal_request,
                timeout=15
            )

            if response.status_code == 200:
                signal_data = response.json()
                signal = signal_data.get('signal')

                if signal:
                    print(f"   🎯 Signal Generated: {signal.get('side', 'UNKNOWN')}")
                    print(f"   💯 Confidence: {signal.get('confidence', 0):.2f}")
                    print(f"   📊 VIPER Score: {signal_data.get('viper_score', 0):.2f}")
                    return signal
                else:
                    print("   📊 No signal generated (market conditions not favorable)")
                    return None
            else:
                print(f"   ❌ Signal generation failed: {response.status_code}")
                return None

        except Exception as e:
            return None

    async def validate_signal_with_risk(self, signal: Dict) -> Optional[Dict]:
        """Validate signal with risk management"""
        print("\n⚖️  Validating Signal with Risk Management...")

        try:
            # Get current account balance
            balance_response = requests.get(
                f"{self.exchange_connector_url}/api/balance",
                timeout=10
            )

            if balance_response.status_code != 200:
                return None

            balance_data = balance_response.json()
            account_balance = balance_data.get('free', 0)

            if account_balance <= 0:
                return None

            # Calculate position size based on risk
            entry_price = signal.get('entry_price', 0)
            if entry_price <= 0:
                return None

            # Calculate position size (risk per trade)
            position_value = account_balance * self.risk_per_trade
            position_size = position_value / entry_price

            # Apply maximum position size limit
            max_position_value = account_balance * self.max_position_size_percent
            if position_value > max_position_value:
                position_value = max_position_value
                position_size = max_position_value / entry_price

            print(f"   💰 Account Balance: ${account_balance:.2f}")
            print(f"   📊 Position Size: {position_size:.6f} BTC")
            print(f"   💵 Position Value: ${position_value:.2f}")
            print(f"   🎯 Risk per Trade: {(self.risk_per_trade * 100):.1f}%")

            # Calculate TP/SL/TSL levels
            tp_sl_request = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'entry_price': entry_price,
                'stop_loss_percent': 0.02,  # 2%
                'take_profit_percent': 0.06,  # 6%
                'trailing_stop_percent': 0.01  # 1%
            }

            tp_sl_response = requests.post(
                f"{self.risk_manager_url}/api/tp-sl-tsl/calculate",
                json=tp_sl_request,
                timeout=10
            )

            if tp_sl_response.status_code == 200:
                tp_sl_data = tp_sl_response.json()
                levels = tp_sl_data['levels']

                validated_signal = {
                    'symbol': signal['symbol'],
                    'side': signal['side'],
                    'size': position_size,
                    'entry_price': entry_price,
                    'stop_loss': levels['stop_loss'],
                    'take_profit': levels['take_profit'],
                    'trailing_stop': levels['trailing_stop'],
                    'trailing_activation': levels['trailing_activation'],
                    'confidence': signal.get('confidence', 0),
                    'risk_amount': position_value * self.risk_per_trade,
                    'potential_profit': position_value * 0.06  # 6% take profit
                }

                print(f"   📉 Stop Loss: ${levels['stop_loss']:.2f}")
                print(f"   📈 Take Profit: ${levels['take_profit']:.2f}")
                print(f"   🎯 Trailing Stop: ${levels['trailing_stop']:.2f}")
                print(f"   🚀 Risk Amount: ${validated_signal['risk_amount']:.2f}")
                print(f"   💎 Potential Profit: ${validated_signal['potential_profit']:.2f}")

                return validated_signal
            else:
                print(f"   ❌ TP/SL calculation failed: {tp_sl_response.status_code}")
                return None

        except Exception as e:
            return None

    async def execute_trade_order(self, validated_signal: Dict) -> Optional[Dict]:
        """Execute the trade with TP/SL/TSL orders"""

        try:
            # Create the complete order
            order_request = {
                'symbol': validated_signal['symbol'],
                'side': validated_signal['side'],
                'size': validated_signal['size'],
                'entry_price': validated_signal['entry_price'],
                'stop_loss': validated_signal['stop_loss'],
                'take_profit': validated_signal['take_profit'],
                'trailing_stop': validated_signal['trailing_stop'],
                'trailing_activation': validated_signal['trailing_activation']
            }

            response = requests.post(
                f"{self.order_lifecycle_url}/api/tp-sl-tsl/create-order",
                json=order_request,
                timeout=15
            )

            if response.status_code == 200:
                order_result = response.json()
                order = order_result['order']
                placement = order_result['placement_results']


                if placement['success']:
                    if placement.get('main_order'):
                        print(f"      • Main Order: {placement['main_order']}")
                    if placement.get('stop_loss_order'):
                        print(f"      • Stop Loss: {placement['stop_loss_order']}")
                    if placement.get('take_profit_order'):
                        print(f"      • Take Profit: {placement['take_profit_order']}")
                else:
                    print(f"   ⚠️  Some orders failed: {placement['errors']}")

                return order_result
            else:
                print(f"   ❌ Order execution failed: {response.status_code}")
                return None

        except Exception as e:
            return None

    async def monitor_position(self, symbol: str, duration_minutes: int = 5) -> Dict:
        """Monitor position and handle TP/SL/TSL triggers"""
        print(f"\n👀 Monitoring Position for {duration_minutes} minutes...")

        start_time = datetime.now()
        monitoring_results = {
            'symbol': symbol,
            'start_time': start_time.isoformat(),
            'price_updates': [],
            'actions_taken': [],
            'final_status': 'monitoring'
        }

        print(f"   ⏰ Start Time: {start_time.strftime('%H:%M:%S')}")
        print(f"   📊 Monitoring {symbol} for price movements and triggers...")

        try:
            while (datetime.now() - start_time).seconds < (duration_minutes * 60):
                # Get current market data
                market_data = await self.get_market_data()
                if not market_data:
                    print("   ⚠️  Cannot get market data, continuing...")
                    await asyncio.sleep(10)
                    continue

                current_price = market_data.get('last', market_data.get('close', 0))

                # Update position with current price
                update_request = {
                    'symbol': symbol,
                    'current_price': current_price
                }

                response = requests.post(
                    f"{self.risk_manager_url}/api/tp-sl-tsl/update-price",
                    json=update_request,
                    timeout=10
                )

                if response.status_code == 200:
                    update_result = response.json()

                    # Record price update
                    monitoring_results['price_updates'].append({
                        'timestamp': datetime.now().isoformat(),
                        'price': current_price,
                        'action': update_result.get('action_taken')
                    })

                    if update_result.get('action_taken'):
                        action = update_result['action_taken']
                        print(f"\n   🎯 ACTION TRIGGERED: {action['action']}")

                        monitoring_results['actions_taken'].append(action)

                        if action['action'] in ['TAKE_PROFIT', 'STOP_LOSS', 'TRAILING_STOP']:
                            monitoring_results['final_status'] = 'closed'
                            print(f"   🔔 Position closed via {action['action']}")
                            break
                    else:
                        position = update_result.get('position')
                        if position:
                            pnl = position.get('unrealized_pnl', 0)
                            print(f"   📊 Price: ${current_price:.2f} | P&L: ${pnl:.2f}", end='\r')
                else:
                    print(f"   ⚠️  Price update failed: {response.status_code}")

                await asyncio.sleep(5)  # Check every 5 seconds

        except KeyboardInterrupt:
            monitoring_results['final_status'] = 'stopped'
        except Exception as e:
            monitoring_results['final_status'] = 'error'

        # Get final position status
        try:
            status_response = requests.get(
                f"{self.order_lifecycle_url}/api/tp-sl-tsl/status/{symbol}",
                timeout=10
            )

            if status_response.status_code == 200:
                final_status = status_response.json()
                monitoring_results['final_position'] = final_status
            else:
                print(f"\n   ⚠️  Could not get final position status")

        except Exception as e:
            print(f"\n   ⚠️  Error getting final status: {e}")

        return monitoring_results

    async def run_complete_workflow(self):
        """Run the complete trading workflow"""
        print("🎯 Starting Complete VIPER Trading Workflow...")

        # Step 1: Check system status
        if not await self.check_system_status():
            print("   python scripts/start_microservices.py start")
            return

        # Step 2: Get market data
        market_data = await self.get_market_data()
        if not market_data:
            return

        # Step 3: Generate trading signal
        signal = await self.generate_trading_signal(market_data)
        if not signal:
            print("\n📊 No trading signal generated. Market conditions may not be favorable.")
            print("   This is normal - the VIPER strategy is conservative.")
            return

        # Step 4: Validate signal with risk management
        validated_signal = await self.validate_signal_with_risk(signal)
        if not validated_signal:
            return

        # Step 5: Execute trade with TP/SL/TSL
        order_result = await self.execute_trade_order(validated_signal)
        if not order_result:
            return

        symbol = validated_signal['symbol']

        # Step 6: Monitor position (optional - comment out for production)
        print("\n⏳ Would you like to monitor this position for TP/SL/TSL triggers?")
        print("   (This will run for 5 minutes and can be stopped with Ctrl+C)")
        try:
            monitoring_results = await self.monitor_position(symbol, duration_minutes=5)

            print(f"   📊 Status: {monitoring_results['final_status']}")
            print(f"   📈 Price Updates: {len(monitoring_results['price_updates'])}")
            print(f"   🎯 Actions Taken: {len(monitoring_results['actions_taken'])}")

            if monitoring_results['actions_taken']:
                for action in monitoring_results['actions_taken']:
                    print(f"      • {action['action']} at ${action['price']} (P&L: ${action['pnl']:.2f})")

        except KeyboardInterrupt:

        print("🎉 COMPLETE VIPER TRADING WORKFLOW FINISHED!")
        print("\n🚀 The VIPER trading system is fully operational!")
        print("   Ready for live trading with complete risk management!")

async def main():
    """Main function"""
    workflow = TradingWorkflow()
    await workflow.run_complete_workflow()

if __name__ == "__main__":
    asyncio.run(main())
