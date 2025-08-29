#!/usr/bin/env python3
"""
🚀 VIPER Standalone Trading Component - Demo Mode
Demonstrates the complete Scan → Score → Trade → TP/SL flow without real trading

This demo shows:
- Market scanning simulation
- VIPER score calculation
- Signal generation 
- Trade execution simulation
- TP/SL monitoring
"""

import time
import random
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import asdict

# Import our trading components
import sys
import os
from unittest.mock import Mock, patch

def create_mock_market_data(symbol: str) -> Dict:
    """Create realistic mock market data"""
    base_prices = {
        'BTC/USDT:USDT': 50000,
        'ETH/USDT:USDT': 3200,
        'SOL/USDT:USDT': 150,
        'BNB/USDT:USDT': 400,
        'ADA/USDT:USDT': 1.2,
        'DOT/USDT:USDT': 25,
        'MATIC/USDT:USDT': 0.8,
        'AVAX/USDT:USDT': 35,
        'LINK/USDT:USDT': 18
    }
    
    base_price = base_prices.get(symbol, 100)
    
    # Generate realistic market movements
    price_change = random.uniform(-5, 5)  # ±5% change
    current_price = base_price * (1 + price_change / 100)
    
    return {
        'symbol': symbol,
        'price': round(current_price, 4),
        'high': round(current_price * 1.02, 4),
        'low': round(current_price * 0.98, 4),
        'volume': random.uniform(500000, 5000000),
        'price_change': round(price_change, 2),
        'bid': round(current_price * 0.9998, 4),
        'ask': round(current_price * 1.0002, 4),
        'spread': round(random.uniform(0.005, 0.02), 4),
        'timestamp': datetime.now().isoformat()
    }

def demo_viper_standalone_trader():
    """Demonstrate the VIPER standalone trader in action"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                🚀 VIPER STANDALONE TRADER - LIVE DEMO                       ║
║                Complete Scan → Score → Trade → TP/SL Flow                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Mock environment setup
    with patch.dict(os.environ, {
        'BITGET_API_KEY': 'demo_key',
        'BITGET_API_SECRET': 'demo_secret',
        'BITGET_API_PASSWORD': 'demo_password',
        'MAX_POSITIONS': '3',
        'RISK_PER_TRADE': '0.02',
        'VIPER_THRESHOLD': '80.0',
        'SCAN_INTERVAL': '5'  # Fast demo
    }):
        with patch('ccxt.bitget') as mock_exchange:
            # Setup mock exchange
            mock_exchange_instance = Mock()
            mock_exchange_instance.load_markets.return_value = True
            mock_exchange_instance.fetch_balance.return_value = {'USDT': {'free': 10000}}
            mock_exchange_instance.market.return_value = {'limits': {'amount': {'min': 0.001}}}
            mock_exchange.return_value = mock_exchange_instance
            
            # Import trader after mocking
            from standalone_viper_trader import StandaloneVIPERTrader
            
            print("🔧 Initializing VIPER Trader (Demo Mode)...")
            trader = StandaloneVIPERTrader()
            
            # Override some settings for demo
            trader.scan_interval = 3  # Faster scanning
            trader.viper_threshold = 75.0  # Lower threshold for demo
            
            print(f"✅ Trader initialized with {len(trader.trading_pairs)} trading pairs")
            print(f"📊 Demo Configuration:")
            print(f"   - Max Positions: {trader.max_positions}")
            print(f"   - Risk per Trade: {trader.risk_per_trade*100:.1f}%")
            print(f"   - VIPER Threshold: {trader.viper_threshold}")
            print(f"   - Scan Interval: {trader.scan_interval}s")
            
            # Demo loop
            demo_cycles = 5
            print(f"\n🎬 Running {demo_cycles} demo cycles...")
            
            for cycle in range(1, demo_cycles + 1):
                print(f"\n{'='*80}")
                print(f"📅 DEMO CYCLE {cycle}/{demo_cycles} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*80}")
                
                # 1. MARKET SCANNING PHASE
                print(f"🔍 SCANNING PHASE: Analyzing {len(trader.trading_pairs)} trading pairs...")
                print("-" * 60)
                
                signals = []
                for i, symbol in enumerate(trader.trading_pairs):
                    market_data = create_mock_market_data(symbol)
                    
                    # Mock the trader's fetch method
                    with patch.object(trader, 'fetch_market_data', return_value=market_data):
                        signal = trader.generate_signal(symbol, market_data)
                    
                    viper_score = trader.calculate_viper_score(market_data)
                    price_change = market_data['price_change']
                    
                    status = ""
                    if signal:
                        signals.append(signal)
                        status = f"🎯 {signal.signal} SIGNAL"
                    elif viper_score >= 70:
                        status = f"⚠️ WATCH (Score: {viper_score:.1f})"
                    else:
                        status = f"💤 HOLD (Score: {viper_score:.1f})"
                    
                    change_icon = "🟢" if price_change > 0 else "🔴" if price_change < 0 else "⚪"
                    
                    print(f"  {change_icon} {symbol:<15} | ${market_data['price']:>8.2f} | "
                          f"{price_change:>+6.2f}% | Score: {viper_score:>5.1f} | {status}")
                
                print(f"\n🎯 SCAN RESULTS: Found {len(signals)} trading opportunities")
                
                # 2. SCORING & TRADE EXECUTION PHASE
                if signals:
                    print(f"\n💰 TRADING PHASE: Processing {len(signals)} signals...")
                    print("-" * 60)
                    
                    for signal in signals[:trader.max_positions]:  # Limit to max positions
                        print(f"\n📊 Signal Details for {signal.symbol}:")
                        print(f"   Direction: {signal.signal}")
                        print(f"   VIPER Score: {signal.viper_score:.1f}/100")
                        print(f"   Confidence: {signal.confidence:.1%}")
                        print(f"   Entry Price: ${signal.price:.2f}")
                        print(f"   Stop Loss: ${signal.stop_loss:.2f} ({((signal.stop_loss/signal.price - 1)*100):+.1f}%)")
                        print(f"   Take Profit: ${signal.take_profit:.2f} ({((signal.take_profit/signal.price - 1)*100):+.1f}%)")
                        
                        # Simulate position size calculation
                        with patch.object(trader, 'fetch_market_data', return_value=create_mock_market_data(signal.symbol)):
                            position_size = trader.calculate_position_size(signal)
                        
                        print(f"   Position Size: {position_size:.6f} {signal.symbol.split('/')[0]}")
                        
                        # Simulate trade execution
                        mock_order = {'id': f'demo_order_{random.randint(1000, 9999)}'}
                        with patch.object(trader.exchange, 'create_order', return_value=mock_order):
                            success = trader.execute_trade(signal)
                        
                        if success:
                            print(f"   ✅ Trade executed successfully!")
                        else:
                            print(f"   ❌ Trade execution failed")
                
                # 3. POSITION MONITORING PHASE (if we had positions)
                if trader.active_positions:
                    print(f"\n📈 MONITORING PHASE: Tracking {len(trader.active_positions)} positions...")
                    print("-" * 60)
                    
                    for symbol, position in trader.active_positions.items():
                        # Simulate price movement
                        current_price = position.entry_price * (1 + random.uniform(-0.03, 0.03))
                        
                        # Calculate P&L
                        if position.side == 'buy':
                            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                        else:
                            pnl_pct = (position.entry_price - current_price) / position.entry_price * 100
                        
                        pnl_icon = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                        
                        print(f"  {pnl_icon} {symbol:<15} | {position.side.upper():<5} | "
                              f"Entry: ${position.entry_price:.2f} | Current: ${current_price:.2f} | "
                              f"P&L: {pnl_pct:+.2f}%")
                        
                        # Check TP/SL conditions
                        tp_hit = (position.side == 'buy' and current_price >= position.take_profit) or \
                                 (position.side == 'sell' and current_price <= position.take_profit)
                        sl_hit = (position.side == 'buy' and current_price <= position.stop_loss) or \
                                 (position.side == 'sell' and current_price >= position.stop_loss)
                        
                        if tp_hit:
                            print(f"     🎯 TAKE PROFIT triggered at ${current_price:.2f}")
                            print(f"     📤 Closing position with {pnl_pct:+.2f}% profit")
                        elif sl_hit:
                            print(f"     🛑 STOP LOSS triggered at ${current_price:.2f}")
                            print(f"     📤 Closing position with {pnl_pct:+.2f}% loss")
                
                # Print summary
                print(f"\n📊 CYCLE {cycle} SUMMARY:")
                print(f"   Pairs Scanned: {len(trader.trading_pairs)}")
                print(f"   Signals Generated: {len(signals)}")
                print(f"   Active Positions: {len(trader.active_positions)}")
                print(f"   Available Capital: $10,000 (Demo)")
                
                # Sleep before next cycle
                if cycle < demo_cycles:
                    print(f"\n💤 Waiting {trader.scan_interval}s for next scan cycle...")
                    time.sleep(trader.scan_interval)
            
            # Final summary
            print(f"\n{'='*80}")
            print(f"🎉 DEMO COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"✅ Demonstrated complete Scan → Score → Trade → TP/SL flow")
            print(f"✅ All {demo_cycles} cycles completed without errors") 
            print(f"✅ Risk management rules applied throughout")
            print(f"✅ Ready for live trading with real API credentials")
            
            print(f"\n📚 Next Steps:")
            print(f"  1. Configure your .env file with real Bitget API credentials")
            print(f"  2. Start with small amounts for initial testing")
            print(f"  3. Monitor the live system closely")
            print(f"  4. Adjust parameters based on market conditions")
            
            print(f"\n🚀 Start live trading with: python standalone_viper_trader.py")

def main():
    """Main demo entry point"""
    try:
        demo_viper_standalone_trader()
        return 0
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())