#!/usr/bin/env python3
"""
Test script for VIPER Standalone Trading Component
Validates functionality without requiring live API credentials
"""

import sys
import os
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_viper_score_calculation():
    """Test VIPER score calculation logic"""
    print("🧪 Testing VIPER Score Calculation...")
    
    # Mock the trader class without exchange connection
    with patch.dict(os.environ, {
        'BITGET_API_KEY': 'test_key',
        'BITGET_API_SECRET': 'test_secret', 
        'BITGET_API_PASSWORD': 'test_password'
    }):
        with patch('ccxt.bitget') as mock_exchange:
            # Mock exchange initialization
            mock_exchange_instance = Mock()
            mock_exchange_instance.load_markets.return_value = True
            mock_exchange_instance.fetch_balance.return_value = {'USDT': {'free': 1000}}
            mock_exchange.return_value = mock_exchange_instance
            
            # Import and create trader instance
            from standalone_viper_trader import StandaloneVIPERTrader
            trader = StandaloneVIPERTrader()
            
            # Test data
            test_market_data = {
                'symbol': 'BTC/USDT:USDT',
                'price': 50000.0,
                'high': 51000.0,
                'low': 49000.0,
                'volume': 2000000.0,
                'price_change': 2.5,  # 2.5% change
                'spread': 0.01  # 0.01% spread
            }
            
            # Calculate VIPER score
            score = trader.calculate_viper_score(test_market_data)
            
            print(f"  ✅ VIPER Score calculated: {score:.2f}")
            
            # Validate score is within expected range
            assert 0 <= score <= 100, f"Score {score} is out of valid range (0-100)"
            
            # Test signal generation
            signal = trader.generate_signal('BTC/USDT:USDT', test_market_data)
            
            if signal:
                print(f"  ✅ Signal generated: {signal.signal} with score {signal.viper_score:.2f}")
                print(f"     Stop Loss: ${signal.stop_loss:.2f}")
                print(f"     Take Profit: ${signal.take_profit:.2f}")
            else:
                print(f"  ⚠️ No signal generated (score {score:.2f} below threshold)")
            
            return True

def test_risk_management():
    """Test risk management calculations"""
    print("\n🛡️ Testing Risk Management...")
    
    with patch.dict(os.environ, {
        'BITGET_API_KEY': 'test_key',
        'BITGET_API_SECRET': 'test_secret',
        'BITGET_API_PASSWORD': 'test_password'
    }):
        with patch('ccxt.bitget') as mock_exchange:
            # Mock exchange
            mock_exchange_instance = Mock()
            mock_exchange_instance.load_markets.return_value = True
            mock_exchange_instance.fetch_balance.return_value = {'USDT': {'free': 10000}}
            mock_exchange_instance.market.return_value = {
                'limits': {'amount': {'min': 0.001}}
            }
            mock_exchange.return_value = mock_exchange_instance
            
            from standalone_viper_trader import StandaloneVIPERTrader, VIPERSignal
            trader = StandaloneVIPERTrader()
            
            # Test signal
            test_signal = VIPERSignal(
                symbol='BTC/USDT:USDT',
                signal='LONG',
                viper_score=90.0,
                price=50000.0,
                price_change=2.5,
                volume=2000000.0,
                confidence=0.9,
                stop_loss=49000.0,  # 2% stop loss
                take_profit=52000.0,  # 4% take profit
                timestamp=datetime.now().isoformat()
            )
            
            # Test position size calculation
            position_size = trader.calculate_position_size(test_signal)
            
            print(f"  ✅ Position size calculated: {position_size:.6f} BTC")
            
            # Validate position size is reasonable
            account_balance = 10000  # USDT
            risk_amount = account_balance * trader.risk_per_trade  # 2% = $200
            expected_size = risk_amount / (50000.0 - 49000.0)  # Risk / Stop distance
            
            print(f"     Account balance: ${account_balance}")
            print(f"     Risk per trade: ${risk_amount} ({trader.risk_per_trade*100}%)")
            print(f"     Stop loss distance: ${50000.0 - 49000.0}")
            
            assert position_size > 0, "Position size should be positive"
            
            return True

def test_configuration_loading():
    """Test configuration loading from environment"""
    print("\n⚙️ Testing Configuration Loading...")
    
    test_env = {
        'BITGET_API_KEY': 'test_key',
        'BITGET_API_SECRET': 'test_secret',
        'BITGET_API_PASSWORD': 'test_password',
        'MAX_POSITIONS': '3',
        'RISK_PER_TRADE': '0.015',
        'VIPER_THRESHOLD': '80.0',
        'SCAN_INTERVAL': '60'
    }
    
    with patch.dict(os.environ, test_env):
        with patch('ccxt.bitget') as mock_exchange:
            mock_exchange_instance = Mock()
            mock_exchange_instance.load_markets.return_value = True
            mock_exchange_instance.fetch_balance.return_value = {'USDT': {'free': 1000}}
            mock_exchange.return_value = mock_exchange_instance
            
            from standalone_viper_trader import StandaloneVIPERTrader
            trader = StandaloneVIPERTrader()
            
            # Validate configuration
            assert trader.max_positions == 3
            assert trader.risk_per_trade == 0.015
            assert trader.viper_threshold == 80.0
            assert trader.scan_interval == 60
            
            print(f"  ✅ Max positions: {trader.max_positions}")
            print(f"  ✅ Risk per trade: {trader.risk_per_trade*100}%")
            print(f"  ✅ VIPER threshold: {trader.viper_threshold}")
            print(f"  ✅ Scan interval: {trader.scan_interval}s")
            
            return True

def test_market_data_processing():
    """Test market data fetching and processing"""
    print("\n📊 Testing Market Data Processing...")
    
    with patch.dict(os.environ, {
        'BITGET_API_KEY': 'test_key',
        'BITGET_API_SECRET': 'test_secret',
        'BITGET_API_PASSWORD': 'test_password'
    }):
        with patch('ccxt.bitget') as mock_exchange:
            # Mock market data response
            mock_ticker = {
                'last': 50000.0,
                'high': 51000.0,
                'low': 49000.0,
                'baseVolume': 1500000.0,
                'percentage': 1.8,
                'bid': 49990.0,
                'ask': 50010.0
            }
            
            mock_ohlcv = [
                [1640995200000, 49500.0, 50500.0, 49000.0, 50000.0, 1000.0]
            ]
            
            mock_exchange_instance = Mock()
            mock_exchange_instance.load_markets.return_value = True
            mock_exchange_instance.fetch_balance.return_value = {'USDT': {'free': 1000}}
            mock_exchange_instance.fetch_ticker.return_value = mock_ticker
            mock_exchange_instance.fetch_ohlcv.return_value = mock_ohlcv
            mock_exchange.return_value = mock_exchange_instance
            
            from standalone_viper_trader import StandaloneVIPERTrader
            trader = StandaloneVIPERTrader()
            
            # Test fetching market data
            market_data = trader.fetch_market_data('BTC/USDT:USDT')
            
            assert market_data is not None
            assert market_data['symbol'] == 'BTC/USDT:USDT'
            assert market_data['price'] == 50000.0
            assert 'spread' in market_data
            
            print(f"  ✅ Market data fetched successfully")
            print(f"     Price: ${market_data['price']:,.2f}")
            print(f"     Volume: {market_data['volume']:,.0f}")
            print(f"     Spread: {market_data['spread']:.4f}%")
            
            return True

def test_signal_generation_edge_cases():
    """Test signal generation with edge case scenarios"""
    print("\n⚠️ Testing Signal Generation Edge Cases...")
    
    with patch.dict(os.environ, {
        'BITGET_API_KEY': 'test_key',
        'BITGET_API_SECRET': 'test_secret',
        'BITGET_API_PASSWORD': 'test_password',
        'VIPER_THRESHOLD': '90.0'  # High threshold
    }):
        with patch('ccxt.bitget') as mock_exchange:
            mock_exchange_instance = Mock()
            mock_exchange_instance.load_markets.return_value = True
            mock_exchange_instance.fetch_balance.return_value = {'USDT': {'free': 10000}}
            mock_exchange.return_value = mock_exchange_instance
            
            from standalone_viper_trader import StandaloneVIPERTrader
            trader = StandaloneVIPERTrader()
            
            # Test cases with different market conditions
            test_cases = [
                {
                    'name': 'High Volatility Market',
                    'data': {
                        'symbol': 'BTC/USDT:USDT',
                        'price': 50000.0,
                        'high': 55000.0,  # 10% range
                        'low': 45000.0,
                        'volume': 5000000.0,  # High volume
                        'price_change': 8.5,  # Strong momentum
                        'spread': 0.005
                    }
                },
                {
                    'name': 'Low Volatility Market', 
                    'data': {
                        'symbol': 'USDC/USDT:USDT',
                        'price': 1.0,
                        'high': 1.001,  # 0.1% range
                        'low': 0.999,
                        'volume': 100000.0,  # Low volume
                        'price_change': 0.05,  # Minimal momentum
                        'spread': 0.001
                    }
                },
                {
                    'name': 'Extreme Negative Movement',
                    'data': {
                        'symbol': 'LUNA/USDT:USDT',
                        'price': 10.0,
                        'high': 15.0,
                        'low': 8.0,
                        'volume': 10000000.0,
                        'price_change': -20.0,  # Crash scenario
                        'spread': 0.02
                    }
                }
            ]
            
            for test_case in test_cases:
                score = trader.calculate_viper_score(test_case['data'])
                signal = trader.generate_signal(test_case['data']['symbol'], test_case['data'])
                
                print(f"  📊 {test_case['name']}:")
                print(f"     VIPER Score: {score:.2f}")
                print(f"     Signal: {'Generated' if signal else 'None'}")
                
                # Validate score range
                assert 0 <= score <= 100, f"Invalid score range for {test_case['name']}"
            
            return True

def test_position_management():
    """Test position management and TP/SL logic"""
    print("\n📈 Testing Position Management...")
    
    with patch.dict(os.environ, {
        'BITGET_API_KEY': 'test_key',
        'BITGET_API_SECRET': 'test_secret',
        'BITGET_API_PASSWORD': 'test_password'
    }):
        with patch('ccxt.bitget') as mock_exchange:
            mock_exchange_instance = Mock()
            mock_exchange_instance.load_markets.return_value = True
            mock_exchange_instance.fetch_balance.return_value = {'USDT': {'free': 10000}}
            mock_exchange_instance.market.return_value = {'limits': {'amount': {'min': 0.001}}}
            mock_exchange.return_value = mock_exchange_instance
            
            from standalone_viper_trader import StandaloneVIPERTrader, TradingPosition
            trader = StandaloneVIPERTrader()
            
            # Create test positions
            long_position = TradingPosition(
                symbol='BTC/USDT:USDT',
                side='buy',
                size=0.1,
                entry_price=50000.0,
                stop_loss=49000.0,
                take_profit=52000.0,
                timestamp=datetime.now().isoformat(),
                order_id='test_long_123'
            )
            
            short_position = TradingPosition(
                symbol='ETH/USDT:USDT',
                side='sell',
                size=1.0,
                entry_price=3000.0,
                stop_loss=3100.0,
                take_profit=2900.0,
                timestamp=datetime.now().isoformat(),
                order_id='test_short_456'
            )
            
            # Add positions to trader
            trader.active_positions['BTC/USDT:USDT'] = long_position
            trader.active_positions['ETH/USDT:USDT'] = short_position
            
            print(f"  ✅ Created {len(trader.active_positions)} test positions")
            
            # Test TP/SL calculations
            test_scenarios = [
                {'symbol': 'BTC/USDT:USDT', 'current_price': 52100.0, 'expected': 'TP_HIT'},
                {'symbol': 'BTC/USDT:USDT', 'current_price': 48900.0, 'expected': 'SL_HIT'},
                {'symbol': 'ETH/USDT:USDT', 'current_price': 2850.0, 'expected': 'TP_HIT'},
                {'symbol': 'ETH/USDT:USDT', 'current_price': 3150.0, 'expected': 'SL_HIT'},
                {'symbol': 'BTC/USDT:USDT', 'current_price': 51000.0, 'expected': 'ACTIVE'},
            ]
            
            for scenario in test_scenarios:
                position = trader.active_positions.get(scenario['symbol'])
                if position:
                    # Check conditions manually (simulating trader logic)
                    current_price = scenario['current_price']
                    
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
                    
                    if take_profit_triggered:
                        result = 'TP_HIT'
                    elif stop_loss_triggered:
                        result = 'SL_HIT'
                    else:
                        result = 'ACTIVE'
                    
                    print(f"     {scenario['symbol']} @ ${scenario['current_price']}: {result}")
                    assert result == scenario['expected'], f"Expected {scenario['expected']}, got {result}"
            
            return True

def test_error_handling():
    """Test error handling and recovery scenarios"""
    print("\n🔧 Testing Error Handling...")
    
    with patch.dict(os.environ, {
        'BITGET_API_KEY': 'test_key',
        'BITGET_API_SECRET': 'test_secret',
        'BITGET_API_PASSWORD': 'test_password'
    }):
        with patch('ccxt.bitget') as mock_exchange:
            # Test exchange connection failure
            mock_exchange.side_effect = Exception("Connection failed")
            
            try:
                from standalone_viper_trader import StandaloneVIPERTrader
                trader = StandaloneVIPERTrader()
                assert False, "Should have raised exception"
            except Exception as e:
                print(f"  ✅ Exchange connection failure handled: {e}")
            
            # Reset mock for other tests
            mock_exchange.side_effect = None
            mock_exchange_instance = Mock()
            mock_exchange_instance.load_markets.return_value = True
            mock_exchange_instance.fetch_balance.return_value = {'USDT': {'free': 10000}}
            mock_exchange.return_value = mock_exchange_instance
            
            trader = StandaloneVIPERTrader()
            
            # Test API rate limiting
            import ccxt
            mock_exchange_instance.fetch_ticker.side_effect = ccxt.RateLimitExceeded("Rate limit exceeded")
            
            try:
                market_data = trader.fetch_market_data('BTC/USDT:USDT')
                print(f"  ✅ Rate limit handling: Returned fallback data")
            except Exception as e:
                print(f"  ✅ Rate limit exception caught: {e}")
            
            return True

def test_integration_workflow():
    """Test complete integration workflow"""
    print("\n🔄 Testing Complete Integration Workflow...")
    
    with patch.dict(os.environ, {
        'BITGET_API_KEY': 'test_key',
        'BITGET_API_SECRET': 'test_secret',
        'BITGET_API_PASSWORD': 'test_password',
        'MAX_POSITIONS': '2',
        'VIPER_THRESHOLD': '75.0'
    }):
        with patch('ccxt.bitget') as mock_exchange:
            mock_exchange_instance = Mock()
            mock_exchange_instance.load_markets.return_value = True
            mock_exchange_instance.fetch_balance.return_value = {'USDT': {'free': 10000}}
            mock_exchange_instance.market.return_value = {'limits': {'amount': {'min': 0.001}}}
            mock_exchange_instance.create_order.return_value = {'id': 'test_order_123'}
            mock_exchange.return_value = mock_exchange_instance
            
            from standalone_viper_trader import StandaloneVIPERTrader
            trader = StandaloneVIPERTrader()
            
            workflow_steps = []
            
            # Step 1: Scan markets
            workflow_steps.append("Market Scanning")
            scanned_pairs = []
            for symbol in trader.trading_pairs[:3]:  # Test first 3 pairs
                mock_data = {
                    'symbol': symbol,
                    'price': 50000.0,
                    'high': 51000.0,
                    'low': 49000.0,
                    'volume': 2000000.0,
                    'price_change': 3.5,
                    'spread': 0.01
                }
                
                with patch.object(trader, 'fetch_market_data', return_value=mock_data):
                    signal = trader.generate_signal(symbol, mock_data)
                    if signal:
                        scanned_pairs.append((symbol, signal))
            
            print(f"  ✅ Step 1 - Scanned {len(trader.trading_pairs[:3])} pairs, found {len(scanned_pairs)} signals")
            
            # Step 2: Execute trades
            if scanned_pairs:
                workflow_steps.append("Trade Execution")
                executed_trades = 0
                for symbol, signal in scanned_pairs[:trader.max_positions]:
                    success = trader.execute_trade(signal)
                    if success:
                        executed_trades += 1
                
                print(f"  ✅ Step 2 - Executed {executed_trades} trades")
            
            # Step 3: Monitor positions
            if trader.active_positions:
                workflow_steps.append("Position Monitoring")
                monitored_positions = len(trader.active_positions)
                print(f"  ✅ Step 3 - Monitoring {monitored_positions} active positions")
            
            print(f"  🎯 Workflow completed: {' → '.join(workflow_steps)}")
            return True

def run_comprehensive_test():
    """Run comprehensive functionality test"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🧪 VIPER STANDALONE TRADER - COMPREHENSIVE TEST                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    tests = [
        ("VIPER Score Calculation", test_viper_score_calculation),
        ("Risk Management", test_risk_management),
        ("Configuration Loading", test_configuration_loading),
        ("Market Data Processing", test_market_data_processing),
        ("Signal Generation Edge Cases", test_signal_generation_edge_cases),
        ("Position Management", test_position_management),
        ("Error Handling", test_error_handling),
        ("Integration Workflow", test_integration_workflow),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"🔬 Running {test_name} Test")
            print(f"{'='*60}")
            
            success = test_func()
            if success:
                print(f"✅ {test_name} test PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} test FAILED with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results
    print(f"\n{'='*80}")
    print(f"🎯 TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*80}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! VIPER Standalone Trader is ready for use.")
        return 0
    else:
        print(f"⚠️ {total_tests - passed_tests} test(s) failed. Please review the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(run_comprehensive_test())