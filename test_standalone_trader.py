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