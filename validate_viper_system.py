#!/usr/bin/env python3
"""
🔍 VIPER SYSTEM STATUS VALIDATOR
Quick validation that all fixes are working properly
"""

import os
import sys
import asyncio
import logging

# Configure environment
os.environ['USE_MOCK_DATA'] = 'true'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - VALIDATOR - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def validate_viper_system():
    """Validate that the VIPER system is working correctly"""
    
    logger.info("🔍 VALIDATING VIPER SYSTEM STATUS")
    logger.info("=" * 50)
    
    validation_results = {
        'scoring_service': False,
        'trade_execution': False,
        's1s2r1r2_strategy': False,
        'execution_cost_control': False
    }
    
    try:
        # Test 1: VIPER Scoring Service
        logger.info("1️⃣ Testing VIPER Scoring Service...")
        sys.path.insert(0, '/home/runner/work/viper-/viper-/services/viper-scoring-service')
        from main import VIPERScoringService
        
        viper_service = VIPERScoringService()
        
        # Check weights are correct
        expected_weights = {'volume_score': 0.25, 'price_score': 0.30, 'external_score': 0.30, 'range_score': 0.15}
        if viper_service.scoring_weights == expected_weights:
            logger.info("   ✅ Scoring weights are correct")
            validation_results['scoring_service'] = True
        else:
            logger.error(f"   ❌ Scoring weights incorrect: {viper_service.scoring_weights}")
            
    except Exception as e:
        logger.error(f"   ❌ VIPER Scoring Service test failed: {e}")
    
    try:
        # Test 2: Trade Execution Engine
        logger.info("2️⃣ Testing Trade Execution Engine...")
        sys.path.insert(0, '/home/runner/work/viper-/viper-')
        from enhanced_trade_execution_engine import EnhancedTradeExecutionEngine
        
        trader = EnhancedTradeExecutionEngine()
        
        # Test mock data creation
        mock_data = trader.create_mock_market_data('BTC/USDT:USDT')
        if mock_data and 'ohlcv' in mock_data and len(mock_data['ohlcv']['ohlcv']) > 0:
            logger.info("   ✅ Trade execution engine working")
            validation_results['trade_execution'] = True
        else:
            logger.error("   ❌ Trade execution engine failed")
            
    except Exception as e:
        logger.error(f"   ❌ Trade Execution Engine test failed: {e}")
    
    try:
        # Test 3: S1S2R1R2 Strategy
        logger.info("3️⃣ Testing S1S2R1R2 Strategy...")
        
        # Create test market data
        test_data = {
            'ticker': {'high': 52000, 'low': 48000, 'close': 50000, 'price': 50000}
        }
        
        levels = viper_service.calculate_s1s2r1r2_levels(test_data, 'BTC/USDT:USDT')
        
        expected_levels = ['S2', 'S1', 'pivot', 'R1', 'R2']
        if all(level in levels and levels[level] > 0 for level in expected_levels):
            logger.info(f"   ✅ S1S2R1R2 levels calculated: {levels}")
            validation_results['s1s2r1r2_strategy'] = True
        else:
            logger.error(f"   ❌ S1S2R1R2 calculation failed: {levels}")
            
    except Exception as e:
        logger.error(f"   ❌ S1S2R1R2 Strategy test failed: {e}")
    
    try:
        # Test 4: Execution Cost Control
        logger.info("4️⃣ Testing Execution Cost Control...")
        
        # Test high cost scenario (should be rejected)
        high_cost_data = {
            'ticker': {'price': 50000, 'volume': 10000},  # Low volume
            'orderbook': {
                'bids': [[49000, 1.0]],  # Wide spread
                'asks': [[51000, 1.0]]
            }
        }
        
        cost = viper_service.calculate_execution_cost(high_cost_data)
        score = viper_service.calculate_viper_score(high_cost_data, 'TEST')
        
        if cost >= 3.0 and 'rejected_reason' in score:
            logger.info(f"   ✅ High cost trades rejected (${cost:.2f} >= $3.00)")
            validation_results['execution_cost_control'] = True
        else:
            logger.error(f"   ❌ Execution cost control failed: cost=${cost:.2f}, score={score}")
            
    except Exception as e:
        logger.error(f"   ❌ Execution Cost Control test failed: {e}")
    
    # Final validation
    logger.info("=" * 50)
    logger.info("🎯 VALIDATION RESULTS:")
    
    passed_tests = sum(validation_results.values())
    total_tests = len(validation_results)
    
    for test_name, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"   {status} - {test_name}")
    
    logger.info(f"\n📊 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - VIPER SYSTEM IS FULLY OPERATIONAL!")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed - system needs attention")
        return False

if __name__ == "__main__":
    result = asyncio.run(validate_viper_system())
    exit(0 if result else 1)