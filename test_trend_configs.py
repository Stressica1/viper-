#!/usr/bin/env python3
"""
🧪 TREND CONFIGURATION TESTER
Test different MA lengths, ATR multipliers, and stability settings
Find optimal parameters for stable trend detection
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict
from advanced_trend_detector import AdvancedTrendDetector, TrendConfig, TrendDirection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendConfigTester:
    """Test different trend detection configurations"""
    
    def __init__(self):
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'BNBUSDT']
        self.results = {}
        
    async def test_configuration(self, config_name: str, config: TrendConfig) -> Dict:
        """Test a specific configuration"""
        logger.info(f"\n🧪 Testing Configuration: {config_name}")
        logger.info(f"   MA: {config.fast_ma_length}-{config.slow_ma_length}-{config.trend_ma_length}")
        logger.info(f"   ATR: {config.atr_length} x {config.atr_multiplier}")
        logger.info(f"   Stability: {config.min_trend_bars} bars, {config.trend_change_threshold*100}% threshold")
        
        detector = AdvancedTrendDetector(config)
        
        if not await detector.initialize_exchange():
            return {"error": "Failed to connect to exchange"}
        
        results = {
            "config_name": config_name,
            "config": config,
            "symbol_results": {},
            "summary": {
                "strong_signals": 0,
                "moderate_signals": 0,
                "weak_signals": 0,
                "neutral_signals": 0,
                "avg_confidence": 0.0,
                "trend_distribution": {}
            }
        }
        
        total_confidence = 0
        signal_count = 0
        
        for symbol in self.test_symbols:
            try:
                # Test single timeframe
                signal = await detector.detect_trend(symbol, '1h')
                
                if signal:
                    # Test multi-timeframe
                    mtf_signals = await detector.multi_timeframe_analysis(symbol)
                    consensus = detector.get_consensus_trend(mtf_signals)
                    
                    symbol_result = {
                        "single_tf": {
                            "direction": signal.direction.value,
                            "strength": signal.strength.value,
                            "confidence": signal.confidence,
                            "atr_value": signal.atr_value,
                            "ma_alignment": signal.ma_alignment
                        },
                        "multi_tf": {
                            "consensus_direction": consensus.direction.value if consensus else "NONE",
                            "consensus_confidence": consensus.confidence if consensus else 0.0,
                            "timeframe_count": len(mtf_signals)
                        }
                    }
                    
                    results["symbol_results"][symbol] = symbol_result
                    
                    # Update summary
                    if signal.strength.value >= 4:
                        results["summary"]["strong_signals"] += 1
                    elif signal.strength.value >= 3:
                        results["summary"]["moderate_signals"] += 1
                    elif signal.strength.value >= 2:
                        results["summary"]["weak_signals"] += 1
                    else:
                        results["summary"]["neutral_signals"] += 1
                    
                    total_confidence += signal.confidence
                    signal_count += 1
                    
                    # Track trend distribution
                    direction = signal.direction.value
                    if direction not in results["summary"]["trend_distribution"]:
                        results["summary"]["trend_distribution"][direction] = 0
                    results["summary"]["trend_distribution"][direction] += 1
                    
                    logger.info(f"   📊 {symbol}: {signal.direction.value} ({signal.strength.value}/5) "
                               f"Conf:{signal.confidence:.2f} ATR:{signal.atr_value:.6f}")
                
            except Exception as e:
                logger.error(f"   ❌ Error testing {symbol}: {e}")
                results["symbol_results"][symbol] = {"error": str(e)}
        
        # Calculate average confidence
        if signal_count > 0:
            results["summary"]["avg_confidence"] = total_confidence / signal_count
        
        detector.exchange.close()
        return results
    
    async def run_comprehensive_test(self):
        """Run comprehensive test with multiple configurations"""
        logger.info("🚀 Starting Comprehensive Trend Detection Testing")
        logger.info(f"📊 Testing symbols: {', '.join(self.test_symbols)}")
        
        # Define test configurations
        test_configs = {
            "AGGRESSIVE": TrendConfig(
                fast_ma_length=8,
                slow_ma_length=21,
                trend_ma_length=89,
                atr_length=14,
                atr_multiplier=1.5,
                min_trend_bars=3,
                trend_change_threshold=0.015
            ),
            "BALANCED": TrendConfig(
                fast_ma_length=21,
                slow_ma_length=50,
                trend_ma_length=200,
                atr_length=14,
                atr_multiplier=2.0,
                min_trend_bars=5,
                trend_change_threshold=0.02
            ),
            "CONSERVATIVE": TrendConfig(
                fast_ma_length=34,
                slow_ma_length=89,
                trend_ma_length=233,
                atr_length=21,
                atr_multiplier=2.5,
                min_trend_bars=8,
                trend_change_threshold=0.03
            ),
            "HIGH_ATR": TrendConfig(
                fast_ma_length=21,
                slow_ma_length=50,
                trend_ma_length=200,
                atr_length=14,
                atr_multiplier=3.0,  # Higher ATR multiplier
                min_trend_bars=5,
                trend_change_threshold=0.02
            ),
            "FAST_MA": TrendConfig(
                fast_ma_length=13,
                slow_ma_length=34,
                trend_ma_length=144,
                atr_length=14,
                atr_multiplier=2.0,
                min_trend_bars=4,
                trend_change_threshold=0.018
            )
        }
        
        # Test each configuration
        all_results = {}
        for config_name, config in test_configs.items():
            try:
                results = await self.test_configuration(config_name, config)
                all_results[config_name] = results
                
                # Brief pause between tests
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Failed to test configuration {config_name}: {e}")
                all_results[config_name] = {"error": str(e)}
        
        # Analyze and compare results
        self.analyze_results(all_results)
        
        return all_results
    
    def analyze_results(self, all_results: Dict):
        """Analyze and compare all configuration results"""
        logger.info("\n" + "="*80)
        logger.info("📊 TREND DETECTION CONFIGURATION ANALYSIS")
        logger.info("="*80)
        
        # Compare configurations
        comparison = []
        
        for config_name, results in all_results.items():
            if "error" in results:
                logger.error(f"❌ {config_name}: {results['error']}")
                continue
            
            summary = results["summary"]
            config = results["config"]
            
            # Calculate performance score
            performance_score = (
                summary["strong_signals"] * 3 +
                summary["moderate_signals"] * 2 +
                summary["weak_signals"] * 1
            ) / len(self.test_symbols)
            
            comparison.append({
                "config_name": config_name,
                "performance_score": performance_score,
                "avg_confidence": summary["avg_confidence"],
                "strong_signals": summary["strong_signals"],
                "trend_diversity": len(summary["trend_distribution"]),
                "config": config
            })
            
            # Detailed results
            logger.info(f"\n🎯 {config_name} Configuration:")
            logger.info(f"   📈 Performance Score: {performance_score:.2f}")
            logger.info(f"   🎲 Average Confidence: {summary['avg_confidence']:.2f}")
            logger.info(f"   💪 Strong Signals: {summary['strong_signals']}/{len(self.test_symbols)}")
            logger.info(f"   📊 Moderate Signals: {summary['moderate_signals']}/{len(self.test_symbols)}")
            logger.info(f"   📉 Trend Distribution: {summary['trend_distribution']}")
        
        # Find best configuration
        if comparison:
            best_config = max(comparison, key=lambda x: x["performance_score"] + x["avg_confidence"])
            
            logger.info(f"\n🏆 BEST PERFORMING CONFIGURATION: {best_config['config_name']}")
            logger.info(f"   📊 Performance Score: {best_config['performance_score']:.2f}")
            logger.info(f"   🎯 Average Confidence: {best_config['avg_confidence']:.2f}")
            logger.info(f"   ⚙️  Settings:")
            best = best_config['config']
            logger.info(f"      MA Lengths: {best.fast_ma_length}-{best.slow_ma_length}-{best.trend_ma_length}")
            logger.info(f"      ATR: {best.atr_length} x {best.atr_multiplier}")
            logger.info(f"      Stability: {best.min_trend_bars} bars, {best.trend_change_threshold*100}% threshold")
            
            # Recommended .env settings
            logger.info(f"\n📝 RECOMMENDED .ENV SETTINGS:")
            logger.info(f"FAST_MA_LENGTH={best.fast_ma_length}")
            logger.info(f"SLOW_MA_LENGTH={best.slow_ma_length}")
            logger.info(f"TREND_MA_LENGTH={best.trend_ma_length}")
            logger.info(f"ATR_LENGTH={best.atr_length}")
            logger.info(f"ATR_MULTIPLIER={best.atr_multiplier}")
            logger.info(f"MIN_TREND_BARS={best.min_trend_bars}")
            logger.info(f"TREND_CHANGE_THRESHOLD={best.trend_change_threshold}")

async def main():
    """Main test function"""
    tester = TrendConfigTester()
    results = await tester.run_comprehensive_test()
    
    logger.info("\n✅ Trend Detection Configuration Testing Complete!")
    logger.info("🎯 Use the recommended settings above for optimal performance.")

if __name__ == "__main__":
    asyncio.run(main())
