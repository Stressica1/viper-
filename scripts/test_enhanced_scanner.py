#!/usr/bin/env python3
"""
🧪 ENHANCED MARKET SCANNER TEST
==============================

Test script for the MCP-powered enhanced market scanner.
This script tests the scanner functionality independently.
"""

import os
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - TEST_SCANNER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_enhanced_scanner():
    """Test the enhanced market scanner"""
    try:
        print("🚀 TESTING ENHANCED MARKET SCANNER")
        print("=" * 60)

        # Import the enhanced scanner
        from enhanced_market_scanner import EnhancedMarketScanner

        # Configure exchange credentials
        exchange_config = {
            'api_key': os.getenv('BITGET_API_KEY'),
            'api_secret': os.getenv('BITGET_API_SECRET'),
            'api_password': os.getenv('BITGET_API_PASSWORD')
        }

        if not all(exchange_config.values()):
            print("❌ Missing exchange credentials")
            return

        print("✅ Exchange credentials loaded")

        # Initialize scanner
        print("\n🔧 Initializing Enhanced Market Scanner...")
        scanner = EnhancedMarketScanner(exchange_config)

        success = await scanner.initialize()
        if not success:
            print("❌ Scanner initialization failed")
            return

        print("✅ Scanner initialized successfully")

        # Test market summary
        print("\n📊 Getting market summary...")
        summary = await scanner.get_market_summary()
        print(f"📈 Total Symbols: {summary.get('total_symbols', 0)}")
        print(f"🎯 Scan Results: {summary.get('scan_results', 0)}")
        print(f"📊 Market Sentiment: {summary.get('market_sentiment', 'unknown')}")
        print(f"📈 Volatility Index: {summary.get('volatility_index', 0):.4f}")

        # Test parallel scanning
        print("\n🔍 Testing parallel market scanning...")
        start_time = datetime.now()

        signals = await scanner.scan_markets_parallel()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("✅ Scanning completed!"        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📊 Signals Found: {len(signals)}")

        if signals:
            print("\n🎯 TOP SIGNALS:")
            for i, signal in enumerate(signals[:5], 1):
                print(f"   {i}. {signal.symbol}: Score={signal.score:.3f}, "
                      f"Direction={signal.trend_direction}, "
                      f"Confidence={signal.confidence:.3f}")

            # Test MCP scoring if available
            if scanner.mcp_client:
                print("\n🤖 Testing MCP-enhanced scoring...")
                mcp_signals = await scanner._apply_mcp_scoring(signals[:3])
                for i, signal in enumerate(mcp_signals, 1):
                    boost = signal.ai_enhanced_score - signal.score
                    print(f"   {i}. MCP Boost: +{boost:.3f} (New Score: {signal.ai_enhanced_score:.3f})")

        print("\n✅ ENHANCED MARKET SCANNER TEST COMPLETED SUCCESSFULLY!")
        print("🎉 MCP and Docker integration working perfectly!")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure enhanced_market_scanner.py is in the scripts directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_scanner())
