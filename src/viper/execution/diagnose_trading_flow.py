#!/usr/bin/env python3
"""
# Rocket VIPER TRADING FLOW DIAGNOSTIC
Comprehensive diagnostic tool for scan/score/trade/TP/SL flow

This diagnostic will:
    pass
# Check Test pair discovery and filtering
# Check Test market data fetching (OHLCV)
# Check Test VIPER scoring system
# Check Test trade execution simulation
# Check Test TP/SL logic
# Check Identify bottlenecks and issues
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import ccxt

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - FLOW_DIAG - %(levelname)s - %(message)s'
()
logger = logging.getLogger(__name__)"""

class TradingFlowDiagnostic:
    """Comprehensive diagnostic for the trading flow""""""

    def __init__(self):
        self.exchange = None
        self.pairs_data = []
        self.qualified_pairs = []
        self.scoring_results = []
        self.trade_simulations = []

        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()

        # Initialize exchange
        self._setup_exchange()

    def _setup_exchange(self):
        """Setup exchange connection""""""
        try:
            api_key = os.getenv('BITGET_API_KEY')
            api_secret = os.getenv('BITGET_API_SECRET')
            api_password = os.getenv('BITGET_API_PASSWORD')

            if not all([api_key, api_secret, api_password]):
                logger.error("# X Missing API credentials")
                return False

            self.exchange = ccxt.bitget({)
                'apiKey': api_key,
                'secret': api_secret,
                'password': api_password,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
(            })

            # Test connection
            self.exchange.load_markets()
            logger.info(f"# Check Exchange connected: {len(self.exchange.markets)} markets")
            return True

        except Exception as e:
            logger.error(f"# X Exchange setup failed: {e}")
            return False

    async def run_complete_diagnostic(self):
        """Run complete trading flow diagnostic""""""

        try:
            # Step 1: Pair Discovery
            await self.diagnose_pair_discovery()

            # Step 2: Pair Filtering
            await self.diagnose_pair_filtering()

            # Step 3: Market Data Fetching
            await self.diagnose_market_data()

            # Step 4: VIPER Scoring
            await self.diagnose_viper_scoring()

            # Step 5: Trade Execution
            await self.diagnose_trade_execution()

            # Step 6: TP/SL Logic
            await self.diagnose_tp_sl_logic()

            # Step 7: Generate Report
            self.generate_diagnostic_report()

        except Exception as e:
            logger.error(f"# X Diagnostic failed: {e}")
            import traceback
            traceback.print_exc()

    async def diagnose_pair_discovery(self):
        """Diagnose pair discovery phase""""""

        try:
            # Discover all USDT swap pairs
            all_symbols = [symbol for symbol in self.exchange.markets.keys() if symbol.endswith('USDT:USDT')]
            logger.info(f"# Chart Found {len(all_symbols)} USDT swap pairs")

            # Sample first 10 pairs for detailed analysis
            sample_pairs = all_symbols[:10]
            logger.info(f"# Search Analyzing first {len(sample_pairs)} pairs...")

            for symbol in sample_pairs:
                try:
                    market_info = self.exchange.markets[symbol]
                    leverage = market_info.get('leverage', {}).get('max', 1)

                    pair_data = {
                        'symbol': symbol,
                        'leverage': leverage,
                        'active': market_info.get('active', False),
                        'precision': market_info.get('precision', {}),
                        'limits': market_info.get('limits', {})
                    }

                    self.pairs_data.append(pair_data)
                    logger.info(f"# Check {symbol}: Leverage={leverage}x, Active={pair_data['active']}")

                except Exception as e:
                    logger.warning(f"# Warning Could not analyze {symbol}: {e}")

            logger.info(f"# Check Pair discovery completed: {len(self.pairs_data)} pairs analyzed")

        except Exception as e:
            logger.error(f"# X Pair discovery failed: {e}")

    async def diagnose_pair_filtering(self):
        """Diagnose pair filtering phase""""""

        try:
            # Filtering criteria
            criteria = {
                'min_volume_threshold': 10000,  # $10K
                'min_leverage_required': 1,     # 1x
                'max_spread_threshold': 0.001,  # 0.1%
                'require_price': True
            }

            logger.info(f"# Target Filtering Criteria: {criteria}")

            qualified_count = 0
            rejected_reasons = {}

            for pair_data in self.pairs_data:
                try:
                    symbol = pair_data['symbol']

                    # Get ticker data (synchronous call)
                    ticker = self.exchange.fetch_ticker(symbol)

                    # Extract metrics
                    volume_24h = ticker.get('quoteVolume', 0)
                    spread = abs(ticker.get('ask', 0) - ticker.get('bid', 0)) / max(ticker.get('bid', 1), 0.000001)
                    leverage = pair_data.get('leverage', 1)
                    price = ticker.get('last', 0)

                    # Check each criterion
                    reasons = []

                    if volume_24h < criteria['min_volume_threshold']:
                        reasons.append(f"Volume too low: ${volume_24h:,.0f} < ${criteria['min_volume_threshold']:,.0f}")

                    if leverage < criteria['min_leverage_required']:
                        reasons.append(f"Leverage too low: {leverage}x < {criteria['min_leverage_required']}x")

                    if spread > criteria['max_spread_threshold']:
                        reasons.append(".4f")

                    if criteria['require_price'] and price <= 0:
                        reasons.append(f"Invalid price: {price}")

                    if not reasons:  # All criteria passed
                        qualified_count += 1
                        self.qualified_pairs.append({)
                            'symbol': symbol,
                            'volume_24h': volume_24h,
                            'spread': spread,
                            'leverage': leverage,
                            'price': price
(                        })
                        logger.info(f"# Check QUALIFIED: {symbol} (Vol: ${volume_24h:,.0f}, Spread: {spread:.4f})")
                    else:
                        # Track rejection reasons
                        for reason in reasons:
                            if reason not in rejected_reasons:
                                rejected_reasons[reason] = 0
                            rejected_reasons[reason] += 1

                except Exception as e:
                    logger.warning(f"# Warning Could not filter {pair_data['symbol']}: {e}")

            logger.info(f"# Target Filtering Results: {qualified_count} qualified, {len(self.pairs_data) - qualified_count} rejected")

            # Show top rejection reasons
            if rejected_reasons:
                logger.info("# Chart Top Rejection Reasons:")
                for reason, count in sorted(rejected_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.info(f"   • {reason}: {count} pairs")

        except Exception as e:
            logger.error(f"# X Pair filtering diagnostic failed: {e}")

    async def diagnose_market_data(self):
        """Diagnose market data fetching (OHLCV)""""""

        try:
            if not self.qualified_pairs:
                logger.warning("# Warning No qualified pairs to test market data")
                return

            # Test OHLCV fetching for qualified pairs
            test_pairs = self.qualified_pairs[:3]  # Test first 3 pairs
            timeframes = ['1h', '4h', '1d']

            for pair_data in test_pairs:
                symbol = pair_data['symbol']
                logger.info(f"# Chart Testing OHLCV for {symbol}")

                for timeframe in timeframes:
                    try:
                        # Test OHLCV fetch (synchronous call)
                        start_time = time.time()
                        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                        fetch_time = time.time() - start_time

                        if ohlcv and len(ohlcv) > 0:
                            logger.info(f"   # Check {timeframe}: {len(ohlcv)} candles in {fetch_time:.2f}s")
                        else:
                            logger.warning(f"   # Warning {timeframe}: No data returned")

                    except Exception as e:
                        logger.error(f"   # X {timeframe}: Failed - {e}")

        except Exception as e:
            logger.error(f"# X Market data diagnostic failed: {e}")

    async def diagnose_viper_scoring(self):
        """Diagnose VIPER scoring system""""""

        try:
            if not self.qualified_pairs:
                logger.warning("# Warning No qualified pairs to test scoring")
                return

            # Import VIPER scorer
            from viper_async_trader import ViperAsyncTrader

            # Create scorer instance with exchange
            scorer = ViperAsyncTrader()
            scorer.exchange = self.exchange  # Set the exchange connection

            # Test scoring on qualified pairs
            for pair_data in self.qualified_pairs[:3]:  # Test first 3 pairs
                symbol = pair_data['symbol']
                logger.info(f"# Target Testing VIPER scoring for {symbol}")

                try:
                    # Create test data for scoring
                    test_data = {
                        'symbol': symbol,
                        'price': pair_data.get('price', 1.0),
                        'volume': pair_data.get('volume_24h', 0),
                        'change': 2.5,  # Assume 2.5% change
                        'high': pair_data.get('price', 1.0) * 1.05,
                        'low': pair_data.get('price', 1.0) * 0.95
                    }

                    # Get VIPER score
                    opportunities = await scorer.scan_opportunities()

                    if opportunities:
                        logger.info(f"   # Chart Found {len(opportunities)} total opportunities")
                        # Show all opportunities found
                        for opp in opportunities[:5]:  # Show first 5
                            logger.info(f"   # Target {opp.symbol}: Score {opp.score:.2f}/100 ({opp.recommended_side})")

                        # Find best opportunity for our symbol or similar
                        best_opp = None
                        for opp in opportunities:
                            if opp.symbol == symbol or symbol in opp.symbol:
                                best_opp = opp
                                break

                        if not best_opp and opportunities:
                            # Use the highest scoring opportunity as fallback
                            best_opp = max(opportunities, key=lambda x: x.score)

                        if best_opp:
                            self.scoring_results.append({)
                                'symbol': symbol,
                                'score': best_opp.score,
                                'side': best_opp.recommended_side,
                                'confidence': best_opp.confidence
(                            })
                            logger.info(f"   # Check VIPER Score: {best_opp.score:.2f}/100 ({best_opp.recommended_side})")
                        else:
                            logger.info(f"   # Warning No opportunities available")
                    else:
                        logger.warning(f"   # Warning Scoring failed for {symbol}")

                except Exception as e:
                    logger.error(f"   # X Scoring failed for {symbol}: {e}")

        except Exception as e:
            logger.error(f"# X VIPER scoring diagnostic failed: {e}")

    async def diagnose_trade_execution(self):
        """Diagnose trade execution logic""""""

        try:
            if not self.scoring_results:
                logger.warning("# Warning No scoring results to test trade execution")
                return

            # Test trade execution simulation
            for result in self.scoring_results[:2]:  # Test first 2 results
                symbol = result['symbol']
                score = result['score']
                side = result['side']

                if score >= 75:  # Only test high-confidence trades
                    logger.info(f"💰 Testing trade execution for {symbol} ({side})")

                    try:
                        # Simulate trade parameters
                        price = 50000 if 'BTC' in symbol else 3000  # Sample prices
                        balance = 1000  # $1000 balance
                        risk_per_trade = 0.02  # 2%

                        # Calculate position size
                        risk_amount = balance * risk_per_trade
                        stop_loss_distance = price * 0.05  # 5% stop loss
                        position_size = risk_amount / stop_loss_distance

                        # Simulate TP/SL levels
                        if side == 'LONG':
                            tp_price = price * 1.03  # 3% take profit
                            sl_price = price * 0.95  # 5% stop loss
                        else:
                            tp_price = price * 0.97  # 3% take profit
                            sl_price = price * 1.05  # 5% stop loss

                        trade_sim = {
                            'symbol': symbol,
                            'side': side,
                            'entry_price': price,
                            'position_size': position_size,
                            'risk_amount': risk_amount,
                            'tp_price': tp_price,
                            'sl_price': sl_price,
                            'potential_pnl': position_size * (tp_price - price) if side == 'LONG' else position_size * (price - tp_price),
                            'max_loss': risk_amount
                        }

                        self.trade_simulations.append(trade_sim)

                        logger.info(f"   # Check Simulated {side} trade:")
                        logger.info(f"      Entry: ${price:.2f}, Size: {position_size:.6f}")
                        logger.info(f"      TP: ${tp_price:.2f}, SL: ${sl_price:.2f}")
                        logger.info(f"      Potential P&L: ${trade_sim['potential_pnl']:.2f}")

                    except Exception as e:
                        logger.error(f"   # X Trade simulation failed for {symbol}: {e}")
                else:
                    logger.info(f"   ⏭️ Skipping {symbol} (Score: {score:.1f} < 75)")

        except Exception as e:
            logger.error(f"# X Trade execution diagnostic failed: {e}")

    async def diagnose_tp_sl_logic(self):
        """Diagnose TP/SL logic""""""

        try:
            if not self.trade_simulations:
                logger.warning("# Warning No trade simulations to test TP/SL")
                return

            # Test TP/SL logic for each simulated trade
            for trade in self.trade_simulations:
                symbol = trade['symbol']
                side = trade['side']
                entry_price = trade['entry_price']
                tp_price = trade['tp_price']
                sl_price = trade['sl_price']

                logger.info(f"# Target Testing TP/SL for {symbol} {side}")

                # Simulate different price scenarios
                scenarios = [
                    entry_price * 1.05,  # +5% (should hit TP for LONG, SL for SHORT)
                    entry_price * 0.95,  # -5% (should hit SL for LONG, TP for SHORT)
                    entry_price * 1.02,  # +2% (partial move)
                    entry_price * 0.98   # -2% (partial move)
                ]

                for i, test_price in enumerate(scenarios, 1):
                    try:
                        if side == 'LONG':
                            if test_price >= tp_price:
                                result = "TP HIT"
                                pnl = trade['potential_pnl']
                            elif test_price <= sl_price:
                                result = "SL HIT"
                                pnl = -trade['max_loss']
                            else:
                                result = "POSITION OPEN"
                                pnl = trade['position_size'] * (test_price - entry_price)
                        else:  # SHORT
                            if test_price <= tp_price:
                                result = "TP HIT"
                                pnl = trade['potential_pnl']
                            elif test_price >= sl_price:
                                result = "SL HIT"
                                pnl = -trade['max_loss']
                            else:
                                result = "POSITION OPEN"
                                pnl = trade['position_size'] * (entry_price - test_price)

                        logger.info(f"   Scenario {i}: ${test_price:.2f} → {result} (P&L: ${pnl:.2f})")

                    except Exception as e:
                        logger.error(f"   # X TP/SL test failed for scenario {i}: {e}")

        except Exception as e:
            logger.error(f"# X TP/SL diagnostic failed: {e}")

    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report"""

        report = {
            'diagnostic_timestamp': datetime.now().isoformat(),
            'total_pairs_discovered': len(self.pairs_data),
            'qualified_pairs': len(self.qualified_pairs),
            'scoring_tests': len(self.scoring_results),
            'trade_simulations': len(self.trade_simulations),
            'issues_found': [],
            'recommendations': []
        }

        # Analyze results"""
        if len(self.qualified_pairs) == 0:
            report['issues_found'].append("No pairs qualified - filtering criteria too strict")
            report['recommendations'].append("Reduce volume threshold or leverage requirements")

        if len(self.scoring_results) == 0:
            report['issues_found'].append("VIPER scoring system not working")
            report['recommendations'].append("Check VIPER scoring implementation")

        if len(self.trade_simulations) == 0:
            report['issues_found'].append("Trade execution simulation failed")
            report['recommendations'].append("Fix trade execution logic")

        # Save detailed report
        with open('trading_flow_diagnostic_report.json', 'w') as f:
            json.dump({)
                'summary': report,
                'pairs_data': self.pairs_data[:10],  # First 10 pairs
                'qualified_pairs': self.qualified_pairs,
                'scoring_results': self.scoring_results,
                'trade_simulations': self.trade_simulations
(            }, f, indent=2, default=str)

        # Display summary
        print(f"   Pairs Discovered: {len(self.pairs_data)}")
        print(f"   Pairs Qualified: {len(self.qualified_pairs)}")
        print(f"   Trade Simulations: {len(self.trade_simulations)}")
        print(f"   Issues Found: {len(report['issues_found'])}")

        if report['issues_found']:
            for issue in report['issues_found']:
        if report['recommendations']:
            for rec in report['recommendations']:
        print(f"\n📄 Detailed report saved: trading_flow_diagnostic_report.json")

        # Overall assessment
        if len(report['issues_found']) == 0:
            pass
        else:
            print(f"\n# Warning STATUS: {len(report['issues_found'])} ISSUES NEED ATTENTION")

async def main():
    """Main diagnostic function"""
    diagnostic = TradingFlowDiagnostic()"""

    if diagnostic.exchange:
        await diagnostic.run_complete_diagnostic()
    else:
        print("# X Cannot run diagnostic - exchange connection failed")
        print("Please ensure BITGET_API_KEY, BITGET_API_SECRET, and BITGET_API_PASSWORD are set in .env")

if __name__ == "__main__":
    asyncio.run(main())
