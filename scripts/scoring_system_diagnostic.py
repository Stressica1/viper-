#!/usr/bin/env python3
"""
# Rocket VIPER Trading System - Comprehensive Scoring System Diagnostic
Complete analysis and diagnosis of the VIPER scoring system and components

Features:
- VIPER algorithm analysis and validation
- Signal processor diagnostic
- Scoring system performance testing
- Real-time score calculation testing
- Component connectivity verification
- Scoring system optimization recommendations
"""

import os
import sys
import json
import time
import requests
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from enum import Enum
import ccxt

# Load environment variables
BITGET_API_KEY = os.getenv('BITGET_API_KEY', '')
BITGET_API_SECRET = os.getenv('BITGET_API_SECRET', '')
BITGET_API_PASSWORD = os.getenv('BITGET_API_PASSWORD', '')

class ScoringSystemStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    FAILED = "FAILED"

class VIPERScoringDiagnostic:
    """
    Comprehensive diagnostic tool for VIPER scoring system
    """

    def __init__(self):
        # Use current working directory or environment variable for better portability
        self.project_root = os.getenv('VIPER_PROJECT_ROOT', os.getcwd())
        self.diagnostic_report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'INITIALIZING',
            'components': {},
            'scoring_algorithm': {},
            'performance_metrics': {},
            'issues': [],
            'recommendations': []
        }

        # Initialize exchange connection
        self.exchange = None
        self.initialize_exchange()

    def initialize_exchange(self):
        """Initialize Bitget exchange connection"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': BITGET_API_KEY,
                'secret': BITGET_API_SECRET,
                'password': BITGET_API_PASSWORD,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                },
                'sandbox': False,
            })
            self.exchange.loadMarkets()
        except Exception as e:
            self.diagnostic_report['issues'].append(f"Exchange initialization failed: {e}")

    def print_header(self):
        """Print diagnostic header"""
#==============================================================================#
# # Search VIPER SCORING SYSTEM DIAGNOSTIC - COMPLETE ANALYSIS                        #
# Comprehensive diagnosis of VIPER scoring algorithm and components             #
#==============================================================================#
""")

    def check_service_health(self, service_name: str, port: int) -> Dict:
        """Check if a service is healthy"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                return {
                    'status': 'HEALTHY',
                    'response_time': response.elapsed.total_seconds(),
                    'data': response.json()
                }
            else:
                return {
                    'status': 'DEGRADED',
                    'response_code': response.status_code,
                    'error': 'Non-200 response'
                }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'DOWN',
                'error': str(e)
            }

    def diagnose_signal_processor(self) -> Dict:
        """Diagnose signal processor service"""

        result = {
            'service_status': self.check_service_health('signal-processor', 8006),
            'algorithm_analysis': {},
            'performance_metrics': {},
            'issues': [],
            'recommendations': []
        }

        # Check if service is running via Docker
        try:
            ps_result = subprocess.run(['docker', 'ps', '--filter', 'name=signal-processor', '--format', 'json'],
                                     capture_output=True, text=True, timeout=10)

            if ps_result.returncode == 0 and ps_result.stdout.strip():
                result['docker_status'] = 'RUNNING'
                container_info = json.loads(ps_result.stdout.strip())
                result['container_id'] = container_info.get('ID', 'Unknown')
            else:
                result['docker_status'] = 'NOT_RUNNING'
                result['issues'].append("Signal processor container not running")
        except Exception as e:
            result['docker_status'] = 'ERROR'
            result['issues'].append(f"Failed to check Docker status: {e}")

        # Analyze VIPER algorithm implementation
        result['algorithm_analysis'] = self.analyze_viper_algorithm()

        return result

    def analyze_viper_algorithm(self) -> Dict:
        """Analyze VIPER scoring algorithm implementation"""
        analysis = {
            'components': {},
            'weights': {},
            'thresholds': {},
            'issues': [],
            'performance': {}
        }

        try:
            # Read signal processor code
            signal_processor_path = f"{self.project_root}/services/signal-processor/main.py"
            with open(signal_processor_path, 'r') as f:
                code = f.read()

            # Extract VIPER calculation logic
            if 'def calculate_viper_score' in code:
                analysis['components']['volume_score'] = 'Volume Analysis (V)' in code
                analysis['components']['price_score'] = 'Price Action (P)' in code
                analysis['components']['spread_score'] = 'External Factors (E)' in code
                analysis['components']['range_score'] = 'Range Analysis (R)' in code

                # Extract weights
                if 'volume_score * 0.3' in code:
                    analysis['weights']['volume'] = 0.3
                if 'price_score * 0.3' in code:
                    analysis['weights']['price'] = 0.3
                if 'spread_score * 0.2' in code:
                    analysis['weights']['spread'] = 0.2
                if 'range_score * 0.2' in code:
                    analysis['weights']['range'] = 0.2

                # Extract thresholds
                if 'VIPER_THRESHOLD' in code:
                    analysis['thresholds']['signal_threshold'] = 'Found in environment variables'

                analysis['implementation_status'] = 'COMPLETE'
            else:
                analysis['issues'].append("VIPER calculation function not found")
                analysis['implementation_status'] = 'INCOMPLETE'

        except Exception as e:
            analysis['issues'].append(f"Failed to analyze algorithm: {e}")
            analysis['implementation_status'] = 'ERROR'

        return analysis

    def test_score_calculation(self) -> Dict:
        """Test VIPER score calculation with real market data"""

        result = {
            'test_pairs': [],
            'scores_generated': 0,
            'high_confidence_signals': 0,
            'average_score': 0,
            'performance_metrics': {},
            'issues': []
        }

        if not self.exchange:
            result['issues'].append("Exchange connection not available")
            return result

        try:
            # Test with top 10 pairs by volume
            markets = list(self.exchange.markets.values())[:10]
            scores = []

            for market in markets:
                symbol = market['symbol']
                try:
                    # Get market data
                    ticker = self.exchange.fetch_ticker(symbol)
                    order_book = self.exchange.fetch_order_book(symbol, limit=5)

                    # Simulate VIPER calculation
                    market_data = {
                        'ticker': ticker,
                        'orderbook': order_book
                    }

                    viper_score = self.calculate_viper_score(symbol, market_data)

                    pair_result = {
                        'symbol': symbol,
                        'viper_score': viper_score,
                        'price': ticker.get('last', 0),
                        'volume': ticker.get('quoteVolume', 0),
                        'change': ticker.get('percentage', 0)
                    }

                    result['test_pairs'].append(pair_result)
                    scores.append(viper_score)

                    if viper_score >= 85:
                        result['high_confidence_signals'] += 1

                except Exception as e:
                    result['issues'].append(f"Failed to calculate score for {symbol}: {e}")

            if scores:
                result['scores_generated'] = len(scores)
                result['average_score'] = sum(scores) / len(scores)
                result['performance_metrics'] = {
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'median_score': sorted(scores)[len(scores)//2]
                }

        except Exception as e:
            result['issues'].append(f"Score calculation test failed: {e}")

        return result

    def calculate_viper_score(self, symbol: str, market_data: Dict) -> float:
        """Calculate VIPER score (replicated from signal processor)"""
        try:
            ticker = market_data.get('ticker', {})
            orderbook = market_data.get('orderbook', {})

            if not ticker:
                return 0.0

            # Volume Analysis (V)
            volume = ticker.get('quoteVolume', 0) or ticker.get('volume', 0)
            volume_score = min(volume / 1000000, 100)

            # Price Action (P)
            price_change = ticker.get('percentage', 0) or ticker.get('change', 0)
            price_score = max(0, min(100, 50 + price_change * 10))

            # External Factors (E) - Spread
            spread = abs(orderbook.get('asks', [0])[0] - orderbook.get('bids', [0])[0]) if orderbook.get('asks') and orderbook.get('bids') else 0
            spread_score = max(0, 100 - spread * 1000)

            # Range Analysis (R)
            high = ticker.get('high', 0)
            low = ticker.get('low', 0)
            current = ticker.get('last', ticker.get('close', 0))

            if high and low and current and high != low:
                range_score = ((current - low) / (high - low)) * 100
            else:
                range_score = 50

            # Calculate weighted score
            viper_score = (
                volume_score * 0.3 +
                price_score * 0.3 +
                spread_score * 0.2 +
                range_score * 0.2
            )

            return min(100, max(0, viper_score))

        except Exception as e:
            return 0.0

    def diagnose_scoring_pipeline(self) -> Dict:
        """Diagnose complete scoring pipeline"""
        print("🔬 Diagnosing Complete Scoring Pipeline...")

        pipeline = {
            'data_ingestion': {},
            'score_calculation': {},
            'signal_generation': {},
            'performance_monitoring': {},
            'issues': [],
            'recommendations': []
        }

        # Test data ingestion
        pipeline['data_ingestion'] = self.test_data_ingestion()

        # Test score calculation
        pipeline['score_calculation'] = self.test_score_calculation()

        # Test signal generation
        pipeline['signal_generation'] = self.test_signal_generation()

        # Performance monitoring
        pipeline['performance_monitoring'] = self.monitor_scoring_performance()

        return pipeline

    def test_data_ingestion(self) -> Dict:
        """Test market data ingestion"""
        result = {
            'exchange_connection': 'UNKNOWN',
            'market_data_fetching': 'UNKNOWN',
            'data_quality': {},
            'issues': []
        }

        if self.exchange:
            result['exchange_connection'] = 'SUCCESS'
            try:
                # Test market data fetching
                markets = self.exchange.loadMarkets()
                result['market_count'] = len(markets)

                # Test ticker data
                symbol = 'BTC/USDT:USDT'
                if symbol in markets:
                    ticker = self.exchange.fetch_ticker(symbol)
                    result['market_data_fetching'] = 'SUCCESS'
                    result['data_quality'] = {
                        'has_price': ticker.get('last') is not None,
                        'has_volume': ticker.get('volume') is not None,
                        'has_change': ticker.get('percentage') is not None
                    }
                else:
                    result['issues'].append("BTC/USDT pair not available")
            except Exception as e:
                result['issues'].append(f"Data ingestion test failed: {e}")
        else:
            result['exchange_connection'] = 'FAILED'
            result['issues'].append("Exchange connection not available")

        return result

    def test_signal_generation(self) -> Dict:
        """Test signal generation pipeline"""
        result = {
            'signal_processor_status': 'UNKNOWN',
            'signal_generation_test': {},
            'threshold_testing': {},
            'issues': []
        }

        # Check signal processor status
        sp_status = self.check_service_health('signal-processor', 8006)
        result['signal_processor_status'] = sp_status['status']

        if sp_status['status'] == 'HEALTHY':
            try:
                # Test signal generation endpoint if available
                response = requests.post('http://localhost:8006/generate-signal',
                                       json={'symbol': 'BTC/USDT:USDT'},
                                       timeout=10)
                if response.status_code == 200:
                    result['signal_generation_test'] = response.json()
                else:
                    result['issues'].append("Signal generation endpoint returned error")
            except Exception as e:
                result['issues'].append(f"Signal generation test failed: {e}")

        return result

    def monitor_scoring_performance(self) -> Dict:
        """Monitor scoring system performance"""
        result = {
            'calculation_speed': {},
            'memory_usage': {},
            'accuracy_metrics': {},
            'issues': []
        }

        try:
            # Test calculation speed
            start_time = time.time()
            scores = []

            for i in range(10):  # Test 10 calculations
                symbol = 'BTC/USDT:USDT'
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    order_book = self.exchange.fetch_order_book(symbol, limit=5)
                    market_data = {'ticker': ticker, 'orderbook': order_book}
                    score = self.calculate_viper_score(symbol, market_data)
                    scores.append(score)
                except Exception:
                    continue

            end_time = time.time()
            calculation_time = end_time - start_time

            if scores:
                result['calculation_speed'] = {
                    'total_time': calculation_time,
                    'average_time': calculation_time / len(scores),
                    'calculations_per_second': len(scores) / calculation_time
                }

        except Exception as e:
            result['issues'].append(f"Performance monitoring failed: {e}")

        return result

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on diagnostic findings"""
        recommendations = []

        # Check component status
        if self.diagnostic_report['components'].get('signal_processor', {}).get('service_status', {}).get('status') != 'HEALTHY':
            recommendations.append("# Tool Restart signal processor service")
            recommendations.append("# Tool Check signal processor logs for errors")

        # Check algorithm implementation
        algo_analysis = self.diagnostic_report.get('scoring_algorithm', {}).get('algorithm_analysis', {})
        if algo_analysis.get('implementation_status') != 'COMPLETE':
            recommendations.append("# Tool Complete VIPER algorithm implementation")
            recommendations.append("# Tool Add missing scoring components")

        # Check scoring performance
        perf_metrics = self.diagnostic_report.get('performance_metrics', {})
        if perf_metrics.get('calculation_speed', {}).get('calculations_per_second', 0) < 10:
            recommendations.append("⚡ Optimize scoring calculation performance")
            recommendations.append("⚡ Consider caching market data")

        # Check data quality
        data_ingestion = self.diagnostic_report.get('scoring_pipeline', {}).get('data_ingestion', {})
        if not all(data_ingestion.get('data_quality', {}).values()):
            recommendations.append("# Chart Improve market data quality")
            recommendations.append("# Chart Add data validation and fallback mechanisms")

        return recommendations

    def run_complete_diagnostic(self):
        """Run complete diagnostic suite"""
        print("# Rocket Starting Complete VIPER Scoring System Diagnostic...")

        # Component diagnosis
        self.diagnostic_report['components']['signal_processor'] = self.diagnose_signal_processor()

        # Scoring pipeline diagnosis
        self.diagnostic_report['scoring_pipeline'] = self.diagnose_scoring_pipeline()

        # Performance analysis
        self.diagnostic_report['performance_metrics'] = self.monitor_scoring_performance()

        # Generate recommendations
        self.diagnostic_report['recommendations'] = self.generate_recommendations()

        # Determine overall system status
        component_statuses = []
        for component in self.diagnostic_report['components'].values():
            if isinstance(component, dict) and 'service_status' in component:
                status = component['service_status'].get('status')
                if status:
                    component_statuses.append(status)

        if 'DOWN' in component_statuses or 'FAILED' in component_statuses:
            self.diagnostic_report['system_status'] = 'CRITICAL'
        elif 'DEGRADED' in component_statuses:
            self.diagnostic_report['system_status'] = 'DEGRADED'
        elif component_statuses and all(s == 'HEALTHY' for s in component_statuses):
            self.diagnostic_report['system_status'] = 'HEALTHY'
        else:
            self.diagnostic_report['system_status'] = 'UNKNOWN'

        return self.diagnostic_report

    def print_diagnostic_report(self):
        """Print comprehensive diagnostic report"""
        report = self.diagnostic_report


        # System Status
        status = report['system_status']
        status_color = {
            'HEALTHY': '🟢',
            'DEGRADED': '🟡',
            'CRITICAL': '🔴',
            'UNKNOWN': '⚪'
        }
        print(f"\n# Chart OVERALL SYSTEM STATUS: {status_color.get(status, '⚪')} {status}")

        # Component Status
        for component_name, component_data in report.get('components', {}).items():
            if 'service_status' in component_data:
                status = component_data['service_status']['status']
                status_icon = '# Check' if status == 'HEALTHY' else '# X' if status == 'DOWN' else '# Warning'
                print(f"  {status_icon} {component_name}: {status}")

        # Algorithm Analysis
        algo_analysis = report.get('scoring_algorithm', {}).get('algorithm_analysis', {})
        if algo_analysis:
            print(f"  📈 Implementation Status: {algo_analysis.get('implementation_status', 'UNKNOWN')}")
            for comp, present in algo_analysis.get('components', {}).items():
                icon = '# Check' if present else '# X'

            for factor, weight in algo_analysis.get('weights', {}).items():

        # Performance Metrics
        perf = report.get('performance_metrics', {})
        if perf.get('calculation_speed'):
            speed = perf['calculation_speed']
        # Scoring Pipeline
        pipeline = report.get('scoring_pipeline', {})
        if pipeline.get('score_calculation'):
            calc = pipeline['score_calculation']
            print(f"  # Target Scores Generated: {calc.get('scores_generated', 0)}")
            print(f"  🔴 High Confidence Signals: {calc.get('high_confidence_signals', 0)}")

        # Issues Found
        issues = report.get('issues', [])
        if issues:
            for issue in issues:

        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            for rec in recommendations:


def main():
    """Main diagnostic function"""
    diagnostic = VIPERScoringDiagnostic()
    diagnostic.print_header()

    try:
        # Run complete diagnostic
        report = diagnostic.run_complete_diagnostic()

        # Print detailed report
        diagnostic.print_diagnostic_report()

        # Save diagnostic report - ensure directory exists and use relative path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(diagnostic.project_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        report_file = os.path.join(reports_dir, f"scoring_system_diagnostic_{timestamp}.json")
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"💾 Detailed report saved to: {report_file}")
        except Exception as save_e:
            # Try to save in current directory as fallback
            try:
                fallback_file = f"scoring_system_diagnostic_{timestamp}.json"
                with open(fallback_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"💾 Fallback report saved to: {fallback_file}")
            except Exception as fallback_e:
                print(f"# X Fallback save also failed: {fallback_e}")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
