#!/usr/bin/env python3
"""
🚀 VIPER Trading System - Simple Web Dashboard
Minimal web interface to monitor trading signals and market data

Features:
- Real-time signal display
- Market data monitoring
- Simple performance tracking
- No Docker required
"""

import os
import json
import time
import requests
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from threading import Thread
from typing import Dict, List, Optional, Any

# Global variables for tracking
trading_signals = []
market_data = {}
system_stats = {
    'signals_generated': 0,
    'scans_completed': 0,
    'last_update': None,
    'system_status': 'Running'
}

app = Flask(__name__)

class SimpleTradingMonitor:
    """Simple trading monitor that fetches data and generates signals"""

    def __init__(self):
        self.symbols = self.fetch_all_trading_pairs()
        self.base_url = "https://api.bitget.com"
        self.is_running = False

    def fetch_all_trading_pairs(self):
        """Fetch ALL available trading pairs from Bitget"""
        try:
            spot_instruments_url = f"{self.base_url}/api/v2/spot/public/symbols"

            response = requests.get(spot_instruments_url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get('code') == '00000' and data.get('data'):
                # Get all USDT pairs that are trading
                all_pairs = []
                for symbol_data in data['data']:
                    if (symbol_data.get('quoteCoin') == 'USDT' and
                        symbol_data.get('status') == 'online' and
                        symbol_data.get('minTradeAmount', 0) > 0):

                        symbol_name = f"{symbol_data.get('baseCoin')}/USDT:USDT"
                        all_pairs.append(symbol_name)

                print(f"📊 Dashboard monitoring {len(all_pairs)} USDT trading pairs")

                # Limit to top 100 most liquid pairs for dashboard performance
                if len(all_pairs) > 100:
                    print(f"📊 Limiting dashboard to top 100 pairs (found {len(all_pairs)} total)")
                    all_pairs = all_pairs[:100]

                return sorted(all_pairs)

            return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT']  # Fallback

        except Exception as e:
            print(f"❌ Error fetching trading pairs: {e}")
            return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT']  # Fallback

    def fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch current market data"""
        try:
            ticker_url = f"{self.base_url}/api/v2/spot/market/tickers"
            response = requests.get(ticker_url, timeout=5)
            response.raise_for_status()

            data = response.json()
            if data.get('code') == '00000' and data.get('data'):
                for ticker in data['data']:
                    if ticker.get('symbol') == symbol.replace('/', '').replace(':USDT', ''):
                        return {
                            'symbol': symbol,
                            'price': float(ticker.get('last', 0)),
                            'high': float(ticker.get('high24h', 0)),
                            'low': float(ticker.get('low24h', 0)),
                            'volume': float(ticker.get('volume24h', 0)),
                            'price_change': float(ticker.get('change', 0)),
                            'timestamp': datetime.now().isoformat()
                        }
            return None
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def calculate_viper_score(self, market_data: Dict) -> float:
        """Calculate VIPER score"""
        try:
            price_change = market_data.get('price_change', 0)
            volume = market_data.get('volume', 0)
            high_low_range = market_data.get('high', 1) - market_data.get('low', 0)

            volume_score = min(volume / 1000000, 100)
            price_score = abs(price_change) * 100
            range_score = (high_low_range / market_data.get('price', 1)) * 100

            viper_score = (volume_score * 0.4) + (price_score * 0.3) + (range_score * 0.3)
            return min(viper_score, 100)
        except:
            return 0

    def generate_signal(self, symbol: str, viper_score: float, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal"""
        try:
            if viper_score >= 85:  # VIPER threshold
                price_change = market_data.get('price_change', 0)

                signal = None
                if price_change > 0.5:
                    signal = "LONG"
                elif price_change < -0.5:
                    signal = "SHORT"

                if signal:
                    return {
                        'id': len(trading_signals) + 1,
                        'symbol': symbol,
                        'signal': signal,
                        'viper_score': viper_score,
                        'price': market_data.get('price', 0),
                        'price_change': market_data.get('price_change', 0),
                        'volume': market_data.get('volume', 0),
                        'timestamp': datetime.now().isoformat(),
                        'confidence': min(viper_score / 100, 1.0),
                        'stop_loss': market_data.get('price', 0) * (0.98 if signal == "LONG" else 1.02),
                        'take_profit': market_data.get('price', 0) * (1.03 if signal == "LONG" else 0.97)
                    }
            return None
        except:
            return None

    def run_monitoring_loop(self):
        """Main monitoring loop"""
        self.is_running = True
        print("🚀 VIPER Trading Monitor Started")

        while self.is_running:
            try:
                # Update system stats
                system_stats['scans_completed'] += 1
                system_stats['last_update'] = datetime.now().isoformat()

                # Fetch market data for all symbols
                for symbol in self.symbols:
                    data = self.fetch_market_data(symbol)
                    if data:
                        market_data[symbol] = data

                        # Calculate VIPER score
                        viper_score = self.calculate_viper_score(data)

                        # Check for trading signals
                        signal = self.generate_signal(symbol, viper_score, data)
                        if signal:
                            trading_signals.append(signal)
                            system_stats['signals_generated'] += 1

                            print(f"🚨 SIGNAL GENERATED: {signal['symbol']} {signal['signal']} "
                                  f"(VIPER: {signal['viper_score']:.1f})")

                time.sleep(30)  # Scan every 30 seconds

            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(10)

# HTML Template for the dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🚀 VIPER Trading Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .signal { background: #e8f5e8; border-left: 5px solid #4CAF50; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .signal.long { background: #e8f5e8; border-left-color: #4CAF50; }
        .signal.short { background: #ffebee; border-left-color: #f44336; }
        .market-data { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
        .refresh-btn { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        .refresh-btn:hover { background: #5a6fd8; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 VIPER Trading Dashboard</h1>
            <p>Real-time signal generation and market monitoring</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <h3>📊 Signals Generated</h3>
                <div style="font-size: 2em; color: #4CAF50;" id="signals-count">{{ signals_generated }}</div>
            </div>
            <div class="stat-card">
                <h3>🔍 Market Scans</h3>
                <div style="font-size: 2em; color: #2196F3;" id="scans-count">{{ scans_completed }}</div>
            </div>
            <div class="stat-card">
                <h3>📈 System Status</h3>
                <div style="font-size: 1.2em; color: #4CAF50;" id="system-status">{{ system_status }}</div>
            </div>
            <div class="stat-card">
                <h3>🕒 Last Update</h3>
                <div style="font-size: 0.9em; color: #666;" id="last-update">{{ last_update or 'Never' }}</div>
            </div>
        </div>

        <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>

        <div class="market-data">
            <h2>📊 Current Market Data</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Price</th>
                        <th>24h Change</th>
                        <th>Volume</th>
                        <th>VIPER Score</th>
                    </tr>
                </thead>
                <tbody id="market-table">
                    <!-- Market data will be inserted here -->
                </tbody>
            </table>
        </div>

        <div class="market-data">
            <h2>🚨 Trading Signals</h2>
            <div id="signals-container">
                <!-- Trading signals will be inserted here -->
            </div>
        </div>
    </div>

    <script>
        function refreshData() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    updateStats(data.system_stats);
                    updateMarketData(data.market_data);
                    updateSignals(data.signals);
                })
                .catch(error => console.error('Error refreshing data:', error));
        }

        function updateStats(stats) {
            document.getElementById('signals-count').textContent = stats.signals_generated;
            document.getElementById('scans-count').textContent = stats.scans_completed;
            document.getElementById('system-status').textContent = stats.system_status;
            document.getElementById('last-update').textContent = stats.last_update ?
                new Date(stats.last_update).toLocaleString() : 'Never';
        }

        function updateMarketData(marketData) {
            const table = document.getElementById('market-table');
            table.innerHTML = '';

            Object.values(marketData).forEach(data => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${data.symbol}</td>
                    <td>$${parseFloat(data.price).toFixed(4)}</td>
                    <td style="color: ${data.price_change >= 0 ? 'green' : 'red'};">${data.price_change >= 0 ? '+' : ''}${data.price_change.toFixed(2)}%</td>
                    <td>${parseFloat(data.volume).toLocaleString()}</td>
                    <td>${calculateViperScore(data).toFixed(1)}</td>
                `;
                table.appendChild(row);
            });
        }

        function updateSignals(signals) {
            const container = document.getElementById('signals-container');
            container.innerHTML = '';

            if (signals.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #666;">No trading signals generated yet. Waiting for market opportunities...</p>';
                return;
            }

            signals.slice(-5).reverse().forEach(signal => {  // Show last 5 signals
                const signalDiv = document.createElement('div');
                signalDiv.className = `signal ${signal.signal.toLowerCase()}`;
                signalDiv.innerHTML = `
                    <h3>🚨 Signal #${signal.id}: ${signal.symbol} ${signal.signal}</h3>
                    <p><strong>VIPER Score:</strong> ${signal.viper_score.toFixed(1)}/100 | <strong>Confidence:</strong> ${(signal.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Price:</strong> $${parseFloat(signal.price).toFixed(4)} | <strong>Change:</strong> ${signal.price_change >= 0 ? '+' : ''}${signal.price_change.toFixed(2)}%</p>
                    <p><strong>Volume:</strong> ${parseFloat(signal.volume).toLocaleString()}</p>
                    <p><strong>Stop Loss:</strong> $${parseFloat(signal.stop_loss).toFixed(4)} | <strong>Take Profit:</strong> $${parseFloat(signal.take_profit).toFixed(4)}</p>
                    <p><small>🕒 ${new Date(signal.timestamp).toLocaleString()}</small></p>
                `;
                container.appendChild(signalDiv);
            });
        }

        function calculateViperScore(data) {
            const volumeScore = Math.min(data.volume / 1000000, 100);
            const priceScore = Math.abs(data.price_change) * 100;
            const rangeScore = ((data.high - data.low) / data.price) * 100;
            return Math.min((volumeScore * 0.4) + (priceScore * 0.3) + (rangeScore * 0.3), 100);
        }

        // Auto-refresh every 10 seconds
        setInterval(refreshData, 10000);

        // Initial load
        refreshData();
    </script>
</body>
</html>
"""

monitor = SimpleTradingMonitor()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE,
                                signals_generated=system_stats['signals_generated'],
                                scans_completed=system_stats['scans_completed'],
                                system_status=system_stats['system_status'],
                                last_update=system_stats['last_update'])

@app.route('/api/data')
def get_data():
    """API endpoint for dashboard data"""
    return jsonify({
        'system_stats': system_stats,
        'market_data': market_data,
        'signals': trading_signals[-10:]  # Last 10 signals
    })

def start_monitoring():
    """Start the monitoring thread"""
    monitor_thread = Thread(target=monitor.run_monitoring_loop, daemon=True)
    monitor_thread.start()

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🚀 VIPER TRADING DASHBOARD - LIVE MONITORING                                 ║
║ 🔥 Real-time Signals | 📊 Market Data | 🎯 Performance Tracking                ║
║ 🌐 Web Interface | 📈 Live Updates | 🚨 Instant Alerts                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    print("📊 Starting monitoring system...")
    start_monitoring()

    print("🌐 Starting web dashboard...")
    print("📱 Access your dashboard at: http://localhost:5000")
    print("🔄 Dashboard auto-refreshes every 10 seconds")
    print("🚨 Trading signals will appear here when conditions are met")
    print("=" * 80)

    app.run(host='0.0.0.0', port=5000, debug=False)
