#!/usr/bin/env python3
"""
🚀 VIPER Standalone Trading Component - Enhanced Visual Demo
Static 3x2x3 Panel Layout with Header and Footer

Panel Layout:
Header: System Status & Time
┌─────────────┬─────────────┬─────────────┐
│ Panel 1     │ Panel 2     │ Panel 3     │
│ Market Scan │ VIPER Score │ Signals     │
├─────────────┴─────────────┴─────────────┤
│ Panel 4                 │ Panel 5       │
│ Active Positions        │ Trading Stats │
├─────────────┬───────────────────────────┤
│ Panel 6     │ Panel 7     │ Panel 8     │
│ Recent      │ Risk Mgmt   │ System      │
│ Trades      │             │ Logs        │
└─────────────┴─────────────┴─────────────┘
Footer: Controls & Status
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock, patch
from dataclasses import dataclass
import threading


@dataclass
class PanelData:
    """Data structure for panel information"""
    title: str
    content: List[str]
    width: int
    height: int


class EnhancedVIPERDisplay:
    """Enhanced display with static 3x2x3 panel layout"""
    
    def __init__(self):
        # Panel dimensions (characters)
        self.panel_width = 25
        self.panel_height = 8
        self.display_width = 80
        
        # Sample data for demonstration
        self.trading_pairs = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'BNB/USDT:USDT', 'ADA/USDT:USDT', 'DOT/USDT:USDT',
            'MATIC/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT'
        ]
        
        # Demo state variables
        self.active_positions = {}
        self.trading_stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
        self.recent_trades = []
        self.system_logs = []
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def create_border_line(self, positions: str) -> str:
        """Create border lines for panel layout
        
        Args:
            positions: String indicating line type:
                'top': ┌─┬─┬─┐
                'mid1': ├─┴─┴─┤ 
                'mid2': ├─┬───┤
                'bot': └─┴─┴─┘
        """
        if positions == 'top':
            return "┌─────────────────────────┬─────────────────────────┬─────────────────────────┐"
        elif positions == 'mid1':
            return "├─────────────────────────┴─────────────────────────┴─────────────────────────┤"
        elif positions == 'mid2':
            return "├─────────────────────────┬───────────────────────────────────────────────────┤"
        elif positions == 'mid3':
            return "├─────────────────────────┼─────────────────────────┬─────────────────────────┤"
        elif positions == 'bot':
            return "└─────────────────────────┴─────────────────────────┴─────────────────────────┘"
        else:
            return "│" + " " * (self.display_width - 2) + "│"
    
    def format_panel_line(self, left: str = "", center: str = "", right: str = "", 
                         wide_left: str = "", narrow_right: str = "", layout: str = "three") -> str:
        """Format a line with appropriate panel layout"""
        if layout == "three":
            # Three equal panels
            left_padded = f"│ {left:<23} "
            center_padded = f"│ {center:<23} "
            right_padded = f"│ {right:<23} │"
            return left_padded + center_padded + right_padded
        elif layout == "two":
            # Two panels (wide left, narrow right)
            left_padded = f"│ {wide_left:<47} "
            right_padded = f"│ {narrow_right:<23} │"
            return left_padded + right_padded
        else:
            return f"│ {left:<{self.display_width-4}} │"
    
    def generate_market_scan_data(self) -> Dict:
        """Generate mock market scanning data"""
        scan_data = {}
        for symbol in self.trading_pairs:
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
            price_change = random.uniform(-8, 8)
            current_price = base_price * (1 + price_change / 100)
            
            scan_data[symbol] = {
                'price': round(current_price, 4),
                'change': round(price_change, 2),
                'volume': random.uniform(500000, 5000000),
                'viper_score': random.uniform(20, 95)
            }
        
        return scan_data
    
    def generate_signals(self, scan_data: Dict) -> List[Dict]:
        """Generate trading signals from scan data"""
        signals = []
        for symbol, data in scan_data.items():
            if data['viper_score'] > 80:  # High score threshold
                signal_type = "LONG" if data['change'] > 0 else "SHORT"
                signals.append({
                    'symbol': symbol,
                    'type': signal_type,
                    'score': data['viper_score'],
                    'confidence': random.uniform(0.7, 0.95)
                })
        return signals
    
    def update_positions(self, signals: List[Dict]):
        """Update active positions based on signals"""
        # Add new positions (limit to 3)
        for signal in signals[:3]:
            if len(self.active_positions) < 3:
                self.active_positions[signal['symbol']] = {
                    'side': signal['type'],
                    'entry_price': random.uniform(45000, 55000),
                    'current_price': random.uniform(45000, 55000),
                    'size': round(random.uniform(0.001, 0.1), 6),
                    'pnl_pct': round(random.uniform(-3, 6), 2),
                    'time': datetime.now().strftime("%H:%M:%S")
                }
        
        # Update existing positions
        for symbol, position in self.active_positions.items():
            # Simulate price movement
            position['current_price'] *= (1 + random.uniform(-0.02, 0.02))
            # Recalculate P&L
            if position['side'] == 'LONG':
                position['pnl_pct'] = ((position['current_price'] - position['entry_price']) / position['entry_price']) * 100
            else:
                position['pnl_pct'] = ((position['entry_price'] - position['current_price']) / position['entry_price']) * 100
            position['pnl_pct'] = round(position['pnl_pct'], 2)
    
    def update_stats(self):
        """Update trading statistics"""
        if random.random() < 0.3:  # 30% chance to close a position
            if self.active_positions:
                symbol = list(self.active_positions.keys())[0]
                position = self.active_positions.pop(symbol)
                
                # Add to recent trades
                self.recent_trades.insert(0, {
                    'symbol': symbol,
                    'side': position['side'],
                    'pnl': position['pnl_pct'],
                    'time': datetime.now().strftime("%H:%M:%S")
                })
                
                # Keep only last 5 trades
                self.recent_trades = self.recent_trades[:5]
                
                # Update stats
                self.trading_stats['total_trades'] += 1
                if position['pnl_pct'] > 0:
                    self.trading_stats['profitable_trades'] += 1
                
                self.trading_stats['total_pnl'] += position['pnl_pct']
                
                if self.trading_stats['total_trades'] > 0:
                    self.trading_stats['win_rate'] = (
                        self.trading_stats['profitable_trades'] / 
                        self.trading_stats['total_trades'] * 100
                    )
    
    def update_logs(self):
        """Update system logs"""
        log_messages = [
            "Market scan completed",
            "Signal generated for BTC",
            "Position updated",
            "Risk check passed",
            "TP/SL monitoring active",
            "API rate limit OK",
            "Balance updated",
            "System health: OK"
        ]
        
        if random.random() < 0.7:  # 70% chance to add log
            new_log = {
                'time': datetime.now().strftime("%H:%M:%S"),
                'message': random.choice(log_messages)
            }
            self.system_logs.insert(0, new_log)
            self.system_logs = self.system_logs[:5]  # Keep last 5 logs
    
    def render_display(self, scan_data: Dict, signals: List[Dict]):
        """Render the complete display with 3x2x3 panel layout"""
        
        # Header
        header_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"🚀 VIPER TRADING SYSTEM - LIVE DEMO │ {header_time} │ STATUS: ACTIVE"
        
        print("=" * self.display_width)
        print(f" {header:^{self.display_width-2}} ")
        print("=" * self.display_width)
        print()
        
        # Top border
        print(self.create_border_line('top'))
        
        # Panel headers (row 1)
        print(self.format_panel_line(
            "🔍 MARKET SCANNER", 
            "📊 VIPER SCORES", 
            "🎯 SIGNALS", 
            layout="three"
        ))
        
        # Panel content (rows 2-7)
        for i in range(6):
            if i == 0:
                # First content row
                scan_items = list(scan_data.items())[:3]
                scan_line = " | ".join([f"{s[0].split('/')[0]}: {s[1]['change']:+.1f}%" for s in scan_items])
                
                # Top 3 scores
                top_scores = sorted(scan_data.items(), key=lambda x: x[1]['viper_score'], reverse=True)[:3]
                score_line = " | ".join([f"{s[0].split('/')[0]}: {s[1]['viper_score']:.0f}" for s in top_scores])
                
                # Active signals
                signal_line = f"Active: {len(signals)}" if signals else "No signals"
                
                print(self.format_panel_line(
                    scan_line[:23], 
                    score_line[:23], 
                    signal_line[:23], 
                    layout="three"
                ))
            elif i < 5:
                # Remaining content rows
                if i <= len(scan_data) // 3:
                    start_idx = i * 3
                    scan_items = list(scan_data.items())[start_idx:start_idx+3]
                    if scan_items:
                        scan_line = " | ".join([f"{s[0].split('/')[0]}: ${s[1]['price']:.0f}" for s in scan_items])
                    else:
                        scan_line = ""
                else:
                    scan_line = ""
                
                # VIPER score details
                if i <= len(top_scores):
                    if i == 1:
                        score_line = "Volume: High Activity"
                    elif i == 2:
                        score_line = "Price: Strong Momentum"  
                    elif i == 3:
                        score_line = "External: Positive"
                    elif i == 4:
                        score_line = "Range: Volatile"
                    else:
                        score_line = ""
                else:
                    score_line = ""
                
                # Signal details
                if signals and i <= len(signals):
                    if i-1 < len(signals):
                        signal = signals[i-1]
                        signal_line = f"{signal['symbol'].split('/')[0]}: {signal['type']}"
                    else:
                        signal_line = ""
                else:
                    signal_line = ""
                
                print(self.format_panel_line(
                    scan_line[:23], 
                    score_line[:23], 
                    signal_line[:23], 
                    layout="three"
                ))
            else:
                # Empty content row
                print(self.format_panel_line("", "", "", layout="three"))
        
        # Middle border 1 (transition to 2-panel layout)
        print(self.create_border_line('mid1'))
        
        # Two-panel section header
        print(self.format_panel_line(
            wide_left="📈 ACTIVE POSITIONS", 
            narrow_right="📊 TRADING STATS", 
            layout="two"
        ))
        
        # Two-panel content (rows)
        for i in range(6):
            if i == 0:
                # First content row
                positions_line = f"Count: {len(self.active_positions)}/3"
                stats_line = f"Win Rate: {self.trading_stats['win_rate']:.1f}%"
            elif i == 1:
                # Position details or stats
                if self.active_positions:
                    first_pos = list(self.active_positions.values())[0]
                    positions_line = f"Latest: {first_pos['pnl_pct']:+.2f}% P&L"
                else:
                    positions_line = "No active positions"
                stats_line = f"Total P&L: {self.trading_stats['total_pnl']:+.2f}%"
            elif i < len(self.active_positions) + 2:
                # Show position details
                pos_items = list(self.active_positions.items())
                if i-2 < len(pos_items):
                    symbol, pos = pos_items[i-2]
                    pnl_icon = "🟢" if pos['pnl_pct'] > 0 else "🔴" if pos['pnl_pct'] < 0 else "⚪"
                    positions_line = f"{pnl_icon} {symbol.split('/')[0]}: {pos['pnl_pct']:+.2f}%"
                else:
                    positions_line = ""
                stats_line = f"Trades: {self.trading_stats['total_trades']}"
            else:
                positions_line = ""
                stats_line = ""
            
            print(self.format_panel_line(
                wide_left=positions_line, 
                narrow_right=stats_line, 
                layout="two"
            ))
        
        # Middle border 2 (transition back to 3-panel layout)
        print(self.create_border_line('mid2'))
        
        # Bottom three-panel section header
        print(self.format_panel_line(
            "📋 RECENT TRADES", 
            "🛡️ RISK MANAGEMENT", 
            "📝 SYSTEM LOGS", 
            layout="three"
        ))
        
        # Bottom three-panel content
        for i in range(6):
            if i == 0:
                # First content row
                trade_line = f"Count: {len(self.recent_trades)}"
                risk_line = "Risk/Trade: 2.0%"
                log_line = f"Events: {len(self.system_logs)}"
            elif i <= len(self.recent_trades) and self.recent_trades:
                # Show recent trades
                if i-1 < len(self.recent_trades):
                    trade = self.recent_trades[i-1]
                    pnl_icon = "🟢" if trade['pnl'] > 0 else "🔴"
                    trade_line = f"{pnl_icon} {trade['symbol'].split('/')[0]}: {trade['pnl']:+.1f}%"
                else:
                    trade_line = ""
                
                # Risk management info
                if i == 1:
                    risk_line = "Max Positions: 3"
                elif i == 2:
                    risk_line = "Stop Loss: 2%"
                elif i == 3:
                    risk_line = "Take Profit: 4%"
                elif i == 4:
                    risk_line = "Balance: $10,000"
                else:
                    risk_line = "Status: Active"
                
                # System logs
                if i-1 < len(self.system_logs) and self.system_logs:
                    log = self.system_logs[i-1]
                    log_line = f"{log['time']}: {log['message'][:15]}"
                else:
                    log_line = ""
            else:
                trade_line = ""
                risk_line = ""
                log_line = ""
            
            print(self.format_panel_line(
                trade_line[:23], 
                risk_line[:23], 
                log_line[:23], 
                layout="three"
            ))
        
        # Bottom border
        print(self.create_border_line('bot'))
        
        # Footer
        footer = "🔧 CONTROLS: Ctrl+C to Stop │ ⚡ SCAN INTERVAL: 5s │ 💰 DEMO MODE"
        print()
        print("=" * self.display_width)
        print(f" {footer:^{self.display_width-2}} ")
        print("=" * self.display_width)

    def run_demo(self):
        """Run the enhanced visual demo"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║            🚀 VIPER ENHANCED VISUAL DEMO - 3x2x3 PANEL LAYOUT               ║
║                     Static Display with Live Data Simulation                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        
        print("🎬 Starting enhanced visual demo...")
        print("📱 Display will refresh every 5 seconds")
        print("🛑 Press Ctrl+C to stop the demo")
        print()
        
        input("Press Enter to start the demo...")
        
        try:
            cycle = 0
            while True:
                cycle += 1
                
                # Clear screen for clean display
                self.clear_screen()
                
                # Generate fresh data
                scan_data = self.generate_market_scan_data()
                signals = self.generate_signals(scan_data)
                
                # Update positions and stats
                self.update_positions(signals)
                self.update_stats()
                self.update_logs()
                
                # Render the complete display
                self.render_display(scan_data, signals)
                
                # Show cycle info
                print(f"\n🔄 Demo Cycle: {cycle} │ Runtime: {cycle * 5}s")
                
                # Wait before next update
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Demo stopped by user")
            print("✅ Thanks for trying the VIPER Enhanced Visual Demo!")


def main():
    """Main entry point for enhanced demo"""
    try:
        demo = EnhancedVIPERDisplay()
        demo.run_demo()
        return 0
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())