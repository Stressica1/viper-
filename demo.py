#!/usr/bin/env python3
"""
🎮 VIPER TRADING BOT - DEMO MODE
Test the system safely without real API keys or money!

This demo mode shows how the system works using simulated data.
Perfect for testing and learning before live trading.
"""

import os
import sys
import subprocess
import time
import random
from pathlib import Path

# Colors for better output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Print demo header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}🎮 VIPER TRADING BOT - DEMO MODE{Colors.END}")
    print(f"{Colors.BOLD}Safe testing with simulated data - NO REAL MONEY!{Colors.END}")
    print("="*60)

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def simulate_market_data():
    """Simulate market data processing"""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOGE/USDT']
    
    print(f"\n{Colors.BOLD}📊 Market Data Streaming (Simulated){Colors.END}")
    print("-" * 40)
    
    for i in range(10):
        symbol = random.choice(symbols)
        price = random.uniform(0.1, 50000)
        change = random.uniform(-5, 5)
        
        if change > 0:
            color = Colors.GREEN
            arrow = "↗"
        else:
            color = Colors.RED  
            arrow = "↘"
        
        print(f"{symbol:12} ${price:8.2f} {color}{arrow} {change:+5.2f}%{Colors.END}")
        time.sleep(0.5)

def simulate_signal_generation():
    """Simulate signal generation"""
    print(f"\n{Colors.BOLD}🎯 VIPER Signal Generation (Simulated){Colors.END}")
    print("-" * 40)
    
    signals = [
        ("BTC/USDT", "BUY", 87, "Strong bullish momentum"),
        ("ETH/USDT", "BUY", 82, "Breaking resistance level"),
        ("SOL/USDT", "SELL", 78, "Overbought conditions"),
        ("ADA/USDT", "BUY", 85, "Trend reversal pattern"),
    ]
    
    for symbol, action, score, reason in signals:
        if action == "BUY":
            color = Colors.GREEN
        else:
            color = Colors.RED
        
        print(f"{symbol:12} {color}{action:4}{Colors.END} Score: {score}/100 - {reason}")
        time.sleep(1)

def simulate_trading():
    """Simulate trading execution"""
    print(f"\n{Colors.BOLD}💰 Trade Execution (Simulated){Colors.END}")
    print("-" * 40)
    
    trades = [
        ("BTC/USDT", "BUY", 0.05, 45000.00, "Market entry"),
        ("ETH/USDT", "BUY", 0.8, 2500.00, "Breakout trade"),
        ("SOL/USDT", "SELL", 10.0, 95.50, "Take profit"),
    ]
    
    balance = 10000.00  # Simulated starting balance
    
    print(f"Starting Balance: ${balance:,.2f}")
    print()
    
    for symbol, action, qty, price, note in trades:
        trade_value = qty * price
        
        if action == "BUY":
            balance -= trade_value
            color = Colors.GREEN
            print(f"{color}📈 {action} {qty} {symbol} @ ${price:.2f}{Colors.END}")
        else:
            balance += trade_value  
            color = Colors.RED
            print(f"{color}📉 {action} {qty} {symbol} @ ${price:.2f}{Colors.END}")
        
        print(f"   Trade Value: ${trade_value:,.2f}")
        print(f"   Note: {note}")
        print(f"   New Balance: ${balance:,.2f}")
        print()
        time.sleep(1.5)
    
    profit = balance - 10000
    if profit > 0:
        print(f"{Colors.GREEN}💰 Profit: +${profit:,.2f} ({profit/10000*100:.1f}%){Colors.END}")
    else:
        print(f"{Colors.RED}📉 Loss: ${profit:,.2f} ({profit/10000*100:.1f}%){Colors.END}")

def simulate_risk_management():
    """Simulate risk management"""
    print(f"\n{Colors.BOLD}🛡️  Risk Management (Simulated){Colors.END}")
    print("-" * 40)
    
    print_info("Position limit: 15 max positions")
    print_info("Risk per trade: 2% of account balance")
    print_info("Daily loss limit: 3% of account")
    print_info("Stop-loss: Automatic based on ATR")
    print_info("Take-profit: 2:1 risk/reward ratio")
    
    # Simulate position monitoring
    positions = [
        ("BTC/USDT", "LONG", 0.05, 45000, 43200, "+4.2%"),
        ("ETH/USDT", "LONG", 0.8, 2500, 2580, "+3.2%"), 
        ("SOL/USDT", "SHORT", 10, 95.5, 92.1, "+3.6%"),
    ]
    
    print("\nActive Positions:")
    for symbol, side, qty, entry, current, pnl in positions:
        if "+" in pnl:
            color = Colors.GREEN
        else:
            color = Colors.RED
        print(f"  {symbol:12} {side:5} {qty:6} @ ${entry:8.2f} → ${current:8.2f} {color}{pnl}{Colors.END}")

def run_demo():
    """Run the complete demo"""
    print_header()
    
    print(f"\n{Colors.YELLOW}This demo shows how the VIPER trading system works.{Colors.END}")
    print(f"{Colors.YELLOW}No real money or API keys are used - everything is simulated!{Colors.END}")
    
    print(f"\n{Colors.BLUE}Press Enter to start demo or Ctrl+C to exit...{Colors.END}")
    try:
        input()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo cancelled{Colors.END}")
        return
    
    # Run demo components
    print_success("🚀 Starting VIPER Demo System...")
    time.sleep(2)
    
    print_success("📊 Market data streaming started")
    time.sleep(1)
    
    print_success("🤖 VIPER algorithm initialized") 
    time.sleep(1)
    
    print_success("🛡️  Risk management active")
    time.sleep(1)
    
    print_success("💱 Exchange connection established (simulated)")
    time.sleep(1)
    
    # Show system components working
    simulate_market_data()
    simulate_signal_generation()
    simulate_risk_management()
    simulate_trading()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"{Colors.BOLD}{Colors.GREEN}🎉 DEMO COMPLETED!{Colors.END}")
    print(f"\n{Colors.BLUE}What you just saw:{Colors.END}")
    print(f"✅ Real-time market data processing")
    print(f"✅ VIPER algorithm signal generation") 
    print(f"✅ Automated trade execution")
    print(f"✅ Risk management and position monitoring")
    print(f"✅ Performance tracking and reporting")
    
    print(f"\n{Colors.YELLOW}Ready for live trading?{Colors.END}")
    print(f"1. Add your Bitget API keys to .env file")
    print(f"2. Run: python start_trading.py")
    print(f"3. Start making real money! 💰")
    
    print(f"\n{Colors.RED}Remember: Live trading uses real money!{Colors.END}")

def main():
    """Main demo function"""
    try:
        run_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo stopped by user{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Demo error: {e}{Colors.END}")

if __name__ == "__main__":
    main()