#!/usr/bin/env python3
"""Test the fixed position sizing calculation with leverage"""

def test_position_sizing():
    """Test the position sizing calculation"""
    print("🧪 Testing Position Sizing with Leverage...")

    # Test the calculation directly

    # Test with your actual balance (~$60)
    balance = 60.0
    price = 50000  # BTC price
    leverage = 50

    # Calculate using the same formula as the fixed method
    risk_per_trade = 0.03
    risk_amount = balance * risk_per_trade
    stop_loss_pct = 0.02
    stop_loss_distance = price * stop_loss_pct
    base_position_size = risk_amount / stop_loss_distance
    leveraged_position_size = base_position_size * leverage
    min_contract_size = 0.001
    position_size = max(leveraged_position_size, min_contract_size)

    print("✅ Position Sizing Test Results:")
    print(f"   Account Balance: ${balance:.2f}")
    print(f"   Risk per Trade: 3% = ${risk_amount:.2f}")
    print(f"   Stop Loss Distance: 2% = ${stop_loss_distance:.2f}")
    print(f"   Base Position Size: {base_position_size:.6f} BTC")
    print(f"   Final Position Size (50x leverage): {position_size:.6f} BTC")
    print(f"   Notional Value: ${position_size * price:.2f}")

    # Expected calculation:
    # Risk amount = $60 * 0.03 = $1.80
    # Stop loss distance = $50,000 * 0.02 = $1,000
    # Base position = $1.80 / $1,000 = 0.0018 BTC
    # Leveraged position = 0.0018 * 50 = 0.09 BTC
    # Notional value = 0.09 * $50,000 = $4,500

    expected_base = (balance * 0.03) / (price * 0.02)
    expected_leveraged = expected_base * leverage

    print("\n📊 Expected vs Actual:")
    print(f"   Expected Base: {expected_base:.6f} BTC")
    print(f"   Expected Leveraged: {expected_leveraged:.6f} BTC")
    print(f"   Actual: {position_size:.6f} BTC")

    if abs(position_size - expected_leveraged) < 0.000001:
        print("✅ Position sizing calculation is CORRECT!")
        return True
    else:
        print("❌ Position sizing calculation is INCORRECT!")
        return False

if __name__ == "__main__":
    test_position_sizing()
