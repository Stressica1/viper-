# Fixed Margin Position Sizing Implementation

## Overview

This implementation changes the VIPER V2 Risk-Optimized Trading Job from a percentage-based risk system to a fixed margin system as specified in the requirements.

## Changes Made

### Before (Percentage-based system):
- Used 2% of account balance as risk per trade
- Position size varied significantly based on account balance
- Example: $10,000 balance → $200 risk → ~$10,000 margin with leverage

### After (Fixed margin system):
- Uses fixed $2 margin per position
- With 50x leverage → exactly $100 notional value per position
- Consistent position sizing regardless of account balance

## Key Benefits

1. **Consistency**: Every position uses exactly $2 margin and $100 notional value
2. **Simplicity**: No complex percentage calculations based on account balance
3. **Predictability**: Position sizes are directly determined by asset price
4. **Risk Control**: Maximum risk per position is fixed at $2

## Technical Implementation

### Position Sizing Formula:
```
fixed_margin = $2
leverage = 50x
notional_value = fixed_margin × leverage = $2 × 50 = $100
position_size = notional_value ÷ asset_price = $100 ÷ price
```

### Examples:
- **BTC at $50,000**: Position size = 100/50000 = 0.002 BTC
- **ETH at $3,000**: Position size = 100/3000 = 0.0333 ETH
- **ADA at $1**: Position size = 100/1 = 100 ADA

## Files Modified

1. **`src/viper/execution/v2_risk_optimized_trading_job.py`**
   - Updated `calculate_v2_position_size()` method
   - Added `fixed_margin_per_position` parameter
   - Updated logging and position tracking
   - Modified status reporting

2. **`src/viper/risk/enhanced_risk_manager.py`**
   - Added fixed margin mode support
   - New `_calculate_fixed_margin_position_size()` method
   - Enhanced compatibility with main trading system

## Testing

All tests pass:
- ✅ Fixed $2 margin per position
- ✅ Exactly $100 notional value with 50x leverage
- ✅ Consistent across all price levels
- ✅ Proper integration with enhanced risk manager
- ✅ Backwards compatibility maintained

## Deployment Notes

The system maintains backwards compatibility and will fall back to basic fixed margin calculation if enhanced components are not available. All existing functionality remains intact while providing the new fixed margin benefits.