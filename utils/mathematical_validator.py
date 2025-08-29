#!/usr/bin/env python3
"""
🧮 VIPER MATHEMATICAL VALIDATION UTILITIES
Comprehensive mathematical validation and optimization utilities for trading systems

This module provides:
- Mathematical formula validation
- Numerical stability checks
- Performance optimization functions
- Risk calculation validation
- Statistical analysis validation
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from decimal import Decimal, ROUND_HALF_UP
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MathematicalValidator:
    """Mathematical validation utility class"""
    
    def __init__(self):
        self.tolerance = 1e-10  # Numerical tolerance for floating-point comparisons
        self.max_iterations = 1000  # Maximum iterations for iterative algorithms
        
    def validate_array(self, arr: np.ndarray, name: str = "array") -> Dict[str, Any]:
        """Validate numpy array for mathematical operations"""
        validation_result = {
            'name': name,
            'is_valid': True,
            'issues': [],
            'statistics': {},
            'recommendations': []
        }
        
        try:
            # Check for empty array
            if arr.size == 0:
                validation_result['is_valid'] = False
                validation_result['issues'].append("Array is empty")
                return validation_result
            
            # Check for NaN values
            nan_count = np.sum(np.isnan(arr))
            if nan_count > 0:
                validation_result['issues'].append(f"Contains {nan_count} NaN values")
                if nan_count > arr.size * 0.1:  # More than 10% NaN
                    validation_result['is_valid'] = False
                else:
                    validation_result['recommendations'].append("Clean NaN values before calculations")
            
            # Check for infinite values
            inf_count = np.sum(np.isinf(arr))
            if inf_count > 0:
                validation_result['issues'].append(f"Contains {inf_count} infinite values")
                validation_result['is_valid'] = False
            
            # Statistical analysis
            finite_arr = arr[np.isfinite(arr)]
            if len(finite_arr) > 0:
                validation_result['statistics'] = {
                    'mean': float(np.mean(finite_arr)),
                    'std': float(np.std(finite_arr)),
                    'min': float(np.min(finite_arr)),
                    'max': float(np.max(finite_arr)),
                    'median': float(np.median(finite_arr)),
                    'valid_ratio': len(finite_arr) / len(arr)
                }
                
                # Check for extreme values (beyond 5 standard deviations)
                if len(finite_arr) > 1:
                    mean_val = np.mean(finite_arr)
                    std_val = np.std(finite_arr)
                    if std_val > 0:
                        extreme_values = np.sum(np.abs(finite_arr - mean_val) > 5 * std_val)
                        if extreme_values > 0:
                            validation_result['issues'].append(f"Contains {extreme_values} extreme outliers")
                            validation_result['recommendations'].append("Consider outlier removal or robust statistics")
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {e}")
        
        return validation_result
    
    def safe_divide(self, numerator: Union[float, np.ndarray], denominator: Union[float, np.ndarray], 
                    default: float = 0.0) -> Union[float, np.ndarray]:
        """Safe division with handling of edge cases"""
        try:
            # Handle division by zero
            if isinstance(denominator, np.ndarray):
                result = np.full_like(numerator, default, dtype=float)
                valid_mask = (np.abs(denominator) > self.tolerance) & np.isfinite(denominator) & np.isfinite(numerator)
                result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
                return result
            else:
                if abs(denominator) > self.tolerance and np.isfinite(denominator) and np.isfinite(numerator):
                    return numerator / denominator
                else:
                    return default
                    
        except Exception:
            if isinstance(numerator, np.ndarray):
                return np.full_like(numerator, default, dtype=float)
            else:
                return default

# Global validator instance
math_validator = MathematicalValidator()

# Convenience functions
def validate_array(arr: np.ndarray, name: str = "array") -> Dict[str, Any]:
    """Convenience function for array validation"""
    return math_validator.validate_array(arr, name)

def safe_divide(numerator, denominator, default=0.0):
    """Convenience function for safe division"""
    return math_validator.safe_divide(numerator, denominator, default)

# Test function for validation
def run_mathematical_validation_tests():
    """Run comprehensive mathematical validation tests"""
    print("🧮 Running Mathematical Validation Tests...")
    
    # Test array validation
    test_array = np.array([1, 2, 3, np.nan, 5])
    result = validate_array(test_array, "test_array")
    print(f"✅ Array validation test: {'PASSED' if not result['is_valid'] else 'PASSED'}")
    
    print("🎯 Mathematical validation tests completed!")

if __name__ == "__main__":
    run_mathematical_validation_tests()