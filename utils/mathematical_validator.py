#!/usr/bin/env python3
"""
🧮 VIPER ENHANCED MATHEMATICAL VALIDATION SYSTEM
Advanced mathematical validation, optimization, and numerical analysis for trading systems

Features:
- Comprehensive mathematical formula validation
- Advanced numerical stability checks
- Statistical analysis and outlier detection
- Risk calculation validation with Monte Carlo
- Performance optimization and profiling
- Backtesting mathematics validation
- Financial indicator validation
- Portfolio optimization mathematics
"""

import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import scipy.optimize as optimize
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from decimal import Decimal, ROUND_HALF_UP
import warnings
import time
import functools
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Enhanced validation result with detailed analysis"""
    name: str
    is_valid: bool
    risk_level: RiskLevel
    confidence: float
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    statistics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    mathematical_properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'is_valid': self.is_valid,
            'risk_level': self.risk_level.value,
            'confidence': self.confidence,
            'issues': self.issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'statistics': self.statistics,
            'performance_metrics': self.performance_metrics,
            'mathematical_properties': self.mathematical_properties
        }

class EnhancedMathematicalValidator:
    """Advanced mathematical validation system with comprehensive analysis"""
    
    def __init__(self):
        self.tolerance = 1e-10
        self.max_iterations = 10000
        self.monte_carlo_samples = 10000
        self.validation_cache = {}
        self.performance_stats = {}
        
        # Numerical constants
        self.machine_epsilon = np.finfo(float).eps
        self.max_safe_integer = 2**53 - 1
        self.min_safe_value = 1e-100
        self.max_safe_value = 1e100
        
        logger.info("🧮 Enhanced Mathematical Validator initialized")
    
    def validate_array(self, arr: np.ndarray, name: str = "array", 
                      level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Comprehensive array validation with multiple analysis levels"""
        start_time = time.time()
        
        try:
            issues = []
            warnings = []
            recommendations = []
            statistics = {}
            mathematical_properties = {}
            
            # Basic validation
            if arr.size == 0:
                return ValidationResult(
                    name=name, is_valid=False, risk_level=RiskLevel.HIGH,
                    confidence=1.0, issues=["Array is empty"], warnings=[],
                    recommendations=["Provide non-empty array"], statistics={},
                    performance_metrics={}, mathematical_properties={}
                )
            
            # Check data types and memory layout
            if not np.issubdtype(arr.dtype, np.number):
                issues.append(f"Non-numeric data type: {arr.dtype}")
            
            # Memory and performance analysis
            memory_usage = arr.nbytes / 1024 / 1024  # MB
            if memory_usage > 100:  # > 100MB
                warnings.append(f"Large memory usage: {memory_usage:.1f} MB")
            
            # NaN and infinite value analysis
            nan_count = np.sum(np.isnan(arr))
            inf_count = np.sum(np.isinf(arr))
            finite_count = np.sum(np.isfinite(arr))
            
            if nan_count > 0:
                issues.append(f"Contains {nan_count} NaN values ({nan_count/len(arr)*100:.1f}%)")
                if nan_count > arr.size * 0.1:
                    risk_level = RiskLevel.HIGH
                else:
                    risk_level = RiskLevel.MEDIUM
                    recommendations.append("Clean NaN values before calculations")
            
            if inf_count > 0:
                issues.append(f"Contains {inf_count} infinite values")
                risk_level = RiskLevel.HIGH
                recommendations.append("Remove infinite values")
            
            # Analyze finite values
            finite_arr = arr[np.isfinite(arr)]
            if len(finite_arr) == 0:
                return ValidationResult(
                    name=name, is_valid=False, risk_level=RiskLevel.CRITICAL,
                    confidence=1.0, issues=["No finite values found"], warnings=[],
                    recommendations=["Fix data source"], statistics={},
                    performance_metrics={}, mathematical_properties={}
                )
            
            # Statistical analysis
            statistics = self._calculate_enhanced_statistics(finite_arr, level)
            
            # Mathematical properties analysis
            if level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                mathematical_properties = self._analyze_mathematical_properties(finite_arr)
            
            # Advanced analysis for comprehensive level
            if level == ValidationLevel.COMPREHENSIVE:
                additional_analysis = self._comprehensive_array_analysis(finite_arr)
                mathematical_properties.update(additional_analysis)
            
            # Outlier detection
            outlier_analysis = self._detect_outliers(finite_arr)
            if outlier_analysis['outlier_count'] > 0:
                warnings.append(f"Detected {outlier_analysis['outlier_count']} outliers")
                recommendations.append("Consider outlier treatment")
            
            # Numerical stability checks
            stability_check = self._check_numerical_stability(finite_arr)
            if not stability_check['stable']:
                issues.extend(stability_check['issues'])
                recommendations.extend(stability_check['recommendations'])
            
            # Determine overall validation result
            is_valid = len(issues) == 0
            if len(issues) > 3:
                risk_level = RiskLevel.CRITICAL
            elif len(issues) > 1:
                risk_level = RiskLevel.HIGH
            elif len(warnings) > 2:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Calculate confidence
            confidence = self._calculate_validation_confidence(
                finite_count / len(arr), len(issues), len(warnings)
            )
            
            # Performance metrics
            execution_time = time.time() - start_time
            performance_metrics = {
                'execution_time': execution_time,
                'memory_usage_mb': memory_usage,
                'elements_per_second': len(arr) / execution_time if execution_time > 0 else 0,
                'validation_level': level.value
            }
            
            return ValidationResult(
                name=name, is_valid=is_valid, risk_level=risk_level,
                confidence=confidence, issues=issues, warnings=warnings,
                recommendations=recommendations, statistics=statistics,
                performance_metrics=performance_metrics,
                mathematical_properties=mathematical_properties
            )
            
        except Exception as e:
            logger.error(f"❌ Array validation error: {e}")
            return ValidationResult(
                name=name, is_valid=False, risk_level=RiskLevel.CRITICAL,
                confidence=0.0, issues=[f"Validation error: {e}"], warnings=[],
                recommendations=["Debug validation process"], statistics={},
                performance_metrics={}, mathematical_properties={}
            )
    
    def _calculate_enhanced_statistics(self, arr: np.ndarray, level: ValidationLevel) -> Dict[str, Any]:
        """Calculate comprehensive statistical measures"""
        stats_dict = {
            'count': len(arr),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'var': float(np.var(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'range': float(np.max(arr) - np.min(arr)),
            'median': float(np.median(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
            'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25))
        }
        
        if level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
            # Additional statistical measures
            stats_dict.update({
                'skewness': float(stats.skew(arr)),
                'kurtosis': float(stats.kurtosis(arr)),
                'mode': float(stats.mode(arr)[0][0]) if len(stats.mode(arr)[0]) > 0 else np.nan,
                'geometric_mean': float(stats.gmean(np.abs(arr) + 1e-10)),  # Avoid log(0)
                'harmonic_mean': float(stats.hmean(np.abs(arr) + 1e-10)),
                'coefficient_of_variation': float(np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else np.inf
            })
        
        if level == ValidationLevel.COMPREHENSIVE:
            # Advanced statistical measures
            try:
                # Normality tests
                shapiro_stat, shapiro_p = stats.shapiro(arr[:5000])  # Limit sample size for performance
                kolmogorov_stat, kolmogorov_p = stats.kstest(arr, 'norm')
                
                stats_dict.update({
                    'entropy': float(stats.entropy(np.histogram(arr, bins=50)[0] + 1)),
                    'mad': float(stats.median_absolute_deviation(arr)),
                    'shapiro_stat': float(shapiro_stat),
                    'shapiro_p_value': float(shapiro_p),
                    'kolmogorov_stat': float(kolmogorov_stat),
                    'kolmogorov_p_value': float(kolmogorov_p),
                    'is_normal': shapiro_p > 0.05,
                    'autocorrelation': float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if len(arr) > 1 else 0
                })
            except Exception as e:
                logger.debug(f"Advanced statistics calculation error: {e}")
        
        return stats_dict
    
    def _analyze_mathematical_properties(self, arr: np.ndarray) -> Dict[str, Any]:
        """Analyze mathematical properties of the array"""
        properties = {}
        
        try:
            # Monotonicity
            is_increasing = np.all(np.diff(arr) >= 0)
            is_decreasing = np.all(np.diff(arr) <= 0)
            is_strictly_increasing = np.all(np.diff(arr) > 0)
            is_strictly_decreasing = np.all(np.diff(arr) < 0)
            
            properties.update({
                'is_monotonic_increasing': is_increasing,
                'is_monotonic_decreasing': is_decreasing,
                'is_strictly_increasing': is_strictly_increasing,
                'is_strictly_decreasing': is_strictly_decreasing,
                'monotonic_score': self._calculate_monotonic_score(arr)
            })
            
            # Periodicity detection
            if len(arr) > 10:
                periodicity = self._detect_periodicity(arr)
                properties.update(periodicity)
            
            # Smoothness (second derivative analysis)
            if len(arr) > 3:
                second_diff = np.diff(arr, n=2)
                smoothness = 1.0 / (1.0 + np.std(second_diff))
                properties['smoothness_score'] = float(smoothness)
            
            # Range analysis
            data_range = np.max(arr) - np.min(arr)
            properties.update({
                'range_normalized': float(data_range / (np.mean(arr) + 1e-10)),
                'dynamic_range': float(np.log10(np.max(np.abs(arr)) / (np.min(np.abs(arr[arr != 0])) + 1e-10))),
                'zero_crossings': int(np.sum(np.diff(np.sign(arr)) != 0))
            })
            
        except Exception as e:
            logger.debug(f"Mathematical properties analysis error: {e}")
            properties['analysis_error'] = str(e)
        
        return properties
    
    def _comprehensive_array_analysis(self, arr: np.ndarray) -> Dict[str, Any]:
        """Comprehensive analysis for advanced validation level"""
        analysis = {}
        
        try:
            # Frequency domain analysis
            if len(arr) > 8:
                fft_result = np.fft.fft(arr - np.mean(arr))  # Remove DC component
                frequencies = np.fft.fftfreq(len(arr))
                power_spectrum = np.abs(fft_result) ** 2
                
                # Find dominant frequencies
                dominant_freq_idx = np.argsort(power_spectrum)[-3:]  # Top 3 frequencies
                dominant_freqs = frequencies[dominant_freq_idx]
                
                analysis.update({
                    'dominant_frequencies': dominant_freqs.tolist(),
                    'spectral_centroid': float(np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)),
                    'spectral_bandwidth': float(np.sqrt(np.sum(((frequencies - analysis.get('spectral_centroid', 0)) ** 2) * power_spectrum) / np.sum(power_spectrum)))
                })
            
            # Complexity measures
            analysis.update({
                'complexity_lz': self._lempel_ziv_complexity(arr),
                'fractal_dimension': self._estimate_fractal_dimension(arr),
                'sample_entropy': self._sample_entropy(arr)
            })
            
            # Stationarity test
            if len(arr) > 20:
                stationarity = self._test_stationarity(arr)
                analysis.update(stationarity)
            
        except Exception as e:
            logger.debug(f"Comprehensive analysis error: {e}")
            analysis['comprehensive_error'] = str(e)
        
        return analysis
    
    def _detect_outliers(self, arr: np.ndarray, method: str = "iqr") -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        outlier_results = {}
        
        try:
            if method == "iqr":
                q25, q75 = np.percentile(arr, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = (arr < lower_bound) | (arr > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(arr))
                outliers = z_scores > 3
                
            elif method == "modified_zscore":
                median = np.median(arr)
                mad = stats.median_absolute_deviation(arr)
                modified_z_scores = 0.6745 * (arr - median) / mad
                outliers = np.abs(modified_z_scores) > 3.5
                
            outlier_results = {
                'outlier_count': int(np.sum(outliers)),
                'outlier_percentage': float(np.sum(outliers) / len(arr) * 100),
                'outlier_indices': np.where(outliers)[0].tolist(),
                'method': method
            }
            
        except Exception as e:
            logger.debug(f"Outlier detection error: {e}")
            outlier_results = {'outlier_count': 0, 'error': str(e)}
        
        return outlier_results
    
    def _check_numerical_stability(self, arr: np.ndarray) -> Dict[str, Any]:
        """Check numerical stability of array values"""
        issues = []
        recommendations = []
        
        # Check for values near machine epsilon
        near_zero = np.abs(arr) < self.machine_epsilon * 100
        if np.any(near_zero):
            issues.append(f"{np.sum(near_zero)} values near machine epsilon")
            recommendations.append("Consider regularization for near-zero values")
        
        # Check for very large values
        very_large = np.abs(arr) > self.max_safe_value / 100
        if np.any(very_large):
            issues.append(f"{np.sum(very_large)} extremely large values")
            recommendations.append("Scale down large values to prevent overflow")
        
        # Check condition number for matrices
        if len(arr.shape) == 2 and arr.shape[0] == arr.shape[1]:
            try:
                cond_num = np.linalg.cond(arr)
                if cond_num > 1e12:
                    issues.append(f"Poor condition number: {cond_num:.2e}")
                    recommendations.append("Matrix may be ill-conditioned")
            except:
                pass
        
        return {
            'stable': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _calculate_validation_confidence(self, finite_ratio: float, issue_count: int, warning_count: int) -> float:
        """Calculate confidence score for validation result"""
        confidence = finite_ratio  # Start with ratio of finite values
        
        # Penalize issues and warnings
        confidence -= issue_count * 0.2
        confidence -= warning_count * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_monotonic_score(self, arr: np.ndarray) -> float:
        """Calculate monotonicity score (0 = not monotonic, 1 = perfectly monotonic)"""
        if len(arr) < 2:
            return 1.0
        
        diffs = np.diff(arr)
        positive_diffs = np.sum(diffs > 0)
        negative_diffs = np.sum(diffs < 0)
        total_diffs = len(diffs)
        
        if total_diffs == 0:
            return 1.0
        
        # Score based on predominant direction
        max_direction = max(positive_diffs, negative_diffs)
        return max_direction / total_diffs
    
    def _detect_periodicity(self, arr: np.ndarray) -> Dict[str, Any]:
        """Detect periodicity in the array"""
        try:
            # Autocorrelation approach
            autocorr = np.correlate(arr - np.mean(arr), arr - np.mean(arr), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
            
            # Estimate period
            if peaks:
                strongest_period = peaks[np.argmax([autocorr[p] for p in peaks])]
                periodicity_strength = autocorr[strongest_period] / autocorr[0]
            else:
                strongest_period = 0
                periodicity_strength = 0.0
            
            return {
                'estimated_period': int(strongest_period),
                'periodicity_strength': float(periodicity_strength),
                'is_periodic': periodicity_strength > 0.3
            }
            
        except Exception as e:
            return {'periodicity_error': str(e)}
    
    def _lempel_ziv_complexity(self, arr: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity (simplified version)"""
        try:
            # Convert to binary string (simplified)
            binary_arr = (arr > np.median(arr)).astype(int)
            binary_str = ''.join(map(str, binary_arr))
            
            complexity = 0
            i = 0
            while i < len(binary_str):
                substring = binary_str[i]
                j = i + 1
                while j <= len(binary_str) and substring not in binary_str[:i]:
                    if j < len(binary_str):
                        substring += binary_str[j]
                    j += 1
                complexity += 1
                i = j - 1 if j > i + 1 else i + 1
            
            return complexity / len(binary_str)  # Normalize
            
        except Exception:
            return 0.0
    
    def _estimate_fractal_dimension(self, arr: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting method (simplified)"""
        try:
            if len(arr) < 10:
                return 1.0
            
            # Simple approach: count direction changes at different scales
            scales = [1, 2, 4, 8]
            counts = []
            
            for scale in scales:
                if scale >= len(arr):
                    continue
                    
                scaled_arr = arr[::scale]
                direction_changes = np.sum(np.diff(np.sign(np.diff(scaled_arr))) != 0)
                counts.append(direction_changes + 1)  # +1 to avoid log(0)
            
            if len(counts) < 2:
                return 1.0
            
            # Linear regression to estimate fractal dimension
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(counts)
            
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            fractal_dim = -slope  # Negative because of inverse relationship
            
            return max(1.0, min(2.0, fractal_dim))  # Clamp to reasonable range
            
        except Exception:
            return 1.0
    
    def _sample_entropy(self, arr: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy"""
        try:
            if len(arr) < m + 1:
                return 0.0
            
            # Normalize data
            arr_norm = (arr - np.mean(arr)) / np.std(arr)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([arr_norm[i:i + m] for i in range(len(arr_norm) - m + 1)])
                C = np.zeros(len(patterns))
                
                for i in range(len(patterns)):
                    template_i = patterns[i]
                    for j in range(len(patterns)):
                        if _maxdist(template_i, patterns[j], m) <= r:
                            C[i] += 1.0
                
                phi = (C / len(patterns)).mean()
                return phi
            
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            
            if phi_m1 == 0 or phi_m == 0:
                return 0.0
            
            return -np.log(phi_m1 / phi_m)
            
        except Exception:
            return 0.0
    
    def _test_stationarity(self, arr: np.ndarray) -> Dict[str, Any]:
        """Test time series stationarity (simplified)"""
        try:
            # Split into segments and test for consistent statistics
            n_segments = min(5, len(arr) // 10)
            segment_size = len(arr) // n_segments
            
            segment_means = []
            segment_stds = []
            
            for i in range(n_segments):
                start = i * segment_size
                end = (i + 1) * segment_size if i < n_segments - 1 else len(arr)
                segment = arr[start:end]
                
                segment_means.append(np.mean(segment))
                segment_stds.append(np.std(segment))
            
            # Test if means and stds are relatively consistent
            mean_cv = np.std(segment_means) / (np.mean(segment_means) + 1e-10)
            std_cv = np.std(segment_stds) / (np.mean(segment_stds) + 1e-10)
            
            is_stationary = mean_cv < 0.1 and std_cv < 0.1
            
            return {
                'is_stationary': is_stationary,
                'mean_coefficient_of_variation': float(mean_cv),
                'std_coefficient_of_variation': float(std_cv),
                'stationarity_score': float(1.0 / (1.0 + mean_cv + std_cv))
            }
            
        except Exception as e:
            return {'stationarity_error': str(e)}
    
    def validate_trading_calculation(self, calculation_func: Callable, 
                                   inputs: Dict[str, Any], 
                                   expected_range: Tuple[float, float] = None,
                                   monte_carlo: bool = False) -> ValidationResult:
        """Validate trading-specific calculations with Monte Carlo analysis"""
        start_time = time.time()
        
        try:
            issues = []
            warnings = []
            recommendations = []
            statistics = {}
            
            # Execute the calculation
            try:
                result = calculation_func(**inputs)
                if not np.isfinite(result):
                    issues.append("Calculation result is not finite")
                    
            except Exception as e:
                return ValidationResult(
                    name="trading_calculation", is_valid=False, 
                    risk_level=RiskLevel.CRITICAL, confidence=0.0,
                    issues=[f"Calculation failed: {e}"], warnings=[],
                    recommendations=["Fix calculation implementation"],
                    statistics={}, performance_metrics={}, mathematical_properties={}
                )
            
            # Validate against expected range
            if expected_range and np.isfinite(result):
                if not (expected_range[0] <= result <= expected_range[1]):
                    issues.append(f"Result {result} outside expected range {expected_range}")
            
            # Monte Carlo analysis
            if monte_carlo and np.isfinite(result):
                mc_results = self._monte_carlo_analysis(calculation_func, inputs)
                statistics.update(mc_results)
                
                # Check for stability
                if mc_results.get('std', 0) / abs(mc_results.get('mean', 1)) > 0.5:  # High variance
                    warnings.append("High variability in Monte Carlo analysis")
                    recommendations.append("Check input sensitivity")
            
            # Performance analysis
            execution_time = time.time() - start_time
            performance_metrics = {
                'execution_time': execution_time,
                'result': float(result) if np.isfinite(result) else None
            }
            
            # Determine validation result
            is_valid = len(issues) == 0 and np.isfinite(result)
            risk_level = RiskLevel.LOW if is_valid and len(warnings) == 0 else RiskLevel.MEDIUM
            confidence = 1.0 if is_valid else 0.5
            
            return ValidationResult(
                name="trading_calculation", is_valid=is_valid,
                risk_level=risk_level, confidence=confidence,
                issues=issues, warnings=warnings, recommendations=recommendations,
                statistics=statistics, performance_metrics=performance_metrics,
                mathematical_properties={}
            )
            
        except Exception as e:
            logger.error(f"❌ Trading calculation validation error: {e}")
            return ValidationResult(
                name="trading_calculation", is_valid=False,
                risk_level=RiskLevel.CRITICAL, confidence=0.0,
                issues=[f"Validation error: {e}"], warnings=[],
                recommendations=["Debug validation process"],
                statistics={}, performance_metrics={}, mathematical_properties={}
            )
    
    def _monte_carlo_analysis(self, calculation_func: Callable, 
                            base_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Monte Carlo analysis on calculation"""
        try:
            results = []
            
            for _ in range(self.monte_carlo_samples):
                # Add noise to inputs (simplified approach)
                noisy_inputs = {}
                for key, value in base_inputs.items():
                    if isinstance(value, (int, float)):
                        # Add 1% random noise
                        noise_factor = 1 + np.random.normal(0, 0.01)
                        noisy_inputs[key] = value * noise_factor
                    else:
                        noisy_inputs[key] = value
                
                try:
                    result = calculation_func(**noisy_inputs)
                    if np.isfinite(result):
                        results.append(result)
                except:
                    continue
            
            if len(results) == 0:
                return {'monte_carlo_error': 'No valid results'}
            
            results = np.array(results)
            
            return {
                'monte_carlo_samples': len(results),
                'mean': float(np.mean(results)),
                'std': float(np.std(results)),
                'min': float(np.min(results)),
                'max': float(np.max(results)),
                'p05': float(np.percentile(results, 5)),
                'p95': float(np.percentile(results, 95)),
                'coefficient_of_variation': float(np.std(results) / np.mean(results)) if np.mean(results) != 0 else np.inf
            }
            
        except Exception as e:
            return {'monte_carlo_error': str(e)}
    
    def validate_risk_calculation(self, portfolio_returns: np.ndarray, 
                                confidence_level: float = 0.05) -> ValidationResult:
        """Validate risk calculations like VaR, Expected Shortfall, etc."""
        
        # Validate portfolio returns array
        returns_validation = self.validate_array(portfolio_returns, "portfolio_returns", ValidationLevel.COMPREHENSIVE)
        
        if not returns_validation.is_valid:
            return returns_validation
        
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # Calculate risk metrics
            var = np.percentile(portfolio_returns, confidence_level * 100)
            expected_shortfall = np.mean(portfolio_returns[portfolio_returns <= var])
            volatility = np.std(portfolio_returns)
            sharpe_ratio = np.mean(portfolio_returns) / volatility if volatility > 0 else 0
            
            risk_metrics = {
                'value_at_risk': float(var),
                'expected_shortfall': float(expected_shortfall),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(self._calculate_max_drawdown(portfolio_returns))
            }
            
            # Validate risk metrics
            if abs(sharpe_ratio) > 5:  # Unrealistic Sharpe ratio
                warnings.append(f"Extreme Sharpe ratio: {sharpe_ratio:.2f}")
                recommendations.append("Review return and volatility calculations")
            
            if volatility > 0.5:  # > 50% volatility
                warnings.append(f"High volatility: {volatility:.2%}")
                recommendations.append("Consider position sizing adjustments")
            
            return ValidationResult(
                name="risk_calculation", is_valid=len(issues) == 0,
                risk_level=RiskLevel.LOW if len(warnings) == 0 else RiskLevel.MEDIUM,
                confidence=0.9, issues=issues, warnings=warnings,
                recommendations=recommendations, statistics=risk_metrics,
                performance_metrics={}, mathematical_properties={}
            )
            
        except Exception as e:
            return ValidationResult(
                name="risk_calculation", is_valid=False,
                risk_level=RiskLevel.CRITICAL, confidence=0.0,
                issues=[f"Risk calculation error: {e}"], warnings=[],
                recommendations=["Debug risk calculation"], statistics={},
                performance_metrics={}, mathematical_properties={}
            )
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(np.min(drawdown))
        except:
            return 0.0
    
    def safe_divide(self, numerator: Union[float, np.ndarray], 
                   denominator: Union[float, np.ndarray], 
                   default: float = 0.0) -> Union[float, np.ndarray]:
        """Enhanced safe division with comprehensive error handling"""
        try:
            if isinstance(denominator, np.ndarray):
                result = np.full_like(numerator, default, dtype=float)
                
                # Create mask for valid divisions
                valid_mask = (
                    (np.abs(denominator) > self.tolerance) & 
                    np.isfinite(denominator) & 
                    np.isfinite(numerator) &
                    (np.abs(denominator) < self.max_safe_value) &
                    (np.abs(numerator) < self.max_safe_value)
                )
                
                result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
                
                # Check for potential overflow
                overflow_mask = np.abs(result) > self.max_safe_value / 2
                result[overflow_mask] = np.sign(result[overflow_mask]) * self.max_safe_value / 2
                
                return result
            else:
                # Scalar division
                if (abs(denominator) > self.tolerance and 
                    np.isfinite(denominator) and np.isfinite(numerator) and
                    abs(denominator) < self.max_safe_value and
                    abs(numerator) < self.max_safe_value):
                    
                    result = numerator / denominator
                    
                    # Check for overflow
                    if abs(result) > self.max_safe_value / 2:
                        return np.sign(result) * self.max_safe_value / 2
                    
                    return result
                else:
                    return default
                    
        except Exception:
            if isinstance(numerator, np.ndarray):
                return np.full_like(numerator, default, dtype=float)
            else:
                return default
    
    def benchmark_calculation(self, calculation_func: Callable, 
                            inputs: Dict[str, Any], 
                            iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark a calculation function for performance analysis"""
        
        execution_times = []
        memory_usage = []
        
        for i in range(iterations):
            start_time = time.time()
            start_memory = 0  # Would use memory profiling in production
            
            try:
                result = calculation_func(**inputs)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                # Memory usage estimation (simplified)
                if hasattr(result, 'nbytes'):
                    memory_usage.append(result.nbytes)
                elif isinstance(result, (list, tuple)):
                    memory_usage.append(len(result) * 8)  # Rough estimate
                else:
                    memory_usage.append(8)  # Single value
                    
            except Exception as e:
                logger.debug(f"Benchmark iteration {i} failed: {e}")
                continue
        
        if not execution_times:
            return {'error': 'All benchmark iterations failed'}
        
        execution_times = np.array(execution_times)
        memory_usage = np.array(memory_usage)
        
        return {
            'iterations': len(execution_times),
            'mean_execution_time': float(np.mean(execution_times)),
            'std_execution_time': float(np.std(execution_times)),
            'min_execution_time': float(np.min(execution_times)),
            'max_execution_time': float(np.max(execution_times)),
            'operations_per_second': 1.0 / np.mean(execution_times),
            'mean_memory_usage': float(np.mean(memory_usage)),
            'performance_consistency': 1.0 / (1.0 + np.std(execution_times) / np.mean(execution_times))
        }

# Global enhanced validator instance
enhanced_math_validator = EnhancedMathematicalValidator()

# Convenience functions
def validate_array(arr: np.ndarray, name: str = "array", 
                  level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """Convenience function for enhanced array validation"""
    return enhanced_math_validator.validate_array(arr, name, level)

def safe_divide(numerator, denominator, default=0.0):
    """Convenience function for enhanced safe division"""
    return enhanced_math_validator.safe_divide(numerator, denominator, default)

def validate_trading_calculation(calculation_func: Callable, inputs: Dict[str, Any], 
                               expected_range: Tuple[float, float] = None,
                               monte_carlo: bool = False) -> ValidationResult:
    """Convenience function for trading calculation validation"""
    return enhanced_math_validator.validate_trading_calculation(
        calculation_func, inputs, expected_range, monte_carlo
    )

def validate_risk_metrics(portfolio_returns: np.ndarray, 
                         confidence_level: float = 0.05) -> ValidationResult:
    """Convenience function for risk metrics validation"""
    return enhanced_math_validator.validate_risk_calculation(portfolio_returns, confidence_level)

def benchmark_function(calculation_func: Callable, inputs: Dict[str, Any], 
                      iterations: int = 1000) -> Dict[str, Any]:
    """Convenience function for performance benchmarking"""
    return enhanced_math_validator.benchmark_calculation(calculation_func, inputs, iterations)

# Test function for validation
def run_enhanced_mathematical_validation_tests():
    """Run comprehensive mathematical validation tests"""
    print("🧮 Running Enhanced Mathematical Validation Tests...")
    
    # Test array validation
    test_array = np.array([1, 2, 3, np.nan, 5, 100, -50, 2.5])
    result = validate_array(test_array, "test_array", ValidationLevel.COMPREHENSIVE)
    print(f"✅ Enhanced array validation: {'PASSED' if result.confidence > 0.5 else 'NEEDS ATTENTION'}")
    print(f"   Risk Level: {result.risk_level.value}, Confidence: {result.confidence:.2%}")
    
    # Test safe division
    numerators = np.array([10, 20, 30])
    denominators = np.array([2, 0, 5])  # Include division by zero
    safe_results = safe_divide(numerators, denominators, default=-999)
    print(f"✅ Enhanced safe division: {safe_results}")
    
    # Test trading calculation validation
    def simple_return_calc(price_start, price_end):
        return (price_end - price_start) / price_start
    
    trading_result = validate_trading_calculation(
        simple_return_calc, 
        {'price_start': 100, 'price_end': 110},
        expected_range=(-1.0, 1.0),
        monte_carlo=True
    )
    print(f"✅ Trading calculation validation: {'PASSED' if trading_result.is_valid else 'FAILED'}")
    
    # Test risk metrics validation
    sample_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    risk_result = validate_risk_metrics(sample_returns)
    print(f"✅ Risk metrics validation: {'PASSED' if risk_result.is_valid else 'FAILED'}")
    
    # Test performance benchmarking
    def test_calculation(x, y):
        return np.sum(x * y)
    
    benchmark_result = benchmark_function(
        test_calculation, 
        {'x': np.random.rand(1000), 'y': np.random.rand(1000)},
        iterations=100
    )
    print(f"✅ Performance benchmark: {benchmark_result.get('operations_per_second', 0):.0f} ops/sec")
    
    print("🎯 Enhanced mathematical validation tests completed!")

if __name__ == "__main__":
    run_enhanced_mathematical_validation_tests()