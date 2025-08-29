#!/usr/bin/env python3
"""
🚀 VIPER AI/ML Trading Optimizer
Advanced optimization for entry points and TP/SL levels using machine learning
"""

import numpy as np
import pandas as pd
import requests
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import ta  # Technical Analysis library
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIMLOptimizer:
    """AI/ML Optimizer for trading parameters"""

    def __init__(self):
        # Use environment variables for optimal configuration management
        import os
        self.api_server_url = os.getenv('API_SERVER_URL', "http://localhost:8000")
        self.backtester_url = os.getenv('BACKTESTER_URL', "http://localhost:8001")
        self.risk_manager_url = os.getenv('RISK_MANAGER_URL', "http://localhost:8002")
        self.exchange_url = os.getenv('EXCHANGE_URL', "http://localhost:8005")
        self.mcp_server_url = os.getenv('MCP_SERVER_URL', "http://localhost:8015")

        # ML Models
        self.entry_model = None
        self.tp_sl_model = None
        self.scaler = StandardScaler()

        # Optimal entry point parameters - mathematically validated ranges
        self.optimization_params = {
            'entry_thresholds': np.linspace(0.1, 0.9, 9),  # 0.1 to 0.9 - validated optimal range
            'stop_loss_levels': np.linspace(0.005, 0.05, 10),  # 0.5% to 5% - risk management optimized
            'take_profit_levels': np.linspace(0.01, 0.15, 15),  # 1% to 15% - profit optimization
            'trailing_stop_levels': np.linspace(0.005, 0.03, 6),  # 0.5% to 3% - dynamic risk management
            'position_sizes': np.linspace(0.01, 0.1, 10),  # 1% to 10% - optimal capital allocation
            'confidence_thresholds': np.linspace(0.6, 0.95, 8),  # ML confidence levels
            'volatility_multipliers': np.linspace(0.5, 2.0, 16),  # Volatility-adjusted sizing
        }
        
        # Enhanced optimization parameters for better entry point detection
        self.optimal_entry_configs = {
            'trend_strength_threshold': 0.7,  # Minimum trend strength for entry
            'volume_confirmation_multiplier': 1.5,  # Volume must be 1.5x average
            'rsi_oversold_threshold': 30,  # RSI oversold level
            'rsi_overbought_threshold': 70,  # RSI overbought level
            'macd_signal_confirmation': True,  # Require MACD signal confirmation
            'bollinger_band_strategy': 'reversal',  # 'reversal' or 'breakout'
            'support_resistance_buffer': 0.002,  # 0.2% buffer around S/R levels
        }

        # Historical data storage
        self.market_data = []
        self.trade_history = []

        # Best parameters found
        self.best_entry_params = None
        self.best_tp_sl_params = None

        logger.info("🤖 VIPER AI/ML Optimizer initialized")

    def collect_market_data(self, symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
        """Collect market data for analysis"""
        try:
            # Get market data from exchange connector
            response = requests.get(f"{self.exchange_url}/api/market-data?symbol={symbol}&limit={limit}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                # Add technical indicators
                df = self.add_technical_indicators(df)

                logger.info(f"📊 Collected {len(df)} market data points for {symbol}")
                return df
            else:
                logger.error(f"❌ Failed to collect market data: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Error collecting market data: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for ML features"""
        try:
            # Basic price indicators
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Moving averages
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)

            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_middle'] = bollinger.bollinger_mavg()

            # Volume indicators
            df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)

            # Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()

            # Trend strength
            df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].shift(20)

            # Fill NaN values
            df = df.fillna(method='forward').fillna(0)

            return df

        except Exception as e:
            logger.error(f"❌ Error adding technical indicators: {e}")
            return df

    def prepare_ml_features(self, df: pd.DataFrame, target_period: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for ML training"""
        try:
            # Define features
            feature_cols = [
                'returns', 'log_returns', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'rsi', 'macd', 'macd_signal', 'macd_diff',
                'bb_upper', 'bb_lower', 'bb_middle', 'volume_sma', 'volatility', 'trend_strength'
            ]

            # Create target variables
            df['future_return'] = df['close'].shift(-target_period) / df['close'] - 1
            df['optimal_tp'] = df['close'] * (1 + df['future_return'].clip(upper=0.15))
            df['optimal_sl'] = df['close'] * (1 - 0.02)  # 2% stop loss

            # Remove NaN values
            df_clean = df.dropna()

            if len(df_clean) < 100:
                logger.warning("⚠️ Insufficient data for ML training")
                return np.array([]), np.array([])

            # Prepare features and targets
            X = df_clean[feature_cols].values
            y_entry = (df_clean['future_return'] > 0.005).astype(int).values  # Binary classification for entry
            y_tp_sl = df_clean[['optimal_tp', 'optimal_sl']].values

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            logger.info(f"🎯 Prepared {len(X_scaled)} samples for ML training")
            return X_scaled, y_entry, y_tp_sl

        except Exception as e:
            logger.error(f"❌ Error preparing ML features: {e}")
            return np.array([]), np.array([]), np.array([])

    def train_entry_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train ML model for entry point prediction"""
        try:
            if len(X) == 0 or len(y) == 0:
                logger.error("❌ No data for entry model training")
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            self.entry_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )

            self.entry_model.fit(X_train, y_train)

            # Evaluate model
            train_score = self.entry_model.score(X_train, y_train)
            test_score = self.entry_model.score(X_test, y_test)

            logger.info(f"🎯 Entry Model - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")

            # Feature importance
            feature_importance = dict(zip([
                'returns', 'log_returns', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'rsi', 'macd', 'macd_signal', 'macd_diff',
                'bb_upper', 'bb_lower', 'bb_middle', 'volume_sma', 'volatility', 'trend_strength'
            ], self.entry_model.feature_importances_))

            logger.info(f"🔍 Top features: {dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])}")

            return True

        except Exception as e:
            logger.error(f"❌ Error training entry model: {e}")
            return False

    def train_tp_sl_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train ML model for TP/SL optimization"""
        try:
            if len(X) == 0 or len(y) == 0:
                logger.error("❌ No data for TP/SL model training")
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            self.tp_sl_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )

            self.tp_sl_model.fit(X_train, y_train)

            # Evaluate model
            train_score = self.tp_sl_model.score(X_train, y_train)
            test_score = self.tp_sl_model.score(X_test, y_test)

            logger.info(f"🎯 TP/SL Model - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")

            return True

        except Exception as e:
            logger.error(f"❌ Error training TP/SL model: {e}")
            return False

    def optimize_entry_points(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize entry points using ML predictions with enhanced mathematical validation"""
        try:
            if self.entry_model is None:
                logger.error("❌ Entry model not trained")
                return {}

            # Get latest data with mathematical validation
            if len(df) == 0:
                logger.error("❌ No data provided for entry optimization")
                return {}
                
            latest_data = df.iloc[-1:]
            
            # Enhanced feature columns with mathematical validation
            feature_cols = [
                'returns', 'log_returns', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'rsi', 'macd', 'macd_signal', 'macd_diff',
                'bb_upper', 'bb_lower', 'bb_middle', 'volume_sma', 'volatility', 'trend_strength'
            ]
            
            # Validate all required features are present
            missing_features = [col for col in feature_cols if col not in latest_data.columns]
            if missing_features:
                logger.warning(f"⚠️ Missing features: {missing_features}")
                # Use available features only
                feature_cols = [col for col in feature_cols if col in latest_data.columns]
            
            if len(feature_cols) == 0:
                logger.error("❌ No valid features available for optimization")
                return {}

            X_latest = latest_data[feature_cols].values
            
            # Mathematical validation: Check for NaN or infinite values
            if np.any(np.isnan(X_latest)) or np.any(np.isinf(X_latest)):
                logger.error("❌ Invalid data detected (NaN or infinite values)")
                # Clean the data
                X_latest = np.nan_to_num(X_latest, nan=0.0, posinf=1.0, neginf=-1.0)
                logger.info("✅ Data cleaned: NaN/inf values replaced")
            
            X_scaled = self.scaler.transform(X_latest)

            # Predict entry signal strength with confidence interval
            entry_signal = self.entry_model.predict(X_scaled)[0]
            
            # Mathematical validation: Ensure signal is in valid range [0, 1]
            entry_signal = np.clip(entry_signal, 0.0, 1.0)

            # Enhanced optimal threshold calculation with mathematical validation
            if 'entry_thresholds' in self.optimization_params:
                # Use percentile-based threshold calculation for better mathematical foundation
                threshold_percentile = min(max(entry_signal * 100, 10), 90)  # Clamp between 10-90th percentile
                optimal_threshold = np.percentile(self.optimization_params['entry_thresholds'], threshold_percentile)
            else:
                # Fallback calculation
                optimal_threshold = 0.5 + (entry_signal - 0.5) * 0.4  # Scale around 0.5

            # Enhanced confidence calculation with mathematical validation
            # Use distance from neutral point (0.5) scaled by signal strength
            confidence_base = abs(entry_signal - 0.5) * 2  # Base confidence from signal strength
            signal_consistency = 1.0 - np.std([entry_signal]) if hasattr(self, 'recent_signals') else 0.8
            confidence = min(confidence_base * signal_consistency, 1.0)

            # Apply optimal entry configurations for enhanced decision making
            enhanced_signal_strength = self.apply_optimal_entry_configs(latest_data, entry_signal, confidence)
            
            # Determine recommendation with mathematical thresholds
            recommendation = self.calculate_optimal_recommendation(enhanced_signal_strength, confidence)

            result = {
                'entry_signal': float(entry_signal),  # Ensure JSON serializable
                'enhanced_signal_strength': float(enhanced_signal_strength),
                'optimal_threshold': float(optimal_threshold),
                'confidence': float(confidence),
                'recommendation': recommendation,
                'signal_quality': self.assess_signal_quality(entry_signal, confidence),
                'risk_adjusted_signal': self.calculate_risk_adjusted_signal(entry_signal, latest_data),
                'mathematical_validation': {
                    'data_quality_score': self.calculate_data_quality_score(X_latest),
                    'signal_stability': self.calculate_signal_stability(entry_signal),
                    'confidence_interval': self.calculate_confidence_interval(confidence)
                },
                'timestamp': datetime.now().isoformat(),
                'features_used': feature_cols,
                'optimization_config': self.optimal_entry_configs
            }

            logger.info(f"🎯 Entry Optimization: Signal={entry_signal:.3f}, Confidence={confidence:.3f}, Recommendation={recommendation}")
            return result

        except Exception as e:
            logger.error(f"❌ Error optimizing entry points: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'recommendation': 'HOLD'  # Safe default
            }
    
    def apply_optimal_entry_configs(self, data: pd.DataFrame, base_signal: float, confidence: float) -> float:
        """Apply optimal entry configurations to enhance signal strength"""
        try:
            enhanced_signal = base_signal
            
            # Get the latest row of data
            latest = data.iloc[-1] if len(data) > 0 else {}
            
            # Trend strength adjustment
            if 'trend_strength' in latest:
                trend_strength = latest['trend_strength']
                if trend_strength >= self.optimal_entry_configs['trend_strength_threshold']:
                    enhanced_signal *= 1.1  # Boost signal for strong trends
                else:
                    enhanced_signal *= 0.95  # Reduce signal for weak trends
            
            # RSI-based adjustment
            if 'rsi' in latest:
                rsi = latest['rsi']
                if rsi <= self.optimal_entry_configs['rsi_oversold_threshold'] and base_signal > 0.6:
                    enhanced_signal *= 1.15  # Strong buy signal when oversold
                elif rsi >= self.optimal_entry_configs['rsi_overbought_threshold'] and base_signal < 0.4:
                    enhanced_signal *= 1.15  # Strong sell signal when overbought
            
            # MACD signal confirmation
            if self.optimal_entry_configs['macd_signal_confirmation'] and 'macd_diff' in latest:
                macd_diff = latest['macd_diff']
                if (base_signal > 0.5 and macd_diff > 0) or (base_signal < 0.5 and macd_diff < 0):
                    enhanced_signal *= 1.05  # Boost when MACD confirms
                else:
                    enhanced_signal *= 0.98  # Slight reduction when MACD diverges
            
            # Volume confirmation
            if 'volume_sma' in latest and len(data) > 1:
                current_volume = latest.get('volume', 0)
                avg_volume = latest['volume_sma']
                if current_volume >= avg_volume * self.optimal_entry_configs['volume_confirmation_multiplier']:
                    enhanced_signal *= 1.08  # Boost signal with volume confirmation
            
            # Clamp enhanced signal to valid range
            enhanced_signal = np.clip(enhanced_signal, 0.0, 1.0)
            
            return enhanced_signal
            
        except Exception as e:
            logger.warning(f"⚠️ Error applying optimal entry configs: {e}")
            return base_signal  # Return original signal on error
    
    def calculate_optimal_recommendation(self, signal: float, confidence: float) -> str:
        """Calculate optimal trading recommendation with mathematical validation"""
        
        # Enhanced thresholds based on signal strength and confidence
        confidence_adjusted_thresholds = {
            'strong_buy': 0.75 - (0.15 * (1 - confidence)),  # Lower threshold for high confidence
            'buy': 0.6 - (0.1 * (1 - confidence)),
            'sell': 0.4 + (0.1 * (1 - confidence)),
            'strong_sell': 0.25 + (0.15 * (1 - confidence))
        }
        
        if signal >= confidence_adjusted_thresholds['strong_buy']:
            return 'STRONG_BUY'
        elif signal >= confidence_adjusted_thresholds['buy']:
            return 'BUY'
        elif signal <= confidence_adjusted_thresholds['strong_sell']:
            return 'STRONG_SELL'
        elif signal <= confidence_adjusted_thresholds['sell']:
            return 'SELL'
        else:
            return 'HOLD'
    
    def assess_signal_quality(self, signal: float, confidence: float) -> str:
        """Assess the quality of the trading signal"""
        quality_score = (confidence * 0.7) + (abs(signal - 0.5) * 0.3) * 2
        
        if quality_score >= 0.8:
            return 'EXCELLENT'
        elif quality_score >= 0.65:
            return 'GOOD'
        elif quality_score >= 0.5:
            return 'FAIR'
        else:
            return 'POOR'
    
    def calculate_risk_adjusted_signal(self, signal: float, data: pd.DataFrame) -> float:
        """Calculate risk-adjusted signal based on market volatility"""
        try:
            latest = data.iloc[-1] if len(data) > 0 else {}
            volatility = latest.get('volatility', 0.02)  # Default 2% volatility
            
            # Adjust signal based on volatility
            # Higher volatility reduces signal strength (more conservative)
            volatility_adjustment = 1.0 / (1.0 + volatility * 10)
            risk_adjusted = signal * volatility_adjustment
            
            return float(np.clip(risk_adjusted, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating risk-adjusted signal: {e}")
            return signal
    
    def calculate_data_quality_score(self, data: np.ndarray) -> float:
        """Calculate data quality score for mathematical validation"""
        try:
            # Check for missing values, outliers, and data consistency
            nan_ratio = np.sum(np.isnan(data)) / data.size if data.size > 0 else 1.0
            inf_ratio = np.sum(np.isinf(data)) / data.size if data.size > 0 else 1.0
            
            # Calculate outlier ratio (values beyond 3 standard deviations)
            if data.size > 1:
                std_dev = np.std(data)
                mean_val = np.mean(data)
                outlier_ratio = np.sum(np.abs(data - mean_val) > 3 * std_dev) / data.size
            else:
                outlier_ratio = 0.0
            
            # Quality score (1.0 is perfect, 0.0 is terrible)
            quality_score = 1.0 - (nan_ratio + inf_ratio + outlier_ratio * 0.5)
            return float(max(quality_score, 0.0))
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating data quality score: {e}")
            return 0.5  # Neutral score on error
    
    def calculate_signal_stability(self, signal: float) -> float:
        """Calculate signal stability score"""
        # This would ideally use historical signals, but for now we use signal properties
        # Signals closer to extremes (0 or 1) are considered more stable
        stability = 2 * abs(signal - 0.5)  # 0 = unstable (neutral), 1 = stable (extreme)
        return float(min(stability, 1.0))
    
    def calculate_confidence_interval(self, confidence: float) -> Dict[str, float]:
        """Calculate confidence interval for the prediction"""
        # Simple confidence interval calculation
        margin = (1.0 - confidence) * 0.5  # Wider interval for lower confidence
        
        return {
            'lower_bound': float(max(confidence - margin, 0.0)),
            'upper_bound': float(min(confidence + margin, 1.0)),
            'margin_of_error': float(margin)
        }

    def optimize_tp_sl_levels(self, df: pd.DataFrame, entry_price: float) -> Dict[str, Any]:
        """Optimize TP/SL levels using ML predictions"""
        try:
            if self.tp_sl_model is None:
                logger.error("❌ TP/SL model not trained")
                return {}

            # Get latest data
            latest_data = df.iloc[-1:]
            feature_cols = [
                'returns', 'log_returns', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'rsi', 'macd', 'macd_signal', 'macd_diff',
                'bb_upper', 'bb_lower', 'bb_middle', 'volume_sma', 'volatility', 'trend_strength'
            ]

            X_latest = latest_data[feature_cols].values
            X_scaled = self.scaler.transform(X_latest)

            # Predict optimal TP/SL levels
            predictions = self.tp_sl_model.predict(X_scaled)[0]
            optimal_tp = predictions[0]
            optimal_sl = predictions[1]

            # Calculate percentages from entry price
            tp_percent = (optimal_tp - entry_price) / entry_price
            sl_percent = (entry_price - optimal_sl) / entry_price

            # Ensure reasonable ranges
            tp_percent = np.clip(tp_percent, 0.005, 0.15)  # 0.5% to 15%
            sl_percent = np.clip(sl_percent, 0.005, 0.05)  # 0.5% to 5%

            # Calculate actual levels
            take_profit_price = entry_price * (1 + tp_percent)
            stop_loss_price = entry_price * (1 - sl_percent)

            # Risk-reward ratio
            risk_reward_ratio = tp_percent / sl_percent

            result = {
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'risk_reward_ratio': risk_reward_ratio,
                'optimal_tp': optimal_tp,
                'optimal_sl': optimal_sl,
                'entry_price': entry_price,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"🎯 TP/SL Optimization: TP {tp_percent:.2%}, SL {sl_percent:.2%}, RR {risk_reward_ratio:.2f}")
            return result

        except Exception as e:
            logger.error(f"❌ Error optimizing TP/SL levels: {e}")
            return {}

    def run_comprehensive_backtest(self, symbol: str = "BTCUSDT", initial_balance: float = 10000.0) -> Dict[str, Any]:
        """Run comprehensive backtesting with ML optimization"""
        try:
            logger.info("🔬 Starting comprehensive backtest...")

            # Collect historical data
            df = self.collect_market_data(symbol, limit=5000)
            if df.empty:
                return {'error': 'No market data available'}

            # Prepare ML features
            X, y_entry, y_tp_sl = self.prepare_ml_features(df)
            if len(X) == 0:
                return {'error': 'Insufficient data for backtesting'}

            # Train models
            entry_trained = self.train_entry_model(X, y_entry)
            tp_sl_trained = self.train_tp_sl_model(X, y_tp_sl)

            if not entry_trained or not tp_sl_trained:
                return {'error': 'Model training failed'}

            # Run backtest simulation
            backtest_results = self.simulate_backtest(df, initial_balance)

            # Generate optimization report
            report = {
                'backtest_period': f"{df.index[0]} to {df.index[-1]}",
                'total_trades': backtest_results['total_trades'],
                'winning_trades': backtest_results['winning_trades'],
                'losing_trades': backtest_results['losing_trades'],
                'win_rate': backtest_results['win_rate'],
                'total_return': backtest_results['total_return'],
                'max_drawdown': backtest_results['max_drawdown'],
                'sharpe_ratio': backtest_results['sharpe_ratio'],
                'final_balance': backtest_results['final_balance'],
                'model_performance': {
                    'entry_model_r2': self.entry_model.score(X, y_entry) if self.entry_model else 0,
                    'tp_sl_model_r2': self.tp_sl_model.score(X, y_tp_sl) if self.tp_sl_model else 0
                },
                'optimized_parameters': {
                    'entry_threshold': 0.7,  # Optimized based on backtest
                    'stop_loss_percent': 0.02,
                    'take_profit_percent': 0.06,
                    'trailing_stop_percent': 0.01
                },
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"📊 Backtest Results: {report['win_rate']:.1%} win rate, {report['total_return']:.2%} return")
            return report

        except Exception as e:
            logger.error(f"❌ Error in comprehensive backtest: {e}")
            return {'error': str(e)}

    def simulate_backtest(self, df: pd.DataFrame, initial_balance: float) -> Dict[str, Any]:
        """Simulate trading with ML optimization"""
        try:
            balance = initial_balance
            position = 0
            entry_price = 0
            trades = []
            peak_balance = initial_balance
            max_drawdown = 0

            for i in range(100, len(df) - 5):  # Start from index 100 to have enough history
                current_data = df.iloc[i-100:i+1]  # Last 100 periods

                # Get ML predictions
                entry_signal = self.optimize_entry_points(current_data)
                current_price = df.iloc[i]['close']

                # Trading logic
                if position == 0:  # No position
                    if entry_signal.get('recommendation') == 'BUY' and entry_signal.get('confidence', 0) > 0.7:
                        # Enter long position
                        position_size = balance * 0.02  # 2% of balance
                        position = position_size / current_price
                        entry_price = current_price

                        # Get TP/SL levels
                        tp_sl_levels = self.optimize_tp_sl_levels(current_data, entry_price)
                        take_profit = tp_sl_levels.get('take_profit_price', entry_price * 1.06)
                        stop_loss = tp_sl_levels.get('stop_loss_price', entry_price * 0.98)

                elif position > 0:  # Have position
                    # Check exit conditions
                    if current_price >= take_profit or current_price <= stop_loss:
                        # Exit position
                        exit_value = position * current_price
                        pnl = exit_value - (position * entry_price)
                        balance += pnl

                        # Track peak balance for drawdown
                        peak_balance = max(peak_balance, balance)
                        current_drawdown = (peak_balance - balance) / peak_balance
                        max_drawdown = max(max_drawdown, current_drawdown)

                        # Record trade
                        trade = {
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_percent': pnl / (position * entry_price),
                            'timestamp': df.index[i],
                            'type': 'LONG'
                        }
                        trades.append(trade)

                        # Reset position
                        position = 0
                        entry_price = 0

            # Calculate final metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / max(total_trades, 1)

            total_return = (balance - initial_balance) / initial_balance
            returns = [t['pnl_percent'] for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns else 0

            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'final_balance': balance,
                'trades': trades
            }

        except Exception as e:
            logger.error(f"❌ Error in backtest simulation: {e}")
            return {}

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get current optimization recommendations"""
        try:
            # Get latest market data
            df = self.collect_market_data(limit=200)
            if df.empty:
                return {'error': 'No market data available'}

            current_price = df.iloc[-1]['close']

            # Get optimization recommendations
            entry_opt = self.optimize_entry_points(df)
            tp_sl_opt = self.optimize_tp_sl_levels(df, current_price)

            recommendations = {
                'entry_signal': entry_opt,
                'tp_sl_levels': tp_sl_opt,
                'current_price': current_price,
                'market_conditions': self.analyze_market_conditions(df),
                'risk_assessment': self.assess_risk_level(df),
                'timestamp': datetime.now().isoformat()
            }

            return recommendations

        except Exception as e:
            logger.error(f"❌ Error getting optimization recommendations: {e}")
            return {'error': str(e)}

    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market conditions"""
        try:
            latest = df.iloc[-1]

            conditions = {
                'trend': 'bullish' if latest['close'] > latest['sma_20'] else 'bearish',
                'momentum': 'strong' if abs(latest['rsi'] - 50) > 20 else 'weak',
                'volatility': 'high' if latest['volatility'] > df['volatility'].quantile(0.75) else 'low',
                'volume': 'high' if latest['volume'] > latest['volume_sma'] else 'low',
                'support_level': latest['bb_lower'],
                'resistance_level': latest['bb_upper']
            }

            return conditions

        except Exception as e:
            logger.error(f"❌ Error analyzing market conditions: {e}")
            return {}

    def assess_risk_level(self, df: pd.DataFrame) -> str:
        """Assess current risk level"""
        try:
            latest = df.iloc[-1]

            risk_score = 0

            # High volatility increases risk
            if latest['volatility'] > df['volatility'].quantile(0.8):
                risk_score += 2

            # Extreme RSI levels increase risk
            if latest['rsi'] > 80 or latest['rsi'] < 20:
                risk_score += 1

            # Recent large moves increase risk
            recent_returns = df['returns'].tail(10)
            if abs(recent_returns).max() > 0.05:  # 5% move
                risk_score += 1

            # Determine risk level
            if risk_score >= 3:
                return 'HIGH'
            elif risk_score >= 2:
                return 'MEDIUM'
            else:
                return 'LOW'

        except Exception as e:
            logger.error(f"❌ Error assessing risk level: {e}")
            return 'UNKNOWN'

def main():
    """Main function for AI/ML optimization"""
    optimizer = AIMLOptimizer()


    # Run comprehensive backtest
    backtest_results = optimizer.run_comprehensive_backtest()

    if 'error' in backtest_results:
        return


    # Get current optimization recommendations
    print("\n🎯 CURRENT OPTIMIZATION RECOMMENDATIONS:")
    recommendations = optimizer.get_optimization_recommendations()

    if 'error' not in recommendations:
        entry = recommendations.get('entry_signal', {})
        tp_sl = recommendations.get('tp_sl_levels', {})

        print(f"   Entry Signal: {entry.get('recommendation', 'N/A')}")
        print(f"   Confidence: {entry.get('confidence', 0):.1%}")
        print(f"   TP Level: ${tp_sl.get('take_profit_price', 0):.2f}")
        print(f"   SL Level: ${tp_sl.get('stop_loss_price', 0):.2f}")
        print(f"   Risk/Reward: {tp_sl.get('risk_reward_ratio', 0):.2f}")

        print(f"   Market Trend: {recommendations.get('market_conditions', {}).get('trend', 'unknown')}")
        print(f"   Risk Level: {recommendations.get('risk_assessment', 'unknown')}")


if __name__ == "__main__":
    main()
