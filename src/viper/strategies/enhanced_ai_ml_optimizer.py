#!/usr/bin/env python3
"""
🚀 ENHANCED VIPER AI/ML Trading Optimizer
Advanced optimization with ensemble methods and sophisticated feature engineering

This enhanced version includes:
- Ensemble ML models for better predictions
- Advanced feature engineering with 50+ technical indicators
- Multi-timeframe analysis integration
- Hyperparameter optimization
- Market regime detection
- Performance monitoring and adaptive learning
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import ta  # Technical Analysis library
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ENHANCED_AI_ML - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedAIMLOptimizer:
    """Enhanced AI/ML Optimizer with ensemble methods and advanced features"""

    def __init__(self):
        # Use environment variables for optimal configuration management
        import os
        self.api_server_url = os.getenv('API_SERVER_URL', "http://localhost:8000")
        self.backtester_url = os.getenv('BACKTESTER_URL', "http://localhost:8001")
        self.risk_manager_url = os.getenv('RISK_MANAGER_URL', "http://localhost:8002")
        self.exchange_url = os.getenv('EXCHANGE_URL', "http://localhost:8005")
        self.mcp_server_url = os.getenv('MCP_SERVER_URL', "http://localhost:8015")

        # Ensemble ML Models
        self.entry_ensemble = None
        self.tp_sl_ensemble = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = SelectKBest(score_func=f_regression, k=30)

        # Enhanced optimization parameters
        self.optimization_params = {
            'entry_thresholds': np.linspace(0.1, 0.9, 9),
            'stop_loss_levels': np.linspace(0.005, 0.05, 10),
            'take_profit_levels': np.linspace(0.01, 0.15, 15),
            'trailing_stop_levels': np.linspace(0.005, 0.03, 6),
            'position_sizes': np.linspace(0.01, 0.1, 10),
            'confidence_thresholds': np.linspace(0.6, 0.95, 8),
            'volatility_multipliers': np.linspace(0.5, 2.0, 16),
        }

        # Advanced market regime detection
        self.market_regime_config = {
            'trending_threshold': 0.7,
            'ranging_threshold': 0.3,
            'high_volatility_threshold': 0.8,
            'low_volatility_threshold': 0.2,
        }

        # Performance tracking
        self.model_performance = {
            'training_sessions': 0,
            'best_score': 0.0,
            'last_training_time': None,
            'feature_importance': {},
            'model_versions': []
        }

        logger.info("🤖 Enhanced VIPER AI/ML Optimizer initialized with ensemble methods")

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive set of technical features"""
        try:
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Multiple timeframe moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

            # RSI variations
            df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
            df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            df['rsi_21'] = ta.momentum.rsi(df['close'], window=21)

            # MACD variations
            macd_params = [(8, 21, 5), (12, 26, 9), (19, 39, 9)]
            for fast, slow, signal in macd_params:
                macd = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
                df[f'macd_{fast}_{slow}'] = macd.macd()
                df[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
                df[f'macd_diff_{fast}_{slow}'] = macd.macd_diff()

            # Bollinger Bands with different standard deviations
            for std_dev in [1.5, 2.0, 2.5]:
                bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=std_dev)
                df[f'bb_upper_{std_dev}'] = bollinger.bollinger_hband()
                df[f'bb_lower_{std_dev}'] = bollinger.bollinger_lband()
                df[f'bb_width_{std_dev}'] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / bollinger.bollinger_mavg()

            # Volume features
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['volume_trend'] = df['volume_sma_20'].pct_change(5)

            # Volatility measures
            df['volatility_10'] = df['returns'].rolling(window=10).std()
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            df['volatility_50'] = df['returns'].rolling(window=50).std()

            # Trend strength indicators
            df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].shift(20)
            df['trend_slope'] = (df['sma_20'] - df['sma_20'].shift(5)) / 5
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            df['momentum_20'] = df['close'] - df['close'].shift(20)

            # Support/Resistance levels
            df['support_20'] = df['low'].rolling(window=20).min()
            df['resistance_20'] = df['high'].rolling(window=20).max()
            df['support_resistance_ratio'] = (df['close'] - df['support_20']) / (df['resistance_20'] - df['support_20'])

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            # Williams %R
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])

            # Commodity Channel Index
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])

            # Average True Range
            df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            df['atr_21'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=21)

            # Rate of Change
            df['roc_5'] = ta.momentum.roc(df['close'], window=5)
            df['roc_10'] = ta.momentum.roc(df['close'], window=10)
            df['roc_20'] = ta.momentum.roc(df['close'], window=20)

            # Price patterns
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['range_ratio'] = (df['high'] - df['low']) / df['close']
            df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
            df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)

            # Cross-sectional features
            df['ema_20_vs_50'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
            df['macd_vs_signal'] = df['macd_12_26'] - df['macd_signal_12_26']
            df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].shift(5)

            # Advanced momentum indicators
            df['trix'] = ta.trend.trix(df['close'])
            df['tsi'] = ta.momentum.tsi(df['close'])

            # Fill NaN values
            df = df.fillna(method='forward').fillna(0)

            return df

        except Exception as e:
            logger.error(f"❌ Error creating advanced features: {e}")
            return df

    def detect_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime"""
        try:
            latest = df.iloc[-1]

            # Trend strength analysis
            trend_strength = latest.get('trend_strength', 0.5)
            trend_slope = abs(latest.get('trend_slope', 0)) * 100

            # Volatility analysis
            volatility_20 = latest.get('volatility_20', 0.02)
            volatility_trend = df['volatility_20'].tail(10).mean()

            # Volume analysis
            volume_ratio = latest.get('volume_ratio', 1.0)

            # Determine regime
            regime = {
                'is_trending': trend_strength > self.market_regime_config['trending_threshold'],
                'is_high_volatility': volatility_20 > df['volatility_20'].quantile(0.8),
                'is_low_volatility': volatility_20 < df['volatility_20'].quantile(0.2),
                'trend_direction': 'bullish' if latest.get('trend_slope', 0) > 0 else 'bearish',
                'volatility_regime': 'high' if volatility_20 > df['volatility_20'].quantile(0.75) else 'normal',
                'volume_regime': 'high' if volume_ratio > 1.5 else 'normal',
                'regime_confidence': min(trend_strength + (1 - volatility_20 * 50), 1.0)
            }

            return regime

        except Exception as e:
            logger.error(f"❌ Error detecting market regime: {e}")
            return {'regime': 'unknown', 'confidence': 0.5}

    def prepare_enhanced_ml_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced ML feature preparation with market regime awareness"""
        try:
            # Create advanced features
            df_enhanced = self.create_advanced_features(df)

            # Detect market regime
            regime_info = self.detect_market_regime(df_enhanced)

            # Create multi-horizon targets
            df_enhanced['future_return_1'] = df_enhanced['close'].shift(-1) / df_enhanced['close'] - 1
            df_enhanced['future_return_5'] = df_enhanced['close'].shift(-5) / df_enhanced['close'] - 1
            df_enhanced['future_return_10'] = df_enhanced['close'].shift(-10) / df_enhanced['close'] - 1

            # Enhanced targets
            df_enhanced['optimal_tp'] = df_enhanced['close'] * (1 + df_enhanced['future_return_5'].clip(upper=0.15))
            df_enhanced['optimal_sl'] = df_enhanced['close'] * (1 - 0.02)

            # Multi-signal targets
            df_enhanced['entry_signal'] = ((df_enhanced['future_return_1'] > 0.002) &
                                         (df_enhanced['future_return_5'] > 0.01)).astype(int)

            # Remove NaN values
            df_clean = df_enhanced.dropna()

            if len(df_clean) < 200:
                logger.warning("⚠️ Insufficient data for enhanced ML training")
                return np.array([]), np.array([])

            # Get all feature columns
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                          'future_return_1', 'future_return_5', 'future_return_10',
                          'optimal_tp', 'optimal_sl', 'entry_signal']

            feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

            # Prepare features and targets
            X = df_clean[feature_cols].values
            y_entry = df_clean['entry_signal'].values
            y_tp_sl = df_clean[['optimal_tp', 'optimal_sl']].values

            # Advanced feature scaling with outlier handling
            X_scaled = self._advanced_feature_scaling(X)

            # Feature selection
            X_selected = self.feature_selector.fit_transform(X_scaled, y_entry)

            logger.info(f"🎯 Enhanced ML features prepared: {len(X_selected)} samples, {X_selected.shape[1]} selected features")
            logger.info(f"📊 Market regime detected: {regime_info}")

            return X_selected, y_entry, y_tp_sl

        except Exception as e:
            logger.error(f"❌ Error preparing enhanced ML features: {e}")
            return np.array([]), np.array([]), np.array([])

    def _advanced_feature_scaling(self, X: np.ndarray) -> np.ndarray:
        """Advanced feature scaling with outlier handling"""
        try:
            # Remove outliers using IQR method
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Clip outliers
            X_clipped = np.clip(X, lower_bound, upper_bound)

            # Scale features
            X_scaled = self.scaler.fit_transform(X_clipped)

            return X_scaled

        except Exception as e:
            logger.error(f"❌ Error in advanced scaling: {e}")
            return self.scaler.fit_transform(X)

    def create_ensemble_models(self) -> bool:
        """Create ensemble models for better predictions"""
        try:
            # Base models for entry prediction
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )

            gb_model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )

            et_model = ExtraTreesRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )

            # Voting ensemble for entry prediction
            self.entry_ensemble = VotingRegressor([
                ('rf', rf_model),
                ('gb', gb_model),
                ('et', et_model)
            ])

            # Stacking ensemble for TP/SL prediction
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('et', ExtraTreesRegressor(n_estimators=100, random_state=42))
            ]

            self.tp_sl_ensemble = StackingRegressor(
                estimators=base_models,
                final_estimator=GradientBoostingRegressor(n_estimators=50, random_state=42)
            )

            logger.info("✅ Ensemble models created successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error creating ensemble models: {e}")
            return False

    def train_enhanced_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train enhanced ensemble models"""
        try:
            logger.info("🚀 Starting enhanced model training...")

            # Prepare enhanced features
            X, y_entry, y_tp_sl = self.prepare_enhanced_ml_features(df)

            if len(X) == 0:
                return {'error': 'No data for training'}

            # Create ensemble models
            if not self.create_ensemble_models():
                return {'error': 'Failed to create ensemble models'}

            # Split data
            X_train, X_test, y_entry_train, y_entry_test, y_tp_sl_train, y_tp_sl_test = train_test_split(
                X, y_entry, y_tp_sl, test_size=0.2, random_state=42
            )

            # Train entry ensemble
            logger.info("🎯 Training entry ensemble...")
            self.entry_ensemble.fit(X_train, y_entry_train)

            # Train TP/SL ensemble
            logger.info("🎯 Training TP/SL ensemble...")
            self.tp_sl_ensemble.fit(X_train, y_tp_sl_train)

            # Evaluate models
            entry_pred = self.entry_ensemble.predict(X_test)
            tp_sl_pred = self.tp_sl_ensemble.predict(X_test)

            entry_r2 = r2_score(y_entry_test, entry_pred)
            entry_mae = mean_absolute_error(y_entry_test, entry_pred)
            tp_sl_r2 = r2_score(y_tp_sl_test, tp_sl_pred)
            tp_sl_mae = mean_absolute_error(y_tp_sl_test, tp_sl_pred)

            # Feature importance
            if hasattr(self.entry_ensemble, 'feature_importances_'):
                feature_importance = dict(zip(range(len(self.feature_selector.get_support())), self.entry_ensemble.feature_importances_))
            else:
                # Get feature importance from RandomForest in ensemble
                rf_model = self.entry_ensemble.named_estimators_['rf']
                feature_importance = dict(zip(range(len(self.feature_selector.get_support())), rf_model.feature_importances_))

            # Update performance tracking
            self.model_performance.update({
                'training_sessions': self.model_performance['training_sessions'] + 1,
                'best_score': max(self.model_performance['best_score'], entry_r2),
                'last_training_time': datetime.now(),
                'feature_importance': feature_importance
            })

            results = {
                'entry_model_r2': float(entry_r2),
                'entry_model_mae': float(entry_mae),
                'tp_sl_model_r2': float(tp_sl_r2),
                'tp_sl_model_mae': float(tp_sl_mae),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X.shape[1],
                'model_performance': self.model_performance,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"✅ Enhanced models trained - Entry R²: {entry_r2:.4f}, TP/SL R²: {tp_sl_r2:.4f}")
            return results

        except Exception as e:
            logger.error(f"❌ Error training enhanced models: {e}")
            return {'error': str(e)}

    def optimize_entry_points_enhanced(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced entry point optimization using ensemble models"""
        try:
            if self.entry_ensemble is None:
                logger.error("❌ Entry ensemble not trained")
                return {}

            # Prepare features
            X, _, _ = self.prepare_enhanced_ml_features(df)

            if len(X) == 0:
                return {'error': 'No data for optimization'}

            # Get latest data
            latest_data = X[-1:]

            # Predict entry signal
            entry_signal = self.entry_ensemble.predict(latest_data)[0]

            # Predict TP/SL levels
            tp_sl_predictions = self.tp_sl_ensemble.predict(latest_data)[0]
            optimal_tp = tp_sl_predictions[0]
            optimal_sl = tp_sl_predictions[1]

            # Calculate confidence based on model agreement
            model_predictions = []
            for name, model in self.entry_ensemble.named_estimators_.items():
                pred = model.predict(latest_data)[0]
                model_predictions.append(pred)

            prediction_std = np.std(model_predictions)
            confidence = max(0.1, 1.0 - prediction_std * 2)  # Lower std = higher confidence

            # Market regime adjustment
            regime_info = self.detect_market_regime(df)
            regime_multiplier = 1.0
            if regime_info.get('is_high_volatility'):
                regime_multiplier = 0.8  # Reduce confidence in high volatility
            elif regime_info.get('is_trending'):
                regime_multiplier = 1.1  # Increase confidence in trending markets

            confidence *= regime_multiplier
            confidence = np.clip(confidence, 0.0, 1.0)

            # Generate recommendation
            if confidence > 0.7:
                if entry_signal > 0.6:
                    recommendation = 'STRONG_BUY'
                elif entry_signal > 0.5:
                    recommendation = 'BUY'
                elif entry_signal < 0.4:
                    recommendation = 'STRONG_SELL'
                elif entry_signal < 0.5:
                    recommendation = 'SELL'
                else:
                    recommendation = 'HOLD'
            else:
                recommendation = 'HOLD'

            result = {
                'entry_signal': float(entry_signal),
                'optimal_tp': float(optimal_tp),
                'optimal_sl': float(optimal_sl),
                'confidence': float(confidence),
                'recommendation': recommendation,
                'market_regime': regime_info,
                'model_agreement': float(1.0 - prediction_std),
                'ensemble_prediction_std': float(prediction_std),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"🎯 Enhanced Entry Optimization: Signal={entry_signal:.3f}, Confidence={confidence:.3f}, Recommendation={recommendation}")
            return result

        except Exception as e:
            logger.error(f"❌ Error in enhanced entry optimization: {e}")
            return {
                'error': str(e),
                'recommendation': 'HOLD',
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Enhanced AI/ML optimization example"""

    optimizer = EnhancedAIMLOptimizer()

    print("🎯 Features: Ensemble models, advanced features, market regime detection")

if __name__ == "__main__":
    main()
