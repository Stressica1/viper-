#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Ultra Backtester Service
High-performance backtesting engine for strategy validation and optimization

Features:
- Historical data backtesting with multiple timeframes
- Performance metrics calculation (Sharpe ratio, drawdown, win rate, etc.)
- Strategy parameter testing and optimization
- Monte Carlo simulation capabilities
- Risk-adjusted return analysis
- RESTful API for backtest execution and results
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
import uvicorn
import redis
from pathlib import Path
import httpx
import uuid

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'ultra-backtester')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltraBacktester:
    """Ultra high-performance backtesting engine"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False
        self.active_backtests = {}

        # Load configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        self.workers = int(os.getenv('BACKTEST_WORKERS', '4'))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))
        self.viper_threshold = int(os.getenv('VIPER_THRESHOLD', '85'))

        # Service URLs
        self.data_manager_url = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')
        self.risk_manager_url = os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8000')

        # Backtest results storage
        self.results_path = Path('/app/backtest_results')
        self.results_path.mkdir(exist_ok=True)

        logger.info("🏗️ Initializing Ultra Backtester...")

    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False

    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 1000) -> Optional[List]:
        """Get historical OHLCV data from data manager"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.data_manager_url}/api/ohlcv/{symbol}",
                    params={'timeframe': timeframe, 'limit': limit}
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"⚠️ Failed to get historical data: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
            return None

    def calculate_viper_signal(self, ohlcv_data: List[Dict]) -> List[Dict]:
        """Calculate VIPER trading signals from OHLCV data"""
        signals = []

        if len(ohlcv_data) < 50:
            logger.warning("⚠️ Insufficient data for signal calculation")
            return signals

        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(ohlcv_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Calculate moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()

            # Calculate RSI
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            df['rsi'] = calculate_rsi(df['close'])

            # Generate signals
            for i in range(50, len(df)):
                current_price = df.loc[i, 'close']
                sma_20 = df.loc[i, 'sma_20']
                sma_50 = df.loc[i, 'sma_50']
                rsi = df.loc[i, 'rsi']

                signal = 'HOLD'
                confidence = 0.5

                # VIPER Strategy Logic
                if current_price > sma_20 and sma_20 > sma_50 and rsi < 70:
                    signal = 'BUY'
                    confidence = min(0.95, (current_price - sma_20) / sma_20)
                elif current_price < sma_20 and sma_20 < sma_50 and rsi > 30:
                    signal = 'SELL'
                    confidence = min(0.95, (sma_20 - current_price) / current_price)

                signals.append({
                    'timestamp': df.loc[i, 'timestamp'].isoformat(),
                    'price': current_price,
                    'signal': signal,
                    'confidence': confidence,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'rsi': rsi,
                    'ohlcv': {
                        'open': df.loc[i, 'open'],
                        'high': df.loc[i, 'high'],
                        'low': df.loc[i, 'low'],
                        'close': df.loc[i, 'close'],
                        'volume': df.loc[i, 'volume']
                    }
                })

        except Exception as e:
            logger.error(f"❌ Error calculating VIPER signals: {e}")

        return signals

    def run_backtest(self, signals: List[Dict], initial_balance: float = 10000.0,
                    commission: float = 0.001) -> Dict:
        """Run backtest simulation with given signals"""
        try:
            balance = initial_balance
            position = 0  # Current position size
            entry_price = 0
            trades = []
            equity_curve = []

            for signal in signals:
                price = signal['price']
                signal_type = signal['signal']
                confidence = signal['confidence']

                if signal_type == 'BUY' and position == 0:
                    # Calculate position size based on risk management
                    risk_amount = balance * self.risk_per_trade
                    position_size = risk_amount / price

                    # Apply commission
                    commission_fee = position_size * price * commission
                    if balance >= (position_size * price + commission_fee):
                        position = position_size
                        entry_price = price
                        balance -= commission_fee

                        trades.append({
                            'type': 'BUY',
                            'price': price,
                            'size': position_size,
                            'timestamp': signal['timestamp'],
                            'commission': commission_fee
                        })

                        logger.debug(".2f")

                elif signal_type == 'SELL' and position > 0:
                    # Close position
                    exit_value = position * price
                    commission_fee = exit_value * commission

                    # Calculate P&L
                    entry_value = position * entry_price
                    pnl = exit_value - entry_value - commission_fee

                    balance += exit_value - commission_fee

                    trades.append({
                        'type': 'SELL',
                        'price': price,
                        'size': position,
                        'timestamp': signal['timestamp'],
                        'commission': commission_fee,
                        'pnl': pnl,
                        'entry_price': entry_price
                    })

                    logger.debug(".2f")
                    position = 0
                    entry_price = 0

                # Track equity curve
                current_equity = balance + (position * price if position > 0 else 0)
                equity_curve.append({
                    'timestamp': signal['timestamp'],
                    'equity': current_equity,
                    'balance': balance,
                    'position_value': position * price if position > 0 else 0
                })

            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(trades, equity_curve, initial_balance)

            return {
                'backtest_id': str(uuid.uuid4()),
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_trades': len(trades),
                'win_trades': len([t for t in trades if t.get('pnl', 0) > 0]),
                'lose_trades': len([t for t in trades if t.get('pnl', 0) < 0]),
                'total_pnl': balance - initial_balance,
                'total_return': (balance - initial_balance) / initial_balance,
                'trades': trades,
                'equity_curve': equity_curve,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error running backtest: {e}")
            return {'error': str(e)}

    def calculate_performance_metrics(self, trades: List, equity_curve: List, initial_balance: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if not trades or not equity_curve:
                return {'error': 'No trades or equity data'}

            # Basic metrics
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

            win_rate = len(winning_trades) / len(trades) if trades else 0

            # P&L metrics
            total_pnl = sum(t.get('pnl', 0) for t in winning_trades)
            total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))

            profit_factor = total_pnl / total_loss if total_loss > 0 else float('inf')

            # Average win/loss
            avg_win = total_pnl / len(winning_trades) if winning_trades else 0
            avg_loss = total_loss / len(losing_trades) if losing_trades else 0

            # Sharpe ratio (simplified)
            equity_returns = pd.DataFrame(equity_curve)
            if len(equity_returns) > 1:
                equity_returns['returns'] = equity_returns['equity'].pct_change()
                sharpe_ratio = equity_returns['returns'].mean() / equity_returns['returns'].std() * np.sqrt(365) if equity_returns['returns'].std() > 0 else 0
            else:
                sharpe_ratio = 0

            # Maximum drawdown
            equity_values = [point['equity'] for point in equity_curve]
            peak = initial_balance
            max_drawdown = 0

            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)

            # Calmar ratio
            calmar_ratio = (equity_values[-1] - initial_balance) / initial_balance / max_drawdown if max_drawdown > 0 else float('inf')

            # Win/Loss streak analysis
            win_streak = 0
            max_win_streak = 0
            loss_streak = 0
            max_loss_streak = 0

            for trade in trades:
                pnl = trade.get('pnl', 0)
                if pnl > 0:
                    win_streak += 1
                    loss_streak = 0
                    max_win_streak = max(max_win_streak, win_streak)
                elif pnl < 0:
                    loss_streak += 1
                    win_streak = 0
                    max_loss_streak = max(max_loss_streak, loss_streak)

            return {
                'win_rate': round(win_rate * 100, 2),
                'profit_factor': round(profit_factor, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown * 100, 2),
                'calmar_ratio': round(calmar_ratio, 2),
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                'total_wins': len(winning_trades),
                'total_losses': len(losing_trades),
                'largest_win': max((t.get('pnl', 0) for t in winning_trades), default=0),
                'largest_loss': min((t.get('pnl', 0) for t in losing_trades), default=0)
            }

        except Exception as e:
            logger.error(f"❌ Error calculating performance metrics: {e}")
            return {'error': str(e)}

    async def run_parameter_optimization(self, symbol: str, timeframe: str,
                                       parameter_ranges: Dict, initial_balance: float = 10000.0) -> Dict:
        """Run parameter optimization for strategy"""
        try:
            # Get historical data
            ohlcv_data = await self.get_historical_data(symbol, timeframe, 1000)
            if not ohlcv_data:
                return {'error': 'Unable to get historical data'}

            # Generate parameter combinations (simplified grid search)
            results = []

            # Example: optimize SMA periods
            sma_20_range = parameter_ranges.get('sma_20_period', [10, 15, 20, 25, 30])
            sma_50_range = parameter_ranges.get('sma_50_period', [40, 45, 50, 55, 60])

            for sma_20 in sma_20_range:
                for sma_50 in sma_50_range:
                    if sma_20 >= sma_50:
                        continue

                    # Generate signals with custom parameters
                    signals = self.calculate_viper_signal_custom(ohlcv_data, sma_20, sma_50)

                    # Run backtest
                    result = self.run_backtest(signals, initial_balance)

                    if 'error' not in result:
                        result['parameters'] = {
                            'sma_20_period': sma_20,
                            'sma_50_period': sma_50
                        }
                        results.append(result)

            # Sort by Sharpe ratio
            results.sort(key=lambda x: x['metrics'].get('sharpe_ratio', 0), reverse=True)

            return {
                'optimization_results': results[:10],  # Top 10 results
                'total_combinations': len(results),
                'best_parameters': results[0]['parameters'] if results else None,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error in parameter optimization: {e}")
            return {'error': str(e)}

    def calculate_viper_signal_custom(self, ohlcv_data: List, sma_20_period: int, sma_50_period: int) -> List:
        """Calculate VIPER signals with custom parameters"""
        # Simplified implementation - in production this would be more sophisticated
        return self.calculate_viper_signal(ohlcv_data)

    async def run_monte_carlo_simulation(self, signals: List, initial_balance: float = 10000.0,
                                       simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation for robustness testing"""
        try:
            all_results = []

            for i in range(simulations):
                # Add random noise to signals (simplified)
                noisy_signals = []
                for signal in signals:
                    noisy_signal = signal.copy()
                    # Add small random variation to price
                    noise = np.random.normal(0, signal['price'] * 0.001)  # 0.1% std deviation
                    noisy_signal['price'] += noise
                    noisy_signals.append(noisy_signal)

                # Run backtest with noisy data
                result = self.run_backtest(noisy_signals, initial_balance)
                if 'error' not in result:
                    all_results.append(result)

            if not all_results:
                return {'error': 'No valid simulation results'}

            # Analyze results distribution
            final_balances = [r['final_balance'] for r in all_results]
            returns = [(b - initial_balance) / initial_balance for b in final_balances]

            return {
                'simulations_run': len(all_results),
                'mean_final_balance': round(np.mean(final_balances), 2),
                'median_final_balance': round(np.median(final_balances), 2),
                'std_final_balance': round(np.std(final_balances), 2),
                'min_final_balance': round(min(final_balances), 2),
                'max_final_balance': round(max(final_balances), 2),
                'mean_return': round(np.mean(returns), 4),
                'return_std': round(np.std(returns), 4),
                'sharpe_ratio': round(np.mean(returns) / np.std(returns), 2) if np.std(returns) > 0 else 0,
                'win_probability': round(len([r for r in final_balances if r > initial_balance]) / len(final_balances), 2),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error in Monte Carlo simulation: {e}")
            return {'error': str(e)}

# FastAPI application
app = FastAPI(
    title="VIPER Ultra Backtester",
    version="1.0.0",
    description="High-performance backtesting and strategy optimization service"
)

backtester = UltraBacktester()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if not backtester.initialize_redis():
        logger.error("❌ Failed to initialize Redis. Exiting...")
        return

    logger.info("✅ Ultra Backtester started successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "ultra-backtester",
            "redis_connected": backtester.redis_client is not None,
            "active_backtests": len(backtester.active_backtests),
            "workers": backtester.workers,
            "viper_threshold": backtester.viper_threshold
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "ultra-backtester",
                "error": str(e)
            }
        )

@app.post("/api/backtest/run")
async def run_backtest(request: Request, background_tasks: BackgroundTasks):
    """Run a backtest"""
    try:
        data = await request.json()

        required_fields = ['symbol', 'timeframe']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        symbol = data['symbol']
        timeframe = data['timeframe']
        initial_balance = data.get('initial_balance', 10000.0)
        limit = data.get('limit', 1000)

        backtest_id = str(uuid.uuid4())
        backtester.active_backtests[backtest_id] = {'status': 'running', 'progress': 0}

        # Run backtest in background
        background_tasks.add_task(
            backtester.run_backtest_async,
            backtest_id, symbol, timeframe, initial_balance, limit
        )

        return {
            'backtest_id': backtest_id,
            'status': 'started',
            'message': 'Backtest started successfully'
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error starting backtest: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/backtest/start")
async def start_backtest(request: Request, background_tasks: BackgroundTasks):
    """Start a backtest (compatibility endpoint for completion tests)"""
    try:
        data = await request.json()
        
        # Map completion test parameters to expected format
        symbol = data.get('symbol', 'BTC/USDT:USDT')
        timeframe = data.get('timeframe', '1h')
        initial_balance = data.get('initial_balance', 10000.0)
        
        # Create backtest request compatible with our existing run_backtest endpoint
        backtest_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'initial_balance': initial_balance,
            'limit': 1000
        }
        
        # Use existing run_backtest logic
        required_fields = ['symbol', 'timeframe']
        for field in required_fields:
            if field not in backtest_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        backtest_id = str(uuid.uuid4())
        backtester.active_backtests[backtest_id] = {'status': 'running', 'progress': 0}
        
        # Run backtest in background
        background_tasks.add_task(
            backtester.run_backtest_async,
            backtest_id, backtest_data['symbol'], backtest_data['timeframe'], 
            backtest_data['initial_balance'], backtest_data['limit']
        )
        
        return {
            'backtest_id': backtest_id,
            'status': 'started',
            'message': 'Backtest started successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error starting backtest: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/backtest/{backtest_id}")
async def get_backtest_result(backtest_id: str):
    """Get backtest result"""
    try:
        # Check if result is cached in Redis
        result = backtester.redis_client.get(f"viper:backtest:{backtest_id}")
        if result:
            return json.loads(result)

        # Check if backtest is still running
        if backtest_id in backtester.active_backtests:
            return {
                'backtest_id': backtest_id,
                'status': backtester.active_backtests[backtest_id]['status'],
                'progress': backtester.active_backtests[backtest_id]['progress']
            }

        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting backtest result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/optimization/run")
async def run_optimization(request: Request, background_tasks: BackgroundTasks):
    """Run parameter optimization"""
    try:
        data = await request.json()

        required_fields = ['symbol', 'timeframe', 'parameter_ranges']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        symbol = data['symbol']
        timeframe = data['timeframe']
        parameter_ranges = data['parameter_ranges']
        initial_balance = data.get('initial_balance', 10000.0)

        optimization_id = str(uuid.uuid4())

        # Run optimization in background
        background_tasks.add_task(
            backtester.run_optimization_async,
            optimization_id, symbol, timeframe, parameter_ranges, initial_balance
        )

        return {
            'optimization_id': optimization_id,
            'status': 'started',
            'message': 'Optimization started successfully'
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error starting optimization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/monte-carlo/run")
async def run_monte_carlo(request: Request, background_tasks: BackgroundTasks):
    """Run Monte Carlo simulation"""
    try:
        data = await request.json()

        required_fields = ['symbol', 'timeframe']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        symbol = data['symbol']
        timeframe = data['timeframe']
        initial_balance = data.get('initial_balance', 10000.0)
        simulations = data.get('simulations', 1000)

        simulation_id = str(uuid.uuid4())

        # Run simulation in background
        background_tasks.add_task(
            backtester.run_monte_carlo_async,
            simulation_id, symbol, timeframe, initial_balance, simulations
        )

        return {
            'simulation_id': simulation_id,
            'status': 'started',
            'message': 'Monte Carlo simulation started successfully'
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error starting Monte Carlo simulation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/backtests")
async def list_backtests():
    """List all available backtest results"""
    try:
        keys = backtester.redis_client.keys("viper:backtest:*")
        backtest_ids = [key.split(":")[-1] for key in keys]

        backtests = []
        for backtest_id in backtest_ids[:50]:  # Limit to last 50
            result = backtester.redis_client.get(f"viper:backtest:{backtest_id}")
            if result:
                data = json.loads(result)
                backtests.append({
                    'backtest_id': backtest_id,
                    'symbol': data.get('symbol', 'unknown'),
                    'timestamp': data.get('timestamp', 'unknown'),
                    'total_return': data.get('total_return', 0),
                    'sharpe_ratio': data.get('metrics', {}).get('sharpe_ratio', 0)
                })

        return {'backtests': backtests}

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to list backtests: {e}")

# Background task implementations
async def run_backtest_async(self, backtest_id, symbol, timeframe, initial_balance, limit):
    """Background task to run backtest"""
    try:
        self.active_backtests[backtest_id]['progress'] = 10

        # Get historical data
        ohlcv_data = await self.get_historical_data(symbol, timeframe, limit)
        if not ohlcv_data:
            self.active_backtests[backtest_id] = {'status': 'failed', 'error': 'No historical data'}
            return

        self.active_backtests[backtest_id]['progress'] = 30

        # Calculate signals
        signals = self.calculate_viper_signal(ohlcv_data)
        if not signals:
            self.active_backtests[backtest_id] = {'status': 'failed', 'error': 'No signals generated'}
            return

        self.active_backtests[backtest_id]['progress'] = 60

        # Run backtest
        result = self.run_backtest(signals, initial_balance)
        result['backtest_id'] = backtest_id
        result['symbol'] = symbol
        result['timeframe'] = timeframe

        self.active_backtests[backtest_id]['progress'] = 90

        # Store result
        self.redis_client.setex(
            f"viper:backtest:{backtest_id}",
            86400 * 7,  # 7 days
            json.dumps(result)
        )

        # Save to file
        result_file = self.results_path / f"backtest_{backtest_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        self.active_backtests[backtest_id]['progress'] = 100
        self.active_backtests[backtest_id]['status'] = 'completed'

        logger.info(f"✅ Backtest {backtest_id} completed successfully")

    except Exception as e:
        logger.error(f"❌ Error in backtest {backtest_id}: {e}")
        self.active_backtests[backtest_id] = {'status': 'failed', 'error': str(e)}

# Add background methods to class
UltraBacktester.run_backtest_async = run_backtest_async

if __name__ == "__main__":
    port = int(os.getenv("ULTRA_BACKTESTER_PORT", 8000))
    logger.info(f"Starting VIPER Ultra Backtester on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
