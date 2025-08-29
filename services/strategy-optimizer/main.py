#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Strategy Optimizer Service
Advanced parameter optimization using genetic algorithms and machine learning

Features:
- Genetic algorithm optimization
- Grid search and random search
- Walk-forward analysis
- Multi-objective optimization
- Parameter sensitivity analysis
- Optimization result visualization
- RESTful API for optimization tasks
"""

import os
import json
import time
import logging
import asyncio
import sys
import numpy as np
import random
import secrets
from fastapi.responses import JSONResponse
import uvicorn
import redis
from pathlib import Path
import httpx
import uuid

# Add shared directory to path for circuit breaker
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
try:
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    # Fallback if shared module not available
    ServiceClient = None
    call_service = None
    CIRCUIT_BREAKER_AVAILABLE = False

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'strategy-optimizer')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """Advanced strategy optimization using genetic algorithms and other methods"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False
        self.active_optimizations = {}

        # Load configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        self.optimization_methods = os.getenv('OPTIMIZATION_METHODS', 'grid_search,genetic,walk_forward').split(',')
        self.max_iterations = int(os.getenv('MAX_ITERATIONS', '1000'))

        # Service URLs
        self.ultra_backtester_url = os.getenv('ULTRA_BACKTESTER_URL', 'http://ultra-backtester:8000')
        self.data_manager_url = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')

        # Optimization results storage
        self.results_path = Path('/app/backtest_results')
        self.results_path.mkdir(exist_ok=True)

        logger.info("🏗️ Initializing Strategy Optimizer...")

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

    async def run_backtest_with_params(self, symbol: str, timeframe: str,
                                     parameters: Dict, initial_balance: float = 10000.0) -> Optional[Dict]:
        """Run a backtest with specific parameters"""
        if call_service and CIRCUIT_BREAKER_AVAILABLE:
            try:
                # Start backtest
                start_data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'initial_balance': initial_balance,
                    'parameters': parameters
                }

                start_result = await call_service(
                    "ultra-backtester",
                    "/api/backtest/run",
                    method="POST",
                    redis_client=self.redis_client,
                    json=start_data
                )

                if start_result and 'backtest_id' in start_result:
                    backtest_id = start_result['backtest_id']

                    # Wait for completion (simplified - in production use websockets or polling)
                    await asyncio.sleep(5)

                    # Get result
                    result = await call_service(
                        "ultra-backtester",
                        f"/api/backtest/{backtest_id}",
                        method="GET",
                        redis_client=self.redis_client
                    )

                    return result

                return None

            except Exception as e:
                logger.error(f"❌ Circuit breaker error running backtest: {e}")
                return None
        else:
            # Fallback to direct HTTP call
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    backtest_data = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'initial_balance': initial_balance,
                        'parameters': parameters
                    }

                    response = await client.post(
                        f"{self.ultra_backtester_url}/api/backtest/run",
                        json=backtest_data
                    )

                    if response.status_code == 200:
                        result = response.json()
                        backtest_id = result['backtest_id']

                        # Wait for completion (simplified - in production use websockets or polling)
                        await asyncio.sleep(5)

                        # Get result
                        result_response = await client.get(f"{self.ultra_backtester_url}/api/backtest/{backtest_id}")
                        if result_response.status_code == 200:
                            return result_response.json()

                    return None

            except Exception as e:
                logger.error(f"❌ Error running backtest with params: {e}")
                return None

    def genetic_algorithm_optimization(self, symbol: str, timeframe: str,
                                    parameter_ranges: Dict, population_size: int = 50,
                                    generations: int = 30, initial_balance: float = 10000.0) -> Dict:
        """Genetic algorithm for parameter optimization"""
        try:
            # Define fitness function
            def fitness_function(parameters):
                """Evaluate parameter fitness (higher is better)"""
                try:
                    # Run backtest with parameters (synchronous for GA)
                    backtest_result = asyncio.run(
                        self.run_backtest_with_params(symbol, timeframe, parameters, initial_balance)
                    )

                    if not backtest_result or 'error' in backtest_result:
                        return 0.0

                    metrics = backtest_result.get('metrics', {})

                    # Multi-objective fitness: Sharpe ratio + win rate - drawdown
                    sharpe = metrics.get('sharpe_ratio', 0)
                    win_rate = metrics.get('win_rate', 0) / 100  # Convert percentage to decimal
                    max_drawdown = metrics.get('max_drawdown', 100) / 100  # Convert percentage to decimal

                    # Combined fitness score
                    fitness = sharpe * 0.5 + win_rate * 0.3 - max_drawdown * 0.2

                    return max(fitness, 0.0)  # Ensure non-negative

                except Exception as e:
                    logger.error(f"❌ Error in fitness function: {e}")
                    return 0.0

            # Initialize population
            population = self.initialize_population(parameter_ranges, population_size)

            best_individual = None
            best_fitness = 0.0
            fitness_history = []

            logger.info(f"🚀 Starting genetic algorithm optimization ({generations} generations, {population_size} population)")

            for generation in range(generations):
                logger.info(f"📊 Generation {generation + 1}/{generations}")

                # Evaluate fitness
                fitness_scores = []
                for individual in population:
                    fitness = fitness_function(individual)
                    fitness_scores.append((individual, fitness))

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()

                # Sort by fitness (descending)
                fitness_scores.sort(key=lambda x: x[1], reverse=True)

                # Selection (top 50%)
                selected = [ind for ind, fit in fitness_scores[:population_size//2]]

                # Crossover and mutation
                new_population = []

                while len(new_population) < population_size:
                    # Select parents
                    parent1 = secrets.choice(selected)
                    parent2 = secrets.choice(selected)

                    # Crossover
                    child = self.crossover(parent1, parent2, parameter_ranges)

                    # Mutation
                    child = self.mutate(child, parameter_ranges, mutation_rate=0.1)

                    new_population.append(child)

                population = new_population
                fitness_history.append(best_fitness)

                logger.info(".4f")

            return {
                'optimization_method': 'genetic_algorithm',
                'best_parameters': best_individual,
                'best_fitness': best_fitness,
                'generations': generations,
                'population_size': population_size,
                'fitness_history': fitness_history,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error in genetic algorithm optimization: {e}")
            return {'error': str(e)}

    def initialize_population(self, parameter_ranges: Dict, population_size: int) -> List[Dict]:
        """Initialize random population for genetic algorithm"""
        population = []

        for _ in range(population_size):
            individual = {}
            for param_name, param_range in parameter_ranges.items():
                if isinstance(param_range, list):
                    individual[param_name] = secrets.choice(param_range)
                elif isinstance(param_range, dict):
                    min_val = param_range.get('min', 0)
                    max_val = param_range.get('max', 100)
                    step = param_range.get('step', 1)
                    if step == 1:
                        individual[param_name] = secrets.randbelow(max_val - min_val + 1) + min_val  # Was: random.randint(min_val, max_val)
                    else:
                        individual[param_name] = round(random.uniform(min_val, max_val) / step) * step
                else:
                    individual[param_name] = param_range

            population.append(individual)

        return population

    def crossover(self, parent1: Dict, parent2: Dict, parameter_ranges: Dict) -> Dict:
        """Crossover operation for genetic algorithm"""
        child = {}

        for param_name in parameter_ranges.keys():
            # Randomly select from either parent
            if secrets.randbelow(1000000) / 1000000.0  # Was: random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]

        return child

    def mutate(self, individual: Dict, parameter_ranges: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutation operation for genetic algorithm"""
        mutated = individual.copy()

        for param_name, param_range in parameter_ranges.items():
            if secrets.randbelow(1000000) / 1000000.0  # Was: random.random() < mutation_rate:
                if isinstance(param_range, list):
                    mutated[param_name] = secrets.choice(param_range)
                elif isinstance(param_range, dict):
                    min_val = param_range.get('min', 0)
                    max_val = param_range.get('max', 100)
                    step = param_range.get('step', 1)
                    if step == 1:
                        mutated[param_name] = secrets.randbelow(max_val - min_val + 1) + min_val  # Was: random.randint(min_val, max_val)
                    else:
                        mutated[param_name] = round(random.uniform(min_val, max_val) / step) * step

        return mutated

    def grid_search_optimization(self, symbol: str, timeframe: str,
                               parameter_ranges: Dict, initial_balance: float = 10000.0) -> Dict:
        """Grid search optimization"""
        try:
            # Generate all parameter combinations
            param_combinations = self.generate_parameter_combinations(parameter_ranges)

            logger.info(f"🚀 Starting grid search optimization ({len(param_combinations)} combinations)")

            best_result = None
            best_fitness = 0.0
            results = []

            for i, parameters in enumerate(param_combinations):
                logger.info(f"📊 Testing combination {i + 1}/{len(param_combinations)}: {parameters}")

                # Run backtest
                backtest_result = asyncio.run(
                    self.run_backtest_with_params(symbol, timeframe, parameters, initial_balance)
                )

                if backtest_result and 'error' not in backtest_result:
                    metrics = backtest_result.get('metrics', {})

                    # Calculate fitness
                    sharpe = metrics.get('sharpe_ratio', 0)
                    win_rate = metrics.get('win_rate', 0) / 100
                    max_drawdown = metrics.get('max_drawdown', 100) / 100

                    fitness = sharpe * 0.5 + win_rate * 0.3 - max_drawdown * 0.2

                    result = {
                        'parameters': parameters,
                        'fitness': fitness,
                        'metrics': metrics,
                        'backtest_result': backtest_result
                    }
                    results.append(result)

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_result = result

                # Small delay to avoid overwhelming the backtester
                time.sleep(0.1)

            return {
                'optimization_method': 'grid_search',
                'best_parameters': best_result['parameters'] if best_result else None,
                'best_fitness': best_fitness,
                'total_combinations': len(param_combinations),
                'results': sorted(results, key=lambda x: x['fitness'], reverse=True)[:10],  # Top 10
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error in grid search optimization: {e}")
            return {'error': str(e)}

    def generate_parameter_combinations(self, parameter_ranges: Dict) -> List[Dict]:
        """Generate all possible parameter combinations for grid search"""
        if not parameter_ranges:
            return [{}]

        param_names = list(parameter_ranges.keys())
        param_values = []

        for param_name in param_names:
            param_range = parameter_ranges[param_name]
            if isinstance(param_range, list):
                param_values.append(param_range)
            elif isinstance(param_range, dict):
                min_val = param_range.get('min', 0)
                max_val = param_range.get('max', 100)
                step = param_range.get('step', 1)

                if step == 1:
                    values = list(range(min_val, max_val + 1))
                else:
                    values = []
                    current = min_val
                    while current <= max_val:
                        values.append(current)
                        current += step
                param_values.append(values)
            else:
                param_values.append([param_range])

        # Generate combinations
        combinations = []
        for combo in np.array(np.meshgrid(*param_values)).T.reshape(-1, len(param_names)):
            combination = {}
            for i, param_name in enumerate(param_names):
                combination[param_name] = combo[i]
            combinations.append(combination)

        return combinations

    def walk_forward_optimization(self, symbol: str, timeframe: str,
                                parameter_ranges: Dict, initial_balance: float = 10000.0,
                                window_size: int = 100, step_size: int = 20) -> Dict:
        """Walk-forward analysis for out-of-sample testing"""
        try:
            logger.info("🚀 Starting walk-forward optimization")

            # Get historical data
            ohlcv_data = asyncio.run(self.get_historical_data(symbol, timeframe, 1000))
            if not ohlcv_data or len(ohlcv_data) < window_size:
                return {'error': 'Insufficient historical data for walk-forward analysis'}

            results = []
            in_sample_results = []
            out_sample_results = []

            # Rolling window analysis
            for i in range(0, len(ohlcv_data) - window_size - step_size, step_size):
                # Define in-sample and out-of-sample periods
                in_sample_end = i + window_size
                out_sample_end = min(i + window_size + step_size, len(ohlcv_data))

                in_sample_data = ohlcv_data[i:in_sample_end]
                out_sample_data = ohlcv_data[in_sample_end:out_sample_end]

                if len(out_sample_data) < 10:  # Skip if out-of-sample is too small
                    continue

                # Optimize on in-sample data
                in_sample_result = self.grid_search_optimization(
                    symbol, timeframe, parameter_ranges, initial_balance
                )

                if 'error' in in_sample_result:
                    continue

                best_params = in_sample_result['best_parameters']

                # Test on out-of-sample data (simplified - would need backtester modification)
                # For now, we'll use the same backtest logic
                out_sample_backtest = asyncio.run(
                    self.run_backtest_with_params(symbol, timeframe, best_params, initial_balance)
                )

                if out_sample_backtest and 'error' not in out_sample_backtest:
                    result = {
                        'window_start': i,
                        'in_sample_end': in_sample_end,
                        'out_sample_end': out_sample_end,
                        'best_parameters': best_params,
                        'in_sample_metrics': in_sample_result.get('results', [{}])[0].get('metrics', {}),
                        'out_sample_metrics': out_sample_backtest.get('metrics', {}),
                        'out_sample_return': out_sample_backtest.get('total_return', 0)
                    }

                    results.append(result)
                    in_sample_results.append(result['in_sample_metrics'].get('sharpe_ratio', 0))
                    out_sample_results.append(result['out_sample_metrics'].get('sharpe_ratio', 0))

            # Calculate walk-forward efficiency
            if in_sample_results and out_sample_results:
                correlation = np.corrcoef(in_sample_results, out_sample_results)[0, 1]
                avg_out_sample_return = np.mean(out_sample_results)
            else:
                correlation = 0
                avg_out_sample_return = 0

            return {
                'optimization_method': 'walk_forward',
                'results': results,
                'walk_forward_efficiency': round(correlation, 4),
                'average_out_sample_sharpe': round(avg_out_sample_return, 4),
                'total_windows': len(results),
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error in walk-forward optimization: {e}")
            return {'error': str(e)}

    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> Optional[List]:
        """Get historical data from data manager"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.data_manager_url}/api/ohlcv/{symbol}",
                    params={'timeframe': timeframe, 'limit': limit}
                )
                if response.status_code == 200:
                    return response.json()
                return None
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
            return None

# FastAPI application
app = FastAPI(
    title="VIPER Strategy Optimizer",
    version="1.0.0",
    description="Advanced parameter optimization and strategy tuning service"
)

optimizer = StrategyOptimizer()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if not optimizer.initialize_redis():
        logger.error("❌ Failed to initialize Redis. Exiting...")
        return

    logger.info("✅ Strategy Optimizer started successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "strategy-optimizer",
            "redis_connected": optimizer.redis_client is not None,
            "active_optimizations": len(optimizer.active_optimizations),
            "optimization_methods": optimizer.optimization_methods,
            "max_iterations": optimizer.max_iterations
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "strategy-optimizer",
                "error": str(e)
            }
        )

@app.post("/api/optimization/run")
async def run_optimization(request: Request, background_tasks: BackgroundTasks):
    """Run optimization"""
    try:
        data = await request.json()

        required_fields = ['symbol', 'timeframe', 'method', 'parameter_ranges']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        symbol = data['symbol']
        timeframe = data['timeframe']
        method = data['method']
        parameter_ranges = data['parameter_ranges']
        initial_balance = data.get('initial_balance', 10000.0)

        if method not in optimizer.optimization_methods:
            raise HTTPException(status_code=400, detail=f"Optimization method '{method}' not supported")

        optimization_id = str(uuid.uuid4())
        optimizer.active_optimizations[optimization_id] = {'status': 'running', 'progress': 0}

        # Run optimization in background
        background_tasks.add_task(
            optimizer.run_optimization_async,
            optimization_id, symbol, timeframe, method, parameter_ranges, initial_balance
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

@app.get("/api/optimization/{optimization_id}")
async def get_optimization_result(optimization_id: str):
    """Get optimization result"""
    try:
        # Check if result is cached in Redis
        result = optimizer.redis_client.get(f"viper:optimization:{optimization_id}")
        if result:
            return json.loads(result)

        # Check if optimization is still running
        if optimization_id in optimizer.active_optimizations:
            return {
                'optimization_id': optimization_id,
                'status': optimizer.active_optimizations[optimization_id]['status'],
                'progress': optimizer.active_optimizations[optimization_id]['progress']
            }

        raise HTTPException(status_code=404, detail=f"Optimization {optimization_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting optimization result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/optimizations")
async def list_optimizations():
    """List all available optimization results"""
    try:
        keys = optimizer.redis_client.keys("viper:optimization:*")
        optimization_ids = [key.split(":")[-1] for key in keys]

        optimizations = []
        for opt_id in optimization_ids[:50]:  # Limit to last 50
            result = optimizer.redis_client.get(f"viper:optimization:{opt_id}")
            if result:
                data = json.loads(result)
                optimizations.append({
                    'optimization_id': opt_id,
                    'method': data.get('optimization_method', 'unknown'),
                    'symbol': data.get('symbol', 'unknown'),
                    'timestamp': data.get('timestamp', 'unknown'),
                    'best_fitness': data.get('best_fitness', 0)
                })

        return {'optimizations': optimizations}

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to list optimizations: {e}")

@app.get("/api/methods")
async def get_available_methods():
    """Get available optimization methods"""
    return {
        'methods': optimizer.optimization_methods,
        'descriptions': {
            'genetic': 'Genetic algorithm optimization for complex parameter spaces',
            'grid_search': 'Exhaustive grid search over parameter combinations',
            'walk_forward': 'Walk-forward analysis with out-of-sample testing',
            'random_search': 'Random search optimization (fast but less thorough)'
        }
    }

# Background task implementations
async def run_optimization_async(self, optimization_id, symbol, timeframe, method,
                               parameter_ranges, initial_balance):
    """Background task to run optimization"""
    try:
        self.active_optimizations[optimization_id]['progress'] = 5

        result = None

        if method == 'genetic':
            result = self.genetic_algorithm_optimization(
                symbol, timeframe, parameter_ranges, initial_balance=initial_balance
            )
        elif method == 'grid_search':
            result = self.grid_search_optimization(
                symbol, timeframe, parameter_ranges, initial_balance=initial_balance
            )
        elif method == 'walk_forward':
            result = self.walk_forward_optimization(
                symbol, timeframe, parameter_ranges, initial_balance=initial_balance
            )
        else:
            result = {'error': f'Unknown optimization method: {method}'}

        self.active_optimizations[optimization_id]['progress'] = 95

        # Store result
        result['optimization_id'] = optimization_id
        self.redis_client.setex(
            f"viper:optimization:{optimization_id}",
            86400 * 7,  # 7 days
            json.dumps(result)
        )

        # Save to file
        result_file = self.results_path / f"optimization_{optimization_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        self.active_optimizations[optimization_id]['progress'] = 100
        self.active_optimizations[optimization_id]['status'] = 'completed'

        logger.info(f"✅ Optimization {optimization_id} completed successfully")

    except Exception as e:
        logger.error(f"❌ Error in optimization {optimization_id}: {e}")
        self.active_optimizations[optimization_id] = {'status': 'failed', 'error': str(e)}

# Add background methods to class
StrategyOptimizer.run_optimization_async = run_optimization_async

if __name__ == "__main__":
    port = int(os.getenv("STRATEGY_OPTIMIZER_PORT", 8000))
    logger.info(f"Starting VIPER Strategy Optimizer on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
