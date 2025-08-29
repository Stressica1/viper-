#!/usr/bin/env python3
"""
🎯 MASSIVE BACKTEST ORCHESTRATOR - 50 Pairs × 200 Configs
GitHub MCP-powered comprehensive backtesting system

This orchestrator provides:
✅ 50 trading pairs simultaneous backtesting
✅ 200 configuration variations optimization
✅ Parallel processing with resource management
✅ Real-time progress monitoring
✅ GitHub MCP integration for results
✅ Distributed processing capabilities
"""

import sys
import json
import asyncio
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import gc
from dataclasses import dataclass, asdict
from itertools import product
import aiofiles

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import existing components
from github_mcp_integration import GitHubMCPIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MASSIVE_BACKTEST - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('massive_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MassiveBacktestConfig:
    """Configuration for massive backtesting operation"""
    name: str
    trading_pairs: List[str]
    timeframes: List[str]
    ma_configs: List[Dict[str, int]]
    atr_configs: List[Dict[str, float]]
    risk_configs: List[Dict[str, float]]
    trend_configs: List[Dict[str, float]]
    entry_filters: List[Dict[str, float]]
    historical_days: int = 90
    max_concurrent_pairs: int = 5
    max_memory_mb: int = 4096
    timeout_seconds: int = 3600

@dataclass
class ProcessingStats:
    """Real-time processing statistics"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_tasks: int = 0
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    memory_usage_mb: float = 0.0
    cpu_usage_pct: float = 0.0

class MassiveBacktestOrchestrator:
    """
    Orchestrator for massive-scale backtesting operations
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or project_root / "massive_backtest_config.json"
        self.config = None
        self.processing_stats = ProcessingStats()
        self.results = []
        self.failed_tasks = []
        self.optimizer = None
        self.github_mcp = GitHubMCPIntegration()

        # Load configuration
        self._load_configuration()

        # Processing queues
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()

        logger.info("🎯 Massive Backtest Orchestrator initialized")
        logger.info(f"📊 Configuration: {len(self.config.trading_pairs)} pairs × {self._calculate_total_configs()} configs")

    def _load_configuration(self):
        """Load massive backtest configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)

            massive_config = config_data['massive_backtest_configuration']

            # Extract configuration variations
            variations = massive_config['configuration_variations']

            self.config = MassiveBacktestConfig(
                name=massive_config['name'],
                trading_pairs=massive_config['trading_pairs'],
                timeframes=massive_config['timeframes'],
                ma_configs=variations['moving_average_configs'],
                atr_configs=variations['atr_configs'],
                risk_configs=variations['risk_configs'],
                trend_configs=variations['trend_configs'],
                entry_filters=variations['entry_filters'],
                historical_days=massive_config['backtest_parameters']['historical_days'],
                max_concurrent_pairs=massive_config['processing_strategy']['max_concurrent_pairs']
            )

            logger.info("✅ Configuration loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise

    def _calculate_total_configs(self) -> int:
        """Calculate total number of configuration combinations"""
        if not self.config:
            return 0

        total = (len(self.config.ma_configs) *
                len(self.config.atr_configs) *
                len(self.config.risk_configs) *
                len(self.config.trend_configs) *
                len(self.config.entry_filters))

        return total

    def _generate_config_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible configuration combinations"""
        combinations = []

        # Create all combinations using cartesian product
        for ma, atr, risk, trend, entry_filter in product(
            self.config.ma_configs,
            self.config.atr_configs,
            self.config.risk_configs,
            self.config.trend_configs,
            self.config.entry_filters
        ):

            # Merge all configuration parameters
            config = {
                **ma,
                **atr,
                **risk,
                **trend,
                **entry_filter
            }

            # Create unique config ID
            config_id = f"MA{ma['fast_ma']}_{ma['slow_ma']}_{ma['trend_ma']}_ATR{atr['atr_length']}_{atr['atr_multiplier']}_TP{risk['take_profit_pct']}_SL{risk['stop_loss_pct']}_TS{risk['trailing_stop_pct']}_TB{trend['min_trend_bars']}_TCT{trend['trend_change_threshold']}"

            config['config_id'] = config_id
            combinations.append(config)

        logger.info(f"📊 Generated {len(combinations)} configuration combinations")
        return combinations

    async def initialize_optimizer(self):
        """Initialize the backtesting optimizer"""
        try:
            self.optimizer = MCPBacktestingOptimizer()
            await self.optimizer.initialize_exchange()
            logger.info("✅ Backtesting optimizer initialized")
        except Exception as e:
            logger.error(f"❌ Optimizer initialization failed: {e}")
            raise

    async def create_massive_backtest_task(self) -> str:
        """
        Create a massive backtesting task via MCP

        Returns:
            Task ID for tracking
        """
        try:
            task_id = f"massive_backtest_{len(self.config.trading_pairs)}p_{self._calculate_total_configs()}c_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create comprehensive task configuration
            task_config = {
                'task_id': task_id,
                'task_type': 'massive_backtest',
                'name': self.config.name,
                'trading_pairs': len(self.config.trading_pairs),
                'configurations': self._calculate_total_configs(),
                'total_combinations': len(self.config.trading_pairs) * self._calculate_total_configs(),
                'estimated_duration_hours': (len(self.config.trading_pairs) * self._calculate_total_configs() * 30) / 3600,  # Rough estimate
                'created_at': datetime.now().isoformat(),
                'status': 'created'
            }

            # Log task creation to GitHub MCP
            await self.github_mcp.log_system_performance({
                'massive_backtest_task_created': task_id,
                'total_pairs': len(self.config.trading_pairs),
                'total_configs': self._calculate_total_configs(),
                'estimated_duration_hours': task_config['estimated_duration_hours'],
                'task_config': task_config
            })

            logger.info(f"✅ Massive backtest task created: {task_id}")
            logger.info(f"📊 Total combinations: {task_config['total_combinations']:,}")
            logger.info(f"⏱️ Estimated duration: {task_config['estimated_duration_hours']:.1f} hours")

            return task_id

        except Exception as e:
            logger.error(f"❌ Task creation failed: {e}")
            return None

    async def run_massive_backtest(self, task_id: str) -> Dict[str, Any]:
        """
        Execute the massive backtesting operation

        Args:
            task_id: MCP task identifier

        Returns:
            Comprehensive results summary
        """
        try:
            logger.info("🚀 STARTING MASSIVE BACKTEST OPERATION")
            logger.info("=" * 60)

            # Initialize processing stats
            self.processing_stats.start_time = datetime.now()
            self.processing_stats.total_tasks = len(self.config.trading_pairs) * self._calculate_total_configs()

            # Create all task combinations
            await self._populate_task_queue()

            # Start processing workers
            results = await self._run_parallel_processing()

            # Generate comprehensive results
            summary = await self._generate_massive_results_summary(task_id, results)

            # Update final stats
            self.processing_stats.completed_tasks = len(results)
            self.processing_stats.failed_tasks = len(self.failed_tasks)

            logger.info("✅ MASSIVE BACKTEST OPERATION COMPLETED")
            logger.info(f"📊 Results: {len(results)} successful, {len(self.failed_tasks)} failed")

            return summary

        except Exception as e:
            logger.error(f"❌ Massive backtest failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def _populate_task_queue(self):
        """Populate the task queue with all combinations"""
        try:
            combinations = self._generate_config_combinations()

            total_tasks = 0
            for pair in self.config.trading_pairs:
                for timeframe in self.config.timeframes:
                    for config in combinations:
                        task = {
                            'symbol': pair,
                            'timeframe': timeframe,
                            'config': config,
                            'config_id': config['config_id'],
                            'task_id': f"{pair}_{timeframe}_{config['config_id']}"
                        }

                        await self.task_queue.put(task)
                        total_tasks += 1

            logger.info(f"📋 Task queue populated with {total_tasks} tasks")

        except Exception as e:
            logger.error(f"❌ Failed to populate task queue: {e}")
            raise

    async def _run_parallel_processing(self) -> List[Dict[str, Any]]:
        """Run parallel processing with resource management"""
        try:
            results = []
            semaphore = asyncio.Semaphore(self.config.max_concurrent_pairs)

            async def worker():
                while True:
                    try:
                        # Get next task
                        task = await self.task_queue.get()

                        # Acquire semaphore for resource management
                        async with semaphore:
                            self.processing_stats.active_tasks += 1

                            # Process task
                            result = await self._process_single_backtest(task)

                            if result:
                                results.append(result)
                                self.processing_stats.completed_tasks += 1
                            else:
                                self.failed_tasks.append(task)
                                self.processing_stats.failed_tasks += 1

                            self.processing_stats.active_tasks -= 1

                            # Update progress
                            await self._update_progress()

                            # Memory management
                            if len(results) % 100 == 0:
                                gc.collect()

                        self.task_queue.task_done()

                    except asyncio.QueueEmpty:
                        break
                    except Exception as e:
                        logger.error(f"❌ Worker error: {e}")
                        self.processing_stats.active_tasks -= 1
                        continue

            # Create worker tasks
            workers = []
            for i in range(self.config.max_concurrent_pairs):
                worker_task = asyncio.create_task(worker())
                workers.append(worker_task)

            # Wait for all tasks to complete
            await asyncio.gather(*workers, return_exceptions=True)

            return results

        except Exception as e:
            logger.error(f"❌ Parallel processing failed: {e}")
            return []

    async def _process_single_backtest(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single backtest task"""
        try:
            symbol = task['symbol']
            timeframe = task['timeframe']
            config = task['config']

            logger.debug(f"🔍 Processing: {symbol} {timeframe} {config['config_id']}")

            # Create optimization parameters from config
            from mcp_backtesting_optimizer import OptimizationParameters

            opt_params = OptimizationParameters(
                fast_ma_length=config['fast_ma'],
                slow_ma_length=config['slow_ma'],
                atr_length=config['atr_length'],
                atr_multiplier=config['atr_multiplier'],
                take_profit_pct=config['take_profit_pct'],
                stop_loss_pct=config['stop_loss_pct'],
                trailing_stop_pct=config['trailing_stop_pct'],
                min_trend_bars=config['min_trend_bars'],
                trend_change_threshold=config['trend_change_threshold']
            )

            # Run backtest
            result = await self.optimizer.run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                config=opt_params,
                days=self.config.historical_days
            )

            if result:
                result_dict = asdict(result)
                result_dict['config_id'] = config['config_id']
                result_dict['task_id'] = task['task_id']

                return result_dict
            else:
                logger.warning(f"⚠️ No result for {symbol} {timeframe} {config['config_id']}")
                return None

        except Exception as e:
            logger.error(f"❌ Single backtest failed: {e}")
            return None

    async def _update_progress(self):
        """Update and log progress"""
        try:
            total_processed = self.processing_stats.completed_tasks + self.processing_stats.failed_tasks
            progress_pct = (total_processed / self.processing_stats.total_tasks) * 100

            # Calculate estimated completion time
            if self.processing_stats.start_time and total_processed > 0:
                elapsed = datetime.now() - self.processing_stats.start_time
                avg_time_per_task = elapsed.total_seconds() / total_processed
                remaining_tasks = self.processing_stats.total_tasks - total_processed
                estimated_remaining = remaining_tasks * avg_time_per_task

                self.processing_stats.estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining)

            # Update memory and CPU stats
            self.processing_stats.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.processing_stats.cpu_usage_pct = psutil.cpu_percent()

            # Log progress every 100 tasks
            if total_processed % 100 == 0:
                logger.info(f"📊 Progress: {progress_pct:.1f}% ({total_processed}/{self.processing_stats.total_tasks})")
                logger.info(f"   Active: {self.processing_stats.active_tasks}, Memory: {self.processing_stats.memory_usage_mb:.1f}MB")

                # Log to GitHub MCP
                await self.github_mcp.log_system_performance({
                    'massive_backtest_progress': True,
                    'progress_pct': progress_pct,
                    'completed': self.processing_stats.completed_tasks,
                    'failed': self.processing_stats.failed_tasks,
                    'active': self.processing_stats.active_tasks,
                    'memory_mb': self.processing_stats.memory_usage_mb,
                    'cpu_pct': self.processing_stats.cpu_usage_pct,
                    'estimated_completion': self.processing_stats.estimated_completion.isoformat() if self.processing_stats.estimated_completion else None
                })

        except Exception as e:
            logger.error(f"❌ Progress update failed: {e}")

    async def _generate_massive_results_summary(self, task_id: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive results summary"""
        try:
            logger.info("📄 Generating massive results summary...")

            if not results:
                return {'status': 'no_results', 'message': 'No backtest results generated'}

            # Analyze results by various metrics
            summary = {
                'task_id': task_id,
                'generated_at': datetime.now().isoformat(),
                'total_results': len(results),
                'total_failed': len(self.failed_tasks),
                'processing_stats': asdict(self.processing_stats),

                # Best performers by different metrics
                'best_by_sharpe_ratio': self._find_best_results(results, 'sharpe_ratio', reverse=True),
                'best_by_win_rate': self._find_best_results(results, 'win_rate', reverse=True),
                'best_by_total_pnl': self._find_best_results(results, 'total_pnl', reverse=True),
                'best_by_profit_factor': self._find_best_results(results, 'profit_factor', reverse=True),

                # Overall statistics
                'overall_stats': self._calculate_overall_statistics(results),

                # Pair-specific analysis
                'pair_performance': self._analyze_pair_performance(results),

                # Timeframe analysis
                'timeframe_performance': self._analyze_timeframe_performance(results),

                # Success rate analysis
                'success_analysis': self._analyze_success_rates(results)
            }

            # Save comprehensive results
            await self._save_massive_results(summary)

            # Log summary to GitHub MCP
            await self.github_mcp.log_system_performance({
                'massive_backtest_completed': True,
                'task_id': task_id,
                'total_results': len(results),
                'best_sharpe': summary['best_by_sharpe_ratio'][0]['sharpe_ratio'] if summary['best_by_sharpe_ratio'] else 0,
                'best_win_rate': summary['best_by_win_rate'][0]['win_rate'] if summary['best_by_win_rate'] else 0,
                'processing_time_seconds': (datetime.now() - self.processing_stats.start_time).total_seconds() if self.processing_stats.start_time else 0
            })

            logger.info("✅ Massive results summary generated")
            return summary

        except Exception as e:
            logger.error(f"❌ Results summary generation failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def _find_best_results(self, results: List[Dict[str, Any]], metric: str, limit: int = 10, reverse: bool = True) -> List[Dict[str, Any]]:
        """Find best performing results by metric"""
        try:
            # Filter out results with invalid metrics
            valid_results = [r for r in results if r.get(metric) is not None and not (isinstance(r[metric], float) and (math.isnan(r[metric]) or math.isinf(r[metric])))]

            if not valid_results:
                return []

            # Sort by metric
            sorted_results = sorted(valid_results, key=lambda x: x[metric], reverse=reverse)
            return sorted_results[:limit]

        except Exception as e:
            logger.error(f"❌ Best results analysis failed: {e}")
            return []

    def _calculate_overall_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall statistics across all results"""
        try:
            if not results:
                return {}

            # Extract metrics
            win_rates = [r['win_rate'] for r in results if r.get('win_rate') is not None]
            total_pnls = [r['total_pnl'] for r in results if r.get('total_pnl') is not None]
            sharpe_ratios = [r['sharpe_ratio'] for r in results if r.get('sharpe_ratio') is not None and not math.isnan(r['sharpe_ratio'])]
            profit_factors = [r['profit_factor'] for r in results if r.get('profit_factor') is not None and r['profit_factor'] != float('inf')]

            return {
                'avg_win_rate': sum(win_rates) / len(win_rates) if win_rates else 0,
                'median_win_rate': sorted(win_rates)[len(win_rates)//2] if win_rates else 0,
                'best_win_rate': max(win_rates) if win_rates else 0,
                'worst_win_rate': min(win_rates) if win_rates else 0,

                'avg_total_pnl': sum(total_pnls) / len(total_pnls) if total_pnls else 0,
                'total_positive_pnl': len([p for p in total_pnls if p > 0]),
                'best_total_pnl': max(total_pnls) if total_pnls else 0,
                'worst_total_pnl': min(total_pnls) if total_pnls else 0,

                'avg_sharpe_ratio': sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0,
                'best_sharpe_ratio': max(sharpe_ratios) if sharpe_ratios else 0,

                'avg_profit_factor': sum(profit_factors) / len(profit_factors) if profit_factors else 0,
                'best_profit_factor': max(profit_factors) if profit_factors else 0
            }

        except Exception as e:
            logger.error(f"❌ Overall statistics calculation failed: {e}")
            return {}

    def _analyze_pair_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by trading pair"""
        try:
            pair_stats = {}

            for result in results:
                symbol = result.get('symbol')
                if not symbol:
                    continue

                if symbol not in pair_stats:
                    pair_stats[symbol] = {
                        'total_tests': 0,
                        'profitable_tests': 0,
                        'avg_win_rate': 0,
                        'avg_pnl': 0,
                        'best_sharpe': 0,
                        'win_rates': []
                    }

                pair_stats[symbol]['total_tests'] += 1

                if result.get('total_pnl', 0) > 0:
                    pair_stats[symbol]['profitable_tests'] += 1

                if result.get('win_rate'):
                    pair_stats[symbol]['win_rates'].append(result['win_rate'])

                if result.get('sharpe_ratio') and result['sharpe_ratio'] > pair_stats[symbol]['best_sharpe']:
                    pair_stats[symbol]['best_sharpe'] = result['sharpe_ratio']

            # Calculate averages
            for symbol, stats in pair_stats.items():
                if stats['win_rates']:
                    stats['avg_win_rate'] = sum(stats['win_rates']) / len(stats['win_rates'])
                stats['profitability_pct'] = (stats['profitable_tests'] / stats['total_tests']) * 100 if stats['total_tests'] > 0 else 0

            return pair_stats

        except Exception as e:
            logger.error(f"❌ Pair performance analysis failed: {e}")
            return {}

    def _analyze_timeframe_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by timeframe"""
        try:
            timeframe_stats = {}

            for result in results:
                timeframe = result.get('timeframe')
                if not timeframe:
                    continue

                if timeframe not in timeframe_stats:
                    timeframe_stats[timeframe] = {
                        'total_tests': 0,
                        'profitable_tests': 0,
                        'avg_win_rate': 0,
                        'avg_pnl': 0,
                        'win_rates': []
                    }

                timeframe_stats[timeframe]['total_tests'] += 1

                if result.get('total_pnl', 0) > 0:
                    timeframe_stats[timeframe]['profitable_tests'] += 1

                if result.get('win_rate'):
                    timeframe_stats[timeframe]['win_rates'].append(result['win_rate'])

            # Calculate averages
            for timeframe, stats in timeframe_stats.items():
                if stats['win_rates']:
                    stats['avg_win_rate'] = sum(stats['win_rates']) / len(stats['win_rates'])
                stats['profitability_pct'] = (stats['profitable_tests'] / stats['total_tests']) * 100 if stats['total_tests'] > 0 else 0

            return timeframe_stats

        except Exception as e:
            logger.error(f"❌ Timeframe performance analysis failed: {e}")
            return {}

    def _analyze_success_rates(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze success rates and patterns"""
        try:
            # Categorize by success metrics
            high_win_rate = [r for r in results if r.get('win_rate', 0) >= 60]
            profitable = [r for r in results if r.get('total_pnl', 0) > 0]
            high_sharpe = [r for r in results if r.get('sharpe_ratio', 0) >= 1.0]

            return {
                'high_win_rate_configs': len(high_win_rate),
                'profitable_configs': len(profitable),
                'high_sharpe_configs': len(high_sharpe),
                'success_rate_pct': (len(profitable) / len(results)) * 100 if results else 0,
                'elite_configs': len([r for r in results if r.get('win_rate', 0) >= 60 and r.get('total_pnl', 0) > 0 and r.get('sharpe_ratio', 0) >= 1.0])
            }

        except Exception as e:
            logger.error(f"❌ Success rate analysis failed: {e}")
            return {}

    async def _save_massive_results(self, summary: Dict[str, Any]):
        """Save comprehensive results to files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"massive_backtest_results_{timestamp}"

            # Save JSON summary
            json_file = f"{base_filename}.json"
            async with aiofiles.open(json_file, 'w') as f:
                await f.write(json.dumps(summary, indent=2, default=str))

            # Save CSV summary
            csv_file = f"{base_filename}_summary.csv"
            await self._save_csv_summary(summary, csv_file)

            # Save top performers
            top_performers_file = f"{base_filename}_top_performers.json"
            await self._save_top_performers(summary, top_performers_file)

            logger.info(f"💾 Results saved: {json_file}, {csv_file}, {top_performers_file}")

        except Exception as e:
            logger.error(f"❌ Results saving failed: {e}")

    async def _save_csv_summary(self, summary: Dict[str, Any], filename: str):
        """Save CSV summary of results"""
        try:
            csv_content = "Metric,Value\n"

            # Overall stats
            overall = summary.get('overall_stats', {})
            for key, value in overall.items():
                csv_content += f"{key},{value}\n"

            # Success analysis
            success = summary.get('success_analysis', {})
            for key, value in success.items():
                csv_content += f"{key},{value}\n"

            async with aiofiles.open(filename, 'w') as f:
                await f.write(csv_content)

        except Exception as e:
            logger.error(f"❌ CSV summary save failed: {e}")

    async def _save_top_performers(self, summary: Dict[str, Any], filename: str):
        """Save top performing configurations"""
        try:
            top_performers = {
                'best_sharpe_ratio': summary.get('best_by_sharpe_ratio', [])[:5],
                'best_win_rate': summary.get('best_by_win_rate', [])[:5],
                'best_total_pnl': summary.get('best_by_total_pnl', [])[:5],
                'best_profit_factor': summary.get('best_by_profit_factor', [])[:5]
            }

            async with aiofiles.open(filename, 'w') as f:
                await f.write(json.dumps(top_performers, indent=2, default=str))

        except Exception as e:
            logger.error(f"❌ Top performers save failed: {e}")

# MCP Task Functions
async def create_massive_backtest_task() -> str:
    """
    Create a massive backtesting task via MCP

    Returns:
        Task ID
    """
    try:
        orchestrator = MassiveBacktestOrchestrator()
        await orchestrator.initialize_optimizer()
        return await orchestrator.create_massive_backtest_task()

    except Exception as e:
        logger.error(f"❌ MCP task creation failed: {e}")
        return None

async def run_massive_backtest_operation(task_id: str) -> Dict[str, Any]:
    """
    Run the complete massive backtesting operation

    Args:
        task_id: MCP task identifier

    Returns:
        Comprehensive results
    """
    try:
        orchestrator = MassiveBacktestOrchestrator()
        await orchestrator.initialize_optimizer()
        return await orchestrator.run_massive_backtest(task_id)

    except Exception as e:
        logger.error(f"❌ Massive backtest operation failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def get_massive_backtest_status() -> Dict[str, Any]:
    """
    Get current massive backtest status

    Returns:
        Status information
    """
    try:
        # This would typically query the MCP system for status
        return {
            'status': 'running',
            'message': 'Massive backtest operation in progress',
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# Main execution
async def main():
    """Main massive backtesting execution"""

    try:
        # Create massive backtest task
        task_id = await create_massive_backtest_task()

        if not task_id:
            return 1


        # Get configuration info
        orchestrator = MassiveBacktestOrchestrator()
        total_pairs = len(orchestrator.config.trading_pairs)
        total_configs = orchestrator._calculate_total_configs()
        total_combinations = total_pairs * total_configs * len(orchestrator.config.timeframes)

        print(f"   Timeframes: {len(orchestrator.config.timeframes)}")
        print(f"   Total Combinations: {total_combinations:,}")
        print(f"   Estimated Duration: {(total_combinations * 30) / 3600:.1f} hours")

        # Confirm execution
        confirm = input(f"\n🚀 Execute massive backtest with {total_combinations:,} combinations? (yes/no): ").strip().lower()

        if confirm != 'yes':
            return 0

        # Run the massive backtest

        start_time = datetime.now()
        results = await run_massive_backtest_operation(task_id)
        end_time = datetime.now()

        print(f"Duration: {(end_time - start_time).total_seconds() / 3600:.2f} hours")

        if results.get('status') == 'error':
            print(f"❌ Error: {results.get('error', 'Unknown error')}")
            return 1

        # Display summary
        summary = results
        overall_stats = summary.get('overall_stats', {})

        print(f"   Total Results: {summary.get('total_results', 0):,}")
        print(f"   Failed Tasks: {summary.get('total_failed', 0)}")
        print(f"   Average Win Rate: {overall_stats.get('avg_win_rate', 0):.1f}%")
        print(f"   Best Win Rate: {overall_stats.get('best_win_rate', 0):.1f}%")
        print(f"   Profitable Configs: {summary.get('success_analysis', {}).get('profitable_configs', 0)}")

        if summary.get('best_by_sharpe_ratio'):
            best_sharpe = summary['best_by_sharpe_ratio'][0]
            print(f"   Symbol: {best_sharpe.get('symbol', 'N/A')}")
            print(f"   Timeframe: {best_sharpe.get('timeframe', 'N/A')}")
            print(f"   Sharpe Ratio: {best_sharpe.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {best_sharpe.get('win_rate', 0):.1f}%")
            print(f"   Total P&L: ${best_sharpe.get('total_pnl', 0):.2f}")


        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
