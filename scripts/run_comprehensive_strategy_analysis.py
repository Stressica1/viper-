#!/usr/bin/env python3
"""
🚀 MASTER STRATEGY ANALYSIS RUNNER
Complete workflow for comprehensive strategy backtesting and visualization

This script:
✅ Runs comprehensive backtesting for lower timeframes (30min and under)
✅ Analyzes and ranks all strategies
✅ Creates enhanced interactive dashboards
✅ Generates detailed performance reports
✅ Provides best strategy recommendations for live trading
"""

import asyncio
import sys
from pathlib import Path
import logging
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import our backtesting modules
try:
    from comprehensive_strategy_backtester import ComprehensiveStrategyBacktester
    from enhanced_strategy_dashboard import EnhancedStrategyDashboard, create_enhanced_summary_report
except ImportError as e:
    print("Please ensure all required dependencies are installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - STRATEGY_ANALYSIS - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

console = Console()

class StrategyAnalysisWorkflow:
    """
    Complete workflow for strategy analysis and backtesting
    """
    
    def __init__(self):
        self.backtester = None
        self.dashboard = None
        self.results_path = Path("backtest_results")
        self.results_path.mkdir(exist_ok=True)
        
        console.print("🚀 Strategy Analysis Workflow initialized")
    
    async def run_complete_analysis(self) -> dict:
        """Run the complete strategy analysis workflow"""
        
        # Display welcome message
        welcome_text = Text()
        welcome_text.append("🚀 COMPREHENSIVE STRATEGY ANALYSIS\n", style="bold blue")
        welcome_text.append("🎯 Focus: Lower Timeframes (30min and under)\n", style="yellow")
        welcome_text.append("📊 Testing multiple strategies with advanced metrics\n", style="green")
        welcome_text.append("🎨 Creating enhanced visualizations and reports", style="magenta")
        
        console.print(Panel(welcome_text, title="VIPER STRATEGY ANALYSIS", border_style="blue"))
        
        start_time = datetime.now()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=False
        ) as progress:
            
            # Step 1: Initialize backtester
            task1 = progress.add_task("🔧 Initializing backtesting engine...", total=None)
            self.backtester = ComprehensiveStrategyBacktester()
            progress.update(task1, description="✅ Backtesting engine ready")
            
            # Step 2: Run comprehensive backtesting
            task2 = progress.add_task("🔄 Running comprehensive backtests...", total=None)
            backtest_results = await self.backtester.run_comprehensive_backtest()
            progress.update(task2, description="✅ Backtesting completed")
            
            # Step 3: Initialize dashboard
            task3 = progress.add_task("🎨 Setting up visualization dashboard...", total=None)
            self.dashboard = EnhancedStrategyDashboard()
            progress.update(task3, description="✅ Dashboard initialized")
            
            # Step 4: Load results into dashboard
            task4 = progress.add_task("📊 Loading results for visualization...", total=None)
            # Find the latest results file
            results_files = list(self.results_path.glob("comprehensive_backtest_report_*.json"))
            if results_files:
                latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
                self.dashboard.load_backtest_results(str(latest_file))
            progress.update(task4, description="✅ Results loaded")
            
            # Step 5: Create interactive dashboard
            task5 = progress.add_task("🌐 Creating interactive dashboard...", total=None)
            dashboard_file = self.dashboard.create_interactive_dashboard()
            progress.update(task5, description="✅ Interactive dashboard created")
            
            # Step 6: Generate final report
            task6 = progress.add_task("📄 Generating comprehensive report...", total=None)
            final_report = await self._generate_final_report()
            progress.update(task6, description="✅ Final report generated")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Display completion message
        completion_text = Text()
        completion_text.append("🎉 ANALYSIS COMPLETED SUCCESSFULLY!\n\n", style="bold green")
        completion_text.append(f"⏱️  Total Execution Time: {execution_time:.2f} seconds\n", style="blue")
        completion_text.append(f"📊 Total Backtests: {backtest_results.get('total_backtests', 'N/A')}\n", style="cyan")
        completion_text.append(f"🏆 Best Strategy: {backtest_results.get('best_strategy', {}).get('overall_best', {}).get('strategy_name', 'N/A')}\n", style="yellow")
        completion_text.append(f"🌐 Dashboard: {dashboard_file}\n", style="magenta")
        completion_text.append(f"📄 Report: {final_report.get('report_file', 'N/A')}", style="green")
        
        console.print(Panel(completion_text, title="✅ ANALYSIS COMPLETE", border_style="green"))
        
        return {
            'backtest_results': backtest_results,
            'dashboard_file': dashboard_file,
            'final_report': final_report,
            'execution_time': execution_time
        }
    
    async def _generate_final_report(self) -> dict:
        """Generate comprehensive final report"""
        
        # Display strategy performance table
        console.print("\n📊 [bold blue]DETAILED STRATEGY PERFORMANCE[/bold blue]")
        if self.dashboard and self.dashboard.backtest_results:
            table = self.dashboard.generate_strategy_report_table()
            console.print(table)
        
        # Create summary report
        console.print("\n📄 [bold blue]EXECUTIVE SUMMARY[/bold blue]")
        if self.dashboard:
            summary = create_enhanced_summary_report(self.dashboard)
            console.print(summary)
        
        # Generate recommendations for live trading
        recommendations = self._generate_live_trading_recommendations()
        
        # Save final report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'Comprehensive Strategy Analysis',
            'focus': 'Lower Timeframes (30min and under)',
            'recommendations': recommendations,
            'summary': summary if self.dashboard else "No summary available"
        }
        
        report_file = self.results_path / f"final_strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return {
            'report_file': str(report_file),
            'recommendations': recommendations,
            'summary': summary if self.dashboard else "No summary available"
        }
    
    def _generate_live_trading_recommendations(self) -> dict:
        """Generate specific recommendations for live trading implementation"""
        
        if not self.dashboard or not self.dashboard.backtest_results:
            return {"error": "No backtest results available for recommendations"}
        
        # Analyze results by timeframe
        timeframe_analysis = {}
        
        for tf in ['5m', '15m', '30m']:
            tf_results = [r for r in self.dashboard.backtest_results if r['timeframe'] == tf]
            if tf_results:
                # Sort by risk-adjusted return
                tf_sorted = sorted(tf_results, key=lambda x: x['total_return'] / max(x['max_drawdown'], 0.01), reverse=True)
                
                best_strategy = tf_sorted[0]
                timeframe_analysis[tf] = {
                    'recommended_strategy': best_strategy['strategy_name'],
                    'expected_return': best_strategy['total_return'],
                    'max_drawdown': best_strategy['max_drawdown'],
                    'win_rate': best_strategy['win_rate'],
                    'sharpe_ratio': best_strategy['sharpe_ratio'],
                    'total_trades': best_strategy['total_trades'],
                    'risk_level': 'Low' if best_strategy['max_drawdown'] < 0.05 else 'Medium' if best_strategy['max_drawdown'] < 0.15 else 'High'
                }
        
        # Overall best strategy
        all_sorted = sorted(self.dashboard.backtest_results, 
                          key=lambda x: x['total_return'] / max(x['max_drawdown'], 0.01), reverse=True)
        overall_best = all_sorted[0]
        
        recommendations = {
            'overall_best': {
                'strategy': overall_best['strategy_name'],
                'timeframe': overall_best['timeframe'],
                'expected_annual_return': overall_best['total_return'] * 4,  # Annualized approximation
                'risk_metrics': {
                    'max_drawdown': overall_best['max_drawdown'],
                    'sharpe_ratio': overall_best['sharpe_ratio'],
                    'win_rate': overall_best['win_rate']
                },
                'implementation_notes': [
                    "Start with small position sizes",
                    "Monitor performance closely for first week",
                    "Implement proper risk management (2% risk per trade)",
                    "Use trailing stops for profit protection"
                ]
            },
            'by_timeframe': timeframe_analysis,
            'risk_management': {
                'max_portfolio_risk': '30-35% capital utilization',
                'position_sizing': '2% risk per trade',
                'max_positions': 15,
                'stop_loss': 'Use ATR-based stops',
                'take_profit': 'Risk:reward ratio of 1:2 minimum'
            }
        }
        
        # Display recommendations
        rec_text = Text()
        rec_text.append("🎯 LIVE TRADING RECOMMENDATIONS\n\n", style="bold green")
        rec_text.append(f"🏆 OVERALL BEST: {overall_best['strategy_name']} ({overall_best['timeframe']})\n", style="bold cyan")
        rec_text.append(f"📈 Expected Return: {overall_best['total_return']:.2%}\n", style="blue")
        rec_text.append(f"📉 Max Drawdown: {overall_best['max_drawdown']:.2%}\n", style="red")
        rec_text.append(f"🎯 Win Rate: {overall_best['win_rate']:.1%}\n", style="green")
        rec_text.append(f"📊 Sharpe Ratio: {overall_best['sharpe_ratio']:.2f}\n\n", style="magenta")
        
        rec_text.append("📋 IMPLEMENTATION CHECKLIST:\n", style="bold yellow")
        for note in recommendations['overall_best']['implementation_notes']:
            rec_text.append(f"   ✅ {note}\n", style="dim")
        
        console.print(Panel(rec_text, title="🚀 READY FOR LIVE TRADING", border_style="green"))
        
        return recommendations

async def main():
    """Main execution function"""
    try:
        # Create and run workflow
        workflow = StrategyAnalysisWorkflow()
        results = await workflow.run_complete_analysis()
        
        # Final success message
        console.print("\n" + "="*60)
        console.print("🎉 [bold green]COMPREHENSIVE STRATEGY ANALYSIS COMPLETE![/bold green]")
        console.print("="*60)
        console.print(f"🌐 [cyan]Interactive Dashboard: {results['dashboard_file']}[/cyan]")
        console.print(f"📄 [yellow]Final Report: {results['final_report']['report_file']}[/yellow]")
        console.print(f"⏱️  [blue]Total Time: {results['execution_time']:.2f} seconds[/blue]")
        console.print(f"🚀 [green]Ready for live trading deployment![/green]")
        console.print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in strategy analysis: {e}")
        console.print(f"❌ [bold red]Analysis failed: {e}[/bold red]")
        return None

if __name__ == "__main__":
    # Ensure required dependencies
    try:
        from rich.console import Console
    except ImportError as e:
        print("Please install with: pip install numpy pandas matplotlib plotly seaborn rich")
        sys.exit(1)
    
    # Run the analysis
    asyncio.run(main())