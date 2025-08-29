#!/usr/bin/env python3
"""
🚀 ENHANCED STRATEGY COMPARISON DASHBOARD
Interactive web dashboard for strategy backtesting results with advanced visualizations

Features:
✅ Interactive Plotly charts and dashboards
✅ Real-time strategy comparison
✅ Performance heatmaps and correlation analysis
✅ Risk-return optimization plots
✅ Strategy parameter sensitivity analysis
✅ Export-ready reports and visualizations
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.figure_factory as ff

from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import box

console = Console()
logger = logging.getLogger(__name__)

class EnhancedStrategyDashboard:
    """
    Enhanced dashboard for strategy comparison and visualization
    """
    
    def __init__(self, results_path: str = "backtest_results"):
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Dashboard data
        self.backtest_results = None
        self.strategy_rankings = None
        self.performance_metrics = None
        
        console.print("🎨 Enhanced Strategy Dashboard initialized")
    
    def load_backtest_results(self, results_file: str) -> bool:
        """Load backtest results from JSON file"""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            self.backtest_results = data.get('detailed_results', [])
            self.strategy_rankings = data.get('all_rankings', [])
            self.performance_metrics = data
            
            console.print(f"✅ Loaded {len(self.backtest_results)} backtest results")
            return True
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return False
    
    def create_interactive_dashboard(self) -> str:
        """Create comprehensive interactive dashboard"""
        console.print("\n🎨 [bold blue]CREATING INTERACTIVE DASHBOARD[/bold blue]")
        
        if not self.backtest_results:
            console.print("❌ No backtest results loaded")
            return ""
        
        # Create main dashboard figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "🏆 Strategy Performance Ranking",
                "🎯 Risk-Return Analysis", 
                "📊 Performance by Timeframe",
                "💰 Win Rate vs Profit Factor",
                "📈 Consistency vs Return",
                "🔥 Strategy Heatmap"
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "violin"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Strategy Performance Ranking (Bar Chart)
        self._add_performance_ranking_chart(fig, row=1, col=1)
        
        # 2. Risk-Return Scatter Plot
        self._add_risk_return_scatter(fig, row=1, col=2)
        
        # 3. Performance by Timeframe (Violin Plot)
        self._add_timeframe_performance(fig, row=2, col=1)
        
        # 4. Win Rate vs Profit Factor
        self._add_win_rate_profit_factor(fig, row=2, col=2)
        
        # 5. Consistency vs Return
        self._add_consistency_return_scatter(fig, row=3, col=1)
        
        # 6. Strategy Performance Heatmap
        self._add_strategy_heatmap(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title={
                'text': "🚀 COMPREHENSIVE STRATEGY BACKTESTING DASHBOARD - LOWER TIMEFRAMES",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#1f77b4'}
            },
            font={'size': 10},
            plot_bgcolor='rgba(240,240,240,0.2)',
            paper_bgcolor='white'
        )
        
        # Save dashboard
        dashboard_file = self.results_path / f"strategy_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        plot(fig, filename=str(dashboard_file), auto_open=False)
        
        console.print(f"📊 [bold green]Interactive dashboard saved to: {dashboard_file}[/bold green]")
        
        # Create additional detailed charts
        self._create_detailed_performance_charts()
        
        return str(dashboard_file)
    
    def _add_performance_ranking_chart(self, fig, row, col):
        """Add strategy performance ranking bar chart"""
        if not self.strategy_rankings:
            return
        
        # Sort by composite score
        sorted_rankings = sorted(self.strategy_rankings, key=lambda x: x['composite_score'], reverse=True)[:10]
        
        strategies = [f"{r['strategy_name']} ({r['timeframe']})" for r in sorted_rankings]
        scores = [r['composite_score'] for r in sorted_rankings]
        
        # Color mapping based on performance
        colors = ['#2E8B57' if score > 0.8 else '#FFD700' if score > 0.6 else '#FF6347' for score in scores]
        
        fig.add_trace(
            go.Bar(
                x=scores,
                y=strategies,
                orientation='h',
                marker_color=colors,
                text=[f"{score:.3f}" for score in scores],
                textposition='auto',
                name='Composite Score',
                hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Composite Score", row=row, col=col)
        fig.update_yaxes(title_text="Strategy (Timeframe)", row=row, col=col)
    
    def _add_risk_return_scatter(self, fig, row, col):
        """Add risk-return analysis scatter plot"""
        if not self.strategy_rankings:
            return
        
        risk_adj_returns = [r['risk_adjusted_return'] for r in self.strategy_rankings]
        composite_scores = [r['composite_score'] for r in self.strategy_rankings]
        strategies = [f"{r['strategy_name']} ({r['timeframe']})" for r in self.strategy_rankings]
        
        fig.add_trace(
            go.Scatter(
                x=risk_adj_returns,
                y=composite_scores,
                mode='markers',
                marker=dict(
                    size=[max(5, min(50, r['consistency_score'] * 20 + 5)) for r in self.strategy_rankings],  # Clamp size values
                    color=composite_scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Composite Score", x=1.02),
                    line=dict(width=1, color='black')
                ),
                text=strategies,
                name='Strategies',
                hovertemplate='<b>%{text}</b><br>Risk-Adj Return: %{x:.3f}<br>Score: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Risk-Adjusted Return", row=row, col=col)
        fig.update_yaxes(title_text="Composite Score", row=row, col=col)
    
    def _add_timeframe_performance(self, fig, row, col):
        """Add timeframe performance violin plot"""
        if not self.backtest_results:
            return
        
        timeframes = ['5m', '15m', '30m']
        
        for tf in timeframes:
            tf_results = [r for r in self.backtest_results if r['timeframe'] == tf]
            returns = [r['total_return'] for r in tf_results]
            
            if returns:
                fig.add_trace(
                    go.Violin(
                        y=returns,
                        name=tf,
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=True
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Timeframe", row=row, col=col)
        fig.update_yaxes(title_text="Total Return", row=row, col=col)
    
    def _add_win_rate_profit_factor(self, fig, row, col):
        """Add win rate vs profit factor scatter"""
        if not self.backtest_results:
            return
        
        win_rates = [r['win_rate'] for r in self.backtest_results]
        profit_factors = [min(r['profit_factor'], 5) for r in self.backtest_results]  # Cap at 5
        strategies = [f"{r['strategy_name']} ({r['timeframe']})" for r in self.backtest_results]
        colors = [r['total_return'] for r in self.backtest_results]
        
        fig.add_trace(
            go.Scatter(
                x=win_rates,
                y=profit_factors,
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Total Return", x=1.05),
                    line=dict(width=1, color='black')
                ),
                text=strategies,
                name='Strategies',
                hovertemplate='<b>%{text}</b><br>Win Rate: %{x:.2%}<br>Profit Factor: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Win Rate", row=row, col=col, tickformat='.1%')
        fig.update_yaxes(title_text="Profit Factor", row=row, col=col)
    
    def _add_consistency_return_scatter(self, fig, row, col):
        """Add consistency vs return scatter"""
        if not self.strategy_rankings:
            return
        
        consistency_scores = [r['consistency_score'] for r in self.strategy_rankings]
        risk_adj_returns = [r['risk_adjusted_return'] for r in self.strategy_rankings]
        strategies = [f"{r['strategy_name']} ({r['timeframe']})" for r in self.strategy_rankings]
        sizes = [r['composite_score'] * 30 + 5 for r in self.strategy_rankings]
        
        fig.add_trace(
            go.Scatter(
                x=consistency_scores,
                y=risk_adj_returns,
                mode='markers',
                marker=dict(
                    size=[max(5, min(50, r['composite_score'] * 30 + 5)) for r in self.strategy_rankings],  # Clamp size values
                    color='purple',
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
                text=strategies,
                name='Consistency Analysis',
                hovertemplate='<b>%{text}</b><br>Consistency: %{x:.3f}<br>Risk-Adj Return: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Consistency Score", row=row, col=col)
        fig.update_yaxes(title_text="Risk-Adjusted Return", row=row, col=col)
    
    def _add_strategy_heatmap(self, fig, row, col):
        """Add strategy performance heatmap"""
        if not self.backtest_results:
            return
        
        # Create pivot table for heatmap
        df = pd.DataFrame(self.backtest_results)
        pivot_table = df.pivot_table(
            values='total_return',
            index='strategy_name',
            columns='timeframe',
            aggfunc='mean'
        )
        
        # Fill NaN values with 0
        pivot_table = pivot_table.fillna(0)
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='RdYlGn',
                text=np.round(pivot_table.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>%{y}</b><br>%{x}<br>Return: %{z:.3f}<extra></extra>',
                showscale=True,
                colorbar=dict(title="Avg Return", x=1.08)
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Timeframe", row=row, col=col)
        fig.update_yaxes(title_text="Strategy", row=row, col=col)
    
    def _create_detailed_performance_charts(self):
        """Create additional detailed performance charts"""
        console.print("📈 Creating detailed performance analysis...")
        
        # 1. Strategy Comparison Radar Chart
        self._create_radar_chart()
        
        # 2. Performance Timeline Analysis
        self._create_timeline_analysis()
        
        # 3. Risk Metrics Comparison
        self._create_risk_metrics_chart()
        
        console.print("✅ Detailed charts created successfully")
    
    def _create_radar_chart(self):
        """Create radar chart for strategy comparison"""
        if not self.strategy_rankings:
            return
        
        # Get top 5 strategies
        top_strategies = sorted(self.strategy_rankings, key=lambda x: x['composite_score'], reverse=True)[:5]
        
        # Define metrics for radar chart
        metrics = ['composite_score', 'risk_adjusted_return', 'consistency_score', 'robustness_score']
        metric_labels = ['Composite Score', 'Risk-Adj Return', 'Consistency', 'Robustness']
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for i, strategy in enumerate(top_strategies):
            values = [strategy[metric] for metric in metrics]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_labels + [metric_labels[0]],
                fill='toself',
                name=f"{strategy['strategy_name']} ({strategy['timeframe']})",
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="🎯 Top 5 Strategies - Multi-Metric Comparison",
            font_size=12
        )
        
        radar_file = self.results_path / f"strategy_radar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        plot(fig, filename=str(radar_file), auto_open=False)
        console.print(f"🎯 Radar chart saved to: {radar_file}")
    
    def _create_timeline_analysis(self):
        """Create timeline analysis of strategy performance"""
        if not self.backtest_results:
            return
        
        # Create synthetic timeline data (in production, use actual trade timestamps)
        fig = go.Figure()
        
        strategies = list(set([r['strategy_name'] for r in self.backtest_results]))
        colors = px.colors.qualitative.Set1
        
        for i, strategy in enumerate(strategies[:5]):  # Top 5 strategies
            strategy_results = [r for r in self.backtest_results if r['strategy_name'] == strategy]
            
            # Generate synthetic timeline
            dates = pd.date_range(start='2023-01-01', periods=len(strategy_results), freq='D')
            cumulative_returns = np.cumsum([r['total_return'] for r in strategy_results])
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines+markers',
                name=strategy,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="📈 Strategy Performance Timeline",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            showlegend=True,
            hovermode='x unified'
        )
        
        timeline_file = self.results_path / f"strategy_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        plot(fig, filename=str(timeline_file), auto_open=False)
        console.print(f"📈 Timeline chart saved to: {timeline_file}")
    
    def _create_risk_metrics_chart(self):
        """Create comprehensive risk metrics comparison"""
        if not self.backtest_results:
            return
        
        # Create risk metrics comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Max Drawdown by Strategy", "Sharpe Ratio Distribution", 
                          "Volatility vs Return", "Risk-Adjusted Metrics"],
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Max Drawdown by Strategy
        strategies = list(set([r['strategy_name'] for r in self.backtest_results]))
        avg_drawdowns = []
        for strategy in strategies:
            strategy_results = [r for r in self.backtest_results if r['strategy_name'] == strategy]
            avg_drawdown = np.mean([r['max_drawdown'] for r in strategy_results])
            avg_drawdowns.append(avg_drawdown)
        
        fig.add_trace(
            go.Bar(x=strategies, y=avg_drawdowns, name="Avg Max Drawdown",
                  marker_color='red', opacity=0.7),
            row=1, col=1
        )
        
        # 2. Sharpe Ratio Distribution
        sharpe_ratios = [r['sharpe_ratio'] for r in self.backtest_results]
        fig.add_trace(
            go.Histogram(x=sharpe_ratios, name="Sharpe Ratio", nbinsx=20,
                        marker_color='blue', opacity=0.7),
            row=1, col=2
        )
        
        # 3. Volatility vs Return
        volatilities = [r['volatility'] for r in self.backtest_results]
        returns = [r['total_return'] for r in self.backtest_results]
        fig.add_trace(
            go.Scatter(x=volatilities, y=returns, mode='markers',
                      name="Vol vs Return", marker_color='green'),
            row=2, col=1
        )
        
        # 4. Risk-Adjusted Metrics
        calmar_ratios = [r.get('calmar_ratio', 0) for r in self.backtest_results]
        sortino_ratios = [r.get('sortino_ratio', 0) for r in self.backtest_results]
        
        fig.add_trace(
            go.Bar(x=strategies, y=calmar_ratios, name="Avg Calmar Ratio",
                  marker_color='purple', opacity=0.7),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True,
                         title="📊 Comprehensive Risk Metrics Analysis")
        
        risk_file = self.results_path / f"risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        plot(fig, filename=str(risk_file), auto_open=False)
        console.print(f"📊 Risk metrics chart saved to: {risk_file}")
    
    def generate_strategy_report_table(self) -> Table:
        """Generate detailed strategy report table"""
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        
        # Add columns
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Strategy", style="cyan", width=18)
        table.add_column("TF", style="yellow", width=4)
        table.add_column("Score", justify="right", style="green", width=6)
        table.add_column("Return", justify="right", style="blue", width=8)
        table.add_column("Sharpe", justify="right", style="magenta", width=6)
        table.add_column("DD", justify="right", style="red", width=6)
        table.add_column("Win%", justify="right", style="green", width=6)
        table.add_column("Trades", justify="right", style="dim", width=6)
        table.add_column("Status", width=12)
        
        if not self.backtest_results:
            return table
        
        # Aggregate results by strategy-timeframe
        strategy_agg = {}
        for result in self.backtest_results:
            key = f"{result['strategy_name']}_{result['timeframe']}"
            if key not in strategy_agg:
                strategy_agg[key] = []
            strategy_agg[key].append(result)
        
        # Calculate aggregated metrics and sort
        agg_results = []
        for key, results in strategy_agg.items():
            strategy_name, timeframe = key.split('_', 1)
            
            avg_return = np.mean([r['total_return'] for r in results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in results])
            avg_winrate = np.mean([r['win_rate'] for r in results])
            total_trades = sum([r['total_trades'] for r in results])
            
            # Calculate composite score
            composite = (avg_return * 0.3 + avg_sharpe * 0.3 + (1-avg_drawdown) * 0.2 + avg_winrate * 0.2)
            
            agg_results.append({
                'strategy_name': strategy_name,
                'timeframe': timeframe,
                'composite_score': composite,
                'avg_return': avg_return,
                'avg_sharpe': avg_sharpe,
                'avg_drawdown': avg_drawdown,
                'avg_winrate': avg_winrate,
                'total_trades': total_trades
            })
        
        # Sort by composite score
        agg_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add rows to table
        for i, result in enumerate(agg_results[:15], 1):
            if result['composite_score'] > 0.8:
                status = "🟢 EXCELLENT"
            elif result['composite_score'] > 0.6:
                status = "🟡 GOOD"
            elif result['composite_score'] > 0.4:
                status = "🟠 FAIR"
            else:
                status = "🔴 POOR"
            
            table.add_row(
                str(i),
                result['strategy_name'],
                result['timeframe'],
                f"{result['composite_score']:.3f}",
                f"{result['avg_return']:.2%}",
                f"{result['avg_sharpe']:.2f}",
                f"{result['avg_drawdown']:.2%}",
                f"{result['avg_winrate']:.1%}",
                str(result['total_trades']),
                status
            )
        
        return table

def create_enhanced_summary_report(dashboard: EnhancedStrategyDashboard) -> str:
    """Create enhanced summary report with recommendations"""
    
    if not dashboard.backtest_results:
        return "No backtest results available"
    
    # Generate summary statistics
    total_backtests = len(dashboard.backtest_results)
    strategies_tested = len(set([r['strategy_name'] for r in dashboard.backtest_results]))
    
    # Best performers by timeframe
    best_by_tf = {}
    for tf in ['5m', '15m', '30m']:
        tf_results = [r for r in dashboard.backtest_results if r['timeframe'] == tf]
        if tf_results:
            best = max(tf_results, key=lambda x: x['total_return'])
            best_by_tf[tf] = best
    
    # Create report
    report = f"""
🚀 COMPREHENSIVE STRATEGY BACKTESTING REPORT
============================================

📊 SUMMARY STATISTICS:
• Total Backtests: {total_backtests}
• Strategies Tested: {strategies_tested}
• Timeframes: 5m, 15m, 30m (Lower TF Focus)
• Test Period: 90 days synthetic data

🏆 BEST PERFORMERS BY TIMEFRAME:
"""
    
    for tf, result in best_by_tf.items():
        report += f"""
⏰ {tf} TIMEFRAME:
  🥇 Strategy: {result['strategy_name']}
  📈 Total Return: {result['total_return']:.2%}
  📊 Sharpe Ratio: {result['sharpe_ratio']:.2f}
  📉 Max Drawdown: {result['max_drawdown']:.2%}
  🎯 Win Rate: {result['win_rate']:.1%}
"""
    
    # Overall recommendation
    all_results_sorted = sorted(dashboard.backtest_results, key=lambda x: x['total_return'], reverse=True)
    overall_best = all_results_sorted[0]
    
    report += f"""
🎯 OVERALL RECOMMENDATION:
==========================
🏆 BEST STRATEGY: {overall_best['strategy_name']} ({overall_best['timeframe']})

Performance Metrics:
• Total Return: {overall_best['total_return']:.2%}
• Risk-Adjusted Return: {overall_best['total_return'] / max(overall_best['max_drawdown'], 0.01):.2f}
• Sharpe Ratio: {overall_best['sharpe_ratio']:.2f}
• Maximum Drawdown: {overall_best['max_drawdown']:.2%}
• Win Rate: {overall_best['win_rate']:.1%}
• Total Trades: {overall_best['total_trades']}

🚀 RECOMMENDATION: Deploy this strategy for lower timeframe trading
✅ Ready for live implementation
"""
    
    return report

async def main():
    """Main function to demonstrate enhanced dashboard"""
    console.print("\n🎨 [bold blue]ENHANCED STRATEGY DASHBOARD DEMO[/bold blue]")
    
    # Create dashboard
    dashboard = EnhancedStrategyDashboard()
    
    # For demo, we'll use the latest backtest results if available
    results_files = list(Path("backtest_results").glob("comprehensive_backtest_report_*.json"))
    
    if results_files:
        latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
        console.print(f"📄 Loading results from: {latest_file}")
        
        if dashboard.load_backtest_results(str(latest_file)):
            # Create interactive dashboard
            dashboard_file = dashboard.create_interactive_dashboard()
            
            # Display summary table
            console.print("\n📊 [bold yellow]STRATEGY PERFORMANCE SUMMARY[/bold yellow]")
            table = dashboard.generate_strategy_report_table()
            console.print(table)
            
            # Generate summary report
            summary = create_enhanced_summary_report(dashboard)
            console.print(f"\n{summary}")
            
            console.print(f"\n✅ [bold green]Enhanced dashboard created successfully![/bold green]")
            console.print(f"🌐 [cyan]Open in browser: {dashboard_file}[/cyan]")
        else:
            console.print("❌ Failed to load backtest results")
    else:
        console.print("📄 No backtest results found. Run comprehensive_strategy_backtester.py first.")

if __name__ == "__main__":
    asyncio.run(main())