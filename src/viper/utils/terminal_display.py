"""
🎨 VIPER Terminal Display Enhancement
Rich terminal interface for better user experience and AI interaction
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.tree import Tree
    from rich.rule import Rule
    from rich.columns import Columns
    from rich.status import Status
    from rich import box
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to basic console
    class Console:
        def print(self, *args, **kwargs):
        
        def rule(self, title=""):
            if title:

# Global console instance
console = Console() if RICH_AVAILABLE else Console()

class ViperTerminal:
    """Enhanced terminal interface for VIPER trading system"""
    
    def __init__(self):
        self.console = console
        self.theme = {
            'success': 'green',
            'error': 'red', 
            'warning': 'yellow',
            'info': 'blue',
            'profit': 'bright_green',
            'loss': 'bright_red',
            'neutral': 'white'
        }
    
    def print_banner(self):
        """Display VIPER system banner"""
        banner_text = """
    ██#   ██#██#██████# ███████#██████# 
    ██#   ██#██#██#==██#██#====#██#==██#
    ██#   ██#██#██████##█████#  ██████##
    #██# ██##██#██#===# ██#==#  ██#==██#
     #████## ██#██#     ███████#██#  ██#
      #===#  #=##=#     #======##=#  #=#
                                        
    # Rocket AI-Powered Trading System v2.5.4
        """
        
        if RICH_AVAILABLE:
            panel = Panel(
                banner_text,
                title="🤖 VIPER Trading Bot",
                subtitle="Automated Trading System",
                border_style="bright_blue",
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
    
    def print_system_status(self, status_data: Dict[str, Any]):
        """Display comprehensive system status"""
        if not RICH_AVAILABLE:
            for key, value in status_data.items():
            return
            
        # Create status table
        table = Table(title="# Tool System Status", box=box.ROUNDED)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Details", style="white")
        
        for component, info in status_data.items():
            status_color = "green" if info.get('healthy', False) else "red"
            status_icon = "# Check" if info.get('healthy', False) else "# X"
            
            table.add_row(
                component,
                f"[{status_color}]{status_icon} {info.get('status', 'Unknown')}[/]",
                info.get('details', '')
            )
        
        self.console.print(table)
    
    def print_trading_summary(self, summary: Dict[str, Any]):
        """Display trading performance summary"""
        if not RICH_AVAILABLE:
            for key, value in summary.items():
            return
            
        # Create trading summary layout
        layout = Layout()
        layout.split_row(
            Layout(name="metrics", ratio=2),
            Layout(name="positions", ratio=1)
        )
        
        # Metrics table
        metrics_table = Table(title="# Chart Performance Metrics", box=box.SIMPLE_HEAD)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right")
        metrics_table.add_column("Change", justify="right")
        
        # Add metrics with color coding
        for metric, data in summary.get('metrics', {}).items():
            value = data.get('value', '0')
            change = data.get('change', '0%')
            
            # Color based on performance
            if 'profit' in metric.lower() or 'pnl' in metric.lower():
                color = 'green' if float(value.replace('$', '').replace('%', '')) > 0 else 'red'
            else:
                color = 'white'
                
            change_color = 'green' if change.startswith('+') else 'red' if change.startswith('-') else 'white'
            
            metrics_table.add_row(
                metric,
                f"[{color}]{value}[/]",
                f"[{change_color}]{change}[/]"
            )
        
        layout["metrics"].update(metrics_table)
        
        # Positions table
        positions_table = Table(title="💼 Active Positions", box=box.SIMPLE_HEAD)
        positions_table.add_column("Symbol", style="cyan")
        positions_table.add_column("Side", justify="center")
        positions_table.add_column("Size", justify="right")
        positions_table.add_column("P&L", justify="right")
        
        for position in summary.get('positions', []):
            pnl_color = 'green' if position['pnl'] > 0 else 'red'
            side_color = 'green' if position['side'] == 'LONG' else 'red'
            
            positions_table.add_row(
                position['symbol'],
                f"[{side_color}]{position['side']}[/]",
                str(position['size']),
                f"[{pnl_color}]{position['pnl']:+.2f}%[/]"
            )
        
        layout["positions"].update(positions_table)
        self.console.print(layout)
    
    def show_progress(self, tasks: List[str], title: str = "Processing"):
        """Show progress for multiple tasks"""
        if not RICH_AVAILABLE:
            for i, task in enumerate(tasks, 1):
            return
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            
            # Add tasks
            task_ids = []
            for task_desc in tasks:
                task_id = progress.add_task(f"[cyan]{task_desc}", total=100)
                task_ids.append(task_id)
            
            # Simulate progress (in real implementation, update based on actual progress)
            import time
            for i in range(100):
                for task_id in task_ids:
                    progress.update(task_id, advance=1)
                time.sleep(0.02)
    
    def display_error(self, error_msg: str, details: Optional[str] = None):
        """Display error with formatting"""
        if not RICH_AVAILABLE:
            if details:
            return
            
        panel = Panel(
            f"[red]{error_msg}[/]\n\n{details or ''}", 
            title="# X Error",
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def display_warning(self, warning_msg: str, details: Optional[str] = None):
        """Display warning with formatting"""
        if not RICH_AVAILABLE:
            if details:
            return
            
        panel = Panel(
            f"[yellow]{warning_msg}[/]\n\n{details or ''}", 
            title="# Warning Warning",
            border_style="yellow",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def display_success(self, success_msg: str, details: Optional[str] = None):
        """Display success message with formatting"""
        if not RICH_AVAILABLE:
            if details:
            return
            
        panel = Panel(
            f"[green]{success_msg}[/]\n\n{details or ''}", 
            title="# Check Success",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def display_config_summary(self, config: Dict[str, Any]):
        """Display configuration summary"""
        if not RICH_AVAILABLE:
            for section, values in config.items():
                for key, value in values.items():
            return
            
        tree = Tree("# Tool Configuration Summary")
        
        for section, values in config.items():
            section_branch = tree.add(f"[bold cyan]{section.title()}[/]")
            
            for key, value in values.items():
                # Mask sensitive information
                if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                    display_value = '***masked***'
                else:
                    display_value = str(value)
                    
                section_branch.add(f"{key}: [white]{display_value}[/]")
        
        self.console.print(tree)
    
    def live_dashboard(self, update_func, refresh_rate: float = 1.0):
        """Create live updating dashboard"""
        if not RICH_AVAILABLE:
            print("Live dashboard not available - Rich library not installed")
            return
            
        def generate_layout():
            return update_func()
        
        with Live(generate_layout(), refresh_per_second=refresh_rate, console=self.console) as live:
            try:
                while True:
                    live.update(generate_layout())
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Dashboard stopped by user[/]")
    
    def display_repository_structure(self, root_path: str = "."):
        """Display repository structure as tree"""
        if not RICH_AVAILABLE:
            print("Repository structure display not available")
            return
            
        tree = Tree(f"📁 [bold blue]{Path(root_path).name}[/] Repository Structure")
        
        def add_to_tree(path: Path, tree_node):
            try:
                items = sorted(path.iterdir())
                # Separate directories and files
                dirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
                files = [item for item in items if item.is_file() and not item.name.startswith('.')]
                
                # Add directories first
                for directory in dirs:
                    dir_node = tree_node.add(f"📁 [cyan]{directory.name}[/]")
                    if len(list(directory.iterdir())) > 0:  # Has contents
                        add_to_tree(directory, dir_node)
                
                # Add files
                for file in files[:10]:  # Limit to first 10 files per directory
                    icon = "🐍" if file.suffix == ".py" else "📄"
                    tree_node.add(f"{icon} {file.name}")
                
                if len(files) > 10:
                    tree_node.add(f"... and {len(files) - 10} more files")
                    
            except PermissionError:
                tree_node.add("[red]Permission denied[/]")
        
        add_to_tree(Path(root_path), tree)
        self.console.print(tree)

# Global terminal instance
terminal = ViperTerminal()

# Convenience functions for easy import
def print_banner():
    terminal.print_banner()

def print_status(status_data):
    terminal.print_system_status(status_data)

def print_trading_summary(summary):
    terminal.print_trading_summary(summary)

def show_progress(tasks, title="Processing"):
    terminal.show_progress(tasks, title)

def display_error(msg, details=None):
    terminal.display_error(msg, details)

def display_warning(msg, details=None):
    terminal.display_warning(msg, details)

def display_success(msg, details=None):
    terminal.display_success(msg, details)

def display_config(config):
    terminal.display_config_summary(config)

def display_repo_structure(path="."):
    terminal.display_repository_structure(path)