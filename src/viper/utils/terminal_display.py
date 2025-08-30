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
            print(*args, **kwargs)
        
        def rule(self, title=""):
            if title:
                print(f"\n=== {title} ===")
            else:
                print("=" * 50)

# Global console instance
console = Console() if RICH_AVAILABLE else Console()

class ViperTerminal:
    """Enhanced terminal interface for VIPER trading system"""
    
    def __init__(self):
        self.console = console
        self.theme = {
            'success': 'bright_green',
            'error': 'bright_red', 
            'warning': 'bright_yellow',
            'info': 'bright_blue',
            'profit': 'bold green',
            'loss': 'bold red',
            'neutral': 'white',
            'accent': 'bright_cyan',
            'subtle': 'dim white',
            'header': 'bold bright_blue'
        }
        # Enhanced icon set
        self.icons = {
            'success': '✅',
            'error': '❌', 
            'warning': '⚠️',
            'info': 'ℹ️',
            'rocket': '🚀',
            'chart': '📈',
            'money': '💰',
            'shield': '🛡️',
            'gear': '⚙️',
            'lightning': '⚡',
            'target': '🎯',
            'fire': '🔥',
            'diamond': '💎',
            'brain': '🧠',
            'robot': '🤖',
            'eye': '👁️',
            'lock': '🔒',
            'key': '🗝️',
            'database': '🗄️',
            'network': '🌐',
            'signal': '📡',
            'trend_up': '↗️',
            'trend_down': '↘️'
        }
    
    def print_banner(self):
        """Display VIPER system banner"""
        banner_text = """
    [bold bright_blue]██╗   ██╗██╗██████╗ ███████╗██████╗[/] 
    [bold bright_blue]██║   ██║██║██╔══██╗██╔════╝██╔══██╗[/]
    [bold bright_blue]██║   ██║██║██████╔╝█████╗  ██████╔╝[/]
    [bold bright_blue]╚██╗ ██╔╝██║██╔═══╝ ██╔══╝  ██╔══██╗[/]
    [bold bright_blue] ╚████╔╝ ██║██║     ███████╗██║  ██║[/]
    [bold bright_blue]  ╚═══╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝[/]
                                        
    [bold bright_cyan]🚀 AI-Powered Trading System v2.5.4[/]
    [dim]Intelligent • Adaptive • Profitable[/]
        """
        
        if RICH_AVAILABLE:
            panel = Panel(
                banner_text,
                title=f"{self.icons['robot']} [bold bright_cyan]VIPER Trading Bot[/]",
                subtitle=f"[dim]{self.icons['lightning']} High-Performance Automated Trading[/]",
                border_style="bright_blue",
                padding=(1, 2),
                expand=False
            )
            self.console.print(panel)
        else:
            print(banner_text)
    
    def print_system_status(self, status_data: Dict[str, Any]):
        """Display comprehensive system status"""
        if not RICH_AVAILABLE:
            for key, value in status_data.items():
                print(f"{key}: {value}")
            return
            
        # Create status table with enhanced design
        table = Table(
            title=f"{self.icons['gear']} [bold bright_cyan]System Health Monitor[/]", 
            box=box.ROUNDED,
            show_header=True,
            header_style="bold bright_blue",
            border_style="bright_blue"
        )
        table.add_column("🔧 Component", style="bright_cyan", no_wrap=True, width=20)
        table.add_column("📊 Status", justify="center", width=15)
        table.add_column("📝 Details", style="dim white", width=40)
        table.add_column("⏱️ Uptime", justify="center", width=10)
        
        # Enhanced status mapping
        status_icons = {
            'Connected': f"{self.icons['success']} Online",
            'Active': f"{self.icons['lightning']} Active", 
            'Scanning': f"{self.icons['eye']} Scanning",
            'Warning': f"{self.icons['warning']} Warning",
            'Error': f"{self.icons['error']} Error",
            'Offline': "🔌 Offline",
            'Starting': "⏳ Starting"
        }
        
        for component, info in status_data.items():
            healthy = info.get('healthy', False)
            status = info.get('status', 'Unknown')
            details = info.get('details', 'No details available')
            uptime = info.get('uptime', 'N/A')
            
            # Color coding based on health
            if healthy:
                status_color = "bright_green"
                row_style = None
            elif status == 'Warning':
                status_color = "bright_yellow" 
                row_style = "dim"
            else:
                status_color = "bright_red"
                row_style = "dim"
            
            status_display = status_icons.get(status, f"❓ {status}")
            
            table.add_row(
                f"{self.icons.get('gear', '⚙️')} {component}",
                f"[{status_color}]{status_display}[/]",
                f"[dim]{details}[/]",
                f"[dim]{uptime}[/]",
                style=row_style
            )
        
        self.console.print(table)
    
    def print_trading_summary(self, summary: Dict[str, Any]):
        """Display trading performance summary"""
        if not RICH_AVAILABLE:
            for key, value in summary.items():
                print(f"{key}: {value}")
            return
            
        # Create enhanced trading summary layout
        layout = Layout()
        layout.split_row(
            Layout(name="metrics", ratio=3),
            Layout(name="positions", ratio=2)
        )
        
        # Enhanced metrics table with USDT focus
        metrics_table = Table(
            title=f"{self.icons['chart']} [bold bright_green]USDT Pairs Performance Dashboard[/]", 
            box=box.ROUNDED,
            header_style="bold bright_blue",
            border_style="bright_green"
        )
        metrics_table.add_column("📊 Metric", style="bright_cyan", width=16)
        metrics_table.add_column("💰 USDT Value", justify="right", width=12)
        metrics_table.add_column("📈 Change", justify="right", width=10)
        metrics_table.add_column("🎯 Status", justify="center", width=8)
        
        # Add metrics with enhanced color coding and status indicators
        for metric, data in summary.get('metrics', {}).items():
            value = data.get('value', '0')
            change = data.get('change', '0%')
            
            # Add USDT symbol if it's a monetary value
            if any(term in metric.lower() for term in ['p&l', 'profit', 'balance', 'value']):
                if not value.endswith('USDT') and not value.startswith('$'):
                    value = f"{value} USDT"
            
            # Determine colors and status based on metric type and value
            if 'P&L' in metric or 'profit' in metric.lower():
                value_color = 'bright_green' if value.startswith('+') else 'bright_red'
                status_icon = self.icons['money'] if value.startswith('+') else '💸'
            elif 'Win Rate' in metric:
                rate = float(value.replace('%', '').replace(' USDT', ''))
                value_color = 'bright_green' if rate > 60 else 'yellow' if rate > 40 else 'bright_red'
                status_icon = self.icons['target'] if rate > 60 else '🎯'
            elif 'Sharpe' in metric:
                ratio = float(value.replace(' USDT', ''))
                value_color = 'bright_green' if ratio > 1.5 else 'yellow' if ratio > 1.0 else 'bright_red'
                status_icon = self.icons['diamond'] if ratio > 1.5 else '💎'
            else:
                value_color = 'white'
                status_icon = self.icons['info']
                
            change_color = 'bright_green' if change.startswith('+') else 'bright_red' if change.startswith('-') else 'white'
            
            metrics_table.add_row(
                f"{self.icons.get('chart', '📈')} {metric}",
                f"[{value_color}][bold]{value}[/bold][/]",
                f"[{change_color}]{change}[/]",
                status_icon
            )
        
        layout["metrics"].update(metrics_table)
        
        # Enhanced positions table with USDT pair highlighting
        positions_table = Table(
            title=f"{self.icons['shield']} [bold bright_blue]Active USDT Swap Positions[/]", 
            box=box.ROUNDED,
            header_style="bold bright_blue",
            border_style="bright_blue"
        )
        positions_table.add_column("🪙 USDT Pair", style="bright_cyan", width=12)
        positions_table.add_column("📈 Side", justify="center", width=8)
        positions_table.add_column("📊 Size", justify="right", width=8)
        positions_table.add_column("💰 P&L (USDT)", justify="right", width=12)
        positions_table.add_column("🎯", justify="center", width=3)
        
        for position in summary.get('positions', []):
            symbol = position['symbol']
            pnl = position['pnl']
            pnl_color = 'bright_green' if pnl > 0 else 'bright_red'
            side = position['side']
            side_color = 'bright_green' if side == 'LONG' else 'bright_red'
            side_icon = '📈' if side == 'LONG' else '📉'
            
            # Highlight USDT in symbol
            if 'USDT' in symbol:
                symbol_display = symbol.replace('USDT', '[bold yellow]USDT[/bold yellow]')
            else:
                symbol_display = symbol
            
            # Status icon based on P&L
            status_icon = self.icons['fire'] if abs(pnl) > 5 else self.icons['target'] if pnl > 0 else '❄️'
            
            positions_table.add_row(
                f"[bright_cyan]{symbol_display}[/]",
                f"[{side_color}]{side_icon} {side}[/]",
                f"[dim]{position['size']}[/]",
                f"[{pnl_color}][bold]{pnl:+.2f} USDT[/bold][/]",
                status_icon
            )
        
        layout["positions"].update(positions_table)
        self.console.print(layout)

    def display_usdt_pairs_scan_status(self, scan_data: Dict[str, Any]):
        """Display USDT pairs scanning status with enhanced visuals"""
        if not RICH_AVAILABLE:
            for key, value in scan_data.items():
                print(f"{key}: {value}")
            return
            
        # Create scan status layout
        layout = Layout()
        layout.split_column(
            Layout(name="scan_header", size=3),
            Layout(name="scan_progress", size=8),
            Layout(name="scan_results", size=6)
        )
        
        # Scan header
        header_text = f"[bold bright_cyan]🔍 USDT Swap Pairs Scanner Status[/]\n"
        header_text += f"[dim]Scanning Bitget USDT perpetual swap contracts only[/]"
        layout["scan_header"].update(Panel(header_text, border_style="bright_cyan"))
        
        # Scan progress
        progress_table = Table(
            title=f"{self.icons['gear']} Scanning Progress",
            box=box.ROUNDED,
            header_style="bold bright_blue"
        )
        progress_table.add_column("📊 Metric", style="bright_cyan", width=20)
        progress_table.add_column("🔢 Value", justify="right", width=15)
        progress_table.add_column("📈 Status", justify="center", width=10)
        
        scan_metrics = {
            "Total USDT Pairs Found": scan_data.get('total_pairs', 0),
            "Pairs Scanned": scan_data.get('pairs_scanned', 0),
            "High Volume Pairs": scan_data.get('high_volume_pairs', 0),
            "Opportunities Found": scan_data.get('opportunities', 0),
            "Active Positions": scan_data.get('active_positions', 0),
            "Scan Progress": f"{scan_data.get('progress', 0):.1f}%"
        }
        
        for metric, value in scan_metrics.items():
            # Color coding based on metric type
            if "Opportunities" in metric or "High Volume" in metric:
                value_color = 'bright_green' if value > 0 else 'dim'
                status_icon = self.icons['target'] if value > 0 else '⏳'
            elif "Progress" in metric:
                progress_val = float(str(value).replace('%', ''))
                value_color = 'bright_green' if progress_val >= 100 else 'yellow' if progress_val >= 50 else 'white'
                status_icon = '✅' if progress_val >= 100 else '🔄' if progress_val >= 50 else '⏳'
            elif "Positions" in metric:
                value_color = 'bright_blue' if value > 0 else 'dim'
                status_icon = self.icons['shield'] if value > 0 else '📋'
            else:
                value_color = 'white'
                status_icon = self.icons['info']
            
            progress_table.add_row(
                f"{self.icons.get('chart', '📊')} {metric}",
                f"[{value_color}][bold]{value}[/bold][/]",
                status_icon
            )
        
        layout["scan_progress"].update(progress_table)
        
        # Top USDT pairs results
        top_pairs = scan_data.get('top_pairs', [])[:5]  # Show top 5
        if top_pairs:
            results_table = Table(
                title=f"{self.icons['fire']} Top USDT Pairs by Volume",
                box=box.MINIMAL,
                header_style="bold bright_green"
            )
            results_table.add_column("🥇 Rank", justify="center", width=6)
            results_table.add_column("🪙 USDT Pair", style="bright_cyan", width=15)
            results_table.add_column("💰 Volume", justify="right", width=12)
            results_table.add_column("📈 Score", justify="right", width=8)
            results_table.add_column("🎯", justify="center", width=3)
            
            for i, pair in enumerate(top_pairs, 1):
                symbol = pair.get('symbol', 'N/A')
                volume = pair.get('volume', 0)
                score = pair.get('viper_score', 0)
                
                # Rank styling
                rank_color = 'gold' if i == 1 else 'bright_white' if i == 2 else 'yellow' if i == 3 else 'white'
                rank_icon = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i}.'
                
                # Highlight USDT
                if 'USDT' in symbol:
                    symbol_display = symbol.replace('USDT', '[bold yellow]USDT[/bold yellow]')
                else:
                    symbol_display = symbol
                
                # Volume formatting
                if volume >= 1000000:
                    volume_str = f"${volume/1000000:.1f}M"
                elif volume >= 1000:
                    volume_str = f"${volume/1000:.0f}K"
                else:
                    volume_str = f"${volume:.0f}"
                
                # Score color
                score_color = 'bright_green' if score >= 80 else 'yellow' if score >= 60 else 'white'
                status_icon = self.icons['fire'] if score >= 90 else self.icons['target'] if score >= 70 else '⭐'
                
                results_table.add_row(
                    f"[{rank_color}]{rank_icon}[/]",
                    f"[bright_cyan]{symbol_display}[/]",
                    f"[bright_green]{volume_str}[/]",
                    f"[{score_color}]{score:.0f}[/]",
                    status_icon
                )
            
            layout["scan_results"].update(results_table)
        else:
            no_results = Panel(
                "[dim]No USDT pairs scanned yet...[/]\n[yellow]Scanning in progress...[/]",
                title="📋 Results",
                border_style="dim"
            )
            layout["scan_results"].update(no_results)
        
        self.console.print(layout)

    def display_api_connection_status(self, connection_data: Dict[str, Any]):
        """Display API connection status with enhanced error reporting"""
        if not RICH_AVAILABLE:
            for service, status in connection_data.items():
                print(f"{service}: {status}")
            return
            
        # Create connection status table
        connection_table = Table(
            title=f"{self.icons['network']} [bold bright_cyan]Bitget API Connection Status[/]",
            box=box.ROUNDED,
            header_style="bold bright_blue",
            border_style="bright_cyan"
        )
        connection_table.add_column("🔗 Service", style="bright_cyan", width=20)
        connection_table.add_column("📡 Status", justify="center", width=12)
        connection_table.add_column("📊 Details", style="dim", width=30)
        connection_table.add_column("🕐 Last Check", justify="center", width=12)
        connection_table.add_column("🎯", justify="center", width=4)
        
        for service, data in connection_data.items():
            status = data.get('status', 'Unknown')
            details = data.get('details', 'No details')
            last_check = data.get('last_check', 'Never')
            error_count = data.get('error_count', 0)
            
            # Status color and icon
            if status == 'Connected':
                status_color = 'bright_green'
                status_icon = self.icons['success']
                service_icon = self.icons['network']
            elif status == 'Connecting':
                status_color = 'yellow'
                status_icon = '🔄'
                service_icon = self.icons['gear']
            elif status == 'Error':
                status_color = 'bright_red'
                status_icon = self.icons['error']
                service_icon = '💥'
            else:
                status_color = 'dim'
                status_icon = '❓'
                service_icon = self.icons['warning']
            
            # Service-specific icons
            if 'bitget' in service.lower():
                service_icon = '₿'
            elif 'api' in service.lower():
                service_icon = self.icons['key']
            elif 'market' in service.lower():
                service_icon = self.icons['chart']
            elif 'swap' in service.lower():
                service_icon = '🔄'
            
            # Error indicator
            final_icon = f"{status_icon}"
            if error_count > 0:
                final_icon += f" ({error_count})"
            
            connection_table.add_row(
                f"{service_icon} [bright_cyan]{service}[/]",
                f"[{status_color}]{status}[/]",
                f"[dim]{details}[/]",
                f"[dim]{last_check}[/]",
                final_icon
            )
        
        self.console.print(connection_table)
        
        # Show additional connection tips if there are errors
        error_services = [s for s, d in connection_data.items() if d.get('status') == 'Error']
        if error_services:
            tips_text = "[bold yellow]🔧 Connection Troubleshooting:[/]\n\n"
            tips_text += "• Check your [bold]BITGET_API_KEY[/], [bold]BITGET_API_SECRET[/], and [bold]BITGET_API_PASSWORD[/] in .env\n"
            tips_text += "• Verify API permissions include [bold]Futures Trading[/] and [bold]Read[/] access\n"
            tips_text += "• Ensure your IP is whitelisted in Bitget API settings\n"
            tips_text += "• Check network connectivity to [bold]api.bitget.com[/]\n"
            tips_text += "• For USDT swaps, ensure [bold]USDT-M Futures[/] is enabled"
            
            tips_panel = Panel(
                tips_text,
                title=f"{self.icons['warning']} [bold yellow]API Connection Issues Detected[/]",
                border_style="yellow",
                padding=(1, 2)
            )
            self.console.print(tips_panel)
    
    def show_progress(self, tasks: List[str], title: str = "Processing"):
        """Show progress for multiple tasks"""
        if not RICH_AVAILABLE:
            for i, task in enumerate(tasks, 1):
                print(f"[{i}/{len(tasks)}] {task}")
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
            print(f"ERROR: {error_msg}")
            if details:
                print(f"Details: {details}")
            return
            
        panel = Panel(
            f"[bold bright_red]{error_msg}[/bold bright_red]\n\n[dim]{details or ''}[/dim]", 
            title=f"{self.icons['error']} [bold bright_red]System Error[/]",
            border_style="bright_red",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def display_warning(self, warning_msg: str, details: Optional[str] = None):
        """Display warning with formatting"""
        if not RICH_AVAILABLE:
            print(f"WARNING: {warning_msg}")
            if details:
                print(f"Details: {details}")
            return
            
        panel = Panel(
            f"[bold bright_yellow]{warning_msg}[/bold bright_yellow]\n\n[dim]{details or ''}[/dim]", 
            title=f"{self.icons['warning']} [bold bright_yellow]System Warning[/]",
            border_style="bright_yellow",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def display_success(self, success_msg: str, details: Optional[str] = None):
        """Display success message with formatting"""
        if not RICH_AVAILABLE:
            print(f"SUCCESS: {success_msg}")
            if details:
                print(f"Details: {details}")
            return
            
        panel = Panel(
            f"[bold bright_green]{success_msg}[/bold bright_green]\n\n[dim]{details or ''}[/dim]", 
            title=f"{self.icons['success']} [bold bright_green]Operation Successful[/]",
            border_style="bright_green",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def display_config_summary(self, config: Dict[str, Any]):
        """Display configuration summary"""
        if not RICH_AVAILABLE:
            for section, values in config.items():
                print(f"\n{section.upper()}:")
                for key, value in values.items():
                    # Mask sensitive information
                    if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                        display_value = '***masked***'
                    else:
                        display_value = str(value)
                    print(f"  {key}: {display_value}")
            return
            
        tree = Tree(f"{self.icons['gear']} [bold bright_cyan]System Configuration[/]")
        
        for section, values in config.items():
            # Use appropriate icons for sections
            section_icon = {
                'trading': self.icons['chart'],
                'api': self.icons['key'], 
                'risk': self.icons['shield'],
                'database': self.icons['database'],
                'network': self.icons['network']
            }.get(section.lower(), self.icons['gear'])
            
            section_branch = tree.add(f"[bold bright_blue]{section_icon} {section.title()}[/]")
            
            for key, value in values.items():
                # Mask sensitive information
                if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                    display_value = f"[dim]{self.icons['lock']} ***masked***[/dim]"
                else:
                    display_value = f"[white]{str(value)}[/white]"
                    
                # Add appropriate icons for common config keys
                key_icon = ''
                if 'risk' in key.lower():
                    key_icon = self.icons['shield']
                elif 'limit' in key.lower() or 'max' in key.lower():
                    key_icon = self.icons['warning']
                elif 'profit' in key.lower():
                    key_icon = self.icons['money']
                elif 'key' in key.lower() or 'secret' in key.lower():
                    key_icon = self.icons['lock']
                else:
                    key_icon = self.icons['gear']
                    
                section_branch.add(f"[dim]{key_icon}[/dim] [cyan]{key}[/]: {display_value}")
        
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

    def display_live_metrics(self, metrics: Dict[str, Any]):
        """Display live trading metrics in a compact format"""
        if not RICH_AVAILABLE:
            for key, value in metrics.items():
                print(f"{key}: {value}")
            return
            
        # Create a compact metrics panel
        metrics_grid = Table.grid(padding=1)
        metrics_grid.add_column(style="cyan", justify="right")
        metrics_grid.add_column(style="white", justify="left")
        
        for key, value in metrics.items():
            # Format value based on type
            if isinstance(value, float):
                if 'pnl' in key.lower() or 'profit' in key.lower():
                    color = 'bright_green' if value > 0 else 'bright_red'
                    formatted_value = f"[{color}]{value:+.2f}%[/]"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
                
            metrics_grid.add_row(f"[dim]{key}:[/]", formatted_value)
        
        panel = Panel(
            metrics_grid,
            title=f"{self.icons['lightning']} [bold]Live Metrics[/]",
            border_style="bright_cyan",
            padding=(0, 1)
        )
        self.console.print(panel)

    def display_trade_execution_log(self, trades: List[Dict[str, Any]]):
        """Display recent trade executions"""
        if not RICH_AVAILABLE:
            for trade in trades:
                print(f"Trade: {trade}")
            return
            
        table = Table(
            title=f"{self.icons['target']} [bold bright_cyan]Recent Trade Executions[/]",
            box=box.MINIMAL,
            header_style="bold bright_blue"
        )
        table.add_column("⏰ Time", style="dim", width=8)
        table.add_column("🪙 Symbol", style="bright_cyan", width=10)
        table.add_column("📈 Action", justify="center", width=8)
        table.add_column("💰 Amount", justify="right", width=10)
        table.add_column("💹 Price", justify="right", width=12)
        table.add_column("🎯 Status", justify="center", width=8)
        
        for trade in trades[-10:]:  # Show last 10 trades
            action = trade.get('action', 'UNKNOWN')
            status = trade.get('status', 'PENDING')
            
            action_color = 'bright_green' if action == 'BUY' else 'bright_red'
            action_icon = '📈' if action == 'BUY' else '📉'
            
            status_color = 'bright_green' if status == 'FILLED' else 'yellow' if status == 'PENDING' else 'bright_red'
            status_icon = self.icons['success'] if status == 'FILLED' else '⏳' if status == 'PENDING' else self.icons['error']
            
            table.add_row(
                f"[dim]{trade.get('time', 'N/A')}[/]",
                f"[bright_cyan]{trade.get('symbol', 'N/A')}[/]",
                f"[{action_color}]{action_icon} {action}[/]",
                f"[white]{trade.get('amount', 'N/A')}[/]",
                f"[white]${trade.get('price', 'N/A')}[/]",
                f"[{status_color}]{status_icon}[/]"
            )
        
        self.console.print(table)

    def display_risk_dashboard(self, risk_data: Dict[str, Any]):
        """Display comprehensive risk management dashboard"""
        if not RICH_AVAILABLE:
            for key, value in risk_data.items():
                print(f"{key}: {value}")
            return
            
        # Create risk dashboard layout
        layout = Layout()
        layout.split_column(
            Layout(name="risk_meters", size=8),
            Layout(name="risk_limits", size=6)
        )
        
        # Risk meters (top section)
        risk_grid = Table.grid(padding=2)
        risk_grid.add_column()
        risk_grid.add_column()
        risk_grid.add_column()
        
        # Create risk level indicators
        portfolio_risk = risk_data.get('portfolio_risk', 0)
        daily_loss = risk_data.get('daily_loss', 0)
        exposure = risk_data.get('exposure', 0)
        
        def get_risk_bar(value, max_value, label):
            percentage = min(value / max_value * 100, 100)
            bar_color = 'bright_green' if percentage < 50 else 'yellow' if percentage < 80 else 'bright_red'
            filled = int(percentage / 10)
            bar = '█' * filled + '░' * (10 - filled)
            return f"[{bar_color}]{bar}[/] [dim]{percentage:.1f}%[/]\n[bold]{label}[/]"
        
        risk_grid.add_row(
            Panel(get_risk_bar(portfolio_risk, 100, "Portfolio Risk"), title="🛡️ Risk Level", border_style="blue"),
            Panel(get_risk_bar(abs(daily_loss), 5, "Daily Loss"), title="📉 Daily Loss", border_style="yellow"),
            Panel(get_risk_bar(exposure, 100, "Exposure"), title="💼 Exposure", border_style="cyan")
        )
        
        layout["risk_meters"].update(risk_grid)
        
        # Risk limits table (bottom section)
        limits_table = Table(
            title=f"{self.icons['shield']} Risk Limits & Controls",
            box=box.ROUNDED,
            header_style="bold bright_blue"
        )
        limits_table.add_column("🛡️ Control", style="bright_cyan")
        limits_table.add_column("⚙️ Limit", justify="center")
        limits_table.add_column("📊 Current", justify="center")
        limits_table.add_column("🚨 Status", justify="center")
        
        limits = risk_data.get('limits', {})
        for control, data in limits.items():
            limit = data.get('limit', 'N/A')
            current = data.get('current', 'N/A')
            status = data.get('status', 'OK')
            
            status_color = 'bright_green' if status == 'OK' else 'yellow' if status == 'WARNING' else 'bright_red'
            status_icon = self.icons['success'] if status == 'OK' else self.icons['warning'] if status == 'WARNING' else self.icons['error']
            
            limits_table.add_row(
                control,
                str(limit),
                str(current),
                f"[{status_color}]{status_icon} {status}[/]"
            )
        
        layout["risk_limits"].update(limits_table)
        self.console.print(layout)

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

def display_usdt_scan_status(scan_data):
    terminal.display_usdt_pairs_scan_status(scan_data)

def display_api_status(connection_data):
    terminal.display_api_connection_status(connection_data)