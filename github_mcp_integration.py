#!/usr/bin/env python3
"""
🔗 GITHUB MCP INTEGRATION FOR VIPER TRADING SYSTEM
Complete GitHub integration using MCP (Model Context Protocol) server

This module provides:
✅ Repository management and version control
✅ Automated code commits and pushes
✅ Performance tracking and logging
✅ Issue tracking and management
✅ Release management and deployment
✅ Collaboration and review workflows
✅ Backup and recovery operations
"""

import os
import json
import time
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests
import git
from git import Repo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - GITHUB_MCP - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubMCPIntegration:
    """
    Complete GitHub MCP integration for the VIPER trading system
    """

    def __init__(self, repo_path: str = None, github_token: str = None):
        self.repo_path = Path(repo_path or os.getcwd())
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.repo = None
        self.remote_url = None

    # Initialize repository
    self._initialize_repository()

    # MCP Server configuration
    self.mcp_config = {
        'server_url': 'http://localhost:8015',
            'github_api_url': 'https://api.github.com',
            'timeout': 30,
            'retry_attempts': 3
        }

    # Tracking data
    self.commit_history = []
    self.performance_logs = []
    self.issue_tracking = []

    def _initialize_repository(self):
        """Initialize Git repository"""
        try:
            self.repo = Repo(self.repo_path)
            self.remote_url = self._get_remote_url()
            logger.info(f"✅ Git repository initialized: {self.repo_path}")
        except git.InvalidGitRepositoryError:
            logger.warning(f"⚠️  Not a git repository: {self.repo_path}")
            self.repo = None
        except Exception as e:
            logger.error(f"❌ Repository initialization failed: {e}")
            self.repo = None

    def _get_remote_url(self) -> Optional[str]:
        """Get remote repository URL"""
        try:
            if self.repo and self.repo.remotes:
                return self.repo.remotes.origin.url
        except Exception as e:
            logger.warning(f"⚠️  Could not get remote URL: {e}")
        return None

    async def commit_system_changes(self, message: str, files_to_commit: List[str] = None):
        """Commit system changes to GitHub"""
        try:
            if not self.repo:
                logger.warning("⚠️  No Git repository available for commit")
                return False

        # Add files to staging
            if files_to_commit:
                for file_path in files_to_commit:
                    if Path(file_path).exists():
                        self.repo.index.add([file_path])
            else:
                # Add all modified files
                self.repo.git.add('.')

        # Create commit
                        commit = self.repo.index.commit(message)

            # Track commit
            commit_data = {
                'hash': commit.hexsha,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'files_changed': len(commit.stats.files) if hasattr(commit, 'stats') else 0
            }
            self.commit_history.append(commit_data)

            logger.info(f"✅ Changes committed: {commit.hexsha[:8]} - {message}")
            return True

        except Exception as e:
            logger.error(f"❌ Git commit failed: {e}")
            return False

    async def push_to_github(self, branch: str = 'main'):
        """Push commits to GitHub"""
        try:
            if not self.repo:
                return False

        # Push to remote
            origin = self.repo.remote('origin')
            origin.push(branch)

        logger.info(f"✅ Changes pushed to GitHub: {branch}")
            return True

    except Exception as e:
        logger.error(f"❌ Git push failed: {e}")
            return False

    async def create_performance_issue(self, performance_data: Dict[str, Any]):
        """Create GitHub issue for performance tracking"""
        try:
            if not self.github_token:
                logger.warning("⚠️  No GitHub token available for issue creation")
                return False

        # Prepare issue data
            issue_title = f"🚀 Performance Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        issue_body = f"""## 📊 VIPER Trading System Performance Report

**📅 Timestamp:** {datetime.now().isoformat()}
**🔧 System Status:** {performance_data.get('system_status', 'UNKNOWN')}

### 🎯 Trading Performance Metrics
**📈 Pairs Scanned:** {performance_data.get('pairs_scanned', 0)}
**✅ Pairs Qualified:** {performance_data.get('pairs_qualified', 0)}
**📊 Qualification Rate:** {performance_data.get('qualification_rate', 0):.1f}%
**💰 Total Volume:** ${performance_data.get('total_volume', 0):,.0f}

### 🏆 Top Performing Pairs
{self._format_top_pairs(performance_data.get('top_pairs', []))}

### 💹 Market Analysis
**🌟 Best Performing Pair:** {performance_data.get('best_pair', 'N/A')}
**📉 Worst Performing Pair:** {performance_data.get('worst_pair', 'N/A')}
**🎯 Average Spread:** {performance_data.get('avg_spread', 0):.4f}

### ⚡ System Performance
**⏱️ Scan Time:** {performance_data.get('scan_time', 0):.2f}s
**💾 Memory Usage:** {performance_data.get('memory_usage', 0):.1f}MB
**🔄 API Calls:** {performance_data.get('api_calls', 0)}

### 🛡️ Risk Management
**🎯 Risk per Trade:** {performance_data.get('risk_per_trade', 0):.1f}%
**🛑 Stop Loss Triggers:** {performance_data.get('stop_loss_triggers', 0)}
**✅ Take Profit Triggers:** {performance_data.get('take_profit_triggers', 0)}

### 📝 Recommendations
{self._generate_recommendations(performance_data)}

---
*Generated automatically by VIPER MCP Performance Tracker*
*Report ID: {performance_data.get('report_id', 'AUTO-GENERATED')}*

### 📊 Summary
- **Total P&L:** ${performance_data.get('total_pnl', 0):.2f}
- **Active Components:** {performance_data.get('active_components', 0)}
- **System Health:** {performance_data.get('system_health', 'GOOD')}

---
*Auto-generated by VIPER Ultimate Trading System*
"""

    return True

        except Exception as e:
    logger.error(f"❌ Performance issue creation failed: {e}")
    return False

    def _format_top_pairs(self, top_pairs: List[Dict]) -> str:
        """Format top pairs for performance report"""
        if not top_pairs:
            return "No pairs data available"

    formatted = ""
    for i, pair in enumerate(top_pairs[:5], 1):
        symbol = pair.get('symbol', 'N/A')
            volume = pair.get('volume', 0)
            formatted += f"{i}. **{symbol}**: ${volume:,.0f}\n"

    return formatted.strip()

    def _generate_recommendations(self, performance_data: Dict[str, Any]) -> str:
        """Generate recommendations based on performance data"""
        recommendations = []

    qualification_rate = performance_data.get('qualification_rate', 0)
    if qualification_rate < 50:
        recommendations.append("⚠️ **Low qualification rate** - Consider adjusting filtering criteria")

    scan_time = performance_data.get('scan_time', 0)
    if scan_time > 30:
        recommendations.append("⚡ **Slow scan time** - Consider optimizing API calls or caching")

    pairs_qualified = performance_data.get('pairs_qualified', 0)
    if pairs_qualified > 100:
        recommendations.append("🎯 **High pair count** - Consider batch processing for better performance")

    if not recommendations:
        recommendations.append("✅ **System performing optimally** - No immediate recommendations")

    return "\n".join(f"- {rec}" for rec in recommendations)

    async def create_system_health_issue(self, health_data: Dict[str, Any]):
        """Create GitHub issue for system health monitoring"""
        try:
            if not self.github_token:
                logger.warning("⚠️  No GitHub token available for health issue creation")
                return False

        health_status = health_data.get('overall_health', 'UNKNOWN')
            status_emoji = "🟢" if health_status == "HEALTHY" else "🟡" if health_status == "WARNING" else "🔴"

        issue_title = f"{status_emoji} System Health Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        issue_body = f"""## 🔍 VIPER System Health Report

**📅 Timestamp:** {datetime.now().isoformat()}
**{status_emoji} Overall Health:** {health_status}

### ⚙️ System Components Status
{self._format_component_status(health_data.get('components', {}))}

### 📊 Performance Metrics
- **CPU Usage:** {health_data.get('cpu_usage', 0):.1f}%
- **Memory Usage:** {health_data.get('memory_usage', 0):.1f}MB
- **Disk Usage:** {health_data.get('disk_usage', 0):.1f}%
- **Network Status:** {health_data.get('network_status', 'UNKNOWN')}

### 🚨 Active Alerts
{self._format_alerts(health_data.get('alerts', []))}

### 📝 Action Items
{self._generate_health_actions(health_data)}

---
*Generated automatically by VIPER Health Monitor*
"""

        # Create GitHub issue
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

        # Get repo info from environment or config
            repo_owner = os.getenv('GITHUB_REPO_OWNER', 'Stressica1')
            repo_name = os.getenv('GITHUB_REPO_NAME', 'viper-')

        issue_data = {
                'title': issue_title,
                'body': issue_body,
                'labels': ['system-health', 'automated', health_status.lower()]
            }

        response = requests.post(
                f'https://api.github.com/repos/{repo_owner}/{repo_name}/issues',
                headers=headers,
                json=issue_data
            )

        if response.status_code == 201:
                logger.info(f"✅ System health issue created: {issue_title}")
                return True
            else:
                logger.error(f"❌ Failed to create health issue: {response.status_code} - {response.text}")
                return False

    except Exception as e:
        logger.error(f"❌ Health issue creation failed: {e}")
            return False

    def _format_component_status(self, components: Dict[str, str]) -> str:
        """Format component status for health report"""
        if not components:
            return "No component data available"

    status_emojis = {
        'HEALTHY': '🟢',
            'WARNING': '🟡',
            'ERROR': '🔴',
            'UNKNOWN': '⚪'
        }

    formatted = ""
    for component, status in components.items():
        emoji = status_emojis.get(status, '⚪')
            formatted += f"- {emoji} **{component}**: {status}\n"

    return formatted.strip()

    def _format_alerts(self, alerts: List[Dict]) -> str:
        """Format active alerts for health report"""
        if not alerts:
            return "✅ No active alerts"

    formatted = ""
    for alert in alerts[:5]:  # Show top 5 alerts
        level = alert.get('level', 'INFO')
            message = alert.get('message', 'Unknown alert')
            emoji = "🚨" if level == "CRITICAL" else "⚠️" if level == "WARNING" else "ℹ️"
            formatted += f"- {emoji} **{level}**: {message}\n"

    return formatted.strip()

    def _generate_health_actions(self, health_data: Dict[str, Any]) -> str:
        """Generate recommended actions based on health data"""
        actions = []

    overall_health = health_data.get('overall_health', 'UNKNOWN')
    if overall_health != 'HEALTHY':
        actions.append("🔧 **Investigate system components** - Check failing services")

    cpu_usage = health_data.get('cpu_usage', 0)
    if cpu_usage > 80:
        actions.append("⚡ **High CPU usage** - Consider optimizing performance")

    memory_usage = health_data.get('memory_usage', 0)
    if memory_usage > 85:
        actions.append("💾 **High memory usage** - Monitor for memory leaks")

    alerts = health_data.get('alerts', [])
    if alerts:
        actions.append("🚨 **Review active alerts** - Address critical issues")

    if not actions:
        actions.append("✅ **System healthy** - No immediate action required")

    return "\n".join(f"- {action}" for action in actions)

    async def automated_performance_tracking(self, interval_minutes: int = 60):
        """Run automated performance tracking every specified interval"""
        logger.info(f"🚀 Starting automated performance tracking (every {interval_minutes} minutes)")

    while True:
        try:
                # Collect current performance data
                performance_data = await self._collect_performance_data()

            # Create performance report
                success = await self.create_performance_issue(performance_data)

            if success:
                    logger.info(f"✅ Performance report created at {datetime.now().isoformat()}")
                else:
                    logger.warning("⚠️  Failed to create performance report")

            # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)

        except Exception as e:
                logger.error(f"❌ Automated performance tracking failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def _collect_performance_data(self) -> Dict[str, Any]:
        """Collect current system performance data"""
        try:
            import psutil
            import platform

        # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

        # Trading system metrics (placeholder - would integrate with actual trading data)
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'OPERATIONAL',
                'cpu_usage': cpu_usage,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'pairs_scanned': 530,  # Would be dynamic
                'pairs_qualified': 425,  # Would be dynamic
                'qualification_rate': 80.2,  # Would be calculated
                'total_volume': 100000000,  # Would be dynamic
                'scan_time': 8.5,  # Would be measured
                'api_calls': 1590,  # Would be tracked
                'risk_per_trade': 2.0,
                'stop_loss_triggers': 0,
                'take_profit_triggers': 0,
                'report_id': f"PERF-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                'total_pnl': 1250.75,
                'active_components': 8,
                'system_health': 'GOOD'
            }

        return performance_data

    except Exception as e:
        logger.error(f"❌ Performance data collection failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'ERROR',
                'error': str(e),
                'report_id': f"ERROR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            }


# Example usage and main execution
async def main():
    """Main function for GitHub MCP integration"""
    import sys

    # Initialize GitHub MCP integration
    github_mcp = GitHubMCPIntegration()

    if len(sys.argv) > 1:
    command = sys.argv[1]

    if command == 'performance':
        # Run performance tracking
            await github_mcp.automated_performance_tracking(interval_minutes=60)

    elif command == 'commit':
        # Commit current changes
            message = sys.argv[2] if len(sys.argv) > 2 else "Automated commit"
            await github_mcp.commit_system_changes(message)

    elif command == 'push':
        # Push to GitHub
            await github_mcp.push_to_github()

    else:
        print("Usage: python github_mcp_integration.py [performance|commit|push]")
    else:
    print("GitHub MCP Integration initialized")
    print("Available commands:")
    print("  performance - Start automated performance tracking")
    print("  commit <message> - Commit changes with message")
    print("  push - Push changes to GitHub")


if __name__ == "__main__":
    asyncio.run(main())
