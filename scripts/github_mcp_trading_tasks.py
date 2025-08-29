#!/usr/bin/env python3
"""
🐙 GITHUB MCP TRADING TASKS - Automated Trading Task Management
============================================================

GitHub MCP integration for automated trading task creation and management.

Features:
- Automated GitHub issue creation for trading operations
- Real-time performance monitoring via GitHub
- Alert system integration with GitHub notifications
- Strategy optimization task automation
- Task status tracking and updates

Author: VIPER Development Team
Version: 1.0.0
Date: 2025-01-29
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class GitHubTask:
    """Represents a GitHub issue/task"""
    title: str
    body: str
    labels: List[str]
    assignees: List[str]
    milestone: Optional[str] = None
    issue_number: Optional[int] = None
    status: str = "open"  # open, closed, in_progress
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

@dataclass
class TradingAlert:
    """Represents a trading alert"""
    alert_type: str  # 'performance', 'risk', 'error', 'opportunity'
    severity: str    # 'low', 'medium', 'high', 'critical'
    title: str
    description: str
    strategy_name: Optional[str] = None
    symbol: Optional[str] = None
    value: Optional[float] = None
    timestamp: str = ""

class GitHubMCPTradingTasks:
    """GitHub MCP integration for trading task management"""

    def __init__(self, github_token: str = None, repo_owner: str = None, repo_name: str = None):
        self.github_token = github_token or os.getenv('GITHUB_PAT')
        self.repo_owner = repo_owner or os.getenv('GITHUB_OWNER', 'Stressica1')
        self.repo_name = repo_name or os.getenv('GITHUB_REPO', 'viper-')
        self.base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        self.session = requests.Session()

        if self.github_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            })

        self.tasks: Dict[str, GitHubTask] = {}
        self.active_alerts: List[TradingAlert] = []

    def create_trading_task(self, task_type: str, title: str, description: str,
                          strategy_name: str = None, priority: str = "medium",
                          labels: List[str] = None) -> Optional[int]:
        """Create a new GitHub issue for trading task"""
        if not self.github_token:
            return None

        if labels is None:
            labels = ["trading", task_type]

        # Add priority and strategy labels
        labels.extend([f"priority-{priority}"])
        if strategy_name:
            labels.append(f"strategy-{strategy_name.lower().replace(' ', '-')}")

        issue_data = {
            "title": f"# Rocket {title}",
            "body": self._format_task_body(task_type, description, strategy_name),
            "labels": labels
        }

        try:
            response = self.session.post(f"{self.base_url}/issues", json=issue_data)

            if response.status_code == 201:
                issue = response.json()
                issue_number = issue['number']

                # Store task info
                task = GitHubTask(
                    title=title,
                    body=description,
                    labels=labels,
                    assignees=[],
                    issue_number=issue_number,
                    status="open",
                    created_at=datetime.now().isoformat()
                )
                self.tasks[f"task_{issue_number}"] = task

                print(f"# Check Created GitHub task #{issue_number}: {title}")
                return issue_number
            else:
                print(f"# X Failed to create GitHub task: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            return None

    def create_performance_monitoring_task(self, strategy_name: str, metrics: Dict[str, Any]) -> Optional[int]:
        """Create task for performance monitoring"""
        title = f"# Chart Performance Monitoring: {strategy_name}"

        description = f"""
## Performance Monitoring Task

**Strategy:** {strategy_name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Current Metrics
- **Win Rate:** {metrics.get('win_rate', 'N/A')}%
- **Sharpe Ratio:** {metrics.get('sharpe_ratio', 'N/A')}
- **Total Return:** {metrics.get('total_return', 'N/A')}%
- **Max Drawdown:** {metrics.get('max_drawdown', 'N/A')}%

### Tasks
- [ ] Review strategy performance metrics
- [ ] Analyze recent trades and outcomes
- [ ] Check for optimization opportunities
- [ ] Update strategy parameters if needed
- [ ] Schedule next performance review

### Alerts
{self._check_performance_alerts(strategy_name, metrics)}
        """

        return self.create_trading_task(
            "performance",
            title,
            description,
            strategy_name=strategy_name,
            priority="medium",
            labels=["monitoring", "performance", "automated"]
        )

    def create_risk_alert_task(self, alert: TradingAlert) -> Optional[int]:
        """Create task for risk alerts"""
        title = f"🚨 Risk Alert: {alert.title}"

        description = f"""
## Risk Alert

**Type:** {alert.alert_type}
**Severity:** {alert.severity.upper()}
**Strategy:** {alert.strategy_name or 'Portfolio'}
**Symbol:** {alert.symbol or 'N/A'}
**Value:** {alert.value or 'N/A'}
**Time:** {alert.timestamp}

### Description
{alert.description}

### Required Actions
- [ ] Assess risk impact on portfolio
- [ ] Review position sizing and limits
- [ ] Consider position adjustments
- [ ] Update risk management parameters
- [ ] Monitor for further alerts

### Risk Assessment
{self._assess_alert_risk(alert)}
        """

        priority = "high" if alert.severity in ["high", "critical"] else "medium"

        return self.create_trading_task(
            "risk",
            title,
            description,
            strategy_name=alert.strategy_name,
            priority=priority,
            labels=["alert", "risk", f"severity-{alert.severity}"]
        )

    def create_strategy_optimization_task(self, strategy_name: str, recommendations: List[str]) -> Optional[int]:
        """Create task for strategy optimization"""
        title = f"# Tool Strategy Optimization: {strategy_name}"

        recommendations_text = "\n".join(f"- [ ] {rec}" for rec in recommendations)

        description = f"""
## Strategy Optimization Task

**Strategy:** {strategy_name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Optimization Recommendations
{recommendations_text}

### Implementation Plan
1. [ ] Analyze current strategy performance
2. [ ] Implement recommended changes
3. [ ] Backtest optimization changes
4. [ ] Deploy to live trading (if successful)
5. [ ] Monitor post-optimization performance

### Risk Considerations
- [ ] Ensure optimization doesn't increase risk excessively
- [ ] Validate changes with historical data
- [ ] Monitor for overfitting in backtests
- [ ] Plan rollback strategy if needed

### Success Metrics
- [ ] Improved Sharpe ratio
- [ ] Better risk-adjusted returns
- [ ] Consistent performance improvement
- [ ] No increase in maximum drawdown
        """

        return self.create_trading_task(
            "optimization",
            title,
            description,
            strategy_name=strategy_name,
            priority="medium",
            labels=["optimization", "strategy", "improvement"]
        )

    def create_live_trading_task(self, operation: str, details: Dict[str, Any]) -> Optional[int]:
        """Create task for live trading operations"""
        title = f"💰 Live Trading: {operation}"

        description = f"""
## Live Trading Operation

**Operation:** {operation}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Details
{json.dumps(details, indent=2)}

### Checklist
- [ ] Verify trading conditions
- [ ] Check account balance and limits
- [ ] Confirm risk management settings
- [ ] Execute trading operation
- [ ] Monitor position performance
- [ ] Update trading records

### Risk Controls
- [ ] Position size within limits
- [ ] Risk per trade validated
- [ ] Emergency stop accessible
- [ ] Monitoring systems active
        """

        return self.create_trading_task(
            "live_trading",
            title,
            description,
            priority="high",
            labels=["live-trading", operation.lower(), "execution"]
        )

    def update_task_status(self, issue_number: int, status: str, comment: str = None) -> bool:
        """Update GitHub issue status"""
        if not self.github_token:
            return False

        try:
            # Add comment if provided
            if comment:
                comment_data = {"body": comment}
                self.session.post(f"{self.base_url}/issues/{issue_number}/comments", json=comment_data)

            # Update labels based on status
            if status == "in_progress":
                # Add "in-progress" label
                update_data = {"labels": ["in-progress"]}
                self.session.patch(f"{self.base_url}/issues/{issue_number}", json=update_data)
            elif status == "completed":
                # Close the issue
                update_data = {"state": "closed"}
                self.session.patch(f"{self.base_url}/issues/{issue_number}", json=update_data)

            # Update local task status
            task_key = f"task_{issue_number}"
            if task_key in self.tasks:
                self.tasks[task_key].status = status
                self.tasks[task_key].updated_at = datetime.now().isoformat()

            return True

        except Exception as e:
            return False

    def get_open_tasks(self) -> List[Dict[str, Any]]:
        """Get all open GitHub issues/tasks"""
        if not self.github_token:
            return []

        try:
            params = {
                "state": "open",
                "labels": "trading",
                "sort": "created",
                "direction": "desc"
            }

            response = self.session.get(f"{self.base_url}/issues", params=params)

            if response.status_code == 200:
                issues = response.json()
                return [{
                    'number': issue['number'],
                    'title': issue['title'],
                    'state': issue['state'],
                    'labels': [label['name'] for label in issue['labels']],
                    'created_at': issue['created_at'],
                    'updated_at': issue['updated_at']
                } for issue in issues]
            else:
                print(f"# X Failed to get tasks: {response.status_code}")
                return []

        except Exception as e:
            return []

    def create_daily_performance_report(self, dashboard_data: Dict[str, Any]) -> Optional[int]:
        """Create daily performance report task"""
        title = f"📈 Daily Performance Report: {datetime.now().strftime('%Y-%m-%d')}"

        portfolio_summary = dashboard_data.get('portfolio_summary', {})
        strategy_table = dashboard_data.get('strategy_table', 'No data available')

        description = f"""
## Daily Performance Report

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Generated:** {datetime.now().strftime('%H:%M:%S')}

### Portfolio Summary
- **Total Strategies:** {portfolio_summary.get('total_strategies', 'N/A')}
- **Active Strategies:** {portfolio_summary.get('active_strategies', 'N/A')}
- **Portfolio Value:** ${portfolio_summary.get('total_portfolio_value', 0):,.2f}
- **Daily P&L:** ${portfolio_summary.get('daily_pnl', 0):,.2f}
- **Total P&L:** ${portfolio_summary.get('total_pnl', 0):,.2f}
- **Portfolio Return:** {portfolio_summary.get('portfolio_return', 0):.1f}%

### Strategy Performance
```
{strategy_table}
```

### Daily Tasks
- [ ] Review portfolio performance
- [ ] Check for strategy alerts
- [ ] Monitor risk metrics
- [ ] Update strategy weights if needed
- [ ] Plan next day's trading activities

### Notes
_Daily performance review completed. All systems operational._
        """

        return self.create_trading_task(
            "daily_report",
            title,
            description,
            priority="low",
            labels=["daily", "report", "performance", "automated"]
        )

    def create_weekly_optimization_review(self, recommendations: List[str]) -> Optional[int]:
        """Create weekly strategy optimization review"""
        title = f"🔄 Weekly Strategy Optimization Review: Week of {datetime.now().strftime('%Y-%m-%d')}"

        recommendations_text = "\n".join(f"- [ ] {rec}" for rec in recommendations)

        description = f"""
## Weekly Strategy Optimization Review

**Week Of:** {datetime.now().strftime('%Y-%m-%d')}
**Generated:** {datetime.now().strftime('%H:%M:%S')}

### Optimization Recommendations
{recommendations_text}

### Review Checklist
- [ ] Analyze weekly performance trends
- [ ] Review strategy correlations
- [ ] Assess risk-adjusted returns
- [ ] Check for market regime changes
- [ ] Evaluate position sizing effectiveness
- [ ] Review drawdown management
- [ ] Update strategy parameters
- [ ] Plan A/B testing for optimizations

### Implementation Priority
1. **High Priority:** Risk management improvements
2. **Medium Priority:** Performance optimizations
3. **Low Priority:** Minor parameter tweaks

### Success Metrics
- [ ] Maintain Sharpe ratio > 1.5
- [ ] Keep max drawdown < 10%
- [ ] Achieve win rate > 55%
- [ ] Improve profit factor > 1.8
        """

        return self.create_trading_task(
            "weekly_review",
            title,
            description,
            priority="medium",
            labels=["weekly", "review", "optimization", "strategy"]
        )

    def _format_task_body(self, task_type: str, description: str, strategy_name: str = None) -> str:
        """Format task body with standard template"""
        strategy_info = f"**Strategy:** {strategy_name}" if strategy_name else ""

        return f"""
## Trading Task

**Type:** {task_type}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{strategy_info}

### Description
{description}

### Status
- [ ] Task created
- [ ] In progress
- [ ] Completed
- [ ] Verified

### Notes
_Automatically generated by VIPER Trading System_
        """

    def _check_performance_alerts(self, strategy_name: str, metrics: Dict[str, Any]) -> str:
        """Check for performance alerts"""
        alerts = []

        if metrics.get('sharpe_ratio', 0) < 1.5:
            alerts.append(f"# Warning  Sharpe ratio below target: {metrics.get('sharpe_ratio', 0):.2f} < 1.5")

        if metrics.get('win_rate', 0) < 55:
            alerts.append(f"# Warning  Win rate below target: {metrics.get('win_rate', 0):.1f}% < 55%")

        if metrics.get('max_drawdown', 0) < -10:
            alerts.append(f"🚨 High drawdown: {metrics.get('max_drawdown', 0):.1f}% < -10%")

        if not alerts:
            alerts.append("# Check All metrics within acceptable ranges")

        return "\n".join(alerts)

    def _assess_alert_risk(self, alert: TradingAlert) -> str:
        """Assess risk level of alert"""
        if alert.severity == "critical":
            return """
🚨 **CRITICAL RISK LEVEL**
- Immediate action required
- Consider pausing strategy
- Review all positions
- Contact risk management team
            """
        elif alert.severity == "high":
            return """
# Warning  **HIGH RISK LEVEL**
- Urgent attention needed
- Monitor closely
- Consider position adjustments
- Review risk limits
            """
        elif alert.severity == "medium":
            return """
# Chart **MEDIUM RISK LEVEL**
- Monitor situation
- Review in next update
- Consider minor adjustments
- Track for escalation
            """
        else:
            return """
ℹ️  **LOW RISK LEVEL**
- Monitor as part of routine
- No immediate action required
- Log for trend analysis
            """

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='GitHub MCP Trading Tasks Manager')
    parser.add_argument('--list', '-l', action='store_true', help='List all open trading tasks')
    parser.add_argument('--create', '-c', help='Create task: performance|risk|optimization|trading')
    parser.add_argument('--strategy', '-s', help='Strategy name for task')
    parser.add_argument('--title', '-t', help='Task title')
    parser.add_argument('--description', '-d', help='Task description')
    parser.add_argument('--daily-report', action='store_true', help='Create daily performance report')
    parser.add_argument('--weekly-review', action='store_true', help='Create weekly optimization review')

    args = parser.parse_args()

    # Initialize GitHub MCP client
    github_client = GitHubMCPTradingTasks()

    if not github_client.github_token:
        print("# X GitHub token not configured. Set GITHUB_PAT environment variable.")
        sys.exit(1)

    if args.list:
        tasks = github_client.get_open_tasks()
        if tasks:
            for task in tasks:
                print(f"  #{task['number']}: {task['title']} ({task['state']})")
        else:

    elif args.create:
        if not args.title or not args.description:
            print("# X Title and description required for task creation")
            sys.exit(1)

        issue_number = github_client.create_trading_task(
            args.create,
            args.title,
            args.description,
            strategy_name=args.strategy
        )

        if issue_number:
        else:

    elif args.daily_report:
        # Mock dashboard data for demonstration
        mock_data = {
            'portfolio_summary': {
                'total_strategies': 5,
                'active_strategies': 5,
                'total_portfolio_value': 100000.0,
                'daily_pnl': 1250.50,
                'total_pnl': 15750.75,
                'portfolio_return': 15.75
            },
            'strategy_table': "Mock strategy performance table"
        }

        issue_number = github_client.create_daily_performance_report(mock_data)
        if issue_number:
            print(f"# Check Created daily report task #{issue_number}")

    elif args.weekly_review:
        recommendations = [
            "Optimize position sizing based on volatility",
            "Review correlation between strategies",
            "Update stop-loss levels",
            "Implement dynamic leverage adjustment"
        ]

        issue_number = github_client.create_weekly_optimization_review(recommendations)
        if issue_number:
            print(f"# Check Created weekly review task #{issue_number}")

    else:

if __name__ == '__main__':
    main()
