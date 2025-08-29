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
            issue_title = f"Performance Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            issue_body = f"""## Performance Report

**Timestamp:** {datetime.now().isoformat()}
**System Status:** {performance_data.get('system_status', 'UNKNOWN')}

### Key Metrics
- **Uptime:** {performance_data.get('system_uptime', 0):.1f} seconds
- **CPU Usage:** {performance_data.get('cpu_usage', 0):.1f}%
- **Memory Usage:** {performance_data.get('memory_usage', 0):.1f}%
- **Active Components:** {performance_data.get('active_components', 0)}

### Trading Performance
- **Trades Executed:** {performance_data.get('total_trades_executed', 0)}
- **Win Rate:** {performance_data.get('win_rate', 0):.1f}%
- **Total P&L:** ${performance_data.get('total_pnl', 0):.2f}

### System Health
- **Active Threads:** {performance_data.get('active_threads', 0)}
- **Network Connections:** {performance_data.get('network_connections', 0)}

---
*Auto-generated by VIPER Ultimate Trading System*
"""

            # Create issue via GitHub API
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            issue_data = {
                'title': issue_title,
                'body': issue_body,
                'labels': ['performance', 'automated', 'viper-system']
            }

            # Extract repo info from remote URL
            if self.remote_url:
                repo_info = self._extract_repo_info(self.remote_url)
                if repo_info:
                    api_url = f"{self.mcp_config['github_api_url']}/repos/{repo_info['owner']}/{repo_info['repo']}/issues"

                    response = requests.post(api_url, headers=headers, json=issue_data)

                    if response.status_code == 201:
                        issue_number = response.json().get('number')
                        logger.info(f"✅ Performance issue created: #{issue_number}")
                        return True
                    else:
                        logger.error(f"❌ GitHub API error: {response.status_code} - {response.text}")

            return False

        except Exception as e:
            logger.error(f"❌ Performance issue creation failed: {e}")
            return False

    async def log_system_performance(self, performance_data: Dict[str, Any]):
        """Log system performance to GitHub"""
        try:
            # Add timestamp
            performance_data['logged_at'] = datetime.now().isoformat()

            # Store in local tracking
            self.performance_logs.append(performance_data)

            # Create local performance log file
            log_file = self.repo_path / f"performance_{datetime.now().strftime('%Y%m%d')}.json"

            with open(log_file, 'a') as f:
                json.dump(performance_data, f, default=str)
                f.write('\n')

            # Commit performance log
            await self.commit_system_changes(
                f"Performance log update - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                [str(log_file)]
            )

            # Push to GitHub
            await self.push_to_github()

            logger.info("✅ System performance logged to GitHub")
            return True

        except Exception as e:
            logger.error(f"❌ Performance logging failed: {e}")
            return False

    async def create_release(self, version: str, release_notes: str):
        """Create a GitHub release"""
        try:
            if not self.github_token:
                return False

            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            release_data = {
                'tag_name': f'v{version}',
                'name': f'VIPER Trading System v{version}',
                'body': release_notes,
                'draft': False,
                'prerelease': False
            }

            if self.remote_url:
                repo_info = self._extract_repo_info(self.remote_url)
                if repo_info:
                    api_url = f"{self.mcp_config['github_api_url']}/repos/{repo_info['owner']}/{repo_info['repo']}/releases"

                    response = requests.post(api_url, headers=headers, json=release_data)

                    if response.status_code == 201:
                        logger.info(f"✅ Release v{version} created on GitHub")
                        return True
                    else:
                        logger.error(f"❌ GitHub release creation failed: {response.status_code}")

            return False

        except Exception as e:
            logger.error(f"❌ Release creation failed: {e}")
            return False

    async def backup_system_state(self, system_state: Dict[str, Any]):
        """Backup system state to GitHub"""
        try:
            # Create backup file
            backup_file = self.repo_path / f"system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(backup_file, 'w') as f:
                json.dump(system_state, f, indent=2, default=str)

            # Commit backup
            await self.commit_system_changes(
                f"System state backup - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                [str(backup_file)]
            )

            # Push to GitHub
            await self.push_to_github()

            logger.info("✅ System state backed up to GitHub")
            return True

        except Exception as e:
            logger.error(f"❌ System backup failed: {e}")
            return False

    async def track_system_changes(self, change_description: str, affected_files: List[str]):
        """Track system changes for version control"""
        try:
            change_data = {
                'timestamp': datetime.now().isoformat(),
                'description': change_description,
                'affected_files': affected_files,
                'commit_hash': None
            }

            # Commit the changes
            success = await self.commit_system_changes(change_description, affected_files)

            if success and self.repo:
                # Get the latest commit hash
                latest_commit = self.repo.head.commit
                change_data['commit_hash'] = latest_commit.hexsha

            # Track the change
            self.commit_history.append(change_data)

            logger.info(f"✅ System change tracked: {change_description}")
            return True

        except Exception as e:
            logger.error(f"❌ Change tracking failed: {e}")
            return False

    async def get_repository_status(self) -> Dict[str, Any]:
        """Get comprehensive repository status"""
        try:
            status = {
                'repository_path': str(self.repo_path),
                'remote_url': self.remote_url,
                'is_git_repo': self.repo is not None,
                'total_commits': len(self.commit_history),
                'performance_logs': len(self.performance_logs),
                'last_commit': None,
                'uncommitted_changes': False,
                'branch_info': {}
            }

            if self.repo:
                # Get last commit info
                try:
                    last_commit = self.repo.head.commit
                    status['last_commit'] = {
                        'hash': last_commit.hexsha,
                        'message': last_commit.message,
                        'author': last_commit.author.name,
                        'date': last_commit.date.isoformat()
                    }
                except Exception:
                    pass

                # Check for uncommitted changes
                status['uncommitted_changes'] = self.repo.is_dirty()

                # Get branch info
                try:
                    current_branch = self.repo.active_branch.name
                    status['branch_info'] = {
                        'current_branch': current_branch,
                        'remote_tracking': str(self.repo.active_branch.tracking_branch) if self.repo.active_branch.tracking_branch else None
                    }
                except Exception:
                    pass

            return status

        except Exception as e:
            logger.error(f"❌ Repository status check failed: {e}")
            return {'error': str(e)}

    async def create_feature_branch(self, feature_name: str) -> bool:
        """Create a new feature branch"""
        try:
            if not self.repo:
                return False

            # Create and checkout new branch
            branch_name = f"feature/{feature_name.replace(' ', '_').lower()}"
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()

            logger.info(f"✅ Feature branch created: {branch_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Feature branch creation failed: {e}")
            return False

    async def merge_feature_branch(self, feature_branch: str, target_branch: str = 'main') -> bool:
        """Merge feature branch into target branch"""
        try:
            if not self.repo:
                return False

            # Switch to target branch
            self.repo.git.checkout(target_branch)

            # Merge feature branch
            self.repo.git.merge(feature_branch)

            # Delete feature branch
            self.repo.git.branch('-d', feature_branch)

            logger.info(f"✅ Feature branch merged: {feature_branch} -> {target_branch}")
            return True

        except Exception as e:
            logger.error(f"❌ Branch merge failed: {e}")
            return False

    def _extract_repo_info(self, remote_url: str) -> Optional[Dict[str, str]]:
        """Extract repository owner and name from remote URL"""
        try:
            # Handle different URL formats
            if 'github.com' in remote_url:
                if remote_url.startswith('https://'):
                    # https://github.com/owner/repo.git
                    parts = remote_url.replace('https://github.com/', '').replace('.git', '').split('/')
                elif remote_url.startswith('git@'):
                    # git@github.com:owner/repo.git
                    parts = remote_url.replace('git@github.com:', '').replace('.git', '').split('/')
                else:
                    return None

                if len(parts) >= 2:
                    return {
                        'owner': parts[0],
                        'repo': parts[1]
                    }

        except Exception as e:
            logger.warning(f"⚠️  Could not extract repo info: {e}")

        return None

    async def get_commit_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent commit history"""
        try:
            if not self.repo:
                return []

            commits = []
            for commit in self.repo.iter_commits(max_count=limit):
                commits.append({
                    'hash': commit.hexsha,
                    'message': commit.message,
                    'author': commit.author.name,
                    'date': commit.date.isoformat(),
                    'files_changed': len(commit.stats.files) if hasattr(commit, 'stats') else 0
                })

            return commits

        except Exception as e:
            logger.error(f"❌ Commit history retrieval failed: {e}")
            return []

    async def create_pull_request(self, title: str, description: str, head_branch: str, base_branch: str = 'main') -> bool:
        """Create a pull request"""
        try:
            if not self.github_token:
                return False

            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            pr_data = {
                'title': title,
                'body': description,
                'head': head_branch,
                'base': base_branch
            }

            if self.remote_url:
                repo_info = self._extract_repo_info(self.remote_url)
                if repo_info:
                    api_url = f"{self.mcp_config['github_api_url']}/repos/{repo_info['owner']}/{repo_info['repo']}/pulls"

                    response = requests.post(api_url, headers=headers, json=pr_data)

                    if response.status_code == 201:
                        pr_number = response.json().get('number')
                        logger.info(f"✅ Pull request created: #{pr_number}")
                        return True
                    else:
                        logger.error(f"❌ Pull request creation failed: {response.status_code}")

            return False

        except Exception as e:
            logger.error(f"❌ Pull request creation failed: {e}")
            return False

    async def commit_and_push(self, message: str, files: List[str] = None) -> bool:
        """
        Commit and push changes to GitHub repository

        Args:
            message: Commit message
            files: List of files to commit (None for all changes)

        Returns:
            bool: Success status
        """
        try:
            if not self.repo:
                logger.error("❌ Repository not initialized")
                return False

            # Add files to staging
            if files:
                for file_path in files:
                    if os.path.exists(file_path):
                        self.repo.index.add([file_path])
                        logger.info(f"✅ Added to staging: {file_path}")
            else:
                # Add all changes
                self.repo.git.add(A=True)
                logger.info("✅ Added all changes to staging")

            # Check if there are changes to commit
            if not self.repo.is_dirty() and len(self.repo.untracked_files) == 0:
                logger.info("ℹ️ No changes to commit")
                return True

            # Create commit
            commit = self.repo.index.commit(message)
            logger.info(f"✅ Changes committed: {commit.hexsha[:8]}")

            # Push to remote
            if self.remote_url:
                try:
                    origin = self.repo.remote('origin')
                    origin.push()
                    logger.info("✅ Changes pushed to remote repository")
                    return True
                except Exception as e:
                    logger.error(f"❌ Push failed: {e}")
                    return False
            else:
                logger.warning("⚠️ No remote repository configured")
                return True

        except Exception as e:
            logger.error(f"❌ Commit and push failed: {e}")
            return False

# Example usage and testing functions
async def test_github_integration():
    """Test GitHub MCP integration"""
    print("🔗 Testing GitHub MCP Integration...")

    # Initialize integration
    github_mcp = GitHubMCPIntegration()

    # Test repository status
    status = await github_mcp.get_repository_status()
    print(f"📊 Repository Status: {json.dumps(status, indent=2)}")

    # Test commit history
    commits = await github_mcp.get_commit_history(5)
    print(f"📝 Recent Commits: {len(commits)} found")

    # Test performance logging
    test_performance = {
        'system_status': 'TESTING',
        'cpu_usage': 45.2,
        'memory_usage': 67.8,
        'total_trades_executed': 150,
        'win_rate': 68.5,
        'total_pnl': 1250.75
    }

    success = await github_mcp.log_system_performance(test_performance)
    print(f"📈 Performance logging: {'SUCCESS' if success else 'FAILED'}")

    print("✅ GitHub MCP integration test completed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_github_integration())
