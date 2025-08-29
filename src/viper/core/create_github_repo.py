#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - GitHub Repository Creation & Upload Script
Uses GitHub MCP server to create a new repository and upload the complete system
"""

import os
import asyncio
import httpx
import sys
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GitHubMCPCreator:
    """GitHub MCP client for repository creation and upload"""

    def __init__(self):
        self.github_pat = os.getenv('GITHUB_PAT', '')
        self.github_owner = os.getenv('GITHUB_OWNER', '')
        self.github_repo = os.getenv('GITHUB_REPO', 'viper-trading-bot-complete')
        self.mcp_base_url = "http://localhost:8001"  # GitHub Manager MCP Server

        # Validate configuration
        if not self.github_pat or self.github_pat == 'github_pat_your_personal_access_token_here':
            print("# X ERROR: Please set a valid GITHUB_PAT in your .env file")
            print("   Get a token from: https://github.com/settings/tokens")
            sys.exit(1)

        if not self.github_owner or self.github_owner == 'your_github_username_here':
            print("# X ERROR: Please set a valid GITHUB_OWNER in your .env file")
            sys.exit(1)

    async def create_repository(self) -> Dict[str, Any]:
        """Create a new GitHub repository using MCP server"""
        print(f"   Repository: {self.github_owner}/{self.github_repo}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Create repository data
                repo_data = {
                    "name": self.github_repo,
                    "description": "# Rocket Complete VIPER Trading Bot - Ultra High-Performance Algorithmic Trading Platform with 20 Microservices",
                    "private": False,
                    "auto_init": True,
                    "license_template": "mit"
                }

                # GitHub API call through MCP server
                response = await client.post(
                    f"{self.mcp_base_url}/github/create-repository",
                    json=repo_data,
                    headers={"Authorization": f"Bearer {self.github_pat}"}
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"   URL: https://github.com/{self.github_owner}/{self.github_repo}")
                    return resul
                else:
                    print(f"# X Failed to create repository: {response.status_code}")
                    return {"error": response.text}

        except Exception as e:
            return {"error": str(e)}

    async def get_all_files(self, directory: str = ".") -> List[Dict[str, str]]:
        """Get all files in the project directory"""

        files_data = []
        project_root = Path(directory)

        # Files and directories to exclude
        exclude_patterns = {
            '__pycache__',
            '.git',
            'node_modules',
            '.pytest_cache',
            '*.pyc',
            '*.pyo',
            '*.log',
            '.env',
            '.env.backup*',
            '*.tmp',
            '*.swp',
            '*.bak'
        }

        for file_path in project_root.rglob('*'):
            if file_path.is_file():
                # Check if file should be excluded
                exclude_file = False
                for pattern in exclude_patterns:
                    if pattern in str(file_path) or file_path.name.endswith(pattern.strip('*')):
                        exclude_file = True
                        break

                if not exclude_file:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Get relative path from project root
                        relative_path = str(file_path.relative_to(project_root))

                        files_data.append({
                            "path": relative_path,
                            "content": content
                        })

                    except Exception as e:
                        print(f"# Warning  Warning: Could not read {file_path}: {e}")

        print(f"# Check Found {len(files_data)} files to upload")
        return files_data

    async def upload_files(self, files_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Upload all files to the GitHub repository"""
        print("📤 Uploading files to GitHub repository...")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Upload files in batches to avoid payload size limits
                batch_size = 50
                uploaded_count = 0

                for i in range(0, len(files_data), batch_size):
                    batch = files_data[i:i + batch_size]
                    print(f"   Uploading batch {i//batch_size + 1}/{(len(files_data) + batch_size - 1)//batch_size}...")

                    upload_data = {
                        "owner": self.github_owner,
                        "repo": self.github_repo,
                        "branch": "main",
                        "files": batch,
                        "message": f"# Rocket VIPER Trading Bot - Complete System Upload (Batch {i//batch_size + 1})"
                    }

                    response = await client.post(
                        f"{self.mcp_base_url}/github/upload-files",
                        json=upload_data,
                        headers={"Authorization": f"Bearer {self.github_pat}"}
                    )

                    if response.status_code == 200:
                        uploaded_count += len(batch)
                    else:
                        print(f"# X Failed to upload batch: {response.status_code}")
                        return {"error": response.text}

                return {
                    "status": "success",
                    "files_uploaded": uploaded_count,
                    "repository_url": f"https://github.com/{self.github_owner}/{self.github_repo}"
                }

        except Exception as e:
            return {"error": str(e)}

    async def create_readme_task(self) -> Dict[str, Any]:
        """Create a task/issue for repository setup"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                task_data = {
                    "title": "# Rocket VIPER Trading Bot - Repository Setup Complete",
                    "body": """## # Party VIPER Trading Bot Repository Created!

This repository contains the complete VIPER Trading Bot system with 20 microservices.

### # Construction **System Architecture:**
- **17 Microservices** - Complete trading pipeline
- **Docker Containerization** - Production-ready deployment
- **MCP Integration** - AI agent support
- **Enterprise Security** - Encrypted vault and access control
- **Real-time Processing** - Sub-second latency trading

### # Rocket **Quick Start:**
```bash
# Clone the repository
git clone https://github.com/{self.github_owner}/{self.github_repo}.git
cd {self.github_repo}

# Start all services
python main.py

# Access dashboard
open http://localhost:8000
```

### # Chart **Key Features:**
- # Check Ultra High-Performance Algorithmic Trading
- # Check Real-time Market Data Streaming
- # Check Advanced Risk Management (2% per trade rule)
- # Check Backtesting Engine with Predictive Ranges
- # Check 50x Leverage Support with Position Limits
- # Check Enterprise Logging & Monitoring

### 🤖 **AI Integration:**
- MCP Server for AI agent communication
- GitHub Project Management integration
- Automated trading signal generation

---
**Status:** 🟢 Repository ready for use
**Created by:** VIPER System Setup Script
""",
                    "labels": ["enhancement", "documentation", "setup-complete"],
                    "assignees": []
                }

                response = await client.post(
                    f"{self.mcp_base_url}/github/create-task",
                    json=task_data,
                    headers={"Authorization": f"Bearer {self.github_pat}"}
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"# Warning  Warning: Could not create task: {response.status_code}")
                    return {"warning": response.text}

        except Exception as e:
            return {"warning": str(e)}

async def main():
    """Main function to create repository and upload codebase"""
    print("# Rocket VIPER Trading Bot - GitHub Repository Creation & Upload")

    # Initialize GitHub MCP creator
    creator = GitHubMCPCreator()

    # Step 1: Create repository
    repo_result = await creator.create_repository()

    if "error" in repo_result:
        print("# X Repository creation failed. Please check your GitHub PAT and try again.")
        return

    # Step 2: Get all project files
    files_data = await creator.get_all_files()

    if not files_data:
        return

    # Step 3: Upload files
    upload_result = await creator.upload_files(files_data)

    if "error" in upload_result:
        return

    # Step 4: Create setup task
    await creator.create_readme_task()

    # Success summary
    print("# Party SUCCESS! VIPER Trading Bot repository created and uploaded!")
    print(f"📁 Repository: https://github.com/{creator.github_owner}/{creator.github_repo}")
    print(f"# Chart Files uploaded: {upload_result.get('files_uploaded', 0)}")
    print("   4. Access dashboard: http://localhost:8000")

if __name__ == "__main__":
    asyncio.run(main())
