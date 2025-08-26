#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - GitHub Project Management MCP Server
Dedicated MCP server for GitHub task management and project operations
"""

import os
import asyncio
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubProjectManager:
    """GitHub Project Management MCP Server"""
    
    def __init__(self):
        self.app = FastAPI(
            title="GitHub Project Manager MCP Server",
            description="MCP Server for GitHub task management and project operations",
            version="1.0.0"
        )
        
        # GitHub Configuration
        self.github_pat = os.getenv('GITHUB_PAT', '')
        self.github_owner = os.getenv('GITHUB_OWNER', '')
        self.github_repo = os.getenv('GITHUB_REPO', '')
        
        # Validate configuration
        if not self.github_pat:
            logger.warning("GitHub PAT not configured")
        if not self.github_owner or not self.github_repo:
            logger.warning("GitHub repository not fully configured")
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "GitHub Project Manager MCP Server",
                "status": "operational",
                "version": "1.0.0"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "github_configured": bool(self.github_pat),
                "repository": f"{self.github_owner}/{self.github_repo}" if self.github_owner and self.github_repo else "Not configured"
            }
        
        @self.app.post("/github/create-task")
        async def create_github_task(request: Request):
            """Create a GitHub task/issue"""
            try:
                data = await request.json()
                return await self.create_github_issue(data)
            except Exception as e:
                logger.error(f"Error creating GitHub task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/github/tasks")
        async def list_github_tasks(status: str = "open"):
            """List GitHub tasks/issues"""
            try:
                return await self.list_github_issues(status)
            except Exception as e:
                logger.error(f"Error listing GitHub tasks: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/github/update-task")
        async def update_github_task(request: Request):
            """Update a GitHub task/issue"""
            try:
                data = await request.json()
                return await self.update_github_issue(data)
            except Exception as e:
                logger.error(f"Error updating GitHub task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/github/projects")
        async def list_github_projects():
            """List GitHub projects"""
            try:
                return await self.list_github_projects()
            except Exception as e:
                logger.error(f"Error listing GitHub projects: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def create_github_issue(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a GitHub issue via MCP"""
        try:
            if not self.github_pat:
                return {"error": "GitHub PAT not configured", "status": "error"}

            headers = {
                "Authorization": f"token {self.github_pat}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json"
            }

            issue_data = {
                "title": data.get("title", "New Task"),
                "body": data.get("body", ""),
                "labels": data.get("labels", ["task"]),
                "assignees": data.get("assignees", [])
            }

            url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/issues"
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=issue_data, headers=headers)

            if response.status_code == 201:
                issue = response.json()
                logger.info(f"GitHub issue created: #{issue['number']} - {issue['title']}")
                return {
                    "status": "success",
                    "operation": "create_github_issue",
                    "issue": {
                        "number": issue["number"],
                        "title": issue["title"],
                        "url": issue["html_url"],
                        "state": issue["state"]
                    }
                }
            else:
                error_msg = response.text
                logger.error(f"GitHub API error: {response.status_code} - {error_msg}")
                return {"status": "error", "error": error_msg}

        except Exception as e:
            logger.error(f"GitHub issue creation error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def list_github_issues(self, status: str = "open") -> Dict[str, Any]:
        """List GitHub issues via MCP"""
        try:
            if not self.github_pat:
                return {"error": "GitHub PAT not configured", "status": "error"}

            headers = {
                "Authorization": f"token {self.github_pat}",
                "Accept": "application/vnd.github.v3+json"
            }

            params = {"state": status}
            url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/issues"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                issues = response.json()
                logger.info(f"Retrieved {len(issues)} GitHub issues")
                return {
                    "status": "success",
                    "operation": "list_github_issues",
                    "count": len(issues),
                    "issues": [
                        {
                            "number": issue["number"],
                            "title": issue["title"],
                            "state": issue["state"],
                            "url": issue["html_url"],
                            "labels": [label["name"] for label in issue["labels"]]
                        }
                        for issue in issues
                    ]
                }
            else:
                error_msg = response.text
                logger.error(f"GitHub API error: {response.status_code} - {error_msg}")
                return {"status": "error", "error": error_msg}

        except Exception as e:
            logger.error(f"GitHub issue listing error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def update_github_issue(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a GitHub issue via MCP"""
        try:
            if not self.github_pat:
                return {"error": "GitHub PAT not configured", "status": "error"}

            issue_number = data.get("issue_number")
            if not issue_number:
                return {"error": "Issue number is required", "status": "error"}

            headers = {
                "Authorization": f"token {self.github_pat}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json"
            }

            update_data = {}
            if "title" in data:
                update_data["title"] = data["title"]
            if "body" in data:
                update_data["body"] = data["body"]
            if "state" in data:
                update_data["state"] = data["state"]
            if "labels" in data:
                update_data["labels"] = data["labels"]

            url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/issues/{issue_number}"
            async with httpx.AsyncClient() as client:
                response = await client.patch(url, json=update_data, headers=headers)

            if response.status_code == 200:
                issue = response.json()
                logger.info(f"GitHub issue updated: #{issue['number']} - {issue['title']}")
                return {
                    "status": "success",
                    "operation": "update_github_issue",
                    "issue": {
                        "number": issue["number"],
                        "title": issue["title"],
                        "url": issue["html_url"],
                        "state": issue["state"]
                    }
                }
            else:
                error_msg = response.text
                logger.error(f"GitHub API error: {response.status_code} - {error_msg}")
                return {"status": "error", "error": error_msg}

        except Exception as e:
            logger.error(f"GitHub issue update error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def list_github_projects(self) -> Dict[str, Any]:
        """List GitHub projects"""
        try:
            if not self.github_pat:
                return {"error": "GitHub PAT not configured", "status": "error"}

            headers = {
                "Authorization": f"token {self.github_pat}",
                "Accept": "application/vnd.github.v3+json"
            }

            url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/projects"
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)

            if response.status_code == 200:
                projects = response.json()
                logger.info(f"Retrieved {len(projects)} GitHub projects")
                return {
                    "status": "success",
                    "operation": "list_github_projects",
                    "count": len(projects),
                    "projects": [
                        {
                            "id": project["id"],
                            "name": project["name"],
                            "number": project["number"],
                            "state": project["state"]
                        }
                        for project in projects
                    ]
                }
            else:
                error_msg = response.text
                logger.error(f"GitHub API error: {response.status_code} - {error_msg}")
                return {"status": "error", "error": error_msg}

        except Exception as e:
            logger.error(f"GitHub project listing error: {e}")
            return {"status": "error", "error": str(e)}
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the MCP server"""
        import uvicorn
        logger.info(f"Starting GitHub Project Manager MCP Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    manager = GitHubProjectManager()
    manager.run()
