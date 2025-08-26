#!/usr/bin/env python3
"""
🚀 VIPER Trading System - MCP Activation Diagnostic
Diagnoses and tests GitHub MCP server activation
"""

import os
import json
import subprocess
import time
from pathlib import Path

class MCPDiagnostic:
    """
    Comprehensive diagnostic tool for GitHub MCP activation
    """

    def __init__(self):
        self.workspace_root = Path.cwd()
        self.vscode_dir = self.workspace_root / ".vscode"
        self.config_files = [
            self.vscode_dir / "settings.json",
            self.vscode_dir / "mcp.json"
        ]

    def check_docker_status(self) -> dict:
        """Check Docker installation and status"""
        print("🔍 Checking Docker status...")

        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                print(f"✅ Docker is installed: {result.stdout.strip()}")

                # Check if Docker is running
                result = subprocess.run(
                    ["docker", "ps"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    return {
                        "installed": True,
                        "running": True,
                        "version": result.stdout.strip()
                    }
                else:
                    print("❌ Docker daemon is not running")
                    return {
                        "installed": True,
                        "running": False,
                        "error": "Docker daemon not running"
                    }
            else:
                print("❌ Docker is not installed or not in PATH")
                return {
                    "installed": False,
                    "running": False,
                    "error": "Docker not found"
                }

        except FileNotFoundError:
            print("❌ Docker command not found")
            return {
                "installed": False,
                "running": False,
                "error": "Docker not found"
            }
        except subprocess.TimeoutExpired:
            print("❌ Docker command timed out")
            return {
                "installed": False,
                "running": False,
                "error": "Docker timeout"
            }

    def check_config_files(self) -> dict:
        """Check MCP configuration files"""
        print("\n🔍 Checking MCP configuration files...")

        results = {}

        for config_file in self.config_files:
            if config_file.exists():
                print(f"✅ Found config file: {config_file}")

                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)

                    # Validate MCP configuration
                    if "mcp" in config:
                        mcp_config = config["mcp"]
                        results[str(config_file)] = {
                            "exists": True,
                            "valid": True,
                            "has_servers": "servers" in mcp_config,
                            "has_github_server": "github" in mcp_config.get("servers", {}),
                            "config": mcp_config
                        }
                        print(f"   ✅ Valid MCP configuration found")
                    else:
                        results[str(config_file)] = {
                            "exists": True,
                            "valid": False,
                            "error": "No MCP configuration found"
                        }
                        print(f"   ❌ No MCP configuration in file")

                except json.JSONDecodeError as e:
                    results[str(config_file)] = {
                        "exists": True,
                        "valid": False,
                        "error": f"Invalid JSON: {e}"
                    }
                    print(f"   ❌ Invalid JSON in config file: {e}")
            else:
                results[str(config_file)] = {
                    "exists": False,
                    "valid": False
                }
                print(f"❌ Config file not found: {config_file}")

        return results

    def test_mcp_server_connection(self) -> dict:
        """Test MCP server connection with a sample token"""
        print("\n🔍 Testing MCP server connection...")

        # Test command from configuration
        test_command = [
            "docker", "run", "-i", "--rm",
            "-e", "GITHUB_PERSONAL_ACCESS_TOKEN=test_token",
            "ghcr.io/github/github-mcp-server"
        ]

        print(f"🧪 Testing command: {' '.join(test_command)}")

        try:
            result = subprocess.run(
                test_command,
                input="test",  # Provide minimal input
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("✅ MCP server started successfully")
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print(f"❌ MCP server failed with code {result.returncode}")
                print(f"   STDERR: {result.stderr}")
                return {
                    "success": False,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }

        except subprocess.TimeoutExpired:
            print("❌ MCP server test timed out")
            return {
                "success": False,
                "error": "Timeout"
            }
        except Exception as e:
            print(f"❌ MCP server test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def generate_activation_commands(self) -> list:
        """Generate proper MCP activation commands"""
        print("\n🔧 Generating MCP activation commands...")

        commands = []

        # Command 1: Direct Docker run with token prompt
        commands.append({
            "name": "Direct Docker Run",
            "description": "Run MCP server directly with Docker",
            "command": "docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server",
            "setup": "Set GITHUB_TOKEN environment variable first"
        })

        # Command 2: Docker run with interactive token input
        commands.append({
            "name": "Interactive Token Input",
            "description": "Run MCP server with token input",
            "command": "echo $GITHUB_TOKEN | docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=/dev/stdin ghcr.io/github/github-mcp-server",
            "setup": "Set GITHUB_TOKEN environment variable first"
        })

        # Command 3: Using the MCP config files
        commands.append({
            "name": "VS Code MCP Integration",
            "description": "Use VS Code MCP configuration files",
            "command": "code --enable-proposed-api github.vscode-github-mcp",
            "setup": "Restart VS Code and use MCP commands from palette"
        })

        # Command 4: Background service mode
        commands.append({
            "name": "Background Service Mode",
            "description": "Run MCP server in background",
            "command": "docker run -d --name github-mcp -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server",
            "setup": "Set GITHUB_TOKEN environment variable first"
        })

        return commands

    def create_github_token_guide(self) -> str:
        """Create a guide for obtaining GitHub token"""
        guide = """
# GitHub Personal Access Token Setup

## Steps to Create a GitHub PAT:

1. **Go to GitHub Settings**
   - Visit: https://github.com/settings/tokens
   - Click "Generate new token (classic)"

2. **Configure Token Permissions**
   - Note: `classic` tokens are required for MCP
   - Select scopes:
     - [x] `read:packages` - Access GitHub packages
     - [x] `repo` - Full control of private repositories
     - [x] `user` - Read user profile data

3. **Generate and Save Token**
   - Click "Generate token"
   - **IMPORTANT**: Copy the token immediately
   - Token will look like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

4. **Set Environment Variable**
   ```bash
   export GITHUB_TOKEN=ghp_your_token_here
   ```

5. **Test Token**
   ```bash
   curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user
   ```

## Security Notes:
- Never commit tokens to version control
- Use environment variables, not hardcoded values
- Rotate tokens regularly
- Use the minimum required permissions

**Current Token Status**: Check if GITHUB_TOKEN is set
"""

        return guide

    def run_full_diagnostic(self) -> dict:
        """Run comprehensive MCP diagnostic"""
        print("🚀 Starting MCP Activation Diagnostic")
        print("=" * 50)

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "docker_status": self.check_docker_status(),
            "config_files": self.check_config_files(),
            "server_test": self.test_mcp_server_connection(),
            "activation_commands": self.generate_activation_commands(),
            "token_guide": self.create_github_token_guide()
        }

        print("\n" + "=" * 50)
        print("📋 DIAGNOSTIC SUMMARY")
        print("=" * 50)

        # Docker status
        docker = results["docker_status"]
        if docker["installed"] and docker["running"]:
            print("✅ Docker: Ready")
        else:
            print("❌ Docker: Issues detected")
            if "error" in docker:
                print(f"   Error: {docker['error']}")

        # Config files
        config_issues = 0
        for file_path, status in results["config_files"].items():
            if status["exists"] and status["valid"]:
                print(f"✅ Config: {file_path}")
            else:
                print(f"❌ Config: {file_path} - Issues detected")
                config_issues += 1

        # Server test
        server_test = results["server_test"]
        if server_test["success"]:
            print("✅ Server Test: Passed")
        else:
            print("❌ Server Test: Failed")
            if "error" in server_test:
                print(f"   Error: {server_test['error']}")

        print(f"\n📚 Found {len(results['activation_commands'])} activation methods")

        return results

def main():
    """Main diagnostic function"""
    diagnostic = MCPDiagnostic()
    results = diagnostic.run_full_diagnostic()

    # Save detailed results
    with open("mcp_diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save activation guide
    with open("MCP_ACTIVATION_GUIDE.md", "w", encoding='utf-8') as f:
        f.write("# GitHub MCP Activation Guide\n\n")
        f.write(results["token_guide"])
        f.write("\n## Activation Commands\n\n")

        for cmd in results["activation_commands"]:
            f.write(f"### {cmd['name']}\n")
            f.write(f"**Description**: {cmd['description']}\n\n")
            f.write("**Setup**:\n")
            f.write(f"```\n{cmd['setup']}\n```\n\n")
            f.write("**Command**:\n")
            f.write(f"```\n{cmd['command']}\n```\n\n")

    print("\n📄 Results saved:")
    print("  📊 JSON data: mcp_diagnostic_results.json")
    print("  📋 Guide: MCP_ACTIVATION_GUIDE.md")
    # Print key findings
    print("\n🎯 KEY FINDINGS:")
    docker_status = results["docker_status"]
    if not docker_status["installed"]:
        print("❌ ACTION REQUIRED: Install Docker")
    elif not docker_status["running"]:
        print("❌ ACTION REQUIRED: Start Docker daemon")

    config_files = results["config_files"]
    valid_configs = sum(1 for status in config_files.values() if status["exists"] and status["valid"])
    if valid_configs == 0:
        print("❌ ACTION REQUIRED: Create MCP configuration files")

    server_test = results["server_test"]
    if not server_test["success"]:
        print("❌ ACTION REQUIRED: Obtain valid GitHub token")

    print("\n✅ Next Steps:")
    print("1. Follow MCP_ACTIVATION_GUIDE.md")
    print("2. Set GITHUB_TOKEN environment variable")
    print("3. Test with: docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server")

if __name__ == "__main__":
    main()
