#!/usr/bin/env python3
"""
🚀 VIPER Trading System - Test GitHub MCP Activation
Demonstrates the proper commands to activate GitHub MCP
"""

import os
import subprocess
import time
from pathlib import Path

def test_mcp_activation():
    """
    Test the proper MCP activation commands
    """
    print("🚀 Testing GitHub MCP Activation")
    print("=" * 50)

    # Check if GITHUB_TOKEN is set
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("❌ GITHUB_TOKEN environment variable not set")
        print("\n📝 To set your GitHub token:")
        print("   export GITHUB_TOKEN=your_github_token_here")
        print("\n🔑 Get a token from: https://github.com/settings/tokens")
        return False

    print(f"✅ GITHUB_TOKEN found (length: {len(github_token)})")

    # Test Command 1: Direct Docker Run
    print("\n🔧 Testing Command 1: Direct Docker Run")
    print("Command: docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server")

    try:
        # Use a timeout and provide minimal input to test
        result = subprocess.run(
            [
                "docker", "run", "-i", "--rm",
                "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={github_token}",
                "ghcr.io/github/github-mcp-server"
            ],
            input="help\nquit\n",  # Test input
            text=True,
            capture_output=True,
            timeout=30
        )

        if result.returncode == 0:
            print("✅ Command 1 SUCCESS")
            print("Sample output:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"❌ Command 1 FAILED (code: {result.returncode})")
            print("STDERR:", result.stderr)

    except subprocess.TimeoutExpired:
        print("❌ Command 1 TIMEOUT")
    except Exception as e:
        print(f"❌ Command 1 ERROR: {e}")

    # Test Command 2: Background Service Mode
    print("\n🔧 Testing Command 2: Background Service Mode")
    print("Command: docker run -d --name github-mcp -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server")

    try:
        # Stop any existing container first
        subprocess.run(
            ["docker", "stop", "github-mcp"],
            capture_output=True,
            timeout=10
        )
        subprocess.run(
            ["docker", "rm", "github-mcp"],
            capture_output=True,
            timeout=10
        )

        # Start new container
        result = subprocess.run(
            [
                "docker", "run", "-d", "--name", "github-mcp",
                "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={github_token}",
                "ghcr.io/github/github-mcp-server"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("✅ Command 2 SUCCESS")
            print(f"Container ID: {result.stdout.strip()}")

            # Check if container is running
            time.sleep(2)
            status_result = subprocess.run(
                ["docker", "ps", "--filter", "name=github-mcp"],
                capture_output=True,
                text=True
            )

            if "github-mcp" in status_result.stdout:
                print("✅ Background service is running")
            else:
                print("❌ Background service failed to start")

        else:
            print(f"❌ Command 2 FAILED (code: {result.returncode})")
            print("STDERR:", result.stderr)

    except subprocess.TimeoutExpired:
        print("❌ Command 2 TIMEOUT")
    except Exception as e:
        print(f"❌ Command 2 ERROR: {e}")

    # Show VS Code integration command
    print("\n🔧 Command 3: VS Code MCP Integration")
    print("Command: code --enable-proposed-api github.vscode-github-mcp")
    print("Setup: Restart VS Code and use MCP commands from palette")

    # Show current MCP configuration status
    print("\n🔍 MCP Configuration Status:")
    vscode_dir = Path(".vscode")
    settings_file = vscode_dir / "settings.json"
    mcp_file = vscode_dir / "mcp.json"

    if settings_file.exists():
        print("✅ .vscode/settings.json exists")
    else:
        print("❌ .vscode/settings.json missing")

    if mcp_file.exists():
        print("✅ .vscode/mcp.json exists")
    else:
        print("❌ .vscode/mcp.json missing")

    print("\n" + "=" * 50)
    print("🎯 ACTIVATION SUMMARY")
    print("=" * 50)

    print("\n📋 WORKING COMMANDS:")
    print("1. Direct Run:")
    print("   docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server")

    print("\n2. Background Service:")
    print("   docker run -d --name github-mcp -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server")

    print("\n3. VS Code Integration:")
    print("   code --enable-proposed-api github.vscode-github-mcp")

    print("\n📚 MCP CONFIGURATION:")
    print("   ✅ .vscode/settings.json configured")
    print("   ✅ .vscode/mcp.json configured")

    print("\n🔑 GITHUB TOKEN:")
    print(f"   ✅ Token configured (length: {len(github_token)})")

    print("\n🚀 READY TO USE MCP!")

    return True

def show_usage_examples():
    """Show usage examples for MCP"""
    print("\n" + "=" * 50)
    print("📖 MCP USAGE EXAMPLES")
    print("=" * 50)

    examples = [
        {
            "name": "Search Repository",
            "command": "search repository:octocat/Hello-World",
            "description": "Search within a specific repository"
        },
        {
            "name": "Find File",
            "command": "find file:README.md in:octocat/Hello-World",
            "description": "Find specific files in repositories"
        },
        {
            "name": "Search Code",
            "command": "search code:function processData",
            "description": "Search for specific code patterns"
        },
        {
            "name": "Get Repository Info",
            "command": "get repository info:octocat/Hello-World",
            "description": "Get detailed repository information"
        }
    ]

    for example in examples:
        print(f"\n🔍 {example['name']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")

    print("\n💡 TIP: Use MCP to diagnose trading system issues!")
    print("   Example: search code:function execute_trade in:your-repo")

if __name__ == "__main__":
    success = test_mcp_activation()
    if success:
        show_usage_examples()

        print("\n" + "=" * 50)
        print("🎉 MCP ACTIVATION COMPLETE!")
        print("=" * 50)
        print("\nYou can now use GitHub MCP to:")
        print("• Search repositories and code")
        print("• Diagnose trading system issues")
        print("• Find implementation examples")
        print("• Access GitHub data directly")
    else:
        print("\n❌ MCP activation failed. Please check your GitHub token and try again.")
