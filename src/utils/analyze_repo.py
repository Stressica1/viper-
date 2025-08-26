#!/usr/bin/env python3
"""
Repository Structure Analysis for VIPER Trading Bot Reorganization
"""

import os
import json
from pathlib import Path

def analyze_repository():
    """Analyze current repository structure"""
    print("🔍 VIPER Trading Bot - Repository Structure Analysis")
    print("=" * 60)

    base_path = Path(".")

    # Get all items in root directory
    all_items = []
    for item in sorted(os.listdir(".")):
        if not item.startswith('.'):
            item_path = Path(item)
            if item_path.is_dir():
                # Count items in directory
                try:
                    sub_items = len([f for f in item_path.iterdir() if not f.name.startswith('.')])
                    all_items.append(f"{item}/ ({sub_items} items)")
                except:
                    all_items.append(f"{item}/ (access denied)")
            else:
                all_items.append(item)

    print("📁 Current Root Directory Structure:")
    for item in all_items:
        print(f"  {item}")

    print()
    print("📊 Repository Statistics:")

    # Count directories and files
    dirs = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]
    files = [f for f in os.listdir('.') if os.path.isfile(f) and not f.startswith('.')]

    print(f"  📂 Directories: {len(dirs)}")
    print(f"  📄 Files: {len(files)}")

    # Analyze services directory
    if 'services' in dirs:
        services_path = Path('services')
        if services_path.exists():
            service_dirs = [d for d in services_path.iterdir() if d.is_dir()]
            print(f"  🔧 Microservices: {len(service_dirs)}")
            print("    Services:")
            for service in sorted([d.name for d in service_dirs]):
                print(f"      - {service}")

    # Analyze MCP integration status
    print()
    print("🤖 MCP Integration Status:")
    mcp_files = [
        'viper_mcp_client.py',
        'MCP_INTEGRATION_GUIDE.md',
        'MCP_IMPLEMENTATION_README.md',
        'services/mcp-server/main.py',
        'services/mcp-server/Dockerfile',
        'services/mcp-server/requirements.txt',
        '.cursor/mcp.json'
    ]

    for mcp_file in mcp_files:
        status = "✅" if Path(mcp_file).exists() else "❌"
        print(f"  {mcp_file}: {status}")

    print()
    print("🎯 Analysis Complete!")
    print("📋 Ready to proceed with repository reorganization.")

if __name__ == "__main__":
    analyze_repository()
