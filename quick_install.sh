#!/bin/bash
# VIPER Trading Bot - One-Line Installer
# Usage: curl -sSL https://raw.githubusercontent.com/Stressica1/viper-/main/quick_install.sh | bash

set -e

echo "🚀 VIPER Trading Bot - Quick Install"
echo "===================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    echo "Please install Python 3.11+ from https://python.org"
    exit 1
fi

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is required but not found"
    echo "Please install Git from https://git-scm.com"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not found"
    echo "Please install Docker Desktop from https://docker.com"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Clone repository
if [ ! -d "viper-" ]; then
    echo "📦 Cloning VIPER repository..."
    git clone https://github.com/Stressica1/viper-.git
fi

cd viper-

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install --user -r requirements.txt

# Setup environment
echo "⚙️ Setting up environment..."
cp .env.template .env

echo ""
echo "🎉 VIPER Installation Complete!"
echo ""
echo "Next steps:"
echo "1. Configure API keys: python scripts/configure_api.py"
echo "2. Start system: python scripts/start_microservices.py start"
echo "3. Open dashboard: http://localhost:8000"
echo ""
echo "For detailed guide, see: INSTALLATION.md"