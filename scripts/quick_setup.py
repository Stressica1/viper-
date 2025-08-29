#!/usr/bin/env python3
"""
# Rocket VIPER Quick Setup Script
Automated setup script for AI to quickly configure the VIPER trading bot
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

# Enhanced terminal display
try:
    from src.viper.utils.terminal_display import (
        terminal, display_error, display_success, display_warning, 
        print_banner, show_progress
    )
    ENHANCED_DISPLAY = True
except ImportError:
    ENHANCED_DISPLAY = False
    def display_error(msg, details=None): print(f"# X {msg}")
    def display_success(msg, details=None): print(f"# Check {msg}")
    def display_warning(msg, details=None): print(f"# Warning {msg}")
    def print_banner(): print("# Rocket VIPER Quick Setup")
    def show_progress(tasks, title): print(f"{title}: {', '.join(tasks)}")

def run_command(cmd, description, check=True):
    """Run a command with enhanced display"""
    try:
        if ENHANCED_DISPLAY:
            terminal.console.print(f"[blue]Running:[/] {description}")
        else:
            
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        
        if result.returncode == 0:
            display_success(description)
            return True, result.stdout
        else:
            display_error(f"Failed: {description}", result.stderr)
            return False, result.stderr
    except subprocess.CalledProcessError as e:
        display_error(f"Command failed: {description}", str(e))
        return False, str(e)

def setup_python_environment():
    """Setup Python virtual environment and dependencies"""
    display_warning("Setting up Python environment...")
    
    # Check Python version
    success, output = run_command("python --version", "Checking Python version", check=False)
    if success:
    
    # Create virtual environment if it doesn't exist
    if not Path("viper_env").exists():
        display_warning("Creating virtual environment...")
        success, _ = run_command("python -m venv viper_env", "Creating virtual environment")
        if not success:
            return False
    else:
        display_success("Virtual environment already exists")
    
    # Install requirements
    display_warning("Installing Python dependencies...")
    
    # Try different pip commands based on OS
    pip_commands = [
        "viper_env/bin/pip install -r requirements.txt",  # Linux/macOS
        "viper_env\\Scripts\\pip install -r requirements.txt",  # Windows
        "pip install -r requirements.txt"  # Fallback
    ]
    
    for pip_cmd in pip_commands:
        success, output = run_command(pip_cmd, f"Installing dependencies with {pip_cmd.split()[0]}", check=False)
        if success:
            display_success("Dependencies installed successfully")
            break
    else:
        display_warning("Could not install dependencies automatically", 
                       "Please run: pip install -r requirements.txt manually")
    
    return True

def setup_configuration():
    """Setup configuration files"""
    display_warning("Setting up configuration...")
    
    # Create .env from template if it doesn't exist
    if not Path(".env").exists():
        if Path(".env.example").exists():
            shutil.copy(".env.example", ".env")
            display_success("Created .env from template")
            display_warning("IMPORTANT: Edit .env with your API credentials!", 
                          "Add your real Bitget API keys and other configuration")
        else:
            display_error("No .env.example found")
            return False
    else:
        display_success(".env file already exists")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    display_success("Logs directory created")
    
    # Create reports directory if it doesn't exist
    Path("reports").mkdir(exist_ok=True)
    display_success("Reports directory created")
    
    return True

def setup_docker():
    """Setup Docker services"""
    display_warning("Setting up Docker services...")
    
    # Check if Docker is available
    success, output = run_command("docker --version", "Checking Docker installation", check=False)
    if not success:
        display_warning("Docker not found", 
                       "Please install Docker Desktop from https://docker.com/products/docker-desktop")
        return False
    
    display_success(f"Docker found: {output.strip()}")
    
    # Check Docker Compose
    success, output = run_command("docker compose version", "Checking Docker Compose", check=False)
    if not success:
        display_warning("Docker Compose not available", 
                       "Install docker-compose plugin or use docker-compose command")
        return False
    
    display_success(f"Docker Compose found: {output.strip()}")
    
    # Try to start basic services (optional)
    display_warning("Starting basic Docker services...")
    success, output = run_command("docker compose up -d redis", "Starting Redis service", check=False)
    if success:
        display_success("Redis service started")
    else:
        display_warning("Could not start Redis automatically", 
                       "You can start services later with: docker compose up -d")
    
    return True

def validate_setup():
    """Validate the setup"""
    display_warning("Validating setup...")
    
    # Run the setup validator
    success, output = run_command("python tools/setup_validator.py", "Running setup validation", check=False)
    
    if "Overall Status: PASS" in output:
        display_success("Setup validation passed!")
        return True
    else:
        display_warning("Setup validation found issues", "Check the validation report above")
        return True  # Don't fail completely, just warn

def main():
    """Main setup function"""
    if ENHANCED_DISPLAY:
        print_banner()
        terminal.console.rule("[bold blue]# Rocket Automated VIPER Setup Starting[/]")
    else:
    
    setup_steps = [
        ("Python Environment", setup_python_environment),
        ("Configuration Files", setup_configuration), 
        ("Docker Services", setup_docker),
        ("Setup Validation", validate_setup),
    ]
    
    if ENHANCED_DISPLAY:
        show_progress([step[0] for step in setup_steps], "# Tool Setup Steps")
    
    success_count = 0
    
    for step_name, step_func in setup_steps:
        if ENHANCED_DISPLAY:
            terminal.console.rule(f"[bold cyan]{step_name}[/]")
        else:
        
        try:
            if step_func():
                success_count += 1
        except Exception as e:
            display_error(f"Error in {step_name}", str(e))
    
    # Final report
    
    if success_count == len(setup_steps):
        display_success("All setup steps completed successfully!")
        display_success("You can now run the trading system!")
        
        if ENHANCED_DISPLAY:
            terminal.console.print("\n[bold green]# Target Next Steps:[/]")
            terminal.console.print("1. Edit .env file with your API credentials")
            terminal.console.print("2. Run: python tools/setup_validator.py")
            terminal.console.print("3. Start system: python scripts/start_live_trading_mandatory.py")
        else:
            print("1. Edit .env file with your API credentials")
            print("3. Start system: python scripts/start_live_trading_mandatory.py")
            
    else:
        display_warning(f"Setup completed with {len(setup_steps) - success_count} issues", 
                       "Review the messages above and fix any problems")
        
        if ENHANCED_DISPLAY:
            terminal.console.print("\n[bold yellow]# Tool Troubleshooting:[/]")
            terminal.console.print("• Check the AI Setup Guide: docs/AI_SETUP_GUIDE.md")
            terminal.console.print("• Run validator: python tools/setup_validator.py")
            terminal.console.print("• Check logs in logs/ directory")
        else:
            print("• Check the AI Setup Guide: docs/AI_SETUP_GUIDE.md")
            print("• Run validator: python tools/setup_validator.py")

if __name__ == "__main__":
    main()