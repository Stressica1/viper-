#!/usr/bin/env python3
"""
🔗 MANDATORY DOCKER & MCP SYSTEM WRAPPER
Universal wrapper that enforces Docker/MCP requirements for ALL VIPER system operations

This wrapper:
    pass
# Check ENFORCES Docker and MCP requirements before ANY system operation
# Check Integrates ALL modules through unified Docker/MCP framework
# Check Prevents execution if mandatory requirements not met
# Check Connects all trading operations through MCP GitHub integration
# Check Provides unified entry point for all system features

EVERY MODULE MUST GO THROUGH THIS WRAPPER!
"""

import os
import sys
import time
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import importlib
import traceback
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import mandatory enforcement system
from docker_mcp_enforcer import enforce_docker_mcp_requirements, get_enforcer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MANDATORY_WRAPPER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MandatoryDockerMCPWrapper:
    """
    Universal wrapper that enforces Docker/MCP requirements for all operations
    NO MODULE CAN EXECUTE WITHOUT PASSING THROUGH THIS WRAPPER!
    """"""
    
    def __init__(self):
        self.enforcer = get_enforcer()
        self.github_mcp = None
        self.integrated_modules = {}
        self.operation_history = []
        
        # Module registry - ALL modules must be registered here
        self.available_modules = {
            # Main Trading Modules
            'main': {
                'module': 'main',
                'class': 'SimpleVIPERTrader',
                'description': 'Main VIPER Trading Bot',
                'requires_mcp': True,
                'requires_github': True
            },
            'master_live_trading_job': {
                'module': 'master_live_trading_job',
                'class': 'MasterLiveTradingJob', 
                'description': 'Master Live Trading System',
                'requires_mcp': True,
                'requires_github': True
            },
            'mcp_live_trading_connector': {
                'module': 'mcp_live_trading_connector',
                'class': 'MCPLiveTradingConnector',
                'description': 'MCP Live Trading Connector',
                'requires_mcp': True,
                'requires_github': True
            },
            'enhanced_system_integrator': {
                'module': 'enhanced_system_integrator', 
                'class': 'EnhancedSystemIntegrator',
                'description': 'System Integration Framework',
                'requires_mcp': True,
                'requires_github': True
            },
            'viper_async_trader': {
                'module': 'viper_async_trader',
                'class': 'ViperAsyncTrader',
                'description': 'Async VIPER Trading System',
                'requires_mcp': True,
                'requires_github': True
            },
            
            # MCP and Brain Modules
            'mcp_brain_controller': {
                'module': 'mcp_brain_controller',
                'class': 'MCPBrainController',
                'description': 'MCP Brain Controller',
                'requires_mcp': True,
                'requires_github': True
            },
            'github_mcp_integration': {
                'module': 'github_mcp_integration',
                'class': 'GitHubMCPIntegration',
                'description': 'GitHub MCP Integration',
                'requires_mcp': True,
                'requires_github': True
            },
            
            # Analysis and Optimization Modules
            'ai_ml_optimizer': {
                'module': 'ai_ml_optimizer',
                'class': 'AIMLOptimizer',
                'description': 'AI/ML Trade Optimizer',
                'requires_mcp': True,
                'requires_github': True
            },
            'comprehensive_backtester': {
                'module': 'comprehensive_backtester',
                'class': 'ComprehensiveBacktester', 
                'description': 'Comprehensive Backtesting System',
                'requires_mcp': True,
                'requires_github': True
            },
            
            # Monitoring and Validation Modules
            'performance_monitoring_system': {
                'module': 'performance_monitoring_system',
                'class': 'PerformanceMonitoringSystem',
                'description': 'Performance Monitoring System',
                'requires_mcp': True,
                'requires_github': True
            },
            'comprehensive_verification_system': {
                'module': 'comprehensive_verification_system',
                'class': 'ComprehensiveVerificationSystem',
                'description': 'System Verification Framework',
                'requires_mcp': True,
                'requires_github': True
            }
        }
    
    def execute_with_enforcement(self, module_name: str, operation: str = 'main', *args, **kwargs) -> Any:
        """Execute any module operation with mandatory Docker/MCP enforcement"""
        logger.info(f"🔒 ENFORCING REQUIREMENTS FOR: {module_name}.{operation}")
        
        # STEP 1: MANDATORY ENFORCEMENT
        if not enforce_docker_mcp_requirements():
            logger.error("💀 ENFORCEMENT FAILED - OPERATION BLOCKED!")
            sys.exit(1)
        
        # STEP 2: Initialize GitHub MCP Integration
        if not self._initialize_github_mcp():
            logger.error("💀 GITHUB MCP INITIALIZATION FAILED - OPERATION BLOCKED!")
            sys.exit(1)
        
        # STEP 3: Validate module registration
        if module_name not in self.available_modules:
            logger.error(f"💀 MODULE {module_name} NOT REGISTERED - OPERATION BLOCKED!")
            sys.exit(1)
        
        module_info = self.available_modules[module_name]
        
        # STEP 4: Load and execute module with MCP integration
        try:
            logger.info(f"# Rocket EXECUTING {module_info['description']}")
            
            # Import and initialize module
            module = importlib.import_module(module_info['module'])
            
            if 'class' in module_info:
                # Initialize class-based module
                module_class = getattr(module, module_info['class'])
                instance = module_class()
                
                # Inject MCP integration if required
                if module_info.get('requires_mcp', False):
                    self._inject_mcp_integration(instance, module_name)
                
                # Execute operation
                if hasattr(instance, operation):
                    result = getattr(instance, operation)(*args, **kwargs)
                elif operation == 'main' and hasattr(instance, 'run'):
                    result = instance.run(*args, **kwargs)
                else:
                    logger.error(f"💀 OPERATION {operation} NOT FOUND IN {module_name}")
                    sys.exit(1)
            else:
                # Execute function-based module
                if hasattr(module, operation):
                    result = getattr(module, operation)(*args, **kwargs)
                else:
                    result = module.main(*args, **kwargs) if hasattr(module, 'main') else None
            
            # STEP 5: Log operation to GitHub MCP
            self._log_operation_to_github(module_name, operation, True)
            
            logger.info(f"# Check {module_name}.{operation} COMPLETED SUCCESSFULLY")
            return result
            
        except Exception as e:
            logger.error(f"💀 {module_name}.{operation} FAILED: {e}")
            self._log_operation_to_github(module_name, operation, False, str(e))
            traceback.print_exc()
            sys.exit(1)
    
    def _initialize_github_mcp(self) -> bool:
        """Initialize GitHub MCP integration (mandatory)""""""
        if self.github_mcp is not None:
            return True
            
        try:
import github_mcp_integration

            self.github_mcp = github_mcp_integration.GitHubMCPIntegration()
            logger.info("# Check GitHub MCP Integration: INITIALIZED")
            return True
        except Exception as e:
            logger.error(f"# X GitHub MCP Integration failed: {e}")
            return False
    
    def _inject_mcp_integration(self, instance: Any, module_name: str):
        """Inject MCP integration into module instance""""""
        try:
            # Inject GitHub MCP integration
            if hasattr(instance, '__dict__'):
                instance.github_mcp = self.github_mcp
                instance.mcp_enforcer = self.enforcer
                logger.info(f"# Check MCP integration injected into {module_name}")
            
            # Add MCP logging wrapper to key methods
            self._wrap_methods_with_mcp_logging(instance, module_name)
            
        except Exception as e:
            logger.warning(f"# Warning MCP injection warning for {module_name}: {e}")
    
    def _wrap_methods_with_mcp_logging(self, instance: Any, module_name: str):
        """Wrap instance methods with MCP logging""""""
        try:
            # Common methods to wrap
            methods_to_wrap = ['trade', 'run', 'execute', 'start', 'process', 'analyze', 'optimize']
            
            for method_name in methods_to_wrap:
                if hasattr(instance, method_name):
                    original_method = getattr(instance, method_name)
                    wrapped_method = self._create_mcp_logged_method(original_method, module_name, method_name)
                    setattr(instance, method_name, wrapped_method)
                    logger.debug(f"# Check Wrapped {module_name}.{method_name} with MCP logging")
                    
        except Exception as e:
            logger.warning(f"# Warning Method wrapping warning for {module_name}: {e}")
    
    def _create_mcp_logged_method(self, original_method: Callable, module_name: str, method_name: str):
        """Create MCP-logged version of a method""""""
        def wrapped_method(*args, **kwargs):
            start_time = datetime.now()
            try:
                # Log start to GitHub MCP
                if self.github_mcp:
                    asyncio.create_task(self.github_mcp.log_system_performance(}))
                        'module': module_name,
                        'method': method_name,
                        'status': 'started',
                        'timestamp': start_time.isoformat()
((                    }))
                
                # Execute original method
                result = original_method(*args, **kwargs)
                
                # Log success to GitHub MCP
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                if self.github_mcp:
                    asyncio.create_task(self.github_mcp.log_system_performance(}))
                        'module': module_name,
                        'method': method_name,
                        'status': 'completed',
                        'duration_seconds': duration,
                        'timestamp': end_time.isoformat()
((                    }))
                
                return result
                
            except Exception as e:
                # Log error to GitHub MCP
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                if self.github_mcp:
                    asyncio.create_task(self.github_mcp.log_system_performance(}))
                        'module': module_name,
                        'method': method_name,
                        'status': 'failed',
                        'error': str(e),
                        'duration_seconds': duration,
                        'timestamp': end_time.isoformat()
((                    }))
                
                raise
        
        return wrapped_method
    
    def _log_operation_to_github(self, module_name: str, operation: str, )
(                               success: bool, error: str = None):
        """Log operation result to GitHub MCP""""""
        try:
            if self.github_mcp:
                operation_data = {
                    'module': module_name,
                    'operation': operation,
                    'success': success,
                    'timestamp': datetime.now().isoformat(),
                    'enforcer_status': self.enforcer.get_system_status()
                }
                
                if error:
                    operation_data['error'] = error
                
                # Store in operation history
                self.operation_history.append(operation_data)
                
                # Commit to GitHub (async)
                commit_message = f"{'# Check' if success else '# X'} {module_name}.{operation} - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
                asyncio.create_task(self.github_mcp.commit_system_changes(commit_message))
                
        except Exception as e:
            logger.warning(f"# Warning GitHub logging warning: {e}")
    
    def start_docker_services(self) -> bool:
        """Start Docker services if not running"""
        logger.info("🐳 Starting Docker services...")
        
        try:
            # Start docker-compose services
            result = subprocess.run(['docker', 'compose', 'up', '-d'], )
(                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("# Check Docker services started successfully")
                time.sleep(30)  # Wait for services to initialize
                return True
            else:
                logger.error(f"# X Docker services start failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"# X Error starting Docker services: {e}")
            return False
    
    def get_available_modules(self) -> Dict[str, Any]
        """Get list of available modules"""
        return self.available_modules"""
    
    def get_operation_history(self) -> List[Dict[str, Any]]
        """Get operation execution history"""
        return self.operation_history"""
    
    def get_system_status(self) -> Dict[str, Any]
        """Get complete system status"""
        return {:
            'enforcer_status': self.enforcer.get_system_status(),
            'github_mcp_initialized': self.github_mcp is not None,
            'available_modules': len(self.available_modules),
            'operations_executed': len(self.operation_history),
            'last_operation': self.operation_history[-1] if self.operation_history else None
        }

# Global wrapper instance
_wrapper_instance = None"""

def get_wrapper() -> MandatoryDockerMCPWrapper:
    """Get global wrapper instance"""
    global _wrapper_instance"""
    if _wrapper_instance is None:
        _wrapper_instance = MandatoryDockerMCPWrapper()
    return _wrapper_instance

def execute_module(module_name: str, operation: str = 'main', *args, **kwargs) -> Any:
    """
    MANDATORY ENTRY POINT - Execute any module with Docker/MCP enforcement
    ALL MODULE EXECUTIONS MUST GO THROUGH THIS FUNCTION!
    """
    wrapper = get_wrapper()
    return wrapper.execute_with_enforcement(module_name, operation, *args, **kwargs)


def start_system_with_enforcement() -> bool:
    """Start complete system with mandatory enforcement"""
    wrapper = get_wrapper()

    # Start Docker services
    if not wrapper.start_docker_services():
        logger.error("💀 CANNOT START DOCKER SERVICES - SYSTEM BLOCKED!")
        return False

    # Enforce all requirements
    if not enforce_docker_mcp_requirements():
        logger.error("💀 ENFORCEMENT FAILED - SYSTEM BLOCKED!")
        return False

    logger.info("🎉 SYSTEM STARTED WITH MANDATORY ENFORCEMENT ACTIVE!")
    return True

if __name__ == "__main__":
    pass
    
    if len(sys.argv) < 2:
        print("Usage: python mandatory_docker_mcp_wrapper.py <module_name> [operation] [args...]")
        wrapper = get_wrapper()
        for name, info in wrapper.get_available_modules().items():
            print(f"  {name}: {info['description']}")
        sys.exit(1)
    
    module_name = sys.argv[1]
    operation = sys.argv[2] if len(sys.argv) > 2 else 'main'
    args = sys.argv[3:] if len(sys.argv) > 3 else []
    
    # Execute with enforcement
    try:
        result = execute_module(module_name, operation, *args)
        print(f"# Check {module_name}.{operation} completed successfully!")
        if result:
            pass
    except SystemExit:
        print(f"💀 {module_name}.{operation} execution blocked by enforcement!")
        sys.exit(1)