#!/usr/bin/env python3
"""
🚀 OPTIMAL MCP SERVER CONFIGURATION
Advanced configuration settings for maximum performance and reliability
"""

import os
from typing import Dict, Any

def get_optimal_mcp_config() -> Dict[str, Any]:
    """Get optimal MCP server configuration with environment variable support"""
    
    return {
        # Core Server Settings
        "server": {
            "host": os.getenv('MCP_HOST', '0.0.0.0'),
            "port": int(os.getenv('MCP_PORT', '8015')),
            "workers": int(os.getenv('MCP_WORKERS', '4')),
            "max_connections": int(os.getenv('MCP_MAX_CONNECTIONS', '100')),
            "timeout": int(os.getenv('MCP_TIMEOUT', '30'))
        },
        
        # Performance Optimization
        "performance": {
            "connection_pool_size": int(os.getenv('MCP_POOL_SIZE', '20')),
            "max_concurrent_requests": int(os.getenv('MCP_MAX_CONCURRENT', '10')),
            "request_timeout": int(os.getenv('MCP_REQUEST_TIMEOUT', '15')),
            "keepalive_timeout": int(os.getenv('MCP_KEEPALIVE_TIMEOUT', '5')),
            "max_request_size": int(os.getenv('MCP_MAX_REQUEST_SIZE', '16777216'))  # 16MB
        },
        
        # Retry and Resilience
        "resilience": {
            "retry_attempts": int(os.getenv('MCP_RETRY_ATTEMPTS', '3')),
            "retry_delay": float(os.getenv('MCP_RETRY_DELAY', '1.5')),
            "backoff_multiplier": float(os.getenv('MCP_BACKOFF_MULTIPLIER', '2.0')),
            "circuit_breaker_threshold": int(os.getenv('MCP_CIRCUIT_BREAKER_THRESHOLD', '5')),
            "circuit_breaker_timeout": int(os.getenv('MCP_CIRCUIT_BREAKER_TIMEOUT', '60'))
        },
        
        # Health Check Configuration
        "health_check": {
            "enabled": os.getenv('MCP_HEALTH_CHECK_ENABLED', 'true').lower() == 'true',
            "interval": int(os.getenv('MCP_HEALTH_CHECK_INTERVAL', '30')),
            "timeout": int(os.getenv('MCP_HEALTH_CHECK_TIMEOUT', '5')),
            "failure_threshold": int(os.getenv('MCP_HEALTH_FAILURE_THRESHOLD', '3')),
            "recovery_threshold": int(os.getenv('MCP_HEALTH_RECOVERY_THRESHOLD', '2'))
        },
        
        # Security Settings
        "security": {
            "enable_cors": os.getenv('MCP_ENABLE_CORS', 'true').lower() == 'true',
            "cors_origins": os.getenv('MCP_CORS_ORIGINS', '*').split(','),
            "api_key_required": os.getenv('MCP_API_KEY_REQUIRED', 'false').lower() == 'true',
            "rate_limit_enabled": os.getenv('MCP_RATE_LIMIT_ENABLED', 'true').lower() == 'true',
            "max_requests_per_minute": int(os.getenv('MCP_MAX_REQUESTS_PER_MINUTE', '300'))
        },
        
        # Logging Configuration
        "logging": {
            "level": os.getenv('MCP_LOG_LEVEL', 'INFO'),
            "format": os.getenv('MCP_LOG_FORMAT', 'json'),
            "file_logging": os.getenv('MCP_FILE_LOGGING', 'true').lower() == 'true',
            "log_rotation": os.getenv('MCP_LOG_ROTATION', 'true').lower() == 'true',
            "max_log_size_mb": int(os.getenv('MCP_MAX_LOG_SIZE_MB', '100')),
            "backup_count": int(os.getenv('MCP_LOG_BACKUP_COUNT', '5'))
        },
        
        # Service Discovery
        "services": {
            'api-server': {
                'url': os.getenv('API_SERVER_URL', 'http://api-server:8000'),
                'health_endpoint': '/health',
                'timeout': 10,
                'retry_attempts': 3
            },
            'data-manager': {
                'url': os.getenv('DATA_MANAGER_URL', 'http://data-manager:8001'),
                'health_endpoint': '/health',
                'timeout': 15,
                'retry_attempts': 3
            },
            'exchange-connector': {
                'url': os.getenv('EXCHANGE_CONNECTOR_URL', 'http://exchange-connector:8005'),
                'health_endpoint': '/health',
                'timeout': 20,
                'retry_attempts': 5
            },
            'risk-manager': {
                'url': os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8002'),
                'health_endpoint': '/health',
                'timeout': 10,
                'retry_attempts': 3
            },
            'monitoring-service': {
                'url': os.getenv('MONITORING_SERVICE_URL', 'http://monitoring-service:8010'),
                'health_endpoint': '/health',
                'timeout': 5,
                'retry_attempts': 2
            }
        },
        
        # Cache Configuration
        "cache": {
            "enabled": os.getenv('MCP_CACHE_ENABLED', 'true').lower() == 'true',
            "redis_url": os.getenv('REDIS_URL', 'redis://redis:6379'),
            "default_ttl": int(os.getenv('MCP_CACHE_TTL', '300')),  # 5 minutes
            "max_memory": os.getenv('MCP_CACHE_MAX_MEMORY', '256mb'),
            "eviction_policy": os.getenv('MCP_CACHE_EVICTION_POLICY', 'allkeys-lru')
        },
        
        # Metrics and Monitoring
        "metrics": {
            "enabled": os.getenv('MCP_METRICS_ENABLED', 'true').lower() == 'true',
            "prometheus_endpoint": os.getenv('MCP_PROMETHEUS_ENDPOINT', '/metrics'),
            "custom_metrics": os.getenv('MCP_CUSTOM_METRICS', 'true').lower() == 'true',
            "export_interval": int(os.getenv('MCP_METRICS_EXPORT_INTERVAL', '15'))
        },
        
        # Advanced Features
        "features": {
            "async_processing": os.getenv('MCP_ASYNC_PROCESSING', 'true').lower() == 'true',
            "background_tasks": os.getenv('MCP_BACKGROUND_TASKS', 'true').lower() == 'true',
            "auto_scaling": os.getenv('MCP_AUTO_SCALING', 'false').lower() == 'true',
            "load_balancing": os.getenv('MCP_LOAD_BALANCING', 'false').lower() == 'true'
        }
    }

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and optimize configuration values"""
    
    # Ensure port is in valid range
    if not 1024 <= config["server"]["port"] <= 65535:
        config["server"]["port"] = 8015
    
    # Ensure worker count is reasonable
    import os
    cpu_count = os.cpu_count() or 4
    if config["server"]["workers"] > cpu_count * 2:
        config["server"]["workers"] = min(cpu_count * 2, 8)
    
    # Validate timeout values
    if config["server"]["timeout"] < 1:
        config["server"]["timeout"] = 30
    if config["performance"]["request_timeout"] < 1:
        config["performance"]["request_timeout"] = 15
    
    # Ensure connection limits are reasonable
    if config["performance"]["max_concurrent_requests"] < 1:
        config["performance"]["max_concurrent_requests"] = 10
    if config["performance"]["connection_pool_size"] < 5:
        config["performance"]["connection_pool_size"] = 20
    
    return config

def get_production_config() -> Dict[str, Any]:
    """Get production-optimized configuration"""
    config = get_optimal_mcp_config()
    
    # Production overrides
    production_overrides = {
        "server": {
            "workers": min(os.cpu_count() or 4, 8),
            "max_connections": 200
        },
        "performance": {
            "connection_pool_size": 50,
            "max_concurrent_requests": 25,
            "request_timeout": 30
        },
        "resilience": {
            "retry_attempts": 5,
            "circuit_breaker_threshold": 10
        },
        "logging": {
            "level": "INFO",
            "file_logging": True,
            "log_rotation": True
        },
        "security": {
            "api_key_required": True,
            "rate_limit_enabled": True,
            "max_requests_per_minute": 600
        }
    }
    
    # Merge production overrides
    for section, overrides in production_overrides.items():
        if section in config:
            config[section].update(overrides)
    
    return validate_config(config)

def get_development_config() -> Dict[str, Any]:
    """Get development-optimized configuration"""
    config = get_optimal_mcp_config()
    
    # Development overrides
    development_overrides = {
        "server": {
            "workers": 1,
            "max_connections": 50
        },
        "performance": {
            "connection_pool_size": 10,
            "max_concurrent_requests": 5
        },
        "logging": {
            "level": "DEBUG",
            "file_logging": False
        },
        "security": {
            "api_key_required": False,
            "rate_limit_enabled": False
        },
        "health_check": {
            "interval": 10
        }
    }
    
    # Merge development overrides
    for section, overrides in development_overrides.items():
        if section in config:
            config[section].update(overrides)
    
    return validate_config(config)

# Export configurations based on environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development').lower()

if ENVIRONMENT == 'production':
    OPTIMAL_MCP_CONFIG = get_production_config()
elif ENVIRONMENT == 'development':
    OPTIMAL_MCP_CONFIG = get_development_config()
else:
    OPTIMAL_MCP_CONFIG = get_optimal_mcp_config()

# Make the config easily importable
__all__ = ['OPTIMAL_MCP_CONFIG', 'get_optimal_mcp_config', 'validate_config', 'get_production_config', 'get_development_config']