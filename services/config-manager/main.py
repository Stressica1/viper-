#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Configuration Manager Service
Centralized configuration management for all trading parameters and settings

Features:
- Centralized parameter management
- Environment variable validation
- Configuration validation and schema enforcement
- Real-time configuration updates
- Configuration history and rollback
- Service-specific configuration profiles
"""

import os
import json
import logging
import hashlib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import uvicorn
import redis
from pathlib import Path
import yaml

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'config-manager')

# Backup configuration from environment variables
BACKUP_INTERVAL_HOURS = int(os.getenv('BACKUP_INTERVAL_HOURS', '24'))
BACKUP_RETENTION_DAYS = int(os.getenv('BACKUP_RETENTION_DAYS', '30'))

# Configuration paths
CONFIG_DIR = Path("/app/config")
ENV_FILE = Path("/app/.env")

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingConfig(BaseModel):
    """Trading configuration schema"""
    # VIPER Scoring
    viper_threshold_high: float = Field(default=85, ge=0, le=100)
    viper_threshold_medium: float = Field(default=70, ge=0, le=100)
    signal_cooldown: int = Field(default=300, ge=30, le=3600)  # seconds
    max_signals_per_symbol: int = Field(default=3, ge=1, le=10)

    # Risk Management
    risk_per_trade: float = Field(default=0.02, ge=0.001, le=0.1)  # 0.1% to 10%
    max_position_size_percent: float = Field(default=0.10, ge=0.01, le=1.0)  # 1% to 100%
    daily_loss_limit: float = Field(default=0.03, ge=0.001, le=0.1)  # 0.1% to 10%
    enable_auto_stops: bool = Field(default=True)
    max_positions: int = Field(default=15, ge=1, le=50)
    max_leverage: int = Field(default=50, ge=1, le=125)

    # Market Data
    enable_data_streaming: bool = Field(default=True)
    streaming_interval: int = Field(default=5, ge=1, le=300)  # seconds
    rate_limit_delay: float = Field(default=0.1, ge=0.01, le=1.0)  # seconds
    batch_size: int = Field(default=50, ge=10, le=200)
    cache_ttl: int = Field(default=300, ge=60, le=3600)  # seconds

    # Event System
    event_retention_seconds: int = Field(default=3600, ge=300, le=86400)
    max_events_per_channel: int = Field(default=1000, ge=100, le=10000)
    health_check_interval: int = Field(default=30, ge=10, le=300)
    dead_letter_ttl: int = Field(default=86400, ge=3600, le=604800)

    # Scanning
    scan_all_pairs: bool = Field(default=True)
    max_pairs_limit: int = Field(default=500, ge=50, le=2000)
    scan_interval_seconds: int = Field(default=300, ge=60, le=3600)
    leverage_scan_enabled: bool = Field(default=True)

class ServiceConfig(BaseModel):
    """Service-specific configuration"""
    service_name: str
    port: int
    url: str
    enabled: bool = True
    dependencies: List[str] = []
    config_overrides: Dict[str, Any] = {}

class ConfigurationManager:
    """Centralized configuration management service"""

    def __init__(self):
        self.redis_client = None
        self.current_config = {}
        self.config_history = []
        self.service_configs = {}
        self.is_running = False

        # Default trading configuration
        self.default_trading_config = TradingConfig()

        logger.info("⚙️ Configuration Manager initialized")

    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("# Check Redis connection established")
            return True
        except Exception as e:
            logger.error(f"# X Failed to connect to Redis: {e}")
            return False

    def load_config_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        try:
            config = {}

            # Load trading configuration
            trading_config = TradingConfig(
                viper_threshold_high=float(os.getenv('VIPER_THRESHOLD_HIGH', '85')),
                viper_threshold_medium=float(os.getenv('VIPER_THRESHOLD_MEDIUM', '70')),
                signal_cooldown=int(os.getenv('SIGNAL_COOLDOWN', '300')),
                max_signals_per_symbol=int(os.getenv('MAX_SIGNALS_PER_SYMBOL', '3')),
                risk_per_trade=float(os.getenv('RISK_PER_TRADE', '0.02')),
                max_position_size_percent=float(os.getenv('MAX_POSITION_SIZE_PERCENT', '0.10')),
                daily_loss_limit=float(os.getenv('DAILY_LOSS_LIMIT', '0.03')),
                enable_auto_stops=os.getenv('ENABLE_AUTO_STOPS', 'true').lower() == 'true',
                max_positions=int(os.getenv('MAX_POSITIONS', '15')),
                max_leverage=int(os.getenv('MAX_LEVERAGE', '50')),
                enable_data_streaming=os.getenv('ENABLE_DATA_STREAMING', 'true').lower() == 'true',
                streaming_interval=int(os.getenv('STREAMING_INTERVAL', '5')),
                rate_limit_delay=float(os.getenv('RATE_LIMIT_DELAY', '0.1')),
                batch_size=int(os.getenv('BATCH_SIZE', '50')),
                cache_ttl=int(os.getenv('CACHE_TTL', '300')),
                event_retention_seconds=int(os.getenv('EVENT_RETENTION_SECONDS', '3600')),
                max_events_per_channel=int(os.getenv('MAX_EVENTS_PER_CHANNEL', '1000')),
                health_check_interval=int(os.getenv('HEALTH_CHECK_INTERVAL', '30')),
                dead_letter_ttl=int(os.getenv('DEAD_LETTER_TTL', '86400')),
                scan_all_pairs=os.getenv('SCAN_ALL_PAIRS', 'true').lower() == 'true',
                max_pairs_limit=int(os.getenv('MAX_PAIRS_LIMIT', '500')),
                scan_interval_seconds=int(os.getenv('SCAN_INTERVAL_SECONDS', '300')),
                leverage_scan_enabled=os.getenv('LEVERAGE_SCAN_ENABLED', 'true').lower() == 'true'
            )

            config['trading'] = trading_config.dict()

            # Load service configurations
            services = [
                {
                    'service_name': 'market-data-manager',
                    'port': int(os.getenv('MARKET_DATA_MANAGER_PORT', '8003')),
                    'url': os.getenv('MARKET_DATA_MANAGER_URL', 'http://market-data-manager:8003'),
                    'dependencies': []
                },
                {
                    'service_name': 'viper-scoring-service',
                    'port': int(os.getenv('VIPER_SCORING_SERVICE_PORT', '8009')),
                    'url': os.getenv('VIPER_SCORING_SERVICE_URL', 'http://viper-scoring-service:8009'),
                    'dependencies': ['market-data-manager']
                },
                {
                    'service_name': 'live-trading-engine',
                    'port': int(os.getenv('LIVE_TRADING_ENGINE_PORT', '8007')),
                    'url': os.getenv('LIVE_TRADING_ENGINE_URL', 'http://live-trading-engine:8007'),
                    'dependencies': ['viper-scoring-service', 'risk-manager']
                },
                {
                    'service_name': 'risk-manager',
                    'port': int(os.getenv('RISK_MANAGER_PORT', '8002')),
                    'url': os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8002'),
                    'dependencies': ['market-data-manager']
                },
                {
                    'service_name': 'unified-scanner',
                    'port': int(os.getenv('UNIFIED_SCANNER_PORT', '8011')),
                    'url': os.getenv('UNIFIED_SCANNER_URL', 'http://unified-scanner:8011'),
                    'dependencies': ['market-data-manager', 'viper-scoring-service']
                },
                {
                    'service_name': 'event-system',
                    'port': int(os.getenv('EVENT_SYSTEM_PORT', '8010')),
                    'url': os.getenv('EVENT_SYSTEM_URL', 'http://event-system:8010'),
                    'dependencies': []
                }
            ]

            config['services'] = {service['service_name']: service for service in services}

            logger.info("# Check Configuration loaded from environment")
            return config

        except Exception as e:
            logger.error(f"# X Error loading configuration from environment: {e}")
            return {}

    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema"""
        try:
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': []
            }

            # Validate trading configuration
            if 'trading' in config:
                try:
                    TradingConfig(**config['trading'])
                    logger.info("# Check Trading configuration validated")
                except Exception as e:
                    validation_results['valid'] = False
                    validation_results['errors'].append(f"Trading config validation failed: {e}")

            # Validate service configurations
            if 'services' in config:
                for service_name, service_config in config['services'].items():
                    try:
                        ServiceConfig(**service_config)
                    except Exception as e:
                        validation_results['valid'] = False
                        validation_results['errors'].append(f"Service {service_name} config validation failed: {e}")

            # Cross-validation
            if 'trading' in config and 'services' in config:
                trading_config = config['trading']

                # Check risk management consistency
                if trading_config.get('risk_per_trade', 0) * trading_config.get('max_positions', 0) > 1.0:
                    validation_results['warnings'].append(
                        "Warning: Total risk exposure may exceed 100% with current settings"
                    )

                # Check service dependencies
                for service_name, service_config in config['services'].items():
                    for dependency in service_config.get('dependencies', []):
                        if dependency not in config['services']:
                            validation_results['errors'].append(
                                f"Service {service_name} depends on {dependency} which is not configured"
                            )

            return validation_results

        except Exception as e:
            logger.error(f"# X Configuration validation error: {e}")
            return {'valid': False, 'errors': [str(e)], 'warnings': []}

    def save_configuration(self, config: Dict[str, Any], user: str = "system") -> bool:
        """Save configuration with history tracking"""
        try:
            # Validate configuration
            validation = self.validate_configuration(config)
            if not validation['valid']:
                logger.error(f"# X Configuration validation failed: {validation['errors']}")
                return False

            # Create configuration snapshot
            config_snapshot = {
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'user': user,
                'version': len(self.config_history) + 1,
                'hash': hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
            }

            # Save to history
            self.config_history.append(config_snapshot)

            # Keep only last 10 versions
            if len(self.config_history) > 10:
                self.config_history = self.config_history[-10:]

            # Save to Redis
            self.redis_client.set('viper:current_config', json.dumps(config))
            self.redis_client.set('viper:config_history', json.dumps(self.config_history))

            # Update current config
            self.current_config = config

            # Publish configuration update event
            update_event = {
                'type': 'config_updated',
                'timestamp': datetime.now().isoformat(),
                'user': user,
                'version': config_snapshot['version']
            }

            self.redis_client.publish('config_events', json.dumps(update_event))

            logger.info(f"# Check Configuration saved (version {config_snapshot['version']})")
            return True

        except Exception as e:
            logger.error(f"# X Error saving configuration: {e}")
            return False

    def load_configuration(self) -> Dict[str, Any]:
        """Load current configuration"""
        try:
            # Try to load from Redis first
            config_data = self.redis_client.get('viper:current_config')
            if config_data:
                config = json.loads(config_data)
                self.current_config = config
                logger.info("# Check Configuration loaded from Redis")
                return config

            # Fallback to loading from environment
            config = self.load_config_from_env()
            if config:
                self.save_configuration(config, "system")
                return config

            logger.error("# X No configuration available")
            return {}

        except Exception as e:
            logger.error(f"# X Error loading configuration: {e}")
            return {}

    def get_service_config(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific service"""
        try:
            if service_name in self.current_config.get('services', {}):
                service_config = self.current_config['services'][service_name].copy()

                # Merge with trading configuration overrides
                trading_config = self.current_config.get('trading', {})
                service_config['trading_config'] = trading_config

                return service_config

            return None

        except Exception as e:
            logger.error(f"# X Error getting service config for {service_name}: {e}")
            return None

    def update_trading_parameter(self, parameter: str, value: Any, user: str = "system") -> bool:
        """Update a specific trading parameter"""
        try:
            if 'trading' not in self.current_config:
                self.current_config['trading'] = {}

            # Validate the parameter exists in our schema
            temp_config = TradingConfig(**self.current_config['trading'])
            if not hasattr(temp_config, parameter):
                logger.error(f"# X Unknown trading parameter: {parameter}")
                return False

            # Update the parameter
            self.current_config['trading'][parameter] = value

            # Validate the updated configuration
            validation = self.validate_configuration(self.current_config)
            if not validation['valid']:
                logger.error(f"# X Parameter update validation failed: {validation['errors']}")
                return False

            # Save the updated configuration
            return self.save_configuration(self.current_config, user)

        except Exception as e:
            logger.error(f"# X Error updating trading parameter: {e}")
            return False

    def get_configuration_history(self, limit: int = 10) -> List[Dict]:
        """Get configuration change history"""
        try:
            return self.config_history[-limit:] if self.config_history else []
        except Exception as e:
            logger.error(f"# X Error getting configuration history: {e}")
            return []

    def export_configuration(self, format: str = "json") -> str:
        """Export current configuration"""
        try:
            if format.lower() == "yaml":
                return yaml.dump(self.current_config, default_flow_style=False)
            else:
                return json.dumps(self.current_config, indent=2)
        except Exception as e:
            logger.error(f"# X Error exporting configuration: {e}")
            return ""

    def import_configuration(self, config_data: str, format: str = "json", user: str = "system") -> bool:
        """Import configuration from string"""
        try:
            if format.lower() == "yaml":
                config = yaml.safe_load(config_data)
            else:
                config = json.loads(config_data)

            return self.save_configuration(config, user)

        except Exception as e:
            logger.error(f"# X Error importing configuration: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get system configuration status"""
        try:
            return {
                'config_loaded': bool(self.current_config),
                'config_version': len(self.config_history),
                'last_update': self.config_history[-1]['timestamp'] if self.config_history else None,
                'services_configured': len(self.current_config.get('services', {})),
                'trading_parameters': len(self.current_config.get('trading', {})),
                'redis_connected': self.redis_client is not None,
                'backup_interval_hours': BACKUP_INTERVAL_HOURS,
                'backup_retention_days': BACKUP_RETENTION_DAYS
            }
        except Exception as e:
            logger.error(f"# X Error getting system status: {e}")
            return {'error': str(e)}
    
    def create_backup(self) -> bool:
        """Create a configuration backup using environment settings"""
        try:
            import datetime
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_key = f"viper:config_backup:{timestamp}"
            
            backup_data = {
                'timestamp': timestamp,
                'config': self.current_config,
                'metadata': {
                    'backup_interval_hours': BACKUP_INTERVAL_HOURS,
                    'retention_days': BACKUP_RETENTION_DAYS,
                    'created_by': 'config_manager_auto_backup'
                }
            }
            
            # Store backup in Redis
            self.redis_client.setex(
                backup_key, 
                BACKUP_RETENTION_DAYS * 24 * 3600,  # Convert days to seconds
                json.dumps(backup_data)
            )
            
            logger.info(f"# Check Configuration backup created: {backup_key}")
            return True
            
        except Exception as e:
            logger.error(f"# X Error creating backup: {e}")
            return False
    
    def cleanup_old_backups(self) -> int:
        """Clean up backups older than retention period"""
        try:
            import datetime
            
            cutoff_timestamp = datetime.datetime.now() - datetime.timedelta(days=BACKUP_RETENTION_DAYS)
            cutoff_str = cutoff_timestamp.strftime("%Y%m%d_%H%M%S")
            
            # Get all backup keys
            backup_keys = self.redis_client.keys("viper:config_backup:*")
            deleted_count = 0
            
            for key in backup_keys:
                # Extract timestamp from key
                timestamp_str = key.decode('utf-8').split(':')[-1]
                if timestamp_str < cutoff_str:
                    self.redis_client.delete(key)
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"# Check Cleaned up {deleted_count} old backups")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"# X Error cleaning up backups: {e}")
            return 0

# FastAPI application
app = FastAPI(
    title="VIPER Configuration Manager",
    version="1.0.0",
    description="Centralized configuration management service"
)

config_manager = ConfigurationManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if not config_manager.initialize_redis():
        logger.error("# X Failed to initialize Redis")
        return

    # Load initial configuration
    config_manager.load_configuration()

    logger.info("# Check Configuration Manager started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = config_manager.get_system_status()
        return {
            "status": "healthy" if status.get('config_loaded') else "degraded",
            "service": "config-manager",
            "redis_connected": config_manager.redis_client is not None,
            **status
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "config-manager",
                "error": str(e)
            }
        )

@app.get("/api/config")
async def get_configuration():
    """Get current configuration"""
    try:
        return config_manager.current_config
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get configuration: {e}")

@app.post("/api/config")
async def save_configuration(request: Request, user: str = Query("api", description="User making the change")):
    """Save new configuration"""
    try:
        config_data = await request.json()
        success = config_manager.save_configuration(config_data, user)

        if success:
            return {"status": "saved", "message": "Configuration updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Configuration validation failed")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {e}")

@app.get("/api/config/service/{service_name}")
async def get_service_configuration(service_name: str):
    """Get configuration for a specific service"""
    try:
        service_config = config_manager.get_service_config(service_name)
        if service_config:
            return service_config
        else:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get service config: {e}")

@app.put("/api/config/trading/{parameter}")
async def update_trading_parameter(
    parameter: str,
    value: str = Query(..., description="New parameter value"),
    user: str = Query("api", description="User making the change")
):
    """Update a specific trading parameter"""
    try:
        # Convert value to appropriate type
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
        elif value.lower() in ['true', 'false']:
            value = value.lower() == 'true'

        success = config_manager.update_trading_parameter(parameter, value, user)

        if success:
            return {"status": "updated", "parameter": parameter, "value": value}
        else:
            raise HTTPException(status_code=400, detail=f"Parameter update failed for {parameter}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {e}")

@app.get("/api/config/history")
async def get_configuration_history(limit: int = Query(10, ge=1, le=50)):
    """Get configuration change history"""
    try:
        history = config_manager.get_configuration_history(limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get history: {e}")

@app.get("/api/config/export")
async def export_configuration(format: str = Query("json", description="Export format (json or yaml)")):
    """Export current configuration"""
    try:
        exported = config_manager.export_configuration(format)
        return {"configuration": exported, "format": format}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Export failed: {e}")

@app.post("/api/config/import")
async def import_configuration(
    request: Request,
    format: str = Query("json", description="Import format (json or yaml)"),
    user: str = Query("api", description="User importing the configuration")
):
    """Import configuration"""
    try:
        config_data = await request.text()
        success = config_manager.import_configuration(config_data, format, user)

        if success:
            return {"status": "imported", "message": "Configuration imported successfully"}
        else:
            raise HTTPException(status_code=400, detail="Configuration import failed")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {e}")

@app.get("/api/config/validate")
async def validate_current_configuration():
    """Validate current configuration"""
    try:
        validation = config_manager.validate_configuration(config_manager.current_config)
        return validation
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Validation failed: {e}")

@app.get("/api/status")
async def get_system_status():
    """Get system configuration status"""
    try:
        return config_manager.get_system_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get status: {e}")

@app.post("/api/backup/create")
async def create_configuration_backup():
    """Create a configuration backup"""
    try:
        success = config_manager.create_backup()
        if success:
            return {
                "status": "created",
                "message": "Configuration backup created successfully",
                "backup_settings": {
                    "interval_hours": BACKUP_INTERVAL_HOURS,
                    "retention_days": BACKUP_RETENTION_DAYS
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Backup creation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {e}")

@app.post("/api/backup/cleanup")
async def cleanup_old_backups():
    """Clean up old configuration backups"""
    try:
        deleted_count = config_manager.cleanup_old_backups()
        return {
            "status": "completed",
            "deleted_backups": deleted_count,
            "retention_policy": f"{BACKUP_RETENTION_DAYS} days"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e}")

if __name__ == "__main__":
    port = int(os.getenv("CONFIG_MANAGER_PORT", 8012))
    logger.info(f"Starting VIPER Configuration Manager on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
