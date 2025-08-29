#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Monitoring Service
System metrics collection, alerting, and performance monitoring

Features:
- Real-time metrics collection from all services
- Alert system with configurable thresholds
- Performance monitoring and anomaly detection
- Dashboard data aggregation
- Health status monitoring
- Log aggregation and analysis
- RESTful API for monitoring data
"""

import os
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
import uvicorn
import redis
import httpx
import psutil

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'monitoring-service')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringService:
    """Monitoring service for system metrics and alerting"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False

        # Load configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        self.prometheus_endpoint = os.getenv('PROMETHEUS_ENDPOINT', '9090')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

        # Service configurations
        self.services = {
            'api-server': {'port': 8000, 'url': 'http://api-server:8000'},
            'ultra-backtester': {'port': 8001, 'url': 'http://ultra-backtester:8000'},
            'risk-manager': {'port': 8002, 'url': 'http://risk-manager:8000'},
            'data-manager': {'port': 8003, 'url': 'http://data-manager:8000'},
            'strategy-optimizer': {'port': 8004, 'url': 'http://strategy-optimizer:8000'},
            'exchange-connector': {'port': 8005, 'url': 'http://exchange-connector:8000'},
            'monitoring-service': {'port': 8006, 'url': 'http://monitoring-service:8000'}
        }

        # Alert configurations loaded from environment variables
        self.alert_rules = {
            'service_down': {'threshold': 0, 'duration': 30, 'enabled': True},
            'high_cpu_usage': {
                'threshold': float(os.getenv('ALERT_HIGH_CPU_THRESHOLD', '80.0')), 
                'duration': 60, 
                'enabled': True
            },
            'high_memory_usage': {
                'threshold': float(os.getenv('ALERT_HIGH_MEMORY_THRESHOLD', '85.0')), 
                'duration': 60, 
                'enabled': True
            },
            'low_disk_space': {
                'threshold': float(os.getenv('ALERT_LOW_DISK_THRESHOLD', '90.0')), 
                'duration': 300, 
                'enabled': True
            },
            'risk_limit_breached': {'enabled': True},
            'api_rate_limit': {'threshold': 100, 'duration': 60, 'enabled': True}
        }

        # Metrics storage
        self.metrics_history = {}
        self.active_alerts = []

        logger.info("# Construction Initializing Monitoring Service...")

    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("# Check Redis connection established")
            return True
        except Exception as e:
            logger.error(f"# X Failed to connect to Redis: {e}")
            return False

    async def check_service_health(self, service_name: str) -> Dict:
        """Check health of a specific service"""
        try:
            service_config = self.services.get(service_name, {})
            service_url = service_config.get('url', '')

            if not service_url:
                return {
                    'service': service_name,
                    'status': 'unknown',
                    'response_time': None,
                    'error': 'Service URL not configured',
                    'timestamp': datetime.now().isoformat()
                }

            start_time = time.time()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service_url}/health")
                response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                if response.status_code == 200:
                    health_data = response.json()
                    return {
                        'service': service_name,
                        'status': 'healthy',
                        'response_time': round(response_time, 2),
                        'details': health_data,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'service': service_name,
                        'status': 'unhealthy',
                        'response_time': round(response_time, 2),
                        'error': f'HTTP {response.status_code}',
                        'timestamp': datetime.now().isoformat()
                    }

        except httpx.TimeoutException:
            return {
                'service': service_name,
                'status': 'timeout',
                'response_time': None,
                'error': 'Request timeout',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'service': service_name,
                'status': 'error',
                'response_time': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def collect_system_metrics(self) -> Dict:
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used
            memory_total = memory.total

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            disk_total = disk.total

            # Network metrics (simplified)
            net_io = psutil.net_io_counters()
            bytes_sent = net_io.bytes_sent
            bytes_recv = net_io.bytes_recv

            return {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'usage_percent': memory_percent,
                    'used_bytes': memory_used,
                    'total_bytes': memory_total,
                    'free_bytes': memory.free
                },
                'disk': {
                    'usage_percent': disk_percent,
                    'free_bytes': disk_free,
                    'total_bytes': disk_total
                },
                'network': {
                    'bytes_sent': bytes_sent,
                    'bytes_recv': bytes_recv
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"# X Error collecting system metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    async def collect_all_service_metrics(self) -> Dict:
        """Collect health metrics from all services"""
        service_metrics = {}
        unhealthy_services = []

        for service_name in self.services.keys():
            health = await self.check_service_health(service_name)
            service_metrics[service_name] = health

            if health['status'] != 'healthy':
                unhealthy_services.append(service_name)

        # Collect system metrics
        system_metrics = self.collect_system_metrics()

        # Calculate overall system health
        total_services = len(self.services)
        healthy_services = total_services - len(unhealthy_services)
        system_health = (healthy_services / total_services) * 100

        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'system_health': round(system_health, 2),
            'total_services': total_services,
            'healthy_services': healthy_services,
            'unhealthy_services': unhealthy_services,
            'services': service_metrics,
            'system': system_metrics
        }

        return metrics_data

    def check_alerts(self, metrics_data: Dict) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []

        try:
            # Service down alerts
            for service_name, service_data in metrics_data.get('services', {}).items():
                if service_data.get('status') != 'healthy':
                    alert_key = f"service_down_{service_name}"
                    if self.should_trigger_alert(alert_key):
                        alerts.append({
                            'type': 'service_down',
                            'severity': 'critical',
                            'service': service_name,
                            'message': f'Service {service_name} is {service_data.get("status", "unknown")}',
                            'details': service_data,
                            'timestamp': datetime.now().isoformat()
                        })

            # System resource alerts
            system_metrics = metrics_data.get('system', {})

            # CPU usage alert
            cpu_usage = system_metrics.get('cpu', {}).get('usage_percent', 0)
            if cpu_usage > self.alert_rules['high_cpu_usage']['threshold']:
                alert_key = "high_cpu_usage"
                if self.should_trigger_alert(alert_key):
                    alerts.append({
                        'type': 'high_cpu_usage',
                        'severity': 'warning',
                        'message': f'High CPU usage: {cpu_usage:.1f}%',
                        'threshold': self.alert_rules['high_cpu_usage']['threshold'],
                        'current_value': cpu_usage,
                        'timestamp': datetime.now().isoformat()
                    })

            # Memory usage alert
            memory_usage = system_metrics.get('memory', {}).get('usage_percent', 0)
            if memory_usage > self.alert_rules['high_memory_usage']['threshold']:
                alert_key = "high_memory_usage"
                if self.should_trigger_alert(alert_key):
                    alerts.append({
                        'type': 'high_memory_usage',
                        'severity': 'warning',
                        'message': f'High memory usage: {memory_usage:.1f}%',
                        'threshold': self.alert_rules['high_memory_usage']['threshold'],
                        'current_value': memory_usage,
                        'timestamp': datetime.now().isoformat()
                    })

            # Disk space alert
            disk_usage = system_metrics.get('disk', {}).get('usage_percent', 0)
            if disk_usage > self.alert_rules['low_disk_space']['threshold']:
                alert_key = "low_disk_space"
                if self.should_trigger_alert(alert_key):
                    alerts.append({
                        'type': 'low_disk_space',
                        'severity': 'critical',
                        'message': f'Low disk space: {disk_usage:.1f}% used',
                        'threshold': self.alert_rules['low_disk_space']['threshold'],
                        'current_value': disk_usage,
                        'timestamp': datetime.now().isoformat()
                    })

        except Exception as e:
            logger.error(f"# X Error checking alerts: {e}")
            alerts.append({
                'type': 'monitoring_error',
                'severity': 'critical',
                'message': f'Error in alert checking: {e}',
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def should_trigger_alert(self, alert_key: str) -> bool:
        """Check if alert should be triggered based on cooldown"""
        try:
            # Get alert rule
            rule_name = alert_key.split('_')[0]
            rule = self.alert_rules.get(rule_name, {})
            if not rule.get('enabled', True):
                return False

            # Check if alert was recently triggered
            last_alert = self.redis_client.get(f"viper:last_alert:{alert_key}")
            if last_alert:
                last_alert_time = datetime.fromisoformat(last_alert)
                cooldown_period = timedelta(seconds=rule.get('duration', 300))
                if datetime.now() - last_alert_time < cooldown_period:
                    return False

            # Update last alert time
            self.redis_client.setex(
                f"viper:last_alert:{alert_key}",
                rule.get('duration', 300),
                datetime.now().isoformat()
            )

            return True

        except Exception as e:
            logger.error(f"# X Error checking alert cooldown: {e}")
            return True  # Default to triggering alert on error

    def get_dashboard_data(self) -> Dict:
        """Aggregate data for dashboard display"""
        try:
            # Get latest metrics
            latest_metrics = self.redis_client.get('viper:latest_metrics')
            if latest_metrics:
                metrics_data = json.loads(latest_metrics)
            else:
                metrics_data = {}

            # Get recent alerts
            alerts = self.redis_client.lrange('viper:alerts', 0, 9)  # Last 10 alerts
            recent_alerts = []
            for alert in alerts:
                try:
                    recent_alerts.append(json.loads(alert))
                except Exception:
                    recent_alerts.append({'raw': alert})

            # Get service status summary
            services_status = {}
            for service_name, service_data in metrics_data.get('services', {}).items():
                services_status[service_name] = {
                    'status': service_data.get('status', 'unknown'),
                    'response_time': service_data.get('response_time'),
                    'last_check': service_data.get('timestamp')
                }

            # Get system resource summary
            system_data = metrics_data.get('system', {})
            system_summary = {
                'cpu_usage': system_data.get('cpu', {}).get('usage_percent', 0),
                'memory_usage': system_data.get('memory', {}).get('usage_percent', 0),
                'disk_usage': system_data.get('disk', {}).get('usage_percent', 0),
                'uptime': self.get_system_uptime()
            }

            # Get trading performance summary (from risk manager if available)
            trading_summary = {
                'daily_pnl': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'risk_score': 85
            }

            return {
                'timestamp': datetime.now().isoformat(),
                'system_health': metrics_data.get('system_health', 0),
                'services_status': services_status,
                'system_summary': system_summary,
                'trading_summary': trading_summary,
                'recent_alerts': recent_alerts,
                'active_alerts_count': len([a for a in recent_alerts if a.get('active', True)])
            }

        except Exception as e:
            logger.error(f"# X Error getting dashboard data: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def get_system_uptime(self) -> str:
        """Get system uptime"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                uptime_timedelta = timedelta(seconds=uptime_seconds)
                days = uptime_timedelta.days
                hours = uptime_timedelta.seconds // 3600
                minutes = (uptime_timedelta.seconds % 3600) // 60
                return f"{days}d {hours}h {minutes}m"
        except Exception:
            return "unknown"

    async def start_monitoring(self):
        """Start monitoring loop"""
        logger.info("# Rocket Starting monitoring loop...")
        self.is_running = True

        while self.is_running:
            try:
                # Collect metrics from all services
                metrics_data = await self.collect_all_service_metrics()

                # Store latest metrics
                self.redis_client.setex(
                    'viper:latest_metrics',
                    300,  # 5 minutes
                    json.dumps(metrics_data)
                )

                # Check for alerts
                alerts = self.check_alerts(metrics_data)

                # Store alerts
                for alert in alerts:
                    self.redis_client.lpush('viper:alerts', json.dumps(alert))
                    self.redis_client.expire('viper:alerts', 86400)  # 24 hours

                    logger.warning(f"🚨 Alert: {alert['type']} - {alert['message']}")

                # Store metrics history (keep last 100 entries)
                history_key = f"viper:metrics_history:{int(time.time())}"
                self.redis_client.setex(history_key, 86400, json.dumps(metrics_data))

                # Clean old history entries (keep last 100)
                history_keys = self.redis_client.keys("viper:metrics_history:*")
                if len(history_keys) > 100:
                    # Remove oldest entries
                    history_keys.sort()
                    for old_key in history_keys[:-100]:
                        self.redis_client.delete(old_key)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"# X Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    def stop(self):
        """Stop the monitoring service"""
        logger.info("🛑 Stopping Monitoring Service...")
        self.is_running = False

# FastAPI application
app = FastAPI(
    title="VIPER Monitoring Service",
    version="1.0.0",
    description="System monitoring, metrics collection, and alerting service"
)

monitor = MonitoringService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if not monitor.initialize_redis():
        logger.error("# X Failed to initialize Redis. Exiting...")
        return

    # Start monitoring in background task
    asyncio.create_task(monitor.start_monitoring())
    logger.info("# Check Monitoring Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    monitor.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "monitoring-service",
            "redis_connected": monitor.redis_client is not None,
            "monitoring_running": monitor.is_running,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "monitoring-service",
                "error": str(e)
            }
        )

@app.get("/api/metrics")
async def get_metrics():
    """Get current system metrics"""
    try:
        latest_metrics = monitor.redis_client.get('viper:latest_metrics')
        if latest_metrics:
            return json.loads(latest_metrics)
        else:
            # Return current metrics if none cached
            return await monitor.collect_all_service_metrics()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get metrics: {e}")

@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard data"""
    try:
        return monitor.get_dashboard_data()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get dashboard data: {e}")

@app.get("/api/services/{service_name}/health")
async def get_service_health(service_name: str):
    """Get health status of a specific service"""
    try:
        if service_name not in monitor.services:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")

        health = await monitor.check_service_health(service_name)
        return health
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get service health: {e}")

@app.get("/api/services/health")
async def get_all_services_health():
    """Get health status of all services"""
    try:
        metrics_data = await monitor.collect_all_service_metrics()
        return metrics_data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get services health: {e}")

@app.get("/api/alerts")
async def get_alerts(limit: int = Query(50, description="Number of alerts to retrieve", ge=1, le=100)):
    """Get recent alerts"""
    try:
        alerts = monitor.redis_client.lrange('viper:alerts', 0, limit - 1)
        parsed_alerts = []

        for alert in alerts:
            try:
                parsed_alerts.append(json.loads(alert))
            except Exception:
                parsed_alerts.append({'raw': alert})

        return {'alerts': parsed_alerts, 'count': len(parsed_alerts)}

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get alerts: {e}")

@app.delete("/api/alerts")
async def clear_alerts():
    """Clear all alerts"""
    try:
        monitor.redis_client.delete('viper:alerts')
        return {'status': 'cleared', 'message': 'All alerts cleared'}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to clear alerts: {e}")

@app.get("/api/metrics/history")
async def get_metrics_history(hours: int = Query(24, description="Hours of history to retrieve", ge=1, le=168)):
    """Get metrics history"""
    try:
        cutoff_time = int(time.time()) - (hours * 3600)
        history_keys = monitor.redis_client.keys("viper:metrics_history:*")

        history_data = []
        for key in history_keys:
            try:
                timestamp = int(key.split(':')[-1])
                if timestamp >= cutoff_time:
                    data = monitor.redis_client.get(key)
                    if data:
                        history_data.append(json.loads(data))
            except Exception:
                continue

        # Sort by timestamp
        history_data.sort(key=lambda x: x.get('timestamp', ''))

        return {'history': history_data, 'count': len(history_data)}

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get metrics history: {e}")

@app.get("/api/system/resources")
async def get_system_resources():
    """Get current system resource usage"""
    try:
        return monitor.collect_system_metrics()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get system resources: {e}")

@app.get("/api/config/alerts")
async def get_alert_config():
    """Get alert configuration"""
    return {
        'alert_rules': monitor.alert_rules,
        'services': list(monitor.services.keys())
    }

@app.post("/api/config/alerts")
async def update_alert_config(request: Request):
    """Update alert configuration"""
    try:
        data = await request.json()

        if 'alert_rules' in data:
            monitor.alert_rules.update(data['alert_rules'])

        return {
            'status': 'updated',
            'alert_rules': monitor.alert_rules
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to update alert config: {e}")

if __name__ == "__main__":
    port = int(os.getenv("MONITORING_SERVICE_PORT", 8000))
    logger.info(f"Starting VIPER Monitoring Service on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
