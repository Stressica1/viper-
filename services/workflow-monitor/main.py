#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Workflow Monitor Service
Comprehensive workflow validation and health monitoring for the entire trading system

Features:
    pass
- End-to-end workflow validation
- Service health monitoring
- Performance metrics collection
- Alert generation and escalation
- Workflow bottleneck detection
- Automated recovery procedures
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
import requests
import psutil

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'workflow-monitor')

# Monitoring Configuration
HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))  # seconds
WORKFLOW_VALIDATION_INTERVAL = int(os.getenv('WORKFLOW_VALIDATION_INTERVAL', '300'))  # 5 minutes
PERFORMANCE_MONITORING_INTERVAL = int(os.getenv('PERFORMANCE_MONITORING_INTERVAL', '60'))  # 1 minute
ALERT_ESCALATION_TIME = int(os.getenv('ALERT_ESCALATION_TIME', '600'))  # 10 minutes
MAX_WORKFLOW_LATENCY = float(os.getenv('MAX_WORKFLOW_LATENCY', '30.0'))  # seconds

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceHealth:
    """Service health monitoring"""

    def __init__(self, name: str, url: str, dependencies: List[str] = None):
        self.name = name
        self.url = url
        self.dependencies = dependencies or []
        self.status = "unknown"
        self.last_check = None
        self.response_time = 0.0
        self.error_count = 0
        self.consecutive_failures = 0
        self.last_error = None

    def check_health(self) -> Dict[str, Any]
        """Check service health""":"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.url}/health", timeout=10)
            response_time = time.time() - start_time

            if response.status_code == 200:
                health_data = response.json()
                self.status = health_data.get('status', 'unknown')
                self.response_time = response_time
                self.consecutive_failures = 0
                self.last_error = None
            else:
                self.status = "unhealthy"
                self.error_count += 1
                self.consecutive_failures += 1
                self.last_error = f"HTTP {response.status_code}: {response.text}"

            self.last_check = datetime.now()

        except requests.exceptions.RequestException as e:
            self.status = "unhealthy"
            self.error_count += 1
            self.consecutive_failures += 1
            self.last_error = str(e)
            self.last_check = datetime.now()
        except Exception as e:
            self.status = "error"
            self.error_count += 1
            self.consecutive_failures += 1
            self.last_error = str(e)
            self.last_check = datetime.now()

        return self.get_health_status()

    def get_health_status(self) -> Dict[str, Any]
        """Get current health status"""
        return {:
            'service': self.name,
            'status': self.status,
            'url': self.url,
            'dependencies': self.dependencies,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'response_time': round(self.response_time, 3),
            'error_count': self.error_count,
            'consecutive_failures': self.consecutive_failures,
            'last_error': self.last_error,
            'healthy': self.status in ['healthy', 'ok']
        }"""

class WorkflowStep:
    """Workflow step definition""""""

    def __init__(self, name: str, description: str, service: str, endpoint: str, expected_latency: float = 5.0):
        self.name = name
        self.description = description
        self.service = service
        self.endpoint = endpoint
        self.expected_latency = expected_latency
        self.last_execution = None
        self.last_latency = 0.0
        self.success_count = 0
        self.failure_count = 0
        self.average_latency = 0.0

    def execute_step(self, service_urls: Dict[str, str]) -> Dict[str, Any]
        """Execute workflow step""":"""
        try:
            if self.service not in service_urls:
                return {
                    'step': self.name,
                    'status': 'error',
                    'error': f'Service {self.service} not available',
                    'latency': 0.0
                }

            start_time = time.time()
            url = f"{service_urls[self.service]}{self.endpoint}"
            response = requests.get(url, timeout=15)
            latency = time.time() - start_time

            self.last_execution = datetime.now()
            self.last_latency = latency

            if response.status_code == 200:
                self.success_count += 1
                status = 'success'
            else:
                self.failure_count += 1
                status = 'failure'

            # Update average latency
            total_executions = self.success_count + self.failure_count
            if total_executions > 0:
                self.average_latency = ((self.average_latency * (total_executions - 1)) + latency) / total_executions

            return {
                'step': self.name,
                'status': status,
                'latency': round(latency, 3),
                'expected_latency': self.expected_latency,
                'response_code': response.status_code,
                'within_sla': latency <= self.expected_latency
            }

        except Exception as e:
            self.failure_count += 1
            return {
                'step': self.name,
                'status': 'error',
                'error': str(e),
                'latency': 0.0
            }

class WorkflowMonitor:
    """Comprehensive workflow monitoring service""""""

    def __init__(self):
        self.redis_client = None
        self.is_running = False
        self.services = {}
        self.workflows = {}
        self.alerts = []
        self.performance_metrics = {}
        self.service_urls = {}

        # Initialize service configurations
        self.initialize_services()

        logger.info("# Chart Workflow Monitor initialized")

    def initialize_redis(self) -> bool:
        """Initialize Redis connection""""""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("# Check Redis connection established")
            return True
        except Exception as e:
            logger.error(f"# X Failed to connect to Redis: {e}")
            return False

    def initialize_services(self):
        """Initialize service health monitors"""
        self.service_urls = {
            'market-data-manager': os.getenv('MARKET_DATA_MANAGER_URL', 'http://market-data-manager:8003'),
            'viper-scoring-service': os.getenv('VIPER_SCORING_SERVICE_URL', 'http://viper-scoring-service:8009'),
            'live-trading-engine': os.getenv('LIVE_TRADING_ENGINE_URL', 'http://live-trading-engine:8007'),
            'risk-manager': os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8002'),
            'unified-scanner': os.getenv('UNIFIED_SCANNER_URL', 'http://unified-scanner:8011'),
            'event-system': os.getenv('EVENT_SYSTEM_URL', 'http://event-system:8010'),
            'config-manager': os.getenv('CONFIG_MANAGER_URL', 'http://config-manager:8012')
        }

        # Initialize service health monitors
        for service_name, url in self.service_urls.items():
            dependencies = self.get_service_dependencies(service_name)
            self.services[service_name] = ServiceHealth(service_name, url, dependencies)

        # Initialize workflow definitions
        self.initialize_workflows()"""

    def get_service_dependencies(self, service_name: str) -> List[str]
        """Get service dependencies"""
        dependencies = {:
            'viper-scoring-service': ['market-data-manager'],
            'live-trading-engine': ['viper-scoring-service', 'risk-manager'],
            'unified-scanner': ['market-data-manager', 'viper-scoring-service'],
            'risk-manager': ['market-data-manager']
        }
        return dependencies.get(service_name, [])"""

    def initialize_workflows(self):
        """Initialize workflow definitions"""
        self.workflows = {
            'market_data_flow': [
                WorkflowStep('fetch_ticker', 'Fetch market ticker data', 'market-data-manager', '/api/market/BTC/USDT:USDT', 2.0),
                WorkflowStep('fetch_orderbook', 'Fetch orderbook data', 'market-data-manager', '/api/market/BTC/USDT:USDT', 3.0),
                WorkflowStep('fetch_ohlcv', 'Fetch OHLCV data', 'market-data-manager', '/api/market/BTC/USDT:USDT', 3.0)
            ],
            'signal_generation_flow': [
                WorkflowStep('get_market_data', 'Get market data for scoring', 'market-data-manager', '/api/market/BTC/USDT:USDT', 2.0),
                WorkflowStep('calculate_score', 'Calculate VIPER score', 'viper-scoring-service', '/api/config', 5.0),
                WorkflowStep('generate_signal', 'Generate trading signal', 'viper-scoring-service', '/api/config', 3.0)
            ],
            'trading_execution_flow': [
                WorkflowStep('risk_check', 'Check risk limits', 'risk-manager', '/api/limits', 2.0),
                WorkflowStep('position_sizing', 'Calculate position size', 'risk-manager', '/api/limits', 3.0),
                WorkflowStep('execute_trade', 'Execute trade', 'live-trading-engine', '/health', 5.0)
            ],
            'scanning_workflow': [
                WorkflowStep('discover_pairs', 'Discover trading pairs', 'unified-scanner', '/api/symbols', 10.0),
                WorkflowStep('scan_pair_data', 'Scan pair data', 'unified-scanner', '/api/opportunities', 15.0),
                WorkflowStep('generate_signals', 'Generate scanning signals', 'unified-scanner', '/api/opportunities', 10.0)
            ]
        }

    async def check_service_health(self):
        """Check health of all services"""
        while self.is_running:"""
            try:
                health_status = {}
                unhealthy_services = []

                for service_name, service in self.services.items():
                    health = service.check_health()
                    health_status[service_name] = health

                    if not health['healthy']:
                        unhealthy_services.append(service_name)

                        # Generate alert for unhealthy service
                        alert = {
                            'type': 'service_unhealthy',
                            'service': service_name,
                            'severity': 'high' if service.consecutive_failures > 3 else 'medium',
                            'message': f"Service {service_name} is unhealthy: {health.get('last_error', 'Unknown error')}",
                            'timestamp': datetime.now().isoformat(),
                            'failure_count': service.consecutive_failures
                        }

                        self.generate_alert(alert)

                # Check for dependency issues
                for service_name, service in self.services.items():
                    for dependency in service.dependencies:
                        if dependency in health_status and not health_status[dependency]['healthy']:
                            alert = {
                                'type': 'dependency_failure',
                                'service': service_name,
                                'dependency': dependency,
                                'severity': 'high',
                                'message': f"Service {service_name} dependency {dependency} is unhealthy",
                                'timestamp': datetime.now().isoformat()
                            }
                            self.generate_alert(alert)

                # Publish health status
                self.redis_client.setex('system_health', 60, json.dumps(health_status))

                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"# X Error in health check loop: {e}")
                await asyncio.sleep(10)

    def validate_workflow(self, workflow_name: str) -> Dict[str, Any]
        """Validate a complete workflow""":"""
        try:
            if workflow_name not in self.workflows:
                return {'error': f'Workflow {workflow_name} not found'}

            workflow = self.workflows[workflow_name]
            results = []

            start_time = time.time()
            for step in workflow:
                result = step.execute_step(self.service_urls)
                results.append(result)

            total_latency = time.time() - start_time

            # Analyze results
            failed_steps = [r for r in results if r['status'] in ['error', 'failure']]
            sla_violations = [r for r in results if not r.get('within_sla', True)]

            workflow_result = {
                'workflow': workflow_name,
                'timestamp': datetime.now().isoformat(),
                'total_latency': round(total_latency, 3),
                'max_allowed_latency': MAX_WORKFLOW_LATENCY,
                'within_sla': total_latency <= MAX_WORKFLOW_LATENCY,
                'steps': results,
                'failed_steps': len(failed_steps),
                'sla_violations': len(sla_violations),
                'status': 'failed' if failed_steps else 'success'
            }

            # Generate alerts for workflow failures
            if failed_steps:
                alert = {
                    'type': 'workflow_failure',
                    'workflow': workflow_name,
                    'severity': 'high',
                    'message': f"Workflow {workflow_name} failed with {len(failed_steps)} failed steps",
                    'failed_steps': [step['step'] for step in failed_steps],
                    'timestamp': datetime.now().isoformat()
                }
                self.generate_alert(alert)

            return workflow_result

        except Exception as e:
            logger.error(f"# X Error validating workflow {workflow_name}: {e}")
            return {'error': str(e)}

    async def run_workflow_validations(self):
        """Run periodic workflow validations"""
        while self.is_running:"""
            try:
                for workflow_name in self.workflows.keys():
                    logger.info(f"# Search Validating workflow: {workflow_name}")
                    result = self.validate_workflow(workflow_name)

                    # Store result in Redis
                    self.redis_client.setex(f"workflow:{workflow_name}", 600, json.dumps(result))

                    # Publish validation result
                    self.redis_client.publish('workflow_validations', json.dumps(}))
                        'workflow': workflow_name,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
((                    }))

                    await asyncio.sleep(1)  # Small delay between workflows

                await asyncio.sleep(WORKFLOW_VALIDATION_INTERVAL)

            except Exception as e:
                logger.error(f"# X Error in workflow validation loop: {e}")
                await asyncio.sleep(60)

    async def collect_performance_metrics(self):
        """Collect system performance metrics"""
        while self.is_running:"""
            try:
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'system': {
                        'cpu_percent': psutil.cpu_percent(interval=1),
                        'memory_percent': psutil.virtual_memory().percent,
                        'disk_usage': psutil.disk_usage('/').percent
                    },
                    'redis': self.get_redis_metrics(),
                    'services': {}
                }

                # Collect service-specific metrics
                for service_name, service in self.services.items():
                    if hasattr(service, 'response_time'):
                        metrics['services'][service_name] = {
                            'response_time': service.response_time,
                            'error_count': service.error_count,
                            'status': service.status
                        }

                # Store metrics
                self.redis_client.lpush('performance_metrics', json.dumps(metrics))
                self.redis_client.ltrim('performance_metrics', 0, 100)  # Keep last 100 entries

                # Check for performance issues
                if metrics['system']['cpu_percent'] > 90:
                    self.generate_alert(})
                        'type': 'high_cpu_usage',
                        'severity': 'medium',
                        'message': f"High CPU usage: {metrics['system']['cpu_percent']}%",
                        'timestamp': datetime.now().isoformat()
(                    })

                if metrics['system']['memory_percent'] > 90:
                    self.generate_alert(})
                        'type': 'high_memory_usage',
                        'severity': 'medium',
                        'message': f"High memory usage: {metrics['system']['memory_percent']}%",
                        'timestamp': datetime.now().isoformat()
(                    })

                await asyncio.sleep(PERFORMANCE_MONITORING_INTERVAL)

            except Exception as e:
                logger.error(f"# X Error collecting performance metrics: {e}")
                await asyncio.sleep(60)

    def get_redis_metrics(self) -> Dict[str, Any]
        """Get Redis performance metrics""":"""
        try:
            info = self.redis_client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'total_connections_received': info.get('total_connections_received', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"# X Error getting Redis metrics: {e}")
            return {}

    def generate_alert(self, alert: Dict[str, Any]):
        """Generate and escalate alerts""""""
        try:
            alert_id = f"{alert['type']}_{int(time.time())}"
            alert['id'] = alert_id
            alert['escalation_level'] = 1

            # Add alert to active alerts
            self.alerts.append(alert)

            # Keep only recent alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts = [a for a in self.alerts
                          if datetime.fromisoformat(a['timestamp']) > cutoff_time]:
            # Publish alert
            self.redis_client.publish('system_alerts', json.dumps(alert))
            self.redis_client.lpush('alert_history', json.dumps(alert))
            self.redis_client.expire('alert_history', 86400)  # 24 hours

            logger.warning(f"🚨 ALERT [{alert['severity'].upper()}]: {alert['message']}")

        except Exception as e:
            logger.error(f"# X Error generating alert: {e}")

    def get_system_overview(self) -> Dict[str, Any]
        """Get comprehensive system overview""":"""
        try:
            # Get latest workflow results
            workflow_status = {}
            for workflow_name in self.workflows.keys():
                result = self.redis_client.get(f"workflow:{workflow_name}")
                if result:
                    workflow_status[workflow_name] = json.loads(result)

            # Get service health
            service_health = {}
            for service_name, service in self.services.items():
                service_health[service_name] = service.get_health_status()

            # Get performance metrics
            latest_metrics = self.redis_client.lrange('performance_metrics', 0, 0)
            performance = json.loads(latest_metrics[0]) if latest_metrics else {}

            # Calculate system health score
            healthy_services = sum(1 for s in service_health.values() if s['healthy'])
            total_services = len(service_health)
            health_score = (healthy_services / total_services) * 100 if total_services > 0 else 0

            successful_workflows = sum(1 for w in workflow_status.values())
(                                     if w.get('status') == 'success')
            total_workflows = len(workflow_status)
            workflow_score = (successful_workflows / total_workflows) * 100 if total_workflows > 0 else 0

            overall_score = (health_score + workflow_score) / 2

            return {
                'timestamp': datetime.now().isoformat(),
                'system_health': {
                    'overall_score': round(overall_score, 2),
                    'health_score': round(health_score, 2),
                    'workflow_score': round(workflow_score, 2),
                    'status': 'healthy' if overall_score >= 80 else 'degraded' if overall_score >= 60 else 'critical'
                },
                'services': {
                    'total': total_services,
                    'healthy': healthy_services,
                    'unhealthy': total_services - healthy_services,
                    'details': service_health
                },
                'workflows': {
                    'total': total_workflows,
                    'successful': successful_workflows,
                    'failed': total_workflows - successful_workflows,
                    'details': workflow_status
                },
                'performance': performance,
                'active_alerts': len([a for a in self.alerts)
(                                    if (datetime.now() - datetime.fromisoformat(a['timestamp'])).seconds < 3600])
            }

        except Exception as e:
            logger.error(f"# X Error getting system overview: {e}")
            return {'error': str(e)}

    def start_background_tasks(self):
        """Start background monitoring tasks"""
        # Start health monitoring
        asyncio.create_task(self.check_service_health())

        # Start workflow validation
        asyncio.create_task(self.run_workflow_validations())

        # Start performance monitoring
        asyncio.create_task(self.collect_performance_metrics())

        logger.info("# Target Background monitoring tasks started")

    def start(self):
        """Start the workflow monitor service""""""
        try:
            logger.info("# Rocket Starting Workflow Monitor Service...")

            # Connect to Redis
            if not self.initialize_redis():
                raise Exception("Failed to connect to Redis")

            # Start background tasks
            self.is_running = True
            self.start_background_tasks()

            # Keep main thread alive
            while self.is_running:
                # Publish periodic system overview
                overview = self.get_system_overview()
                self.redis_client.setex('system_overview', 60, json.dumps(overview))

                # Check for alert escalation
                self.check_alert_escalation()

                asyncio.run(asyncio.sleep(60))  # Update every minute

        except KeyboardInterrupt:
            logger.info("⏹️ Stopping Workflow Monitor Service...")
            self.stop()
        except Exception as e:
            logger.error(f"# X Workflow Monitor Service error: {e}")
            self.stop()

    def check_alert_escalation(self):
        """Check and escalate alerts if needed""""""
        try:
            current_time = datetime.now()

            for alert in self.alerts:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                time_since_alert = (current_time - alert_time).seconds

                # Escalate if alert is older than escalation time and not already escalated
                if (time_since_alert > ALERT_ESCALATION_TIME and):
(                    alert.get('escalation_level', 1) < 3)
                    alert['escalation_level'] = alert.get('escalation_level', 1) + 1
                    alert['escalated_at'] = current_time.isoformat()

                    # Publish escalated alert
                    self.redis_client.publish('escalated_alerts', json.dumps(alert))

                    logger.error(f"🚨 ESCALATED ALERT [{alert['severity'].upper()}]: {alert['message']}")

        except Exception as e:
            logger.error(f"# X Error checking alert escalation: {e}")

    def stop(self):
        """Stop the workflow monitor service"""
        self.is_running = False
        logger.info("# Check Workflow Monitor Service stopped")

# FastAPI application
app = FastAPI()
    title="VIPER Workflow Monitor",
    version="1.0.0",
    description="Comprehensive workflow validation and health monitoring service"
()

workflow_monitor = WorkflowMonitor()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup""""""
    if not workflow_monitor.initialize_redis():
        logger.error("# X Failed to initialize Redis")
        return

    # Start background tasks
    asyncio.create_task(asyncio.to_thread(workflow_monitor.start_background_tasks))
    logger.info("# Check Workflow Monitor Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    workflow_monitor.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint""""""
    try:
        overview = workflow_monitor.get_system_overview()
        return {
            "status": overview['system_health']['status'],
            "service": "workflow-monitor",
            "redis_connected": workflow_monitor.redis_client is not None,
            "monitoring_active": workflow_monitor.is_running,
            **overview['system_health']
        }
    except Exception as e:
        return JSONResponse()
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "workflow-monitor",
                "error": str(e)
            }
(        )

@app.get("/api/overview")
async def get_system_overview():
    """Get comprehensive system overview""""""
    try:
        return workflow_monitor.get_system_overview()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get overview: {e}")

@app.post("/api/workflow/validate/{workflow_name}")
async def validate_workflow(workflow_name: str):
    """Validate a specific workflow""""""
    try:
        result = workflow_monitor.validate_workflow(workflow_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Validation failed: {e}")

@app.get("/api/workflows")
async def get_workflow_status():
    """Get status of all workflows""""""
    try:
        workflows = {}
        for workflow_name in workflow_monitor.workflows.keys():
            result = workflow_monitor.redis_client.get(f"workflow:{workflow_name}")
            if result:
                workflows[workflow_name] = json.loads(result)
        return {"workflows": workflows, "count": len(workflows)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get workflow status: {e}")

@app.get("/api/services/health")
async def get_service_health():
    """Get health status of all services""""""
    try:
        health_status = {}
        for service_name, service in workflow_monitor.services.items():
            health_status[service_name] = service.get_health_status()
        return {"services": health_status, "count": len(health_status)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get service health: {e}")

@app.get("/api/alerts")
async def get_alerts(hours: int = Query(24, description="Hours of alert history", ge=1, le=168))
    """Get recent alerts""""""
    try:
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in workflow_monitor.alerts
                        if datetime.fromisoformat(alert['timestamp']) > cutoff_time]:
        return {
            "alerts": recent_alerts,
            "count": len(recent_alerts),
            "hours": hours
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get alerts: {e}")

@app.get("/api/performance")
async def get_performance_metrics(limit: int = Query(10, description="Number of recent metrics", ge=1, le=100))
    """Get recent performance metrics""""""
    try:
        metrics = workflow_monitor.redis_client.lrange('performance_metrics', 0, limit - 1)
        parsed_metrics = [json.loads(metric) for metric in metrics]
        return {
            "metrics": parsed_metrics,
            "count": len(parsed_metrics)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get performance metrics: {e}")

@app.delete("/api/alerts")
async def clear_alerts():
    """Clear all active alerts""""""
    try:
        workflow_monitor.alerts.clear()
        return {"status": "cleared", "message": "All alerts cleared"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to clear alerts: {e}")

if __name__ == "__main__":
    port = int(os.getenv("WORKFLOW_MONITOR_PORT", 8013))
    logger.info(f"Starting VIPER Workflow Monitor on port {port}")
    uvicorn.run()
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
(    )
