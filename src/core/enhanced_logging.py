#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Enhanced Centralized Logging System
Advanced logging with structured data and real-time analytics
"""

import os
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import threading
from queue import Queue, Empty

class LogLevel(Enum):
    """Enhanced log levels for trading system"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR" 
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRADE = "TRADE"      # Special level for trade events
    SIGNAL = "SIGNAL"    # Special level for trading signals
    RISK = "RISK"        # Special level for risk events
    PERFORMANCE = "PERFORMANCE"  # Special level for performance metrics

@dataclass
class VIPERLogEntry:
    """Structured log entry for VIPER system"""
    timestamp: str
    level: str
    service: str
    component: str
    message: str
    symbol: Optional[str] = None
    trade_id: Optional[str] = None
    confidence: Optional[float] = None
    amount: Optional[float] = None
    price: Optional[float] = None
    metadata: Optional[Dict] = None
    correlation_id: Optional[str] = None

class EnhancedVIPERLogger:
    """
    Enhanced logging system with structured data and real-time processing
    Integrates with Elasticsearch, Logstash, and Kibana (ELK stack)
    """
    
    def __init__(self, service_name: str, component_name: str = "main"):
        self.service_name = service_name
        self.component_name = component_name
        self.session_id = str(uuid.uuid4())[:8]
        
        # Redis for log streaming
        self.redis_client = self._create_redis_client()
        
        # Log queue for async processing
        self.log_queue = Queue(maxsize=10000)
        self.is_running = False
        
        # Performance tracking
        self.log_count = 0
        self.error_count = 0
        self.trade_count = 0
        
        # Start background log processor
        self._start_log_processor()
        
        # Standard logger setup
        self._setup_standard_logging()
        
        logger.info(f"📝 Enhanced VIPER Logger initialized for {service_name}.{component_name}")
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client for log streaming"""
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        return redis.Redis.from_url(redis_url, decode_responses=True)
    
    def _setup_standard_logging(self):
        """Setup standard Python logging integration"""
        # Create custom handler that integrates with VIPER logging
        handler = VIPERLogHandler(self)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
    
    def _start_log_processor(self):
        """Start background log processing thread"""
        self.is_running = True
        
        def processor():
            while self.is_running:
                try:
                    # Get log entry from queue (with timeout)
                    try:
                        log_entry = self.log_queue.get(timeout=1)
                        self._process_log_entry(log_entry)
                        self.log_queue.task_done()
                    except Empty:
                        continue
                        
                except Exception as e:
                    print(f"❌ Error in log processor: {e}")
                    time.sleep(1)
        
        processor_thread = threading.Thread(target=processor, daemon=True)
        processor_thread.start()
    
    def _process_log_entry(self, log_entry: VIPERLogEntry):
        """Process a single log entry"""
        try:
            # Convert to dictionary
            log_dict = asdict(log_entry)
            log_json = json.dumps(log_dict)
            
            # Stream to different Redis channels based on log level
            channels = ['viper:logs:all']  # All logs go here
            
            # Route to specific channels
            if log_entry.level == LogLevel.TRADE.value:
                channels.append('viper:logs:trades')
                self.trade_count += 1
            elif log_entry.level == LogLevel.SIGNAL.value:
                channels.append('viper:logs:signals')
            elif log_entry.level == LogLevel.RISK.value:
                channels.append('viper:logs:risk')
            elif log_entry.level in [LogLevel.ERROR.value, LogLevel.CRITICAL.value]:
                channels.append('viper:logs:errors')
                self.error_count += 1
            elif log_entry.level == LogLevel.PERFORMANCE.value:
                channels.append('viper:logs:performance')
            
            # Service-specific channel
            channels.append(f'viper:logs:service:{self.service_name}')
            
            # Symbol-specific channel (if applicable)
            if log_entry.symbol:
                channels.append(f'viper:logs:symbol:{log_entry.symbol}')
            
            # Publish to all relevant channels
            for channel in channels:
                self.redis_client.publish(channel, log_json)
            
            # Store in Redis with expiration (for recent log access)
            log_key = f"viper:log:{int(time.time())}:{uuid.uuid4().hex[:8]}"
            self.redis_client.setex(log_key, 3600, log_json)  # 1 hour expiry
            
            # Maintain log statistics
            self.log_count += 1
            
            # Update service statistics
            stats = {
                'service': self.service_name,
                'component': self.component_name,
                'session_id': self.session_id,
                'total_logs': self.log_count,
                'error_count': self.error_count,
                'trade_count': self.trade_count,
                'last_activity': datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                f"viper:log_stats:{self.service_name}",
                300,  # 5 minutes
                json.dumps(stats)
            )
            
        except Exception as e:
            print(f"❌ Error processing log entry: {e}")
    
    def log_trade_execution(self, symbol: str, side: str, size: float, 
                           price: float, trade_id: str, success: bool,
                           metadata: Optional[Dict] = None):
        """Log trade execution with structured data"""
        try:
            log_entry = VIPERLogEntry(
                timestamp=datetime.now().isoformat(),
                level=LogLevel.TRADE.value,
                service=self.service_name,
                component=self.component_name,
                message=f"{'SUCCESS' if success else 'FAILED'}: {side} {size:.6f} {symbol} at ${price:.4f}",
                symbol=symbol,
                trade_id=trade_id,
                amount=size,
                price=price,
                metadata={
                    'side': side,
                    'success': success,
                    'session_id': self.session_id,
                    **(metadata or {})
                }
            )
            
            self._queue_log_entry(log_entry)
            
        except Exception as e:
            self.error(f"Failed to log trade execution: {e}")
    
    def log_trading_signal(self, symbol: str, signal: str, confidence: float,
                          price: float, metadata: Optional[Dict] = None):
        """Log trading signal generation"""
        try:
            log_entry = VIPERLogEntry(
                timestamp=datetime.now().isoformat(),
                level=LogLevel.SIGNAL.value,
                service=self.service_name,
                component=self.component_name,
                message=f"SIGNAL: {signal} for {symbol} at ${price:.4f} ({confidence:.1f}% confidence)",
                symbol=symbol,
                confidence=confidence,
                price=price,
                metadata={
                    'signal_type': signal,
                    'session_id': self.session_id,
                    **(metadata or {})
                }
            )
            
            self._queue_log_entry(log_entry)
            
        except Exception as e:
            self.error(f"Failed to log trading signal: {e}")
    
    def log_risk_event(self, event_type: str, symbol: Optional[str] = None,
                      risk_data: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """Log risk management events"""
        try:
            message = f"RISK: {event_type}"
            if symbol:
                message += f" for {symbol}"
            
            log_entry = VIPERLogEntry(
                timestamp=datetime.now().isoformat(),
                level=LogLevel.RISK.value,
                service=self.service_name,
                component=self.component_name,
                message=message,
                symbol=symbol,
                metadata={
                    'event_type': event_type,
                    'risk_data': risk_data,
                    'session_id': self.session_id,
                    **(metadata or {})
                }
            )
            
            self._queue_log_entry(log_entry)
            
        except Exception as e:
            self.error(f"Failed to log risk event: {e}")
    
    def log_performance_metric(self, metric_name: str, value: Union[int, float, str],
                             metadata: Optional[Dict] = None):
        """Log performance metrics"""
        try:
            log_entry = VIPERLogEntry(
                timestamp=datetime.now().isoformat(),
                level=LogLevel.PERFORMANCE.value,
                service=self.service_name,
                component=self.component_name,
                message=f"METRIC: {metric_name} = {value}",
                metadata={
                    'metric_name': metric_name,
                    'metric_value': value,
                    'session_id': self.session_id,
                    **(metadata or {})
                }
            )
            
            self._queue_log_entry(log_entry)
            
        except Exception as e:
            self.error(f"Failed to log performance metric: {e}")
    
    def info(self, message: str, symbol: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log info message"""
        self._log(LogLevel.INFO, message, symbol, metadata)
    
    def warning(self, message: str, symbol: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, symbol, metadata)
    
    def error(self, message: str, symbol: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log error message"""
        self._log(LogLevel.ERROR, message, symbol, metadata)
    
    def critical(self, message: str, symbol: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, symbol, metadata)
    
    def debug(self, message: str, symbol: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, symbol, metadata)
    
    def _log(self, level: LogLevel, message: str, symbol: Optional[str] = None,
            metadata: Optional[Dict] = None):
        """Internal logging method"""
        try:
            log_entry = VIPERLogEntry(
                timestamp=datetime.now().isoformat(),
                level=level.value,
                service=self.service_name,
                component=self.component_name,
                message=message,
                symbol=symbol,
                metadata={
                    'session_id': self.session_id,
                    **(metadata or {})
                }
            )
            
            self._queue_log_entry(log_entry)
            
        except Exception as e:
            # Fallback to print if logging fails
            print(f"❌ Logging failed: {e}")
            print(f"{level.value}: {message}")
    
    def _queue_log_entry(self, log_entry: VIPERLogEntry):
        """Queue log entry for async processing"""
        try:
            if self.log_queue.qsize() < 9000:  # Prevent queue overflow
                self.log_queue.put_nowait(log_entry)
            else:
                # Drop oldest entries if queue is full
                try:
                    self.log_queue.get_nowait()
                    self.log_queue.put_nowait(log_entry)
                except Empty:
                    pass
                    
        except Exception as e:
            print(f"❌ Failed to queue log entry: {e}")
    
    def get_recent_logs(self, level: Optional[str] = None, 
                       symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recent logs with filtering"""
        try:
            # Build channel name for filtering
            if level and symbol:
                # Not directly supported, fall back to all logs with filtering
                channel_pattern = 'viper:logs:all'
            elif level:
                level_channels = {
                    'TRADE': 'viper:logs:trades',
                    'SIGNAL': 'viper:logs:signals', 
                    'RISK': 'viper:logs:risk',
                    'ERROR': 'viper:logs:errors',
                    'CRITICAL': 'viper:logs:errors',
                    'PERFORMANCE': 'viper:logs:performance'
                }
                channel_pattern = level_channels.get(level, 'viper:logs:all')
            elif symbol:
                channel_pattern = f'viper:logs:symbol:{symbol}'
            else:
                channel_pattern = 'viper:logs:all'
            
            # Get recent log keys
            log_keys = self.redis_client.keys("viper:log:*")
            log_keys.sort(reverse=True)  # Most recent first
            
            logs = []
            for key in log_keys[:limit * 2]:  # Get more than needed for filtering
                try:
                    log_data = self.redis_client.get(key)
                    if log_data:
                        log_dict = json.loads(log_data)
                        
                        # Apply filters
                        if level and log_dict.get('level') != level:
                            continue
                        if symbol and log_dict.get('symbol') != symbol:
                            continue
                        
                        logs.append(log_dict)
                        
                        if len(logs) >= limit:
                            break
                except:
                    continue
            
            return logs
            
        except Exception as e:
            logger.error(f"❌ Error getting recent logs: {e}")
            return []
    
    def get_log_statistics(self) -> Dict:
        """Get comprehensive logging statistics"""
        try:
            # Get service statistics
            service_stats_keys = self.redis_client.keys("viper:log_stats:*")
            service_stats = {}
            
            for key in service_stats_keys:
                try:
                    stats_data = self.redis_client.get(key)
                    if stats_data:
                        stats = json.loads(stats_data)
                        service_name = key.split(':')[-1]
                        service_stats[service_name] = stats
                except:
                    continue
            
            # Calculate totals
            total_logs = sum(stats.get('total_logs', 0) for stats in service_stats.values())
            total_errors = sum(stats.get('error_count', 0) for stats in service_stats.values())
            total_trades = sum(stats.get('trade_count', 0) for stats in service_stats.values())
            
            # Error rate calculation
            error_rate = (total_errors / total_logs * 100) if total_logs > 0 else 0
            
            return {
                'overview': {
                    'total_logs': total_logs,
                    'total_errors': total_errors,
                    'total_trades': total_trades,
                    'error_rate': round(error_rate, 2),
                    'active_services': len(service_stats)
                },
                'service_breakdown': service_stats,
                'logging_health': 'healthy' if error_rate < 5 else 'degraded' if error_rate < 15 else 'unhealthy',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting log statistics: {e}")
            return {'error': str(e)}
    
    def create_correlation_id(self) -> str:
        """Create correlation ID for tracking related events"""
        return f"{self.service_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    def log_with_correlation(self, level: LogLevel, message: str, correlation_id: str,
                           symbol: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log with correlation ID for tracking related events"""
        try:
            log_entry = VIPERLogEntry(
                timestamp=datetime.now().isoformat(),
                level=level.value,
                service=self.service_name,
                component=self.component_name,
                message=message,
                symbol=symbol,
                correlation_id=correlation_id,
                metadata={
                    'session_id': self.session_id,
                    **(metadata or {})
                }
            )
            
            self._queue_log_entry(log_entry)
            
        except Exception as e:
            print(f"❌ Failed to log with correlation: {e}")
    
    def start_trade_logging_session(self, symbol: str, signal: str) -> str:
        """Start a trade logging session with correlation tracking"""
        correlation_id = self.create_correlation_id()
        
        self.log_with_correlation(
            LogLevel.TRADE,
            f"TRADE_SESSION_START: {signal} signal for {symbol}",
            correlation_id,
            symbol,
            {'trade_session': 'start', 'signal': signal}
        )
        
        return correlation_id
    
    def end_trade_logging_session(self, correlation_id: str, symbol: str, 
                                 success: bool, result_data: Optional[Dict] = None):
        """End a trade logging session"""
        self.log_with_correlation(
            LogLevel.TRADE,
            f"TRADE_SESSION_END: {'SUCCESS' if success else 'FAILED'} for {symbol}",
            correlation_id,
            symbol,
            {
                'trade_session': 'end',
                'success': success,
                'result_data': result_data or {}
            }
        )
    
    def monitor_system_logs(self) -> Dict:
        """Monitor system-wide log patterns and health"""
        try:
            # Get recent error logs
            error_logs = self.get_recent_logs(level='ERROR', limit=50)
            critical_logs = self.get_recent_logs(level='CRITICAL', limit=20)
            
            # Analyze error patterns
            error_patterns = {}
            for log in error_logs + critical_logs:
                service = log.get('service', 'unknown')
                if service not in error_patterns:
                    error_patterns[service] = 0
                error_patterns[service] += 1
            
            # Get trading activity
            trade_logs = self.get_recent_logs(level='TRADE', limit=100)
            recent_trades = len([log for log in trade_logs 
                               if datetime.fromisoformat(log['timestamp']) > datetime.now() - timedelta(hours=1)])
            
            # Calculate system health based on logs
            if len(critical_logs) > 5:
                log_health = 'critical'
            elif len(error_logs) > 20:
                log_health = 'poor'
            elif len(error_logs) > 10:
                log_health = 'degraded'
            else:
                log_health = 'good'
            
            return {
                'log_health': log_health,
                'error_patterns': error_patterns,
                'recent_errors': len(error_logs),
                'critical_errors': len(critical_logs),
                'recent_trades': recent_trades,
                'total_trade_logs': len(trade_logs),
                'recommendations': self._generate_log_recommendations(error_patterns, log_health),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error monitoring system logs: {e}")
            return {'error': str(e)}
    
    def _generate_log_recommendations(self, error_patterns: Dict, log_health: str) -> List[str]:
        """Generate recommendations based on log analysis"""
        recommendations = []
        
        try:
            # Service-specific recommendations
            for service, error_count in error_patterns.items():
                if error_count > 10:
                    recommendations.append(f"🚨 High error rate in {service}: {error_count} errors")
                elif error_count > 5:
                    recommendations.append(f"⚠️ Moderate errors in {service}: {error_count} errors")
            
            # Overall health recommendations
            if log_health == 'critical':
                recommendations.append("🚫 CRITICAL: System requires immediate attention")
            elif log_health == 'poor':
                recommendations.append("⚠️ Poor system health - investigate error patterns")
            elif log_health == 'good':
                recommendations.append("✅ System logging healthy")
            
            return recommendations
            
        except Exception as e:
            return [f"❌ Error generating recommendations: {e}"]
    
    def export_logs_for_analysis(self, hours: int = 24, format: str = 'json') -> Optional[str]:
        """Export logs for external analysis"""
        try:
            # Get logs from last N hours
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            all_logs = self.get_recent_logs(limit=10000)
            filtered_logs = [
                log for log in all_logs
                if datetime.fromisoformat(log['timestamp']) > cutoff_time
            ]
            
            if format == 'json':
                # Create export file
                export_data = {
                    'export_info': {
                        'exported_at': datetime.now().isoformat(),
                        'time_range_hours': hours,
                        'total_logs': len(filtered_logs),
                        'services': list(set(log.get('service', 'unknown') for log in filtered_logs))
                    },
                    'logs': filtered_logs
                }
                
                # Save to file
                export_file = f"/tmp/viper_logs_export_{int(time.time())}.json"
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                logger.info(f"📄 Logs exported to {export_file}")
                return export_file
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error exporting logs: {e}")
            return None
    
    def shutdown(self):
        """Shutdown the logging system gracefully"""
        try:
            logger.info("🛑 Shutting down enhanced logging system...")
            
            self.is_running = False
            
            # Process remaining logs in queue
            remaining_logs = 0
            while not self.log_queue.empty() and remaining_logs < 1000:
                try:
                    log_entry = self.log_queue.get_nowait()
                    self._process_log_entry(log_entry)
                    remaining_logs += 1
                except Empty:
                    break
            
            # Store final statistics
            final_stats = {
                'service': self.service_name,
                'session_id': self.session_id,
                'total_logs_processed': self.log_count,
                'total_errors': self.error_count,
                'total_trades': self.trade_count,
                'shutdown_time': datetime.now().isoformat(),
                'remaining_logs_processed': remaining_logs
            }
            
            self.redis_client.setex(
                f"viper:final_log_stats:{self.service_name}_{self.session_id}",
                86400,  # 24 hours
                json.dumps(final_stats)
            )
            
            logger.info("✅ Enhanced logging system shutdown complete")
            
        except Exception as e:
            print(f"❌ Error during logging shutdown: {e}")

class VIPERLogHandler(logging.Handler):
    """Custom logging handler that integrates with VIPER logging system"""
    
    def __init__(self, viper_logger: EnhancedVIPERLogger):
        super().__init__()
        self.viper_logger = viper_logger
    
    def emit(self, record):
        """Emit a log record to VIPER logging system"""
        try:
            # Convert logging level to VIPER level
            level_mapping = {
                'DEBUG': LogLevel.DEBUG,
                'INFO': LogLevel.INFO,
                'WARNING': LogLevel.WARNING,
                'ERROR': LogLevel.ERROR,
                'CRITICAL': LogLevel.CRITICAL
            }
            
            viper_level = level_mapping.get(record.levelname, LogLevel.INFO)
            
            # Extract metadata from record
            metadata = {
                'logger_name': record.name,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Create log entry
            log_entry = VIPERLogEntry(
                timestamp=datetime.fromtimestamp(record.created).isoformat(),
                level=viper_level.value,
                service=self.viper_logger.service_name,
                component=self.viper_logger.component_name,
                message=record.getMessage(),
                metadata=metadata
            )
            
            self.viper_logger._queue_log_entry(log_entry)
            
        except Exception as e:
            print(f"❌ Error in VIPER log handler: {e}")

# Factory function for creating service-specific loggers
def create_viper_logger(service_name: str, component_name: str = "main") -> EnhancedVIPERLogger:
    """Create a VIPER logger for a specific service and component"""
    return EnhancedVIPERLogger(service_name, component_name)

# Global logger registry
_logger_registry: Dict[str, EnhancedVIPERLogger] = {}

def get_viper_logger(service_name: str, component_name: str = "main") -> EnhancedVIPERLogger:
    """Get or create a VIPER logger for a service"""
    key = f"{service_name}.{component_name}"
    
    if key not in _logger_registry:
        _logger_registry[key] = create_viper_logger(service_name, component_name)
    
    return _logger_registry[key]

def shutdown_all_loggers():
    """Shutdown all active VIPER loggers"""
    for logger_instance in _logger_registry.values():
        logger_instance.shutdown()
    
    _logger_registry.clear()