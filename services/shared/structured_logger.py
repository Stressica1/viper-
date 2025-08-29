#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Structured Logger
Unified logging utility for all microservices

Features:
- Structured logging with correlation IDs
- Centralized log aggregation
- Multiple log levels and formats
- Redis-based log transport
- Performance monitoring integration
- Error tracking and alerting
"""

import os
import json
import logging
import time
from datetime import datetime
from collections import deque
import redis

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'unknown-service')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

class StructuredLogger:
    """Structured logging utility for microservices"""

    def __init__(self, service_name: str = None):
        self.service_name = service_name or SERVICE_NAME
        self.redis_client = None
        self.log_buffer = deque(maxlen=1000)  # Local buffer for reliability
        self.correlation_id = self._generate_correlation_id()
        self.trace_stack = []
        self.performance_metrics = {}
        self.error_counts = {}
        self.is_connected = False

        # Setup Redis connection
        self.connect_redis()

        # Configure standard Python logging
        self._setup_standard_logging()

    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def connect_redis(self):
        """Connect to Redis for log transport"""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            self.is_connected = True
        except Exception as e:
            self.is_connected = False

    def _setup_standard_logging(self):
        """Setup standard Python logging with structured format"""
        log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

        # Create custom formatter
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'service': self.__class__.__name__.replace('StructuredFormatter', '').replace('<', '').replace('>', ''),
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                    'correlation_id': getattr(record, 'correlation_id', None),
                    'trace_id': getattr(record, 'trace_id', None),
                    'data': getattr(record, 'data', {})
                }
                return json.dumps(log_entry)

        # Setup logger
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(log_level)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add structured handler
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)

    def log(self, level: str, message: str, correlation_id: str = None,
            trace_id: str = None, data: Dict = None, **kwargs):
        """Log a structured message"""
        try:
            # Create log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'service': self.service_name,
                'level': level.upper(),
                'message': message,
                'correlation_id': correlation_id or self.correlation_id,
                'trace_id': trace_id or self._get_current_trace_id(),
                'data': data or {},
                **kwargs
            }

            # Add to local buffer
            self.log_buffer.append(log_entry)

            # Update error counts
            if level.upper() == 'ERROR':
                error_key = data.get('error_type', 'unknown') if data else 'unknown'
                self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

            # Send to centralized logger via Redis
            if self.is_connected and self.redis_client:
                channel = f'logs:{self.service_name}'
                self.redis_client.publish(channel, json.dumps(log_entry))

            # Also log to standard Python logging
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            extra_data = {
                'correlation_id': log_entry['correlation_id'],
                'trace_id': log_entry['trace_id'],
                'data': log_entry['data']
            }
            log_method(message, extra=extra_data)

        except Exception as e:
            # Fallback to simple logging if structured logging fails
            print(f"[{self.service_name}] {level.upper()}: {message} | Error: {e}")

    def _get_current_trace_id(self) -> str:
        """Get current trace ID from stack"""
        return self.trace_stack[-1] if self.trace_stack else self.correlation_id

    def start_trace(self, operation: str) -> str:
        """Start a new trace for operation tracking"""
        trace_id = f"{operation}_{int(time.time() * 1000)}"
        self.trace_stack.append(trace_id)

        self.log('DEBUG', f'Starting operation: {operation}',
                trace_id=trace_id,
                data={'operation': operation, 'type': 'trace_start'})

        return trace_id

    def end_trace(self, trace_id: str, success: bool = True, data: Dict = None):
        """End a trace"""
        if self.trace_stack and self.trace_stack[-1] == trace_id:
            self.trace_stack.pop()

        self.log('DEBUG', f'Ending trace: {trace_id}',
                trace_id=trace_id,
                data={
                    'success': success,
                    'type': 'trace_end',
                    **(data or {})
                })

    def trace_operation(self, operation: str):
        """Decorator for tracing operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                trace_id = self.start_trace(operation)
                try:
                    result = func(*args, **kwargs)
                    self.end_trace(trace_id, success=True,
                                 data={'result_type': str(type(result).__name__)})
                    return result
                except Exception as e:
                    self.end_trace(trace_id, success=False,
                                 data={'error': str(e), 'error_type': type(e).__name__})
                    raise
            return wrapper
        return decorator

    def info(self, message: str, correlation_id: str = None, data: Dict = None, **kwargs):
        """Log info level message"""
        self.log('INFO', message, correlation_id, data=data, **kwargs)

    def debug(self, message: str, correlation_id: str = None, data: Dict = None, **kwargs):
        """Log debug level message"""
        self.log('DEBUG', message, correlation_id, data=data, **kwargs)

    def warning(self, message: str, correlation_id: str = None, data: Dict = None, **kwargs):
        """Log warning level message"""
        self.log('WARNING', message, correlation_id, data=data, **kwargs)

    def error(self, message: str, correlation_id: str = None, data: Dict = None, **kwargs):
        """Log error level message"""
        self.log('ERROR', message, correlation_id, data=data, **kwargs)

    def critical(self, message: str, correlation_id: str = None, data: Dict = None, **kwargs):
        """Log critical level message"""
        self.log('CRITICAL', message, correlation_id, data=data, **kwargs)

    def performance_start(self, operation: str) -> str:
        """Start performance monitoring"""
        perf_id = f"perf_{operation}_{int(time.time() * 1000)}"
        self.performance_metrics[perf_id] = {
            'operation': operation,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage()
        }
        return perf_id

    def performance_end(self, perf_id: str, data: Dict = None):
        """End performance monitoring"""
        if perf_id not in self.performance_metrics:
            return

        start_data = self.performance_metrics[perf_id]
        end_time = time.time()
        end_memory = self._get_memory_usage()

        perf_data = {
            'operation': start_data['operation'],
            'duration': end_time - start_data['start_time'],
            'memory_delta': end_memory - start_data['start_memory'],
            'start_memory': start_data['start_memory'],
            'end_memory': end_memory,
            'type': 'performance',
            **(data or {})
        }

        self.log('INFO', f'Performance: {start_data["operation"]}', data=perf_data)
        del self.performance_metrics[perf_id]

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def log_request(self, method: str, endpoint: str, status_code: int,
                   duration: float, correlation_id: str = None, user_id: str = None):
        """Log HTTP request"""
        self.log('INFO', f'{method} {endpoint} -> {status_code}',
                correlation_id=correlation_id,
                data={
                    'method': method,
                    'endpoint': endpoint,
                    'status_code': status_code,
                    'duration': duration,
                    'user_id': user_id,
                    'type': 'http_request'
                })

    def log_trade(self, symbol: str, side: str, amount: float, price: float,
                 order_type: str, correlation_id: str = None):
        """Log trading activity"""
        self.log('INFO', f'Trade: {side.upper()} {amount} {symbol} @ {price}',
                correlation_id=correlation_id,
                data={
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'order_type': order_type,
                    'type': 'trade'
                })

    def log_error(self, error: Exception, context: str = "",
                 correlation_id: str = None, data: Dict = None):
        """Log exception with context"""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': self._get_traceback(error),
            'type': 'exception',
            **(data or {})
        }

        self.log('ERROR', f'Exception in {context}: {str(error)}',
                correlation_id=correlation_id, data=error_data)

    def _get_traceback(self, error: Exception) -> str:
        """Get formatted traceback"""
        import traceback
        return ''.join(traceback.format_exception(type(error), error, error.__traceback__))

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_types': self.error_counts.copy(),
            'timestamp': datetime.now().isoformat()
        }

    def get_log_summary(self) -> Dict[str, Any]:
        """Get log summary"""
        return {
            'service': self.service_name,
            'buffered_logs': len(self.log_buffer),
            'correlation_id': self.correlation_id,
            'active_traces': len(self.trace_stack),
            'performance_metrics': len(self.performance_metrics),
            'error_summary': self.get_error_summary(),
            'timestamp': datetime.now().isoformat()
        }

# Global logger instance
logger = StructuredLogger()

# Convenience functions for easy importing
def get_logger(service_name: str = None) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(service_name)

def log_info(message: str, **kwargs):
    """Convenience function for info logging"""
    logger.info(message, **kwargs)

def log_error(message: str, **kwargs):
    """Convenience function for error logging"""
    logger.error(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Convenience function for warning logging"""
    logger.warning(message, **kwargs)

def log_debug(message: str, **kwargs):
    """Convenience function for debug logging"""
    logger.debug(message, **kwargs)

def trace_operation(operation: str):
    """Convenience decorator for tracing operations"""
    return logger.trace_operation(operation)

# Example usage in services:
"""
from shared.structured_logger import get_logger, log_info, log_error, trace_operation

# Get service-specific logger
logger = get_logger('my-service')

# Basic logging
log_info("Service started", data={'version': '1.0.0'})

# Error logging
try:
    risky_operation()
except Exception as e:
    log_error("Operation failed", error=e, context="risky_operation")

# Performance tracing
@trace_operation("database_query")
def query_database():
    return db.query("SELECT * FROM trades")

# Request logging
logger.log_request("POST", "/orders", 200, 0.125, correlation_id="abc123")

# Trade logging
logger.log_trade("BTC/USDT", "buy", 0.001, 45000, "market")
"""
