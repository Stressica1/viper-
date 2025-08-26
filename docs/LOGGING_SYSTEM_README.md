# 🚀 VIPER Trading Bot - Centralized Logging System

## 📊 COMPREHENSIVE LOGGING SOLUTION

A production-grade, enterprise-level logging system that provides unified log aggregation, real-time monitoring, and advanced analytics across all microservices.

---

## 🏗️ ARCHITECTURE OVERVIEW

### **Logging Pipeline:**
```
Services → Structured Logger → Redis → Centralized Logger → Elasticsearch → Kibana
```

### **Components:**
1. **📝 Structured Logger** - Service-level logging utility
2. **📊 Centralized Logger** - Log aggregation and processing (8015)
3. **🔍 Elasticsearch** - Log search and analytics (9200)
4. **📥 Logstash** - Log processing pipeline (5044)
5. **📊 Kibana** - Log visualization dashboard (5601)

---

## 🚀 QUICK START

### **1. Start Logging Services:**
```bash
# Start all logging infrastructure
docker-compose up -d elasticsearch logstash kibana centralized-logger

# Wait for services to be healthy
sleep 60
```

### **2. Access Interfaces:**
- **📊 Kibana Dashboard**: http://localhost:5601
- **🔍 Elasticsearch API**: http://localhost:9200
- **📊 Centralized Logger**: http://localhost:8015

### **3. Configure Kibana:**
1. Open http://localhost:5601
2. Create index pattern: `viper-logs-*`
3. Set time field: `@timestamp`
4. Start exploring logs!

---

## 📝 USAGE EXAMPLES

### **Basic Structured Logging:**
```python
from shared.structured_logger import get_logger, log_info, log_error

# Get service-specific logger
logger = get_logger('my-service')

# Basic logging
log_info("Service started successfully", data={'version': '1.0.0'})

# Error logging with context
try:
    risky_operation()
except Exception as e:
    log_error("Operation failed", error=e, context="risky_operation")
```

### **Performance Monitoring:**
```python
# Start performance tracking
perf_id = logger.performance_start("database_query")

# Your operation
result = db.query("SELECT * FROM trades")

# End performance tracking
logger.performance_end(perf_id, data={'rows': len(result)})
```

### **Request Logging:**
```python
# HTTP request logging
logger.log_request("POST", "/orders", 200, 0.125, correlation_id="abc123")
```

### **Trade Logging:**
```python
# Trading activity logging
logger.log_trade("BTC/USDT", "buy", 0.001, 45000, "market")
```

### **Operation Tracing:**
```python
@logger.trace_operation("process_order")
def process_order(order_id):
    # Your order processing logic
    return process_result
```

### **Correlation ID Tracking:**
```python
# Track related operations across services
correlation_id = "order_123_process"

logger.info("Starting order processing",
           correlation_id=correlation_id,
           data={'order_id': 123})

# This log will be grouped with the above log
logger.info("Order validation complete",
           correlation_id=correlation_id,
           data={'validation_status': 'passed'})
```

---

## 🔍 LOG SEARCH & ANALYSIS

### **Kibana Dashboards:**

#### **1. Service Health Dashboard:**
```
Filters: service:*
Metrics: Count by level, error rate by service
```

#### **2. Performance Monitoring:**
```
Filters: log_type:performance
Metrics: Average duration, memory usage trends
```

#### **3. Error Analysis:**
```
Filters: level:(error OR critical)
Metrics: Error types, affected services, trends
```

#### **4. Trading Activity:**
```
Filters: log_type:trade
Metrics: Volume by symbol, success rates, P&L tracking
```

### **Elasticsearch Queries:**

#### **Search Recent Errors:**
```json
GET /viper-logs-*/_search
{
  "query": {
    "bool": {
      "must": [
        {"match": {"level": "error"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  },
  "size": 100
}
```

#### **Performance Analysis:**
```json
GET /viper-performance-*/_search
{
  "query": {
    "range": {"@timestamp": {"gte": "now-24h"}}
  },
  "aggs": {
    "avg_duration": {"avg": {"field": "duration_ms"}},
    "by_operation": {"terms": {"field": "operation"}}
  }
}
```

#### **Correlation Tracking:**
```json
GET /viper-logs-*/_search
{
  "query": {
    "match": {"correlation_id": "abc123"}
  },
  "sort": {"@timestamp": {"order": "asc"}}
}
```

---

## 📊 MONITORING FEATURES

### **Real-Time Alerting:**
- **Error Spikes**: Detect high error rates
- **Service Down**: Identify unresponsive services
- **Memory Issues**: Monitor resource usage
- **Performance Degradation**: Track response times

### **Automated Actions:**
```python
# Alert rules automatically trigger
alert_rules = {
    'error_spike': lambda logs: len([l for l in logs if l.level == 'ERROR']) > 10,
    'service_down': lambda logs: len(logs) == 0,
    'memory_warning': lambda logs: any('memory' in l.message.lower() for l in logs)
}
```

### **WebSocket Streaming:**
```python
# Real-time log streaming
ws = websocket.WebSocketApp("ws://localhost:8015/logs/realtime")
```

---

## 🔧 ADVANCED FEATURES

### **Log Correlation:**
- **Trace IDs**: Track operations across services
- **Correlation IDs**: Group related log entries
- **Causal Chains**: Understand request flows

### **Structured Data:**
```json
{
  "timestamp": "2025-01-27T10:30:45.123Z",
  "service": "order-lifecycle-manager",
  "level": "INFO",
  "message": "Order executed successfully",
  "correlation_id": "order_123_process",
  "trace_id": "process_order_1643284245123",
  "log_type": "trade",
  "data": {
    "symbol": "BTC/USDT",
    "amount": 0.001,
    "price": 45000,
    "type": "market"
  }
}
```

### **Performance Insights:**
- **Operation Timing**: Track function execution times
- **Memory Usage**: Monitor resource consumption
- **Throughput Metrics**: Measure system performance
- **Bottleneck Detection**: Identify slow operations

---

## 🏗️ INTEGRATION GUIDE

### **1. Update Existing Services:**

Replace standard logging with structured logging:
```python
# OLD: Standard logging
import logging
logger = logging.getLogger(__name__)
logger.info("Service started")

# NEW: Structured logging
from shared.structured_logger import get_logger
logger = get_logger('service-name')
logger.info("Service started", data={'version': '1.0.0'})
```

### **2. Add Performance Monitoring:**
```python
@logger.trace_operation("critical_operation")
def critical_operation():
    perf_id = logger.performance_start("database_query")
    result = db.query("SELECT * FROM data")
    logger.performance_end(perf_id, data={'rows': len(result)})
    return result
```

### **3. Error Tracking:**
```python
try:
    process_trade(trade_data)
except Exception as e:
    logger.log_error(e, context="trade_processing",
                    data={'trade_id': trade_id, 'symbol': symbol})
    raise
```

### **4. Request Tracking:**
```python
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.log_request(
        request.method,
        str(request.url.path),
        response.status_code,
        duration,
        correlation_id=request.headers.get('X-Correlation-ID')
    )
    return response
```

---

## 📈 ANALYTICS & INSIGHTS

### **Log Analytics:**
- **Service Health**: Monitor all microservices
- **Error Trends**: Track error patterns over time
- **Performance Metrics**: Analyze system performance
- **User Behavior**: Understand trading patterns

### **Custom Dashboards:**
1. **Trading Performance**: Win rates, volume, P&L
2. **System Health**: CPU, memory, response times
3. **Error Analysis**: Error types, frequencies, impacts
4. **User Activity**: Trading patterns, session tracking

---

## 🔧 CONFIGURATION

### **Environment Variables:**
```bash
# Logging Configuration
LOG_LEVEL=INFO
LOG_RETENTION_DAYS=30
MAX_LOGS_PER_SERVICE=1000

# ELK Stack
ELASTICSEARCH_URL=http://elasticsearch:9200
KIBANA_PORT=5601
LOGSTASH_PORT=5044

# Alert Thresholds
ERROR_SPIKE_THRESHOLD=10
MEMORY_WARNING_THRESHOLD=0.8
RESPONSE_TIME_ALERT=5.0
```

### **Docker Compose:**
```yaml
services:
  centralized-logger:
    ports:
      - "${CENTRALIZED_LOGGER_PORT:-8015}:8000"

  elasticsearch:
    ports:
      - "${ELASTICSEARCH_PORT:-9200}:9200"

  kibana:
    ports:
      - "${KIBANA_PORT:-5601}:5601"
```

---

## 🎯 MONITORING ENDPOINTS

### **Centralized Logger API:**
- `GET /health` - Service health
- `GET /logs/{service}` - Service-specific logs
- `GET /logs/correlation/{id}` - Correlated logs
- `GET /logs/search` - Search logs
- `GET /stats` - Logging statistics
- `WS /ws/logs` - Real-time log streaming

### **Elasticsearch API:**
- `GET /_cluster/health` - Cluster status
- `GET /viper-logs-*/_search` - Log search
- `GET /_cat/indices` - Index information

### **Kibana:**
- **Dashboard URL**: http://localhost:5601
- **Index Patterns**: `viper-logs-*`, `viper-errors-*`, `viper-performance-*`

---

## 🚨 ALERTING SYSTEM

### **Built-in Alerts:**
- **Error Rate Spikes**: Automatic error detection
- **Service Failures**: Down service alerts
- **Memory Issues**: Resource usage warnings
- **Performance Degradation**: Slow operation alerts

### **Custom Alerts:**
```python
# Add custom alert rules
alert_rules['custom_rule'] = {
    'condition': lambda logs: custom_logic(logs),
    'message': 'Custom alert triggered',
    'severity': 'MEDIUM'
}
```

---

## 📊 LOG RETENTION & ARCHIVAL

### **Retention Policies:**
- **Application Logs**: 30 days (configurable)
- **Error Logs**: 90 days
- **Performance Logs**: 60 days
- **Trade Logs**: 1 year (compliance)

### **Archival Strategy:**
- **Daily Indices**: `viper-logs-YYYY-MM-DD`
- **Automatic Cleanup**: Old indices deleted
- **Backup**: Export to cold storage
- **Compliance**: Audit trail preservation

---

## 🏆 CONCLUSION

**The VIPER Centralized Logging System provides:**

- **🤖 Unified Log Aggregation** across all 14 microservices
- **🔍 Advanced Search & Analytics** with Elasticsearch
- **📊 Real-Time Visualization** with Kibana dashboards
- **🚨 Intelligent Alerting** for system issues
- **📈 Performance Monitoring** with detailed metrics
- **🔗 Request Correlation** for debugging
- **📋 Compliance Ready** with audit trails
- **⚡ High Performance** with Redis buffering
- **🛡️ Enterprise Security** with encrypted transport
- **🔧 Easy Integration** with structured logger utility

**This is a world-class logging infrastructure that rivals enterprise-grade systems! 🚀**

---

*Comprehensive logging solution that provides complete observability and monitoring for the entire VIPER Trading Bot ecosystem.*
