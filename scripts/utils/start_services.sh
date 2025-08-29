#!/bin/bash
# VIPER Trading System - Service Startup Script

set -e

echo "🚀 VIPER Service Startup"
echo "======================="

# Start infrastructure services first
echo "🏗️  Starting infrastructure..."
docker run -d --name viper-redis --network viper-network -p 6379:6379 redis:7-alpine redis-server --appendonly yes || echo "Redis already running"

# Wait for Redis
echo "⏳ Waiting for Redis..."
sleep 5

# Start core services
echo "🔧 Starting core services..."
SERVICES=(
    "credential-vault:8008"
    "exchange-connector:8005"
    "data-manager:8003"
    "api-server:8000"
    "risk-manager:8002"
    "signal-processor:8011"
    "order-lifecycle-manager:8013"
    "live-trading-engine:8007"
    "ultra-backtester:8001"
    "strategy-optimizer:8004"
    "monitoring-service:8006"
    "centralized-logger:8015"
    "alert-system:8012"
    "position-synchronizer:8014"
    "market-data-streamer:8010"
    "github-manager:8001"
    "trading-optimizer:8002"
)

for service_info in "${SERVICES[@]}"; do
    service=$(echo $service_info | cut -d: -f1)
    port=$(echo $service_info | cut -d: -f2)
    
    echo "🚀 Starting $service on port $port..."
    docker run -d \
        --name "viper-$service" \
        --network viper-network \
        -p "$port:$port" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/config:/app/config" \
        -v "$(pwd)/.env:/app/.env" \
        --env-file .env \
        "viper-$service" || echo "⚠️  $service may already be running"
done

echo "🎉 All services started!"
echo ""
echo "📊 Service Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep viper
