#!/bin/bash
# VIPER Trading System - Individual Service Builder
# Works around Docker path space issues

set -e

echo "🚀 VIPER Individual Service Builder"
echo "==================================="

# Service list in dependency order
SERVICES=(
    "redis"
    "credential-vault"
    "exchange-connector"
    "data-manager"
    "api-server"
    "risk-manager"
    "signal-processor"
    "order-lifecycle-manager"
    "live-trading-engine"
    "ultra-backtester"
    "strategy-optimizer"
    "monitoring-service"
    "centralized-logger"
    "alert-system"
    "position-synchronizer"
    "market-data-streamer"
    "github-manager"
    "trading-optimizer"
)

# Build each service
echo "🏗️  Building services..."
for service in "${SERVICES[@]}"; do
    if [ "$service" = "redis" ] || [ "$service" = "prometheus" ] || [ "$service" = "grafana" ] || [ "$service" = "elasticsearch" ] || [ "$service" = "logstash" ]; then
        echo "⏭️  Skipping pre-built service: $service"
        continue
    fi
    
    if [ -d "services/$service" ]; then
        echo "🔨 Building $service..."
        docker build -t "viper-$service" -f "services/$service/Dockerfile" "services/$service/"
        echo "✅ $service built successfully"
    else
        echo "⚠️  Service directory not found: $service"
    fi
done

echo "🎉 All services built successfully!"
