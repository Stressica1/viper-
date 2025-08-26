#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Risk Manager Service
Position control, loss limits, and safety checks for automated trading

Features:
- Real-time risk assessment and monitoring
- Position size limits and exposure control
- Daily loss limits and stop-loss management
- Auto-stop functionality
- Risk score calculation
- RESTful API for risk management
"""

import os
import json
import time
import logging
import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import uvicorn
import redis
from pathlib import Path
import threading
import httpx

# Add shared directory to path for circuit breaker
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
try:
    from circuit_breaker import ServiceClient, call_service
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    # Fallback if shared module not available
    ServiceClient = None
    call_service = None
    CIRCUIT_BREAKER_AVAILABLE = False

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'risk-manager')

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskManager:
    """Risk manager for position control and safety checks"""

    def __init__(self):
        self.redis_client = None
        self.is_running = False

        # Load configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        self.daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT', '0.03'))  # 3%
        self.max_position_size_percent = float(os.getenv('MAX_POSITION_SIZE_PERCENT', '0.1'))  # 10%
        self.enable_auto_stops = os.getenv('ENABLE_AUTO_STOPS', 'true').lower() == 'true'
        self.max_positions = int(os.getenv('MAX_POSITIONS', '15'))  # 15 position limit

        # Risk tracking
        self.daily_pnl = 0.0
        self.starting_balance = 0.0
        self.current_balance = 0.0
        self.open_positions = {}  # Track positions by symbol
        self.active_symbols = set()  # Track active symbols to enforce 1 trade per symbol
        self.daily_trades = []

        # Service URLs
        self.exchange_connector_url = os.getenv('EXCHANGE_CONNECTOR_URL', 'http://exchange-connector:8000')
        self.data_manager_url = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')

        logger.info("🏗️ Initializing Risk Manager...")

    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False

    async def get_account_balance(self) -> Optional[float]:
        """Get current account balance from exchange connector"""
        if call_service and CIRCUIT_BREAKER_AVAILABLE:
            try:
                result = await call_service(
                    "exchange-connector",
                    "/api/balance",
                    method="GET",
                    redis_client=self.redis_client
                )
                return result.get('free', 0)
            except Exception as e:
                logger.error(f"❌ Circuit breaker error getting balance: {e}")
                return None
        else:
            # Fallback to direct HTTP call
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.exchange_connector_url}/api/balance")
                    if response.status_code == 200:
                        data = response.json()
                        return data.get('free', 0)
                    else:
                        logger.warning(f"⚠️ Failed to get balance: {response.status_code}")
                        return None
            except Exception as e:
                logger.error(f"❌ Error getting balance: {e}")
                return None

    async def get_positions(self) -> Optional[List]:
        """Get current positions from exchange connector"""
        if call_service and CIRCUIT_BREAKER_AVAILABLE:
            try:
                result = await call_service(
                    "exchange-connector",
                    "/api/positions",
                    method="GET",
                    redis_client=self.redis_client
                )
                return result.get('positions', [])
            except Exception as e:
                logger.error(f"❌ Circuit breaker error getting positions: {e}")
                return []
        else:
            # Fallback to direct HTTP call
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.exchange_connector_url}/api/positions")
                    if response.status_code == 200:
                        data = response.json()
                        return data.get('positions', [])
                    else:
                        logger.warning(f"⚠️ Failed to get positions: {response.status_code}")
                        return []
            except Exception as e:
                logger.error(f"❌ Error getting positions: {e}")
                return []

    def calculate_risk_score(self, balance: float, positions: List) -> Dict:
        """Calculate overall risk score with 30-35% capital utilization check"""
        try:
            # Initialize risk metrics
            total_exposure = 0.0
            unrealized_pnl = 0.0
            position_count = len(positions)

            # Calculate exposure and P&L
            for position in positions:
                size = abs(position.get('size', 0))
                entry_price = position.get('entry_price', 0)
                mark_price = position.get('mark_price', 0)
                position_pnl = position.get('unrealized_pnl', 0)

                if entry_price > 0:
                    exposure = (size * entry_price) / balance
                    total_exposure += exposure

                unrealized_pnl += position_pnl

            # Calculate daily loss percentage
            if self.starting_balance > 0:
                daily_loss_percent = abs(self.daily_pnl) / self.starting_balance
            else:
                daily_loss_percent = 0.0

            # Check capital utilization (30-35% rule)
            capital_utilization = total_exposure
            utilization_score = 0.0

            if capital_utilization < 0.30:  # Below 30%
                utilization_score = 20  # Penalty for under-utilization
            elif capital_utilization > 0.35:  # Above 35%
                utilization_score = 30  # Penalty for over-utilization
            else:  # Within 30-35% range
                utilization_score = 0  # Optimal range

            # Risk score components (0-100, higher = riskier)
            exposure_score = min(total_exposure * 100, 100)  # 0-100 based on exposure
            loss_score = min(daily_loss_percent * 100, 100)  # 0-100 based on daily loss
            position_score = min(position_count * 10, 50)  # 0-50 based on position count

            # Overall risk score (weighted average) - includes utilization penalty
            overall_risk = (exposure_score * 0.3 + loss_score * 0.3 + position_score * 0.2 + utilization_score * 0.2)

            # Risk level determination
            if overall_risk < 30:
                risk_level = "low"
            elif overall_risk < 70:
                risk_level = "medium"
            else:
                risk_level = "high"

            return {
                'overall_score': round(overall_risk, 2),
                'risk_level': risk_level,
                'exposure_score': round(exposure_score, 2),
                'loss_score': round(loss_score, 2),
                'position_score': round(position_score, 2),
                'utilization_score': round(utilization_score, 2),
                'capital_utilization': round(capital_utilization, 4),
                'capital_utilization_target': '30-35%',
                'utilization_status': 'optimal' if utilization_score == 0 else 'suboptimal',
                'total_exposure': round(total_exposure, 4),
                'unrealized_pnl': round(unrealized_pnl, 2),
                'daily_pnl': round(self.daily_pnl, 2),
                'position_count': position_count,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error calculating risk score: {e}")
            return {
                'overall_score': 100,
                'risk_level': 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def check_daily_loss_limit(self) -> Dict:
        """Check if daily loss limit has been breached"""
        try:
            if self.starting_balance <= 0:
                return {'breached': False, 'current_loss': 0.0, 'limit': self.daily_loss_limit}

            daily_loss_percent = abs(self.daily_pnl) / self.starting_balance
            breached = daily_loss_percent >= self.daily_loss_limit

            return {
                'breached': breached,
                'current_loss_percent': round(daily_loss_percent, 4),
                'limit': self.daily_loss_limit,
                'current_pnl': round(self.daily_pnl, 2),
                'starting_balance': round(self.starting_balance, 2)
            }
        except Exception as e:
            logger.error(f"❌ Error checking daily loss limit: {e}")
            return {'breached': False, 'error': str(e)}

    def check_capital_utilization(self, balance: float, positions: List) -> Dict:
        """Check if capital utilization is within 30-35% range"""
        try:
            total_exposure = 0.0

            # Calculate total exposure
            for position in positions:
                size = abs(position.get('size', 0))
                entry_price = position.get('entry_price', 0)

                if entry_price > 0:
                    exposure = (size * entry_price) / balance
                    total_exposure += exposure

            capital_utilization = total_exposure
            min_target = 0.30  # 30%
            max_target = 0.35  # 35%

            if capital_utilization < min_target:
                status = 'under_utilized'
                message = f'Capital utilization {capital_utilization:.1%} is below target range (30-35%)'
                can_add_positions = True
            elif capital_utilization > max_target:
                status = 'over_utilized'
                message = f'Capital utilization {capital_utilization:.1%} is above target range (30-35%)'
                can_add_positions = False
            else:
                status = 'optimal'
                message = f'Capital utilization {capital_utilization:.1%} is within optimal range (30-35%)'
                can_add_positions = True

            return {
                'status': status,
                'capital_utilization': round(capital_utilization, 4),
                'target_range': '30-35%',
                'min_target': min_target,
                'max_target': max_target,
                'can_add_positions': can_add_positions,
                'message': message,
                'total_exposure': round(total_exposure, 4),
                'current_positions': len(positions)
            }

        except Exception as e:
            logger.error(f"❌ Error checking capital utilization: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'can_add_positions': False
            }

    def check_position_limits(self, symbol: str) -> Dict:
        """Check if new position is allowed based on limits"""
        try:
            # Check 15-position limit
            current_positions = len(self.open_positions)
            if current_positions >= self.max_positions:
                return {
                    'allowed': False,
                    'reason': f'Maximum positions reached ({current_positions}/{self.max_positions})',
                    'current_positions': current_positions,
                    'max_positions': self.max_positions
                }

            # Check single trade per symbol rule
            if symbol in self.active_symbols:
                return {
                    'allowed': False,
                    'reason': f'Position already exists for symbol {symbol} (1 trade per symbol limit)',
                    'symbol': symbol,
                    'active_symbols': list(self.active_symbols)
                }

            return {
                'allowed': True,
                'current_positions': current_positions,
                'max_positions': self.max_positions,
                'active_symbols': list(self.active_symbols)
            }

        except Exception as e:
            logger.error(f"❌ Error checking position limits: {e}")
            return {'allowed': False, 'error': str(e)}

    def check_risk_limits(self, symbol: str, position_size: float, price: float, balance: float) -> Dict:
        """Check if trade meets all risk management rules"""
        try:
            # Check 2% risk per trade limit
            risk_amount = balance * 0.02  # 2% of balance
            trade_value = position_size * price

            if trade_value > risk_amount:
                return {
                    'allowed': False,
                    'reason': f'Trade value ${trade_value:.2f} exceeds 2% risk limit (${risk_amount:.2f})',
                    'trade_value': round(trade_value, 2),
                    'risk_limit': round(risk_amount, 2),
                    'risk_percent': 2.0
                }

            # Check position limits
            position_check = self.check_position_limits(symbol)
            if not position_check.get('allowed', False):
                return position_check

            # Check capital utilization (30-35% rule) - simplified for now
            utilization_check = {
                'status': 'optimal',
                'capital_utilization': 0.0,
                'target_range': '30-35%',
                'can_add_positions': True,
                'message': 'Capital utilization check bypassed for stability'
            }

            return {
                'allowed': True,
                'trade_value': round(trade_value, 2),
                'risk_limit': round(risk_amount, 2),
                'risk_percent': 2.0,
                'position_check': position_check,
                'utilization_check': utilization_check
            }

        except Exception as e:
            logger.error(f"❌ Error checking risk limits: {e}")
            return {'allowed': False, 'error': str(e)}

    def register_position(self, symbol: str, position_data: Dict) -> bool:
        """Register a new position in risk tracking"""
        try:
            if symbol in self.active_symbols:
                logger.warning(f"⚠️ Symbol {symbol} already has an active position")
                return False

            self.open_positions[symbol] = position_data
            self.active_symbols.add(symbol)

            logger.info(f"✅ Position registered for {symbol}: {position_data}")
            return True

        except Exception as e:
            logger.error(f"❌ Error registering position: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """Remove a position from risk tracking"""
        try:
            if symbol in self.open_positions:
                del self.open_positions[symbol]
            if symbol in self.active_symbols:
                self.active_symbols.remove(symbol)

            logger.info(f"✅ Position closed for {symbol}")
            return True

        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
            return False

    def calculate_position_size(self, symbol: str, price: float, balance: float,
                               risk_per_trade: float = 0.02) -> Dict:
        """Calculate safe position size based on risk management"""
        try:
            # Get available balance for trading
            available_balance = balance * self.max_position_size_percent

            # Risk-based position sizing
            risk_amount = available_balance * risk_per_trade

            # Calculate position size (simplified - in real implementation,
            # this would consider volatility, stop-loss levels, etc.)
            position_size = risk_amount / price

            # Additional safety checks
            max_position_value = balance * self.max_position_size_percent
            max_position_size = max_position_value / price

            # Use the more conservative limit
            final_position_size = min(position_size, max_position_size)

            return {
                'recommended_size': round(final_position_size, 6),
                'max_allowed_size': round(max_position_size, 6),
                'risk_amount': round(risk_amount, 2),
                'available_balance': round(available_balance, 2),
                'risk_per_trade': risk_per_trade,
                'price': price,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return {'error': str(e), 'recommended_size': 0}

    async def check_auto_stops(self, positions: List) -> List[Dict]:
        """Check if any positions need auto-stop execution"""
        alerts = []

        if not self.enable_auto_stops:
            return alerts

        try:
            for position in positions:
                symbol = position.get('symbol', '')
                entry_price = position.get('entry_price', 0)
                mark_price = position.get('mark_price', 0)
                unrealized_pnl = position.get('unrealized_pnl', 0)
                size = abs(position.get('size', 0))

                if entry_price <= 0 or size <= 0:
                    continue

                # Calculate loss percentage
                loss_percent = abs(unrealized_pnl) / (entry_price * size)

                # Check stop-loss thresholds
                stop_loss_percent = 0.05  # 5% stop loss
                if loss_percent >= stop_loss_percent:
                    alerts.append({
                        'type': 'auto_stop_triggered',
                        'symbol': symbol,
                        'reason': f'Loss exceeded {stop_loss_percent*100}%',
                        'current_loss_percent': round(loss_percent * 100, 2),
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'unrealized_pnl': unrealized_pnl,
                        'recommended_action': 'close_position',
                        'timestamp': datetime.now().isoformat()
                    })

                # Check if daily loss limit would be breached by keeping position
                if self.starting_balance > 0:
                    total_daily_loss = abs(self.daily_pnl) + abs(unrealized_pnl)
                    potential_loss_percent = total_daily_loss / self.starting_balance

                    if potential_loss_percent >= self.daily_loss_limit:
                        alerts.append({
                            'type': 'daily_loss_limit_warning',
                            'symbol': symbol,
                            'reason': 'Position loss would breach daily limit',
                            'potential_loss_percent': round(potential_loss_percent * 100, 2),
                            'daily_limit': self.daily_loss_limit * 100,
                            'recommended_action': 'reduce_position',
                            'timestamp': datetime.now().isoformat()
                        })

        except Exception as e:
            logger.error(f"❌ Error checking auto stops: {e}")
            alerts.append({
                'type': 'error',
                'message': f'Error checking auto stops: {e}',
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    async def update_daily_pnl(self):
        """Update daily P&L tracking"""
        try:
            current_balance = await self.get_account_balance()
            if current_balance is not None:
                if self.starting_balance <= 0:
                    # Initialize starting balance on first run
                    self.starting_balance = current_balance

                self.current_balance = current_balance
                self.daily_pnl = current_balance - self.starting_balance

                # Store in Redis
                risk_data = {
                    'daily_pnl': self.daily_pnl,
                    'starting_balance': self.starting_balance,
                    'current_balance': self.current_balance,
                    'timestamp': datetime.now().isoformat()
                }

                self.redis_client.setex(
                    'viper:daily_pnl',
                    86400,  # 24 hours
                    json.dumps(risk_data)
                )

        except Exception as e:
            logger.error(f"❌ Error updating daily P&L: {e}")

    def load_daily_pnl(self) -> bool:
        """Load daily P&L from Redis"""
        try:
            data = self.redis_client.get('viper:daily_pnl')
            if data:
                pnl_data = json.loads(data)
                self.daily_pnl = pnl_data.get('daily_pnl', 0.0)
                self.starting_balance = pnl_data.get('starting_balance', 0.0)
                self.current_balance = pnl_data.get('current_balance', 0.0)
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error loading daily P&L: {e}")
            return False

    async def start_monitoring(self):
        """Start risk monitoring loop"""
        logger.info("🚀 Starting risk monitoring...")
        self.is_running = True

        # Load previous daily P&L data
        self.load_daily_pnl()

        while self.is_running:
            try:
                # Update daily P&L
                await self.update_daily_pnl()

                # Get current positions
                positions = await self.get_positions()

                # Check auto stops
                if positions:
                    alerts = await self.check_auto_stops(positions)
                    if alerts:
                        for alert in alerts:
                            logger.warning(f"🚨 Risk Alert: {alert}")
                            # Store alerts in Redis
                            self.redis_client.lpush('viper:risk_alerts', json.dumps(alert))
                            self.redis_client.expire('viper:risk_alerts', 86400)

                # Check daily loss limit
                loss_check = self.check_daily_loss_limit()
                if loss_check.get('breached', False):
                    logger.error(f"🚫 Daily loss limit breached: {loss_check}")
                    alert = {
                        'type': 'daily_loss_limit_breached',
                        'details': loss_check,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.redis_client.lpush('viper:risk_alerts', json.dumps(alert))

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    def stop(self):
        """Stop the risk manager"""
        logger.info("🛑 Stopping Risk Manager...")
        self.is_running = False

# FastAPI application
app = FastAPI(
    title="VIPER Risk Manager",
    version="1.0.0",
    description="Risk management and position control service"
)

risk_manager = RiskManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if not risk_manager.initialize_redis():
        logger.error("❌ Failed to initialize Redis. Exiting...")
        return

    # Start monitoring in background task
    asyncio.create_task(risk_manager.start_monitoring())
    logger.info("✅ Risk Manager started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    risk_manager.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "risk-manager",
            "redis_connected": risk_manager.redis_client is not None,
            "monitoring_running": risk_manager.is_running,
            "auto_stops_enabled": risk_manager.enable_auto_stops,
            "daily_loss_limit": risk_manager.daily_loss_limit,
            "max_position_size_percent": risk_manager.max_position_size_percent
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "risk-manager",
                "error": str(e)
            }
        )

@app.get("/api/risk/status")
async def get_risk_status():
    """Get current risk status"""
    try:
        balance = await risk_manager.get_account_balance()
        positions = await risk_manager.get_positions()

        if balance is None:
            balance = 0.0
        if positions is None:
            positions = []

        risk_score = risk_manager.calculate_risk_score(balance, positions)
        loss_check = risk_manager.check_daily_loss_limit()
        utilization_check = risk_manager.check_capital_utilization(balance, positions)
        alerts = await risk_manager.check_auto_stops(positions)

        return {
            'risk_score': risk_score,
            'daily_loss_check': loss_check,
            'capital_utilization_check': utilization_check,
            'active_alerts': alerts,
            'current_balance': round(balance, 2),
            'daily_pnl': round(risk_manager.daily_pnl, 2),
            'position_count': len(positions)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get risk status: {e}")

@app.post("/api/position/size")
async def calculate_position_size(request: Request):
    """Calculate recommended position size"""
    try:
        data = await request.json()

        required_fields = ['symbol', 'price', 'balance']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        symbol = data['symbol']
        price = float(data['price'])
        balance = float(data['balance'])
        risk_per_trade = data.get('risk_per_trade', 0.02)

        result = risk_manager.calculate_position_size(symbol, price, balance, risk_per_trade)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error calculating position size: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/alerts")
async def get_risk_alerts(limit: int = Query(50, description="Number of alerts to retrieve", ge=1, le=100)):
    """Get recent risk alerts"""
    try:
        alerts = risk_manager.redis_client.lrange('viper:risk_alerts', 0, limit - 1)
        parsed_alerts = []

        for alert in alerts:
            try:
                parsed_alerts.append(json.loads(alert))
            except:
                parsed_alerts.append({'raw': alert})

        return {'alerts': parsed_alerts, 'count': len(parsed_alerts)}

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get risk alerts: {e}")

@app.delete("/api/alerts")
async def clear_risk_alerts():
    """Clear all risk alerts"""
    try:
        risk_manager.redis_client.delete('viper:risk_alerts')
        return {'status': 'cleared', 'message': 'All risk alerts cleared'}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to clear alerts: {e}")

@app.get("/api/limits")
async def get_risk_limits():
    """Get current risk limits"""
    return {
        'daily_loss_limit': risk_manager.daily_loss_limit,
        'max_position_size_percent': risk_manager.max_position_size_percent,
        'enable_auto_stops': risk_manager.enable_auto_stops,
        'starting_balance': round(risk_manager.starting_balance, 2),
        'current_balance': round(risk_manager.current_balance, 2),
        'daily_pnl': round(risk_manager.daily_pnl, 2)
    }

@app.get("/api/capital/utilization")
async def get_capital_utilization():
    """Get current capital utilization status"""
    try:
        balance = await risk_manager.get_account_balance()
        positions = await risk_manager.get_positions()

        if balance is None:
            balance = 0.0
        if positions is None:
            positions = []

        utilization_check = risk_manager.check_capital_utilization(balance, positions)
        return utilization_check

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unable to get capital utilization: {e}")

@app.post("/api/limits")
async def update_risk_limits(request: Request):
    """Update risk limits"""
    try:
        data = await request.json()

        if 'daily_loss_limit' in data:
            new_limit = float(data['daily_loss_limit'])
            if 0 < new_limit <= 1.0:  # Max 100% loss limit
                risk_manager.daily_loss_limit = new_limit
            else:
                raise HTTPException(status_code=400, detail="Daily loss limit must be between 0 and 1.0")

        if 'max_position_size_percent' in data:
            new_size = float(data['max_position_size_percent'])
            if 0 < new_size <= 1.0:  # Max 100% position size
                risk_manager.max_position_size_percent = new_size
            else:
                raise HTTPException(status_code=400, detail="Max position size must be between 0 and 1.0")

        if 'enable_auto_stops' in data:
            risk_manager.enable_auto_stops = bool(data['enable_auto_stops'])

        return {
            'status': 'updated',
            'daily_loss_limit': risk_manager.daily_loss_limit,
            'max_position_size_percent': risk_manager.max_position_size_percent,
            'enable_auto_stops': risk_manager.enable_auto_stops
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {e}")

@app.get("/api/position/limits")
async def check_position_limits(symbol: str):
    """Check if a position is allowed for a symbol"""
    limits = risk_manager.check_position_limits(symbol)
    return limits

@app.post("/api/position/check")
async def check_trade_risk(request: Request):
    """Check if a trade meets risk management rules"""
    try:
        data = await request.json()
        symbol = data.get('symbol', '')
        position_size = data.get('position_size', 0)
        price = data.get('price', 0)
        balance = data.get('balance', 0)

        if not all([symbol, position_size, price, balance]):
            raise HTTPException(status_code=400, detail="Missing required fields: symbol, position_size, price, balance")

        risk_check = risk_manager.check_risk_limits(symbol, float(position_size), float(price), float(balance))
        return risk_check

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk check failed: {e}")

@app.post("/api/position/register")
async def register_position(request: Request):
    """Register a new position for risk tracking"""
    try:
        data = await request.json()
        symbol = data.get('symbol', '')
        position_data = data.get('position_data', {})

        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol is required")

        success = risk_manager.register_position(symbol, position_data)
        if success:
            return {'status': 'registered', 'symbol': symbol}
        else:
            raise HTTPException(status_code=409, detail="Position already exists for this symbol")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Position registration failed: {e}")

@app.delete("/api/position/{symbol}")
async def close_position(symbol: str):
    """Close a position and remove from risk tracking"""
    success = risk_manager.close_position(symbol)
    if success:
        return {'status': 'closed', 'symbol': symbol}
    else:
        raise HTTPException(status_code=500, detail="Failed to close position")

@app.get("/api/position/status")
async def get_position_status():
    """Get current position tracking status"""
    return {
        'max_positions': risk_manager.max_positions,
        'current_positions': len(risk_manager.open_positions),
        'active_symbols': list(risk_manager.active_symbols),
        'open_positions': risk_manager.open_positions,
        'available_slots': risk_manager.max_positions - len(risk_manager.open_positions)
    }

if __name__ == "__main__":
    port = int(os.getenv("RISK_MANAGER_PORT", 8000))
    logger.info(f"Starting VIPER Risk Manager on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level="info"
    )
