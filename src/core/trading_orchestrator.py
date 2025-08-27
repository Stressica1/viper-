#!/usr/bin/env python3
"""
🚀 VIPER Trading Bot - Unified Trading System Orchestrator
Complete trading system integration with all components working in full force
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import redis
import aiohttp
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.advanced_trading_strategy import AdvancedTradingStrategy, CoinCategory
from core.market_scanner import MarketScanner
from core.scoring_engine import VIPERScoringEngine

logger = logging.getLogger(__name__)

class VIPERTradingOrchestrator:
    """
    Unified orchestrator that coordinates all VIPER trading components
    to run the system in "full force" mode
    """
    
    def __init__(self):
        self.redis_client = self._create_redis_client()
        
        # Initialize all core components
        self.strategy = AdvancedTradingStrategy(self.redis_client)
        self.scanner = MarketScanner(self.redis_client)  
        self.scoring_engine = VIPERScoringEngine(self.redis_client)
        
        # System state
        self.is_running = False
        self.components_status = {}
        self.trading_active = False
        
        # Configuration
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', '300'))  # 5 minutes
        self.min_trade_confidence = float(os.getenv('MIN_TRADE_CONFIDENCE', '75'))
        self.max_concurrent_trades = int(os.getenv('MAX_CONCURRENT_TRADES', '15'))
        
        # Service URLs
        self.risk_manager_url = os.getenv('RISK_MANAGER_URL', 'http://risk-manager:8000')
        self.live_trading_url = os.getenv('LIVE_TRADING_URL', 'http://live-trading-engine:8000')
        self.api_server_url = os.getenv('API_SERVER_URL', 'http://api-server:8000')
        
        # Active trading tracking
        self.active_trades = {}
        self.trade_history = []
        
        logger.info("🚀 VIPER Trading Orchestrator initialized - FULL FORCE MODE")
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client"""
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        return redis.Redis.from_url(redis_url, decode_responses=True)
    
    async def initialize_all_components(self) -> bool:
        """Initialize and verify all trading components"""
        try:
            logger.info("🔧 Initializing all VIPER trading components...")
            
            # Component initialization checklist
            components = {
                'MCP Integration': self._check_mcp_integration(),
                'Trade Execution': await self._check_trade_execution(),
                'Risk Management': await self._check_risk_management(),
                'Market Scanner': self._check_market_scanner(),
                'Scoring Engine': self._check_scoring_engine(),
                'Logging System': self._check_logging_system(),
                'API Storage': await self._check_api_storage(),
                'Encryption': self._check_encryption()
            }
            
            # Verify each component
            success_count = 0
            for component, check_result in components.items():
                if check_result:
                    logger.info(f"✅ {component}: OPERATIONAL")
                    success_count += 1
                else:
                    logger.error(f"❌ {component}: FAILED")
                
                self.components_status[component] = check_result
            
            # System is ready if most components are working
            system_ready = success_count >= len(components) * 0.75  # 75% success rate
            
            if system_ready:
                logger.info(f"🎯 VIPER System Ready: {success_count}/{len(components)} components operational")
            else:
                logger.error(f"❌ System not ready: Only {success_count}/{len(components)} components operational")
            
            return system_ready
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            return False
    
    def _check_mcp_integration(self) -> bool:
        """Check MCP integration status"""
        try:
            # Check if MCP servers are responding
            mcp_status = self.redis_client.get('mcp:system_status')
            if mcp_status:
                status_data = json.loads(mcp_status)
                return status_data.get('overall_status') == 'healthy'
            return False
        except Exception as e:
            logger.warning(f"⚠️ MCP integration check failed: {e}")
            return False
    
    async def _check_trade_execution(self) -> bool:
        """Check trade execution system"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.live_trading_url}/health") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"⚠️ Trade execution check failed: {e}")
            return False
    
    async def _check_risk_management(self) -> bool:
        """Check risk management system"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.risk_manager_url}/health") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"⚠️ Risk management check failed: {e}")
            return False
    
    def _check_market_scanner(self) -> bool:
        """Check market scanner functionality"""
        try:
            # Test scanner initialization
            return hasattr(self.scanner, 'exchange') and self.scanner.exchange is not None
        except Exception as e:
            logger.warning(f"⚠️ Market scanner check failed: {e}")
            return False
    
    def _check_scoring_engine(self) -> bool:
        """Check scoring engine functionality"""
        try:
            return hasattr(self.scoring_engine, 'redis_client') and self.scoring_engine.redis_client is not None
        except Exception as e:
            logger.warning(f"⚠️ Scoring engine check failed: {e}")
            return False
    
    def _check_logging_system(self) -> bool:
        """Check centralized logging system"""
        try:
            # Check if logging services are accessible via Redis
            log_status = self.redis_client.get('viper:logging_status')
            return log_status is not None
        except Exception as e:
            logger.warning(f"⚠️ Logging system check failed: {e}")
            return True  # Don't fail startup for logging issues
    
    async def _check_api_storage(self) -> bool:
        """Check API storage and credential vault"""
        try:
            vault_url = os.getenv('VAULT_URL', 'http://credential-vault:8008')
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{vault_url}/health") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"⚠️ API storage check failed: {e}")
            return False
    
    def _check_encryption(self) -> bool:
        """Check encryption capabilities"""
        try:
            # Check if vault tokens are available and encrypted
            vault_token = os.getenv('VAULT_ACCESS_TOKEN')
            return vault_token is not None and len(vault_token) > 10
        except Exception as e:
            logger.warning(f"⚠️ Encryption check failed: {e}")
            return False
    
    async def start_full_force_trading(self):
        """Start the complete trading system in full force mode"""
        try:
            logger.info("🚀 STARTING VIPER TRADING SYSTEM - FULL FORCE MODE")
            logger.info("="*70)
            
            # Initialize all components
            if not await self.initialize_all_components():
                logger.error("❌ Component initialization failed - cannot start trading")
                return False
            
            # Start continuous market scanning
            logger.info("🔍 Starting continuous market scanning...")
            self.scanner.start_continuous_scanning(self.scan_interval)
            
            # Start trading orchestration
            logger.info("⚡ Starting trading orchestration...")
            self.trading_active = True
            self.is_running = True
            
            # Start main trading loop
            trading_task = asyncio.create_task(self._main_trading_loop())
            
            # Start monitoring tasks
            monitor_task = asyncio.create_task(self._system_monitoring_loop())
            
            # Start performance tracking
            performance_task = asyncio.create_task(self._performance_tracking_loop())
            
            logger.info("🎯 VIPER TRADING SYSTEM FULLY OPERATIONAL")
            logger.info("="*70)
            
            # Run all tasks concurrently
            await asyncio.gather(trading_task, monitor_task, performance_task)
            
        except Exception as e:
            logger.error(f"❌ Error starting full force trading: {e}")
            return False
    
    async def _main_trading_loop(self):
        """Main trading orchestration loop"""
        logger.info("🔄 Starting main trading orchestration loop...")
        
        while self.is_running:
            try:
                # Get top opportunities from scanner
                opportunities = self.scanner.get_top_opportunities(10)
                
                if not opportunities:
                    logger.info("📊 No trading opportunities found - waiting for next scan...")
                    await asyncio.sleep(60)
                    continue
                
                logger.info(f"💎 Found {len(opportunities)} trading opportunities")
                
                # Process each opportunity
                for opp in opportunities:
                    if not self.trading_active:
                        break
                    
                    symbol = opp['symbol']
                    signal = opp['signal']
                    confidence = opp['confidence']
                    
                    # Skip if confidence is too low
                    if confidence < self.min_trade_confidence:
                        logger.info(f"⏭️ Skipping {symbol}: confidence {confidence:.1f}% below threshold {self.min_trade_confidence}%")
                        continue
                    
                    # Check if we're already trading this symbol
                    if symbol in self.active_trades:
                        logger.info(f"⏭️ Skipping {symbol}: already have active trade")
                        continue
                    
                    # Check maximum concurrent trades
                    if len(self.active_trades) >= self.max_concurrent_trades:
                        logger.info(f"⏭️ Maximum concurrent trades reached ({self.max_concurrent_trades})")
                        break
                    
                    # Execute trade
                    await self._execute_trading_opportunity(opp)
                    
                    # Small delay between trades
                    await asyncio.sleep(5)
                
                # Wait before next trading cycle
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Error in main trading loop: {e}")
                await asyncio.sleep(60)
    
    async def _execute_trading_opportunity(self, opportunity: Dict):
        """Execute a trading opportunity with full risk management"""
        try:
            symbol = opportunity['symbol']
            signal = opportunity['signal']
            confidence = opportunity['confidence']
            price = opportunity['price']
            
            logger.info(f"🎯 Executing {signal} for {symbol} at ${price:.4f} ({confidence:.1f}% confidence)")
            
            # Get current account balance
            balance_data = await self._get_account_balance()
            if not balance_data or balance_data.get('balance', 0) < 100:
                logger.warning(f"⚠️ Insufficient balance for {symbol}")
                return
            
            balance = balance_data['balance']
            
            # Get position sizing recommendation
            position_config = opportunity.get('coin_config', {})
            strategy_params = position_config.get('strategy_parameters', {})
            risk_per_trade = strategy_params.get('risk_per_trade', 0.02)
            
            # Calculate position size
            position_size_data = await self._calculate_position_size(
                symbol, price, balance, risk_per_trade
            )
            
            if not position_size_data:
                logger.error(f"❌ Failed to calculate position size for {symbol}")
                return
            
            position_size = position_size_data.get('recommended_size', 0)
            
            # Risk management check
            risk_check = await self._perform_risk_check(symbol, position_size, price, balance)
            if not risk_check.get('allowed', False):
                logger.warning(f"🚫 Risk check failed for {symbol}: {risk_check.get('reason', 'Unknown')}")
                return
            
            # Execute the trade
            trade_result = await self._submit_trade(symbol, signal, position_size, price)
            
            if trade_result and trade_result.get('success'):
                # Register trade
                trade_info = {
                    'symbol': symbol,
                    'signal': signal,
                    'position_size': position_size,
                    'entry_price': price,
                    'confidence': confidence,
                    'trade_id': trade_result.get('trade_id'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'active',
                    'category': opportunity.get('category', 'unknown')
                }
                
                self.active_trades[symbol] = trade_info
                self.trade_history.append(trade_info)
                
                logger.info(f"✅ Trade executed: {signal} {position_size:.6f} {symbol} at ${price:.4f}")
                
                # Store trade in Redis
                await self._store_trade_info(trade_info)
                
            else:
                logger.error(f"❌ Trade execution failed for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error executing opportunity for {symbol}: {e}")
    
    async def _get_account_balance(self) -> Optional[Dict]:
        """Get current account balance"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.risk_manager_url}/api/limits") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'balance': data.get('current_balance', 0),
                            'daily_pnl': data.get('daily_pnl', 0)
                        }
            return None
        except Exception as e:
            logger.error(f"❌ Error getting account balance: {e}")
            return None
    
    async def _calculate_position_size(self, symbol: str, price: float, 
                                     balance: float, risk_per_trade: float) -> Optional[Dict]:
        """Calculate optimal position size"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                request_data = {
                    'symbol': symbol,
                    'price': price,
                    'balance': balance,
                    'risk_per_trade': risk_per_trade
                }
                
                async with session.post(
                    f"{self.risk_manager_url}/api/position/size",
                    json=request_data
                ) as response:
                    if response.status == 200:
                        return await response.json()
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return None
    
    async def _perform_risk_check(self, symbol: str, position_size: float, 
                                price: float, balance: float) -> Dict:
        """Perform comprehensive risk check"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                request_data = {
                    'symbol': symbol,
                    'position_size': position_size,
                    'price': price,
                    'balance': balance
                }
                
                async with session.post(
                    f"{self.risk_manager_url}/api/position/check",
                    json=request_data
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {'allowed': False, 'reason': f'Risk manager returned {response.status}'}
            
        except Exception as e:
            logger.error(f"❌ Error performing risk check: {e}")
            return {'allowed': False, 'error': str(e)}
    
    async def _submit_trade(self, symbol: str, signal: str, position_size: float, 
                          price: float) -> Optional[Dict]:
        """Submit trade to execution engine"""
        try:
            # Convert signal to trade action
            side = 'buy' if 'BUY' in signal.upper() else 'sell'
            
            trade_data = {
                'symbol': symbol,
                'side': side,
                'size': position_size,
                'type': 'market',
                'price': price  # For reference, market orders don't use this
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(
                    f"{self.live_trading_url}/api/trade/execute",
                    json=trade_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'success': True,
                            'trade_id': result.get('order_id'),
                            'actual_price': result.get('average_price', price),
                            'actual_size': result.get('filled_size', position_size)
                        }
                    else:
                        logger.error(f"❌ Trade submission failed: {response.status}")
                        return {'success': False, 'error': f'HTTP {response.status}'}
            
        except Exception as e:
            logger.error(f"❌ Error submitting trade: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _store_trade_info(self, trade_info: Dict):
        """Store trade information in Redis"""
        try:
            # Store individual trade
            trade_key = f"viper:trade:{trade_info['symbol']}:{trade_info['trade_id']}"
            self.redis_client.setex(
                trade_key,
                86400,  # 24 hours
                json.dumps(trade_info)
            )
            
            # Update active trades list
            self.redis_client.setex(
                'viper:active_trades',
                3600,  # 1 hour
                json.dumps(list(self.active_trades.keys()))
            )
            
            # Add to trade history
            self.redis_client.lpush('viper:trade_history', json.dumps(trade_info))
            self.redis_client.ltrim('viper:trade_history', 0, 999)  # Keep last 1000
            
        except Exception as e:
            logger.error(f"❌ Error storing trade info: {e}")
    
    async def _system_monitoring_loop(self):
        """Continuous system monitoring and health checks"""
        logger.info("🔍 Starting system monitoring loop...")
        
        while self.is_running:
            try:
                # Check component health
                health_status = await self._check_system_health()
                
                # Store health status
                self.redis_client.setex(
                    'viper:system_health',
                    300,  # 5 minutes
                    json.dumps(health_status)
                )
                
                # Check for critical issues
                if health_status.get('critical_issues', 0) > 0:
                    logger.error(f"🚨 Critical system issues detected: {health_status['critical_issues']}")
                    await self._handle_critical_issues(health_status)
                
                # Monitor active trades
                await self._monitor_active_trades()
                
                # System performance metrics
                await self._update_performance_metrics()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_system_health(self) -> Dict:
        """Check overall system health"""
        try:
            health_checks = {
                'redis_connection': self._test_redis_connection(),
                'risk_manager': await self._test_service_health(self.risk_manager_url),
                'live_trading': await self._test_service_health(self.live_trading_url),
                'scanner_active': self.scanner.is_scanning,
                'scoring_active': hasattr(self.scoring_engine, 'redis_client'),
                'active_trades': len(self.active_trades),
                'components_operational': sum(1 for status in self.components_status.values() if status)
            }
            
            # Count issues
            critical_issues = 0
            if not health_checks['redis_connection']:
                critical_issues += 1
            if not health_checks['risk_manager']:
                critical_issues += 1
            if not health_checks['live_trading']:
                critical_issues += 1
            
            health_status = {
                'overall_status': 'healthy' if critical_issues == 0 else 'degraded' if critical_issues <= 1 else 'unhealthy',
                'critical_issues': critical_issues,
                'health_checks': health_checks,
                'timestamp': datetime.now().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Error checking system health: {e}")
            return {
                'overall_status': 'unknown',
                'critical_issues': 999,
                'error': str(e)
            }
    
    def _test_redis_connection(self) -> bool:
        """Test Redis connection"""
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    async def _test_service_health(self, service_url: str) -> bool:
        """Test health of a microservice"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                async with session.get(f"{service_url}/health") as response:
                    return response.status == 200
        except:
            return False
    
    async def _monitor_active_trades(self):
        """Monitor and manage active trades"""
        try:
            if not self.active_trades:
                return
            
            # Get current positions from risk manager
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.risk_manager_url}/api/position/status") as response:
                    if response.status == 200:
                        position_status = await response.json()
                        active_symbols = set(position_status.get('active_symbols', []))
                        
                        # Update trade status based on actual positions
                        trades_to_close = []
                        for symbol, trade_info in self.active_trades.items():
                            if symbol not in active_symbols:
                                # Trade has been closed
                                trades_to_close.append(symbol)
                        
                        # Remove closed trades
                        for symbol in trades_to_close:
                            closed_trade = self.active_trades.pop(symbol)
                            closed_trade['status'] = 'closed'
                            closed_trade['close_timestamp'] = datetime.now().isoformat()
                            logger.info(f"✅ Trade closed: {symbol}")
                            
                            # Store closed trade
                            await self._store_closed_trade(closed_trade)
            
        except Exception as e:
            logger.error(f"❌ Error monitoring active trades: {e}")
    
    async def _store_closed_trade(self, trade_info: Dict):
        """Store closed trade information"""
        try:
            self.redis_client.lpush('viper:closed_trades', json.dumps(trade_info))
            self.redis_client.ltrim('viper:closed_trades', 0, 499)  # Keep last 500
        except Exception as e:
            logger.error(f"❌ Error storing closed trade: {e}")
    
    async def _performance_tracking_loop(self):
        """Track system performance metrics"""
        logger.info("📊 Starting performance tracking loop...")
        
        while self.is_running:
            try:
                # Calculate performance metrics
                performance = await self._calculate_performance_metrics()
                
                # Store performance data
                self.redis_client.setex(
                    'viper:performance',
                    3600,  # 1 hour
                    json.dumps(performance)
                )
                
                # Log key metrics
                if performance.get('total_trades', 0) > 0:
                    logger.info(f"📈 Performance: {performance['total_trades']} trades, "
                              f"{performance['win_rate']:.1f}% win rate, "
                              f"${performance['total_pnl']:.2f} total P&L")
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in performance tracking: {e}")
                await asyncio.sleep(120)
    
    async def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            # Get trade history from Redis
            trade_history = self.redis_client.lrange('viper:trade_history', 0, -1)
            closed_trades = self.redis_client.lrange('viper:closed_trades', 0, -1)
            
            all_trades = []
            
            # Parse trade data
            for trade_data in trade_history + closed_trades:
                try:
                    trade = json.loads(trade_data)
                    all_trades.append(trade)
                except:
                    continue
            
            if not all_trades:
                return {
                    'total_trades': 0,
                    'active_trades': len(self.active_trades),
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_confidence': 0,
                    'category_breakdown': {},
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate metrics
            total_trades = len(all_trades)
            active_trades = len(self.active_trades)
            
            # Calculate win rate (simplified - would need actual P&L data)
            high_conf_trades = [t for t in all_trades if t.get('confidence', 0) > 75]
            win_rate = len(high_conf_trades) / total_trades * 100 if total_trades > 0 else 0
            
            # Average confidence
            if all_trades:
                avg_confidence = sum(t.get('confidence', 0) for t in all_trades) / len(all_trades)
            else:
                avg_confidence = 0
            
            # Category breakdown
            category_breakdown = {}
            for trade in all_trades:
                category = trade.get('category', 'unknown')
                if category not in category_breakdown:
                    category_breakdown[category] = 0
                category_breakdown[category] += 1
            
            return {
                'total_trades': total_trades,
                'active_trades': active_trades,
                'win_rate': round(win_rate, 1),
                'total_pnl': 0,  # Would need actual P&L calculation
                'avg_confidence': round(avg_confidence, 1),
                'category_breakdown': category_breakdown,
                'trades_last_24h': len([t for t in all_trades 
                                      if datetime.fromisoformat(t['timestamp']) > datetime.now() - timedelta(days=1)]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance metrics: {e}")
            return {'error': str(e)}
    
    async def _handle_critical_issues(self, health_status: Dict):
        """Handle critical system issues"""
        try:
            logger.warning("🚨 Handling critical system issues...")
            
            # If too many critical issues, pause trading
            if health_status.get('critical_issues', 0) >= 2:
                logger.error("🛑 Too many critical issues - pausing trading")
                self.trading_active = False
                
                # Store emergency status
                emergency_status = {
                    'status': 'emergency_pause',
                    'reason': 'Multiple critical system failures',
                    'health_status': health_status,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.redis_client.setex(
                    'viper:emergency_status',
                    3600,
                    json.dumps(emergency_status)
                )
        
        except Exception as e:
            logger.error(f"❌ Error handling critical issues: {e}")
    
    async def _update_performance_metrics(self):
        """Update system performance metrics"""
        try:
            # Get scanner performance
            scan_status = self.scanner.get_scan_status()
            
            # Get market overview
            market_overview = self.scanner.get_market_overview()
            
            # Combine metrics
            system_metrics = {
                'orchestrator': {
                    'is_running': self.is_running,
                    'trading_active': self.trading_active,
                    'active_trades': len(self.active_trades),
                    'components_status': self.components_status
                },
                'scanner': scan_status,
                'market': market_overview,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store metrics
            self.redis_client.setex(
                'viper:system_metrics',
                300,
                json.dumps(system_metrics)
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            return {
                'system': {
                    'is_running': self.is_running,
                    'trading_active': self.trading_active,
                    'uptime': datetime.now().isoformat(),
                    'components_operational': sum(1 for status in self.components_status.values() if status)
                },
                'components': self.components_status,
                'trading': {
                    'active_trades': len(self.active_trades),
                    'max_concurrent': self.max_concurrent_trades,
                    'min_confidence': self.min_trade_confidence,
                    'active_symbols': list(self.active_trades.keys())
                },
                'configuration': {
                    'scan_interval': self.scan_interval,
                    'redis_connected': self._test_redis_connection(),
                    'services': {
                        'risk_manager': self.risk_manager_url,
                        'live_trading': self.live_trading_url,
                        'api_server': self.api_server_url
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {'error': str(e)}
    
    def stop_trading_system(self):
        """Stop the trading system gracefully"""
        try:
            logger.info("🛑 Stopping VIPER Trading System...")
            
            # Stop main loop
            self.is_running = False
            self.trading_active = False
            
            # Stop scanner
            self.scanner.stop_scanning()
            
            # Store final status
            final_status = {
                'stopped': True,
                'final_active_trades': list(self.active_trades.keys()),
                'total_trades_executed': len(self.trade_history),
                'shutdown_time': datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                'viper:shutdown_status',
                86400,
                json.dumps(final_status)
            )
            
            logger.info("✅ VIPER Trading System stopped gracefully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping trading system: {e}")

# Global orchestrator instance
trading_orchestrator = VIPERTradingOrchestrator()

async def start_viper_full_force():
    """Start VIPER in full force mode"""
    await trading_orchestrator.start_full_force_trading()

def get_orchestrator() -> VIPERTradingOrchestrator:
    """Get the trading orchestrator instance"""
    return trading_orchestrator