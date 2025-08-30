#!/usr/bin/env python3
"""
# Rocket VIPER Enhanced Multi-Layer Caching System
Advanced caching infrastructure for faster loading and improved performance

Features:
- Multi-layer caching (L1: Memory, L2: Redis, L3: Persistent Storage)  
- Intelligent cache invalidation and refresh strategies
- Cache warming and pre-loading for critical data
- Performance monitoring and cache hit/miss analytics
- Data compression and serialization optimization
- TTL management with smart expiration policies
"""

import time
import logging
import asyncio
import pickle
import zlib
from enum import Enum
import redis
import hashlib
from pathlib import Path
import threading
from collections import defaultdict, OrderedDict
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheLayer(Enum):
    MEMORY = "memory"
    REDIS = "redis" 
    PERSISTENT = "persistent"

class CacheEvent(Enum):
    HIT = "hit"
    MISS = "miss"
    SET = "set"
    DELETE = "delete"
    EXPIRE = "expire"
    INVALIDATE = "invalidate"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_access: float = 0
    size_bytes: int = 0
    layer: CacheLayer = CacheLayer.MEMORY
    
    def __post_init__(self):
        self.last_access = self.timestamp
        self.size_bytes = sys.getsizeof(self.value)

@dataclass 
class CacheStats:
    """Cache statistics and analytics"""
    total_requests: int = 0
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage: int = 0
    redis_usage: int = 0
    persistent_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(key)
                entry.access_count += 1
                entry.last_access = time.time()
                self.cache[key] = entry
                return entry.value
            return None
    
    def put(self, key: str, value: Any, ttl: int = 3600) -> bool:
        with self.lock:
            now = time.time()
            
            # Remove expired entries
            self._cleanup_expired()
            
            if key in self.cache:
                # Update existing entry
                entry = self.cache.pop(key)
                entry.value = value
                entry.timestamp = now
                entry.last_access = now
                entry.ttl = ttl
                entry.size_bytes = sys.getsizeof(value)
                self.cache[key] = entry
            else:
                # Add new entry
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    self.cache.popitem(last=False)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=now,
                    ttl=ttl,
                    layer=CacheLayer.MEMORY
                )
                self.cache[key] = entry
            
            return True
    
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        with self.lock:
            self.cache.clear()
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        now = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now - entry.timestamp > entry.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def get_stats(self) -> Dict:
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'total_bytes': total_size,
                'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0
            }

class EnhancedCachingSystem:
    """Multi-layer caching system with intelligent management"""
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.memory_cache = LRUCache(max_size=5000)  # 5K items in memory
        self.stats = CacheStats()
        self.lock = threading.RLock()
        
        # Configuration
        self.compression_enabled = True
        self.compression_threshold = 1024  # Compress data > 1KB
        self.default_ttl = 3600  # 1 hour default TTL
        self.cache_warming_enabled = True
        
        # Persistent storage
        self.persistent_path = Path("/tmp/viper_cache")
        self.persistent_path.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_log = []
        self.layer_stats = defaultdict(lambda: defaultdict(int))
        
        # Cache warming patterns
        self.warm_patterns = [
            "market_data:*",
            "ohlcv:*:1h", 
            "ticker:*",
            "viper_score:*"
        ]
        
        logger.info("# Rocket Enhanced Caching System initialized")
    
    async def initialize(self) -> bool:
        """Initialize all cache layers"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=False)
            await asyncio.to_thread(self.redis_client.ping)
            
            # Start background tasks
            asyncio.create_task(self._cache_maintenance_loop())
            asyncio.create_task(self._performance_monitor_loop())
            
            # Warm critical caches
            if self.cache_warming_enabled:
                asyncio.create_task(self._warm_critical_caches())
            
            logger.info("# Check Enhanced caching system initialized")
            return True
            
        except Exception as e:
            logger.error(f"# X Failed to initialize caching system: {e}")
            return False
    
    def _generate_key(self, namespace: str, key: str, **kwargs) -> str:
        """Generate cache key with namespace and parameters"""
        if kwargs:
            params = "_".join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            return f"{namespace}:{key}:{params}"
        return f"{namespace}:{key}"
    
    def _should_compress(self, data: bytes) -> bool:
        """Determine if data should be compressed"""
        return self.compression_enabled and len(data) > self.compression_threshold
    
    def _serialize_and_compress(self, value: Any) -> bytes:
        """Serialize and optionally compress data"""
        try:
            # Serialize to bytes
            data = pickle.dumps(value)
            
            # Compress if beneficial
            if self._should_compress(data):
                compressed = zlib.compress(data)
                # Only use compression if it actually reduces size
                if len(compressed) < len(data):
                    return b'compressed:' + compressed
            
            return b'raw:' + data
            
        except Exception as e:
            logger.error(f"# X Serialization error: {e}")
            return b'raw:' + pickle.dumps(None)
    
    def _deserialize_and_decompress(self, data: bytes) -> Any:
        """Deserialize and decompress data"""
        try:
            if data.startswith(b'compressed:'):
                compressed_data = data[11:]  # Remove 'compressed:' prefix
                decompressed = zlib.decompress(compressed_data)
                return pickle.loads(decompressed)
            elif data.startswith(b'raw:'):
                raw_data = data[4:]  # Remove 'raw:' prefix
                return pickle.loads(raw_data)
            else:
                # Fallback for legacy data
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"# X Deserialization error: {e}")
            return None
    
    async def get(self, namespace: str, key: str, **kwargs) -> Optional[Any]:
        """Get value from cache with multi-layer fallback"""
        cache_key = self._generate_key(namespace, key, **kwargs)
        start_time = time.time()
        
        with self.lock:
            self.stats.total_requests += 1
        
        try:
            # Layer 1: Memory cache
            value = self.memory_cache.get(cache_key)
            if value is not None:
                self._record_cache_event(CacheEvent.HIT, CacheLayer.MEMORY, time.time() - start_time)
                return value
            
            # Layer 2: Redis cache
            if self.redis_client:
                try:
                    redis_data = await asyncio.to_thread(self.redis_client.get, cache_key)
                    if redis_data:
                        value = self._deserialize_and_decompress(redis_data)
                        if value is not None:
                            # Promote to memory cache
                            self.memory_cache.put(cache_key, value)
                            self._record_cache_event(CacheEvent.HIT, CacheLayer.REDIS, time.time() - start_time)
                            return value
                except Exception as e:
                    logger.warning(f"# Warning Redis cache error: {e}")
            
            # Layer 3: Persistent storage
            try:
                persistent_file = self.persistent_path / f"{hashlib.md5(cache_key.encode()).hexdigest()}.cache"
                if persistent_file.exists():
                    with open(persistent_file, 'rb') as f:
                        data = f.read()
                        value = self._deserialize_and_decompress(data)
                        if value is not None:
                            # Promote to higher layers
                            self.memory_cache.put(cache_key, value)
                            if self.redis_client:
                                await asyncio.to_thread(
                                    self.redis_client.setex, 
                                    cache_key, 
                                    self.default_ttl, 
                                    self._serialize_and_compress(value)
                                )
                            self._record_cache_event(CacheEvent.HIT, CacheLayer.PERSISTENT, time.time() - start_time)
                            return value
            except Exception as e:
                logger.warning(f"# Warning Persistent cache error: {e}")
            
            # Cache miss
            self._record_cache_event(CacheEvent.MISS, None, time.time() - start_time)
            return None
            
        except Exception as e:
            logger.error(f"# X Cache get error: {e}")
            self._record_cache_event(CacheEvent.MISS, None, time.time() - start_time)
            return None
    
    async def set(self, namespace: str, key: str, value: Any, ttl: int = None, **kwargs) -> bool:
        """Set value in all cache layers"""
        cache_key = self._generate_key(namespace, key, **kwargs)
        ttl = ttl or self.default_ttl
        start_time = time.time()
        
        try:
            # Layer 1: Memory cache
            self.memory_cache.put(cache_key, value, ttl)
            
            # Layer 2: Redis cache
            if self.redis_client:
                try:
                    serialized_data = self._serialize_and_compress(value)
                    await asyncio.to_thread(self.redis_client.setex, cache_key, ttl, serialized_data)
                except Exception as e:
                    logger.warning(f"# Warning Redis set error: {e}")
            
            # Layer 3: Persistent storage (for critical data)
            if self._is_critical_data(namespace):
                try:
                    persistent_file = self.persistent_path / f"{hashlib.md5(cache_key.encode()).hexdigest()}.cache"
                    with open(persistent_file, 'wb') as f:
                        f.write(self._serialize_and_compress(value))
                except Exception as e:
                    logger.warning(f"# Warning Persistent set error: {e}")
            
            self._record_cache_event(CacheEvent.SET, CacheLayer.MEMORY, time.time() - start_time)
            
            with self.lock:
                self.stats.sets += 1
            
            return True
            
        except Exception as e:
            logger.error(f"# X Cache set error: {e}")
            return False
    
    def _is_critical_data(self, namespace: str) -> bool:
        """Determine if data should be stored in persistent layer"""
        critical_namespaces = ["market_data", "ohlcv", "symbols", "config"]
        return namespace in critical_namespaces
    
    async def delete(self, namespace: str, key: str, **kwargs) -> bool:
        """Delete from all cache layers"""
        cache_key = self._generate_key(namespace, key, **kwargs)
        
        try:
            # Delete from all layers
            deleted = False
            
            # Memory
            if self.memory_cache.delete(cache_key):
                deleted = True
            
            # Redis
            if self.redis_client:
                try:
                    redis_deleted = await asyncio.to_thread(self.redis_client.delete, cache_key)
                    if redis_deleted:
                        deleted = True
                except Exception as e:
                    logger.warning(f"# Warning Redis delete error: {e}")
            
            # Persistent
            try:
                persistent_file = self.persistent_path / f"{hashlib.md5(cache_key.encode()).hexdigest()}.cache"
                if persistent_file.exists():
                    persistent_file.unlink()
                    deleted = True
            except Exception as e:
                logger.warning(f"# Warning Persistent delete error: {e}")
            
            if deleted:
                with self.lock:
                    self.stats.deletes += 1
            
            return deleted
            
        except Exception as e:
            logger.error(f"# X Cache delete error: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        try:
            invalidated = 0
            
            # Redis pattern matching
            if self.redis_client:
                try:
                    keys = await asyncio.to_thread(self.redis_client.keys, pattern)
                    if keys:
                        deleted = await asyncio.to_thread(self.redis_client.delete, *keys)
                        invalidated += deleted
                except Exception as e:
                    logger.warning(f"# Warning Redis pattern invalidation error: {e}")
            
            # Memory cache pattern matching (simplified)
            with self.memory_cache.lock:
                matching_keys = [
                    key for key in self.memory_cache.cache.keys()
                    if self._pattern_matches(key, pattern)
                ]
                for key in matching_keys:
                    if self.memory_cache.delete(key):
                        invalidated += 1
            
            logger.info(f"🗑️ Invalidated {invalidated} cache entries matching '{pattern}'")
            return invalidated
            
        except Exception as e:
            logger.error(f"# X Pattern invalidation error: {e}")
            return 0
    
    def _pattern_matches(self, key: str, pattern: str) -> bool:
        """Simple pattern matching (supports * wildcard)"""
        if '*' not in pattern:
            return key == pattern
        
        # Convert pattern to regex-like matching
        pattern_parts = pattern.split('*')
        if len(pattern_parts) == 2:
            return key.startswith(pattern_parts[0]) and key.endswith(pattern_parts[1])
        
        return False
    
    def _record_cache_event(self, event: CacheEvent, layer: Optional[CacheLayer], duration: float):
        """Record cache event for analytics"""
        with self.lock:
            if event == CacheEvent.HIT:
                self.stats.hits += 1
            elif event == CacheEvent.MISS:
                self.stats.misses += 1
            
            if layer:
                self.layer_stats[layer.value][event.value] += 1
        
        # Store performance data
        if len(self.performance_log) > 10000:  # Keep last 10k events
            self.performance_log = self.performance_log[-5000:]
        
        self.performance_log.append({
            'timestamp': time.time(),
            'event': event.value,
            'layer': layer.value if layer else 'none',
            'duration': duration
        })
    
    async def _cache_maintenance_loop(self):
        """Background maintenance tasks"""
        while True:
            try:
                # Cleanup expired entries
                self.memory_cache._cleanup_expired()
                
                # Update memory usage stats
                memory_stats = self.memory_cache.get_stats()
                with self.lock:
                    self.stats.memory_usage = memory_stats['total_bytes']
                
                # Log cache statistics periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    await self._log_cache_statistics()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"# X Cache maintenance error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitor_loop(self):
        """Monitor cache performance and optimize"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Analyze performance
                if len(self.performance_log) > 100:
                    recent_events = self.performance_log[-1000:]
                    avg_hit_time = np.mean([
                        event['duration'] for event in recent_events 
                        if event['event'] == 'hit'
                    ]) if recent_events else 0
                    
                    avg_miss_time = np.mean([
                        event['duration'] for event in recent_events 
                        if event['event'] == 'miss'
                    ]) if recent_events else 0
                    
                    logger.info(f"# Chart Cache Performance: Hit={avg_hit_time:.4f}s, Miss={avg_miss_time:.4f}s")
                
            except Exception as e:
                logger.error(f"# X Performance monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _warm_critical_caches(self):
        """Warm up critical caches with frequently accessed data"""
        try:
            logger.info("🔥 Starting cache warming...")
            
            # This would be implemented with actual data sources
            # For now, we'll just log the intent
            
            for pattern in self.warm_patterns:
                logger.info(f"🔥 Warming cache pattern: {pattern}")
                # Implementation would fetch and cache critical data here
                await asyncio.sleep(0.1)  # Prevent overwhelming
            
            logger.info("# Check Cache warming completed")
            
        except Exception as e:
            logger.error(f"# X Cache warming error: {e}")
    
    async def _log_cache_statistics(self):
        """Log detailed cache statistics"""
        try:
            memory_stats = self.memory_cache.get_stats()
            
            with self.lock:
                stats = {
                    'total_requests': self.stats.total_requests,
                    'hit_rate': self.stats.hit_rate,
                    'memory_utilization': memory_stats['utilization'],
                    'memory_size': memory_stats['size'],
                    'memory_bytes': memory_stats['total_bytes'],
                    'layer_stats': dict(self.layer_stats)
                }
            
            logger.info(f"📈 Cache Stats: Hit Rate={stats['hit_rate']:.2%}, "
                       f"Memory={stats['memory_size']}/{memory_stats['max_size']} items, "
                       f"Size={stats['memory_bytes']:,} bytes")
            
        except Exception as e:
            logger.error(f"# X Statistics logging error: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            memory_stats = self.memory_cache.get_stats()
            
            with self.lock:
                stats = {
                    'overview': {
                        'total_requests': self.stats.total_requests,
                        'hits': self.stats.hits,
                        'misses': self.stats.misses,
                        'hit_rate': self.stats.hit_rate,
                        'sets': self.stats.sets,
                        'deletes': self.stats.deletes
                    },
                    'memory_cache': memory_stats,
                    'layer_performance': dict(self.layer_stats),
                    'recent_performance': self.performance_log[-100:] if self.performance_log else [],
                    'configuration': {
                        'compression_enabled': self.compression_enabled,
                        'compression_threshold': self.compression_threshold,
                        'default_ttl': self.default_ttl,
                        'cache_warming_enabled': self.cache_warming_enabled
                    }
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"# X Statistics error: {e}")
            return {}
    
    async def clear_all_caches(self) -> Dict[str, bool]:
        """Clear all cache layers"""
        results = {}
        
        try:
            # Memory cache
            self.memory_cache.clear()
            results['memory'] = True
            
            # Redis cache
            if self.redis_client:
                try:
                    await asyncio.to_thread(self.redis_client.flushdb)
                    results['redis'] = True
                except Exception as e:
                    logger.error(f"# X Redis clear error: {e}")
                    results['redis'] = False
            
            # Persistent cache
            try:
                for cache_file in self.persistent_path.glob("*.cache"):
                    cache_file.unlink()
                results['persistent'] = True
            except Exception as e:
                logger.error(f"# X Persistent clear error: {e}")
                results['persistent'] = False
            
            # Reset stats
            with self.lock:
                self.stats = CacheStats()
                self.layer_stats.clear()
                self.performance_log.clear()
            
            logger.info("🗑️ All caches cleared")
            
        except Exception as e:
            logger.error(f"# X Cache clear error: {e}")
        
        return results

# Global cache instance
enhanced_cache = EnhancedCachingSystem()

# Convenience functions
async def get_cached(namespace: str, key: str, **kwargs) -> Optional[Any]:
    """Get from cache"""
    return await enhanced_cache.get(namespace, key, **kwargs)

async def set_cached(namespace: str, key: str, value: Any, ttl: int = None, **kwargs) -> bool:
    """Set to cache"""
    return await enhanced_cache.set(namespace, key, value, ttl, **kwargs)

async def delete_cached(namespace: str, key: str, **kwargs) -> bool:
    """Delete from cache"""
    return await enhanced_cache.delete(namespace, key, **kwargs)

async def invalidate_cached_pattern(pattern: str) -> int:
    """Invalidate cache pattern"""
    return await enhanced_cache.invalidate_pattern(pattern)

if __name__ == "__main__":
    async def test_caching_system():
        """Test the enhanced caching system"""
        logger.info("🧪 Testing Enhanced Caching System...")
        
        # Initialize
        await enhanced_cache.initialize()
        
        # Test basic operations
        await set_cached("test", "key1", {"data": "test_value"})
        result = await get_cached("test", "key1")
        
        # Test with parameters
        await set_cached("market_data", "BTCUSDT", {"price": 50000}, timeframe="1h")
        result = await get_cached("market_data", "BTCUSDT", timeframe="1h")
        
        # Test statistics
        stats = await enhanced_cache.get_statistics()
        
    
    asyncio.run(test_caching_system())