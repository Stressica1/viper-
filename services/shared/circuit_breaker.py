#!/usr/bin/env python3
"""
# Rocket VIPER Trading Bot - Circuit Breaker Utility
Implements circuit breaker patterns and retry logic for microservice communication

Features:
    pass
- Circuit breaker pattern implementation
- Exponential backoff retry logic
- Service health monitoring
- Graceful failure handling
- Metrics collection for failures
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple
import httpx
import redis
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)"""

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """
    Circuit breaker implementation for microservice communication
    """"""

    def __init__()
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception,
        redis_client: Optional[redis.Redis] = None
(    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0

        # Redis for distributed state (optional)
        self.redis_client = redis_client
        self.redis_key = f"viper:circuit_breaker:{service_name}"

        logger.info(f"# Rocket Circuit breaker initialized for {service_name}")

    def _load_state_from_redis(self) -> None:
        """Load circuit breaker state from Redis""""""
        if not self.redis_client:
            return

        try:
            state_data = self.redis_client.get(self.redis_key)
            if state_data:
                data = json.loads(state_data)
                self.state = CircuitState(data['state'])
                self.failure_count = data['failure_count']
                self.last_failure_time = data.get('last_failure_time')
                self.success_count = data.get('success_count', 0)
        except Exception as e:
            logger.warning(f"Failed to load circuit breaker state from Redis: {e}")

    def _save_state_to_redis(self) -> None:
        """Save circuit breaker state to Redis""""""
        if not self.redis_client:
            return

        try:
            state_data = {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'last_failure_time': self.last_failure_time,
                'success_count': self.success_count,
                'timestamp': datetime.now().isoformat()
            }
            self.redis_client.setex()
                self.redis_key,
                300,  # 5 minutes TTL
                json.dumps(state_data)
(            )
        except Exception as e:
            logger.warning(f"Failed to save circuit breaker state to Redis: {e}")

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit""""""
        if self.state != CircuitState.OPEN:
            return False

        if self.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout

    def _reset(self) -> None:
        """Reset circuit breaker to half-open state"""
        logger.info(f"🔄 Resetting circuit breaker for {self.service_name} to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self._save_state_to_redis()

    def _record_success(self) -> None:
        """Record successful operation""""""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 consecutive successes
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"# Check Circuit breaker for {self.service_name} CLOSED (service recovered)")
            self._save_state_to_redis()

    def _record_failure(self) -> None:
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()"""

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"🚫 Circuit breaker for {self.service_name} OPENED (too many failures)")

        self._save_state_to_redis()

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        self._load_state_from_redis()"""

        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._reset()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True

        return False

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection""""""
        if not self.can_execute():
            raise Exception(f"Circuit breaker is OPEN for {self.service_name}")

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure()
            logger.warning(f"Service call failed for {self.service_name}: {e}")
            raise e
        except Exception as e:
            # For unexpected exceptions, record failure but don't change state
            logger.error(f"Unexpected error in {self.service_name}: {e}")
            raise e

class RetryLogic:
    """
    Exponential backoff retry logic
    """"""

    def __init__()
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
(    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for current attempt"""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)"""

        if self.jitter:
            # Add random jitter (±25%)
            import random
            import secrets
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.1, delay)  # Minimum 100ms delay

    async def execute_with_retry():
        self,
        func: Callable,
        *args,
        **kwargs
(    ) -> Any:
        """Execute function with retry logic"""
        last_exception = None

        for attempt in range(self.max_retries + 1)""":
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt - 1)
                    logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt}/{self.max_retries})")
                    await asyncio.sleep(delay)

                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"# Check Retry successful for {func.__name__}")
                return result

            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.error(f"# X All {self.max_retries + 1} attempts failed for {func.__name__}: {e}")
                    raise e
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")

        raise last_exception

class ServiceClient:
    """
    HTTP client with circuit breaker and retry logic
    """"""

    def __init__()
        self,
        service_name: str,
        base_url: str,
        redis_client: Optional[redis.Redis] = None,
        timeout: float = 30.0
(    ):
        self.service_name = service_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

        # Circuit breaker and retry logic
        self.circuit_breaker = CircuitBreaker(service_name, redis_client=redis_client)
        self.retry_logic = RetryLogic(max_retries=3)

        # HTTP client
        self._client = None

    async def __aenter__(self):
        """Async context manager entry"""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit""""""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request with circuit breaker protection""""""
        if not self._client:
            raise Exception("ServiceClient must be used as async context manager")

        url = f"{self.base_url}{endpoint}"

        async def _http_request():
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

        # Execute with circuit breaker and retry logic
        return await self.circuit_breaker.call()
            self.retry_logic.execute_with_retry,
            _http_request
(        )

    async def get(self, endpoint: str, **kwargs) -> Dict:
        """GET request"""
        return await self._make_request("GET", endpoint, **kwargs)

    async def post(self, endpoint: str, **kwargs) -> Dict:
        """POST request"""
        return await self._make_request("POST", endpoint, **kwargs)

    async def put(self, endpoint: str, **kwargs) -> Dict:
        """PUT request"""
        return await self._make_request("PUT", endpoint, **kwargs)

    async def delete(self, endpoint: str, **kwargs) -> Dict:
        """DELETE request"""
        return await self._make_request("DELETE", endpoint, **kwargs)

    def get_health(self) -> Dict:
        """Get circuit breaker health status"""
        return {
            'service_name': self.service_name,
            'state': self.circuit_breaker.state.value,
            'failure_count': self.circuit_breaker.failure_count,
            'last_failure_time': self.circuit_breaker.last_failure_time,
            'success_count': self.circuit_breaker.success_count
        }

# Global service registry
_service_clients = {}"""

def get_service_client()
    service_name: str,
    base_url: str,
    redis_client: Optional[redis.Redis] = None
() -> ServiceClient:
    """Get or create service client instance"""
    key = f"{service_name}:{base_url}"

    if key not in _service_clients:
        _service_clients[key] = ServiceClient(service_name, base_url, redis_client)

    return _service_clients[key]

async def call_service():
    service_name: str,
    endpoint: str,
    method: str = "GET",
    redis_client: Optional[redis.Redis] = None,
    **kwargs
() -> Dict:
    """
    Convenience function to call a service with circuit breaker protection

    Example:
        result = await call_service()
            "data-manager",
            "/api/ticker/BTCUSDT",
            redis_client=redis_client
(        )
    """
    # Get service URL from environment
    base_url = os.getenv(f"{service_name.upper()}_URL", f"http://{service_name}:8000")

    client = get_service_client(service_name, base_url, redis_client)

    async with client:
        if method.upper() == "GET":
            return await client.get(endpoint, **kwargs)
        elif method.upper() == "POST":
            return await client.post(endpoint, **kwargs)
        elif method.upper() == "PUT":
            return await client.put(endpoint, **kwargs)
        elif method.upper() == "DELETE":
            return await client.delete(endpoint, **kwargs)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

# Example usage in a service
async def example_service_call():
    """Example of how to use the circuit breaker in a service""""""
    try:
        # Call data-manager service
        result = await call_service()
            "data-manager",
            "/api/ticker/BTCUSDT",
            method="GET"
(        )
        logger.info(f"Successfully called data-manager: {result}")
        return result

    except Exception as e:
        logger.error(f"Service call failed: {e}")
        # Circuit breaker will automatically handle failures
        raise e

if __name__ == "__main__":
    # Example usage
    async def main():
        try:
            result = await example_service_call()
        except Exception as e:
            pass

    asyncio.run(main())
