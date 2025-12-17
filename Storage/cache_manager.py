"""
Cache Manager for OSINT AI System
Redis-based caching for events, results, and rate limiting
"""

import logging
import json
import pickle
from typing import Any, Optional, List, Dict, cast, Callable, TypeVar
import redis
from functools import wraps
import hashlib

from Data.settings import Settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages Redis cache for the OSINT AI system"""
    
    def __init__(self):
        self.settings = Settings()
        self.redis_client: Optional[redis.Redis] = None
        
        try:
            self.redis_client = redis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                db=self.settings.REDIS_DB,
                decode_responses=False,  # We'll handle encoding
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis: {self.settings.REDIS_HOST}:{self.settings.REDIS_PORT}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _make_key(self, prefix: str, identifier: str) -> str:
        """Create a cache key"""
        return f"osint:{prefix}:{identifier}"
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = 3600,
        prefix: str = "general"
    ) -> bool:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache (will be pickled)
            ttl: Time to live in seconds (default 1 hour)
            prefix: Key prefix for organization
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._make_key(prefix, key)
            serialized = pickle.dumps(value)
            
            if ttl:
                self.redis_client.setex(cache_key, ttl, serialized)
            else:
                self.redis_client.set(cache_key, serialized)
            
            logger.debug(f"Cached: {cache_key} (TTL: {ttl}s)")
            return True
        
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def get(self, key: str, prefix: str = "general") -> Optional[Any]:
        """
        Get a value from cache
        
        Args:
            key: Cache key
            prefix: Key prefix
            
        Returns:
            Cached value or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._make_key(prefix, key)
            serialized = self.redis_client.get(cache_key)
            
            if serialized:
                value = pickle.loads(cast(bytes, serialized))
                logger.debug(f"Cache hit: {cache_key}")
                return value
            else:
                logger.debug(f"Cache miss: {cache_key}")
                return None
        
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    def delete(self, key: str, prefix: str = "general") -> bool:
        """Delete a key from cache"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._make_key(prefix, key)
            self.redis_client.delete(cache_key)
            logger.debug(f"Deleted from cache: {cache_key}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def exists(self, key: str, prefix: str = "general") -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._make_key(prefix, key)
            exists_raw = self.redis_client.exists(cache_key)
            return bool(exists_raw)
        
        except Exception as e:
            logger.error(f"Error checking cache existence: {e}")
            return False
    
    def set_json(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = 3600,
        prefix: str = "json"
    ) -> bool:
        """Set JSON data in cache"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._make_key(prefix, key)
            json_str = json.dumps(value)
            
            if ttl:
                self.redis_client.setex(cache_key, ttl, json_str)
            else:
                self.redis_client.set(cache_key, json_str)
            
            return True
        
        except Exception as e:
            logger.error(f"Error setting JSON cache: {e}")
            return False
    
    def get_json(self, key: str, prefix: str = "json") -> Optional[Dict[str, Any]]:
        """Get JSON data from cache"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._make_key(prefix, key)
            raw_json = self.redis_client.get(cache_key)
            
            if raw_json:
                json_str = raw_json.decode() if isinstance(raw_json, (bytes, bytearray)) else str(raw_json)
                return json.loads(json_str)
            return None
        
        except Exception as e:
            logger.error(f"Error getting JSON from cache: {e}")
            return None
    
    def cache_event(self, event_id: str, event_data: Dict[str, Any], ttl: int = 86400) -> bool:
        """Cache an event (24 hour default TTL)"""
        return self.set_json(event_id, event_data, ttl=ttl, prefix="event")
    
    def get_cached_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get cached event"""
        return self.get_json(event_id, prefix="event")
    
    def cache_classification(
        self,
        event_id: str,
        classification: Dict[str, Any],
        ttl: int = 86400
    ) -> bool:
        """Cache event classification result"""
        return self.set_json(event_id, classification, ttl=ttl, prefix="classification")
    
    def get_cached_classification(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get cached classification"""
        return self.get_json(event_id, prefix="classification")
    
    # ==================== URL Deduplication ====================
    
    def is_url_seen(self, url: str, ttl: int = 86400) -> bool:
        """
        Check if URL has been seen before (for deduplication)
        
        Args:
            url: URL to check
            ttl: How long to remember this URL (default 24 hours)
        """
        if not self.redis_client:
            return False
        
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_key = self._make_key("url", url_hash)
        
        exists = bool(self.redis_client.exists(cache_key))
        
        if not exists:
            # Mark as seen
            self.redis_client.setex(cache_key, ttl, "1")
        
        return exists
    
    def mark_url_seen(self, url: str, ttl: int = 86400) -> bool:
        """Mark a URL as seen"""
        if not self.redis_client:
            return False
        
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_key = self._make_key("url", url_hash)
        self.redis_client.setex(cache_key, ttl, "1")
        return True
    
    # ==================== Rate Limiting ====================
    
    def check_rate_limit(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """
        Check if request is within rate limit
        
        Args:
            identifier: Unique identifier (e.g., API key, IP)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            True if within limit, False if exceeded
        """
        if not self.redis_client:
            return True  # Allow if Redis unavailable
        
        try:
            key = self._make_key("ratelimit", identifier)
            
            # Increment counter
            current = int(self.redis_client.incr(key))
            
            # Set expiry on first request
            if current == 1:
                self.redis_client.expire(key, window_seconds)
            
            return current <= max_requests
        
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow on error
    
    def get_rate_limit_remaining(
        self,
        identifier: str,
        max_requests: int
    ) -> int:
        """Get remaining requests in current window"""
        if not self.redis_client:
            return max_requests
        
        try:
            key = self._make_key("ratelimit", identifier)
            current_raw: Any = self.redis_client.get(key)
            current = int(current_raw or 0)
            return max(0, max_requests - current)
        
        except Exception as e:
            logger.error(f"Error getting rate limit: {e}")
            return max_requests
    
    # ==================== List Operations ====================
    
    def push_to_list(self, list_name: str, value: Any, max_length: Optional[int] = None) -> bool:
        """
        Push value to a list (queue)
        
        Args:
            list_name: Name of the list
            value: Value to push
            max_length: Maximum list length (oldest items removed)
        """
        if not self.redis_client:
            return False
        
        try:
            key = self._make_key("list", list_name)
            serialized = pickle.dumps(value)
            
            self.redis_client.lpush(key, serialized)
            
            # Trim if max length specified
            if max_length:
                self.redis_client.ltrim(key, 0, max_length - 1)
            
            return True
        
        except Exception as e:
            logger.error(f"Error pushing to list: {e}")
            return False
    
    def get_list(self, list_name: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get items from a list"""
        if not self.redis_client:
            return []
        
        try:
            key = self._make_key("list", list_name)
            items = self.redis_client.lrange(key, start, end)
            return [pickle.loads(cast(bytes, item)) for item in items]
        
        except Exception as e:
            logger.error(f"Error getting list: {e}")
            return []
    
    def get_list_length(self, list_name: str) -> int:
        """Get length of a list"""
        if not self.redis_client:
            return 0
        
        try:
            key = self._make_key("list", list_name)
            return int(self.redis_client.llen(key))
        
        except Exception as e:
            logger.error(f"Error getting list length: {e}")
            return 0
    
    # ==================== Statistics ====================
    
    def increment_counter(self, counter_name: str, amount: int = 1) -> int:
        """Increment a counter"""
        if not self.redis_client:
            return 0
        
        try:
            key = self._make_key("counter", counter_name)
            return int(self.redis_client.incrby(key, amount))
        
        except Exception as e:
            logger.error(f"Error incrementing counter: {e}")
            return 0
    
    def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        if not self.redis_client:
            return 0
        
        try:
            key = self._make_key("counter", counter_name)
            value: Any = self.redis_client.get(key)
            return int(value) if value else 0
        
        except Exception as e:
            logger.error(f"Error getting counter: {e}")
            return 0
    
    def reset_counter(self, counter_name: str) -> bool:
        """Reset counter to 0"""
        return self.delete(counter_name, prefix="counter")
    
    # ==================== Cache Patterns ====================
    
    def get_or_set(
        self,
        key: str,
        fetch_function: Callable[[], Any],
        ttl: int = 3600,
        prefix: str = "general"
    ) -> Any:
        """
        Get from cache or fetch and cache if not exists
        
        Args:
            key: Cache key
            fetch_function: Function to call if cache miss
            ttl: Time to live
            prefix: Key prefix
        """
        # Try to get from cache
        cached = self.get(key, prefix=prefix)
        if cached is not None:
            return cached
        
        # Fetch fresh data
        value = fetch_function()
        
        # Cache for next time
        self.set(key, value, ttl=ttl, prefix=prefix)
        
        return value
    
    # ==================== Cleanup ====================
    
    def flush_all(self) -> bool:
        """Flush all cache (use with caution!)"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.flushdb()
            logger.warning("Flushed entire cache database")
            return True
        
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
    
    def flush_prefix(self, prefix: str) -> int:
        """Delete all keys with a given prefix"""
        if not self.redis_client:
            return 0
        
        try:
            pattern = f"osint:{prefix}:*"
            keys = list(self.redis_client.keys(pattern))
            
            if keys:
                count = int(self.redis_client.delete(*keys))
                logger.info(f"Deleted {count} keys with prefix '{prefix}'")
                return count
            
            return 0
        
        except Exception as e:
            logger.error(f"Error flushing prefix: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"status": "disconnected"}
        
        try:
            info: Dict[str, Any] = self.redis_client.info()
            
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "unknown"),
                "total_keys": int(self.redis_client.dbsize() or 0),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0)
            }
        
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def health_check(self) -> bool:
        """Check if Redis is healthy"""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.ping())
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


# ==================== Decorators ====================

F = TypeVar("F", bound=Callable[..., Any])


def cache_result(ttl: int = 3600, prefix: str = "function"):
    """
    Decorator to cache function results
    
    Usage:
        @cache_result(ttl=600, prefix="api")
        def expensive_function(arg1, arg2):
            # ... expensive computation
            return result
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = CacheManager()
            
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = ":".join(key_parts)
            
            # Try cache
            cached = cache.get(cache_key, prefix=prefix)
            if cached is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached
            
            # Compute
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, ttl=ttl, prefix=prefix)
            
            return result
        
        return cast(F, wrapper)
    return decorator


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize cache
    cache = CacheManager()
    
    # Test connection
    if cache.health_check():
        print("✅ Redis connection successful")
        
        # Test basic operations
        cache.set("test_key", {"data": "test"}, ttl=60)
        result = cache.get("test_key")
        print(f"Cached data: {result}")
        
        # Test URL deduplication
        url = "https://example.com/article/123"
        is_duplicate = cache.is_url_seen(url)
        print(f"URL seen before: {is_duplicate}")
        
        # Test rate limiting
        within_limit = cache.check_rate_limit("test_user", max_requests=5, window_seconds=60)
        print(f"Within rate limit: {within_limit}")
        
        # Get stats
        stats = cache.get_stats()
        print(f"\nCache Stats:")
        print(f"  Status: {stats['status']}")
        print(f"  Total Keys: {stats.get('total_keys', 0)}")
        print(f"  Memory Used: {stats.get('used_memory', 'unknown')}")
    else:
        print("❌ Redis connection failed")
