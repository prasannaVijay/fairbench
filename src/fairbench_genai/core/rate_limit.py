"""Async token-bucket rate limiter.

Book: Chapter 5, "Run configuration" / "Orchestrator implementation".

A semaphore bounds *concurrency*; it does not enforce a request rate. When calls
fail fast (e.g. a 400 content-policy rejection in well under a second), a pool of
concurrent workers can blow past a per-minute limit and trigger 429s. The token
bucket is the authoritative throttle: workers ``await acquire()`` before each
call, and the bucket refills at ``rpm`` requests per minute.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """A simple async token bucket sized in requests per minute."""

    def __init__(self, rpm: int) -> None:
        self.rate = max(rpm, 1) / 60.0        # tokens per second
        self.capacity = float(max(rpm, 1))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
            self._updated = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self.rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
                self._updated = time.monotonic()
            else:
                self._tokens -= 1.0
