import asyncio

import pytest

from log_api.protection import (
    RateLimitConfig,
    RateLimiterUnavailable,
    RedisMetricsStore,
    RedisRateLimiter,
)


class FakeRedis:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.closed = False

    async def eval(self, *args):
        self.calls.append(args)
        return self.result

    async def aclose(self):
        self.closed = True


class FakePipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def pfadd(self, *args):
        self.calls.append(("pfadd", *args))

    def expire(self, *args):
        self.calls.append(("expire", *args))

    def hgetall(self, *args):
        self.calls.append(("hgetall", *args))

    def pfcount(self, *args):
        self.calls.append(("pfcount", *args))

    async def execute(self):
        return self.result


class FakeMetricsRedis:
    def __init__(self, results):
        self.results = list(results)
        self.pipelines = []

    def pipeline(self, **_kwargs):
        pipeline = FakePipeline(self.results.pop(0))
        self.pipelines.append(pipeline)
        return pipeline


def test_redis_limiter_uses_prefixed_key_and_returns_decision():
    client = FakeRedis([0, 7, 30])
    config = RateLimitConfig(
        requests=30,
        window_seconds=60,
        redis_url="redis://example",
        key_prefix="sift:test-ratelimit",
    )
    limiter = RedisRateLimiter(config, client=client)

    allowed, retry_after = asyncio.run(limiter.check("user:123"))

    assert allowed is False
    assert retry_after == 7
    assert client.calls[0][1:3] == (1, "sift:test-ratelimit:user:123")


def test_redis_limiter_closes_client():
    client = FakeRedis([1, 0, 1])
    limiter = RedisRateLimiter(
        RateLimitConfig(redis_url="redis://example"),
        client=client,
    )

    asyncio.run(limiter.close())

    assert client.closed is True


@pytest.mark.parametrize(
    ("env_name", "value", "message"),
    [
        ("RATE_LIMIT_REQUESTS", "0", "RATE_LIMIT_REQUESTS"),
        ("RATE_LIMIT_WINDOW_SECONDS", "0", "RATE_LIMIT_WINDOW_SECONDS"),
    ],
)
def test_rate_limit_config_rejects_invalid_values(monkeypatch, env_name, value, message):
    monkeypatch.setenv(env_name, value)

    with pytest.raises(RuntimeError, match=message):
        RateLimitConfig.from_env()


def test_rate_limit_config_requires_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    with pytest.raises(RuntimeError, match="REDIS_URL"):
        RateLimitConfig.from_env()


def test_redis_errors_are_translated_to_limiter_unavailable():
    class BrokenRedis:
        async def eval(self, *args):
            from redis.exceptions import ConnectionError

            raise ConnectionError("redis is down")

    limiter = RedisRateLimiter(
        RateLimitConfig(redis_url="redis://example"),
        client=BrokenRedis(),
    )

    async def exercise():
        with pytest.raises(RateLimiterUnavailable):
            await limiter.check("user:123")

    asyncio.run(exercise())


def test_metrics_store_counts_pseudonymous_visitor_without_raw_subject():
    client = FakeMetricsRedis([[1, True]])
    store = RedisMetricsStore(client, "sift:test-metrics", 3600)

    asyncio.run(store.record_visitor("user-123"))

    calls = client.pipelines[0].calls
    visitor_values = [call[2] for call in calls if call[0] == "pfadd"]
    assert len(visitor_values) == 1
    assert all("user-123" not in value for value in visitor_values)


def test_metrics_snapshot_includes_unique_visitors():
    client = FakeMetricsRedis(
        [[
            {"requests": "2", "errors": "0", "latency_sum_ms": "20"},
            {"requests": "2", "errors": "0", "latency_sum_ms": "20"},
            3,
        ]]
    )
    store = RedisMetricsStore(client, "sift:test-metrics", 3600)

    snapshot = asyncio.run(store.snapshot(1))

    assert snapshot["totals"]["unique_visitors"] == 3
