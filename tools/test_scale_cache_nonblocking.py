"""
Concurrency test: prove SerialScaleReader.get_latest_weight() never blocks
even when read_weight() holds _lock during simulated serial I/O.

Pass criteria:
  - All get_latest_weight() calls return in < 1 ms
  - Max latency < 1 ms
  - Correct cached value returned throughout

Run:
  python tools/test_scale_cache_nonblocking.py
"""

import sys
import threading
import time
from pathlib import Path
from statistics import mean, median, quantiles

# Allow running from repo root or from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.scale_reader import SerialScaleReader

BLOCK_DURATION   = 2.0    # seconds the "serial read" holds _lock
SAMPLE_COUNT     = 200
LATENCY_LIMIT_MS = 1.0    # each call must be faster than this
SEED_VALUE       = 42.0   # pre-seeded cached weight


def _hold_lock(reader: SerialScaleReader, duration: float) -> None:
    """Simulate a blocking serial read by holding _lock for `duration` seconds."""
    with reader._lock:
        time.sleep(duration)


def run_test() -> bool:
    reader = SerialScaleReader(port="/dev/null", timeout=2.0, retries=5)

    # Pre-seed _latest via _cache_lock (same path as real serial update)
    with reader._cache_lock:
        reader._latest = SEED_VALUE

    # Start the "blocking serial read" thread
    blocker = threading.Thread(target=_hold_lock, args=(reader, BLOCK_DURATION), daemon=True)
    blocker.start()

    # Give the blocker a moment to acquire _lock
    time.sleep(0.01)

    latencies_ms: list[float] = []
    wrong_values = 0

    for _ in range(SAMPLE_COUNT):
        t0 = time.perf_counter()
        val = reader.get_latest_weight()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)
        if val != SEED_VALUE:
            wrong_values += 1

    blocker.join(timeout=BLOCK_DURATION + 1)

    max_ms  = max(latencies_ms)
    avg_ms  = mean(latencies_ms)
    med_ms  = median(latencies_ms)
    p99_ms  = quantiles(latencies_ms, n=100)[98]

    print(f"  Samples  : {SAMPLE_COUNT}")
    print(f"  Max      : {max_ms:.3f} ms  (limit {LATENCY_LIMIT_MS:.1f} ms)")
    print(f"  Avg      : {avg_ms:.3f} ms")
    print(f"  Median   : {med_ms:.3f} ms")
    print(f"  p99      : {p99_ms:.3f} ms")
    print(f"  Wrong val: {wrong_values} / {SAMPLE_COUNT}")

    passed = max_ms < LATENCY_LIMIT_MS and wrong_values == 0
    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("SerialScaleReader.get_latest_weight() — non-blocking test")
    print("=" * 60)
    print(f"  Blocker holds _lock for {BLOCK_DURATION}s while we sample …\n")

    ok = run_test()

    print()
    if ok:
        print("PASS — get_latest_weight() is non-blocking.")
        sys.exit(0)
    else:
        print("FAIL — get_latest_weight() blocked or returned wrong value.")
        sys.exit(1)
