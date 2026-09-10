"""Process-local ownership of shared, read-only dataset statistics.

Consumers must not mutate or close returned xarrays. Registration determines
freshness; invalidation drops references without closing live consumer views.
Methods run on the server event loop, like CacheHurrier itself.
"""
import anyio

from ctapdash.io.paths import DatasetPaths
from ctapdash.io.xarray import load_xarray


class DatasetStatsCache:
    def __init__(self):
        self._entries = {}
        self._locks = {}

    def get_cached(self, root):
        return self._entries.get(DatasetPaths(root).root)

    def invalidate(self, root):
        root = DatasetPaths(root).root
        self._entries.pop(root, None)
        self._locks.pop(root, None)

    async def get(self, root):
        from ctapdash.io.cache import REGISTERED

        paths = DatasetPaths(root)
        root = paths.root
        while True:
            cached = self._entries.get(root)
            if cached is not None:
                return cached
            lock = self._locks.setdefault(root, anyio.Lock())
            async with lock:
                if self._locks.get(root) is not lock:
                    continue
                cached = self._entries.get(root)
                if cached is not None:
                    return cached
                hurrier = REGISTERED.get(str(root))
                if hurrier is None:
                    raise RuntimeError(f"Dataset is not registered for stats: {root}")
                await hurrier.hurry("stats", root)
                if self._locks.get(root) is not lock:
                    continue
                # Loading only maps data and reads small metadata. Keep publication
                # on the event loop so invalidation cannot interleave with it.
                stats = load_xarray(paths.stats, mode="r")
                self._entries[root] = stats
                return stats


DATASET_STATS = DatasetStatsCache()
