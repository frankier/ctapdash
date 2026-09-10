import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import xarray as xr
from bokeh.document import Document

from ctapdash.io.cache import REGISTERED
from ctapdash.io.paths import DatasetPaths, ObservationData
from ctapdash.io.stats_cache import DATASET_STATS, DatasetStatsCache
from ctapdash.io.xarray import save_xarray
from venn_ts.plot import _stats_extrema, venn_time_series_bokeh


def stats_fixture():
    return xr.Dataset(
        {"min": (("recording", "channel"), [[-2., -1.], [0., 1.]]),
         "max": (("recording", "channel"), [[2., 1.], [3., 4.]])},
        coords={"channel": ["Cz", "Pz"], "participant": ("recording", ["p", "p"]),
                "step": ("recording", [1, 2])},
    )


def test_shared_mapping_and_retry(tmp_path, monkeypatch):
    path = DatasetPaths(tmp_path).stats
    path.parent.mkdir()
    save_xarray(stats_fixture(), path)
    cache = DatasetStatsCache()
    hurrier = SimpleNamespace(hurry=AsyncMock(side_effect=[RuntimeError("worker failed"), None]))
    monkeypatch.setitem(REGISTERED, str(tmp_path), hurrier)

    async def run():
        with pytest.raises(RuntimeError, match="worker failed"):
            await cache.get(tmp_path)
        a, b = await asyncio.gather(cache.get(tmp_path), cache.get(tmp_path / "."))
        assert a is b is cache.get_cached(tmp_path)
        assert not a["min"].data.flags.writeable
        backing = a["min"].data
        while not isinstance(backing, np.memmap) and backing.base is not None:
            backing = backing.base
        assert isinstance(backing, np.memmap)
        assert hurrier.hurry.await_count == 2
        cache.invalidate(tmp_path)
        assert cache.get_cached(tmp_path) is None
        # Invalidating ownership must not close a consumer's mapping.
        np.testing.assert_array_equal(a["min"], [[-2, -1], [0, 1]])
    asyncio.run(run())


def test_invalidation_during_load(tmp_path, monkeypatch):
    cache = DatasetStatsCache()
    import ctapdash.io.stats_cache as module
    loaded = []
    def load(*args, **kwargs):
        result = stats_fixture()
        loaded.append(result)
        return result
    monkeypatch.setattr(module, "load_xarray", load)

    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        async def hurry(*args):
            started.set()
            await release.wait()
        monkeypatch.setitem(REGISTERED, str(tmp_path), SimpleNamespace(hurry=hurry))
        old = asyncio.create_task(cache.get(tmp_path))
        await started.wait()
        cache.invalidate(tmp_path)
        new = asyncio.create_task(cache.get(tmp_path))
        release.set()
        a, b = await asyncio.gather(old, new)
        assert a is b is cache.get_cached(tmp_path)
        assert len(loaded) == 1
    asyncio.run(run())


def test_unregistered_and_load_failure_retry(tmp_path, monkeypatch):
    cache = DatasetStatsCache()
    async def run():
        with pytest.raises(RuntimeError, match="not registered"):
            await cache.get(tmp_path)
        monkeypatch.setitem(REGISTERED, str(tmp_path), SimpleNamespace(hurry=AsyncMock()))
        with pytest.raises(FileNotFoundError):
            await cache.get(tmp_path)
        path = DatasetPaths(tmp_path).stats
        path.parent.mkdir()
        save_xarray(stats_fixture(), path)
        assert await cache.get(tmp_path) is cache.get_cached(tmp_path)
    asyncio.run(run())


def test_stats_selection_and_invalid_bounds():
    stats = stats_fixture()
    np.testing.assert_array_equal(_stats_extrema(stats, "p", "1", ["Pz", "Cz"]), [[-1, 1], [-2, 2]])
    with pytest.raises(ValueError, match="found 0"):
        _stats_extrema(stats, "missing", 1, ["Cz"])
    with pytest.raises(ValueError, match="found 2"):
        _stats_extrema(xr.concat([stats, stats], dim="recording"), "p", 1, ["Cz"])
    with pytest.raises(KeyError):
        _stats_extrema(stats, "p", 1, ["missing"])
    for value in [np.nan, np.inf, 3.]:
        modified = stats.copy(deep=True)
        modified["min"].data[0, 0] = value
        with pytest.raises(ValueError, match="Invalid min/max"):
            _stats_extrema(modified, "p", 1, ["Cz"])
    stats["min"].data[0, 0] = 2.
    np.testing.assert_array_equal(_stats_extrema(stats, "p", 1, ["Cz"]), [[2., 2.]])


@pytest.mark.parametrize("destroy,error", [(False, False), (True, False), (False, True)])
def test_direct_bokeh_miss(tmp_path, monkeypatch, destroy, error):
    import venn_ts.plot as plot
    dataset = SimpleNamespace(source_path=tmp_path, get_steps=lambda: [(1, tmp_path)])
    monkeypatch.setattr(ObservationData, "from_bokeh_doc", lambda doc: dataset)
    monkeypatch.setattr(DATASET_STATS, "get_cached", lambda root: None)
    built = []
    monkeypatch.setattr(plot, "_build_comparison", lambda *args: built.append(args))
    doc = Document()
    async def get(root):
        if destroy:
            for callback in doc.session_destroyed_callbacks:
                callback(None)
        if error:
            raise RuntimeError("stats worker failed")
        return stats_fixture()
    monkeypatch.setattr(DATASET_STATS, "get", get)
    venn_time_series_bokeh(doc)
    callback = next(iter(doc.session_callbacks))
    asyncio.run(callback.callback())
    assert bool(built) == (not destroy and not error)
    if error:
        assert "stats worker failed" in doc.roots[0].text
    if destroy:
        assert doc.roots[0].text == "Loading comparison statistics..."


def test_registration_and_shutdown_invalidate(tmp_path, monkeypatch):
    from multiprocessing import Pipe
    from ctapdash.io.cache import CacheHurrier
    parent, worker = Pipe()
    process = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: False)
    hurrier = CacheHurrier(SimpleNamespace(process=process), parent)
    monkeypatch.setattr(DATASET_STATS, "_entries", {tmp_path: stats_fixture()})
    monkeypatch.setattr(DATASET_STATS, "_locks", {})
    try:
        hurrier.add_dataset(tmp_path)
        assert DATASET_STATS.get_cached(tmp_path) is None
        assert worker.recv() == ("add_dataset", str(tmp_path))
        # A redundant registration during scanning doesn't discard a live entry.
        stats = stats_fixture()
        DATASET_STATS._entries[tmp_path] = stats
        hurrier.add_dataset(tmp_path)
        assert DATASET_STATS.get_cached(tmp_path) is stats
        hurrier.scanning.clear()
        hurrier.add_dataset(tmp_path)
        assert DATASET_STATS.get_cached(tmp_path) is None
        assert worker.recv() == ("add_dataset", str(tmp_path))
        DATASET_STATS._entries[tmp_path] = stats
        asyncio.run(hurrier.close())
        assert DATASET_STATS.get_cached(tmp_path) is None
        assert str(tmp_path) not in REGISTERED
        np.testing.assert_array_equal(stats["min"], [[-2, -1], [0, 1]])
    finally:
        REGISTERED.pop(str(tmp_path), None)
        parent.close()
        worker.close()


def test_http_viewer_waits_for_stats(tmp_path, monkeypatch):
    import ctapdash.webapp as webapp
    request = SimpleNamespace(query_params={"participant": "p"})
    dataset = SimpleNamespace(source_path=tmp_path, get_steps=lambda: [])
    monkeypatch.setattr(webapp, "participant_context", lambda *args, **kwargs: {"source": "s", "participant": "p"})
    monkeypatch.setattr(ObservationData, "from_request", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(webapp, "_wait_metadata", AsyncMock())
    get = AsyncMock(return_value=stats_fixture())
    monkeypatch.setattr(DATASET_STATS, "get", get)
    def embed(*args, **kwargs):
        get.assert_awaited_once_with(tmp_path)
        return "embedded"
    monkeypatch.setattr(webapp, "bokeh_document", embed)
    monkeypatch.setattr(webapp.templates, "TemplateResponse", lambda *args, **kwargs: kwargs["context"])
    assert asyncio.run(webapp.venn_time_series(request))["venn_time_series"] == "embedded"
