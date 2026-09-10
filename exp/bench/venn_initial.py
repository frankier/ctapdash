"""Benchmark initial Bokeh autoload response, without a browser or tile requests.

Run: .venv/bin/python exp/bench/venn_initial.py [--profile] [--runs N]
Imports/server startup are outside the request timer. Each request creates a
fresh session; caches are not cleared and no system page cache is flushed.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

import argparse  # noqa: E402
import asyncio  # noqa: E402
import cProfile  # noqa: E402
import json  # noqa: E402
from time import perf_counter  # noqa: E402
from functools import wraps  # noqa: E402

from ctapdash.config import SETTINGS, load_from_file  # noqa: E402
from venn_ts import range_series, renderer  # noqa: E402
from venn_ts.plot import venn_time_series_bokeh  # noqa: E402
from venn_ts.plot import _build_comparison  # noqa: F401,E402 - ensures importability for instrumentation
import venn_ts.plot as venn_plot  # noqa: E402
from bokeh.server import asgi  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true')
parser.add_argument('--runs', type=int, default=6)
args = parser.parse_args()
# Resolve conf.toml relative to the project root so the bench works from any cwd.
_conf = _ROOT / "conf.toml"
load_from_file(str(_conf) if _conf.exists() else "conf.toml")

# Pre-seed the process-local stats cache from existing files so the Bokeh
# document can be built synchronously. Without a CacheWarmer registration
# DATASET_STATS.get() would raise and the session would only contain a
# "Loading..." Div, causing the `plots==1` assertion to fail.
try:
    from ctapdash.io.paths import DatasetPaths
    from ctapdash.io.stats_cache import DATASET_STATS
    from ctapdash.io.xarray import load_xarray

    for _src_path in SETTINGS.sources.values():
        try:
            _stats_path = DatasetPaths(Path(_src_path)).stats
            _root = DatasetPaths(Path(_src_path)).root
            if _root not in DATASET_STATS._entries and _stats_path.exists():
                DATASET_STATS._entries[_root] = load_xarray(_stats_path)  # type: ignore[arg-type]
        except Exception:
            pass
except Exception:
    pass

timings = {}

def instrument(obj, name, label=None):
    original = getattr(obj, name)
    @wraps(original)
    def wrapped(*a, **kw):
        start = perf_counter()
        try:
            return original(*a, **kw)
        finally:
            timings.setdefault(label or name, []).append(perf_counter()-start)
    setattr(obj, name, wrapped)

instrument(venn_plot, '_build_comparison', 'build_document')
instrument(range_series.RecordingTileSource, '__init__', 'open_source')
instrument(range_series.RecordingTileSource, 'finite_extrema')
instrument(range_series, 'validate_tile_source_alignment')
import ctapdash.io.eeglab as eeglab  # noqa: E402
import ctapdash.io.pyramid as pyramid  # noqa: E402
instrument(eeglab, 'read_eeglab')
instrument(pyramid, 'load_pyramid')
instrument(asgi, 'bundle_for_objs_and_resources', 'bundle_resources')

def no_tiles(*a, **kw):
    raise AssertionError('Tile loading is outside this benchmark')
renderer.TileCoordinator._load = no_tiles

async def _ensure_psicat_stats():
    """Ensure the psicat stats needed by the benchmark are available."""
    try:
        from ctapdash.io.paths import DatasetPaths
        from ctapdash.io.stats_cache import DATASET_STATS
        from ctapdash.io.xarray import load_xarray
        psicat_path = SETTINGS.sources.get("psicat")
        if not psicat_path:
            return
        root = DatasetPaths(Path(psicat_path)).root
        if DATASET_STATS.get_cached(root) is not None:
            return
        stats_path = DatasetPaths(Path(psicat_path)).stats
        if stats_path.exists():
            try:
                DATASET_STATS._entries[root] = load_xarray(stats_path)  # type: ignore[arg-type]
                return
            except Exception:
                pass
        # Missing file: build it via the cache worker so DATASET_STATS.get()
        # would succeed and the Bokeh handler builds synchronously.
        from ctapdash.io.cache import CacheWarmer
        async with CacheWarmer() as hurrier:  # type: ignore[no-untyped-call]
            for p in SETTINGS.sources.values():
                hurrier.add_dataset(p)
            await hurrier.hurry("stats", root)
            # Worker wrote the file; now load it into the local cache.
            if stats_path.exists():
                DATASET_STATS._entries[root] = load_xarray(stats_path)  # type: ignore[arg-type]
            else:
                # Fallback: direct computation if worker did not materialize
                from ctapdash.stats import precompute_descriptive_statistics
                await asyncio.to_thread(precompute_descriptive_statistics, root)
                DATASET_STATS._entries[root] = load_xarray(stats_path)  # type: ignore[arg-type]
    except Exception:
        pass

async def main():
    await _ensure_psicat_stats()
    app = asgi.BokehASGI({'/venn-time-series': venn_time_series_bokeh})
    results = []
    try:
        for i in range(args.runs):
            timings.clear()
            messages = []
            async def send(message):
                messages.append(message)
            async def receive():
                return {'type':'http.request','body':b'', 'more_body':False}
            scope = dict(type='http', asgi={'version':'3.0'}, http_version='1.1',
                         method='GET', scheme='http', path='/venn-time-series/autoload.js',
                         raw_path=b'/venn-time-series/autoload.js', root_path='',
                         query_string=b'bokeh-autoload-element=benchmark&bokeh-app-path=/venn-time-series&source=psicat&participant=1061P_psicat_intake',
                         headers=[(b'host', b'localhost')], server=('localhost',80), client=('127.0.0.1',12345))
            profiler = cProfile.Profile() if args.profile else None
            start = perf_counter()
            if profiler:
                profiler.enable()
            await app(scope, receive, send)
            # If stats were not cached, the Bokeh document builds asynchronously
            # via add_next_tick_callback. Poll briefly until the renderer appears
            # so the benchmark still measures a completed document.
            # When stats are pre-seeded above this loop is a no-op.
            deadline = perf_counter() + 5.0
            plots = []
            while perf_counter() < deadline:
                sessions = app.core.applications['/venn-time-series'].sessions
                if sessions:
                    session = list(sessions)[-1]
                    models = list(session.document.models)
                    plots = [m for m in models if isinstance(m, renderer.VennTimeSeriesRenderer)]
                    if len(plots) == 1:
                        # Also ensure any in-flight next-tick callback had a chance
                        # to record its `build_document` timing.
                        if timings.get('build_document'):
                            break
                        # Synchronous path records timing immediately; allow
                        # async path one more iteration for timings to appear.
                        if perf_counter() + 0.2 > deadline:
                            break
                    # If we have a Div error container, surface it early.
                    if plots:
                        break
                await asyncio.sleep(0.05)
                # If the document never gains a renderer, fall through to assertion
                if plots:
                    break
            if profiler:
                profiler.disable()
            total = perf_counter()-start
            assert messages[0]['status']==200, messages
            sessions = app.core.applications['/venn-time-series'].sessions
            session = list(sessions)[-1]
            models = list(session.document.models)
            plots = [m for m in models if isinstance(m, renderer.VennTimeSeriesRenderer)]
            assert len(plots)==1, f'Expected an actual plot, not an error container: {[type(r).__name__+":"+getattr(r,"text","")[:300] for r in session.document.roots]} timings={timings}'
            assert plots[0].tile_requests==0
            result = {'run':i+1, 'total_s':total, 'phases_s':{k:sum(v) for k,v in timings.items()},
                      'calls':{k:len(v) for k,v in timings.items()}, 'models':len(models),
                      'bytes':sum(len(m.get('body',b'')) for m in messages)}
            results.append(result)
            print(json.dumps(result), flush=True)
            if profiler:
                out_prof = _ROOT / f'exp/bench/results/venn-initial-{i+1}.prof'
                out_prof.parent.mkdir(parents=True, exist_ok=True)
                profiler.dump_stats(str(out_prof))
        out_path = _ROOT / f'exp/bench/results/{("profiled" if args.profile else "timings")}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results,indent=2))
    finally:
        await app.core.stop()

asyncio.run(main())
