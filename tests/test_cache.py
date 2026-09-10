import asyncio
import os
from multiprocessing import Pipe
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anyio
import numpy as np
import pytest

from ctapdash.io.cache import (
    CacheHurrier, CacheWarmer, JobQueue, REGISTERED, job_key, scan_dataset,
)
from ctapdash.io.paths import DatasetPaths, ObservationData
from ctapdash.io.recording import RecordingData
from tests.dummy_data import write_eeglab


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_recording_paths_and_observation(tmp_path, monkeypatch):
    dataset = DatasetPaths(tmp_path)
    recording = dataset.recording("nested/1_load/p.set")
    assert dataset.recording(recording.set) == recording
    assert recording.transpose == tmp_path / ".ctapdash_cache/nested/1_load/p.ctapdash_order"
    assert recording.pyramid == tmp_path / ".ctapdash_cache/nested/1_load/p.pyramid"
    assert recording.rangepyramid == tmp_path / ".ctapdash_cache/nested/1_load/p.rangepyramid"
    assert recording.metadata == tmp_path / "nested/1_load/.p.pkl"
    assert dataset.stats == tmp_path / ".ctapdash_cache/stats.xm"
    monkeypatch.chdir(tmp_path.parent)
    assert ObservationData(tmp_path.name).source_path == tmp_path
    with pytest.raises(ValueError):
        dataset.recording("../outside.set")
    with pytest.raises(ValueError):
        ObservationData(tmp_path).get_recording(1)
    (tmp_path / "1_load").mkdir()
    (tmp_path / "1_load/p.set").touch()
    assert ObservationData(tmp_path, "p").get_recording("1").paths.set == tmp_path / "1_load/p.set"
    with pytest.raises(ValueError, match="Unknown"):
        ObservationData(tmp_path, "p").get_recording(2)
    with pytest.raises(FileNotFoundError, match="not ready"):
        RecordingData(recording).open_transpose()
    with pytest.raises(FileNotFoundError, match="not ready"):
        RecordingData(recording).open_pyramid()


def test_registration_freshness_snapshot(tmp_path):
    path = write_eeglab(tmp_path / "1_load")
    paths = DatasetPaths(tmp_path).recording(path)
    timestamp = path.stat().st_mtime_ns
    snapshot = scan_dataset(tmp_path)
    assert len(snapshot) == 5
    assert all(job.state == "pending" for job in snapshot.values())
    for job in snapshot.values():
        job.output.parent.mkdir(parents=True, exist_ok=True)
        job.output.touch()
        os.utime(job.output, ns=(timestamp, timestamp))
    assert all(job.state == "ready" for job in scan_dataset(tmp_path).values())
    os.utime(paths.set, ns=(timestamp + 1_000_000, timestamp + 1_000_000))
    assert all(job.state == "pending" for job in scan_dataset(tmp_path).values())
    # Files outside numbered immediate step directories don't affect stats.
    write_eeglab(tmp_path / "nested" / "other")
    stats = scan_dataset(tmp_path)[job_key("stats", tmp_path)]
    assert stats.recordings == (paths.relative,)
    os.utime(paths.dataset.stats, ns=(timestamp + 2_000_000, timestamp + 2_000_000))
    assert scan_dataset(tmp_path)[job_key("stats", tmp_path)].state == "ready"


def test_queue_dedup_priorities_and_failure(tmp_path, monkeypatch):
    first = write_eeglab(tmp_path / "1_load", stem="a")
    second = write_eeglab(tmp_path / "1_load", stem="b")
    queue = JobQueue()
    queue.register(tmp_path)
    pending = list(queue.pending)
    queue.register(tmp_path / ".")
    assert queue.pending == pending
    target = job_key("pyramid", (tmp_path, second))
    queue.prioritize(target)
    assert queue.pending[:3] == [job_key(kind, (tmp_path, second))
                                 for kind in ("metadata", "transpose", "pyramid")]
    built = []

    def build(job):
        built.append(job.key)
        if job.key == job_key("metadata", (tmp_path, second)):
            raise ValueError("bad metadata")
    monkeypatch.setattr("ctapdash.io.cache.build_job", build)
    messages = []
    while queue.pending:
        queue.run_next(messages.append)
    assert queue.jobs[target].state == "failed"
    assert target not in built
    assert queue.jobs[job_key("stats", tmp_path)].state == "failed"
    assert queue.jobs[job_key("pyramid", (tmp_path, first))].state == "ready"
    assert sum(message[2] in ("ready", "failed") for message in messages) == 9


@pytest.mark.anyio
async def test_dispatch_concurrent_waiters_and_failure(tmp_path):
    write_eeglab(tmp_path / "1_load")
    parent, worker = Pipe()
    process = SimpleNamespace(is_alive=lambda: True)
    hurrier = CacheHurrier(SimpleNamespace(process=process), parent)
    hurrier.add_dataset(tmp_path)
    hurrier.add_dataset(tmp_path)
    assert worker.recv() == ("add_dataset", str(tmp_path))
    assert not worker.poll()
    assert hurrier.status()["state"] == "scanning"
    snapshot = scan_dataset(tmp_path)
    key = job_key("stats", tmp_path)
    worker.send(("snapshot", str(tmp_path), snapshot))
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(hurrier.dispatch)
            first = asyncio.create_task(hurrier.hurry("stats", tmp_path))
            second = asyncio.create_task(hurrier.hurry("stats", tmp_path))
            await anyio.sleep(.1)
            assert not first.done() and not second.done()
            worker.send(("job", key, "running", None))
            worker.send(("job", key, "ready", None))
            with anyio.fail_after(2):
                await first
                await second
            assert hurrier.status()["completed"] == 1
            failure = job_key("pyramid", (tmp_path, "1_load/raw.set"))
            worker.send(("job", failure, "failed", "bad pyramid"))
            with pytest.raises(RuntimeError, match="bad pyramid"):
                await hurrier.hurry("pyramid", (tmp_path, "1_load/raw.set"))
            hurrier.closed = True
    finally:
        REGISTERED.pop(str(tmp_path), None)
        parent.close()
        worker.close()


@pytest.mark.anyio
async def test_real_worker_builds_and_reuses_raw_and_epoch_caches(tmp_path):
    for epochs in (1, 2):
        write_eeglab(tmp_path / "1_load", epochs, n_times=512)
    async with CacheWarmer() as warmer:
        process = warmer.warmer.process
        warmer.add_dataset(tmp_path)
        with anyio.fail_after(90):
            for stem in ("raw", "epochs"):
                for kind in ("pyramid", "rangepyramid"):
                    await warmer.hurry(kind, (tmp_path, f"1_load/{stem}.set"))
            await warmer.hurry("stats", tmp_path)
        assert warmer.status()["state"] == "idle"
        assert warmer.status()["completed"] == warmer.status()["total"] == 9
        recording = ObservationData(tmp_path, "raw").get_recording(1)
        np.testing.assert_array_equal(recording.open_transpose(), recording.read_metadata().mmap())
        tree, groups = recording.open_pyramid()
        assert groups == ("/factor_8", "/factor_64")
        tree.close()
        before = recording.paths.transpose.stat().st_mtime_ns
        warmer.add_dataset(tmp_path)
        await warmer.hurry("stats", tmp_path)
        assert warmer.status()["total"] == 0
        assert recording.paths.transpose.stat().st_mtime_ns == before
        # Existing stale stats must be awaited and replaced, even though the path exists.
        os.utime(recording.paths.dataset.stats, ns=(0, 0))
        warmer.add_dataset(tmp_path)
        await warmer.hurry("stats", tmp_path)
        assert warmer.status()["total"] == 1
        assert recording.paths.dataset.stats.stat().st_mtime_ns > 0
        write_eeglab(tmp_path / "1_load", stem="raw", n_times=100)
        warmer.add_dataset(tmp_path)
        with anyio.fail_after(30):
            await warmer.hurry("stats", tmp_path)
            await warmer.hurry("pyramid", (tmp_path, "1_load/raw.set"))
            await warmer.hurry("rangepyramid", (tmp_path, "1_load/raw.set"))
        assert warmer.status()["completed"] == warmer.status()["total"] == 5
        assert recording.open_transpose().shape == (2, 100)
        tree, groups = recording.open_pyramid()
        assert tree[groups[-1]].ds.sizes["time"] == 1
        tree.close()
    assert not process.is_alive()
    assert str(tmp_path) not in REGISTERED


@pytest.mark.anyio
async def test_empty_dataset_and_worker_death(tmp_path):
    async with CacheWarmer() as warmer:
        warmer.add_dataset(tmp_path)
        with anyio.fail_after(30):
            await warmer.hurry("stats", tmp_path)
        assert warmer.status()["state"] == "idle"
        warmer.warmer.process.terminate()
        with anyio.fail_after(3):
            while warmer.worker_error is None:
                await anyio.sleep(.05)
        with pytest.raises(RuntimeError):
            await warmer.hurry("stats", tmp_path)
        assert warmer.status()["state"] == "failed"
        with pytest.raises(RuntimeError):
            warmer.add_dataset(tmp_path)
        assert warmer.status()["state"] == "failed"


def test_setup_registers_manual_and_picked_sources(tmp_path, monkeypatch):
    from ctapdash import setup_ui
    from ctapdash.config import SETTINGS
    from unittest.mock import Mock
    warmer = Mock()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(cache_hurrier=warmer)))
    monkeypatch.setattr(SETTINGS, "sources", {})
    monkeypatch.setattr(setup_ui, "_read_form", AsyncMock(return_value={"path": str(tmp_path)}))
    monkeypatch.setattr(setup_ui, "_render", lambda *args, **kwargs: kwargs)
    asyncio.run(setup_ui.setup_add(request))
    warmer.add_dataset.assert_called_once_with(str(tmp_path))
    warmer.reset_mock()
    window = SimpleNamespace(create_file_dialog=lambda *args, **kwargs: [str(tmp_path)])
    monkeypatch.setattr(setup_ui.desktop, "WINDOW", window)
    with patch.dict("sys.modules", {"webview": SimpleNamespace(FOLDER_DIALOG=1)}):
        asyncio.run(setup_ui.setup_pick(request))
    warmer.add_dataset.assert_called_once_with(str(tmp_path))


def test_newest_relevant_set_and_single_mtime_pass(tmp_path, monkeypatch):
    first = write_eeglab(tmp_path / "1_load", stem="a")
    second = write_eeglab(tmp_path / "2_clean", stem="a")
    for path, timestamp in ((first, 100), (second, 200)):
        os.utime(path, ns=(timestamp, timestamp))
    stats = DatasetPaths(tmp_path).stats
    stats.parent.mkdir()
    stats.touch()
    os.utime(stats, ns=(150, 150))
    counts = {}
    original = Path.stat

    def count_stat(path, *args, **kwargs):
        if path.suffix == ".set":
            counts[path] = counts.get(path, 0) + 1
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", count_stat)
    snapshot = scan_dataset(tmp_path)
    assert counts == {first: 1, second: 1}
    assert snapshot[job_key("stats", tmp_path)].state == "pending"
    os.utime(stats, ns=(200, 200))
    # Changes to external samples alone are deliberately outside the policy.
    os.utime(second.with_suffix(".fdt"), ns=(1000, 1000))
    assert scan_dataset(tmp_path)[job_key("stats", tmp_path)].state == "ready"


@pytest.mark.anyio
async def test_registered_reads_require_readiness_and_skip_stat(tmp_path, monkeypatch):
    from ctapdash.io.eeglab import read_eeglab
    path = write_eeglab(tmp_path / "1_load")
    read_eeglab(path)
    recording = ObservationData(tmp_path, "raw").get_recording(1)
    parent, worker = Pipe()
    hurrier = CacheHurrier(SimpleNamespace(), parent)
    REGISTERED[str(tmp_path)] = hurrier
    try:
        with pytest.raises(FileNotFoundError, match="metadata is not ready"):
            recording.read_metadata()
        hurrier.jobs = scan_dataset(tmp_path)
        assert hurrier.ready("metadata", (tmp_path, path))
        with patch("ctapdash.io.eeglab.os.stat", side_effect=AssertionError("Repeated stat")):
            assert recording.read_metadata().ch_names == ["A", "B"]
    finally:
        REGISTERED.pop(str(tmp_path), None)
        parent.close()
        worker.close()


def test_atomic_replacement_preserves_previous_output_on_failure(tmp_path):
    from ctapdash.io.utils import atomic_write
    destination = tmp_path / "artifact"
    destination.write_text("old")
    with pytest.raises(ValueError):
        with atomic_write(destination, overwrite=True) as staging:
            staging.write_text("incomplete")
            raise ValueError("build failed")
    assert destination.read_text() == "old"
    assert list(tmp_path.iterdir()) == [destination]
    with atomic_write(destination, overwrite=True) as staging:
        staging.write_text("new")
    assert destination.read_text() == "new"
    assert list(tmp_path.iterdir()) == [destination]


def test_stats_requires_metadata_even_with_ready_transpose(tmp_path, monkeypatch):
    path = write_eeglab(tmp_path / "1_load")
    paths = DatasetPaths(tmp_path).recording(path)
    paths.transpose.parent.mkdir(parents=True)
    paths.transpose.touch()
    queue = JobQueue()
    queue.register(tmp_path)
    stats = job_key("stats", tmp_path)
    queue.prioritize(stats)
    built = []

    def fail(job):
        built.append(job.key)
        raise ValueError("cannot read metadata")
    monkeypatch.setattr("ctapdash.io.cache.build_job", fail)
    while queue.pending:
        queue.run_next(lambda message: None)
    assert stats not in built
    assert queue.jobs[stats].state == "failed"


@pytest.mark.parametrize("state,operation,total,completed,visible", [
    ("scanning", None, 5, 0, False),
    ("idle", None, 5, 5, False),
    ("failed", None, 5, 2, False),
    ("warming", None, 5, 2, False),
    ("warming", "transpose", 0, 0, False),
    ("warming", "transpose", 5, 5, False),
    ("warming", "transpose", 5, 2, True),
])
def test_progress_template_only_shows_actual_progress(state, operation, total, completed, visible):
    from ctapdash.webapp import templates
    from html.parser import HTMLParser

    class Indicator(HTMLParser):
        attributes = None
        def handle_starttag(self, tag, attrs):
            if tag == "aside":
                self.attributes = dict(attrs)

    template = templates.env.get_template("_cache_progress.html")
    html = template.render(status=dict(state=state, operation=operation, total=total,
                                       completed=completed, dataset="<example>", error="failure"))
    indicator = Indicator()
    indicator.feed(html)
    assert ("hidden" not in indicator.attributes) == visible
    assert "failure" not in html
    if visible:
        assert "&lt;example&gt;" in html
    else:
        assert "<progress" not in html


@pytest.mark.anyio
async def test_status_broadcast_initial_update_and_unsubscribe(tmp_path):
    parent, worker = Pipe()
    hurrier = CacheHurrier(SimpleNamespace(), parent)
    first, second = hurrier.statuses(), hurrier.statuses()
    try:
        assert (await anext(first))["state"] == "idle"
        assert (await anext(second))["state"] == "idle"
        task_a = asyncio.create_task(anext(first))
        task_b = asyncio.create_task(anext(second))
        await anyio.sleep(0)
        assert not task_a.done() and not task_b.done()
        hurrier.add_dataset(tmp_path)
        with anyio.fail_after(2):
            assert (await task_a)["state"] == "scanning"
            assert (await task_b)["state"] == "scanning"
        await first.aclose()
        task_b = asyncio.create_task(anext(second))
        hurrier.scanning.clear()
        hurrier._wake()
        with anyio.fail_after(2):
            assert (await task_b)["state"] == "idle"
    finally:
        await first.aclose()
        await second.aclose()
        REGISTERED.pop(str(tmp_path), None)
        parent.close()
        worker.close()
