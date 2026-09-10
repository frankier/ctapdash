"""Registration-time cache validation and a single-reader worker dispatcher."""
from dataclasses import dataclass
from multiprocessing import Pipe, Process
from pathlib import Path
import pickle

import anyio

from ctapdash.io.paths import DatasetPaths


# Available to synchronous viewers in the server process. Never shared with workers.
REGISTERED = {}


def job_key(kind, file):
    if kind == "stats":
        return kind, str(DatasetPaths(file).root), ""
    root, relative = file
    paths = DatasetPaths(root).recording(relative)
    return kind, str(paths.dataset.root), str(paths.relative)


@dataclass
class Job:
    key: tuple
    output: Path
    dependencies: tuple = ()
    state: str = "pending"
    error: str | None = None
    recordings: tuple = ()


def scan_dataset(directory):
    """Capture source mtimes once and classify all artifacts without building."""
    dataset = DatasetPaths(directory)
    if not dataset.root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset.root}")
    inputs = [(dataset.recording(path), path.stat().st_mtime_ns)
              for path in sorted(dataset.root.rglob("*.set"))
              if dataset.cache not in path.parents]
    jobs = {}
    stats_inputs = []
    stats_mtimes = []

    def add(kind, paths, output, mtime, dependencies=(), recordings=()):
        key = (kind, str(dataset.root), str(paths.relative) if paths else "")
        try:
            fresh = output.stat().st_mtime_ns >= mtime
        except FileNotFoundError:
            fresh = False
        jobs[key] = Job(key, output, dependencies, "ready" if fresh else "pending",
                        recordings=recordings)
        return key

    for paths, mtime in inputs:
        metadata = add("metadata", paths, paths.metadata, mtime)
        transpose = add("transpose", paths, paths.transpose, mtime, (metadata,))
        add("pyramid", paths, paths.pyramid, mtime, (metadata, transpose))
        add("rangepyramid", paths, paths.rangepyramid, mtime, (metadata, transpose))
        # Match describe_dataset: immediate .set files in numbered step folders.
        if len(paths.relative.parts) == 2 and paths.relative.parts[0][0].isnumeric():
            stats_inputs.append(paths.relative)
            stats_mtimes.append(mtime)
    dependencies = tuple((kind, str(dataset.root), str(path))
                         for path in stats_inputs for kind in ("metadata", "transpose"))
    add("stats", None, dataset.stats, max(stats_mtimes, default=0), dependencies,
        tuple(stats_inputs))
    return jobs


def build_job(job):
    from ctapdash.io.eeglab import read_eeglab
    from ctapdash.io.recording import RecordingData
    from ctapdash.io.transpose import write_transpose
    from ctapdash.io.pyramid import mne_to_pyramid, mne_to_rangepyramid
    from ctapdash.io.utils import atomic_write
    from ctapdash.stats import precompute_descriptive_statistics

    kind, root, relative = job.key
    job.output.parent.mkdir(parents=True, exist_ok=True)
    if kind == "metadata":
        eeg = read_eeglab(Path(root) / relative, use_cache=False)
        with atomic_write(job.output, overwrite=True) as staging:
            with staging.open("wb") as stream:
                pickle.dump(eeg, stream)
    elif kind == "transpose":
        write_transpose(root, relative, metadata_validated=True)
    elif kind == "stats":
        precompute_descriptive_statistics(root, recordings=job.recordings,
                                          metadata_validated=True)
    else:
        recording = RecordingData(DatasetPaths(root).recording(relative), True)
        array = recording.open_transpose(return_xarray=True)
        builder = mne_to_pyramid if kind == "pyramid" else mne_to_rangepyramid
        with atomic_write(job.output, dir=True, overwrite=True) as staging:
            builder(array, staging, [8, 8])


class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.pending = []

    def register(self, directory):
        root = str(DatasetPaths(directory).root)
        if any(key[1] == root for key in self.pending):
            return {key: job for key, job in self.jobs.items() if key[1] == root}
        snapshot = scan_dataset(root)
        self.jobs = {key: job for key, job in self.jobs.items() if key[1] != root}
        self.jobs.update(snapshot)
        # Metadata, transposes, stats, then pyramids. Priorities preserve dependencies.
        order = {"metadata": 0, "transpose": 1, "stats": 2, "pyramid": 3, "rangepyramid": 4}
        self.pending.extend(sorted((key for key, job in snapshot.items()
                                    if job.state == "pending"), key=lambda key: order[key[0]]))
        return snapshot

    def prioritize(self, key):
        ordered = []

        def visit(current):
            if current not in self.jobs or current in ordered:
                return
            for dependency in self.jobs[current].dependencies:
                visit(dependency)
            if current in self.pending:
                ordered.append(current)
        visit(key)
        self.pending = ordered + [item for item in self.pending if item not in ordered]

    def run_next(self, send):
        key = self.pending.pop(0)
        job = self.jobs[key]
        try:
            for dependency in job.dependencies:
                if self.jobs[dependency].state != "ready":
                    raise RuntimeError(f"Dependency failed: {dependency}")
            job.state = "running"
            send(("job", key, "running", None))
            build_job(job)
            job.state = "ready"
        except Exception as error:
            job.state = "failed"
            job.error = f"{type(error).__name__}: {error}"
        send(("job", key, job.state, job.error))


def _cache_builder_loop(conn):
    queue = JobQueue()
    try:
        while True:
            if not queue.pending or conn.poll():
                command, value = conn.recv()
                if command == "exit":
                    return
                if command == "add_dataset":
                    try:
                        snapshot = queue.register(value)
                        conn.send(("snapshot", value, snapshot))
                    except Exception as error:
                        conn.send(("scan_failed", value, f"{type(error).__name__}: {error}"))
                elif command == "hurry":
                    queue.prioritize(value)
                # Drain registration/priority requests before choosing the next job.
                continue
            queue.run_next(conn.send)
    except (EOFError, BrokenPipeError):
        return
    finally:
        conn.close()


class CacheHurrier:
    def __init__(self, warmer, conn):
        self.warmer = warmer
        self.conn = conn
        self.jobs = {}
        self.required = set()
        self.scanning = set()
        self.errors = {}
        self.waiters = {}
        self.closed = False
        self.worker_error = None
        self.status_changed = anyio.Event()

    def add_dataset(self, dataset_dir):
        root = str(DatasetPaths(dataset_dir).root)
        if root in self.scanning or any(key[1] == root and job.state in ("pending", "running")
                                        for key, job in self.jobs.items()):
            return
        self._send(("add_dataset", root))
        REGISTERED[root] = self
        self.scanning.add(root)
        self.errors.pop(root, None)
        self._wake()

    def _send(self, message):
        if self.worker_error or self.closed:
            raise RuntimeError(self.worker_error or "Cache worker closed")
        try:
            self.conn.send(message)
        except (OSError, EOFError) as error:
            self.worker_error = "Cache worker disconnected"
            self._wake()
            raise RuntimeError(self.worker_error) from error

    def ready(self, kind, file):
        key = job_key(kind, file)
        return key[1] not in self.scanning and key in self.jobs and self.jobs[key].state == "ready"

    def _wake(self):
        self.status_changed.set()
        self.status_changed = anyio.Event()
        for event in self.waiters.values():
            event.set()
        self.waiters.clear()

    async def hurry(self, kind, file):
        key = job_key(kind, file)
        if REGISTERED.get(key[1]) is not self:
            self.add_dataset(key[1])
        self._send(("hurry", key))
        while True:
            if self.worker_error or self.closed:
                raise RuntimeError(self.worker_error or "Cache worker closed")
            if key[1] in self.errors:
                raise RuntimeError(self.errors[key[1]])
            if key[1] not in self.scanning:
                job = self.jobs.get(key)
                if job is None:
                    raise RuntimeError(f"Cache job not found: {key}")
                if job.state == "ready":
                    return
                if job.state == "failed":
                    raise RuntimeError(job.error)
            event = self.waiters.setdefault(key, anyio.Event())
            await event.wait()

    def status(self):
        running = next((job.key for job in self.jobs.values() if job.state == "running"), None)
        failed = [job for job in self.jobs.values() if job.state == "failed"]
        completed = sum(self.jobs[key].state == "ready" for key in self.required)
        active = bool(self.scanning or any(job.state in ("pending", "running") for job in self.jobs.values()))
        error = self.worker_error or next(iter(self.errors.values()), None) or (failed[0].error if failed else None)
        return {"state": "scanning" if self.scanning else "warming" if active else "failed" if error else "idle",
                "completed": completed, "total": len(self.required),
                "failed": len(failed), "error": error,
                "dataset": Path(running[1]).name if running else
                           (Path(sorted(self.scanning)[0]).name if self.scanning else None),
                "operation": running[0] if running else None}

    async def statuses(self):
        """Broadcast snapshots to each subscriber without consuming worker messages."""
        previous = None
        while not self.closed:
            changed = self.status_changed
            snapshot = self.status()
            if snapshot != previous:
                previous = snapshot
                yield snapshot
            await changed.wait()

    async def dispatch(self):
        try:
            while not self.closed:
                while self.conn.poll():
                    message = self.conn.recv()
                    if message[0] == "snapshot":
                        _, root, snapshot = message
                        self.jobs = {key: job for key, job in self.jobs.items() if key[1] != root}
                        self.required = {key for key in self.required if key[1] != root}
                        self.jobs.update(snapshot)
                        self.required.update(key for key, job in snapshot.items() if job.state == "pending")
                        self.scanning.discard(root)
                    elif message[0] == "scan_failed":
                        _, root, error = message
                        self.scanning.discard(root)
                        self.errors[root] = error
                    else:
                        _, key, state, error = message
                        self.jobs[key].state = state
                        self.jobs[key].error = error
                    self._wake()
                if not self.warmer.process.is_alive():
                    raise RuntimeError("Cache worker exited unexpectedly")
                await anyio.sleep(0.05)
        except (EOFError, OSError, RuntimeError) as error:
            if not self.closed:
                self.worker_error = str(error) or "Cache worker disconnected"
                self.scanning.clear()
                for job in self.jobs.values():
                    if job.state in ("pending", "running"):
                        job.state, job.error = "failed", self.worker_error
                self._wake()

    async def close(self):
        self.closed = True
        self._wake()
        for root in list(REGISTERED):
            if REGISTERED[root] is self:
                del REGISTERED[root]
        try:
            self.conn.send(("exit", None))
        except (BrokenPipeError, OSError):
            pass
        process = self.warmer.process
        await anyio.to_thread.run_sync(process.join, 3)
        if process.is_alive():
            process.terminate()
            await anyio.to_thread.run_sync(process.join, 3)
        self.conn.close()


class CacheWarmer:
    def __init__(self):
        self.process = None
        self.hurrier = None

    async def __aenter__(self):
        my_conn, their_conn = Pipe()
        self.process = Process(target=_cache_builder_loop, args=(their_conn,))
        self.process.start()
        their_conn.close()
        self.hurrier = CacheHurrier(self, my_conn)
        self.task_group = anyio.create_task_group()
        await self.task_group.__aenter__()
        self.task_group.start_soon(self.hurrier.dispatch)
        return self.hurrier

    async def __aexit__(self, exc_type, exc, tb):
        with anyio.CancelScope(shield=True):
            await self.hurrier.close()
        await self.task_group.__aexit__(exc_type, exc, tb)
        self.hurrier = None
        self.process = None
