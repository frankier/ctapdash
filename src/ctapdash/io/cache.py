import anyio
from glob import glob
from multiprocessing import Pipe, Process
from pathlib import Path

from ctapdash.io.pyramid import mne_to_pyramid, mne_to_rangepyramid
from ctapdash.stats import precompute_descriptive_statistics


def _cache_builder_loop(conn):
    from ctapdash.io.transpose import transpose_file_path, write_transpose
    from ctapdash.io.eeglab import mmap_eeglab, read_eeglab
    from ctapdash.io.paths import cache_path
    jobs = []
    dataset_eeglab_files = {}

    def add_dataset(dataset_dir):
        dataset_dir = Path(dataset_dir)
        dataset_eeglab_files[dataset_dir] = eeglab_files = glob("**/*.set", root_dir=dataset_dir, recursive=True)
        for eeglab_file in eeglab_files:
            jobs.append(("transpose", (dataset_dir, eeglab_file)))
        jobs.append(("stats", dataset_dir))
        for eeglab_file in eeglab_files:
            jobs.append(("pyramid", (dataset_dir, eeglab_file)))
            jobs.append(("rangepyramid", (dataset_dir, eeglab_file)))

    def internal_hurry_tranpose(file):
        try:
            transpose_idx = jobs.index(("transpose", file))
        except ValueError:
            conn.send((job, file, "NOTFOUNDTRANSPOSE"))
        else:
            if transpose_idx > cur_idx:
                del jobs[transpose_idx]
                jobs.insert(cur_idx, ("transpose", file))

    def process_request():
        job, file = conn.recv()
        if job == "exit":
            return True
        if job == "add_dataset":
            add_dataset(file)
        try:
            hurry_idx = jobs.index((job, file))
        except ValueError:
            conn.send((job, file, "NOTFOUND"))
        else:
            if hurry_idx < cur_idx:
                conn.send((job, file))
            else:
                del jobs[hurry_idx]
                jobs.insert(cur_idx, (job, file))
                if job in ("pyramid", "rangepyramid"):
                    internal_hurry_tranpose(file)
                if job == "stats":
                    eeglab_files = dataset_eeglab_files[file]
                    for eeglab_file in eeglab_files:
                        internal_hurry_tranpose((file, eeglab_file))

        return False

    cur_idx = 0
    while 1:
        if cur_idx >= len(jobs) or conn.poll():
            if process_request():
                break
        job, file = jobs[cur_idx]
        if job == "transpose":
            write_transpose(Path(file[0]), Path(file[1]))
        elif job == "stats":
            precompute_descriptive_statistics(file)
        elif job in ("pyramid", "rangepyramid"):
            dataset_dir, in_path = Path(file[0]), Path(file[1])
            full_in_path = dataset_dir / in_path
            if job == "pyramid":
                out_path = cache_path(dataset_dir) / in_path.with_suffix(".pyramid")
            else:
                out_path = cache_path(dataset_dir) / in_path.with_suffix(".rangepyramid")
            if out_path.exists() and out_path.stat().st_mtime >= full_in_path.stat().st_mtime:
                continue
            eeg = read_eeglab(full_in_path, mmap=True)
            arr = mmap_eeglab(
                eeg,
                return_xarray=True,
                data_fname=transpose_file_path(dataset_dir, in_path),
                ctapdash_order=True,
            )
            if job == "pyramid":
                mne_to_pyramid(arr, out_path, [8, 8])
            else:
                mne_to_rangepyramid(arr, out_path, [8, 8])
        conn.send((job, file))
        cur_idx += 1


class CacheHurrier:
    def __init__(self, warmer, conn):
        self.warmer = warmer
        self.conn = conn
        self.resps = []

    async def wait_for(self, expected):
        try:
            found_idx = self.resps.index(expected)
        except ValueError:
            pass
        else:
            self.resps.pop(found_idx)
            return
        while 1:
            resp = await anyio.to_thread.run_sync(self.conn.recv)
            if resp[:2] == expected:
                if len(resp) == 3:
                    if resp[2] == "NOTFOUND":
                        raise RuntimeError(f"Cache job {expected} not found")
                    if resp[2] == "NOTFOUNDTRANSPOSE":
                        raise RuntimeError(f"Cache job {expected} not found (transpose missing)")
                return
            else:
                self.resps.append(resp)

    async def hurry(self, job, file):
        self.conn.send((job, file))
        await self.wait_for((job, file))

    def add_dataset(self, dataset_dir):
        self.conn.send(("add_dataset", dataset_dir))

    async def close(self):
        self.conn.send(("exit", None))
        with anyio.move_on_after(3) as timeout:
            await anyio.to_thread.run_sync(self.process.join)
        if timeout.cancelled_caught:
            self.process.terminate()
        self.conn.close()


class CacheWarmer:
    def __init__(self):
        self.process = None
        self.hurrier = None

    async def __aenter__(self):
        my_conn, their_conn = Pipe()
        self.process = Process(target=_cache_builder_loop, args=(their_conn,))
        await anyio.to_thread.run_sync(self.process.start)
        self.hurrier = CacheHurrier(self, my_conn)
        return self.hurrier

    async def __aexit__(self):
        await self.hurrier.close()
        self.hurrier = None
        self.process = None
