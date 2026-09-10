from pathlib import Path

from ctapdash.io.eeglab import read_eeglab
from ctapdash.io.paths import cache_path
from ctapdash.io.utils import atomic_write


def write_transpose(dataset_dir, eeglab_file):
    full_eeglab_file = dataset_dir / eeglab_file
    dest = transpose_file_path(dataset_dir, eeglab_file)
    if dest.exists() and dest.stat().st_mtime >= full_eeglab_file.stat().st_mtime:
        return
    eeg = read_eeglab(full_eeglab_file, mmap=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # mmap() yields (ch, [epochs,] time); write it in C order so that
    # mmap_eeglab(..., ctapdash_order=True) can map it back without a copy.
    arr = eeg.mmap()
    with atomic_write(dest) as staging:
        arr.tofile(staging)


def transpose_file_path(dataset_dir, eeglab_file):
    eeglab_file = Path(eeglab_file)
    if eeglab_file.is_absolute():
        eeglab_file = eeglab_file.relative_to(dataset_dir)
    return cache_path(dataset_dir) / eeglab_file.with_suffix(".ctapdash_order")
