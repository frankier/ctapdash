from ctapdash.io.paths import DatasetPaths, transpose_file_path  # compatibility export
from ctapdash.io.recording import RecordingData
from ctapdash.io.utils import atomic_write


def write_transpose(dataset_dir, eeglab_file, *, metadata_validated=False):
    recording = RecordingData(
        DatasetPaths(dataset_dir).recording(eeglab_file), metadata_validated
    )
    dest = recording.paths.transpose
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = recording.read_metadata().mmap()
    with atomic_write(dest, overwrite=True) as staging:
        arr.tofile(staging)
