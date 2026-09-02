import sys

from mne import BaseEpochs
from pathlib import Path

from ctapdash.io import read_eeglab
from ctapdash.pyramid import mne_to_pyramid, mne_to_rangepyramid


for in_path in sys.argv[1:]:
    print("Processing", in_path)
    eeg = read_eeglab(Path(in_path))
    arr = eeg.mmap(return_xarray=True)
    if not arr.data.flags["C_CONTIGUOUS"]:
        arr = arr.copy(data=arr.data.copy(order="C"))
    mne_to_pyramid(arr, in_path + ".pyramid", [8, 8])
    mne_to_rangepyramid(arr, in_path + ".rangepyramid", [8, 8])
