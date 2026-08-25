from mne.io import read_raw_eeglab, read_epochs_eeglab, read_raw_fif
from mne import read_epochs
from glob import glob

import warnings
import time
import sys


def is_epoched(path):
    import pymatreader
    return len(pymatreader.read_mat(path, "epoch").get("epoch", ())) > 0


def read_eeglab(path):
    with warnings.catch_warnings(action="ignore"):
        if is_epoched(path):
            return read_epochs_eeglab(path)
        else:
            return read_raw_eeglab(path, preload=False)


def read_fif(path, preload=False):
    import warnings
    with warnings.catch_warnings(action="ignore"):
        try:
            return read_epochs(path, preload=preload)
        except Exception:
            return read_raw_fif(path, preload=preload)


for set_file in glob(sys.argv[1] + "/**/*.set", recursive=True):
    start = time.time()
    eeg = read_eeglab(set_file)
    print("Time taken: {:.2f} seconds".format(time.time() - start))
    print()

    fif_file = set_file + ".fif"
    print(fif_file, "preload=False")
    start = time.time()
    eeg = read_fif(fif_file, preload=False)
    print("Time taken: {:.2f} seconds".format(time.time() - start))
    print()

    print(fif_file, "preload=True")
    start = time.time()
    eeg = read_fif(fif_file, preload=True)
    print("Time taken: {:.2f} seconds".format(time.time() - start))
    print()
