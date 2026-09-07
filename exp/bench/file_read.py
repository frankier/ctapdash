from mne.io import read_raw_eeglab, read_epochs_eeglab, read_raw_fif
from mne import read_epochs
from glob import glob

from ctapdash.io import MmapRawEEGLAB, MmapEpochEEGLAB

import pickle
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


def read_eeglab_mmap(path):
    with warnings.catch_warnings(action="ignore"):
        if is_epoched(path):
            return MmapEpochEEGLAB(path)
        else:
            return MmapRawEEGLAB(path)


for set_file in sys.argv[1:]:
    print("#", set_file)
    print("Default eeglab")
    start = time.time()
    eeg = read_eeglab(set_file)
    print("Time taken: {:.2f} seconds".format(time.time() - start))
    print()

    print("Mmap eeglab")
    start = time.time()
    eeg = read_eeglab_mmap(set_file)
    arr = eeg.mmap(return_xarray=True)
    print("Time taken: {:.2f} seconds".format(time.time() - start))
    print()

    pickle.dump(eeg, open(set_file + ".pkl", "wb"))

    print("Pickled load")
    start = time.time()
    eeg = pickle.load(open(set_file + ".pkl", "rb"))
    arr = eeg.mmap(return_xarray=True)
    print("Time taken: {:.2f} seconds".format(time.time() - start))
    print()

    eeg.set_annotations(None)
    pickle.dump(eeg, open(set_file + ".bare.pkl", "wb"))

    print("Pickled load (no annotations)")
    start = time.time()
    eeg = pickle.load(open(set_file + ".bare.pkl", "rb"))
    pickle_end = time.time()
    arr = eeg.mmap() # return_xarray=True
    print("Pickle load : {:.2f} seconds".format(pickle_end - start))
    print("Time taken: {:.2f} seconds".format(time.time() - start))
    print()

    print("fif preload=False")
    fif_file = set_file + ".fif"
    print(fif_file, "preload=False")
    start = time.time()
    eeg = read_fif(fif_file, preload=False)
    print("Time taken: {:.2f} seconds".format(time.time() - start))
    print()

    print("fif preload=True")
    print(fif_file, "preload=True")
    start = time.time()
    eeg = read_fif(fif_file, preload=True)
    print("Time taken: {:.2f} seconds".format(time.time() - start))
    print()
    print()
