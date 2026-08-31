import sys

from mne import BaseEpochs
from pathlib import Path

from ctapdash.io import read_eeglab
from ctapdash.pyramid import mne_to_pyramid


for in_path in sys.argv[2:]:
    eeg = read_eeglab(Path(in_path))
    if isinstance(eeg, BaseEpochs):
        continue
    mne_to_pyramid(eeg, in_path + ".pyramid", [1, 4])
