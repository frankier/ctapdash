from ctapdash.plotting.time_series import venn_time_series
from ctapdash.io.eeglab import read_eeglab

from pathlib import Path
import sys


eegs = []
for in_path in sys.argv[1:]:
    eeg = read_eeglab(Path(in_path))
    eegs.append(eeg)
venn_time_series(eegs, 2560, 1440, ())
