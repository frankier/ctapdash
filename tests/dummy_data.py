"""Deterministic synthetic CTAP files; no participant data is copied."""
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat
from PIL import Image, ImageDraw


def write_eeglab(tmp_path, n_epochs=1, *, stem=None, n_times=4):
    """Write external float32 samples and top-level EEGLAB MAT metadata."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    n_channels = 2
    data = np.arange(
        n_epochs * n_channels * n_times, dtype=np.float32
    ).reshape(n_epochs, n_channels, n_times)
    stem = stem or ("epochs" if n_epochs > 1 else "raw")
    fdt_path = tmp_path / f"{stem}.fdt"
    data.transpose(1, 2, 0).ravel(order="F").tofile(fdt_path)

    chanlocs = np.rec.fromarrays([["A", "B"]], names=["labels"])
    eeg = {
        "trials": n_epochs,
        "srate": 100.0,
        "nbchan": n_channels,
        "data": fdt_path.name,
        "chanlocs": chanlocs,
        "pnts": n_times,
        "xmin": -0.01 if n_epochs > 1 else 0.0,
        "xmax": (n_times - 1) / 100.0 - (0.01 if n_epochs > 1 else 0),
    }
    if n_epochs > 1:
        eeg["event"] = np.rec.fromarrays(
            [np.arange(n_epochs) * n_times + 1.0, ["event"] * n_epochs],
            names=["latency", "type"],
        )
        eeg["epoch"] = np.rec.fromarrays(
            [["event"] * n_epochs], names=["eventtype"]
        )
    else:
        eeg["event"] = np.empty(0)
        eeg["epoch"] = np.empty(0)

    set_path = tmp_path / f"{stem}.set"
    savemat(set_path, eeg, appendmat=False, oned_as="row")
    return set_path


def write_dataset(directory):
    """Write a small TAPPED-style dataset and return its TOML config path."""
    directory = Path(directory).resolve()
    root = directory / "TAPPED"
    participant = "1001P_dummy_intake"
    for step in ("1_load", "2_rough_clean", "3_fine_clean"):
        write_eeglab(root / step, stem=participant, n_times=500)
    # Include epochs as well as continuous recordings in the real readers/viewers.
    write_eeglab(root / "3_fine_clean", 2, stem="1002P_dummy_intake", n_times=100)
    log = root / "logs" / "CTAP_load_data" / f"{participant}.log"
    log.parent.mkdir(parents=True)
    log.write_text("Dummy CTAP pipeline\nLoaded 2 channels and 500 samples.\nProcessing completed.\n")
    for folder, names in {
        "CTAP_peek_data/set1_fun1": [f"{participant}-badChan-scalp.png", f"{participant}-chs1-2.png"],
        "CTAP_blink2event/set3_fun1": [f"{participant}_blink_ERP.png"],
    }.items():
        target = root / "quality_control" / folder
        target.mkdir(parents=True)
        for name in names:
            image = Image.new("RGB", (160, 80), "white")
            ImageDraw.Draw(image).line([(10, 40), (40, 15), (80, 65), (150, 40)], fill="navy", width=3)
            image.save(target / name)
    config = directory / "conf.toml"
    config.write_text('[sources]\n"dummy" = ' + json.dumps(str(root)) + "\n")
    return config


def series(values, times=None, *, name="recording:channel", levels=None):
    from ctapdash.plotting.venn_ts.range_series import ArrayRangeSeries

    values = np.asarray(values, dtype=np.float64)
    if times is None:
        times = np.arange(len(values), dtype=np.float64) * 0.25
    return ArrayRangeSeries(values, times, channel_identity=name, levels=levels)


class FakeMmapRecording:
    ch_names = ["Fz", "Cz", "Pz"]

    def __init__(self, offset=0, times=None):
        self.offset = offset
        self._times = np.arange(12, dtype=float) * 0.25 if times is None else times
        self.mmap_calls = 0

    def mmap(self, *, return_xarray=False):
        import xarray as xr

        self.mmap_calls += 1
        values = np.arange(36, dtype=np.float32).reshape(3, 12) + self.offset
        return xr.DataArray(
            values,
            coords=(self.ch_names, self._times),
            dims=("ch", "time"),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(write_dataset(parser.parse_args().directory))
