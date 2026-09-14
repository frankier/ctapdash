from unittest.mock import patch

import mne
import numpy as np
import pytest
import xarray as xr
from scipy.io import loadmat, savemat

from ctapdash.channels import channel_metadata
from ctapdash.io.eeglab import CtapRawEEGLAB, CtapEpochEEGLAB
from ctapdash.plotting.stats_heatmap import participant_descriptive_heatmap, _split_heatmap_channels
from tests.dummy_data import write_eeglab


def recording(tmp_path, *, epochs=1, names=("Fz", "Cz"), types=("EEG", "Custom"), positions=None):
    path = write_eeglab(tmp_path, n_epochs=epochs)
    eeg = loadmat(path, simplify_cells=True)
    coords = positions if positions is not None else [(np.nan,) * 3] * 2
    eeg["chanlocs"] = np.rec.fromarrays(
        [list(names), list(types), *np.asarray(coords).T],
        names=["labels", "type", "X", "Y", "Z"],
    )
    savemat(path, {key: value for key, value in eeg.items() if not key.startswith("__")})
    return (CtapEpochEEGLAB if epochs > 1 else CtapRawEEGLAB)(path)


def test_union_types_bads_and_first_usable_position(tmp_path):
    early = recording(tmp_path / "1")
    late = recording(tmp_path / "2", names=("Fz", "Pz"), types=("Changed", "EOG"))
    early.info["bads"] = ["Cz"]
    early.info["chs"][1]["loc"][:3] = [0, 0, .09]
    late.info["chs"][0]["loc"][:3] = [0, .04, .08]
    late.info["chs"][1]["loc"][:3] = [0, -.04, .08]
    with patch("ctapdash.channels._head_geometry", return_value=({"points": {}, "outlines": [], "message": ""}, {})) as project:
        result = channel_metadata([(2, late), (1, early)], bads_by_step={2: ["Pz", "unknown"]}, groups={"Middle": ["Fz", "Cz", "unknown"]})
    assert result["channels"] == [{"name": "Fz", "type": "EEG"}, {"name": "Cz", "type": "Custom"}, {"name": "Pz", "type": "EOG"}]
    assert result["bads"] == {"1": ["Cz"], "2": ["Pz"]}
    assert result["groups"] == {"Middle": ["Fz", "Cz"]}
    assert result["steps"] == {"1": ["Fz", "Cz"], "2": ["Fz", "Pz"]}
    np.testing.assert_array_equal(project.call_args.args[0]["Fz"], [0, .04, .08])
    assert channel_metadata([(1, early)], bads_by_step={1: []})["bads"]["1"] == []


@pytest.mark.parametrize("epochs", [1, 2])
def test_ctap_metadata_projects_without_loading_samples(tmp_path, epochs):
    raw = recording(tmp_path, epochs=epochs)
    montage = mne.channels.make_standard_montage("biosemi64").get_positions()["ch_pos"]
    for name, ch in zip(raw.ch_names, raw.info["chs"]):
        ch["loc"][:3] = montage[name]
    with patch.object(type(raw), "mmap", side_effect=AssertionError("Loaded samples")):
        result = channel_metadata([(1, raw)])
    assert set(result["head"]["points"]) == {"Fz", "Cz"}
    assert len(result["head"]["outlines"]) == 4
    assert result["head"]["message"] == ""
    assert np.asarray(list(result["head"]["points"].values())).shape == (2, 2)


def test_regional_grouping_and_stable_geometry(tmp_path):
    from ctapdash.channels import _head_geometry
    montage = mne.channels.make_standard_montage("biosemi64").get_positions()["ch_pos"]
    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "T7", "T8"]
    positions = {name: montage[name] for name in names}
    head, regions = _head_geometry(positions)
    assert set(sum(regions.values(), [])) == set(names)
    assert _head_geometry(positions)[0] == head
    assert head["points"]["Fp1"][1] < head["points"]["O1"][1]
    assert head["points"]["Fp1"][0] < head["points"]["Fp2"][0]


def test_missing_and_failed_positions_retain_checklist(tmp_path):
    raw = recording(tmp_path)
    missing = channel_metadata([(1, raw)])
    assert not missing["head"]["points"]
    assert "No usable" in missing["head"]["message"]
    raw.info["chs"][0]["loc"][:3] = [.02, .03, .08]
    with patch("ctapdash.channels._head_geometry", side_effect=ValueError("bad geometry")):
        failed = channel_metadata([(1, raw)])
    assert failed["channels"] == missing["channels"]
    assert "could not be projected" in failed["head"]["message"]


def test_heatmap_filters_before_normalizing_and_handles_empty():
    stats = xr.Dataset({"mean": (("recording", "channel"), [[0., 10., 100.], [1., np.nan, 90.]])},
                       coords={"channel": ["A", "B", "C"], "participant": ("recording", ["p", "p"]), "step": ("recording", [1, 2])})
    steps = [(1, None), (2, None)]
    full = participant_descriptive_heatmap(steps, "p", stats)
    filtered = participant_descriptive_heatmap(steps, "p", stats, ["B", "A", "unknown"])
    assert filtered["channels"] == ["A", "B"]
    assert filtered["rows"][0]["range"]["maximum"] == "10"
    assert filtered["rows"][0]["cells"][1]["style"] != full["rows"][0]["cells"][1]["style"]
    assert filtered["rows"][1]["cells"][1]["title"] == "Not available"
    empty = participant_descriptive_heatmap(steps, "p", stats, [])
    assert empty["channels"] == [] and empty["groups"] == []
    assert _split_heatmap_channels([]) == (0, [])
