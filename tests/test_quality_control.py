from pathlib import Path

import pytest

from ctapdash.io.paths import collect_qc, qc_step_options, qc_to_tree


def test_collect_qc_matches_flat_and_recording_directory_plots(tmp_path):
    plots = [
        "CTAP_fir_filter/set1_fun5/subject_filter.png",
        "CTAP_peek_data/set3_fun5/subject/EEG_allchan_stats.png",
    ]
    for relative in [
        *plots,
        "CTAP_peek_data/set3_fun5/subject_other/EEG_allchan_stats.png",
        "CTAP_peek_data/set3_fun5/subject/signal.mat",
        "CTAP_peek_data/set3_fun5/subject/peek_times.txt",
    ]:
        path = tmp_path / "quality_control" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert collect_qc(tmp_path, "subject") == [Path(plot) for plot in plots]


@pytest.mark.parametrize("participant", ["subject", "subject_other"])
def test_collect_qc_filter_diagnostics(tmp_path, participant):
    for step in ("set1_fun5", "set1_fun6"):
        for function in ("CTAP_fir_filter", "CTAP_peek_data"):
            directory = tmp_path / "quality_control" / function / step
            directory.mkdir(parents=True)
            for name in (
                "filter_effects_subject.png",
                "filter_effects_subject_other.png",
                "filter_properties.png",
                "filter_properties.mat",
                "unrelated.png",
            ):
                (directory / name).touch()

    qc = collect_qc(tmp_path, participant)
    expected = [
        Path("CTAP_fir_filter") / step / name
        for step in ("set1_fun5", "set1_fun6")
        for name in (f"filter_effects_{participant}.png", "filter_properties.png")
    ]
    assert qc == expected
    assert qc_to_tree(qc) == {
        "CTAP_fir_filter": {
            step: ({}, {path.name: path for path in expected if path.parts[1] == step})
            for step in ("set1_fun5", "set1_fun6")
        }
    }


def test_qc_names_prefer_logged_ids_and_fall_back_when_ambiguous(tmp_path):
    for directory in ("1_load", "2_rough_clean", "02_other", "3_fine_clean"):
        (tmp_path / directory).mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "runlog_a.txt").write_text(
        "STEP SET 3_fine_clean\n"
        "Saving diagnostics to 'C:\\analysis\\quality_control\\CTAP_peek_data\\set1_fun2\\case'\n"
        "Saving diagnostics to '/analysis/quality_control/CTAP_peek_data/set3_fun7-badchans/case'\n"
    )
    (logs / "runlog_b.txt").write_text(
        "STEP SET 2_rough_clean\n"
        "Saving diagnostics to '/analysis/quality_control/CTAP_peek_data/set3_fun7-badchans/case'\n"
    )
    steps = ["set1_fun1", "set1_fun2", "set2_fun1", "set3_fun7-badchans", "set4_fun1"]

    assert qc_step_options(tmp_path, "CTAP_peek_data", steps) == {
        "1_load": [("set1_fun1", "Step 1")],
        "3_fine_clean": [("set1_fun2", "Step 2")],
        "Step set 2": [("set2_fun1", "Step 1")],
        "Step set 3": [("set3_fun7-badchans", "Step 7 · badchans")],
        "Step set 4": [("set4_fun1", "Step 1")],
    }
