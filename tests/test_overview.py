import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from ctapdash.config import SETTINGS
from ctapdash.webapp import (
    _descriptive_heatmap,
    _participant_step_rows,
    _split_heatmap_channels,
    participant_overview_fragment,
    participant_statistics_fragment,
    templates,
)


def test_uses_viridis_tiles_tooltips_and_a_final_range():
    summary = xr.Dataset(
        {
            "nobs": (("recording", "channel"), [[4.0, 4.0], [4.0, np.nan]]),
            "mean": (("recording", "channel"), [[0.0, 5.0], [10.0, np.nan]]),
        },
        coords={"recording": [1, 2], "channel": ["Fz", "Cz"]},
    )

    heatmap = _descriptive_heatmap(summary)

    assert heatmap["channels"] == ["Fz", "Cz"]
    assert len(heatmap["rows"]) == 2
    assert [(row["statistic"], row["recording"]) for row in heatmap["rows"]] == [
        ("mean", 1),
        ("mean", 2),
    ]
    low = heatmap["rows"][0]["cells"][0]
    high = heatmap["rows"][1]["cells"][0]
    assert low["style"] == "background-color: #440154"
    assert high["style"] == "background-color: #fde725"
    assert high["title"] == "10"
    assert "display" not in high
    assert "class" not in high
    assert heatmap["rows"][1]["cells"][1]["title"] == "Not available"
    assert heatmap["rows"][0]["range"]["minimum"] == "0"
    assert heatmap["rows"][0]["range"]["maximum"] == "10"
    assert "linear-gradient" in heatmap["rows"][0]["range"]["style"]


def test_splits_long_channel_lists_and_preserves_sensor_groups():
    channels = [
        *(f"A{i}" for i in range(1, 33)),
        *(f"B{i}" for i in range(1, 33)),
        "HEOG1",
        "HEOG2",
        "VEOG1",
        "VEOG2",
        "L_MASTOID",
        "R_MASTOID",
    ]

    maximum, groups = _split_heatmap_channels(channels)

    assert maximum == 32
    assert groups == [(0, 32), (32, 64), (64, 70)]


def test_small_sensor_groups_share_one_band():
    channels = [
        *(f"HEOG{i}" for i in range(1, 3)),
        *(f"VEOG{i}" for i in range(1, 3)),
        "L_MASTOID",
        "R_MASTOID",
    ]

    maximum, groups = _split_heatmap_channels(channels)

    assert maximum == 6
    assert groups == [(0, 6)]


def test_uses_the_largest_sensor_group_as_the_column_count():
    channels = [
        *(f"A{i}" for i in range(1, 13)),
        *(f"B{i}" for i in range(1, 21)),
    ]

    maximum, groups = _split_heatmap_channels(channels)

    assert maximum == 20
    assert groups == [(0, 12), (12, 32)]


def test_heatmap_repeats_rows_for_each_channel_group():
    channels = [*(f"A{i}" for i in range(1, 33)), *(f"B{i}" for i in range(1, 3))]
    summary = xr.Dataset(
        {"mean": (("recording", "channel"), [np.arange(len(channels))])},
        coords={"recording": [1], "channel": channels},
    )

    heatmap = _descriptive_heatmap(summary)

    assert heatmap["maximum_columns"] == 32
    assert [len(group["channels"]) for group in heatmap["groups"]] == [32, 2]
    assert [group["padding"] for group in heatmap["groups"]] == [0, 30]
    assert [len(group["rows"][0]["cells"]) for group in heatmap["groups"]] == [32, 2]


def test_heatmap_naturally_orders_channels_before_grouping():
    expected = [f"A{i}" for i in range(1, 33)]
    channels = sorted(expected)
    summary = xr.Dataset(
        {"mean": (("recording", "channel"), [np.arange(len(channels))])},
        coords={"recording": [1], "channel": channels},
    )

    heatmap = _descriptive_heatmap(summary)

    assert heatmap["groups"][0]["channels"] == expected
    a2_index = heatmap["groups"][0]["channels"].index("A2")
    assert (
        heatmap["groups"][0]["rows"][0]["cells"][a2_index]["title"]
        == str(channels.index("A2"))
    )


def test_fragment_uses_one_table_and_hides_filtered_step_column():
    channels = [*(f"A{i}" for i in range(1, 33)), "Cz"]
    summary = xr.Dataset(
        {"min": (("recording", "channel"), [np.arange(len(channels))])},
        coords={"recording": [4], "channel": channels},
    )

    rendered = templates.env.get_template("participant_statistics.html").render(
        descriptive_heatmap=_descriptive_heatmap(summary), show_step=False
    )
    compact = " ".join(rendered.split())

    assert rendered.count("<table") == 1
    assert compact.count("> Stats </th>") == 1
    assert compact.count("> Min </th>") == 2
    assert compact.count("> Range </th>") == 1
    assert 'colspan="31"' in compact
    assert "> Step </th>" not in compact


def test_new_statistics_filter_replaces_the_in_flight_load():
    source, _, _ = templates.env.loader.get_source(
        templates.env, "participant_overview.html"
    )

    assert source.count('hx-sync="#channel-statistics-table:replace"') == 2


def test_step_rows_strip_dataset_root_and_include_observations():
    root = Path("/data/dataset")
    steps = [
        (1, root / "1_import"),
        (3, root / "nested" / "3_filter"),
    ]

    with (
        patch("ctapdash.webapp.read_eeglab", side_effect=[object(), object()]),
        patch("ctapdash.webapp._observation_count", side_effect=[100, 24]),
    ):
        rows = _participant_step_rows(root, steps, "sub-01")

    assert rows == [
        {"number": 1, "directory": "1_import", "observations": 100},
        {"number": 3, "directory": "nested/3_filter", "observations": 24},
    ]


def test_overview_does_not_calculate_descriptive_statistics(monkeypatch):
    request = SimpleNamespace(
        query_params={"source": "example", "participant": "sub-01"}
    )
    steps = [(1, Path("/data/1_import"))]
    step_rows = [{"number": 1, "directory": "1_import", "observations": 100}]
    rendered = object()

    monkeypatch.setattr(SETTINGS, "sources", {"example": "/data"})
    with (
        patch("ctapdash.webapp.collect_logs", return_value=[]),
        patch("ctapdash.webapp.collect_qc", return_value=[]),
        patch("ctapdash.webapp.get_steps_for_participant", return_value=steps),
        patch("ctapdash.webapp._participant_step_rows", return_value=step_rows),
        patch("ctapdash.webapp.describe") as describe,
        patch(
            "ctapdash.webapp.run_in_threadpool",
            side_effect=lambda function, *args: function(*args),
        ),
        patch(
            "ctapdash.webapp.templates.TemplateResponse", return_value=rendered
        ) as template_response,
    ):
        response = asyncio.run(participant_overview_fragment(request))

    assert response is rendered
    describe.assert_not_called()
    context = template_response.call_args.kwargs["context"]
    assert context["step_rows"][0]["observations"] == 100
    assert "descriptive_heatmap" not in context


def test_statistics_fragment_can_filter_to_one_step(monkeypatch):
    request = SimpleNamespace(
        query_params={
            "source": "example",
            "participant": "sub-01",
            "step": "3",
        }
    )
    steps = [(1, Path("/data/1_import")), (3, Path("/data/3_filter"))]
    heatmap = {"channels": ["Cz"], "rows": []}
    rendered = object()

    monkeypatch.setattr(SETTINGS, "sources", {"example": "/data"})
    with (
        patch("ctapdash.webapp.get_steps_for_participant", return_value=steps),
        patch(
            "ctapdash.webapp._participant_descriptive_heatmap",
            return_value=heatmap,
        ) as build_heatmap,
        patch(
            "ctapdash.webapp.run_in_threadpool",
            side_effect=lambda function, *args: function(*args),
        ),
        patch(
            "ctapdash.webapp.templates.TemplateResponse", return_value=rendered
        ) as template_response,
    ):
        response = asyncio.run(participant_statistics_fragment(request))

    assert response is rendered
    build_heatmap.assert_called_once_with([steps[1]], "sub-01")
    assert template_response.call_args.args[1] == "participant_statistics.html"
    assert (
        template_response.call_args.kwargs["context"]["descriptive_heatmap"] is heatmap
    )
    assert not template_response.call_args.kwargs["context"]["show_step"]
