import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
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


class DescriptiveHeatmapTests(unittest.TestCase):
    def test_uses_viridis_tiles_tooltips_and_a_final_range(self):
        summary = xr.Dataset(
            {
                "nobs": (("recording", "channel"), [[4.0, 4.0], [4.0, np.nan]]),
                "mean": (("recording", "channel"), [[0.0, 5.0], [10.0, np.nan]]),
            },
            coords={"recording": [1, 2], "channel": ["Fz", "Cz"]},
        )

        heatmap = _descriptive_heatmap(summary)

        self.assertEqual(heatmap["channels"], ["Fz", "Cz"])
        self.assertEqual(len(heatmap["rows"]), 2)
        self.assertEqual(
            [(row["statistic"], row["recording"]) for row in heatmap["rows"]],
            [("mean", 1), ("mean", 2)],
        )
        low = heatmap["rows"][0]["cells"][0]
        high = heatmap["rows"][1]["cells"][0]
        self.assertEqual(low["style"], "background-color: #440154")
        self.assertEqual(high["style"], "background-color: #fde725")
        self.assertEqual(high["title"], "10")
        self.assertNotIn("display", high)
        self.assertNotIn("class", high)
        self.assertEqual(heatmap["rows"][1]["cells"][1]["title"], "Not available")
        self.assertEqual(heatmap["rows"][0]["range"]["minimum"], "0")
        self.assertEqual(heatmap["rows"][0]["range"]["maximum"], "10")
        self.assertIn("linear-gradient", heatmap["rows"][0]["range"]["style"])

    def test_splits_long_channel_lists_and_preserves_sensor_groups(self):
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

        self.assertEqual(maximum, 32)
        self.assertEqual(groups, [(0, 32), (32, 64), (64, 70)])

    def test_small_sensor_groups_share_one_band(self):
        channels = [
            *(f"HEOG{i}" for i in range(1, 3)),
            *(f"VEOG{i}" for i in range(1, 3)),
            "L_MASTOID",
            "R_MASTOID",
        ]

        maximum, groups = _split_heatmap_channels(channels)

        self.assertEqual(maximum, 6)
        self.assertEqual(groups, [(0, 6)])

    def test_uses_the_largest_sensor_group_as_the_column_count(self):
        channels = [
            *(f"A{i}" for i in range(1, 13)),
            *(f"B{i}" for i in range(1, 21)),
        ]

        maximum, groups = _split_heatmap_channels(channels)

        self.assertEqual(maximum, 20)
        self.assertEqual(groups, [(0, 12), (12, 32)])

    def test_heatmap_repeats_rows_for_each_channel_group(self):
        channels = [*(f"A{i}" for i in range(1, 33)), *(f"B{i}" for i in range(1, 3))]
        summary = xr.Dataset(
            {"mean": (("recording", "channel"), [np.arange(len(channels))])},
            coords={"recording": [1], "channel": channels},
        )

        heatmap = _descriptive_heatmap(summary)

        self.assertEqual(heatmap["maximum_columns"], 32)
        self.assertEqual([len(group["channels"]) for group in heatmap["groups"]], [32, 2])
        self.assertEqual([group["padding"] for group in heatmap["groups"]], [0, 30])
        self.assertEqual(
            [len(group["rows"][0]["cells"]) for group in heatmap["groups"]],
            [32, 2],
        )

    def test_heatmap_naturally_orders_channels_before_grouping(self):
        expected = [f"A{i}" for i in range(1, 33)]
        channels = sorted(expected)
        summary = xr.Dataset(
            {"mean": (("recording", "channel"), [np.arange(len(channels))])},
            coords={"recording": [1], "channel": channels},
        )

        heatmap = _descriptive_heatmap(summary)

        self.assertEqual(heatmap["groups"][0]["channels"], expected)
        a2_index = heatmap["groups"][0]["channels"].index("A2")
        self.assertEqual(
            heatmap["groups"][0]["rows"][0]["cells"][a2_index]["title"],
            str(channels.index("A2")),
        )

    def test_fragment_uses_one_table_and_hides_filtered_step_column(self):
        channels = [*(f"A{i}" for i in range(1, 33)), "Cz"]
        summary = xr.Dataset(
            {"min": (("recording", "channel"), [np.arange(len(channels))])},
            coords={"recording": [4], "channel": channels},
        )

        rendered = templates.env.get_template("participant_statistics.html").render(
            descriptive_heatmap=_descriptive_heatmap(summary), show_step=False
        )
        compact = " ".join(rendered.split())

        self.assertEqual(rendered.count("<table"), 1)
        self.assertEqual(compact.count("> Stats </th>"), 1)
        self.assertEqual(compact.count("> Min </th>"), 2)
        self.assertEqual(compact.count("> Range </th>"), 1)
        self.assertIn('colspan="31"', compact)
        self.assertNotIn("> Step </th>", compact)


class ParticipantOverviewTests(unittest.TestCase):
    def test_new_statistics_filter_replaces_the_in_flight_load(self):
        source, _, _ = templates.env.loader.get_source(
            templates.env, "participant_overview.html"
        )

        self.assertEqual(
            source.count('hx-sync="#channel-statistics-table:replace"'), 2
        )

    def test_step_rows_strip_dataset_root_and_include_observations(self):
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

        self.assertEqual(
            rows,
            [
                {"number": 1, "directory": "1_import", "observations": 100},
                {"number": 3, "directory": "nested/3_filter", "observations": 24},
            ],
        )

    def test_overview_does_not_calculate_descriptive_statistics(self):
        request = SimpleNamespace(
            query_params={"source": "example", "participant": "sub-01"}
        )
        steps = [(1, Path("/data/1_import"))]
        rendered = object()

        with (
            patch.dict(SETTINGS.sources, {"example": "/data"}, clear=True),
            patch("ctapdash.webapp.collect_logs", return_value=[]),
            patch("ctapdash.webapp.collect_qc", return_value=[]),
            patch("ctapdash.webapp.get_steps_for_participant", return_value=steps),
            patch(
                "ctapdash.webapp._participant_step_rows",
                return_value=[
                    {"number": 1, "directory": "1_import", "observations": 100}
                ],
            ),
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

        self.assertIs(response, rendered)
        describe.assert_not_called()
        context = template_response.call_args.kwargs["context"]
        self.assertEqual(context["step_rows"][0]["observations"], 100)
        self.assertNotIn("descriptive_heatmap", context)

    def test_statistics_fragment_can_filter_to_one_step(self):
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

        with (
            patch.dict(SETTINGS.sources, {"example": "/data"}, clear=True),
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

        self.assertIs(response, rendered)
        build_heatmap.assert_called_once_with([steps[1]], "sub-01")
        self.assertEqual(
            template_response.call_args.args[1], "participant_statistics.html"
        )
        self.assertIs(
            template_response.call_args.kwargs["context"]["descriptive_heatmap"],
            heatmap,
        )
        self.assertFalse(template_response.call_args.kwargs["context"]["show_step"])


if __name__ == "__main__":
    unittest.main()
