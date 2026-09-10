"""Run with: bokeh serve --show exp/venn_bokeh.py --args --series A.set Cz --series B.set Cz

The experiment requires Bokeh's shared WebGL canvas, matching the dashboard.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import sys

from bokeh.document import Document
from bokeh.models import Range1d
from bokeh.plotting import figure

from venn_ts.venn import VennManifest, parse_series_args
from venn_ts.range_series import (
    RecordingTileSource,
    validate_tile_source_alignment,
)
from venn_ts.renderer import TileCoordinator, VennTimeSeriesRenderer


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebGL Venn time-series viewer")
    parser.add_argument(
        "--series",
        nargs=2,
        action="append",
        metavar=("PATH", "CHANNEL"),
        required=True,
        help="recording path and channel name or zero-based index (exactly twice)",
    )
    args = parser.parse_args(argv)
    try:
        args.series = parse_series_args(args.series)
    except ValueError as error:
        parser.error(str(error))
    return args


def build_document(document: Document, argv: list[str]) -> None:
    args = parse_args(argv)
    selections = args.series
    sources = tuple(RecordingTileSource(selection.path) for selection in selections)
    # Keep the command-line experiment's ability to compare differently named
    # channels while using the same multichannel tile path as the web app.
    common = validate_tile_source_alignment(*sources)
    source_channels = ([selections[0].channel], [selections[1].channel])
    extrema = [
        source.finite_extrema(source.indices(channels), common)[0]
        for source, channels in zip(sources, source_channels)
    ]
    y_start = min(value[0] for value in extrema)
    y_end = max(value[1] for value in extrema)
    if y_start == y_end:
        padding = abs(y_start) * 0.05 or 1.0
        y_start -= padding
        y_end += padding
    factors = tuple(sorted(set(sources[0].range_factors) & set(sources[1].range_factors)))
    page_size = 2048
    manifest = VennManifest(
        dataset_version=sha256("|".join(source.dataset_version for source in sources).encode()).hexdigest()[:20],
        sample_count=common,
        time_start=sources[0].time_start,
        time_end=sources[0].times[common - 1],
        sample_interval=(sources[0].sample_interval + sources[1].sample_interval) / 2,
        source_factors=factors,
        page_size=page_size,
        page_counts=tuple(
            (min(source.level_length("venn", factor) for source in sources) + page_size - 1) // page_size
            for factor in factors
        ),
        initial_y_start=y_start,
        initial_y_end=y_end,
    )
    renderer = VennTimeSeriesRenderer(
        dataset_version=manifest.dataset_version,
        sample_count=manifest.sample_count,
        time_start=manifest.time_start,
        time_end=manifest.time_end,
        sample_interval=manifest.sample_interval,
        source_factors=list(manifest.source_factors),
        page_counts=list(manifest.page_counts),
        range_factors=list(manifest.source_factors),
        range_page_counts=list(manifest.page_counts),
        channel_names=[f"{selections[0].channel} / {selections[1].channel}"],
        amplitude_scales=[1.0],
        amplitude_offsets=[0.0],
        channel_y_mins=[manifest.initial_y_start],
        channel_y_maxs=[manifest.initial_y_end],
        lines_visible=False,
    )
    plot = figure(
        x_range=Range1d(manifest.time_start, manifest.time_end),
        y_range=Range1d(manifest.initial_y_start, manifest.initial_y_end),
        tools="pan,xwheel_zoom,ywheel_zoom,box_zoom,reset,save",
        active_scroll="xwheel_zoom",
        output_backend="webgl",
        sizing_mode="stretch_both",
        title=f"{selections[0].path}:{selections[0].channel} / {selections[1].path}:{selections[1].channel}",
    )
    plot.renderers.append(renderer)
    document.add_root(plot)
    document.title = "Venn time series"
    mailbox = TileCoordinator(
        document,
        renderer,
        sources,
        renderer.channel_names,
        source_channels=source_channels,
    )
    def close_session(_context):
        mailbox.close()

    document.on_session_destroyed(close_session)


from bokeh.io import curdoc

build_document(curdoc(), sys.argv[1:])
