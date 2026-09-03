"""Run with: bokeh serve --show exp/venn_bokeh.py --args --series A.set Cz --series B.set Cz

Add ``--output-backend webgl`` to render directly into Bokeh's shared WebGL
canvas.  The default canvas backend uses GPU rendering followed by CPU
readback so that Bokeh retains ownership of its 2D canvas.
"""

from __future__ import annotations

import argparse
import sys

from bokeh.document import Document
from bokeh.models import Range1d
from bokeh.plotting import figure

from ctapdash.venn import RecordingRangeSeries, build_manifest, parse_series_args
from ctapdash.vennrender import PageMailbox, VennTimeSeriesRenderer


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
    parser.add_argument(
        "--output-backend",
        choices=("canvas", "webgl"),
        default="canvas",
        help="Bokeh backend; selects Venn readback or shared-WebGL composition",
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
    series = tuple(
        RecordingRangeSeries(selection.path, selection.channel)
        for selection in selections
    )
    manifest = build_manifest(series[0], series[1])
    renderer = VennTimeSeriesRenderer(manifest=manifest)
    plot = figure(
        x_range=Range1d(manifest.time_start, manifest.time_end),
        y_range=Range1d(manifest.initial_y_start, manifest.initial_y_end),
        tools="pan,xwheel_zoom,ywheel_zoom,box_zoom,reset,save",
        active_scroll="xwheel_zoom",
        output_backend=args.output_backend,
        sizing_mode="stretch_both",
        title=f"{series[0].channel_identity} / {series[1].channel_identity}",
    )
    plot.renderers.append(renderer)
    document.add_root(plot)
    document.title = "Venn time series"
    mailbox = PageMailbox(document, renderer, series)
    document.on_session_destroyed(lambda _context, mailbox=mailbox: mailbox.close())


from bokeh.io import curdoc

build_document(curdoc(), sys.argv[1:])
