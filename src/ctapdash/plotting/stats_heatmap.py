import re

from matplotlib import colormaps
from matplotlib.colors import to_hex
from natsort import natsorted
import numpy as np

from ctapdash.io import read_eeglab
from ctapdash.stats import describe_mne


STATISTIC_LABELS = {
    "min": "Min",
    "max": "Max",
    "mean": "Mean",
    "variance": "Var",
    "skewness": "Skew",
    "kurtosis": "Kurt",
}

MAX_HEATMAP_CHANNELS = 32
MIN_HEATMAP_GROUP_CHANNELS = 10
NUMBERED_CHANNEL_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>[0-9]+)$")


def _display_statistic(value):
    return f"{value:.4g}"


def _viridis_gradient():
    stops = ", ".join(
        f"{to_hex(colormaps['viridis'](position))} {position:.0%}"
        for position in np.linspace(0, 1, 9)
    )
    return f"background: linear-gradient(to right, {stops})"


def _split_heatmap_channels(channels, limit=MAX_HEATMAP_CHANNELS):
    """Return the table width and slices, preserving substantial sensor groups."""
    runs = []
    index = 0
    while index < len(channels):
        match = NUMBERED_CHANNEL_RE.match(str(channels[index]))
        if match is None:
            index += 1
            continue

        prefix = match.group("prefix")
        previous_number = int(match.group("number"))
        run_end = index + 1
        while run_end < len(channels):
            next_match = NUMBERED_CHANNEL_RE.match(str(channels[run_end]))
            if (
                next_match is None
                or next_match.group("prefix") != prefix
                or int(next_match.group("number")) != previous_number + 1
            ):
                break
            previous_number += 1
            run_end += 1
        if run_end - index >= MIN_HEATMAP_GROUP_CHANNELS:
            runs.append((index, run_end))
        index = run_end

    largest_group = max((end - start for start, end in runs), default=len(channels))
    maximum = min(limit, largest_group)
    slices = []

    def append_chunks(start, end):
        slices.extend(
            (chunk_start, min(chunk_start + maximum, end))
            for chunk_start in range(start, end, maximum)
        )

    loose_start = 0
    for start, end in runs:
        append_chunks(loose_start, start)
        append_chunks(start, end)
        loose_start = end
    append_chunks(loose_start, len(channels))
    return maximum, slices


def _descriptive_heatmap(summary):
    """Convert a descriptive-statistics Dataset into a Jinja table model."""
    channel_order = []
    index = 0
    while index < len(summary.channel):
        match = NUMBERED_CHANNEL_RE.match(str(summary.channel.values[index]))
        if match is None:
            channel_order.append(index)
            index += 1
            continue

        prefix = match.group("prefix")
        run_end = index + 1
        while run_end < len(summary.channel):
            next_match = NUMBERED_CHANNEL_RE.match(
                str(summary.channel.values[run_end])
            )
            if next_match is None or next_match.group("prefix") != prefix:
                break
            run_end += 1
        channel_order.extend(
            natsorted(
                range(index, run_end),
                key=lambda item: str(summary.channel.values[item]),
            )
        )
        index = run_end
    summary = summary.isel(channel=channel_order)
    rows = []
    for statistic, variable in summary.data_vars.items():
        if statistic == "nobs":
            continue
        values = np.asarray(variable.values, dtype=float)
        finite = np.isfinite(values)
        low = high = None
        if finite.any():
            low = values[finite].min()
            high = values[finite].max()
        value_range = {
            "minimum": "—" if low is None else _display_statistic(low),
            "maximum": "—" if high is None else _display_statistic(high),
            "style": _viridis_gradient() if low is not None else "background: #e2e8f0",
        }

        for recording_index, recording in enumerate(summary.recording.values):
            cells = []
            for channel_index in range(len(summary.channel)):
                value = values[recording_index, channel_index]
                missing = not finite[recording_index, channel_index]
                if missing:
                    style = "background-color: #e2e8f0"
                    title = "Not available"
                else:
                    normalized = 0.0 if high == low else (value - low) / (high - low)
                    style = f"background-color: {to_hex(colormaps['viridis'](normalized))}"
                    title = _display_statistic(value)
                cells.append(
                    {
                        "style": style,
                        "title": title,
                    }
                )
            rows.append(
                {
                    "statistic": statistic,
                    "statistic_label": STATISTIC_LABELS.get(
                        statistic, statistic.replace("_", " ").title()
                    ),
                    "recording": recording.item()
                    if hasattr(recording, "item")
                    else recording,
                    "cells": cells,
                    "range": value_range,
                }
            )

    channels = [str(channel) for channel in summary.channel.values]
    maximum_columns, channel_slices = _split_heatmap_channels(channels)
    groups = []
    for start, end in channel_slices:
        groups.append(
            {
                "channels": channels[start:end],
                "rows": [dict(row, cells=row["cells"][start:end]) for row in rows],
                "padding": maximum_columns - (end - start),
            }
        )

    return {
        "channels": channels,
        "rows": rows,
        "groups": groups,
        "maximum_columns": maximum_columns,
    }


def participant_descriptive_heatmap(steps, participant):
    recordings = [
        read_eeglab(step_path / (participant + ".set"))
        for _, step_path in steps
    ]
    summary = describe_mne(*recordings).assign_coords(
        recording=[step_num for step_num, _ in steps]
    )
    return _descriptive_heatmap(summary)
