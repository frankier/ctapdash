from ..channel_grouping import channel_order, split_channel_columns as _split_heatmap_channels

from matplotlib import colormaps
from matplotlib.colors import to_hex
import numpy as np



STATISTIC_LABELS = {
    "min": "Min",
    "max": "Max",
    "mean": "Mean",
    "variance": "Var",
    "skewness": "Skew",
    "kurtosis": "Kurt",
}

MAX_HEATMAP_CHANNELS = 32



def _display_statistic(value):
    return f"{value:.4g}"


def _viridis_gradient():
    stops = ", ".join(
        f"{to_hex(colormaps['viridis'](position))} {position:.0%}"
        for position in np.linspace(0, 1, 9)
    )
    return f"background: linear-gradient(to right, {stops})"



def _descriptive_heatmap(summary):
    """Convert a descriptive-statistics Dataset into a Jinja table model."""
    summary = summary.isel(channel=channel_order(summary.channel.values))
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


def participant_descriptive_heatmap(steps, participant, stats, channels=None):
    selected = [
        (step_num, index)
        for step_num, _ in steps
        for index in np.flatnonzero(
            (stats["participant"].values == participant)
            & (stats["step"].values == step_num)
        )
    ]
    summary = stats.isel(recording=[index for _, index in selected]).assign_coords(
        recording=[step_num for step_num, _ in selected]
    )
    if channels is not None:
        selected_names = set(channels)
        summary = summary.sel(channel=[name for name in summary.channel.values if str(name) in selected_names])
    return _descriptive_heatmap(summary)
