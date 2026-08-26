import re
from importlib.resources import files
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex
from mne import BaseEpochs
from mne.io import BaseRaw
from natsort import natsorted

from starlette_htmx.middleware import HtmxMiddleware
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from ctapdash.config import SETTINGS
from ctapdash.io import read_eeglab
from ctapdash.stats import describe_mne
from ctapdash.plotting.mne import set_onionskin_eeg, OnionskinMNEBrowseFigure
from ctapdash.middleware import GlobalRequestMiddleware
from mplbed import mplbed_starlette, safe_html


SCALP_REGEX = re.compile("(?P<stem>[^-]+)-badChan-scalp.png")
CH_REGEX = re.compile("(?P<stem>.+)-chs(?P<ch_start>[0-9]+)-(?P<ch_end>[0-9]+).png")

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

# importlib.resources works both from a normal install and from inside a
# PyInstaller onedir bundle, where the frozen loader reports a real directory.
_PKG = files("ctapdash")
TEMPLATES_DIR = str(_PKG / "templates")
STATIC_DIR = str(_PKG / "static")


def sources_context(request):
    ctx = {
        "sources": SETTINGS.sources,
    }
    source = request.query_params.get("source", "")
    ctx["source"] = source
    if source:
        source_path = Path(SETTINGS.sources[source])
        ctx.update(collect_steps(source_path))
    participant = request.query_params.get("participant")
    if participant is not None:
        ctx["participant"] = participant
    if source and participant:
        ctx["source_participant_qs"] = f"?source={source}&participant={participant}"
    return ctx


templates = Jinja2Templates(directory=TEMPLATES_DIR, context_processors=[sources_context])


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


def _participant_descriptive_heatmap(steps, participant):
    recordings = [
        read_eeglab(step_path / (participant + ".set"))
        for _, step_path in steps
    ]
    summary = describe_mne(*recordings, impl="numba").assign_coords(
        recording=[step_num for step_num, _ in steps]
    )
    return _descriptive_heatmap(summary)


def _observation_count(instance):
    if isinstance(instance, BaseEpochs):
        return len(instance) * len(instance.times)
    if isinstance(instance, BaseRaw):
        return instance.n_times
    raise TypeError(f"Expected MNE Raw or Epochs, got {type(instance).__name__}")


def _participant_step_rows(root_path, steps, participant):
    rows = []
    for step_num, step_path in steps:
        instance = read_eeglab(step_path / (participant + ".set"))
        rows.append(
            {
                "number": step_num,
                "directory": str(step_path.relative_to(root_path)),
                "observations": _observation_count(instance),
            }
        )
    return rows


def get_steps_for_participant(root_path, participant):
    steps = []
    for subdir in root_path.iterdir():
        if not subdir.name[0].isnumeric():
            continue
        path = subdir / (participant + ".set")
        if not path.exists():
            continue
        step_num = int(subdir.name.split("_", 1)[0])
        steps.append((step_num, subdir))
    steps = natsorted(steps)
    return steps


def collect_steps(root_path):
    from collections import Counter

    steps = []
    files = Counter()

    for subdir in root_path.iterdir():
        if subdir.name[0].isnumeric():
            step_num = int(subdir.name.split("_", 1)[0])
            steps.append((step_num, subdir))
            for filename in subdir.iterdir():
                if not filename.suffix == ".set":
                    continue
                files[filename.stem] += 1

    steps.sort()
    files = [(-count, filename) for filename, count in files.items()]
    files.sort()

    return {
        "participants": [(filename, -count) for count, filename in files],
        "steps": steps,
    }


async def index(request):
    return templates.TemplateResponse(
        request,
        'index.html',
        context={
            "sources": SETTINGS.sources,
        }
    )


def collect_logs(source_path, participant):
    logs = []
    log_dir = source_path / "logs"
    for root, dirs, files in log_dir.walk():
        for file in files:
            if not file.startswith(participant):
                continue
            file_path = root / file
            rel_path = file_path.relative_to(source_path)
            logs.append(str(rel_path))
    return logs


def collect_qc(source_path, participant):
    qc = []
    qc_dir = source_path / "quality_control"
    for root, dirs, files in qc_dir.walk():
        for file in files:
            if not file.startswith(participant):
                continue
            file_path = root / file
            rel_path = file_path.relative_to(qc_dir)
            qc.append(rel_path)
    qc = natsorted(qc)
    return qc


def qc_to_tree(qcs):
    tree = {}
    for qc in qcs:
        tree.setdefault(qc.parts[0], []).append(qc)

    def group_channels(values):
        groups = {}
        rest = {}
        for value, path in values:
            match = SCALP_REGEX.match(value)
            if match:
                stem = match.group("stem")
                groups.setdefault(stem, {})["scalp"] = (value, path)
                continue
            match = CH_REGEX.match(value)
            if match:
                stem = match.group("stem")
                ch_start = int(match.group("ch_start"))
                ch_end = int(match.group("ch_end"))
                groups.setdefault(stem, {}).setdefault("chs", []).append((ch_start, ch_end, value, path))
                groups[stem]["chs"].sort()
                continue
            rest[value] = path
        return groups, rest

    def form_sets(peek_list):
        peek_dict = {}
        rest = {}
        for directory in peek_list:
            first_seg = directory.parts[1]
            if first_seg.startswith("set"):
                peek_dict.setdefault(directory.parts[1], []).append((directory.parts[-1], directory))
            else:
                sub_peek_dict, rest_dict = form_sets(directory)
                peek_dict.update(sub_peek_dict)
                rest.update(rest_dict)
        return peek_dict, rest

    new_tree = {}
    for root, peek_list in tree.items():
        peek_dict, rest = form_sets(peek_list)
        
        assert len(rest) == 0
        peek_dict = {k: group_channels(v) for k, v in peek_dict.items()}
        new_tree[root] = peek_dict

    return new_tree


async def participant_steps_fragment(request):
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    yaxis = request.query_params.get("yaxis", "overdraw")
    source_path = Path(SETTINGS.sources[source])
    steps = get_steps_for_participant(source_path, participant)
    context = {
        "steps": steps,
        "view": "steps",
        "yaxis": yaxis,
        "yaxis_options": []
    }
    step = request.query_params.get("step", "")
    if step:
        steps_dict = dict(steps)
        try:
            step = int(step)
        except ValueError:
            raise HTTPException(status_code=404, detail="Step must be integer")
        if step not in steps_dict:
            raise HTTPException(status_code=404, detail="Step not found")
        has_prev = (step - 1) in steps_dict
        context["has_prev"] = has_prev
        onionskin = has_prev and request.query_params.get("onionskin") == "onionskin"
        context["onionskin"] = onionskin
        step_full = steps_dict[step]
        path = step_full / (participant + ".set")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        eeg = read_eeglab(path)
        if onionskin:
            path = steps_dict[step - 1] / (participant + ".set")
            if not path.exists():
                raise HTTPException(status_code=404, detail="Path not found")
            prev_eeg = read_eeglab(path)
            set_onionskin_eeg(prev_eeg)
        if yaxis == "normalize":
            scalings = "auto"
        else:
            scalings = None
        context["yaxis_options"].extend(["overdraw", "normalize"])
        if isinstance(eeg, BaseEpochs):
            fig = eeg.plot(show=False, scalings=scalings, figure_class=OnionskinMNEBrowseFigure)
        else:
            context["yaxis_options"].append("clamp")
            if yaxis == "clip":
                clipping = "clamp"
            else:
                clipping = None
            fig = eeg.plot(show=False, scalings=scalings, clipping=clipping, figure_class=OnionskinMNEBrowseFigure)
        context["eeg_fig"] = safe_html.figure_html(fig, on_close="msg_discrete", prevent_default_navigation=True)
        context["current_step"] = step
    return templates.TemplateResponse(
        request,
        'participant_steps.html',
        context=context,
    )


def trim(img):
    """Crop away the uniform border, as ImageMagick's trim() did.

    The reference colour is the top-left pixel, matching ImageMagick. Unlike
    ImageMagick there is no fuzz tolerance, which is fine for CTAP's flat-
    background QC plots.
    """
    from PIL import Image, ImageChops

    if img.mode in ("RGBA", "LA"):
        bbox = img.getchannel("A").getbbox()
        if bbox:
            return img.crop(bbox)
    rgb = img.convert("RGB")
    background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
    bbox = ImageChops.difference(rgb, background).getbbox()
    return img.crop(bbox) if bbox else img


def encode_qc(source_path, path):
    from PIL import Image
    import base64
    import io

    filename = source_path / "quality_control" / path
    with Image.open(filename) as img:
        img.load()
        out = io.BytesIO()
        trim(img).save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode("utf-8")


def map_encode_qc(source_path, val):
    if isinstance(val, Path):
        return encode_qc(source_path, val)
    elif isinstance(val, list):
        return [map_encode_qc(source_path, v) for v in val]
    elif isinstance(val, tuple):
        return tuple(map_encode_qc(source_path, v) for v in val)
    elif isinstance(val, dict):
        return {k: map_encode_qc(source_path, v) for k, v in val.items()}
    else:
        return val


async def participant_peeks_fragment(request):
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    source_path = Path(SETTINGS.sources[source])
    qcs = collect_qc(source_path, participant)
    tree = qc_to_tree(qcs)
    peek_param = request.query_params.get("peek")
    set_param = request.query_params.get("set")
    if set_param is None:
        peek_tree = tree.get(peek_param)
        if peek_tree is not None and len(peek_tree) > 0:
            set_param = list(peek_tree.keys())[0]
    bit_param = request.query_params.get("bit")
    if bit_param is None:
        groupsrest = tree.get(peek_param, {}).get(set_param)
        if groupsrest is not None:
            groups, rest = groupsrest
            if len(groups) > 0:
                bit_param = list(groups.keys())[0]
            if len(rest) > 0:
                bit_param = list(rest.keys())[0]
    qcs = [str(qc) for qc in qcs]
    context = {
        "view": "peeks",
        "qcs": qcs,
        "tree": tree,
        "peek_param": peek_param,
        "set_param": set_param,
        "bit_param": bit_param,
    }
    if peek_param:
        groups, rest = tree.get(peek_param, {}).get(set_param, ({}, {}))
        if bit_param in groups:
            context.update({
                "qc_type": "eeg",
                "eeg": map_encode_qc(source_path, groups[bit_param]),
            })
        elif bit_param in rest:
            path = rest[bit_param]
            context.update({
                "qc_type": "image",
                "path": str(path),
                "encoded_string": encode_qc(source_path, path),
            })
        elif "set" in request.query_params and "bit" in request.query_params:
            raise HTTPException(status_code=404)
    return templates.TemplateResponse(
        request,
        'participant_peeks.html',
        context=context
    )


async def participant_overview_fragment(request):
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    source_path = Path(SETTINGS.sources[source])
    logs = collect_logs(source_path, participant)
    qc = collect_qc(source_path, participant)
    steps = get_steps_for_participant(source_path, participant)
    step_rows = await run_in_threadpool(
        _participant_step_rows, source_path, steps, participant
    )
    return templates.TemplateResponse(
        request,
        'participant_overview.html',
        context={
            "view": "overview",
            "logs": logs,
            "qc": qc,
            "steps": steps,
            "step_rows": step_rows,
        }
    )


async def participant_statistics_fragment(request):
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    source_path = Path(SETTINGS.sources[source])
    steps = get_steps_for_participant(source_path, participant)
    selected_step = request.query_params.get("step", "all")
    selected_steps = steps
    if selected_step != "all":
        try:
            step_num = int(selected_step)
        except ValueError:
            raise HTTPException(status_code=404, detail="Step must be integer or all")
        steps_by_number = dict(steps)
        if step_num not in steps_by_number:
            raise HTTPException(status_code=404, detail="Step not found")
        selected_steps = [(step_num, steps_by_number[step_num])]

    descriptive_heatmap = None
    if selected_steps:
        descriptive_heatmap = await run_in_threadpool(
            _participant_descriptive_heatmap, selected_steps, participant
        )
    return templates.TemplateResponse(
        request,
        "participant_statistics.html",
        context={
            "descriptive_heatmap": descriptive_heatmap,
            "show_step": selected_step == "all",
        },
    )


async def participant_log(request):
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    source_path = Path(SETTINGS.sources[source])
    logs = collect_logs(source_path, participant)
    ctx = {
        "view": "logs",
        "logs": logs,
    }
    if "log" in request.query_params:
        log = request.query_params["log"]
        log_path = source_path / log
        with open(log_path) as f:
            content = f.read()
        ctx.update({
            "current_log_file": log,
            "content": content,
        })
    return templates.TemplateResponse(
        request,
        'participant_log.html',
        context=ctx
    )


def create_app(debug=False):
    from ctapdash.setup_ui import RequireConfigMiddleware, setup_routes

    app = Starlette(
        debug=debug,
        routes=[
            Route('/', index, name="index"),
            Mount('/static', app=StaticFiles(directory=STATIC_DIR), name="static"),
            Route('/participant/overview', participant_overview_fragment, name="participant_overview"),
            Route('/participant/statistics', participant_statistics_fragment, name="participant_statistics"),
            Route('/participant/steps', participant_steps_fragment, name="participant_steps"),
            Route('/participant/peeks', participant_peeks_fragment, name="participant_peeks"),
            Route('/participant/log', participant_log, name="participant_log"),
            *setup_routes(),
        ],
        middleware=[
            Middleware(RequireConfigMiddleware),
            Middleware(HtmxMiddleware),
            Middleware(GlobalRequestMiddleware),
        ],
    )
    # Installs MplbedMiddleware (which does its own /webagg routing), registers
    # the mplbed_head context processor, and selects the webaggext backend.
    mplbed_starlette.setup(app, templates=templates, prefix="/webagg")
    return app
