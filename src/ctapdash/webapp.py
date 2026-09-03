from collections.abc import AsyncGenerator
from functools import partial
from contextlib import asynccontextmanager
import re
from importlib.resources import files
from pathlib import Path
from ctapdash.pyramid import load_pyramid
import panel.io.resources as panel_resources

from mne import BaseEpochs
from mne.io import BaseRaw

from bokeh.server.asgi import BokehASGI
from starlette_htmx.middleware import HtmxMiddleware
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from ctapdash.config import SETTINGS
from ctapdash.io import read_eeglab, ObservationData
from ctapdash.plotting.mne import set_onionskin_eeg, OnionskinMNEBrowseFigure
from ctapdash.middleware import GlobalRequestMiddleware
from mplbed import mplbed_starlette, safe_html


# importlib.resources works both from a normal install and from inside a
# PyInstaller onedir bundle, where the frozen loader reports a real directory.
_PKG = files("ctapdash")
TEMPLATES_DIR = str(_PKG / "templates")
STATIC_DIR = str(_PKG / "static")
panel_resources.RESOURCE_MODE = "cdn"


def sources_context(request):
    ctx = {
        "sources": SETTINGS.sources,
    }
    source = request.query_params.get("source", "")
    ctx["source"] = source
    if source:
        ctx.update(ObservationData.from_request(request).get_all_steps())
    participant = request.query_params.get("participant")
    if participant is not None:
        ctx["participant"] = participant
    if source and participant:
        ctx["source_participant_qs"] = f"?source={source}&participant={participant}"
    return ctx


templates = Jinja2Templates(directory=TEMPLATES_DIR, context_processors=[sources_context])


def _observation_count(instance):
    if isinstance(instance, BaseEpochs):
        return len(instance) * len(instance.times)
    if isinstance(instance, BaseRaw):
        return instance.n_times
    raise TypeError(f"Expected MNE Raw or Epochs, got {type(instance).__name__}")


def _participant_step_rows(root_path, steps, participant):
    rows = []
    for step_num, step_path in steps:
        path = step_path / (participant + ".set")
        instance = read_eeglab(path)
        print(path, instance)
        rows.append(
            {
                "number": step_num,
                "directory": str(step_path.relative_to(root_path)),
                "observations": _observation_count(instance),
            }
        )
    return rows


async def index(request):
    return templates.TemplateResponse(
        request,
        'index.html',
        context={
            "sources": SETTINGS.sources,
        }
    )


def participant_context(request):
    dataset = ObservationData.from_request(request)
    steps = dataset.get_steps()
    return {
        "source": dataset.source,
        "participant": dataset.participant,
        "steps": steps,
    }


async def participant_steps_fragment(request):
    context = participant_context(request)
    context["view"] = "steps"
    yaxis = request.query_params.get("yaxis", "overdraw")
    context = {
        **context,
        "yaxis": yaxis,
        "yaxis_options": []
    }
    participant = context["participant"]
    steps = context["steps"]
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


def bokeh_document(request, path, *args, **kwargs):
    from bokeh.embed import server_document
    from markupsafe import Markup

    url = str(request.url_for("bokeh", path=path))
    return Markup(server_document(url, relative_urls=True, *args, **kwargs))


async def time_series(request):
    context = participant_context(request)
    context["view"] = "time_series"
    context["time_series"] = bokeh_document(request, "/time-series", arguments={
        "source": context["source"],
        "participant": context["participant"],
    })

    return templates.TemplateResponse(
        request,
        'time_series.html',
        context=context
    )


async def hv_viewer(request):
    context = participant_context(request)
    context["view"] = "hv_viewer"
    context["hv_plot"] = bokeh_document(request, "/hv-viewer", arguments={
        "source": context["source"],
        "participant": context["participant"],
    })

    return templates.TemplateResponse(
        request,
        'hv_viewer.html',
        context=context
    )


def maybe_int(value):
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def time_series_bokeh(doc):
    import numpy as np
    import xarray as xr
    from bokeh.layouts import column
    from bokeh.events import RangesUpdate
    from bokeh.models import (
        ColumnDataSource,
        FixedTicker,
        FullscreenTool,
        HoverTool,
        Range1d,
        WheelZoomTool,
    )
    from bokeh.plotting import figure

    dataset = ObservationData.from_bokeh_doc(doc)
    steps = dataset.get_steps()

    pyramid_path = steps[0][-1] / (dataset.participant + ".set.pyramid")
    ts_dt = xr.open_datatree(pyramid_path, engine="zarr")

    X_PADDING = 0.2  # buffer x-range to reduce update latency with pans and zoom-outs

    def extract_ds(ts_dt, level, channels=None):
        """Extract a dataset at a specific level"""
        ds = ts_dt[str(level)].ds
        return ds if channels is None else ds.sel(ch=channels)

    groups = tuple(
        sorted(
            (group for group in ts_dt.groups if "time" in ts_dt[group].ds),
            key=lambda group: ts_dt[group].ds["time"].size,
            reverse=True,
        )
    )
    num_levels = len(groups)
    coarsest_level = groups[-1]
    time_da = extract_ds(ts_dt, coarsest_level)["time"]
    channels = ts_dt[coarsest_level].ds["ch"].values
    num_channels = len(channels)
    x_range = (time_da.min().item(), time_da.max().item())

    plot = figure(
        x_range=Range1d(*x_range),
        y_range=Range1d(-2, num_channels + 1),
        x_axis_label="Time (s)",
        y_axis_label="Channel",
        min_height=600,
        sizing_mode="stretch_both",
        tools="pan,box_zoom,reset,save",
        active_drag="box_zoom",
        #output_backend="webgl",
    )
    plot.yaxis.ticker = FixedTicker(ticks=list(range(num_channels)))
    plot.yaxis.major_label_overrides = {
        index: str(channel) for index, channel in enumerate(channels)
    }
    plot.ygrid.grid_line_color = None

    wheel_zoom = WheelZoomTool(dimensions="width")
    plot.toolbar.logo = None
    plot.add_tools(wheel_zoom)
    plot.add_tools(FullscreenTool())
    plot.toolbar.active_scroll = wheel_zoom

    hover = HoverTool(
        tooltips=[
            ("ch", "$name"),
            ("time", "$x{0.000} s"),
            ("amplitude", "$y{0.000} µV"),
        ],
        mode="mouse",
        renderers=[],
    )
    plot.add_tools(hover)

    sources = {}
    amplitude_ranges = {}
    for index, channel in enumerate(channels):
        channel_name = str(channel)
        source = ColumnDataSource(data={"time": [], "amplitude": []})
        amplitude_range = Range1d(0, 1)
        channel_plot = plot.subplot(
            x_source=plot.x_range,
            x_target=plot.x_range,
            y_source=amplitude_range,
            # Match HoloViews' subcoordinate_scale=4 around each channel.
            y_target=Range1d(index - 2, index + 2),
        )
        renderer = channel_plot.line(
            "time",
            "amplitude",
            source=source,
            name=channel_name,
            color="black",
            line_width=1,
        )
        hover.renderers.append(renderer)
        sources[channel_name] = source
        amplitude_ranges[channel_name] = amplitude_range

    def update_plot(x_range, width=None, height=None):
        x_padding = (x_range[1] - x_range[0]) * X_PADDING
        time_slice = slice(x_range[0] - x_padding, x_range[1] + x_padding)

        # Pick the nearest level that has at least one sample per horizontal
        # pixel. Before browser layout is known, start with the coarsest level.
        if not width:
            pyramid_level = num_levels - 1
            size = time_da.size
        else:
            sizes = np.array(
                [
                    extract_ds(ts_dt, pyramid_level)["time"].sel(time=time_slice).size
                    for pyramid_level in groups
                ]
            )
            diffs = sizes - width
            pyramid_level = np.argmin(np.where(diffs >= 0, diffs, np.inf))  # nearest higher-res
            size = sizes[pyramid_level]

        plot.title.text = (
            f"[Pyramid Level {pyramid_level} ({x_range[0]:.2f}s - {x_range[1]:.2f}s)]   "
            f"[Time Samples: {size}]  [Plot Size WxH: {width}x{height}]"
        )

        ds = (
            extract_ds(ts_dt, groups[pyramid_level], channels)
            .sel(time=time_slice)
            .load()
        )
        for channel in ds["ch"].values.tolist():
            channel_name = str(channel)
            channel_data = ds.sel(ch=channel)
            amplitudes = np.asarray(channel_data["data"].values)
            sources[channel_name].data = {
                "time": np.asarray(channel_data["time"].values),
                "amplitude": amplitudes,
            }

            finite = amplitudes[np.isfinite(amplitudes)]
            if finite.size:
                start, end = float(finite.min()), float(finite.max())
                if start == end:
                    padding = abs(start) * 0.1 or 1.0
                    start, end = start - padding, end + padding
                amplitude_ranges[channel_name].start = start
                amplitude_ranges[channel_name].end = end

    def plot_dimension(name):
        try:
            return getattr(plot, name) or None
        except ValueError:  # Bokeh raises while layout dimensions are unset.
            return None

    def update_from_viewport(width=None):
        update_plot(
            (plot.x_range.start, plot.x_range.end),
            width or plot_dimension("inner_width"),
            plot_dimension("inner_height"),
        )

    def update_from_range(event):
        update_from_viewport()

    def update_from_size(attr, old, new):
        update_from_viewport(width=new)

    plot.on_event(RangesUpdate, update_from_range)
    plot.on_change("inner_width", update_from_size)

    update_plot(x_range)
    root = column(
        plot,
        sizing_mode="stretch_both",
    )
    doc.add_root(root)


def hv_viewer_bokeh(doc):
    """EEG viewer based on hvPlot (see
    https://hvplot.holoviz.org/user_guide/Large_Timeseries.html).

    Renders channels as vertically stacked curves with ``subcoordinate_y``.
    A Panel dropdown switches between plain WebGL rendering of a downsampled
    pyramid level and Datashader rasterization of the full-resolution data.
    """
    import panel as pn
    import xarray as xr
    import hvplot.xarray  # noqa: F401  (registers the .hvplot accessor)

    dataset = ObservationData.from_bokeh_doc(doc)
    steps = dataset.get_steps()

    ts_dt, groups = load_pyramid(steps[0][-1] / (dataset.participant + ".set"))
    finest_level, coarsest_level = groups[0], groups[-1]

    channels = [str(ch) for ch in ts_dt[coarsest_level].ds["ch"].values]

    # WebGL rendering ships all selected samples to the browser, so use the
    # finest pyramid level that stays below a sane per-channel point budget.
    WEBGL_MAX_POINTS_PER_CHANNEL = 200_000
    webgl_level = coarsest_level
    for level in groups:  # finest -> coarsest
        if ts_dt[level].ds["time"].size <= WEBGL_MAX_POINTS_PER_CHANNEL:
            webgl_level = level
            break

    yticks = [(index, channel) for index, channel in enumerate(channels)]
    common_opts = dict(
        x="time",
        y="data",
        by="ch",
        subcoordinate_y=True,
        responsive=True,
        min_height=600,
        line_width=1,
        yticks=yticks,
        xlabel="Time (s)",
        ylabel="Channel",
        tools=["xwheel_zoom"],
        title=dataset.participant,
    )

    def make_plot(render_mode):
        if render_mode == "Datashader":
            # Rasterize the full-resolution level server-side; only an image
            # sized to the viewport is sent to the browser.
            return (
                ts_dt[finest_level].ds["data"]
                .hvplot.line(
                    rasterize=True,
                    cmap=["black"],
                    colorbar=False,
                    **common_opts,
                )
            )
        return (
            ts_dt[webgl_level].ds["data"]
            .hvplot.line(
                color="black",
                hover_tooltips=[
                    ("Channel", "$label"),
                    ("Time", "$x{0.000} s"),
                    ("Amplitude", "$y{0.000} µV"),
                ],
                **common_opts,
            )
        )

    render_mode = pn.widgets.Select(
        options=["WebGL", "Datashader"],
        value="WebGL",
        name="Render mode",
    )
    plot_pane = pn.panel(make_plot(render_mode.value))
    render_mode.param.watch(
        lambda event: setattr(plot_pane, "object", make_plot(event.new)),
        "value",
    )

    layout = pn.Column(
        pn.Row(render_mode),
        plot_pane,
        sizing_mode="stretch_both",
    )
    doc.add_root(layout.get_root(doc))


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
    from ctapdash.io import qc_to_tree

    dataset = ObservationData.from_request(request)
    qcs = dataset.get_qc()
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
                "eeg": map_encode_qc(dataset.source_path, groups[bit_param]),
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
    dataset = ObservationData.from_request(request)
    logs = dataset.get_logs()
    qc = dataset.get_qc()
    steps = dataset.get_steps()
    step_rows = await run_in_threadpool(
        _participant_step_rows, dataset.source_path, steps, dataset.participant
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
    from ctapdash.plotting.stats_heatmap import participant_descriptive_heatmap
    dataset = ObservationData.from_request(request)
    steps = dataset.get_steps()
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
            participant_descriptive_heatmap, selected_steps, dataset.participant
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
    dataset = ObservationData.from_request(request)
    logs = dataset.get_logs()
    ctx = {
        "view": "logs",
        "logs": logs,
    }
    if "log" in request.query_params:
        log = request.query_params["log"]
        log_path = dataset.source_path / log
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

    bokeh_application = BokehASGI({
        "/time-series": time_series_bokeh,
        "/hv-viewer": hv_viewer_bokeh,
    })

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None, None]:
        # Mounted Starlette applications don't receive lifespan events. Start and
        # stop Bokeh from the parent application's lifespan instead.
        await bokeh_application.core.start()
        try:
            yield
        finally:
            await bokeh_application.core.stop()

    app = Starlette(
        debug=debug,
        routes=[
            Route('/', index, name="index"),
            Mount('/static', app=StaticFiles(directory=STATIC_DIR), name="static"),
            Route('/participant/overview', participant_overview_fragment, name="participant_overview"),
            Route('/participant/statistics', participant_statistics_fragment, name="participant_statistics"),
            Route('/participant/steps', participant_steps_fragment, name="participant_steps"),
            Route('/participant/time-series', time_series, name="time_series"),
            Route('/participant/hv-viewer', hv_viewer, name="hv_viewer"),
            Route('/participant/peeks', participant_peeks_fragment, name="participant_peeks"),
            Route('/participant/log', participant_log, name="participant_log"),
            *setup_routes(),
            Mount("/bokeh", bokeh_application, name="bokeh"),
        ],
        middleware=[
            Middleware(RequireConfigMiddleware),
            Middleware(HtmxMiddleware),
            Middleware(GlobalRequestMiddleware),
        ],
        lifespan=lifespan
    )
    # Installs MplbedMiddleware (which does its own /webagg routing), registers
    # the mplbed_head context processor, and selects the webaggext backend.
    mplbed_starlette.setup(app, templates=templates, prefix="/webagg")
    return app
