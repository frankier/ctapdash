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
        step_full = steps_dict[step]
        path = step_full / (participant + ".set")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        eeg = read_eeglab(path, mmap=False, use_cache=False)
        if yaxis == "normalize":
            scalings = "auto"
        else:
            scalings = None
        context["yaxis_options"].extend(["overdraw", "normalize"])
        if isinstance(eeg, BaseEpochs):
            fig = eeg.plot(show=False, scalings=scalings)
        else:
            context["yaxis_options"].append("clamp")
            if yaxis == "clip":
                clipping = "clamp"
            else:
                clipping = None
            fig = eeg.plot(show=False, scalings=scalings, clipping=clipping)
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


async def venn_time_series(request):
    """Serve the multichannel processing-step comparison viewer."""
    context = participant_context(request)
    context["view"] = "venn_time_series"
    context["venn_time_series"] = bokeh_document(
        request,
        "/venn-time-series",
        arguments={
            "source": context["source"],
            "participant": context["participant"],
        },
    )
    return templates.TemplateResponse(
        request,
        "venn_time_series.html",
        context=context,
    )


def maybe_int(value):
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _comparison_channel_geometry(extrema, mode):
    """Map source amplitudes into stacked channel-lane coordinates.

    The first recording supplies the nominal range. This makes excursions
    introduced by a later processing step visible according to ``mode``.
    """
    if mode not in {"overplot", "stretch", "normalize"}:
        raise ValueError(f"Unknown plotting mode: {mode}")

    prepared = []
    for a_min, a_max, b_min, b_max in extrema:
        nominal_min, nominal_max = float(a_min), float(a_max)
        if nominal_min == nominal_max:
            padding = abs(nominal_min) * 0.05 or 1.0
            nominal_min -= padding
            nominal_max += padding
        actual_min = min(nominal_min, float(b_min))
        actual_max = max(nominal_max, float(b_max))
        prepared.append((nominal_min, nominal_max, actual_min, actual_max))

    geometry = [None] * len(prepared)
    if mode == "stretch":
        cursor = 0.0
        # Put the first selector item at the top of the chart.
        for index in reversed(range(len(prepared))):
            nominal_min, nominal_max, actual_min, actual_max = prepared[index]
            scale = 0.8 / (nominal_max - nominal_min)
            nominal_mid = (nominal_min + nominal_max) / 2
            low = min(-0.4, (actual_min - nominal_mid) * scale)
            high = max(0.4, (actual_max - nominal_mid) * scale)
            offset = cursor + 0.05 - low - nominal_mid * scale
            geometry[index] = {
                "scale": scale,
                "offset": offset,
                "center": nominal_mid * scale + offset,
                "actual_min": actual_min,
                "actual_max": actual_max,
                "outside": actual_min < nominal_min or actual_max > nominal_max,
            }
            cursor += high - low + 0.1
        y_range = (0.0, max(cursor, 1.0))
    else:
        for index, (nominal_min, nominal_max, actual_min, actual_max) in enumerate(prepared):
            center = len(prepared) - index - 0.5
            source_min, source_max = (
                (actual_min, actual_max)
                if mode == "normalize"
                else (nominal_min, nominal_max)
            )
            scale = 0.8 / (source_max - source_min)
            geometry[index] = {
                "scale": scale,
                "offset": center - ((source_min + source_max) / 2) * scale,
                "center": center,
                "actual_min": actual_min,
                "actual_max": actual_max,
                "outside": actual_min < nominal_min or actual_max > nominal_max,
            }
        y_range = (0.0, max(float(len(prepared)), 1.0))
    return geometry, y_range


def venn_time_series_bokeh(doc):
    """Build one WebGL figure containing Venn and line comparison layers."""
    from base64 import b64encode
    from hashlib import sha256

    from bokeh.layouts import column, row
    from bokeh.models import (
        CheckboxButtonGroup,
        CustomAction,
        CustomJS,
        Dialog,
        Div,
        FixedTicker,
        HoverTool,
        InlineStyleSheet,
        MultiChoice,
        Range1d,
        RangeSlider,
        Select,
        Span,
        Toggle,
        WheelZoomTool,
    )
    from bokeh.plotting import figure
    from markupsafe import escape

    from ctapdash.plotting.venn_ts.range_series import (
        RecordingTileSource,
        validate_tile_source_alignment,
    )
    from ctapdash.plotting.venn_ts.renderer import TileCoordinator, VennTimeSeriesRenderer

    dataset = ObservationData.from_bokeh_doc(doc)
    steps = dataset.get_steps()
    if not steps:
        doc.add_root(Div(text="No processing steps are available for this participant."))
        return

    steps_by_number = dict(steps)
    step_values = [str(number) for number, _path in steps]
    step_a = Select(title="Step A (red)", options=step_values, value=step_values[0])
    step_b = Select(
        title="Step B (blue)",
        options=step_values,
        value=step_values[1] if len(step_values) > 1 else step_values[0],
    )
    plotting_mode = Select(
        title="Plotting mode",
        options=["overplot", "stretch", "normalize"],
        value="overplot",
    )
    layers = CheckboxButtonGroup(labels=["Venn", "Lines"], active=[0, 1])
    status = Div(text="", sizing_mode="stretch_width")
    plot_holder = column(sizing_mode="stretch_both")
    source_cache = {}
    active_coordinator = [None]
    active_renderer = [None]
    active_plot = [None]
    viewport = {"x": None, "y": None}

    def path_for(value):
        return steps_by_number[int(value)] / (dataset.participant + ".set")

    def get_source(value):
        path = path_for(value).resolve()
        if path not in source_cache:
            source_cache[path] = RecordingTileSource(path)
        return source_cache[path]

    try:
        initial_source = get_source(step_a.value)
        initial_channels = list(initial_source.channels)
    except Exception as error:
        doc.add_root(Div(text=f"<strong>Unable to open comparison data:</strong> {escape(str(error))}"))
        return

    channel_choice = MultiChoice(
        title="Channels",
        options=initial_channels,
        value=initial_channels,
    )
    channel_toggle = Toggle(
        label="Show channels", active=False, name="channel-dialog-toggle",
    )
    channel_dialog = Dialog(
        title="Channels",
        content=channel_choice,
        visible=False,
        closable=True,
        close_action="hide",
        movable="both",
    )

    def toggle_channels(_attr, _old, visible):
        channel_dialog.visible = visible
        channel_toggle.label = "Hide channels" if visible else "Show channels"

    def sync_channel_toggle(_attr, _old, visible):
        if channel_toggle.active != visible:
            channel_toggle.active = visible

    channel_toggle.on_change("active", toggle_channels)
    channel_dialog.on_change("visible", sync_channel_toggle)

    normal_sidebar_open = Toggle(active=True, visible=False, name="normal-sidebar-open")
    fullscreen_sidebar_open = Toggle(active=False, visible=False, name="fullscreen-sidebar-open")
    sidebar_toggle = Toggle(
        label="« Hide controls", active=True, width=110, name="sidebar-toggle",
    )
    controls_sidebar = column(
        step_a,
        step_b,
        plotting_mode,
        layers,
        channel_toggle,
        width=220,
    )
    sidebar_shell = column(
        sidebar_toggle, controls_sidebar, width=220, sizing_mode="stretch_height",
    )
    plot_panel = column(status, plot_holder, sizing_mode="stretch_both")
    viewer_frame = column(
        row(sidebar_shell, plot_panel, sizing_mode="stretch_both"),
        min_height=695,
        sizing_mode="stretch_both",
        stylesheets=[InlineStyleSheet(css="""
            :host {
                background: white;
            }
            :host(:fullscreen) {
                width: 100vw !important;
                height: 100vh !important;
                overflow: auto;
                padding: 8px;
            }
        """)],
    )
    sidebar_toggle.js_on_change("active", CustomJS(
        args={
            "controls_sidebar": controls_sidebar,
            "sidebar_shell": sidebar_shell,
            "viewer_frame": viewer_frame,
            "normal_state": normal_sidebar_open,
            "fullscreen_state": fullscreen_sidebar_open,
        },
        code="""
            const frame_view = Bokeh.index.find_one(viewer_frame)
            const fullscreen = document.fullscreenElement === frame_view?.el
            const state = fullscreen ? fullscreen_state : normal_state
            state.active = cb_obj.active
            controls_sidebar.visible = cb_obj.active
            sidebar_shell.width = cb_obj.active ? 160 : 30
            cb_obj.label = cb_obj.active ? "« Hide controls" : "»"
        """,
    ))

    def page_counts(sources, layer, factors, common):
        counts = []
        for factor in factors:
            length = min(
                *(source.level_length(layer, factor) for source in sources),
                common // factor,
            )
            counts.append((length + 2047) // 2048)
        return counts

    def style_axes(plot, channels, geometry, mode):
        ticks, labels = [], {}
        for channel, item in zip(channels, geometry):
            center = item["center"]
            ticks.append(center)
            labels[center] = channel
            if mode == "normalize" and item["outside"]:
                values = [item["actual_min"], item["actual_max"]]
                if item["actual_min"] <= 0 <= item["actual_max"]:
                    values.insert(1, 0.0)
                for value in values:
                    tick = value * item["scale"] + item["offset"]
                    ticks.append(tick)
                    label = f"{value:.3g}"
                    labels[tick] = f"{channel} · {label}" if abs(tick - center) < 1e-12 else label
            plot.add_layout(Span(
                location=center - 0.4,
                dimension="width",
                line_color="#dddddd",
                line_width=1,
            ))
            plot.add_layout(Span(
                location=center + 0.4,
                dimension="width",
                line_color="#dddddd",
                line_width=1,
            ))
        plot.yaxis.ticker = FixedTicker(ticks=ticks)
        plot.yaxis.major_label_overrides = labels
        plot.yaxis.axis_label = "Channel"
        plot.ygrid.grid_line_color = None
        plot.xaxis.axis_label = "Time (s)"
        plot.toolbar.logo = None
        xwheel = WheelZoomTool(dimensions="width")
        plot.add_tools(xwheel, WheelZoomTool(dimensions="height"))
        plot.toolbar.active_scroll = xwheel

    def install_toolbar_sizing(plot):
        resize = CustomJS(
            name="toolbar-button-resizer",
            args={"toolbar": plot.toolbar},
            code="""
                const apply_size = (attempt = 0) => {
                    const toolbar_view = Bokeh.index.find_one(toolbar)
                    if (toolbar_view == null) {
                        if (attempt == 0)
                            requestAnimationFrame(() => apply_size(1))
                        return
                    }
                    if (toolbar_view._ctap_buttons_sized)
                        return
                    toolbar_view._ctap_buttons_sized = true
                    for (const button_view of toolbar_view.tool_button_views) {
                        const style = button_view.el.style
                        style.setProperty("--button-width", "40px", "important")
                        style.setProperty("--button-height", "40px", "important")
                        style.setProperty("width", "40px", "important")
                        style.setProperty("height", "40px", "important")
                    }
                }
                apply_size()
            """,
        )
        # This fires once as each initial or rebuilt plot completes layout.
        # The callback guard prevents later responsive layouts from reapplying it.
        plot.js_on_change("inner_width", resize)

    def preserved_range(saved, bounds, fallback):
        """Fit a previous viewport into new bounds without resetting its span."""
        if saved is None:
            return fallback
        bound_start, bound_end = bounds
        start, end = saved
        span = end - start
        bound_span = bound_end - bound_start
        if span <= 0 or span >= bound_span:
            return bounds
        if start < bound_start:
            start, end = bound_start, bound_start + span
        if end > bound_end:
            start, end = bound_end - span, bound_end
        return start, end

    def channel_count_action(plot_range, bounds, channel_count, delta):
        label = f"{delta:+d} channel"
        sign_path = "M3 12h7M6.5 8.5v7" if delta > 0 else "M3 12h7"
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <g fill="none" stroke="#747679" stroke-width="2.5"
               stroke-linecap="round" stroke-linejoin="round">
                <path d="{sign_path}"/>
                <path d="M14 9l3-3v13M14 19h6"/>
            </g>
        </svg>"""
        icon = "data:image/svg+xml;base64," + b64encode(svg.encode()).decode()
        return CustomAction(
            description=label,
            icon=icon,
            callback=CustomJS(
                args={"plot_range": plot_range},
                code=f"""
                    const lower = {bounds[0]}
                    const upper = {bounds[1]}
                    const channel_size = (upper - lower) / {channel_count}
                    const current_span = plot_range.end - plot_range.start
                    const target_span = Math.max(
                        channel_size,
                        Math.min(upper - lower, current_span + ({delta}) * channel_size),
                    )
                    let start = (plot_range.start + plot_range.end - target_span) / 2
                    let end = start + target_span
                    if (start < lower) {{
                        start = lower
                        end = lower + target_span
                    }}
                    if (end > upper) {{
                        end = upper
                        start = upper - target_span
                    }}
                    plot_range.start = start
                    plot_range.end = end
                """,
            ),
        )

    def range_scrollbar(plot_range, *, start, end, value, orientation, **kwargs):
        stylesheets = []
        if orientation == "vertical":
            # Reserve half a handle at each end of the track. Without this,
            # noUiSlider positions the end handles outside the visible box.
            stylesheets.append(InlineStyleSheet(css="""
                :host {
                    overflow: visible !important;
                }
                .bk-input-group {
                    box-sizing: border-box !important;
                    height: 100% !important;
                    padding: 7px 0 !important;
                    overflow: visible !important;
                }
                .noUi-target.noUi-vertical {
                    flex: 1 1 auto !important;
                    height: 100% !important;
                    min-height: 0 !important;
                    margin-top: 0 !important;
                    margin-bottom: 0 !important;
                }
                .noUi-vertical .noUi-handle {
                    top: auto !important;
                    bottom: var(--handle-right) !important;
                }
            """))
        scrollbar = RangeSlider(
            start=start,
            end=end,
            value=value,
            step=max((end - start) / 10_000, 1e-12),
            orientation=orientation,
            direction="rtl" if orientation == "vertical" else "ltr",
            show_value=False,
            tooltips=False,
            bar_color="#d1d5db",
            stylesheets=stylesheets,
            **kwargs,
        )
        scrollbar.js_on_change("value", CustomJS(
            args={"plot_range": plot_range},
            code="""
                const [start, end] = cb_obj.value
                if (plot_range.start !== start) plot_range.start = start
                if (plot_range.end !== end) plot_range.end = end
            """,
        ))
        sync_scrollbar = CustomJS(
            args={"scrollbar": scrollbar, "plot_range": plot_range},
            code="""
                const [old_start, old_end] = scrollbar.value
                if (old_start !== plot_range.start || old_end !== plot_range.end)
                    scrollbar.value = [plot_range.start, plot_range.end]
            """,
        )
        plot_range.js_on_change("start", sync_scrollbar)
        plot_range.js_on_change("end", sync_scrollbar)
        return scrollbar

    def set_layers(_attr, _old, _new):
        renderer = active_renderer[0]
        if renderer is not None:
            renderer.venn_visible = 0 in layers.active
            renderer.lines_visible = 1 in layers.active

    def rebuild(_attr, _old, _new):
        previous_plot = active_plot[0]
        if previous_plot is not None:
            viewport["x"] = (previous_plot.x_range.start, previous_plot.x_range.end)
            viewport["y"] = (previous_plot.y_range.start, previous_plot.y_range.end)
        if active_coordinator[0] is not None:
            active_coordinator[0].close()
            active_coordinator[0] = None
        status.text = "Loading..."
        try:
            source_a, source_b = get_source(step_a.value), get_source(step_b.value)
            common = validate_tile_source_alignment(source_a, source_b)
            common_channels = [
                channel for channel in source_a.channels if channel in source_b.channel_index
            ]
            channel_choice.options = common_channels
            channels = [channel for channel in channel_choice.value if channel in common_channels]
            if list(channel_choice.value) != channels:
                channel_choice.value = channels
                return
            if not channels:
                active_renderer[0] = None
                active_plot[0] = None
                plot_holder.children = [Div(text="Select at least one channel.")]
                status.text = ""
                return

            indices_a = source_a.indices(channels)
            indices_b = source_b.indices(channels)
            extrema_a = source_a.finite_extrema(indices_a, common)
            extrema_b = source_b.finite_extrema(indices_b, common)
            extrema = [(*a, *b) for a, b in zip(extrema_a, extrema_b)]
            geometry, y_range = _comparison_channel_geometry(extrema, plotting_mode.value)
            scales = [item["scale"] for item in geometry]
            offsets = [item["offset"] for item in geometry]
            channel_y_mins = [
                item["actual_min"] * item["scale"] + item["offset"] for item in geometry
            ]
            channel_y_maxs = [
                item["actual_max"] * item["scale"] + item["offset"] for item in geometry
            ]
            range_factors = sorted(set(source_a.range_factors) & set(source_b.range_factors))
            line_factors = sorted(set(source_a.line_factors) & set(source_b.line_factors))
            version = sha256(
                f"{source_a.dataset_version}|{source_b.dataset_version}|{channels}|{common}".encode()
            ).hexdigest()[:20]
            renderer = VennTimeSeriesRenderer(
                dataset_version=version,
                sample_count=common,
                time_start=max(source_a.time_start, source_b.time_start),
                time_end=min(source_a.times[common - 1], source_b.times[common - 1]),
                sample_interval=(source_a.sample_interval + source_b.sample_interval) / 2,
                source_factors=range_factors,
                page_counts=page_counts((source_a, source_b), "venn", range_factors, common),
                range_factors=range_factors,
                range_page_counts=page_counts((source_a, source_b), "venn", range_factors, common),
                line_factors=line_factors,
                line_page_counts=page_counts((source_a, source_b), "line", line_factors, common),
                channel_names=channels,
                amplitude_scales=scales,
                amplitude_offsets=offsets,
                channel_y_mins=channel_y_mins,
                channel_y_maxs=channel_y_maxs,
                venn_visible=0 in layers.active,
                lines_visible=1 in layers.active,
            )
            visible_channel_count = min(6, len(channels))
            if visible_channel_count == len(channels):
                initial_y_range = y_range
            else:
                # Channel order is top-to-bottom, while Bokeh y coordinates
                # increase bottom-to-top. Put the first selected channels in
                # the initial fixed-height viewport.
                initial_y_range = (
                    (geometry[visible_channel_count - 1]["center"]
                     + geometry[visible_channel_count]["center"]) / 2,
                    y_range[1],
                )
            plot_height = max(180, visible_channel_count * 100 + 60)
            x_bounds = (renderer.time_start, renderer.time_end)
            initial_x_range = preserved_range(viewport["x"], x_bounds, x_bounds)
            initial_y_range = preserved_range(viewport["y"], y_range, initial_y_range)
            plot = figure(
                x_range=Range1d(
                    *initial_x_range,
                    bounds=x_bounds,
                ),
                y_range=Range1d(*initial_y_range, bounds=y_range),
                height=plot_height,
                min_height=plot_height,
                sizing_mode="stretch_both",
                tools="box_zoom,reset,save",
                active_drag="box_zoom",
                output_backend="webgl",
            )
            plot.renderers.append(renderer)
            line_a = plot.multi_line(
                xs="xs", ys="ys", source=renderer.line_source_a,
                line_color="red", line_width=1, line_alpha=0.9,
                name=f"step {step_a.value}",
            )
            line_b = plot.multi_line(
                xs="xs", ys="ys", source=renderer.line_source_b,
                line_color="blue", line_width=1, line_alpha=0.9,
                name=f"step {step_b.value}",
            )
            plot.add_tools(HoverTool(
                renderers=[line_a, line_b],
                tooltips=[("Step", "$name"), ("Channel", "@channel")],
                mode="mouse",
            ))
            style_axes(plot, channels, geometry, plotting_mode.value)
            plot.add_tools(
                channel_count_action(plot.y_range, y_range, len(channels), 1),
                channel_count_action(plot.y_range, y_range, len(channels), -1),
            )
            renderer.js_on_change("error", CustomJS(
                args={"status": status},
                code="status.text = cb_obj.error ? `<strong>${cb_obj.error}</strong>` : ''",
            ))
            coordinator = TileCoordinator(doc, renderer, (source_a, source_b), channels)
            horizontal_scrollbar = range_scrollbar(
                plot.x_range,
                start=renderer.time_start,
                end=renderer.time_end,
                value=(plot.x_range.start, plot.x_range.end),
                orientation="horizontal",
                height=35,
                sizing_mode="stretch_width",
            )
            vertical_scrollbar = range_scrollbar(
                plot.y_range,
                start=y_range[0],
                end=y_range[1],
                value=(plot.y_range.start, plot.y_range.end),
                orientation="vertical",
                width=45,
                min_height=plot_height,
                sizing_mode="stretch_height",
            )
            plot_frame = column(
                row(
                    plot,
                    vertical_scrollbar,
                    min_height=plot_height,
                    sizing_mode="stretch_both",
                ),
                horizontal_scrollbar,
                min_height=plot_height + 35,
                sizing_mode="stretch_both",
            )
            plot.add_tools(CustomAction(
                description="Fullscreen",
                icon="fullscreen",
                callback=CustomJS(args={
                    "viewer_frame": viewer_frame,
                    "controls_sidebar": controls_sidebar,
                    "sidebar_shell": sidebar_shell,
                    "sidebar_toggle": sidebar_toggle,
                    "normal_state": normal_sidebar_open,
                    "fullscreen_state": fullscreen_sidebar_open,
                }, code="""
                    const view = Bokeh.index.find_one(viewer_frame)
                    const element = view?.el
                    if (element == null) return
                    const sync_sidebar = () => {
                        const fullscreen = document.fullscreenElement === element
                        const state = fullscreen ? fullscreen_state : normal_state
                        controls_sidebar.visible = state.active
                        sidebar_shell.width = state.active ? 160 : 30
                        sidebar_toggle.active = state.active
                        sidebar_toggle.label = state.active ? "« Hide controls" : "»"
                    }
                    if (element._ctap_sidebar_fullscreen_listener == null) {
                        element._ctap_sidebar_fullscreen_listener = sync_sidebar
                        document.addEventListener("fullscreenchange", sync_sidebar)
                    }
                    if (document.fullscreenElement === element)
                        void document.exitFullscreen()
                    else if (document.fullscreenElement == null)
                        void element.requestFullscreen()
                """),
            ))
            install_toolbar_sizing(plot)
            active_renderer[0] = renderer
            active_coordinator[0] = coordinator
            active_plot[0] = plot
            plot_holder.children = [plot_frame]
            status.text = ""
        except Exception as error:
            active_renderer[0] = None
            active_plot[0] = None
            plot_holder.children = [Div(text=f"<strong>Unable to build comparison:</strong> {escape(str(error))}")]
            status.text = ""

    layers.on_change("active", set_layers)
    for widget in (step_a, step_b, channel_choice, plotting_mode):
        widget.on_change("value", rebuild)

    rebuild(None, None, None)
    doc.add_root(viewer_frame)
    doc.add_root(channel_dialog)
    doc.title = "Processing-step time series comparison"

    def close_session(_context):
        if active_coordinator[0] is not None:
            active_coordinator[0].close()

    doc.on_session_destroyed(close_session)


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
        "/venn-time-series": venn_time_series_bokeh,
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
            Route('/participant/venn-time-series', venn_time_series, name="venn_time_series"),
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
