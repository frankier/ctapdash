from ctapdash.io.paths import ObservationData


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

#: Nominal plot-frame width in device pixels used for the deepest-zoom floor.
NOMINAL_FRAME_WIDTH = 1200
#: Deepest zoom keeps each sample within this many pixels.
MAX_PIXELS_PER_SAMPLE = 16


def min_zoom_span(sample_interval: float, time_end: float, data_span: float):
    """Smallest x span the UI may zoom in to.

    Zooming stops at the first of two walls: each sample spans at most
    ``MAX_PIXELS_PER_SAMPLE`` pixels, and each pixel must advance the x
    position by at least one float32 ULP at the worst-case time magnitude so
    the shader's pixel-to-time mapping cannot collapse.  The floor never
    exceeds the data span, so short recordings stay fully zoomable.
    """
    import numpy as np

    if sample_interval <= 0 or data_span <= 0:
        return 0.0
    per_sample = NOMINAL_FRAME_WIDTH * sample_interval / MAX_PIXELS_PER_SAMPLE
    per_ulp = NOMINAL_FRAME_WIDTH * float(np.spacing(np.float32(abs(time_end))))
    return min(max(per_sample, per_ulp), data_span)


def floored_range(viewport, floor, bounds):
    """Widen a preserved viewport that is narrower than the zoom floor."""
    start, end = viewport
    if floor <= 0 or end - start >= floor:
        return viewport
    lower, upper = bounds
    start = max(lower, (start + end) / 2 - floor / 2)
    end = min(upper, start + floor)
    if end - start < floor:
        start = end - floor
    return start, end


def _build_comparison(doc, dataset, stats):
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
        Range1d,
        Select,
        Toggle,
        WheelZoomTool,
    )
    from bokeh.plotting import figure
    from markupsafe import escape

    from venn_ts.range_series import (
        RecordingTileSource,
        validate_tile_source_alignment,
    )
    from venn_ts.renderer import TileCoordinator, VennTimeSeriesRenderer

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
    layers = CheckboxButtonGroup(labels=["Venn", "Lines"], active=[0])
    guides_toggle = Toggle(label="Guides", active=True, name="guides-toggle")
    active_guides = [None]
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
            source_cache[path] = RecordingTileSource(path, recording_data=dataset.get_recording(value))
        return source_cache[path]

    try:
        initial_source = get_source(step_a.value)
        initial_channels = list(initial_source.channels)
    except Exception as error:
        doc.add_root(Div(text=f"<strong>Unable to open comparison data:</strong> {escape(str(error))}"))
        raise

    import json
    from ctapdash.channels import participant_channel_metadata
    from venn_ts.channel_selector import ChannelSelector

    channel_metadata = participant_channel_metadata(dataset)
    channel_choice = ChannelSelector(
        metadata_json=json.dumps(channel_metadata),
        available=initial_channels,
        value=[channel["name"] for channel in channel_metadata["channels"]],
        width=900, height=520,
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
    channel_dialog.js_on_change("visible", CustomJS(args={"dialog": channel_dialog}, code="""
        queueMicrotask(() => {
            if (!dialog.visible) return
            const el = Bokeh.index.find_one(dialog)?.el
            const fullscreen = document.fullscreenElement
            if (el != null && fullscreen != null)
                (fullscreen.shadowRoot ?? fullscreen).append(el)
        })
    """))

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
        guides_toggle,
        channel_toggle,
        width=220,
    )
    sidebar_shell = column(
        sidebar_toggle, controls_sidebar, width=230, sizing_mode="stretch_height",
        styles={"padding-top": "10px", "padding-left": "10px"},
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
            sidebar_shell.width = cb_obj.active ? 230 : 40
            cb_obj.width = cb_obj.active ? 110 : 30
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

    def set_guides(_attr, _old, _new):
        if active_guides[0] is None:
            return
        renderer, axis, guide_labels = active_guides[0]
        renderer.visible = guides_toggle.active
        labels = dict(guide_labels) if guides_toggle.active else {}
        # Channel names take precedence when zero coincides with a guide.
        labels.update(axis.channel_labels)
        axis.ticker.ticks = sorted(labels)
        axis.major_label_overrides = labels

    guides_toggle.on_change("active", set_guides)

    def style_axes(plot, channels, geometry):
        from venn_ts.channel_axis import ChannelAxis

        old_axis = plot.yaxis[0]
        plot.left.remove(old_axis)
        axis = ChannelAxis(
            channel_labels={item["offset"]: channel for channel, item in zip(channels, geometry)},
            ticker=FixedTicker(),
            avoid_overlap=True,
        )
        plot.add_layout(axis, "left")
        guide_positions = [item["center"] + delta for item in geometry for delta in (-0.4, 0.4)]
        guide_labels = {
            position: f"{(position - item['offset']) / item['scale']:.3g}"
            for item in geometry
            for position in (item["center"] - 0.4, item["center"] + 0.4)
        }
        guides = plot.hspan(
            y=guide_positions,
            line_color="#dddddd",
            line_width=1,
            level="underlay",
        )
        active_guides[0] = (guides, axis, guide_labels)
        set_guides(None, None, None)
        plot.yaxis.axis_label = None
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
        active_guides[0] = None
        status.text = "Loading..."
        try:
            source_a, source_b = get_source(step_a.value), get_source(step_b.value)
            common = validate_tile_source_alignment(source_a, source_b)
            common_channels = [
                channel for channel in source_a.channels if channel in source_b.channel_index
            ]
            channel_choice.available = common_channels
            channel_choice.bads_json = json.dumps({
                step: channel_metadata["bads"].get(step, [])
                for step in dict.fromkeys((step_a.value, step_b.value))
            })
            selected_names = set(channel_choice.value)
            channels = [channel for channel in common_channels if channel in selected_names]
            if not channels:
                active_renderer[0] = None
                active_plot[0] = None
                plot_holder.children = [Div(text="Select at least one channel.")]
                status.text = ""
                return

            extrema_a = _stats_extrema(stats, dataset.participant, step_a.value, channels)
            extrema_b = _stats_extrema(stats, dataset.participant, step_b.value, channels)
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
                level="glyph",
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
            visible_channel_count = min(16, len(channels))
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
            plot_height = max(180, visible_channel_count * 50 + 60)
            x_bounds = (renderer.time_start, renderer.time_end)
            x_floor = min_zoom_span(
                renderer.sample_interval, renderer.time_end, renderer.time_end - renderer.time_start
            )
            initial_x_range = floored_range(
                preserved_range(viewport["x"], x_bounds, x_bounds), x_floor, x_bounds
            )
            initial_y_range = preserved_range(viewport["y"], y_range, initial_y_range)
            plot = figure(
                x_range=Range1d(
                    *initial_x_range,
                    bounds=x_bounds,
                    min_interval=x_floor or None,
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
            style_axes(plot, channels, geometry)
            plot.add_tools(
                channel_count_action(plot.y_range, y_range, len(channels), -1),
                channel_count_action(plot.y_range, y_range, len(channels), 1),
            )
            renderer.js_on_change("error", CustomJS(
                args={"status": status},
                code="status.text = cb_obj.error ? `<strong>${cb_obj.error}</strong>` : ''",
            ))
            coordinator = TileCoordinator(doc, renderer, (source_a, source_b), channels)
            from venn_ts.navigation import navigation_frame

            plot.name = "comparison-plot"
            plot_frame = navigation_frame(plot, x_bounds, y_range, min_interval=x_floor or None)
            plot.add_tools(CustomAction(
                description="Fullscreen",
                icon="fullscreen",
                callback=CustomJS(args={
                    "viewer_frame": viewer_frame,
                    "channel_dialog": channel_dialog,
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
                        const dialog_el = Bokeh.index.find_one(channel_dialog)?.el
                        if (channel_dialog.visible && dialog_el != null)
                            (fullscreen ? view.shadow_el : document.body).append(dialog_el)
                        const state = fullscreen ? fullscreen_state : normal_state
                        controls_sidebar.visible = state.active
                        sidebar_shell.width = state.active ? 230 : 40
                        sidebar_toggle.active = state.active
                        sidebar_toggle.width = state.active ? 110 : 30
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


def _stats_extrema(stats, participant, step, channels):
    """Select whole-recording bounds in the viewer's channel order."""
    import numpy as np

    matches = np.flatnonzero(
        (stats["participant"].values == participant)
        & (stats["step"].values == int(step))
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one stats recording for {participant}, step {step}; found {len(matches)}")
    recording = stats.isel(recording=int(matches[0])).sel(channel=channels)
    extrema = np.column_stack((recording["min"].values, recording["max"].values))
    if not np.all(np.isfinite(extrema)) or np.any(extrema[:, 0] > extrema[:, 1]):
        raise ValueError(f"Invalid min/max stats for {participant}, step {step}")
    return extrema


def venn_time_series_bokeh(doc):
    """Build immediately with cached stats, or await them without blocking Bokeh."""
    from bokeh.models import Div
    from markupsafe import escape
    from ctapdash.io.stats_cache import DATASET_STATS

    dataset = ObservationData.from_bokeh_doc(doc)
    if not dataset.get_steps():
        doc.add_root(Div(text="No processing steps are available for this participant."))
        return
    stats = DATASET_STATS.get_cached(dataset.source_path)
    if stats is not None:
        _build_comparison(doc, dataset, stats)
        return

    loading = Div(text="Loading comparison statistics...")
    doc.add_root(loading)
    destroyed = False

    def close_session(_context):
        nonlocal destroyed
        destroyed = True

    doc.on_session_destroyed(close_session)

    async def initialize():
        if destroyed:
            return
        try:
            stats = await DATASET_STATS.get(dataset.source_path)
            if destroyed:
                return
            doc.remove_root(loading)
            _build_comparison(doc, dataset, stats)
        except Exception as error:
            if not destroyed:
                loading.text = f"<strong>Unable to load comparison statistics:</strong> {escape(str(error))}"
                if loading not in doc.roots:
                    doc.add_root(loading)

    doc.add_next_tick_callback(initialize)
