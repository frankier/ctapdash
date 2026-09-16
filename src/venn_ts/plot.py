from ctapdash.io.paths import ObservationData


def _comparison_channel_geometry(extrema, mode):
    """Map source amplitudes into stacked channel-lane coordinates.

    The first recording supplies the nominal range. This makes excursions
    introduced by a later processing step visible according to ``mode``.
    """
    if mode not in {"overplot", "stretch", "normalize"}:
        raise ValueError(f"Unknown plotting mode: {mode}")

    prepared = []
    for bounds in extrema:
        a_min, a_max = bounds[:2]
        nominal_min, nominal_max = float(a_min), float(a_max)
        if nominal_min == nominal_max:
            padding = abs(nominal_min) * 0.05 or 1.0
            nominal_min -= padding
            nominal_max += padding
        actual_min = min(nominal_min, *map(float, bounds[::2]))
        actual_max = max(nominal_max, *map(float, bounds[1::2]))
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
        for index, (nominal_min, nominal_max, actual_min, actual_max) in enumerate(
            prepared
        ):
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
        Button,
        CheckboxButtonGroup,
        CheckboxGroup,
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
    )
    from venn_ts.renderer import TileCoordinator, VennTimeSeriesRenderer
    from venn_ts.domain import ComparisonDomain
    from venn_ts.domain_axis import configure_domain_axis, update_epoch_markers

    steps = dataset.get_steps()
    if not steps:
        doc.add_root(
            Div(text="No processing steps are available for this participant.")
        )
        return

    steps_by_number = dict(steps)
    step_values = [str(number) for number, _path in steps]
    from venn_ts.palettes import PALETTES, PaletteSelector, hull_color

    step_selectors = [
        Select(
            title=f"Step {'ABC'[i]}",
            options=step_values,
            value=step_values[min(i, len(step_values) - 1)],
            name=f"step-{'abc'[i]}",
            width=177,
        )
        for i in range(3)
    ]
    step_count = [2]
    swatches = [Div(width=22, height=28, margin=(25, 0, 0, 0)) for _ in range(3)]
    step_rows = [
        row(swatch, selector, visible=i < 2)
        for i, (swatch, selector) in enumerate(zip(swatches, step_selectors))
    ]
    remove_step = Button(
        label="−",
        name="remove-step",
        width=45,
        html_attributes={"title": "Remove step"},
    )
    add_step = Button(
        label="+", name="add-step", width=45, html_attributes={"title": "Add step"}
    )
    palette_select = PaletteSelector(name="palette-selector", width=210, height=78)
    hull_toggle = CheckboxGroup(
        labels=["Show min-max hull"], active=[0], name="hull-toggle"
    )
    original_theme = doc.theme

    def colors():
        return PALETTES[palette_select.value]

    def style_selectors():
        for i, swatch in enumerate(swatches):
            swatch.text = (
                f'<span title="Step {"ABC"[i]} color" style="display:inline-block;'
                f"width:14px;height:14px;border-radius:50%;border:1px solid #888;"
                f'background:{colors()[1 << i]}"></span>'
            )
        doc.theme = "contrast" if colors()[0] == "#000000" else original_theme

    style_selectors()
    plotting_mode = Select(
        title="Plotting mode",
        options=["overplot", "stretch", "normalize"],
        value="overplot",
    )
    domain_mode = Select(
        title="Domain",
        options=[("union", "Union"), ("intersection", "Intersection")],
        value="union",
        name="domain-mode",
    )
    epoch_toggle = CheckboxGroup(
        labels=["Show epoch starts"],
        active=[],
        visible=False,
        name="epoch-starts-toggle",
    )
    epoch_preferences = {"union": False, "intersection": True}
    epoch_markers = []
    epoch_choices = {}
    current_pair = [None]
    reset_domain_view = [False]
    layers = CheckboxButtonGroup(labels=["Venn", "Lines"], active=[0])
    guides_toggle = Toggle(label="Guides", active=True, name="guides-toggle")
    active_guides = [None]
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
            source_cache[path] = RecordingTileSource(
                path, recording_data=dataset.get_recording(value)
            )
        return source_cache[path]

    try:
        initial_source = get_source(step_selectors[0].value)
        initial_channels = list(initial_source.channels)
    except Exception as error:
        doc.add_root(
            Div(
                text=f"<strong>Unable to open comparison data:</strong> {escape(str(error))}"
            )
        )
        raise

    import json
    from ctapdash.channels import participant_channel_metadata
    from venn_ts.channel_selector import ChannelSelector

    channel_metadata = participant_channel_metadata(dataset)
    channel_choice = ChannelSelector(
        metadata_json=json.dumps(channel_metadata),
        available=initial_channels,
        value=[channel["name"] for channel in channel_metadata["channels"]],
        width=900,
        height=520,
    )
    channel_toggle = Toggle(
        label="Show channels",
        active=False,
        name="channel-dialog-toggle",
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
    channel_dialog.js_on_change(
        "visible",
        CustomJS(
            args={"dialog": channel_dialog},
            code="""
        queueMicrotask(() => {
            if (!dialog.visible) return
            const el = Bokeh.index.find_one(dialog)?.el
            const fullscreen = document.fullscreenElement
            if (el != null && fullscreen != null)
                (fullscreen.shadowRoot ?? fullscreen).append(el)
        })
    """,
        ),
    )

    normal_sidebar_open = Toggle(active=True, visible=False, name="normal-sidebar-open")
    fullscreen_sidebar_open = Toggle(
        active=False, visible=False, name="fullscreen-sidebar-open"
    )
    sidebar_toggle = Toggle(
        label="« Hide controls",
        active=True,
        width=110,
        name="sidebar-toggle",
    )
    controls_sidebar = column(
        *step_rows,
        row(remove_step, add_step),
        palette_select,
        hull_toggle,
        plotting_mode,
        domain_mode,
        epoch_toggle,
        layers,
        guides_toggle,
        channel_toggle,
        width=220,
    )
    sidebar_shell = column(
        sidebar_toggle,
        controls_sidebar,
        width=230,
        sizing_mode="stretch_height",
        styles={"padding-top": "10px", "padding-left": "10px"},
    )
    plot_panel = column(plot_holder, sizing_mode="stretch_both")
    viewer_frame = column(
        row(sidebar_shell, plot_panel, sizing_mode="stretch_both"),
        min_height=695,
        sizing_mode="stretch_both",
        stylesheets=[
            InlineStyleSheet(
                css="""
            :host {
                background: white;
            }
            :host(:fullscreen) {
                width: 100vw !important;
                height: 100vh !important;
                overflow: auto;
                padding: 8px;
            }
        """
            )
        ],
    )
    sidebar_toggle.js_on_change(
        "active",
        CustomJS(
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
        ),
    )

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
            channel_labels={
                item["offset"]: channel for channel, item in zip(channels, geometry)
            },
            ticker=FixedTicker(),
            avoid_overlap=True,
        )
        plot.add_layout(axis, "left")
        guide_positions = [
            item["center"] + delta for item in geometry for delta in (-0.4, 0.4)
        ]
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
        epoch_markers.clear()
        if reset_domain_view[0]:
            viewport["x"] = None
            reset_domain_view[0] = False
        if active_renderer[0] is not None:
            active_renderer[0].status = "Loading..."
        try:
            selected_steps = [
                widget.value for widget in step_selectors[: step_count[0]]
            ]
            sources = tuple(get_source(step) for step in selected_steps)
            pair = tuple(selected_steps)
            if current_pair[0] != pair:
                epoch_choices.clear()
                current_pair[0] = pair
            epoch_toggle.visible = any(source.is_epoched for source in sources)
            domain = ComparisonDomain(sources, domain_mode.value, epoch_choices)
            common = domain.sample_count
            common_channels = [
                channel
                for channel in sources[0].channels
                if all(channel in source.channel_index for source in sources)
            ]
            channel_choice.available = common_channels
            channel_choice.bads_json = json.dumps(
                {
                    step: channel_metadata["bads"].get(step, [])
                    for step in dict.fromkeys(selected_steps)
                }
            )
            selected_names = set(channel_choice.value)
            channels = [
                channel for channel in common_channels if channel in selected_names
            ]
            if not channels:
                active_renderer[0] = None
                active_plot[0] = None
                plot_holder.children = [Div(text="Select at least one channel.")]
                return

            step_extrema = [
                _stats_extrema(stats, dataset.participant, step, channels)
                for step in selected_steps
            ]
            extrema = [
                tuple(value for bounds in channel for value in bounds)
                for channel in zip(*step_extrema)
            ]
            geometry, y_range = _comparison_channel_geometry(
                extrema, plotting_mode.value
            )
            scales = [item["scale"] for item in geometry]
            offsets = [item["offset"] for item in geometry]
            channel_y_mins = [
                item["actual_min"] * item["scale"] + item["offset"] for item in geometry
            ]
            channel_y_maxs = [
                item["actual_max"] * item["scale"] + item["offset"] for item in geometry
            ]
            range_factors = sorted(
                set.intersection(*(set(source.range_factors) for source in sources))
            )
            line_factors = sorted(
                set.intersection(*(set(source.line_factors) for source in sources))
            )
            version = sha256(
                f"{[source.dataset_version for source in sources]}|{channels}|{domain.mode}|{domain.segments}".encode()
            ).hexdigest()[:20]
            renderer = VennTimeSeriesRenderer(
                level="glyph",
                step_count=len(sources),
                palette=colors(),
                hull_visible=bool(hull_toggle.active),
                hull_color=hull_color(colors()[0]),
                dataset_version=version,
                sample_count=common,
                time_start=domain.start,
                time_end=domain.end,
                sample_interval=domain.interval,
                segment_starts=[s.display_start for s in domain.segments],
                segment_counts=[s.count for s in domain.segments],
                segment_sample_starts=[domain.sample_start(s) for s in domain.segments],
                segment_revisions=[0] * len(domain.segments),
                source_factors=range_factors,
                page_counts=domain.page_counts(range_factors, 2048),
                range_factors=range_factors,
                range_page_counts=domain.page_counts(range_factors, 2048),
                line_factors=line_factors,
                line_page_counts=domain.page_counts(line_factors, 2048),
                epoch_pagers=json.dumps(
                    [
                        dict(
                            segment=i,
                            side=side,
                            count=len(segment.choices(side)),
                            index=segment.selection(side),
                            start=segment.display_start,
                            end=segment.display_start + segment.count * domain.interval,
                        )
                        for i, segment in enumerate(domain.segments)
                        for side in range(len(sources))
                        if domain.mode == "union" and len(segment.choices(side)) > 1
                    ]
                ),
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
                    (
                        geometry[visible_channel_count - 1]["center"]
                        + geometry[visible_channel_count]["center"]
                    )
                    / 2,
                    y_range[1],
                )
            plot_height = max(180, visible_channel_count * 50 + 60)
            x_bounds = (renderer.time_start, renderer.time_end)
            x_floor = min_zoom_span(
                renderer.sample_interval,
                renderer.time_end,
                renderer.time_end - renderer.time_start,
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
                min_border_top=29,
                background_fill_color=colors()[0],
            )
            plot.renderers.append(renderer)
            line_renderers = [
                plot.multi_line(
                    xs="xs",
                    ys="ys",
                    source=source,
                    line_color=colors()[1 << i],
                    line_width=1,
                    line_alpha=1,
                    name=f"step {step}",
                )
                for i, (step, source) in enumerate(
                    zip(
                        selected_steps,
                        (
                            renderer.line_source_a,
                            renderer.line_source_b,
                            renderer.line_source_c,
                        ),
                    )
                )
            ]
            plot.add_tools(
                HoverTool(
                    renderers=line_renderers,
                    tooltips=[("Step", "$name"), ("Channel", "@channel")],
                    mode="mouse",
                )
            )
            style_axes(plot, channels, geometry)
            epoch_markers.extend(
                configure_domain_axis(
                    plot, domain, epoch_visible=bool(epoch_toggle.active)
                )
            )
            update_epoch_markers(
                plot, domain, bool(epoch_toggle.active), epoch_markers, colors()
            )
            plot.add_tools(
                channel_count_action(plot.y_range, y_range, len(channels), -1),
                channel_count_action(plot.y_range, y_range, len(channels), 1),
            )
            coordinator = TileCoordinator(
                doc, renderer, sources, channels, domain=domain
            )

            def select_epoch(_attr, _old, selection):
                if len(selection) != 3:
                    return
                segment, side, index = selection
                if (
                    0 <= segment < len(domain.segments)
                    and 0 <= side < len(sources)
                    and 0 <= index < len(domain.segments[segment].choices(side))
                ):
                    epoch_choices[(segment, side)] = index
                    coordinator.select_epoch(segment, side, index)
                    pagers = json.loads(renderer.epoch_pagers)
                    for pager in pagers:
                        if pager["segment"] == segment and pager["side"] == side:
                            pager["index"] = index
                    renderer.epoch_pagers = json.dumps(pagers, separators=(",", ":"))
                    update_epoch_markers(
                        plot, domain, bool(epoch_toggle.active), epoch_markers, colors()
                    )

            renderer.on_change("epoch_selection", select_epoch)
            from venn_ts.navigation import navigation_frame

            plot.name = "comparison-plot"
            plot_frame = navigation_frame(
                plot, x_bounds, y_range, min_interval=x_floor or None
            )
            minimap = plot_frame.select_one({"name": "time-minimap"})
            configure_domain_axis(minimap, domain, epochs=False)
            plot.add_tools(
                CustomAction(
                    description="Fullscreen",
                    icon="fullscreen",
                    callback=CustomJS(
                        args={
                            "viewer_frame": viewer_frame,
                            "channel_dialog": channel_dialog,
                            "controls_sidebar": controls_sidebar,
                            "sidebar_shell": sidebar_shell,
                            "sidebar_toggle": sidebar_toggle,
                            "normal_state": normal_sidebar_open,
                            "fullscreen_state": fullscreen_sidebar_open,
                        },
                        code="""
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
                """,
                    ),
                )
            )
            install_toolbar_sizing(plot)
            active_renderer[0] = renderer
            active_coordinator[0] = coordinator
            active_plot[0] = plot
            plot_holder.children = [plot_frame]
        except Exception as error:
            active_renderer[0] = None
            active_plot[0] = None
            plot_holder.children = [
                Div(
                    text=f"<strong>Unable to build comparison:</strong> {escape(str(error))}"
                )
            ]

    def set_epochs(_attr, _old, active):
        epoch_preferences[domain_mode.value] = bool(active)
        for marker in epoch_markers:
            marker.visible = bool(active)

    def set_domain(_attr, _old, mode):
        epoch_toggle.active = [0] if epoch_preferences[mode] else []
        epoch_choices.clear()
        reset_domain_view[0] = True
        rebuild(None, None, None)

    # Show feedback immediately in the existing strip while Python rebuilds.
    for widget in (*step_selectors, channel_choice, plotting_mode, domain_mode):
        widget.js_on_change(
            "value",
            CustomJS(
                args=dict(holder=plot_holder),
                code="""
            for (const model of holder.references()) {
                if (model.type === "venn_ts.renderer.VennTimeSeriesRenderer")
                    model.status = "Loading..."
            }
        """,
            ),
        )

    epoch_toggle.on_change("active", set_epochs)
    domain_mode.on_change("value", set_domain)
    layers.on_change("active", set_layers)
    for widget in (*step_selectors, channel_choice, plotting_mode):
        widget.on_change("value", rebuild)

    def change_count(delta):
        step_count[0] = max(1, min(3, step_count[0] + delta))
        for i, step_row in enumerate(step_rows):
            step_row.visible = i < step_count[0]
        remove_step.disabled = step_count[0] == 1
        add_step.disabled = step_count[0] == 3
        rebuild(None, None, None)

    def change_palette(_attr, _old, _new):
        style_selectors()
        rebuild(None, None, None)

    def change_hull(_attr, _old, active):
        if active_renderer[0] is not None:
            active_renderer[0].hull_visible = bool(active)

    remove_step.on_click(lambda: change_count(-1))
    add_step.on_click(lambda: change_count(1))
    palette_select.on_change("value", change_palette)
    hull_toggle.on_change("active", change_hull)

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
        raise ValueError(
            f"Expected one stats recording for {participant}, step {step}; found {len(matches)}"
        )
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
        doc.add_root(
            Div(text="No processing steps are available for this participant.")
        )
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
