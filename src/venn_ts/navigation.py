"""Frame-aligned range controls with independent, full-recording axes."""
from bokeh.events import DocumentReady
from bokeh.layouts import column, row
from bokeh.models import CustomJS, FixedTicker, InlineStyleSheet, Range1d, RangeSlider
from bokeh.plotting import figure

from venn_ts.channel_axis import ChannelAxis


def _scrollbar(plot_range, bounds, orientation):
    vertical = orientation == "vertical"
    scrollbar = RangeSlider(
        name=f"{orientation}-range-scrollbar",
        start=bounds[0], end=bounds[1],
        value=(plot_range.start, plot_range.end),
        step=max((bounds[1] - bounds[0]) / 10_000, 1e-12),
        orientation=orientation, direction="rtl" if vertical else "ltr",
        show_value=False, tooltips=False, bar_color="#d1d5db",
        width=28 if vertical else 100, height=100 if vertical else 28,
        margin=0, sizing_mode="fixed",
        stylesheets=[InlineStyleSheet(css="""
            :host { overflow: visible !important; }
            .bk-slider-title { display: none !important; }
            .bk-input-group {
                box-sizing: border-box !important;
                height: 100% !important;
                padding: 0 !important;
                overflow: visible !important;
            }
            .noUi-target { border: 0 !important; }
            .noUi-target.noUi-vertical {
                flex: 1 1 auto !important;
                height: 100% !important;
                min-height: 0 !important;
                margin: 0 10px !important;
            }
            .noUi-target.noUi-horizontal {
                margin: 10px 0 !important;
                width: 100% !important;
            }
            .noUi-vertical .noUi-handle {
                top: auto !important;
                bottom: var(--handle-right) !important;
            }
        """)],
    )
    scrollbar.js_on_change("value", CustomJS(args={"plot_range": plot_range}, code="""
        const [start, end] = cb_obj.value
        if (plot_range.start !== start) plot_range.start = start
        if (plot_range.end !== end) plot_range.end = end
    """))
    sync = CustomJS(args={"scrollbar": scrollbar, "plot_range": plot_range}, code="""
        const [start, end] = scrollbar.value
        if (start !== plot_range.start || end !== plot_range.end)
            scrollbar.value = [plot_range.start, plot_range.end]
    """)
    plot_range.js_on_change("start", sync)
    plot_range.js_on_change("end", sync)
    return scrollbar


def navigation_frame(plot, x_bounds, y_bounds):
    """Keep tracks and overview scales on the plot's actual frame, after layout.

    The overview ranges are deliberately separate from the viewport ranges.
    Only the handle positions change when the user pans or zooms.
    """
    plot.margin = 0
    horizontal = _scrollbar(plot.x_range, x_bounds, "horizontal")
    vertical = _scrollbar(plot.y_range, y_bounds, "vertical")
    common = dict(tools="", toolbar_location=None, min_border=0, margin=0,
                  outline_line_color=None, sizing_mode="fixed")
    time_axis = figure(
        name="time-minimap", x_range=Range1d(*x_bounds), y_range=(0, 1),
        width=124, height=45, frame_width=100, frame_height=0,
        min_border_left=12, min_border_right=12,
        **common,
    )
    time_axis.yaxis.visible = False
    time_axis.grid.visible = False
    time_axis.xaxis.axis_label = "Time (s)"
    channel_axis = figure(
        name="channel-minimap", x_range=(0, 1), y_range=Range1d(*y_bounds),
        width=65, height=100, frame_width=0, frame_height=100,
        y_axis_location=None, x_axis_location=None, **common,
    )
    axis = plot.yaxis[0]
    channel_axis.add_layout(ChannelAxis(
        avoid_overlap=True, truncate_labels=True,
        ticker=FixedTicker(ticks=list(axis.ticker.ticks)),
        channel_labels=dict(axis.channel_labels),
        major_label_overrides=dict(axis.major_label_overrides),
    ), "right")
    channel_axis.grid.visible = False
    # A fixed side strip and bottom strip cannot change the viewport's size
    # in response to the measurements we copy back into their children.
    side = row(vertical, channel_axis, width=100, spacing=0, margin=0,
               sizing_mode="stretch_height")
    bottom = column(horizontal, time_axis, height=73, spacing=0, margin=0,
                    sizing_mode="stretch_width")
    layout = column(
        row(plot, side, spacing=0, margin=0, sizing_mode="stretch_both"),
        bottom, spacing=0, margin=0, min_height=plot.min_height + 73,
        sizing_mode="stretch_both",
    )
    align = CustomJS(name="align-range-navigation", args=dict(
        plot=plot, horizontal=horizontal, vertical=vertical,
        time_axis=time_axis, channel_axis=channel_axis,
    ), code="""
        const align = () => {
            const view = Bokeh.index.find_one(plot)
            if (view == null) return
            const {left, top, width, height} = view.frame.bbox
            if (width <= 0 || height <= 0) return
            const set = (model, key, value) => {
                if (JSON.stringify(model[key]) !== JSON.stringify(value)) model[key] = value
            }
            set(horizontal, "width", width)
            set(horizontal, "margin", [0, 0, 0, left])
            set(time_axis, "width", width + 24)
            set(time_axis, "frame_width", width)
            set(time_axis, "margin", [0, 0, 0, left - 12])
            set(vertical, "height", height)
            set(vertical, "margin", [top, 0, 0, 0])
            set(channel_axis, "height", height)
            set(channel_axis, "frame_height", height)
            set(channel_axis, "margin", [top, 0, 0, 0])
        }
        requestAnimationFrame(align)
    """)
    for property_name in ("inner_width", "inner_height", "outer_width", "outer_height"):
        plot.js_on_change(property_name, align)
    plot.js_on_event(DocumentReady, align)
    return layout
