"""Frame-aligned range controls with independent, full-recording axes."""

from bokeh.events import DocumentReady
from bokeh.layouts import column, row
from bokeh.models import CustomJS, FixedTicker, Range1d
from bokeh.plotting import figure

from workingtitle.bokeh.browser import ON_PLOT_FRAME
from workingtitle.bokeh.ranges import range_scrollbar

from venn_ts.channel_axis import ChannelAxis


def navigation_frame(plot, x_bounds, y_bounds, min_interval=None):
    """Keep tracks and overview scales on the plot's actual frame, after layout.

    The overview ranges are deliberately separate from the viewport ranges.
    Only the handle positions change when the user pans or zooms.
    """
    plot.margin = 0
    horizontal = range_scrollbar(plot.x_range, x_bounds, "horizontal", min_interval)
    vertical = range_scrollbar(plot.y_range, y_bounds, "vertical")
    common = dict(
        tools="",
        toolbar_location=None,
        min_border=0,
        margin=0,
        outline_line_color=None,
        sizing_mode="fixed",
    )
    time_axis = figure(
        name="time-minimap",
        x_range=Range1d(*x_bounds),
        y_range=(0, 1),
        width=124,
        height=45,
        frame_width=100,
        frame_height=0,
        min_border_left=12,
        min_border_right=12,
        **common,
    )
    time_axis.yaxis.visible = False
    time_axis.grid.visible = False
    time_axis.xaxis.axis_label = "Time (s)"
    channel_axis = figure(
        name="channel-minimap",
        x_range=(0, 1),
        y_range=Range1d(*y_bounds),
        width=65,
        height=100,
        frame_width=0,
        frame_height=100,
        y_axis_location=None,
        x_axis_location=None,
        **common,
    )
    axis = plot.yaxis[0]
    channel_axis.add_layout(
        ChannelAxis(
            avoid_overlap=True,
            truncate_labels=True,
            ticker=FixedTicker(ticks=list(axis.channel_labels)),
            channel_labels=dict(axis.channel_labels),
            major_label_overrides=dict(axis.channel_labels),
        ),
        "right",
    )
    channel_axis.grid.visible = False
    # Keep navigation scales readable when the document uses the contrast theme.
    for overview in (time_axis, channel_axis):
        overview.background_fill_color = "white"
        overview.border_fill_color = "white"
        # Bokeh ignores assignments equal to the current default. Set each
        # alpha away from its default first so 1 is stored as an override.
        for alpha in (0, 1):
            overview.axis.major_tick_line_alpha = alpha
            overview.axis.minor_tick_line_alpha = alpha
            overview.axis.axis_line_alpha = alpha
    # A fixed side strip and bottom strip cannot change the viewport's size
    # in response to the measurements we copy back into their children.
    side = row(
        vertical,
        channel_axis,
        width=100,
        spacing=0,
        margin=0,
        sizing_mode="stretch_height",
    )
    bottom = column(
        horizontal,
        time_axis,
        height=73,
        spacing=0,
        margin=0,
        sizing_mode="stretch_width",
    )
    layout = column(
        row(plot, side, spacing=0, margin=0, sizing_mode="stretch_both"),
        bottom,
        spacing=0,
        margin=0,
        min_height=plot.min_height + 73,
        sizing_mode="stretch_both",
    )
    align = CustomJS(
        name="align-range-navigation",
        args=dict(
            plot=plot,
            horizontal=horizontal,
            vertical=vertical,
            time_axis=time_axis,
            channel_axis=channel_axis,
        ),
        code=ON_PLOT_FRAME
        + """
        onPlotFrame(plot, 'rangeNavigationFrame', () => plot, (_view, {left, top, width, height}) => {
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
        });
    """,
    )
    for property_name in ("inner_width", "inner_height", "outer_width", "outer_height"):
        plot.js_on_change(property_name, align)
    plot.js_on_event(DocumentReady, align)
    return layout
