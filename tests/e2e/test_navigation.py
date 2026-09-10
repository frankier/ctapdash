"""Check actual track geometry and range linking, including responsive layout."""
import pytest
from bokeh.embed import file_html
from bokeh.models import FixedTicker, Range1d
from bokeh.plotting import figure
from bokeh.resources import INLINE

from venn_ts.build import ensure_extension_built
from venn_ts.channel_axis import ChannelAxis
from venn_ts.navigation import navigation_frame


GEOMETRY = """() => {
    const doc = Bokeh.documents[0];
    const model = name => doc.get_model_by_name(name);
    const view = name => Bokeh.index.find_one(model(name));
    const frame = name => {
        const v = view(name), b = v.frame.bbox, r = v.el.getBoundingClientRect();
        return {left:r.left+b.left, right:r.left+b.right,
                top:r.top+b.top, bottom:r.top+b.bottom};
    };
    const track = name => view(name).shadow_el.querySelector('.noUi-base')
        .getBoundingClientRect().toJSON();
    const p = model('navigation-test');
    return {plot:frame('navigation-test'),
            horizontal:track('horizontal-range-scrollbar'),
            vertical:track('vertical-range-scrollbar'),
            time:frame('time-minimap'), channel:frame('channel-minimap'),
            x:[p.x_range.start,p.x_range.end], y:[p.y_range.start,p.y_range.end],
            fullX:[model('time-minimap').x_range.start,model('time-minimap').x_range.end],
            fullY:[model('channel-minimap').y_range.start,model('channel-minimap').y_range.end]};
}"""


def assert_aligned(page):
    # Layout measurements update the sibling controls on the next animation frame.
    page.wait_for_function("""() => {
        const d = Bokeh.documents[0], p = d.get_model_by_name('navigation-test');
        return d.is_idle && p.inner_width === d.get_model_by_name('horizontal-range-scrollbar').width
            && p.inner_height === d.get_model_by_name('vertical-range-scrollbar').height;
    }""")
    g = page.evaluate(GEOMETRY)
    for dimension in ("left", "right"):
        assert g["horizontal"][dimension] == pytest.approx(g["plot"][dimension], abs=1)
        assert g["time"][dimension] == pytest.approx(g["plot"][dimension], abs=1)
    for dimension in ("top", "bottom"):
        assert g["vertical"][dimension] == pytest.approx(g["plot"][dimension], abs=1)
        assert g["channel"][dimension] == pytest.approx(g["plot"][dimension], abs=1)
    assert g["time"]["top"] > g["horizontal"]["bottom"]
    assert g["channel"]["right"] > g["vertical"]["right"]
    assert g["fullX"] == [0, 10]
    assert g["fullY"] == [0, 8]
    return g


def test_navigation_tracks_match_plot_and_keep_full_ranges(page):
    ensure_extension_built()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    labels = {i + 0.5: name for i, name in enumerate([
        "A888", "A1", "LongChannelName", "ExtremelyLongChannelNameToAbbreviate",
        "Fp1", "ReferenceLeft", "A88", "EMG",
    ])}
    plot = figure(name="navigation-test", height=520, min_height=300,
                  sizing_mode="stretch_both", x_range=Range1d(2, 5),
                  y_range=Range1d(2, 6))
    plot.left.remove(plot.yaxis[0])
    plot.add_layout(ChannelAxis(ticker=FixedTicker(ticks=list(labels)),
                               major_label_overrides=labels, channel_labels=labels), "left")
    plot.line([0, 10], [0, 8])
    layout = navigation_frame(plot, (0, 10), (0, 8))
    page.set_content(file_html(layout, INLINE))
    page.add_style_tag(content="html, body {width:100%;height:100%;margin:0}")
    page.wait_for_function("window.Bokeh?.documents[0]?.is_idle")
    original = assert_aligned(page)
    for orientation in ("horizontal", "vertical"):
        handle = page.evaluate("""orientation => {
            const m = Bokeh.documents[0].get_model_by_name(orientation+'-range-scrollbar');
            const el = Bokeh.index.find_one(m).shadow_el.querySelectorAll('.noUi-handle')[1];
            return el.getBoundingClientRect().toJSON();
        }""", orientation)
        x, y = handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + (30 if orientation == "horizontal" else 0),
                        y - (30 if orientation == "vertical" else 0), steps=5)
        page.mouse.up()
        current = assert_aligned(page)
        axis = "x" if orientation == "horizontal" else "y"
        assert current[axis][1] > original[axis][1]

    page.set_viewport_size({"width": 980, "height": 820})
    page.wait_for_function("""oldRight => {
        const p = Bokeh.documents[0].get_model_by_name('navigation-test');
        const v = Bokeh.index.find_one(p);
        return v.el.getBoundingClientRect().left + v.frame.bbox.right !== oldRight;
    }""", arg=original["plot"]["right"])
    resized = assert_aligned(page)
    assert resized["plot"]["right"] != original["plot"]["right"]
    page.evaluate("""() => {
        const p = Bokeh.documents[0].get_model_by_name('navigation-test');
        p.x_range.setv({start:0,end:10}); p.y_range.setv({start:0,end:8});
    }""")
    assert_aligned(page)
    values = page.evaluate("""() => ['horizontal', 'vertical'].map(orientation =>
        Bokeh.documents[0].get_model_by_name(orientation+'-range-scrollbar').value)""")
    assert values == [[0, 10], [0, 8]]
    assert not errors
