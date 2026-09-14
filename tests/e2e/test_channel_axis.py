"""Exercise text measurement and axis layout in a real canvas."""
from bokeh.embed import file_html
from bokeh.models import FixedTicker, Range1d
from bokeh.plotting import figure
from bokeh.resources import INLINE

from venn_ts.build import ensure_extension_built
from venn_ts.channel_axis import ChannelAxis


def test_channel_labels_fit_without_moving_plot(page):
    ensure_extension_built()
    labels = {
        0.5: "A888", 1.5: "A1", 2.5: "LongChannelName",
        3.5: "ReallyExtremelyLongChannelNameThatNeedsTruncation",
        4.5: "Fp1", 5.5: "ReferenceLeft", 6.5: "A88", 7.5: "EMG",
    }
    plot = figure(width=800, height=520, y_range=Range1d(0, 8), tools="",
                  name="channel-axis-test")
    plot.left.remove(plot.yaxis[0])
    plot.add_layout(ChannelAxis(
        ticker=FixedTicker(ticks=list(labels)), channel_labels=labels,
        major_label_overrides=labels, axis_label="Channel",
    ), "left")
    plot.line([0, 1], [0, 8])
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.evaluate("""() => {
        window.labelDraws = [];
        const original = CanvasRenderingContext2D.prototype.fillText;
        CanvasRenderingContext2D.prototype.fillText = function(text, ...args) {
            const t = this.getTransform();
            window.labelDraws.push({text, rotated: Math.abs(t.b) > 0.1, align: this.textAlign});
            return original.call(this, text, ...args);
        };
    }""")
    page.set_content(file_html(plot, INLINE))
    page.wait_for_function("window.Bokeh?.documents[0]?.is_idle")
    draws = page.evaluate("window.labelDraws")
    assert any(d["text"] == "A888" and not d["rotated"] and d["align"] == "right" for d in draws)
    assert any(d["text"].endswith("...") and d["rotated"] for d in draws)
    assert any(d["text"].startswith("Long") and d["rotated"] for d in draws)
    before = page.evaluate("""() => {
        const plot = Bokeh.documents[0].get_model_by_name('channel-axis-test');
        const left = Bokeh.index.find_one(plot).frame.bbox.left;
        plot.y_range.start = 4;
        plot.y_range.end = 8;
        return left;
    }""")
    page.wait_for_function("Bokeh.documents[0].is_idle")
    after = page.evaluate("""() => Bokeh.index.find_one(
        Bokeh.documents[0].get_model_by_name('channel-axis-test')).frame.bbox.left""")
    assert before == after
    assert not errors


def test_dense_minimap_labels_keep_endpoints_and_do_not_overlap(page):
    ensure_extension_built()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    labels = {i + 0.5: f"A{i}" + ("LongChannelName" if i % 2 else "") for i in range(80)}
    plot = figure(width=70, height=250, frame_height=250, frame_width=0,
                  min_border=0, margin=0, y_range=Range1d(0, 80), x_range=(0, 1),
                  x_axis_location=None, y_axis_location=None, tools="",
                  toolbar_location=None)
    plot.add_layout(ChannelAxis(
        ticker=FixedTicker(ticks=list(labels)), channel_labels=labels,
        major_label_overrides=labels, avoid_overlap=True, truncate_labels=True,
    ), "right")
    page.evaluate("""() => {
        window.draws = [];
        const original = CanvasRenderingContext2D.prototype.fillText;
        CanvasRenderingContext2D.prototype.fillText = function(text, ...args) {
            if (/^A[0-9]+/.test(text)) {
                const t = this.getTransform();
                window.draws.push({text, y:t.f + args[1]*t.d, rotated: Math.abs(t.b) > 0.1, width: this.measureText(text).width});
            }
            return original.call(this, text, ...args);
        };
    }""")
    page.set_content(file_html(plot, INLINE))
    page.wait_for_function("window.Bokeh?.documents[0]?.is_idle")
    draws = page.evaluate("Array.from(new Map(window.draws.map(d => [d.text, d])).values())")
    assert not errors
    assert any(d["text"] == "A0" for d in draws)
    assert any(d["text"].startswith("A79") and d["text"].endswith("...") for d in draws)
    assert all(not d["rotated"] and d["width"] <= 48 for d in draws)
    assert len(draws) >= 12
    positions = sorted(d["y"] for d in draws)
    assert positions[0] >= 6
    assert positions[-1] <= 245
    assert all(b - a >= 16 for a, b in zip(positions, positions[1:]))
