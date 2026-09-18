"""The real VennDiff viewer with continuous and overlapping epoched files."""

import numpy as np
import pytest
from playwright.sync_api import expect

PARTICIPANT = "1003P_epochs_intake"
MODELS = "Bokeh.documents.flatMap(d => [...d.all_models])"


def state(page):
    return page.evaluate(f"""() => {{
        const models = {MODELS};
        const renderer = models.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        const plot = models.find(m => m.name === 'comparison-plot' && Bokeh.index.find_one(m) != null);
        return {{ready: renderer?.ready, error: renderer?.error,
            version: renderer?.dataset_version, id: renderer?.id, revisions: renderer?.segment_revisions, requests: renderer?.tile_requests, count: renderer?.sample_count,
            starts: renderer?.segment_starts, counts: renderer?.segment_counts,
            pagers: JSON.parse(renderer?.epoch_pagers ?? '[]'),
            range: plot ? [plot.x_range.start, plot.x_range.end] : null,
            breaks: models.filter(m => m.name === 'domain-break' && Bokeh.index.find_one(m) != null).length,
            epochs: models.filter(m => m.name === 'epoch-start' && Bokeh.index.find_one(m) != null).map(m => m.visible)}};
    }}""")


def ready(page, previous=None):
    page.wait_for_function(
        f"""previous => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        return m?.ready && !m.error && (previous == null || m.dataset_version !== previous);
    }}""",
        arg=previous,
        timeout=60000,
    )


def choose(page, name, value):
    previous = state(page)["version"]
    page.evaluate(
        f"""([name, value]) => {{
        {MODELS}.find(m => m.name === name).value = value;
    }}""",
        [name, value],
    )
    ready(page, previous)


def open_viewer(page, dashboard_url):
    page.goto(
        f"{dashboard_url}/participant/overview?source=dummy&participant={PARTICIPANT}"
    )
    page.locator("#viewer").get_by_role("link", name="Venndiff EEG viewer").click()
    expect(page.get_by_text("Combine time axes using...", exact=True)).to_be_visible(
        timeout=60000
    )
    ready(page)


def test_mixed_domains_epoch_preferences_and_paging(page, dashboard_url):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    open_viewer(page, dashboard_url)
    page.evaluate(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        m.lines_visible = true;
    }}""")
    page.wait_for_function(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        return m.ready && m.line_source_b.data.ys.length > 0;
    }}""")

    def selected_data():
        return page.evaluate(f"""() => {{
            const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
            const view = Bokeh.index.find_one(m);
            const ranges = [...view.epoch_tiles.entries()].filter(([, t]) => t.layer === 'venn')
                .flatMap(([key]) => view.range_parts(key).flatMap(([, t]) => Array.from(t.ranges)));
            return {{ranges, lines: Array.from(m.line_source_b.data.ys, r => Array.from(r)),
                lines_a: Array.from(m.line_source_a.data.ys, r => Array.from(r)),
                times: Array.from(m.line_source_b.data.xs, r => Array.from(r))}};
        }}""")

    before_data = selected_data()
    initial = state(page)
    assert initial["count"] == 500
    assert initial["range"] == pytest.approx([0, 5])
    checkbox = page.get_by_label("Show epoch starts", exact=True)
    expect(checkbox).not_to_be_checked()
    expect(page.locator(".epoch-pager")).to_contain_text(["B 1/2"])
    previous = initial["version"]
    page.get_by_role("button", name="Next epoch B, region 3", exact=True).click()
    page.wait_for_function(
        f"() => {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null)?.segment_revisions[2] === 1"
    )
    ready(page)
    switched = state(page)
    assert switched["id"] == initial["id"]
    assert switched["version"] == previous
    assert switched["requests"] == initial["requests"]  # Selection is local.
    after_data = selected_data()
    assert before_data["ranges"] != after_data["ranges"]
    assert before_data["ranges"][0::4] == after_data["ranges"][0::4]  # A unchanged.
    assert before_data["ranges"][1::4] == after_data["ranges"][1::4]
    assert any(
        not np.array_equal(a, b, equal_nan=True)
        for a, b in zip(before_data["lines"], after_data["lines"])
    )
    for name in ("lines_a", "times"):
        assert len(before_data[name]) == len(after_data[name])
        for a, b in zip(before_data[name], after_data[name]):
            np.testing.assert_array_equal(a, b)
    expected_revisions = [0] * len(initial["counts"])
    expected_revisions[2] = 1
    assert switched["revisions"] == expected_revisions
    expect(page.locator(".epoch-pager")).to_contain_text(["B 2/2"])
    expect(
        page.get_by_role("button", name="Next epoch B, region 3", exact=True)
    ).to_be_disabled()
    assert state(page)["range"] == initial["range"]

    checkbox.check()
    assert state(page)["version"] == initial["version"]
    paged = state(page)["version"]
    page.wait_for_function(
        f"() => {MODELS}.filter(m => m.name === 'epoch-start' && Bokeh.index.find_one(m) != null).every(m => m.visible)"
    )
    assert state(page)["version"] == paged

    choose(page, "domain-mode", "intersection")
    shared = state(page)
    assert shared["count"] == 300  # Repeats the overlapping epoch interval.
    assert shared["range"] == pytest.approx([0, 3])
    assert shared["breaks"] >= 2
    assert shared["epochs"] and all(shared["epochs"])
    expect(checkbox).to_be_checked()
    expect(page.locator(".epoch-pager")).to_have_count(0)
    checkbox.uncheck()
    choose(page, "domain-mode", "union")
    expect(checkbox).to_be_checked()  # Union's explicit choice is remembered.
    choose(page, "domain-mode", "intersection")
    expect(checkbox).not_to_be_checked()

    # Original-time labels on both main and overview axes use the same mapping.
    labels = page.evaluate(f"""async () => {{
        const p = {MODELS}.find(m => m.name === 'comparison-plot' && Bokeh.index.find_one(m) != null);
        const formatter = p.below[0].formatter;
        return await formatter.doFormat([0.5, 1, 1.5, 2, 2.5], {{loc: 0}});
    }}""")
    assert labels == ["1", "1", "1.5", "3", "3.5"]
    assert errors == []


def test_two_epoched_domains_and_navigation(page, dashboard_url):
    open_viewer(page, dashboard_url)
    previous = state(page)["version"]
    page.evaluate(f"""() => {{
        const models = {MODELS};
        models.find(m => m.name === 'step-a').value = '2';
        models.find(m => m.name === 'step-b').value = '3';
    }}""")
    ready(page, previous)
    page.wait_for_function(
        f"() => {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null)?.sample_count === 275"
    )
    assert state(page)["breaks"] >= 2  # Both axes mark the omitted gap.
    choose(page, "domain-mode", "intersection")
    assert state(page)["count"] == 225
    page.evaluate(f"""() => {{
        const models = {MODELS};
        models.find(m => m.name === 'horizontal-range-scrollbar' && Bokeh.index.find_one(m) != null).value = [.8, 1.8];
        models.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null).lines_visible = true;
    }}""")
    ready(page)
    page.wait_for_function(
        f"() => {MODELS}.find(m => m.name === 'comparison-plot' && Bokeh.index.find_one(m) != null).x_range.start > .7"
    )
    assert state(page)["range"] == pytest.approx([0.8, 1.8])
    choose(page, "domain-mode", "union")
    assert state(page)["range"] == pytest.approx([0.5, 3.25])


def test_pager_gutter_overlap_and_hover_priority(page, dashboard_url):
    open_viewer(page, dashboard_url)
    labels = page.evaluate(f"""() => {{
        const models = {MODELS};
        const p = models.find(m => m.name === 'comparison-plot' && Bokeh.index.find_one(m) != null);
        const outer = models.find(m => m.name === 'time-minimap' && Bokeh.index.find_one(m) != null);
        return [p.below[0].axis_label, outer.below[0].axis_label];
    }}""")
    assert labels == [None, "Time (s)"]
    page.evaluate(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        const item = JSON.parse(m.epoch_pagers)[0];
        m.epoch_pagers = JSON.stringify([item, {{...item, side: 0, start: item.start + .1, end: item.end + .1}}]);
    }}""")
    pagers = page.locator(".epoch-pager")
    expect(pagers).to_have_count(2)
    boxes = [pagers.nth(i).bounding_box() for i in range(2)]
    assert boxes[0]["y"] == boxes[1]["y"]
    assert boxes[1]["x"] < boxes[0]["x"] + boxes[0]["width"]
    # Hover the exposed edge of the older indicator, then leave the gutter.
    page.mouse.move(boxes[0]["x"] + 2, boxes[0]["y"] + 5)
    expect(pagers.nth(0)).to_have_css("z-index", "6")
    page.mouse.move(0, 0)
    expect(pagers.nth(0)).to_have_css("z-index", "5")
    expect(pagers.nth(1)).to_have_css("z-index", "5")
    assert (
        page.evaluate(
            f"""() => {MODELS}.find(m => m.name === 'comparison-plot' && Bokeh.index.find_one(m) != null).min_border_top"""
        )
        == 29
    )


def test_coarse_tiles_follow_original_grid_across_compacted_segments(
    page, dashboard_url
):
    open_viewer(page, dashboard_url)
    choose(page, "domain-mode", "intersection")
    # Small pages expose the extra bucket created by a nonzero epoch phase.
    page.evaluate(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        m.page_size = 2;
        m.range_factors = [64];
        m.line_factors = [64];
        m.lines_visible = false;
        m.lines_visible = true;
    }}""")
    page.wait_for_function(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        return m?.ready && !m.error && m.current_lod === 64;
    }}""")
    metadata = page.evaluate(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        const view = Bokeh.index.find_one(m);
        return [...view.epoch_tiles.values()].filter(t => t.layer === 'venn' && t.factor === 64 && t.channel_page === 0)
            .map(t => ({{page: t.page, start: t.start, end: t.end, fragments: t.fragments}})).sort((a,b) => a.page-b.page);
    }}""")
    # Fixed 128-sample display pages, independent of the three epoch boundaries.
    assert [tile["page"] for tile in metadata] == [0, 1, 2]
    assert [tile["start"] for tile in metadata] == pytest.approx([0, 1.28, 2.56])
    assert metadata[-1]["end"] == pytest.approx(3)
    assert metadata[0]["fragments"][0]["time"] == pytest.approx(-0.5)
    assert metadata[0]["fragments"][0]["end"] == pytest.approx(1)
    assert metadata[0]["fragments"][1]["time"] == pytest.approx(0.64)
    assert page.evaluate(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        const view = Bokeh.index.find_one(m), range = view.coordinates.x_source;
        for (const [key, batch] of view.epoch_tiles) {{
            if (batch.layer !== 'venn' || batch.factor !== 64) continue;
            for (const [, tile] of view.range_parts(key)) {{
                let upload, calls = 0;
                const gl = {{getParameter:()=>16, activeTexture:()=>{{}}, bindTexture:()=>{{}},
                    texImage2D:(...args)=>{{upload=args.at(-1); calls++}}}};
                const gpu = {{lookup_signature:'',lookup_width:1,lookup_height:1,bytes:8}};
                view.update_lookup(tile,gpu,gl,33); // Exercise a multi-row texture.
                view.update_lookup(tile,gpu,gl,33);
                if (calls !== 1 || gpu.lookup_height !== 3) return false;
                for (let pixel=0;pixel<33;pixel++) {{
                    const lo=range.start+pixel/33*(range.end-range.start)-tile.time_start;
                    const hi=range.start+(pixel+1)/33*(range.end-range.start)-tile.time_start;
                    const expected=[];
                    for (let i=0;i<tile.entry_count;i++)
                        if (tile.edges[2*i+1]>lo && tile.edges[2*i]<hi) expected.push(i);
                    const actual=Array.from({{length:upload[2*pixel+1]-upload[2*pixel]}},(_,i)=>upload[2*pixel]+i);
                    if (JSON.stringify(actual)!==JSON.stringify(expected)) return false;
                }}
            }}
        }}
        return true;
    }}""")


def test_cached_epoch_switch_does_not_depend_on_server_round_trip(page, dashboard_url):
    open_viewer(page, dashboard_url)
    page.evaluate(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        m.lines_visible = true;
    }}""")
    page.wait_for_function(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        return m.ready && m.line_source_b.data.ys.length > 0;
    }}""")
    before = page.evaluate(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        m.syncable = false; // No selection/pager changes will reach Python.
        return {{requests:m.request_seq, lines:JSON.stringify(m.line_source_b.data.ys)}};
    }}""")
    page.get_by_role("button", name="Next epoch B, region 3", exact=True).click()
    page.wait_for_function(
        f"""before => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        return m.ready && JSON.stringify(m.line_source_b.data.ys) !== before;
    }}""",
        arg=before["lines"],
    )
    after = state(page)
    assert not any(after["revisions"])  # No server-side selection callback ran.
    assert (
        page.evaluate(
            f"""() => {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null).request_seq"""
        )
        == before["requests"]
    )
    assert after["pagers"][0]["index"] == 1


def test_shared_marker_strip_and_loading_overlay(page, dashboard_url):
    open_viewer(page, dashboard_url)

    def geometry():
        return page.evaluate(f"""() => {{
            const models = {MODELS};
            const plot = models.find(m => m.name === 'comparison-plot' && Bokeh.index.find_one(m));
            const view = Bokeh.index.find_one(plot);
            const rect = view.el.getBoundingClientRect();
            const markers = plot.renderers.filter(m => m.type === 'venn_ts.domain_axis.DomainMarkers');
            return {{top: rect.top + view.frame.bbox.top, gutter: plot.min_border_top,
                markers: markers.length, labels: markers.flatMap(m => m.labels)}};
        }}""")

    initial = geometry()
    choose(page, "domain-mode", "intersection")
    shared = geometry()
    assert shared["top"] == pytest.approx(initial["top"], abs=1)
    assert shared["gutter"] == 29
    assert shared["markers"] == 2
    assert "↶" in shared["labels"]
    assert "//" in shared["labels"]
    assert "B" in shared["labels"]

    # Loading covers the same strip and never moves the plot frame.
    page.evaluate(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m));
        m.status = 'Loading...';
    }}""")
    overlay = page.locator(".comparison-status")
    expect(overlay).to_be_visible()
    box = overlay.bounding_box()
    assert box["y"] + box["height"] == pytest.approx(shared["top"], abs=1)
    assert geometry()["top"] == shared["top"]
    page.evaluate(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m));
        m.status = '';
    }}""")
    expect(overlay).not_to_be_visible()

    # Comparing the continuous step with itself keeps the same reserved strip.
    page.evaluate(f"""() => {{
        const models = {MODELS};
        const a = models.find(m => m.name === 'step-a');
        models.find(m => m.name === 'step-b').value = a.value;
    }}""")
    page.wait_for_function(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m));
        return m?.ready && m.segment_counts.length === 1;
    }}""")
    assert geometry()["top"] == pytest.approx(initial["top"], abs=1)
