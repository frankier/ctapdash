"""Shared selector interactions and real HTMX/Bokeh integration."""
from playwright.sync_api import expect

from .test_pages import QUERY, visit


METADATA = {
    "stateKey": "component-test",
    "channels": [{"name": name, "type": kind} for name, kind in [("Fz", "EEG"), ("Cz", "EEG"), ("Pz", "EEG"), ("Aux", "Custom")]],
    "groups": {"Middle": ["Fz", "Cz"]}, "regions": {"Frontal": ["Fz"], "Parietal": ["Pz"]},
    "head": {"points": {"Fz": [0, -.05], "Cz": [0, 0], "Pz": [0, .05]},
             "outlines": [[[-.09, 0], [0, -.09], [.09, 0], [0, .09], [-.09, 0]]], "message": ""},
    "bads": {"1": ["Cz"], "2": ["Cz"]}, "steps": {"1": ["Fz", "Cz", "Pz", "Aux"]},
}


def mount(page, dashboard_url, metadata=METADATA):
    visit(page, dashboard_url, "/setup")
    page.wait_for_function("customElements.get('channel-selector') !== undefined")
    page.evaluate("""metadata => {
        const selector = document.createElement('channel-selector');
        selector.id = 'test-selector';
        selector.configure(metadata, metadata.channels.map(c => c.name), metadata.bads);
        document.body.replaceChildren(selector);
        window.changes = [];
        selector.addEventListener('channel-selection-change', e => window.changes.push(e.detail.selectedNames));
    }""", metadata)
    return page.locator("channel-selector")


def test_groups_search_bad_markers_and_keyboard(page, dashboard_url):
    selector = mount(page, dashboard_url)
    cz = selector.get_by_role("checkbox", name="Cz — Bad in steps 1, 2", exact=True)
    expect(cz).to_be_checked()
    cz.focus(); page.keyboard.press("Space")
    group = selector.get_by_role("checkbox", name="EEG (2/3)", exact=True)
    expect(group).to_have_js_property("indeterminate", True)
    selector.get_by_role("searchbox").fill("Fz")
    selector.get_by_role("button", name="Clear", exact=True).click()
    assert selector.evaluate("s => s.selectedNames") == ["Pz", "Aux"]
    selector.get_by_role("searchbox").fill("")
    selector.get_by_role("combobox", name="Group channels by").select_option("custom")
    expect(selector.get_by_role("checkbox", name="Middle (0/2)", exact=True)).not_to_be_checked()
    selector.get_by_role("checkbox", name="Middle (0/2)", exact=True).click()
    assert selector.evaluate("s => s.selectedNames") == ["Fz", "Cz", "Pz", "Aux"]
    selector.get_by_role("tab", name="Grouped list").focus(); page.keyboard.press("ArrowRight")
    expect(selector.get_by_role("tab", name="Head diagram")).to_be_focused()
    expect(selector.get_by_role("checkbox", name="Cz — Bad in steps 1, 2", exact=True)).to_have_attribute("aria-checked", "true")


def test_head_click_lasso_and_programmatic_updates(page, dashboard_url, tmp_path):
    selector = mount(page, dashboard_url)
    selector.get_by_role("tab", name="Head diagram").click()
    selector.get_by_role("checkbox", name="Fz", exact=True).click()
    assert selector.evaluate("s => s.selectedNames") == ["Cz", "Pz", "Aux"]
    selector.get_by_role("combobox", name="Lasso action").select_option("deselect")
    # Draw a closed polygon around Cz only, using SVG coordinates.
    corners = selector.locator("svg").evaluate("""svg => [[-.015,-.015],[.015,-.015],[.015,.015],[-.015,.015],[-.015,-.015]].map(([x,y]) => {
        const p = new DOMPoint(x,y).matrixTransform(svg.getScreenCTM()); return [p.x,p.y];
    })""")
    before = page.evaluate("window.changes.length")
    page.mouse.move(*corners[0]); page.mouse.down()
    for corner in corners[1:]:
        page.mouse.move(*corner, steps=3)
    page.mouse.up()
    assert selector.evaluate("s => s.selectedNames") == ["Pz", "Aux"]
    assert page.evaluate("window.changes.length") == before + 1
    selector.evaluate("s => { s.selectedNames = ['Fz']; s.availableNames = ['Cz', 'Pz']; }")
    expect(selector.get_by_role("checkbox", name="Fz — Unavailable in the displayed processing steps", exact=True)).to_have_attribute("aria-disabled", "true")
    assert selector.evaluate("s => s.selectedNames") == ["Fz"]
    assert page.evaluate("window.changes.length") == before + 1
    selector.evaluate("s => s.availableNames = ['Fz', 'Cz', 'Pz', 'Aux']")
    expect(selector.get_by_role("checkbox", name="Fz", exact=True)).to_have_attribute("aria-checked", "true")
    selector.screenshot(path=str(tmp_path / "head-selector.png"))


def test_storage_isolation_missing_positions_and_cleanup(page, dashboard_url):
    selector = mount(page, dashboard_url)
    selector.get_by_role("button", name="Clear", exact=True).click()
    selector = mount(page, dashboard_url)
    assert selector.evaluate("s => s.selectedNames") == []
    other = dict(METADATA, stateKey="different-participant", head={"points": {}, "outlines": [], "message": "No usable positions"})
    selector = mount(page, dashboard_url, other)
    assert len(selector.evaluate("s => s.selectedNames")) == 4
    selector.get_by_role("tab", name="Head diagram").click()
    expect(selector.get_by_text("No usable positions", exact=True)).to_be_visible()
    assert page.evaluate("""() => {
        const s = document.querySelector('channel-selector'); s.remove();
        window.dispatchEvent(new CustomEvent('ctap-channel-state', {detail: {key: 'different-participant', names: []}}));
        return s.selectedNames.length;
    }""") == 4


def test_heatmap_empty_selection_navigation_and_venndiff(page, dashboard_url):
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    visit(page, dashboard_url, "/participant/overview" + QUERY)
    expect(page.locator("#channel-statistics-table table")).to_be_visible(timeout=30000)
    page.get_by_role("button", name="Show channels", exact=True).click()
    selector = page.locator("channel-selector")
    selector.get_by_role("checkbox", name="B", exact=True).uncheck()
    page.get_by_role("button", name="Close channels", exact=True).click()
    expect(page.locator('#channel-statistics-table th[title="B"]')).to_have_count(0)
    page.locator("#statistics-step").select_option("2")
    expect(page.locator('#channel-statistics-table th[title="A"]')).to_be_visible()
    expect(page.locator('#channel-statistics-table th[title="B"]')).to_have_count(0)
    page.locator("#viewer").get_by_role("link", name="Venndiff EEG viewer").click()
    page.wait_for_function("""() => window.Bokeh?.documents.some(doc => [...doc.all_models].some(m => m.type === 'venn_ts.channel_selector.ChannelSelector' && JSON.stringify(m.value) === '["A"]'))""", timeout=60000)
    page.get_by_role("button", name="Show channels", exact=True).click()
    expect(page.locator("channel-selector").get_by_role("checkbox", name="B", exact=True)).not_to_be_checked()
    page.locator("channel-selector").get_by_role("checkbox", name="B", exact=True).check()
    page.wait_for_function("""() => Bokeh.documents.some(doc => [...doc.all_models].some(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && m.channel_names.length === 2))""")
    page.get_by_role("button", name="Hide channels", exact=True).click()
    page.get_by_title("Fullscreen", exact=True).click()
    page.wait_for_function("document.fullscreenElement !== null")
    page.get_by_role("button", name="»", exact=True).click()
    page.get_by_role("button", name="Show channels", exact=True).click()
    expect(page.locator("channel-selector").get_by_role("checkbox", name="B", exact=True)).to_be_visible()
    page.evaluate("document.exitFullscreen()")
    page.wait_for_function("document.fullscreenElement === null")
    page.locator("#viewer").get_by_role("link", name="Overview", exact=True).click()
    page.get_by_role("button", name="Show channels", exact=True).click()
    page.locator("channel-selector").get_by_role("button", name="Clear", exact=True).click()
    page.get_by_role("button", name="Close channels", exact=True).click()
    expect(page.locator("#channel-statistics-table")).to_have_text("Empty selection")
    assert errors == []


def test_mne_head_diagram(page, dashboard_url, tmp_path):
    import mne
    from ctapdash.channels import _head_geometry

    positions = mne.channels.make_standard_montage("biosemi64").get_positions()["ch_pos"]
    head, regions = _head_geometry(positions)
    metadata = dict(METADATA, channels=[{"name": name, "type": "EEG"} for name in positions],
                    head=head, regions=regions, stateKey="biosemi64")
    selector = mount(page, dashboard_url, metadata)
    selector.get_by_role("tab", name="Head diagram").click()
    expect(selector.locator("svg .sensor")).to_have_count(64)
    selector.screenshot(path=str(tmp_path / "mne-head-selector.png"))
    assert page.evaluate("""() => {
        const standalone = document.createElement('channel-selector');
        standalone.channels = [{name: 'A', type: 'EEG'}];
        const defaults = [standalone.selectedNames, standalone.availableNames];
        standalone.selectedNames = []; standalone.availableNames = [];
        standalone.channels = [{name: 'A', type: 'EEG'}, {name: 'B', type: 'EEG'}];
        return [defaults, standalone.selectedNames, standalone.availableNames];
    }""") == [[["A"], ["A"]], [], []]
