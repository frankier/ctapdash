from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from bokeh.document import Document
from bokeh.models import (
    CheckboxButtonGroup,
    CustomAction,
    Dialog,
    HoverTool,
    MultiChoice,
    PanTool,
    RangeSlider,
    Select as BkSelect,
    Toggle,
)
from bokeh.models.renderers import GlyphRenderer
from bokeh.plotting._figure import figure as BkFigure

from ctapdash.io import paths
from ctapdash.config import SETTINGS
from venn_ts.renderer import VennTimeSeriesRenderer


def test_comparison_channel_geometry_modes():
    from venn_ts.plot import _comparison_channel_geometry
    extrema = [(-1, 1, -3, 2), (-2, 2, -1, 1)]

    overplot, overplot_range = _comparison_channel_geometry(extrema, "overplot")
    assert overplot_range == (0.0, 2.0)
    assert overplot[0]["outside"] is True
    assert overplot[0]["actual_min"] * overplot[0]["scale"] + overplot[0]["offset"] < 1.1

    stretch, stretch_range = _comparison_channel_geometry(extrema, "stretch")
    assert stretch_range[1] > overplot_range[1]
    first_low = stretch[0]["actual_min"] * stretch[0]["scale"] + stretch[0]["offset"]
    second_high = stretch[1]["actual_max"] * stretch[1]["scale"] + stretch[1]["offset"]
    assert first_low > second_high

    normalized, normalized_range = _comparison_channel_geometry(extrema, "normalize")
    assert normalized_range == overplot_range
    for item in normalized:
        mapped_span = (item["actual_max"] - item["actual_min"]) * item["scale"]
        assert mapped_span == pytest.approx(0.8)


def test_comparison_view_uses_one_webgl_figure_with_overlaid_layers(tmp_path):
    from venn_ts.range_series import RecordingTileSource
    from venn_ts import venn_time_series_bokeh

    class FakeRecording:
        ch_names = [f"Ch{index}" for index in range(8)]

        def __init__(self, offset):
            self.offset = offset

        def mmap(self, *, return_xarray=False):
            times = np.arange(20, dtype=float) * 0.1
            values = np.vstack(
                [np.sin(times + index / 10) for index in range(len(self.ch_names))]
            ) + self.offset
            return xr.DataArray(values, coords=(self.ch_names, times), dims=("ch", "time"))

    steps = []
    for number in (1, 2):
        step = tmp_path / f"{number:02d}_step"
        step.mkdir()
        (step / "p.set").touch()
        steps.append((number, step))
    dataset = paths.ObservationData(tmp_path, "p", "fake")
    dataset.get_steps = lambda: steps

    def fake_source(path, *, recording_data=None):
        offset = int(Path(path).parent.name[:2])
        return RecordingTileSource(path, recording=FakeRecording(offset))

    with (
        patch.object(
            paths.ObservationData,
            "from_bokeh_doc",
            classmethod(lambda cls, doc: dataset),
        ),
        patch(
            "venn_ts.range_series.RecordingTileSource",
            side_effect=fake_source,
        ),
    ):
        doc = Document()
        venn_time_series_bokeh(doc)

    figure = doc.roots[0].select_one({"type": BkFigure})
    custom = list(figure.select({"type": VennTimeSeriesRenderer}))
    glyphs = list(figure.select({"type": GlyphRenderer}))
    layer_control = doc.roots[0].select_one({"type": CheckboxButtonGroup})
    channel_choice = doc.select_one({"type": MultiChoice})
    channel_toggle = doc.get_model_by_name("channel-dialog-toggle")
    channel_dialog = doc.select_one({"type": Dialog})
    scrollbars = list(doc.roots[0].select({"type": RangeSlider}))
    scrollbars_by_orientation = {
        scrollbar.orientation: scrollbar for scrollbar in scrollbars
    }
    assert figure.output_backend == "webgl"
    assert figure.height == 660
    assert figure.min_height == 660
    assert figure.sizing_mode == "stretch_both"
    assert (figure.y_range.start, figure.y_range.end) == (2, 8)
    assert len(custom) == 1
    assert custom[0].channel_tile_size == 1
    assert len(glyphs) == 2
    assert all(glyph.data_source in {custom[0].line_source_a, custom[0].line_source_b} for glyph in glyphs)
    assert layer_control.labels == ["Venn", "Lines"]
    assert layer_control.active == [0, 1]
    assert channel_dialog.visible is False
    assert channel_dialog.close_action == "hide"
    assert channel_toggle.label == "Show channels"
    channel_toggle.active = True
    assert channel_dialog.visible is True
    assert channel_toggle.label == "Hide channels"
    channel_dialog.visible = False
    assert channel_toggle.active is False
    assert {scrollbar.orientation for scrollbar in scrollbars} == {"horizontal", "vertical"}
    assert scrollbars_by_orientation["horizontal"].value == pytest.approx((0, 1.9))
    assert scrollbars_by_orientation["vertical"].value == (2, 8)
    assert scrollbars_by_orientation["vertical"].direction == "rtl"
    assert scrollbars_by_orientation["vertical"].min_height == 660
    assert scrollbars_by_orientation["vertical"].sizing_mode == "stretch_height"
    assert not list(figure.select({"type": PanTool}))
    layer_control.active = [1]
    assert custom[0].venn_visible is False
    assert custom[0].lines_visible is True
    layer_control.active = [0]
    assert custom[0].venn_visible is True
    assert custom[0].lines_visible is False

    actions = list(figure.select({"type": CustomAction}))
    assert {action.description for action in actions} == {
        "+1 channel", "-1 channel", "Fullscreen",
    }
    channel_actions = [action for action in actions if "channel" in action.description]
    assert all(
        action.icon.startswith("data:image/svg+xml;base64,")
        for action in channel_actions
    )
    assert figure.toolbar.stylesheets == []
    toolbar_resizer = doc.get_model_by_name("toolbar-button-resizer")
    assert toolbar_resizer.args["toolbar"] is figure.toolbar
    assert 'setProperty("--button-width", "40px", "important")' in toolbar_resizer.code
    assert 'setProperty("width", "40px", "important")' in toolbar_resizer.code
    vertical_css = "\n".join(
        stylesheet.css
        for stylesheet in scrollbars_by_orientation["vertical"].stylesheets
    )
    assert "padding: 7px 0 !important" in vertical_css
    assert "height: 100% !important" in vertical_css
    assert "top: auto !important" in vertical_css
    assert "bottom: var(--handle-right) !important" in vertical_css
    fullscreen = next(action for action in actions if action.description == "Fullscreen")
    fullscreen_frame = fullscreen.callback.args["viewer_frame"]
    assert fullscreen_frame.min_height == 695
    assert fullscreen_frame.sizing_mode == "stretch_both"
    assert len(list(fullscreen_frame.select({"type": RangeSlider}))) == 2
    sidebar_toggle = doc.get_model_by_name("sidebar-toggle")
    normal_sidebar_open = doc.get_model_by_name("normal-sidebar-open")
    fullscreen_sidebar_open = doc.get_model_by_name("fullscreen-sidebar-open")
    assert sidebar_toggle.active is True
    assert sidebar_toggle.label == "« Hide controls"
    assert normal_sidebar_open.active is True
    assert fullscreen_sidebar_open.active is False
    assert fullscreen.callback.args["controls_sidebar"].visible is True

    figure.x_range.start, figure.x_range.end = 0.4, 1.2
    figure.y_range.start, figure.y_range.end = 3.0, 7.0
    plotting_mode = next(
        select for select in doc.roots[0].select({"type": BkSelect})
        if select.title == "Plotting mode"
    )
    plotting_mode.value = "normalize"
    rebuilt = doc.roots[0].select_one({"type": BkFigure})
    assert (rebuilt.x_range.start, rebuilt.x_range.end) == pytest.approx((0.4, 1.2))
    assert (rebuilt.y_range.start, rebuilt.y_range.end) == pytest.approx((3.0, 7.0))

    step_a = next(
        select for select in doc.roots[0].select({"type": BkSelect})
        if select.title == "Step A (red)"
    )
    step_a.value = "2"
    rebuilt = doc.roots[0].select_one({"type": BkFigure})
    assert (rebuilt.x_range.start, rebuilt.x_range.end) == pytest.approx((0.4, 1.2))
    assert (rebuilt.y_range.start, rebuilt.y_range.end) == pytest.approx((3.0, 7.0))

    channel_choice.value = channel_choice.value[:-1]
    rebuilt = doc.roots[0].select_one({"type": BkFigure})
    assert (rebuilt.x_range.start, rebuilt.x_range.end) == pytest.approx((0.4, 1.2))
    assert rebuilt.y_range.end - rebuilt.y_range.start == pytest.approx(4.0)


@pytest.fixture
def pyramid_source(tmp_path):
    """A fake source directory holding one step with a 2-level pyramid."""
    step = tmp_path / "01_test"
    step.mkdir()
    (step / "p.set").write_text("")

    rng = np.random.default_rng(0)
    t = np.linspace(0, 100, 100_000)
    chs = [f"Fp{i}" for i in range(8)]
    data = rng.normal(0, 50, (8, 100_000))
    for factor in (1, 10):
        sl = slice(None, None, factor)
        xr.Dataset(
            {"data": (("ch", "time"), data[:, sl])},
            coords={"ch": chs, "time": t[sl]},
        ).to_zarr(step / "p.set.pyramid", group=f"factor_{factor}", mode="a")

    SETTINGS.sources["fake"] = str(tmp_path)
    yield tmp_path
    del SETTINGS.sources["fake"]


def test_pyramid_groups_finest_to_coarsest(pyramid_source):
    from ctapdash.io.pyramid import _pyramid_groups
    ts_dt = xr.open_datatree(
        pyramid_source / "01_test" / "p.set.pyramid",
        engine="zarr",
        consolidated=True,
    )
    assert _pyramid_groups(ts_dt) == ("/factor_1", "/factor_10")
