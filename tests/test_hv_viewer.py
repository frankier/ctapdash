from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from bokeh.document import Document
from bokeh.models import CheckboxButtonGroup, HoverTool, Select as BkSelect
from bokeh.models.renderers import GlyphRenderer
from bokeh.plotting._figure import figure as BkFigure

from ctapdash import webapp
from ctapdash.config import SETTINGS
from ctapdash.plotting.venn_ts.renderer import VennTimeSeriesRenderer


def test_comparison_channel_geometry_modes():
    extrema = [(-1, 1, -3, 2), (-2, 2, -1, 1)]

    overplot, overplot_range = webapp._comparison_channel_geometry(extrema, "overplot")
    assert overplot_range == (0.0, 2.0)
    assert overplot[0]["outside"] is True
    assert overplot[0]["actual_min"] * overplot[0]["scale"] + overplot[0]["offset"] < 1.1

    stretch, stretch_range = webapp._comparison_channel_geometry(extrema, "stretch")
    assert stretch_range[1] > overplot_range[1]
    first_low = stretch[0]["actual_min"] * stretch[0]["scale"] + stretch[0]["offset"]
    second_high = stretch[1]["actual_max"] * stretch[1]["scale"] + stretch[1]["offset"]
    assert first_low > second_high

    normalized, normalized_range = webapp._comparison_channel_geometry(extrema, "normalize")
    assert normalized_range == overplot_range
    for item in normalized:
        mapped_span = (item["actual_max"] - item["actual_min"]) * item["scale"]
        assert mapped_span == pytest.approx(0.8)


def test_comparison_view_uses_one_webgl_figure_with_overlaid_layers(tmp_path):
    from ctapdash.plotting.venn_ts.range_series import RecordingTileSource

    class FakeRecording:
        ch_names = ["Fz", "Cz"]

        def __init__(self, offset):
            self.offset = offset

        def mmap(self, *, return_xarray=False):
            times = np.arange(20, dtype=float) * 0.1
            values = np.vstack((np.sin(times), np.cos(times))) + self.offset
            return xr.DataArray(values, coords=(self.ch_names, times), dims=("ch", "time"))

    steps = []
    for number in (1, 2):
        step = tmp_path / f"{number:02d}_step"
        step.mkdir()
        (step / "p.set").touch()
        steps.append((number, step))
    dataset = webapp.ObservationData(tmp_path, "p", "fake")
    dataset.get_steps = lambda: steps

    def fake_source(path):
        offset = int(Path(path).parent.name[:2])
        return RecordingTileSource(path, recording=FakeRecording(offset))

    with (
        patch.object(
            webapp.ObservationData,
            "from_bokeh_doc",
            classmethod(lambda cls, doc: dataset),
        ),
        patch(
            "ctapdash.plotting.venn_ts.range_series.RecordingTileSource",
            side_effect=fake_source,
        ),
    ):
        doc = Document()
        webapp.venn_time_series_bokeh(doc)

    figure = doc.roots[0].select_one({"type": BkFigure})
    custom = list(figure.select({"type": VennTimeSeriesRenderer}))
    glyphs = list(figure.select({"type": GlyphRenderer}))
    layer_control = doc.roots[0].select_one({"type": CheckboxButtonGroup})
    assert figure.output_backend == "webgl"
    assert len(custom) == 1
    assert custom[0].channel_tile_size == 1
    assert len(glyphs) == 2
    assert all(glyph.data_source in {custom[0].line_source_a, custom[0].line_source_b} for glyph in glyphs)
    assert layer_control.labels == ["Venn", "Lines"]
    assert layer_control.active == [0, 1]
    layer_control.active = [1]
    assert custom[0].venn_visible is False
    assert custom[0].lines_visible is True
    layer_control.active = [0]
    assert custom[0].venn_visible is True
    assert custom[0].lines_visible is False


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
    from ctapdash.pyramid import _pyramid_groups
    ts_dt = xr.open_datatree(
        pyramid_source / "01_test" / "p.set.pyramid",
        engine="zarr",
        consolidated=True,
    )
    assert _pyramid_groups(ts_dt) == ("/factor_1", "/factor_10")


def test_time_series_bokeh_accepts_unnamed_data_array(tmp_path):
    """Generated pyramids contain an unnamed DataArray, not a ``data`` variable."""
    step = tmp_path / "01_test"
    step.mkdir()
    (step / "p.set").write_text("")

    times = np.linspace(0, 1, 100)
    channels = ["Fp1", "Fp2"]
    values = np.arange(200, dtype=float).reshape(2, 100)
    for factor in (1, 10):
        xr.DataArray(
            values[:, ::factor],
            coords=(channels, times[::factor]),
            dims=("ch", "time"),
        ).to_zarr(step / "p.set.pyramid", group=f"factor_{factor}", mode="a")

    with patch.object(
        webapp.ObservationData,
        "from_bokeh_doc",
        classmethod(lambda cls, doc: webapp.ObservationData(tmp_path, "p", "fake")),
    ):
        doc = Document()
        webapp.time_series_bokeh(doc)

    fig = doc.roots[0].select_one({"type": BkFigure})
    assert len(fig.renderers) == 2
    assert {len(renderer.data_source.data["time"]) for renderer in fig.renderers} == {10}
    assert all(
        renderer.data_source.data["amplitude"].ndim == 1
        for renderer in fig.renderers
    )


def test_hv_viewer_bokeh_webgl(pyramid_source):
    with patch.object(
        webapp.ObservationData,
        "from_bokeh_doc",
        classmethod(lambda cls, doc: webapp.ObservationData(pyramid_source, "p", "fake")),
    ):
        doc = Document()
        webapp.hv_viewer_bokeh(doc)

    assert len(doc.roots) == 1
    select = doc.roots[0].select_one({"type": BkSelect})
    assert select.options == ["WebGL", "Datashader"]
    assert select.value == "WebGL"

    fig = doc.roots[0].select_one({"type": BkFigure})
    # One line per channel...
    assert len(fig.renderers) == 8
    # ...from the finest pyramid level, which fits the WebGL point budget.
    assert {len(r.data_source.data["time"]) for r in fig.renderers} == {100_000}
    assert any(isinstance(tool, HoverTool) for tool in fig.tools)
