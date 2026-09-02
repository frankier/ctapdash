from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from bokeh.document import Document
from bokeh.models import HoverTool, Select as BkSelect
from bokeh.plotting._figure import figure as BkFigure

from ctapdash import webapp
from ctapdash.config import SETTINGS


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
    ts_dt = xr.open_datatree(
        pyramid_source / "01_test" / "p.set.pyramid",
        engine="zarr",
        consolidated=True,
    )
    assert webapp._pyramid_groups(ts_dt) == ("/factor_1", "/factor_10")


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
