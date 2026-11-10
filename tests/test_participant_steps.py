import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from starlette.requests import Request

from ctapdash import webapp


def test_steps_fragment_reports_deferred_sample_read_error(tmp_path, monkeypatch):
    step_dir = tmp_path / "1_load"
    step_dir.mkdir()
    path = step_dir / "participant.set"
    path.touch()
    monkeypatch.setattr(
        webapp,
        "participant_context",
        lambda request: {"participant": "participant", "step_sets": [(1, step_dir)]},
    )
    monkeypatch.setattr(
        webapp.ObservationData,
        "from_request",
        lambda request: SimpleNamespace(source_path=tmp_path),
    )
    monkeypatch.setattr(webapp, "_wait_metadata", AsyncMock())
    monkeypatch.setattr(webapp, "_recording_problems", lambda *args: {})
    plot = Mock(side_effect=OSError("truncated sample data"))
    read_eeglab = Mock(return_value=SimpleNamespace(plot=plot))
    monkeypatch.setattr(webapp, "read_eeglab", read_eeglab)
    template_response = Mock()
    monkeypatch.setattr(webapp.templates, "TemplateResponse", template_response)
    request = Request({"type": "http", "query_string": b"step_set=1"})

    asyncio.run(webapp.participant_step_sets_fragment(request))

    read_eeglab.assert_called_once_with(path, mmap=False, use_cache=False)
    plot.assert_called_once_with(show=False, scalings=None, clipping=None)
    assert template_response.call_args.args == (request, "participant_step_sets.html")
    context = template_response.call_args.kwargs["context"]
    assert context["current_step_set"] == 1
    assert context["recording_problems"] == {
        1: "1_load/participant.set: truncated sample data"
    }
    assert "eeg_fig" not in context
