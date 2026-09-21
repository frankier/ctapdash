"""The build self-check must fail when the Windows CLR bridge cannot load."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ctapdash import smoke


@pytest.mark.parametrize(
    "platform,backend,expected",
    [
        ("win32", SimpleNamespace(BrowserView=object), 0),
        ("win32", None, 1),  # None in sys.modules makes the import fail.
        ("linux", None, 0),
        ("darwin", None, 0),
    ],
)
def test_native_backend_smoke_check(monkeypatch, capsys, platform, backend, expected):
    monkeypatch.setattr(smoke, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setitem(sys.modules, "webview.platforms.winforms", backend)
    monkeypatch.setattr(smoke, "HTTP_CHECKS", [])
    monkeypatch.setattr(smoke, "IMPORT_CHECKS", [])
    monkeypatch.setattr(smoke, "_check_backend", lambda failures: None)
    monkeypatch.setattr(smoke, "_check_mne_plot", lambda failures: None)
    server = Mock(url="http://127.0.0.1:1234/")
    monkeypatch.setattr(
        smoke.desktop, "ServerThread", Mock(return_value=Mock(start=lambda: server))
    )

    assert smoke.run_smoke_test(object(), object()) == expected
    server.stop.assert_called_once_with()
    output = capsys.readouterr().out
    if platform == "win32":
        assert "Windows native backend" in output
    else:
        assert "Windows native backend" not in output
