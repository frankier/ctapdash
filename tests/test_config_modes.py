"""Configuration mode and managed source persistence."""

import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.exceptions import HTTPException

from ctapdash import cli, config, setup_ui
from ctapdash.config import SETTINGS


@pytest.fixture
def managed_path(tmp_path, monkeypatch):
    path = tmp_path / "config" / "config.toml"
    monkeypatch.setattr(config, "managed_config_path", lambda: path)
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    monkeypatch.setattr(SETTINGS, "sources", {})
    monkeypatch.setattr(SETTINGS, "managed", True)
    monkeypatch.setattr(SETTINGS, "loaded_from", None)
    return path


def test_managed_startup_restores_sources(managed_path, tmp_path):
    assert cli.prepare(Namespace(config=None)) == {}
    assert SETTINGS.managed and not SETTINGS.configured
    config.save_managed_sources({"sample": str(tmp_path)})
    assert managed_path.exists()

    SETTINGS.sources = {}
    config.load_startup()
    assert SETTINGS.managed
    assert SETTINGS.sources == {"sample": str(tmp_path)}


def test_explicit_paths_are_immutable(managed_path, tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.toml"
    explicit.write_text('[sources]\n"fixed" = "/data/fixed"\n')
    monkeypatch.setenv(config.ENV_VAR, str(explicit))
    config.load_startup()
    assert not SETTINGS.managed
    assert SETTINGS.sources == {"fixed": "/data/fixed"}
    with pytest.raises(ValueError, match="read only"):
        config.save_managed_sources({})

    monkeypatch.setenv(config.ENV_VAR, str(managed_path))
    assert cli.prepare(Namespace(config=explicit)) == {config.ENV_VAR: str(explicit)}
    assert not SETTINGS.managed
    assert SETTINGS.loaded_from == explicit


def test_failed_replace_keeps_sources_and_cleans_temporary_file(
    managed_path, tmp_path, monkeypatch
):
    SETTINGS.sources = {"first": str(tmp_path)}

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(config.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        config.save_managed_sources({"second": str(tmp_path)})
    assert SETTINGS.sources == {"first": str(tmp_path)}
    assert not list(managed_path.parent.iterdir())


def test_setup_changes_persist_and_immutable_routes_reject(
    managed_path, tmp_path, monkeypatch
):
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(cache_hurrier=Mock()))
    )
    form = {"path": str(tmp_path), "name": "sample", "modal": "1"}
    monkeypatch.setattr(setup_ui, "_read_form", AsyncMock(side_effect=lambda _: form))
    monkeypatch.setattr(setup_ui, "_render", lambda _request, **kwargs: kwargs)

    added = asyncio.run(setup_ui.setup_add(request))
    assert added == {"modal": True, "changed": True}
    assert SETTINGS.sources == {"sample": str(tmp_path)}
    request.app.state.cache_hurrier.add_dataset.assert_called_once_with(str(tmp_path))

    form = {"name": "sample", "modal": "1"}
    removed = asyncio.run(setup_ui.setup_remove(request))
    assert removed == {"modal": True, "changed": True}
    assert SETTINGS.sources == {}
    assert managed_path.read_text() == "[sources]\n"

    SETTINGS.managed = False
    with pytest.raises(HTTPException) as error:
        asyncio.run(setup_ui.setup_add(request))
    assert error.value.status_code == 403


def test_setup_write_error_keeps_visible_sources(managed_path, tmp_path, monkeypatch):
    SETTINGS.sources = {"first": str(tmp_path)}
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(cache_hurrier=Mock()))
    )
    monkeypatch.setattr(
        setup_ui,
        "_read_form",
        AsyncMock(return_value={"path": str(tmp_path), "name": "second", "modal": "1"}),
    )
    monkeypatch.setattr(setup_ui, "_render", lambda _request, **kwargs: kwargs)
    monkeypatch.setattr(
        config, "save_managed_sources", Mock(side_effect=OSError("disk full"))
    )

    result = asyncio.run(setup_ui.setup_add(request))
    assert result == {"modal": True, "error": "disk full"}
    assert SETTINGS.sources == {"first": str(tmp_path)}
    request.app.state.cache_hurrier.add_dataset.assert_not_called()
