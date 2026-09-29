"""The app's asset plan and its wiring to the shared desktop runner."""

from argparse import Namespace

import pytest

from ctapdash import cli
from ctapdash.build import ASSETS


def test_reload_filter_accepts_declared_sources_and_python():
    from uvicorn.config import Config
    from uvicorn.supervisors.watchfilesreload import FileFilter

    config = Config(
        "ctapdash.asgi:create_app_from_env",
        factory=True,
        reload=True,
        reload_dirs=[str(ASSETS.root)],
    )
    accepts = ASSETS.watch_filter(FileFilter(config))

    for relative in (
        "src/js/index.ts",
        "src/css/index.css",
        "src/ctapdash/templates/base.html",
        "src/venn_ts/vennrender.ts",
        "package-lock.json",
        "src/ctapdash/webapp.py",
    ):
        assert accepts(ASSETS.root / relative), relative


def test_reload_filter_rejects_generated_and_dependency_trees():
    from uvicorn.config import Config
    from uvicorn.supervisors.watchfilesreload import FileFilter

    config = Config(
        "ctapdash.asgi:create_app_from_env",
        factory=True,
        reload=True,
        reload_dirs=[str(ASSETS.root)],
    )
    accepts = ASSETS.watch_filter(FileFilter(config))

    for relative in (
        "README.md",
        "node_modules/x/y.ts",
        "src/venn_ts/node_modules/x.ts",
        "src/venn_ts/dist/venn_ts.js",
        "src/ctapdash/static/generated/index.js",
        ".desktop-build/frontend.json",
    ):
        assert not accepts(ASSETS.root / relative), relative


def test_build_extension_reports_toolchain_failure_as_unavailable(monkeypatch):
    from workingtitle.desktop import BuildUnavailable

    import venn_ts.build as venn
    from ctapdash import build

    def fail(**kwargs):
        raise venn.ExtensionBuildFailed("no node")

    monkeypatch.setattr(venn, "build_extension_bundle", fail)
    with pytest.raises(BuildUnavailable, match="no node"):
        build.build_extension()


def test_build_extension_propagates_integrity_errors(monkeypatch):
    import venn_ts.build as venn
    from ctapdash import build

    def broken(**kwargs):
        raise RuntimeError("missing modules")

    monkeypatch.setattr(venn, "build_extension_bundle", broken)
    # A broken bundle must never be treated as a fallback candidate.
    with pytest.raises(RuntimeError, match="missing modules"):
        build.build_extension()


def test_installed_assets_are_validation_only():
    from ctapdash.build import _installed_assets

    plan = _installed_assets()
    assert plan.source is False
    outputs = {output for step in plan.steps for output in step.outputs}
    assert "ctapdash/static/generated/index.js" in outputs
    assert "venn_ts/dist/venn_ts.json" in outputs


def test_prepare_returns_environment_override(tmp_path, monkeypatch):
    from ctapdash import config

    path = tmp_path / "conf.toml"
    path.write_text("[sources]\n")
    monkeypatch.setattr(config, "load_from_file", lambda p: None)
    assert cli.prepare(Namespace(config=path)) == {config.ENV_VAR: str(path)}


def test_prepare_rejects_missing_config(tmp_path):
    with pytest.raises(ValueError, match="no such configuration file"):
        cli.prepare(Namespace(config=tmp_path / "missing.toml"))


def test_main_wires_the_shared_runner(monkeypatch):
    captured = {}

    def fake_run_cli(spec, argv, **kwargs):
        captured.update(spec=spec, argv=argv, **kwargs)
        return 0

    monkeypatch.setattr(cli, "run_cli", fake_run_cli)
    assert cli.main(["--config", "conf.toml"]) == 0
    assert captured["spec"] is cli.APP
    assert captured["assets"] is ASSETS
    assert captured["session"] is cli.SESSION
    assert captured["smoke_test"] is cli.smoke_test
