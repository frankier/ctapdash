"""Reloader wiring for the compiled TypeScript sources."""

from pathlib import Path

from ctapdash import desktop


def make_tree(root: Path) -> None:
    for path in (
        root / "node_modules",
        root / ".venv",
        root / "build",
        root / "dist",
        root / "src/venn_ts/node_modules",
        root / "src/venn_ts/dist",
    ):
        path.mkdir(parents=True)


def test_watches_typescript_but_not_dependencies(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    make_tree(root)
    monkeypatch.setattr(desktop, "__file__", str(root / "src/ctapdash/desktop.py"))

    includes, excludes = desktop._typescript_reload_options()

    assert includes == ["*.ts"]
    assert set(excludes) == {
        str(root / "node_modules"),
        str(root / ".venv"),
        str(root / "build"),
        str(root / "dist"),
        str(root / "src/venn_ts/node_modules"),
        str(root / "src/venn_ts/dist"),
    }


def test_uvicorn_filter_accepts_extension_sources(monkeypatch, tmp_path):
    from uvicorn.config import Config
    from uvicorn.supervisors.watchfilesreload import FileFilter

    root = tmp_path.resolve()
    make_tree(root)
    monkeypatch.setattr(desktop, "__file__", str(root / "src/ctapdash/desktop.py"))
    includes, excludes = desktop._typescript_reload_options()
    config = Config(
        "ctapdash.asgi:create_app_from_env",
        factory=True,
        reload=True,
        reload_includes=includes,
        reload_excludes=excludes,
    )
    watch_filter = FileFilter(config)

    for relative in (
        "src/venn_ts/plot.py",
        "src/venn_ts/vennrender.ts",
        "src/js/index.ts",
    ):
        assert watch_filter(root / relative), relative
    for relative in (
        "README.md",
        "src/css/index.css",
        "src/venn_ts/node_modules/x/y.ts",
        "src/venn_ts/dist/lib/vennrender.d.ts",
        "node_modules/x/y.ts",
        ".venv/lib/x.ts",
        "dist/app/foo.ts",
    ):
        assert not watch_filter(root / relative), relative


def test_restarts_rebuild_before_spawning(monkeypatch):
    import ctapdash.build as build
    from uvicorn.supervisors import ChangeReload

    calls = []
    monkeypatch.setattr(build, "ensure_built", lambda: calls.append("build"))
    monkeypatch.setattr(ChangeReload, "restart", lambda self: calls.append("restart"))

    desktop._BuildOnReload.restart(object.__new__(desktop._BuildOnReload))

    assert calls == ["build", "restart"]
