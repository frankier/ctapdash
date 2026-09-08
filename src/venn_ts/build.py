"""Keep the compiled BokehJS bundle for this extension up to date.

Bokeh serves custom extension models from ``dist/``, but nothing on the
normal import path rebuilds the bundle when the TypeScript sources change,
so a stale bundle shows up as ``Cannot find module './...'`` in the browser.
``ensure_extension_built()`` compares the bundle's modification time against
the sources and rebuilds through ``bokeh.ext.build`` (the internal API the
``bokeh build`` command line wraps) when anything is newer.

Frozen builds ship a bundle made at packaging time and have no node runtime,
so the bundling process sets :data:`SKIP_ENV_VAR` to make this module a
no-op there; see ``rthook_extbuild.py`` and ``ctapdash.spec``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Set to a non-empty value other than "0" to skip the check entirely.
SKIP_ENV_VAR = "CTAPDASH_SKIP_EXTENSION_BUILD"

# Files that participate in the build. The TypeScript sources are found
# dynamically so new files are picked up without touching this module.
_TOP_LEVEL_SOURCES = ("bokeh.ext.json", "package.json", "tsconfig.json")
_PRUNE_DIRS = ("node_modules", "dist", "__pycache__")


def _extension_dir() -> Path:
    return Path(__file__).resolve().parent


def _source_files(base_dir: Path) -> list[Path]:
    sources = [base_dir / name for name in _TOP_LEVEL_SOURCES]
    for path in base_dir.rglob("*.ts"):
        if not any(part in _PRUNE_DIRS for part in path.relative_to(base_dir).parts):
            sources.append(path)
    return [path for path in sources if path.exists()]


def _is_up_to_date(base_dir: Path) -> bool:
    bundle = base_dir / "dist" / "venn_ts.json"
    if not bundle.exists():
        return False
    bundle_mtime = bundle.stat().st_mtime
    return all(path.stat().st_mtime <= bundle_mtime for path in _source_files(base_dir))


def _missing_modules(base_dir: Path) -> set[str]:
    """Report source modules that the compiler silently left out of the bundle.

    The linker resolves imports after compilation and can drop a module while
    still exiting with status 0. This happened when a stale ``dist/lib/x/``
    directory made ``./x`` ambiguous, leaving ``require('./x')`` dangling in
    the bundle and the browser failing with ``Cannot find module './x'``.
    """
    bundle = base_dir / "dist" / "venn_ts.json"
    if not bundle.exists():
        return set()
    with bundle.open() as io:
        import json

        manifest = json.load(io)
    bundled = {
        Path(artifact["module"]["base_path"]).with_suffix("").as_posix()
        for artifact in manifest["artifacts"]
    }
    sources = {
        path.relative_to(base_dir).with_suffix("").as_posix()
        for path in _source_files(base_dir)
        if path.suffix == ".ts"
    }
    return sources - bundled


def ensure_extension_built(*, force: bool = False, verbose: bool = False) -> None:
    """Rebuild the extension bundle if any source is newer than it.

    A failure is tolerated when a (possibly stale) bundle already exists so
    that a broken or missing node toolchain does not take the whole dashboard
    down; without any bundle there is nothing to serve, so it raises.
    """
    if os.environ.get(SKIP_ENV_VAR, "").strip() not in ("", "0"):
        return

    base_dir = _extension_dir()
    if not force and _is_up_to_date(base_dir):
        return

    # Imported lazily: pulling in the compiler plumbing is wasted work on
    # every launch when the bundle is already current.
    from bokeh.ext import build

    if not build(base_dir, verbose=verbose):
        bundle = base_dir / "dist" / "venn_ts.json"
        if bundle.exists():
            print(
                f"warning: could not rebuild the {base_dir.name} extension bundle; "
                "continuing with the existing one (it may be out of date)",
                file=sys.stderr,
            )
            return
        raise RuntimeError(
            f"could not build the {base_dir.name} extension bundle in "
            f"{base_dir}; check that node and npm are installed"
        )

    missing = _missing_modules(base_dir)
    if missing:
        raise RuntimeError(
            f"the {base_dir.name} extension bundle is missing modules "
            f"{sorted(missing)}; delete {base_dir / 'dist'} and rebuild "
            "(this is usually caused by stale files under dist/lib)"
        )


if __name__ == "__main__":
    ensure_extension_built(force="--force" in sys.argv, verbose=True)
