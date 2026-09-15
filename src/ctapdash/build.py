"""Build browser assets and the Bokeh extension before startup or packaging."""
import os
from pathlib import Path
import subprocess
import sys

SKIP_ENV_VAR = "CTAPDASH_SKIP_EXTENSION_BUILD"


def ensure_built(*, force=False, verbose=False):
    if os.environ.get(SKIP_ENV_VAR, "").strip() not in ("", "0"):
        return
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    sources = [path for directory in ("src/js", "src/css")
               for path in (root / directory).rglob("*") if path.is_file()]
    outputs = [package / "static/generated/index.js",
               package / "static/generated/index.css"]
    sources += [root / name for name in ("package.json", "package-lock.json")]
    current = (all(path.exists() for path in outputs)
               and max(path.stat().st_mtime for path in sources)
               <= min(path.stat().st_mtime for path in outputs))
    if force or not current:
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        # Written by every npm install, so this is the cheapest proof that the
        # tree is populated at all.
        if not (root / "node_modules/.package-lock.json").exists():
            subprocess.run([npm, "ci", "--ignore-scripts"], cwd=root, check=True)
        subprocess.run([npm, "run", "build"], cwd=root, check=True)
    from venn_ts.build import ensure_extension_built
    ensure_extension_built(force=force, verbose=verbose)


if __name__ == "__main__":
    ensure_built(force="--force" in sys.argv, verbose=True)
