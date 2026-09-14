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
    sources = list((package / "frontend").glob("*.ts"))
    outputs = [package / "static/generated" / source.with_suffix(".js").name for source in sources]
    sources += [root / name for name in ("package.json", "package-lock.json", "tsconfig.json")]
    sources = [path for path in sources if path.exists()]
    current = (outputs and all(path.exists() for path in outputs)
               and max(path.stat().st_mtime for path in sources)
               <= min(path.stat().st_mtime for path in outputs))
    if force or not current:
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        if not (root / "node_modules/typescript").exists():
            subprocess.run([npm, "ci", "--ignore-scripts"], cwd=root, check=True)
        subprocess.run([npm, "run", "build"], cwd=root, check=True)
    from venn_ts.build import ensure_extension_built
    ensure_extension_built(force=force, verbose=verbose)


if __name__ == "__main__":
    ensure_built(force="--force" in sys.argv, verbose=True)
