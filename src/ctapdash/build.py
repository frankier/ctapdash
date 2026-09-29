"""Frontend and Bokeh-extension build plan.

``workingtitle.desktop.AssetPlan`` owns the incremental logic, npm handling,
fingerprints, and reload watching. This module only declares what the dashboard
compiles and how. The same plan drives source startup, reload rebuilds, and
PyInstaller packing, so none of them can disagree about what is fresh.
"""

from __future__ import annotations

import sys
from pathlib import Path

from workingtitle.desktop import AssetPlan, BuildUnavailable, NpmBuild, PythonBuild

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_FRONTEND_OUTPUTS = (
    "src/ctapdash/static/generated/index.js",
    "src/ctapdash/static/generated/index.css",
)
_EXTENSION_INPUTS = (
    "src/venn_ts/**/*.ts",
    "src/venn_ts/bokeh.ext.json",
    "src/venn_ts/package.json",
    "src/venn_ts/package-lock.json",
    "src/venn_ts/tsconfig.json",
)
_EXTENSION_OUTPUTS = (
    "src/venn_ts/dist/venn_ts.json",
    "src/venn_ts/dist/venn_ts.js",
)


def build_extension():
    """Compile the Bokeh extension for the asset plan.

    A toolchain failure becomes ``BuildUnavailable``: development keeps the
    stale bundle and retries next launch, while packaging (strict) fails. A
    bundle that fails its integrity check is never safe to serve and always
    propagates.
    """
    from venn_ts.build import ExtensionBuildFailed, build_extension_bundle

    try:
        build_extension_bundle()
    except ExtensionBuildFailed as exc:
        raise BuildUnavailable(str(exc)) from exc


def _source_steps():
    return (
        NpmBuild(
            name="frontend",
            # Tailwind scans the templates, so they are inputs too.
            inputs=(
                "src/js/**",
                "src/css/**",
                "src/ctapdash/templates/**",
            ),
            outputs=_FRONTEND_OUTPUTS,
        ),
        PythonBuild(
            name="venn-ts",
            inputs=_EXTENSION_INPUTS,
            outputs=_EXTENSION_OUTPUTS,
            action="ctapdash.build:build_extension",
        ),
    )


def _installed_assets():
    """Validate assets as a wheel or PyInstaller bundle ships them.

    The packages are siblings under the import root, so one root covers both.
    """
    from importlib.resources import files

    root = Path(str(files("ctapdash"))).parent
    return AssetPlan(
        root=root,
        source=False,
        steps=(
            NpmBuild(
                name="frontend",
                inputs=(),
                outputs=(
                    "ctapdash/static/generated/index.js",
                    "ctapdash/static/generated/index.css",
                ),
            ),
            PythonBuild(
                name="venn-ts",
                inputs=(),
                outputs=(
                    "venn_ts/dist/venn_ts.json",
                    "venn_ts/dist/venn_ts.js",
                ),
                action="ctapdash.build:build_extension",
            ),
        ),
    )


if getattr(sys, "frozen", False):
    ASSETS = _installed_assets()
else:
    ASSETS = AssetPlan(root=PROJECT_ROOT, steps=_source_steps())


if __name__ == "__main__":
    ASSETS.ensure_built(force="--force" in sys.argv)
