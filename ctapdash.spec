# PyInstaller spec for the CTAP Dashboard desktop app.
#
# onedir, not onefile: the bundle carries numpy, scipy, matplotlib's mpl-data
# and MNE, so onefile would re-extract several hundred megabytes to a temp
# directory on every single launch. onedir is also the only sane basis for a
# macOS .app bundle.
#
# The generic onedir/macOS/backend/data plumbing lives in
# workingtitle.desktop.freezing. This file only declares the recipe.
#
# Build with:  uv run pyinstaller ctapdash.spec --noconfirm --clean

import sys
from pathlib import Path

# The spec runs from the project root; put the source tree on the path so it
# can drive the extension build and read the app declaration.
sys.path.insert(0, "src")

from ctapdash.build import ASSETS
from ctapdash.desktop import APP
from workingtitle.desktop.freezing import (
    BundleSpec,
    PackageData,
    Submodules,
    build_bundle,
)


def keep_mne_module(name):
    """Keep core MNE families, but not downloaders, commands, or tests."""
    return not (
        name == "mne.datasets"
        or name.startswith("mne.datasets.")
        or name == "mne.commands"
        or name.startswith("mne.commands.")
        or "tests" in name.split(".")
    )


def runtime_data(entry):
    """Drop build-only Bokeh files and non-stub MNE data.

    Upstream hooks also collect package data, so this runs after Analysis.
    Keep runtime Bokeh bundles, templates, fonts and the compiled venn_ts
    extension. Only the build-time compiler/types/modules and maps are removed.
    Normalize Windows TOC paths too.
    """
    name = entry[0].replace("\\", "/")
    if name.startswith("bokeh/server/static/"):
        return not (
            name.startswith("bokeh/server/static/lib/")
            or name.startswith("bokeh/server/static/js/lib/")
            or name == "bokeh/server/static/js/compiler.js"
            or name.endswith((".d.ts", ".map"))
        )
    if name.startswith("mne/data/"):
        return name.endswith(".pyi")
    return True


recipe = BundleSpec(
    app=APP,
    entrypoint="src/ctapdash/__main__.py",
    root=Path(SPECPATH),
    packages=(
        # templates/ and static/ live inside the package and are found through
        # importlib.resources, so they must land at ctapdash/... in the bundle.
        PackageData("ctapdash"),
        # The venn_ts extension is served from its compiled bundle; without
        # these files the browser fails with "Cannot find module './vennrender'".
        PackageData(
            "venn_ts",
            includes=("bokeh.ext.json", "package.json", "dist/**"),
        ),
        # mplbed reads webaggext.js through importlib.resources and serves
        # matplotlib's backends/web_backend and mpl-data as static dirs.
        PackageData("mplbed"),
        # MNE builds its namespace at import time from .pyi stubs via
        # lazy_loader; if these are missing, `import mne` fails outright.
        PackageData("mne", includes=("**/*.pyi",)),
        # EEG montage/layout data is needed; anatomical surfaces, MEG helmets,
        # coil definitions and the large icos.fif.gz under data/ are not.
        PackageData(
            "mne",
            includes=("icons/**", "html_templates/**", "channels/data/**"),
        ),
    ),
    metadata=("mne", "mplbed"),
    submodules=(Submodules("mne", filter=keep_mne_module),),
    hiddenimports=(
        # Selected via matplotlib.use("module://mplbed.webaggext._impl").
        "mplbed.webaggext._impl",
        "mplbed.integration.starlette",
        "matplotlib.backends.backend_webagg_core",
        "matplotlib.backends.backend_agg",
        "mne.viz._mpl_figure",
    ),
    excludes=(
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "qtpy",
        "IPython",
        "pytest",
        "notebook",
        "marimo",
        "wand",
        "quart",
        "cefpython3",
        "jnius",
        "matplotlib.backends.backend_tk",
        "matplotlib.backends._backend_tk",
        "PIL._tkinter_finder",
    ),
    # Collects matplotlib data/metadata and installs the shared runtime hook
    # that puts its font cache under a writable per-executable directory.
    matplotlib=True,
    bundle_identifier="fi.helsinki.hipercog.ctapdash",
    # CLR reads the host executable's config before pythonnet loads. This must
    # sit beside the EXE: collecting it as data puts it in _internal instead.
    adjacent_files={"win32": ("ci/windows/ctapdash.exe.config",)},
    data_filter=runtime_data,
)

bundle = build_bundle(recipe, globals(), assets=ASSETS)
