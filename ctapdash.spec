# PyInstaller spec for the CTAP Dashboard desktop app.
#
# onedir, not onefile: the bundle carries numpy, scipy, matplotlib's mpl-data
# and MNE, so onefile would re-extract several hundred megabytes to a temp
# directory on every single launch. onedir is also the only sane basis for a
# macOS .app bundle.
#
# Build with:  uv run pyinstaller ctapdash.spec --noconfirm --clean

import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

datas = []
# templates/ and static/ live inside the package and are found through
# importlib.resources, so they must land at ctapdash/... in the bundle.
datas += collect_data_files("ctapdash")
# mplbed reads webaggext.js through importlib.resources.
datas += collect_data_files("mplbed")
# mplbed serves matplotlib's backends/web_backend and mpl-data as static dirs.
datas += collect_data_files("matplotlib")
# MNE builds its namespace at import time from .pyi stubs via lazy_loader; if
# these are missing, `import mne` fails outright.
datas += collect_data_files("mne", includes=["**/*.pyi"])
datas += collect_data_files(
    "mne",
    includes=["data/**", "icons/**", "html_templates/**", "channels/data/**"],
)
datas += copy_metadata("mne")
datas += copy_metadata("matplotlib")
datas += copy_metadata("mplbed")

hiddenimports = []
# lazy_loader defeats static analysis for the whole mne namespace.
hiddenimports += collect_submodules("mne")
# uvicorn picks its protocol/loop/lifespan implementations by string.
hiddenimports += collect_submodules("uvicorn")
hiddenimports += [
    # Selected via matplotlib.use("module://mplbed.webaggext._impl").
    "mplbed.webaggext._impl",
    "mplbed.integration.starlette",
    "matplotlib.backends.backend_webagg_core",
    "matplotlib.backends.backend_agg",
    "mne.viz._mpl_figure",
    "websockets",
    "anyio._backends._asyncio",
    "encodings.idna",
]

excludes = [
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
]

if IS_LINUX:
    # No native window on Linux; see ctapdash/desktop.py.
    excludes += ["webview", "gi", "pythonnet", "clr"]
else:
    # PyInstaller pulls in every pywebview backend it can find, used or not.
    excludes += ["webview.platforms.gtk", "webview.platforms.qt", "webview.platforms.android"]
    if IS_MACOS:
        hiddenimports += ["webview.platforms.cocoa"]
    else:
        hiddenimports += ["webview.platforms.edgechromium", "webview.platforms.winforms"]


a = Analysis(
    ["src/ctapdash/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=["rthook_mplconfig.py"],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ctapdash",
    debug=False,
    strip=False,
    upx=False,
    # Keep the console for now: while this packaging is new, a crash with no
    # console is an invisible flash. Revisit once builds are reliably green.
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ctapdash",
)

if IS_MACOS:
    app = BUNDLE(
        coll,
        name="CTAP Dashboard.app",
        bundle_identifier="fi.helsinki.hipercog.ctapdash",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "CFBundleShortVersionString": "0.1.0",
        },
    )
