"""Self-check used to validate frozen builds.

A plain request to `/` would pass even in a bundle where MNE is unusable: the
heavy machinery is only reached when a participant is opened, and MNE resolves
its submodules lazily. This exercises those paths directly, because missing
modules are the failure mode PyInstaller actually produces.
"""

import urllib.error
import urllib.request

from ctapdash import desktop


HTTP_CHECKS = [
    # base.html renders mplbed_head(), which needs the middleware's request
    # context, so this covers the whole head-injection path.
    ("setup", "mpl.js"),
    ("static/img/flask.gif", None),
    ("static/vendor/htmx.min.js", None),
    ("static/vendor/tailwind-browser.js", None),
    ("webagg/mpl.js", None),
    ("webagg/_static/js/mpl.js", None),
]

IMPORT_CHECKS = [
    # MNE attaches its namespace from .pyi stubs at runtime; if PyInstaller
    # dropped them, every one of these fails.
    ("mne.io.read_raw_eeglab", "mne.io", "read_raw_eeglab"),
    ("mne.read_epochs", "mne", "read_epochs"),
    ("mne.viz._mpl_figure", "mne.viz._mpl_figure", "MNEBrowseFigure"),
    ("ctapdash.plotting.mne", "ctapdash.plotting.mne", "OnionskinMNEBrowseFigure"),
    ("ctapdash.io", "ctapdash.io", "read_eeglab"),
    ("scipy.io.loadmat", "scipy.io", "loadmat"),
    ("scipy.stats.describe", "scipy.stats", "describe"),
    ("xarray.Dataset", "xarray", "Dataset"),
    ("PIL.Image", "PIL.Image", "open"),
    ("PIL.ImageChops", "PIL.ImageChops", "difference"),
    (
        "uvicorn websockets protocol",
        "uvicorn.protocols.websockets.websockets_impl",
        "WebSocketProtocol",
    ),
]


def _check_http(url, failures, contains=None):
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            assert response.status == 200, response.status
            body = response.read()
            assert body, "empty body"
        if contains is not None:
            assert contains.encode() in body, f"{contains!r} missing from body"
    except Exception as err:
        failures.append(f"GET {url}: {err!r}")
        print(f"FAIL GET {url}: {err!r}", flush=True)
    else:
        print(f"OK   GET {url}", flush=True)


def _check_import(label, module, attr, failures):
    try:
        imported = __import__(module, fromlist=[attr])
        getattr(imported, attr)
    except Exception as err:
        failures.append(f"import {label}: {err!r}")
        print(f"FAIL import {label}: {err!r}", flush=True)
    else:
        print(f"OK   import {label}", flush=True)


def _check_backend(failures):
    """Render a figure through mplbed's backend.

    The backend is selected by the string "module://mplbed.webaggext._impl",
    which static analysis cannot see, so this is exactly the kind of thing a
    bundle drops. Building the HTML itself needs a live request context, which
    the GET of /setup below already exercises via mplbed_head().
    """
    try:
        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.pyplot import _get_backend_mod

        backend = matplotlib.get_backend()
        assert "mplbed" in backend, f"unexpected backend {backend!r}"
        figure = Figure()
        figure.gca().plot([0, 1], [0, 1])
        manager = _get_backend_mod().new_figure_manager_given_figure(id(figure), figure)
        manager.canvas.draw()
    except Exception as err:
        failures.append(f"mplbed backend: {err!r}")
        print(f"FAIL mplbed backend: {err!r}", flush=True)
    else:
        print("OK   mplbed backend", flush=True)


def _check_mne_plot(failures):
    """Render an MNE browse figure on synthetic data.

    This is the heaviest code path in the app (mne.viz plus the matplotlib
    backend) and the one a bundle is most likely to break. Synthetic data keeps
    it runnable in CI with no fixtures.
    """
    try:
        import numpy as np
        import mne

        info = mne.create_info(["a", "b", "c"], sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((3, 500)), info, verbose="error")
        figure = raw.plot(show=False, verbose="error")
        figure.canvas.draw()
    except Exception as err:
        failures.append(f"mne plot: {err!r}")
        print(f"FAIL mne plot: {err!r}", flush=True)
    else:
        print("OK   mne plot", flush=True)


def _check_describe(failures):
    """Exercise the public MNE-to-xarray statistics path."""
    try:
        import mne
        import numpy as np

        from ctapdash.stats import describe

        info = mne.create_info(["a", "b"], sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(np.arange(20.0).reshape(2, 10), info, verbose="error")
        summary = describe(raw)
        assert summary.sizes == {"recording": 1, "channel": 2}
        assert list(summary.data_vars) == [
            "nobs",
            "min",
            "max",
            "mean",
            "variance",
            "skewness",
            "kurtosis",
        ]
    except Exception as err:
        failures.append(f"describe: {err!r}")
        print(f"FAIL describe: {err!r}", flush=True)
    else:
        print("OK   describe", flush=True)


def run_smoke_test(app, sock):
    server = desktop.ServerThread(app, sock, log_level="warning").start()
    failures = []
    try:
        print(f"Serving at {server.url}", flush=True)
        for path, contains in HTTP_CHECKS:
            _check_http(server.url + path, failures, contains=contains)
        for label, module, attr in IMPORT_CHECKS:
            _check_import(label, module, attr, failures)
        _check_backend(failures)
        _check_mne_plot(failures)
        _check_describe(failures)
    finally:
        server.stop()

    if failures:
        print("\nSMOKE TEST FAILED:", flush=True)
        for failure in failures:
            print(f" - {failure}", flush=True)
        return 1
    print("\nSMOKE TEST PASSED", flush=True)
    return 0
