"""Running the dashboard as a desktop application.

Windows and macOS get a native pywebview window. Linux does not: pywebview
there needs PyGObject/WebKitGTK, which cannot be frozen reliably, so the
frozen Linux binary serves and opens the system browser instead.
"""

import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from uvicorn.supervisors import ChangeReload


# The active pywebview Window, or None when running browser-backed. setup_ui
# reads this to decide between a native folder dialog and a text input.
WINDOW = None


def native_window_supported():
    return not sys.platform.startswith("linux")


def bind_socket(host="127.0.0.1", port=0):
    """Bind a listening socket up front, so the port we report is the port we use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    sock.set_inheritable(True)
    return sock


def socket_url(sock):
    host, port = sock.getsockname()[:2]
    return f"http://{host}:{port}/"


class _Server(uvicorn.Server):
    def install_signal_handlers(self):
        # Signal handlers can only be installed on the main thread, and the
        # main thread belongs to pywebview.
        pass


class ServerThread:
    def __init__(self, app, sock, log_level="warning"):
        self.sock = sock
        self.server = _Server(uvicorn.Config(app, log_level=log_level, ws="websockets"))
        self.thread = threading.Thread(
            target=self.server.run,
            kwargs={"sockets": [sock]},
            daemon=True,
            name="uvicorn",
        )

    @property
    def url(self):
        return socket_url(self.sock)

    def start(self, timeout=60.0):
        self.thread.start()
        deadline = time.monotonic() + timeout
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("Server thread died during startup")
            if time.monotonic() > deadline:
                raise TimeoutError("Server did not start in time")
            time.sleep(0.05)
        return self

    def stop(self, timeout=10.0):
        self.server.should_exit = True
        self.thread.join(timeout)
        if self.thread.is_alive():
            self.server.force_exit = True
            self.thread.join(5.0)


def run_window(app, sock, title="CTAP Dashboard", log_level="warning"):
    global WINDOW
    import webview

    server = ServerThread(app, sock, log_level=log_level).start()
    WINDOW = webview.create_window(title, server.url, width=1400, height=900)
    try:
        # Must own the main thread; macOS will not run a UI anywhere else.
        webview.start()
    finally:
        WINDOW = None
        server.stop()


def run_browser(app, sock, open_browser=True, log_level="info"):
    server = ServerThread(app, sock, log_level=log_level).start()
    print(f"CTAP Dashboard running at {server.url}  (Ctrl-C to quit)", flush=True)
    if open_browser:
        webbrowser.open(server.url)
    try:
        while server.thread.is_alive():
            server.thread.join(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def _typescript_reload_options():
    """Watch patterns and exclusions that keep the compiled code fresh.

    uvicorn watches ``*.py`` by default, but the Bokeh application is compiled
    from TypeScript. Without ``*.ts`` in the includes, edits to the extension
    neither restart the server nor rebuild the BokehJS bundle. Build outputs,
    virtualenvs and ``node_modules`` trees are full of ``.ts`` and ``.d.ts``
    files and must stay out, or a build would immediately trigger the next
    reload.
    """
    package_dir = Path(__file__).resolve().parent
    source_dir = package_dir.parent
    root = source_dir.parent
    extension_dir = source_dir / "venn_ts"
    excludes = [
        str(path)
        for path in (
            root / "node_modules",
            root / ".venv",
            root / "build",
            root / "dist",
            extension_dir / "node_modules",
            extension_dir / "dist",
        )
        if path.is_dir()
    ]
    return ["*.ts"], excludes


class _BuildOnReload(ChangeReload):
    """Rebuild compiled browser assets before each restart.

    Adding TypeScript to the watched patterns restarts the server, but the
    subprocess still serves the bundle built at startup. Rebuilding here, in
    the parent, means every restart picks up the edited sources; the frozen
    build's skip flag turns this into a no-op (see rthook_extbuild.py).
    """

    def restart(self):
        from ctapdash.build import ensure_built

        ensure_built()
        super().restart()


def run_reload(host="127.0.0.1", port=0, open_browser=True, log_level="info"):
    """Serve with uvicorn's reloader, restarting when source files change.

    The reloader runs the application in a spawned subprocess, so unlike
    run_browser and run_window it takes an app factory import string rather
    than the app object. Binding the socket here keeps the reported port
    authoritative and lets every restart share the same socket.

    Python sources restart the server out of the box. The Bokeh extension's
    TypeScript sources are watched explicitly and compiled before each restart
    (see _typescript_reload_options and _BuildOnReload).
    """
    sock = bind_socket(host, port)
    reload_includes, reload_excludes = _typescript_reload_options()
    config = uvicorn.Config(
        "ctapdash.asgi:create_app_from_env",
        factory=True,
        reload=True,
        reload_includes=reload_includes,
        reload_excludes=reload_excludes,
        log_level=log_level,
        ws="websockets",
    )
    server = uvicorn.Server(config)
    url = socket_url(sock)
    print(
        f"CTAP Dashboard running at {url}  (reload enabled, Ctrl-C to quit)", flush=True
    )
    if open_browser:
        webbrowser.open(url)
    try:
        _BuildOnReload(config, target=server.run, sockets=[sock]).run()
    except KeyboardInterrupt:
        pass
    finally:
        # ChangeReload closes the sockets it was given; this is a no-op then.
        sock.close()
