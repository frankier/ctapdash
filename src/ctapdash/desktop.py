"""Declarative identity for the shared desktop runtime.

All launching, window, socket, fallback, and reload mechanics live in
``workingtitle.desktop``. This module only says what the dashboard is called
and which application factory serves it. ``SESSION`` is the launch-scoped
window handle; the application receives it through ``create_app``.
"""

from workingtitle.desktop import AppSpec, DesktopSession, WindowSpec

APP = AppSpec(
    name="ctapdash",
    title="CTAP Dashboard",
    description="A dashboard for viewing the outputs of CTAP pipelines.",
    factory="ctapdash.asgi:create_app_from_env",
    debug_env_var="CTAPDASH_DEBUG",
    window=WindowSpec(width=1400, height=900),
    # Linux has no frozen native-window backend: pywebview needs
    # PyGObject/WebKitGTK there, so the shared runtime opens the browser.
    native_platforms=("win32", "darwin"),
)

SESSION = DesktopSession(APP)
