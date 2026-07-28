"""First-run source picker.

Sources can come from --config, from CTAPDASH_SETTINGS, or from this UI. There
is deliberately no global config location: anything added here lives in memory
for the session unless the user explicitly saves it somewhere.
"""

import secrets
from pathlib import Path
from urllib.parse import parse_qsl

from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import RedirectResponse
from starlette.routing import Route

from ctapdash import config, desktop
from ctapdash.config import SETTINGS


# The server is loopback-bound, but any web page the user visits can POST to
# 127.0.0.1. Mutating routes therefore require a token only our own pages know.
TOKEN = secrets.token_urlsafe(32)

_EXEMPT_PREFIXES = ("/setup", "/static", "/webagg")


class RequireConfigMiddleware:
    """Send every page to /setup until at least one source is configured."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not SETTINGS.configured:
            path = scope["path"]
            if not path.startswith(_EXEMPT_PREFIXES):
                await RedirectResponse("/setup")(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def _read_form(request):
    """Parse an urlencoded form body.

    Starlette's request.form() would pull in python-multipart, which these
    plain key/value forms do not need.
    """
    body = (await request.body()).decode("utf-8")
    form = dict(parse_qsl(body, keep_blank_values=True))
    if not secrets.compare_digest(form.get("token", ""), TOKEN):
        raise HTTPException(status_code=403, detail="Bad or missing token")
    return form


def _render(request, **extra):
    from ctapdash.webapp import templates

    return templates.TemplateResponse(
        request,
        "setup.html",
        context={
            "token": TOKEN,
            "native": desktop.WINDOW is not None,
            "loaded_from": SETTINGS.loaded_from,
            **extra,
        },
    )


def _add_source(directory, name=None):
    """Add a directory, deriving a unique name from it if none is given."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    name = (name or directory.name or str(directory)).strip()
    candidate = name
    suffix = 2
    while candidate in SETTINGS.sources and SETTINGS.sources[candidate] != str(directory):
        candidate = f"{name}-{suffix}"
        suffix += 1
    SETTINGS.sources[candidate] = str(directory)
    return candidate


async def setup_page(request):
    return _render(request)


async def setup_add(request):
    form = await _read_form(request)
    try:
        _add_source(form.get("path", ""), form.get("name") or None)
    except ValueError as err:
        return _render(request, error=str(err))
    return _render(request)


async def setup_pick(request):
    """Open the platform folder picker. Only reachable with a native window."""
    form = await _read_form(request)
    window = desktop.WINDOW
    if window is None:
        raise HTTPException(status_code=400, detail="No native window")
    import webview

    # create_file_dialog blocks until the user dismisses it.
    chosen = await run_in_threadpool(
        window.create_file_dialog, webview.FOLDER_DIALOG, allow_multiple=True
    )
    for directory in chosen or ():
        _add_source(directory)
    return _render(request)


async def setup_remove(request):
    form = await _read_form(request)
    SETTINGS.sources.pop(form.get("name", ""), None)
    return _render(request)


async def setup_save(request):
    form = await _read_form(request)
    target = form.get("path", "").strip()
    window = desktop.WINDOW
    if not target and window is not None:
        import webview

        chosen = await run_in_threadpool(
            window.create_file_dialog,
            webview.SAVE_DIALOG,
            save_filename="conf.toml",
        )
        if isinstance(chosen, (list, tuple)):
            chosen = chosen[0] if chosen else None
        target = chosen or ""
    if not target:
        return _render(request, error="No destination given")
    try:
        saved = config.save_to_file(Path(target).expanduser())
    except OSError as err:
        return _render(request, error=str(err))
    return _render(request, saved_to=saved)


def setup_routes():
    return [
        Route("/setup", setup_page, name="setup"),
        Route("/setup/add", setup_add, methods=["POST"], name="setup_add"),
        Route("/setup/pick", setup_pick, methods=["POST"], name="setup_pick"),
        Route("/setup/remove", setup_remove, methods=["POST"], name="setup_remove"),
        Route("/setup/save", setup_save, methods=["POST"], name="setup_save"),
    ]
