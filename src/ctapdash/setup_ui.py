"""Managed source picker, shared by first-run setup and the dashboard dialog."""

import secrets
from pathlib import Path
from urllib.parse import parse_qsl

from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import RedirectResponse
from starlette.routing import Route

from ctapdash import config
from ctapdash.config import SETTINGS


def _session(request):
    """The launch's desktop session, or None in tests and bare ASGI runs."""
    return getattr(request.app.state, "desktop", None)


# The server is loopback-bound, but any web page the user visits can POST to
# 127.0.0.1. Mutating routes therefore require a token only our own pages know.
TOKEN = secrets.token_urlsafe(32)

_EXEMPT_PREFIXES = ("/setup", "/static", "/webagg", "/cache-status")


class RequireConfigMiddleware:
    """Send every page to /setup until at least one source is configured."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and SETTINGS.managed and not SETTINGS.configured:
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


def _render(request, *, modal=False, changed=False, **extra):
    from ctapdash.webapp import templates

    session = _session(request)
    response = templates.TemplateResponse(
        request,
        "setup.html",
        context={
            "token": TOKEN,
            "native": session is not None and session.native,
            "modal": modal,
            **extra,
        },
    )
    if changed:
        response.headers["HX-Trigger"] = "sources-changed"
    return response


def _add_source(directory, name=None, sources=None):
    """Return a new source map, deriving a unique name when needed."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    name = (name or directory.name or str(directory)).strip()
    candidate = name
    suffix = 2
    sources = (SETTINGS.sources if sources is None else sources).copy()
    while candidate in sources and sources[candidate] != str(directory):
        candidate = f"{name}-{suffix}"
        suffix += 1
    sources[candidate] = str(directory)
    return candidate, sources


def _require_managed():
    if not SETTINGS.managed:
        raise HTTPException(status_code=403, detail="Configuration is read only")


async def setup_page(request):
    if not SETTINGS.managed:
        return RedirectResponse("/")
    return _render(request, modal=request.query_params.get("modal") == "1")


async def setup_add(request):
    _require_managed()
    form = await _read_form(request)
    modal = form.get("modal") == "1"
    try:
        name, sources = _add_source(form.get("path", ""), form.get("name") or None)
        changed = sources != SETTINGS.sources
        if changed:
            config.save_managed_sources(sources)
        request.app.state.cache_hurrier.add_dataset(SETTINGS.sources[name])
    except (ValueError, OSError) as err:
        return _render(request, modal=modal, error=str(err))
    return _render(request, modal=modal, changed=changed)


async def setup_pick(request):
    """Open the platform folder picker. Only reachable with a native window."""
    # _read_form validates the CSRF token; the form carries no other fields.
    _require_managed()
    form = await _read_form(request)
    modal = form.get("modal") == "1"
    session = _session(request)
    if session is None or not session.native:
        raise HTTPException(status_code=400, detail="No native window")

    # The dialog blocks until the user dismisses it.
    chosen = await run_in_threadpool(session.open_folder, allow_multiple=True)
    sources = SETTINGS.sources.copy()
    picked_paths = []
    try:
        for directory in chosen:
            name, sources = _add_source(directory, sources=sources)
            picked_paths.append(sources[name])
        changed = sources != SETTINGS.sources
        if changed:
            config.save_managed_sources(sources)
    except (ValueError, OSError) as err:
        return _render(request, modal=modal, error=str(err))
    for path in picked_paths:
        request.app.state.cache_hurrier.add_dataset(path)
    return _render(request, modal=modal, changed=changed)


async def setup_remove(request):
    _require_managed()
    form = await _read_form(request)
    modal = form.get("modal") == "1"
    sources = SETTINGS.sources.copy()
    sources.pop(form.get("name", ""), None)
    changed = sources != SETTINGS.sources
    try:
        if changed:
            config.save_managed_sources(sources)
    except OSError as err:
        return _render(request, modal=modal, error=str(err))
    return _render(request, modal=modal, changed=changed)


def setup_routes():
    return [
        Route("/setup", setup_page, name="setup"),
        Route("/setup/add", setup_add, methods=["POST"], name="setup_add"),
        Route("/setup/pick", setup_pick, methods=["POST"], name="setup_pick"),
        Route("/setup/remove", setup_remove, methods=["POST"], name="setup_remove"),
    ]
