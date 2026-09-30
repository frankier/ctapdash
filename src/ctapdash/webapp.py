from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.resources import files
import importlib.util
from pathlib import Path
from ctapdash.io.cache import CacheWarmer
from ctapdash.io.paths import ObservationData, DatasetPaths
from ctapdash.io.recording import RecordingData

from mne import BaseEpochs
from mne.io import BaseRaw

from bokeh.server.asgi import BokehASGI
from starlette_htmx.middleware import HtmxMiddleware
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.routing import Route, Mount, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from ctapdash.config import SETTINGS
from ctapdash.io.eeglab import read_eeglab
from ctapdash.io.stats_cache import DATASET_STATS
from ctapdash.middleware import GlobalRequestMiddleware
from mplbed import mplbed_starlette, safe_html
from pydantic import BaseModel
from workingtitle.pydanticstarlette import (
    int_or,
    json_list_of,
    query_params,
    require_found,
)

from starlette.websockets import WebSocketDisconnect
import anyio


# importlib.resources works both from a normal install and from inside a
# PyInstaller onedir bundle, where the frozen loader reports a real directory.
_PKG = files("ctapdash")
TEMPLATES_DIR = str(_PKG / "templates")
STATIC_DIR = str(_PKG / "static")

# This is a guard against a Panel bug with BokehASGI if we accidentally end up with Panel installed.
# Can probably be removed at some point.
if importlib.util.find_spec("panel") is not None:
    import panel.io.resources as panel_resources

    panel_resources.RESOURCE_MODE = "cdn"


def sources_context(request):
    ctx = {
        "sources": SETTINGS.sources,
        "managed_config": SETTINGS.managed,
    }
    source = request.query_params.get("source", "")
    ctx["source"] = source
    if source:
        ctx.update(ObservationData.from_request(request).get_all_steps())
    participant = request.query_params.get("participant")
    if participant is not None:
        ctx["participant"] = participant
    if source:
        ctx["source_qs"] = f"?source={source}"
    if source and participant:
        ctx["source_participant_qs"] = f"?source={source}&participant={participant}"
    return ctx


safe_templates = Jinja2Templates(directory=TEMPLATES_DIR)

templates = Jinja2Templates(
    directory=TEMPLATES_DIR, context_processors=[sources_context]
)


class PeeksParams(BaseModel):
    """Peek/set/bit selection; None means "not present in the query"."""

    peek: str | None = None
    set: str | None = None
    bit: str | None = None


class StepsFragmentParams(BaseModel):
    """The steps fragment: a step number, or "" when none was given."""

    step: int_or("") = ""
    yaxis: str = "overdraw"


class StatisticsParams(BaseModel):
    """The statistics fragment: a step number or "all", plus a channel filter."""

    step: int_or("all") = "all"
    channels: json_list_of(str, message="must be a JSON list of names") | None = None


class LogParams(BaseModel):
    """The log viewer: which log file to show (checked against the dataset)."""

    log: str | None = None


def _observation_count(instance):
    if isinstance(instance, BaseEpochs):
        return len(instance) * len(instance.times)
    if isinstance(instance, BaseRaw):
        return instance.n_times
    raise TypeError(f"Expected MNE Raw or Epochs, got {type(instance).__name__}")


def _participant_step_rows(root_path, steps, participant):
    rows = []
    for step_num, step_path in steps:
        path = step_path / (participant + ".set")
        instance = RecordingData(
            DatasetPaths(root_path).recording(path)
        ).read_metadata()
        rows.append(
            {
                "number": step_num,
                "directory": str(step_path.relative_to(root_path)),
                "observations": _observation_count(instance),
            }
        )
    return rows


@query_params()
async def index(request):
    return templates.TemplateResponse(
        request,
        "index.html",
        context={
            "sources": SETTINGS.sources,
        },
    )


def participant_context(request, default_participant=None):
    dataset = ObservationData.from_request(
        request, default_participant=default_participant
    )
    result = {
        "source": dataset.source,
    }
    if dataset.participant is not None:
        steps = dataset.get_steps()
        result.update(
            {
                "participant": dataset.participant,
                "steps": steps,
            }
        )
    return result


@query_params()
async def dataset_overview(request):
    context = participant_context(request)
    context["view"] = "dataset-overview"
    dataset = ObservationData.from_request(request)
    participant_steps = {}
    participants = dataset.get_all_steps()["participants"]
    for participant, _ in participants:
        participant_dataset = dataset.with_participant(participant)
        steps = participant_dataset.get_steps()
        await _wait_metadata(request, participant_dataset, steps)
        step_rows = await run_in_threadpool(
            _participant_step_rows, participant_dataset.source_path, steps, participant
        )
        participant_steps[participant] = step_rows
    context["participant_steps"] = participant_steps
    context["participant_totals"] = {
        participant: sum(row["observations"] for row in rows)
        for participant, rows in participant_steps.items()
    }
    return templates.TemplateResponse(
        request,
        "dataset_overview.html",
        context=context,
    )


@query_params()
async def participant_select(request):
    context = participant_context(request)
    context["view"] = "participant-select"
    return templates.TemplateResponse(
        request,
        "participant_select.html",
        context=context,
    )


@query_params(StepsFragmentParams)
async def participant_steps_fragment(request, params):
    context = participant_context(request)
    context["view"] = "steps"
    yaxis = params.yaxis
    context = {**context, "yaxis": yaxis, "yaxis_options": []}
    participant = context["participant"]
    steps = context["steps"]
    step = params.step
    if step != "":
        steps_dict = dict(steps)
        if step not in steps_dict:
            raise HTTPException(status_code=404, detail="Step not found")
        has_prev = (step - 1) in steps_dict
        context["has_prev"] = has_prev
        step_full = steps_dict[step]
        path = step_full / (participant + ".set")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        eeg = read_eeglab(path, mmap=False, use_cache=False)
        if yaxis == "normalize":
            scalings = "auto"
        else:
            scalings = None
        context["yaxis_options"].extend(["overdraw", "normalize"])
        if isinstance(eeg, BaseEpochs):
            fig = eeg.plot(show=False, scalings=scalings)
        else:
            context["yaxis_options"].append("clamp")
            if yaxis == "clip":
                clipping = "clamp"
            else:
                clipping = None
            fig = eeg.plot(show=False, scalings=scalings, clipping=clipping)
        context["eeg_fig"] = safe_html.figure_html(
            fig, on_close="msg_discrete", prevent_default_navigation=True
        )
        context["current_step"] = step
    return templates.TemplateResponse(
        request,
        "participant_steps.html",
        context=context,
    )


def bokeh_document(request, path, *args, **kwargs):
    from bokeh.embed import server_document
    from markupsafe import Markup

    url = str(request.url_for("bokeh", path=path))
    return Markup(server_document(url, relative_urls=True, *args, **kwargs))


@query_params()
async def venn_time_series(request):
    """Serve the multichannel processing-step comparison viewer."""
    # If there's not participant, just pick the first one
    default_participant = None
    context = participant_context(request, default_participant=default_participant)
    dataset = ObservationData.from_request(
        request, default_participant=default_participant
    )
    await _wait_metadata(request, dataset, dataset.get_steps())
    await DATASET_STATS.get(dataset.source_path)
    context["view"] = "venn_time_series"
    context["venn_time_series"] = bokeh_document(
        request,
        "/venn-time-series",
        arguments={
            "source": context["source"],
            "participant": context["participant"],
        },
    )
    return templates.TemplateResponse(
        request,
        "venn_time_series.html",
        context=context,
    )


def trim(img):
    """Crop away the uniform border, as ImageMagick's trim() did.

    The reference colour is the top-left pixel, matching ImageMagick. Unlike
    ImageMagick there is no fuzz tolerance, which is fine for CTAP's flat-
    background QC plots.
    """
    from PIL import Image, ImageChops

    if img.mode in ("RGBA", "LA"):
        bbox = img.getchannel("A").getbbox()
        if bbox:
            return img.crop(bbox)
    rgb = img.convert("RGB")
    background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
    bbox = ImageChops.difference(rgb, background).getbbox()
    return img.crop(bbox) if bbox else img


def encode_qc(source_path, path):
    from PIL import Image
    import base64
    import io

    filename = source_path / "quality_control" / path
    with Image.open(filename) as img:
        img.load()
        out = io.BytesIO()
        trim(img).save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode("utf-8")


def map_encode_qc(source_path, val):
    if isinstance(val, Path):
        return encode_qc(source_path, val)
    elif isinstance(val, list):
        return [map_encode_qc(source_path, v) for v in val]
    elif isinstance(val, tuple):
        return tuple(map_encode_qc(source_path, v) for v in val)
    elif isinstance(val, dict):
        return {k: map_encode_qc(source_path, v) for k, v in val.items()}
    else:
        return val


@query_params(PeeksParams)
async def participant_peeks_fragment(request, params):
    from ctapdash.io.paths import qc_to_tree

    dataset = ObservationData.from_request(request)
    qcs = dataset.get_qc()
    tree = qc_to_tree(qcs)
    peek_param = params.peek
    set_param = params.set
    if set_param is None:
        peek_tree = tree.get(peek_param)
        if peek_tree is not None and len(peek_tree) > 0:
            set_param = list(peek_tree.keys())[0]
    bit_param = params.bit
    if bit_param is None:
        groupsrest = tree.get(peek_param, {}).get(set_param)
        if groupsrest is not None:
            groups, rest = groupsrest
            if len(groups) > 0:
                bit_param = list(groups.keys())[0]
            if len(rest) > 0:
                bit_param = list(rest.keys())[0]
    qcs = [str(qc) for qc in qcs]
    context = {
        "view": "peeks",
        "qcs": qcs,
        "tree": tree,
        "peek_param": peek_param,
        "set_param": set_param,
        "bit_param": bit_param,
    }
    if peek_param:
        groups, rest = tree.get(peek_param, {}).get(set_param, ({}, {}))
        if bit_param in groups:
            context.update(
                {
                    "qc_type": "eeg",
                    "eeg": map_encode_qc(dataset.source_path, groups[bit_param]),
                }
            )
        elif bit_param in rest:
            path = rest[bit_param]
            context.update(
                {
                    "qc_type": "image",
                    "path": str(path),
                    "encoded_string": encode_qc(dataset.source_path, path),
                }
            )
        elif params.set is not None and params.bit is not None:
            raise HTTPException(status_code=404)
    return templates.TemplateResponse(
        request, "participant_peeks.html", context=context
    )


async def _wait_metadata(request, dataset, steps):
    for _number, directory in steps:
        await request.app.state.cache_hurrier.hurry(
            "metadata",
            (dataset.source_path, directory / (dataset.participant + ".set")),
        )


async def cache_status(websocket):
    await websocket.accept()
    template = templates.env.get_template("_cache_progress.html")
    warmer = websocket.app.state.cache_hurrier
    async with anyio.create_task_group() as group:

        async def send_progress():
            try:
                async for status in warmer.statuses():
                    await websocket.send_text(template.render(status=status))
            except WebSocketDisconnect, OSError:
                pass
            finally:
                group.cancel_scope.cancel()

        group.start_soon(send_progress)
        try:
            # Detect disconnects even when there are no new progress events.
            async for _message in websocket.iter_text():
                pass
        finally:
            group.cancel_scope.cancel()


@query_params()
async def participant_overview_fragment(request):
    dataset = ObservationData.from_request(request)
    logs = dataset.get_logs()
    qc = dataset.get_qc()
    steps = dataset.get_steps()
    await _wait_metadata(request, dataset, steps)
    from ctapdash.channels import participant_channel_metadata

    channel_metadata = await run_in_threadpool(participant_channel_metadata, dataset)
    step_rows = await run_in_threadpool(
        _participant_step_rows, dataset.source_path, steps, dataset.participant
    )
    return templates.TemplateResponse(
        request,
        "participant_overview.html",
        context={
            "view": "overview",
            "logs": logs,
            "qc": qc,
            "steps": steps,
            "step_rows": step_rows,
            "channel_metadata": channel_metadata,
        },
    )


@query_params(StatisticsParams)
async def participant_statistics_fragment(request, params):
    from ctapdash.plotting.stats_heatmap import participant_descriptive_heatmap

    dataset = ObservationData.from_request(request)
    steps = dataset.get_steps()
    selected_steps = steps
    if params.step != "all":
        steps_by_number = dict(steps)
        if params.step not in steps_by_number:
            raise HTTPException(status_code=404, detail="Step not found")
        selected_steps = [(params.step, steps_by_number[params.step])]
    stats = await DATASET_STATS.get(dataset.source_path)
    channels = params.channels
    descriptive_heatmap = None
    if selected_steps:
        descriptive_heatmap = await run_in_threadpool(
            participant_descriptive_heatmap,
            selected_steps,
            dataset.participant,
            stats,
            channels,
        )
    return templates.TemplateResponse(
        request,
        "participant_statistics.html",
        context={
            "descriptive_heatmap": descriptive_heatmap,
            "show_step": params.step == "all",
        },
    )


@query_params(LogParams)
async def participant_log(request, params):
    dataset = ObservationData.from_request(request)
    logs = dataset.get_logs()
    ctx = {
        "view": "logs",
        "logs": logs,
    }
    if params.log is not None:
        log = require_found(params.log, logs, "Log")
        log_path = dataset.source_path / log
        with open(log_path) as f:
            content = f.read()
        ctx.update(
            {
                "current_log_file": log,
                "content": content,
            }
        )
    return templates.TemplateResponse(request, "participant_log.html", context=ctx)


async def bad_request(request: Request, exc: HTTPException):
    return safe_templates.TemplateResponse(request, "400.html", status_code=400)


async def not_found(request: Request, exc: HTTPException):
    return safe_templates.TemplateResponse(request, "404.html", status_code=404)


async def server_error(request: Request, exc: Exception):
    import traceback
    from textwrap import dedent
    from urllib.parse import urlencode

    show_stacktrace = SETTINGS.managed
    if show_stacktrace:
        stacktrace = "".join(
            traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__,
            )
        )
        report_url = "https://github.com/frankier/ctapdash/issues/new?"
        report_url += urlencode(
            (
                ("title", "Runtime error with stacktrace: " + type(exc).__name__),
                ("assignee", "frankier"),
                (
                    "body",
                    dedent("""\
                ## Steps to reproduce

                What were you doing when the error occurred?

                ## Stacktrace
                ```
                PASTE THE STACKTRACE HERE
                ```
                """),
                ),
                ("labels[]", ("bug", "errorpage")),
            )
        )
    else:
        stacktrace = ""
        report_url = ""
    return safe_templates.TemplateResponse(
        request,
        "500.html",
        context={
            "stacktrace": stacktrace,
            "report_url": report_url,
        },
        status_code=500,
    )


exception_handlers = {400: bad_request, 404: not_found, 500: server_error}


def create_app(debug=False, session=None):
    from ctapdash.setup_ui import RequireConfigMiddleware, setup_routes
    from venn_ts import venn_time_series_bokeh

    bokeh_application = BokehASGI(
        {
            "/venn-time-series": venn_time_series_bokeh,
        }
    )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
        async with CacheWarmer() as hurrier:
            for path in SETTINGS.sources.values():
                hurrier.add_dataset(path)
            app.state.cache_hurrier = hurrier
            # Mounted Starlette applications don't receive lifespan events. Start and
            # stop Bokeh from the parent application's lifespan instead.
            await bokeh_application.core.start()
            try:
                yield
            finally:
                await bokeh_application.core.stop()

    app = Starlette(
        debug=debug,
        routes=[
            Route("/", index, name="index"),
            WebSocketRoute("/cache-status", cache_status, name="cache_status"),
            Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static"),
            Route("/overview", dataset_overview, name="dataset_overview"),
            Route("/participant", participant_select, name="participant_select"),
            Route(
                "/participant/overview",
                participant_overview_fragment,
                name="participant_overview",
            ),
            Route(
                "/participant/statistics",
                participant_statistics_fragment,
                name="participant_statistics",
            ),
            Route(
                "/participant/steps",
                participant_steps_fragment,
                name="participant_steps",
            ),
            Route(
                "/participant/venn-time-series",
                venn_time_series,
                name="venn_time_series",
            ),
            Route(
                "/participant/peeks",
                participant_peeks_fragment,
                name="participant_peeks",
            ),
            Route("/participant/log", participant_log, name="participant_log"),
            *setup_routes(),
            Mount("/bokeh", bokeh_application, name="bokeh"),
        ],
        middleware=[
            Middleware(RequireConfigMiddleware),
            Middleware(HtmxMiddleware),
            Middleware(GlobalRequestMiddleware),
        ],
        exception_handlers=exception_handlers,
        lifespan=lifespan,
    )
    # Installs MplbedMiddleware (which does its own /webagg routing), registers
    # the mplbed_head context processor, and selects the webaggext backend.
    mplbed_starlette.setup(app, templates=templates, prefix="/webagg")
    # The setup UI reaches native file dialogs through this launch-scoped
    # object; it is None in browser/server mode and in tests.
    app.state.desktop = session
    return app
