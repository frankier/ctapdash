import re
from importlib.resources import files
from pathlib import Path

from mne import BaseEpochs
from mne.io import BaseRaw

from starlette_htmx.middleware import HtmxMiddleware
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from ctapdash.config import SETTINGS
from ctapdash.io import read_eeglab, ObservationData
from ctapdash.stats import describe_mne
from ctapdash.plotting.mne import set_onionskin_eeg, OnionskinMNEBrowseFigure
from ctapdash.middleware import GlobalRequestMiddleware
from mplbed import mplbed_starlette, safe_html



# importlib.resources works both from a normal install and from inside a
# PyInstaller onedir bundle, where the frozen loader reports a real directory.
_PKG = files("ctapdash")
TEMPLATES_DIR = str(_PKG / "templates")
STATIC_DIR = str(_PKG / "static")


def sources_context(request):
    ctx = {
        "sources": SETTINGS.sources,
    }
    source = request.query_params.get("source", "")
    ctx["source"] = source
    if source:
        ctx.update(ObservationData.from_request(request).get_all_steps())
    participant = request.query_params.get("participant")
    if participant is not None:
        ctx["participant"] = participant
    if source and participant:
        ctx["source_participant_qs"] = f"?source={source}&participant={participant}"
    return ctx


templates = Jinja2Templates(directory=TEMPLATES_DIR, context_processors=[sources_context])


def _observation_count(instance):
    if isinstance(instance, BaseEpochs):
        return len(instance) * len(instance.times)
    if isinstance(instance, BaseRaw):
        return instance.n_times
    raise TypeError(f"Expected MNE Raw or Epochs, got {type(instance).__name__}")


def _participant_step_rows(root_path, steps, participant):
    rows = []
    for step_num, step_path in steps:
        instance = read_eeglab(step_path / (participant + ".set"))
        rows.append(
            {
                "number": step_num,
                "directory": str(step_path.relative_to(root_path)),
                "observations": _observation_count(instance),
            }
        )
    return rows


async def index(request):
    return templates.TemplateResponse(
        request,
        'index.html',
        context={
            "sources": SETTINGS.sources,
        }
    )


def participant_context(request):
    dataset = ObservationData.from_request(request)
    steps = dataset.get_steps()
    return {
        "source": dataset.source,
        "participant": dataset.participant,
        "steps": steps,
        "view": "steps",
    }


async def participant_steps_fragment(request):
    context = participant_context(request)
    yaxis = request.query_params.get("yaxis", "overdraw")
    context = {
        **context,
        "yaxis": yaxis,
        "yaxis_options": []
    }
    participant = context["participant"]
    steps = context["steps"]
    step = request.query_params.get("step", "")
    if step:
        steps_dict = dict(steps)
        try:
            step = int(step)
        except ValueError:
            raise HTTPException(status_code=404, detail="Step must be integer")
        if step not in steps_dict:
            raise HTTPException(status_code=404, detail="Step not found")
        has_prev = (step - 1) in steps_dict
        context["has_prev"] = has_prev
        onionskin = has_prev and request.query_params.get("onionskin") == "onionskin"
        context["onionskin"] = onionskin
        step_full = steps_dict[step]
        path = step_full / (participant + ".set")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        eeg = read_eeglab(path)
        if onionskin:
            path = steps_dict[step - 1] / (participant + ".set")
            if not path.exists():
                raise HTTPException(status_code=404, detail="Path not found")
            prev_eeg = read_eeglab(path)
            set_onionskin_eeg(prev_eeg)
        if yaxis == "normalize":
            scalings = "auto"
        else:
            scalings = None
        context["yaxis_options"].extend(["overdraw", "normalize"])
        if isinstance(eeg, BaseEpochs):
            fig = eeg.plot(show=False, scalings=scalings, figure_class=OnionskinMNEBrowseFigure)
        else:
            context["yaxis_options"].append("clamp")
            if yaxis == "clip":
                clipping = "clamp"
            else:
                clipping = None
            fig = eeg.plot(show=False, scalings=scalings, clipping=clipping, figure_class=OnionskinMNEBrowseFigure)
        context["eeg_fig"] = safe_html.figure_html(fig, on_close="msg_discrete", prevent_default_navigation=True)
        context["current_step"] = step
    return templates.TemplateResponse(
        request,
        'participant_steps.html',
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


async def participant_peeks_fragment(request):
    from ctapdash.io import qc_to_tree

    dataset = ObservationData.from_request(request)
    qcs = dataset.get_qc()
    tree = qc_to_tree(qcs)
    peek_param = request.query_params.get("peek")
    set_param = request.query_params.get("set")
    if set_param is None:
        peek_tree = tree.get(peek_param)
        if peek_tree is not None and len(peek_tree) > 0:
            set_param = list(peek_tree.keys())[0]
    bit_param = request.query_params.get("bit")
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
            context.update({
                "qc_type": "eeg",
                "eeg": map_encode_qc(dataset.source_path, groups[bit_param]),
            })
        elif bit_param in rest:
            path = rest[bit_param]
            context.update({
                "qc_type": "image",
                "path": str(path),
                "encoded_string": encode_qc(source_path, path),
            })
        elif "set" in request.query_params and "bit" in request.query_params:
            raise HTTPException(status_code=404)
    return templates.TemplateResponse(
        request,
        'participant_peeks.html',
        context=context
    )


async def participant_overview_fragment(request):
    dataset = ObservationData.from_request(request)
    logs = dataset.get_logs()
    qc = dataset.get_qc()
    steps = dataset.get_steps()
    step_rows = await run_in_threadpool(
        _participant_step_rows, dataset.source_path, steps, dataset.participant
    )
    return templates.TemplateResponse(
        request,
        'participant_overview.html',
        context={
            "view": "overview",
            "logs": logs,
            "qc": qc,
            "steps": steps,
            "step_rows": step_rows,
        }
    )


async def participant_statistics_fragment(request):
    from ctapdash.plotting.stats_heatmap import participant_descriptive_heatmap
    dataset = ObservationData.from_request(request)
    steps = dataset.get_steps()
    selected_step = request.query_params.get("step", "all")
    selected_steps = steps
    if selected_step != "all":
        try:
            step_num = int(selected_step)
        except ValueError:
            raise HTTPException(status_code=404, detail="Step must be integer or all")
        steps_by_number = dict(steps)
        if step_num not in steps_by_number:
            raise HTTPException(status_code=404, detail="Step not found")
        selected_steps = [(step_num, steps_by_number[step_num])]

    descriptive_heatmap = None
    if selected_steps:
        descriptive_heatmap = await run_in_threadpool(
            participant_descriptive_heatmap, selected_steps, dataset.participant
        )
    return templates.TemplateResponse(
        request,
        "participant_statistics.html",
        context={
            "descriptive_heatmap": descriptive_heatmap,
            "show_step": selected_step == "all",
        },
    )


async def participant_log(request):
    dataset = ObservationData.from_request(request)
    logs = dataset.get_logs()
    ctx = {
        "view": "logs",
        "logs": logs,
    }
    if "log" in request.query_params:
        log = request.query_params["log"]
        log_path = dataset.source_path / log
        with open(log_path) as f:
            content = f.read()
        ctx.update({
            "current_log_file": log,
            "content": content,
        })
    return templates.TemplateResponse(
        request,
        'participant_log.html',
        context=ctx
    )


def create_app(debug=False):
    from ctapdash.setup_ui import RequireConfigMiddleware, setup_routes

    app = Starlette(
        debug=debug,
        routes=[
            Route('/', index, name="index"),
            Mount('/static', app=StaticFiles(directory=STATIC_DIR), name="static"),
            Route('/participant/overview', participant_overview_fragment, name="participant_overview"),
            Route('/participant/statistics', participant_statistics_fragment, name="participant_statistics"),
            Route('/participant/steps', participant_steps_fragment, name="participant_steps"),
            Route('/participant/peeks', participant_peeks_fragment, name="participant_peeks"),
            Route('/participant/log', participant_log, name="participant_log"),
            *setup_routes(),
        ],
        middleware=[
            Middleware(RequireConfigMiddleware),
            Middleware(HtmxMiddleware),
            Middleware(GlobalRequestMiddleware),
        ],
    )
    # Installs MplbedMiddleware (which does its own /webagg routing), registers
    # the mplbed_head context processor, and selects the webaggext backend.
    mplbed_starlette.setup(app, templates=templates, prefix="/webagg")
    return app
