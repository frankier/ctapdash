from os import environ
from pathlib import Path
from mne.io import read_raw_eeglab, read_epochs_eeglab
import tomlkit

from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route, Mount
from starlette.templating import Jinja2Templates
from ctapdash.plots import set_onionskin_eeg, perform_monkeypatch
from starlette_webagg import get_head_content, get_app as get_webagg_app, figure_html
from starlette_webagg.utils import composed_lifespan


perform_monkeypatch()

with open(environ["CTAPDASH_SETTINGS"]) as f:
    SETTINGS = tomlkit.parse(f.read())

def load_sources():
    sources_settings = SETTINGS["sources"]
    return dict(sources_settings.items())


SOURCES = load_sources()
print("Sources:", SOURCES)


templates = Jinja2Templates(directory="templates")


def read_eeglab(path):
    import warnings
    with warnings.catch_warnings(action="ignore"):
        try:
            return read_epochs_eeglab(path)
        except ValueError:
            return read_raw_eeglab(path, preload=True)


def read_eegs(root_path, participant, steps):
    eegs = []
    step_num_to_eeg_idx = {}
    for eeg_idx, (step_num, step) in enumerate(steps):
        path = root_path / step / participant
        if not path.exists():
            print(f"Not found: {path}")
            continue
        eeg = read_eeglab(path)
        eegs.append(eeg)
        step_num_to_eeg_idx[step_num] = eeg_idx
    return eegs, step_num_to_eeg_idx


def collect_steps(root_path):
    from collections import Counter

    steps = []
    files = Counter()

    for subdir in root_path.iterdir():
        if subdir.name[0].isnumeric():
            step_num = int(subdir.name.split("_", 1)[0])
            steps.append((step_num, subdir))
            for filename in subdir.iterdir():
                if not filename.suffix == ".set":
                    continue
                files[filename.stem] += 1

    steps.sort()
    files = [(-count, filename) for filename, count in files.items()]
    files.sort()

    return {
        "participants": [(filename, -count) for count, filename in files],
        "steps": steps,
    }


async def index(request):
    return templates.TemplateResponse(
        request,
        'index.html',
        context={
            'head_content': get_head_content(request, core=True),
            "sources": SOURCES,
            "display_selector": False,
        }
    )


async def participant_selector(request):
    source = request.query_params["source"]
    context = {}
    if source == "":
        context["display_selector"] = False
    else:
        context["display_selector"] = True
        source_path = Path(SOURCES[source])
        context.update(collect_steps(source_path))
    return templates.TemplateResponse(
        request,
        'participant_select_fragment.html',
        context=context
    )


def collect_logs(source_path, participant):
    logs = []
    log_dir = source_path / "logs"
    for root, dirs, files in log_dir.walk():
        for file in files:
            if not file.startswith(participant):
                continue
            file_path = root / file
            rel_path = file_path.relative_to(source_path)
            logs.append(str(rel_path))
    return logs


def collect_qc(source_path, participant):
    qc = []
    qc_dir = source_path / "quality_control"
    for root, dirs, files in qc_dir.walk():
        for file in files:
            if not file.startswith(participant):
                continue
            file_path = root / file
            rel_path = file_path.relative_to(source_path)
            qc.append(str(rel_path))
    return qc


async def participant_steps_fragment(request):
    return templates.TemplateResponse(
        request,
        'participant_steps.html',
        context={
            "view": "steps",
        }
    )


async def participant_peeks_fragment(request):
    import base64

    print("participant_peeks_fragment")
    from pprint import pprint
    pprint(request.query_params)
    print("/participant_peeks_fragment")
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    img = request.query_params["img"]
    source_path = Path(SOURCES[source])
    encoded_string = None
    with open(source_path / img, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode("utf-8")
    qcs = collect_qc(source_path, participant)
    return templates.TemplateResponse(
        request,
        'participant_peeks.html',
        context={
            "view": "peeks",
            "qc": img,
            "qcs": qcs,
            "encoded_string": encoded_string,
        }
    )


async def participant_overview_fragment(request):
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    source_path = Path(SOURCES[source])
    logs = collect_logs(source_path, participant)
    qc = collect_qc(source_path, participant)
    return templates.TemplateResponse(
        request,
        'participant_overview.html',
        context={
            "view": "overview",
            "logs": logs,
            "qc": qc,
        }
    )


async def participant_log(request):
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    source_path = Path(SOURCES[source])
    logs = collect_logs(source_path, participant)
    print("logs")
    print(logs)
    if "log" in request.query_params:
        log = request.query_params["log"]
        log_path = source_path / log
        with open(log_path) as f:
            content = f.read()
        return templates.TemplateResponse(
            request,
            'participant_log.html',
            context={
                "view": "logs",
                "logs": logs,
                "current_log_file": log,
                "content": content,
            }
        )
    else:
        return templates.TemplateResponse(
            request,
            'participant_log.html',
            context={
                "view": "logs",
                "logs": logs,
            }
        )


app = Starlette(debug=True, routes=[
    Route('/', index, name="index"),
    Route('/fragments/participant/selector', participant_selector, name="participant_selector"),
    Route('/fragments/participant/overview', participant_overview_fragment, name="participant_overview"),
    Route('/fragments/participant/steps', participant_steps_fragment, name="participant_steps"),
    Route('/fragments/participant/peeks', participant_peeks_fragment, name="participant_peeks"),
    Route('/fragments/participant/log', participant_log, name="participant_log"),
    Mount("/webagg", app=get_webagg_app(), name="webagg"),
], lifespan=composed_lifespan())
