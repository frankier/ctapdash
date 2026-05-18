import re
from os import environ
from pathlib import Path
from mne.io import read_raw_eeglab, read_epochs_eeglab
from mne import BaseEpochs
import tomlkit
from natsort import natsorted

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.responses import HTMLResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from ctapdash.plots import set_onionskin_eeg, perform_monkeypatch
from starlette_webagg import get_head_content, get_app as get_webagg_app, figure_html
from starlette_webagg.utils import composed_lifespan


SCALP_REGEX = re.compile("(?P<stem>[^-]+)-badChan-scalp.png")
CH_REGEX = re.compile("(?P<stem>.+)-chs(?P<ch_start>[0-9]+)-(?P<ch_end>[0-9]+).png")


perform_monkeypatch()

with open(environ["CTAPDASH_SETTINGS"]) as f:
    SETTINGS = tomlkit.parse(f.read())

def load_sources():
    sources_settings = SETTINGS["sources"]
    return dict(sources_settings.items())


SOURCES = load_sources()


def webagg_context(request):
    return {
        'head_webagg': get_head_content(request, core=True),
    }


templates = Jinja2Templates(directory="templates", context_processors=[webagg_context])


def read_eeglab(path):
    import warnings
    with warnings.catch_warnings(action="ignore"):
        try:
            return read_epochs_eeglab(path)
        except ValueError:
            return read_raw_eeglab(path, preload=True)


def get_steps_for_participant(root_path, participant):
    steps = []
    for subdir in root_path.iterdir():
        if not subdir.name[0].isnumeric():
            continue
        path = subdir / (participant + ".set")
        if not path.exists():
            continue
        step_num = int(subdir.name.split("_", 1)[0])
        steps.append((step_num, subdir))
    steps = natsorted(steps)
    return steps


def read_eegs(root_path, participant, steps):
    eegs = []
    step_num_to_eeg_idx = {}
    for eeg_idx, (step_num, step) in enumerate(steps):
        path = root_path / step / (participant + ".set")
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
            rel_path = file_path.relative_to(qc_dir)
            qc.append(rel_path)
    qc = natsorted(qc)
    return qc


def qc_to_tree(qcs):
    tree = {}
    for qc in qcs:
        tree.setdefault(qc.parts[0], []).append(qc)

    def group_channels(values):
        groups = {}
        rest = {}
        for value, path in values:
            match = SCALP_REGEX.match(value)
            if match:
                stem = match.group("stem")
                groups.setdefault(stem, {})["scalp"] = (value, path)
                continue
            match = CH_REGEX.match(value)
            if match:
                stem = match.group("stem")
                ch_start = int(match.group("ch_start"))
                ch_end = int(match.group("ch_end"))
                groups.setdefault(stem, {}).setdefault("chs", []).append((ch_start, ch_end, value, path))
                groups[stem]["chs"].sort()
                continue
            rest[value] = path
        return groups, rest

    def form_sets(peek_list):
        peek_dict = {}
        rest = {}
        for directory in peek_list:
            first_seg = directory.parts[1]
            if first_seg.startswith("set"):
                peek_dict.setdefault(directory.parts[1], []).append((directory.parts[-1], directory))
            else:
                sub_peek_dict, rest_dict = form_sets(directory)
                peek_dict.update(sub_peek_dict)
                rest.update(rest_dict)
        return peek_dict, rest

    new_tree = {}
    for root, peek_list in tree.items():
        peek_dict, rest = form_sets(peek_list)
        
        assert len(rest) == 0
        peek_dict = {k: group_channels(v) for k, v in peek_dict.items()}
        new_tree[root] = peek_dict

    return new_tree


async def participant_steps_fragment(request):
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    yaxis = request.query_params.get("yaxis", "overdraw")
    source_path = Path(SOURCES[source])
    steps = get_steps_for_participant(source_path, participant)
    context = {
        "steps": steps,
        "view": "steps",
        "yaxis": yaxis,
        "yaxis_options": []
    }
    if "step" in request.query_params:
        steps_dict = dict(steps)
        step = request.query_params["step"]
        step = int(step)
        if step not in steps_dict:
            raise HTTPException(status_code=404, detail="Step not found")
        step_full = steps_dict[step]
        path = step_full / (participant + ".set")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        eeg = read_eeglab(path)
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
        context["eeg_fig"] = figure_html(request.app, fig)
        context["current_step"] = step
    return templates.TemplateResponse(
        request,
        'participant_steps.html',
        context=context,
    )


def encode_qc(source_path, path):
    from wand.image import Image
    import base64

    filename = source_path / "quality_control" / path
    with Image(filename=filename) as img:
        img.trim()
        return base64.b64encode(img.make_blob("png")).decode("utf-8")


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
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    source_path = Path(SOURCES[source])
    qcs = collect_qc(source_path, participant)
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
                "eeg": map_encode_qc(source_path, groups[bit_param]),
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
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    source_path = Path(SOURCES[source])
    logs = collect_logs(source_path, participant)
    qc = collect_qc(source_path, participant)
    steps = get_steps_for_participant(source_path, participant)
    return templates.TemplateResponse(
        request,
        'participant_overview.html',
        context={
            "view": "overview",
            "logs": logs,
            "qc": qc,
            "steps": steps,
        }
    )


async def participant_log(request):
    source = request.query_params["source"]
    participant = request.query_params["participant"]
    source_path = Path(SOURCES[source])
    logs = collect_logs(source_path, participant)
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
    Mount('/static', app=StaticFiles(directory='static'), name="static"),
    Route('/fragments/participant/selector', participant_selector, name="participant_selector"),
    Route('/fragments/participant/overview', participant_overview_fragment, name="participant_overview"),
    Route('/fragments/participant/steps', participant_steps_fragment, name="participant_steps"),
    Route('/fragments/participant/peeks', participant_peeks_fragment, name="participant_peeks"),
    Route('/fragments/participant/log', participant_log, name="participant_log"),
    Mount("/webagg", app=get_webagg_app(), name="webagg"),
], lifespan=composed_lifespan())
