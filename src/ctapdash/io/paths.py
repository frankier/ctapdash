from natsort import natsorted

from dataclasses import dataclass
from pathlib import Path
import re

from pydantic import BaseModel
from workingtitle.pydanticstarlette import FileStem, ParamError, parse_query

SCALP_REGEX = re.compile(r"(?P<stem>[^-]+)-badChan-scalp.png")
CH_REGEX = re.compile(r"(?P<stem>.+)-chs(?P<ch_start>[0-9]+)-(?P<ch_end>[0-9]+).png")
QC_STEP_REGEX = re.compile(r"set([0-9]+)_fun([0-9]+)(?:-(.+))?")


def is_recording_file(path, root):
    """Ignore hidden paths and files without a plausible MATLAB header."""
    from ctapdash.io.eeglab.validation import set_appears_valid

    return (
        path.suffix == ".set"
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and path.is_file()
        and set_appears_valid(path)
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
                groups.setdefault(stem, {}).setdefault("chs", []).append(
                    (ch_start, ch_end, value, path)
                )
                groups[stem]["chs"].sort()
                continue
            rest[value] = path
        return groups, rest

    new_tree = {}
    for function, paths in tree.items():
        steps = {}
        for path in paths:
            if len(path.parts) >= 3 and path.parts[1].startswith("set"):
                steps.setdefault(path.parts[1], []).append((path.name, path))
        new_tree[function] = {
            key: group_channels(plots) for key, plots in steps.items()
        }

    return new_tree


def qc_step_options(root, steps):
    """Group QC step labels by step set, retaining the original selection keys."""
    directories = {}
    for path in root.iterdir():
        number = path.name.split("_", 1)[0]
        if number.isascii() and number.isdecimal() and path.is_dir():
            directories.setdefault(int(number), []).append(path.name)

    options = {}
    for key in steps:
        match = QC_STEP_REGEX.fullmatch(key)
        if match is None:
            options.setdefault("Other steps", []).append((key, key))
            continue
        step_set, step, suffix = match.groups()
        names = directories.get(int(step_set), [])
        group = names[0] if len(names) == 1 else f"Step set {step_set}"
        label = f"Step {step}"
        if suffix:
            label += f" · {suffix}"
        options.setdefault(group, []).append((key, label))
    return options


def get_step_sets_for_participant(root_path, participant):
    step_sets = []
    for subdir in root_path.iterdir():
        if not subdir.name[0].isnumeric():
            continue
        path = subdir / (participant + ".set")
        if not is_recording_file(path, root_path):
            continue
        step_set_number = int(subdir.name.split("_", 1)[0])
        step_sets.append((step_set_number, subdir))
    step_sets = natsorted(step_sets)
    return step_sets


def collect_step_sets(root_path):
    from collections import Counter

    step_sets = []
    files = Counter()

    for subdir in root_path.iterdir():
        if subdir.name[0].isnumeric():
            step_set_number = int(subdir.name.split("_", 1)[0])
            step_sets.append((step_set_number, subdir))
            for filename in subdir.iterdir():
                if not is_recording_file(filename, root_path):
                    continue
                files[filename.stem] += 1

    step_sets.sort()
    files = [(-count, filename) for filename, count in files.items()]
    files.sort()

    return {
        "participants": [(filename, -count) for count, filename in files],
        "step_sets": step_sets,
    }


class RequestQuery(BaseModel):
    """The source/participant pair every page is keyed on."""

    source: str | None = None
    participant: FileStem | None = None


class ObservationData:
    def __init__(self, source_path, participant=None, source=None):
        self.source_path = DatasetPaths(source_path).root
        self.participant = participant
        self.source = source

    def with_participant(self, participant):
        return ObservationData(self.source_path, participant, self.source)

    @classmethod
    def from_source(cls, source, participant=None):
        from ctapdash.config import SETTINGS
        from pathlib import Path

        if source not in SETTINGS.sources:
            raise ParamError(f"Unknown source: {source}")
        source_path = Path(SETTINGS.sources[source])
        return cls(source_path, participant, source)

    @classmethod
    def from_request(cls, request, default_participant=None):
        query = parse_query(request.query_params, RequestQuery)
        if query.source is None:
            raise ParamError("Source must be specified in the request.")
        participant = query.participant
        if participant is None:
            participant = default_participant
        return cls.from_source(query.source, participant)

    @classmethod
    def from_bokeh_doc(cls, doc):
        def get_arg(name):
            args = doc.session_context.request.arguments
            val = args.get(name)
            if not val:
                return None
            return val[-1].decode()

        source = get_arg("source")
        participant = get_arg("participant")
        return cls.from_source(source, participant)

    def require_participant(self):
        if self.participant is None:
            raise ValueError("Participant must be specified for this operation.")

    def get_logs(self):
        self.require_participant()
        return collect_logs(self.source_path, self.participant)

    def get_qc(self):
        self.require_participant()
        return collect_qc(self.source_path, self.participant)

    def get_step_sets(self, *, usable=False):
        self.require_participant()
        step_sets = get_step_sets_for_participant(self.source_path, self.participant)
        if usable:
            from ctapdash.io.cache import REGISTERED

            warmer = REGISTERED.get(str(self.source_path))
            if warmer is not None:
                step_sets = [
                    (number, directory)
                    for number, directory in step_sets
                    if warmer.problem(
                        (self.source_path, directory / (self.participant + ".set"))
                    )
                    is None
                ]
        return step_sets

    def get_recording(self, step_set):
        from ctapdash.io.recording import RecordingData

        self.require_participant()
        step_sets = dict(self.get_step_sets())
        try:
            directory = step_sets[int(step_set)]
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f"Unknown step set: {step_set}") from error
        return RecordingData(
            DatasetPaths(self.source_path).recording(
                directory / (self.participant + ".set")
            )
        )

    def get_all_step_sets(self):
        return collect_step_sets(self.source_path)


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def cache(self):
        return self.root / ".ctapdash_cache"

    @property
    def stats(self):
        return self.cache / "stats.xm"

    def recording(self, path):
        return RecordingPaths(self, path)


@dataclass(frozen=True)
class RecordingPaths:
    dataset: DatasetPaths
    relative: Path

    def __post_init__(self):
        path = Path(self.relative)
        absolute = (self.dataset.root / path).resolve()
        relative = absolute.relative_to(self.dataset.root)
        if relative.suffix != ".set":
            raise ValueError(f"Expected .set file, got {path}")
        object.__setattr__(self, "relative", relative)

    @property
    def set(self):
        return self.dataset.root / self.relative

    @property
    def metadata(self):
        return metadata_file_path(self.set)

    @property
    def transpose(self):
        return self.dataset.cache / self.relative.with_suffix(".ctapdash_order")

    @property
    def pyramid(self):
        return self.dataset.cache / self.relative.with_suffix(".pyramid")

    @property
    def rangepyramid(self):
        return self.dataset.cache / self.relative.with_suffix(".rangepyramid")


def metadata_file_path(path):
    path = Path(path)
    return path.parent / ("." + path.stem + ".pkl")


def cache_path(dataset_dir):
    return DatasetPaths(dataset_dir).cache


def transpose_file_path(dataset_dir, eeglab_file):
    return DatasetPaths(dataset_dir).recording(eeglab_file).transpose


def stats_file_path(dataset_dir):
    return DatasetPaths(dataset_dir).stats
