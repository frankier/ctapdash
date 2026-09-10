from natsort import natsorted

from dataclasses import dataclass
from pathlib import Path
import re

SCALP_REGEX = re.compile(r"(?P<stem>[^-]+)-badChan-scalp.png")
CH_REGEX = re.compile(r"(?P<stem>.+)-chs(?P<ch_start>[0-9]+)-(?P<ch_end>[0-9]+).png")


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


class ObservationData:
    def __init__(self, source_path, participant=None, source=None):
        self.source_path = DatasetPaths(source_path).root
        self.participant = participant
        self.source = source

    @classmethod
    def from_source(cls, source, participant=None):
        from ctapdash.config import SETTINGS
        from pathlib import Path

        source_path = Path(SETTINGS.sources[source])
        return cls(source_path, participant, source)

    @classmethod
    def from_request(cls, request, default_participant=None):
        source = request.query_params.get("source")
        if source is None:
            raise ValueError("Source must be specified in the request.")
        participant = request.query_params.get("participant")
        if participant is None:
            participant = default_participant
        return cls.from_source(source, participant)

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

    def get_steps(self):
        self.require_participant()
        return get_steps_for_participant(self.source_path, self.participant)

    def get_recording(self, step):
        from ctapdash.io.recording import RecordingData

        self.require_participant()
        steps = dict(self.get_steps())
        try:
            directory = steps[int(step)]
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f"Unknown processing step: {step}") from error
        return RecordingData(DatasetPaths(self.source_path).recording(
            directory / (self.participant + ".set")
        ))

    def get_all_steps(self):
        return collect_steps(self.source_path)


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
