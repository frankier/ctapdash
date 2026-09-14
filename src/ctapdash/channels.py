"""Participant channel metadata from raw EEGLAB montages."""

import json
import logging

from .head_geometry import _head_geometry

logger = logging.getLogger(__name__)


def channel_metadata(recordings, *, bads_by_step=None, groups=None):
    """Build a stable union from ``(step, Ctap*EEGLAB)`` pairs.

    Supplied bad lists override metadata only for the specified steps. Unknown
    names are ignored. Custom named groups can overlap, like type/region groups.
    """
    channels = {}
    geometries = []
    steps = {}
    bads = {}
    overrides = {str(k): v for k, v in (bads_by_step or {}).items()}
    for step, recording in sorted(recordings, key=lambda item: int(item[0])):
        key = str(step)
        steps[key] = list(recording.ch_names)
        bads[key] = list(dict.fromkeys(
            name for name in overrides.get(key, recording.info["bads"])
            if name in recording.ch_names
        ))
        for name, label in zip(
            recording.ch_names, recording.raw_ch_types, strict=True
        ):
            channels.setdefault(name, {"name": name, "type": label})
        if recording.raw_montage is not None:
            geometries.append(recording.raw_montage)
    head = {"points": {}, "outlines": [], "message": "No usable sensor positions are available."}
    regions = {}
    for montage in geometries:
        try:
            projected, partition = _head_geometry(montage)
            if not projected["points"]:
                continue
            if not head["points"]:
                head = projected
            else:
                for view in (head, head["front"]):
                    source = projected if view is head else projected["front"]
                    for name, point in source["points"].items():
                        view["points"].setdefault(name, point)
            assigned = {name for names in regions.values() for name in names}
            for label, names in partition.items():
                regions.setdefault(label, []).extend(name for name in names if name not in assigned)
        except Exception:
            logger.warning("Unable to project channel positions", exc_info=True)
            if not head["points"]:
                head["message"] = "Sensor positions could not be projected. Use the channel list."
    # Montage-only references are not selectable recording channels.
    for view in (head, head.get("front", {})):
        if "points" in view:
            view["points"] = {name: point for name, point in view["points"].items() if name in channels}
    regions = {label: [name for name in names if name in channels] for label, names in regions.items()}
    return {
        "channels": list(channels.values()), "head": head, "regions": regions,
        "groups": {label: [name for name in names if name in channels]
                   for label, names in (groups or {}).items()},
        "steps": steps, "bads": bads,
    }


def participant_channel_metadata(dataset, *, bads_by_step=None, groups=None):
    result = channel_metadata(
        ((step, dataset.get_recording(step).read_metadata()) for step, _ in dataset.get_steps()),
        bads_by_step=bads_by_step, groups=groups,
    )
    result["stateKey"] = json.dumps([str(dataset.source_path), dataset.participant])
    return result
