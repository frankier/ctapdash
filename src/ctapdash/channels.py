"""Participant channel metadata and the isolated MNE topomap compatibility layer.

The Ctap EEGLAB readers already expose MNE Info, including montage coordinates;
no sample loading or additional reader methods are needed for these helpers.
"""

import json
import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def _head_geometry(positions):
    """Keep private MNE APIs here, returning JSON-compatible SVG coordinates."""
    import mne
    from mne.channels.channels import _divide_to_regions
    from mne.viz.topomap import _get_pos_outlines

    info = mne.create_info(list(positions), 1.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=positions, coord_frame="head"))
    picks = np.arange(len(positions))
    xy, outlines = _get_pos_outlines(info, picks, sphere=None)
    if not np.isfinite(xy).all():
        raise ValueError("Sensor projection produced invalid coordinates")
    # SVG's y axis points down. Transform outlines and sensors identically.
    points = {name: [float(x), float(-y)] for name, (x, y) in zip(positions, xy)}
    paths = [
        np.column_stack((outlines[key][0], -outlines[key][1])).tolist()
        for key in ("head", "nose", "ear_left", "ear_right")
        if key in outlines
    ]
    regions = {}
    # MNE's equal-size regional partition needs at least four sensors.
    if len(positions) >= 4:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            regions = {
                label: [info.ch_names[int(i)] for i in indices]
                for label, indices in _divide_to_regions(info, add_stim=False).items()
                if len(indices)
            }
    return {"points": points, "outlines": paths, "message": ""}, regions


def channel_metadata(recordings, *, bads_by_step=None, groups=None):
    """Build a stable union from ``(step, Ctap*EEGLAB)`` pairs.

    Supplied bad lists override metadata only for the specified steps. Unknown
    names are ignored. Custom named groups can overlap, like type/region groups.
    """
    channels = {}
    positions = {}
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
        for name, label, ch in zip(
            recording.ch_names, recording.raw_ch_types, recording.info["chs"], strict=True
        ):
            channels.setdefault(name, {"name": name, "type": label})
            pos = np.asarray(ch["loc"][:3], dtype=float)
            if name not in positions and np.isfinite(pos).all() and np.linalg.norm(pos) > 0:
                positions[name] = pos.copy()
    head = {"points": {}, "outlines": [], "message": "No usable sensor positions are available."}
    regions = {}
    if positions:
        try:
            head, regions = _head_geometry(positions)
        except Exception:
            logger.warning("Unable to project channel positions", exc_info=True)
            head["message"] = "Sensor positions could not be projected. Use the channel list."
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
