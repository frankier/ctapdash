"""Montage-based selector geometry, including non-EEG sensors.

The spherical flattening below is adapted from MNE-Python's
channels/layout.py::_auto_topomap_coords (BSD-3-Clause, copyright the
MNE-Python contributors). It operates on montage positions directly so that
Info's channel-type filtering cannot discard EOG or other auxiliary sensors.
"""

# Copyright 2011-2025 MNE-Python authors
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings

import mne
import numpy as np
from mne.channels import transform_to_head
from mne.channels.channels import _divide_to_regions
from mne.viz.topomap import _make_head_outlines
from mne.utils import _check_sphere


def _head_geometry(raw_montage):
    """Return top, front and back SVG views of every usable montage channel.

    Transform a copy using the montage fiducials before projecting. Missing or
    zero positions stay in the checklist, never at an invented sensor location.
    Front view looks at the face: the subject's right is on the viewer's left.
    """
    montage = raw_montage.copy()
    if montage.get_positions()["coord_frame"] != "head":
        montage = transform_to_head(montage)
    positions = {
        name: np.asarray(pos, dtype=float)
        for name, pos in montage.get_positions()["ch_pos"].items()
        if np.isfinite(pos).all() and np.linalg.norm(pos) > 0
    }
    if not positions:
        return {
            "points": {},
            "outlines": [],
            "message": "No usable sensor positions are available.",
        }, {}
    xyz = np.array(list(positions.values()))
    info = mne.create_info(list(positions), 1.0, "eeg")
    for ch, pos in zip(info["chs"], xyz):
        ch["loc"][:3] = pos
    with info._unlock():
        info["dig"] = montage.dig
    # Preserve MNE's automatic fit to extra head-shape points when available.
    sphere = _check_sphere(None, info)
    centred = xyz - sphere[:3]
    radius = np.linalg.norm(centred, axis=1)
    azimuth = np.arctan2(centred[:, 1], centred[:, 0])
    polar = np.arccos(
        np.clip(
            np.divide(
                centred[:, 2], radius, out=np.ones_like(radius), where=radius > 0
            ),
            -1.0,
            1.0,
        )
    )
    distance = radius * polar / (np.pi / 2.0)
    xy = np.column_stack((distance * np.cos(azimuth), distance * np.sin(azimuth)))
    xy += sphere[:2]
    outlines = _make_head_outlines(sphere, xy, "head", sphere[:2])
    points = {name: [float(x), float(-y)] for name, (x, y) in zip(positions, xy)}
    paths = [
        np.column_stack((outlines[key][0], -outlines[key][1])).tolist()
        for key in ("head", "nose", "ear_left", "ear_right")
    ]
    # Orthographic front projection preserves vertical separation of EOG pairs.
    front_points = {
        name: [float(-x), float(-z)]
        for name, (x, y, z) in zip(positions, centred)
        if y >= 0
    }
    back_points = {
        name: [float(x), float(-z)]
        for name, (x, y, z) in zip(positions, centred)
        if y < 0
    }
    angle = np.linspace(0, 2 * np.pi, 101)
    r = sphere[3]
    front_paths = [
        np.column_stack((r * np.cos(angle), r * np.sin(angle))).tolist(),
        [[-0.012, -0.008], [0.0, 0.012], [0.012, -0.008]],
        [[-0.025, 0.045], [0.0, 0.05], [0.025, 0.045]],
    ]
    for x in (-0.035, 0.035):
        front_paths.append(
            np.column_stack(
                (x + 0.015 * np.cos(angle), -0.025 + 0.007 * np.sin(angle))
            ).tolist()
        )
    regions = {}
    if len(positions) >= 4:
        # Region partitioning only reads loc; include all montage channel types.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            regions = {
                label: [info.ch_names[int(i)] for i in indices]
                for label, indices in _divide_to_regions(info, add_stim=False).items()
                if len(indices)
            }
    return {
        "points": points,
        "outlines": paths,
        "message": "",
        "front": {"points": front_points, "outlines": front_paths, "message": ""},
        "back": {"points": back_points, "outlines": front_paths[:1], "message": ""},
    }, regions
