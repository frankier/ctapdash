"""Selection-independent epoch pages and a bounded binary transport.

Each source epoch is read once per page; segment options reference slices of
that block. A/B choices are independent, never a Cartesian product. The wire
packet is a padded UTF-8 JSON header followed by little-endian float32 data.
Transport chunks may split anywhere in this packet (including inside a float).
"""

import json
import struct

import numpy as np

TRANSPORT_BYTES = 1024 * 1024


def build_tile(
    domain, source_indices, channels, factor, page_size, page, channel_page, layer
):
    if layer not in ("venn", "lines"):
        raise ValueError(f"Unknown tile layer {layer!r}")
    ch_start, ch_stop = channels
    if ch_start >= ch_stop:
        raise IndexError("Channel tile is outside the selection")
    fragments = list(domain.batch_fragments(factor, page_size, page))
    if layer == "lines":
        # Halo samples allow the browser to interpolate exactly at page edges.
        # Never extend beyond a segment into a different epoch or compacted gap.
        fragments = [
            (
                sid,
                s,
                max(0, first - 1),
                min(domain.bucket_geometry(s, factor)[1], last + 1),
                lo,
                hi,
            )
            for sid, s, first, last, lo, hi in fragments
        ]
    spans = {}
    for _, segment, first, last, _, _ in fragments:
        bucket = domain.sample_start(segment) // factor
        for side in range(len(domain.sources)):
            for selected in segment.choices(side):
                key = side, selected
                low, high = bucket + first, bucket + last
                if key in spans:
                    low, high = min(low, spans[key][0]), max(high, spans[key][1])
                spans[key] = low, high
    blocks, parts, size = {}, [], 0
    components = 2 if layer == "venn" else 1
    for (side, selected), (low, high) in spans.items():
        source = domain.sources[side]
        segment = source.segments[selected]
        source_bucket = round(segment.time_start / domain.interval) // factor
        values = (
            source.read_segment(
                segment,
                source_indices[side][ch_start:ch_stop],
                0,
                segment.sample_count,
                factor,
                low - source_bucket,
                high - source_bucket,
                layer,
            )
            .astype("<f4", copy=False)
            .ravel()
        )
        blocks[side, selected] = dict(
            offset=size, stride=(high - low) * components, bucket=low
        )
        parts.append(values)
        size += values.size
    rows = []
    for sid, segment, first, last, local_low, local_high in fragments:
        bucket = domain.sample_start(segment) // factor + first
        options = []
        for side in range(len(domain.sources)):
            choices = []
            for selected in segment.choices(side):
                block = blocks[side, selected]
                choices.append(
                    [
                        block["offset"] + (bucket - block["bucket"]) * components,
                        block["stride"],
                    ]
                )
            options.append(choices)
        grid_start, _ = domain.bucket_geometry(segment, factor)
        rows.append(
            dict(
                segment=sid,
                segment_start=segment.display_start,
                count=last - first,
                start=segment.display_start + local_low * domain.interval,
                end=segment.display_start + local_high * domain.interval,
                time=grid_start + first * factor * domain.interval,
                options=options,
            )
        )
    low = domain.start + page * page_size * factor * domain.interval
    return dict(
        layer=layer,
        factor=factor,
        page=page,
        channel_page=channel_page,
        channel_start=ch_start,
        channel_count=ch_stop - ch_start,
        start=low,
        end=min(domain.end, low + page_size * factor * domain.interval),
        step=factor * domain.interval,
        fragments=rows,
    ), np.concatenate(parts) if parts else np.empty(0, "<f4")


def pack_tiles(tiles):
    headers, arrays, offset = [], [], 0
    for header, values in tiles:
        headers.append(dict(header, offset=offset, length=values.size))
        arrays.append(values)
        offset += values.size
    header = json.dumps(headers, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 4)
    return b"".join(
        [struct.pack("<I", len(header)), header, *(a.tobytes() for a in arrays)]
    )
