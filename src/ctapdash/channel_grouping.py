"""Shared natural ordering and substantial numbered channel runs."""

import re
from natsort import natsorted

MIN_GROUP_CHANNELS = 10
NUMBERED_CHANNEL_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>[0-9]+)$")


def split_channel_columns(channels, limit=32):
    """Return the table width and slices, preserving substantial sensor groups."""
    if not len(channels):
        return 0, []
    runs = []
    index = 0
    while index < len(channels):
        match = NUMBERED_CHANNEL_RE.match(str(channels[index]))
        if match is None:
            index += 1
            continue

        prefix = match.group("prefix")
        previous_number = int(match.group("number"))
        run_end = index + 1
        while run_end < len(channels):
            next_match = NUMBERED_CHANNEL_RE.match(str(channels[run_end]))
            if (
                next_match is None
                or next_match.group("prefix") != prefix
                or int(next_match.group("number")) != previous_number + 1
            ):
                break
            previous_number += 1
            run_end += 1
        if run_end - index >= MIN_GROUP_CHANNELS:
            runs.append((index, run_end))
        index = run_end

    largest_group = max((end - start for start, end in runs), default=len(channels))
    maximum = min(limit, largest_group)
    slices = []

    def append_chunks(start, end):
        slices.extend(
            (chunk_start, min(chunk_start + maximum, end))
            for chunk_start in range(start, end, maximum)
        )

    loose_start = 0
    for start, end in runs:
        append_chunks(loose_start, start)
        append_chunks(start, end)
        loose_start = end
    append_chunks(loose_start, len(channels))
    return maximum, slices


def channel_order(channels):
    channel_order = []
    index = 0
    while index < len(channels):
        match = NUMBERED_CHANNEL_RE.match(str(channels[index]))
        if match is None:
            channel_order.append(index)
            index += 1
            continue

        prefix = match.group("prefix")
        run_end = index + 1
        while run_end < len(channels):
            next_match = NUMBERED_CHANNEL_RE.match(str(channels[run_end]))
            if next_match is None or next_match.group("prefix") != prefix:
                break
            run_end += 1
        channel_order.extend(
            natsorted(
                range(index, run_end),
                key=lambda item: str(channels[item]),
            )
        )
        index = run_end
    return channel_order
