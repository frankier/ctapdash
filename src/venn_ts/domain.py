"""Plan comparisons in original time and map their pieces to a compact axis."""

from dataclasses import dataclass
from bisect import bisect_right
from itertools import product
import numpy as np


@dataclass(frozen=True)
class DomainSegment:
    original_start: float
    count: int
    display_start: float
    choices_a: tuple[int, ...]
    choices_b: tuple[int, ...]
    selected_a: int = 0
    selected_b: int = 0

    choices_c: tuple[int, ...] = ()
    selected_c: int = 0

    def choices(self, side):
        return (self.choices_a, self.choices_b, self.choices_c)[side]

    def selection(self, side):
        return (self.selected_a, self.selected_b, self.selected_c)[side]

    def selected(self, side):
        choices = self.choices(side)
        return choices[self.selection(side)] if choices else None


class ComparisonDomain:
    def __init__(self, sources, mode="union", selections=None):
        if mode not in ("union", "intersection"):
            raise ValueError(f"Unknown domain mode: {mode}")
        if not 1 <= len(sources) <= 3:
            raise ValueError("Select one to three steps")
        self.mode = mode
        self.sources = sources
        self.interval = sources[0].sample_interval
        tolerance = abs(self.interval) * 1e-5
        if any(
            abs(self.interval - source.sample_interval) > tolerance
            for source in sources
        ):
            raise ValueError("The selected steps have incompatible sampling intervals")
        for source in sources:
            if not source.segments:
                raise ValueError(
                    f"Cannot align {source.path.name}: no original-time domain"
                )
        origin = min(s.time_start for source in sources for s in source.segments)
        intervals = []
        for source in sources:
            rows = []
            for i, segment in enumerate(source.segments):
                offset = (segment.time_start - origin) / self.interval
                if abs(offset - round(offset)) > 1e-5:
                    raise ValueError(
                        "The selected steps have incompatible original-time sampling grids"
                    )
                start = int(round(offset))
                rows.append((start, start + segment.sample_count, i))
            intervals.append(rows)
        pieces = []
        if mode == "intersection":
            for combination in product(*intervals):
                start = max(row[0] for row in combination)
                end = min(row[1] for row in combination)
                if start < end:
                    pieces.append((start, end, *((row[2],) for row in combination)))
            pieces.sort(key=lambda p: (p[0], *p[2:]))
        else:
            # Sweep endpoints rather than materializing a sample-sized time grid.
            events = {}
            for side, rows in enumerate(intervals):
                for start, end, index in rows:
                    events.setdefault(start, []).append((side, index, True))
                    events.setdefault(end, []).append((side, index, False))
            points = sorted(events)
            active = [set() for _ in sources]
            for start, end in zip(points, points[1:]):
                for side, index, add in events[start]:
                    if add:
                        active[side].add(index)
                    else:
                        active[side].discard(index)
                if any(active):
                    pieces.append(
                        (start, end, *(tuple(sorted(side)) for side in active))
                    )
        if not pieces:
            raise ValueError(
                "The selected files have no shared times (empty intersection). Choose Union."
            )
        self.segments = []
        display = origin
        selections = selections or {}
        for i, (start, end, *choices) in enumerate(pieces):
            a, b, c = choices + [()] * (3 - len(choices))
            self.segments.append(
                DomainSegment(
                    origin + start * self.interval,
                    end - start,
                    display,
                    a,
                    b,
                    min(selections.get((i, 0), 0), max(0, len(a) - 1)),
                    min(selections.get((i, 1), 0), max(0, len(b) - 1)),
                    c,
                    min(selections.get((i, 2), 0), max(0, len(c) - 1)),
                )
            )
            display += (end - start) * self.interval
        self.start, self.end = origin, display
        self.sample_count = sum(s.count for s in self.segments)
        self._page_offsets = {}
        self._display_offsets = np.cumsum(
            [0, *(s.count for s in self.segments)]
        ).tolist()

    def sample_start(self, segment):
        return round(segment.original_start / self.interval)

    def bucket_geometry(self, segment, factor):
        phase = self.sample_start(segment) % factor
        count = (phase + segment.count + factor - 1) // factor
        return segment.display_start - phase * self.interval, count

    def page_counts(self, factors, page_size):
        """Fixed display-time pages, independent of epoch boundaries."""
        return [
            (self.sample_count + f * page_size - 1) // (f * page_size) for f in factors
        ]

    def batch_fragments(self, factor, page_size, page):
        """Intersect a display page with segments, retaining original-grid buckets."""
        low = page * page_size * factor
        high = min(self.sample_count, low + page_size * factor)
        if low < 0 or low >= high:
            raise IndexError("Domain tile is outside the selected domain")
        index = bisect_right(self._display_offsets, low) - 1
        while index < len(self.segments) and self._display_offsets[index] < high:
            segment = self.segments[index]
            local_low = max(0, low - self._display_offsets[index])
            local_high = min(segment.count, high - self._display_offsets[index])
            sample = self.sample_start(segment)
            phase = sample % factor
            first = (phase + local_low) // factor
            last = (phase + local_high + factor - 1) // factor
            yield index, segment, first, last, local_low, local_high
            index += 1

    def segment_page_counts(self, factors, page_size):
        return [
            sum(
                (self.bucket_geometry(s, factor)[1] + page_size - 1) // page_size
                for s in self.segments
            )
            for factor in factors
        ]

    def page(self, factor, page_size, page):
        key = (factor, page_size)
        if key not in self._page_offsets:
            counts = [
                (self.bucket_geometry(s, factor)[1] + page_size - 1) // page_size
                for s in self.segments
            ]
            self._page_offsets[key] = np.cumsum([0, *counts]).tolist()
        offsets = self._page_offsets[key]
        index = bisect_right(offsets, page) - 1
        if page < 0 or index >= len(self.segments):
            raise IndexError("Domain tile is outside the selected domain")
        segment = self.segments[index]
        length = self.bucket_geometry(segment, factor)[1]
        core_start = (page - offsets[index]) * page_size
        core_stop = min(core_start + page_size, length)
        start = max(0, core_start - 1)
        stop = min(length, core_stop + 1)
        return index, segment, start, stop, core_start, core_stop

    def read(self, segment, side, channels, factor, start, stop, layer):
        bucket_start, _ = self.bucket_geometry(segment, factor)
        times = (
            bucket_start
            + np.arange(start, stop, dtype=np.int64) * factor * self.interval
        )
        if layer != "venn":
            times = np.maximum(times, segment.display_start)
        selected = segment.selected(side)
        if selected is None:
            shape = (
                (len(channels), len(times), 2)
                if layer == "venn"
                else (len(channels), len(times))
            )
            return times, np.full(shape, np.nan, dtype=np.float32)
        source = self.sources[side]
        source_segment = source.segments[selected]
        offset = int(
            round((segment.original_start - source_segment.time_start) / self.interval)
        )
        values = source.read_segment(
            source_segment, channels, offset, segment.count, factor, start, stop, layer
        )
        return times, values

    def epoch_starts(self):
        markers = {}
        for segment in self.segments:
            for side in range(len(self.sources)):
                selected = segment.selected(side)
                if selected is None:
                    continue
                source = self.sources[side].segments[selected]
                if (
                    source.epoch is not None
                    and abs(source.time_start - segment.original_start)
                    < self.interval * 1e-5
                ):
                    markers.setdefault(segment.display_start, set()).add(side)
        return markers
