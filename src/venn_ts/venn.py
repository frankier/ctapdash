"""Data and reference-rendering support for the WebGL Venn renderer.

The browser only knows about uniformly-spaced min/max ranges.  This module is
the deliberately boring boundary between recordings/range pyramids and that
wire representation.  Keeping it independent of Bokeh also makes the pixel
contract straightforward to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_PAGE_SIZE = 2048
DEFAULT_MAX_RANGES_PER_PIXEL = 32
DEFAULT_PREFETCH_PAGES = 1
DEFAULT_LOD_HYSTERESIS = 0.20
SHADER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SeriesSelection:
    path: Path
    channel: str


@dataclass(frozen=True)
class PageMetadata:
    source_factor: int
    page_index: int
    series_id: int
    offset: int
    length: int
    core_start: int
    core_length: int
    time_start: float
    time_step: float


@dataclass(frozen=True)
class RangePage:
    metadata: PageMetadata
    minimum: NDArray[np.float32]
    maximum: NDArray[np.float32]
    valid: NDArray[np.uint8]


@dataclass(frozen=True)
class VennManifest:
    dataset_version: str
    sample_count: int
    time_start: float
    time_end: float
    sample_interval: float
    source_factors: tuple[int, ...]
    page_size: int
    page_counts: tuple[int, ...]
    initial_y_start: float
    initial_y_end: float
    shader_schema_version: int = SHADER_SCHEMA_VERSION


def parse_series_args(values: Sequence[Sequence[str]]) -> tuple[SeriesSelection, SeriesSelection]:
    """Validate values collected by repeatable ``--series PATH CHANNEL``."""
    if len(values) != 2:
        raise ValueError(f"--series must be supplied exactly twice (got {len(values)})")
    if any(len(item) != 2 for item in values):
        raise ValueError("each --series requires PATH and CHANNEL")
    return tuple(SeriesSelection(Path(item[0]), item[1]) for item in values)  # type: ignore[return-value]


def validate_alignment(a: RangeSeries, b: RangeSeries) -> int:
    """Return the common sample length after enforcing the time contract."""
    common = min(a.sample_count, b.sample_count)
    if common < 2:
        raise ValueError(
            f"{a.channel_identity} and {b.channel_identity} need at least two common samples"
        )
    interval_tolerance = max(
        abs(a.sample_interval) * 1e-6,
        abs(b.sample_interval) * 1e-6,
        np.finfo(np.float64).eps * 16,
    )
    if abs(a.sample_interval - b.sample_interval) > interval_tolerance:
        raise ValueError(
            "incompatible sample intervals for "
            f"{a.channel_identity} ({a.sample_interval:g}) and "
            f"{b.channel_identity} ({b.sample_interval:g})"
        )
    at = np.asarray(a.times[:common], dtype=np.float64)
    bt = np.asarray(b.times[:common], dtype=np.float64)
    if at.shape != bt.shape:
        raise ValueError("internal error: truncated time arrays have different shapes")
    tolerance = max(interval_tolerance, abs(a.sample_interval) * 1e-5)
    mismatches = np.flatnonzero(np.abs(at - bt) > tolerance)
    if len(mismatches):
        index = int(mismatches[0])
        raise ValueError(
            f"time mismatch at sample {index}: {a.channel_identity}={at[index]:.17g}, "
            f"{b.channel_identity}={bt[index]:.17g} (tolerance {tolerance:g})"
        )
    return common


def shared_source_factors(a: RangeSeries, b: RangeSeries) -> tuple[int, ...]:
    factors = sorted(set(a.source_factors).intersection(b.source_factors))
    if not factors or factors[0] != 1:
        raise ValueError("both series must expose raw samples at source factor 1")
    return tuple(factors)


def build_manifest(
    a: RangeSeries,
    b: RangeSeries,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> VennManifest:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    common = validate_alignment(a, b)
    factors = shared_source_factors(a, b)
    amin, amax = a.finite_extrema(common)
    bmin, bmax = b.finite_extrema(common)
    y_start, y_end = min(amin, bmin), max(amax, bmax)
    if y_start == y_end:
        padding = max(abs(y_start) * 0.05, 1.0)
        y_start -= padding
        y_end += padding
    counts = tuple((common // factor + page_size - 1) // page_size for factor in factors)
    version_text = "|".join((a.dataset_version, b.dataset_version, str(common)))
    version = sha256(version_text.encode()).hexdigest()[:20]
    return VennManifest(
        dataset_version=version,
        sample_count=common,
        time_start=float(a.times[0]),
        time_end=float(a.times[common - 1]),
        sample_interval=(a.sample_interval + b.sample_interval) / 2,
        source_factors=factors,
        page_size=page_size,
        page_counts=counts,
        initial_y_start=y_start,
        initial_y_end=y_end,
    )


def load_page(
    series: RangeSeries,
    *,
    series_id: int,
    source_factor: int,
    page_index: int,
    common_length: int,
    page_size: int,
) -> RangePage:
    """Load a page with a one-entry gutter on each available side."""
    if source_factor not in series.source_factors:
        raise ValueError(f"source factor {source_factor} is unavailable")
    level_length = min(series.level_length(source_factor), common_length // source_factor)
    page_count = (level_length + page_size - 1) // page_size
    if not 0 <= page_index < page_count:
        raise IndexError(f"page {page_index} outside [0, {page_count})")
    core_start = page_index * page_size
    core_stop = min(core_start + page_size, level_length)
    data_start = max(0, core_start - 1)
    data_stop = min(level_length, core_stop + 1)
    ranges = np.asarray(series.slice_ranges(source_factor, data_start, data_stop))
    finite = np.isfinite(ranges).all(axis=1)
    infinite = np.isinf(ranges).any(axis=1)
    if np.any(infinite):
        local = int(np.flatnonzero(infinite)[0])
        raise ValueError(
            f"{series.channel_identity} contains an infinite range at source entry "
            f"{data_start + local} (factor {source_factor})"
        )
    valid = finite & (ranges[:, 0] <= ranges[:, 1])
    minimum = np.where(valid, ranges[:, 0], 0).astype(np.float32, copy=False)
    maximum = np.where(valid, ranges[:, 1], 0).astype(np.float32, copy=False)
    metadata = PageMetadata(
        source_factor=source_factor,
        page_index=page_index,
        series_id=series_id,
        offset=0,
        length=len(ranges),
        core_start=core_start - data_start,
        core_length=core_stop - core_start,
        time_start=series.time_start + data_start * source_factor * series.sample_interval,
        time_step=source_factor * series.sample_interval,
    )
    return RangePage(metadata, minimum, maximum, valid.astype(np.uint8))


def pack_pages(pages: Iterable[RangePage]) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """Pack pages into separate equal-length data and metadata columns."""
    page_list = list(pages)
    minima: list[NDArray[np.float32]] = []
    maxima: list[NDArray[np.float32]] = []
    validity: list[NDArray[np.uint8]] = []
    metadata_rows: list[PageMetadata] = []
    offset = 0
    for page in page_list:
        if not (len(page.minimum) == len(page.maximum) == len(page.valid)):
            raise ValueError("page data columns must have equal lengths")
        minima.append(np.asarray(page.minimum, dtype=np.float32))
        maxima.append(np.asarray(page.maximum, dtype=np.float32))
        validity.append(np.asarray(page.valid, dtype=np.uint8))
        metadata_rows.append(
            PageMetadata(**{**page.metadata.__dict__, "offset": offset})
        )
        offset += len(page.minimum)

    data = {
        "minimum": np.concatenate(minima) if minima else np.empty(0, np.float32),
        "maximum": np.concatenate(maxima) if maxima else np.empty(0, np.float32),
        "valid": np.concatenate(validity) if validity else np.empty(0, np.uint8),
    }
    fields = PageMetadata.__dataclass_fields__
    metadata = {
        name: np.asarray(
            [getattr(row, name) for row in metadata_rows],
            dtype=np.float64 if name in {"time_start", "time_step"} else np.int32,
        )
        for name in fields
    }
    return data, metadata


def reference_raster(
    ranges_a: ArrayLike,
    ranges_b: ArrayLike,
    *,
    width: int,
    height: int,
    y_start: float,
    y_end: float,
) -> NDArray[np.uint8]:
    """Rasterize the shader contract to top-origin, non-antialiased RGBA."""
    a = np.asarray(ranges_a)
    b = np.asarray(ranges_b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("range inputs must have the same (n, 2) shape")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    output = np.zeros((height, width, 4), dtype=np.uint8)
    scale = len(a) / width
    finite_a = np.isfinite(a).all(axis=1)
    finite_b = np.isfinite(b).all(axis=1)
    for x in range(width):
        start = int(scale * x)
        stop = max(start + 1, int(scale * (x + 1)))
        stop = min(stop, len(a))
        if start >= len(a):
            continue
        av = a[start:stop][finite_a[start:stop]]
        bv = b[start:stop][finite_b[start:stop]]
        amin, amax = (float(av[:, 0].min()), float(av[:, 1].max())) if len(av) else (1, 0)
        bmin, bmax = (float(bv[:, 0].min()), float(bv[:, 1].max())) if len(bv) else (1, 0)
        for y in range(height):
            value = y_start + (y_end - y_start) * (1 - y / height)
            inside_a = amin <= value <= amax
            inside_b = bmin <= value <= bmax
            if inside_a and inside_b:
                output[y, x] = (0, 0, 0, 255)
            elif inside_a:
                output[y, x] = (255, 0, 0, 255)
            elif inside_b:
                output[y, x] = (0, 0, 255, 255)
    return output
