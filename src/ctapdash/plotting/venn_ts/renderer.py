"""Bokeh model and server-side page mailbox for the Venn renderer."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

import numpy as np

from bokeh.core.properties import (
    Bool,
    Color,
    Float,
    Int,
    Instance,
    List,
    String,
    Tuple,
)
from bokeh.document import Document
from bokeh.models import ColumnDataSource, Renderer

from ctapdash.plotting.venn_ts.venn import VennManifest, load_page, pack_pages
from ctapdash.plotting.venn_ts.range_series import RangeSeries
from ctapdash.plotting.venn_ts.range_series import RecordingTileSource


EMPTY_PAGE_DATA = {"minimum": [], "maximum": [], "valid": []}
EMPTY_METADATA = {
    "source_factor": [],
    "page_index": [],
    "series_id": [],
    "offset": [],
    "length": [],
    "core_start": [],
    "core_length": [],
    "time_start": [],
    "time_step": [],
}
EMPTY_RANGE_TILE_DATA = {
    "minimum_a": [], "maximum_a": [], "minimum_b": [], "maximum_b": [],
    "valid_a": [], "valid_b": [],
}
EMPTY_RANGE_TILE_METADATA = {
    "factor": [], "x_page": [], "channel_page": [], "offset": [], "length": [],
    "channel_start": [], "channel_count": [], "entry_count": [], "data_start": [],
    "core_start": [], "core_length": [], "time_start": [], "time_step": [],
}
EMPTY_LINE_TILE_DATA = {"time": [], "value_a": [], "value_b": []}
EMPTY_LINE_TILE_METADATA = {
    "factor": [], "x_page": [], "channel_page": [], "channel_index": [],
    "offset": [], "length": [],
}


class VennTimeSeriesRenderer(Renderer):
    """Pixel Venn renderer whose bulk data is supplied through a page mailbox."""

    __implementation__ = "vennrender.ts"

    request_seq = Int(default=0, help="Monotonic client page-request sequence")
    requested_pages = List(Tuple(Int, Int), default=[], help="(factor, page) requests")
    response_seq = Int(default=0, help="Sequence whose response is currently published")
    response_ack = Int(default=0, help="Last response consumed by the browser")
    response_generation = Int(default=0, help="Viewport generation of published pages")
    response_tiles = List(
        Tuple(String, Int, Int, Int),
        default=[],
        help="Tile keys covered by the current response, including failed tiles",
    )
    response_error = String(default="", help="Failure associated with the current response")
    page_source = Instance(ColumnDataSource)
    page_metadata_source = Instance(ColumnDataSource)
    dataset_version = String(default="")
    sample_count = Int(default=0)
    time_start = Float(default=0)
    time_end = Float(default=0)
    sample_interval = Float(default=1)
    source_factors = List(Int, default=[])
    page_size = Int(default=2048)
    page_counts = List(Int, default=[])
    shader_schema_version = Int(default=1)
    color_a = Color(default="red")
    color_b = Color(default="blue")
    color_overlap = Color(default="black")
    amplitude_scale = Float(
        default=1.0,
        help="Scale mapping source amplitudes into the plot's y coordinates",
    )
    amplitude_offset = Float(
        default=0.0,
        help="Offset mapping source amplitudes into the plot's y coordinates",
    )
    max_ranges_per_pixel = Int(default=32)
    prefetch_pages = Int(default=1)
    lod_hysteresis = Float(default=0.2)
    rendered_cache_bytes = Int(default=64 * 1024 * 1024)
    data_cache_bytes = Int(default=64 * 1024 * 1024)
    ready = Bool(default=False, help="All exact visible pages have been painted")
    error = String(default="", help="Actionable renderer or protocol failure")
    composition_mode = String(
        default="pending",
        help="Active client composition path: bokeh_webgl or readback",
    )
    current_lod = Int(default=1)
    cpu_cache_bytes = Int(default=0)
    gpu_cache_bytes = Int(default=0)
    cache_hits = Int(default=0)
    cache_misses = Int(default=0)
    requested_tiles = List(
        Tuple(String, Int, Int, Int),
        default=[],
        help="(layer, factor, x page, channel page) tile requests",
    )
    range_tile_source = Instance(ColumnDataSource)
    range_tile_metadata_source = Instance(ColumnDataSource)
    line_tile_source = Instance(ColumnDataSource)
    line_tile_metadata_source = Instance(ColumnDataSource)
    line_source_a = Instance(ColumnDataSource)
    line_source_b = Instance(ColumnDataSource)
    channel_names = List(String, default=[])
    amplitude_scales = List(Float, default=[])
    amplitude_offsets = List(Float, default=[])
    channel_y_mins = List(Float, default=[])
    channel_y_maxs = List(Float, default=[])
    range_factors = List(Int, default=[])
    range_page_counts = List(Int, default=[])
    line_factors = List(Int, default=[])
    line_page_counts = List(Int, default=[])
    channel_tile_size = Int(default=1)
    venn_visible = Bool(default=True)
    lines_visible = Bool(default=True)
    shader_compilations = Int(default=0)
    tile_requests = Int(default=0)
    last_tile_latency_ms = Float(default=0)
    last_paint_ms = Float(default=0)

    def __init__(self, *, manifest: VennManifest | None = None, **kwargs) -> None:
        kwargs.setdefault("page_source", ColumnDataSource(data=EMPTY_PAGE_DATA))
        kwargs.setdefault("page_metadata_source", ColumnDataSource(data=EMPTY_METADATA))
        kwargs.setdefault("range_tile_source", ColumnDataSource(data=EMPTY_RANGE_TILE_DATA))
        kwargs.setdefault(
            "range_tile_metadata_source",
            ColumnDataSource(data=EMPTY_RANGE_TILE_METADATA),
        )
        kwargs.setdefault("line_tile_source", ColumnDataSource(data=EMPTY_LINE_TILE_DATA))
        kwargs.setdefault(
            "line_tile_metadata_source",
            ColumnDataSource(data=EMPTY_LINE_TILE_METADATA),
        )
        kwargs.setdefault(
            "line_source_a",
            ColumnDataSource(
                data={"xs": [], "ys": [], "channel": []}, syncable=False
            ),
        )
        kwargs.setdefault(
            "line_source_b",
            ColumnDataSource(
                data={"xs": [], "ys": [], "channel": []}, syncable=False
            ),
        )
        kwargs.setdefault("level", "image")
        if manifest is not None:
            kwargs.update(
                dataset_version=manifest.dataset_version,
                sample_count=manifest.sample_count,
                time_start=manifest.time_start,
                time_end=manifest.time_end,
                sample_interval=manifest.sample_interval,
                source_factors=list(manifest.source_factors),
                page_size=manifest.page_size,
                page_counts=list(manifest.page_counts),
                range_factors=list(manifest.source_factors),
                range_page_counts=list(manifest.page_counts),
                shader_schema_version=manifest.shader_schema_version,
            )
        super().__init__(**kwargs)


class PageMailbox:
    """Load page requests off the document lock and publish atomically."""

    def __init__(
        self,
        document: Document,
        renderer: VennTimeSeriesRenderer,
        series: tuple[RangeSeries, RangeSeries],
        *,
        workers: int = 2,
    ) -> None:
        self.document = document
        self.renderer = renderer
        self.series = series
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="venn-pages")
        self._closed = False
        self._lock = Lock()
        renderer.on_change("request_seq", self._request)

    def _request(self, _attr: str, _old: int, seq: int) -> None:
        requests = tuple(tuple(item) for item in self.renderer.requested_pages)
        common_length = self.renderer.sample_count
        page_size = self.renderer.page_size
        future = self._executor.submit(
            self._load, seq, requests, common_length, page_size
        )
        future.add_done_callback(self._loaded)

    def _load(
        self,
        seq: int,
        requests: tuple[tuple[int, int], ...],
        common_length: int | None = None,
        page_size: int | None = None,
    ):
        common_length = self.renderer.sample_count if common_length is None else common_length
        page_size = self.renderer.page_size if page_size is None else page_size
        pages = []
        for factor, page_index in requests:
            for series_id, source in enumerate(self.series):
                pages.append(
                    load_page(
                        source,
                        series_id=series_id,
                        source_factor=factor,
                        page_index=page_index,
                        common_length=common_length,
                        page_size=page_size,
                    )
                )
        return seq, pack_pages(pages)

    def _loaded(self, future: Future) -> None:
        try:
            seq, (data, metadata) = future.result()
        except Exception as error:
            message = f"page request failed: {error}"

            def publish_error() -> None:
                if not self._closed:
                    self.renderer.error = message

            self.document.add_next_tick_callback(publish_error)
            return

        def publish() -> None:
            if self._closed:
                return
            self.renderer.page_source.data = data
            self.renderer.page_metadata_source.data = metadata
            self.renderer.response_generation = seq
            self.renderer.response_seq = seq

        self.document.add_next_tick_callback(publish)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.renderer.remove_on_change("request_seq", self._request)
        self._executor.shutdown(wait=False, cancel_futures=True)


class TileCoordinator:
    """Serve all channel/time tiles for one comparison through one mailbox."""

    def __init__(
        self,
        document: Document,
        renderer: VennTimeSeriesRenderer,
        sources: tuple[RecordingTileSource, RecordingTileSource],
        channels: list[str],
        *,
        source_channels: tuple[list[str], list[str]] | None = None,
        workers: int = 2,
    ) -> None:
        self.document = document
        self.renderer = renderer
        self.sources = sources
        self.channels = tuple(channels)
        selections = source_channels or (channels, channels)
        self.source_indices = tuple(
            source.indices(selected) for source, selected in zip(sources, selections)
        )
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="venn-tiles")
        self._closed = False
        self._lock = Lock()
        self._pending_responses = {}
        self._publishing_response = None
        self._next_response_seq = renderer.request_seq + 1
        renderer.on_change("request_seq", self._request)
        renderer.on_change("response_ack", self._acknowledge)

    def _request(self, _attr: str, _old: int, seq: int) -> None:
        requests = tuple(tuple(item) for item in self.renderer.requested_tiles)
        self.renderer.tile_requests += len(requests)
        future = self._executor.submit(self._load, seq, requests)
        future.add_done_callback(
            lambda completed, request_seq=seq, requested=requests:
                self._loaded(completed, request_seq, requested)
        )

    def _channel_slice(self, channel_page: int) -> tuple[int, int]:
        start = channel_page * self.renderer.channel_tile_size
        return start, min(start + self.renderer.channel_tile_size, len(self.channels))

    def _bounds(self, source, layer: str, factor: int, x_page: int):
        length = min(
            source.level_length(layer, factor),
            self.renderer.sample_count // factor,
        )
        core_start = x_page * self.renderer.page_size
        core_stop = min(core_start + self.renderer.page_size, length)
        data_start = max(0, core_start - 1)
        data_stop = min(length, core_stop + 1)
        if core_start >= core_stop:
            raise IndexError(f"tile x page {x_page} is outside the {layer} level")
        return data_start, data_stop, core_start - data_start, core_stop - core_start

    def _range_tile(self, factor: int, x_page: int, channel_page: int):
        channel_start, channel_stop = self._channel_slice(channel_page)
        if channel_start >= channel_stop:
            raise IndexError(f"channel page {channel_page} is outside the selection")
        bounds = self._bounds(self.sources[0], "venn", factor, x_page)
        data_start, data_stop, core_start, core_length = bounds
        arrays = []
        for source, indices in zip(self.sources, self.source_indices):
            other_bounds = self._bounds(source, "venn", factor, x_page)
            if other_bounds != bounds:
                raise ValueError("The selected range-pyramid levels are not aligned")
            arrays.append(
                source.slice_ranges(
                    factor,
                    indices[channel_start:channel_stop],
                    data_start,
                    data_stop,
                )
            )
        a, b = arrays
        if a.shape != b.shape:
            raise ValueError("The selected range tiles have different shapes")
        valid_a = np.isfinite(a).all(axis=-1) & (a[..., 0] <= a[..., 1])
        valid_b = np.isfinite(b).all(axis=-1) & (b[..., 0] <= b[..., 1])
        return {
            "factor": factor,
            "x_page": x_page,
            "channel_page": channel_page,
            "channel_start": channel_start,
            "channel_count": channel_stop - channel_start,
            "entry_count": a.shape[1],
            "data_start": data_start,
            "core_start": core_start,
            "core_length": core_length,
            "time_start": self.renderer.time_start + data_start * factor * self.renderer.sample_interval,
            "time_step": factor * self.renderer.sample_interval,
            "minimum_a": np.where(valid_a, a[..., 0], 0).astype(np.float32).ravel(),
            "maximum_a": np.where(valid_a, a[..., 1], 0).astype(np.float32).ravel(),
            "minimum_b": np.where(valid_b, b[..., 0], 0).astype(np.float32).ravel(),
            "maximum_b": np.where(valid_b, b[..., 1], 0).astype(np.float32).ravel(),
            "valid_a": valid_a.astype(np.uint8).ravel(),
            "valid_b": valid_b.astype(np.uint8).ravel(),
        }

    def _line_tile(self, factor: int, x_page: int, channel_page: int):
        channel_start, channel_stop = self._channel_slice(channel_page)
        if channel_start >= channel_stop:
            raise IndexError(f"channel page {channel_page} is outside the selection")
        bounds = self._bounds(self.sources[0], "line", factor, x_page)
        data_start, data_stop, _core_start, _core_length = bounds
        results = []
        for source, indices in zip(self.sources, self.source_indices):
            results.append(
                source.slice_lines(
                    factor,
                    indices[channel_start:channel_stop],
                    data_start,
                    data_stop,
                )
            )
        (times_a, values_a), (times_b, values_b) = results
        if values_a.shape != values_b.shape or not np.allclose(times_a, times_b):
            raise ValueError("The selected line-pyramid tiles are not aligned")
        rows = []
        for local, channel_index in enumerate(range(channel_start, channel_stop)):
            rows.append({
                "factor": factor,
                "x_page": x_page,
                "channel_page": channel_page,
                "channel_index": channel_index,
                "time": np.asarray(times_a, dtype=np.float64),
                "value_a": np.asarray(values_a[local], dtype=np.float32),
                "value_b": np.asarray(values_b[local], dtype=np.float32),
            })
        return rows

    @staticmethod
    def _pack_ranges(tiles):
        data = {name: [] for name in EMPTY_RANGE_TILE_DATA}
        metadata = {name: [] for name in EMPTY_RANGE_TILE_METADATA}
        offset = 0
        for tile in tiles:
            length = len(tile["minimum_a"])
            for name in data:
                data[name].append(tile[name])
            values = {
                **tile,
                "offset": offset,
                "length": length,
            }
            for name in metadata:
                metadata[name].append(values[name])
            offset += length
        packed_data = {
            name: np.concatenate(parts) if parts else np.empty(0, np.uint8 if name.startswith("valid") else np.float32)
            for name, parts in data.items()
        }
        packed_metadata = {
            name: np.asarray(values, dtype=np.float64 if name in {"time_start", "time_step"} else np.int32)
            for name, values in metadata.items()
        }
        return packed_data, packed_metadata

    @staticmethod
    def _pack_lines(rows):
        data = {name: [] for name in EMPTY_LINE_TILE_DATA}
        metadata = {name: [] for name in EMPTY_LINE_TILE_METADATA}
        offset = 0
        for row in rows:
            length = len(row["time"])
            for name in data:
                data[name].append(row[name])
            values = {**row, "offset": offset, "length": length}
            for name in metadata:
                metadata[name].append(values[name])
            offset += length
        packed_data = {
            name: np.concatenate(parts) if parts else np.empty(0, np.float64 if name == "time" else np.float32)
            for name, parts in data.items()
        }
        packed_metadata = {
            name: np.asarray(values, dtype=np.int32) for name, values in metadata.items()
        }
        return packed_data, packed_metadata

    def _load(self, seq, requests):
        ranges, lines = [], []
        for layer, factor, x_page, channel_page in requests:
            if layer == "venn":
                ranges.append(self._range_tile(factor, x_page, channel_page))
            elif layer == "lines":
                lines.extend(self._line_tile(factor, x_page, channel_page))
            else:
                raise ValueError(f"Unknown tile layer {layer!r}")
        return seq, self._pack_ranges(ranges), self._pack_lines(lines)

    def _loaded(self, future: Future, seq: int, requests) -> None:
        try:
            loaded_seq, (range_data, range_metadata), (line_data, line_metadata) = future.result()
            if loaded_seq != seq:
                raise ValueError(f"tile response {loaded_seq} does not match request {seq}")
            error_message = ""
        except Exception as error:
            range_data = {name: [] for name in EMPTY_RANGE_TILE_DATA}
            range_metadata = {name: [] for name in EMPTY_RANGE_TILE_METADATA}
            line_data = {name: [] for name in EMPTY_LINE_TILE_DATA}
            line_metadata = {name: [] for name in EMPTY_LINE_TILE_METADATA}
            error_message = f"tile request failed: {error}"

        def enqueue() -> None:
            if self._closed:
                return
            self._pending_responses[seq] = (
                tuple(requests), range_data, range_metadata,
                line_data, line_metadata, error_message,
            )
            self._publish_next_response()
        self.document.add_next_tick_callback(enqueue)

    def _publish_next_response(self) -> None:
        if self._closed or self._publishing_response is not None:
            return
        response = self._pending_responses.pop(self._next_response_seq, None)
        if response is None:
            return
        requests, range_data, range_metadata, line_data, line_metadata, error = response
        seq = self._next_response_seq
        self._publishing_response = seq
        self.renderer.range_tile_source.data = range_data
        self.renderer.range_tile_metadata_source.data = range_metadata
        self.renderer.line_tile_source.data = line_data
        self.renderer.line_tile_metadata_source.data = line_metadata
        self.renderer.response_tiles = list(requests)
        self.renderer.response_error = error
        self.renderer.response_generation = seq
        self.renderer.response_seq = seq

    def _acknowledge(self, _attr: str, _old: int, seq: int) -> None:
        if seq != self._publishing_response:
            return
        self._publishing_response = None
        self._next_response_seq = seq + 1
        self._publish_next_response()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.renderer.remove_on_change("request_seq", self._request)
        self.renderer.remove_on_change("response_ack", self._acknowledge)
        self._executor.shutdown(wait=False, cancel_futures=True)
