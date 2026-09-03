"""Bokeh model and server-side page mailbox for the Venn renderer."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

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


class VennTimeSeriesRenderer(Renderer):
    """Pixel Venn renderer whose bulk data is supplied through a page mailbox."""

    __implementation__ = "vennrender.ts"

    request_seq = Int(default=0, help="Monotonic client page-request sequence")
    requested_pages = List(Tuple(Int, Int), default=[], help="(factor, page) requests")
    response_seq = Int(default=0, help="Sequence whose response is currently published")
    response_generation = Int(default=0, help="Viewport generation of published pages")
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

    def __init__(self, *, manifest: VennManifest, **kwargs) -> None:
        kwargs.setdefault("page_source", ColumnDataSource(data=EMPTY_PAGE_DATA))
        kwargs.setdefault("page_metadata_source", ColumnDataSource(data=EMPTY_METADATA))
        kwargs.setdefault("level", "image")
        kwargs.update(
            dataset_version=manifest.dataset_version,
            sample_count=manifest.sample_count,
            time_start=manifest.time_start,
            time_end=manifest.time_end,
            sample_interval=manifest.sample_interval,
            source_factors=list(manifest.source_factors),
            page_size=manifest.page_size,
            page_counts=list(manifest.page_counts),
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
