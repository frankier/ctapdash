import {Renderer, RendererView} from "@bokehjs/models/renderers/renderer"
import {ColumnDataSource} from "@bokehjs/models/sources/column_data_source"
import type {Context2d} from "@bokehjs/core/util/canvas"
import type {Color} from "@bokehjs/core/types"
import * as p from "@bokehjs/core/properties"

type EpochPager = {
    segment: number
    side: number
    index: number
    count: number
    start: number
    end: number
}

type EpochFragment = {
    segment: number
    segment_start: number
    count: number
    start: number
    end: number
    time: number
    options: [number, number][][] // per side: [block offset, channel stride]
}
type EpochTile = {
    layer: string
    factor: number
    page: number
    channel_page: number
    channel_start: number
    channel_count: number
    start: number
    end: number
    step: number
    fragments: EpochFragment[]
    values: Float32Array
    bytes: number
    used: number
}
type SelectedRanges = {signature: string; parts: [string, RangeTile][]; bytes: number}

type RangeTile = {
    edges?: Float32Array
    clip_start: number
    clip_end: number
    factor: number
    x_page: number
    channel_page: number
    channel_start: number
    channel_count: number
    entry_count: number
    data_start: number
    core_start: number
    core_length: number
    time_start: number
    time_step: number
    ranges: Float32Array
    ranges_c?: Float32Array
    valid: Uint8Array
    bytes: number
    used: number
}

type LineTile = {
    factor: number
    x_page: number
    channel_page: number
    channel_index: number
    time: Float64Array
    a: Float32Array
    b: Float32Array
    c?: Float32Array
    bytes: number
    used: number
}

type GpuTile = {
    lookup_signature: string
    lookup_width: number
    lookup_height: number
    edges: WebGLTexture
    ranges: WebGLTexture
    ranges_c: WebGLTexture
    valid: WebGLTexture
    bytes: number
    used: number
}

type GLResources = {
    gl: WebGLRenderingContext
    program: WebGLProgram
    buffer: WebGLBuffer
}

type InternalReglWrapper = {_regl?: {_refresh?: () => void}}

const vertex_source = `
attribute vec2 a_position;
void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
`

const fragment_source = `
precision highp float;
uniform sampler2D u_ranges;
uniform sampler2D u_ranges_c;
uniform sampler2D u_valid;
uniform sampler2D u_edges;
uniform bool u_irregular;
uniform vec2 u_lookup_size;
uniform vec2 u_texture_size;
uniform float u_entry_count;
uniform float u_channel_row;
uniform float u_source_time_start;
uniform float u_source_time_step;
uniform float u_x_start;
uniform float u_x_end;
uniform float u_y_start;
uniform float u_y_end;
uniform vec2 u_frame_origin;
uniform vec2 u_frame_size;
uniform float u_amplitude_scale;
uniform float u_amplitude_offset;
uniform vec4 u_palette[8];
uniform vec4 u_hull_color;
uniform bool u_hull_visible;

vec2 texel_position(float index) {
  return (vec2(index + 0.5, u_channel_row + 0.5)) / u_texture_size;
}

void main() {
  float local_x = gl_FragCoord.x - u_frame_origin.x;
  float x0 = u_x_start + (local_x - 0.5) / u_frame_size.x * (u_x_end - u_x_start);
  float x1 = u_x_start + (local_x + 0.5) / u_frame_size.x * (u_x_end - u_x_start);
  // Aggregate the entries that start inside this pixel's x span.  When the
  // zoom is so deep that no entry starts in the span, fall back to the
  // previous entry so every pixel inside the data still shows something.
  float first = floor((min(x0, x1) - u_source_time_start) / u_source_time_step);
  float next_entry = floor((max(x0, x1) - u_source_time_start) / u_source_time_step);
  float last = next_entry == first ? first + 1.0 : next_entry;
  first = clamp(first, 0.0, u_entry_count);
  last = clamp(last, first, u_entry_count);

  if (u_irregular) {
    // The CPU caches the clipped bucket span for each screen column. Keeping
    // binary searches out of the fragment shader also makes software WebGL fast.
    float column = floor(local_x);
    vec2 position = vec2(mod(column, u_lookup_size.x) + 0.5,
                         floor(column / u_lookup_size.x) + 0.5) / u_lookup_size;
    vec4 span = texture2D(u_edges, position);
    first = span.r;
    last = span.a;
  }
  float amin = 3.402823466e38;
  float amax = -3.402823466e38;
  float bmin = 3.402823466e38;
  float bmax = -3.402823466e38;
  float cmin = 3.402823466e38;
  float cmax = -3.402823466e38;
  bool has_c = false;
  bool has_a = false;
  bool has_b = false;
  for (int offset = 0; offset < 2048; offset++) {
    float index = first + float(offset);
    if (index >= last) break;
    vec2 position = texel_position(index);
    vec4 ranges = texture2D(u_ranges, position);
    vec4 validity = texture2D(u_valid, position);
    vec4 c = texture2D(u_ranges_c, position);
    if (c.b > 0.0) {
      cmin = min(cmin, c.r);
      cmax = max(cmax, c.g);
      has_c = true;
    }
    if (validity.r > 0.0) {
      amin = min(amin, ranges.r);
      amax = max(amax, ranges.g);
      has_a = true;
    }
    if (validity.a > 0.0) {
      bmin = min(bmin, ranges.b);
      bmax = max(bmax, ranges.a);
      has_b = true;
    }
  }
  float local_y = gl_FragCoord.y - u_frame_origin.y;
  float plot_y = u_y_start + local_y / u_frame_size.y * (u_y_end - u_y_start);
  // The pixel row covers a value extent, not a point, so degenerate
  // (zero-height) ranges still light the single row whose extent contains
  // their value.
  float row = (u_y_end - u_y_start) / u_frame_size.y;
  float v0 = (plot_y - 0.5*row - u_amplitude_offset) / u_amplitude_scale;
  float v1 = (plot_y + 0.5*row - u_amplitude_offset) / u_amplitude_scale;
  float v_lo = min(v0, v1);
  float v_hi = max(v0, v1);
  bool inside_a = has_a && amin <= v_hi && amax >= v_lo;
  bool inside_b = has_b && bmin <= v_hi && bmax >= v_lo;
  bool inside_c = has_c && cmin <= v_hi && cmax >= v_lo;
  int bits = (inside_a ? 1 : 0) + (inside_b ? 2 : 0) + (inside_c ? 4 : 0);
  // Constant indices keep this compatible with WebGL 1 uniform-array rules.
  if (bits == 1) gl_FragColor = u_palette[1];
  else if (bits == 2) gl_FragColor = u_palette[2];
  else if (bits == 3) gl_FragColor = u_palette[3];
  else if (bits == 4) gl_FragColor = u_palette[4];
  else if (bits == 5) gl_FragColor = u_palette[5];
  else if (bits == 6) gl_FragColor = u_palette[6];
  else if (bits == 7) gl_FragColor = u_palette[7];
  else if (u_hull_visible && (has_a || has_b || has_c) &&
           min(amin, min(bmin, cmin)) <= v_hi && max(amax, max(bmax, cmax)) >= v_lo)
    gl_FragColor = u_hull_color;
  else discard;
}
`

function numeric_column(source: ColumnDataSource, name: string): ArrayLike<number> {
    const column = source.get_column(name)
    if (column == null) throw new Error(`missing tile column '${name}'`)
    return column as ArrayLike<number>
}

function css_color(color: Color): [number, number, number, number] {
    if (Array.isArray(color)) {
        if (typeof color[0] == "number") {
            const rgba = color as [number, number, number, number?]
            return [rgba[0] / 255, rgba[1] / 255, rgba[2] / 255, (rgba[3] ?? 255) / 255]
        }
        return css_color(color[0])
    }
    if (typeof color == "number")
        return [
            (color & 255) / 255,
            ((color >>> 8) & 255) / 255,
            ((color >>> 16) & 255) / 255,
            ((color >>> 24) & 255) / 255,
        ]
    const named: {[key: string]: string} = {red: "#ff0000", blue: "#0000ff", black: "#000000"}
    const value = named[color.toLowerCase()] ?? color
    const match = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(value)
    if (match == null) throw new Error(`unsupported Venn color '${color}'`)
    const rgb = parseInt(match[1], 16)
    const alpha = match[2] == null ? 255 : parseInt(match[2], 16)
    return [((rgb >>> 16) & 255) / 255, ((rgb >>> 8) & 255) / 255, (rgb & 255) / 255, alpha / 255]
}

export class VennTimeSeriesRendererView extends RendererView {
    declare model: VennTimeSeriesRenderer
    private pager_elements: {el: HTMLDivElement; item: EpochPager}[] | null = null
    private status_element: HTMLDivElement | null = null
    private pagers_dirty = true
    private epoch_tiles = new Map<string, EpochTile>()
    private selected_ranges = new Map<string, SelectedRanges>()
    private selection_serial = 0
    private epoch_chunks: Uint8Array[] = []
    private resources: GLResources | null = null
    private range_tiles = new Map<string, RangeTile>()
    private line_tiles = new Map<string, LineTile>()
    private gpu_tiles = new Map<string, GpuTile>()
    private in_flight = new Set<string>()
    private frame_request: number | null = null
    private use_counter = 0
    private selected_range_factor = 1
    private selected_line_factor = 1
    private context_canvas: HTMLCanvasElement | null = null
    private requested_at = new Map<string, number>()
    private failure_counts = new Map<string, number>()

    override initialize(): void {
        super.initialize()
    }

    override connect_signals(): void {
        super.connect_signals()
        const {x_source, y_source} = this.coordinates
        this.connect(x_source.change, () => this.schedule())
        this.connect(y_source.change, () => this.schedule())
        this.connect(this.model.properties.response_generation.change, () =>
            this.consume_response(),
        )
        this.connect(this.model.properties.epoch_pagers.change, () => {
            this.pagers_dirty = true
            this.schedule()
        })
        for (const property of [
            this.model.properties.venn_visible,
            this.model.properties.lines_visible,
            this.model.properties.amplitude_scales,
            this.model.properties.amplitude_offsets,
            this.model.properties.channel_y_mins,
            this.model.properties.channel_y_maxs,
            this.model.properties.palette,
            this.model.properties.hull_visible,
            this.model.properties.hull_color,
            this.model.properties.color_a,
            this.model.properties.color_b,
            this.model.properties.color_overlap,
            this.model.properties.data_cache_bytes,
            this.model.properties.rendered_cache_bytes,
        ])
            this.connect(property.change, () => this.schedule())
        for (const property of [
            this.model.properties.ready,
            this.model.properties.error,
            this.model.properties.status,
        ])
            this.connect(property.change, () => this.update_status())
        this.schedule()
    }

    private update_status(): void {
        if (this.status_element == null) {
            const el = document.createElement("div")
            el.className = "comparison-status"
            el.setAttribute("role", "status")
            el.style.cssText =
                "position:absolute;z-index:10;height:26px;box-sizing:border-box;background:white;font:13px sans-serif;line-height:26px;padding:0 4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
            this.plot_view.shadow_el.append(el)
            this.status_element = el
        }
        const el = this.status_element,
            bbox = this.plot_view.frame.bbox
        const message =
            this.model.error || this.model.status || (this.model.ready ? "" : "Loading...")
        el.textContent = message
        el.title = message
        el.style.display = message ? "block" : "none"
        el.style.left = `${bbox.x}px`
        el.style.top = `${Math.max(0, bbox.y - 26)}px`
        el.style.width = `${bbox.width}px`
    }

    private schedule(): void {
        if (this.frame_request != null) return
        this.frame_request = requestAnimationFrame(() => {
            this.frame_request = null
            this.plan_requests()
            this.update_lines()
            this.update_pagers()
            this.request_paint()
        })
    }

    private update_pagers(): void {
        this.update_status()
        if (this.pager_elements == null || this.pagers_dirty) {
            this.pagers_dirty = false
            const existing = new Map(
                (this.pager_elements ?? []).map((p) => [`${p.item.segment}:${p.item.side}`, p]),
            )
            this.pager_elements = []
            const items: EpochPager[] = JSON.parse(this.model.epoch_pagers)
            for (const item of items) {
                const key = `${item.segment}:${item.side}`
                const previous = existing.get(key)
                if (previous != null) {
                    existing.delete(key)
                    if (previous.item.index != item.index || previous.item.count != item.count) {
                        Object.assign(previous.item, item)
                        previous.el.querySelector("span")!.textContent =
                            `${"ABC"[item.side]} ${item.index + 1}/${item.count}`
                        const buttons = previous.el.querySelectorAll("button")
                        buttons[0].disabled = item.index == 0
                        buttons[1].disabled = item.index + 1 >= item.count
                    } else Object.assign(previous.item, item)
                    this.pager_elements.push(previous)
                    continue
                }
                const el = document.createElement("div")
                el.className = "epoch-pager"
                el.style.cssText =
                    "position:absolute;z-index:5;white-space:nowrap;font:11px sans-serif;background:white;border:1px solid #ddd;border-radius:3px;padding:1px"
                // Later controls overlap earlier controls; hover priority is temporary.
                el.addEventListener("mouseenter", () => {
                    el.style.zIndex = "6"
                })
                el.addEventListener("mouseleave", () => {
                    el.style.zIndex = "5"
                })
                const label = "ABC"[item.side]
                el.style.color = String(
                    this.model.palette[1 << item.side] ?? (item.side == 0 ? "#c00" : "#00c"),
                )
                for (const direction of [-1, 0, 1]) {
                    if (direction == 0) {
                        const text = document.createElement("span")
                        text.textContent = `${label} ${item.index + 1}/${item.count}`
                        el.append(text)
                    } else {
                        const button = document.createElement("button")
                        button.textContent = direction == -1 ? "‹" : "›"
                        button.style.cssText =
                            "font:14px sans-serif;padding:0 4px;background:transparent;border:0;cursor:pointer;color:inherit"
                        button.disabled =
                            item.index + direction < 0 || item.index + direction >= item.count
                        button.setAttribute(
                            "aria-label",
                            `${direction == -1 ? "Previous" : "Next"} epoch ${label}, region ${item.segment + 1}`,
                        )
                        button.addEventListener("click", () => {
                            const items = JSON.parse(this.model.epoch_pagers)
                            for (const pager of items)
                                if (pager.segment == item.segment && pager.side == item.side)
                                    pager.index = item.index + direction
                            this.model.epoch_pagers = JSON.stringify(items)
                            this.model.ready = false
                            this.model.epoch_selection = [
                                item.segment,
                                item.side,
                                item.index + direction,
                            ]
                        })
                        el.append(button)
                    }
                }
                this.plot_view.shadow_el.append(el)
                this.pager_elements.push({el, item})
            }
            for (const pager of existing.values()) pager.el.remove()
        }
        const range = this.coordinates.x_source,
            bbox = this.plot_view.frame.bbox
        // Batch layout reads before position writes. Hundreds of overlapping
        // epoch controls otherwise force a browser layout for every indicator.
        for (const {el, item} of this.pager_elements)
            el.style.display =
                Math.min(item.end, range.end) <= Math.max(item.start, range.start)
                    ? "none"
                    : "block"
        const widths = this.pager_elements.map((p) => p.el.offsetWidth)
        for (let i = 0; i < this.pager_elements.length; i++) {
            const {el, item} = this.pager_elements[i]
            const low = Math.max(item.start, range.start),
                high = Math.min(item.end, range.end)
            if (high <= low) continue
            const center =
                bbox.x + (((low + high) / 2 - range.start) / (range.end - range.start)) * bbox.width
            const width = widths[i]
            const left = Math.min(bbox.right - width, Math.max(bbox.x, center - width / 2))
            el.style.left = `${left}px`
            el.style.top = `${Math.max(0, bbox.y - 26)}px`
        }
    }

    private choose_factor(factors: number[]): number {
        const width = Math.max(1, this.plot_view.frame.bbox.width * this.canvas.pixel_ratio)
        const samples_per_pixel =
            Math.abs(this.coordinates.x_source.end - this.coordinates.x_source.start) /
            this.model.sample_interval /
            width
        let selected = factors[0] ?? 1
        for (const factor of factors)
            if (factor <= Math.max(1, samples_per_pixel)) selected = factor
        return selected
    }

    private page_count(layer: string, factor: number): number {
        const factors = layer == "venn" ? this.model.range_factors : this.model.line_factors
        const counts = layer == "venn" ? this.model.range_page_counts : this.model.line_page_counts
        const index = factors.indexOf(factor)
        if (this.model.segment_counts.length != 0)
            return Math.ceil(this.model.sample_count / (factor * this.model.page_size))
        return index < 0 ? 0 : counts[index]
    }

    private visible_x_pages(layer: string, factor: number, prefetch = 0): number[] {
        const range = this.coordinates.x_source
        const low = Math.max(this.model.time_start, Math.min(range.start, range.end))
        const high = Math.min(this.model.time_end, Math.max(range.start, range.end))
        if (high < low) return []
        const first_entry = Math.max(
            0,
            Math.floor((low - this.model.time_start) / (factor * this.model.sample_interval)),
        )
        const last_entry = Math.max(
            first_entry,
            Math.floor((high - this.model.time_start) / (factor * this.model.sample_interval)),
        )
        const count = this.page_count(layer, factor)
        const first = Math.max(0, Math.floor(first_entry / this.model.page_size) - prefetch)
        const last = Math.min(count - 1, Math.floor(last_entry / this.model.page_size) + prefetch)
        const pages = []
        for (let page = first; page <= last; page++) pages.push(page)
        return pages
    }

    private visible_channel_pages(prefetch = 0): number[] {
        const range = this.coordinates.y_source
        const low = Math.min(range.start, range.end),
            high = Math.max(range.start, range.end)
        const visible = []
        for (let index = 0; index < this.model.channel_names.length; index++) {
            const bottom = this.model.channel_y_mins[index] ?? -Infinity
            const top = this.model.channel_y_maxs[index] ?? Infinity
            if (top >= low && bottom <= high)
                visible.push(Math.floor(index / this.model.channel_tile_size))
        }
        const pages = new Set<number>()
        const count = Math.ceil(this.model.channel_names.length / this.model.channel_tile_size)
        for (const page of visible)
            for (
                let candidate = Math.max(0, page - prefetch);
                candidate <= Math.min(count - 1, page + prefetch);
                candidate++
            )
                pages.add(candidate)
        return [...pages]
    }

    private tile_key(layer: string, factor: number, x_page: number, channel_page: number): string {
        return `${this.model.dataset_version}:${layer}:${factor}:${this.model.page_size}:${x_page}:${channel_page}`
    }

    private response_key(
        layer: string,
        factor: number,
        x_page: number,
        channel_page: number,
    ): string {
        return this.tile_key(layer, factor, x_page, channel_page)
    }

    private has_tile(layer: string, factor: number, x_page: number, channel_page: number): boolean {
        const key = this.tile_key(layer, factor, x_page, channel_page)
        if (this.epoch_tiles.has(key)) return true
        return layer == "venn"
            ? this.range_tiles.has(key)
            : this.has_line_page(factor, x_page, channel_page)
    }

    private has_line_page(factor: number, x_page: number, channel_page: number): boolean {
        const start = channel_page * this.model.channel_tile_size
        const stop = Math.min(start + this.model.channel_tile_size, this.model.channel_names.length)
        for (let channel = start; channel < stop; channel++)
            if (
                !this.line_tiles.has(
                    `${this.tile_key("lines", factor, x_page, channel_page)}:${channel}`,
                )
            )
                return false
        return start < stop
    }

    private plan_requests(): void {
        this.selected_range_factor = this.choose_factor(this.model.range_factors)
        this.selected_line_factor = this.choose_factor(this.model.line_factors)
        this.model.current_lod = this.model.venn_visible
            ? this.selected_range_factor
            : this.selected_line_factor
        const channel_pages = this.visible_channel_pages(1)
        const candidates: [string, number, number, number][] = []
        let hits = 0,
            misses = 0
        for (const [layer, enabled, factor] of [
            ["venn", this.model.venn_visible, this.selected_range_factor],
            ["lines", this.model.lines_visible, this.selected_line_factor],
        ] as [string, boolean, number][]) {
            if (!enabled) continue
            for (const x_page of this.visible_x_pages(layer, factor, 1))
                for (const channel_page of channel_pages) {
                    const key = this.tile_key(layer, factor, x_page, channel_page)
                    if (this.has_tile(layer, factor, x_page, channel_page)) hits += 1
                    else {
                        misses += 1
                        if (!this.in_flight.has(key) && (this.failure_counts.get(key) ?? 0) < 3)
                            candidates.push([layer, factor, x_page, channel_page])
                    }
                }
        }
        this.model.cache_hits = hits
        this.model.cache_misses = misses
        candidates.sort((a, b) => {
            // Fill one channel from left to right before proceeding downwards.
            // Pair the two layers at each x page so the line and Venn views become
            // useful together instead of one layer monopolizing early batches.
            const channel_order = a[3] - b[3]
            if (channel_order != 0) return channel_order
            const x_order = a[2] - b[2]
            if (x_order != 0) return x_order
            const layer_order = (a[0] == "venn" ? 0 : 1) - (b[0] == "venn" ? 0 : 1)
            return layer_order
        })
        if (this.in_flight.size >= 16) return
        const batch = candidates.slice(0, Math.min(8, 16 - this.in_flight.size))
        if (batch.length == 0) {
            this.model.ready = this.in_flight.size == 0 && misses == 0
            return
        }
        for (const [layer, factor, x_page, channel_page] of batch) {
            const key = this.tile_key(layer, factor, x_page, channel_page)
            this.in_flight.add(key)
            this.requested_at.set(key, performance.now())
        }
        this.model.ready = false
        this.model.requested_tiles = batch
        this.model.request_seq += 1
    }

    private consume_response(): void {
        const response_tiles = this.model.response_tiles
        const failed = this.model.response_error != ""
        try {
            if (failed) this.model.error = this.model.response_error
            else {
                if (this.model.segment_counts.length != 0) this.consume_epoch_chunk()
                else {
                    this.consume_ranges()
                    this.consume_lines()
                }
                if (!this.model.response_final) return
                this.evict_cpu_cache()
                this.model.error = ""
            }
        } catch (error) {
            this.model.error = `Venn tile protocol error: ${error}`
        } finally {
            if (this.model.response_final)
                for (const [layer, factor, x_page, channel_page] of response_tiles) {
                    const key = this.response_key(layer, factor, x_page, channel_page)
                    const current = key == this.tile_key(layer, factor, x_page, channel_page)
                    const succeeded = !failed && this.has_tile(layer, factor, x_page, channel_page)
                    if (succeeded) this.failure_counts.delete(key)
                    else if (current)
                        this.failure_counts.set(key, (this.failure_counts.get(key) ?? 0) + 1)
                    const requested = this.requested_at.get(key)
                    if (requested != null)
                        this.model.last_tile_latency_ms = performance.now() - requested
                    this.requested_at.delete(key)
                    this.in_flight.delete(key)
                }
            if (failed) this.epoch_chunks = []
            this.model.response_ack = this.model.response_generation
            this.schedule()
        }
    }

    private consume_epoch_chunk(): void {
        const payload = numeric_column(this.model.epoch_batch_source, "payload")
        this.epoch_chunks.push(Uint8Array.from(payload))
        if (!this.model.response_final) return
        const packet = new Uint8Array(this.epoch_chunks.reduce((n, p) => n + p.length, 0))
        let offset = 0
        for (const chunk of this.epoch_chunks) {
            packet.set(chunk, offset)
            offset += chunk.length
        }
        this.epoch_chunks = []
        const header_size = new DataView(packet.buffer).getUint32(0, true)
        const headers = JSON.parse(new TextDecoder().decode(packet.subarray(4, 4 + header_size)))
        const values = new Float32Array(packet.buffer, 4 + header_size)
        for (const header of headers) {
            const key = this.tile_key(header.layer, header.factor, header.page, header.channel_page)
            const data = values.slice(header.offset, header.offset + header.length)
            this.drop_selected(key)
            this.epoch_tiles.set(key, {
                ...header,
                values: data,
                bytes: data.byteLength + JSON.stringify(header).length * 2,
                used: ++this.use_counter,
            })
        }
    }

    private choices(): Map<number, number[]> {
        const selections = new Map<number, number[]>()
        for (const item of JSON.parse(this.model.epoch_pagers)) {
            const pair = selections.get(item.segment) ?? [0, 0, 0]
            pair[item.side] = item.index
            selections.set(item.segment, pair)
        }
        return selections
    }

    private drop_selected(key: string): void {
        for (const [draw_key] of this.selected_ranges.get(key)?.parts ?? []) {
            const gpu = this.gpu_tiles.get(draw_key)
            if (gpu != null && this.resources != null) {
                const gl = this.resources.gl
                gl.deleteTexture(gpu.ranges)
                gl.deleteTexture(gpu.ranges_c)
                gl.deleteTexture(gpu.valid)
                gl.deleteTexture(gpu.edges)
                this.gpu_tiles.delete(draw_key)
            }
        }
        this.selected_ranges.delete(key)
    }

    private range_parts(key: string): [string, RangeTile][] {
        const batch = this.epoch_tiles.get(key)
        if (batch == null) {
            const tile = this.range_tiles.get(key)
            return tile == null ? [] : [[key, tile]]
        }
        batch.used = ++this.use_counter
        const choices = this.choices()
        const signature = batch.fragments
            .map((f) => (choices.get(f.segment) ?? [0, 0, 0]).join(","))
            .join(";")
        const previous = this.selected_ranges.get(key)
        if (previous?.signature == signature) return previous.parts
        this.drop_selected(key)
        const parts: [string, RangeTile][] = []
        // Only selected choices reach the GPU. Pack adjacent segment pieces into
        // bounded textures; explicit endpoints retain exact clipping and gaps.
        let entries: {fragment: EpochFragment; index: number; start: number; end: number}[] = []
        const flush = () => {
            if (entries.length == 0) return
            const count = entries.length,
                origin = entries[0].start
            const ranges = new Float32Array(count * batch.channel_count * 4)
            const ranges_c = new Float32Array(count * batch.channel_count * 4)
            const valid = new Uint8Array(count * batch.channel_count * 2)
            const edges = new Float32Array(count * 2)
            for (let i = 0; i < count; i++) {
                const entry = entries[i],
                    f = entry.fragment
                edges[i * 2] = entry.start - origin
                edges[i * 2 + 1] = entry.end - origin
                const selected = choices.get(f.segment) ?? [0, 0, 0]
                for (let side = 0; side < this.model.step_count; side++) {
                    const option = f.options[side]?.[selected[side]]
                    if (option == null) continue
                    for (let ch = 0; ch < batch.channel_count; ch++) {
                        const source = option[0] + ch * option[1] + entry.index * 2
                        const lo = batch.values[source],
                            hi = batch.values[source + 1]
                        if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo > hi) continue
                        const target = ch * count + i
                        if (side == 2) {
                            ranges_c[target * 4] = lo
                            ranges_c[target * 4 + 1] = hi
                            ranges_c[target * 4 + 2] = 1
                        } else {
                            ranges[target * 4 + side * 2] = lo
                            ranges[target * 4 + side * 2 + 1] = hi
                            valid[target * 2 + side] = 1
                        }
                    }
                }
            }
            const tile: RangeTile = {
                factor: batch.factor,
                x_page: batch.page,
                channel_page: batch.channel_page,
                channel_start: batch.channel_start,
                channel_count: batch.channel_count,
                entry_count: count,
                data_start: 0,
                core_start: 0,
                core_length: count,
                clip_start: origin,
                clip_end: entries[count - 1].end,
                time_start: origin,
                time_step: batch.step,
                ranges,
                ranges_c,
                valid,
                edges,
                bytes:
                    ranges.byteLength + ranges_c.byteLength + valid.byteLength + edges.byteLength,
                used: ++this.use_counter,
            }
            parts.push([`${key}:selected:${++this.selection_serial}`, tile])
            entries = []
        }
        for (const f of batch.fragments) {
            for (let i = 0; i < f.count; i++) {
                const start = Math.max(f.start, f.time + i * batch.step)
                const end = Math.min(f.end, f.time + (i + 1) * batch.step)
                if (end <= start) continue
                entries.push({fragment: f, index: i, start, end})
                if (entries.length == 2048) flush()
            }
        }
        flush()
        this.selected_ranges.set(key, {
            signature,
            parts,
            bytes: parts.reduce((n, p) => n + p[1].bytes, 0),
        })
        return parts
    }

    private epoch_lines(batch: EpochTile, channel: number): LineTile[] {
        batch.used = ++this.use_counter
        const choices = this.choices(),
            rows: LineTile[] = []
        for (const f of batch.fragments) {
            const selected = choices.get(f.segment) ?? [0, 0, 0]
            const times = Array.from({length: f.count}, (_, i) =>
                Math.max(f.segment_start, f.time + i * batch.step),
            )
            const points: [number, number, number][] = []
            // Clip interpolated lines at page edges, sharing an identical endpoint
            // with the adjacent page. Segment boundaries always remain separate.
            for (let i = 0; i < times.length; i++) {
                if (i > 0)
                    for (const edge of [f.start, f.end]) {
                        if (times[i - 1] < edge && edge < times[i])
                            points.push([
                                edge,
                                i - 1,
                                (edge - times[i - 1]) / (times[i] - times[i - 1]),
                            ])
                    }
                if (times[i] >= f.start && times[i] <= f.end) points.push([times[i], i, 0])
            }
            const time = Float64Array.from(points, (p) => p[0])
            const a = new Float32Array(points.length).fill(NaN),
                b = new Float32Array(points.length).fill(NaN),
                c = new Float32Array(points.length).fill(NaN)
            for (let side = 0; side < this.model.step_count; side++) {
                const option = f.options[side]?.[selected[side]]
                if (option == null) continue
                const offset = option[0] + (channel - batch.channel_start) * option[1]
                const output = [a, b, c][side]
                for (let i = 0; i < points.length; i++) {
                    const [, index, fraction] = points[i]
                    const value = batch.values[offset + index]
                    output[i] =
                        fraction == 0
                            ? value
                            : value + fraction * (batch.values[offset + index + 1] - value)
                }
            }
            rows.push({
                factor: batch.factor,
                x_page: batch.page,
                channel_page: batch.channel_page,
                channel_index: channel,
                time,
                a,
                b,
                c,
                bytes: time.byteLength + a.byteLength + b.byteLength + c.byteLength,
                used: batch.used,
            })
        }
        // One polyline per tile/channel. NaN separators retain epoch/gap breaks
        // without creating hundreds of tiny Bokeh glyph rows per channel.
        const count = rows.reduce((n, row) => n + row.time.length + 1, 0)
        const time = new Float64Array(count).fill(NaN)
        const a = new Float32Array(count).fill(NaN),
            b = new Float32Array(count).fill(NaN),
            c = new Float32Array(count).fill(NaN)
        let offset = 0
        for (const row of rows) {
            time.set(row.time, offset)
            a.set(row.a, offset)
            b.set(row.b, offset)
            if (row.c != null) c.set(row.c, offset)
            offset += row.time.length + 1
        }
        return [
            {
                factor: batch.factor,
                x_page: batch.page,
                channel_page: batch.channel_page,
                channel_index: channel,
                time,
                a,
                b,
                c,
                used: batch.used,
                bytes: time.byteLength + a.byteLength + b.byteLength + c.byteLength,
            },
        ]
    }

    private consume_ranges(): void {
        const data = this.model.range_tile_source,
            metadata = this.model.range_tile_metadata_source
        const amin = numeric_column(data, "minimum_a"),
            amax = numeric_column(data, "maximum_a")
        const bmin = numeric_column(data, "minimum_b"),
            bmax = numeric_column(data, "maximum_b")
        const va = numeric_column(data, "valid_a"),
            vb = numeric_column(data, "valid_b")
        const factors = numeric_column(metadata, "factor"),
            xpages = numeric_column(metadata, "x_page")
        const cpages = numeric_column(metadata, "channel_page"),
            offsets = numeric_column(metadata, "offset")
        const lengths = numeric_column(metadata, "length"),
            cstarts = numeric_column(metadata, "channel_start")
        const ccounts = numeric_column(metadata, "channel_count"),
            ecounts = numeric_column(metadata, "entry_count")
        const dstarts = numeric_column(metadata, "data_start"),
            corestarts = numeric_column(metadata, "core_start")
        const corelengths = numeric_column(metadata, "core_length"),
            tstarts = numeric_column(metadata, "time_start")
        const tsteps = numeric_column(metadata, "time_step")
        const clip_starts = numeric_column(metadata, "clip_start"),
            clip_ends = numeric_column(metadata, "clip_end")
        for (let row = 0; row < factors.length; row++) {
            const length = lengths[row],
                offset = offsets[row]
            const ranges = new Float32Array(length * 4),
                valid = new Uint8Array(length * 2)
            for (let index = 0; index < length; index++) {
                ranges[index * 4] = amin[offset + index]
                ranges[index * 4 + 1] = amax[offset + index]
                ranges[index * 4 + 2] = bmin[offset + index]
                ranges[index * 4 + 3] = bmax[offset + index]
                valid[index * 2] = va[offset + index]
                valid[index * 2 + 1] = vb[offset + index]
            }
            const key = this.response_key("venn", factors[row], xpages[row], cpages[row])
            if (key != this.tile_key("venn", factors[row], xpages[row], cpages[row])) continue
            this.range_tiles.set(key, {
                factor: factors[row],
                x_page: xpages[row],
                channel_page: cpages[row],
                channel_start: cstarts[row],
                channel_count: ccounts[row],
                entry_count: ecounts[row],
                data_start: dstarts[row],
                core_start: corestarts[row],
                core_length: corelengths[row],
                clip_start: clip_starts[row],
                clip_end: clip_ends[row],
                time_start: tstarts[row],
                time_step: tsteps[row],
                ranges,
                valid,
                bytes: ranges.byteLength + valid.byteLength,
                used: ++this.use_counter,
            })
        }
    }

    private consume_lines(): void {
        const data = this.model.line_tile_source,
            metadata = this.model.line_tile_metadata_source
        const time = numeric_column(data, "time"),
            a = numeric_column(data, "value_a"),
            b = numeric_column(data, "value_b")
        const factors = numeric_column(metadata, "factor"),
            xpages = numeric_column(metadata, "x_page")
        const cpages = numeric_column(metadata, "channel_page"),
            channels = numeric_column(metadata, "channel_index")
        const offsets = numeric_column(metadata, "offset"),
            lengths = numeric_column(metadata, "length")
        for (let row = 0; row < factors.length; row++) {
            const offset = offsets[row],
                length = lengths[row]
            const times = new Float64Array(length),
                av = new Float32Array(length),
                bv = new Float32Array(length)
            for (let index = 0; index < length; index++) {
                times[index] = time[offset + index]
                av[index] = a[offset + index]
                bv[index] = b[offset + index]
            }
            const page_key = this.response_key("lines", factors[row], xpages[row], cpages[row])
            if (page_key != this.tile_key("lines", factors[row], xpages[row], cpages[row])) continue
            this.line_tiles.set(`${page_key}:${channels[row]}`, {
                factor: factors[row],
                x_page: xpages[row],
                channel_page: cpages[row],
                channel_index: channels[row],
                time: times,
                a: av,
                b: bv,
                bytes: times.byteLength + av.byteLength + bv.byteLength,
                used: ++this.use_counter,
            })
        }
    }

    private evict_cpu_cache(): void {
        let bytes = 0
        for (const tile of this.range_tiles.values()) bytes += tile.bytes
        for (const tile of this.line_tiles.values()) bytes += tile.bytes
        for (const tile of this.epoch_tiles.values()) bytes += tile.bytes
        for (const tile of this.selected_ranges.values()) bytes += tile.bytes
        const entries: [string, RangeTile | LineTile | EpochTile, "range" | "line" | "epoch"][] = []
        for (const [key, tile] of this.range_tiles) entries.push([key, tile, "range"])
        for (const [key, tile] of this.line_tiles) entries.push([key, tile, "line"])
        for (const [key, tile] of this.epoch_tiles) entries.push([key, tile, "epoch"])
        entries.sort((a, b) => a[1].used - b[1].used)
        for (const [key, tile, kind] of entries) {
            if (bytes <= this.model.data_cache_bytes) break
            if (kind == "range") this.range_tiles.delete(key)
            else if (kind == "line") this.line_tiles.delete(key)
            else {
                this.epoch_tiles.delete(key)
                bytes -= this.selected_ranges.get(key)?.bytes ?? 0
                this.drop_selected(key)
            }
            bytes -= tile.bytes
        }
        this.model.cpu_cache_bytes = bytes
    }

    private update_lines(): void {
        if (!this.model.lines_visible) {
            this.model.line_source_a.data = {xs: [], ys: [], channel: []}
            this.model.line_source_b.data = {xs: [], ys: [], channel: []}
            this.model.line_source_c.data = {xs: [], ys: [], channel: []}
            return
        }
        const xs: Float64Array[] = [],
            ays: Float32Array[] = [],
            bys: Float32Array[] = [],
            cys: Float32Array[] = [],
            names: string[] = []
        const xpages = this.visible_x_pages("lines", this.selected_line_factor)
        const cpages = this.visible_channel_pages()
        for (const channel_page of cpages)
            for (const x_page of xpages) {
                const page_key = this.tile_key(
                    "lines",
                    this.selected_line_factor,
                    x_page,
                    channel_page,
                )
                const start = channel_page * this.model.channel_tile_size
                const stop = Math.min(
                    start + this.model.channel_tile_size,
                    this.model.channel_names.length,
                )
                for (let channel = start; channel < stop; channel++) {
                    const epoch = this.epoch_tiles.get(page_key)
                    const legacy = this.line_tiles.get(`${page_key}:${channel}`)
                    const rows =
                        epoch != null
                            ? this.epoch_lines(epoch, channel)
                            : legacy == null
                              ? []
                              : [legacy]
                    for (const tile of rows) {
                        tile.used = ++this.use_counter
                        const ay = new Float32Array(tile.a.length),
                            by = new Float32Array(tile.b.length),
                            cy = new Float32Array(tile.a.length).fill(NaN)
                        const scale = this.model.amplitude_scales[channel],
                            offset = this.model.amplitude_offsets[channel]
                        for (let index = 0; index < tile.a.length; index++) {
                            ay[index] = tile.a[index] * scale + offset
                            by[index] = tile.b[index] * scale + offset
                            if (tile.c != null) cy[index] = tile.c[index] * scale + offset
                        }
                        xs.push(tile.time)
                        ays.push(ay)
                        bys.push(by)
                        cys.push(cy)
                        names.push(this.model.channel_names[channel])
                    }
                }
            }
        this.model.line_source_a.data = {xs, ys: ays, channel: names}
        this.model.line_source_b.data = {xs, ys: bys, channel: names}
        this.model.line_source_c.data = {xs, ys: cys, channel: names}
    }

    private compile_shader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
        const shader = gl.createShader(type)
        if (shader == null) throw new Error("Bokeh WebGL could not allocate a shader")
        gl.shaderSource(shader, source)
        gl.compileShader(shader)
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const message = gl.getShaderInfoLog(shader) ?? "shader compilation failed"
            gl.deleteShader(shader)
            throw new Error(message)
        }
        return shader
    }

    private get_resources(): GLResources {
        const state = this.canvas.webgl
        if (state == null) throw new Error("Bokeh's WebGL output backend is required")
        const gl = state.canvas.getContext("webgl")
        if (gl == null) throw new Error("Bokeh's shared WebGL context is unavailable")
        if (this.resources?.gl == gl && gl.isProgram(this.resources.program)) return this.resources
        this.delete_resources()
        if (gl.getExtension("OES_texture_float") == null)
            throw new Error("floating-point textures are unavailable")
        const vertex = this.compile_shader(gl, gl.VERTEX_SHADER, vertex_source)
        const fragment = this.compile_shader(gl, gl.FRAGMENT_SHADER, fragment_source)
        const program = gl.createProgram()
        if (program == null) throw new Error("Bokeh WebGL could not allocate a shader program")
        gl.attachShader(program, vertex)
        gl.attachShader(program, fragment)
        gl.linkProgram(program)
        gl.deleteShader(vertex)
        gl.deleteShader(fragment)
        if (!gl.getProgramParameter(program, gl.LINK_STATUS))
            throw new Error(gl.getProgramInfoLog(program) ?? "shader link failed")
        const buffer = gl.createBuffer()
        if (buffer == null) throw new Error("Bokeh WebGL could not allocate geometry")
        this.resources = {gl, program, buffer}
        this.model.shader_compilations += 1
        if (this.context_canvas != state.canvas) {
            this.context_canvas?.removeEventListener("webglcontextlost", this.context_lost)
            this.context_canvas?.removeEventListener("webglcontextrestored", this.context_restored)
            this.context_canvas = state.canvas
            this.context_canvas.addEventListener("webglcontextlost", this.context_lost)
            this.context_canvas.addEventListener("webglcontextrestored", this.context_restored)
        }
        return this.resources
    }

    private context_lost = (event: Event): void => {
        event.preventDefault()
        this.resources = null
        this.gpu_tiles.clear()
        this.model.gpu_cache_bytes = 0
        this.model.ready = false
    }

    private context_restored = (): void => {
        this.resources = null
        this.schedule()
    }

    private upload_tile(key: string, tile: RangeTile, resources: GLResources): GpuTile {
        const cached = this.gpu_tiles.get(key)
        if (cached != null) {
            cached.used = ++this.use_counter
            return cached
        }
        const {gl} = resources
        const ranges = gl.createTexture(),
            ranges_c = gl.createTexture(),
            valid = gl.createTexture(),
            edges = gl.createTexture()
        if (ranges == null || ranges_c == null || valid == null || edges == null)
            throw new Error("Bokeh WebGL could not allocate tile textures")
        gl.bindTexture(gl.TEXTURE_2D, ranges)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            tile.entry_count,
            tile.channel_count,
            0,
            gl.RGBA,
            gl.FLOAT,
            tile.ranges,
        )
        gl.bindTexture(gl.TEXTURE_2D, ranges_c)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            tile.entry_count,
            tile.channel_count,
            0,
            gl.RGBA,
            gl.FLOAT,
            tile.ranges_c ?? new Float32Array(tile.ranges.length),
        )
        gl.bindTexture(gl.TEXTURE_2D, valid)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.LUMINANCE_ALPHA,
            tile.entry_count,
            tile.channel_count,
            0,
            gl.LUMINANCE_ALPHA,
            gl.UNSIGNED_BYTE,
            tile.valid,
        )
        gl.bindTexture(gl.TEXTURE_2D, edges)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.LUMINANCE_ALPHA,
            1,
            1,
            0,
            gl.LUMINANCE_ALPHA,
            gl.FLOAT,
            new Float32Array(2),
        )
        const gpu: GpuTile = {
            lookup_signature: "",
            lookup_width: 1,
            lookup_height: 1,
            ranges,
            ranges_c,
            valid,
            edges,
            bytes: tile.ranges.byteLength * 2 + tile.valid.byteLength + 8,
            used: ++this.use_counter,
        }
        this.gpu_tiles.set(key, gpu)
        this.evict_gpu_cache(gl)
        return gpu
    }

    private update_lookup(
        tile: RangeTile,
        gpu: GpuTile,
        gl: WebGLRenderingContext,
        width: number,
    ): void {
        if (tile.edges == null) return
        const range = this.coordinates.x_source
        const signature = `${range.start}:${range.end}:${width}`
        if (gpu.lookup_signature == signature) return
        const texture_width = Math.min(width, gl.getParameter(gl.MAX_TEXTURE_SIZE) as number)
        const height = Math.ceil(width / texture_width)
        const lookup = new Float32Array(texture_width * height * 2)
        const edges = tile.edges,
            n = tile.entry_count
        let first = 0,
            last = 0
        for (let pixel = 0; pixel < width; pixel++) {
            const x0 = range.start + (pixel / width) * (range.end - range.start) - tile.time_start
            const x1 =
                range.start + ((pixel + 1) / width) * (range.end - range.start) - tile.time_start
            if (range.end < range.start) {
                first = 0
                last = 0
            }
            while (first < n && edges[first * 2 + 1] <= Math.min(x0, x1)) first++
            last = Math.max(first, last)
            while (last < n && edges[last * 2] < Math.max(x0, x1)) last++
            lookup[pixel * 2] = first
            lookup[pixel * 2 + 1] = last
        }
        gl.activeTexture(gl.TEXTURE2)
        gl.bindTexture(gl.TEXTURE_2D, gpu.edges)
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.LUMINANCE_ALPHA,
            texture_width,
            height,
            0,
            gl.LUMINANCE_ALPHA,
            gl.FLOAT,
            lookup,
        )
        gpu.bytes += lookup.byteLength - gpu.lookup_width * gpu.lookup_height * 8
        gpu.lookup_width = texture_width
        gpu.lookup_height = height
        gpu.lookup_signature = signature
    }

    private evict_gpu_cache(gl: WebGLRenderingContext): void {
        let bytes = 0
        for (const tile of this.gpu_tiles.values()) bytes += tile.bytes
        const entries = [...this.gpu_tiles.entries()].sort((a, b) => a[1].used - b[1].used)
        for (const [key, tile] of entries) {
            if (bytes <= this.model.rendered_cache_bytes) break
            gl.deleteTexture(tile.ranges)
            gl.deleteTexture(tile.ranges_c)
            gl.deleteTexture(tile.valid)
            gl.deleteTexture(tile.edges)
            this.gpu_tiles.delete(key)
            bytes -= tile.bytes
        }
        this.model.gpu_cache_bytes = bytes
    }

    private clip_x(
        value: number,
        canvas_width: number,
        origin_x: number,
        frame_width: number,
    ): number {
        const range = this.coordinates.x_source
        return (
            (2 * (origin_x + ((value - range.start) / (range.end - range.start)) * frame_width)) /
                canvas_width -
            1
        )
    }

    private clip_y(
        value: number,
        canvas_height: number,
        origin_y: number,
        frame_height: number,
    ): number {
        const range = this.coordinates.y_source
        return (
            (2 * (origin_y + ((value - range.start) / (range.end - range.start)) * frame_height)) /
                canvas_height -
            1
        )
    }

    protected override _paint(_ctx: Context2d): void {
        this.update_pagers()
        if (!this.model.venn_visible) return
        const paint_started = performance.now()
        try {
            const resources = this.get_resources(),
                {gl, program, buffer} = resources
            const state = this.canvas.webgl!,
                canvas = state.canvas,
                bbox = this.plot_view.frame.bbox,
                ratio = this.canvas.pixel_ratio
            const frame_width = Math.max(1, Math.round(bbox.width * ratio)),
                frame_height = Math.max(1, Math.round(bbox.height * ratio))
            const origin_x = Math.round(bbox.x * ratio),
                origin_y = Math.round(canvas.height - (bbox.y + bbox.height) * ratio)
            gl.viewport(0, 0, canvas.width, canvas.height)
            gl.enable(gl.SCISSOR_TEST)
            gl.scissor(origin_x, origin_y, frame_width, frame_height)
            gl.disable(gl.BLEND)
            gl.disable(gl.DEPTH_TEST)
            gl.useProgram(program)
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
            const position = gl.getAttribLocation(program, "a_position")
            gl.enableVertexAttribArray(position)
            gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0)
            const uniform1i = (name: string, value: number) =>
                gl.uniform1i(gl.getUniformLocation(program, name), value)
            const uniform1f = (name: string, value: number) =>
                gl.uniform1f(gl.getUniformLocation(program, name), value)
            uniform1i("u_ranges", 0)
            uniform1i("u_valid", 1)
            uniform1f("u_x_start", this.coordinates.x_source.start)
            uniform1f("u_x_end", this.coordinates.x_source.end)
            uniform1f("u_y_start", this.coordinates.y_source.start)
            uniform1f("u_y_end", this.coordinates.y_source.end)
            gl.uniform2f(gl.getUniformLocation(program, "u_frame_origin"), origin_x, origin_y)
            gl.uniform2f(gl.getUniformLocation(program, "u_frame_size"), frame_width, frame_height)
            const palette =
                this.model.palette.length == 8
                    ? this.model.palette
                    : [
                          "#ffffff",
                          this.model.color_a,
                          this.model.color_b,
                          this.model.color_overlap,
                          "#00ff00",
                          "#ffff00",
                          "#00ffff",
                          "#ffffff",
                      ]
            gl.uniform4fv(
                gl.getUniformLocation(program, "u_palette[0]"),
                palette.flatMap(css_color),
            )
            gl.uniform4fv(
                gl.getUniformLocation(program, "u_hull_color"),
                css_color(this.model.hull_color),
            )
            uniform1i("u_hull_visible", this.model.hull_visible ? 1 : 0)
            uniform1i("u_ranges_c", 3)
            const xpages = this.visible_x_pages("venn", this.selected_range_factor),
                cpages = this.visible_channel_pages()
            for (const channel_page of cpages)
                for (const x_page of xpages) {
                    const key = this.tile_key(
                        "venn",
                        this.selected_range_factor,
                        x_page,
                        channel_page,
                    )
                    for (const [draw_key, tile] of this.range_parts(key)) {
                        tile.used = ++this.use_counter
                        const gpu = this.upload_tile(draw_key, tile, resources)
                        this.update_lookup(tile, gpu, gl, frame_width)
                        gl.activeTexture(gl.TEXTURE0)
                        gl.bindTexture(gl.TEXTURE_2D, gpu.ranges)
                        gl.activeTexture(gl.TEXTURE1)
                        gl.bindTexture(gl.TEXTURE_2D, gpu.valid)
                        gl.activeTexture(gl.TEXTURE3)
                        gl.bindTexture(gl.TEXTURE_2D, gpu.ranges_c)
                        gl.activeTexture(gl.TEXTURE2)
                        gl.bindTexture(gl.TEXTURE_2D, gpu.edges)
                        uniform1i("u_edges", 2)
                        gl.uniform2f(
                            gl.getUniformLocation(program, "u_lookup_size"),
                            gpu.lookup_width,
                            gpu.lookup_height,
                        )
                        uniform1i("u_irregular", tile.edges == null ? 0 : 1)
                        gl.uniform2f(
                            gl.getUniformLocation(program, "u_texture_size"),
                            tile.entry_count,
                            tile.channel_count,
                        )
                        uniform1f("u_entry_count", tile.entry_count)
                        uniform1f("u_source_time_start", tile.time_start)
                        uniform1f("u_source_time_step", tile.time_step)
                        const x0 = tile.clip_start
                        const x1 = tile.clip_end
                        for (let local = 0; local < tile.channel_count; local++) {
                            const channel = tile.channel_start + local
                            const y0 = this.model.channel_y_mins[channel],
                                y1 = this.model.channel_y_maxs[channel]
                            const left = this.clip_x(x0, canvas.width, origin_x, frame_width),
                                right = this.clip_x(x1, canvas.width, origin_x, frame_width)
                            const bottom = this.clip_y(y0, canvas.height, origin_y, frame_height),
                                top = this.clip_y(y1, canvas.height, origin_y, frame_height)
                            gl.bufferData(
                                gl.ARRAY_BUFFER,
                                new Float32Array([
                                    left,
                                    bottom,
                                    right,
                                    bottom,
                                    left,
                                    top,
                                    left,
                                    top,
                                    right,
                                    bottom,
                                    right,
                                    top,
                                ]),
                                gl.DYNAMIC_DRAW,
                            )
                            uniform1f("u_channel_row", local)
                            uniform1f("u_amplitude_scale", this.model.amplitude_scales[channel])
                            uniform1f("u_amplitude_offset", this.model.amplitude_offsets[channel])
                            gl.drawArrays(gl.TRIANGLES, 0, 6)
                        }
                    }
                }
            gl.disableVertexAttribArray(position)
            gl.disable(gl.SCISSOR_TEST)
            const refresh = (this.canvas.webgl!.regl_wrapper as unknown as InternalReglWrapper)
                ._regl?._refresh
            if (refresh == null) throw new Error("Bokeh's regl refresh hook is unavailable")
            refresh()
            this.model.composition_mode = "bokeh_webgl"
            this.model.last_paint_ms = performance.now() - paint_started
        } catch (error) {
            this.model.error = `Venn rendering requires Bokeh WebGL with float textures: ${error}`
        }
    }

    private delete_resources(): void {
        if (this.resources != null) {
            const {gl, program, buffer} = this.resources
            for (const tile of this.gpu_tiles.values()) {
                gl.deleteTexture(tile.ranges)
                gl.deleteTexture(tile.ranges_c)
                gl.deleteTexture(tile.valid)
                gl.deleteTexture(tile.edges)
            }
            gl.deleteBuffer(buffer)
            gl.deleteProgram(program)
        }
        this.gpu_tiles.clear()
        this.resources = null
        this.model.gpu_cache_bytes = 0
    }

    override remove(): void {
        if (this.frame_request != null) cancelAnimationFrame(this.frame_request)
        this.context_canvas?.removeEventListener("webglcontextlost", this.context_lost)
        this.context_canvas?.removeEventListener("webglcontextrestored", this.context_restored)
        for (const pager of this.pager_elements ?? []) pager.el.remove()
        this.status_element?.remove()
        this.delete_resources()
        this.range_tiles.clear()
        this.line_tiles.clear()
        this.epoch_tiles.clear()
        this.selected_ranges.clear()
        this.epoch_chunks = []
        this.in_flight.clear()
        super.remove()
    }
}

export namespace VennTimeSeriesRenderer {
    export type Attrs = p.AttrsOf<Props>
    export type Props = Renderer.Props & {
        request_seq: p.Property<number>
        requested_pages: p.Property<[number, number][]>
        requested_tiles: p.Property<[string, number, number, number][]>
        response_seq: p.Property<number>
        response_ack: p.Property<number>
        response_generation: p.Property<number>
        response_tiles: p.Property<[string, number, number, number][]>
        response_error: p.Property<string>
        response_final: p.Property<boolean>
        epoch_batch_source: p.Property<ColumnDataSource>
        page_source: p.Property<ColumnDataSource>
        page_metadata_source: p.Property<ColumnDataSource>
        range_tile_source: p.Property<ColumnDataSource>
        range_tile_metadata_source: p.Property<ColumnDataSource>
        line_tile_source: p.Property<ColumnDataSource>
        line_tile_metadata_source: p.Property<ColumnDataSource>
        line_source_a: p.Property<ColumnDataSource>
        line_source_b: p.Property<ColumnDataSource>
        line_source_c: p.Property<ColumnDataSource>
        dataset_version: p.Property<string>
        status: p.Property<string>
        epoch_pagers: p.Property<string>
        epoch_selection: p.Property<number[]>
        segment_revisions: p.Property<number[]>
        segment_starts: p.Property<number[]>
        segment_counts: p.Property<number[]>
        segment_sample_starts: p.Property<number[]>
        sample_count: p.Property<number>
        time_start: p.Property<number>
        time_end: p.Property<number>
        sample_interval: p.Property<number>
        source_factors: p.Property<number[]>
        page_size: p.Property<number>
        page_counts: p.Property<number[]>
        range_factors: p.Property<number[]>
        range_page_counts: p.Property<number[]>
        line_factors: p.Property<number[]>
        line_page_counts: p.Property<number[]>
        channel_names: p.Property<string[]>
        amplitude_scales: p.Property<number[]>
        amplitude_offsets: p.Property<number[]>
        channel_y_mins: p.Property<number[]>
        channel_y_maxs: p.Property<number[]>
        channel_tile_size: p.Property<number>
        shader_schema_version: p.Property<number>
        step_count: p.Property<number>
        palette: p.Property<Color[]>
        hull_visible: p.Property<boolean>
        hull_color: p.Property<Color>
        color_a: p.Property<Color>
        color_b: p.Property<Color>
        color_overlap: p.Property<Color>
        amplitude_scale: p.Property<number>
        amplitude_offset: p.Property<number>
        prefetch_pages: p.Property<number>
        lod_hysteresis: p.Property<number>
        rendered_cache_bytes: p.Property<number>
        data_cache_bytes: p.Property<number>
        ready: p.Property<boolean>
        error: p.Property<string>
        current_lod: p.Property<number>
        composition_mode: p.Property<string>
        venn_visible: p.Property<boolean>
        lines_visible: p.Property<boolean>
        cpu_cache_bytes: p.Property<number>
        gpu_cache_bytes: p.Property<number>
        cache_hits: p.Property<number>
        cache_misses: p.Property<number>
        shader_compilations: p.Property<number>
        tile_requests: p.Property<number>
        last_tile_latency_ms: p.Property<number>
        last_paint_ms: p.Property<number>
    }
}

export interface VennTimeSeriesRenderer extends VennTimeSeriesRenderer.Attrs {}

export class VennTimeSeriesRenderer extends Renderer {
    static __module__ = "venn_ts.renderer"
    declare properties: VennTimeSeriesRenderer.Props
    declare __view_type__: VennTimeSeriesRendererView
    static {
        this.prototype.default_view = VennTimeSeriesRendererView
        this.define<VennTimeSeriesRenderer.Props>(
            ({Bool, Color, Float, Int, List, Ref, Str, Tuple}) => ({
                request_seq: [Int, 0],
                requested_pages: [List(Tuple(Int, Int)), []],
                requested_tiles: [List(Tuple(Str, Int, Int, Int)), []],
                response_seq: [Int, 0],
                response_ack: [Int, 0],
                response_generation: [Int, 0],
                response_tiles: [List(Tuple(Str, Int, Int, Int)), []],
                response_error: [Str, ""],
                response_final: [Bool, true],
                epoch_batch_source: [Ref(ColumnDataSource)],
                page_source: [Ref(ColumnDataSource)],
                page_metadata_source: [Ref(ColumnDataSource)],
                range_tile_source: [Ref(ColumnDataSource)],
                range_tile_metadata_source: [Ref(ColumnDataSource)],
                line_tile_source: [Ref(ColumnDataSource)],
                line_tile_metadata_source: [Ref(ColumnDataSource)],
                line_source_a: [Ref(ColumnDataSource)],
                line_source_b: [Ref(ColumnDataSource)],
                line_source_c: [Ref(ColumnDataSource)],
                dataset_version: [Str, ""],
                status: [Str, ""],
                epoch_pagers: [Str, "[]"],
                epoch_selection: [List(Int), []],
                segment_revisions: [List(Int), []],
                segment_starts: [List(Float), []],
                segment_counts: [List(Int), []],
                segment_sample_starts: [List(Int), []],
                sample_count: [Int, 0],
                time_start: [Float, 0],
                time_end: [Float, 0],
                sample_interval: [Float, 1],
                source_factors: [List(Int), []],
                page_size: [Int, 2048],
                page_counts: [List(Int), []],
                range_factors: [List(Int), []],
                range_page_counts: [List(Int), []],
                line_factors: [List(Int), []],
                line_page_counts: [List(Int), []],
                channel_names: [List(Str), []],
                amplitude_scales: [List(Float), []],
                amplitude_offsets: [List(Float), []],
                channel_y_mins: [List(Float), []],
                channel_y_maxs: [List(Float), []],
                channel_tile_size: [Int, 1],
                shader_schema_version: [Int, 3],
                step_count: [Int, 2],
                palette: [List(Color), []],
                hull_visible: [Bool, false],
                hull_color: [Color, "#dddddd"],
                color_a: [Color, "red"],
                color_b: [Color, "blue"],
                color_overlap: [Color, "black"],
                amplitude_scale: [Float, 1],
                amplitude_offset: [Float, 0],
                prefetch_pages: [Int, 1],
                lod_hysteresis: [Float, 0.2],
                rendered_cache_bytes: [Int, 64 * 1024 * 1024],
                data_cache_bytes: [Int, 64 * 1024 * 1024],
                ready: [Bool, false],
                error: [Str, ""],
                current_lod: [Int, 1],
                composition_mode: [Str, "pending"],
                venn_visible: [Bool, true],
                lines_visible: [Bool, true],
                cpu_cache_bytes: [Int, 0],
                gpu_cache_bytes: [Int, 0],
                cache_hits: [Int, 0],
                cache_misses: [Int, 0],
                shader_compilations: [Int, 0],
                tile_requests: [Int, 0],
                last_tile_latency_ms: [Float, 0],
                last_paint_ms: [Float, 0],
            }),
        )
    }
}
