import {Renderer, RendererView} from "models/renderers/renderer"
import {ColumnDataSource} from "models/sources/column_data_source"
import type {Context2d} from "core/util/canvas"
import type {Color} from "core/types"
import * as p from "core/properties"

type RangeTile = {
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
  bytes: number
  used: number
}

type GpuTile = {
  ranges: WebGLTexture
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
uniform sampler2D u_valid;
uniform vec2 u_texture_size;
uniform float u_entry_count;
uniform float u_channel_row;
uniform int u_max_aggregation;
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
uniform vec4 u_color_a;
uniform vec4 u_color_b;
uniform vec4 u_color_overlap;

vec2 texel_position(float index) {
  return (vec2(index + 0.5, u_channel_row + 0.5)) / u_texture_size;
}

void main() {
  float local_x = gl_FragCoord.x - u_frame_origin.x;
  float x0 = u_x_start + (local_x - 0.5) / u_frame_size.x * (u_x_end - u_x_start);
  float x1 = u_x_start + (local_x + 0.5) / u_frame_size.x * (u_x_end - u_x_start);
  float first = floor((min(x0, x1) - u_source_time_start) / u_source_time_step);
  float next_entry = floor((max(x0, x1) - u_source_time_start) / u_source_time_step);
  float last = next_entry == first ? first + 1.0 : next_entry;
  first = clamp(first, 0.0, u_entry_count);
  last = clamp(last, first, min(u_entry_count, first + float(u_max_aggregation)));

  float amin = 3.402823466e38;
  float amax = -3.402823466e38;
  float bmin = 3.402823466e38;
  float bmax = -3.402823466e38;
  bool has_a = false;
  bool has_b = false;
  for (int offset = 0; offset < 128; offset++) {
    float index = first + float(offset);
    if (index >= last) break;
    vec2 position = texel_position(index);
    vec4 ranges = texture2D(u_ranges, position);
    vec4 validity = texture2D(u_valid, position);
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
  float value = (plot_y - u_amplitude_offset) / u_amplitude_scale;
  bool inside_a = has_a && amin <= value && value <= amax;
  bool inside_b = has_b && bmin <= value && value <= bmax;
  if (inside_a && inside_b) gl_FragColor = u_color_overlap;
  else if (inside_a) gl_FragColor = u_color_a;
  else if (inside_b) gl_FragColor = u_color_b;
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
      return [rgba[0]/255, rgba[1]/255, rgba[2]/255, (rgba[3] ?? 255)/255]
    }
    return css_color(color[0])
  }
  if (typeof color == "number")
    return [(color & 255)/255, ((color >>> 8) & 255)/255, ((color >>> 16) & 255)/255, ((color >>> 24) & 255)/255]
  const named: {[key: string]: string} = {red: "#ff0000", blue: "#0000ff", black: "#000000"}
  const value = named[color.toLowerCase()] ?? color
  const match = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(value)
  if (match == null) throw new Error(`unsupported Venn color '${color}'`)
  const rgb = parseInt(match[1], 16)
  const alpha = match[2] == null ? 255 : parseInt(match[2], 16)
  return [((rgb >>> 16) & 255)/255, ((rgb >>> 8) & 255)/255, (rgb & 255)/255, alpha/255]
}

export class VennTimeSeriesRendererView extends RendererView {
  declare model: VennTimeSeriesRenderer
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

  override initialize(): void {
    super.initialize()
  }

  override connect_signals(): void {
    super.connect_signals()
    const {x_source, y_source} = this.coordinates
    this.connect(x_source.change, () => this.schedule())
    this.connect(y_source.change, () => this.schedule())
    this.connect(this.model.properties.response_seq.change, () => this.consume_response())
    for (const property of [
      this.model.properties.venn_visible, this.model.properties.lines_visible,
      this.model.properties.amplitude_scales, this.model.properties.amplitude_offsets,
      this.model.properties.channel_y_mins, this.model.properties.channel_y_maxs,
      this.model.properties.color_a, this.model.properties.color_b,
      this.model.properties.color_overlap, this.model.properties.data_cache_bytes,
      this.model.properties.rendered_cache_bytes,
    ]) this.connect(property.change, () => this.schedule())
    this.schedule()
  }

  private schedule(): void {
    if (this.frame_request != null) return
    this.frame_request = requestAnimationFrame(() => {
      this.frame_request = null
      this.plan_requests()
      this.update_lines()
      this.request_paint()
    })
  }

  private choose_factor(factors: number[]): number {
    const width = Math.max(1, this.plot_view.frame.bbox.width*this.canvas.pixel_ratio)
    const samples_per_pixel = Math.abs(this.coordinates.x_source.end - this.coordinates.x_source.start) /
      this.model.sample_interval / width
    let selected = factors[0] ?? 1
    for (const factor of factors) if (factor <= Math.max(1, samples_per_pixel)) selected = factor
    return selected
  }

  private page_count(layer: string, factor: number): number {
    const factors = layer == "venn" ? this.model.range_factors : this.model.line_factors
    const counts = layer == "venn" ? this.model.range_page_counts : this.model.line_page_counts
    const index = factors.indexOf(factor)
    return index < 0 ? 0 : counts[index]
  }

  private visible_x_pages(layer: string, factor: number, prefetch = 0): number[] {
    const range = this.coordinates.x_source
    const low = Math.max(this.model.time_start, Math.min(range.start, range.end))
    const high = Math.min(this.model.time_end, Math.max(range.start, range.end))
    if (high < low) return []
    const first_entry = Math.max(0, Math.floor((low - this.model.time_start)/(factor*this.model.sample_interval)))
    const last_entry = Math.max(first_entry, Math.floor((high - this.model.time_start)/(factor*this.model.sample_interval)))
    const count = this.page_count(layer, factor)
    const first = Math.max(0, Math.floor(first_entry/this.model.page_size) - prefetch)
    const last = Math.min(count - 1, Math.floor(last_entry/this.model.page_size) + prefetch)
    const pages = []
    for (let page = first; page <= last; page++) pages.push(page)
    return pages
  }

  private visible_channel_pages(prefetch = 0): number[] {
    const range = this.coordinates.y_source
    const low = Math.min(range.start, range.end), high = Math.max(range.start, range.end)
    const visible = []
    for (let index = 0; index < this.model.channel_names.length; index++) {
      const bottom = this.model.channel_y_mins[index] ?? -Infinity
      const top = this.model.channel_y_maxs[index] ?? Infinity
      if (top >= low && bottom <= high) visible.push(Math.floor(index/this.model.channel_tile_size))
    }
    const pages = new Set<number>()
    const count = Math.ceil(this.model.channel_names.length/this.model.channel_tile_size)
    for (const page of visible)
      for (let candidate = Math.max(0, page - prefetch); candidate <= Math.min(count - 1, page + prefetch); candidate++)
        pages.add(candidate)
    return [...pages]
  }

  private tile_key(layer: string, factor: number, x_page: number, channel_page: number): string {
    return `${this.model.dataset_version}:${layer}:${factor}:${x_page}:${channel_page}`
  }

  private has_tile(layer: string, factor: number, x_page: number, channel_page: number): boolean {
    const key = this.tile_key(layer, factor, x_page, channel_page)
    return layer == "venn" ? this.range_tiles.has(key) : this.has_line_page(factor, x_page, channel_page)
  }

  private has_line_page(factor: number, x_page: number, channel_page: number): boolean {
    const start = channel_page*this.model.channel_tile_size
    const stop = Math.min(start + this.model.channel_tile_size, this.model.channel_names.length)
    for (let channel = start; channel < stop; channel++)
      if (!this.line_tiles.has(`${this.tile_key("lines", factor, x_page, channel_page)}:${channel}`)) return false
    return start < stop
  }

  private plan_requests(): void {
    this.selected_range_factor = this.choose_factor(this.model.range_factors)
    this.selected_line_factor = this.choose_factor(this.model.line_factors)
    this.model.current_lod = this.model.venn_visible ? this.selected_range_factor : this.selected_line_factor
    const channel_pages = this.visible_channel_pages(1)
    const candidates: [string, number, number, number][] = []
    let hits = 0, misses = 0
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
            if (!this.in_flight.has(key)) candidates.push([layer, factor, x_page, channel_page])
          }
        }
    }
    this.model.cache_hits = hits; this.model.cache_misses = misses
    const x_center = (this.coordinates.x_source.start + this.coordinates.x_source.end)/2
    candidates.sort((a, b) => {
      const ac = (a[2] + 0.5)*this.model.page_size*a[1]*this.model.sample_interval + this.model.time_start
      const bc = (b[2] + 0.5)*this.model.page_size*b[1]*this.model.sample_interval + this.model.time_start
      return Math.abs(ac - x_center) - Math.abs(bc - x_center)
    })
    if (this.in_flight.size >= 16) return
    const batch = candidates.slice(0, Math.min(8, 16 - this.in_flight.size))
    if (batch.length == 0) {
      this.model.ready = this.in_flight.size == 0
      return
    }
    for (const [layer, factor, x_page, channel_page] of batch) {
      const key = this.tile_key(layer, factor, x_page, channel_page)
      this.in_flight.add(key); this.requested_at.set(key, performance.now())
    }
    this.model.ready = false
    this.model.requested_tiles = batch
    this.model.request_seq += 1
  }

  private consume_response(): void {
    try {
      this.consume_ranges()
      this.consume_lines()
      this.evict_cpu_cache()
      this.schedule()
    } catch (error) {
      this.model.error = `Venn tile protocol error: ${error}`
    }
  }

  private consume_ranges(): void {
    const data = this.model.range_tile_source, metadata = this.model.range_tile_metadata_source
    const amin = numeric_column(data, "minimum_a"), amax = numeric_column(data, "maximum_a")
    const bmin = numeric_column(data, "minimum_b"), bmax = numeric_column(data, "maximum_b")
    const va = numeric_column(data, "valid_a"), vb = numeric_column(data, "valid_b")
    const factors = numeric_column(metadata, "factor"), xpages = numeric_column(metadata, "x_page")
    const cpages = numeric_column(metadata, "channel_page"), offsets = numeric_column(metadata, "offset")
    const lengths = numeric_column(metadata, "length"), cstarts = numeric_column(metadata, "channel_start")
    const ccounts = numeric_column(metadata, "channel_count"), ecounts = numeric_column(metadata, "entry_count")
    const dstarts = numeric_column(metadata, "data_start"), corestarts = numeric_column(metadata, "core_start")
    const corelengths = numeric_column(metadata, "core_length"), tstarts = numeric_column(metadata, "time_start")
    const tsteps = numeric_column(metadata, "time_step")
    for (let row = 0; row < factors.length; row++) {
      const length = lengths[row], offset = offsets[row]
      const ranges = new Float32Array(length*4), valid = new Uint8Array(length*2)
      for (let index = 0; index < length; index++) {
        ranges[index*4] = amin[offset + index]; ranges[index*4 + 1] = amax[offset + index]
        ranges[index*4 + 2] = bmin[offset + index]; ranges[index*4 + 3] = bmax[offset + index]
        valid[index*2] = va[offset + index]; valid[index*2 + 1] = vb[offset + index]
      }
      const key = this.tile_key("venn", factors[row], xpages[row], cpages[row])
      this.range_tiles.set(key, {
        factor: factors[row], x_page: xpages[row], channel_page: cpages[row],
        channel_start: cstarts[row], channel_count: ccounts[row], entry_count: ecounts[row],
        data_start: dstarts[row], core_start: corestarts[row], core_length: corelengths[row],
        time_start: tstarts[row], time_step: tsteps[row], ranges, valid,
        bytes: ranges.byteLength + valid.byteLength, used: ++this.use_counter,
      })
      const requested = this.requested_at.get(key)
      if (requested != null) this.model.last_tile_latency_ms = performance.now() - requested
      this.requested_at.delete(key)
      this.in_flight.delete(key)
    }
  }

  private consume_lines(): void {
    const data = this.model.line_tile_source, metadata = this.model.line_tile_metadata_source
    const time = numeric_column(data, "time"), a = numeric_column(data, "value_a"), b = numeric_column(data, "value_b")
    const factors = numeric_column(metadata, "factor"), xpages = numeric_column(metadata, "x_page")
    const cpages = numeric_column(metadata, "channel_page"), channels = numeric_column(metadata, "channel_index")
    const offsets = numeric_column(metadata, "offset"), lengths = numeric_column(metadata, "length")
    const completed = new Set<string>()
    for (let row = 0; row < factors.length; row++) {
      const offset = offsets[row], length = lengths[row]
      const times = new Float64Array(length), av = new Float32Array(length), bv = new Float32Array(length)
      for (let index = 0; index < length; index++) {
        times[index] = time[offset + index]; av[index] = a[offset + index]; bv[index] = b[offset + index]
      }
      const page_key = this.tile_key("lines", factors[row], xpages[row], cpages[row])
      this.line_tiles.set(`${page_key}:${channels[row]}`, {
        factor: factors[row], x_page: xpages[row], channel_page: cpages[row], channel_index: channels[row],
        time: times, a: av, b: bv, bytes: times.byteLength + av.byteLength + bv.byteLength,
        used: ++this.use_counter,
      })
      completed.add(page_key)
    }
    for (const key of completed) {
      const requested = this.requested_at.get(key)
      if (requested != null) this.model.last_tile_latency_ms = performance.now() - requested
      this.requested_at.delete(key); this.in_flight.delete(key)
    }
  }

  private evict_cpu_cache(): void {
    let bytes = 0
    for (const tile of this.range_tiles.values()) bytes += tile.bytes
    for (const tile of this.line_tiles.values()) bytes += tile.bytes
    const entries: [string, RangeTile | LineTile, "range" | "line"][] = []
    for (const [key, tile] of this.range_tiles) entries.push([key, tile, "range"])
    for (const [key, tile] of this.line_tiles) entries.push([key, tile, "line"])
    entries.sort((a, b) => a[1].used - b[1].used)
    for (const [key, tile, kind] of entries) {
      if (bytes <= this.model.data_cache_bytes) break
      if (kind == "range") this.range_tiles.delete(key)
      else this.line_tiles.delete(key)
      bytes -= tile.bytes
    }
    this.model.cpu_cache_bytes = bytes
  }

  private update_lines(): void {
    if (!this.model.lines_visible) {
      this.model.line_source_a.data = {xs: [], ys: [], channel: []}
      this.model.line_source_b.data = {xs: [], ys: [], channel: []}
      return
    }
    const xs: Float64Array[] = [], ays: Float32Array[] = [], bys: Float32Array[] = [], names: string[] = []
    const xpages = this.visible_x_pages("lines", this.selected_line_factor)
    const cpages = this.visible_channel_pages()
    for (const channel_page of cpages) for (const x_page of xpages) {
      const page_key = this.tile_key("lines", this.selected_line_factor, x_page, channel_page)
      const start = channel_page*this.model.channel_tile_size
      const stop = Math.min(start + this.model.channel_tile_size, this.model.channel_names.length)
      for (let channel = start; channel < stop; channel++) {
        const tile = this.line_tiles.get(`${page_key}:${channel}`)
        if (tile == null) continue
        tile.used = ++this.use_counter
        const ay = new Float32Array(tile.a.length), by = new Float32Array(tile.b.length)
        const scale = this.model.amplitude_scales[channel], offset = this.model.amplitude_offsets[channel]
        for (let index = 0; index < tile.a.length; index++) {
          ay[index] = tile.a[index]*scale + offset
          by[index] = tile.b[index]*scale + offset
        }
        xs.push(tile.time); ays.push(ay); bys.push(by); names.push(this.model.channel_names[channel])
      }
    }
    this.model.line_source_a.data = {xs, ys: ays, channel: names}
    this.model.line_source_b.data = {xs, ys: bys, channel: names}
  }

  private compile_shader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type)
    if (shader == null) throw new Error("Bokeh WebGL could not allocate a shader")
    gl.shaderSource(shader, source); gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader) ?? "shader compilation failed"
      gl.deleteShader(shader); throw new Error(message)
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
    gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program)
    gl.deleteShader(vertex); gl.deleteShader(fragment)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "shader link failed")
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
    event.preventDefault(); this.resources = null; this.gpu_tiles.clear(); this.model.gpu_cache_bytes = 0; this.model.ready = false
  }

  private context_restored = (): void => { this.resources = null; this.schedule() }

  private upload_tile(key: string, tile: RangeTile, resources: GLResources): GpuTile {
    const cached = this.gpu_tiles.get(key)
    if (cached != null) { cached.used = ++this.use_counter; return cached }
    const {gl} = resources
    const ranges = gl.createTexture(), valid = gl.createTexture()
    if (ranges == null || valid == null) throw new Error("Bokeh WebGL could not allocate tile textures")
    gl.bindTexture(gl.TEXTURE_2D, ranges)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, tile.entry_count, tile.channel_count, 0, gl.RGBA, gl.FLOAT, tile.ranges)
    gl.bindTexture(gl.TEXTURE_2D, valid)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE_ALPHA, tile.entry_count, tile.channel_count, 0, gl.LUMINANCE_ALPHA, gl.UNSIGNED_BYTE, tile.valid)
    const gpu = {ranges, valid, bytes: tile.ranges.byteLength + tile.valid.byteLength, used: ++this.use_counter}
    this.gpu_tiles.set(key, gpu); this.evict_gpu_cache(gl); return gpu
  }

  private evict_gpu_cache(gl: WebGLRenderingContext): void {
    let bytes = 0
    for (const tile of this.gpu_tiles.values()) bytes += tile.bytes
    const entries = [...this.gpu_tiles.entries()].sort((a, b) => a[1].used - b[1].used)
    for (const [key, tile] of entries) {
      if (bytes <= this.model.rendered_cache_bytes) break
      gl.deleteTexture(tile.ranges); gl.deleteTexture(tile.valid); this.gpu_tiles.delete(key); bytes -= tile.bytes
    }
    this.model.gpu_cache_bytes = bytes
  }

  private clip_x(value: number, canvas_width: number, origin_x: number, frame_width: number): number {
    const range = this.coordinates.x_source
    return 2*(origin_x + (value - range.start)/(range.end - range.start)*frame_width)/canvas_width - 1
  }

  private clip_y(value: number, canvas_height: number, origin_y: number, frame_height: number): number {
    const range = this.coordinates.y_source
    return 2*(origin_y + (value - range.start)/(range.end - range.start)*frame_height)/canvas_height - 1
  }

  protected override _paint(_ctx: Context2d): void {
    if (!this.model.venn_visible) return
    const paint_started = performance.now()
    try {
      const resources = this.get_resources(), {gl, program, buffer} = resources
      const state = this.canvas.webgl!, canvas = state.canvas, bbox = this.plot_view.frame.bbox, ratio = this.canvas.pixel_ratio
      const frame_width = Math.max(1, Math.round(bbox.width*ratio)), frame_height = Math.max(1, Math.round(bbox.height*ratio))
      const origin_x = Math.round(bbox.x*ratio), origin_y = Math.round(canvas.height - (bbox.y + bbox.height)*ratio)
      gl.viewport(0, 0, canvas.width, canvas.height); gl.enable(gl.SCISSOR_TEST); gl.scissor(origin_x, origin_y, frame_width, frame_height)
      gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST); gl.useProgram(program); gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
      const position = gl.getAttribLocation(program, "a_position")
      gl.enableVertexAttribArray(position); gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0)
      const uniform1i = (name: string, value: number) => gl.uniform1i(gl.getUniformLocation(program, name), value)
      const uniform1f = (name: string, value: number) => gl.uniform1f(gl.getUniformLocation(program, name), value)
      uniform1i("u_ranges", 0); uniform1i("u_valid", 1)
      uniform1i("u_max_aggregation", Math.min(128, Math.max(1, this.model.max_ranges_per_pixel)))
      uniform1f("u_x_start", this.coordinates.x_source.start); uniform1f("u_x_end", this.coordinates.x_source.end)
      uniform1f("u_y_start", this.coordinates.y_source.start); uniform1f("u_y_end", this.coordinates.y_source.end)
      gl.uniform2f(gl.getUniformLocation(program, "u_frame_origin"), origin_x, origin_y)
      gl.uniform2f(gl.getUniformLocation(program, "u_frame_size"), frame_width, frame_height)
      gl.uniform4fv(gl.getUniformLocation(program, "u_color_a"), css_color(this.model.color_a))
      gl.uniform4fv(gl.getUniformLocation(program, "u_color_b"), css_color(this.model.color_b))
      gl.uniform4fv(gl.getUniformLocation(program, "u_color_overlap"), css_color(this.model.color_overlap))
      const xpages = this.visible_x_pages("venn", this.selected_range_factor), cpages = this.visible_channel_pages()
      for (const channel_page of cpages) for (const x_page of xpages) {
        const key = this.tile_key("venn", this.selected_range_factor, x_page, channel_page)
        const tile = this.range_tiles.get(key)
        if (tile == null) continue
        tile.used = ++this.use_counter
        const gpu = this.upload_tile(key, tile, resources)
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, gpu.ranges)
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, gpu.valid)
        gl.uniform2f(gl.getUniformLocation(program, "u_texture_size"), tile.entry_count, tile.channel_count)
        uniform1f("u_entry_count", tile.entry_count)
        uniform1f("u_source_time_start", tile.time_start); uniform1f("u_source_time_step", tile.time_step)
        const x0 = this.model.time_start + x_page*this.model.page_size*tile.factor*this.model.sample_interval
        const x1 = Math.min(this.model.time_end, x0 + tile.core_length*tile.factor*this.model.sample_interval)
        for (let local = 0; local < tile.channel_count; local++) {
          const channel = tile.channel_start + local
          const y0 = this.model.channel_y_mins[channel], y1 = this.model.channel_y_maxs[channel]
          const left = this.clip_x(x0, canvas.width, origin_x, frame_width), right = this.clip_x(x1, canvas.width, origin_x, frame_width)
          const bottom = this.clip_y(y0, canvas.height, origin_y, frame_height), top = this.clip_y(y1, canvas.height, origin_y, frame_height)
          gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([left,bottom, right,bottom, left,top, left,top, right,bottom, right,top]), gl.DYNAMIC_DRAW)
          uniform1f("u_channel_row", local); uniform1f("u_amplitude_scale", this.model.amplitude_scales[channel])
          uniform1f("u_amplitude_offset", this.model.amplitude_offsets[channel]); gl.drawArrays(gl.TRIANGLES, 0, 6)
        }
      }
      gl.disableVertexAttribArray(position); gl.disable(gl.SCISSOR_TEST)
      const refresh = (this.canvas.webgl!.regl_wrapper as unknown as InternalReglWrapper)._regl?._refresh
      if (refresh == null) throw new Error("Bokeh's regl refresh hook is unavailable")
      refresh(); this.model.composition_mode = "bokeh_webgl"; this.model.error = ""
      this.model.last_paint_ms = performance.now() - paint_started
    } catch (error) {
      this.model.error = `Venn rendering requires Bokeh WebGL with float textures: ${error}`
    }
  }

  private delete_resources(): void {
    if (this.resources != null) {
      const {gl, program, buffer} = this.resources
      for (const tile of this.gpu_tiles.values()) { gl.deleteTexture(tile.ranges); gl.deleteTexture(tile.valid) }
      gl.deleteBuffer(buffer); gl.deleteProgram(program)
    }
    this.gpu_tiles.clear(); this.resources = null; this.model.gpu_cache_bytes = 0
  }

  override remove(): void {
    if (this.frame_request != null) cancelAnimationFrame(this.frame_request)
    this.context_canvas?.removeEventListener("webglcontextlost", this.context_lost)
    this.context_canvas?.removeEventListener("webglcontextrestored", this.context_restored)
    this.delete_resources(); this.range_tiles.clear(); this.line_tiles.clear(); this.in_flight.clear(); super.remove()
  }
}

export namespace VennTimeSeriesRenderer {
  export type Attrs = p.AttrsOf<Props>
  export type Props = Renderer.Props & {
    request_seq: p.Property<number>; requested_pages: p.Property<[number, number][]>
    requested_tiles: p.Property<[string, number, number, number][]>
    response_seq: p.Property<number>; response_generation: p.Property<number>
    page_source: p.Property<ColumnDataSource>; page_metadata_source: p.Property<ColumnDataSource>
    range_tile_source: p.Property<ColumnDataSource>; range_tile_metadata_source: p.Property<ColumnDataSource>
    line_tile_source: p.Property<ColumnDataSource>; line_tile_metadata_source: p.Property<ColumnDataSource>
    line_source_a: p.Property<ColumnDataSource>; line_source_b: p.Property<ColumnDataSource>
    dataset_version: p.Property<string>; sample_count: p.Property<number>
    time_start: p.Property<number>; time_end: p.Property<number>; sample_interval: p.Property<number>
    source_factors: p.Property<number[]>; page_size: p.Property<number>; page_counts: p.Property<number[]>
    range_factors: p.Property<number[]>; range_page_counts: p.Property<number[]>
    line_factors: p.Property<number[]>; line_page_counts: p.Property<number[]>
    channel_names: p.Property<string[]>; amplitude_scales: p.Property<number[]>; amplitude_offsets: p.Property<number[]>
    channel_y_mins: p.Property<number[]>; channel_y_maxs: p.Property<number[]>; channel_tile_size: p.Property<number>
    shader_schema_version: p.Property<number>; color_a: p.Property<Color>; color_b: p.Property<Color>
    color_overlap: p.Property<Color>; amplitude_scale: p.Property<number>; amplitude_offset: p.Property<number>
    max_ranges_per_pixel: p.Property<number>; prefetch_pages: p.Property<number>; lod_hysteresis: p.Property<number>
    rendered_cache_bytes: p.Property<number>; data_cache_bytes: p.Property<number>
    ready: p.Property<boolean>; error: p.Property<string>; current_lod: p.Property<number>
    composition_mode: p.Property<string>; venn_visible: p.Property<boolean>; lines_visible: p.Property<boolean>
    cpu_cache_bytes: p.Property<number>; gpu_cache_bytes: p.Property<number>; cache_hits: p.Property<number>
    cache_misses: p.Property<number>; shader_compilations: p.Property<number>; tile_requests: p.Property<number>
    last_tile_latency_ms: p.Property<number>; last_paint_ms: p.Property<number>
  }
}

export interface VennTimeSeriesRenderer extends VennTimeSeriesRenderer.Attrs {}

export class VennTimeSeriesRenderer extends Renderer {
  declare properties: VennTimeSeriesRenderer.Props
  declare __view_type__: VennTimeSeriesRendererView
  static {
    this.prototype.default_view = VennTimeSeriesRendererView
    this.define<VennTimeSeriesRenderer.Props>(({Bool, Color, Float, Int, List, Ref, Str, Tuple}) => ({
      request_seq: [Int, 0], requested_pages: [List(Tuple(Int, Int)), []],
      requested_tiles: [List(Tuple(Str, Int, Int, Int)), []], response_seq: [Int, 0], response_generation: [Int, 0],
      page_source: [Ref(ColumnDataSource)], page_metadata_source: [Ref(ColumnDataSource)],
      range_tile_source: [Ref(ColumnDataSource)], range_tile_metadata_source: [Ref(ColumnDataSource)],
      line_tile_source: [Ref(ColumnDataSource)], line_tile_metadata_source: [Ref(ColumnDataSource)],
      line_source_a: [Ref(ColumnDataSource)], line_source_b: [Ref(ColumnDataSource)],
      dataset_version: [Str, ""], sample_count: [Int, 0], time_start: [Float, 0], time_end: [Float, 0], sample_interval: [Float, 1],
      source_factors: [List(Int), []], page_size: [Int, 2048], page_counts: [List(Int), []],
      range_factors: [List(Int), []], range_page_counts: [List(Int), []], line_factors: [List(Int), []], line_page_counts: [List(Int), []],
      channel_names: [List(Str), []], amplitude_scales: [List(Float), []], amplitude_offsets: [List(Float), []],
      channel_y_mins: [List(Float), []], channel_y_maxs: [List(Float), []], channel_tile_size: [Int, 8],
      shader_schema_version: [Int, 1], color_a: [Color, "red"], color_b: [Color, "blue"], color_overlap: [Color, "black"],
      amplitude_scale: [Float, 1], amplitude_offset: [Float, 0], max_ranges_per_pixel: [Int, 32], prefetch_pages: [Int, 1], lod_hysteresis: [Float, 0.2],
      rendered_cache_bytes: [Int, 64*1024*1024], data_cache_bytes: [Int, 64*1024*1024], ready: [Bool, false], error: [Str, ""],
      current_lod: [Int, 1], composition_mode: [Str, "pending"], venn_visible: [Bool, true], lines_visible: [Bool, true],
      cpu_cache_bytes: [Int, 0], gpu_cache_bytes: [Int, 0], cache_hits: [Int, 0], cache_misses: [Int, 0],
      shader_compilations: [Int, 0], tile_requests: [Int, 0],
      last_tile_latency_ms: [Float, 0], last_paint_ms: [Float, 0],
    }))
  }
}
