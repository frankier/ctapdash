import {Renderer, RendererView} from "models/renderers/renderer"
import {ColumnDataSource} from "models/sources/column_data_source"
import type {Context2d} from "core/util/canvas"
import type {Color} from "core/types"
import * as p from "core/properties"

type Page = {
  factor: number
  page: number
  series: number
  data_start: number
  core_start: number
  core_length: number
  minimum: Float32Array
  maximum: Float32Array
  valid: Uint8Array
  bytes: number
  used: number
}

const vertex_source = `#version 300 es
precision highp float;
const vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0); }
`

const fragment_source = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_ranges_a;
uniform sampler2D u_ranges_b;
uniform sampler2D u_valid_a;
uniform sampler2D u_valid_b;
uniform int u_texture_width;
uniform int u_entry_count;
uniform int u_max_aggregation;
uniform float u_source_time_start;
uniform float u_source_time_step;
uniform float u_x_start;
uniform float u_x_end;
uniform float u_y_start;
uniform float u_y_end;
uniform vec2 u_size;
uniform vec4 u_color_a;
uniform vec4 u_color_b;
uniform vec4 u_color_overlap;
out vec4 out_color;

ivec2 texel_position(int index) {
  return ivec2(index % u_texture_width, index / u_texture_width);
}

void main() {
  float x0 = u_x_start + (gl_FragCoord.x - 0.5) / u_size.x * (u_x_end - u_x_start);
  float x1 = u_x_start + (gl_FragCoord.x + 0.5) / u_size.x * (u_x_end - u_x_start);
  int first = int(floor((min(x0, x1) - u_source_time_start) / u_source_time_step));
  int next = int(floor((max(x0, x1) - u_source_time_start) / u_source_time_step));
  int last = next == first ? first + 1 : next;
  first = clamp(first, 0, u_entry_count);
  last = clamp(last, first, min(u_entry_count, first + u_max_aggregation));

  float amin = 3.402823466e38;
  float amax = -3.402823466e38;
  float bmin = 3.402823466e38;
  float bmax = -3.402823466e38;
  bool has_a = false;
  bool has_b = false;
  for (int index = 0; index < 128; index++) {
    if (first + index >= last) break;
    ivec2 position = texel_position(first + index);
    if (texelFetch(u_valid_a, position, 0).r > 0.5) {
      vec2 range_a = texelFetch(u_ranges_a, position, 0).rg;
      amin = min(amin, range_a.x);
      amax = max(amax, range_a.y);
      has_a = true;
    }
    if (texelFetch(u_valid_b, position, 0).r > 0.5) {
      vec2 range_b = texelFetch(u_ranges_b, position, 0).rg;
      bmin = min(bmin, range_b.x);
      bmax = max(bmax, range_b.y);
      has_b = true;
    }
  }
  float value = u_y_start + gl_FragCoord.y / u_size.y * (u_y_end - u_y_start);
  bool inside_a = has_a && amin <= value && value <= amax;
  bool inside_b = has_b && bmin <= value && value <= bmax;
  if (inside_a && inside_b) out_color = u_color_overlap;
  else if (inside_a) out_color = u_color_a;
  else if (inside_b) out_color = u_color_b;
  else out_color = vec4(0.0);
}
`

function numeric_column(source: ColumnDataSource, name: string): ArrayLike<number> {
  const column = source.get_column(name)
  if (column == null)
    throw new Error(`missing protocol column '${name}'`)
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
  const named: {[key: string]: string} = {red: "#ff0000", blue: "#0000ff", black: "#000000", transparent: "#00000000"}
  const value = named[color.toLowerCase()] ?? color
  const match = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(value)
  if (match == null)
    throw new Error(`Venn renderer colors must be named red/blue/black or #RRGGBB[AA], got '${color}'`)
  const rgb = parseInt(match[1], 16)
  const alpha = match[2] == null ? 255 : parseInt(match[2], 16)
  return [((rgb >>> 16) & 255)/255, ((rgb >>> 8) & 255)/255, (rgb & 255)/255, alpha/255]
}

export class VennTimeSeriesRendererView extends RendererView {
  declare model: VennTimeSeriesRenderer
  private gl_canvas = document.createElement("canvas")
  private gl: WebGL2RenderingContext | null = null
  private program: WebGLProgram | null = null
  private vao: WebGLVertexArrayObject | null = null
  private textures: WebGLTexture[] = []
  private pages = new Map<string, Page>()
  private in_flight = new Set<string>()
  private frame_request: number | null = null
  private context_lost = false
  private use_counter = 0
  private selected_factor = 1
  private last_samples_per_pixel = 0

  override initialize(): void {
    super.initialize()
    this.gl_canvas.addEventListener("webglcontextlost", this.on_context_lost)
    this.gl_canvas.addEventListener("webglcontextrestored", this.on_context_restored)
    const gl = this.gl_canvas.getContext("webgl2", {
      alpha: true, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: true, preserveDrawingBuffer: true,
    })
    if (gl == null) {
      this.model.error = "Venn time-series rendering requires WebGL2, but this browser could not create a WebGL2 context."
      throw new Error(this.model.error)
    }
    this.gl = gl
    this.initialize_gl()
  }

  override connect_signals(): void {
    super.connect_signals()
    const {x_source, y_source} = this.coordinates
    this.connect(x_source.change, () => this.schedule_viewport())
    this.connect(y_source.change, () => this.schedule_viewport())
    this.connect(this.model.properties.response_seq.change, () => this.consume_response())
    for (const property of [this.model.properties.color_a, this.model.properties.color_b,
      this.model.properties.color_overlap, this.model.properties.max_ranges_per_pixel,
      this.model.properties.data_cache_bytes])
      this.connect(property.change, () => this.schedule_viewport())
    this.schedule_viewport()
  }

  private compile_shader(type: number, source: string): WebGLShader {
    const gl = this.gl!
    const shader = gl.createShader(type)
    if (shader == null) throw new Error("WebGL2 could not allocate a shader")
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader) ?? "unknown shader compilation failure"
      gl.deleteShader(shader)
      throw new Error(message)
    }
    return shader
  }

  private initialize_gl(): void {
    const gl = this.gl!
    try {
      const vertex = this.compile_shader(gl.VERTEX_SHADER, vertex_source)
      const fragment = this.compile_shader(gl.FRAGMENT_SHADER, fragment_source)
      const program = gl.createProgram()
      if (program == null) throw new Error("WebGL2 could not allocate a shader program")
      gl.attachShader(program, vertex)
      gl.attachShader(program, fragment)
      gl.linkProgram(program)
      gl.deleteShader(vertex)
      gl.deleteShader(fragment)
      if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        throw new Error(gl.getProgramInfoLog(program) ?? "unknown shader link failure")
      this.program = program
      this.vao = gl.createVertexArray()
      gl.disable(gl.BLEND)
      gl.disable(gl.DEPTH_TEST)
      this.model.error = ""
    } catch (error) {
      this.model.error = `Venn WebGL2 initialization failed: ${error}`
      throw error
    }
  }

  private on_context_lost = (event: Event): void => {
    event.preventDefault()
    this.context_lost = true
    this.textures = []
    this.program = null
    this.vao = null
    this.model.gpu_cache_bytes = 0
    this.model.ready = false
  }

  private on_context_restored = (): void => {
    this.context_lost = false
    this.initialize_gl()
    this.schedule_viewport()
  }

  private schedule_viewport(): void {
    if (this.frame_request != null) return
    this.frame_request = requestAnimationFrame(() => {
      this.frame_request = null
      this.plan_requests()
      this.request_paint()
    })
  }

  private factor_page_count(factor: number): number {
    const index = this.model.source_factors.indexOf(factor)
    return index < 0 ? 0 : this.model.page_counts[index]
  }

  private page_key(factor: number, page: number, series: number): string {
    return `${this.model.dataset_version}:${factor}:${page}:${series}`
  }

  private pair_key(factor: number, page: number): string { return `${factor}:${page}` }

  private choose_factor(width: number): number {
    const range = this.coordinates.x_source
    const samples_per_pixel = Math.abs(range.end - range.start) / this.model.sample_interval / Math.max(width, 1)
    let factor = 1
    for (const candidate of this.model.source_factors)
      if (candidate <= Math.max(samples_per_pixel, 1)) factor = candidate
    if (this.model.source_factors.includes(this.selected_factor)) {
      const boundary = Math.max(this.selected_factor, factor)
      const change = Math.abs(samples_per_pixel - this.last_samples_per_pixel)
      if (factor != this.selected_factor && change <= boundary*this.model.lod_hysteresis)
        factor = this.selected_factor
    }
    this.last_samples_per_pixel = samples_per_pixel
    this.selected_factor = factor
    this.model.current_lod = factor
    return factor
  }

  private visible_pages(factor: number, prefetch = 0): number[] {
    const x = this.coordinates.x_source
    const low = Math.max(this.model.time_start, Math.min(x.start, x.end))
    const high = Math.min(this.model.time_end, Math.max(x.start, x.end))
    if (high < low) return []
    const step = this.model.sample_interval*factor
    const first_entry = Math.max(0, Math.floor((low - this.model.time_start)/step))
    const last_entry = Math.max(first_entry, Math.floor((high - this.model.time_start)/step))
    const count = this.factor_page_count(factor)
    const first = Math.max(0, Math.floor(first_entry/this.model.page_size) - prefetch)
    const last = Math.min(count - 1, Math.floor(last_entry/this.model.page_size) + prefetch)
    const pages = []
    for (let page = first; page <= last; page++) pages.push(page)
    return pages
  }

  private has_pair(factor: number, page: number): boolean {
    return this.pages.has(this.page_key(factor, page, 0)) && this.pages.has(this.page_key(factor, page, 1))
  }

  private plan_requests(): void {
    const width = Math.max(1, Math.round(this.plot_view.frame.bbox.width*this.canvas.pixel_ratio))
    const factor = this.choose_factor(width)
    const visible = this.visible_pages(factor)
    const center = visible.length ? (visible[0] + visible[visible.length - 1])/2 : 0
    const candidates = this.visible_pages(factor, this.model.prefetch_pages)
      .filter((page) => !this.has_pair(factor, page) && !this.in_flight.has(this.pair_key(factor, page)))
      .sort((a, b) => Math.abs(a - center) - Math.abs(b - center))
    if (candidates.length != 0) {
      const batch = candidates.slice(0, 4)
      for (const page of batch) this.in_flight.add(this.pair_key(factor, page))
      this.model.ready = false
      this.model.requested_pages = batch.map((page) => [factor, page])
      this.model.request_seq += 1
    }
  }

  private consume_response(): void {
    try {
      const metadata = this.model.page_metadata_source
      const data = this.model.page_source
      const minimum = numeric_column(data, "minimum")
      const maximum = numeric_column(data, "maximum")
      const valid = numeric_column(data, "valid")
      const factors = numeric_column(metadata, "source_factor")
      const indices = numeric_column(metadata, "page_index")
      const series_ids = numeric_column(metadata, "series_id")
      const offsets = numeric_column(metadata, "offset")
      const lengths = numeric_column(metadata, "length")
      const core_starts = numeric_column(metadata, "core_start")
      const core_lengths = numeric_column(metadata, "core_length")
      for (let row = 0; row < factors.length; row++) {
        const factor = factors[row], page_index = indices[row], series = series_ids[row]
        const offset = offsets[row], length = lengths[row]
        const mins = new Float32Array(length), maxes = new Float32Array(length), validity = new Uint8Array(length)
        for (let index = 0; index < length; index++) {
          mins[index] = minimum[offset + index]
          maxes[index] = maximum[offset + index]
          validity[index] = valid[offset + index]
        }
        const core_start = core_starts[row]
        const page: Page = {
          factor, page: page_index, series,
          data_start: page_index*this.model.page_size - core_start,
          core_start, core_length: core_lengths[row], minimum: mins, maximum: maxes, valid: validity,
          bytes: mins.byteLength + maxes.byteLength + validity.byteLength, used: ++this.use_counter,
        }
        this.pages.set(this.page_key(factor, page_index, series), page)
        this.in_flight.delete(this.pair_key(factor, page_index))
      }
      this.evict_cpu_cache()
      this.schedule_viewport()
    } catch (error) {
      this.model.error = `Venn page protocol error: ${error}`
    }
  }

  private evict_cpu_cache(): void {
    let bytes = 0
    for (const page of this.pages.values()) bytes += page.bytes
    const pinned = new Set<string>()
    for (const factor of this.model.source_factors)
      for (const page of this.visible_pages(factor))
        for (const series of [0, 1]) pinned.add(this.page_key(factor, page, series))
    const evictable = [...this.pages.entries()].filter(([key]) => !pinned.has(key)).sort((a, b) => a[1].used - b[1].used)
    for (const [key, page] of evictable) {
      if (bytes <= this.model.data_cache_bytes) break
      this.pages.delete(key)
      bytes -= page.bytes
    }
    this.model.cpu_cache_bytes = bytes
  }

  private best_factor(): number | null {
    const candidates = [this.selected_factor, ...this.model.source_factors]
    for (const factor of candidates) {
      const visible = this.visible_pages(factor)
      if (visible.length != 0 && visible.some((page) => this.has_pair(factor, page))) return factor
    }
    return null
  }

  private build_viewport_arrays(factor: number): {
    start: number, count: number, a: Float32Array, b: Float32Array,
    valid_a: Uint8Array, valid_b: Uint8Array, complete: boolean,
  } | null {
    const visible = this.visible_pages(factor)
    if (visible.length == 0) return null
    const start = visible[0]*this.model.page_size
    const level_count = Math.floor(this.model.sample_count/factor)
    const stop = Math.min(level_count, (visible[visible.length - 1] + 1)*this.model.page_size)
    const count = stop - start
    const a = new Float32Array(count*2), b = new Float32Array(count*2)
    const valid_a = new Uint8Array(count), valid_b = new Uint8Array(count)
    let complete = true
    for (const page_index of visible) {
      for (const series of [0, 1]) {
        const page = this.pages.get(this.page_key(factor, page_index, series))
        if (page == null) { complete = false; continue }
        page.used = ++this.use_counter
        const target = series == 0 ? a : b
        const target_valid = series == 0 ? valid_a : valid_b
        for (let core = 0; core < page.core_length; core++) {
          const source_index = page.core_start + core
          const target_index = page.data_start + source_index - start
          if (target_index < 0 || target_index >= count) continue
          target[target_index*2] = page.minimum[source_index]
          target[target_index*2 + 1] = page.maximum[source_index]
          target_valid[target_index] = page.valid[source_index]
        }
      }
    }
    return {start, count, a, b, valid_a, valid_b, complete}
  }

  private make_texture(internal: number, format: number, type: number, width: number, height: number, data: ArrayBufferView): WebGLTexture {
    const gl = this.gl!, texture = gl.createTexture()
    if (texture == null) throw new Error("WebGL2 could not allocate a data texture")
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, type, data)
    this.textures.push(texture)
    return texture
  }

  private clear_textures(): void {
    if (this.gl != null && !this.context_lost)
      for (const texture of this.textures) this.gl.deleteTexture(texture)
    this.textures = []
    this.model.gpu_cache_bytes = 0
  }

  protected override _paint(ctx: Context2d): void {
    if (this.context_lost || this.gl == null || this.program == null) return
    const bbox = this.plot_view.frame.bbox, ratio = this.canvas.pixel_ratio
    const width = Math.max(1, Math.round(bbox.width*ratio)), height = Math.max(1, Math.round(bbox.height*ratio))
    if (this.gl_canvas.width != width || this.gl_canvas.height != height) {
      this.gl_canvas.width = width; this.gl_canvas.height = height
    }
    const gl = this.gl
    gl.viewport(0, 0, width, height)
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    const factor = this.best_factor()
    if (factor != null) {
      const arrays = this.build_viewport_arrays(factor)
      if (arrays != null && arrays.count > 0) {
        this.clear_textures()
        const max_width = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number
        const texture_width = Math.min(max_width, arrays.count), texture_height = Math.ceil(arrays.count/texture_width)
        const entries = texture_width*texture_height
        const pad_float = (input: Float32Array, components: number): Float32Array => {
          if (input.length == entries*components) return input
          const output = new Float32Array(entries*components); output.set(input); return output
        }
        const pad_byte = (input: Uint8Array): Uint8Array => {
          if (input.length == entries) return input
          const output = new Uint8Array(entries); output.set(input); return output
        }
        const textures = [
          this.make_texture(gl.RG32F, gl.RG, gl.FLOAT, texture_width, texture_height, pad_float(arrays.a, 2)),
          this.make_texture(gl.RG32F, gl.RG, gl.FLOAT, texture_width, texture_height, pad_float(arrays.b, 2)),
          this.make_texture(gl.R8, gl.RED, gl.UNSIGNED_BYTE, texture_width, texture_height, pad_byte(arrays.valid_a)),
          this.make_texture(gl.R8, gl.RED, gl.UNSIGNED_BYTE, texture_width, texture_height, pad_byte(arrays.valid_b)),
        ]
        this.model.gpu_cache_bytes = entries*18
        gl.useProgram(this.program); gl.bindVertexArray(this.vao)
        const uniform1i = (name: string, value: number) => gl.uniform1i(gl.getUniformLocation(this.program!, name), value)
        const uniform1f = (name: string, value: number) => gl.uniform1f(gl.getUniformLocation(this.program!, name), value)
        for (let unit = 0; unit < textures.length; unit++) {
          gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, textures[unit])
        }
        uniform1i("u_ranges_a", 0); uniform1i("u_ranges_b", 1); uniform1i("u_valid_a", 2); uniform1i("u_valid_b", 3)
        uniform1i("u_texture_width", texture_width); uniform1i("u_entry_count", arrays.count)
        uniform1i("u_max_aggregation", Math.min(128, Math.max(1, this.model.max_ranges_per_pixel)))
        uniform1f("u_source_time_start", this.model.time_start + arrays.start*factor*this.model.sample_interval)
        uniform1f("u_source_time_step", factor*this.model.sample_interval)
        uniform1f("u_x_start", this.coordinates.x_source.start); uniform1f("u_x_end", this.coordinates.x_source.end)
        uniform1f("u_y_start", this.coordinates.y_source.start); uniform1f("u_y_end", this.coordinates.y_source.end)
        gl.uniform2f(gl.getUniformLocation(this.program, "u_size"), width, height)
        gl.uniform4fv(gl.getUniformLocation(this.program, "u_color_a"), css_color(this.model.color_a))
        gl.uniform4fv(gl.getUniformLocation(this.program, "u_color_b"), css_color(this.model.color_b))
        gl.uniform4fv(gl.getUniformLocation(this.program, "u_color_overlap"), css_color(this.model.color_overlap))
        gl.drawArrays(gl.TRIANGLES, 0, 3)
        if (factor == this.selected_factor && arrays.complete) this.model.ready = true
      }
    }
    const smoothing = ctx.imageSmoothingEnabled
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(this.gl_canvas, bbox.x, bbox.y, bbox.width, bbox.height)
    ctx.imageSmoothingEnabled = smoothing
  }

  override remove(): void {
    if (this.frame_request != null) cancelAnimationFrame(this.frame_request)
    this.frame_request = null
    this.gl_canvas.removeEventListener("webglcontextlost", this.on_context_lost)
    this.gl_canvas.removeEventListener("webglcontextrestored", this.on_context_restored)
    this.clear_textures()
    if (this.gl != null && !this.context_lost) {
      if (this.vao != null) this.gl.deleteVertexArray(this.vao)
      if (this.program != null) this.gl.deleteProgram(this.program)
    }
    this.pages.clear(); this.in_flight.clear()
    super.remove()
  }
}

export namespace VennTimeSeriesRenderer {
  export type Attrs = p.AttrsOf<Props>
  export type Props = Renderer.Props & {
    request_seq: p.Property<number>; requested_pages: p.Property<[number, number][]>
    response_seq: p.Property<number>; response_generation: p.Property<number>
    page_source: p.Property<ColumnDataSource>; page_metadata_source: p.Property<ColumnDataSource>
    dataset_version: p.Property<string>; sample_count: p.Property<number>
    time_start: p.Property<number>; time_end: p.Property<number>; sample_interval: p.Property<number>
    source_factors: p.Property<number[]>; page_size: p.Property<number>; page_counts: p.Property<number[]>
    shader_schema_version: p.Property<number>; color_a: p.Property<Color>; color_b: p.Property<Color>
    color_overlap: p.Property<Color>; max_ranges_per_pixel: p.Property<number>
    prefetch_pages: p.Property<number>; lod_hysteresis: p.Property<number>
    rendered_cache_bytes: p.Property<number>; data_cache_bytes: p.Property<number>
    ready: p.Property<boolean>; error: p.Property<string>; current_lod: p.Property<number>
    cpu_cache_bytes: p.Property<number>; gpu_cache_bytes: p.Property<number>
    cache_hits: p.Property<number>; cache_misses: p.Property<number>
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
      response_seq: [Int, 0], response_generation: [Int, 0],
      page_source: [Ref(ColumnDataSource)], page_metadata_source: [Ref(ColumnDataSource)],
      dataset_version: [Str, ""], sample_count: [Int, 0], time_start: [Float, 0], time_end: [Float, 0],
      sample_interval: [Float, 1], source_factors: [List(Int), []], page_size: [Int, 2048], page_counts: [List(Int), []],
      shader_schema_version: [Int, 1], color_a: [Color, "red"], color_b: [Color, "blue"], color_overlap: [Color, "black"],
      max_ranges_per_pixel: [Int, 32], prefetch_pages: [Int, 1], lod_hysteresis: [Float, 0.2],
      rendered_cache_bytes: [Int, 64*1024*1024], data_cache_bytes: [Int, 64*1024*1024],
      ready: [Bool, false], error: [Str, ""], current_lod: [Int, 1], cpu_cache_bytes: [Int, 0],
      gpu_cache_bytes: [Int, 0], cache_hits: [Int, 0], cache_misses: [Int, 0],
    }))
  }
}
